"""Fused Dispatcher with Quantization Support for MoE models.

This module provides QuantizedDispatcherCacheGroupedMLP which inherits MegatronModule
and implements custom forward/backward logic for dynamic quantization:

1. Forward: Receive quantized weights + delta/z from CPU optimizer → GPU dequantizes → compute
2. Backward: GPU computes LSQ delta/z updates → returns new values to CPU optimizer

Key design:
- Inherits MegatronModule (NOT FusedDispatcherCacheGroupedMLP)
- All storage managed by optimizer (CPU shared memory + quantization pool)
- Dispatcher only handles computation: prefetch, dequantize, GEMM, LSQ gradients
- Uses optimizer's unified prefetch_expert_data() interface

Architecture:
- Optimizer: CPU shared memory (EP), quantization pool (INT8/INT4), standard Adam for GPU params
- Dispatcher: GPU workspace for dequantized weights, CUDA streams, LSQ gradient computation

Usage:
    # In model initialization
    from megatron.core.transformer.moe.quantized_dispatcher import QuantizedDispatcherCacheGroupedMLP

    # In optimizer
    optimizer = FusedAdamLSQCPUOffloadOptimizer(model, lr=1e-4)
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.distributed as dist
from torch.cuda import nvtx

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.moe.experts import (
    PerSetActivationCache,
    pin_existing_tensor,
    log_memory,
)

# Lazy import for global expert ID encoding to avoid circular import
# The encoding functions are imported when needed, not at module load time
_encode_global_expert_id = None
_decode_global_expert_id = None
_num_experts_per_layer_cache = None

def _get_encode_global_expert_id():
    """Lazy import of encode_global_expert_id to avoid circular import."""
    global _encode_global_expert_id
    if _encode_global_expert_id is None:
        from megatron.core.optimizer.fused_adam_lsq_cpu_offload_optimizer import encode_global_expert_id
        _encode_global_expert_id = encode_global_expert_id
    return _encode_global_expert_id

def _get_decode_global_expert_id():
    """Lazy import of decode_global_expert_id to avoid circular import."""
    global _decode_global_expert_id
    if _decode_global_expert_id is None:
        from megatron.core.optimizer.fused_adam_lsq_cpu_offload_optimizer import decode_global_expert_id
        _decode_global_expert_id = decode_global_expert_id
    return _decode_global_expert_id


class _QuantizedGlobalBufferManager:
    """Global Buffer Manager for QuantizedDispatcherCacheGroupedMLP - all layers share the same buffers.

    Since Transformer layers are executed sequentially, there's no need for each layer
    to have its own GPU workspaces for dequantized weights and delta/z params. This singleton
    manager provides shared buffers that all QuantizedDispatcherCacheGroupedMLP instances can reuse.

    Key difference from _GlobalBufferManager for CacheGroupedMLP:
    - This manager handles mixed precision: BF16 weights + INT8/INT4 quant_w (uint8)
    - Separate buffers for BF16 and quantized weights
    - Delta/z workspace for INT8/INT4 quantization params

    Buffer types:
    - _w1_bf16_gpu_workspace: BF16 weights (for precision=16)
    - _w1_quant_gpu_workspace: uint8 quant_w (for precision=8 or 4)
    - _delta/z_gpu_workspace: float32 delta/z params (for INT8/INT4)
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_initialized(self) -> bool:
        """Check if buffers have been initialized."""
        return self._initialized

    def initialize(
        self,
        num_global_experts: int,
        hidden_size: int,
        fc1_out_features: int,
        ffn_hidden_size: int,
        quant_group_size: int,
        dtype: torch.dtype,
        device: torch.device,
        max_experts_per_set: Optional[int] = None,
    ):
        """Initialize global shared buffers. Creates new buffers if not initialized
        or if current buffers are smaller than needed.

        Args:
            num_global_experts: Total number of experts
            hidden_size: Hidden size
            fc1_out_features: FC1 output features (hidden_size * 2 for GLU)
            ffn_hidden_size: FFN hidden size
            quant_group_size: Quantization group size for delta/z tensors
            dtype: Data type for BF16 weight workspace
            device: Device for buffers
            max_experts_per_set: Maximum number of experts in a single set
        """
        # Default max_experts_per_set to num_global_experts for safety
        if max_experts_per_set is None:
            max_experts_per_set = num_global_experts

        # Compute number of quantization groups
        w1_num_elements = hidden_size * fc1_out_features
        w2_num_elements = ffn_hidden_size * hidden_size
        num_groups_w1 = w1_num_elements // quant_group_size
        num_groups_w2 = w2_num_elements // quant_group_size

        # INT4 quant_w size is half of original (packed)
        w1_quant_size = w1_num_elements // 2  # INT4 packed
        w2_quant_size = w2_num_elements // 2

        # Check if we need to (re)allocate buffers
        need_reallocate = False
        if not self._initialized:
            need_reallocate = True
        else:
            # Check if current buffers are large enough
            if (self._w1_bf16_gpu_workspace is None or
                self._w1_bf16_gpu_workspace.shape[1] < max_experts_per_set or
                self._w1_bf16_gpu_workspace.shape[3] < fc1_out_features):
                need_reallocate = True
            if (self._w1_quant_gpu_workspace is None or
                self._w1_quant_gpu_workspace.shape[1] < max_experts_per_set or
                self._w1_quant_gpu_workspace.shape[3] < fc1_out_features):
                need_reallocate = True
            if (self._delta_w1_gpu_workspace is None or
                self._delta_w1_gpu_workspace.shape[1] < max_experts_per_set or
                self._delta_w1_gpu_workspace.shape[2] < num_groups_w1):
                need_reallocate = True

        if not need_reallocate:
            log_memory(f"_QuantizedGlobalBufferManager.initialize: reusing existing buffers")
            return

        log_memory(f"_QuantizedGlobalBufferManager.initialize: allocating new buffers")

        # GPU workspace for BF16 weights (double-buffered)
        # Shape: [2 buffers, max_experts_per_set, hidden_size, fc1_out_features]
        self._w1_bf16_gpu_workspace = torch.empty(
            2, max_experts_per_set, hidden_size, fc1_out_features,
            dtype=dtype, device=device
        )
        self._w2_bf16_gpu_workspace = torch.empty(
            2, max_experts_per_set, ffn_hidden_size, hidden_size,
            dtype=dtype, device=device
        )

        # GPU workspace for quantized weights (uint8, double-buffered)
        # INT8: same numel, INT4: half numel (packed)
        # Use largest size (INT8) to accommodate both
        self._w1_quant_gpu_workspace = torch.empty(
            2, max_experts_per_set, hidden_size, fc1_out_features,
            dtype=torch.uint8, device=device
        )
        self._w2_quant_gpu_workspace = torch.empty(
            2, max_experts_per_set, ffn_hidden_size, hidden_size,
            dtype=torch.uint8, device=device
        )

        # Delta/z workspace for quantization params (double-buffered)
        # Shape: [2 buffers, max_experts_per_set, num_groups]
        self._delta_w1_gpu_workspace = torch.empty(
            2, max_experts_per_set, num_groups_w1,
            dtype=torch.float32, device=device
        )
        self._z_w1_gpu_workspace = torch.empty(
            2, max_experts_per_set, num_groups_w1,
            dtype=torch.float32, device=device
        )
        self._delta_w2_gpu_workspace = torch.empty(
            2, max_experts_per_set, num_groups_w2,
            dtype=torch.float32, device=device
        )
        self._z_w2_gpu_workspace = torch.empty(
            2, max_experts_per_set, num_groups_w2,
            dtype=torch.float32, device=device
        )

        log_memory(f"_QuantizedGlobalBufferManager.initialize: after workspace allocation "
                   f"(w1_bf16 shape={self._w1_bf16_gpu_workspace.shape}, "
                   f"w1_quant shape={self._w1_quant_gpu_workspace.shape}, "
                   f"delta_w1 shape={self._delta_w1_gpu_workspace.shape})")

        # CUDA streams (shared across layers) - only create once
        if not self._initialized:
            self._load_stream = torch.cuda.Stream()
            self._grad_offload_stream = torch.cuda.Stream()
            self._comm_stream = torch.cuda.Stream()
            self.compute_events = [torch.cuda.Event() for _ in range(2)]

        self._initialized = True

    def cleanup(self):
        """Release all resources. Should be called after all layers are done."""
        if not self._initialized:
            return

        # Synchronize and clean up CUDA streams
        if self._load_stream is not None:
            self._load_stream.synchronize()
            self._load_stream = None
        if self._grad_offload_stream is not None:
            self._grad_offload_stream.synchronize()
            self._grad_offload_stream = None
        if self._comm_stream is not None:
            self._comm_stream.synchronize()
            self._comm_stream = None

        # Release buffers
        self._w1_bf16_gpu_workspace = None
        self._w2_bf16_gpu_workspace = None
        self._w1_quant_gpu_workspace = None
        self._w2_quant_gpu_workspace = None
        self._delta_w1_gpu_workspace = None
        self._z_w1_gpu_workspace = None
        self._delta_w2_gpu_workspace = None
        self._z_w2_gpu_workspace = None

        self._initialized = False

    @property
    def w1_bf16_gpu_workspace(self):
        return self._w1_bf16_gpu_workspace

    @property
    def w2_bf16_gpu_workspace(self):
        return self._w2_bf16_gpu_workspace

    @property
    def w1_quant_gpu_workspace(self):
        return self._w1_quant_gpu_workspace

    @property
    def w2_quant_gpu_workspace(self):
        return self._w2_quant_gpu_workspace

    @property
    def delta_w1_gpu_workspace(self):
        return self._delta_w1_gpu_workspace

    @property
    def z_w1_gpu_workspace(self):
        return self._z_w1_gpu_workspace

    @property
    def delta_w2_gpu_workspace(self):
        return self._delta_w2_gpu_workspace

    @property
    def z_w2_gpu_workspace(self):
        return self._z_w2_gpu_workspace

    @property
    def load_stream(self):
        return self._load_stream

    @property
    def grad_offload_stream(self):
        return self._grad_offload_stream

    @property
    def comm_stream(self):
        return self._comm_stream


# Global singleton instance
_quantized_global_buffer_manager = _QuantizedGlobalBufferManager()


def cleanup_quantized_global_buffers():
    """Clean up global shared buffers for QuantizedDispatcher.

    This should be called after all QuantizedDispatcherCacheGroupedMLP layers have finished
    their work (e.g., at the end of training or inference).
    """
    _quantized_global_buffer_manager.cleanup()


class QuantizedDispatcherCacheGroupedMLP(MegatronModule):
    """EP-only MoE with Global-Schedule-Based All-to-All Pipeline + Dynamic Quantization.

    Key differences from FusedDispatcherCacheGroupedMLP:
    1. Inherits MegatronModule (NOT FusedDispatcherCacheGroupedMLP)
    2. Forward/backward logic completely custom for quantization
    3. Weight storage managed by optimizer (dispatcher has no weight1/weight2)
    4. GPU workspace for dequantized weights + delta/z params
    5. LSQ delta/z gradient computation during backward

    Quantization Data Flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Forward:                                                        │
    │   CPU optimizer → GPU: quant_weight + delta/z (INT8/INT4)       │
    │                     OR main_weight (BF16)                       │
    │   GPU: Dequantize → weight_bf16 = quant * delta + z            │
    │   GPU: GEMM with BF16 weights                                   │
    │                                                                 │
    │ Backward:                                                       │
    │   GPU: Prefetch + dequantize again (no activation saved)       │
    │   GPU: Recompute forward, compute gradients                     │
    │   GPU: Compute LSQ gradients for delta/z                       │
    │   GPU → CPU: Return delta_new, z_new (not gradients!)          │
    │   CPU optimizer: Store received values                         │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        num_global_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        layer_number: Optional[int] = None,
        quant_group_size: int = 128,
        lr_quant: float = 1e-4,
    ):
        """Initialize QuantizedDispatcherCacheGroupedMLP.

        Args:
            num_global_experts: Number of global experts
            config: TransformerConfig
            pg_collection: ProcessGroupCollection
            layer_number: Layer number (used as layer_id for global expert ID)
            quant_group_size: Quantization group size (default: 128)
            lr_quant: Learning rate for LSQ delta/z updates on GPU
        """
        super().__init__(config=config)
        self.config = config
        self.num_global_experts = num_global_experts
        self.layer_number = layer_number
        gg.assert_grouped_gemm_is_available()

        # No TP support - only EP
        assert config.add_bias_linear == False, (
            "bias not supported in Grouped GEMM, please set '--disable-bias-linear' instead."
        )
        assert config.moe_latent_size is None, (
            "MoE latent projection not supported in QuantizedDispatcher."
        )

        # EP group setup
        self.ep_group = pg_collection.ep if pg_collection else None
        self.ep_size = self.ep_group.size() if self.ep_group else 1
        self.ep_rank = dist.get_rank(self.ep_group) if self.ep_group else 0

        # Layer ID for global expert ID encoding
        # NOTE: layer_number starts from 1 in Megatron, but layer_id should start from 0
        self.layer_id = (layer_number - 1) if layer_number is not None else 0

        # Quantization parameters
        self.quant_group_size = quant_group_size
        self.lr_quant = lr_quant

        # Calculate weight shapes
        hidden_size = config.hidden_size
        ffn_hidden_size = config.moe_ffn_hidden_size
        fc1_out_features = ffn_hidden_size
        if config.gated_linear_unit:
            fc1_out_features *= 2
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return config.activation_func(x[0]) * x[1]
            self.activation_func = glu
        else:
            self.activation_func = config.activation_func

        self.hidden_size = hidden_size
        self.fc1_out_features = fc1_out_features
        self.ffn_hidden_size = ffn_hidden_size

        # Reference to optimizer (set by optimizer during initialization)
        self._quant_optimizer: Optional[Any] = None

        # CRITICAL: No weight1/weight2 Parameters!
        # All expert weights managed by optimizer in CPU shared memory

        # Max experts per set (for workspace sizing)
        self.max_experts_per_set = min(16, num_global_experts)

        # ========== Initialize Global Buffer Manager (shared across layers) ==========
        device = torch.cuda.current_device()
        dtype = config.params_dtype

        # Initialize global buffer manager with needed sizes
        # This will reuse existing buffers if they're large enough
        _quantized_global_buffer_manager.initialize(
            num_global_experts=num_global_experts,
            hidden_size=hidden_size,
            fc1_out_features=fc1_out_features,
            ffn_hidden_size=ffn_hidden_size,
            quant_group_size=quant_group_size,
            dtype=dtype,
            device=device,
            max_experts_per_set=self.max_experts_per_set,
        )

        # Use shared workspace from global buffer manager
        # BF16 workspace for precision=16
        self._w1_bf16_gpu_workspace = _quantized_global_buffer_manager.w1_bf16_gpu_workspace
        self._w2_bf16_gpu_workspace = _quantized_global_buffer_manager.w2_bf16_gpu_workspace
        # Quant workspace (uint8) for precision=8/4
        self._w1_quant_gpu_workspace = _quantized_global_buffer_manager.w1_quant_gpu_workspace
        self._w2_quant_gpu_workspace = _quantized_global_buffer_manager.w2_quant_gpu_workspace
        # Delta/z workspace for INT8/INT4
        self._delta_w1_gpu_workspace = _quantized_global_buffer_manager.delta_w1_gpu_workspace
        self._z_w1_gpu_workspace = _quantized_global_buffer_manager.z_w1_gpu_workspace
        self._delta_w2_gpu_workspace = _quantized_global_buffer_manager.delta_w2_gpu_workspace
        self._z_w2_gpu_workspace = _quantized_global_buffer_manager.z_w2_gpu_workspace

        # Use shared streams from global buffer manager
        self._load_stream = _quantized_global_buffer_manager.load_stream
        self._grad_offload_stream = _quantized_global_buffer_manager.grad_offload_stream
        self._comm_stream = _quantized_global_buffer_manager.comm_stream

        # ========== Initialize CUDA Events (per-layer, NOT shared) ==========
        # Events must be per-layer because each layer's forward/backward has independent timing
        self._compute_events = [torch.cuda.Event() for _ in range(2)]
        self._comm_events = [torch.cuda.Event() for _ in range(2)]
        self._scatter_done_events = [torch.cuda.Event() for _ in range(2)]
        self._prep_done_events = [torch.cuda.Event() for _ in range(2)]

        # Per-set events for gradient offload (max sets = num_local_experts)
        num_local_experts = num_global_experts // self.ep_size
        self._grad_ready_events = [torch.cuda.Event() for _ in range(num_local_experts)]

        # CPU optimizer update event (set by optimizer after step())
        self._cpu_update_event: Optional[torch.cuda.Event] = None

        # ========== Per-expert quantization state ==========
        # Key: global_expert_id
        self._quant_handlers: Dict[int, Dict[str, Any]] = {}

        # Updated delta/z to return to CPU optimizer
        # Key: global_expert_id -> (delta_w1_new, z_w1_new, delta_w2_new, z_w2_new)
        self._delta_z_updates: Dict[int, Tuple[torch.Tensor, ...]] = {}

        # Current precisions and global expert IDs per buffer
        self._current_precisions: Dict[int, List[int]] = {}
        self._current_global_expert_ids: Dict[int, List[int]] = {}

        # Activation offload (optional)
        cache_enabled = getattr(config, "moe_enable_expert_weight_cache", True)
        self.activation_offload = (
            getattr(config, "moe_activation_offload", False) and cache_enabled
        )
        self._per_set_activation_cache = PerSetActivationCache(enabled=self.activation_offload)

        log_memory(f"QuantizedDispatcher.__init__: layer {layer_number}, rank {self.ep_rank}")

    def set_quant_optimizer(self, optimizer):
        """Set reference to the quantization optimizer.

        The optimizer provides:
        - prefetch_expert_data(): Unified interface to transfer BF16 or quant+delta/z to GPU
        - receive_gpu_updates(): Store updated delta/z values
        - GlobalQuantizationPool for INT8/INT4 slots

        Args:
            optimizer: FusedAdamLSQCPUOffloadOptimizer instance
        """
        self._quant_optimizer = optimizer

    def set_cpu_update_event(self, event: torch.cuda.Event):
        """Set CPU optimizer update event.

        Called by optimizer after optimizer.step() completes.
        Dispatcher waits on this before prefetching weights.
        """
        self._cpu_update_event = event

    def forward(self, hidden_states: torch.Tensor, routing_map: torch.Tensor,
                probs: torch.Tensor, expert_sets: List[List[int]]) -> Tuple[torch.Tensor, None]:
        """Forward pass using QuantizedDispatcherFunction."""
        return QuantizedDispatcherFunction.apply(
            self, hidden_states, routing_map, probs, expert_sets
        )

    def send_delta_z_updates_to_optimizer(self):
        """Send computed delta/z updates to optimizer.

        Called after backward pass completes.
        Optimizer receives and stores the updated values directly.
        """
        if self._quant_optimizer is None:
            return

        for global_expert_id, updates in self._delta_z_updates.items():
            self._quant_optimizer.receive_gpu_updates(
                global_expert_id,
                *updates,
            )

        # Clear updates for next iteration
        self._delta_z_updates.clear()

    def _phase0_exchange_metadata(
        self,
        routing_map: torch.Tensor,
        expert_sets: List[List[int]],
    ) -> Tuple[torch.Tensor, List[List[List[int]]]]:
        """Exchange global schedule and token distribution across EP ranks.

        CRITICAL: All distributed communication must happen on GPU tensors
        because NCCL backend does not support CPU tensors.

        Returns:
            global_tokens_distribution: [ep_size, num_global_experts] CPU tensor
            global_expert_sets: [ep_size][num_sets][experts] complete schedule
        """
        num_global_experts = self.num_global_experts
        device = routing_map.device
        num_sets = len(expert_sets)

        # ============ Part 1: All-gather token distribution ============
        # Local token count per expert (keep on GPU for NCCL communication)
        local_tokens_per_expert = routing_map.sum(dim=0).long()  # [num_global_experts]

        # All-gather on GPU FIRST to avoid device-host sync stall
        global_tokens_distribution_gpu = torch.empty(
            self.ep_size, num_global_experts, dtype=torch.long, device=device
        )

        if self.ep_size > 1:
            dist.all_gather_into_tensor(
                global_tokens_distribution_gpu,
                local_tokens_per_expert,
                group=self.ep_group
            )
        else:
            global_tokens_distribution_gpu[0] = local_tokens_per_expert

        # Copy to CPU after GPU communication is done
        global_tokens_distribution = global_tokens_distribution_gpu.cpu()

        # ============ Part 2: All-gather expert_sets (global schedule) ============
        # Each rank contributes its expert_sets (experts IT processes per set)
        # We use a fixed-size tensor approach to avoid pickle serialization on CPU

        # First, communicate the number of sets (should be same across all ranks)
        num_sets_tensor = torch.tensor([num_sets], dtype=torch.long, device=device)
        if self.ep_size > 1:
            all_num_sets = torch.empty(self.ep_size, dtype=torch.long, device=device)
            dist.all_gather_into_tensor(
                all_num_sets, num_sets_tensor, group=self.ep_group
            )
            # Verify all ranks have the same number of sets
            if not torch.all(all_num_sets == num_sets):
                raise RuntimeError(
                    f"All ranks must have the same number of expert sets! "
                    f"Got {all_num_sets.tolist()}, expected {num_sets}"
                )

        # Communicate max experts per set for buffer allocation
        max_experts_per_set = max(len(s) for s in expert_sets) if expert_sets else 0
        max_experts_tensor = torch.tensor([max_experts_per_set], dtype=torch.long, device=device)
        if self.ep_size > 1:
            all_max_experts = torch.empty(self.ep_size, dtype=torch.long, device=device)
            dist.all_gather_into_tensor(
                all_max_experts, max_experts_tensor, group=self.ep_group
            )
            global_max_experts = int(all_max_experts.max().item())
        else:
            global_max_experts = max_experts_per_set

        # Create fixed-size tensor for expert_sets communication
        # Shape: [num_sets, global_max_experts]
        # Use -1 as padding for unused slots
        expert_sets_tensor = torch.full(
            (num_sets, global_max_experts), -1, dtype=torch.long, device=device
        )
        for set_idx, exp_ids in enumerate(expert_sets):
            expert_sets_tensor[set_idx, :len(exp_ids)] = torch.tensor(exp_ids, device=device)

        if self.ep_size > 1:
            global_expert_sets_tensor = torch.empty(
                self.ep_size, num_sets, global_max_experts, dtype=torch.long, device=device
            )
            dist.all_gather_into_tensor(
                global_expert_sets_tensor.flatten(),
                expert_sets_tensor.flatten(),
                group=self.ep_group
            )
        else:
            global_expert_sets_tensor = expert_sets_tensor.unsqueeze(0)

        # Convert tensor back to nested lists, removing padding (-1 values)
        global_expert_sets: List[List[List[int]]] = []
        for rank_idx in range(self.ep_size):
            rank_sets: List[List[int]] = []
            for set_idx in range(num_sets):
                expert_ids = global_expert_sets_tensor[rank_idx, set_idx].tolist()
                # Filter out -1 padding values
                expert_ids = [e for e in expert_ids if e >= 0]
                rank_sets.append(expert_ids)
            global_expert_sets.append(rank_sets)

        return global_tokens_distribution, global_expert_sets

    def _compute_splits_for_set(
        self,
        set_idx: int,
        global_expert_sets: List[List[List[int]]],
        global_tokens_distribution: torch.Tensor,
    ) -> Tuple[List[int], List[int]]:
        """Compute send/recv splits for a specific set_idx using GLOBAL schedule.

        send_splits[dest_rank] = tokens WE have that are routed to experts
                                 in global_expert_sets[dest_rank][set_idx]

        recv_splits[src_rank] = tokens src_rank has that are routed to experts
                                in global_expert_sets[OUR_RANK][set_idx]

        Args:
            set_idx: Current set index
            global_expert_sets: [ep_size][num_sets][experts] - complete schedule
            global_tokens_distribution: [ep_size, num_global_experts] token counts

        Returns:
            send_splits: List of token counts to send to each rank
            recv_splits: List of token counts to receive from each rank
        """
        send_splits = []
        recv_splits = []

        for ep_rank in range(self.ep_size):
            # SEND to ep_rank: count tokens we have for experts that ep_rank processes
            dest_experts = global_expert_sets[ep_rank][set_idx]
            send_count = sum(
                global_tokens_distribution[self.ep_rank][exp_id].item()
                for exp_id in dest_experts
            )
            send_splits.append(send_count)

            # RECV from ep_rank: count tokens ep_rank has for experts WE process
            our_experts = global_expert_sets[self.ep_rank][set_idx]
            recv_count = sum(
                global_tokens_distribution[ep_rank][exp_id].item()
                for exp_id in our_experts
            )
            recv_splits.append(recv_count)

        return send_splits, recv_splits

    def _compute_all_splits(
        self,
        num_sets: int,
        global_expert_sets: List[List[List[int]]],
        global_tokens_distribution: torch.Tensor,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Compute all sets' send/recv splits at once.

        Utilizes CPU idle time (e.g., during H2D DMA) to precompute all splits,
        avoiding Forward Main Loop stalls.

        Args:
            num_sets: Total number of sets
            global_expert_sets: [ep_size][num_sets][experts]
            global_tokens_distribution: [ep_size, num_global_experts] CPU tensor

        Returns:
            all_send_splits: List[List[int]], each set's send_splits
            all_recv_splits: List[List[int]], each set's recv_splits
        """
        all_send_splits = []
        all_recv_splits = []

        for set_idx in range(num_sets):
            send_splits, recv_splits = self._compute_splits_for_set(
                set_idx, global_expert_sets, global_tokens_distribution
            )
            all_send_splits.append(send_splits)
            all_recv_splits.append(recv_splits)

        return all_send_splits, all_recv_splits

    def _compute_all_expert_indices(
        self,
        routing_map: torch.Tensor,
        num_global_experts: int,
    ) -> Dict[int, torch.Tensor]:
        """Precompute token indices for ALL experts (one-time, avoiding repeated nonzero calls).

        Args:
            routing_map: [num_tokens, num_global_experts] boolean routing map on GPU
            num_global_experts: Total number of global experts

        Returns:
            expert_indices_map: Dict mapping expert_id -> tensor of token indices
        """
        expert_indices_map = {}
        for exp_id in range(num_global_experts):
            expert_mask = routing_map[:, exp_id]
            indices = expert_mask.nonzero(as_tuple=True)[0]
            # Detach to break reference to routing_map's autograd graph
            expert_indices_map[exp_id] = indices.detach()
        return expert_indices_map

    def _build_send_buffer_for_set(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        set_idx: int,
        global_expert_sets: List[List[List[int]]],
        global_tokens_distribution: torch.Tensor,
        precomputed_send_splits: Optional[List[int]] = None,
        precomputed_expert_indices: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build send buffer for a specific set_idx.

        Extracts and packs tokens (and probs) destined for experts in
        global_expert_sets[ep_rank][set_idx] for each destination rank.

        Args:
            hidden_states: [num_tokens, hidden_size] input tokens
            routing_map: [num_tokens, num_experts] boolean routing map
            probs: [num_tokens, num_experts] routing probabilities
            set_idx: Current set index
            global_expert_sets: [ep_size][num_sets][experts] - complete schedule
            global_tokens_distribution: [ep_size, num_global_experts] token counts
            precomputed_send_splits: Optional precomputed send_splits
            precomputed_expert_indices: Optional dict mapping expert_id -> token indices

        Returns:
            set_send_buffer: [total_send, hidden_size] packed hidden states
            set_send_probs: [total_send] packed probabilities
            set_send_reverse_indices: [total_send] original token indices
            set_send_expert_ids: [total_send] expert_id for each entry
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_size = hidden_states.size(1)

        # Use precomputed splits if provided, otherwise compute
        if precomputed_send_splits is not None:
            send_splits = precomputed_send_splits
        else:
            send_splits, _ = self._compute_splits_for_set(
                set_idx, global_expert_sets, global_tokens_distribution
            )
        total_send = sum(send_splits)

        if total_send == 0:
            return (
                torch.empty(0, hidden_size, dtype=dtype, device=device),
                torch.empty(0, dtype=dtype, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )

        # Allocate output buffers
        set_send_buffer = torch.empty(total_send, hidden_size, dtype=dtype, device=device)
        set_send_probs = torch.empty(total_send, dtype=dtype, device=device)
        set_send_reverse_indices = torch.empty(total_send, dtype=torch.long, device=device)
        set_send_expert_ids = torch.empty(total_send, dtype=torch.long, device=device)

        # Pack tokens per destination rank, iterating through each expert
        current_offset = 0
        for dest_rank in range(self.ep_size):
            dest_experts = global_expert_sets[dest_rank][set_idx]

            for exp_id in dest_experts:
                # Use precomputed indices if available, otherwise call nonzero
                if precomputed_expert_indices is not None and exp_id in precomputed_expert_indices:
                    expert_indices = precomputed_expert_indices[exp_id]
                else:
                    expert_mask = routing_map[:, exp_id]
                    expert_indices = expert_mask.nonzero(as_tuple=True)[0]
                num_tokens_for_expert = len(expert_indices)

                if num_tokens_for_expert > 0:
                    set_send_buffer[current_offset:current_offset + num_tokens_for_expert] = hidden_states[expert_indices]
                    set_send_probs[current_offset:current_offset + num_tokens_for_expert] = probs[expert_indices, exp_id]
                    set_send_reverse_indices[current_offset:current_offset + num_tokens_for_expert] = expert_indices
                    set_send_expert_ids[current_offset:current_offset + num_tokens_for_expert] = exp_id

                current_offset += num_tokens_for_expert

        return set_send_buffer, set_send_probs, set_send_reverse_indices, set_send_expert_ids

    def _prefetch_and_compute_splits_async_quantized(
        self,
        expert_ids: List[int],
        buffer_idx: int,
        num_sets: int,
        global_expert_sets: List[List[List[int]]],
        global_tokens_distribution: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[List[int]], List[List[int]]]:
        """Prefetch weights AND compute all splits, overlapping H2D DMA with CPU work.

        Args:
            expert_ids: List of expert IDs to load
            buffer_idx: Buffer index (0 or 1)
            num_sets: Total number of sets
            global_expert_sets: [ep_size][num_sets][experts]
            global_tokens_distribution: [ep_size, num_global_experts] CPU tensor

        Returns:
            w1_bf16_gpu, w2_bf16_gpu: Views into BF16 GPU workspace (for precision=16)
            w1_quant_gpu, w2_quant_gpu: Views into uint8 GPU workspace (for precision=8/4)
            precisions: List of precision values
            all_send_splits, all_recv_splits: Precomputed splits for all sets
        """
        num_experts = len(expert_ids)
        w1_bf16_gpu = self._w1_bf16_gpu_workspace[buffer_idx, :num_experts]
        w2_bf16_gpu = self._w2_bf16_gpu_workspace[buffer_idx, :num_experts]
        w1_quant_gpu = self._w1_quant_gpu_workspace[buffer_idx, :num_experts]
        w2_quant_gpu = self._w2_quant_gpu_workspace[buffer_idx, :num_experts]
        delta_w1_gpu = self._delta_w1_gpu_workspace[buffer_idx, :num_experts]
        z_w1_gpu = self._z_w1_gpu_workspace[buffer_idx, :num_experts]
        delta_w2_gpu = self._delta_w2_gpu_workspace[buffer_idx, :num_experts]
        z_w2_gpu = self._z_w2_gpu_workspace[buffer_idx, :num_experts]

        all_send_splits = []
        all_recv_splits = []

        with torch.cuda.stream(self._load_stream):
            # Wait for compute completion
            self._compute_events[buffer_idx].wait(self._load_stream)

            # Wait for CPU optimizer update
            if self._cpu_update_event is not None:
                self._cpu_update_event.wait(self._load_stream)

            if self._quant_optimizer is not None:
                # Start H2D copy with separate bf16 and quant buffers
                precisions = self._quant_optimizer.prefetch_expert_data(
                    self.layer_id,
                    expert_ids,
                    w1_bf16_gpu,
                    w2_bf16_gpu,
                    w1_quant_gpu,
                    w2_quant_gpu,
                    delta_w1_gpu,
                    z_w1_gpu,
                    delta_w2_gpu,
                    z_w2_gpu,
                    self._load_stream,
                )
                # Store precisions and global expert IDs
                self._current_precisions[buffer_idx] = precisions
                self._current_global_expert_ids[buffer_idx] = [
                    _get_encode_global_expert_id()(self.layer_id, exp_id, self.num_global_experts)
                    for exp_id in expert_ids
                ]

                # Utilize CPU idle time: compute all splits during DMA transfer
                all_send_splits, all_recv_splits = self._compute_all_splits(
                    num_sets, global_expert_sets, global_tokens_distribution
                )
            else:
                raise RuntimeError("QuantizedDispatcher requires optimizer to be set")

        return w1_bf16_gpu, w2_bf16_gpu, w1_quant_gpu, w2_quant_gpu, precisions, all_send_splits, all_recv_splits

    def _prefetch_quantized_weights_async(
        self,
        expert_ids: List[int],
        buffer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Prefetch expert data from optimizer to GPU.

        Calls optimizer.prefetch_expert_data() which handles:
        - BF16: Direct transfer of main weight to bf16 buffer
        - INT8/INT4: Transfer of quant_w (uint8) + delta/z (float32) to respective buffers

        Args:
            expert_ids: List of local expert IDs to load (within this layer)
            buffer_idx: Buffer index (0 or 1) for double buffering

        Returns:
            Tuple of (w1_bf16_gpu, w2_bf16_gpu, w1_quant_gpu, w2_quant_gpu, precisions)
        """
        num_experts = len(expert_ids)
        w1_bf16_gpu = self._w1_bf16_gpu_workspace[buffer_idx, :num_experts]
        w2_bf16_gpu = self._w2_bf16_gpu_workspace[buffer_idx, :num_experts]
        w1_quant_gpu = self._w1_quant_gpu_workspace[buffer_idx, :num_experts]
        w2_quant_gpu = self._w2_quant_gpu_workspace[buffer_idx, :num_experts]
        delta_w1_gpu = self._delta_w1_gpu_workspace[buffer_idx, :num_experts]
        z_w1_gpu = self._z_w1_gpu_workspace[buffer_idx, :num_experts]
        delta_w2_gpu = self._delta_w2_gpu_workspace[buffer_idx, :num_experts]
        z_w2_gpu = self._z_w2_gpu_workspace[buffer_idx, :num_experts]

        with torch.cuda.stream(self._load_stream):
            # Wait for compute completion
            self._compute_events[buffer_idx].wait(self._load_stream)

            # Wait for CPU optimizer update
            if self._cpu_update_event is not None:
                self._cpu_update_event.wait(self._load_stream)

            if self._quant_optimizer is not None:
                # Use optimizer's unified prefetch interface with separate buffers
                precisions = self._quant_optimizer.prefetch_expert_data(
                    self.layer_id,
                    expert_ids,
                    w1_bf16_gpu,
                    w2_bf16_gpu,
                    w1_quant_gpu,
                    w2_quant_gpu,
                    delta_w1_gpu,
                    z_w1_gpu,
                    delta_w2_gpu,
                    z_w2_gpu,
                    self._load_stream,
                )
                # Store precisions and global expert IDs
                self._current_precisions[buffer_idx] = precisions
                self._current_global_expert_ids[buffer_idx] = [
                    _get_encode_global_expert_id()(self.layer_id, exp_id, self.num_global_experts)
                    for exp_id in expert_ids
                ]
            else:
                raise RuntimeError("QuantizedDispatcher requires optimizer to be set")

        return w1_bf16_gpu, w2_bf16_gpu, w1_quant_gpu, w2_quant_gpu, precisions

    def _dequantize_weights_for_compute(
        self,
        expert_ids: List[int],
        buffer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize weights on GPU for computation.

        For INT8/INT4: Read quant_w from uint8 buffer and dequantize using delta/z
        For BF16: Read directly from bf16 buffer, no dequantization needed

        Args:
            expert_ids: List of local expert IDs
            buffer_idx: Buffer index (0 or 1)

        Returns:
            Tuple of (w1_bf16_gpu, w2_bf16_gpu) ready for GEMM
        """
        precisions = self._current_precisions.get(buffer_idx, [])
        if not precisions:
            raise RuntimeError("No precisions set for buffer_idx")

        num_experts = len(expert_ids)

        # Create output tensors for dequantized weights (BF16 for GEMM)
        w1_bf16 = torch.empty(
            num_experts, self.hidden_size, self.fc1_out_features,
            dtype=torch.bfloat16, device=self._w1_bf16_gpu_workspace.device
        )
        w2_bf16 = torch.empty(
            num_experts, self.ffn_hidden_size, self.hidden_size,
            dtype=torch.bfloat16, device=self._w2_bf16_gpu_workspace.device
        )

        group_size = self.quant_group_size

        for i, (local_exp_id, precision) in enumerate(zip(expert_ids, precisions)):
            if precision == 16:
                # BF16: Read directly from bf16 buffer
                w1_bf16_gpu = self._w1_bf16_gpu_workspace[buffer_idx, i]
                w2_bf16_gpu = self._w2_bf16_gpu_workspace[buffer_idx, i]
                w1_bf16[i] = w1_bf16_gpu
                w2_bf16[i] = w2_bf16_gpu
            elif precision == 8:
                # INT8 dequantization: Read from uint8 buffer
                # w = quant * delta + z
                delta_w1 = self._delta_w1_gpu_workspace[buffer_idx, i]
                z_w1 = self._z_w1_gpu_workspace[buffer_idx, i]
                delta_w2 = self._delta_w2_gpu_workspace[buffer_idx, i]
                z_w2 = self._z_w2_gpu_workspace[buffer_idx, i]

                # Read quant_w from uint8 buffer
                quant_w1_uint8 = self._w1_quant_gpu_workspace[buffer_idx, i]
                quant_w2_uint8 = self._w2_quant_gpu_workspace[buffer_idx, i]

                # Dequantize w1: quant_w1 is stored as uint8
                quant_w1_flat = quant_w1_uint8.flatten().float()
                delta_w1_expanded = delta_w1.repeat_interleave(group_size)
                z_w1_expanded = z_w1.repeat_interleave(group_size)
                w1_dequant = quant_w1_flat * delta_w1_expanded + z_w1_expanded
                w1_bf16[i] = w1_dequant.view(self.hidden_size, self.fc1_out_features).to(torch.bfloat16)

                # Dequantize w2
                quant_w2_flat = quant_w2_uint8.flatten().float()
                delta_w2_expanded = delta_w2.repeat_interleave(group_size)
                z_w2_expanded = z_w2.repeat_interleave(group_size)
                w2_dequant = quant_w2_flat * delta_w2_expanded + z_w2_expanded
                w2_bf16[i] = w2_dequant.view(self.ffn_hidden_size, self.hidden_size).to(torch.bfloat16)

                # Store quant params for backward LSQ computation
                global_expert_id = self._current_global_expert_ids[buffer_idx][i]
                self._quant_handlers[global_expert_id] = {
                    'precision': precision,
                    'quant_w1': quant_w1_uint8.clone(),
                    'quant_w2': quant_w2_uint8.clone(),
                    'delta_w1': delta_w1.clone(),
                    'z_w1': z_w1.clone(),
                    'delta_w2': delta_w2.clone(),
                    'z_w2': z_w2.clone(),
                }
            elif precision == 4:
                # INT4 dequantization: Read from uint8 buffer, unpack + dequant
                delta_w1 = self._delta_w1_gpu_workspace[buffer_idx, i]
                z_w1 = self._z_w1_gpu_workspace[buffer_idx, i]
                delta_w2 = self._delta_w2_gpu_workspace[buffer_idx, i]
                z_w2 = self._z_w2_gpu_workspace[buffer_idx, i]

                # Read packed quant_w from uint8 buffer (only half numel for INT4)
                w1_packed_size = self.hidden_size * self.fc1_out_features // 2
                w2_packed_size = self.ffn_hidden_size * self.hidden_size // 2
                quant_w1_uint8 = self._w1_quant_gpu_workspace[buffer_idx, i].flatten()[:w1_packed_size]
                quant_w2_uint8 = self._w2_quant_gpu_workspace[buffer_idx, i].flatten()[:w2_packed_size]

                # Unpack INT4 w1: each uint8 contains 2 INT4 values
                high_nibbles = (quant_w1_uint8 >> 4) & 0x0F
                low_nibbles = quant_w1_uint8 & 0x0F
                unpacked_w1 = torch.empty(
                    quant_w1_uint8.numel() * 2,
                    dtype=torch.float32, device=quant_w1_uint8.device
                )
                unpacked_w1[0::2] = high_nibbles.float()
                unpacked_w1[1::2] = low_nibbles.float()

                # Dequantize w1
                w1_numel = self.hidden_size * self.fc1_out_features
                w1_dequant = unpacked_w1[:w1_numel] * delta_w1.repeat_interleave(group_size)[:w1_numel] + \
                             z_w1.repeat_interleave(group_size)[:w1_numel]
                w1_bf16[i] = w1_dequant.view(self.hidden_size, self.fc1_out_features).to(torch.bfloat16)

                # Unpack INT4 w2
                high_nibbles_w2 = (quant_w2_uint8 >> 4) & 0x0F
                low_nibbles_w2 = quant_w2_uint8 & 0x0F
                unpacked_w2 = torch.empty(
                    quant_w2_uint8.numel() * 2,
                    dtype=torch.float32, device=quant_w2_uint8.device
                )
                unpacked_w2[0::2] = high_nibbles_w2.float()
                unpacked_w2[1::2] = low_nibbles_w2.float()

                # Dequantize w2
                w2_numel = self.ffn_hidden_size * self.hidden_size
                w2_dequant = unpacked_w2[:w2_numel] * delta_w2.repeat_interleave(group_size)[:w2_numel] + \
                             z_w2.repeat_interleave(group_size)[:w2_numel]
                w2_bf16[i] = w2_dequant.view(self.ffn_hidden_size, self.hidden_size).to(torch.bfloat16)

                # Store quant params for backward LSQ computation
                global_expert_id = self._current_global_expert_ids[buffer_idx][i]
                self._quant_handlers[global_expert_id] = {
                    'precision': precision,
                    'quant_w1': quant_w1_uint8.clone(),
                    'quant_w2': quant_w2_uint8.clone(),
                    'delta_w1': delta_w1.clone(),
                    'z_w1': z_w1.clone(),
                    'delta_w2': delta_w2.clone(),
                    'z_w2': z_w2.clone(),
                }

        return w1_bf16, w2_bf16

    def _compute_lsq_updates(
        self,
        global_expert_id: int,
        grad_w1: torch.Tensor,
        grad_w2: torch.Tensor,
        buffer_idx: int,
        expert_idx: int,
    ):
        """Compute LSQ delta/z updates from gradients.

        LSQ gradient formulas:
        - delta_grad = sum(grad * (quant - z/delta)) per group
        - z_grad = sum(grad) per group

        Update:
        - delta_new = delta - lr * delta_grad
        - z_new = z - lr * z_grad

        Args:
            global_expert_id: Global expert ID
            grad_w1: Gradient for weight1 (BF16)
            grad_w2: Gradient for weight2 (BF16)
            buffer_idx: Buffer index
            expert_idx: Index within the buffer
        """
        handler = self._quant_handlers.get(global_expert_id)
        if handler is None or handler['precision'] == 16:
            # BF16: No LSQ update needed
            return

        precision = handler['precision']
        quant_w1 = handler['quant_w1']
        quant_w2 = handler['quant_w2']
        delta_w1 = handler['delta_w1']
        z_w1 = handler['z_w1']
        delta_w2 = handler['delta_w2']
        z_w2 = handler['z_w2']

        group_size = self.quant_group_size

        # Flatten gradients to float32
        grad_w1_flat = grad_w1.flatten().float()
        grad_w2_flat = grad_w2.flatten().float()

        num_groups_w1 = delta_w1.numel()
        num_groups_w2 = delta_w2.numel()

        # Reshape gradients to groups
        grad_w1_groups = grad_w1_flat.view(num_groups_w1, group_size)
        grad_w2_groups = grad_w2_flat.view(num_groups_w2, group_size)

        if precision == 8:
            # INT8: quant values directly available
            quant_w1_groups = quant_w1.float().view(num_groups_w1, group_size)
            quant_w2_groups = quant_w2.float().view(num_groups_w2, group_size)

            delta_grad_w1 = (grad_w1_groups * quant_w1_groups).sum(dim=1)
            z_grad_w1 = grad_w1_groups.sum(dim=1)
            delta_grad_w2 = (grad_w2_groups * quant_w2_groups).sum(dim=1)
            z_grad_w2 = grad_w2_groups.sum(dim=1)
        else:  # INT4
            # INT4: Need to unpack quant values
            unpacked_w1 = torch.empty(
                quant_w1.numel() * 2, dtype=torch.float32, device=grad_w1.device
            )
            unpacked_w1[0::2] = ((quant_w1 >> 4) & 0x0F).float()
            unpacked_w1[1::2] = (quant_w1 & 0x0F).float()
            quant_w1_groups = unpacked_w1[:grad_w1_flat.numel()].view(num_groups_w1, group_size)

            unpacked_w2 = torch.empty(
                quant_w2.numel() * 2, dtype=torch.float32, device=grad_w2.device
            )
            unpacked_w2[0::2] = ((quant_w2 >> 4) & 0x0F).float()
            unpacked_w2[1::2] = (quant_w2 & 0x0F).float()
            quant_w2_groups = unpacked_w2[:grad_w2_flat.numel()].view(num_groups_w2, group_size)

            delta_grad_w1 = (grad_w1_groups * quant_w1_groups).sum(dim=1)
            z_grad_w1 = grad_w1_groups.sum(dim=1)
            delta_grad_w2 = (grad_w2_groups * quant_w2_groups).sum(dim=1)
            z_grad_w2 = grad_w2_groups.sum(dim=1)

        # Update delta/z
        delta_w1_new = (delta_w1 - self.lr_quant * delta_grad_w1).clamp(min=1e-6)
        z_w1_new = z_w1 - self.lr_quant * z_grad_w1
        delta_w2_new = (delta_w2 - self.lr_quant * delta_grad_w2).clamp(min=1e-6)
        z_w2_new = z_w2 - self.lr_quant * z_grad_w2

        # Update delta/z (keep on GPU, will be async offloaded later)
        self._delta_z_updates[global_expert_id] = (
            delta_w1_new,  # GPU tensor
            z_w1_new,
            delta_w2_new,
            z_w2_new,
        )

    def _offload_grads_to_cpu_async(
        self,
        set_idx: int,
        expert_ids: List[int],
        grad_w1: torch.Tensor,  # GPU tensor [num_experts, ...]
        grad_w2: torch.Tensor,  # GPU tensor [num_experts, ...]
    ):
        """Async offload gradients + LSQ delta/z updates to CPU on _grad_offload_stream.

        Follows FusedDispatcherCacheGroupedMLP pattern:
        1. Wait for gradient computation completion (_grad_ready_events[set_idx])
        2. Async copy grad_w1/grad_w2 to CPU pinned buffer
        3. Async copy delta/z updates to CPU (non_blocking=True)
        4. record_stream to keep GPU memory in use

        Args:
            set_idx: Set index for selecting _grad_ready_events
            expert_ids: List of local expert IDs within this layer
            grad_w1: GPU gradient tensor for weight1 [num_experts, hidden, fc1_out]
            grad_w2: GPU gradient tensor for weight2 [num_experts, ffn_hidden, hidden]
        """
        if self._quant_optimizer is None:
            return

        with torch.cuda.stream(self._grad_offload_stream):
            # Wait for THIS SET's gradient computation to complete
            self._grad_ready_events[set_idx].wait(self._grad_offload_stream)

            nvtx.range_push(f"SET{set_idx}:GRAD_OFFLOAD")
            for i, local_exp_id in enumerate(expert_ids):
                global_expert_id = _get_encode_global_expert_id()(
                    self.layer_id, local_exp_id, self.num_global_experts
                )

                # Flatten gradients to float32
                grad_w1_flat = grad_w1[i].flatten().float()
                grad_w2_flat = grad_w2[i].flatten().float()

                # Async copy gradients to CPU pinned buffer
                self._quant_optimizer.quant_pool.grad_w1[global_expert_id].copy_(
                    grad_w1_flat, non_blocking=True
                )
                self._quant_optimizer.quant_pool.grad_w2[global_expert_id].copy_(
                    grad_w2_flat, non_blocking=True
                )

                # Async offload LSQ delta/z updates (if computed)
                if global_expert_id in self._delta_z_updates:
                    delta_w1_new, z_w1_new, delta_w2_new, z_w2_new = self._delta_z_updates[global_expert_id]
                    # Async copy to CPU
                    self._delta_z_updates[global_expert_id] = (
                        delta_w1_new.to('cpu', non_blocking=True),
                        z_w1_new.to('cpu', non_blocking=True),
                        delta_w2_new.to('cpu', non_blocking=True),
                        z_w2_new.to('cpu', non_blocking=True),
                    )
            nvtx.range_pop()

            # CRITICAL: record_stream AFTER copy_ operations are launched
            grad_w1.record_stream(self._grad_offload_stream)
            grad_w2.record_stream(self._grad_offload_stream)


class QuantizedDispatcherFunction(torch.autograd.Function):
    """Custom autograd function with double-buffered pipeline for QuantizedDispatcher.

    Forward flow:
    1. Phase 0: Exchange global schedule + token distribution
    2. Prologue: Prefetch Set 0 weights + compute all splits + all indices
    3. Main Loop: Double-buffered DISPATCH -> GEMM -> COMBINE -> SCATTER
    4. Epilogue: Sync streams

    Backward flow (reverse order):
    1. Prologue: Prefetch last set weights + REV_COMBINE
    2. Main Loop: Double-buffered recompute + gradients + REV_DISPATCH
    3. Epilogue: Send LSQ updates + sync streams
    """

    @staticmethod
    def forward(
        ctx,
        self: QuantizedDispatcherCacheGroupedMLP,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        expert_sets: List[List[int]],
    ):
        """Forward pass with Prologue + Main Loop pipeline for proper overlap."""
        ctx.self = self
        ctx.expert_sets = expert_sets
        ctx.num_sets = len(expert_sets)

        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_size = hidden_states.shape[-1]
        num_tokens = hidden_states.size(0)
        ctx.hidden_size = hidden_size
        ctx.num_tokens = num_tokens

        # Ensure probs has the same dtype as hidden_states
        if probs.dtype != dtype:
            probs = probs.to(dtype)

        # ========== Phase 0: Exchange Global Schedule + Token Distribution ==========
        global_tokens_distribution, global_expert_sets = self._phase0_exchange_metadata(
            routing_map, expert_sets
        )
        ctx.global_tokens_distribution = global_tokens_distribution
        ctx.global_expert_sets = global_expert_sets

        # ========== Initialize events BEFORE loop ==========
        for i in range(2):
            self._compute_events[i].record(torch.cuda.current_stream())
            self._comm_events[i].record(self._comm_stream)
            self._prep_done_events[i].record(torch.cuda.current_stream())

        # ========== Initialize output tensor ==========
        output = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)

        # ========== Initialize data lists ==========
        dispatched_probs_list = []
        reverse_indices_list = []
        expert_ids_per_set = []
        send_splits_list = []
        recv_splits_list = []
        tokens_per_expert_per_set = []

        current_buffer = 0
        next_buffer = 1

        nvtx.range_push(f"QuantizedDispatcher[L{self.layer_id}]::forward")

        # Initialize precomputed data
        all_send_splits = []
        all_recv_splits = []
        all_expert_indices = {}

        # ==================== Prologue: Prepare Set 0 ====================
        if len(expert_sets) > 0:
            set_0_experts = global_expert_sets[self.ep_rank][0]
            num_sets = len(expert_sets)

            # OPTIMIZATION: Prefetch Set 0 weights + compute ALL splits (parallel!)
            _, _, _, _, _, all_send_splits, all_recv_splits = self._prefetch_and_compute_splits_async_quantized(
                set_0_experts, current_buffer, num_sets,
                global_expert_sets, global_tokens_distribution
            )

            # CRITICAL: Wait for _load_stream to complete before running GPU kernels
            # _prefetch runs on _load_stream and returns immediately
            # We must sync before any GPU operations on main stream to avoid CUDA errors
            # torch.cuda.current_stream().wait_stream(self._load_stream)

            # OPTIMIZATION: Precompute ALL expert indices (GPU kernels on compute stream)
            all_expert_indices = self._compute_all_expert_indices(
                routing_map, self.num_global_experts
            )

            # Store all precomputed splits
            send_splits_list.extend(all_send_splits)
            recv_splits_list.extend(all_recv_splits)

            # Use precomputed splits for Set 0
            send_splits_0 = all_send_splits[0]
            recv_splits_0 = all_recv_splits[0]
            total_send_0 = sum(send_splits_0)
            total_recv_0 = sum(recv_splits_0)

            # Build send buffer using precomputed splits AND indices
            set_send_buffer_0, set_send_probs_0, set_send_reverse_indices_0, set_send_expert_ids_0 = \
                self._build_send_buffer_for_set(
                    hidden_states, routing_map, probs,
                    0, global_expert_sets, global_tokens_distribution,
                    precomputed_send_splits=send_splits_0,
                    precomputed_expert_indices=all_expert_indices
                )

            # CRITICAL: Record event after data preparation completes
            self._prep_done_events[current_buffer].record(torch.cuda.current_stream())

            # Allocate recv buffers for Set 0
            set_recv_buffer_0 = torch.empty(total_recv_0, hidden_size, dtype=dtype, device=device)
            recv_probs_buffer_0 = torch.empty(total_recv_0, dtype=dtype, device=device)

            # Launch Set 0 Dispatch on _comm_stream
            if self.ep_size > 1:
                with torch.cuda.stream(self._comm_stream):
                    self._prep_done_events[current_buffer].wait(self._comm_stream)
                    nvtx.range_push("SET0:DISPATCH")
                    torch.distributed.all_to_all_single(
                        set_recv_buffer_0, set_send_buffer_0,
                        output_split_sizes=recv_splits_0,
                        input_split_sizes=send_splits_0,
                        group=self.ep_group
                    )
                    torch.distributed.all_to_all_single(
                        recv_probs_buffer_0, set_send_probs_0,
                        output_split_sizes=recv_splits_0,
                        input_split_sizes=send_splits_0,
                        group=self.ep_group
                    )
                    nvtx.range_pop()
                    self._comm_events[current_buffer].record(self._comm_stream)
            else:
                # EP=1: No cross-rank communication
                set_recv_buffer_0 = set_send_buffer_0
                recv_probs_buffer_0 = set_send_probs_0

            # Store Set 0 data as "current" for first loop iteration
            current_send_buffer = set_send_buffer_0
            current_send_probs = set_send_probs_0
            current_send_reverse_indices = set_send_reverse_indices_0
            current_send_expert_ids = set_send_expert_ids_0
            current_recv_buffer = set_recv_buffer_0
            current_recv_probs = recv_probs_buffer_0
            current_send_splits = send_splits_0
            current_recv_splits = recv_splits_0
            current_total_send = total_send_0
            current_total_recv = total_recv_0

        # ==================== Main Loop ====================
        for set_idx in range(len(expert_sets)):
            local_experts = global_expert_sets[self.ep_rank][set_idx]

            # Wait for current buffer to be ready (load + comm from prologue/prev iteration)
            torch.cuda.current_stream().wait_stream(self._load_stream)
            if self.ep_size > 1:
                torch.cuda.current_stream().wait_stream(self._comm_stream)

            # ==================== ASYNC: Prepare Next Set (N+1) ====================
            next_set_idx = set_idx + 1
            if next_set_idx < len(expert_sets):
                next_experts = global_expert_sets[self.ep_rank][next_set_idx]

                # 1. Prefetch N+1 weights to next_buffer
                _, _, _, _, precisions = self._prefetch_quantized_weights_async(
                    next_experts, next_buffer
                )

                # 2. Use precomputed splits and indices
                send_splits_next = all_send_splits[next_set_idx]
                recv_splits_next = all_recv_splits[next_set_idx]
                total_send_next = sum(send_splits_next)
                total_recv_next = sum(recv_splits_next)

                set_send_buffer_next, set_send_probs_next, set_send_reverse_indices_next, set_send_expert_ids_next = \
                    self._build_send_buffer_for_set(
                        hidden_states, routing_map, probs,
                        next_set_idx, global_expert_sets, global_tokens_distribution,
                        precomputed_send_splits=send_splits_next,
                        precomputed_expert_indices=all_expert_indices
                    )

                # CRITICAL: Record event after data preparation
                self._prep_done_events[next_buffer].record(torch.cuda.current_stream())

                # Allocate recv buffers for N+1
                set_recv_buffer_next = torch.empty(total_recv_next, hidden_size, dtype=dtype, device=device)
                recv_probs_buffer_next = torch.empty(total_recv_next, dtype=dtype, device=device)

                # 3. Launch N+1 Dispatch on _comm_stream
                if self.ep_size > 1:
                    with torch.cuda.stream(self._comm_stream):
                        # Wait for next_buffer to be free
                        self._scatter_done_events[next_buffer].wait(self._comm_stream)
                        # Wait for data preparation to complete
                        self._prep_done_events[next_buffer].wait(self._comm_stream)
                        nvtx.range_push(f"SET{next_set_idx}:DISPATCH")
                        torch.distributed.all_to_all_single(
                            set_recv_buffer_next, set_send_buffer_next,
                            output_split_sizes=recv_splits_next,
                            input_split_sizes=send_splits_next,
                            group=self.ep_group
                        )
                        torch.distributed.all_to_all_single(
                            recv_probs_buffer_next, set_send_probs_next,
                            output_split_sizes=recv_splits_next,
                            input_split_sizes=send_splits_next,
                            group=self.ep_group
                        )
                        nvtx.range_pop()
                        self._comm_events[next_buffer].record(self._comm_stream)
                else:
                    set_recv_buffer_next = set_send_buffer_next
                    recv_probs_buffer_next = set_send_probs_next

            # ==================== Compute Current Set (N) ====================
            num_local_experts = len(local_experts)

            # Dequantize weights
            w1_bf16, w2_bf16 = self._dequantize_weights_for_compute(
                local_experts, current_buffer
            )

            # Compute tokens_per_expert for GEMM
            tokens_per_expert_list = []
            for exp_id in local_experts:
                global_count = sum(
                    global_tokens_distribution[src_rank][exp_id].item()
                    for src_rank in range(self.ep_size)
                )
                tokens_per_expert_list.append(global_count)
            tokens_per_expert = torch.tensor(tokens_per_expert_list, dtype=torch.long)
            tokens_per_expert_per_set.append(tokens_per_expert)

            # ==================== REPACK: Sort tokens by expert ====================
            if current_total_recv > 0 and num_local_experts > 0 and self.ep_size > 1:
                nvtx.range_push(f"SET{set_idx}:REPACK_PRE")
                repacked_buffer = torch.empty_like(current_recv_buffer)
                repacked_probs = torch.empty_like(current_recv_probs)

                src_offset = 0
                expert_offsets = [0] + list(torch.cumsum(tokens_per_expert, 0)[:-1])

                for src_rank in range(self.ep_size):
                    for i, exp_id in enumerate(local_experts):
                        count = global_tokens_distribution[src_rank][exp_id].item()
                        if count > 0:
                            repacked_buffer[expert_offsets[i]:expert_offsets[i] + count] = \
                                current_recv_buffer[src_offset:src_offset + count]
                            repacked_probs[expert_offsets[i]:expert_offsets[i] + count] = \
                                current_recv_probs[src_offset:src_offset + count]
                            expert_offsets[i] += count
                            src_offset += count

                current_recv_buffer = repacked_buffer
                current_recv_probs = repacked_probs
                nvtx.range_pop()

            # ==================== Step 3: GEMM ====================
            if current_total_recv > 0 and num_local_experts > 0:
                nvtx.range_push(f"SET{set_idx}:GEMM")

                fc1_output = gg.ops.gmm(
                    current_recv_buffer, w1_bf16, tokens_per_expert, trans_b=False
                )
                # Ensure fc1_output has correct dtype
                if fc1_output.dtype != dtype:
                    fc1_output = fc1_output.to(dtype)

                intermediate = self.activation_func(fc1_output) * current_recv_probs.unsqueeze(-1)
                # Ensure intermediate has correct dtype
                if intermediate.dtype != dtype:
                    intermediate = intermediate.to(dtype)

                fc2_output = gg.ops.gmm(
                    intermediate, w2_bf16, tokens_per_expert, trans_b=False
                )
                # Ensure fc2_output has correct dtype
                if fc2_output.dtype != dtype:
                    fc2_output = fc2_output.to(dtype)
                nvtx.range_pop()
            else:
                fc2_output = torch.empty(0, hidden_size, dtype=dtype, device=device)

            # ==================== Step 4: REPACK POST-GEMM ====================
            if current_total_recv > 0 and num_local_experts > 0 and self.ep_size > 1:
                nvtx.range_push(f"SET{set_idx}:REPACK_POST")
                fc2_by_source = torch.empty_like(fc2_output)

                src_offset = 0
                expert_offsets = [0] + list(torch.cumsum(tokens_per_expert, 0)[:-1])

                for src_rank in range(self.ep_size):
                    for i, exp_id in enumerate(local_experts):
                        count = global_tokens_distribution[src_rank][exp_id].item()
                        if count > 0:
                            fc2_by_source[src_offset:src_offset + count] = \
                                fc2_output[expert_offsets[i]:expert_offsets[i] + count]
                            expert_offsets[i] += count
                            src_offset += count

                fc2_output = fc2_by_source
                nvtx.range_pop()

            # Record GEMM+REPACK completion for COMBINE synchronization
            self._compute_events[current_buffer].record(torch.cuda.current_stream())

            # ==================== Step 5: COMBINE ====================
            set_combine_buffer = torch.empty(current_total_send, hidden_size, dtype=dtype, device=device)
            if self.ep_size > 1:
                with torch.cuda.stream(self._comm_stream):
                    self._compute_events[current_buffer].wait(self._comm_stream)
                    nvtx.range_push(f"SET{set_idx}:COMBINE")
                    torch.distributed.all_to_all_single(
                        set_combine_buffer, fc2_output,
                        output_split_sizes=current_send_splits,
                        input_split_sizes=current_recv_splits,
                        group=self.ep_group
                    )
                    nvtx.range_pop()
                    self._comm_events[current_buffer].record(self._comm_stream)
                torch.cuda.current_stream().wait_stream(self._comm_stream)
            else:
                set_combine_buffer = fc2_output

            # ==================== Step 6: SCATTER ====================
            if current_send_reverse_indices.numel() > 0 and set_combine_buffer.numel() > 0:
                nvtx.range_push(f"SET{set_idx}:SCATTER")
                output.index_add_(0, current_send_reverse_indices, set_combine_buffer)
                nvtx.range_pop()

            # Save for backward
            dispatched_probs_list.append(current_recv_probs.detach())
            reverse_indices_list.append(current_send_reverse_indices.detach())
            expert_ids_per_set.append(current_send_expert_ids.detach())

            # Record scatter completion for next iteration's DISPATCH
            self._scatter_done_events[current_buffer].record(torch.cuda.current_stream())

            # ==================== Swap Buffers for Next Iteration ====================
            current_buffer, next_buffer = next_buffer, current_buffer

            # Update current buffers to point to next set's precomputed data
            if next_set_idx < len(expert_sets):
                current_send_buffer = set_send_buffer_next
                current_send_probs = set_send_probs_next
                current_send_reverse_indices = set_send_reverse_indices_next
                current_send_expert_ids = set_send_expert_ids_next
                current_recv_buffer = set_recv_buffer_next
                current_recv_probs = recv_probs_buffer_next
                current_send_splits = send_splits_next
                current_recv_splits = recv_splits_next
                current_total_send = total_send_next
                current_total_recv = total_recv_next

        nvtx.range_pop()

        # ==================== CRITICAL: Sync Load Stream ====================
        self._load_stream.synchronize()

        # Save hidden_states for backward recomputation
        ctx.save_for_backward(hidden_states.detach(), routing_map.detach(), *dispatched_probs_list, *reverse_indices_list, *expert_ids_per_set)

        return output, None

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_bias):
        """Backward pass with Prologue + Main Loop pipeline for proper overlap."""
        del grad_bias

        self: QuantizedDispatcherCacheGroupedMLP = ctx.self
        expert_sets: List[List[int]] = ctx.expert_sets
        global_expert_sets: List[List[List[int]]] = ctx.global_expert_sets
        global_tokens_distribution = ctx.global_tokens_distribution
        hidden_size = ctx.hidden_size
        num_tokens = ctx.num_tokens
        num_sets = ctx.num_sets

        device = grad_output.device
        dtype = grad_output.dtype

        # Retrieve saved tensors - structure: [hidden_states, routing_map, dispatched_probs..., reverse_indices..., expert_ids...]
        hidden_states = ctx.saved_tensors[0]
        routing_map = ctx.saved_tensors[1]
        dispatched_probs = ctx.saved_tensors[2:2+num_sets]
        reverse_indices_list = ctx.saved_tensors[2+num_sets:2+2*num_sets]
        expert_ids_per_set = ctx.saved_tensors[2+2*num_sets:2+3*num_sets]

        # ========== Initialize events BEFORE loop ==========
        for i in range(2):
            self._compute_events[i].record(torch.cuda.current_stream())
            self._comm_events[i].record(self._comm_stream)
            self._scatter_done_events[i].record(torch.cuda.current_stream())
            self._prep_done_events[i].record(torch.cuda.current_stream())

        # Initialize gradient tensors
        grad_input = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
        grad_probs_total = torch.zeros(num_tokens, self.num_global_experts, dtype=dtype, device=device)

        current_buffer = 0
        next_buffer = 1

        nvtx.range_push(f"QuantizedDispatcher[L{self.layer_id}]::backward")

        # Get splits (same as forward)
        all_send_splits, all_recv_splits = self._compute_all_splits(
            num_sets, global_expert_sets, global_tokens_distribution
        )

        # ==================== Prologue: Prepare Last Set (N-1) ====================
        last_set_idx = len(expert_sets) - 1
        if last_set_idx >= 0:
            last_experts = global_expert_sets[self.ep_rank][last_set_idx]

            # Async load last set weights
            self._prefetch_quantized_weights_async(last_experts, current_buffer)

            # Get splits for last set
            current_send_splits = all_send_splits[last_set_idx]
            current_recv_splits = all_recv_splits[last_set_idx]
            current_total_send = sum(current_send_splits)
            current_total_recv = sum(current_recv_splits)

            # Get saved data for last set
            current_dispatched_probs = dispatched_probs[last_set_idx]
            current_reverse_indices = reverse_indices_list[last_set_idx]
            current_expert_ids = expert_ids_per_set[last_set_idx]

            # Compute tokens_per_expert for last set
            tokens_per_expert_list = []
            for exp_id in last_experts:
                global_count = sum(
                    global_tokens_distribution[src_rank][exp_id].item()
                    for src_rank in range(self.ep_size)
                )
                tokens_per_expert_list.append(global_count)
            current_tokens_per_expert = torch.tensor(tokens_per_expert_list, dtype=torch.long)

            # Get grad_output for last set
            current_set_grad_output = grad_output[current_reverse_indices.to(device)] if current_reverse_indices.numel() > 0 else \
                torch.empty(0, hidden_size, dtype=dtype, device=device)

            # CRITICAL: Record event after data preparation
            self._prep_done_events[current_buffer].record(torch.cuda.current_stream())

            # Launch last set's REVERSE COMBINE on _comm_stream
            current_grad_fc2 = torch.empty(current_total_recv, hidden_size, dtype=dtype, device=device)
            if self.ep_size > 1:
                with torch.cuda.stream(self._comm_stream):
                    self._prep_done_events[current_buffer].wait(self._comm_stream)
                    nvtx.range_push(f"SET{last_set_idx}:REV_COMBINE")
                    torch.distributed.all_to_all_single(
                        current_grad_fc2, current_set_grad_output,
                        output_split_sizes=current_recv_splits,
                        input_split_sizes=current_send_splits,
                        group=self.ep_group
                    )
                    nvtx.range_pop()
                    self._comm_events[current_buffer].record(self._comm_stream)
                torch.cuda.current_stream().wait_stream(self._comm_stream)
            else:
                current_grad_fc2 = current_set_grad_output

        # ==================== Main Loop (REVERSE order) ====================
        for set_idx in range(last_set_idx, -1, -1):
            local_experts = global_expert_sets[self.ep_rank][set_idx]

            # Wait for current buffer (load + comm)
            torch.cuda.current_stream().wait_stream(self._load_stream)
            if self.ep_size > 1:
                torch.cuda.current_stream().wait_stream(self._comm_stream)

            # ==================== ASYNC: Prepare Previous Set (set_idx-1) ====================
            prev_set_idx = set_idx - 1
            if prev_set_idx >= 0:
                prev_experts = global_expert_sets[self.ep_rank][prev_set_idx]

                # 1. Prefetch prev set weights to next_buffer
                self._prefetch_quantized_weights_async(prev_experts, next_buffer)

                # 2. Get saved data for prev set
                prev_dispatched_probs = dispatched_probs[prev_set_idx]
                prev_reverse_indices = reverse_indices_list[prev_set_idx]
                prev_expert_ids = expert_ids_per_set[prev_set_idx]
                prev_send_splits = all_send_splits[prev_set_idx]
                prev_recv_splits = all_recv_splits[prev_set_idx]
                prev_total_send = sum(prev_send_splits)
                prev_total_recv = sum(prev_recv_splits)

                # Compute tokens_per_expert for prev set
                prev_tokens_per_expert_list = []
                for exp_id in prev_experts:
                    global_count = sum(
                        global_tokens_distribution[src_rank][exp_id].item()
                        for src_rank in range(self.ep_size)
                    )
                    prev_tokens_per_expert_list.append(global_count)
                prev_tokens_per_expert = torch.tensor(prev_tokens_per_expert_list, dtype=torch.long)

                # 3. Get grad_output for prev set
                prev_set_grad_output = grad_output[prev_reverse_indices.to(device)] if prev_reverse_indices.numel() > 0 else \
                    torch.empty(0, hidden_size, dtype=dtype, device=device)

                # CRITICAL: Record event after data preparation
                self._prep_done_events[next_buffer].record(torch.cuda.current_stream())

                # 4. Launch prev set's REVERSE COMBINE on _comm_stream
                prev_grad_fc2 = torch.empty(prev_total_recv, hidden_size, dtype=dtype, device=device)
                if self.ep_size > 1:
                    with torch.cuda.stream(self._comm_stream):
                        self._scatter_done_events[next_buffer].wait(self._comm_stream)
                        self._prep_done_events[next_buffer].wait(self._comm_stream)
                        nvtx.range_push(f"SET{prev_set_idx}:REV_COMBINE")
                        torch.distributed.all_to_all_single(
                            prev_grad_fc2, prev_set_grad_output,
                            output_split_sizes=prev_recv_splits,
                            input_split_sizes=prev_send_splits,
                            group=self.ep_group
                        )
                        nvtx.range_pop()
                        self._comm_events[next_buffer].record(self._comm_stream)
                    torch.cuda.current_stream().wait_stream(self._comm_stream)
                else:
                    prev_grad_fc2 = prev_set_grad_output

            # ==================== Compute Current Set ====================
            num_local_experts = len(local_experts)

            # Dequantize weights
            w1_bf16, w2_bf16 = self._dequantize_weights_for_compute(
                local_experts, current_buffer
            )

            # Use current saved data
            dispatched_probs_t = current_dispatched_probs
            reverse_indices = current_reverse_indices
            set_expert_ids = current_expert_ids
            tokens_per_expert = current_tokens_per_expert
            grad_fc2 = current_grad_fc2
            total_send = current_total_send
            total_recv = current_total_recv
            send_splits = current_send_splits
            recv_splits = current_recv_splits

            # ==================== REBUILD recv_tokens via all_to_all ====================
            # CRITICAL: In backward, we need the recv-side tokens (expert order) to recompute GEMM
            # The reverse_indices gives us send-side tokens (dest_rank order), not recv-side
            # We need to exchange hidden_states via all_to_all to get recv-side data

            # Build send buffer from hidden_states (same as forward)
            send_tokens = hidden_states[reverse_indices.to(device)] if reverse_indices.numel() > 0 else \
                torch.empty(0, hidden_size, dtype=dtype, device=device)

            # Allocate recv buffer
            recv_tokens = torch.empty(total_recv, hidden_size, dtype=dtype, device=device)

            # Exchange data: send -> recv (same as forward's dispatch)
            # CRITICAL: all ranks MUST participate in all_to_all_single (collective operation)
            # Even if total_recv == 0, we must call it to avoid NCCL deadlock
            if self.ep_size > 1:
                nvtx.range_push(f"SET{set_idx}:BW_DISPATCH")
                torch.distributed.all_to_all_single(
                    recv_tokens, send_tokens,
                    output_split_sizes=recv_splits,
                    input_split_sizes=send_splits,
                    group=self.ep_group
                )
                nvtx.range_pop()
            else:
                recv_tokens = send_tokens

            # ==================== REPACK: Convert recv_tokens to expert order ====================
            if self.ep_size > 1 and total_recv > 0 and num_local_experts > 0:
                nvtx.range_push(f"SET{set_idx}:BW_REPACK_TOKENS")
                repacked_tokens = torch.empty_like(recv_tokens)

                src_offset = 0
                expert_offsets = [0] + list(torch.cumsum(tokens_per_expert, 0)[:-1])

                for src_rank in range(self.ep_size):
                    for i, exp_id in enumerate(local_experts):
                        count = global_tokens_distribution[src_rank][exp_id].item()
                        if count > 0:
                            repacked_tokens[expert_offsets[i]:expert_offsets[i] + count] = \
                                recv_tokens[src_offset:src_offset + count]
                            expert_offsets[i] += count
                            src_offset += count

                recv_tokens = repacked_tokens
                nvtx.range_pop()

            # Now recv_tokens is in expert order, matching forward's GEMM input
            dispatched_tokens = recv_tokens

            # Recompute forward with enable_grad
            if total_recv > 0 and num_local_experts > 0:
                # ==================== REPACK: Convert grad_fc2 from source-rank to expert order ====================
                if self.ep_size > 1:
                    nvtx.range_push(f"SET{set_idx}:BW_REPACK_PRE")
                    grad_fc2_expert_order = torch.empty_like(grad_fc2)
                    src_offset = 0
                    expert_offsets = [0] + list(torch.cumsum(tokens_per_expert, 0)[:-1])

                    for src_rank in range(self.ep_size):
                        for i, exp_id in enumerate(local_experts):
                            count = global_tokens_distribution[src_rank][exp_id].item()
                            if count > 0:
                                grad_fc2_expert_order[expert_offsets[i]:expert_offsets[i] + count] = \
                                    grad_fc2[src_offset:src_offset + count]
                                expert_offsets[i] += count
                                src_offset += count
                    grad_fc2_for_gemm = grad_fc2_expert_order
                    nvtx.range_pop()
                else:
                    grad_fc2_for_gemm = grad_fc2

                nvtx.range_push(f"SET{set_idx}:BW_GEMM")

                with torch.enable_grad():
                    dispatched_tokens_req = dispatched_tokens.detach().requires_grad_(True)
                    w1_req = w1_bf16.detach().requires_grad_(True)
                    w2_req = w2_bf16.detach().requires_grad_(True)
                    probs_req = dispatched_probs_t.detach().requires_grad_(True)

                    fc1 = gg.ops.gmm(dispatched_tokens_req, w1_req, tokens_per_expert, trans_b=False)
                    intermediate = self.activation_func(fc1) * probs_req.unsqueeze(-1)
                    fc2 = gg.ops.gmm(intermediate, w2_req, tokens_per_expert, trans_b=False)

                    grads = torch.autograd.grad(
                        fc2, (dispatched_tokens_req, w1_req, w2_req, probs_req),
                        grad_outputs=grad_fc2_for_gemm,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                nvtx.range_pop()

                grad_input_local = grads[0] if grads[0] is not None else torch.zeros_like(dispatched_tokens)
                grad_w1 = grads[1]
                grad_w2 = grads[2]
                grad_probs = grads[3]

                # ==================== REPACK: Convert gradients from expert to source-rank order ====================
                if self.ep_size > 1:
                    nvtx.range_push(f"SET{set_idx}:BW_REPACK_POST")
                    grad_input_for_dispatch = torch.empty_like(grad_input_local)
                    grad_probs_for_dispatch = torch.empty_like(grad_probs)
                    src_offset = 0
                    expert_offsets = [0] + list(torch.cumsum(tokens_per_expert, 0)[:-1])

                    for src_rank in range(self.ep_size):
                        for i, exp_id in enumerate(local_experts):
                            count = global_tokens_distribution[src_rank][exp_id].item()
                            if count > 0:
                                grad_input_for_dispatch[src_offset:src_offset + count] = \
                                    grad_input_local[expert_offsets[i]:expert_offsets[i] + count]
                                grad_probs_for_dispatch[src_offset:src_offset + count] = \
                                    grad_probs[expert_offsets[i]:expert_offsets[i] + count]
                                expert_offsets[i] += count
                                src_offset += count
                    grad_input_local = grad_input_for_dispatch
                    grad_probs = grad_probs_for_dispatch
                    nvtx.range_pop()

                # Compute LSQ updates for each expert
                if grad_w1 is not None and grad_w2 is not None:
                    self._grad_ready_events[set_idx].record(torch.cuda.current_stream())

                    for i, local_exp_id in enumerate(local_experts):
                        global_expert_id = _get_encode_global_expert_id()(
                            self.layer_id, local_exp_id, self.num_global_experts
                        )
                        self._compute_lsq_updates(
                            global_expert_id,
                            grad_w1[i],
                            grad_w2[i],
                            current_buffer,
                            i,
                        )

                    # Async offload gradients to CPU for optimizer step
                    self._offload_grads_to_cpu_async(
                        set_idx,
                        local_experts,
                        grad_w1,
                        grad_w2,
                    )
            else:
                grad_input_local = torch.empty(0, hidden_size, dtype=dtype, device=device)
                grad_probs = torch.empty(0, dtype=dtype, device=device)

            # ==================== REVERSE DISPATCH ====================
            if total_recv > 0:
                concat_grad_local = torch.cat([
                    grad_input_local,
                    grad_probs.unsqueeze(-1)
                ], dim=-1)
            else:
                concat_grad_local = torch.empty(0, hidden_size + 1, dtype=dtype, device=device)

            concat_grad_remote = torch.empty(total_send, hidden_size + 1, dtype=dtype, device=device)

            # CRITICAL: Record event AFTER concat_grad_local is created
            self._compute_events[current_buffer].record(torch.cuda.current_stream())

            if self.ep_size > 1:
                with torch.cuda.stream(self._comm_stream):
                    self._compute_events[current_buffer].wait(self._comm_stream)
                    nvtx.range_push(f"SET{set_idx}:REV_DISPATCH")
                    torch.distributed.all_to_all_single(
                        concat_grad_remote, concat_grad_local,
                        output_split_sizes=send_splits,
                        input_split_sizes=recv_splits,
                        group=self.ep_group
                    )
                    nvtx.range_pop()
                    self._comm_events[current_buffer].record(self._comm_stream)
                torch.cuda.current_stream().wait_stream(self._comm_stream)
            else:
                concat_grad_remote = concat_grad_local

            # Split combined gradient
            if total_send > 0:
                grad_input_remote = concat_grad_remote[:, :-1]
                grad_probs_remote = concat_grad_remote[:, -1]
            else:
                grad_input_remote = torch.empty(0, hidden_size, dtype=dtype, device=device)
                grad_probs_remote = torch.empty(0, dtype=dtype, device=device)

            # Scatter gradients
            if reverse_indices.numel() > 0 and grad_input_remote.numel() > 0:
                nvtx.range_push(f"SET{set_idx}:BW_SCATTER")
                grad_input.index_add_(0, reverse_indices.to(device), grad_input_remote)
                grad_probs_total.index_put_(
                    (reverse_indices.to(device), set_expert_ids.to(device)),
                    grad_probs_remote,
                    accumulate=True
                )
                nvtx.range_pop()

            # Record scatter completion
            self._scatter_done_events[current_buffer].record(torch.cuda.current_stream())

            # ==================== Swap Buffers ====================
            current_buffer, next_buffer = next_buffer, current_buffer

            # Update current data to prev set's data
            if prev_set_idx >= 0:
                current_dispatched_probs = prev_dispatched_probs
                current_reverse_indices = prev_reverse_indices
                current_expert_ids = prev_expert_ids
                current_tokens_per_expert = prev_tokens_per_expert
                current_send_splits = prev_send_splits
                current_recv_splits = prev_recv_splits
                current_total_send = prev_total_send
                current_total_recv = prev_total_recv
                current_grad_fc2 = prev_grad_fc2

        # ==================== Epilogue: Sync streams ====================
        # if self._grad_offload_stream is not None:
        #     self._grad_offload_stream.synchronize()
        # if self._comm_stream is not None:
        #     self._comm_stream.synchronize()
        # if self._load_stream is not None:
        #     self._load_stream.synchronize()

        nvtx.range_pop()

        # Send LSQ updates to optimizer
        self.send_delta_z_updates_to_optimizer()

        # Clear quant handlers
        self._quant_handlers.clear()

        return None, grad_input, None, grad_probs_total, None