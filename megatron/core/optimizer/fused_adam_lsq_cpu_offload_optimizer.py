"""Fused Adam LSQ CPU Offload Optimizer for MoE models with Dynamic Quantization.

This optimizer uses FusedAdamLSQ for expert weight updates on CPU with dynamic
quantization precision allocation based on expert importance scores.

Key features:
1. Expert weights stored in CPU shared memory (already implemented)
2. Expert gradients offloaded to CPU during backward (already implemented)
3. FusedAdamLSQ for efficient CPU-based optimizer updates with LSQ quantization
4. Dynamic quantization precision: BF16 (top 5%), INT8 (top 30%), INT4 (rest)
5. GPU computes delta/z updates via LSQ learning rules, CPU only stores results

Data Flow:
- Training start: GPU receives BF16 weights → computes quantization → returns quant, delta, z to CPU
- Each step: CPU → GPU: quant_weight + delta/z → GPU dequantizes → compute → returns updated delta/z
- CPU only uses transformation rules for precision upgrade/downgrade (INT4↔INT8)

Usage:
    model = Qwen3MoEModel(config, pg_collection, expert_sets)
    optimizer = FusedAdamLSQCPUOffloadOptimizer(model, lr=1e-4)

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()  # Expert params updated on CPU via FusedAdamLSQ
"""

from typing import Dict, List, Optional, Tuple, Any
import threading
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda import nvtx

from megatron.core import parallel_state

# Import QuantizedDispatcher - required for this optimizer
from megatron.core.transformer.moe.quantized_dispatcher import QuantizedDispatcherCacheGroupedMLP

try:
    from fused_adam_lsq import FusedAdamLSQ
    HAVE_FUSED_ADAM_LSQ = True
except ImportError:
    HAVE_FUSED_ADAM_LSQ = False
    FusedAdamLSQ = None


# ============================================================================
# Global Expert ID Encoding/Decoding Functions
# ============================================================================

def encode_global_expert_id(layer_id: int, local_expert_id: int, num_experts_per_layer: int) -> int:
    """Encode layer_id and local_expert_id into a global expert ID.

    Args:
        layer_id: Layer index (0 to num_layers-1)
        local_expert_id: Expert index within the layer (0 to num_experts_per_layer-1)
        num_experts_per_layer: Number of experts per layer

    Returns:
        Global expert ID: layer_id * num_experts_per_layer + local_expert_id
    """
    return layer_id * num_experts_per_layer + local_expert_id


def decode_global_expert_id(global_expert_id: int, num_experts_per_layer: int) -> Tuple[int, int]:
    """Decode global expert ID into layer_id and local_expert_id.

    Args:
        global_expert_id: Global expert ID
        num_experts_per_layer: Number of experts per layer

    Returns:
        Tuple of (layer_id, local_expert_id)
    """
    layer_id = global_expert_id // num_experts_per_layer
    local_expert_id = global_expert_id % num_experts_per_layer
    return layer_id, local_expert_id


# ============================================================================
# Global Quantization Pool
# ============================================================================

class GlobalQuantizationPool:
    """全局量化存储池 - 所有MoE层共享.

    新设计:
    - delta/z buffer: 所有专家持有, 用float32保存, 不区分INT8/INT4
    - quant weight: 区分INT8/INT4池, 专家动态attach/detach到slot
    - BF16专家不进quant池, 但持有delta/z buffer用于首次step量化

    Memory Layout:
    - Delta/Z: [num_total_experts, num_groups] for all experts (float32)
    - INT8 Quant Pool: [num_int8_slots, numel] for quant_w (uint8)
    - INT4 Quant Pool: [num_int4_slots, numel//2] for quant_w (uint8, packed)
    """

    def __init__(
        self,
        num_layers: int,
        num_experts_per_layer: int,
        weight_shapes: Dict[str, int],
        int8_ratio: float = 0.30,
        int4_ratio: float = 0.65,
        quant_group_size: int = 128,
    ):
        """Initialize global quantization pool.

        Args:
            num_layers: Number of MoE layers
            num_experts_per_layer: Number of experts per layer
            weight_shapes: Dict with 'w1_numel' and 'w2_numel' (elements per expert)
            int8_ratio: Ratio of experts to use INT8 (default 30%)
            int4_ratio: Ratio of experts to use INT4 (default 65%)
            quant_group_size: Quantization group size
        """
        # Calculate total experts and slot counts
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.num_total_experts = num_layers * num_experts_per_layer
        self.num_int8_slots = int(self.num_total_experts * int8_ratio)
        # INT4 slots = remaining experts (确保总和等于num_total_experts)
        # BF16专家不需要slot，使用main_weight
        self.num_int4_slots = self.num_total_experts - self.num_int8_slots

        self.quant_group_size = quant_group_size

        # Calculate group counts
        w1_numel = weight_shapes['w1_numel']
        w2_numel = weight_shapes['w2_numel']
        self.w1_numel = w1_numel
        self.w2_numel = w2_numel
        num_groups_w1 = w1_numel // quant_group_size
        num_groups_w2 = w2_numel // quant_group_size
        self.num_groups_w1 = num_groups_w1
        self.num_groups_w2 = num_groups_w2

        print(f"[DEBUG GlobalQuantizationPool.__init__] num_total_experts={self.num_total_experts}, "
              f"w1_numel={w1_numel}, w2_numel={w2_numel}, "
              f"num_groups_w1={num_groups_w1}, num_groups_w2={num_groups_w2}, "
              f"num_int8_slots={self.num_int8_slots}, num_int4_slots={self.num_int4_slots}")

        # Delta/Z buffers - 所有专家持有, 用于存储GPU计算的delta/z
        # 用float32保存, 不区分INT8/INT4精度
        self.delta_w1 = torch.empty(
            self.num_total_experts, num_groups_w1, dtype=torch.float32, pin_memory=True
        )
        self.z_w1 = torch.empty(
            self.num_total_experts, num_groups_w1, dtype=torch.float32, pin_memory=True
        )
        self.delta_w2 = torch.empty(
            self.num_total_experts, num_groups_w2, dtype=torch.float32, pin_memory=True
        )
        self.z_w2 = torch.empty(
            self.num_total_experts, num_groups_w2, dtype=torch.float32, pin_memory=True
        )

        # Gradient buffers - CPU pinned memory for async D2H
        # 每个专家有自己的slice, 形状与main_weight一致
        self.grad_w1 = torch.empty(
            self.num_total_experts, w1_numel, dtype=torch.float32, pin_memory=True
        )
        self.grad_w2 = torch.empty(
            self.num_total_experts, w2_numel, dtype=torch.float32, pin_memory=True
        )
        self.grad_w1.zero_()
        self.grad_w2.zero_()

        # Pre-allocate INT8 quant pool (只存储quant weight, 不存delta/z)
        self.int8_quant_w1 = torch.empty(
            self.num_int8_slots, w1_numel, dtype=torch.uint8, pin_memory=True
        )
        self.int8_quant_w2 = torch.empty(
            self.num_int8_slots, w2_numel, dtype=torch.uint8, pin_memory=True
        )

        # Pre-allocate INT4 quant pool (quant weight size halved: 2 elements per byte)
        w1_quant_size = w1_numel // 2 + (w1_numel % 2)
        w2_quant_size = w2_numel // 2 + (w2_numel % 2)
        self.int4_quant_w1 = torch.empty(
            self.num_int4_slots, w1_quant_size, dtype=torch.uint8, pin_memory=True
        )
        self.int4_quant_w2 = torch.empty(
            self.num_int4_slots, w2_quant_size, dtype=torch.uint8, pin_memory=True
        )

        # Slot allocation state (只用于quant weight, 不用于delta/z)
        self.int8_free_slots: set = set(range(self.num_int8_slots))
        self.int4_free_slots: set = set(range(self.num_int4_slots))

        # Expert → Slot mapping (global_expert_id -> (precision, slot_idx))
        # 只记录quant slot, delta/z通过global_expert_id直接索引
        self.expert_slot_map: Dict[int, Tuple[int, int]] = {}

        # Track precision for all experts (global_expert_id -> precision)
        self.expert_precision: Dict[int, int] = {}

        # Track which experts have valid quant_w (after first optimizer step)
        self._quant_initialized: set = set()

    def attach_quant_slot(self, global_expert_id: int, precision: int) -> int:
        """Attach an expert to a free quant slot in the specified precision pool.

        只分配quant weight slot, delta/z通过global_expert_id直接索引到统一buffer。

        Args:
            global_expert_id: Global expert ID
            precision: Target precision (8=INT8, 4=INT4)

        Returns:
            Assigned slot index

        Raises:
            RuntimeError: If the specified precision pool has no free slots
        """
        if precision == 8:
            if not self.int8_free_slots:
                raise RuntimeError(
                    "INT8 quant pool exhausted - this should not happen per design"
                )
            slot_idx = min(self.int8_free_slots)
            self.int8_free_slots.remove(slot_idx)
        elif precision == 4:
            if not self.int4_free_slots:
                raise RuntimeError(
                    "INT4 quant pool exhausted - this should not happen per design"
                )
            slot_idx = min(self.int4_free_slots)
            self.int4_free_slots.remove(slot_idx)
        else:
            raise ValueError(f"Invalid precision for quant slot attach: {precision}")

        self.expert_slot_map[global_expert_id] = (precision, slot_idx)
        self.expert_precision[global_expert_id] = precision
        return slot_idx

    def is_quant_initialized(self, global_expert_id: int) -> bool:
        """Check if quant weight has been initialized (after first optimizer step).

        Args:
            global_expert_id: Global expert ID

        Returns:
            True if quant_w is initialized, False otherwise
        """
        return global_expert_id in self._quant_initialized

    def mark_quant_initialized(self, global_expert_id: int):
        """Mark that quant weight has been initialized.

        Args:
            global_expert_id: Global expert ID
        """
        self._quant_initialized.add(global_expert_id)

    def detach_quant_slot(self, global_expert_id: int):
        """Detach an expert from its quant slot, releasing the slot for reuse.

        只释放quant slot, delta/z buffer保留 (用于下次量化)。

        Args:
            global_expert_id: Global expert ID
        """
        if global_expert_id not in self.expert_slot_map:
            return  # Expert not in quant pool (BF16 or not attached)

        precision, slot_idx = self.expert_slot_map[global_expert_id]

        if precision == 8:
            self.int8_free_slots.add(slot_idx)
        elif precision == 4:
            self.int4_free_slots.add(slot_idx)

        del self.expert_slot_map[global_expert_id]
        # Keep expert_precision and delta/z buffer (will be updated on re-attach)

    def get_delta_z(self, global_expert_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get delta/z for an expert from unified buffer.

        所有专家都有delta/z, 通过global_expert_id直接索引。

        Args:
            global_expert_id: Global expert ID

        Returns:
            Tuple of (delta_w1, z_w1, delta_w2, z_w2)
        """
        return (
            self.delta_w1[global_expert_id],
            self.z_w1[global_expert_id],
            self.delta_w2[global_expert_id],
            self.z_w2[global_expert_id],
        )

    def write_delta_z(
        self,
        global_expert_id: int,
        delta_w1: torch.Tensor,
        z_w1: torch.Tensor,
        delta_w2: torch.Tensor,
        z_w2: torch.Tensor,
    ):
        """Write GPU-updated delta/z to unified buffer.

        直接覆盖buffer, 不区分精度。

        Args:
            global_expert_id: Global expert ID
            delta_w1: Updated delta for weight1
            z_w1: Updated z for weight1
            delta_w2: Updated delta for weight2
            z_w2: Updated z for weight2
        """
        self.delta_w1[global_expert_id].copy_(delta_w1)
        self.z_w1[global_expert_id].copy_(z_w1)
        self.delta_w2[global_expert_id].copy_(delta_w2)
        self.z_w2[global_expert_id].copy_(z_w2)

    def get_quant_weight(self, global_expert_id: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get quant weight for an expert from pool slot.

        Args:
            global_expert_id: Global expert ID

        Returns:
            Tuple of (quant_w1, quant_w2)
            Returns (None, None) for BF16 experts or if not attached
        """
        if global_expert_id not in self.expert_slot_map:
            return (None, None)

        precision, slot_idx = self.expert_slot_map[global_expert_id]

        if precision == 8:
            return (
                self.int8_quant_w1[slot_idx],
                self.int8_quant_w2[slot_idx],
            )
        else:  # INT4
            return (
                self.int4_quant_w1[slot_idx],
                self.int4_quant_w2[slot_idx],
            )

    def write_quant_weight(
        self,
        global_expert_id: int,
        quant_w1: Optional[torch.Tensor] = None,
        quant_w2: Optional[torch.Tensor] = None,
    ):
        """Write quantized weight to pool slot.

        Args:
            global_expert_id: Global expert ID
            quant_w1: Quantized weight1
            quant_w2: Quantized weight2
        """
        if global_expert_id not in self.expert_slot_map:
            return  # BF16 expert, no slot to write

        precision, slot_idx = self.expert_slot_map[global_expert_id]

        if precision == 8:
            if quant_w1 is not None:
                self.int8_quant_w1[slot_idx].copy_(quant_w1)
            if quant_w2 is not None:
                self.int8_quant_w2[slot_idx].copy_(quant_w2)
        else:  # INT4
            if quant_w1 is not None:
                self.int4_quant_w1[slot_idx].copy_(quant_w1)
            if quant_w2 is not None:
                self.int4_quant_w2[slot_idx].copy_(quant_w2)

        self._quant_initialized.add(global_expert_id)

    def get_precision(self, global_expert_id: int) -> int:
        """Get current precision for an expert."""
        return self.expert_precision.get(global_expert_id, 16)

    def set_precision(self, global_expert_id: int, precision: int):
        """Set precision record for an expert (used for BF16)."""
        self.expert_precision[global_expert_id] = precision

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics for debugging."""
        return {
            'num_total_experts': self.num_total_experts,
            'num_int8_slots': self.num_int8_slots,
            'num_int4_slots': self.num_int4_slots,
            'int8_slots_used': self.num_int8_slots - len(self.int8_free_slots),
            'int4_slots_used': self.num_int4_slots - len(self.int4_free_slots),
            'int8_free_slots': len(self.int8_free_slots),
            'int4_free_slots': len(self.int4_free_slots),
            'experts_attached': len(self.expert_slot_map),
        }


class FusedAdamLSQCPUOffloadOptimizer:
    """Optimizer for MoE models using FusedAdamLSQ with dynamic quantization.

    Architecture:
    - GPU params (attention, embedding, router): torch.optim.AdamW on GPU
    - Expert params: FusedAdamLSQ on CPU with dynamic INT4/INT8/BF16 precision

    Expert weights are stored in CPU shared memory and loaded to GPU on-demand.
    Expert gradients are automatically offloaded to CPU during backward.
    GPU computes delta/z updates via LSQ rules and returns new values to CPU.
    CPU only uses transformation rules when precision changes (INT4↔INT8).

    Dynamic Quantization Precision:
    - Compute expert scores using exp_avg_sq as Hessian approximation
    - Top 5% experts: BF16 (no quantization)
    - Top 30% experts: INT8
    - Rest: INT4

    Example:
        >>> model = Qwen3MoEModel(config, pg_collection, expert_sets)
        >>> optimizer = FusedAdamLSQCPUOffloadOptimizer(model, lr=1e-4)
        >>>
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        clip_grad: float = 1.0,
        # Dynamic quantization parameters
        score_update_interval: int = 100,
        freq_smoothing_alpha: float = 0.9,
        quant_group_size: int = 128,
        top_bf16_ratio: float = 0.05,
        top_int8_ratio: float = 0.30,
        lr_quant: float = 1e-4,  # Learning rate for delta/z updates (GPU uses this)
        initial_precision: int = 8,  # Initial precision for all experts (16=BF16, 8=INT8, 4=INT4)
    ):
        """Initialize FusedAdamLSQCPUOffloadOptimizer.

        Args:
            model: The MoE model to optimize
            lr: Learning rate
            betas: Adam beta coefficients
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
            warmup_steps: Number of warmup steps for learning rate
            clip_grad: Maximum gradient norm for clipping (0 to disable)
            score_update_interval: Number of steps between score recalculations
            freq_smoothing_alpha: Smoothing factor for call frequency history
            quant_group_size: Number of elements per quantization group
            top_bf16_ratio: Ratio of experts to keep in BF16 (no quantization)
            top_int8_ratio: Ratio of experts to use INT8
            lr_quant: Learning rate for delta/z updates (used by GPU)
            initial_precision: Initial precision for all experts (16/8/4)
        """
        import time
        self._debug_start_time = time.time()
        self._debug_print = lambda msg: print(f"[DEBUG Init {time.time()-self._debug_start_time:.2f}s] {msg}")

        self.base_lr = lr
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.warmup_steps = warmup_steps
        self.clip_grad = clip_grad

        # Dynamic quantization parameters
        self.score_update_interval = score_update_interval
        self.freq_smoothing_alpha = freq_smoothing_alpha
        self.quant_group_size = quant_group_size
        self.top_bf16_ratio = top_bf16_ratio
        self.top_int8_ratio = top_int8_ratio
        self.lr_quant = lr_quant
        self.initial_precision = initial_precision

        # Separate parameters
        self.gpu_params: List[nn.Parameter] = []
        self.expert_modules: List[QuantizedDispatcherCacheGroupedMLP] = []
        self._collect_parameters(model)
        self._debug_print("_collect_parameters done")

        # Get expert count from first module
        if self.expert_modules:
            self.num_global_experts = self.expert_modules[0].num_global_experts
        else:
            self.num_global_experts = 0

        # CUDA Event for CPU-GPU synchronization
        self._cpu_update_done_event = torch.cuda.Event()

        # Get shared gradient offload stream from expert modules
        if self.expert_modules:
            self._grad_offload_stream = self.expert_modules[0]._grad_offload_stream
        else:
            self._grad_offload_stream = None

        # EP group for GPU param gradient allreduce
        self._ep_group = parallel_state.get_expert_model_parallel_group()
        self._ep_size = dist.get_world_size(self._ep_group) if dist.is_initialized() else 1
        self._ep_rank = dist.get_rank(self._ep_group) if dist.is_initialized() else 0
        self._debug_print(f"EP group setup done: ep_size={self._ep_size}, ep_rank={self._ep_rank}")

        # Standard AdamW for GPU parameters
        if self.gpu_params:
            self.gpu_optimizer = torch.optim.AdamW(
                self.gpu_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        else:
            self.gpu_optimizer = None
        self._debug_print(f"GPU optimizer created: {len(self.gpu_params)} GPU params")

        # === Global Quantization Pool ===
        # All layers share a single quantization pool
        self.num_layers = len(self.expert_modules)
        self.num_experts_per_layer = self.num_global_experts

        # Get weight shapes from first module config
        if self.expert_modules:
            first_module = self.expert_modules[0]
            w1_numel = first_module.hidden_size * first_module.fc1_out_features
            w2_numel = first_module.ffn_hidden_size * first_module.hidden_size
            weight_shapes = {'w1_numel': w1_numel, 'w2_numel': w2_numel}
            self._hidden_size = first_module.hidden_size
            self._fc1_out_features = first_module.fc1_out_features
            self._ffn_hidden_size = first_module.ffn_hidden_size
        else:
            weight_shapes = {'w1_numel': 0, 'w2_numel': 0}
            self._hidden_size = 0
            self._fc1_out_features = 0
            self._ffn_hidden_size = 0

        # Create global quantization pool
        self.quant_pool = GlobalQuantizationPool(
            num_layers=self.num_layers,
            num_experts_per_layer=self.num_experts_per_layer,
            weight_shapes=weight_shapes,
            int8_ratio=top_int8_ratio,
            int4_ratio=1.0 - top_bf16_ratio - top_int8_ratio,
            quant_group_size=quant_group_size,
        )
        self._debug_print(f"GlobalQuantizationPool created: {self.num_layers} layers, {self.num_experts_per_layer} experts/layer")

        # Expert scores for dynamic precision allocation (global_expert_id -> score)
        self._expert_scores: Dict[int, float] = {}

        # Call frequency history (global_expert_id -> smoothed frequency)
        self._call_frequency: Dict[int, float] = {}

        # Initialize all experts: attach to pool with initial precision
        self._init_quantization_pool()
        self._debug_print("_init_quantization_pool done")

        # Create main weight storage and Parameters for FusedAdamLSQ
        # QuantizedDispatcher doesn't have weight1/weight2, so optimizer manages them
        # Per-expert Parameters for fine-grained precision control
        self._main_weight_storage: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._per_expert_weight_params: Dict[int, Tuple[nn.Parameter, nn.Parameter]] = {}  # global_expert_id -> (w1_param, w2_param)
        # Keep references to shared memory objects to prevent GC from releasing them
        self._shm_objects: Dict[str, shm.SharedMemory] = {}  # shm_name -> SharedMemory
        self._debug_print("Starting _init_main_weight_storage...")
        self._init_main_weight_storage()
        self._debug_print("_init_main_weight_storage done")

        # FusedAdamLSQ for expert parameters - ONLY rank 0 creates it
        expert_params = list(self._per_expert_weight_params.values())  # Flatten to list of params
        expert_params = [p for pair in expert_params for p in pair]  # (w1, w2) pairs -> flat list
        if expert_params and self._ep_rank == 0:
            if HAVE_FUSED_ADAM_LSQ:
                # Create FusedAdamLSQ with default q_bits=8
                # Per-param q_bits will be set via _inject_pool_delta_z_to_optimizer
                self.cpu_optimizer = FusedAdamLSQ(
                    expert_params,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                    adamw_mode=True,
                    fp32_optimizer_states=True,
                    group_size=quant_group_size,
                    q_bits=8,  # Default q_bits, per-param will override
                )
                self._debug_print(f"FusedAdamLSQ created: {len(expert_params)} expert params")
            else:
                raise ImportError(
                    "FusedAdamLSQ not available. Please install fuse_opt package."
                )
        else:
            self.cpu_optimizer = None
        self._debug_print("FusedAdamLSQ setup complete")

        # Threading support for async CPU-GPU optimizer overlap
        self._cpu_step_done_event = threading.Event()
        self._cpu_step_done_event.set()  # Initially set, so first step doesn't wait
        self._cpu_step_lock = threading.Lock()
        self._cpu_optimizer_thread = None

        # Pass CPU update event to expert modules
        for module in self.expert_modules:
            module.set_cpu_update_event(self._cpu_update_done_event)

    def _collect_parameters(self, model: nn.Module):
        """Collect and separate GPU params from expert params.

        Only recognizes QuantizedDispatcherCacheGroupedMLP modules.
        """
        self.router_modules: List[Any] = []
        for name, module in model.named_modules():
            if isinstance(module, QuantizedDispatcherCacheGroupedMLP):
                self.expert_modules.append(module)
            if hasattr(module, 'router') and hasattr(module.router, 'weight'):
                self.router_modules.append(module.router)

        # GPU params are all non-expert params
        for param in model.parameters():
            if param.requires_grad:
                self.gpu_params.append(param)

    def _init_quantization_pool(self):
        """Initialize global quantization pool.

        新设计:
        - 所有专家都有delta/z buffer (在GlobalQuantizationPool.__init__中自动创建)
        - 只有INT8/INT4专家attach到quant slot
        - BF16专家不attach quant slot, 但delta/z buffer已存在

        initial_precision决定:
        - initial_precision=8: attach到INT8 quant slot
        - initial_precision=4: attach到INT4 quant slot
        - initial_precision=16: 不attach quant slot (使用BF16权重)
        """
        for layer_id in range(self.num_layers):
            for local_expert_id in range(self.num_experts_per_layer):
                global_expert_id = encode_global_expert_id(
                    layer_id, local_expert_id, self.num_experts_per_layer
                )

                # Initialize call frequency
                self._call_frequency[global_expert_id] = 0.0

                # 设置初始精度
                self.quant_pool.set_precision(global_expert_id, self.initial_precision)

                if self.initial_precision in [4, 8]:
                    # Attach to quant slot
                    self.quant_pool.attach_quant_slot(global_expert_id, self.initial_precision)
                # BF16: 不attach quant slot, delta/z buffer已存在

    def _init_main_weight_storage(self):
        """Initialize main weight storage for QuantizedDispatcher.

        For QuantizedDispatcher, the optimizer manages the main BF16 weights
        in CPU shared memory and creates Parameter objects for FusedAdamLSQ.
        """
        import multiprocessing.shared_memory as shm
        from megatron.core.transformer.moe.experts import pin_existing_tensor
        import time

        # Get weight shapes from config
        if not self.expert_modules:
            return

        first_dispatcher = self.expert_modules[0]
        hidden_size = self._hidden_size
        fc1_out_features = self._fc1_out_features
        ffn_hidden_size = self._ffn_hidden_size
        dtype = first_dispatcher.config.params_dtype

        # Create shared memory for each layer's experts
        ep_ranks = dist.get_process_group_ranks(self._ep_group) if dist.is_initialized() else [0]
        base_rank = min(ep_ranks) if ep_ranks else 0
        is_rank_0 = (self._ep_rank == 0)

        self._debug_print(f"_init_main_weight_storage: num_layers={self.num_layers}, experts_per_layer={self.num_experts_per_layer}")

        for layer_id in range(self.num_layers):
            layer_start = time.time()
            # Create shared memory for weight1 and weight2
            shm_suffix = f"_quant_layer{layer_id}_r{base_rank}"

            # Weight1 shared memory
            shm_name_w1 = f"megatron_quant_w1{shm_suffix}"
            size_w1 = self.num_experts_per_layer * hidden_size * fc1_out_features * dtype.itemsize
            self._debug_print(f"  Layer {layer_id}: Creating w1 shm (size={size_w1/1e6:.1f}MB)")

            if is_rank_0:
                try:
                    shm.SharedMemory(name=shm_name_w1).unlink()
                except FileNotFoundError:
                    pass
                shm_w1 = shm.SharedMemory(create=True, size=size_w1, name=shm_name_w1)
                self._shm_objects[shm_name_w1] = shm_w1  # Keep reference to prevent GC
                # Barrier after creating shm (let other ranks attach)
                dist.barrier(group=self._ep_group) if dist.is_initialized() else None
            else:
                # Wait for rank 0 to create shm before attaching
                dist.barrier(group=self._ep_group) if dist.is_initialized() else None
                shm_w1 = shm.SharedMemory(name=shm_name_w1)
                self._shm_objects[shm_name_w1] = shm_w1  # Keep reference to prevent GC
            self._debug_print(f"  Layer {layer_id}: w1 shm ready ({time.time()-layer_start:.2f}s)")

            w1_data = torch.frombuffer(shm_w1.buf, dtype=dtype).view(
                self.num_experts_per_layer, hidden_size, fc1_out_features
            )

            # Weight2 shared memory
            shm_name_w2 = f"megatron_quant_w2{shm_suffix}"
            size_w2 = self.num_experts_per_layer * ffn_hidden_size * hidden_size * dtype.itemsize
            self._debug_print(f"  Layer {layer_id}: Creating w2 shm (size={size_w2/1e6:.1f}MB)")

            if is_rank_0:
                try:
                    shm.SharedMemory(name=shm_name_w2).unlink()
                except FileNotFoundError:
                    pass
                shm_w2 = shm.SharedMemory(create=True, size=size_w2, name=shm_name_w2)
                self._shm_objects[shm_name_w2] = shm_w2  # Keep reference to prevent GC
                # Barrier after creating shm (let other ranks attach)
                dist.barrier(group=self._ep_group) if dist.is_initialized() else None
            else:
                # Wait for rank 0 to create shm before attaching
                dist.barrier(group=self._ep_group) if dist.is_initialized() else None
                shm_w2 = shm.SharedMemory(name=shm_name_w2)
                self._shm_objects[shm_name_w2] = shm_w2  # Keep reference to prevent GC
            self._debug_print(f"  Layer {layer_id}: w2 shm ready ({time.time()-layer_start:.2f}s)")

            w2_data = torch.frombuffer(shm_w2.buf, dtype=dtype).view(
                self.num_experts_per_layer, ffn_hidden_size, hidden_size
            )

            # Initialize weights on rank 0 (GPU-based initialization for speed)
            if is_rank_0 and self.expert_modules and self.expert_modules[0].config.perform_initialization:
                init_start = time.time()
                torch.manual_seed(42)
                device = torch.cuda.current_device()
                self._debug_print(f"  Layer {layer_id}: Allocating GPU tensors...")
                # Initialize all experts on GPU at once (much faster than chunked)
                # Single large allocation instead of many small ones
                temp_w1_gpu = torch.empty(
                    self.num_experts_per_layer, hidden_size, fc1_out_features,
                    dtype=dtype, device=device
                )
                temp_w2_gpu = torch.empty(
                    self.num_experts_per_layer, ffn_hidden_size, hidden_size,
                    dtype=dtype, device=device
                )
                self._debug_print(f"  Layer {layer_id}: Applying init_method...")
                # Apply init methods
                self.expert_modules[0].config.init_method(temp_w1_gpu)
                self.expert_modules[0].config.output_layer_init_method(temp_w2_gpu)
                self._debug_print(f"  Layer {layer_id}: Copying to CPU...")
                # Single copy to CPU
                w1_data.copy_(temp_w1_gpu.cpu())
                w2_data.copy_(temp_w2_gpu.cpu())
                del temp_w1_gpu, temp_w2_gpu
                self._debug_print(f"  Layer {layer_id}: GPU init done ({time.time()-init_start:.2f}s)")
                # Barrier after initialization complete
                dist.barrier(group=self._ep_group) if dist.is_initialized() else None
            else:
                # Wait for rank 0 to finish initialization
                dist.barrier(group=self._ep_group) if dist.is_initialized() else None
            self._debug_print(f"  Layer {layer_id}: After init barrier ({time.time()-layer_start:.2f}s)")

            # Pin shared memory for async transfers
            pin_existing_tensor(w1_data)
            pin_existing_tensor(w2_data)
            self._debug_print(f"  Layer {layer_id}: Pinned memory ({time.time()-layer_start:.2f}s)")

            # Create per-expert Parameter objects for FusedAdamLSQ
            # Each expert has its own Parameter for fine-grained precision control
            for local_exp_id in range(self.num_experts_per_layer):
                global_expert_id = encode_global_expert_id(layer_id, local_exp_id, self.num_experts_per_layer)

                # Create Parameter for each expert's weight slice
                w1_expert_param = nn.Parameter(w1_data[local_exp_id], requires_grad=False)
                w2_expert_param = nn.Parameter(w2_data[local_exp_id], requires_grad=False)
                self._per_expert_weight_params[global_expert_id] = (w1_expert_param, w2_expert_param)

                # Store tensor references for each expert
                self._main_weight_storage[global_expert_id] = (w1_data[local_exp_id], w2_data[local_exp_id])

            self._debug_print(f"  Layer {layer_id}: Created {self.num_experts_per_layer} expert params ({time.time()-layer_start:.2f}s)")

    def update_call_frequency(self, tokens_per_expert: torch.Tensor, layer_id: int = 0):
        """Update expert call frequency with historical smoothing.

        Args:
            tokens_per_expert: [num_global_experts] tensor with token counts per expert
            layer_id: Layer ID for global expert ID encoding (default 0 for single layer)
        """
        for local_expert_id in range(self.num_experts_per_layer):
            global_expert_id = encode_global_expert_id(
                layer_id, local_expert_id, self.num_experts_per_layer
            )
            current_freq = tokens_per_expert[local_expert_id].item()

            # Historical smoothing: freq_new = alpha * freq_old + (1-alpha) * freq_current
            old_freq = self._call_frequency.get(global_expert_id, 0.0)
            self._call_frequency[global_expert_id] = (
                self.freq_smoothing_alpha * old_freq +
                (1 - self.freq_smoothing_alpha) * current_freq
            )

    def compute_expert_scores(self):
        """Compute expert importance scores for dynamic precision allocation.

        Score formula:
        Score_expert = sum_g(h_bar_g * Span(W_g)^2) * call_frequency

        where:
        - h_bar_g: average exp_avg_sq per group (Hessian diagonal approximation)
        - Span(W_g): weight range (max - min) per group
        - call_frequency: smoothed token routing count

        Uses optimizer's second moment (exp_avg_sq) from FusedAdamLSQ.
        Scores computed for all experts across all layers (global ranking).
        """
        print(f"[DEBUG compute_expert_scores] cpu_optimizer={self.cpu_optimizer is not None}")
        if self.cpu_optimizer is None:
            return

        # Iterate over per-expert params
        num_experts_with_state = 0
        for global_expert_id, (w1_param, w2_param) in self._per_expert_weight_params.items():
            # Get optimizer state for weight1 and weight2 params
            state_w1 = self.cpu_optimizer.state.get(w1_param, {})
            state_w2 = self.cpu_optimizer.state.get(w2_param, {})

            exp_avg_sq_w1 = state_w1.get('exp_avg_sq')
            exp_avg_sq_w2 = state_w2.get('exp_avg_sq')

            print(f"[DEBUG compute_expert_scores] global_expert_id={global_expert_id}, state_w1={len(state_w1)}, exp_avg_sq_w1={exp_avg_sq_w1 is not None}")

            if exp_avg_sq_w1 is None:
                # State not initialized yet, skip scoring
                continue

            num_experts_with_state += 1

            # Get expert weights (each param is a single expert)
            w1_expert = w1_param.data
            w2_expert = w2_param.data

            # Compute score for weight1
            score_w1 = 0.0
            if exp_avg_sq_w1 is not None:
                # Reshape to groups
                num_groups = w1_expert.numel() // self.quant_group_size
                if num_groups > 0:
                    w1_groups = w1_expert.view(num_groups, self.quant_group_size)
                    h_bar_w1 = exp_avg_sq_w1.view(num_groups, self.quant_group_size).mean(dim=1)
                    span_w1 = w1_groups.max(dim=1).values - w1_groups.min(dim=1).values
                    score_w1 = (h_bar_w1 * span_w1 ** 2).sum().item()

            # Compute score for weight2
            score_w2 = 0.0
            if exp_avg_sq_w2 is not None:
                num_groups = w2_expert.numel() // self.quant_group_size
                if num_groups > 0:
                    w2_groups = w2_expert.view(num_groups, self.quant_group_size)
                    h_bar_w2 = exp_avg_sq_w2.view(num_groups, self.quant_group_size).mean(dim=1)
                    span_w2 = w2_groups.max(dim=1).values - w2_groups.min(dim=1).values
                    score_w2 = (h_bar_w2 * span_w2 ** 2).sum().item()

            # Combined score with call frequency
            freq = self._call_frequency.get(global_expert_id, 1.0)
            self._expert_scores[global_expert_id] = (score_w1 + score_w2) * freq

        print(f"[DEBUG compute_expert_scores] num_experts_with_state={num_experts_with_state}, _expert_scores={len(self._expert_scores)}")

    def update_quant_precision(self):
        """Update expert precision allocation based on global ranking.

        Precision allocation (all layers unified ranking):
        - Top 5% (top_bf16_ratio): BF16 (no quantization)
        - Top 30% (top_int8_ratio): INT8
        - Rest: INT4
        """
        print(f"[DEBUG update_quant_precision] _expert_scores={self._expert_scores}")
        if not self._expert_scores:
            print("[DEBUG update_quant_precision] NO SCORES, returning")
            return

        # Sort all experts by score (descending) - global ranking across all layers
        scores = sorted(self._expert_scores.items(), key=lambda x: -x[1])
        print(f"[DEBUG update_quant_precision] scores={scores}")

        num_total_experts = self.num_layers * self.num_experts_per_layer
        top_bf16_count = int(num_total_experts * self.top_bf16_ratio)
        # 使用 quant_pool 的 slot 数量，确保与 pool 容量一致
        top_int8_count = self.quant_pool.num_int8_slots
        print(f"[DEBUG update_quant_precision] num_total_experts={num_total_experts}, top_bf16_count={top_bf16_count}, top_int8_count={top_int8_count}")

        for rank, (global_expert_id, score) in enumerate(scores):
            old_precision = self.quant_pool.get_precision(global_expert_id)

            # Determine new precision based on global rank
            if rank < top_bf16_count:
                new_precision = 16  # BF16
            elif rank < top_bf16_count + top_int8_count:
                new_precision = 8   # INT8
            else:
                new_precision = 4   # INT4

            print(f"[DEBUG update_quant_precision] rank={rank}, global_expert_id={global_expert_id}, old={old_precision}, new={new_precision}")

            # Apply precision change if needed
            if old_precision != new_precision:
                self._transition_precision(global_expert_id, old_precision, new_precision)

    def _transition_precision(self, global_expert_id: int, old_precision: int, new_precision: int):
        """Transition expert from old precision to new precision.

        新设计:
        - delta/z buffer不区分精度, 用float32保存
        - 通过变换调整值: INT8→INT4乘16, INT4→INT8除16
        - 只切换quant slot (detach旧slot, attach新slot)

        Args:
            global_expert_id: Global expert ID to transition
            old_precision: Current precision (16/8/4)
            new_precision: Target precision (16/8/4)
        """
        print(f"[DEBUG _transition_precision] global_expert_id={global_expert_id}, old={old_precision}, new={new_precision}")
        k = 16  # 2^(8-4)

        if old_precision == new_precision:
            return  # No change needed

        # INT4 → INT8 upgrade: delta/z值变换
        if old_precision == 4 and new_precision == 8:
            # 获取delta/z (统一buffer)
            delta_w1, z_w1, delta_w2, z_w2 = self.quant_pool.get_delta_z(global_expert_id)

            # 值变换: Δ_int8 = Δ_int4 / 16, Z_int8 = Z_int4 * 16
            delta_w1 /= k
            z_w1 *= k
            delta_w2 /= k
            z_w2 *= k

            # 切换quant slot
            self.quant_pool.detach_quant_slot(global_expert_id)
            self.quant_pool.attach_quant_slot(global_expert_id, 8)

        # INT8 → INT4 downgrade: delta/z值变换
        elif old_precision == 8 and new_precision == 4:
            delta_w1, z_w1, delta_w2, z_w2 = self.quant_pool.get_delta_z(global_expert_id)

            # 值变换: Δ_int4 = Δ_int8 * 16, Z_int4 = round(Z_int8 / 16)
            delta_w1 *= k
            z_w1 = torch.round(z_w1 / k)
            delta_w2 *= k
            z_w2 = torch.round(z_w2 / k)

            # 切换quant slot
            self.quant_pool.detach_quant_slot(global_expert_id)
            self.quant_pool.attach_quant_slot(global_expert_id, 4)

        # BF16 → INT8/INT4: delta/z已由GPU初始化
        elif old_precision == 16 and new_precision in [4, 8]:
            # GPU首次前向计算了delta/z, 已offload到buffer
            # 只需attach quant slot
            slot_idx = self.quant_pool.attach_quant_slot(global_expert_id, new_precision)
            print(f"[DEBUG _transition_precision] BF16→{new_precision}, attached slot_idx={slot_idx}")

        # INT8/INT4 → BF16: detach quant slot, delta/z保留
        elif old_precision in [4, 8] and new_precision == 16:
            self.quant_pool.detach_quant_slot(global_expert_id)
            self.quant_pool.set_precision(global_expert_id, 16)

        # 更新精度记录
        self.quant_pool.expert_precision[global_expert_id] = new_precision
        print(f"[DEBUG _transition_precision] expert_precision updated: {self.quant_pool.expert_precision[global_expert_id]}")

    def receive_gpu_updates(self, global_expert_id: int,
                            delta_w1: torch.Tensor, z_w1: torch.Tensor,
                            delta_w2: torch.Tensor, z_w2: torch.Tensor,
                            quant_w1: Optional[torch.Tensor] = None,
                            quant_w2: Optional[torch.Tensor] = None):
        """Receive updated delta/z from GPU and write to buffer.

        新设计:
        - delta/z直接覆盖统一buffer (不区分精度)
        - quant_w写入对应的quant slot

        Args:
            global_expert_id: Global expert ID
            delta_w1: Updated delta for weight1 (from GPU)
            z_w1: Updated z for weight1 (from GPU)
            delta_w2: Updated delta for weight2 (from GPU)
            z_w2: Updated z for weight2 (from GPU)
            quant_w1: Optional updated quantized weight1 (from GPU)
            quant_w2: Optional updated quantized weight2 (from GPU)
        """
        # 写入统一delta/z buffer
        self.quant_pool.write_delta_z(global_expert_id, delta_w1, z_w1, delta_w2, z_w2)

        # 写入quant weight (如果有)
        if quant_w1 is not None or quant_w2 is not None:
            self.quant_pool.write_quant_weight(global_expert_id, quant_w1, quant_w2)

    def get_quant_params(self, global_expert_id: int) -> Tuple[Optional[torch.Tensor], ...]:
        """Get quantization parameters for an expert.

        返回 (quant_w1, delta_w1, z_w1, quant_w2, delta_w2, z_w2, precision)

        Args:
            global_expert_id: Global expert ID

        Returns:
            Tuple of (quant_w1, delta_w1, z_w1, quant_w2, delta_w2, z_w2, precision)
        """
        precision = self.quant_pool.get_precision(global_expert_id)
        quant_w1, quant_w2 = self.quant_pool.get_quant_weight(global_expert_id)
        delta_w1, z_w1, delta_w2, z_w2 = self.quant_pool.get_delta_z(global_expert_id)
        return (quant_w1, delta_w1, z_w1, quant_w2, delta_w2, z_w2, precision)

    def get_quant_params_local(self, layer_id: int, local_expert_id: int) -> Tuple[Optional[torch.Tensor], ...]:
        """Get quantization parameters using local expert ID (for backward compatibility).

        Args:
            layer_id: Layer ID
            local_expert_id: Local expert ID within the layer

        Returns:
            Tuple of (quant_w1, delta_w1, z_w1, quant_w2, delta_w2, z_w2, precision)
        """
        global_expert_id = encode_global_expert_id(
            layer_id, local_expert_id, self.num_experts_per_layer
        )
        return self.get_quant_params(global_expert_id)

    def _get_main_weight(self, global_expert_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get main BF16 weight for an expert.

        Args:
            global_expert_id: Global expert ID

        Returns:
            Tuple of (w1, w2) BF16 tensors from CPU shared memory
        """
        # QuantizedDispatcher: weights managed by optimizer in _main_weight_storage
        if global_expert_id not in self._main_weight_storage:
            raise RuntimeError(f"Expert {global_expert_id} not found in _main_weight_storage")
        return self._main_weight_storage[global_expert_id]

    def prefetch_expert_data(
        self,
        layer_id: int,
        local_expert_ids: List[int],
        gpu_w1_bf16_buffer: torch.Tensor,
        gpu_w2_bf16_buffer: torch.Tensor,
        gpu_w1_quant_buffer: torch.Tensor,
        gpu_w2_quant_buffer: torch.Tensor,
        gpu_delta_w1_buffer: Optional[torch.Tensor],
        gpu_z_w1_buffer: Optional[torch.Tensor],
        gpu_delta_w2_buffer: Optional[torch.Tensor],
        gpu_z_w2_buffer: Optional[torch.Tensor],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> List[int]:
        """Unified prefetch interface - directly copy to MoE layer GPU workspace.

        新设计:
        - BF16: 只传输BF16权重, 不传输delta/z (GPU计算新delta/z不需要原始值)
        - INT8/INT4: 传输quant_w + delta/z (GPU用delta/z反量化, 计算后更新)
        - quant未初始化时回退到BF16路径

        Args:
            layer_id: Layer ID for global expert ID encoding
            local_expert_ids: List of local expert IDs to prefetch
            gpu_w1_bf16_buffer: GPU buffer for BF16 w1 (dtype=torch.bfloat16)
            gpu_w2_bf16_buffer: GPU buffer for BF16 w2 (dtype=torch.bfloat16)
            gpu_w1_quant_buffer: GPU buffer for INT8/INT4 quant_w1 (dtype=torch.uint8)
            gpu_w2_quant_buffer: GPU buffer for INT8/INT4 quant_w2 (dtype=torch.uint8)
            gpu_delta_w1_buffer: GPU buffer for delta_w1 (dtype=torch.float32)
            gpu_z_w1_buffer: GPU buffer for z_w1 (dtype=torch.float32)
            gpu_delta_w2_buffer: GPU buffer for delta_w2 (dtype=torch.float32)
            gpu_z_w2_buffer: GPU buffer for z_w2 (dtype=torch.float32)
            stream: CUDA stream for async transfer (default: current stream)

        Returns:
            List of precisions [16, 8, 4, ...] for each expert
        """
        if stream is None:
            stream = torch.cuda.current_stream()

        precisions = []

        with torch.cuda.stream(stream):
            for i, local_exp_id in enumerate(local_expert_ids):
                global_expert_id = encode_global_expert_id(
                    layer_id, local_exp_id, self.num_experts_per_layer
                )
                precision = self.quant_pool.get_precision(global_expert_id)
                precisions.append(precision)

                if precision == 16:
                    # BF16: Copy main weight to bf16 buffer
                    w1_cpu, w2_cpu = self._get_main_weight(global_expert_id)
                    # print(f"[DEBUG prefetch] global_exp_id={global_expert_id}, precision={precision}, w1_cpu.shape={w1_cpu.shape}, w1_cpu.is_pinned={w1_cpu.is_pinned()}, gpu_buf.shape={gpu_w1_bf16_buffer[i].shape}, gpu_buf.device={gpu_w1_bf16_buffer[i].device}")
                    gpu_w1_bf16_buffer[i].copy_(w1_cpu, non_blocking=True)
                    gpu_w2_bf16_buffer[i].copy_(w2_cpu, non_blocking=True)
                else:
                    # INT8/INT4: 检查quant是否已初始化
                    quant_initialized = self.quant_pool.is_quant_initialized(global_expert_id)
                    print(f"[DEBUG prefetch] global_expert_id={global_expert_id}, precision={precision}, quant_initialized={quant_initialized}")
                    if not quant_initialized:
                        # Fallback: Use BF16 main weight instead
                        w1_cpu, w2_cpu = self._get_main_weight(global_expert_id)
                        # print(f"[DEBUG prefetch] global_exp_id={global_expert_id}, SLOT NOT INITIALIZED")
                        # print(f"[DEBUG prefetch] w1_cpu: shape={w1_cpu.shape}, dtype={w1_cpu.dtype}, is_pinned={w1_cpu.is_pinned()}, data_ptr={w1_cpu.data_ptr()}")
                        # print(f"[DEBUG prefetch] w2_cpu: shape={w2_cpu.shape}, dtype={w2_cpu.dtype}, is_pinned={w2_cpu.is_pinned()}, data_ptr={w2_cpu.data_ptr()}")
                        # print(f"[DEBUG prefetch] gpu_w1_buf: shape={gpu_w1_bf16_buffer[i].shape}, device={gpu_w1_bf16_buffer[i].device}")
                        # print(f"[DEBUG prefetch] gpu_w2_buf: shape={gpu_w2_bf16_buffer[i].shape}, device={gpu_w2_bf16_buffer[i].device}")
                        # Use blocking copy for first pass to ensure GPU state is stable
                        gpu_w1_bf16_buffer[i].copy_(w1_cpu, non_blocking=False)
                        # print(f"[DEBUG prefetch] w1 copy completed")
                        gpu_w2_bf16_buffer[i].copy_(w2_cpu, non_blocking=False)
                        # print(f"[DEBUG prefetch] w2 copy completed")
                        # Override precision for this forward pass
                        precisions[-1] = 16
                        continue

                    # INT8/INT4: Copy quant_w to uint8 buffer + delta/z to float32 buffers
                    quant_w1, quant_w2 = self.quant_pool.get_quant_weight(global_expert_id)
                    delta_w1, z_w1, delta_w2, z_w2 = self.quant_pool.get_delta_z(global_expert_id)

                    # For INT8: quant_w shape matches original weight shape
                    # For INT4: quant_w is packed (half size), copy directly as flatten
                    if precision == 8:
                        # INT8: Reshape quant weights to match buffer shape for copy
                        quant_w1_2d = quant_w1.view(self._hidden_size, self._fc1_out_features)
                        quant_w2_2d = quant_w2.view(self._ffn_hidden_size, self._hidden_size)
                        gpu_w1_quant_buffer[i].copy_(quant_w1_2d, non_blocking=True)
                        gpu_w2_quant_buffer[i].copy_(quant_w2_2d, non_blocking=True)
                    else:  # INT4
                        # INT4: quant_w is packed, copy flatten form directly
                        # GPU buffer has space for INT8 size, but we only fill half (packed INT4)
                        gpu_w1_quant_buffer[i].flatten()[:quant_w1.numel()].copy_(quant_w1, non_blocking=True)
                        gpu_w2_quant_buffer[i].flatten()[:quant_w2.numel()].copy_(quant_w2, non_blocking=True)

                    # Copy delta/z to float32 buffers (delta/z always exist now)
                    if gpu_delta_w1_buffer is not None:
                        gpu_delta_w1_buffer[i].copy_(delta_w1, non_blocking=True)
                    if gpu_z_w1_buffer is not None:
                        gpu_z_w1_buffer[i].copy_(z_w1, non_blocking=True)
                    if gpu_delta_w2_buffer is not None:
                        gpu_delta_w2_buffer[i].copy_(delta_w2, non_blocking=True)
                    if gpu_z_w2_buffer is not None:
                        gpu_z_w2_buffer[i].copy_(z_w2, non_blocking=True)
        print("precisions:" , precisions)
        return precisions

    def sync_gradient_offload(self):
        """Synchronize gradient offload stream before reading CPU gradients."""
        if self._grad_offload_stream is not None:
            self._grad_offload_stream.synchronize()

    def record_cpu_update_done(self):
        """Record Event signaling CPU optimizer update completion."""
        if self.cpu_optimizer is not None:
            # Touch the weight params to ensure they're updated
            for global_expert_id, (w1_param, w2_param) in self._per_expert_weight_params.items():
                if w1_param is not None:
                    _ = w1_param.data.reshape(-1)[::32].sum().item()
                if w2_param is not None:
                    _ = w2_param.data.reshape(-1)[::32].sum().item()

            self._cpu_update_done_event.record(torch.cuda.current_stream())

    def get_cpu_update_event(self) -> torch.cuda.Event:
        """Get the CPU update completion Event."""
        return self._cpu_update_done_event

    def zero_grad(self, set_to_none: bool = False):
        """Zero all gradients."""
        nvtx.range_push("FusedAdamLSQCPUOffloadOptimizer::zero_grad")

        nvtx.range_push("sync_gradient_offload")
        self.sync_gradient_offload()
        nvtx.range_pop()

        if self.gpu_optimizer is not None:
            self.gpu_optimizer.zero_grad(set_to_none=set_to_none)

        for module in self.expert_modules:
            if hasattr(module, 'zero_grad'):
                module.zero_grad(set_to_none=set_to_none)

        nvtx.range_pop()

    def _inject_pool_delta_z_to_optimizer(self):
        """将GlobalQuantizationPool的delta/z注入到FusedAdamLSQ。

        新设计:
        - 所有专家都有delta/z buffer, 都需要设置
        - INT8/INT4还需要设置quant_buffer
        - BF16只需要设置delta/z (用于下次量化)

        在cpu_optimizer.step()之前调用。
        """
        if self.cpu_optimizer is None:
            return

        # 为每个专家设置per-param精度和量化参数
        for global_expert_id, (w1_param, w2_param) in self._per_expert_weight_params.items():
            precision = self.quant_pool.get_precision(global_expert_id)

            # 获取delta/z (所有专家都有)
            delta_w1, z_w1, delta_w2, z_w2 = self.quant_pool.get_delta_z(global_expert_id)

            if precision in [4, 8]:
                # INT8/INT4: 设置量化参数
                quant_w1, quant_w2 = self.quant_pool.get_quant_weight(global_expert_id)

                # 设置q_bits
                self.cpu_optimizer.set_q_bits(w1_param, precision)
                self.cpu_optimizer.set_q_bits(w2_param, precision)

                # 设置quant_buffer
                if quant_w1 is not None:
                    self.cpu_optimizer.set_quant_buffer(w1_param, quant_w1.flatten())
                if quant_w2 is not None:
                    self.cpu_optimizer.set_quant_buffer(w2_param, quant_w2.flatten())

                # 设置delta/z
                self.cpu_optimizer.set_delta_tensor(w1_param, delta_w1.flatten())
                self.cpu_optimizer.set_z_tensor(w1_param, z_w1.flatten())
                self.cpu_optimizer.set_delta_tensor(w2_param, delta_w2.flatten())
                self.cpu_optimizer.set_z_tensor(w2_param, z_w2.flatten())
            else:
                # BF16: 设置q_bits为None, 但仍设置delta/z (用于下次量化)
                self.cpu_optimizer.set_q_bits(w1_param, None)
                self.cpu_optimizer.set_q_bits(w2_param, None)
                # 清除quant_buffer (BF16不量化)
                if w1_param in self.cpu_optimizer.quant_buffers:
                    del self.cpu_optimizer.quant_buffers[w1_param]
                if w2_param in self.cpu_optimizer.quant_buffers:
                    del self.cpu_optimizer.quant_buffers[w2_param]
                # 设置delta/z (用于下次量化)
                self.cpu_optimizer.set_delta_tensor(w1_param, delta_w1.flatten())
                self.cpu_optimizer.set_z_tensor(w1_param, z_w1.flatten())
                self.cpu_optimizer.set_delta_tensor(w2_param, delta_w2.flatten())
                self.cpu_optimizer.set_z_tensor(w2_param, z_w2.flatten())

    def _set_expert_param_gradients(self):
        """Set expert param gradients from CPU pinned buffer after sync_gradient_offload.

        Called in step() after sync_gradient_offload(), before cpu_optimizer.step().
        The sync_gradient_offload() ensures D2H copy is complete, so quant_pool.grad_w1/w2
        are ready to be used as param.grad.
        """
        if self.cpu_optimizer is None:
            return

        # Set gradients for all expert params
        for global_expert_id, (w1_param, w2_param) in self._per_expert_weight_params.items():
            # Get gradient from CPU pinned buffer and convert to param dtype
            # Reshape to match param shape
            w1_param.grad = self.quant_pool.grad_w1[global_expert_id].to(w1_param.dtype).view_as(w1_param)
            w2_param.grad = self.quant_pool.grad_w2[global_expert_id].to(w2_param.dtype).view_as(w2_param)

    def _sync_optimizer_quant_to_pool(self):
        """Sync optimizer's quant_buffers back to pool after step.

        FusedAdamLSQ.step() writes quant_w to its own quant_buffers.
        This method syncs those back to GlobalQuantizationPool.
        """
        if self.cpu_optimizer is None:
            return

        print(f"[DEBUG _sync_optimizer_quant_to_pool] step_count={self.step_count}")
        for global_expert_id, (w1_param, w2_param) in self._per_expert_weight_params.items():
            precision = self.quant_pool.get_precision(global_expert_id)
            print(f"[DEBUG _sync_optimizer_quant_to_pool] global_expert_id={global_expert_id}, precision={precision}")

            if precision in [4, 8]:
                # Sync quant_buffer from optimizer to pool
                quant_w1 = self.cpu_optimizer.quant_buffers.get(w1_param)
                quant_w2 = self.cpu_optimizer.quant_buffers.get(w2_param)

                print(f"[DEBUG _sync_optimizer_quant_to_pool] quant_w1={quant_w1 is not None}, quant_w2={quant_w2 is not None}")

                if quant_w1 is not None and quant_w2 is not None:
                    self.quant_pool.write_quant_weight(global_expert_id, quant_w1, quant_w2)
                    print(f"[DEBUG _sync_optimizer_quant_to_pool] wrote quant_weight for {global_expert_id}")

    def _cpu_optimizer_step_thread(self):
        """Run CPU optimizer step in a background thread."""
        if self.cpu_optimizer is not None:
            nvtx.range_push("cpu_optimizer_step_thread")
            with self._cpu_step_lock:
                self.cpu_optimizer.step()
                # Touch weight params to ensure they're updated
                for global_expert_id, (w1_param, w2_param) in self._per_expert_weight_params.items():
                    if w1_param is not None:
                        _ = w1_param.data.reshape(-1)[::32].sum().item()
                    if w2_param is not None:
                        _ = w2_param.data.reshape(-1)[::32].sum().item()

                # Sync quant_buffers back to pool
                self._sync_optimizer_quant_to_pool()

                # Clear gradient buffers for next iteration
                self.quant_pool.grad_w1.zero_()
                self.quant_pool.grad_w2.zero_()
            nvtx.range_pop()
        self._cpu_step_done_event.set()

    def step(self):
        """Update parameters with CPU-GPU optimizer overlap and dynamic quantization."""
        nvtx.range_push("FusedAdamLSQCPUOffloadOptimizer::step")

        # Sync gradient offload
        nvtx.range_push("sync_gradient_offload")
        self.sync_gradient_offload()
        nvtx.range_pop()

        # Apply warmup learning rate
        current_lr = self.get_lr()
        self.set_lr(current_lr)

        # Gradient clipping (only for GPU params - expert weight gradients handled by QuantizedDispatcher)
        if self.clip_grad > 0:
            nvtx.range_push("gradient_clipping")
            gpu_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.gpu_params, self.clip_grad
            )
            # Note: Expert weight gradients are computed in QuantizedDispatcher backward
            # and used for LSQ delta/z updates. For FusedAdamLSQ Adam update,
            # gradients are passed via the optimizer's custom interface.
            nvtx.range_pop()

        # Dynamic quantization update (every N steps)
        # Wait for previous CPU optimizer step to complete before computing scores
        # (optimizer state is needed for score computation)
        if self.cpu_optimizer is not None:
            self._cpu_step_done_event.wait()

        if self.step_count > 0 and self.step_count % self.score_update_interval == 2:
            nvtx.range_push("dynamic_quant_update")
            print("=========== update quant in ", self.step_count, " ==============")
            self.compute_expert_scores()
            self.update_quant_precision()
            nvtx.range_pop()

        # CPU-GPU Optimizer Parallel Execution
        if self.cpu_optimizer is not None:
            # Set expert param gradients from CPU pinned buffer
            self._set_expert_param_gradients()

            # 注入pool的delta/z到FusedAdamLSQ（在step之前）
            self._inject_pool_delta_z_to_optimizer()

            self._cpu_step_done_event.clear()
            self._cpu_optimizer_thread = threading.Thread(
                target=self._cpu_optimizer_step_thread,
                daemon=True
            )
            self._cpu_optimizer_thread.start()

        if self.gpu_optimizer is not None:
            nvtx.range_push("gpu_optimizer_step")
            if self._ep_size > 1 and dist.is_initialized():
                for param in self.gpu_params:
                    if param.grad is not None:
                        dist.all_reduce(
                            param.grad,
                            op=dist.ReduceOp.SUM,
                            group=self._ep_group,
                        )
                        param.grad.div_(self._ep_size)

            self.gpu_optimizer.step()
            nvtx.range_pop()

        if self.cpu_optimizer is not None:
            self._cpu_step_done_event.wait()

        if self._ep_size > 1 and dist.is_initialized():
            dist.barrier(group=self._ep_group)

        nvtx.range_push("record_cpu_update_event")
        self.record_cpu_update_done()
        nvtx.range_pop()

        self.step_count += 1
        nvtx.range_pop()

    def get_lr(self) -> float:
        """Get current learning rate with warmup."""
        if self.step_count < self.warmup_steps:
            return self.base_lr * (self.step_count + 1) / self.warmup_steps
        return self.base_lr

    def set_lr(self, lr: float):
        """Set learning rate."""
        self.lr = lr
        if self.gpu_optimizer is not None:
            for param_group in self.gpu_optimizer.param_groups:
                param_group['lr'] = lr
        if self.cpu_optimizer is not None:
            for param_group in self.cpu_optimizer.param_groups:
                param_group['lr'] = lr

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict for checkpointing."""
        state = {
            'step_count': self.step_count,
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'score_update_interval': self.score_update_interval,
            'freq_smoothing_alpha': self.freq_smoothing_alpha,
            'quant_group_size': self.quant_group_size,
            'top_bf16_ratio': self.top_bf16_ratio,
            'top_int8_ratio': self.top_int8_ratio,
            'lr_quant': self.lr_quant,
            'initial_precision': self.initial_precision,
            'num_layers': self.num_layers,
            'num_experts_per_layer': self.num_experts_per_layer,
            '_expert_precision': self.quant_pool.expert_precision,
            '_expert_slot_map': dict(self.quant_pool.expert_slot_map),
            '_quant_initialized': set(self.quant_pool._quant_initialized),
            '_expert_scores': self._expert_scores,
            '_call_frequency': self._call_frequency,
            'pool_stats': self.quant_pool.get_pool_stats(),
        }

        if self.gpu_optimizer is not None:
            state['gpu_optimizer'] = self.gpu_optimizer.state_dict()

        if self.cpu_optimizer is not None:
            state['cpu_optimizer'] = self.cpu_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict from checkpoint."""
        self.step_count = state_dict['step_count']
        self.lr = state_dict.get('lr', self.lr)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.score_update_interval = state_dict.get('score_update_interval', self.score_update_interval)
        self.freq_smoothing_alpha = state_dict.get('freq_smoothing_alpha', self.freq_smoothing_alpha)
        self.quant_group_size = state_dict.get('quant_group_size', self.quant_group_size)
        self.top_bf16_ratio = state_dict.get('top_bf16_ratio', self.top_bf16_ratio)
        self.top_int8_ratio = state_dict.get('top_int8_ratio', self.top_int8_ratio)
        self.lr_quant = state_dict.get('lr_quant', self.lr_quant)
        self.initial_precision = state_dict.get('initial_precision', self.initial_precision)

        # Load pool state
        expert_precision = state_dict.get('_expert_precision', {})
        expert_slot_map = state_dict.get('_expert_slot_map', {})
        quant_initialized = state_dict.get('_quant_initialized', set())
        self._expert_scores = state_dict.get('_expert_scores', {})
        self._call_frequency = state_dict.get('_call_frequency', {})

        # Restore pool state
        for global_expert_id, precision in expert_precision.items():
            self.quant_pool.expert_precision[int(global_expert_id)] = precision

        for global_expert_id, slot_info in expert_slot_map.items():
            precision, slot_idx = slot_info
            self.quant_pool.expert_slot_map[int(global_expert_id)] = (precision, slot_idx)
            # Update free slots
            if precision == 8:
                self.quant_pool.int8_free_slots.discard(slot_idx)
            elif precision == 4:
                self.quant_pool.int4_free_slots.discard(slot_idx)

        # Restore quant initialized state
        for global_expert_id in quant_initialized:
            self.quant_pool._quant_initialized.add(int(global_expert_id))

        if self.gpu_optimizer is not None and 'gpu_optimizer' in state_dict:
            self.gpu_optimizer.load_state_dict(state_dict['gpu_optimizer'])

        if self.cpu_optimizer is not None and 'cpu_optimizer' in state_dict:
            self.cpu_optimizer.load_state_dict(state_dict['cpu_optimizer'])