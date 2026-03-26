# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import copy
import ctypes
import logging
import multiprocessing.shared_memory as shm
from dataclasses import dataclass
from copy import deepcopy
from functools import partial
from math import ceil
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from torch.nn.parameter import Parameter

from megatron.core import tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.jit import jit_fuser
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import MLP, MLPSubmodules, apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_align_size_for_quantization,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_object_for_checkpoint,
    sharded_state_dict_default,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding

    HAVE_TE = True

except ImportError:

    HAVE_TE = False

logger = logging.getLogger(__name__)


# CUDA Host Register utilities for pinning existing CPU tensors (e.g., shared memory)
try:
    _cudart = ctypes.CDLL('libcudart.so')
except OSError:
    try:
        _cudart = ctypes.CDLL('libcudart.so.12')
    except OSError:
        _cudart = None

_CUDA_SUCCESS = 0
_CUDA_HOST_REGISTER_DEFAULT = 0x00


def pin_existing_tensor(tensor: torch.Tensor) -> bool:
    """Pin an existing CPU tensor in-place using cudaHostRegister.

    This allows shared memory tensors (created via torch.frombuffer) to be
    used directly for async DMA transfers to GPU, eliminating the need for
    an intermediate copy to pinned memory.

    Args:
        tensor: A CPU tensor to pin in-place

    Returns:
        True if successfully pinned or already pinned

    Raises:
        RuntimeError: If cudaHostRegister fails
    """
    if tensor.device.type != 'cpu':
        return True
    if tensor.is_pinned():
        return True
    if _cudart is None:
        raise RuntimeError("CUDA runtime library not found")

    ptr = tensor.data_ptr()
    size = tensor.element_size() * tensor.nelement()

    ret = _cudart.cudaHostRegister(
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(size),
        ctypes.c_uint(_CUDA_HOST_REGISTER_DEFAULT)
    )

    if ret != _CUDA_SUCCESS:
        raise RuntimeError(f"cudaHostRegister failed with error code {ret}")

    return True


def unpin_existing_tensor(tensor: torch.Tensor) -> None:
    """Unpin a tensor that was pinned with cudaHostRegister.

    Args:
        tensor: A CPU tensor to unpin
    """
    if tensor.device.type != 'cpu':
        return
    if _cudart is None:
        return

    ptr = tensor.data_ptr()
    _cudart.cudaHostUnregister(ctypes.c_void_p(ptr))


@dataclass
class _ExpertWeightCacheEntry:
    param: Parameter
    cpu_buffer: torch.Tensor
    grad_buffer: Optional[torch.Tensor]
    device: Optional[torch.device]
    is_loaded: bool = False
    hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None


class _ExpertWeightCacheContext(contextlib.AbstractContextManager):
    def __init__(self, cache: "ExpertWeightCache", training: bool):
        self.cache = cache
        self.training = training

    def __enter__(self):
        self.cache.ensure_on_device(training=self.training)
        return self.cache

    def __exit__(self, exc_type, exc, tb):
        return False


class ExpertWeightCache:
    """Caches expert weights on CPU and swaps them to GPU on demand."""

    def __init__(
        self,
        parameters: Sequence[Parameter],
        enabled: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        parameter_groups: Optional[Sequence[Sequence[Parameter]]] = None,
    ):
        self.enabled = enabled
        self.pg_collection = pg_collection
        self._shm_objs: List[shm.SharedMemory] = []
        
        if not enabled:
            self._params = tuple()
            self._groups = None
            return

        if parameter_groups is not None:
            self._groups = [self._unique_parameters(g) for g in parameter_groups]
            # Flatten for initial processing if needed
            self._params = self._unique_parameters([p for g in parameter_groups for p in g])
        else:
            self._params = self._unique_parameters(parameters)
            self._groups = [self._params]

        self._entries: Optional[List[_ExpertWeightCacheEntry]] = None
        self._param_to_entry_idx = {}

        if not self._params:
            self.enabled = False

    @staticmethod
    def _unique_parameters(parameters: Sequence[Parameter]) -> Tuple[Parameter, ...]:
        seen = set()
        unique: List[Parameter] = []
        for param in parameters:
            if param is None or not isinstance(param, Parameter):
                continue
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            unique.append(param)
        return tuple(unique)

    def activate(self, training: bool) -> contextlib.AbstractContextManager:
        if not self.enabled:
            return contextlib.nullcontext()
        # Default activate loads everything (backwards compatibility)
        return _ExpertWeightCacheContext(self, training)

    def activate_group(
        self, group_idx: int, training: bool, device: Optional[torch.device] = None
    ):
        if not self.enabled:
            return
        self._maybe_initialize_entries()
        if self._entries is None or self._groups is None:
            return

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        for param in self._groups[group_idx]:
            entry_idx = self._param_to_entry_idx[id(param)]
            entry = self._entries[entry_idx]

            if entry.is_loaded and entry.device == device and entry.param.device == device:
                continue

            gpu_tensor = entry.cpu_buffer.to(device, non_blocking=True)
            # Use detach() to prevent autograd from tracking the weight swap
            # We want the weights to be treated as constants in this graph
            # and re-loaded if necessary.
            entry.param.data = gpu_tensor
            entry.device = device
            entry.is_loaded = True

    def release_group(self, group_idx: int, copy_data: bool = True):
        if not self.enabled or self._entries is None or self._groups is None:
            return
        for param in self._groups[group_idx]:
            entry_idx = self._param_to_entry_idx[id(param)]
            entry = self._entries[entry_idx]
            self._offload_entry(entry, copy_data=copy_data)

    def offload_param_grad_to_cpu(self, param: Parameter, grad: torch.Tensor):
        """Copy a CUDA grad tensor to a persistent CPU buffer and attach to param.grad.

        This intentionally avoids autograd hooks. It is meant to be called from a
        custom backward implementation after grads are computed.
        """

        if not self.enabled:
            return
        self._maybe_initialize_entries()
        if self._entries is None:
            return

        entry_idx = self._param_to_entry_idx.get(id(param))
        if entry_idx is None:
            return
        entry = self._entries[entry_idx]

        if entry.grad_buffer is None or entry.grad_buffer.shape != grad.shape:
            # Keep a stable pinned CPU tensor to reduce reallocations.
            entry.grad_buffer = torch.empty_like(grad, device='cpu', pin_memory=True)
        entry.grad_buffer.copy_(grad.detach(), non_blocking=True)
        # Only attach CPU grads once the parameter data is on CPU.
        if entry.param.device.type == 'cpu':
            entry.param.grad = entry.grad_buffer

    def ensure_on_device(self, training: bool):
        if not self.enabled:
            return
        self._maybe_initialize_entries()
        if self._entries is None:
            return
        current_device = None
        if torch.cuda.is_available():
            current_device = torch.device("cuda", torch.cuda.current_device())
        for entry in self._entries:
            if current_device is None:
                entry.param.data = entry.cpu_buffer
                entry.device = entry.cpu_buffer.device
                entry.is_loaded = False
                continue
            if entry.is_loaded and entry.device == current_device:
                continue
            gpu_tensor = entry.cpu_buffer.to(current_device, non_blocking=True)
            entry.param.data = gpu_tensor
            entry.device = current_device
            entry.is_loaded = True

    def release(self):
        if not self.enabled or self._entries is None:
            return
        for entry in self._entries:
            self._offload_entry(entry, copy_data=True)

        ep_group = self.pg_collection.ep if self.pg_collection else None
        is_rank_0 = ep_group is None or torch.distributed.get_rank(ep_group) == 0

        # Clear shared memory objects
        for s in self._shm_objs:
            s.close()
            if is_rank_0:
                try:
                    s.unlink()
                except FileNotFoundError:
                    pass
        self._shm_objs.clear()
        self._entries = None

    def reset_to_cpu(self):
        """Force all cached parameters back to CPU buffers."""
        if not self.enabled:
            return
        self._maybe_initialize_entries()
        if self._entries is None:
            return
        for entry in self._entries:
            entry.param.data = entry.cpu_buffer
            entry.device = entry.cpu_buffer.device
            entry.is_loaded = False

    def _offload_entry(self, entry: _ExpertWeightCacheEntry, copy_data: bool):
        if entry.param.device.type != "cpu":
            if copy_data:
                # Use non_blocking copy with pinned memory for faster GPU->CPU transfer
                entry.cpu_buffer.copy_(entry.param.data.detach(), non_blocking=True)
            entry.param.data = entry.cpu_buffer
            entry.device = entry.cpu_buffer.device
        entry.is_loaded = False

    def _maybe_initialize_entries(self):
        if self._entries is not None:
            return
        if not self._params:
            self._entries = None
            return

        ep_group = self.pg_collection.ep if self.pg_collection else None
        use_shared_memory = ep_group is not None and ep_group.size() > 1

        entries: List[_ExpertWeightCacheEntry] = []
        self._param_to_entry_idx = {}
        for i, param in enumerate(self._params):
            if use_shared_memory:
                # Shared memory approach for EP
                # All ranks in ep_group should share the same memory.
                ep_ranks = torch.distributed.get_process_group_ranks(ep_group)
                base_rank = min(ep_ranks)
                shm_name = f"megatron_moe_cache_p{i}_r{base_rank}"

                size = param.numel() * param.element_size()
                is_rank_0 = (torch.distributed.get_rank(ep_group) == 0)

                if is_rank_0:
                    # Create shared memory
                    try:
                        shm.SharedMemory(name=shm_name).unlink()
                    except FileNotFoundError:
                        pass
                    _shm = shm.SharedMemory(create=True, size=size, name=shm_name)
                
                # Ensure Rank 0 has created the segment before others open it
                torch.distributed.barrier(group=ep_group)
                
                if not is_rank_0:
                    _shm = shm.SharedMemory(name=shm_name)
                
                self._shm_objs.append(_shm)

                cpu_buffer = torch.frombuffer(_shm.buf, dtype=param.dtype).view(param.shape)

                if is_rank_0:
                    cpu_buffer.copy_(param.detach().to('cpu'))

                # Wait for rank 0 to finish copying before anyone uses it
                torch.distributed.barrier(group=ep_group)
            else:
                # Create pinned memory buffer for faster CPU<->GPU transfers
                cpu_buffer = torch.empty(
                    param.shape, dtype=param.dtype, device='cpu', pin_memory=True
                )
                if param.device.type == 'cuda':
                    cpu_buffer.copy_(param.data.detach(), non_blocking=True)
                else:
                    cpu_buffer.copy_(param.data.detach())

            param.data = cpu_buffer
            self._param_to_entry_idx[id(param)] = len(entries)
            entries.append(
                _ExpertWeightCacheEntry(
                    param=param,
                    cpu_buffer=cpu_buffer,
                    grad_buffer=None,
                    device=None,
                )
            )
        self._entries = entries

    def prime_cpu_storage(self):
        """Ensure parameters start on CPU until explicitly loaded.
        
        Uses pinned memory for faster CPU<->GPU transfers during expert weight swapping.
        """
        if not self.enabled:
            return
        self._maybe_initialize_entries()



class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."
        assert (
            config.moe_latent_size is None
        ), "MoE latent projection not supported in GroupedMLP yet."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            @jit_fuser
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func
        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and (self.config.fp8 or self.config.fp4):
            raise ValueError(
                "moe_act recompute for fp8 or fp4 cannot work with the legacy GroupedMLP."
            )

        @jit_fuser
        def activation_func_with_probs(x, probs):
            dtype = x.dtype
            res = self.activation_func(x) * probs
            return res.to(dtype)

        self.activation_func_with_probs = activation_func_with_probs

        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        self.dp_group = pg_collection.expt_dp
        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition, self.config.hidden_size, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1, config.init_method, partition_dim=1, is_expert=True
                )
                _initialize_affine_weight_gpu(
                    self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

        def remove_extra_states_check(self, incompatible_keys):
            """
            Remove _extra_state from unexpected keys.
            These keys are for dist ckpt compatibility with SequentialMLP.
            """
            keys = deepcopy(incompatible_keys.unexpected_keys)
            for key in keys:
                if '_extra_state' in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

        cache_enabled = getattr(self.config, "moe_enable_expert_weight_cache", True)
        self.weight_cache = ExpertWeightCache(
            parameters=tuple(self.parameters()), 
            enabled=cache_enabled, 
            pg_collection=pg_collection,
            parameter_groups=[[self.weight1], [self.weight2]]
        )
        if self.weight_cache.enabled:
            self.weight_cache.prime_cpu_storage()

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the GroupedMLP."""
        assert self.config.bf16, "Currently GroupedMLP for MoE only supports bf16."
        
        # Internal function to handle the actual computation
        # We need this to use activation checkpointing
        def custom_forward(
            permuted_local_hidden_states, 
            tokens_per_expert, 
            permuted_probs
        ):
            if permuted_local_hidden_states.nelement() != 0:
                # Ensure tokens_per_expert is on CPU for Grouped GEMM
                tokens_per_expert_cpu = tokens_per_expert.cpu()
                # Reshape the weights for the grouped GEMMs.
                self.weight_cache.activate_group(
                    0, training=self.training, device=permuted_local_hidden_states.device
                )
                w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
                fc1_output = gg.ops.gmm(
                    permuted_local_hidden_states, w1, tokens_per_expert_cpu, trans_b=False
                )
                self.weight_cache.release_group(0, copy_data=False)

                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                
                self.weight_cache.activate_group(
                    1, training=self.training, device=permuted_local_hidden_states.device
                )
                w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert_cpu, trans_b=False)
                self.weight_cache.release_group(1, copy_data=False)
            else:
                # No token is allocated for local experts.
                assert torch.count_nonzero(tokens_per_expert) == 0

                # Make sure params of experts still have gradients even given zero tokens.
                w1 = self.weight1.view(self.config.hidden_size, -1)
                w2 = self.weight2.view(-1, self.config.hidden_size)
                h = torch.matmul(permuted_local_hidden_states, w1)
                h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)
            
            return fc2_output

        if self.activation_recompute:
            # We use the standard checkpoint tool to re-run the whole forward during backward
            # This ensures weights are swapped back in during backward.
            fc2_output = tensor_parallel.checkpoint(
                custom_forward,
                False, # distribute_saved_activations
                permuted_local_hidden_states,
                tokens_per_expert,
                permuted_probs
            )
        else:
            fc2_output = custom_forward(
                permuted_local_hidden_states,
                tokens_per_expert,
                permuted_probs
            )

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        Maps local expert to global experts.
        The sharded_state_dict for the weight parts are compatible with the SequentialMLP,
        whereas the optimizer states are not due to the limitation from weight transposing.
        That is, for finetuning scenario, the checkpoint is compatible with the SequentialMLP.

        When `singleton_local_shards` metadata flag is True, experts are broken down into
        separate tensors and stored under separate global keys. Additionally, similarly to MLP,
        layers with GLU activations are broken down into separate `w` and `v` tensors.
        """
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()
        dp_rank = self.dp_group.rank()
        num_global_experts = ep_size * self.num_local_experts
        local_expert_indices_offset = ep_rank * self.num_local_experts

        prepend_axis_num = len(sharded_offsets)
        replica_id = (0, 0, dp_rank)

        local_ffn_dim_size = (
            self.weight2.numel() // self.num_local_experts // self.config.hidden_size
        )

        def _break_into_individual_experts(
            experts_ten: torch.Tensor,
            key: str,
            tp_offset: Tuple[int, int, int],
            replica_id: ReplicaId,
        ):
            """Breaks experts into individual tensors and stores them under separate global keys"""
            experts_state = []
            assert len(experts_ten) == self.num_local_experts, (
                experts_ten.shape,
                self.num_local_experts,
            )
            for local_expert_idx, expert_ten in enumerate(experts_ten):
                global_expert_idx = local_expert_indices_offset + local_expert_idx
                expert_key = key.replace(
                    f'{prefix}experts.', f'{prefix}experts.{global_expert_idx}.'
                )
                experts_state.append(
                    ShardedTensor.from_rank_offsets(
                        expert_key,
                        expert_ten.contiguous(),
                        *sharded_offsets,
                        tp_offset,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )
                )
            return experts_state

        @torch.no_grad()
        def sh_ten_build_fn(
            key: str,
            t: torch.Tensor,
            replica_id: ReplicaId,
            flattened_range: Optional[slice],
            tp_axis: int,
            with_glu: bool,
        ):
            # TODO: write a generic implementation to cover both cases with and without GLU
            if tp_axis == 1:
                # weight1
                if with_glu:
                    last_dim_size = local_ffn_dim_size * 2
                else:
                    last_dim_size = local_ffn_dim_size
                real_shape = (self.num_local_experts, self.config.hidden_size, last_dim_size)
            elif tp_axis == 0:
                # weight2
                real_shape = (self.num_local_experts, local_ffn_dim_size, self.config.hidden_size)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if flattened_range is None:
                # weights
                t = t.view(real_shape).transpose(-1, -2)
                # change tp_axis due to the transposing
                tp_axis = 1 - tp_axis
                if with_glu:
                    assert tp_axis == 0, tp_axis
                    if singleton_local_shards:
                        w_tensor, v_tensor = torch.chunk(t, 2, -2)
                        w_key = f'{key}_w'
                        v_key = f'{key}_v'
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': {
                                'w': _break_into_individual_experts(
                                    w_tensor,
                                    w_key,
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                                'v': _break_into_individual_experts(
                                    v_tensor,
                                    v_key,
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                            },
                        }
                    else:
                        local_tensors = torch.chunk(t, 2, -2)
                        sub_states = [
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[0].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[1].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_size + tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                        ]
                else:
                    if singleton_local_shards:
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': _break_into_individual_experts(
                                t, key, (prepend_axis_num + tp_axis, tp_rank, tp_size), replica_id
                            ),
                        }
                    else:
                        sub_states = ShardedTensor.from_rank_offsets(
                            key,
                            t.contiguous(),
                            *sharded_offsets,
                            (prepend_axis_num, ep_rank, ep_size),
                            (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                        )
            return sub_states  # pylint: disable=possibly-used-before-assignment

        @torch.no_grad()
        def sh_ten_merge_fn(sub_state_dict, tp_axis: int, with_glu: bool):
            if tp_axis == 1:
                # weight1
                weight_shape = (self.config.hidden_size, -1)
            elif tp_axis == 0:
                # weight2
                weight_shape = (-1, self.config.hidden_size)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if isinstance(sub_state_dict, dict):
                assert sub_state_dict['singleton_local_shards']
                if with_glu:
                    assert isinstance(sub_state_dict['data'], dict)
                    sub_state_dict = torch.cat(
                        (
                            torch.stack(sub_state_dict['data']['w']),
                            torch.stack(sub_state_dict['data']['v']),
                        ),
                        dim=-2,
                    )
                else:
                    assert isinstance(sub_state_dict['data'], list)
                    sub_state_dict = torch.stack(sub_state_dict['data'])
            else:
                if with_glu:
                    sub_state_dict = torch.cat(sub_state_dict, -2)
            return sub_state_dict.transpose(-1, -2).reshape(weight_shape)

        state_dict = self.state_dict(prefix='', keep_vars=True)
        for name, tensor in state_dict.items():
            if name == 'weight1':
                tp_axis = 1
                with_glu = self.config.gated_linear_unit
                wkey = f'{prefix}experts.linear_fc1.weight'
            else:
                tp_axis = 0
                with_glu = False
                wkey = f'{prefix}experts.linear_fc2.weight'

            this_replica_id = list(copy.deepcopy(replica_id))

            sharded_state_dict[f'{prefix}{name}'] = ShardedTensorFactory(
                wkey,
                tensor,
                partial(sh_ten_build_fn, tp_axis=tp_axis, with_glu=with_glu),
                partial(sh_ten_merge_fn, tp_axis=tp_axis, with_glu=with_glu),
                tuple(this_replica_id),
            )

        replica_id = (0, tp_rank, dp_rank)
        # Add fake _extra_state to be compatible with SequentialMLP
        for expert_local_idx in range(self.num_local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            if singleton_local_shards:
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )
            for mod in ['linear_fc1', 'linear_fc2']:
                if singleton_local_shards:
                    expert_key = f'{prefix}experts.{expert_global_idx}.{mod}._extra_state'
                else:
                    expert_key = f'{prefix}experts.{mod}._extra_state'
                sharded_state_dict[f'{prefix}expert{expert_global_idx}.{mod}._extra_state'] = (
                    make_sharded_object_for_checkpoint(
                        None, expert_key, expert_sharded_offsets, replica_id
                    )
                )

        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in Experts.
        Empty implementation for compatibility with SequentialMLP and TEGroupedMLP.
        """
        pass


class TEGroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size
        assert not (
            self.config.add_bias_linear and config.bias_dropout_fusion
        ), "bias_dropout_fusion is not supported in TEGroupedMLP when add_bias_linear=True"

        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.moe_ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.num_local_experts,
            self.input_size if self.config.moe_latent_size is None else self.config.moe_latent_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=True,
            tp_comm_buffer_name='fc1',
            pg_collection=pg_collection,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = build_module(submodules.activation_func, config=self.config)
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.num_local_experts,
            self.config.moe_ffn_hidden_size,
            (
                self.config.hidden_size
                if self.config.moe_latent_size is None
                else self.config.moe_latent_size
            ),
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc2',
            pg_collection=pg_collection,
        )

        self.offload_expert_fc1 = (
            self.config.fine_grained_activation_offloading
            and "expert_fc1" in self.config.offload_modules
        )

        self.offload_moe_act = (
            self.config.fine_grained_activation_offloading
            and "moe_act" in self.config.offload_modules
        )

        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and (self.config.fp8 or self.config.fp4):
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc2)

        # This is to avoid the CPU overhead of multiple d2h copies
        if self.offload_expert_fc1:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc1)

        if self.config.fp8 or self.config.fp4:
            assert HAVE_TE, "FP8 and FP4 requires TE."
            self.quantization_padding = Fp8Padding(self.num_local_experts)
            self.quantization_unpadding = Fp8Unpadding(self.num_local_experts)

    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        if bias_parallel is None:
            return intermediate_parallel
        shape = intermediate_parallel.shape
        return (
            torch.cat(
                [
                    t + b * p
                    for t, b, p in zip(
                        torch.split(intermediate_parallel.view(-1, shape[-1]), tokens_per_expert),
                        bias_parallel,
                        torch.split(permuted_probs, tokens_per_expert),
                    )
                ]
            )
            .view(shape)
            .to(intermediate_parallel.dtype)
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert = tokens_per_expert.tolist()
        if self.config.fp8 or self.config.fp4:
            actual_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        with off_interface(
            self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1"
        ) as permuted_local_hidden_states:
            fc1_output, bias_parallel = self.linear_fc1(
                permuted_local_hidden_states, tokens_per_expert
            )
        if self.offload_expert_fc1:
            fc1_output = off_interface.group_commit(
                fc1_output,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states],
            )

        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            if self.config.use_te_activation_func:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                intermediate_parallel = self.activation_func(intermediate_parallel)
                if permuted_probs is not None:
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * permuted_probs
                    intermediate_parallel = intermediate_parallel.to(original_dtype)
            elif self.config.bias_activation_fusion:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu in TEGroupedMLP."
                    )
            elif (
                self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu
            ):
                assert bias_parallel is None
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )
            else:
                if self.config.gated_linear_unit:

                    def glu(x):
                        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                        if (val := self.config.activation_func_clamp_value) is not None:
                            x_glu = x_glu.clamp(min=None, max=val)
                            x_linear = x_linear.clamp(min=-val, max=val)
                        return self.config.activation_func(x_glu) * (
                            x_linear + self.config.glu_linear_offset
                        )

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
            return intermediate_parallel

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = self.activation_checkpoint.checkpoint(
                    bias_act_func, fc1_output, bias_parallel, permuted_probs
                )
        else:
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = bias_act_func(fc1_output, bias_parallel, permuted_probs)

        output, output_bias = self.linear_fc2(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        # Delay the offload of the moe act until after the linear_fc2 has been computed
        # to make sure the fc1_output is reloaded to GPU before recomputing moe_act.
        if self.offload_moe_act:
            output = off_interface.group_commit(
                output, name="moe_act", forced_released_tensors=[fc1_output]
            )
        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)

        # upad and concat the output
        if self.config.fp8 or self.config.fp4:
            output = self.quantization_unpadding(output, actual_tokens_per_expert)

        output_bias = None

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Maps local expert to global experts.
        The sharded state dict is interchangable with SequentialMLP's.
        """
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(
                module, f'{name}.', sharded_offsets, metadata, tp_group=self.tp_group
            )
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                num_global_experts = self.ep_group.size() * self.num_local_experts
                local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts
                ep_axis = len(sharded_offsets)
                for i in range(self.num_local_experts):
                    if singleton_local_shards:
                        new_sharded_offsets = sharded_offsets
                    else:
                        new_sharded_offsets = (
                            *sharded_offsets,
                            (ep_axis, local_expert_indices_offset + i, num_global_experts),
                        )
                    for k in (f'{name}.weight{i}', f'{name}.bias{i}'):
                        if k in sub_sd:
                            sub_sd[k] = apply_swiglu_sharded_factory(
                                sub_sd[k], new_sharded_offsets, singleton_local_shards
                            )
            if singleton_local_shards:
                replace_prefix_for_sharding(sub_sd, '', f'{prefix}experts.')
            else:
                # Add prefix here to match sequential's keys
                replace_prefix_for_sharding(sub_sd, f'{name}.', f'{prefix}experts.{name}.')
            sharded_state_dict.update({f"{prefix}{k}": v for k, v in sub_sd.items()})
        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in TEGroupedMLP.

        This method executes the backward pass for weight gradients by calling
        backward_dw() on the linear layers in reverse order (fc2 followed by fc1).
        If an error occurs during execution, it is caught and re-raised with a
        descriptive message.
        """
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


class ActivationCache:
    """Cache for input activations on CPU, similar to ExpertWeightCache.

    This class handles offloading MoE input activations to CPU during forward pass
    and loading them back to GPU during backward pass, reducing GPU memory usage.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._cpu_buffer: Optional[torch.Tensor] = None
        self._shape: Optional[Tuple[int, ...]] = None
        self._dtype: Optional[torch.dtype] = None
        # CUDA stream for async D2H transfer (offload to CPU)
        self._offload_stream: Optional[torch.cuda.Stream] = None

    def offload_to_cpu_async(self, tensor: torch.Tensor) -> None:
        """Async copy tensor to pinned CPU memory using dedicated stream.

        This overlaps D2H transfer with GPU compute operations.

        Args:
            tensor: The GPU tensor to offload to CPU.
        """
        if not self.enabled:
            return
        if self._offload_stream is None:
            self._offload_stream = torch.cuda.Stream()
        if self._cpu_buffer is None or self._cpu_buffer.shape != tensor.shape:
            self._cpu_buffer = torch.empty_like(
                tensor.detach(), device='cpu', pin_memory=True
            )
        with torch.cuda.stream(self._offload_stream):
            self._cpu_buffer.copy_(tensor.detach(), non_blocking=True)
        self._shape = tensor.shape
        self._dtype = tensor.dtype

    def wait_offload(self) -> None:
        """Wait for async offload to complete."""
        if not self.enabled or self._offload_stream is None:
            return
        torch.cuda.current_stream().wait_stream(self._offload_stream)

    def offload_to_cpu(self, tensor: torch.Tensor) -> None:
        """Copy tensor to pinned CPU memory (synchronous version for compatibility).

        Args:
            tensor: The GPU tensor to offload to CPU.
        """
        if not self.enabled:
            return
        if self._cpu_buffer is None or self._cpu_buffer.shape != tensor.shape:
            self._cpu_buffer = torch.empty_like(
                tensor.detach(), device='cpu', pin_memory=True
            )
        self._cpu_buffer.copy_(tensor.detach(), non_blocking=True)
        self._shape = tensor.shape
        self._dtype = tensor.dtype

    def load_to_device(self, device: torch.device, non_blocking: bool = True) -> torch.Tensor:
        """Load tensor from CPU to GPU.

        Args:
            device: The target GPU device.
            non_blocking: Whether to perform async transfer.

        Returns:
            The tensor loaded to the specified device.
        """
        if not self.enabled or self._cpu_buffer is None:
            raise RuntimeError("No activation cached")
        return self._cpu_buffer.to(device, non_blocking=non_blocking)

    def clear(self) -> None:
        """Clear the cached activation."""
        self._cpu_buffer = None
        self._shape = None
        self._dtype = None

    @property
    def is_cached(self) -> bool:
        """Check if there is a cached activation."""
        return self._cpu_buffer is not None


class SequentialMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        ctx.self = self

        # Offload activation to CPU if enabled, otherwise save to GPU
        if self.activation_offload:
            self.activation_cache.offload_to_cpu(permuted_local_hidden_states)
            ctx.save_for_backward(tokens_per_expert, permuted_probs)
            ctx.activation_offloaded = True
        else:
            ctx.save_for_backward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)
            ctx.activation_offloaded = False

        tokens_per_expert_list = tokens_per_expert.tolist()
        tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert_list)
        probs_list = torch.split(permuted_probs, tokens_per_expert_list)
        output_local_list = []
        nvtx.range_push("SequentialMLP::forward")
        for expert_idx, (expert, tokens, probs) in enumerate(
            zip(self.local_experts, tokens_list, probs_list)
        ):
            if tokens.numel() == 0:
                output_local_list.append(tokens)
                continue
            # Swap only this expert's weights onto GPU for compute.
            nvtx.range_push(f"expert_{expert_idx}_activate")
            self.weight_cache.activate_group(
                expert_idx, training=self.training, device=tokens.device
            )
            nvtx.range_pop()
            if any(p.device != tokens.device for p in expert.parameters()):
                raise RuntimeError("Expert weights not on expected device after activate_group")
            if self.config.fp8 or self.config.fp4:
                hidden, probs = SequentialMLP._pad_tensor_for_quantization(tokens, probs, self.config)
                nvtx.range_push(f"expert_{expert_idx}_compute")
                output, output_bias = expert(hidden, probs)
                nvtx.range_pop()
                output = output[: tokens.shape[0]]
            else:
                nvtx.range_push(f"expert_{expert_idx}_compute")
                output, output_bias = expert(tokens, probs)
                nvtx.range_pop()
            # Release weights immediately after this expert finishes.
            nvtx.range_push(f"expert_{expert_idx}_release")
            self.weight_cache.release_group(expert_idx, copy_data=False)
            nvtx.range_pop()
            output_local_list.append(output)

        output_local = torch.cat(output_local_list, dim=0)
        nvtx.range_pop()
        output_bias_local = None
        return output_local, output_bias_local

    @staticmethod
    def backward(ctx, grad_output, grad_bias):
        self = ctx.self

        # Load activation from CPU if it was offloaded, otherwise use saved tensor
        if ctx.activation_offloaded:
            tokens_per_expert, permuted_probs = ctx.saved_tensors
            permuted_local_hidden_states = self.activation_cache.load_to_device(
                grad_output.device, non_blocking=True
            )
        else:
            permuted_local_hidden_states, tokens_per_expert, permuted_probs = ctx.saved_tensors

        tokens_per_expert_list = tokens_per_expert.tolist()
        tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert_list)
        probs_list = torch.split(permuted_probs, tokens_per_expert_list)
        grad_list = torch.split(grad_output, tokens_per_expert_list)

        grad_input_list: List[torch.Tensor] = []
        nvtx.range_push("SequentialMLP::backward")
        for expert_idx, (expert, tokens, probs, grad) in enumerate(
            zip(self.local_experts, tokens_list, probs_list, grad_list)
        ):
            if tokens.numel() == 0:
                grad_input_list.append(tokens)
                continue
            # Swap only this expert's weights onto GPU for backward.
            nvtx.range_push(f"expert_{expert_idx}_activate")
            self.weight_cache.activate_group(
                expert_idx, training=True, device=tokens.device
            )
            nvtx.range_pop()
            if any(p.device != tokens.device for p in expert.parameters()):
                raise RuntimeError("Expert weights not on expected device after activate_group")

            nvtx.range_push(f"expert_{expert_idx}_bwd_compute")
            with torch.enable_grad():
                tokens_req = tokens.detach().requires_grad_(True)
                probs_const = probs.detach()
                if self.config.fp8 or self.config.fp4:
                    hidden, probs_pad = SequentialMLP._pad_tensor_for_quantization(
                        tokens_req, probs_const, self.config
                    )
                    output, _ = expert(hidden, probs_pad)
                    output = output[: tokens.shape[0]]
                else:
                    output, _ = expert(tokens_req, probs_const)

                params = tuple(expert.parameters())
                grads = torch.autograd.grad(
                    outputs=output,
                    inputs=(tokens_req,) + params,
                    grad_outputs=grad,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

            grad_tokens = grads[0]
            grad_params = grads[1:]
            nvtx.range_pop()

            # Release weights first so params are back on CPU before attaching CPU grads.
            nvtx.range_push(f"expert_{expert_idx}_release")
            self.weight_cache.release_group(expert_idx, copy_data=False)

            # Offload expert parameter grads to CPU.
            for p, g in zip(params, grad_params):
                if g is None:
                    continue
                self.weight_cache.offload_param_grad_to_cpu(p, g)
            nvtx.range_pop()

            grad_input_list.append(grad_tokens)

        grad_input = torch.cat(grad_input_list, dim=0)
        nvtx.range_pop()
        return None, grad_input, None, None  # self, permuted_local_hidden_states, tokens_per_expert, permuted_probs


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):

        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            super().__init__(config=config)
        else:
            # Local SequentialMLP can still be used here by overriding the ffn_hidden_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.ffn_hidden_size = config.moe_ffn_hidden_size
            super().__init__(config=sequential_mlp_config)

        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        # TODO (Hepteract): expt_dp wont be needed here once distributed checkpoint is refactored
        self.dp_group = pg_collection.expt_dp

        for _ in range(self.num_local_experts):
            expert = MLP(
                self.config,
                submodules,
                ffn_hidden_size=self.config.moe_ffn_hidden_size,
                is_expert=True,
                tp_group=pg_collection.expt_tp,
            )
            self.local_experts.append(expert)
        cache_enabled = getattr(self.config, "moe_enable_expert_weight_cache", True)

        # Group cache entries by expert so we can swap/offload one expert at a time.
        expert_param_groups = [list(expert.parameters()) for expert in self.local_experts]
        self.weight_cache = ExpertWeightCache(
            parameters=tuple(self.parameters()),
            enabled=cache_enabled,
            pg_collection=pg_collection,
            parameter_groups=expert_param_groups,
        )
        if self.weight_cache.enabled:
            self.weight_cache.prime_cpu_storage()

        # Add activation offload support
        self.activation_offload = (
            getattr(self.config, "moe_activation_offload", False)
            and cache_enabled
        )
        self.activation_cache = ActivationCache(enabled=self.activation_offload)

    @staticmethod
    def _pad_tensor_for_quantization(hidden, probs, config):
        """Padding tensor shape to multiples of 16/32."""
        actual_num_tokens = hidden.shape[0]
        divisor = get_align_size_for_quantization(config)
        padded_num_tokens = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if padded_num_tokens > 0:
            pad_tensor = torch.zeros(
                padded_num_tokens, hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            hidden = torch.cat((hidden, pad_tensor), dim=0)
            pad_probs = torch.zeros(padded_num_tokens, dtype=probs.dtype, device=probs.device)
            probs = torch.cat((probs, pad_probs), dim=0)
        return hidden, probs

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the SequentialMLP."""

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        return SequentialMLPFunction.apply(
            self, permuted_local_hidden_states, tokens_per_expert, permuted_probs
        )

    def release_weights(self):
        """Manually release expert weights back to CPU."""
        for expert in self.local_experts:
            expert.cpu()

    def delete_expert_weights(self, expert_idx):
        """Manually delete a specific expert's weights from GPU."""
        if 0 <= expert_idx < len(self.local_experts):
            expert = self.local_experts[expert_idx]
            expert.cpu()  # Moves weights back to CPU, effectively deleting GPU copy

    def backward_dw(self):
        """Backward pass for weight gradients in SequentialMLP."""
        for expert in self.local_experts:
            expert.backward_dw()

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Maps local expert to global experts."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        num_global_experts = self.ep_group.size() * self.num_local_experts
        local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)

        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            if singleton_local_shards:
                expert_sharded_prefix = f'{prefix}experts.{expert_global_idx}.'
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_prefix = f'{prefix}experts.'
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'

                sh_ten.replica_id = (*replica_id[:2], self.dp_group.rank())

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict

class _GlobalBufferManager:
    """Global Buffer Manager for CacheGroupedMLP - all layers share the same buffers.

    Since Transformer layers are executed sequentially, there's no need for each layer
    to have its own set of pinned buffers and GPU workspaces. This singleton manager
    provides shared buffers that all CacheGroupedMLP instances can reuse.
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
        dtype: torch.dtype,
        device: torch.device,
        max_experts_per_set: Optional[int] = None,
    ):
        """Initialize global shared buffers. Creates new buffers if not initialized
        or if current buffers are smaller than needed.

        Args:
            max_experts_per_set: Maximum number of experts in a single set. Defaults to
                num_global_experts if not specified.
        """
        # Default max_experts_per_set to num_global_experts for safety
        if max_experts_per_set is None:
            max_experts_per_set = num_global_experts

        # Check if we need to (re)allocate buffers
        need_reallocate = False
        if not self._initialized:
            need_reallocate = True
        else:
            # Check if current buffers are large enough
            if (self._w1_gpu_workspace is None or
                self._w1_gpu_workspace.shape[2] < fc1_out_features or
                self._w1_gpu_workspace.shape[1] < max_experts_per_set):
                need_reallocate = True

        if not need_reallocate:
            return

        # GPU workspace (double-buffered for async prefetch)
        # Shape: [2 buffers, max_experts_per_set, ...]
        # Note: H2D pinned buffers are no longer needed - shared memory is pinned
        # directly via cudaHostRegister, allowing direct DMA transfer
        self._w1_gpu_workspace = torch.empty(
            2, max_experts_per_set, hidden_size, fc1_out_features,
            dtype=dtype, device=device
        )
        self._w2_gpu_workspace = torch.empty(
            2, max_experts_per_set, ffn_hidden_size, hidden_size,
            dtype=dtype, device=device
        )

        # Note: Gradient pinned buffers are no longer needed - gradient shared memory
        # is pinned directly via cudaHostRegister

        # CUDA streams (shared across layers) - only create once
        if not self._initialized:
            self._load_stream = torch.cuda.Stream()
            self._grad_offload_stream = torch.cuda.Stream()
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

        # Release buffers (set to None for garbage collection)
        self._w1_gpu_workspace = None
        self._w2_gpu_workspace = None

        self._initialized = False

    @property
    def w1_gpu_workspace(self):
        return self._w1_gpu_workspace

    @property
    def w2_gpu_workspace(self):
        return self._w2_gpu_workspace

    @property
    def load_stream(self):
        return self._load_stream

    @property
    def grad_offload_stream(self):
        return self._grad_offload_stream


# Global singleton instance
_global_buffer_manager = _GlobalBufferManager()


def cleanup_global_buffers():
    """Clean up global shared buffers.

    This should be called after all CacheGroupedMLP layers have finished
    their work (e.g., at the end of training or inference).
    """
    _global_buffer_manager.cleanup()


class CacheGroupedMLP(MegatronModule):
    """An implementation of the Experts layer using GroupedGEMM with expert weight caching.

    This class supports:
    - Only EP (Expert Parallelism), no TP (Tensor Parallelism)
    - Each rank can access all global experts (no local_expert_ids concept)
    - Forward takes expert_sets - multiple expert subsets, processed one by one
    - All expert weights stored in CPU shared memory (EP group shared)
    - Optional activation offload to reduce GPU memory usage

    Simplified design: No ExpertWeightCache dependency, directly manages shared memory.
    """

    def __init__(
        self,
        num_global_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_global_experts = num_global_experts
        gg.assert_grouped_gemm_is_available()

        # No TP support - only EP
        assert config.add_bias_linear == False, ( 
            "bias not supported in Grouped GEMM, please set '--disable-bias-linear' instead."
        )
        assert config.moe_latent_size is None, (
            "MoE latent projection not supported in CacheGroupedMLP."
        )

        self.ep_group = pg_collection.ep if pg_collection else None
        self.ep_size = self.ep_group.size() if self.ep_group else 1
        self.ep_rank = torch.distributed.get_rank(self.ep_group) if self.ep_group else 0

        # Setup activation function
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using CacheGroupedMLP.")

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # Note: We don't use jit_fuser here to avoid device propagation issues
        # with dynamically loaded weights. Instead, the activation is applied
        # directly in the forward pass.
        self.activation_func_with_probs = None  # Set to None, handled in forward

        # Calculate weight shapes
        # weight1: [num_global_experts, hidden_size, ffn_hidden_size * (2 if glu else 1)]
        # weight2: [num_global_experts, ffn_hidden_size, hidden_size]
        hidden_size = self.config.hidden_size
        ffn_hidden_size = self.config.moe_ffn_hidden_size

        fc1_out_features = ffn_hidden_size
        if self.config.gated_linear_unit:
            fc1_out_features *= 2

        # Use shared memory for EP multi-rank, pinned memory for single rank
        use_shm = self.ep_group is not None and self.ep_size > 1
        self._shm_w1: Optional[shm.SharedMemory] = None
        self._shm_w2: Optional[shm.SharedMemory] = None
        self._shm_g1: Optional[shm.SharedMemory] = None
        self._shm_g2: Optional[shm.SharedMemory] = None

        if use_shm:
            # Shared memory approach - rank 0 creates, others attach
            ep_ranks = torch.distributed.get_process_group_ranks(self.ep_group)
            base_rank = min(ep_ranks)
            is_rank_0 = (self.ep_rank == 0)

            # weight1 shared memory
            shm_name_w1 = f"megatron_moe_w1_r{base_rank}"
            size_w1 = num_global_experts * hidden_size * fc1_out_features * config.params_dtype.itemsize

            if is_rank_0:
                # Clean up any existing shared memory
                try:
                    shm.SharedMemory(name=shm_name_w1).unlink()
                except FileNotFoundError:
                    pass
                self._shm_w1 = shm.SharedMemory(create=True, size=size_w1, name=shm_name_w1)

            # Ensure rank 0 has created the segment before others attach
            torch.distributed.barrier(group=self.ep_group)

            if not is_rank_0:
                self._shm_w1 = shm.SharedMemory(name=shm_name_w1)

            weight1_data = torch.frombuffer(
                self._shm_w1.buf, dtype=config.params_dtype
            ).view(num_global_experts, hidden_size, fc1_out_features)

            # weight2 shared memory
            shm_name_w2 = f"megatron_moe_w2_r{base_rank}"
            size_w2 = num_global_experts * ffn_hidden_size * hidden_size * config.params_dtype.itemsize

            if is_rank_0:
                try:
                    shm.SharedMemory(name=shm_name_w2).unlink()
                except FileNotFoundError:
                    pass
                self._shm_w2 = shm.SharedMemory(create=True, size=size_w2, name=shm_name_w2)

            torch.distributed.barrier(group=self.ep_group)

            if not is_rank_0:
                self._shm_w2 = shm.SharedMemory(name=shm_name_w2)

            weight2_data = torch.frombuffer(
                self._shm_w2.buf, dtype=config.params_dtype
            ).view(num_global_experts, ffn_hidden_size, hidden_size)

            # Pin the shared memory tensors directly using cudaHostRegister
            # This allows async DMA transfers without intermediate CPU copy
            pin_existing_tensor(weight1_data)
            pin_existing_tensor(weight2_data)

            # Initialize weights only on rank 0
            if config.perform_initialization and is_rank_0:
                with torch.no_grad():
                    # 获取当前进程的 GPU
                    device = torch.cuda.current_device()
                    
                    for i in range(num_global_experts):
                        # 1. 在 GPU 上临时建一个空的 Tensor
                        temp_w1_gpu = torch.empty_like(weight1_data[i], device=device)
                        temp_w2_gpu = torch.empty_like(weight2_data[i], device=device)
                        
                        # 2. 让 GPU 去做极其耗时的随机数初始化 (瞬间完成)
                        config.init_method(temp_w1_gpu)
                        config.output_layer_init_method(temp_w2_gpu)
                        
                        # 3. 把初始化好的结果快速拷回 CPU 的 Shared Memory
                        weight1_data[i].copy_(temp_w1_gpu)
                        weight2_data[i].copy_(temp_w2_gpu)
                        
                        # GPU 临时显存会在循环进入下一次时自动释放

            # Wait for rank 0 to finish initialization
            torch.distributed.barrier(group=self.ep_group)
        else:
            # Single rank: use pinned memory for faster transfers
            weight1_data = torch.empty(
                num_global_experts, hidden_size, fc1_out_features,
                dtype=config.params_dtype, device='cpu', pin_memory=True
            )
            weight2_data = torch.empty(
                num_global_experts, ffn_hidden_size, hidden_size,
                dtype=config.params_dtype, device='cpu', pin_memory=True
            )

            if config.perform_initialization:
                with torch.no_grad():
                    device = torch.cuda.current_device()
                    for i in range(num_global_experts):
                        temp_w1_gpu = torch.empty_like(weight1_data[i], device=device)
                        temp_w2_gpu = torch.empty_like(weight2_data[i], device=device)
                        
                        config.init_method(temp_w1_gpu)
                        config.output_layer_init_method(temp_w2_gpu)

                        weight1_data[i].copy_(temp_w1_gpu)
                        weight2_data[i].copy_(temp_w2_gpu)

        # Create Parameter (data is in shared memory or pinned memory)
        self.weight1 = Parameter(weight1_data)
        self.weight2 = Parameter(weight2_data)

        # Gradient buffers
        # For shared memory case: create in shared memory and pin via cudaHostRegister
        # For single rank case: use regular pinned memory
        if use_shm:
            # Gradient buffer shapes (same as weights)
            size_g1 = num_global_experts * hidden_size * fc1_out_features * config.params_dtype.itemsize
            size_g2 = num_global_experts * ffn_hidden_size * hidden_size * config.params_dtype.itemsize

            # Create gradient shared memory
            shm_name_g1 = f"megatron_moe_g1_r{base_rank}"
            if is_rank_0:
                try:
                    shm.SharedMemory(name=shm_name_g1).unlink()
                except FileNotFoundError:
                    pass
                self._shm_g1 = shm.SharedMemory(create=True, size=size_g1, name=shm_name_g1)

            torch.distributed.barrier(group=self.ep_group)

            if not is_rank_0:
                self._shm_g1 = shm.SharedMemory(name=shm_name_g1)

            self._grad_weight1 = torch.frombuffer(
                self._shm_g1.buf, dtype=config.params_dtype
            ).view(num_global_experts, hidden_size, fc1_out_features)
            self._grad_weight1.zero_()

            shm_name_g2 = f"megatron_moe_g2_r{base_rank}"
            if is_rank_0:
                try:
                    shm.SharedMemory(name=shm_name_g2).unlink()
                except FileNotFoundError:
                    pass
                self._shm_g2 = shm.SharedMemory(create=True, size=size_g2, name=shm_name_g2)

            torch.distributed.barrier(group=self.ep_group)

            if not is_rank_0:
                self._shm_g2 = shm.SharedMemory(name=shm_name_g2)

            self._grad_weight2 = torch.frombuffer(
                self._shm_g2.buf, dtype=config.params_dtype
            ).view(num_global_experts, ffn_hidden_size, hidden_size)
            self._grad_weight2.zero_()

            # Pin gradient shared memory tensors
            pin_existing_tensor(self._grad_weight1)
            pin_existing_tensor(self._grad_weight2)
        else:
            # Single rank: use pinned memory for faster transfers
            self._grad_weight1: Optional[torch.Tensor] = torch.zeros_like(weight1_data, pin_memory=True)
            self._grad_weight2: Optional[torch.Tensor] = torch.zeros_like(weight2_data, pin_memory=True)

        # Setup activation offload
        cache_enabled = getattr(self.config, "moe_enable_expert_weight_cache", True)
        self.activation_offload = (
            getattr(self.config, "moe_activation_offload", False)
            and cache_enabled
        )
        self.activation_cache = ActivationCache(enabled=self.activation_offload)

        # Initialize and use global shared buffers (shared across all layers)
        device = torch.cuda.current_device()
        _global_buffer_manager.initialize(
            num_global_experts=num_global_experts,
            hidden_size=hidden_size,
            fc1_out_features=fc1_out_features,
            ffn_hidden_size=ffn_hidden_size,
            dtype=config.params_dtype,
            device=device,
            max_experts_per_set=16,
        )

        # Reference global buffers (no new allocation - just get references)
        # Note: H2D pinned buffers removed - shared memory is pinned directly
        self._w1_gpu_workspace = _global_buffer_manager.w1_gpu_workspace
        self._w2_gpu_workspace = _global_buffer_manager.w2_gpu_workspace
        self._load_stream = _global_buffer_manager.load_stream
        self._grad_offload_stream = _global_buffer_manager.grad_offload_stream
        
    def _load_expert_weights(self, expert_ids: List[int], device: torch.device):
        """Load expert weights to GPU synchronously.

        With cudaHostRegister-pinned shared memory, transfers directly from
        pinned SHM to GPU without intermediate CPU copy.
        """
        num_experts = len(expert_ids)

        # Get views of the expert weights directly from pinned shared memory
        w1_shm_view = self.weight1.data[expert_ids]
        w2_shm_view = self.weight2.data[expert_ids]

        # GPU workspace
        w1_gpu = self._w1_gpu_workspace[0, :num_experts]
        w2_gpu = self._w2_gpu_workspace[0, :num_experts]

        # Direct DMA transfer from pinned shared memory to GPU
        w1_gpu.copy_(w1_shm_view, non_blocking=True)
        w2_gpu.copy_(w2_shm_view, non_blocking=True)

        return w1_gpu, w2_gpu

    def _prefetch_expert_weights_async(
        self,
        expert_ids: List[int],
        buffer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Async prefetch expert weights to GPU buffer with double buffering.

        Uses a dedicated CUDA stream for async loading to overlap with compute.
        With cudaHostRegister-pinned shared memory, transfers directly from
        SHM to GPU without intermediate CPU copy.

        Args:
            expert_ids: List of expert IDs to load
            buffer_idx: Buffer index (0 or 1) for double buffering

        Returns:
            Tuple of (w1_gpu, w2_gpu) - views into GPU workspace
        """
        num_experts = len(expert_ids)
        w1_gpu = self._w1_gpu_workspace[buffer_idx, :num_experts]
        w2_gpu = self._w2_gpu_workspace[buffer_idx, :num_experts]
        _global_buffer_manager.compute_events[buffer_idx].wait(self._load_stream)
        with torch.cuda.stream(self._load_stream):
            for i, exp_id in enumerate(expert_ids):
                # 从 Pinned Memory 的 View 中零拷贝拉取数据
                w1_gpu[i].copy_(self.weight1.data[exp_id], non_blocking=True)
                w2_gpu[i].copy_(self.weight2.data[exp_id], non_blocking=True)

        return w1_gpu, w2_gpu

    def _offload_grads_to_cpu(
        self,
        expert_ids: List[int],
        grad_w1: torch.Tensor,
        grad_w2: torch.Tensor,
    ):
        """Offload gradients to CPU efficiently using pinned gradient buffers.

        With cudaHostRegister-pinned shared memory, transfers directly from
        GPU to pinned SHM gradient buffers without intermediate CPU copy.

        Uses a dedicated CUDA stream for async D2H transfer, allowing overlap with
        next expert set's weight prefetch.
        """
        with torch.cuda.stream(self._grad_offload_stream):
            for i, exp_id in enumerate(expert_ids):
                # exp_id 是整数，索引返回的是 Pinned Memory 的 View
                # 这样 copy_ 才能真正写回 Shared Memory，且 non_blocking 才能生效
                # 注意：如果同一个 batch 有多个 set 访问同一个 expert，这里需要改成 add_ (梯度累加)
                self._grad_weight1[exp_id].copy_(grad_w1[i], non_blocking=True)
                self._grad_weight2[exp_id].copy_(grad_w2[i], non_blocking=True)
            # Critical: Lock memory lifetime to prevent GPU memory reuse before copy completes
            # Without this, default stream may reallocate grad_w1/grad_w2 memory while
            # async D2H copy is still in progress, corrupting the data
            grad_w1.record_stream(self._grad_offload_stream)
            grad_w2.record_stream(self._grad_offload_stream)
            

    def sync_gradients(self):
        """Synchronize accumulated gradients to parameters.

        This should be called after backward pass to apply the CPU gradients
        to the parameter's .grad attribute for the optimizer.

        Note:
            With external scheduling (each expert computed by one rank), no
            allreduce is needed. Gradients are directly attached to parameters.
        """
        if self._grad_offload_stream is not None:
            self._grad_offload_stream.synchronize()
        self.weight1.grad = self._grad_weight1
        self.weight2.grad = self._grad_weight2

    def zero_grad(self, set_to_none: bool = False):
        """Clear gradient buffers."""
        super().zero_grad(set_to_none)
        # Zero the gradient buffers
        if self._grad_weight1 is not None:
            self._grad_weight1.zero_()
        if self._grad_weight2 is not None:
            self._grad_weight2.zero_()

    def release(self):
        """Release shared memory resources with proper synchronization.

        This method ensures:
        1. All CUDA streams are synchronized before cleanup
        2. Pinned shared memory tensors are unpinned before closing
        3. All ranks close shared memory before any rank unlinks it
        4. Resources are properly cleaned up to avoid zombies and timeouts
        """
        is_rank_0 = self.ep_group is None or self.ep_rank == 0

        # 1. Synchronize CUDA streams (ensure all GPU operations complete)
        # This prevents CUDA context destruction from forcing long syncs
        if self._load_stream is not None:
            self._load_stream.synchronize()
        if self._grad_offload_stream is not None:
            self._grad_offload_stream.synchronize()

        # 2. Unpin shared memory tensors (weights and gradients)
        # Must be done before closing shared memory
        if self.weight1 is not None and self.weight1.device.type == 'cpu':
            try:
                unpin_existing_tensor(self.weight1.data)
            except Exception:
                pass  # Ignore errors during cleanup
        if self.weight2 is not None and self.weight2.device.type == 'cpu':
            try:
                unpin_existing_tensor(self.weight2.data)
            except Exception:
                pass
        if self._grad_weight1 is not None and self._grad_weight1.device.type == 'cpu':
            try:
                unpin_existing_tensor(self._grad_weight1)
            except Exception:
                pass
        if self._grad_weight2 is not None and self._grad_weight2.device.type == 'cpu':
            try:
                unpin_existing_tensor(self._grad_weight2)
            except Exception:
                pass

        # 3. Close shared memory (all ranks must do this)
        if self._shm_w1 is not None:
            self._shm_w1.close()
            self._shm_w1 = None

        if self._shm_w2 is not None:
            self._shm_w2.close()
            self._shm_w2 = None

        if self._shm_g1 is not None:
            self._shm_g1.close()
            self._shm_g1 = None

        if self._shm_g2 is not None:
            self._shm_g2.close()
            self._shm_g2 = None

        # 4. Barrier to ensure all ranks have closed before unlink
        # This prevents race conditions where rank 0 unlinks while others are still accessing
        if self.ep_group is not None:
            torch.distributed.barrier(group=self.ep_group)

        # 5. Only rank 0 unlinks, after all ranks have closed
        if is_rank_0:
            # Get base_rank for proper naming
            if self.ep_group is not None:
                ep_ranks = torch.distributed.get_process_group_ranks(self.ep_group)
                base_rank = min(ep_ranks)
            else:
                base_rank = self.ep_rank

            for suffix in ['w1', 'w2', 'g1', 'g2']:
                try:
                    shm.SharedMemory(name=f"megatron_moe_{suffix}_r{base_rank}").unlink()
                except FileNotFoundError:
                    pass

        # 6. Clean up gradient buffer references
        self._grad_weight1 = None
        self._grad_weight2 = None

        # Note: Don't clean up shared buffers here - they're managed by _global_buffer_manager
        # and will be cleaned up via cleanup_global_buffers() when all layers are done.

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert_per_set: List[torch.Tensor],
        probs_per_set: List[torch.Tensor],
        expert_sets: List[List[int]],
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass processing expert sets sequentially.

        Args:
            hidden_states: All tokens concatenated [total_tokens, hidden_size]
            tokens_per_expert_per_set: Token count per expert per set
            probs_per_set: Probability per token per set
            expert_sets: List of expert ID lists [[e1,e2], [e3,e4], ...]

        Returns:
            Tuple of (output, None) where output is [total_tokens, hidden_size]
        """
        # Flatten probs_per_set into a single tensor for proper gradient propagation
        # PyTorch autograd.Function cannot correctly propagate gradients through List[Tensor]
        probs_flat = torch.cat([p for p in probs_per_set if p.numel() > 0], dim=0)

        # Compute offsets for each set to slice probs_flat in forward/backward
        probs_offsets = []
        offset = 0
        for p in probs_per_set:
            probs_offsets.append(offset)
            if p.numel() > 0:
                offset += p.numel()

        return CacheGroupedMLPFunction.apply(
            self, hidden_states, tokens_per_expert_per_set, probs_flat, probs_offsets, expert_sets
        )

    def backward_dw(self):
        """Performs backward pass for weight gradients.
        Empty implementation for compatibility.
        """
        pass

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Return sharded state dict for checkpointing."""
        # TODO: Implement checkpoint support if needed
        del prefix, sharded_offsets, metadata  # Unused
        raise NotImplementedError(
            "CacheGroupedMLP does not support checkpoint save/load yet."
        )


class CacheGroupedMLPFunction(torch.autograd.Function):
    """Custom autograd function for CacheGroupedMLP with weight swapping."""

    @staticmethod
    def forward(
        ctx,
        self: CacheGroupedMLP,
        hidden_states: torch.Tensor,
        tokens_per_expert_per_set: List[torch.Tensor],
        probs_flat: torch.Tensor,
        probs_offsets: List[int],
        expert_sets: List[List[int]],
    ):
        ctx.self = self
        ctx.expert_sets = expert_sets
        ctx.num_sets = len(expert_sets)
        ctx.probs_offsets = probs_offsets

        # Offload activation to CPU asynchronously (overlaps with weight loading and compute)
        if self.activation_offload:
            self.activation_cache.offload_to_cpu_async(hidden_states)
            # Don't save hidden_states - it's offloaded to CPU via activation_cache
            ctx.save_for_backward(probs_flat, *tokens_per_expert_per_set)
            ctx.activation_offloaded = True
        else:
            ctx.save_for_backward(hidden_states, probs_flat, *tokens_per_expert_per_set)
            ctx.activation_offloaded = False

        device = hidden_states.device
        output_list = []
        token_offset = 0
        probs_offset = 0  # Track position in probs_flat

        nvtx.range_push("CacheGroupedMLP::forward")

        # Double buffering for async prefetch
        current_buffer = 0
        next_buffer = 1
        num_sets = len(expert_sets)

        # Prefetch first set
        if num_sets > 0 and len(expert_sets[0]) > 0:
            self._prefetch_expert_weights_async(expert_sets[0], current_buffer)
        cpu_tokens_per_expert_list = [t.cpu() for t in tokens_per_expert_per_set]
        for set_idx, expert_ids in enumerate(expert_sets):
            # 获取对应的 CPU tensor
            tokens_per_expert_cpu = cpu_tokens_per_expert_list[set_idx]
            # 【关键修复】：在纯 CPU Tensor 上执行 sum() 和 item()，极速返回，0 阻塞！
            num_tokens = int(tokens_per_expert_cpu.sum().item())

            if num_tokens == 0:
                continue

            # 1. Wait for current set's weight loading to complete
            torch.cuda.current_stream().wait_stream(self._load_stream)

            # 2. Get weights from current buffer
            w1_gpu = self._w1_gpu_workspace[current_buffer, :len(expert_ids)]
            w2_gpu = self._w2_gpu_workspace[current_buffer, :len(expert_ids)]

            # 4. Execute current set's compute (on default stream)
            set_hidden_states = hidden_states[token_offset:token_offset + num_tokens]
            tokens_per_expert_cpu = cpu_tokens_per_expert_list[set_idx]
            # Slice probs from flattened tensor using dynamic offset
            probs_gpu = probs_flat[probs_offset:probs_offset + num_tokens].to(device, non_blocking=True)
            probs_offset += num_tokens

            # GroupedGEMM: fc1
            fc1_output = gg.ops.gmm(
                set_hidden_states, w1_gpu, tokens_per_expert_cpu, trans_b=False
            )

            # Activation with probs
            intermediate = self.activation_func(fc1_output) * probs_gpu.unsqueeze(-1)

            # GroupedGEMM: fc2
            fc2_output = gg.ops.gmm(
                intermediate, w2_gpu, tokens_per_expert_cpu, trans_b=False
            )
            _global_buffer_manager.compute_events[current_buffer].record(torch.cuda.current_stream())
            output_list.append(fc2_output)
            token_offset += num_tokens

            # 3. Start prefetching next set (if exists)
            next_set_idx = set_idx + 1
            if next_set_idx < num_sets and len(expert_sets[next_set_idx]) > 0:
                self._prefetch_expert_weights_async(expert_sets[next_set_idx], next_buffer)

            # 5. Swap buffers
            current_buffer, next_buffer = next_buffer, current_buffer

        nvtx.range_pop()

        output = torch.cat(output_list, dim=0) if output_list else torch.empty(
            0, self.config.hidden_size, device=device, dtype=hidden_states.dtype
        )

        return output, None

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_bias):
        """Backward pass.

        Args:
            ctx: Autograd context
            grad_output: Gradient of output
            grad_bias: Gradient of bias (always None since bias is not supported)
        """
        del grad_bias  # Unused - bias not supported
        self: CacheGroupedMLP = ctx.self
        expert_sets: List[List[int]] = ctx.expert_sets
        num_sets: int = ctx.num_sets
        probs_offsets: List[int] = ctx.probs_offsets

        # Load activation from CPU if offloaded, otherwise use saved tensor
        # Saved tensors structure:
        #   activation_offloaded=True:  [probs_flat, tokens_per_expert_per_set...]
        #   activation_offloaded=False: [hidden_states, probs_flat, tokens_per_expert_per_set...]
        if ctx.activation_offloaded:
            # Wait for async offload to complete before loading back
            self.activation_cache.wait_offload()
            saved_tensors = ctx.saved_tensors
            probs_flat = saved_tensors[0]
            tokens_per_expert_per_set = list(saved_tensors[1:])
            hidden_states = self.activation_cache.load_to_device(
                grad_output.device, non_blocking=True
            )
        else:
            saved_tensors = ctx.saved_tensors
            hidden_states = saved_tensors[0]
            probs_flat = saved_tensors[1]
            tokens_per_expert_per_set = list(saved_tensors[2:])

        device = grad_output.device
        grad_input_list = []
        grad_probs_list = []  # Collect gradient for probs
        token_offset = 0
        grad_offset = 0
        probs_offset = 0  # Track position in probs_flat

        nvtx.range_push("CacheGroupedMLP::backward")

        # Double buffering for async prefetch
        current_buffer = 0
        next_buffer = 1

        # Prefetch first set
        if num_sets > 0 and len(expert_sets[0]) > 0:
            self._prefetch_expert_weights_async(expert_sets[0], current_buffer)
        cpu_tokens_per_expert_list = [t.cpu() for t in tokens_per_expert_per_set]
        for set_idx, expert_ids in enumerate(expert_sets):
            # 获取对应的 CPU tensor
            tokens_per_expert_cpu = cpu_tokens_per_expert_list[set_idx]

            # 【关键修复】：在纯 CPU Tensor 上执行 sum() 和 item()，极速返回，0 阻塞！
            num_tokens = int(tokens_per_expert_cpu.sum().item())

            if num_tokens == 0:
                continue

            # 1. Wait for weight loading to complete
            torch.cuda.current_stream().wait_stream(self._load_stream)

            # 2. Get weights from current buffer
            w1_gpu = self._w1_gpu_workspace[current_buffer, :len(expert_ids)]
            w2_gpu = self._w2_gpu_workspace[current_buffer, :len(expert_ids)]

            # 3. Start prefetching next set (if exists)
            next_set_idx = set_idx + 1
            if next_set_idx < num_sets and len(expert_sets[next_set_idx]) > 0:
                self._prefetch_expert_weights_async(expert_sets[next_set_idx], next_buffer)

            # 4. Execute current set's compute
            set_hidden_states = hidden_states[token_offset:token_offset + num_tokens]
            set_grad_output = grad_output[grad_offset:grad_offset + num_tokens]
            tokens_per_expert_cpu = cpu_tokens_per_expert_list[set_idx]
            # Slice probs from flattened tensor using dynamic offset
            probs_gpu = probs_flat[probs_offset:probs_offset + num_tokens].to(device, non_blocking=True)

            with torch.enable_grad():
                # Detach and require grad for recomputation
                # CRITICAL: All inputs must be detached to break from the original computation graph
                # This prevents memory leak from stale graph history
                set_hidden_states_req = set_hidden_states.detach().requires_grad_(True)
                w1_gpu_req = w1_gpu.detach().requires_grad_(True)
                w2_gpu_req = w2_gpu.detach().requires_grad_(True)
                # Detach probs to compute its gradient for Router
                probs_gpu_req = probs_gpu.detach().requires_grad_(True)

                # Forward
                fc1_output = gg.ops.gmm(
                    set_hidden_states_req, w1_gpu_req, tokens_per_expert_cpu, trans_b=False
                )
                intermediate = self.activation_func(fc1_output) * probs_gpu_req.unsqueeze(-1)
                fc2_output = gg.ops.gmm(
                    intermediate, w2_gpu_req, tokens_per_expert_cpu, trans_b=False
                )

                # Compute gradients - include probs_gpu_req for Router gradient
                grads = torch.autograd.grad(
                    outputs=fc2_output,
                    inputs=(set_hidden_states_req, w1_gpu_req, w2_gpu_req, probs_gpu_req),
                    grad_outputs=set_grad_output,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

                grad_input = grads[0]
                grad_w1 = grads[1]
                grad_w2 = grads[2]
                grad_probs = grads[3]  # Gradient for probs

            # Offload gradients to CPU
            self._offload_grads_to_cpu(expert_ids, grad_w1, grad_w2)

            grad_input_list.append(grad_input)
            grad_probs_list.append(grad_probs)
            token_offset += num_tokens
            grad_offset += num_tokens
            probs_offset += num_tokens

            # Record event before buffer swap to prevent WAR hazard
            # This ensures async prefetch on _load_stream won't overwrite current buffer
            # while backward compute is still using it
            _global_buffer_manager.compute_events[current_buffer].record(torch.cuda.current_stream())

            # 5. Swap buffers
            current_buffer, next_buffer = next_buffer, current_buffer

        nvtx.range_pop()

        if grad_input_list:
            grad_input = torch.cat(grad_input_list, dim=0)
        else:
            grad_input = torch.empty(
                0, self.config.hidden_size, device=device, dtype=grad_output.dtype
            )

        # Concatenate probs gradients
        if grad_probs_list:
            grad_probs_flat = torch.cat(grad_probs_list, dim=0)
        else:
            grad_probs_flat = torch.empty(0, device=device, dtype=grad_output.dtype)

        # Return gradients: (self, hidden_states, tokens_per_expert_per_set, probs_flat, probs_offsets, expert_sets)
        return None, grad_input, None, grad_probs_flat, None, None


class FusedDispatcherCacheGroupedMLP(CacheGroupedMLP):
    """EP-only MoE with Global-Schedule-Based All-to-All Pipeline.

    CRITICAL DESIGN: Schedule-Based Synchronized Pipeline
    - Phase 0: All ranks exchange COMPLETE global schedule (global_expert_sets)
    - Phase 1+: Every rank participates in EVERY all_to_all UNCONDITIONALLY

    Key principle: The set_idx loop acts as a natural global barrier.
    Every rank iterates over the same set_idx values and unconditionally
    calls all_to_all_single, preventing deadlocks.

    Architecture:
    - Phase 0: Exchange global schedule + token distribution
    - Phase 1: For each set_idx, UNCONDITIONAL dispatch -> compute -> combine
    - Phase 2: Backward with UNCONDITIONAL reverse all_to_all

    Key features:
    - Real cross-rank communication via torch.distributed.all_to_all_single
    - Three CUDA streams: compute (default), load (PCIe weight), comm (NVLink all-to-all)
    - Memory-efficient backward: saves only dispatched tokens per set, not full hidden_states
    - NO conditional skips in all_to_all - NCCL handles 0-size tensors correctly
    """

    def __init__(
        self,
        num_global_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            num_global_experts=num_global_experts,
            config=config,
            pg_collection=pg_collection,
        )

        # Import buffer manager
        from megatron.core.transformer.moe.fused_pipeline_buffer import (
            get_fused_pipeline_buffer_manager,
        )
        self._pipeline_buffer_manager = get_fused_pipeline_buffer_manager()

        # Communication stream for all_to_all operations
        self._comm_stream = torch.cuda.Stream()

        # Events for stream synchronization (double-buffered)
        self._comm_events = [torch.cuda.Event() for _ in range(2)]
        self._compute_events = [torch.cuda.Event() for _ in range(2)]
        self._load_events = [torch.cuda.Event() for _ in range(2)]

    def _phase0_exchange_metadata(
        self,
        routing_map: torch.Tensor,
        expert_sets: List[List[int]],
    ) -> Tuple[torch.Tensor, List[List[List[int]]]]:
        """Exchange global schedule and token distribution.

        CRITICAL: This function ensures ALL ranks know the COMPLETE schedule
        of which experts every rank processes at each set_idx.

        Args:
            routing_map: [num_tokens, num_global_experts] boolean routing map
            expert_sets: List of expert ID lists - experts THIS rank processes per set

        Returns:
            global_tokens_distribution: [ep_size, num_global_experts] tensor on CPU
                global_tokens_distribution[ep_rank][expert_id] = number of tokens
                from ep_rank that are routed to expert_id
            global_expert_sets: [ep_size][num_sets][experts] - COMPLETE schedule
                global_expert_sets[ep_rank][set_idx] = list of expert IDs
                that ep_rank will process at set_idx
        """
        num_global_experts = self.num_global_experts
        device = routing_map.device
        num_sets = len(expert_sets)

        # ============ Part 1: All-gather token distribution ============
        # Local token count per expert (on GPU)
        local_tokens_per_expert = routing_map.sum(dim=0).long()  # [num_global_experts]

        # All-gather on GPU FIRST to avoid device-host sync stall
        global_tokens_distribution_gpu = torch.empty(
            self.ep_size, num_global_experts, dtype=torch.long, device=device
        )

        if self.ep_size > 1:
            torch.distributed.all_gather_into_tensor(
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
        # We need to communicate variable-length lists, so we use a fixed-size tensor

        # First, communicate the number of sets (should be same across all ranks)
        num_sets_tensor = torch.tensor([num_sets], dtype=torch.long, device=device)
        if self.ep_size > 1:
            all_num_sets = torch.empty(self.ep_size, dtype=torch.long, device=device)
            torch.distributed.all_gather_into_tensor(
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
            torch.distributed.all_gather_into_tensor(
                all_max_experts, max_experts_tensor, group=self.ep_group
            )
            global_max_experts = int(all_max_experts.max().item())
        else:
            global_max_experts = max_experts_per_set

        # Create fixed-size tensor for expert_sets communication
        # Shape: [ep_size, num_sets, global_max_experts]
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
            torch.distributed.all_gather_into_tensor(
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

        CRITICAL: Uses global_expert_sets to know EXACTLY which experts each rank
        processes at this set_idx, enabling correct cross-rank communication.

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

    def _build_send_buffer_for_set(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        set_idx: int,
        global_expert_sets: List[List[List[int]]],
        global_tokens_distribution: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build send buffer for a specific set_idx.

        CRITICAL: This method handles token-expert pairs, NOT unique tokens.
        A token routed to multiple experts will appear multiple times in the buffer,
        once for each expert it's routed to.

        Extracts and packs tokens (and probs) destined for experts in
        global_expert_sets[ep_rank][set_idx] for each destination rank.

        Args:
            hidden_states: [num_tokens, hidden_size] input tokens
            routing_map: [num_tokens, num_experts] boolean routing map
            probs: [num_tokens, num_experts] routing probabilities
            set_idx: Current set index
            global_expert_sets: [ep_size][num_sets][experts] - complete schedule
            global_tokens_distribution: [ep_size, num_global_experts] token counts

        Returns:
            set_send_buffer: [total_send, hidden_size] packed hidden states (one per token-expert pair)
            set_send_probs: [total_send] packed probabilities (one per token-expert pair)
            set_send_reverse_indices: [total_send] original token indices for each entry
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_size = hidden_states.size(1)

        # Compute splits first
        send_splits, _ = self._compute_splits_for_set(
            set_idx, global_expert_sets, global_tokens_distribution
        )
        total_send = sum(send_splits)

        if total_send == 0:
            return (
                torch.empty(0, hidden_size, dtype=dtype, device=device),
                torch.empty(0, dtype=dtype, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )

        # Allocate output buffers
        set_send_buffer = torch.empty(total_send, hidden_size, dtype=dtype, device=device)
        set_send_probs = torch.empty(total_send, dtype=dtype, device=device)
        set_send_reverse_indices = torch.empty(total_send, dtype=torch.long, device=device)

        # Pack tokens per destination rank, iterating through each expert separately
        # This correctly handles tokens routed to multiple experts (one entry per token-expert pair)
        current_offset = 0
        for dest_rank in range(self.ep_size):
            dest_experts = global_expert_sets[dest_rank][set_idx]

            # Iterate through each expert to correctly count token-expert pairs
            for exp_id in dest_experts:
                # Get tokens specifically for THIS expert
                expert_mask = routing_map[:, exp_id]
                expert_indices = expert_mask.nonzero(as_tuple=True)[0]
                num_tokens_for_expert = len(expert_indices)

                if num_tokens_for_expert > 0:
                    # Copy hidden states for these tokens
                    set_send_buffer[current_offset:current_offset + num_tokens_for_expert] = hidden_states[expert_indices]
                    # Copy probs for these tokens to this specific expert
                    set_send_probs[current_offset:current_offset + num_tokens_for_expert] = probs[expert_indices, exp_id]
                    # Record original token indices for scatter after combine
                    set_send_reverse_indices[current_offset:current_offset + num_tokens_for_expert] = expert_indices

                current_offset += num_tokens_for_expert

        return set_send_buffer, set_send_probs, set_send_reverse_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        expert_sets: List[List[int]],
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass with fused dispatcher + expert compute pipeline."""
        return FusedDispatcherCacheGroupedMLPFunction.apply(
            self, hidden_states, routing_map, probs, expert_sets
        )


class FusedDispatcherCacheGroupedMLPFunction(torch.autograd.Function):
    """Custom autograd function with UNCONDITIONAL all_to_all for deadlock-free pipeline.

    CRITICAL DESIGN PRINCIPLE:
    - The set_idx loop acts as a natural global barrier
    - EVERY rank participates in EVERY all_to_all_single call
    - NO conditional skips based on token counts
    - NCCL correctly handles 0-size tensors and [0,0,...] splits

    Forward flow:
    1. Phase 0: Exchange global schedule + token distribution
    2. For each set_idx:
       - Build send buffer for this set's destination experts
       - UNCONDITIONAL all_to_all DISPATCH (hidden + probs)
       - Wait for load + comm streams
       - Compute GEMM
       - UNCONDITIONAL all_to_all COMBINE
       - Scatter results back to original token positions

    Backward flow (reverse order):
    1. For each set_idx (reversed):
       - UNCONDITIONAL reverse all_to_all for grad_fc2
       - Recompute forward, compute gradients
       - UNCONDITIONAL reverse all_to_all for grad_input
       - Scatter gradients back to original positions
    """

    @staticmethod
    def forward(
        ctx,
        self: 'FusedDispatcherCacheGroupedMLP',
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

        # ========== Initialize output tensor ==========
        output = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)

        # ========== Initialize data lists ==========
        dispatched_tokens_list = []
        dispatched_probs_list = []
        reverse_indices_list = []
        send_splits_list = []
        recv_splits_list = []
        tokens_per_expert_per_set = []

        current_buffer = 0
        next_buffer = 1

        nvtx.range_push("FusedDispatcher::forward")

        # ==================== Prologue: Prepare Set 0 ====================
        if len(expert_sets) > 0:
            set_0_experts = global_expert_sets[self.ep_rank][0]

            # Async load Set 0 weights to current_buffer
            self._prefetch_expert_weights_async(set_0_experts, current_buffer)

            # Prepare Set 0 splits and buffers
            send_splits_0, recv_splits_0 = self._compute_splits_for_set(
                0, global_expert_sets, global_tokens_distribution
            )
            send_splits_list.append(send_splits_0)
            recv_splits_list.append(recv_splits_0)

            total_send_0 = sum(send_splits_0)
            total_recv_0 = sum(recv_splits_0)

            set_send_buffer_0, set_send_probs_0, set_send_reverse_indices_0 = \
                self._build_send_buffer_for_set(
                    hidden_states, routing_map, probs,
                    0, global_expert_sets, global_tokens_distribution
                )

            # Allocate recv buffers for Set 0
            set_recv_buffer_0 = torch.empty(total_recv_0, hidden_size, dtype=dtype, device=device)
            recv_probs_buffer_0 = torch.empty(total_recv_0, dtype=dtype, device=device)

            # Launch Set 0 Dispatch on _comm_stream
            if self.ep_size > 1:
                with torch.cuda.stream(self._comm_stream):
                    self._compute_events[current_buffer].wait(self._comm_stream)
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
                    self._comm_events[current_buffer].record(self._comm_stream)
            else:
                # EP=1: No cross-rank communication, just use send buffers directly
                set_recv_buffer_0 = set_send_buffer_0
                recv_probs_buffer_0 = set_send_probs_0

            # Store Set 0 data as "current" for the first loop iteration
            current_send_buffer = set_send_buffer_0
            current_send_probs = set_send_probs_0
            current_send_reverse_indices = set_send_reverse_indices_0
            current_recv_buffer = set_recv_buffer_0
            current_recv_probs = recv_probs_buffer_0
            current_send_splits = send_splits_0
            current_recv_splits = recv_splits_0
            current_total_send = total_send_0
            current_total_recv = total_recv_0

        # ==================== Main Loop ====================
        for set_idx in range(len(expert_sets)):
            # Get experts for current set
            local_experts = global_expert_sets[self.ep_rank][set_idx]

            # Wait for current buffer to be ready (load + comm from prologue or previous iteration)
            torch.cuda.current_stream().wait_stream(self._load_stream)
            if self.ep_size > 1:
                torch.cuda.current_stream().wait_stream(self._comm_stream)

            # ==================== ASYNC: Prepare Next Set (N+1) ====================
            next_set_idx = set_idx + 1
            if next_set_idx < len(expert_sets):
                next_experts = global_expert_sets[self.ep_rank][next_set_idx]

                # 1. Prefetch N+1 weights to next_buffer
                self._prefetch_expert_weights_async(next_experts, next_buffer)

                # 2. Prepare N+1 splits and buffers
                send_splits_next, recv_splits_next = self._compute_splits_for_set(
                    next_set_idx, global_expert_sets, global_tokens_distribution
                )
                send_splits_list.append(send_splits_next)
                recv_splits_list.append(recv_splits_next)

                total_send_next = sum(send_splits_next)
                total_recv_next = sum(recv_splits_next)

                set_send_buffer_next, set_send_probs_next, set_send_reverse_indices_next = \
                    self._build_send_buffer_for_set(
                        hidden_states, routing_map, probs,
                        next_set_idx, global_expert_sets, global_tokens_distribution
                    )

                # Allocate recv buffers for N+1
                set_recv_buffer_next = torch.empty(total_recv_next, hidden_size, dtype=dtype, device=device)
                recv_probs_buffer_next = torch.empty(total_recv_next, dtype=dtype, device=device)

                # 3. Launch N+1 Dispatch on _comm_stream
                if self.ep_size > 1:
                    with torch.cuda.stream(self._comm_stream):
                        # CRITICAL: Wait for next_buffer to be free (previous iteration finished)
                        self._compute_events[next_buffer].wait(self._comm_stream)
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
                        self._comm_events[next_buffer].record(self._comm_stream)
                else:
                    # EP=1: No cross-rank communication, just use send buffers directly
                    set_recv_buffer_next = set_send_buffer_next
                    recv_probs_buffer_next = set_send_probs_next

            # ==================== Compute Current Set (N) ====================
            num_local_experts = len(local_experts)
            w1_gpu = self._w1_gpu_workspace[current_buffer, :num_local_experts]
            w2_gpu = self._w2_gpu_workspace[current_buffer, :num_local_experts]

            # Compute tokens_per_expert for GEMM
            tokens_per_expert_list = []
            for exp_id in local_experts:
                count = routing_map[:, exp_id].sum().item()
                tokens_per_expert_list.append(count)
            tokens_per_expert = torch.tensor(tokens_per_expert_list, dtype=torch.long)
            tokens_per_expert_per_set.append(tokens_per_expert)

            if current_total_recv > 0 and num_local_experts > 0:
                fc1_output = gg.ops.gmm(
                    current_recv_buffer, w1_gpu, tokens_per_expert, trans_b=False
                )
                intermediate = self.activation_func(fc1_output) * current_recv_probs.unsqueeze(-1)
                fc2_output = gg.ops.gmm(
                    intermediate, w2_gpu, tokens_per_expert, trans_b=False
                )
            else:
                fc2_output = torch.empty(0, hidden_size, dtype=dtype, device=device)

            # Record compute completion (protects next_buffer from being overwritten)
            self._compute_events[current_buffer].record(torch.cuda.current_stream())

            # ==================== Combine Current Set (N) ====================
            set_combine_buffer = torch.empty(current_total_send, hidden_size, dtype=dtype, device=device)
            if self.ep_size > 1:
                with torch.cuda.stream(self._comm_stream):
                    self._compute_events[current_buffer].wait(self._comm_stream)
                    torch.distributed.all_to_all_single(
                        set_combine_buffer, fc2_output,
                        output_split_sizes=current_send_splits,  # REVERSED!
                        input_split_sizes=current_recv_splits,
                        group=self.ep_group
                    )
                    self._comm_events[current_buffer].record(self._comm_stream)
                torch.cuda.current_stream().wait_stream(self._comm_stream)
            else:
                set_combine_buffer = fc2_output

            # ==================== Scatter Results ====================
            if current_send_reverse_indices.numel() > 0 and set_combine_buffer.numel() > 0:
                output.index_add_(0, current_send_reverse_indices, set_combine_buffer)

            # Save for backward
            dispatched_tokens_list.append(current_recv_buffer.detach())
            dispatched_probs_list.append(current_recv_probs.detach())
            reverse_indices_list.append(current_send_reverse_indices.detach())

            # ==================== Swap Buffers for Next Iteration ====================
            current_buffer, next_buffer = next_buffer, current_buffer

            # Update current buffers to point to next set's precomputed data
            if next_set_idx < len(expert_sets):
                current_send_buffer = set_send_buffer_next
                current_send_probs = set_send_probs_next
                current_send_reverse_indices = set_send_reverse_indices_next
                current_recv_buffer = set_recv_buffer_next
                current_recv_probs = recv_probs_buffer_next
                current_send_splits = send_splits_next
                current_recv_splits = recv_splits_next
                current_total_send = total_send_next
                current_total_recv = total_recv_next

        nvtx.range_pop()

        # Save for backward
        ctx.save_for_backward(*dispatched_tokens_list, *dispatched_probs_list, *reverse_indices_list)
        ctx.send_splits_list = send_splits_list
        ctx.recv_splits_list = recv_splits_list
        ctx.tokens_per_expert_per_set = tokens_per_expert_per_set

        return output, None

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_bias):
        """Backward pass with Prologue + Main Loop pipeline for proper overlap."""
        del grad_bias

        self: 'FusedDispatcherCacheGroupedMLP' = ctx.self
        expert_sets: List[List[int]] = ctx.expert_sets
        global_expert_sets: List[List[List[int]]] = ctx.global_expert_sets
        send_splits_list = ctx.send_splits_list
        recv_splits_list = ctx.recv_splits_list
        tokens_per_expert_per_set = ctx.tokens_per_expert_per_set
        hidden_size = ctx.hidden_size
        num_tokens = ctx.num_tokens

        # Retrieve saved tensors
        num_sets = ctx.num_sets
        dispatched_tensors = ctx.saved_tensors[:num_sets]
        dispatched_probs = ctx.saved_tensors[num_sets:2*num_sets]
        reverse_indices_list = ctx.saved_tensors[2*num_sets:]

        device = grad_output.device
        dtype = grad_output.dtype

        # Initialize events
        for i in range(2):
            self._compute_events[i].record(torch.cuda.current_stream())
            self._comm_events[i].record(self._comm_stream)

        # Initialize gradient for input tokens
        grad_input = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)

        current_buffer = 0
        next_buffer = 1

        nvtx.range_push("FusedDispatcher::backward")

        # ==================== Prologue: Prepare Last Set (len-1) ====================
        last_set_idx = len(expert_sets) - 1
        if last_set_idx >= 0:
            last_experts = global_expert_sets[self.ep_rank][last_set_idx]

            # Async load last set weights to current_buffer
            self._prefetch_expert_weights_async(last_experts, current_buffer)

            # Get saved data for last set
            current_dispatched_tokens = dispatched_tensors[last_set_idx]
            current_dispatched_probs = dispatched_probs[last_set_idx]
            current_reverse_indices = reverse_indices_list[last_set_idx]
            current_tokens_per_expert = tokens_per_expert_per_set[last_set_idx]
            current_send_splits = send_splits_list[last_set_idx]
            current_recv_splits = recv_splits_list[last_set_idx]
            current_total_send = sum(current_send_splits)
            current_total_recv = sum(current_recv_splits)

            # Get grad_output for last set's tokens
            current_set_grad_output = grad_output[current_reverse_indices.to(device)] if current_reverse_indices.numel() > 0 else \
                torch.empty(0, hidden_size, dtype=dtype, device=device)

            # Launch last set's REVERSE COMBINE on _comm_stream
            current_grad_fc2 = torch.empty(current_total_recv, hidden_size, dtype=dtype, device=device)
            if self.ep_size > 1:
                with torch.cuda.stream(self._comm_stream):
                    torch.distributed.all_to_all_single(
                        current_grad_fc2, current_set_grad_output,
                        output_split_sizes=current_recv_splits,
                        input_split_sizes=current_send_splits,
                        group=self.ep_group
                    )
                    self._comm_events[current_buffer].record(self._comm_stream)
            else:
                # EP=1: No cross-rank communication, use grad_output directly
                current_grad_fc2 = current_set_grad_output

        # ==================== Main Loop (REVERSE order) ====================
        for set_idx in range(last_set_idx, -1, -1):
            local_experts = global_expert_sets[self.ep_rank][set_idx]

            # Wait for current buffer to be ready
            torch.cuda.current_stream().wait_stream(self._load_stream)
            if self.ep_size > 1:
                torch.cuda.current_stream().wait_stream(self._comm_stream)

            # ==================== ASYNC: Prepare Previous Set (set_idx-1) ====================
            prev_set_idx = set_idx - 1
            if prev_set_idx >= 0:
                prev_experts = global_expert_sets[self.ep_rank][prev_set_idx]

                # 1. Prefetch prev set weights to next_buffer
                self._prefetch_expert_weights_async(prev_experts, next_buffer)

                # 2. Get saved data for prev set
                prev_dispatched_tokens = dispatched_tensors[prev_set_idx]
                prev_dispatched_probs = dispatched_probs[prev_set_idx]
                prev_reverse_indices = reverse_indices_list[prev_set_idx]
                prev_tokens_per_expert = tokens_per_expert_per_set[prev_set_idx]
                prev_send_splits = send_splits_list[prev_set_idx]
                prev_recv_splits = recv_splits_list[prev_set_idx]
                prev_total_send = sum(prev_send_splits)
                prev_total_recv = sum(prev_recv_splits)

                # 3. Get grad_output for prev set's tokens
                prev_set_grad_output = grad_output[prev_reverse_indices.to(device)] if prev_reverse_indices.numel() > 0 else \
                    torch.empty(0, hidden_size, dtype=dtype, device=device)

                # 4. Launch prev set's REVERSE COMBINE on _comm_stream
                prev_grad_fc2 = torch.empty(prev_total_recv, hidden_size, dtype=dtype, device=device)
                if self.ep_size > 1:
                    with torch.cuda.stream(self._comm_stream):
                        # CRITICAL: Wait for next_buffer to be free
                        self._compute_events[next_buffer].wait(self._comm_stream)
                        torch.distributed.all_to_all_single(
                            prev_grad_fc2, prev_set_grad_output,
                            output_split_sizes=prev_recv_splits,
                            input_split_sizes=prev_send_splits,
                            group=self.ep_group
                        )
                        self._comm_events[next_buffer].record(self._comm_stream)
                else:
                    # EP=1: No cross-rank communication, use grad_output directly
                    prev_grad_fc2 = prev_set_grad_output

            # ==================== Compute Current Set ====================
            num_local_experts = len(local_experts)
            w1_gpu = self._w1_gpu_workspace[current_buffer, :num_local_experts]
            w2_gpu = self._w2_gpu_workspace[current_buffer, :num_local_experts]

            # Use current set's data (from prologue or previous iteration)
            if set_idx == last_set_idx:
                dispatched_tokens = current_dispatched_tokens
                dispatched_probs_t = current_dispatched_probs
                tokens_per_expert = current_tokens_per_expert
                grad_fc2 = current_grad_fc2
                total_send = current_total_send
                total_recv = current_total_recv
                send_splits = current_send_splits
                recv_splits = current_recv_splits
                set_reverse_indices = current_reverse_indices
            else:
                dispatched_tokens = dispatched_tensors[set_idx]
                dispatched_probs_t = dispatched_probs[set_idx]
                tokens_per_expert = tokens_per_expert_per_set[set_idx]
                grad_fc2 = current_grad_fc2
                total_send = current_total_send
                total_recv = current_total_recv
                send_splits = current_send_splits
                recv_splits = current_recv_splits
                set_reverse_indices = current_reverse_indices

            # Recompute forward with enable_grad
            if total_recv > 0 and num_local_experts > 0:
                with torch.enable_grad():
                    dispatched_tokens_req = dispatched_tokens.detach().requires_grad_(True)
                    w1_gpu_req = w1_gpu.detach().requires_grad_(True)
                    w2_gpu_req = w2_gpu.detach().requires_grad_(True)
                    probs_req = dispatched_probs_t.detach().requires_grad_(True)

                    fc1 = gg.ops.gmm(dispatched_tokens_req, w1_gpu_req, tokens_per_expert, trans_b=False)
                    intermediate = self.activation_func(fc1) * probs_req.unsqueeze(-1)
                    fc2 = gg.ops.gmm(intermediate, w2_gpu_req, tokens_per_expert, trans_b=False)

                    grads = torch.autograd.grad(
                        fc2, (dispatched_tokens_req, w1_gpu_req, w2_gpu_req, probs_req),
                        grad_outputs=grad_fc2,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )

                    grad_input_local = grads[0] if grads[0] is not None else torch.zeros_like(dispatched_tokens)
                    grad_w1 = grads[1]
                    grad_w2 = grads[2]
            else:
                grad_input_local = torch.empty(0, hidden_size, dtype=dtype, device=device)
                grad_w1 = None
                grad_w2 = None

            # Offload weight gradients
            if grad_w1 is not None and grad_w2 is not None:
                self._offload_grads_to_cpu(local_experts, grad_w1, grad_w2)

            # Record compute completion
            self._compute_events[current_buffer].record(torch.cuda.current_stream())

            # ==================== REVERSE DISPATCH ====================
            grad_input_remote = torch.empty(total_send, hidden_size, dtype=dtype, device=device)
            if self.ep_size > 1:
                with torch.cuda.stream(self._comm_stream):
                    self._compute_events[current_buffer].wait(self._comm_stream)
                    torch.distributed.all_to_all_single(
                        grad_input_remote, grad_input_local,
                        output_split_sizes=send_splits,
                        input_split_sizes=recv_splits,
                        group=self.ep_group
                    )
                    self._comm_events[current_buffer].record(self._comm_stream)
                torch.cuda.current_stream().wait_stream(self._comm_stream)
            else:
                grad_input_remote = grad_input_local

            # Scatter gradients back to original token positions
            if set_reverse_indices.numel() > 0 and grad_input_remote.numel() > 0:
                grad_input.index_add_(0, set_reverse_indices.to(device), grad_input_remote)

            # ==================== Swap Buffers ====================
            current_buffer, next_buffer = next_buffer, current_buffer

            # Update current data to prev set's precomputed data
            if prev_set_idx >= 0:
                current_dispatched_tokens = prev_dispatched_tokens
                current_dispatched_probs = prev_dispatched_probs
                current_reverse_indices = prev_reverse_indices
                current_tokens_per_expert = prev_tokens_per_expert
                current_send_splits = prev_send_splits
                current_recv_splits = prev_recv_splits
                current_total_send = prev_total_send
                current_total_recv = prev_total_recv
                current_grad_fc2 = prev_grad_fc2

        nvtx.range_pop()

        return None, grad_input, None, None, None