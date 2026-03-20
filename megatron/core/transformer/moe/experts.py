# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import copy
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
    ):
        """Initialize global shared buffers. Only creates new buffers if not already initialized."""
        if self._initialized:
            return

        # Pinned buffers for H2D transfer (CPU -> GPU async transfer)
        self._w1_h2d_pinned = torch.empty(
            num_global_experts, hidden_size, fc1_out_features,
            dtype=dtype, device='cpu', pin_memory=True
        )
        self._w2_h2d_pinned = torch.empty(
            num_global_experts, ffn_hidden_size, hidden_size,
            dtype=dtype, device='cpu', pin_memory=True
        )

        # GPU workspace (double-buffered for async prefetch)
        # Shape: [2 buffers, max_experts_per_set (8), ...]
        self._w1_gpu_workspace = torch.empty(
            2, 8, hidden_size, fc1_out_features,
            dtype=dtype, device=device
        )
        self._w2_gpu_workspace = torch.empty(
            2, 8, ffn_hidden_size, hidden_size,
            dtype=dtype, device=device
        )

        # Pinned buffers for gradient offload (D2H transfer)
        self._grad_w1_pinned = torch.empty(
            num_global_experts, hidden_size, fc1_out_features,
            dtype=dtype, device='cpu', pin_memory=True
        )
        self._grad_w2_pinned = torch.empty(
            num_global_experts, ffn_hidden_size, hidden_size,
            dtype=dtype, device='cpu', pin_memory=True
        )

        # CUDA streams (shared across layers)
        self._load_stream = torch.cuda.Stream()
        self._grad_offload_stream = torch.cuda.Stream()

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
        self._w1_h2d_pinned = None
        self._w2_h2d_pinned = None
        self._w1_gpu_workspace = None
        self._w2_gpu_workspace = None
        self._grad_w1_pinned = None
        self._grad_w2_pinned = None

        self._initialized = False

    @property
    def w1_h2d_pinned(self):
        return self._w1_h2d_pinned

    @property
    def w2_h2d_pinned(self):
        return self._w2_h2d_pinned

    @property
    def w1_gpu_workspace(self):
        return self._w1_gpu_workspace

    @property
    def w2_gpu_workspace(self):
        return self._w2_gpu_workspace

    @property
    def grad_w1_pinned(self):
        return self._grad_w1_pinned

    @property
    def grad_w2_pinned(self):
        return self._grad_w2_pinned

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

        # Gradient buffers (shared memory for EP, pinned memory for single rank)
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
        )

        # Reference global buffers (no new allocation - just get references)
        self._w1_h2d_pinned_buffer = _global_buffer_manager.w1_h2d_pinned
        self._w2_h2d_pinned_buffer = _global_buffer_manager.w2_h2d_pinned
        self._w1_gpu_workspace = _global_buffer_manager.w1_gpu_workspace
        self._w2_gpu_workspace = _global_buffer_manager.w2_gpu_workspace
        self._grad_w1_pinned_buffer = _global_buffer_manager.grad_w1_pinned
        self._grad_w2_pinned_buffer = _global_buffer_manager.grad_w2_pinned
        self._load_stream = _global_buffer_manager.load_stream
        self._grad_offload_stream = _global_buffer_manager.grad_offload_stream
        
    def _load_expert_weights(self, expert_ids: List[int], device: torch.device):
        num_experts = len(expert_ids)
        
        # 1. CPU -> CPU 聚合（Pageable to Pinned）
        # 将共享内存中散落的专家权重，拷贝到连续的锁页内存中
        # 这一步是 CPU 内部的拷贝，速度极快（受限于内存带宽），且不会阻塞 GPU
        for i, exp_id in enumerate(expert_ids):
            self._w1_h2d_pinned_buffer[i].copy_(self.weight1.data[exp_id])
            self._w2_h2d_pinned_buffer[i].copy_(self.weight2.data[exp_id])

        # 获取当前需要用到的连续内存视图 (View)，不产生实际拷贝
        w1_pinned_view = self._w1_h2d_pinned_buffer[:num_experts]
        w2_pinned_view = self._w2_h2d_pinned_buffer[:num_experts]

        # # 2. 在 GPU 上预分配空间
        # # 注意：如果想极致优化，GPU 端也可以像 Pinned Buffer 一样做预分配和复用，避免每次 empty
        # w1_gpu = torch.empty(num_experts, self.weight1.size(1), self.weight1.size(2), device=device, dtype=self.weight1.dtype)
        # w2_gpu = torch.empty(num_experts, self.weight2.size(1), self.weight2.size(2), device=device, dtype=self.weight2.dtype)
        w1_gpu = self._w1_gpu_workspace[:num_experts]
        w2_gpu = self._w2_gpu_workspace[:num_experts]

        # 3. CPU -> GPU 单次、大块、异步传输 (Pinned to VRAM)
        # 因为源端是真正的锁页内存，这里非阻塞传输会完美生效，nsys 里只会看到一条宽阔的绿带
        w1_gpu.copy_(w1_pinned_view, non_blocking=True)
        w2_gpu.copy_(w2_pinned_view, non_blocking=True)

        return w1_gpu, w2_gpu

    def _prefetch_expert_weights_async(
        self,
        expert_ids: List[int],
        buffer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Async prefetch expert weights to GPU buffer with double buffering.

        Uses a dedicated CUDA stream for async loading to overlap with compute.

        Args:
            expert_ids: List of expert IDs to load
            buffer_idx: Buffer index (0 or 1) for double buffering

        Returns:
            Tuple of (w1_gpu, w2_gpu) - views into GPU workspace
        """
        num_experts = len(expert_ids)

        # 1. CPU -> CPU (pinned memory) - on main thread
        for i, exp_id in enumerate(expert_ids):
            self._w1_h2d_pinned_buffer[i].copy_(self.weight1.data[exp_id])
            self._w2_h2d_pinned_buffer[i].copy_(self.weight2.data[exp_id])

        w1_pinned_view = self._w1_h2d_pinned_buffer[:num_experts]
        w2_pinned_view = self._w2_h2d_pinned_buffer[:num_experts]

        # 2. Async transfer to GPU (on load_stream)
        w1_gpu = self._w1_gpu_workspace[buffer_idx, :num_experts]
        w2_gpu = self._w2_gpu_workspace[buffer_idx, :num_experts]

        with torch.cuda.stream(self._load_stream):
            w1_gpu.copy_(w1_pinned_view, non_blocking=True)
            w2_gpu.copy_(w2_pinned_view, non_blocking=True)

        return w1_gpu, w2_gpu

    def _offload_grads_to_cpu(
        self,
        expert_ids: List[int],
        grad_w1: torch.Tensor,
        grad_w2: torch.Tensor,
    ):
        """Offload gradients to CPU efficiently using pre-allocated Pinned Memory buffer.

        Uses a dedicated CUDA stream for async D2H transfer, allowing overlap with
        next expert set's weight prefetch.
        """
        num_experts = len(expert_ids)

        # Use pre-allocated pinned buffers
        grad_w1_pinned = self._grad_w1_pinned_buffer[:num_experts]
        grad_w2_pinned = self._grad_w2_pinned_buffer[:num_experts]

        # Async D2H on dedicated stream (can overlap with weight prefetch)
        with torch.cuda.stream(self._grad_offload_stream):
            grad_w1_pinned.copy_(grad_w1, non_blocking=True)
            grad_w2_pinned.copy_(grad_w2, non_blocking=True)

        # Wait for D2H to complete before CPU reads
        torch.cuda.current_stream().wait_stream(self._grad_offload_stream)

        # CPU -> CPU copy from pinned buffer to shared memory
        for i, expert_id in enumerate(expert_ids):
            self._grad_weight1[expert_id].copy_(grad_w1_pinned[i])
            self._grad_weight2[expert_id].copy_(grad_w2_pinned[i])
            

    def sync_gradients(self):
        """Synchronize accumulated gradients to parameters.

        This should be called after backward pass to apply the CPU gradients
        to the parameter's .grad attribute for the optimizer.

        Note:
            With external scheduling (each expert computed by one rank), no
            allreduce is needed. Gradients are directly attached to parameters.
        """
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
        2. All ranks close shared memory before any rank unlinks it
        3. Resources are properly cleaned up to avoid zombies and timeouts
        """
        is_rank_0 = self.ep_group is None or self.ep_rank == 0

        # 1. Synchronize CUDA streams (ensure all GPU operations complete)
        # This prevents CUDA context destruction from forcing long syncs
        if self._load_stream is not None:
            self._load_stream.synchronize()
        if self._grad_offload_stream is not None:
            self._grad_offload_stream.synchronize()

        # 2. Close shared memory (all ranks must do this)
        if self._shm_w1 is not None:
            self._shm_w1.close()
            self._shm_w1 = None

        if self._shm_w2 is not None:
            self._shm_w2.close()
            self._shm_w2 = None

        # 3. Barrier to ensure all ranks have closed before unlink
        # This prevents race conditions where rank 0 unlinks while others are still accessing
        if self.ep_group is not None:
            torch.distributed.barrier(group=self.ep_group)

        # 4. Only rank 0 unlinks, after all ranks have closed
        if is_rank_0:
            try:
                shm.SharedMemory(name=f"megatron_moe_w1_r{self.ep_rank}").unlink()
            except FileNotFoundError:
                pass
            try:
                shm.SharedMemory(name=f"megatron_moe_w2_r{self.ep_rank}").unlink()
            except FileNotFoundError:
                pass

        # 5. Clean up gradient buffer references
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
        return CacheGroupedMLPFunction.apply(
            self, hidden_states, tokens_per_expert_per_set, probs_per_set, expert_sets
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
        probs_per_set: List[torch.Tensor],
        expert_sets: List[List[int]],
    ):
        ctx.self = self
        ctx.expert_sets = expert_sets
        ctx.num_sets = len(expert_sets)

        # Offload activation to CPU asynchronously (overlaps with weight loading and compute)
        if self.activation_offload:
            self.activation_cache.offload_to_cpu_async(hidden_states)
            ctx.save_for_backward(*tokens_per_expert_per_set, *probs_per_set)
            ctx.activation_offloaded = True
        else:
            ctx.save_for_backward(hidden_states, *tokens_per_expert_per_set, *probs_per_set)
            ctx.activation_offloaded = False

        device = hidden_states.device
        output_list = []
        token_offset = 0

        nvtx.range_push("CacheGroupedMLP::forward")

        # Double buffering for async prefetch
        current_buffer = 0
        next_buffer = 1
        num_sets = len(expert_sets)

        # Prefetch first set
        if num_sets > 0 and len(expert_sets[0]) > 0:
            self._prefetch_expert_weights_async(expert_sets[0], current_buffer)

        for set_idx, expert_ids in enumerate(expert_sets):
            tokens_per_expert = tokens_per_expert_per_set[set_idx]
            probs = probs_per_set[set_idx]
            num_tokens = int(tokens_per_expert.sum().item())

            if num_tokens == 0:
                continue

            # 1. Wait for current set's weight loading to complete
            torch.cuda.current_stream().wait_stream(self._load_stream)

            # 2. Get weights from current buffer
            w1_gpu = self._w1_gpu_workspace[current_buffer, :len(expert_ids)]
            w2_gpu = self._w2_gpu_workspace[current_buffer, :len(expert_ids)]

            # 3. Start prefetching next set (if exists)
            next_set_idx = set_idx + 1
            if next_set_idx < num_sets and len(expert_sets[next_set_idx]) > 0:
                self._prefetch_expert_weights_async(expert_sets[next_set_idx], next_buffer)

            # 4. Execute current set's compute (on default stream)
            set_hidden_states = hidden_states[token_offset:token_offset + num_tokens]
            tokens_per_expert_cpu = tokens_per_expert.cpu()
            probs_gpu = probs.to(device, non_blocking=True)

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

            output_list.append(fc2_output)
            token_offset += num_tokens

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

        # Load activation from CPU if offloaded, otherwise use saved tensor
        if ctx.activation_offloaded:
            # Wait for async offload to complete before loading back
            self.activation_cache.wait_offload()
            saved_tensors = ctx.saved_tensors
            tokens_per_expert_per_set = list(saved_tensors[:num_sets])
            probs_per_set = list(saved_tensors[num_sets:2*num_sets])
            hidden_states = self.activation_cache.load_to_device(
                grad_output.device, non_blocking=True
            )
        else:
            saved_tensors = ctx.saved_tensors
            hidden_states = saved_tensors[0]
            tokens_per_expert_per_set = list(saved_tensors[1:num_sets+1])
            probs_per_set = list(saved_tensors[num_sets+1:2*num_sets+1])

        device = grad_output.device
        grad_input_list = []
        token_offset = 0
        grad_offset = 0

        nvtx.range_push("CacheGroupedMLP::backward")

        # Double buffering for async prefetch
        current_buffer = 0
        next_buffer = 1

        # Prefetch first set
        if num_sets > 0 and len(expert_sets[0]) > 0:
            self._prefetch_expert_weights_async(expert_sets[0], current_buffer)

        for set_idx, expert_ids in enumerate(expert_sets):
            tokens_per_expert = tokens_per_expert_per_set[set_idx]
            probs = probs_per_set[set_idx]
            num_tokens = int(tokens_per_expert.sum().item())

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
            tokens_per_expert_cpu = tokens_per_expert.cpu()
            probs_gpu = probs.to(device, non_blocking=True)

            with torch.enable_grad():
                # Detach and require grad for recomputation
                set_hidden_states_req = set_hidden_states.detach().requires_grad_(True)
                w1_gpu_req = w1_gpu.detach().requires_grad_(True)
                w2_gpu_req = w2_gpu.detach().requires_grad_(True)

                # Forward
                fc1_output = gg.ops.gmm(
                    set_hidden_states_req, w1_gpu_req, tokens_per_expert_cpu, trans_b=False
                )
                intermediate = self.activation_func(fc1_output) * probs_gpu.unsqueeze(-1)
                fc2_output = gg.ops.gmm(
                    intermediate, w2_gpu_req, tokens_per_expert_cpu, trans_b=False
                )

                # Compute gradients
                grads = torch.autograd.grad(
                    outputs=fc2_output,
                    inputs=(set_hidden_states_req, w1_gpu_req, w2_gpu_req),
                    grad_outputs=set_grad_output,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

                grad_input = grads[0]
                grad_w1 = grads[1]
                grad_w2 = grads[2]

            # Offload gradients to CPU
            self._offload_grads_to_cpu(expert_ids, grad_w1, grad_w2)

            grad_input_list.append(grad_input)
            token_offset += num_tokens
            grad_offset += num_tokens

            # 5. Swap buffers
            current_buffer, next_buffer = next_buffer, current_buffer

        nvtx.range_pop()

        if grad_input_list:
            grad_input = torch.cat(grad_input_list, dim=0)
        else:
            grad_input = torch.empty(
                0, self.config.hidden_size, device=device, dtype=grad_output.dtype
            )

        # Return gradients: (self, hidden_states, tokens_per_expert_per_set, probs_per_set, expert_sets)
        return None, grad_input, None, None, None