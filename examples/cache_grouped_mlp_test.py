#!/usr/bin/env python
"""Distributed test script for CacheGroupedMLP with full MoE pipeline.

This script tests CacheGroupedMLP with real routing and dispatching,
including:
- Attention layers (via GPTModel)
- TopKRouter for routing
- MoEAlltoAllTokenDispatcher for token dispatching
- CacheGroupedMLP for expert computation

Usage:
    # Single-GPU test
    CUDA_VISIBLE_DEVICES=0 python examples/cache_grouped_mlp_test.py \\
        --num-layers 1 --batch-size 2 --seq-len 4096

    # Multi-GPU test (EP=2)
    CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 python examples/cache_grouped_mlp_test.py \\
        --num-layers 1 --batch-size 2 --seq-len 4096

    # With activation offload
    CUDA_VISIBLE_DEVICES=0 python examples/cache_grouped_mlp_test.py \\
        --activation-offload

    # With trace offload for debugging
    CUDA_VISIBLE_DEVICES=0 python examples/cache_grouped_mlp_test.py \\
        --trace-offload
"""

import argparse
import os
import time
from typing import Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import get_te_version

# Import for GPT model test - may not be available in all environments
try:
    from megatron.training.utils import get_ltor_masks_and_position_ids
    HAVE_LTOR_MASKS = True
except ImportError:
    HAVE_LTOR_MASKS = False

# Import CacheGroupedMLP for direct testing
try:
    from megatron.core.transformer.moe.experts import CacheGroupedMLP
    HAVE_CACHE_GROUPED_MLP = True
except ImportError:
    HAVE_CACHE_GROUPED_MLP = False

# Import FusedDispatcherCacheGroupedMLP for testing
try:
    from megatron.core.transformer.moe.experts import FusedDispatcherCacheGroupedMLP
    HAVE_FUSED_DISPATCHER_MLP = True
except ImportError:
    HAVE_FUSED_DISPATCHER_MLP = False


def _init_distributed() -> Tuple[int, int, int]:
    """Initialize distributed process group."""
    # Set default environment variables for single-GPU case
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _initialize_model_parallel(tp: int, pp: int, ep: int) -> None:
    """Initialize model parallel groups."""
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
    )


def _iter_sequential_mlps(module: torch.nn.Module) -> Iterable[SequentialMLP]:
    """Iterate over all SequentialMLP modules in a model."""
    for m in module.modules():
        if isinstance(m, SequentialMLP):
            yield m


def _wrap_cache_calls(model: torch.nn.Module, rank: int) -> List[str]:
    """Wrap weight cache operations for tracing."""
    events: List[str] = []

    for idx, mlp in enumerate(_iter_sequential_mlps(model)):
        cache = mlp.weight_cache

        orig_activate = cache.activate_group
        orig_release = cache.release_group
        orig_offload_grad = cache.offload_param_grad_to_cpu

        def activate_group(
            group_idx: int,
            training: bool = True,
            device: torch.device | None = None,
            *,
            _m=idx,
            _orig_activate=orig_activate,
        ):
            events.append(f"rank{rank}:mlp{_m}:activate({group_idx})")
            return _orig_activate(group_idx, training=training, device=device)

        def release_group(
            group_idx: int, copy_data: bool = True, *, _m=idx, _orig_release=orig_release
        ):
            out = _orig_release(group_idx, copy_data=copy_data)
            events.append(f"rank{rank}:mlp{_m}:release({group_idx})")
            return out

        def offload_param_grad_to_cpu(
            param: torch.nn.Parameter,
            grad: torch.Tensor,
            *,
            _m=idx,
            _orig_offload=orig_offload_grad,
        ):
            out = _orig_offload(param, grad)
            pdev = str(param.device)
            gdev = "None" if (param.grad is None) else str(param.grad.device)
            events.append(f"rank{rank}:mlp{_m}:offload_grad(p={pdev},g={gdev})")
            return out

        cache.activate_group = activate_group  # type: ignore[method-assign]
        cache.release_group = release_group  # type: ignore[method-assign]
        cache.offload_param_grad_to_cpu = offload_param_grad_to_cpu  # type: ignore[method-assign]

    return events


def _assert_moe_params_on_cpu(model: torch.nn.Module) -> None:
    """Verify that MoE expert parameters remain on CPU."""
    for mlp in _iter_sequential_mlps(model):
        for expert in mlp.local_experts:
            for p in expert.parameters():
                if str(p.device) != "cpu":
                    raise AssertionError(f"expected expert param on cpu, got {p.device}")
                if p.grad is not None and str(p.grad.device) != "cpu":
                    raise AssertionError(f"expected expert grad on cpu, got {p.grad.device}")


def _prime_expert_cache_to_cpu(model: torch.nn.Module) -> None:
    """Prime expert cache by loading weights to CPU first."""
    for mlp in _iter_sequential_mlps(model):
        cache = mlp.weight_cache
        if not cache.enabled:
            continue
        for expert_idx, expert in enumerate(mlp.local_experts):
            first_param = next(expert.parameters(), None)
            device = None if first_param is None else first_param.device
            cache.activate_group(expert_idx, training=False, device=device)
            cache.release_group(expert_idx, copy_data=False)


def _move_non_moe_to_cuda(model: torch.nn.Module, device: torch.device) -> None:
    """Move non-MoE parameters to CUDA device."""
    expert_params = set()
    expert_buffers = set()
    for mlp in _iter_sequential_mlps(model):
        for p in mlp.parameters():
            expert_params.add(p)
        for b in mlp.buffers():
            expert_buffers.add(b)

    for p in model.parameters():
        if p in expert_params:
            continue
        p.data = p.data.to(device)

    for b in model.buffers():
        if b in expert_buffers:
            continue
        b.data = b.data.to(device)


def _force_release_expert_cache(model: torch.nn.Module) -> None:
    """Force release all expert cache."""
    for mlp in _iter_sequential_mlps(model):
        cache = mlp.weight_cache
        if not cache.enabled:
            continue
        cache.release()


def _wrap_cache_grouped_mlp_calls(model: 'CacheGroupedMLP', rank: int) -> List[str]:
    """Wrap CacheGroupedMLP cache operations for tracing."""
    events: List[str] = []

    # Wrap weight loading
    orig_load = model._load_expert_weights

    def load_expert_weights(
        expert_ids: List[int],
        device: torch.device,
        *,
        _orig_load=orig_load,
    ):
        events.append(f"rank{rank}:load_experts({expert_ids})")
        return _orig_load(expert_ids, device)

    model._load_expert_weights = load_expert_weights  # type: ignore[method-assign]

    # Wrap gradient offload
    orig_offload = model._offload_grads_to_cpu

    def offload_grads_to_cpu(
        expert_ids: List[int],
        grad_w1: torch.Tensor,
        grad_w2: torch.Tensor,
        *,
        _orig_offload=orig_offload,
    ):
        events.append(f"rank{rank}:offload_grads({expert_ids})")
        return _orig_offload(expert_ids, grad_w1, grad_w2)

    model._offload_grads_to_cpu = offload_grads_to_cpu  # type: ignore[method-assign]

    return events


def _assert_weights_on_cpu(model: 'CacheGroupedMLP') -> None:
    """Verify that expert weights remain on CPU."""
    if model.weight1.device.type != "cpu":
        raise AssertionError(
            f"Expected weight1 on CPU, got {model.weight1.device}"
        )
    if model.weight2.device.type != "cpu":
        raise AssertionError(
            f"Expected weight2 on CPU, got {model.weight2.device}"
        )


def run_gpt_model_test(args: argparse.Namespace) -> int:
    """Run test using GPTModel with SequentialMLP (reference implementation)."""
    if not HAVE_LTOR_MASKS:
        raise RuntimeError("megatron.training.utils not available - cannot run GPT model test")

    rank, world_size, _local_rank = _init_distributed()

    tp_size = 1
    pp_size = 1
    ep_size = world_size
    _initialize_model_parallel(tp=tp_size, pp=pp_size, ep=ep_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)

    num_moe_experts = ep_size * args.num_local_experts

    bf16 = bool(args.bf16 and torch.cuda.is_bf16_supported())
    if args.use_flash_attn and not bf16:
        raise RuntimeError("FlashAttention requires --bf16 in this script.")
    if args.use_flash_attn:
        head_dim = args.hidden_size // args.num_attention_heads
        if args.hidden_size % args.num_attention_heads != 0:
            raise RuntimeError("hidden_size must be divisible by num_attention_heads.")
        if head_dim > 256:
            raise RuntimeError(
                f"FlashAttention requires head_dim <= 256, got {head_dim}. "
                "Increase num_attention_heads or reduce hidden_size."
            )

    use_transformer_engine = args.use_transformer_engine or args.use_flash_attn
    if use_transformer_engine and get_te_version() is None:
        raise RuntimeError("Transformer Engine is required for FlashAttention backend.")
    params_dtype = torch.bfloat16 if bf16 else torch.float32

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.ffn_hidden_size,
        num_moe_experts=num_moe_experts,
        moe_layer_freq=1,
        moe_router_topk=args.moe_router_topk,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type=args.moe_token_dispatcher_type,
        moe_grouped_gemm=False,
        moe_enable_expert_weight_cache=True,
        moe_activation_offload=args.activation_offload,
        use_cpu_initialization=True,
        bf16=bf16,
        params_dtype=params_dtype,
        add_bias_linear=False,
        gated_linear_unit=True,
        sequence_parallel=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.flash if args.use_flash_attn else AttnBackend.auto,
    )

    transformer_layer_spec = get_gpt_decoder_block_spec(
        config, use_transformer_engine=use_transformer_engine
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_len,
    )

    if bf16:
        model = model.bfloat16()

    _move_non_moe_to_cuda(model, torch.device("cuda"))
    _force_release_expert_cache(model)

    model.train()
    _prime_expert_cache_to_cpu(model)

    events: List[str] = []
    if args.trace_offload:
        events = _wrap_cache_calls(model, rank=rank)

    device = torch.device("cuda")
    eod_token = 0
    pad_token = 0

    try:
        dist.barrier()
        if rank == 0:
            print(
                f"GPTModel Test (SequentialMLP): iters={args.iters} world={world_size} "
                f"experts={num_moe_experts} local_experts={args.num_local_experts} "
                f"dtype={'bf16' if bf16 else 'fp32'} "
                f"activation_offload={args.activation_offload}",
                flush=True,
            )

        for it in range(args.iters):
            model.zero_grad(set_to_none=True)

            tokens = torch.randint(
                low=0,
                high=args.vocab_size,
                size=(args.batch_size, args.seq_len),
                device=device,
                dtype=torch.long,
            )
            labels = torch.randint(
                low=0,
                high=args.vocab_size,
                size=(args.batch_size, args.seq_len),
                device=device,
                dtype=torch.long,
            )
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens,
                eod_token=eod_token,
                pad_token=pad_token,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                pad_mask_loss=False,
            )

            t0 = time.time()
            loss = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            loss = loss.float().mean()
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0

            _force_release_expert_cache(model)
            _assert_moe_params_on_cpu(model)

            if rank == 0:
                print(f"iter {it}: loss={loss.item():.6f} time={dt:.3f}s", flush=True)

        dist.barrier()
        if args.trace_offload:
            for line in events[-50:]:
                print(line, flush=True)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        parallel_state.destroy_model_parallel()

    return 0


def run_cache_grouped_mlp_test(args: argparse.Namespace) -> int:
    """Run test using CacheGroupedMLP with real Router and Dispatcher."""
    if not HAVE_CACHE_GROUPED_MLP:
        raise RuntimeError("CacheGroupedMLP not available")

    rank, world_size, _local_rank = _init_distributed()

    # EP only - each rank has access to all global experts
    ep_size = world_size
    _initialize_model_parallel(tp=1, pp=1, ep=ep_size)

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)

    # Setup dtype
    bf16 = bool(args.bf16 and torch.cuda.is_bf16_supported())
    params_dtype = torch.bfloat16 if bf16 else torch.float32

    num_global_experts = args.num_global_experts

    # Create config
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.ffn_hidden_size,
        num_moe_experts=num_global_experts,
        moe_layer_freq=1,
        moe_router_topk=args.moe_router_topk,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type=args.moe_token_dispatcher_type,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=bf16,
        params_dtype=params_dtype,
        moe_enable_expert_weight_cache=True,
        moe_activation_offload=args.activation_offload,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    # Create ProcessGroupCollection from parallel_state (pulls all default groups)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Get EP group from parallel_state
    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_rank = dist.get_rank(ep_group)

    # Calculate local experts
    num_local_experts = num_global_experts // ep_size
    local_expert_indices = [ep_rank * num_local_experts + i for i in range(num_local_experts)]

    # Create Router
    router = TopKRouter(config=config, pg_collection=pg_collection)

    # Create Dispatcher
    dispatcher = MoEAlltoAllTokenDispatcher(
        num_local_experts=num_local_experts,
        local_expert_indices=local_expert_indices,
        config=config,
        pg_collection=pg_collection,
    )

    # Create CacheGroupedMLP (experts)
    experts = CacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=pg_collection,
    )

    if bf16:
        experts = experts.bfloat16()

    experts.train()

    # Setup tracing
    events: List[str] = []
    if args.trace_offload:
        events = _wrap_cache_grouped_mlp_calls(experts, rank=rank)

    device = torch.device("cuda")

    try:
        dist.barrier()

        if rank == 0:
            print(
                f"CacheGroupedMLP Test (real Router+Dispatcher): "
                f"iters={args.iters} world={world_size} "
                f"batch_size={args.batch_size} seq_len={args.seq_len} "
                f"experts={num_global_experts} local_experts={num_local_experts} "
                f"hidden_size={args.hidden_size} ffn_hidden_size={args.ffn_hidden_size} "
                f"dtype={'bf16' if bf16 else 'fp32'} "
                f"activation_offload={args.activation_offload}",
                flush=True,
            )

        total_forward_time = 0.0
        total_backward_time = 0.0

        for it in range(args.iters):
            experts.zero_grad(set_to_none=True)

            # Generate input hidden states [S, B, H] - standard format for router
            batch_size = args.batch_size
            seq_len = args.seq_len
            hidden_size = args.hidden_size

            # Random hidden states
            hidden_states = torch.randn(
                seq_len, batch_size, hidden_size,
                dtype=params_dtype, device=device
            )
            hidden_states.requires_grad_(True)

            # Forward pass timing
            t0 = time.time()

            # 1. Router - get routing probabilities and map
            # Router expects [S, B, H] and returns probs, routing_map
            probs, routing_map = router(hidden_states)

            # 2. Preprocess for dispatcher
            # Flatten to [S*B, H] for dispatcher
            hidden_states_flat = hidden_states.view(-1, hidden_size)
            hidden_states_flat, probs = dispatcher.dispatch_preprocess(
                hidden_states_flat, routing_map, probs
            )

            # 3. Token dispatch (AlltoAll communication)
            hidden_states_flat, probs = dispatcher.token_dispatch(hidden_states_flat, probs)

            # 4. Postprocess dispatch - get expert inputs
            expert_input, tokens_per_expert, permuted_probs = dispatcher.dispatch_postprocess(
                hidden_states_flat, probs
            )

            # 5. Expert computation with CacheGroupedMLP
            # Convert dispatcher output to CacheGroupedMLP format
            # CacheGroupedMLP expects:
            #   - expert_sets: List[List[int]] - e.g., [[e1,e2], [e3,e4], ...]
            #   - tokens_per_expert_per_set: List[torch.Tensor] - token counts per expert per set
            #   - probs_per_set: List[torch.Tensor] - probabilities per token per set
            #
            # The dispatcher returns:
            #   - expert_input: [total_tokens, hidden_size] - all tokens for local experts
            #   - tokens_per_expert: [num_local_experts] - token count per local expert
            #   - permuted_probs: [total_tokens] - probabilities for all tokens

            # Split local experts into smaller sets to properly test double buffering
            # and async weight prefetch. Using 2 experts per set.
            experts_per_set = min(8, len(local_expert_indices))
            expert_sets = []
            tokens_per_expert_per_set = []
            probs_per_set = []

            token_offset = 0
            for i in range(0, len(local_expert_indices), experts_per_set):
                set_experts = local_expert_indices[i:i + experts_per_set]
                expert_sets.append(set_experts)

                # Get token counts for experts in this set
                set_token_counts = tokens_per_expert[i:i + len(set_experts)]
                tokens_per_expert_per_set.append(set_token_counts)

                # Get total tokens for this set
                num_tokens_in_set = int(set_token_counts.sum().item())
                set_probs = permuted_probs[token_offset:token_offset + num_tokens_in_set]
                probs_per_set.append(set_probs)
                token_offset += num_tokens_in_set

            # Call CacheGroupedMLP
            expert_output, _ = experts(
                hidden_states=expert_input,
                tokens_per_expert_per_set=tokens_per_expert_per_set,
                probs_per_set=probs_per_set,
                expert_sets=expert_sets,
            )

            # 6. Combine preprocess
            output = dispatcher.combine_preprocess(expert_output)

            # 7. Token combine (AlltoAll communication)
            output = dispatcher.token_combine(output)

            # 8. Combine postprocess
            output = dispatcher.combine_postprocess(output)

            torch.cuda.synchronize()
            forward_time = time.time() - t0
            total_forward_time += forward_time

            # Backward pass timing
            t0 = time.time()

            # Compute loss and backward
            loss = output.sum()
            loss.backward()

            # Sync gradients for CacheGroupedMLP
            experts.sync_gradients()

            torch.cuda.synchronize()
            backward_time = time.time() - t0
            total_backward_time += backward_time

            # Verify weights on CPU
            _assert_weights_on_cpu(experts)

            # Verify gradients
            assert experts.weight1.grad is not None, "weight1.grad should be set after sync"
            assert experts.weight2.grad is not None, "weight2.grad should be set after sync"

            # Get token statistics
            total_tokens = tokens_per_expert.sum().item() if tokens_per_expert.numel() > 0 else 0
            num_experts_this_iter = len(local_expert_indices)
            num_expert_sets = len(expert_sets)

            print(
                f"rank {ep_rank} iter {it}: experts={num_experts_this_iter} "
                f"sets={num_expert_sets} total_tokens={total_tokens} "
                f"loss={loss.item():.6f} "
                f"forward={forward_time:.3f}s backward={backward_time:.3f}s",
                flush=True,
            )

        dist.barrier()

        # Print summary
        if rank == 0:
            print("\n" + "=" * 60)
            print("Summary:")
            print(f"  Num layers: {args.num_layers}")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Seq length: {args.seq_len}")
            print(f"  Total tokens per iteration: {args.batch_size * args.seq_len}")
            print(f"  Num global experts: {num_global_experts}")
            print(f"  Num local experts: {num_local_experts}")
            print(f"  Router topk: {args.moe_router_topk}")
            print(f"  Avg forward time: {total_forward_time / args.iters:.3f}s")
            print(f"  Avg backward time: {total_backward_time / args.iters:.3f}s")
            print(f"  Total time: {total_forward_time + total_backward_time:.3f}s")
            print("=" * 60)

            if args.trace_offload and events:
                print("\nTrace events (last 50):")
                for line in events[-50:]:
                    print(f"  {line}")

    finally:
        # Clean up shared memory
        if hasattr(experts, 'release'):
            experts.release()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        parallel_state.destroy_model_parallel()

    return 0


def run_fused_dispatcher_mlp_test(args: argparse.Namespace) -> int:
    """Run test using FusedDispatcherCacheGroupedMLP with Router (no external dispatcher).

    FusedDispatcherCacheGroupedMLP handles all_to_all communication internally,
    so no separate token dispatcher is needed.
    """
    if not HAVE_FUSED_DISPATCHER_MLP:
        raise RuntimeError("FusedDispatcherCacheGroupedMLP not available")

    rank, world_size, _local_rank = _init_distributed()

    # EP only - each rank processes a subset of experts
    ep_size = world_size
    _initialize_model_parallel(tp=1, pp=1, ep=ep_size)

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)

    # Setup dtype
    bf16 = bool(args.bf16 and torch.cuda.is_bf16_supported())
    params_dtype = torch.bfloat16 if bf16 else torch.float32

    num_global_experts = args.num_global_experts

    # Create config
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.ffn_hidden_size,
        num_moe_experts=num_global_experts,
        moe_layer_freq=1,
        moe_router_topk=args.moe_router_topk,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type=args.moe_token_dispatcher_type,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=bf16,
        params_dtype=params_dtype,
        moe_enable_expert_weight_cache=True,
        moe_activation_offload=args.activation_offload,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    # Create ProcessGroupCollection from parallel_state
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Get EP group from parallel_state
    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_rank = dist.get_rank(ep_group)

    # Create Router (for routing_map and probs)
    router = TopKRouter(config=config, pg_collection=pg_collection)

    # Create FusedDispatcherCacheGroupedMLP (no external dispatcher needed!)
    experts = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=pg_collection,
    )

    if bf16:
        experts = experts.bfloat16()

    experts.train()

    # Define expert_sets: partition experts across ranks
    # Each rank processes its share of experts
    experts_per_rank = num_global_experts // ep_size
    local_expert_start = ep_rank * experts_per_rank
    expert_sets = [[local_expert_start + i for i in range(experts_per_rank)]]

    # Setup tracing
    events: List[str] = []
    if args.trace_offload:
        events = _wrap_cache_grouped_mlp_calls(experts, rank=rank)

    device = torch.device("cuda")
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    seq_len = args.seq_len

    try:
        dist.barrier()

        if rank == 0:
            print(
                f"FusedDispatcherCacheGroupedMLP Test (fused dispatcher): "
                f"iters={args.iters} world={world_size} "
                f"batch_size={batch_size} seq_len={seq_len} "
                f"experts={num_global_experts} experts_per_rank={experts_per_rank} "
                f"hidden_size={hidden_size} ffn_hidden_size={args.ffn_hidden_size} "
                f"dtype={'bf16' if bf16 else 'fp32'} "
                f"activation_offload={args.activation_offload}",
                flush=True,
            )

        total_forward_time = 0.0
        total_backward_time = 0.0

        for it in range(args.iters):
            experts.zero_grad(set_to_none=True)

            # Generate hidden states [S, B, H] - standard format for router
            hidden_states = torch.randn(
                seq_len, batch_size, hidden_size,
                dtype=params_dtype, device=device
            )
            hidden_states.requires_grad_(True)

            # Forward pass timing
            t0 = time.time()

            # 1. Router - get routing probabilities and map
            # Router expects [S, B, H] and returns probs, routing_map
            probs, routing_map = router(hidden_states)

            # 2. Flatten for FusedDispatcherCacheGroupedMLP
            # hidden_states: [S, B, H] -> [S*B, H]
            hidden_states_flat = hidden_states.view(-1, hidden_size)
            # probs: [S, B, num_experts] -> [S*B, num_experts]
            probs_flat = probs.view(-1, num_global_experts)
            # routing_map: [S, B, num_experts] -> [S*B, num_experts]
            routing_map_flat = routing_map.view(-1, num_global_experts)

            # 3. Call FusedDispatcherCacheGroupedMLP - handles all_to_all internally!
            output, _ = experts(
                hidden_states=hidden_states_flat,
                routing_map=routing_map_flat,
                probs=probs_flat,
                expert_sets=expert_sets,
            )

            torch.cuda.synchronize()
            forward_time = time.time() - t0
            total_forward_time += forward_time

            # Backward pass timing
            t0 = time.time()

            # Compute loss and backward
            loss = output.sum()
            loss.backward()

            # Sync gradients for FusedDispatcherCacheGroupedMLP
            experts.sync_gradients()

            torch.cuda.synchronize()
            backward_time = time.time() - t0
            total_backward_time += backward_time

            # Verify weights on CPU
            _assert_weights_on_cpu(experts)

            # Verify gradients
            assert experts.weight1.grad is not None, "weight1.grad should be set after sync"
            assert experts.weight2.grad is not None, "weight2.grad should be set after sync"

            # Get token statistics
            total_tokens = batch_size * seq_len

            print(
                f"rank {ep_rank} iter {it}: experts={experts_per_rank} "
                f"total_tokens={total_tokens} "
                f"loss={loss.item():.6f} "
                f"forward={forward_time:.3f}s backward={backward_time:.3f}s",
                flush=True,
            )

        dist.barrier()

        # Print summary
        if rank == 0:
            print("\n" + "=" * 60)
            print("Summary:")
            print(f"  Num layers: {args.num_layers}")
            print(f"  Batch size: {batch_size}")
            print(f"  Seq length: {seq_len}")
            print(f"  Total tokens per iteration: {batch_size * seq_len}")
            print(f"  Num global experts: {num_global_experts}")
            print(f"  Experts per rank: {experts_per_rank}")
            print(f"  Router topk: {args.moe_router_topk}")
            print(f"  Avg forward time: {total_forward_time / args.iters:.3f}s")
            print(f"  Avg backward time: {total_backward_time / args.iters:.3f}s")
            print(f"  Total time: {total_forward_time + total_backward_time:.3f}s")
            print("=" * 60)

            if args.trace_offload and events:
                print("\nTrace events (last 50):")
                for line in events[-50:]:
                    print(f"  {line}")

    finally:
        # Clean up
        if hasattr(experts, 'release'):
            experts.release()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        parallel_state.destroy_model_parallel()

    return 0


def main() -> int:
    os.environ.setdefault("NCCL_DEBUG", "ERROR")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    parser = argparse.ArgumentParser(
        description="Distributed test for CacheGroupedMLP with full MoE pipeline"
    )
    # Test mode
    parser.add_argument("--test-mode", type=str, default="cache_grouped_mlp",
                        choices=["gpt_model", "cache_grouped_mlp", "fused_dispatcher_mlp"],
                        help="Test mode: gpt_model (SequentialMLP), cache_grouped_mlp, or fused_dispatcher_mlp")

    # Common args
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-hidden-size", type=int, default=10240)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace-offload", action="store_true")
    parser.add_argument("--activation-offload", action="store_true",
                        help="Enable MoE input activation offload to CPU")

    # GPTModel mode args
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--num-local-experts", type=int, default=64,
                        help="Number of local experts per rank (GPTModel mode)")
    parser.add_argument("--moe-router-topk", type=int, default=8)
    parser.add_argument("--moe-token-dispatcher-type", type=str, default="alltoall")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--use-transformer-engine", action="store_true")

    # CacheGroupedMLP mode args
    parser.add_argument("--num-global-experts", type=int, default=64,
                        help="Number of global experts (CacheGroupedMLP mode)")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    if args.test_mode == "gpt_model":
        return run_gpt_model_test(args)
    elif args.test_mode == "cache_grouped_mlp":
        return run_cache_grouped_mlp_test(args)
    else:  # fused_dispatcher_mlp
        return run_fused_dispatcher_mlp_test(args)


if __name__ == "__main__":
    try:
        torch.cuda.memory._record_memory_history()
        exit_code = main()
    finally:
        torch.cuda.memory._dump_snapshot("cache_grouped_mlp_memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)