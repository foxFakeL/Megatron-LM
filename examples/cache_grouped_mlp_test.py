#!/usr/bin/env python
"""Distributed test script for CacheGroupedMLP.

This script directly tests CacheGroupedMLP in a distributed EP setting,
simulating expert routing by generating random expert_sets and token distributions.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./examples/cache_grouped_mlp_test.sh
    CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 ./examples/cache_grouped_mlp_test.sh
    CUDA_VISIBLE_DEVICES=0 ACTIVATION_OFFLOAD=1 ./examples/cache_grouped_mlp_test.sh
"""

import argparse
import os
import random
import time
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import CacheGroupedMLP
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig


def _init_distributed() -> Tuple[int, int, int]:
    """Initialize distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _initialize_model_parallel(ep: int) -> None:
    """Initialize model parallel groups with EP only."""
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep,
    )


def generate_expert_sets_for_rank(
    num_global_experts: int,
    ep_rank: int,
    ep_size: int,
    num_sets: int,
    avg_tokens_per_expert: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[List[List[int]], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Generate test data for CacheGroupedMLP with external scheduling.

    This simulates external routing where each EP rank only processes experts
    where expert_id % ep_size == ep_rank.

    Args:
        num_global_experts: Total number of global experts
        ep_rank: Current EP rank
        ep_size: Total EP world size
        num_sets: Number of expert sets to generate
        avg_tokens_per_expert: Average tokens per expert
        hidden_size: Hidden dimension size
        dtype: Data type for tensors
        device: Device for tensors

    Returns:
        Tuple of (expert_sets, tokens_per_expert_per_set, probs_per_set, hidden_states)
    """
    # Determine which experts this rank is responsible for
    my_experts = [e for e in range(num_global_experts) if e % ep_size == ep_rank]

    if not my_experts:
        # This rank has no experts to process
        return [], [], [], torch.empty(0, hidden_size, dtype=dtype, device=device)

    # Distribute experts across sets
    experts_per_set = max(1, len(my_experts) // num_sets)
    expert_sets = []
    tokens_per_expert_per_set = []
    probs_per_set = []
    total_tokens = 0

    for i in range(num_sets):
        start = i * experts_per_set
        end = min(start + experts_per_set, len(my_experts))
        expert_set = my_experts[start:end]

        if not expert_set:
            continue

        # Random number of tokens per expert
        num_tokens_per_expert = [
            random.randint(1, avg_tokens_per_expert * 2) for _ in expert_set
        ]
        tokens_per_expert = torch.tensor(num_tokens_per_expert, dtype=torch.long)

        # Generate random probabilities
        set_total_tokens = sum(num_tokens_per_expert)
        probs = torch.rand(set_total_tokens, dtype=dtype)

        expert_sets.append(expert_set)
        tokens_per_expert_per_set.append(tokens_per_expert)
        probs_per_set.append(probs)
        total_tokens += set_total_tokens

    # Generate hidden states
    hidden_states = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)

    return expert_sets, tokens_per_expert_per_set, probs_per_set, hidden_states


def generate_expert_sets(
    num_global_experts: int,
    num_sets: int,
    avg_tokens_per_expert: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[List[List[int]], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Generate test data for CacheGroupedMLP.

    This simulates the routing output from a router, generating:
    - expert_sets: List of expert ID lists
    - tokens_per_expert_per_set: Token counts per expert per set
    - probs_per_set: Router probabilities per token per set
    - hidden_states: Random hidden states for all tokens

    Args:
        num_global_experts: Total number of global experts
        num_sets: Number of expert sets to generate
        avg_tokens_per_expert: Average tokens per expert
        hidden_size: Hidden dimension size
        dtype: Data type for tensors
        device: Device for tensors

    Returns:
        Tuple of (expert_sets, tokens_per_expert_per_set, probs_per_set, hidden_states)
    """
    experts_per_set = max(1, num_global_experts // num_sets)
    expert_sets = []
    tokens_per_expert_per_set = []
    probs_per_set = []
    total_tokens = 0

    for i in range(num_sets):
        # Determine expert range for this set
        start_expert = i * experts_per_set
        end_expert = min(start_expert + experts_per_set, num_global_experts)
        expert_set = list(range(start_expert, end_expert))

        # Random number of tokens per expert
        num_tokens_per_expert = [
            random.randint(1, avg_tokens_per_expert * 2) for _ in expert_set
        ]
        tokens_per_expert = torch.tensor(num_tokens_per_expert, dtype=torch.long)

        # Generate random probabilities
        set_total_tokens = sum(num_tokens_per_expert)
        probs = torch.rand(set_total_tokens, dtype=dtype)

        expert_sets.append(expert_set)
        tokens_per_expert_per_set.append(tokens_per_expert)
        probs_per_set.append(probs)
        total_tokens += set_total_tokens

    # Generate hidden states
    hidden_states = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)

    return expert_sets, tokens_per_expert_per_set, probs_per_set, hidden_states


def _wrap_cache_calls(model: CacheGroupedMLP, rank: int) -> List[str]:
    """Wrap weight cache operations for tracing."""
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


def _assert_weights_on_cpu(model: CacheGroupedMLP) -> None:
    """Verify that expert weights remain on CPU."""
    if model.weight1.device.type != "cpu":
        raise AssertionError(
            f"Expected weight1 on CPU, got {model.weight1.device}"
        )
    if model.weight2.device.type != "cpu":
        raise AssertionError(
            f"Expected weight2 on CPU, got {model.weight2.device}"
        )


def main() -> int:
    os.environ.setdefault("NCCL_DEBUG", "ERROR")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    parser = argparse.ArgumentParser(
        description="Distributed test for CacheGroupedMLP"
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--num-global-experts", type=int, default=64)
    parser.add_argument("--num-sets", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--ffn-hidden-size", type=int, default=2048)
    parser.add_argument("--tokens-per-expert", type=int, default=256)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trace-offload", action="store_true")
    parser.add_argument("--activation-offload", action="store_true",
                        help="Enable MoE input activation offload to CPU")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    # Initialize distributed
    rank, world_size, _local_rank = _init_distributed()

    # EP only - each rank has access to all global experts
    ep_size = world_size
    _initialize_model_parallel(ep=ep_size)

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)

    # Setup dtype
    bf16 = bool(args.bf16 and torch.cuda.is_bf16_supported())
    params_dtype = torch.bfloat16 if bf16 else torch.float32

    # Create config
    config = TransformerConfig(
        num_layers=2,
        hidden_size=args.hidden_size,
        num_attention_heads=128,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.ffn_hidden_size,
        num_moe_experts=args.num_global_experts,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=bf16,
        params_dtype=params_dtype,
        moe_enable_expert_weight_cache=True,
        moe_activation_offload=args.activation_offload,
        use_cpu_initialization=True,
    )

    # Create ProcessGroupCollection for EP
    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_rank = dist.get_rank(ep_group)
    pg_collection = ProcessGroupCollection(ep=ep_group)

    # Create model
    model = CacheGroupedMLP(
        num_global_experts=args.num_global_experts,
        config=config,
        pg_collection=pg_collection,
    )

    if bf16:
        model = model.bfloat16()

    model.train()

    # Setup tracing
    events: List[str] = []
    if args.trace_offload:
        events = _wrap_cache_calls(model, rank=rank)

    device = torch.device("cuda")

    try:
        dist.barrier()
        # Calculate experts per rank
        my_experts = [e for e in range(args.num_global_experts) if e % ep_size == ep_rank]
        all_experts_per_rank = [None] * ep_size
        dist.all_gather_object(all_experts_per_rank, my_experts, group=ep_group)

        if rank == 0:
            print(
                f"CacheGroupedMLP Test: iters={args.iters} world={world_size} "
                f"experts={args.num_global_experts} num_sets={args.num_sets} "
                f"hidden_size={args.hidden_size} ffn_hidden_size={args.ffn_hidden_size} "
                f"dtype={'bf16' if bf16 else 'fp32'} "
                f"activation_offload={args.activation_offload}",
                flush=True,
            )
            print("Expert distribution by EP rank:", flush=True)
            for r, exp in enumerate(all_experts_per_rank):
                print(f"  rank {r}: {len(exp)} experts {exp[:5]}{'...' if len(exp) > 5 else ''}", flush=True)

        total_forward_time = 0.0
        total_backward_time = 0.0

        for it in range(args.iters):
            model.zero_grad(set_to_none=True)

            # Generate test data with external scheduling
            # Each rank only processes experts where expert_id % ep_size == ep_rank
            expert_sets, tokens_per_expert_per_set, probs_per_set, hidden_states = (
                generate_expert_sets_for_rank(
                    num_global_experts=args.num_global_experts,
                    ep_rank=ep_rank,
                    ep_size=ep_size,
                    num_sets=args.num_sets,
                    avg_tokens_per_expert=args.tokens_per_expert,
                    hidden_size=args.hidden_size,
                    dtype=params_dtype,
                    device=device,
                )
            )

            # Skip if no experts for this rank
            if not expert_sets:
                if rank == 0:
                    print(f"iter {it}: no experts for rank {ep_rank}, skipping", flush=True)
                continue

            # Enable gradient computation
            hidden_states.requires_grad_(True)

            # Forward
            t0 = time.time()
            output, _ = model(
                hidden_states=hidden_states,
                tokens_per_expert_per_set=tokens_per_expert_per_set,
                probs_per_set=probs_per_set,
                expert_sets=expert_sets,
            )
            torch.cuda.synchronize()
            forward_time = time.time() - t0
            total_forward_time += forward_time

            # Backward
            t0 = time.time()
            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()
            backward_time = time.time() - t0
            total_backward_time += backward_time

            # Sync gradients from CPU to parameters
            model.sync_gradients()

            # Verify weights on CPU
            _assert_weights_on_cpu(model)

            # Verify gradients for this rank's experts
            assert model.weight1.grad is not None, "weight1.grad should be set after sync"
            assert model.weight2.grad is not None, "weight2.grad should be set after sync"

            # Get token statistics
            total_tokens = sum(t.sum().item() for t in tokens_per_expert_per_set)
            num_experts_this_iter = sum(len(s) for s in expert_sets)

            print(
                f"rank {ep_rank} iter {it}: experts={num_experts_this_iter} "
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
            print(f"  Avg forward time: {total_forward_time / args.iters:.3f}s")
            print(f"  Avg backward time: {total_backward_time / args.iters:.3f}s")
            print(f"  Total time: {total_forward_time + total_backward_time:.3f}s")
            print(f"  External scheduling: each expert computed by one EP rank")
            print("=" * 60)

            if args.trace_offload and events:
                print("\nTrace events (last 50):")
                for line in events[-50:]:
                    print(f"  {line}")

    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        parallel_state.destroy_model_parallel()

    return 0


if __name__ == "__main__":
    try:
        torch.cuda.memory._record_memory_history()
        main()
    finally:
        torch.cuda.memory._dump_snapshot("cache_grouped_mlp_memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)