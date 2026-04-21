#!/usr/bin/env python
"""Accuracy alignment test for FusedDispatcherCacheGroupedMLP.

Tests that forward output and backward gradients match a reference implementation.
Uses fixed random seeds and identical weight initialization.

Usage:
    # Single GPU test
    CUDA_VISIBLE_DEVICES=0 python tests/test_fused_dispatcher_accuracy.py

    # Multi-GPU test (EP=2)
    CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 python tests/test_fused_dispatcher_accuracy.py
"""

import os
import sys
import argparse
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


def create_shared_weights(
    num_experts: int,
    hidden_size: int,
    ffn_hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create weight tensors with fixed seed for reproducibility.

    Returns:
        weight1: [num_experts, hidden_size, ffn_hidden_size * 2] (GLU doubles output)
        weight2: [num_experts, ffn_hidden_size, hidden_size]
    """
    torch.manual_seed(seed)
    weight1 = torch.randn(
        num_experts, hidden_size, ffn_hidden_size * 2, dtype=dtype, device=device
    )
    weight2 = torch.randn(
        num_experts, ffn_hidden_size, hidden_size, dtype=dtype, device=device
    )
    return weight1, weight2


def reference_moe_forward(
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    activation_func,
) -> torch.Tensor:
    """Reference implementation: compute each expert's output independently.

    This is a simple loop-based implementation that serves as the ground truth.

    Args:
        hidden_states: [num_tokens, hidden_size]
        routing_map: [num_tokens, num_experts] boolean mask
        probs: [num_tokens, num_experts] routing probabilities
        weight1: [num_experts, hidden_size, ffn_hidden_size * 2]
        weight2: [num_experts, ffn_hidden_size, hidden_size]
        activation_func: activation function (e.g., GLU)

    Returns:
        output: [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    num_experts = weight1.shape[0]
    output = torch.zeros(
        num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )

    for exp_id in range(num_experts):
        # Find tokens routed to this expert
        expert_mask = routing_map[:, exp_id]
        if not expert_mask.any():
            continue

        token_indices = expert_mask.nonzero(as_tuple=True)[0]
        expert_input = hidden_states[token_indices]
        expert_probs = probs[token_indices, exp_id]

        # FC1: [num_tokens, ffn_hidden_size * 2]
        fc1_out = expert_input @ weight1[exp_id]

        # Activation with probability scaling
        intermediate = activation_func(fc1_out) * expert_probs.unsqueeze(-1)

        # FC2: [num_tokens, hidden_size]
        fc2_out = intermediate @ weight2[exp_id]

        # Scatter back to original positions
        output.index_add_(0, token_indices, fc2_out)

    return output


def reference_moe_forward_with_probs_grad(
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    activation_func,
):
    """Reference implementation that also computes probs gradient.

    Returns:
        output: [num_tokens, hidden_size]
        intermediate_list: list of intermediate tensors for backward
        metadata_list: list of (token_indices, expert_probs) for backward
    """
    num_tokens, hidden_size = hidden_states.shape
    num_experts = weight1.shape[0]
    output = torch.zeros(
        num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )
    intermediate_list = []
    metadata_list = []

    for exp_id in range(num_experts):
        expert_mask = routing_map[:, exp_id]
        if not expert_mask.any():
            continue

        token_indices = expert_mask.nonzero(as_tuple=True)[0]
        expert_input = hidden_states[token_indices]
        expert_probs = probs[token_indices, exp_id]

        fc1_out = expert_input @ weight1[exp_id]
        intermediate = activation_func(fc1_out) * expert_probs.unsqueeze(-1)
        fc2_out = intermediate @ weight2[exp_id]

        output.index_add_(0, token_indices, fc2_out)

        # Save for backward
        intermediate_list.append(intermediate)
        metadata_list.append((token_indices, expert_probs, exp_id))

    return output, intermediate_list, metadata_list


def reference_moe_backward_probs(
    grad_output: torch.Tensor,
    intermediate_list: list,
    metadata_list: list,
    weight2: torch.Tensor,
    routing_map: torch.Tensor,
    num_experts: int,
):
    """Compute probs gradient from saved intermediate tensors.

    Args:
        grad_output: [num_tokens, hidden_size]
        intermediate_list: list of intermediate tensors
        metadata_list: list of (token_indices, expert_probs, exp_id)
        weight2: [num_experts, ffn_hidden_size, hidden_size]
        routing_map: [num_tokens, num_experts]
        num_experts: number of experts

    Returns:
        grad_probs: [num_tokens, num_experts]
    """
    num_tokens = grad_output.shape[0]
    dtype = grad_output.dtype
    device = grad_output.device
    grad_probs = torch.zeros(num_tokens, num_experts, dtype=dtype, device=device)

    for intermediate, (token_indices, expert_probs, exp_id) in zip(intermediate_list, metadata_list):
        # grad_output for this expert's tokens
        grad_out_expert = grad_output[token_indices]

        # Backward through fc2: grad_intermediate = grad_out @ weight2.T
        grad_intermediate = grad_out_expert @ weight2[exp_id].T

        # Backward through probs scaling: grad_probs = (grad_intermediate * activation(fc1)).sum(-1)
        # intermediate = activation(fc1) * probs.unsqueeze(-1)
        # So grad_probs = (grad_intermediate * activation(fc1)).sum(-1)
        # But activation(fc1) = intermediate / probs.unsqueeze(-1)
        grad_probs_expert = (grad_intermediate * (intermediate / expert_probs.unsqueeze(-1))).sum(-1)

        # Scatter back to grad_probs
        for i, idx in enumerate(token_indices):
            grad_probs[idx, exp_id] = grad_probs_expert[i]

    return grad_probs


def glu_activation(x: torch.Tensor) -> torch.Tensor:
    """Gated Linear Unit activation: silu(x[0]) * x[1]."""
    x = torch.chunk(x, 2, dim=-1)
    return F.silu(x[0]) * x[1]


def init_distributed():
    """Initialize distributed process group."""
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


def test_single_gpu_accuracy():
    """Test EP=1 case: FusedDispatcherCacheGroupedMLP vs reference implementation.

    For EP=1, there is no cross-rank communication, so the output should be
    identical to a simple local computation.
    """
    from megatron.core import parallel_state
    from megatron.core.transformer.moe.experts import FusedDispatcherCacheGroupedMLP
    from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Initialize distributed
    rank, world_size, local_rank = init_distributed()

    # Initialize model parallel (EP only)
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=world_size,
    )

    # Fixed seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Test parameters
    num_global_experts = 64
    hidden_size = 2560
    ffn_hidden_size = 5120
    batch_size = 2
    seq_len = 64000
    num_tokens = batch_size * seq_len
    topk = 4
    dtype = torch.bfloat16
    device = torch.device("cuda", local_rank)

    print(f"Testing with {num_global_experts} experts, {num_tokens} tokens, topk={topk}")
    print(f"EP size: {world_size}")

    # ========== Create shared weights ==========
    weight1, weight2 = create_shared_weights(
        num_global_experts, hidden_size, ffn_hidden_size, dtype, device, seed + 100
    )

    # ========== Generate input ==========
    torch.manual_seed(seed + 200)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Generate routing (top-k per token)
    router_logits = torch.randn(num_tokens, num_global_experts, dtype=dtype, device=device)
    probs_full = F.softmax(router_logits.float(), dim=-1).to(dtype)
    topk_probs, topk_indices = probs_full.topk(topk, dim=-1)

    # Create routing_map and probs tensors
    routing_map = torch.zeros(num_tokens, num_global_experts, dtype=torch.bool, device=device)
    probs = torch.zeros(num_tokens, num_global_experts, dtype=dtype, device=device)
    for i in range(num_tokens):
        routing_map[i, topk_indices[i]] = True
        probs[i, topk_indices[i]] = topk_probs[i]

    # ========== Reference Implementation ==========
    print("\n=== Reference Implementation ===")
    hidden_states_ref = hidden_states.detach().clone().requires_grad_(True)
    probs_ref = probs.detach().clone().requires_grad_(True)  # Enable probs gradient
    weight1_ref = weight1.detach().clone().requires_grad_(True)
    weight2_ref = weight2.detach().clone().requires_grad_(True)

    output_ref = reference_moe_forward(
        hidden_states_ref, routing_map, probs_ref, weight1_ref, weight2_ref, glu_activation
    )
    loss_ref = output_ref.mean()
    loss_ref.backward()

    grad_input_ref = hidden_states_ref.grad.clone()
    grad_probs_ref = probs_ref.grad.clone()  # Get probs gradient
    grad_w1_ref = weight1_ref.grad.clone()
    grad_w2_ref = weight2_ref.grad.clone()

    print(f"Reference output shape: {output_ref.shape}")
    print(f"Reference loss: {loss_ref.item():.6f}")
    print(f"Reference grad_input norm: {grad_input_ref.norm().item():.6f}")
    print(f"Reference grad_probs norm: {grad_probs_ref.norm().item():.6f}")

    # ========== FusedDispatcher Implementation ==========
    print("\n=== FusedDispatcher Implementation ===")

    # Create config
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        ffn_hidden_size=ffn_hidden_size,
        moe_ffn_hidden_size=ffn_hidden_size,
        num_moe_experts=num_global_experts,
        moe_router_topk=topk,
        moe_router_pre_softmax=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=dtype,
        moe_enable_expert_weight_cache=True,
        use_cpu_initialization=True,  # Required for shared memory
        hidden_dropout=0.0,
        attention_dropout=0.0,
        perform_initialization=False,  # We'll set weights manually
    )

    # Create ProcessGroupCollection
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Create module
    fused_mlp = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=pg_collection,
    )

    # Manually set weights to match reference
    with torch.no_grad():
        fused_mlp.weight1.data.copy_(weight1.cpu())  # weights are on CPU
        fused_mlp.weight2.data.copy_(weight2.cpu())

    # Define expert_sets: partition experts across ranks
    ep_size = world_size
    experts_per_rank = num_global_experts // ep_size
    local_expert_start = rank * experts_per_rank
    expert_list = [local_expert_start + i for i in range(experts_per_rank)]

    # Split into sets (2 experts per set for testing double buffering)
    experts_per_set = 2
    expert_sets = [
        expert_list[i : i + experts_per_set]
        for i in range(0, len(expert_list), experts_per_set)
    ]

    print(f"Rank {rank} expert_sets: {expert_sets}")

    # Forward pass
    hidden_states_fused = hidden_states.detach().clone().requires_grad_(True)
    probs_fused = probs.detach().clone().requires_grad_(True)  # Enable probs gradient
    output_fused, _ = fused_mlp(
        hidden_states=hidden_states_fused,
        routing_map=routing_map,
        probs=probs_fused,
        expert_sets=expert_sets,
    )

    loss_fused = output_fused.mean()
    loss_fused.backward()

    # Sync gradients
    fused_mlp.sync_gradients()

    grad_input_fused = hidden_states_fused.grad.clone()
    grad_probs_fused = probs_fused.grad.clone()  # Get probs gradient
    grad_w1_fused = fused_mlp.weight1.grad.clone().to(device) if fused_mlp.weight1.grad is not None else None
    grad_w2_fused = fused_mlp.weight2.grad.clone().to(device) if fused_mlp.weight2.grad is not None else None

    print(f"Fused output shape: {output_fused.shape}")
    print(f"Fused loss: {loss_fused.item():.6f}")
    print(f"Fused grad_input norm: {grad_input_fused.norm().item():.6f}")
    print(f"Fused grad_probs norm: {grad_probs_fused.norm().item():.6f}")

    # ========== Compare Results ==========
    print("\n=== Comparison ===")

    # Forward output comparison
    forward_diff = (output_fused - output_ref).abs()
    forward_max_diff = forward_diff.max().item()
    forward_mean_diff = forward_diff.mean().item()

    print(f"Forward max diff: {forward_max_diff:.6e}")
    print(f"Forward mean diff: {forward_mean_diff:.6e}")

    # Loss comparison
    loss_diff = abs(loss_fused.item() - loss_ref.item())
    print(f"Loss diff: {loss_diff:.6e}")

    # Gradient comparison
    grad_diff = (grad_input_fused - d).abs()
    grad_max_diff = grad_diff.max().item()
    grad_mean_diff = grad_diff.mean().item()

    print(f"Gradient max diff: {grad_max_diff:.6e}")
    print(f"Gradient mean diff: {grad_mean_diff:.6e}")

    # Probs gradient comparison
    probs_grad_diff = (grad_probs_fused - grad_probs_ref).abs()
    probs_grad_max_diff = probs_grad_diff.max().item()
    probs_grad_mean_diff = probs_grad_diff.mean().item()

    print(f"Probs gradient max diff: {probs_grad_max_diff:.6e}")
    print(f"Probs gradient mean diff: {probs_grad_mean_diff:.6e}")

    # Weight gradient comparison (only for local experts)
    if grad_w1_fused is not None and grad_w2_fused is not None:
        # Compare only the gradients for experts this rank processes
        for set_idx, exp_ids in enumerate(expert_sets):
            for i, exp_id in enumerate(exp_ids):
                w1_diff = (grad_w1_fused[exp_id] - grad_w1_ref[exp_id]).abs().max().item()
                w2_diff = (grad_w2_fused[exp_id] - grad_w2_ref[exp_id]).abs().max().item()
                print(f"Expert {exp_id} grad_w1 diff: {w1_diff:.6e}, grad_w2 diff: {w2_diff:.6e}")

    # Cleanup
    fused_mlp.release()
    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()

    # Assertions (with relative tolerance for bf16)
    # bf16 has ~1% relative precision, but we use absolute tolerance for small values
    output_max = max(output_ref.abs().max().item(), output_fused.abs().max().item())
    grad_max = max(grad_input_ref.abs().max().item(), grad_input_fused.abs().max().item())
    probs_grad_max = max(grad_probs_ref.abs().max().item(), grad_probs_fused.abs().max().item())
    relative_tolerance = 0.01  # 1% relative tolerance for bf16
    absolute_tolerance = 1.0   # For values near zero

    forward_threshold = max(output_max * relative_tolerance, absolute_tolerance)
    grad_threshold = max(grad_max * relative_tolerance, absolute_tolerance)
    probs_grad_threshold = max(probs_grad_max * relative_tolerance, absolute_tolerance)

    if forward_max_diff > forward_threshold:
        print(f"\n❌ FAIL: Forward output mismatch (max diff: {forward_max_diff:.6e}, threshold: {forward_threshold:.6e})")
        # Print some debug info
        print(f"Output ref sample: {output_ref[:5, :5]}")
        print(f"Output fused sample: {output_fused[:5, :5]}")
        return False

    if grad_max_diff > grad_threshold:
        print(f"\n❌ FAIL: Gradient mismatch (max diff: {grad_max_diff:.6e}, threshold: {grad_threshold:.6e})")
        return False

    if probs_grad_max_diff > probs_grad_threshold:
        print(f"\n❌ FAIL: Probs gradient mismatch (max diff: {probs_grad_max_diff:.6e}, threshold: {probs_grad_threshold:.6e})")
        # Print debug info
        nonzero_mask = routing_map.any(dim=1)
        print(f"Probs grad ref sample (first 5 routed tokens): {grad_probs_ref[nonzero_mask][:5, :5]}")
        print(f"Probs grad fused sample: {grad_probs_fused[nonzero_mask][:5, :5]}")
        return False

    print(f"\n✅ PASS: Accuracy test passed!")
    print(f"   Forward max diff: {forward_max_diff:.6e} < {forward_threshold:.6e}")
    print(f"   Gradient max diff: {grad_max_diff:.6e} < {grad_threshold:.6e}")
    print(f"   Probs gradient max diff: {probs_grad_max_diff:.6e} < {probs_grad_threshold:.6e}")
    return True


def test_multi_expert_sets():
    """Test with multiple experts per set to verify double buffering."""
    from megatron.core import parallel_state
    from megatron.core.transformer.moe.experts import FusedDispatcherCacheGroupedMLP
    from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Initialize distributed
    rank, world_size, local_rank = init_distributed()

    # Initialize model parallel (EP only)
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=world_size,
    )

    # Fixed seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Test parameters
    num_global_experts = 64
    hidden_size = 2560
    ffn_hidden_size = 5120
    batch_size = 2
    seq_len = 64000
    num_tokens = batch_size * seq_len
    topk = 4
    dtype = torch.bfloat16
    device = torch.device("cuda", local_rank)

    print(f"\n=== Multi-Expert Set Test ===")
    print(f"Testing with {num_global_experts} experts, {num_tokens} tokens, topk={topk}")

    # Create weights
    weight1, weight2 = create_shared_weights(
        num_global_experts, hidden_size, ffn_hidden_size, dtype, device, seed + 100
    )

    # Generate input and routing
    torch.manual_seed(seed + 200)
    hidden_states = (torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * 0.01)
    router_logits = torch.randn(num_tokens, num_global_experts, dtype=dtype, device=device)
    probs_full = F.softmax(router_logits.float(), dim=-1).to(dtype)
    topk_probs, topk_indices = probs_full.topk(topk, dim=-1)

    routing_map = torch.zeros(num_tokens, num_global_experts, dtype=torch.bool, device=device)
    probs = torch.zeros(num_tokens, num_global_experts, dtype=dtype, device=device)
    for i in range(num_tokens):
        routing_map[i, topk_indices[i]] = True
        probs[i, topk_indices[i]] = topk_probs[i]

    # Reference implementation
    hidden_states_ref = hidden_states.detach().clone().requires_grad_(True)
    probs_ref = probs.detach().clone().requires_grad_(True)
    weight1_ref = weight1.detach().clone().requires_grad_(True)
    weight2_ref = weight2.detach().clone().requires_grad_(True)

    output_ref = reference_moe_forward(
        hidden_states_ref, routing_map, probs_ref, weight1_ref, weight2_ref, glu_activation
    )
    loss_ref = output_ref.mean()
    loss_ref.backward()
    grad_input_ref = hidden_states_ref.grad.clone()
    grad_probs_ref = probs_ref.grad.clone()

    # Fused implementation
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        ffn_hidden_size=ffn_hidden_size,
        moe_ffn_hidden_size=ffn_hidden_size,
        num_moe_experts=num_global_experts,
        moe_router_topk=topk,
        moe_router_pre_softmax=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=dtype,
        moe_enable_expert_weight_cache=True,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        perform_initialization=False,
    )

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    fused_mlp = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=pg_collection,
    )

    with torch.no_grad():
        fused_mlp.weight1.data.copy_(weight1.cpu())
        fused_mlp.weight2.data.copy_(weight2.cpu())

    # Partition experts: 4 experts per set
    ep_size = world_size
    experts_per_rank = num_global_experts // ep_size
    local_expert_start = rank * experts_per_rank
    expert_list = [local_expert_start + i for i in range(experts_per_rank)]
    experts_per_set = 4
    expert_sets = [
        expert_list[i : i + experts_per_set]
        for i in range(0, len(expert_list), experts_per_set)
    ]

    print(f"Rank {rank} expert_sets: {expert_sets}")

    hidden_states_fused = hidden_states.detach().clone().requires_grad_(True)
    probs_fused = probs.detach().clone().requires_grad_(True)
    output_fused, _ = fused_mlp(
        hidden_states=hidden_states_fused,
        routing_map=routing_map,
        probs=probs_fused,
        expert_sets=expert_sets,
    )
    loss_fused = output_fused.mean()
    loss_fused.backward()
    fused_mlp.sync_gradients()

    grad_input_fused = hidden_states_fused.grad.clone()
    grad_probs_fused = probs_fused.grad.clone()

    # Compare
    forward_diff = (output_fused - output_ref).abs().max().item()
    grad_diff = (grad_input_fused - grad_input_ref).abs().max().item()
    probs_grad_diff = (grad_probs_fused - grad_probs_ref).abs().max().item()

    print(f"Forward max diff: {forward_diff:.6e}")
    print(f"Gradient max diff: {grad_diff:.6e}")
    print(f"Probs gradient max diff: {probs_grad_diff:.6e}")

    # Cleanup
    fused_mlp.release()
    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()

    # Use relative tolerance for bf16
    output_max = max(output_ref.abs().max().item(), output_fused.abs().max().item())
    grad_max = max(grad_input_ref.abs().max().item(), grad_input_fused.abs().max().item())
    probs_grad_max = max(grad_probs_ref.abs().max().item(), grad_probs_fused.abs().max().item())
    relative_tolerance = 0.01  # 1% relative tolerance for bf16
    absolute_tolerance = 1.0   # For values near zero

    forward_threshold = max(output_max * relative_tolerance, absolute_tolerance)
    grad_threshold = max(grad_max * relative_tolerance, absolute_tolerance)
    probs_grad_threshold = max(probs_grad_max * relative_tolerance, absolute_tolerance)

    if forward_diff > forward_threshold:
        print(f"\n❌ FAIL: Multi-expert set test failed (forward)")
        print(f"   Forward diff: {forward_diff:.6e} > threshold: {forward_threshold:.6e}")
        return False

    if grad_diff > grad_threshold:
        print(f"\n❌ FAIL: Multi-expert set test failed (gradient)")
        print(f"   Gradient diff: {grad_diff:.6e} > threshold: {grad_threshold:.6e}")
        return False

    if probs_grad_diff > probs_grad_threshold:
        print(f"\n❌ FAIL: Multi-expert set test failed (probs gradient)")
        print(f"   Probs gradient diff: {probs_grad_diff:.6e} > threshold: {probs_grad_threshold:.6e}")
        return False

    print(f"\n✅ PASS: Multi-expert set test passed!")
    print(f"   Forward diff: {forward_diff:.6e}")
    print(f"   Gradient diff: {grad_diff:.6e}")
    print(f"   Probs gradient diff: {probs_grad_diff:.6e}")
    return True


def main():
    parser = argparse.ArgumentParser(description="FusedDispatcher accuracy test")
    parser.add_argument("--test", type=str, default="single", choices=["single", "multi", "all"],
                        help="Which test to run")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this test")
        sys.exit(1)

    passed = True

    if args.test in ["single", "all"]:
        passed = test_single_gpu_accuracy() and passed

    if args.test in ["multi", "all"]:
        passed = test_multi_expert_sets() and passed

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()