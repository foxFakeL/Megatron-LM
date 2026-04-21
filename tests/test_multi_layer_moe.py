#!/usr/bin/env python
"""Multi-layer MoE test to verify gradient correctness with shared GPU workspace.

This test creates multiple FusedDispatcherCacheGroupedMLP layers to simulate
the actual training scenario where GPU workspace is shared across layers.

Test structure: Transformer-style blocks with residual connections
    x = x + MLP(x)      # MLP replaces attention for gradient flow testing
    x = x + MoE(x)      # FusedDispatcherCacheGroupedMLP with residual

Usage:
    # Single GPU test
    CUDA_VISIBLE_DEVICES=0 python tests/test_multi_layer_moe.py

    # Multi-GPU test (EP=2)
    CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 python tests/test_multi_layer_moe.py
"""

import os
import sys
import argparse
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """Simple MLP block for testing residual connections."""
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_out = self.fc1(x)
        # GLU activation
        chunked = torch.chunk(fc1_out, 2, dim=-1)
        intermediate = F.silu(chunked[0]) * chunked[1]
        return self.fc2(intermediate)


def create_shared_weights(
    num_experts: int,
    hidden_size: int,
    ffn_hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    scale: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create weight tensors with fixed seed for reproducibility."""
    torch.manual_seed(seed)
    # Use scaled initialization similar to transformer models
    weight1 = torch.randn(
        num_experts, hidden_size, ffn_hidden_size * 2, dtype=dtype, device=device
    ) * scale
    weight2 = torch.randn(
        num_experts, ffn_hidden_size, hidden_size, dtype=dtype, device=device
    ) * scale
    return weight1, weight2


def glu_activation(x):
    """GLU activation function."""
    x = torch.chunk(x, 2, dim=-1)
    return F.silu(x[0]) * x[1]


def reference_mlp_forward(
    hidden_states: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
) -> torch.Tensor:
    """Reference MLP forward (GLU activation)."""
    fc1_out = hidden_states @ weight1
    intermediate = glu_activation(fc1_out)
    return intermediate @ weight2


def reference_moe_forward(
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation for single layer."""
    num_tokens, hidden_size = hidden_states.shape
    num_experts = weight1.shape[0]
    output = torch.zeros(
        num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )

    total_experts_used = 0
    total_tokens_processed = 0

    for exp_id in range(num_experts):
        expert_mask = routing_map[:, exp_id]
        if not expert_mask.any():
            continue

        total_experts_used += 1
        token_indices = expert_mask.nonzero(as_tuple=True)[0]
        total_tokens_processed += len(token_indices)
        expert_input = hidden_states[token_indices]
        expert_probs = probs[token_indices, exp_id]

        fc1_out = expert_input @ weight1[exp_id]
        intermediate = glu_activation(fc1_out) * expert_probs.unsqueeze(-1)
        fc2_out = intermediate @ weight2[exp_id]

        output.index_add_(0, token_indices, fc2_out)

    print(f"  Reference: {total_experts_used} experts used, {total_tokens_processed} tokens processed")
    print(f"  Reference output stats: mean={output.mean().item():.6e}, std={output.std().item():.6e}, max={output.max().item():.6e}")

    return output


def reference_multi_layer_forward(
    hidden_states: torch.Tensor,
    routing_maps: List[torch.Tensor],
    probs_list: List[torch.Tensor],
    moe_weights1: List[torch.Tensor],
    moe_weights2: List[torch.Tensor],
    mlp_weights1: List[torch.Tensor],
    mlp_weights2: List[torch.Tensor],
) -> torch.Tensor:
    """Reference implementation for multiple transformer-style blocks.

    Each block: x = x + MLP(x) + MoE(x)
    """
    x = hidden_states
    for i, (routing_map, probs, moe_w1, moe_w2, mlp_w1, mlp_w2) in enumerate(
        zip(routing_maps, probs_list, moe_weights1, moe_weights2, mlp_weights1, mlp_weights2)
    ):
        # MLP block (replaces attention for testing)
        mlp_out = reference_mlp_forward(x, mlp_w1, mlp_w2)
        x = x + mlp_out  # Residual connection

        # MoE block
        moe_out = reference_moe_forward(x, routing_map, probs, moe_w1, moe_w2)
        x = x + moe_out  # Residual connection

        print(f"  Reference Layer {i}: after MLP mean={x.mean().item():.6e}, std={x.std().item():.6e}")
    return x


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


def test_multi_layer_gradient():
    """Test multi-layer MoE gradient correctness with shared GPU workspace."""
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
    num_layers = 5  # Test with 5 layers like actual training
    num_global_experts = 64
    hidden_size = 1024  # Reasonable hidden size
    ffn_hidden_size = 1024  # Match hidden_size for simplicity
    batch_size = 2
    seq_len = 25600  # Smaller for faster testing
    num_tokens = batch_size * seq_len
    topk = 4
    dtype = torch.bfloat16
    device = torch.device("cuda", local_rank)

    # Use smaller weight initialization scale for numerical stability
    weight_scale = 0.02  # Similar to transformer init

    print(f"\n{'='*60}")
    print(f"Multi-Layer MoE Gradient Test (with Residual + MLP)")
    print(f"{'='*60}")
    print(f"Layers: {num_layers}")
    print(f"Experts: {num_global_experts}")
    print(f"Tokens: {num_tokens}")
    print(f"EP size: {world_size}")
    print(f"Structure: x = x + MLP(x) + MoE(x)")
    print(f"{'='*60}\n")

    # Create config
    config = TransformerConfig(
        num_layers=num_layers,
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

    # Create ProcessGroupCollection
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Define expert_sets
    ep_size = world_size
    experts_per_rank = num_global_experts // ep_size
    local_expert_start = rank * experts_per_rank
    expert_list = [local_expert_start + i for i in range(experts_per_rank)]
    experts_per_set = 16
    expert_sets = [
        expert_list[i:i + experts_per_set]
        for i in range(0, len(expert_list), experts_per_set)
    ]

    # ========== Create Multiple Layers ==========
    print("Creating multiple MoE + MLP layers with residual connections...")
    moe_layers = []
    mlp_modules = []
    moe_weights1_ref = []
    moe_weights2_ref = []
    mlp_weights1_ref = []
    mlp_weights2_ref = []

    for layer_idx in range(num_layers):
        # Create MoE layer
        moe_layer = FusedDispatcherCacheGroupedMLP(
            num_global_experts=num_global_experts,
            config=config,
            pg_collection=pg_collection,
            layer_number=layer_idx + 1,
        )
        moe_layers.append(moe_layer)

        # Create MLP module (simple FFN with GLU) - convert to bf16
        mlp = SimpleMLP(hidden_size, ffn_hidden_size).to(device).to(dtype)
        mlp_modules.append(mlp)

        # Create MoE reference weights
        moe_w1, moe_w2 = create_shared_weights(
            num_global_experts, hidden_size, ffn_hidden_size, dtype, device,
            seed + layer_idx * 100, scale=weight_scale
        )
        moe_weights1_ref.append(moe_w1)
        moe_weights2_ref.append(moe_w2)

        # Create MLP reference weights (single expert style)
        torch.manual_seed(seed + layer_idx * 100 + 50)
        mlp_w1 = torch.randn(hidden_size, ffn_hidden_size * 2, dtype=dtype, device=device) * weight_scale
        mlp_w2 = torch.randn(ffn_hidden_size, hidden_size, dtype=dtype, device=device) * weight_scale
        mlp_weights1_ref.append(mlp_w1)
        mlp_weights2_ref.append(mlp_w2)

        # Set MoE layer weights
        with torch.no_grad():
            moe_layer.weight1.data.copy_(moe_w1.cpu())
            moe_layer.weight2.data.copy_(moe_w2.cpu())

        # Set MLP weights
        with torch.no_grad():
            mlp.fc1.weight.data.copy_(mlp_w1.t())  # Linear stores transposed
            mlp.fc2.weight.data.copy_(mlp_w2.t())

    print(f"Created {len(moe_layers)} MoE layers and {len(mlp_modules)} MLP modules\n")

    # ========== Generate Input and Routing ==========
    torch.manual_seed(seed + 1000)
    # Use smaller input values for numerical stability
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * 0.1

    # Generate routing for each layer (different routing per layer)
    # IMPORTANT: Only route to local experts (same as expert_sets)
    # This ensures the test matches the actual MoE behavior
    routing_maps = []
    probs_list = []

    for layer_idx in range(num_layers):
        torch.manual_seed(seed + 2000 + layer_idx * 10)
        # Only generate router logits for local experts
        router_logits = torch.randn(num_tokens, len(expert_list), dtype=dtype, device=device)
        probs_full = F.softmax(router_logits.float(), dim=-1).to(dtype)
        topk_probs, topk_indices_local = probs_full.topk(min(topk, len(expert_list)), dim=-1)

        # Map local indices back to global expert IDs
        topk_indices = torch.tensor([[expert_list[idx] for idx in row] for row in topk_indices_local.tolist()],
                                     dtype=torch.long, device=device)

        routing_map = torch.zeros(num_tokens, num_global_experts, dtype=torch.bool, device=device)
        probs = torch.zeros(num_tokens, num_global_experts, dtype=dtype, device=device)
        for i in range(num_tokens):
            routing_map[i, topk_indices[i]] = True
            probs[i, topk_indices[i]] = topk_probs[i]

        routing_maps.append(routing_map)
        probs_list.append(probs)

        # Debug: check routing
        num_routed = routing_map.sum().item()
        print(f"  Layer {layer_idx}: {num_routed} token-expert pairs routed to {len(expert_list)} local experts")

    # ========== Reference Implementation ==========
    print("=== Running Reference Implementation ===")
    hidden_states_ref = hidden_states.detach().clone().requires_grad_(True)

    # Enable gradients for reference weights
    moe_weights1_ref_grad = [w.detach().clone().requires_grad_(True) for w in moe_weights1_ref]
    moe_weights2_ref_grad = [w.detach().clone().requires_grad_(True) for w in moe_weights2_ref]
    mlp_weights1_ref_grad = [w.detach().clone().requires_grad_(True) for w in mlp_weights1_ref]
    mlp_weights2_ref_grad = [w.detach().clone().requires_grad_(True) for w in mlp_weights2_ref]

    output_ref = reference_multi_layer_forward(
        hidden_states_ref, routing_maps, probs_list,
        moe_weights1_ref_grad, moe_weights2_ref_grad,
        mlp_weights1_ref_grad, mlp_weights2_ref_grad,
    )
    # Use sum() instead of mean() to prevent gradient vanishing across 5 layers
    loss_ref = output_ref.sum()
    loss_ref.backward()

    grad_input_ref = hidden_states_ref.grad.clone()
    grad_moe_w1_ref = [w.grad.clone() for w in moe_weights1_ref_grad]
    grad_moe_w2_ref = [w.grad.clone() for w in moe_weights2_ref_grad]
    grad_mlp_w1_ref = [w.grad.clone() for w in mlp_weights1_ref_grad]
    grad_mlp_w2_ref = [w.grad.clone() for w in mlp_weights2_ref_grad]

    print(f"Reference loss: {loss_ref.item():.6f}")
    print(f"Reference grad_input norm: {grad_input_ref.norm().item():.6f}")
    for i, (gw1, gw2) in enumerate(zip(grad_moe_w1_ref, grad_moe_w2_ref)):
        print(f"  Layer {i} MoE: grad_w1 norm={gw1.norm().item():.6f}, grad_w2 norm={gw2.norm().item():.6f}")
    for i, (gw1, gw2) in enumerate(zip(grad_mlp_w1_ref, grad_mlp_w2_ref)):
        print(f"  Layer {i} MLP: grad_w1 norm={gw1.norm().item():.6f}, grad_w2 norm={gw2.norm().item():.6f}")

    # ========== FusedDispatcher Implementation ==========
    print("\n=== Running FusedDispatcher Implementation ===")

    # Forward pass through all layers with residual connections
    x = hidden_states.detach().clone().requires_grad_(True)

    for layer_idx, (moe_layer, mlp, routing_map, probs) in enumerate(
        zip(moe_layers, mlp_modules, routing_maps, probs_list)
    ):
        # MLP block with residual
        mlp_out = mlp(x)
        x = x + mlp_out

        # MoE block with residual
        moe_out, _ = moe_layer(
            hidden_states=x,
            routing_map=routing_map,
            probs=probs,
            expert_sets=expert_sets,
        )
        x = x + moe_out

        print(f"  Layer {layer_idx}: x mean={x.mean().item():.6e}, std={x.std().item():.6e}")

    # Use sum() instead of mean() to prevent gradient vanishing across 5 layers
    loss = x.sum()
    print(f"FusedDispatcher loss: {loss.item():.6f}")

    # CRITICAL: x is a non-leaf tensor (output of multiple layers)
    # Must call retain_grad() to access .grad after backward
    x.retain_grad()

    # Backward pass
    loss.backward()

    # CRITICAL: Sync CPU gradients to weight.grad for gradient checking
    for moe_layer in moe_layers:
        moe_layer.sync_gradients()

    # x is a non-leaf tensor, use retain_grad() to access its gradient
    grad_input = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
    print(f"FusedDispatcher grad_input norm: {grad_input.norm().item():.6f}")

    # Get gradient norms from each MoE layer
    for layer_idx, moe_layer in enumerate(moe_layers):
        if moe_layer.weight1.grad is not None and moe_layer.weight2.grad is not None:
            # Sum gradients across all experts
            grad_w1 = moe_layer.weight1.grad.sum(dim=0)
            grad_w2 = moe_layer.weight2.grad.sum(dim=0)
            print(f"  Layer {layer_idx} MoE: grad_w1 norm={grad_w1.norm().item():.6f}, grad_w2 norm={grad_w2.norm().item():.6f}")
        else:
            print(f"  Layer {layer_idx} MoE: NO GRADIENTS!")

    # Get gradient norms from each MLP module
    for layer_idx, mlp in enumerate(mlp_modules):
        if mlp.fc1.weight.grad is not None and mlp.fc2.weight.grad is not None:
            grad_w1 = mlp.fc1.weight.grad
            grad_w2 = mlp.fc2.weight.grad
            print(f"  Layer {layer_idx} MLP: grad_w1 norm={grad_w1.norm().item():.6f}, grad_w2 norm={grad_w2.norm().item():.6f}")
        else:
            print(f"  Layer {layer_idx} MLP: NO GRADIENTS!")

    # ========== Compare Results ==========
    print("\n=== Comparison ===")

    # Check loss
    loss_diff = abs(loss.item() - loss_ref.item())
    print(f"Loss difference: {loss_diff:.6e}")

    # Check grad_input
    if grad_input is not None:
        grad_diff = (grad_input - grad_input_ref).norm().item()
        grad_rel_diff = grad_diff / (grad_input_ref.norm().item() + 1e-8)
        print(f"Grad input difference: {grad_diff:.6e} (relative: {grad_rel_diff:.6e})")
    else:
        print("ERROR: No gradient computed for input!")

    # Check for NaN/Inf
    has_nan = torch.isnan(x).any().item() or torch.isinf(x).any().item()
    has_nan_grad = (grad_input is not None and
                    (torch.isnan(grad_input).any().item() or torch.isinf(grad_input).any().item()))

    print(f"\nOutput has NaN/Inf: {has_nan}")
    print(f"Gradient has NaN/Inf: {has_nan_grad}")

    # Final verdict
    print("\n=== Verdict ===")
    if loss_diff < 1e-3 and not has_nan and not has_nan_grad:
        print("PASS: Multi-layer gradient test passed!")
    else:
        print("FAIL: Multi-layer gradient test failed!")
        print(f"  Loss diff: {loss_diff:.6e}")
        print(f"  Output NaN/Inf: {has_nan}")
        print(f"  Gradient NaN/Inf: {has_nan_grad}")

    # Cleanup
    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    test_multi_layer_gradient()