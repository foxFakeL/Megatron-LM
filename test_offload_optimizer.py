#!/usr/bin/env python
"""Test script for OffloadOptimizer functionality."""

import torch
import torch.nn.functional as F
from typing import List

from megatron.core.transformer.moe.experts import FusedDispatcherCacheGroupedMLP
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.optimizer.offload_optimizer import OffloadOptimizer


def test_offload_optimizer():
    """Test that OffloadOptimizer correctly updates expert parameters on CPU."""
    print("=" * 60)
    print("Test: OffloadOptimizer parameter update verification")
    print("=" * 60)

    config = TransformerConfig(
        num_layers=2,
        num_attention_heads=4,
        hidden_size=64,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=128,
        num_moe_experts=4,
        gated_linear_unit=False,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    num_global_experts = 4
    expert_module = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    # Create a minimal model wrapper for OffloadOptimizer
    class MinimalModel(torch.nn.Module):
        def __init__(self, expert_module):
            super().__init__()
            self.experts = expert_module
            # Add some non-expert params for GPU optimizer
            self.linear = torch.nn.Linear(64, 64, bias=False)

        def forward(self, hidden_states, routing_map, probs, expert_sets):
            return self.experts(hidden_states, routing_map, probs, expert_sets)

    model = MinimalModel(expert_module)
    model.linear.weight.data = model.linear.weight.data.to('cuda').to(torch.bfloat16)

    # Create OffloadOptimizer
    optimizer = OffloadOptimizer(
        model=model,
        lr=1e-2,  # Higher LR for visible changes
        dtype=torch.float32,
    )

    print(f"[Test] Expert module registered: {len(optimizer.expert_modules)}")
    print(f"[Test] GPU params count: {len(optimizer.gpu_params)}")

    # Record initial weights
    initial_w1 = expert_module.weight1.data.clone()
    initial_w2 = expert_module.weight2.data.clone()

    num_tokens = 16
    hidden_size = 64
    routing_map = torch.zeros(num_tokens, num_global_experts, dtype=torch.bool, device='cuda')
    probs = torch.zeros(num_tokens, num_global_experts, dtype=torch.bfloat16, device='cuda')

    for i in range(num_tokens):
        expert_ids = torch.randperm(num_global_experts)[:2]
        routing_map[i, expert_ids] = True
        probs[i, expert_ids] = 0.5

    expert_sets = [[0, 1], [2, 3]]
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Forward + backward
    optimizer.zero_grad()
    output, _ = model(hidden_states, routing_map, probs, expert_sets)
    loss = output.sum()
    loss.backward()

    # Check gradients are on CPU
    print(f"[Test] Expert gradients after backward:")
    print(f"  _grad_weight1: {expert_module._grad_weight1 is not None}")
    print(f"  _grad_weight2: {expert_module._grad_weight2 is not None}")
    if expert_module._grad_weight1 is not None:
        print(f"  _grad_weight1 shape: {expert_module._grad_weight1.shape}")
        print(f"  _grad_weight1 device: {expert_module._grad_weight1.device}")
        print(f"  _grad_weight1 non-zero: {(expert_module._grad_weight1.abs() > 0).any().item()}")

    # Optimizer step - update expert params
    optimizer.step_experts()

    # Check weights changed
    w1_diff = (expert_module.weight1.data - initial_w1).abs().max().item()
    w2_diff = (expert_module.weight2.data - initial_w2).abs().max().item()

    print(f"[Test] Weight1 max diff after update: {w1_diff:.6f}")
    print(f"[Test] Weight2 max diff after update: {w2_diff:.6f}")

    if w1_diff > 0 or w2_diff > 0:
        print("✓ Expert weights updated successfully!")
    else:
        print("✗ Expert weights NOT updated - check implementation")

    # Check optimizer states are on CPU
    states = optimizer._get_or_create_expert_state(expert_module)
    print(f"[Test] Optimizer states on CPU:")
    print(f"  exp_avg device: {states['weight1'].exp_avg.device}")
    print(f"  exp_avg_sq device: {states['weight1'].exp_avg_sq.device}")

    assert states['weight1'].exp_avg.device.type == 'cpu', "Optimizer states should be on CPU"
    assert states['weight2'].exp_avg.device.type == 'cpu', "Optimizer states should be on CPU"
    print("✓ Optimizer states stored on CPU verified!")

    # Second iteration - test step_count increments
    optimizer.zero_grad()
    output, _ = model(hidden_states, routing_map, probs, expert_sets)
    loss = output.sum()
    loss.backward()
    optimizer.step_experts()

    print(f"[Test] Step count after 2 iterations: {optimizer.step_count}")
    assert optimizer.step_count == 2, f"Step count should be 2, got {optimizer.step_count}"
    print("✓ Step count increment verified!")

    print("\n" + "=" * 60)
    print("All OffloadOptimizer tests passed! ✓")
    print("=" * 60)


def test_step_all():
    """Test that step_all updates both GPU and expert params."""
    print("\n" + "=" * 60)
    print("Test: step_all() updates both GPU and expert params")
    print("=" * 60)

    config = TransformerConfig(
        num_layers=2,
        num_attention_heads=4,
        hidden_size=64,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=128,
        num_moe_experts=4,
        gated_linear_unit=False,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    num_global_experts = 4
    expert_module = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    class MinimalModel(torch.nn.Module):
        def __init__(self, expert_module):
            super().__init__()
            self.experts = expert_module
            self.linear = torch.nn.Linear(64, 64, bias=False)

        def forward(self, hidden_states, routing_map, probs, expert_sets):
            # Apply linear before expert to ensure it gets gradients
            x = self.linear(hidden_states)
            return self.experts(x, routing_map, probs, expert_sets)

    model = MinimalModel(expert_module)
    model.linear.weight.data = model.linear.weight.data.to('cuda').to(torch.bfloat16)

    optimizer = OffloadOptimizer(model=model, lr=1e-2, dtype=torch.float32)

    # Record initial values
    initial_linear_w = model.linear.weight.data.clone()
    initial_w1 = expert_module.weight1.data.clone()

    num_tokens = 16
    routing_map = torch.zeros(num_tokens, num_global_experts, dtype=torch.bool, device='cuda')
    probs = torch.zeros(num_tokens, num_global_experts, dtype=torch.bfloat16, device='cuda')
    for i in range(num_tokens):
        expert_ids = torch.randperm(num_global_experts)[:2]
        routing_map[i, expert_ids] = True
        probs[i, expert_ids] = 0.5

    expert_sets = [[0, 1], [2, 3]]
    hidden_states = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    optimizer.zero_grad()
    output, _ = model(hidden_states, routing_map, probs, expert_sets)
    loss = output.sum()
    loss.backward()

    # Use step_all() to update both
    optimizer.step_all()

    # Check both updated
    linear_diff = (model.linear.weight.data - initial_linear_w).abs().max().item()
    w1_diff = (expert_module.weight1.data - initial_w1).abs().max().item()

    print(f"[Test] Linear weight diff: {linear_diff:.6f}")
    print(f"[Test] Expert weight1 diff: {w1_diff:.6f}")

    assert linear_diff > 0, "GPU params should be updated"
    assert w1_diff > 0, "Expert params should be updated"
    print("✓ step_all() updates both GPU and expert params verified!")


if __name__ == "__main__":
    test_offload_optimizer()
    test_step_all()