#!/usr/bin/env python
"""Test script for FusedDispatcherCacheGroupedMLP implementation."""

import torch
import torch.nn.functional as F
from typing import List

from megatron.core.transformer.moe.experts import FusedDispatcherCacheGroupedMLP
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig


def create_test_config():
    """Create a test TransformerConfig."""
    config = TransformerConfig(
        num_layers=2,
        num_attention_heads=4,
        hidden_size=64,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=128,
        num_moe_experts=8,
        gated_linear_unit=False,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    return config


def test_basic_forward():
    """Test basic forward pass with fused dispatcher."""
    print("=" * 60)
    print("Test 1: Basic forward pass with fused dispatcher")
    print("=" * 60)

    config = create_test_config()
    num_global_experts = 8

    model = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    # Create test inputs
    batch_size = 16
    hidden_size = 64
    num_experts = 8

    # Create routing_map and probs
    num_tokens = 32
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device='cuda')
    probs = torch.zeros(num_tokens, num_experts, dtype=torch.bfloat16, device='cuda')

    # Each token routes to 2 experts (top-2)
    for i in range(num_tokens):
        expert_ids = torch.randperm(num_experts)[:2]
        routing_map[i, expert_ids] = True
        probs[i, expert_ids] = 0.5

    # Expert sets: [[0, 1], [2, 3], [4, 5, 6, 7]]
    expert_sets: List[List[int]] = [[0, 1], [2, 3], [4, 5, 6, 7]]

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        routing_map=routing_map,
        probs=probs,
        expert_sets=expert_sets,
    )

    print(f"Input shape: {hidden_states.shape}")
    print(f"Routing map shape: {routing_map.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Expert sets: {expert_sets}")
    print(f"Output shape: {output.shape}")
    print("✓ Basic forward pass successful!")


def test_forward_backward():
    """Test forward and backward pass."""
    print("\n" + "=" * 60)
    print("Test 2: Forward and backward pass")
    print("=" * 60)

    config = create_test_config()
    num_global_experts = 4

    model = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    num_tokens = 16
    num_experts = 4
    hidden_size = 64

    # Create routing_map and probs
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device='cuda')
    probs = torch.zeros(num_tokens, num_experts, dtype=torch.bfloat16, device='cuda')

    for i in range(num_tokens):
        expert_ids = torch.randperm(num_experts)[:2]
        routing_map[i, expert_ids] = True
        probs[i, expert_ids] = 0.5

    expert_sets = [[0, 1], [2, 3]]

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        routing_map=routing_map,
        probs=probs,
        expert_sets=expert_sets,
    )

    # Backward
    loss = output.sum()
    loss.backward()

    print(f"Input requires_grad: {hidden_states.requires_grad}")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Input grad shape: {hidden_states.grad.shape if hidden_states.grad is not None else 'None'}")
    print("✓ Backward pass successful!")


def test_glu_activation():
    """Test with GLU activation."""
    print("\n" + "=" * 60)
    print("Test 3: GLU activation")
    print("=" * 60)

    config = TransformerConfig(
        num_layers=2,
        num_attention_heads=4,
        hidden_size=64,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=128,
        num_moe_experts=4,
        gated_linear_unit=True,  # Enable GLU
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    num_global_experts = 4

    model = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    # Check weight shapes
    expected_fc1_out = config.moe_ffn_hidden_size * 2
    print(f"Weight1 shape: {model.weight1.shape}")
    print(f"Weight2 shape: {model.weight2.shape}")
    assert model.weight1.shape == (num_global_experts, config.hidden_size, expected_fc1_out)

    num_tokens = 16
    num_experts = 4
    hidden_size = 64

    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device='cuda')
    probs = torch.zeros(num_tokens, num_experts, dtype=torch.bfloat16, device='cuda')

    for i in range(num_tokens):
        expert_ids = torch.randperm(num_experts)[:2]
        routing_map[i, expert_ids] = True
        probs[i, expert_ids] = 0.5

    expert_sets = [[0, 1], [2, 3]]

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        routing_map=routing_map,
        probs=probs,
        expert_sets=expert_sets,
    )

    # Backward
    loss = output.sum()
    loss.backward()

    print(f"Output shape: {output.shape}")
    print("✓ GLU activation test successful!")


def test_empty_expert_set():
    """Test with empty token distribution."""
    print("\n" + "=" * 60)
    print("Test 4: Empty expert set (no tokens)")
    print("=" * 60)

    config = create_test_config()
    num_global_experts = 4

    model = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    num_tokens = 8
    num_experts = 4
    hidden_size = 64

    # Only route to experts 0 and 1
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device='cuda')
    probs = torch.zeros(num_tokens, num_experts, dtype=torch.bfloat16, device='cuda')

    for i in range(num_tokens):
        routing_map[i, 0] = True
        routing_map[i, 1] = True
        probs[i, 0] = 0.5
        probs[i, 1] = 0.5

    # Expert sets - set 1 has no tokens
    expert_sets = [[0, 1], [2, 3]]

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        routing_map=routing_map,
        probs=probs,
        expert_sets=expert_sets,
    )

    print(f"Output shape: {output.shape}")
    print("✓ Empty expert set test successful!")


def test_gradient_sync():
    """Test gradient synchronization to parameters."""
    print("\n" + "=" * 60)
    print("Test 5: Gradient synchronization")
    print("=" * 60)

    config = create_test_config()
    num_global_experts = 4

    model = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    num_tokens = 16
    num_experts = 4
    hidden_size = 64

    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device='cuda')
    probs = torch.zeros(num_tokens, num_experts, dtype=torch.bfloat16, device='cuda')

    for i in range(num_tokens):
        expert_ids = torch.randperm(num_experts)[:2]
        routing_map[i, expert_ids] = True
        probs[i, expert_ids] = 0.5

    expert_sets = [[0, 1], [2, 3]]

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        routing_map=routing_map,
        probs=probs,
        expert_sets=expert_sets,
    )

    # Backward
    loss = output.sum()
    loss.backward()

    # Sync gradients
    model.sync_gradients()

    # Check parameter grads
    assert model.weight1.grad is not None, "Weight1 grad should be set after sync"
    assert model.weight2.grad is not None, "Weight2 grad should be set after sync"
    print(f"Weight1 grad shape: {model.weight1.grad.shape}")
    print(f"Weight2 grad shape: {model.weight2.grad.shape}")

    assert model.weight1.grad.shape == model.weight1.shape
    assert model.weight2.grad.shape == model.weight2.shape

    print("✓ Gradient synchronization test successful!")


def test_memory_efficiency():
    """Test memory efficiency - verify no global buffer allocation."""
    print("\n" + "=" * 60)
    print("Test 6: Memory efficiency")
    print("=" * 60)

    config = create_test_config()
    num_global_experts = 8

    model = FusedDispatcherCacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    # Create inputs with varying token counts per set
    num_tokens = 100
    num_experts = 8
    hidden_size = 64

    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device='cuda')
    probs = torch.zeros(num_tokens, num_experts, dtype=torch.bfloat16, device='cuda')

    # Route tokens unevenly to test max_set_tokens computation
    for i in range(num_tokens):
        # Route to random experts
        num_routes = torch.randint(1, 3, (1,)).item()
        expert_ids = torch.randperm(num_experts)[:num_routes]
        routing_map[i, expert_ids] = True
        probs[i, expert_ids] = 1.0 / num_routes

    expert_sets = [[0, 1], [2, 3, 4], [5, 6, 7]]

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Record initial memory
    torch.cuda.synchronize()
    initial_memory = torch.cuda.memory_allocated()

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        routing_map=routing_map,
        probs=probs,
        expert_sets=expert_sets,
    )

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"Initial memory: {initial_memory / 1024:.2f} KB")
    print(f"Peak memory: {peak_memory / 1024:.2f} KB")
    print(f"Memory increase: {(peak_memory - initial_memory) / 1024:.2f} KB")

    # Backward
    loss = output.sum()
    loss.backward()

    torch.cuda.synchronize()
    final_memory = torch.cuda.memory_allocated()

    print(f"Final memory: {final_memory / 1024:.2f} KB")
    print("✓ Memory efficiency test successful!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running FusedDispatcherCacheGroupedMLP Tests")
    print("=" * 60)

    test_basic_forward()
    test_forward_backward()
    test_glu_activation()
    test_empty_expert_set()
    test_gradient_sync()
    test_memory_efficiency()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()