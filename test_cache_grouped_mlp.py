#!/usr/bin/env python
"""Test script for CacheGroupedMLP implementation."""

import torch
import torch.nn.functional as F
from megatron.core.transformer.moe.experts import CacheGroupedMLP, ActivationCache
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
    """Test basic forward pass."""
    print("=" * 60)
    print("Test 1: Basic forward pass")
    print("=" * 60)

    config = create_test_config()
    num_global_experts = 8

    model = CacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    # Create test inputs
    batch_size = 16
    hidden_size = 64

    # Expert sets: [[0, 1], [2, 3], [4, 5, 6, 7]]
    expert_sets = [[0, 1], [2, 3], [4, 5, 6, 7]]

    # Token distribution per expert per set
    # Set 0: experts 0, 1 with 4, 4 tokens
    # Set 1: experts 2, 3 with 3, 3 tokens
    # Set 2: experts 4, 5, 6, 7 with 2, 2, 2, 2 tokens
    tokens_per_expert_per_set = [
        torch.tensor([4, 4], dtype=torch.long),
        torch.tensor([3, 3], dtype=torch.long),
        torch.tensor([2, 2, 2, 2], dtype=torch.long),
    ]

    total_tokens = 4 + 4 + 3 + 3 + 2 + 2 + 2 + 2  # = 22
    probs_per_set = [
        torch.ones(8, dtype=torch.bfloat16) * 0.5,  # 4+4=8 tokens
        torch.ones(6, dtype=torch.bfloat16) * 0.5,  # 3+3=6 tokens
        torch.ones(8, dtype=torch.bfloat16) * 0.5,  # 2+2+2+2=8 tokens
    ]

    hidden_states = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        tokens_per_expert_per_set=tokens_per_expert_per_set,
        probs_per_set=probs_per_set,
        expert_sets=expert_sets,
    )

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (total_tokens, hidden_size), f"Expected {(total_tokens, hidden_size)}, got {output.shape}"
    print("✓ Forward pass successful!")


def test_forward_backward():
    """Test forward and backward pass."""
    print("\n" + "=" * 60)
    print("Test 2: Forward and backward pass")
    print("=" * 60)

    config = create_test_config()
    num_global_experts = 4

    model = CacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    # Expert sets: [[0, 1], [2, 3]]
    expert_sets = [[0, 1], [2, 3]]

    # Token distribution
    tokens_per_expert_per_set = [
        torch.tensor([4, 4], dtype=torch.long),
        torch.tensor([4, 4], dtype=torch.long),
    ]

    total_tokens = 16
    probs_per_set = [
        torch.ones(8, dtype=torch.bfloat16) * 0.5,
        torch.ones(8, dtype=torch.bfloat16) * 0.5,
    ]

    hidden_states = torch.randn(total_tokens, config.hidden_size, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        tokens_per_expert_per_set=tokens_per_expert_per_set,
        probs_per_set=probs_per_set,
        expert_sets=expert_sets,
    )

    # Backward
    loss = output.sum()
    loss.backward()

    print(f"Input requires_grad: {hidden_states.requires_grad}")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Input grad shape: {hidden_states.grad.shape if hidden_states.grad is not None else 'None'}")
    assert hidden_states.grad is not None, "Expected gradient for input"
    print("✓ Backward pass successful!")


def test_activation_offload():
    """Test activation offload feature."""
    print("\n" + "=" * 60)
    print("Test 3: Activation offload")
    print("=" * 60)

    config = create_test_config()
    # Enable activation offload via config attribute
    config.moe_activation_offload = True
    config.moe_enable_expert_weight_cache = True

    num_global_experts = 4

    model = CacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    assert model.activation_offload, "Activation offload should be enabled"
    print(f"Activation offload enabled: {model.activation_offload}")

    # Expert sets
    expert_sets = [[0, 1], [2, 3]]

    tokens_per_expert_per_set = [
        torch.tensor([4, 4], dtype=torch.long),
        torch.tensor([4, 4], dtype=torch.long),
    ]

    total_tokens = 16
    probs_per_set = [
        torch.ones(8, dtype=torch.bfloat16) * 0.5,
        torch.ones(8, dtype=torch.bfloat16) * 0.5,
    ]

    hidden_states = torch.randn(total_tokens, config.hidden_size, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        tokens_per_expert_per_set=tokens_per_expert_per_set,
        probs_per_set=probs_per_set,
        expert_sets=expert_sets,
    )

    # Check activation was offloaded
    assert model.activation_cache.is_cached, "Activation should be cached"
    print("✓ Activation offloaded to CPU")

    # Backward
    loss = output.sum()
    loss.backward()

    print(f"Input grad shape: {hidden_states.grad.shape}")
    print("✓ Activation offload test successful!")


def test_glu_activation():
    """Test with GLU activation."""
    print("\n" + "=" * 60)
    print("Test 4: GLU activation")
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

    model = CacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    # Check weight shapes
    # weight1 should have fc1_out_features = ffn_hidden_size * 2 for GLU
    expected_fc1_out = config.moe_ffn_hidden_size * 2
    print(f"Weight1 shape: {model.weight1.shape}")
    print(f"Weight2 shape: {model.weight2.shape}")
    assert model.weight1.shape == (num_global_experts, config.hidden_size, expected_fc1_out)

    expert_sets = [[0, 1], [2, 3]]

    tokens_per_expert_per_set = [
        torch.tensor([4, 4], dtype=torch.long),
        torch.tensor([4, 4], dtype=torch.long),
    ]

    total_tokens = 16
    probs_per_set = [
        torch.ones(8, dtype=torch.bfloat16) * 0.5,
        torch.ones(8, dtype=torch.bfloat16) * 0.5,
    ]

    hidden_states = torch.randn(total_tokens, config.hidden_size, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        tokens_per_expert_per_set=tokens_per_expert_per_set,
        probs_per_set=probs_per_set,
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
    print("Test 5: Empty expert set (no tokens)")
    print("=" * 60)

    config = create_test_config()
    num_global_experts = 4

    model = CacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    # One set with tokens, one without
    expert_sets = [[0, 1], [2, 3]]

    tokens_per_expert_per_set = [
        torch.tensor([4, 4], dtype=torch.long),  # Has tokens
        torch.tensor([0, 0], dtype=torch.long),  # No tokens
    ]

    total_tokens = 8
    probs_per_set = [
        torch.ones(8, dtype=torch.bfloat16) * 0.5,
        torch.ones(0, dtype=torch.bfloat16),  # Empty probs
    ]

    hidden_states = torch.randn(total_tokens, config.hidden_size, dtype=torch.bfloat16, device='cuda')

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        tokens_per_expert_per_set=tokens_per_expert_per_set,
        probs_per_set=probs_per_set,
        expert_sets=expert_sets,
    )

    print(f"Output shape: {output.shape}")
    assert output.shape == (total_tokens, config.hidden_size)
    print("✓ Empty expert set test successful!")


def test_gradient_sync():
    """Test gradient synchronization to parameters."""
    print("\n" + "=" * 60)
    print("Test 6: Gradient synchronization")
    print("=" * 60)

    config = create_test_config()
    num_global_experts = 4

    model = CacheGroupedMLP(
        num_global_experts=num_global_experts,
        config=config,
        pg_collection=None,
    )

    expert_sets = [[0, 1], [2, 3]]

    tokens_per_expert_per_set = [
        torch.tensor([4, 4], dtype=torch.long),
        torch.tensor([4, 4], dtype=torch.long),
    ]

    total_tokens = 16
    probs_per_set = [
        torch.ones(8, dtype=torch.bfloat16) * 0.5,
        torch.ones(8, dtype=torch.bfloat16) * 0.5,
    ]

    hidden_states = torch.randn(total_tokens, config.hidden_size, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Forward
    output, _ = model(
        hidden_states=hidden_states,
        tokens_per_expert_per_set=tokens_per_expert_per_set,
        probs_per_set=probs_per_set,
        expert_sets=expert_sets,
    )

    # Backward
    loss = output.sum()
    loss.backward()

    # Before sync, parameter grads should be None
    assert model.weight1.grad is None, "Weight1 grad should be None before sync"
    assert model.weight2.grad is None, "Weight2 grad should be None before sync"

    # Sync gradients
    model.sync_gradients()

    # After sync, parameter grads should be set
    assert model.weight1.grad is not None, "Weight1 grad should be set after sync"
    assert model.weight2.grad is not None, "Weight2 grad should be set after sync"
    print(f"Weight1 grad shape: {model.weight1.grad.shape}")
    print(f"Weight2 grad shape: {model.weight2.grad.shape}")

    assert model.weight1.grad.shape == model.weight1.shape
    assert model.weight2.grad.shape == model.weight2.shape

    print("✓ Gradient synchronization test successful!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running CacheGroupedMLP Tests")
    print("=" * 60)

    test_basic_forward()
    test_forward_backward()
    test_activation_offload()
    test_glu_activation()
    test_empty_expert_set()
    test_gradient_sync()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()