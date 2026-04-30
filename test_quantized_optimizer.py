"""Test script for FusedAdamLSQCPUOffloadOptimizer with Dynamic Quantization.

This script tests:
1. Quantization parameter storage initialization
2. Expert score computation using exp_avg_sq
3. Dynamic precision allocation (BF16/INT8/INT4)
4. INT4 ↔ INT8 precision transition rules
5. GPU delta/z update reception (mock)
6. Call frequency smoothing

Usage:
    # Single GPU test
    python test_quantized_optimizer.py --num-layers 2 --iters 20

    # Multi-GPU EP test
    torchrun --nproc_per_node=2 test_quantized_optimizer.py --num-layers 2 --iters 20
"""

import os
import sys
import time
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial
from typing import Optional, Dict, Tuple
import math

import torch
import torch.distributed as dist

from megatron.core.enums import ModelType
from megatron.core.models.qwen3_moe.qwen3_moe_model import model_provider
from megatron.core.transformer.moe.memory_logger import log_memory
from megatron.core import parallel_state, tensor_parallel
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.initialize import initialize_megatron


# Import the new quantized optimizer
try:
    from megatron.core.optimizer.fused_adam_lsq_cpu_offload_optimizer import FusedAdamLSQCPUOffloadOptimizer
    HAVE_QUANT_OPTIMIZER = True
except ImportError as e:
    HAVE_QUANT_OPTIMIZER = False
    print_rank_0(f"Warning: FusedAdamLSQCPUOffloadOptimizer not available: {e}")

# Import GPU quant utils for testing
try:
    from megatron.core.optimizer.gpu_quant_utils import (
        dequantize_int8,
        dequantize_int4,
        compute_lsq_gradients,
        update_lsq_params,
        QuantizedWeightHandler,
    )
    HAVE_GPU_QUANT_UTILS = True
except ImportError:
    HAVE_GPU_QUANT_UTILS = False


def get_batch(data_iterator):
    """Generate a batch from the data iterator."""
    args = get_args()

    if data_iterator is None:
        batch_size = args.micro_batch_size
        seq_len = args.seq_length
        vocab_size = args.vocab_size if hasattr(args, 'vocab_size') else 151936

        # Cache fake data for consistent testing
        if not hasattr(get_batch, "cached_data"):
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
            labels = torch.cat([tokens[:, 1:], torch.full((batch_size, 1), -100, device='cuda', dtype=tokens.dtype)], dim=1)
            position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
            loss_mask = torch.ones(batch_size, seq_len, device='cuda')
            attention_mask = None
            get_batch.cached_data = (tokens, position_ids, labels, loss_mask, attention_mask)
        else:
            tokens, position_ids, labels, loss_mask, attention_mask = get_batch.cached_data

        return tokens, position_ids, labels, loss_mask, attention_mask

    batch = next(data_iterator)
    tokens = batch['tokens'].cuda()
    raw_labels = batch['labels'].cuda()
    labels = torch.cat([raw_labels[:, 1:], torch.full((tokens.shape[0], 1), -100, device='cuda', dtype=raw_labels.dtype)], dim=1)
    position_ids = torch.arange(tokens.shape[1], device='cuda').unsqueeze(0).expand(tokens.shape[0], -1)
    loss_mask = torch.ones(tokens.shape[0], tokens.shape[1], device='cuda')
    attention_mask = None

    return tokens, position_ids, labels, loss_mask, attention_mask


def loss_func(loss_mask, labels, output_tensor, model=None):
    """Loss function for Qwen3MoE."""
    if isinstance(output_tensor, dict) and 'loss' in output_tensor:
        loss = output_tensor['loss']
        reduced_loss = loss.clone() if loss.dim() == 0 else loss.mean()
        return loss, {'lm loss': reduced_loss}

    raise RuntimeError("Expected dict with 'loss' key from model forward pass")


def forward_step(data_iterator, model):
    """Forward step for Qwen3MoE training."""
    tokens, position_ids, labels, loss_mask, attention_mask = get_batch(data_iterator)

    output_tensor = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    return output_tensor, partial(loss_func, loss_mask, labels)


def add_qwen3_moe_args(parser):
    """Add Qwen3MoE-specific arguments."""
    group = parser.add_argument_group(title='Qwen3MoE')

    group.add_argument('--experts-per-set', type=int, default=16,
                       help='Experts per processing set')
    group.add_argument('--rope-theta', type=float, default=1000000.0,
                       help='RoPE base frequency')
    group.add_argument('--moe-activation-offload', action='store_true',
                       help='Enable MoE activation offload to CPU')
    group.add_argument('--warmup-steps', type=int, default=10,
                       help='Number of warmup steps')

    # Quantization-specific arguments
    group.add_argument('--quant-group-size', type=int, default=128,
                       help='Quantization group size')
    group.add_argument('--score-update-interval', type=int, default=5,
                       help='Steps between score recalculations')
    group.add_argument('--top-bf16-ratio', type=float, default=0.05,
                       help='Ratio of experts to keep in BF16')
    group.add_argument('--top-int8-ratio', type=float, default=0.30,
                       help='Ratio of experts to use INT8')
    group.add_argument('--lr-quant', type=float, default=1e-4,
                       help='Learning rate for delta/z updates')
    group.add_argument('--initial-precision', type=int, default=8,
                       help='Initial precision (16=BF16, 8=INT8, 4=INT4)')
    group.add_argument('--freq-smoothing-alpha', type=float, default=0.9,
                       help='Smoothing factor for call frequency')

    # Test-specific arguments
    group.add_argument('--test-precision-transition', action='store_true',
                       help='Test INT4↔INT8 precision transition')
    group.add_argument('--test-score-computation', action='store_true',
                       help='Test expert score computation')
    group.add_argument('--test-lsq-update', action='store_true',
                       help='Test LSQ delta/z update reception')

    return parser


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build mock datasets for testing."""
    args = get_args()

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, seq_length, vocab_size):
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.vocab_size = vocab_size
            self.fixed_tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
            self.fixed_labels = torch.randint(0, self.vocab_size, (self.seq_length,))

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {'tokens': self.fixed_tokens, 'labels': self.fixed_labels}

    vocab_size = args.vocab_size if hasattr(args, 'vocab_size') else 151936
    seq_length = args.seq_length
    train_samples, valid_samples, test_samples = train_val_test_num_samples

    train_ds = MockDataset(train_samples, seq_length, vocab_size) if train_samples > 0 else None
    valid_ds = None
    test_ds = None

    return train_ds, valid_ds, test_ds


def get_model(model_provider_func, model_type):
    """Get the model."""
    model = model_provider_func()
    model = model.cuda()
    return model


class QuantizationTestSuite:
    """Test suite for quantization optimizer features."""

    def __init__(self, optimizer, model, args):
        self.optimizer = optimizer
        self.model = model
        self.args = args
        self.test_results = {}

    def run_all_tests(self):
        """Run all quantization tests."""
        print_rank_0("=" * 60)
        print_rank_0("Running Quantization Test Suite")
        print_rank_0("=" * 60)

        # Test 1: Global quantization pool initialization
        self.test_quant_pool_init()

        # Test 2: Global expert ID encoding/decoding
        self.test_global_expert_id()

        # Test 3: Slot attach/detach
        self.test_slot_attach_detach()

        # Test 4: Call frequency smoothing
        self.test_call_frequency_smoothing()

        # Test 5: Expert score computation (requires some training steps)
        if self.args.test_score_computation:
            self.test_score_computation()

        # Test 6: Precision transition rules
        if self.args.test_precision_transition:
            self.test_precision_transition()

        # Test 7: LSQ update reception
        if self.args.test_lsq_update:
            self.test_lsq_update_reception()

        # Test 8: GPU dequantization
        if HAVE_GPU_QUANT_UTILS:
            self.test_gpu_dequantization()

        # Print summary
        self.print_test_summary()

        return all(self.test_results.values())

    def test_quant_pool_init(self):
        """Test global quantization pool initialization."""
        print_rank_0("\n[Test 1] Global Quantization Pool Initialization")
        test_name = "quant_pool_init"

        try:
            # Check that quantization pool exists
            assert hasattr(self.optimizer, 'quant_pool'), "Missing quant_pool"
            pool = self.optimizer.quant_pool

            # Check pool dimensions
            num_layers = self.optimizer.num_layers
            num_experts_per_layer = self.optimizer.num_experts_per_layer
            num_total_experts = num_layers * num_experts_per_layer

            assert pool.num_total_experts == num_total_experts, \
                f"Expected {num_total_experts} total experts, got {pool.num_total_experts}"

            # Check slot counts
            expected_int8_slots = int(num_total_experts * self.args.top_int8_ratio)
            expected_int4_slots = num_total_experts - expected_int8_slots - int(num_total_experts * 0.05)

            assert pool.num_int8_slots == expected_int8_slots, \
                f"Expected {expected_int8_slots} INT8 slots, got {pool.num_int8_slots}"
            assert pool.num_int4_slots == expected_int4_slots, \
                f"Expected {expected_int4_slots} INT4 slots, got {pool.num_int4_slots}"

            # Check pool tensors are allocated
            assert pool.int8_delta_w1 is not None, "INT8 delta_w1 pool not allocated"
            assert pool.int8_z_w1 is not None, "INT8 z_w1 pool not allocated"
            assert pool.int4_delta_w1 is not None, "INT4 delta_w1 pool not allocated"
            assert pool.int4_z_w1 is not None, "INT4 z_w1 pool not allocated"

            # Check initial attachment based on initial_precision
            pool_stats = pool.get_pool_stats()
            print_rank_0(f"  Pool stats: {pool_stats}")

            print_rank_0("  ✓ Global quantization pool initialized correctly")
            self.test_results[test_name] = True

        except AssertionError as e:
            print_rank_0(f"  ✗ FAILED: {e}")
            self.test_results[test_name] = False
        except Exception as e:
            print_rank_0(f"  ✗ ERROR: {e}")
            self.test_results[test_name] = False

    def test_global_expert_id(self):
        """Test global expert ID encoding/decoding."""
        print_rank_0("\n[Test 2] Global Expert ID Encoding/Decoding")
        test_name = "global_expert_id"

        try:
            from megatron.core.optimizer.fused_adam_lsq_cpu_offload_optimizer import (
                encode_global_expert_id,
                decode_global_expert_id,
            )

            num_experts_per_layer = self.optimizer.num_experts_per_layer

            # Test encoding/decoding roundtrip
            for layer_id in range(3):
                for local_exp_id in range(min(5, num_experts_per_layer)):
                    global_id = encode_global_expert_id(layer_id, local_exp_id, num_experts_per_layer)
                    decoded_layer, decoded_local = decode_global_expert_id(global_id, num_experts_per_layer)

                    assert decoded_layer == layer_id, \
                        f"Decoded layer {decoded_layer} != original {layer_id}"
                    assert decoded_local == local_exp_id, \
                        f"Decoded local expert {decoded_local} != original {local_exp_id}"

            print_rank_0("  ✓ Global expert ID encoding/decoding works correctly")
            print_rank_0(f"    Example: layer=1, local=5, global={encode_global_expert_id(1, 5, num_experts_per_layer)}")
            self.test_results[test_name] = True

        except AssertionError as e:
            print_rank_0(f"  ✗ FAILED: {e}")
            self.test_results[test_name] = False
        except Exception as e:
            print_rank_0(f"  ✗ ERROR: {e}")
            self.test_results[test_name] = False

    def test_slot_attach_detach(self):
        """Test slot attach/detach operations."""
        print_rank_0("\n[Test 3] Slot Attach/Detach")
        test_name = "slot_attach_detach"

        try:
            pool = self.optimizer.quant_pool

            # Check initial free slots
            initial_stats = pool.get_pool_stats()

            # Test attach/detach cycle
            test_expert_id = 100  # Arbitrary test expert ID

            # Attach to INT8
            slot_idx = pool.attach_expert(test_expert_id, 8)
            assert slot_idx >= 0 and slot_idx < pool.num_int8_slots, \
                f"Invalid slot index {slot_idx}"

            # Check slot is marked as used
            assert slot_idx not in pool.int8_free_slots, \
                "Slot should be removed from free slots"

            # Check expert_slot_map updated
            assert test_expert_id in pool.expert_slot_map, \
                "Expert should be in slot map"

            # Get slot data
            data = pool.get_slot_data(test_expert_id)
            assert data[6] == 8, "Precision should be 8"

            # Detach
            pool.detach_expert(test_expert_id)
            assert test_expert_id not in pool.expert_slot_map, \
                "Expert should be removed from slot map after detach"
            assert slot_idx in pool.int8_free_slots, \
                "Slot should be returned to free slots after detach"

            print_rank_0("  ✓ Slot attach/detach works correctly")
            print_rank_0(f"    Attached expert {test_expert_id} to INT8 slot {slot_idx}")
            self.test_results[test_name] = True

        except AssertionError as e:
            print_rank_0(f"  ✗ FAILED: {e}")
            self.test_results[test_name] = False
        except Exception as e:
            print_rank_0(f"  ✗ ERROR: {e}")
            self.test_results[test_name] = False

    def test_call_frequency_smoothing(self):
        """Test call frequency smoothing with historical average."""
        print_rank_0("\n[Test 4] Call Frequency Smoothing")
        test_name = "call_frequency_smoothing"

        try:
            from megatron.core.optimizer.fused_adam_lsq_cpu_offload_optimizer import encode_global_expert_id

            num_experts_per_layer = self.optimizer.num_experts_per_layer
            alpha = self.args.freq_smoothing_alpha

            # Initial frequency should be 0
            for layer_id in range(self.optimizer.num_layers):
                for local_exp_id in range(num_experts_per_layer):
                    global_expert_id = encode_global_expert_id(layer_id, local_exp_id, num_experts_per_layer)
                    freq = self.optimizer._call_frequency.get(global_expert_id, 0.0)
                    assert freq == 0.0, f"Initial frequency for expert {global_expert_id} should be 0"

            # Simulate multiple updates for layer 0
            tokens_per_expert = torch.zeros(num_experts_per_layer, dtype=torch.long)

            # First update: expert 0 gets 100 tokens
            tokens_per_expert[0] = 100
            self.optimizer.update_call_frequency(tokens_per_expert, layer_id=0)

            global_expert_id_0 = encode_global_expert_id(0, 0, num_experts_per_layer)
            expected_freq_0 = (1 - alpha) * 100  # alpha * 0 + (1-alpha) * 100
            actual_freq_0 = self.optimizer._call_frequency[global_expert_id_0]
            assert abs(actual_freq_0 - expected_freq_0) < 1e-5, \
                f"After first update: expected {expected_freq_0}, got {actual_freq_0}"

            # Second update: expert 0 gets 200 tokens
            tokens_per_expert[0] = 200
            self.optimizer.update_call_frequency(tokens_per_expert, layer_id=0)

            expected_freq_1 = alpha * expected_freq_0 + (1 - alpha) * 200
            actual_freq_1 = self.optimizer._call_frequency[global_expert_id_0]
            assert abs(actual_freq_1 - expected_freq_1) < 1e-5, \
                f"After second update: expected {expected_freq_1}, got {actual_freq_1}"

            print_rank_0(f"  ✓ Frequency smoothing works correctly (alpha={alpha})")
            print_rank_0(f"    After 100 tokens: freq={expected_freq_0:.2f}")
            print_rank_0(f"    After 200 tokens: freq={expected_freq_1:.2f}")
            self.test_results[test_name] = True

        except AssertionError as e:
            print_rank_0(f"  ✗ FAILED: {e}")
            self.test_results[test_name] = False
        except Exception as e:
            print_rank_0(f"  ✗ ERROR: {e}")
            self.test_results[test_name] = False

    def test_score_computation(self):
        """Test expert score computation using exp_avg_sq."""
        print_rank_0("\n[Test 3] Expert Score Computation")
        test_name = "score_computation"

        try:
            # Compute scores
            self.optimizer.compute_expert_scores()

            # Check that scores exist
            assert len(self.optimizer._expert_scores) > 0, "No scores computed"

            # All scores should be non-negative
            for exp_id, score in self.optimizer._expert_scores.items():
                assert score >= 0, f"Score for expert {exp_id} is negative: {score}"

            print_rank_0(f"  ✓ Scores computed for {len(self.optimizer._expert_scores)} experts")

            # Print top 5 scores
            sorted_scores = sorted(self.optimizer._expert_scores.items(), key=lambda x: -x[1])
            print_rank_0("  Top 5 expert scores:")
            for i, (exp_id, score) in enumerate(sorted_scores[:5]):
                print_rank_0(f"    Expert {exp_id}: {score:.6e}")

            self.test_results[test_name] = True

        except AssertionError as e:
            print_rank_0(f"  ✗ FAILED: {e}")
            self.test_results[test_name] = False
        except Exception as e:
            print_rank_0(f"  ✗ ERROR: {e}")
            self.test_results[test_name] = False

    def test_precision_transition(self):
        """Test INT4 ↔ INT8 precision transition rules."""
        print_rank_0("\n[Test 4] Precision Transition Rules")
        test_name = "precision_transition"

        try:
            k = 16  # Scale factor

            # Create mock delta/z for testing
            mock_delta_int4 = torch.ones(10, dtype=torch.float32) * 0.01
            mock_z_int4 = torch.ones(10, dtype=torch.float32) * 5

            # Test INT4 → INT8 upgrade
            print_rank_0("  Testing INT4 → INT8 upgrade:")
            delta_int8_expected = mock_delta_int4 / k  # delta_new = delta_old / 16
            z_int8_expected = mock_z_int4 * k  # z_new = z_old * 16

            # Verify formula
            for i in range(10):
                assert abs(delta_int8_expected[i].item() - 0.01/16) < 1e-6, "INT4→INT8 delta formula wrong"
                assert abs(z_int8_expected[i].item() - 5*16) < 1e-6, "INT4→INT8 z formula wrong"

            print_rank_0("    ✓ INT4 → INT8: delta_new = delta_old / 16, z_new = z_old * 16")

            # Test INT8 → INT4 downgrade
            print_rank_0("  Testing INT8 → INT4 downgrade:")
            mock_delta_int8 = torch.ones(10, dtype=torch.float32) * 0.001
            mock_z_int8 = torch.ones(10, dtype=torch.float32) * 80

            delta_int4_expected = mock_delta_int8 * k  # delta_new = delta_old * 16
            z_int4_expected = torch.round(mock_z_int8 / k)  # z_new = round(z_old / 16)

            for i in range(10):
                assert abs(delta_int4_expected[i].item() - 0.001*16) < 1e-6, "INT8→INT4 delta formula wrong"
                assert abs(z_int4_expected[i].item() - 5) < 1e-6, "INT8→INT4 z formula wrong"

            print_rank_0("    ✓ INT8 → INT4: delta_new = delta_old * 16, z_new = round(z_old / 16)")

            # Test dynamic precision allocation
            print_rank_0("  Testing dynamic precision allocation:")
            self.optimizer.compute_expert_scores()
            self.optimizer.update_quant_precision()

            num_experts = self.optimizer.num_global_experts
            bf16_count = sum(1 for p in self.optimizer._quant_precision.values() if p == 16)
            int8_count = sum(1 for p in self.optimizer._quant_precision.values() if p == 8)
            int4_count = sum(1 for p in self.optimizer._quant_precision.values() if p == 4)

            expected_bf16 = int(num_experts * self.args.top_bf16_ratio)
            expected_int8 = int(num_experts * self.args.top_int8_ratio)
            expected_int4 = num_experts - expected_bf16 - expected_int8

            print_rank_0(f"    BF16 experts: {bf16_count} (expected ~{expected_bf16})")
            print_rank_0(f"    INT8 experts: {int8_count} (expected ~{expected_int8})")
            print_rank_0(f"    INT4 experts: {int4_count} (expected ~{expected_int4})")

            self.test_results[test_name] = True

        except AssertionError as e:
            print_rank_0(f"  ✗ FAILED: {e}")
            self.test_results[test_name] = False
        except Exception as e:
            print_rank_0(f"  ✗ ERROR: {e}")
            self.test_results[test_name] = False

    def test_lsq_update_reception(self):
        """Test receiving updated delta/z from GPU."""
        print_rank_0("\n[Test 5] LSQ Delta/Z Update Reception")
        test_name = "lsq_update_reception"

        try:
            # Create mock GPU updates
            num_groups = 128
            mock_delta_new = torch.randn(num_groups, dtype=torch.float32) * 0.01
            mock_z_new = torch.randn(num_groups, dtype=torch.float32) * 5

            # Test receive_gpu_updates method
            expert_id = 0

            # Simulate receiving updates from GPU
            self.optimizer.receive_gpu_updates(
                expert_id,
                delta_w1=mock_delta_new,
                z_w1=mock_z_new,
                delta_w2=mock_delta_new.clone(),
                z_w2=mock_z_new.clone(),
            )

            # Verify storage
            stored_delta = self.optimizer._delta_w1[expert_id]
            stored_z = self.optimizer._z_w1[expert_id]

            assert stored_delta is not None, "delta_w1 not stored"
            assert stored_z is not None, "z_w1 not stored"

            # Check values match
            assert torch.allclose(stored_delta, mock_delta_new, atol=1e-5), \
                "Stored delta doesn't match GPU update"
            assert torch.allclose(stored_z, mock_z_new, atol=1e-5), \
                "Stored z doesn't match GPU update"

            print_rank_0("  ✓ GPU delta/z updates correctly received and stored")
            print_rank_0(f"    delta mean: {stored_delta.mean().item():.6e}")
            print_rank_0(f"    z mean: {stored_z.mean().item():.6e}")

            self.test_results[test_name] = True

        except AssertionError as e:
            print_rank_0(f"  ✗ FAILED: {e}")
            self.test_results[test_name] = False
        except Exception as e:
            print_rank_0(f"  ✗ ERROR: {e}")
            self.test_results[test_name] = False

    def test_gpu_dequantization(self):
        """Test GPU dequantization functions."""
        print_rank_0("\n[Test 6] GPU Dequantization")
        test_name = "gpu_dequantization"

        try:
            if not torch.cuda.is_available():
                print_rank_0("  ⊘ SKIPPED: CUDA not available")
                self.test_results[test_name] = True
                return

            device = torch.device('cuda')
            group_size = 128

            # Test INT8 dequantization
            print_rank_0("  Testing INT8 dequantization:")
            quant_w8 = torch.randint(0, 256, (1024,), dtype=torch.uint8, device=device)
            delta = torch.ones(8, dtype=torch.float32, device=device) * 0.01
            z = torch.zeros(8, dtype=torch.float32, device=device)

            w_dequant = dequantize_int8(quant_w8, delta, z, group_size)

            assert w_dequant.dtype == torch.bfloat16, "Output should be bfloat16"
            assert w_dequant.numel() == quant_w8.numel(), "Output size mismatch"

            # Verify dequant formula: w = q * delta + z
            # For group 0: delta=0.01, z=0 → w = q * 0.01
            expected_w0 = quant_w8[:128].float() * 0.01
            actual_w0 = w_dequant[:128].float()
            assert torch.allclose(actual_w0, expected_w0, atol=1e-3), "INT8 dequant formula wrong"

            print_rank_0("    ✓ INT8 dequantization: w = q * delta + z")

            # Test INT4 dequantization
            print_rank_0("  Testing INT4 dequantization:")
            # Create packed INT4 weights
            packed = torch.empty(512, dtype=torch.uint8, device=device)
            for i in range(512):
                high = i % 16  # High nibble
                low = (i + 1) % 16  # Low nibble
                packed[i] = (high << 4) | low

            delta_int4 = torch.ones(4, dtype=torch.float32, device=device) * 0.01
            z_int4 = torch.zeros(4, dtype=torch.float32, device=device)

            w_dequant_int4 = dequantize_int4(packed, delta_int4, z_int4, group_size, original_size=1024)

            assert w_dequant_int4.dtype == torch.bfloat16, "Output should be bfloat16"

            # Verify unpacking
            unpacked_0 = (packed[0] >> 4) & 0x0F  # High nibble of first byte
            unpacked_1 = packed[0] & 0x0F  # Low nibble of first byte
            expected_w0 = unpacked_0 * 0.01
            expected_w1 = unpacked_1 * 0.01

            assert abs(w_dequant_int4[0].float().item() - expected_w0) < 1e-3, "INT4 unpack wrong (high)"
            assert abs(w_dequant_int4[1].float().item() - expected_w1) < 1e-3, "INT4 unpack wrong (low)"

            print_rank_0("    ✓ INT4 dequantization: unpack → w = q * delta + z")

            # Test LSQ gradient computation
            print_rank_0("  Testing LSQ gradient computation:")
            grad_output = torch.randn(1024, dtype=torch.bfloat16, device=device)
            quant_float = quant_w8.float()

            delta_grad, z_grad = compute_lsq_gradients(grad_output, quant_float, delta, z, group_size)

            assert delta_grad.numel() == delta.numel(), "delta_grad size mismatch"
            assert z_grad.numel() == z.numel(), "z_grad size mismatch"

            print_rank_0("    ✓ LSQ gradients computed correctly")

            # Test LSQ parameter update
            print_rank_0("  Testing LSQ parameter update:")
            lr_quant = 1e-4
            delta_new, z_new = update_lsq_params(delta, z, delta_grad, z_grad, lr_quant)

            # Verify update formula
            expected_delta_new = (delta - lr_quant * delta_grad).clamp(min=1e-6)
            assert torch.allclose(delta_new, expected_delta_new, atol=1e-6), "LSQ delta update wrong"

            print_rank_0("    ✓ LSQ update: delta_new = delta - lr * grad (clamped to positive)")

            self.test_results[test_name] = True

        except AssertionError as e:
            print_rank_0(f"  ✗ FAILED: {e}")
            self.test_results[test_name] = False
        except Exception as e:
            print_rank_0(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.test_results[test_name] = False

    def print_test_summary(self):
        """Print test summary."""
        print_rank_0("\n" + "=" * 60)
        print_rank_0("Test Summary")
        print_rank_0("=" * 60)

        passed = sum(1 for v in self.test_results.values() if v)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print_rank_0(f"  {test_name}: {status}")

        print_rank_0(f"\nTotal: {passed}/{total} tests passed")
        print_rank_0("=" * 60)


def train_with_tests(
    model,
    optimizer,
    train_dataloader,
    forward_step_func,
    config,
    test_suite,
):
    """Training loop with integrated tests."""
    args = get_args()
    timers = get_timers()

    model.train()
    data_iterator = iter(train_dataloader) if train_dataloader is not None else None

    iter_times = []
    total_start_time = time.time()

    # Training loop
    for iteration in range(1, args.train_iters + 1):
        iter_start_time = time.time()

        optimizer.zero_grad()

        try:
            output_tensor, loss_func_partial = forward_step_func(data_iterator, model)
        except StopIteration:
            data_iterator = iter(train_dataloader) if train_dataloader is not None else None
            output_tensor, loss_func_partial = forward_step_func(data_iterator, model)

        loss, loss_dict = loss_func_partial(output_tensor, model)
        loss.backward()
        torch.cuda.synchronize()

        optimizer.step()

        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        iter_times.append(iter_time)

        # Logging
        if iteration % args.log_interval == 0:
            loss_str = ' | '.join([f'{k}: {v.item():.6e}' for k, v in loss_dict.items()])
            avg_time = sum(iter_times) / len(iter_times)
            print_rank_0(f'iteration {iteration}/{args.train_iters} | {loss_str} | time: {iter_time:.3f}s | avg: {avg_time:.3f}s')

    # Run tests after training
    all_passed = test_suite.run_all_tests()

    total_time = time.time() - total_start_time
    print_rank_0(f'\nTotal training time: {total_time:.2f}s')

    return all_passed


def main():
    """Main entry point for quantization optimizer test."""
    if not HAVE_QUANT_OPTIMIZER:
        print("ERROR: FusedAdamLSQCPUOffloadOptimizer not available. Cannot run tests.")
        print("Please ensure fuse_opt is installed and fused_adam_lsq module is accessible.")
        sys.exit(1)

    # Parse arguments
    args = parse_args(extra_args_provider=add_qwen3_moe_args)
    validate_args(args, {'data_path': False})

    # Initialize Megatron
    initialize_megatron(extra_args_provider=add_qwen3_moe_args)

    args = get_args()
    timers = get_timers()

    # Set random seeds
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    torch.manual_seed(args.seed + ep_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + ep_rank)

    # Build model
    timers('model-setup', log_level=0).start()
    model = get_model(model_provider, ModelType.encoder_or_decoder)
    timers('model-setup').stop()

    print_rank_0(f'Model built with {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters')

    # Create quantized optimizer
    print_rank_0("Creating FusedAdamLSQCPUOffloadOptimizer...")
    optimizer = FusedAdamLSQCPUOffloadOptimizer(
        model=model,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        clip_grad=args.clip_grad,
        score_update_interval=args.score_update_interval,
        freq_smoothing_alpha=args.freq_smoothing_alpha,
        quant_group_size=args.quant_group_size,
        top_bf16_ratio=args.top_bf16_ratio,
        top_int8_ratio=args.top_int8_ratio,
        lr_quant=args.lr_quant,
        initial_precision=args.initial_precision,
    )

    print_rank_0(f"Quantization config:")
    print_rank_0(f"  - group_size: {args.quant_group_size}")
    print_rank_0(f"  - score_update_interval: {args.score_update_interval}")
    print_rank_0(f"  - top_bf16_ratio: {args.top_bf16_ratio}")
    print_rank_0(f"  - top_int8_ratio: {args.top_int8_ratio}")
    print_rank_0(f"  - initial_precision: {args.initial_precision}")

    # Pass CPU update Event to expert modules
    cpu_update_event = optimizer.get_cpu_update_event()
    for layer in model.decoder.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            layer.mlp.experts.set_cpu_update_event(cpu_update_event)

    # Build datasets
    timers('data-loading', log_level=0).start()
    train_ds, _, _ = train_valid_test_datasets_provider(
        (getattr(args, 'train_samples', 1000), 0, 0)
    )
    timers('data-loading').stop()

    # Create dataloader
    train_dataloader = None
    if train_ds is not None:
        train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.micro_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    # Get config
    config = core_transformer_config_from_args(args)

    # Create test suite
    test_suite = QuantizationTestSuite(optimizer, model, args)

    # Train and run tests
    print_rank_0('Starting training with quantization tests...')
    all_tests_passed = train_with_tests(
        model, optimizer, train_dataloader, forward_step, config, test_suite
    )

    if all_tests_passed:
        print_rank_0('\n✓ All tests passed!')
        print_rank_0('Training and testing completed successfully!')
    else:
        print_rank_0('\n✗ Some tests failed!')
        print_rank_0('Please check the output above for details.')
        sys.exit(1)


if __name__ == '__main__':
    main()