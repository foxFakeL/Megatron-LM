"""Pretrain script for Qwen3MoE with FusedAdamLSQCPUOffloadOptimizer.

This script uses:
- QuantizedDispatcherCacheGroupedMLP for MoE layers with dynamic quantization
- FusedAdamLSQCPUOffloadOptimizer for expert weight updates with INT8/INT4 quantization
- GPU-side dequantization and LSQ delta/z updates

Usage:
    # Single GPU test
    python pretrain_qwen3_moe_quantized.py --num-layers 2 --train-iters 100

    # Multi-GPU EP test
    torchrun --nproc_per_node=2 pretrain_qwen3_moe_quantized.py --num-layers 2 --train-iters 100
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial
from typing import Optional

import torch

from megatron.core.enums import ModelType
from megatron.core.transformer.moe.memory_logger import log_memory
from megatron.core import parallel_state, tensor_parallel
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint, save_checkpoint

# Import quantized model
from megatron.core.models.qwen3_moe.qwen3_moe_quantized_model import model_provider as quantized_model_provider

# Import the quantized optimizer
try:
    from megatron.core.optimizer.fused_adam_lsq_cpu_offload_optimizer import FusedAdamLSQCPUOffloadOptimizer
    HAVE_QUANT_OPTIMIZER = True
except ImportError as e:
    HAVE_QUANT_OPTIMIZER = False
    print(f"Warning: FusedAdamLSQCPUOffloadOptimizer not available: {e}")
    # Fallback to standard optimizer
    from megatron.core.optimizer.deepspeed_cpu_offload_optimizer import DeepSpeedCPUOffloadOptimizer
    FusedAdamLSQCPUOffloadOptimizer = DeepSpeedCPUOffloadOptimizer


def get_batch(data_iterator):
    """Generate a batch from the data iterator."""
    args = get_args()

    if data_iterator is None:
        batch_size = args.micro_batch_size
        seq_len = args.seq_length
        vocab_size = args.vocab_size if hasattr(args, 'vocab_size') else 151936

        # CRITICAL: Cache fake data to ensure model sees the same batch every iteration
        # This allows the model to overfit on this fixed batch for testing loss decrease
        if not hasattr(get_batch, "cached_data"):
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
            labels = torch.cat([tokens[:, 1:], torch.full((batch_size, 1), -100, device='cuda', dtype=tokens.dtype)], dim=1)
            position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
            loss_mask = torch.ones(batch_size, seq_len, device='cuda')
            attention_mask = None

            # Store in function attribute
            get_batch.cached_data = (tokens, position_ids, labels, loss_mask, attention_mask)
        else:
            # Return cached data
            tokens, position_ids, labels, loss_mask, attention_mask = get_batch.cached_data

        return tokens, position_ids, labels, loss_mask, attention_mask

    # Real data
    batch = next(data_iterator)
    tokens = batch['tokens'].cuda()
    raw_labels = batch['labels'].cuda()
    labels = torch.cat([raw_labels[:, 1:], torch.full((tokens.shape[0], 1), -100, device='cuda', dtype=raw_labels.dtype)], dim=1)
    position_ids = batch.get('position_ids', None)
    attention_mask = batch.get('attention_mask', None)
    loss_mask = batch.get('loss_mask', None)

    if loss_mask is not None:
        loss_mask = loss_mask.cuda()

    if position_ids is None:
        position_ids = torch.arange(tokens.shape[1], device='cuda').unsqueeze(0).expand(tokens.shape[0], -1)
    else:
        position_ids = position_ids.cuda()

    return tokens, position_ids, labels, loss_mask, attention_mask


def loss_func(loss_mask, labels, output_tensor, model=None):
    """Loss function for Qwen3MoE.

    Args:
        loss_mask: Unused (Liger handles ignore_index internally)
        labels: Labels already shifted in get_batch (last position = -100)
        output_tensor: Model output - dict with 'loss' key (scalar from Liger)

    Returns:
        loss: Scalar loss for backward
        loss_dict: Dict with 'lm loss' for logging
    """
    if isinstance(output_tensor, dict) and 'loss' in output_tensor:
        loss = output_tensor['loss']
        reduced_loss = loss.clone() if loss.dim() == 0 else loss.mean()
        return loss, {'lm loss': reduced_loss}

    raise RuntimeError("Expected dict with 'loss' key from model forward pass")


def forward_step(data_iterator, model):
    """Forward step for Qwen3MoE training.

    Args:
        data_iterator: Data iterator
        model: Qwen3MoE model

    Returns:
        output_tensor: Model output
        loss_func: Loss function
    """
    args = get_args()
    timers = get_timers()

    # Get batch
    tokens, position_ids, labels, loss_mask, attention_mask = get_batch(data_iterator)
    log_memory("forward_step: after get_batch")

    # Forward pass
    output_tensor = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    log_memory("forward_step: after model forward")

    return output_tensor, partial(loss_func, loss_mask, labels)


def add_qwen3_moe_args(parser):
    """Add Qwen3MoE-specific arguments."""
    group = parser.add_argument_group(title='Qwen3MoE')

    # Qwen3MoE specific parameters
    group.add_argument('--experts-per-set', type=int, default=16,
                       help='Experts per processing set')
    group.add_argument('--rope-theta', type=float, default=1000000.0,
                       help='RoPE base frequency')
    # --qk-layernorm is a standard Megatron arg defined in training/arguments.py

    # MoE optimization
    group.add_argument('--moe-activation-offload', action='store_true',
                       help='Enable MoE activation offload to CPU')
    group.add_argument('--warmup-steps', type=int, default=10,
                       help='Number of warmup steps for learning rate schedule')

    # Quantization-specific arguments
    group.add_argument('--quant-group-size', type=int, default=128,
                       help='Quantization group size')
    group.add_argument('--score-update-interval', type=int, default=50,
                       help='Steps between score recalculations')
    group.add_argument('--top-bf16-ratio', type=float, default=0.05,
                       help='Ratio of experts to keep in BF16')
    group.add_argument('--top-int8-ratio', type=float, default=0.30,
                       help='Ratio of experts to use INT8')
    group.add_argument('--lr-quant', type=float, default=1e-4,
                       help='Learning rate for delta/z updates')
    group.add_argument('--initial-precision', type=int, default=16,
                       help='Initial precision (16=BF16, 8=INT8, 4=INT4). Start with BF16 for stability.')
    group.add_argument('--freq-smoothing-alpha', type=float, default=0.9,
                       help='Smoothing factor for call frequency')
    group.add_argument('--timing-warmup-iters', type=int, default=5,
                       help='Number of warmup iterations to skip for timing statistics')

    return parser


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets.

    Creates mock datasets for testing without real data.
    """
    args = get_args()

    class MockDataset(torch.utils.data.Dataset):
        """Mock dataset for testing."""
        def __init__(self, num_samples, seq_length, vocab_size):
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.vocab_size = vocab_size

            # Generate fixed data ONCE at initialization
            self.fixed_tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
            self.fixed_labels = torch.randint(0, self.vocab_size, (self.seq_length,))
            self.fixed_loss_mask = torch.ones(self.seq_length)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                'tokens': self.fixed_tokens,
                'labels': self.fixed_labels,
                'loss_mask': self.fixed_loss_mask,
            }

    vocab_size = args.vocab_size if hasattr(args, 'vocab_size') else 151936
    seq_length = args.seq_length

    train_samples, valid_samples, test_samples = train_val_test_num_samples

    train_ds = MockDataset(train_samples, seq_length, vocab_size) if (train_samples or 0) > 0 else None
    valid_ds = MockDataset(valid_samples, seq_length, vocab_size) if (valid_samples or 0) > 0 else None
    test_ds = MockDataset(test_samples, seq_length, vocab_size) if (test_samples or 0) > 0 else None

    return train_ds, valid_ds, test_ds


def get_model(model_provider_func, model_type, wrap_with_ddp=False):
    """Get the model.

    Note: wrap_with_ddp is disabled because Expert weights are on CPU.
    """
    args = get_args()

    # Build model using quantized model provider with quantization args
    model = quantized_model_provider(
        pre_process=True,
        post_process=True,
        quant_group_size=args.quant_group_size,
        lr_quant=args.lr_quant,
    )

    # Move to GPU (only GPU params, expert params stay on CPU)
    model = model.cuda()

    return model


def train(
    model,
    optimizer,
    train_dataloader,
    forward_step_func,
    config,
):
    """Custom training loop with FusedAdamLSQCPUOffloadOptimizer support."""
    args = get_args()
    timers = get_timers()

    rank = parallel_state.get_expert_model_parallel_rank()

    model.train()

    # Create data iterator
    data_iterator = iter(train_dataloader) if train_dataloader is not None else None
    epoch_count = 0

    # Timing and loss tracking
    iter_times = []
    loss_history = []
    total_start_time = time.time()

    # Track quantization precision changes
    precision_history = {}

    print_rank_0("=" * 70)
    print_rank_0("Starting Training with Dynamic Quantization")
    print_rank_0("=" * 70)
    print_rank_0(f"Quantization config:")
    print_rank_0(f"  - group_size: {args.quant_group_size}")
    print_rank_0(f"  - score_update_interval: {args.score_update_interval} steps")
    print_rank_0(f"  - top_bf16_ratio: {args.top_bf16_ratio}")
    print_rank_0(f"  - top_int8_ratio: {args.top_int8_ratio}")
    print_rank_0(f"  - initial_precision: {args.initial_precision}")
    print_rank_0(f"  - lr_quant: {args.lr_quant}")
    print_rank_0("=" * 70)

    # Training loop
    for iteration in range(1, args.train_iters + 1):
        iter_start_time = time.time()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        timers('forward-backward', log_level=1).start()
        try:
            output_tensor, loss_func_partial = forward_step_func(data_iterator, model)
        except StopIteration:
            epoch_count += 1
            print_rank_0(f'[Epoch {epoch_count}] Recreating data iterator')
            data_iterator = iter(train_dataloader) if train_dataloader is not None else None
            output_tensor, loss_func_partial = forward_step_func(data_iterator, model)

        loss, loss_dict = loss_func_partial(output_tensor, model)
        timers('forward-backward').stop()

        # Backward pass
        timers('backward', log_level=1).start()
        loss.backward()
        torch.cuda.synchronize()
        timers('backward').stop()

        # Update parameters
        timers('optimizer-step', log_level=1).start()
        optimizer.step()
        timers('optimizer-step').stop()

        # Record iteration time and loss
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        iter_times.append(iter_time)

        # Track loss
        current_loss = loss_dict['lm loss'].item()
        loss_history.append(current_loss)

        # Track precision changes (if score update interval reached)
        if hasattr(optimizer, '_quant_precision') and iteration % args.score_update_interval == 0:
            precision_counts = {}
            for exp_id, prec in optimizer._quant_precision.items():
                precision_counts[prec] = precision_counts.get(prec, 0) + 1
            precision_history[iteration] = precision_counts

        # Logging
        if iteration % args.log_interval == 0:
            loss_str = ' | '.join([f'{k}: {v.item():.6e}' for k, v in loss_dict.items()])

            # Calculate timing statistics
            timing_warmup = getattr(args, 'timing_warmup_iters', 5)
            warmed_times = iter_times[timing_warmup:] if len(iter_times) > timing_warmup else iter_times
            avg_time = sum(warmed_times) / len(warmed_times) if warmed_times else iter_time
            recent_avg_time = sum(iter_times[-min(10, len(iter_times)):]) / min(10, len(iter_times))

            # Calculate throughput
            tokens_per_iter = args.micro_batch_size * args.seq_length
            throughput = tokens_per_iter / iter_time if iter_time > 0 else 0

            # Estimate remaining time
            remaining_iters = args.train_iters - iteration
            est_remaining_time = remaining_iters * avg_time

            # Loss trend analysis
            if len(loss_history) >= 10:
                recent_losses = loss_history[-10:]
                loss_trend = "↓" if recent_losses[-1] < recent_losses[0] else "↑" if recent_losses[-1] > recent_losses[0] else "→"
                loss_change = (recent_losses[-1] - recent_losses[0]) / recent_losses[0] * 100
            else:
                loss_trend = "?"
                loss_change = 0

            warmup_status = f'(warmup: {iteration}/{timing_warmup})' if iteration <= timing_warmup else ''

            # Precision distribution (if available)
            prec_str = ""
            if hasattr(optimizer, '_quant_precision'):
                counts = {}
                for prec in optimizer._quant_precision.values():
                    counts[prec] = counts.get(prec, 0) + 1
                prec_str = f"| BF16:{counts.get(16,0)} INT8:{counts.get(8,0)} INT4:{counts.get(4,0)}"

            print_rank_0(
                f'iter {iteration}/{args.train_iters} | {loss_str} | trend: {loss_trend} ({loss_change:+.1f}%) | '
                f'time: {iter_time:.3f}s | avg: {avg_time:.3f}s {warmup_status} | '
                f'thru: {throughput:.0f} tok/s | ETA: {est_remaining_time/60:.1f}min {prec_str}'
            )

        # Checkpointing
        if args.save and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, None)

    # Final summary
    total_time = time.time() - total_start_time
    timing_warmup = getattr(args, 'timing_warmup_iters', 5)
    warmed_times = iter_times[timing_warmup:] if len(iter_times) > timing_warmup else iter_times

    print_rank_0("=" * 70)
    print_rank_0("Training Summary")
    print_rank_0("=" * 70)
    print_rank_0(f'Total iterations: {args.train_iters}')
    print_rank_0(f'Total time: {total_time:.2f}s ({total_time/60:.2f}min)')

    if len(warmed_times) > 0:
        print_rank_0(f'Avg iteration time (after warmup): {sum(warmed_times)/len(warmed_times):.3f}s')
        print_rank_0(f'Throughput: {len(warmed_times) * args.micro_batch_size * args.seq_length / sum(warmed_times):.1f} tokens/s')

    # Loss analysis
    print_rank_0("\nLoss Analysis:")
    print_rank_0(f"  Initial loss: {loss_history[0]:.6e}")
    print_rank_0(f"  Final loss: {loss_history[-1]:.6e}")
    print_rank_0(f"  Loss change: {(loss_history[-1] - loss_history[0]) / loss_history[0] * 100:+.2f}%")

    # Check if loss decreased
    if loss_history[-1] < loss_history[0]:
        print_rank_0("  ✓ Loss DECREASED - Training is working correctly!")
    else:
        print_rank_0("  ✗ Loss did NOT decrease - Check training configuration")

    # Show loss curve (every 10 iterations)
    print_rank_0("\nLoss Curve (sampled):")
    sample_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in sample_points:
        if i < len(loss_history):
            idx = min(i, len(loss_history) - 1)
            print_rank_0(f"  iter {idx+1}: loss = {loss_history[idx]:.6e}")

    # Precision distribution history
    if precision_history:
        print_rank_0("\nPrecision Distribution History:")
        for iter_num, counts in precision_history.items():
            print_rank_0(f"  iter {iter_num}: BF16={counts.get(16,0)}, INT8={counts.get(8,0)}, INT4={counts.get(4,0)}")

    print_rank_0("=" * 70)

    return loss_dict


def main():
    """Main entry point for Qwen3MoE pretrain with quantization."""

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

    # Build model using quantized model provider
    timers('model-setup', log_level=0).start()
    model = get_model(None, ModelType.encoder_or_decoder)  # model_provider_func unused
    timers('model-setup').stop()

    print_rank_0(f'Model built with {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters')
    print_rank_0(f'Using QuantizedDispatcherCacheGroupedMLP with quant_group_size={args.quant_group_size}, lr_quant={args.lr_quant}')

    # Create optimizer
    if HAVE_QUANT_OPTIMIZER:
        print_rank_0("Using FusedAdamLSQCPUOffloadOptimizer with dynamic quantization")
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
    else:
        print_rank_0("Warning: FusedAdamLSQ not available, using DeepSpeedCPUOffloadOptimizer fallback")
        optimizer = FusedAdamLSQCPUOffloadOptimizer(
            model=model,
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            clip_grad=args.clip_grad,
        )

    rank = parallel_state.get_expert_model_parallel_rank()

    # Setup: Pass CPU update Event and quant_optimizer to expert modules
    cpu_update_event = optimizer.get_cpu_update_event()

    for layer in model.decoder.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            # Set CPU update event for synchronization
            layer.mlp.experts.set_cpu_update_event(cpu_update_event)
            # Set quant_optimizer reference for prefetch_expert_data
            layer.mlp.experts.set_quant_optimizer(optimizer)

    # Load checkpoint if specified
    if args.load is not None:
        timers('checkpoint-load', log_level=0).start()
        load_checkpoint(model, optimizer, None)
        timers('checkpoint-load').stop()

    # Build datasets
    timers('data-loading', log_level=0).start()
    eval_samples = getattr(args, 'eval_samples', 0)
    train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(
        (getattr(args, 'train_samples', 5000000), eval_samples, 0)
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

    rank = parallel_state.get_expert_model_parallel_rank()

    # Train
    train(model, optimizer, train_dataloader, forward_step, config)


if __name__ == '__main__':
    main()