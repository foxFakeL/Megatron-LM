# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pretrain script for Qwen3MoE using standard Megatron infrastructure with DeepSpeedCPUOffloadOptimizer.

This script uses:
- Standard Megatron initialization and argument parsing
- Standard Megatron tokenizer infrastructure
- DeepSpeedCPUOffloadOptimizer for expert weight updates on CPU
- Custom training loop that calls optimizer.step()

Usage:
    # Single GPU
    python pretrain_qwen3_moe_standard.py --num-layers 2 --iters 10

    # Multi-GPU EP
    torchrun --nproc_per_node=2 pretrain_qwen3_moe_standard.py --num-layers 2 --iters 10
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
from megatron.core.models.qwen3_moe.qwen3_moe_model import model_provider
from megatron.core.transformer.moe.memory_logger import log_memory
from megatron.core.optimizer.deepspeed_cpu_offload_optimizer import DeepSpeedCPUOffloadOptimizer
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.core import parallel_state, tensor_parallel


def get_batch(data_iterator):
    """Generate a batch from the data iterator."""
    args = get_args()

    if data_iterator is None:
        batch_size = args.micro_batch_size
        seq_len = args.seq_length
        vocab_size = args.vocab_size if hasattr(args, 'vocab_size') else 151936

        # CRITICAL FIX: Cache fake data to ensure model sees the same batch every iteration
        # This allows the model to overfit on this fixed batch for testing
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
    # Causal LM: labels = next token, last position = -100 (ignore_index for Liger)
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
        # Liger returns scalar directly
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

    # Qwen3MoE specific parameters (not in standard Megatron args)
    group.add_argument('--experts-per-set', type=int, default=16,
                       help='Experts per processing set')
    group.add_argument('--rope-theta', type=float, default=1000000.0,
                       help='RoPE base frequency')

    # MoE optimization
    group.add_argument('--moe-activation-offload', action='store_true',
                       help='Enable MoE activation offload to CPU')
    group.add_argument('--warmup-steps', type=int, default=100,
                       help='Number of warmup steps for learning rate schedule')
    # Note: --clip-grad is already defined in Megatron's standard arguments

    # Timing warmup iterations (skip first N iterations for avg timing calculation)
    group.add_argument('--timing-warmup-iters', type=int, default=3,
                       help='Number of warmup iterations to skip for timing statistics')

    return parser


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets.

    Creates mock datasets for testing without real data.
    In production, replace with real dataset loader.
    """
    args = get_args()

    class MockDataset(torch.utils.data.Dataset):
        """Mock dataset for testing."""
        def __init__(self, num_samples, seq_length, vocab_size):
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.vocab_size = vocab_size

            # CRITICAL FIX: Generate fixed data ONCE at initialization
            # This allows the model to overfit on this fixed batch for testing
            self.fixed_tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
            self.fixed_labels = torch.randint(0, self.vocab_size, (self.seq_length,))
            self.fixed_loss_mask = torch.ones(self.seq_length)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Always return the same fixed data
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

    Note: wrap_with_ddp is disabled by default because Expert weights are on CPU
    (shared memory), which is incompatible with DDP. The DeepSpeedCPUOffloadOptimizer
    handles gradient all-reduce for GPU parameters internally.
    """
    args = get_args()

    # Build model
    model = model_provider_func()

    # Move to GPU (only GPU params, expert params stay on CPU)
    model = model.cuda()

    # Do NOT wrap with DDP - expert weights are on CPU, incompatible with DDP
    # DeepSpeedCPUOffloadOptimizer handles gradient all-reduce for GPU params in step()

    return model


def train(
    model,
    optimizer,
    train_dataloader,
    forward_step_func,
    config,
):
    """Custom training loop with DeepSpeedCPUOffloadOptimizer support."""
    args = get_args()
    timers = get_timers()

    model.train()

    # Create data iterator ONCE at the start (outside the loop!)
    # It will be recreated when exhausted (end of epoch)
    data_iterator = iter(train_dataloader) if train_dataloader is not None else None
    epoch_count = 0

    # Timing statistics
    iter_times = []
    total_start_time = time.time()

    # Training loop
    for iteration in range(1, args.train_iters + 1):
        iter_start_time = time.time()

        # Zero gradients
        # NOTE: optimizer.zero_grad() internally calls sync_gradient_offload()
        # This ensures previous iteration's D2H gradient transfers are complete
        # before zeroing CPU gradient buffers (_grad_weight1/_grad_weight2)
        optimizer.zero_grad()

        # Forward pass - data_iterator is reused across iterations
        # get_batch will call next(data_iterator) internally
        timers('forward-backward', log_level=1).start()
        try:
            output_tensor, loss_func_partial = forward_step_func(data_iterator, model)
        except StopIteration:
            # End of epoch - recreate iterator and try again
            epoch_count += 1
            print_rank_0(f'[Epoch {epoch_count}] Recreating data iterator')
            data_iterator = iter(train_dataloader) if train_dataloader is not None else None
            output_tensor, loss_func_partial = forward_step_func(data_iterator, model)

        loss, loss_dict = loss_func_partial(output_tensor, model)
        timers('forward-backward').stop()

        # Backward pass
        timers('backward', log_level=1).start()
        loss.backward()
        # 🚨🚨🚨 必须加回来！这是你的定海神针 🚨🚨🚨
        # 它强制 Python (CPU) 等待主计算流把所有反向传播算完，
        # 从而确保卸载流（_grad_offload_stream）搬运的梯度是真实有效的！
        torch.cuda.synchronize()
        # NOTE: No sync needed here - optimizer.zero_grad() and optimizer.step()
        # handle _grad_offload_stream synchronization internally
        timers('backward').stop()

        # Update parameters (DeepSpeedCPUOffloadOptimizer handles all synchronization)
        # - step() calls sync_gradient_offload(): waits for D2H gradient transfers
        # - cpu_optimizer.step(): updates CPU weights
        # - record_cpu_update_done(): creates Event for next iteration's prefetch
        timers('optimizer-step', log_level=1).start()
        optimizer.step()
        # NOTE: No sync needed here - prefetch uses Event-based sync
        # via _cpu_update_event.wait() in _prefetch_expert_weights_async
        timers('optimizer-step').stop()

        # Record iteration time
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        iter_times.append(iter_time)

        # Logging with timing info
        if iteration % args.log_interval == 0:
            loss_str = ' | '.join([f'{k}: {v.item():.6e}' for k, v in loss_dict.items()])

            # Calculate timing statistics (skip warmup iterations)
            timing_warmup = getattr(args, 'timing_warmup_iters', 3)
            warmed_times = iter_times[timing_warmup:] if len(iter_times) > timing_warmup else iter_times

            if len(warmed_times) > 0:
                avg_time = sum(warmed_times) / len(warmed_times)
            else:
                avg_time = iter_time  # Use current time if still in warmup

            recent_avg_time = sum(iter_times[-min(10, len(iter_times)):]) / min(10, len(iter_times))

            # Calculate throughput (tokens per second)
            tokens_per_iter = args.micro_batch_size * args.seq_length
            throughput = tokens_per_iter / iter_time if iter_time > 0 else 0

            # Estimate remaining time (use warmed average for better estimate)
            remaining_iters = args.train_iters - iteration
            est_remaining_time = remaining_iters * avg_time

            warmup_status = f'(warmup: {iteration}/{timing_warmup})' if iteration <= timing_warmup else ''
            print_rank_0(
                f'iteration {iteration}/{args.train_iters} | {loss_str} | '
                f'time: {iter_time:.3f}s | avg: {avg_time:.3f}s {warmup_status} | recent_avg: {recent_avg_time:.3f}s | '
                f'throughput: {throughput:.1f} tokens/s | ETA: {est_remaining_time/60:.1f}min'
            )

        # Checkpointing
        if args.save and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, None)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Final timing summary (exclude warmup iterations)
    timing_warmup = getattr(args, 'timing_warmup_iters', 3)
    warmed_times = iter_times[timing_warmup:] if len(iter_times) > timing_warmup else iter_times

    print_rank_0('=' * 60)
    print_rank_0('Training Timing Summary:')
    print_rank_0(f'  Total iterations: {args.train_iters}')
    print_rank_0(f'  Warmup iterations (excluded from avg): {timing_warmup}')
    print_rank_0(f'  Total time: {total_time:.2f}s ({total_time/60:.2f}min)')
    if len(warmed_times) > 0:
        print_rank_0(f'  Average iteration time (after warmup): {sum(warmed_times)/len(warmed_times):.3f}s')
        print_rank_0(f'  Min iteration time (after warmup): {min(warmed_times):.3f}s')
        print_rank_0(f'  Max iteration time (after warmup): {max(warmed_times):.3f}s')
        print_rank_0(f'  Throughput (after warmup): {len(warmed_times) * args.micro_batch_size * args.seq_length / sum(warmed_times):.1f} tokens/s')
    print_rank_0('=' * 60)

    return loss_dict


def main():
    """Main entry point for Qwen3MoE pretrain."""
    # Parse arguments
    args = parse_args(extra_args_provider=add_qwen3_moe_args)
    validate_args(args, {'data_path': False})

    # Initialize Megatron
    initialize_megatron(extra_args_provider=add_qwen3_moe_args)

    args = get_args()
    timers = get_timers()

    # Set random seeds - different rank uses different seed to generate different data
    # Without this, all ranks would generate identical mock data
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    torch.manual_seed(args.seed + ep_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + ep_rank)

    # Build model (DDP disabled - expert weights on CPU, DeepSpeedCPUOffloadOptimizer handles gradient sync)
    timers('model-setup', log_level=0).start()
    model = get_model(model_provider, ModelType.encoder_or_decoder)
    timers('model-setup').stop()

    print_rank_0(f'Model built with {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters')

    # Create DeepSpeedCPUOffloadOptimizer
    # - GPU params (attention, embedding, router): standard AdamW on GPU
    # - Expert params: DeepSpeed CPUAdam on CPU
    # - Expert weights/gradients already on CPU (shared memory)
    optimizer = DeepSpeedCPUOffloadOptimizer(
        model=model,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        clip_grad=args.clip_grad,
    )

    # Setup: Pass CPU update Event to expert modules for synchronization
    # After optimizer.step(), this Event signals that CPU weights are updated
    # Prefetch in forward() will wait on this Event before loading weights
    cpu_update_event = optimizer.get_cpu_update_event()
    for layer in model.decoder.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            layer.mlp.experts.set_cpu_update_event(cpu_update_event)

    # Load checkpoint if specified
    if args.load is not None:
        timers('checkpoint-load', log_level=0).start()
        load_checkpoint(model, optimizer, None)
        timers('checkpoint-load').stop()

    # Build datasets
    timers('data-loading', log_level=0).start()
    # Use train_samples for mock data, eval samples default to 0 if not set
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

    # Train
    print_rank_0('Starting training...')
    train(model, optimizer, train_dataloader, forward_step, config)
    print_rank_0('Training completed!')


if __name__ == '__main__':
    main()