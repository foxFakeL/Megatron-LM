"""Fine-tuning script for Qwen3MoE with FusedAdamLSQCPUOffloadOptimizer.

Loads pre-trained Qwen3-MoE-A2.7B weights from Megatron Core distcp checkpoint
and fine-tunes on the OpenHermes-2.5 dataset.

Usage:
    bash run_shell/run_finetune_quant.sh
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial
from typing import Optional

import torch

from megatron.core.enums import ModelType
from megatron.core.transformer.moe.memory_logger import log_memory
from megatron.core import parallel_state
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import save_checkpoint

from megatron.core.models.qwen3_moe.qwen3_moe_quantized_model import model_provider as quantized_model_provider
from megatron.core.models.qwen3_moe.checkpoint_loader import load_finetune_checkpoint

try:
    from fused_adam_lsq import FusedAdamLSQ
    HAVE_FUSE_OPT = True
except ImportError:
    HAVE_FUSE_OPT = False

try:
    from megatron.core.optimizer.fused_adam_lsq_cpu_offload_optimizer import FusedAdamLSQCPUOffloadOptimizer
    HAVE_QUANT_OPTIMIZER = HAVE_FUSE_OPT  # Need fuse_opt package at runtime too
    if not HAVE_FUSE_OPT:
        print("Warning: fuse_opt package not found, falling back to DeepSpeedCPUOffloadOptimizer")
        from megatron.core.optimizer.deepspeed_cpu_offload_optimizer import DeepSpeedCPUOffloadOptimizer
        FusedAdamLSQCPUOffloadOptimizer = DeepSpeedCPUOffloadOptimizer
except ImportError as e:
    HAVE_QUANT_OPTIMIZER = False
    print(f"Warning: FusedAdamLSQCPUOffloadOptimizer not available: {e}")
    from megatron.core.optimizer.deepspeed_cpu_offload_optimizer import DeepSpeedCPUOffloadOptimizer
    FusedAdamLSQCPUOffloadOptimizer = DeepSpeedCPUOffloadOptimizer


def get_batch(data_iterator):
    """Get a batch from real dataset or return None."""
    args = get_args()

    if data_iterator is None:
        return None, None, None, None, None

    batch = next(data_iterator)
    tokens = batch['tokens'].cuda().to(torch.long)
    labels_raw = batch['labels'].cuda().to(torch.long)
    # Shift labels: predict next token, last position = -100 (ignored)
    labels = torch.cat([
        labels_raw[:, 1:],
        torch.full((tokens.shape[0], 1), -100, device='cuda', dtype=torch.long),
    ], dim=1)
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
    if isinstance(output_tensor, dict) and 'loss' in output_tensor:
        loss = output_tensor['loss']
        reduced_loss = loss.clone() if loss.dim() == 0 else loss.mean()
        return loss, {'lm loss': reduced_loss}
    raise RuntimeError("Expected dict with 'loss' key from model forward pass")


def forward_step(data_iterator, model):
    args = get_args()
    timers = get_timers()

    tokens, position_ids, labels, loss_mask, attention_mask = get_batch(data_iterator)
    log_memory("forward_step: after get_batch")

    output_tensor = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    log_memory("forward_step: after model forward")

    return output_tensor, partial(loss_func, loss_mask, labels)


def add_qwen3_moe_args(parser):
    """Add Qwen3MoE fine-tuning arguments."""
    group = parser.add_argument_group(title='Qwen3MoE Fine-tuning')

    # Model architecture
    group.add_argument('--experts-per-set', type=int, default=16,
                       help='Experts per processing set')
    group.add_argument('--rope-theta', type=float, default=10000000.0,
                       help='RoPE base frequency (Qwen3 uses 10M)')
    # --qk-layernorm is a standard Megatron arg defined in training/arguments.py

    # MoE optimization
    group.add_argument('--moe-activation-offload', action='store_true',
                       help='Enable MoE activation offload to CPU')
    group.add_argument('--warmup-steps', type=int, default=100,
                       help='Number of warmup steps for learning rate schedule')

    # Quantization
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
                       help='Initial precision (16=BF16, 8=INT8, 4=INT4)')
    group.add_argument('--freq-smoothing-alpha', type=float, default=0.9,
                       help='Smoothing factor for call frequency')
    group.add_argument('--timing-warmup-iters', type=int, default=5,
                       help='Warmup iterations to skip for timing statistics')

    # Fine-tuning specific
    group.add_argument('--ckpt-path', type=str, default=None,
                       help='Path to Megatron Core distcp checkpoint directory')
    # --data-path and --train-samples are standard Megatron args

    return parser


class ArrowDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping a HuggingFace Arrow tokenized dataset.

    The Arrow file contains tokenized examples with 'input_ids' and 'attention_mask'.
    Each example is sliced into seq_length chunks.
    """

    def __init__(self, data_path: str, seq_length: int, max_samples: Optional[int] = None):
        self.seq_length = seq_length
        self.max_samples = max_samples

        # Load Arrow dataset
        if os.path.isdir(data_path):
            arrow_files = sorted([
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith('.arrow')
            ])
        else:
            arrow_files = [data_path]

        from datasets import Dataset
        all_data = []
        for arrow_file in arrow_files:
            ds = Dataset.from_file(arrow_file)
            all_data.extend(ds['input_ids'])

        self.data = all_data
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]

        print_rank_0(f"ArrowDataset: {len(self.data)} samples loaded from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Truncate or pad to seq_length
        if len(tokens) >= self.seq_length:
            tokens = tokens[:self.seq_length]
        else:
            tokens = tokens + [0] * (self.seq_length - len(tokens))

        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        return {
            'tokens': tokens_tensor,
            'labels': tokens_tensor,  # standard causal LM: labels = tokens
            'loss_mask': torch.ones(self.seq_length),
        }


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train dataset from Arrow files."""
    args = get_args()
    train_samples, _, _ = train_val_test_num_samples

    data_path = getattr(args, 'data_path', None)
    if data_path is None:
        print_rank_0("Warning: --data-path not set, using mock data")
        return _mock_datasets(args)
    # Standard Megatron --data-path is a list (nargs='*'), take first element
    if isinstance(data_path, list):
        data_path = data_path[0] if data_path else None
    if data_path is None:
        print_rank_0("Warning: --data-path is empty, using mock data")
        return _mock_datasets(args)

    max_samples = getattr(args, 'train_samples', None)
    train_ds = ArrowDataset(
        data_path=data_path,
        seq_length=args.seq_length,
        max_samples=max_samples,
    )
    return train_ds, None, None


def _mock_datasets(args):
    """Fallback mock dataset."""
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, n, slen, vs):
            self.n = n
            self.slen = slen
            self.vs = vs
            self.fixed = torch.randint(0, vs, (slen,))

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return {'tokens': self.fixed, 'labels': self.fixed, 'loss_mask': torch.ones(self.slen)}

    vocab_size = getattr(args, 'vocab_size', 151936)
    seq_len = args.seq_length
    train_samples = getattr(args, 'train_samples', 100)
    return MockDataset(train_samples, seq_len, vocab_size), None, None


def get_model(model_provider_func, model_type, wrap_with_ddp=False):
    """Build model using quantized model provider."""
    args = get_args()
    model = quantized_model_provider(
        pre_process=True,
        post_process=True,
        quant_group_size=args.quant_group_size,
        lr_quant=args.lr_quant,
    )
    model = model.cuda()
    return model


def train(model, optimizer, train_dataloader, forward_step_func, config):
    """Training loop."""
    args = get_args()

    model.train()
    data_iterator = iter(train_dataloader) if train_dataloader is not None else None
    epoch_count = 0

    iter_times = []
    loss_history = []
    total_start_time = time.time()

    print_rank_0("=" * 70)
    print_rank_0("Starting Fine-tuning with Dynamic Quantization")
    print_rank_0(f"  group_size={args.quant_group_size}, score_update={args.score_update_interval}")
    print_rank_0(f"  top_bf16_ratio={args.top_bf16_ratio}, top_int8_ratio={args.top_int8_ratio}")
    print_rank_0(f"  lr={args.lr}, lr_quant={args.lr_quant}")
    print_rank_0("=" * 70)

    for iteration in range(1, args.train_iters + 1):
        iter_start = time.time()

        optimizer.zero_grad()

        try:
            output_tensor, loss_func_partial = forward_step_func(data_iterator, model)
        except StopIteration:
            epoch_count += 1
            print_rank_0(f'[Epoch {epoch_count}] Recreating data iterator')
            data_iterator = iter(train_dataloader) if train_dataloader is not None else None
            output_tensor, loss_func_partial = forward_step_func(data_iterator, model)

        loss, loss_dict = loss_func_partial(output_tensor, model)

        loss.backward()
        torch.cuda.synchronize()

        optimizer.step()

        iter_time = time.time() - iter_start
        iter_times.append(iter_time)
        current_loss = loss_dict['lm loss'].item()
        loss_history.append(current_loss)

        if iteration % args.log_interval == 0:
            loss_str = ' | '.join([f'{k}: {v.item():.6e}' for k, v in loss_dict.items()])
            timing_warmup = getattr(args, 'timing_warmup_iters', 5)
            warmed = iter_times[timing_warmup:] if len(iter_times) > timing_warmup else iter_times
            avg_time = sum(warmed) / len(warmed) if warmed else iter_time
            tokens_per_iter = args.micro_batch_size * args.seq_length
            throughput = tokens_per_iter / iter_time if iter_time > 0 else 0

            loss_trend = "?"
            if len(loss_history) >= 10:
                r = loss_history[-10:]
                loss_trend = "↓" if r[-1] < r[0] else "↑" if r[-1] > r[0] else "→"

            warmup_tag = f'(warmup: {iteration}/{timing_warmup})' if iteration <= timing_warmup else ''

            prec_str = ""
            if hasattr(optimizer, '_quant_precision'):
                counts = {}
                for prec in optimizer._quant_precision.values():
                    counts[prec] = counts.get(prec, 0) + 1
                prec_str = f"| BF16:{counts.get(16,0)} INT8:{counts.get(8,0)} INT4:{counts.get(4,0)}"

            remaining = args.train_iters - iteration
            eta_min = remaining * avg_time / 60

            print_rank_0(
                f'iter {iteration}/{args.train_iters} | {loss_str} | trend: {loss_trend} | '
                f'time: {iter_time:.3f}s | avg: {avg_time:.3f}s {warmup_tag} | '
                f'thru: {throughput:.0f} tok/s | ETA: {eta_min:.1f}min {prec_str}'
            )

        if args.save and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, None)

    # Summary
    total_time = time.time() - total_start_time
    timing_warmup = getattr(args, 'timing_warmup_iters', 5)
    warmed = iter_times[timing_warmup:] if len(iter_times) > timing_warmup else iter_times

    print_rank_0("=" * 70)
    print_rank_0("Fine-tuning Summary")
    print_rank_0(f'  Total iterations: {args.train_iters}')
    print_rank_0(f'  Total time: {total_time/60:.1f}min')
    if warmed:
        print_rank_0(f'  Avg iter (warmed): {sum(warmed)/len(warmed):.3f}s')
        tput = len(warmed) * args.micro_batch_size * args.seq_length / sum(warmed)
        print_rank_0(f'  Throughput: {tput:.0f} tok/s')

    print_rank_0(f'  Initial loss: {loss_history[0]:.4f}')
    print_rank_0(f'  Final loss:   {loss_history[-1]:.4f}')
    print_rank_0("=" * 70)

    return loss_dict


def main():
    """Main entry point."""

    args = parse_args(extra_args_provider=add_qwen3_moe_args)
    validate_args(args, {'data_path': False})

    initialize_megatron(extra_args_provider=add_qwen3_moe_args)

    args = get_args()
    timers = get_timers()

    ep_rank = parallel_state.get_expert_model_parallel_rank()
    torch.manual_seed(args.seed + ep_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + ep_rank)

    # Build model
    timers('model-setup', log_level=0).start()
    model = get_model(None, ModelType.encoder_or_decoder)
    timers('model-setup').stop()

    n_params = sum(p.numel() for p in model.parameters())
    print_rank_0(f'Model built: {n_params / 1e9:.2f}B parameters')
    print_rank_0(f'QK-Norm: {getattr(args, "qk_layernorm", False)}')
    print_rank_0(f'Quant: group_size={args.quant_group_size}, lr_quant={args.lr_quant}')

    # Build datasets early (before optimizer/checkpoint, to fail fast on data issues)
    timers('data-loading', log_level=0).start()
    train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(
        (getattr(args, 'train_samples', 5000000), 0, 0)
    )
    timers('data-loading').stop()

    train_dataloader = None
    if train_ds is not None:
        train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.micro_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        print_rank_0(f"DataLoader: {len(train_ds)} samples, batch_size={args.micro_batch_size}")

    # Create optimizer (allocates CPU shared memory for expert weights)
    print_rank_0("Creating optimizer...")
    if HAVE_QUANT_OPTIMIZER:
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
        optimizer = FusedAdamLSQCPUOffloadOptimizer(
            model=model, lr=args.lr, betas=(0.9, 0.999),
            eps=1e-8, weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps, clip_grad=args.clip_grad,
        )

    # Setup expert modules
    cpu_update_event = optimizer.get_cpu_update_event()
    for layer in model.decoder.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            layer.mlp.experts.set_cpu_update_event(cpu_update_event)
            layer.mlp.experts.set_quant_optimizer(optimizer)

    # Load pre-trained weights from distcp checkpoint
    ckpt_path = getattr(args, 'ckpt_path', None)
    if ckpt_path is not None:
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")
        print_rank_0(f"\nLoading pre-trained weights from: {ckpt_path}")
        timers('checkpoint-load', log_level=0).start()
        result = load_finetune_checkpoint(
            model, ckpt_path,
            optimizer=optimizer if HAVE_QUANT_OPTIMIZER else None,
            num_global_experts=getattr(args, 'num_experts', 128),
        )
        timers('checkpoint-load').stop()
        print_rank_0(f"Checkpoint loaded: {result}")
        log_memory("after checkpoint load")
    else:
        print_rank_0("Warning: --ckpt-path not set, training from random initialization")

    config = core_transformer_config_from_args(args)

    # Train
    train(model, optimizer, train_dataloader, forward_step, config)


if __name__ == '__main__':
    main()
