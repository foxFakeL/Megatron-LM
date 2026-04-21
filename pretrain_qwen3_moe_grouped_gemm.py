# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pretrain script for Qwen3MoE using standard Megatron GPTModel with GroupedMLP.

This script uses ONLY standard Megatron modules:
- GPTModel from megatron.core.models.gpt
- MoELayer with GroupedMLP (standard GroupGEMM implementation)
- MoEAlltoAllTokenDispatcher for token permutation
- TopKRouter for routing

Usage:
    # Single GPU
    python pretrain_qwen3_moe_grouped_gemm.py --num-layers 5 --num-experts 128 --moe-grouped-gemm

    # Multi-GPU EP
    torchrun --nproc_per_node=2 pretrain_qwen3_moe_grouped_gemm.py \
        --num-layers 5 --num-experts 128 --moe-grouped-gemm \
        --expert-model-parallel-size 2 --moe-token-dispatcher-type alltoall
"""

import os
import sys
from functools import partial
from typing import Optional, Tuple, List

import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.training import (
    get_args,
    get_timers,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args


def model_provider(pre_process=True, post_process=True, config=None, pg_collection=None):
    """Build the GPTModel with Qwen3MoE configuration.

    Uses standard Megatron GPTModel with MoE enabled via:
    - num_experts (passed via args.num_experts)
    - moe_grouped_gemm (passed via args.moe_grouped_gemm)
    - moe_token_dispatcher_type (alltoall for EP)

    Args:
        pre_process: Include embedding layer
        post_process: Include output layer
        config: TransformerConfig (optional)
        pg_collection: Process groups (optional)

    Returns:
        GPTModel instance with MoE layers
    """
    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)

    # Use local spec (no TransformerEngine) for pure Megatron implementation
    # This uses GroupedMLP when moe_grouped_gemm=True
    transformer_layer_spec = get_gpt_layer_local_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        multi_latent_attention=args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
        normalization=args.normalization,
    )

    print_rank_0(f'Building Qwen3MoE with standard GPTModel...')
    print_rank_0(f'  num_experts: {args.num_experts}')
    print_rank_0(f'  moe_grouped_gemm: {args.moe_grouped_gemm}')
    print_rank_0(f'  moe_token_dispatcher_type: {args.moe_token_dispatcher_type}')
    print_rank_0(f'  expert_model_parallel_size: {args.expert_model_parallel_size}')

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        pg_collection=pg_collection,
    )

    return model


def get_batch(data_iterator):
    """Generate a batch from the data iterator.

    Args:
        data_iterator: Data iterator or None for mock data

    Returns:
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params
    """
    args = get_args()

    if data_iterator is None:
        # Mock data for testing
        batch_size = args.micro_batch_size
        seq_len = args.seq_length
        vocab_size = args.padded_vocab_size

        # Generate mock data
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        labels = tokens.clone()
        loss_mask = torch.ones(batch_size, seq_len, device='cuda')
        attention_mask = None
        position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        packed_seq_params = None

        return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params

    # Real data
    batch = next(data_iterator)
    tokens = batch['tokens'].cuda()
    labels = batch['labels'].cuda() if 'labels' in batch else tokens.clone()
    loss_mask = batch.get('loss_mask', torch.ones_like(tokens, device='cuda'))
    attention_mask = batch.get('attention_mask', None)
    position_ids = batch.get('position_ids', None)

    if position_ids is None:
        position_ids = torch.arange(tokens.shape[1], device='cuda').unsqueeze(0).expand(tokens.shape[0], -1)

    packed_seq_params = None

    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """Loss function for GPT language model.

    Args:
        loss_mask: Mask for valid tokens
        output_tensor: Loss values from model

    Returns:
        loss: Scalar loss
        num_tokens: Number of valid tokens
        report: Dict with metrics
    """
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator: Input data iterator
        model: GPTModel instance

    Returns:
        output_tensor: Loss tensor
        loss_func_partial: Partial loss function
    """
    args = get_args()
    timers = get_timers()

    # Get batch
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator)
    timers('batch-generator').stop()

    # Forward pass
    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        packed_seq_params=packed_seq_params,
    )

    return output_tensor, partial(loss_func, loss_mask, model=model)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets.

    Creates mock datasets for testing without real data.
    Uses a simple MockDataset that generates random tokens without requiring a tokenizer.

    Args:
        train_val_test_num_samples: Tuple of (train_samples, valid_samples, test_samples)
    """
    args = get_args()

    class MockDataset(torch.utils.data.Dataset):
        """Simple mock dataset that generates random tokens without tokenizer."""
        def __init__(self, num_samples, seq_length, vocab_size):
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.vocab_size = vocab_size

            # Generate fixed random data (same every call for reproducibility)
            self.fixed_tokens = torch.randint(0, vocab_size, (seq_length,), dtype=torch.long)
            self.fixed_labels = self.fixed_tokens.clone()
            self.fixed_loss_mask = torch.ones(seq_length, dtype=torch.float)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                'tokens': self.fixed_tokens,
                'labels': self.fixed_labels,
                'loss_mask': self.fixed_loss_mask,
            }

    vocab_size = args.padded_vocab_size if hasattr(args, 'padded_vocab_size') else 151936
    seq_length = args.seq_length

    train_samples, valid_samples, test_samples = train_val_test_num_samples

    train_ds = MockDataset(train_samples, seq_length, vocab_size) if (train_samples or 0) > 0 else None
    valid_ds = MockDataset(valid_samples, seq_length, vocab_size) if (valid_samples or 0) > 0 else None
    test_ds = MockDataset(test_samples, seq_length, vocab_size) if (test_samples or 0) > 0 else None

    print_rank_0('> built mock datasets for Qwen3MoE (no tokenizer required) ...')

    return train_ds, valid_ds, test_ds


if __name__ == '__main__':
    # Run pretrain with standard Megatron infrastructure
    # All Qwen3-specific parameters are passed via command line in the launch script
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={
            'tokenizer_type': 'NullTokenizer',
        },
    )