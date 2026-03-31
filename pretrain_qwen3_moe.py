#!/usr/bin/env python
"""Pretrain script for Qwen3MoE using FusedDispatcherCacheGroupedMLP.

This script demonstrates training Qwen3MoE model with:
- Full Transformer layers (Attention + MoE FFN)
- FusedDispatcherCacheGroupedMLP for MoE FFN layers
- TopKRouter for token routing
- EP (Expert Parallelism) only, no TP/PP

Usage:
    # Single-GPU test
    CUDA_VISIBLE_DEVICES=0 python pretrain_qwen3_moe.py --iters 10 --num-layers 2

    # Multi-GPU test (EP=2)
    torchrun --nproc_per_node=2 pretrain_qwen3_moe.py --iters 10 --num-layers 2

    # With activation offload
    python pretrain_qwen3_moe.py --activation-offload

Note: Dataset and weights are placeholder (mock data).
"""

import argparse
import os
import time
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.experts import FusedDispatcherCacheGroupedMLP

try:
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import transformer_engine as te
    from transformer_engine.common import recipe
    HAVE_TE_MODULE = True
except ImportError:
    HAVE_TE_MODULE = False


def _init_distributed() -> Tuple[int, int, int]:
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


def _initialize_model_parallel(tp: int, pp: int, ep: int) -> None:
    """Initialize model parallel groups."""
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
    )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, num_heads, head_dim]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_heads, seq_len, head_dim]
        cos: [batch, seq_len, head_dim]
        sin: [batch, seq_len, head_dim]
    """
    # Reshape cos/sin for broadcasting: [batch, seq, head_dim] -> [batch, 1, seq, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    """Multi-headed attention with GQA support for Qwen3MoE."""

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)
        self.rope_theta = getattr(config, 'rope_theta', 1000000.0)

        # QKV projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Transpose for attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply rotary embedding
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat k/v heads if num_key_value_heads < num_heads (GQA)
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Causal mask (lower triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        return self.o_proj(attn_output)


class Qwen3MoEFFN(nn.Module):
    """MoE FFN layer using FusedDispatcherCacheGroupedMLP.

    This module wraps Router + FusedDispatcherCacheGroupedMLP to provide
    a standard FFN-like interface for Transformer layers.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection,
        layer_idx: int,
        expert_sets: List[List[int]],
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.expert_sets = expert_sets
        self.num_global_experts = config.num_moe_experts

        # Router
        self.router = TopKRouter(config=config, pg_collection=pg_collection)

        # Expert MLP
        self.experts = FusedDispatcherCacheGroupedMLP(
            num_global_experts=self.num_global_experts,
            config=config,
            pg_collection=pg_collection,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Router expects [seq_len, batch, hidden_size]
        hidden_states_router = hidden_states.transpose(0, 1).contiguous()

        # Get routing probabilities
        probs, routing_map = self.router(hidden_states_router)

        # Flatten for FusedDispatcherCacheGroupedMLP
        # [S, B, H] -> [S*B, H]
        hidden_states_flat = hidden_states_router.reshape(-1, hidden_size)
        probs_flat = probs.reshape(-1, self.num_global_experts)
        routing_map_flat = routing_map.reshape(-1, self.num_global_experts)

        # Expert computation
        output, _ = self.experts(
            hidden_states=hidden_states_flat,
            routing_map=routing_map_flat,
            probs=probs_flat,
            expert_sets=self.expert_sets,
        )

        # Reshape back to [batch, seq_len, hidden_size]
        output = output.view(seq_len, batch_size, hidden_size).transpose(0, 1)

        return output

    def sync_gradients(self):
        """Sync gradients from CPU to GPU."""
        self.experts.sync_gradients()


class Qwen3MoETransformerBlock(nn.Module):
    """Transformer block for Qwen3MoE with Attention + MoE FFN."""

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection,
        layer_idx: int,
        expert_sets: List[List[int]],
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, 'layernorm_epsilon', 1e-6))

        # Attention
        self.self_attn = Qwen3Attention(config, layer_idx)

        # Pre-FFN norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, 'layernorm_epsilon', 1e-6))

        # MoE FFN
        self.mlp = Qwen3MoEFFN(config, pg_collection, layer_idx, expert_sets)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm + Attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-norm + MoE FFN + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def sync_gradients(self):
        """Sync gradients for MoE expert weights."""
        self.mlp.sync_gradients()


class Qwen3MoEModel(nn.Module):
    """Full Qwen3MoE model with embedding, transformer blocks, and output."""

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection,
        expert_sets: List[List[int]],
        vocab_size: int = 151936,
    ):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            Qwen3MoETransformerBlock(config, pg_collection, i, expert_sets)
            for i in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=getattr(config, 'layernorm_epsilon', 1e-6))

        # Output projection (tied with embedding)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, attention_mask)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Output projection
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )

        return logits, loss

    def sync_gradients(self):
        """Sync gradients for all MoE expert weights."""
        for layer in self.layers:
            layer.sync_gradients()


def create_qwen3_moe_config(args: argparse.Namespace) -> TransformerConfig:
    """Create TransformerConfig with Qwen3MoE parameters."""
    bf16 = bool(args.bf16 and torch.cuda.is_bf16_supported())
    params_dtype = torch.bfloat16 if bf16 else torch.float32

    config = TransformerConfig(
        # Model architecture
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_kv_heads,  # GQA: num_query_groups = num_kv_heads

        # FFN dimensions
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,

        # MoE configuration
        num_moe_experts=args.num_experts,
        moe_router_topk=args.moe_router_topk,
        moe_router_pre_softmax=True,
        moe_layer_freq=1,
        moe_token_dispatcher_type="alltoall",

        # Activation
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,

        # Norm
        layernorm_epsilon=1e-6,

        # Training config
        bf16=bf16,
        params_dtype=params_dtype,
        use_cpu_initialization=True,
        moe_enable_expert_weight_cache=True,
        moe_activation_offload=args.activation_offload,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    return config


def get_expert_sets(ep_rank: int, ep_size: int, num_global_experts: int, experts_per_set: int) -> List[List[int]]:
    """Generate expert_sets for current EP rank."""
    experts_per_rank = num_global_experts // ep_size
    local_expert_start = ep_rank * experts_per_rank
    expert_list = [local_expert_start + i for i in range(experts_per_rank)]

    expert_sets = [
        expert_list[i:i + experts_per_set]
        for i in range(0, len(expert_list), experts_per_set)
    ]

    return expert_sets


def main():
    parser = argparse.ArgumentParser(description="Qwen3MoE Pretrain Demo")

    # Training config
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Qwen3MoE architecture
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--num-attention-heads", type=int, default=64, help="Number of attention heads")
    parser.add_argument("--num-kv-heads", type=int, default=4, help="Number of KV heads (GQA)")
    parser.add_argument("--ffn-hidden-size", type=int, default=12288, help="Dense FFN hidden size")
    parser.add_argument("--moe-ffn-hidden-size", type=int, default=1536, help="Expert FFN hidden size")
    parser.add_argument("--num-experts", type=int, default=128, help="Total number of experts")
    parser.add_argument("--moe-router-topk", type=int, default=8, help="Top-k experts per token")
    parser.add_argument("--experts-per-set", type=int, default=16, help="Experts per processing set")
    parser.add_argument("--vocab-size", type=int, default=151936, help="Vocabulary size")
    parser.add_argument("--max-position-embeddings", type=int, default=40960, help="Max position embeddings")
    parser.add_argument("--rope-theta", type=float, default=1000000.0, help="RoPE theta")

    # MoE optimization
    parser.add_argument("--activation-offload", action="store_true",
                        help="Enable MoE activation offload to CPU")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    # Initialize distributed
    rank, world_size, local_rank = _init_distributed()

    # EP only
    tp_size = 1
    pp_size = 1
    ep_size = world_size
    _initialize_model_parallel(tp=tp_size, pp=pp_size, ep=ep_size)

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)

    # Create config
    config = create_qwen3_moe_config(args)

    # Get EP group
    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_rank = dist.get_rank(ep_group)

    # Create ProcessGroupCollection
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Generate expert_sets for this rank
    expert_sets = get_expert_sets(
        ep_rank=ep_rank,
        ep_size=ep_size,
        num_global_experts=args.num_experts,
        experts_per_set=args.experts_per_set,
    )

    # Create model
    model = Qwen3MoEModel(
        config=config,
        pg_collection=pg_collection,
        expert_sets=expert_sets,
        vocab_size=args.vocab_size,
    )

    if config.bf16:
        model = model.bfloat16()

    model.train()

    # Move non-expert parameters to GPU, keep expert weights on CPU
    # This is critical for FusedDispatcherCacheGroupedMLP which stores weights in CPU shared memory
    device = torch.device("cuda")
    for name, param in model.named_parameters():
        if 'experts.weight1' not in name and 'experts.weight2' not in name:
            param.data = param.data.to(device)

    # Move buffers to GPU as well (e.g., rotary embedding inv_freq)
    for name, buffer in model.named_buffers():
        if buffer is not None:
            buffer.data = buffer.data.to(device)

    # Create optimizer (only optimize non-expert params here, expert params are handled separately)
    # For simplicity, we include all params - gradients for expert weights will be on CPU
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    batch_size = args.micro_batch_size
    seq_len = args.seq_length
    vocab_size = args.vocab_size

    try:
        dist.barrier()

        if rank == 0:
            print("=" * 60)
            print("Qwen3MoE Training Demo with FusedDispatcherCacheGroupedMLP")
            print("=" * 60)
            print(f"  World size (EP): {world_size}")
            print(f"  Num layers: {args.num_layers}")
            print(f"  Hidden size: {args.hidden_size}")
            print(f"  Num attention heads: {args.num_attention_heads}")
            print(f"  Num KV heads (GQA): {args.num_kv_heads}")
            print(f"  Num global experts: {args.num_experts}")
            print(f"  Experts per rank: {args.num_experts // ep_size}")
            print(f"  Router top-k: {args.moe_router_topk}")
            print(f"  Expert FFN hidden size: {args.moe_ffn_hidden_size}")
            print(f"  Vocab size: {vocab_size}")
            print(f"  Batch size: {batch_size}")
            print(f"  Seq length: {seq_len}")
            print(f"  Num iterations: {args.iters}")
            print(f"  Activation offload: {args.activation_offload}")
            print(f"  Dtype: {'bf16' if config.bf16 else 'fp32'}")
            print("=" * 60)

        total_forward_time = 0.0
        total_backward_time = 0.0
        total_optimizer_time = 0.0

        for it in range(args.iters):
            optimizer.zero_grad(set_to_none=True)

            # Generate mock data
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

            # Forward pass
            t0 = time.time()
            logits, loss = model(input_ids, position_ids, labels=labels)
            torch.cuda.synchronize()
            forward_time = time.time() - t0
            total_forward_time += forward_time

            # Backward pass
            t0 = time.time()
            loss.backward()

            # Sync gradients for MoE experts
            model.sync_gradients()

            torch.cuda.synchronize()
            backward_time = time.time() - t0
            total_backward_time += backward_time

            # Optimizer step
            t0 = time.time()
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_time = time.time() - t0
            total_optimizer_time += optimizer_time

            total_tokens = batch_size * seq_len

            if rank == 0 or it == 0:
                print(
                    f"iter {it}: loss={loss.item():.6f} "
                    f"tokens={total_tokens} "
                    f"fwd={forward_time:.3f}s bwd={backward_time:.3f}s "
                    f"opt={optimizer_time:.3f}s",
                    flush=True,
                )

        dist.barrier()

        # Print summary
        if rank == 0:
            print("\n" + "=" * 60)
            print("Training Summary:")
            print(f"  Total iterations: {args.iters}")
            print(f"  Avg forward time: {total_forward_time / args.iters:.3f}s")
            print(f"  Avg backward time: {total_backward_time / args.iters:.3f}s")
            print(f"  Avg optimizer time: {total_optimizer_time / args.iters:.3f}s")
            print("=" * 60)

    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        parallel_state.destroy_model_parallel()

    return 0


if __name__ == "__main__":
    main()