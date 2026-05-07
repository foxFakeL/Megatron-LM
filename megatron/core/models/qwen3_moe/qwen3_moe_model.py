"""Qwen3MoE Model using Megatron Core modules.

This model uses:
- Standard Megatron SelfAttention (with Flash Attention support)
- Custom FusedDispatcherCacheGroupedMLP for MoE layers
- Standard Megatron training infrastructure
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.moe_utils import MoECudaGraphPartialCaptureSignal
from megatron.core.transformer.moe.experts import FusedDispatcherCacheGroupedMLP
from megatron.core.transformer.moe.memory_logger import log_memory

# Liger Kernel for memory-efficient cross entropy
try:
    from liger_kernel.transformers import LigerCrossEntropyLoss
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False
    LigerCrossEntropyLoss = None


@dataclass
class Qwen3MoESubmodules:
    """Submodules for Qwen3MoE layer."""
    pass


class Qwen3MoEMLPLayer(torch.nn.Module):
    """MoE MLP layer using FusedDispatcherCacheGroupedMLP.

    This replaces the standard MLP in MoE layers.
    """

    def __init__(
        self,
        config: TransformerConfig,
        num_global_experts: int,
        pg_collection: ProcessGroupCollection,
        expert_sets: List[List[int]],
        layer_number: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.num_global_experts = num_global_experts
        self.layer_number = layer_number

        # Router (uses config.num_moe_experts for num_experts)
        self.router = TopKRouter(
            config=config,
            pg_collection=pg_collection,
        )
        # Set router's layer_number immediately after creation
        if layer_number is not None:
            self.router.set_layer_number(layer_number)

        # Experts using FusedDispatcherCacheGroupedMLP
        self.experts = FusedDispatcherCacheGroupedMLP(
            num_global_experts=num_global_experts,
            config=config,
            pg_collection=pg_collection,
            layer_number=layer_number,
        )

        self.expert_sets = expert_sets

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
        forward_expert_set=None,
    ) -> Tensor:
        """Forward pass.

        Args:
            hidden_states: [seq_len, batch, hidden_size] (standard Megatron format)

        Returns:
            output: [seq_len, batch, hidden_size]
        """
        # Store input dtype to ensure output has the same dtype
        input_dtype = hidden_states.dtype

        # Ensure hidden_states is the expected dtype (bf16 for grouped_gemm compatibility)
        # This is needed because TransformerEngine attention may output float32
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        # Input is [s, b, h] format
        seq_len, batch_size, hidden_size = hidden_states.shape

        # Router forward - returns (probs, routing_map)
        probs, routing_map = self.router(hidden_states)
        log_memory("Qwen3MoEMLPLayer: after router")

        # Reshape for expert computation: [s*b, h]
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        probs_flat = probs.view(-1, self.num_global_experts)
        routing_map_flat = routing_map.view(-1, self.num_global_experts)

        # Expert computation
        output, _ = self.experts(
            hidden_states=hidden_states_flat,
            routing_map=routing_map_flat,
            probs=probs_flat,
            expert_sets=self.expert_sets if forward_expert_set is None else forward_expert_set,
        )
        log_memory("Qwen3MoEMLPLayer: after experts")

        # Reshape back to [s, b, h]
        output = output.view(seq_len, batch_size, hidden_size)

        # Ensure output has the same dtype as input
        if output.dtype != input_dtype:
            output = output.to(input_dtype)

        return output, None  # (output, bias) - no bias for MoE

    def set_layer_number(self, layer_number: int):
        """Set layer number for router."""
        self.layer_number = layer_number
        if self.router is not None:
            self.router.set_layer_number(layer_number)


class Qwen3MoEModel(LanguageModule):
    """Qwen3MoE Transformer model.

    This model combines:
    - Standard Megatron SelfAttention (with Flash Attention 2 support)
    - FusedDispatcherCacheGroupedMLP for MoE FFN layers
    - RoPE embeddings
    - GQA (Grouped Query Attention)

    Args:
        config: TransformerConfig
        transformer_layer_spec: Layer specification
        vocab_size: Vocabulary size
        max_sequence_length: Maximum sequence length
        num_experts: Number of MoE experts
        experts_per_set: Experts per processing set
        pre_process: Include embedding layer
        post_process: Include output layer
        parallel_output: Keep outputs split across TP ranks
        share_embeddings_and_output_weights: Share embedding and output weights
        position_embedding_type: Position embedding type ('rope' for Qwen3)
        rotary_base: RoPE base frequency
        rotary_percent: Percent of hidden size for RoPE
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        num_experts: int = 128,
        experts_per_set: int = 16,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'rope',
        rotary_base: int = 1000000,
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.transformer_layer_spec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.num_experts = num_experts
        self.experts_per_set = experts_per_set
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # Get process groups
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        self.pg_collection = pg_collection

        # Calculate expert sets for this rank
        # CRITICAL: experts_per_set must be dynamically adjusted based on EP size
        # to ensure proper pipeline overlap. We want ~4 sets per rank for good overlap.
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        experts_per_rank = num_experts // ep_size

        # Dynamic adjustment: always aim for 4 sets per rank for pipeline overlap
        # Target: experts_per_set = experts_per_rank // 4 (if possible)
        # This ensures better pipeline overlap (dispatch-set0 overlaps with compute-set1)
        target_sets_per_rank = 4
        target_experts_per_set = max(1, experts_per_rank // target_sets_per_rank)

        # Only use user's experts_per_set if it results in >= target_sets_per_rank
        user_sets = experts_per_rank // experts_per_set if experts_per_rank % experts_per_set == 0 else 1
        if user_sets < target_sets_per_rank:
            # User's setting has fewer sets than target, use target_experts_per_set
            experts_per_set = target_experts_per_set
            # Ensure divisibility
            while experts_per_rank % experts_per_set != 0 and experts_per_set > 1:
                experts_per_set -= 1

        self.experts_per_set = experts_per_set  # Store adjusted value
        local_expert_start = ep_rank * experts_per_rank
        expert_list = [local_expert_start + i for i in range(experts_per_rank)]
        self.expert_sets = [
            expert_list[i:i + experts_per_set]
            for i in range(0, len(expert_list), experts_per_set)
        ]

        # Embedding layer
        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type='none',  # We use RoPE, so no learned position embedding
            )
            log_memory("Qwen3MoEModel.__init__: after embedding")

        # RoPE
        if position_embedding_type == 'rope':
            # kv_channels = hidden_size / num_attention_heads
            kv_channels = self.config.hidden_size // self.config.num_attention_heads
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=kv_channels,
                rotary_base=rotary_base,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
            )
        else:
            self.rotary_pos_emb = None

        # Transformer blocks
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            post_layer_norm=True,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        log_memory("Qwen3MoEModel.__init__: after decoder")

        # Output layer
        if self.post_process:
            # Use ColumnParallelLinear with proper initialization method
            # This ensures logits are in a reasonable range (not exploding)
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.vocab_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=False,
                gather_output=not self.parallel_output,
                skip_bias_add=False,
                # Skip weight allocation if sharing with embedding
                skip_weight_param_allocation=self.share_embeddings_and_output_weights,
                tp_group=self.pg_collection.tp,
            )

            # Liger Kernel cross entropy loss (memory-efficient)
            if HAS_LIGER:
                # Use softcap to prevent numerical overflow with large logits
                self.liger_loss_fn = LigerCrossEntropyLoss(softcap=30.0)
            else:
                self.liger_loss_fn = None

        # Setup embeddings and output layer weight tying
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        # Replace MLP layers with MoE layers
        self._replace_mlp_with_moe()

        # Log expert configuration for debugging
        if parallel_state.get_expert_model_parallel_rank() == 0:
            print(f"[Qwen3MoE] EP={ep_size}, experts_per_rank={experts_per_rank}, "
                  f"experts_per_set={self.experts_per_set}, num_sets={len(self.expert_sets)}")
        log_memory("Qwen3MoEModel.__init__: after MoE replacement")

    def _replace_mlp_with_moe(self):
        """Replace MLP layers in decoder with MoE layers."""
        for layer in self.decoder.layers:
            if hasattr(layer, 'mlp') and layer.mlp is not None:
                # Create MoE MLP layer
                moe_mlp = Qwen3MoEMLPLayer(
                    config=self.config,
                    num_global_experts=self.num_experts,
                    pg_collection=self.pg_collection,
                    expert_sets=self.expert_sets,
                    layer_number=layer.layer_number,
                )
                layer.mlp = moe_mlp

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor for pipeline parallelism."""
        self.decoder.set_input_tensor(input_tensor)

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """Memory-efficient cross entropy using Liger Kernel.

        Args:
            labels: [batch, seq_len] labels (already shifted, last position = -100)
            logits: [seq, batch, vocab] logits (Megatron format)

        Returns:
            loss: Scalar mean loss over non-ignored positions
        """
        if self.liger_loss_fn is not None:
            vocab_size = logits.shape[-1]

            # logits is [s, b, v], view as [s*b, v] - no copy, just reshape
            logits_flat = logits.view(-1, vocab_size)

            # Transpose labels from [b, s] to [s, b] then flatten
            labels_flat = labels.T.contiguous().view(-1)

            # Liger returns scalar mean loss (reduction='mean' by default)
            loss = self.liger_loss_fn(logits_flat, labels_flat)
            return loss
        else:
            return super().compute_language_model_loss(labels, logits)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] input token ids
            position_ids: [batch, seq_len] position ids
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation

        Returns:
            logits or (logits, loss) if labels provided
        """
        # Embedding
        if self.pre_process:
            hidden_states = self.embedding(
                input_ids=input_ids,
                position_ids=position_ids,
            )
            # Ensure hidden_states is bfloat16 (params_dtype may be set but embedding could output different type)
            # This is critical for TransformerEngine compatibility
            if hidden_states.dtype != torch.bfloat16:
                hidden_states = hidden_states.to(torch.bfloat16)
            log_memory("Qwen3MoEModel.forward: after embedding")
        else:
            hidden_states = input_ids

        # RoPE
        rotary_pos_emb = None
        if self.rotary_pos_emb is not None:
            # hidden_states is [seq, batch, hidden] format after embedding
            # Use shape[0] for sequence length
            seq_len = hidden_states.shape[0]
            rotary_pos_emb = self.rotary_pos_emb(seq_len)

        # Transformer blocks
        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            context=None,
            context_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )
        # Ensure hidden_states is bfloat16 for output_layer compatibility
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        log_memory("Qwen3MoEModel.forward: after decoder")

        # Output
        if self.post_process:
            # Get output weight if sharing with embedding
            output_weight = None
            if self.share_embeddings_and_output_weights:
                output_weight = self.shared_embedding_or_output_weight()

            # ColumnParallelLinear returns (output, bias)
            logits, _ = self.output_layer(hidden_states, weight=output_weight)
            log_memory("Qwen3MoEModel.forward: after output_layer")

            if labels is not None:
                # Compute loss using Liger Kernel (memory-efficient)
                loss = self.compute_language_model_loss(labels, logits)
                # Delete logits to save memory - gradients flow through computation graph
                del logits
                # Return dict with loss for loss_func to process
                return {'loss': loss}

            if self.parallel_output:
                return logits
            else:
                return gather_from_sequence_parallel_region(logits)

        return hidden_states

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with custom handling."""
        return super().load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return state dict."""
        return super().state_dict(destination, prefix, keep_vars)


def model_provider(
    pre_process=True,
    post_process=True,
    config=None,
    pg_collection=None,
) -> Qwen3MoEModel:
    """Model provider function for Megatron training.

    This function is called by the pretrain script to create the model.
    Configure model parameters via command line arguments.

    Returns:
        Qwen3MoEModel instance
    """
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.core.models.qwen3_moe.qwen3_moe_layer_specs import get_qwen3_moe_block_spec
    from megatron.training import get_args

    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)

    # Get model parameters from args
    num_experts = getattr(args, 'num_moe_experts', 128)
    experts_per_set = getattr(args, 'experts_per_set', 16)
    rotary_base = getattr(args, 'rope_theta', 1000000)

    # Get block spec
    # Use TransformerEngine if transformer_impl is set to "transformer_engine" (default)
    use_te = getattr(config, 'transformer_impl', 'local') == 'transformer_engine'
    qk_layernorm = getattr(args, 'qk_layernorm', False) or getattr(config, 'qk_layernorm', False)
    block_spec = get_qwen3_moe_block_spec(
        num_layers=args.num_layers,
        use_te=use_te,
        num_experts=num_experts,
        experts_per_set=experts_per_set,
        qk_layernorm=qk_layernorm,
    )

    model = Qwen3MoEModel(
        config=config,
        transformer_layer_spec=block_spec,
        vocab_size=args.vocab_size if hasattr(args, 'vocab_size') else 151936,
        max_sequence_length=args.max_position_embeddings if hasattr(args, 'max_position_embeddings') else 4096,
        num_experts=num_experts,
        experts_per_set=experts_per_set,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type='rope',
        rotary_base=rotary_base,
        pg_collection=pg_collection,
    )

    return model