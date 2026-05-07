# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Layer specs for Qwen3MoE model with FusedDispatcherCacheGroupedMLP."""

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from megatron.core.extensions.transformer_engine import HAVE_TE, TENorm
if HAVE_TE:
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

try:
    import apex
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


def get_qwen3_moe_layer_spec(
    use_te: bool = False,
    num_experts: Optional[int] = None,
    experts_per_set: int = 16,
    qk_layernorm: bool = False,
) -> TransformerLayerSubmodules:
    """Get layer spec for Qwen3MoE model.

    Uses standard Megatron SelfAttention with custom FusedDispatcherCacheGroupedMLP.

    Args:
        use_te: Whether to use TransformerEngine
        num_experts: Number of experts (required for MoE layers)
        experts_per_set: Number of experts per processing set
        qk_layernorm: Whether to use QK-Norm (Qwen3 has per-head Q/K normalization)

    Returns:
        TransformerLayerSubmodules spec
    """
    if use_te and HAVE_TE:
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend = LocalSpecProvider()

    # Norm implementation
    layernorm_spec = ModuleSpec(module=LNImpl)

    # QK-Norm: Qwen3 uses per-head RMSNorm for Q and K
    # Reference: gpt_layer_specs.py uses backend.layer_norm(for_qk=True)
    qk_norm = backend.layer_norm(for_qk=True)

    # Attention spec - use standard Megatron SelfAttention
    # This includes Flash Attention support automatically
    self_attn_spec = ModuleSpec(
        module=SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=backend.column_parallel_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm if qk_layernorm else IdentityOp,
            k_layernorm=qk_norm if qk_layernorm else IdentityOp,
        ),
    )

    # MLP spec for non-MoE layers (if any)
    mlp_spec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(),
            linear_fc2=backend.row_parallel_linear(),
        ),
    )

    # MLP spec - will be replaced with MoE by model provider
    # We use a standard MLP placeholder since TransformerLayer requires a valid MLP spec
    mlp_spec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(),
            linear_fc2=backend.row_parallel_linear(),
        ),
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=self_attn_spec,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=layernorm_spec,  # Add LayerNorm before MoE to prevent activation explosion
            mlp=mlp_spec,  # Placeholder, will be replaced with MoE
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_qwen3_moe_block_spec(
    num_layers: int,
    use_te: bool = False,
    num_experts: Optional[int] = None,
    experts_per_set: int = 16,
    qk_layernorm: bool = False,
) -> TransformerBlockSubmodules:
    """Get block spec for Qwen3MoE model.

    Args:
        num_layers: Number of transformer layers
        use_te: Whether to use TransformerEngine
        num_experts: Number of experts
        experts_per_set: Experts per processing set
        qk_layernorm: Whether to use QK-Norm

    Returns:
        TransformerBlockSubmodules spec
    """
    layer_spec = get_qwen3_moe_layer_spec(use_te, num_experts, experts_per_set, qk_layernorm)

    # Create list of layer specs
    layer_specs = [layer_spec] * num_layers

    return TransformerBlockSubmodules(
        layer_specs=layer_specs,
        layer_norm=ModuleSpec(module=LNImpl),
    )