"""Load Megatron Core distcp checkpoint into custom Qwen3MoE model.

Handles key mismatches between the standard TE-based checkpoint and our custom model:
- linear_qkv.layer_norm_weight -> input_layernorm.weight (separate norm vs fused)
- Expert weights -> optimizer._main_weight_storage (quantized) or model._weight1/_weight2 (standard)
"""

import logging
import pickle
import re
from typing import Dict, Optional, Tuple

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load as distcp_load
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.moe.memory_logger import log_memory

logger = logging.getLogger(__name__)


def _read_metadata(ckpt_dir: str):
    """Read the checkpoint metadata file."""
    with open(f"{ckpt_dir}/release/.metadata", "rb") as f:
        return pickle.load(f)


def _build_load_sd(metadata) -> Dict[str, ShardedTensor]:
    """Build a sharded_state_dict that requests full tensors for all weights.

    For EP=1, each tensor is loaded as a single full chunk (no sharding).
    """
    sdm = metadata.state_dict_metadata
    sharded_sd = {}
    for key, tensor_meta in sdm.items():
        if not hasattr(tensor_meta, "size"):
            continue  # skip BytesStorageMetadata (_extra_state)
        global_shape = tuple(tensor_meta.size)
        st = ShardedTensor(
            key=key,
            data=None,
            dtype=torch.bfloat16,
            local_shape=global_shape,
            global_shape=global_shape,
            global_offset=(0,) * len(global_shape),
            axis_fragmentations=None,
            replica_id=0,
        )
        sharded_sd[key] = st
    return sharded_sd


def _is_expert_weight(ckpt_key: str) -> bool:
    """Check if this checkpoint key is an expert weight tensor."""
    return "mlp.experts.experts.linear_fc" in ckpt_key


def _parse_layer_idx(ckpt_key: str) -> Optional[int]:
    """Extract layer index from a checkpoint key like 'decoder.layers.N.xxx'."""
    m = re.search(r"decoder\.layers\.(\d+)", ckpt_key)
    return int(m.group(1)) if m else None


def _map_ckpt_key_to_model(ckpt_key: str) -> Optional[str]:
    """Map a checkpoint tensor key to our model's parameter name."""
    if _is_expert_weight(ckpt_key):
        return None

    # Map TE fused QKV layer_norm to separate input_layernorm
    if "linear_qkv.layer_norm_weight" in ckpt_key:
        layer_idx = _parse_layer_idx(ckpt_key)
        return f"decoder.layers.{layer_idx}.input_layernorm.weight"

    return ckpt_key


def load_finetune_checkpoint(
    model,
    ckpt_dir: str,
    strict: bool = False,
    optimizer=None,
    num_global_experts: int = 128,
) -> Dict:
    """Load distcp checkpoint weights into the custom Qwen3MoE model.

    Args:
        model: Qwen3MoEQuantizedModel (or Qwen3MoEModel) instance
        ckpt_dir: Path to checkpoint directory (contains 'release/' subdir)
        strict: If True, raise on missing/unexpected keys
        optimizer: Optional FusedAdamLSQCPUOffloadOptimizer for quantized models
        num_global_experts: Number of global experts (for computing global expert IDs)

    Returns:
        Summary dict with loaded/missing/unexpected key counts
    """
    rank = parallel_state.get_expert_model_parallel_rank()

    ckpt_release_dir = f"{ckpt_dir}/release"
    metadata = _read_metadata(ckpt_dir)
    sharded_sd = _build_load_sd(metadata)

    if rank == 0:
        logger.info(f"Loading checkpoint from {ckpt_release_dir}")
        logger.info(f"  {len(sharded_sd)} tensors to load")

    # Load all tensors via Megatron distcp loader
    loaded = distcp_load(
        sharded_sd,
        ckpt_release_dir,
        strict="return_all" if strict else "assume_ok_unexpected",
    )
    log_memory("checkpoint_loader: after distcp_load")

    if isinstance(loaded, tuple):
        loaded_state, missing_keys, unexpected_keys = loaded
    else:
        loaded_state = loaded
        missing_keys = set()
        unexpected_keys = set()
    if rank == 0:
        logger.info(f"Loaded {len(loaded_state)} tensors from checkpoint")

    model_state = model.state_dict()
    use_optimizer_storage = optimizer is not None

    loaded_count = 0
    skipped_expert = 0
    skipped_extra = 0
    mapped_norm = 0

    for ckpt_key, tensor in loaded_state.items():
        if "_extra_state" in ckpt_key:
            skipped_extra += 1
            continue

        # Expert weights: copy to optimizer storage or model buffer
        if _is_expert_weight(ckpt_key):
            layer_idx = _parse_layer_idx(ckpt_key)
            if layer_idx is not None and layer_idx < len(model.decoder.layers):
                is_w1 = "linear_fc1.weight" in ckpt_key
                if use_optimizer_storage:
                    # Quantized model: copy to optimizer._main_weight_storage
                    # Expert weight shape: [num_experts, out_features, in_features]
                    # Split by expert and store per expert
                    num_experts_in_ckpt = tensor.shape[0]
                    for expert_idx in range(num_experts_in_ckpt):
                        global_expert_id = layer_idx * num_global_experts + expert_idx
                        try:
                            w1_storage, w2_storage = optimizer._get_main_weight(global_expert_id)
                            # ckpt stores [out_features, in_features], optimizer stores [in_features, out_features]
                            if is_w1:
                                w1_storage.copy_(tensor[expert_idx].cpu().T)
                            else:
                                w2_storage.copy_(tensor[expert_idx].cpu().T)
                        except (KeyError, ValueError):
                            skipped_expert += 1
                    loaded_count += num_experts_in_ckpt
                else:
                    # Standard model: copy to _weight1/_weight2 CPU buffers
                    mlp_layer = model.decoder.layers[layer_idx].mlp
                    if hasattr(mlp_layer, 'experts'):
                        try:
                            buf = mlp_layer.experts._weight1 if is_w1 else mlp_layer.experts._weight2
                            buf.copy_(tensor.cpu())
                            loaded_count += 1
                        except AttributeError:
                            skipped_expert += 1
            continue

        # Map key and check if model has this parameter
        model_key = _map_ckpt_key_to_model(ckpt_key)
        if model_key is None:
            skipped_extra += 1
            continue

        if model_key in model_state:
            param = model_state[model_key]
            if tensor.shape == param.shape:
                param.copy_(tensor)
                loaded_count += 1
                if "input_layernorm" in model_key and "layer_norm_weight" in ckpt_key:
                    mapped_norm += 1
            else:
                if rank == 0:
                    logger.warning(
                        f"Shape mismatch: {model_key}: ckpt={tensor.shape} vs model={param.shape}"
                    )
        else:
            if rank == 0:
                logger.debug(f"Model key not found: {model_key}")
            skipped_extra += 1

    # Free loaded tensors to reclaim GPU memory
    del loaded_state

    summary = {
        "loaded": loaded_count,
        "mapped_norm": mapped_norm,
        "skipped_expert": skipped_expert,
        "skipped_extra": skipped_extra,
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
    }

    if rank == 0:
        logger.info(f"Checkpoint loading complete: {summary}")

    return summary
