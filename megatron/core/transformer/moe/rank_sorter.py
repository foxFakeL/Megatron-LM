"""Interfaces for token-to-rank sorting strategies in MoE layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from megatron.core import utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig


class TokenRankSorter(ABC):
    """Abstract base class for token-to-rank sorting strategies.

    A sorter can inspect router outputs and hidden states to decide how tokens should
    be distributed across expert-parallel ranks. Concrete implementations can
    implement arbitrary heuristics (load-aware, latency-aware, etc.).
    """

    def __init__(self, config: TransformerConfig, pg_collection: ProcessGroupCollection):
        self.config = config
        self.pg_collection = pg_collection
        self.ep_group = pg_collection.ep
        self.ep_size = utils.get_pg_size(self.ep_group)

    @abstractmethod
    def assign(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Return an optional rank assignment tensor for the dispatcher.

        Args:
            hidden_states: Original hidden representation of shape [..., hidden_size].
            probs: Router probabilities of shape [tokens, num_experts].
            routing_map: Router selection mask of shape [tokens, num_experts].

        Returns:
            An optional tensor whose content is interpreted by the dispatcher.
            Returning ``None`` keeps the default expert-based routing behaviour.
        """


class IdentityTokenRankSorter(TokenRankSorter):
    """Default sorter that preserves the original expert-based routing map."""

    def assign(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Optional[torch.Tensor]:
        _ = hidden_states, probs, routing_map
        return None


def build_token_rank_sorter(
    config: TransformerConfig, pg_collection: ProcessGroupCollection
) -> TokenRankSorter:
    """Factory helper that instantiates the configured sorter.

    The transformer config can optionally expose ``moe_rank_sorter``. When unset or
    set to ``"identity"``, the default sorter is used. Custom implementations can be
    registered by injecting a callable via ``config.moe_rank_sorter_builder`` that
    returns a :class:`TokenRankSorter` instance.
    """

    if hasattr(config, "moe_rank_sorter_builder") and config.moe_rank_sorter_builder is not None:
        return config.moe_rank_sorter_builder(config=config, pg_collection=pg_collection)

    sorter_name = getattr(config, "moe_rank_sorter", "identity")
    if sorter_name in (None, "identity"):
        return IdentityTokenRankSorter(config=config, pg_collection=pg_collection)

    raise ValueError(f"Unknown moe_rank_sorter '{sorter_name}'. Provide moe_rank_sorter_builder.")
