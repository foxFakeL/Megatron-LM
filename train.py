"""Thin wrapper around pretrain_gpt.py for standard training runs.

This keeps repo training entrypoints consistent while letting users provide
their own launch scripts.
"""

import time
from functools import partial

from gpt_builders import gpt_builder
from megatron.core.enums import ModelType
from megatron.training import inprocess_restart, pretrain, set_startup_timestamps
from model_provider import model_provider

from pretrain_gpt import (
    forward_step,
    get_embedding_ranks,
    train_valid_test_datasets_provider,
)


def main() -> None:
    program_start = time.time()
    main_entry = time.time()
    set_startup_timestamps(program_start=program_start, main_entry=main_entry)

    # Temporary for transition to core datasets.
    train_valid_test_datasets_provider.is_distributed = True

    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    wrapped_pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
