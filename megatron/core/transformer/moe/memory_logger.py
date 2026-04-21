# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Memory logging utility for debugging OOM issues."""

import os
import torch
import torch.distributed as dist

# Global flag to enable/disable memory logging
# Can be controlled via environment variable: MEGATRON_LOG_MEMORY=0 to disable
_MEMORY_LOGGING_ENABLED = os.environ.get('MEGATRON_LOG_MEMORY', '1') == '1'

def set_memory_logging(enabled: bool):
    """Enable or disable memory logging globally."""
    global _MEMORY_LOGGING_ENABLED
    _MEMORY_LOGGING_ENABLED = enabled

def is_memory_logging_enabled() -> bool:
    """Check if memory logging is enabled."""
    return _MEMORY_LOGGING_ENABLED


def log_memory(tag: str, rank: int = None, reset_peak: bool = True):
    """Log GPU memory usage with tag.

    Args:
        tag: Description of current operation
        rank: Optional rank override. If None, uses distributed rank
        reset_peak: Whether to reset peak memory stats after logging
    """
    if not _MEMORY_LOGGING_ENABLED:
        return

    if not torch.cuda.is_available():
        return

    if rank is None:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(f"[Rank {rank}] [{tag}] allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, max={max_allocated:.2f}GB")

    if reset_peak:
        torch.cuda.reset_peak_memory_stats()


def log_memory_summary(tag: str = "Memory Summary", rank: int = None):
    """Print detailed memory summary.

    Args:
        tag: Description for the summary
        rank: Optional rank override
    """
    if not torch.cuda.is_available():
        return

    if rank is None:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

    print(f"\n{'='*60}")
    print(f"[Rank {rank}] {tag}")
    print(f"{'='*60}")
    print(torch.cuda.memory_summary())
    print(f"{'='*60}\n")