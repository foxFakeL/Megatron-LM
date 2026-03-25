# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Pipeline Buffer Manager for MoE Dispatcher + Expert Compute Pipeline.

This module provides a singleton buffer manager that supports the three-stream
pipeline architecture:
- compute_stream: GroupedGEMM computation
- load_stream: PCIe weight loading from CPU pinned memory
- comm_stream: NVLink all-to-all token dispatch

Key features:
- Double-buffered dispatch buffers (pre-allocated based on max_set_tokens)
- Double-buffered weight workspaces (shared with existing _GlobalBufferManager)
- CUDA streams and events for cross-stream synchronization
"""

from typing import List, Optional, Tuple
import torch


class FusedPipelineBufferManager:
    """Global singleton for fused pipeline buffers - shared across all layers.

    Since Transformer layers are executed sequentially, there's no need for each
    layer to have its own set of buffers. This singleton manager provides shared
    buffers that all FusedDispatcherCacheGroupedMLP instances can reuse.

    Double-buffering is used for:
    - dispatch_buffers: Token storage for async all-to-all
    - w1/w2_gpu_workspace: Weight storage for async PCIe transfer

    Triple-stream pipeline:
    - compute_stream (default): GroupedGEMM computation
    - load_stream: PCIe CPU->GPU weight transfer
    - comm_stream: NVLink all-to-all token dispatch
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_initialized(self) -> bool:
        """Check if buffers have been initialized."""
        return self._initialized

    def initialize(
        self,
        num_global_experts: int,
        hidden_size: int,
        fc1_out_features: int,
        ffn_hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        max_experts_per_set: Optional[int] = None,
        max_dispatch_tokens: Optional[int] = None,
    ):
        """Initialize global shared buffers for the fused pipeline.

        Args:
            num_global_experts: Total number of experts
            hidden_size: Model hidden dimension
            fc1_out_features: FC1 output features (ffn_hidden_size * 2 if GLU)
            ffn_hidden_size: FFN hidden dimension
            dtype: Data type for buffers
            device: GPU device
            max_experts_per_set: Maximum experts in a single set
            max_dispatch_tokens: Maximum tokens for dispatch buffer (can grow)
        """
        if self._initialized:
            return

        # Default max_experts_per_set to num_global_experts for safety
        if max_experts_per_set is None:
            max_experts_per_set = num_global_experts

        # GPU workspace for weights (double-buffered for async prefetch)
        # Shape: [2 buffers, max_experts_per_set, ...]
        self._w1_gpu_workspace = torch.empty(
            2, max_experts_per_set, hidden_size, fc1_out_features,
            dtype=dtype, device=device
        )
        self._w2_gpu_workspace = torch.empty(
            2, max_experts_per_set, ffn_hidden_size, hidden_size,
            dtype=dtype, device=device
        )

        # Dispatch buffers (double-buffered for async all-to-all)
        # Allocated lazily or pre-allocated if max_dispatch_tokens is provided
        self._dispatch_buffers: Optional[List[torch.Tensor]] = None
        self._dispatch_probs_buffers: Optional[List[torch.Tensor]] = None
        self._dispatch_buffer_size: int = 0
        self._dispatch_buffer_dtype = dtype
        self._dispatch_buffer_device = device
        self._hidden_size = hidden_size

        if max_dispatch_tokens is not None and max_dispatch_tokens > 0:
            self._allocate_dispatch_buffers(max_dispatch_tokens)

        # CUDA streams for pipeline
        self._compute_stream = torch.cuda.current_stream()  # Default stream
        self._load_stream = torch.cuda.Stream()
        self._comm_stream = torch.cuda.Stream()
        self._grad_offload_stream = torch.cuda.Stream()

        # CUDA events for synchronization (double-buffered)
        self._load_events = [torch.cuda.Event() for _ in range(2)]
        self._comm_events = [torch.cuda.Event() for _ in range(2)]
        self._compute_events = [torch.cuda.Event() for _ in range(2)]

        self._initialized = True

    def _allocate_dispatch_buffers(self, max_tokens: int):
        """Allocate or reallocate dispatch buffers for the given max_tokens.

        This is called during metadata exchange when max_set_tokens is known.
        If the current buffer size is smaller than needed, reallocate.

        Args:
            max_tokens: Maximum number of tokens for any set
        """
        if max_tokens <= self._dispatch_buffer_size:
            return  # Already large enough

        # Free old buffers if they exist
        if self._dispatch_buffers is not None:
            del self._dispatch_buffers
            del self._dispatch_probs_buffers

        # Allocate new double-buffered dispatch buffers
        self._dispatch_buffers = [
            torch.empty(
                max_tokens, self._hidden_size,
                dtype=self._dispatch_buffer_dtype,
                device=self._dispatch_buffer_device
            )
            for _ in range(2)
        ]
        self._dispatch_probs_buffers = [
            torch.empty(
                max_tokens,
                dtype=self._dispatch_buffer_dtype,
                device=self._dispatch_buffer_device
            )
            for _ in range(2)
        ]
        self._dispatch_buffer_size = max_tokens

    def ensure_dispatch_buffers(self, max_tokens: int) -> bool:
        """Ensure dispatch buffers are large enough.

        Args:
            max_tokens: Required maximum tokens

        Returns:
            True if buffers were reallocated, False if already large enough
        """
        if max_tokens > self._dispatch_buffer_size:
            self._allocate_dispatch_buffers(max_tokens)
            return True
        return False

    def cleanup(self):
        """Release all resources. Should be called after all layers are done."""
        if not self._initialized:
            return

        # Synchronize and clean up CUDA streams
        if self._load_stream is not None:
            self._load_stream.synchronize()
        if self._comm_stream is not None:
            self._comm_stream.synchronize()
        if self._grad_offload_stream is not None:
            self._grad_offload_stream.synchronize()

        # Release buffers
        self._w1_gpu_workspace = None
        self._w2_gpu_workspace = None
        self._dispatch_buffers = None
        self._dispatch_probs_buffers = None
        self._dispatch_buffer_size = 0

        self._initialized = False

    # Properties for buffer access
    @property
    def w1_gpu_workspace(self) -> torch.Tensor:
        return self._w1_gpu_workspace

    @property
    def w2_gpu_workspace(self) -> torch.Tensor:
        return self._w2_gpu_workspace

    @property
    def dispatch_buffers(self) -> Optional[List[torch.Tensor]]:
        return self._dispatch_buffers

    @property
    def dispatch_probs_buffers(self) -> Optional[List[torch.Tensor]]:
        return self._dispatch_probs_buffers

    @property
    def dispatch_buffer_size(self) -> int:
        return self._dispatch_buffer_size

    # Properties for stream access
    @property
    def compute_stream(self) -> torch.cuda.Stream:
        return self._compute_stream

    @property
    def load_stream(self) -> torch.cuda.Stream:
        return self._load_stream

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        return self._comm_stream

    @property
    def grad_offload_stream(self) -> torch.cuda.Stream:
        return self._grad_offload_stream

    # Properties for event access
    @property
    def load_events(self) -> List[torch.cuda.Event]:
        return self._load_events

    @property
    def comm_events(self) -> List[torch.cuda.Event]:
        return self._comm_events

    @property
    def compute_events(self) -> List[torch.cuda.Event]:
        return self._compute_events


# Global singleton instance
_fused_pipeline_buffer_manager = FusedPipelineBufferManager()


def get_fused_pipeline_buffer_manager() -> FusedPipelineBufferManager:
    """Get the global FusedPipelineBufferManager singleton."""
    return _fused_pipeline_buffer_manager


def cleanup_fused_pipeline_buffers():
    """Clean up global fused pipeline buffers.

    This should be called after all FusedDispatcherCacheGroupedMLP layers
    have finished their work.
    """
    _fused_pipeline_buffer_manager.cleanup()