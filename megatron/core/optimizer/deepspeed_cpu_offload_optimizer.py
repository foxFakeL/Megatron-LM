# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""DeepSpeed CPU Offload Optimizer for MoE models.

This optimizer uses DeepSpeed's CPUAdam for expert weight updates on CPU,
leveraging the existing CPU weight/gradient management in FusedDispatcherCacheGroupedMLP.

Key features:
1. Expert weights stored in CPU shared memory (already implemented)
2. Expert gradients offloaded to CPU during backward (already implemented)
3. DeepSpeed CPUAdam for efficient CPU-based optimizer updates (SIMD optimized)
4. GPU params use standard AdamW

Usage:
    model = Qwen3MoEModel(config, pg_collection, expert_sets)
    optimizer = DeepSpeedCPUOffloadOptimizer(model, lr=1e-4)

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()  # Expert params updated on CPU via DeepSpeed
"""

from typing import Dict, List, Optional, Tuple, Any
import threading

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda import nvtx

from megatron.core import parallel_state
from megatron.core.transformer.moe.experts import FusedDispatcherCacheGroupedMLP

try:
    from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
    HAVE_DEEPSPEED = True
except ImportError:
    HAVE_DEEPSPEED = False
    # Fallback to PyTorch AdamW on CPU
    DeepSpeedCPUAdam = torch.optim.AdamW


class DeepSpeedCPUOffloadOptimizer:
    """Optimizer for MoE models using DeepSpeed CPUAdam for expert weights.

    Architecture:
    - GPU params (attention, embedding, router): torch.optim.AdamW on GPU
    - Expert params: DeepSpeed CPUAdam on CPU

    Expert weights are stored in CPU shared memory and loaded to GPU on-demand.
    Expert gradients are automatically offloaded to CPU during backward.
    DeepSpeed CPUAdam performs optimizer updates directly on CPU.

    Example:
        >>> model = Qwen3MoEModel(config, pg_collection, expert_sets)
        >>> optimizer = DeepSpeedCPUOffloadOptimizer(model, lr=1e-4)
        >>>
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        clip_grad: float = 1.0,
    ):
        """Initialize DeepSpeedCPUOffloadOptimizer.

        Args:
            model: The MoE model to optimize
            lr: Learning rate
            betas: Adam beta coefficients
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
            warmup_steps: Number of warmup steps for learning rate
            clip_grad: Maximum gradient norm for clipping (0 to disable)
        """
        self.base_lr = lr  # 保存原始目标学习率，用于 warmup 计算
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.warmup_steps = warmup_steps
        self.clip_grad = clip_grad

        # Separate parameters
        self.gpu_params: List[nn.Parameter] = []
        self.expert_modules: List[FusedDispatcherCacheGroupedMLP] = []
        self._collect_parameters(model)

        # CUDA Event for CPU-GPU synchronization
        # After optimizer.step(), this Event signals that CPU weights are updated
        # The next iteration's prefetch will wait on this Event
        # Initialize it immediately so it can be passed to expert modules before training starts
        self._cpu_update_done_event = torch.cuda.Event()

        # Get shared gradient offload stream from expert modules
        # Gradients are offloaded asynchronously on this stream
        # Must sync before reading CPU gradient buffers in optimizer.step()
        if self.expert_modules:
            self._grad_offload_stream = self.expert_modules[0]._grad_offload_stream
        else:
            self._grad_offload_stream = None

        # EP group for GPU param gradient allreduce
        # IMPORTANT: Get EP rank BEFORE creating CPU optimizer - only rank 0 creates it
        self._ep_group = parallel_state.get_expert_model_parallel_group()
        self._ep_size = dist.get_world_size(self._ep_group) if dist.is_initialized() else 1
        self._ep_rank = dist.get_rank(self._ep_group) if dist.is_initialized() else 0

        # Standard AdamW for GPU parameters
        if self.gpu_params:
            self.gpu_optimizer = torch.optim.AdamW(
                self.gpu_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        else:
            self.gpu_optimizer = None

        # DeepSpeed CPUAdam for expert parameters - ONLY rank 0 creates it
        # Expert weights are in shared CPU memory, gradients are accumulated there
        # Only rank 0 needs to hold the optimizer and perform updates
        # Other ranks will see updated weights via shared memory
        expert_params = self._get_expert_params()
        if expert_params and self._ep_rank == 0:
            if HAVE_DEEPSPEED:
                self.cpu_optimizer = DeepSpeedCPUAdam(
                    expert_params,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                    adamw_mode=True,  # Use AdamW (decoupled weight decay)
                )
            else:
                # Fallback to PyTorch AdamW on CPU
                self.cpu_optimizer = torch.optim.AdamW(
                    expert_params,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
        else:
            self.cpu_optimizer = None

        # EP group for GPU param gradient allreduce
        self._ep_group = parallel_state.get_expert_model_parallel_group()
        self._ep_size = dist.get_world_size(self._ep_group) if dist.is_initialized() else 1
        self._ep_rank = dist.get_rank(self._ep_group) if dist.is_initialized() else 0

        # Threading support for async CPU-GPU optimizer overlap
        # CPU optimizer runs in a background thread while GPU optimizer runs on main thread
        self._cpu_step_done_event = threading.Event()
        self._cpu_step_lock = threading.Lock()
        self._cpu_optimizer_thread = None

        # Pass CPU update event to expert modules
        for module in self.expert_modules:
            module.set_cpu_update_event(self._cpu_update_done_event)

    def _collect_parameters(self, model: nn.Module):
        """Collect and separate GPU params from expert params."""
        expert_param_ids = set()

        # Find all expert modules and router modules
        self.router_modules: List[Any] = []
        for name, module in model.named_modules():
            if isinstance(module, FusedDispatcherCacheGroupedMLP):
                self.expert_modules.append(module)
                # Mark expert parameters
                if hasattr(module, 'weight1') and module.weight1 is not None:
                    expert_param_ids.add(id(module.weight1))
                if hasattr(module, 'weight2') and module.weight2 is not None:
                    expert_param_ids.add(id(module.weight2))
            # Collect routers for debugging
            if hasattr(module, 'router') and hasattr(module.router, 'weight'):
                self.router_modules.append(module.router)

        # Collect non-expert parameters (GPU params)
        for param in model.parameters():
            if id(param) not in expert_param_ids and param.requires_grad:
                self.gpu_params.append(param)

    def _get_expert_params(self) -> List[nn.Parameter]:
        """Get expert parameters (CPU tensors)."""
        params = []
        for module in self.expert_modules:
            if hasattr(module, 'weight1') and module.weight1 is not None:
                params.append(module.weight1)
            if hasattr(module, 'weight2') and module.weight2 is not None:
                params.append(module.weight2)
        return params

    def _attach_expert_gradients(self):
        """Attach CPU gradient buffers to expert parameters.

        Expert gradients are stored in _grad_weight1 and _grad_weight2 (CPU tensors).
        DeepSpeed CPUAdam expects param.grad to be set.
        """
        for module in self.expert_modules:
            # _grad_weight1 and _grad_weight2 are CPU tensors
            # They need to be attached as .grad for the optimizer
            if hasattr(module, '_grad_weight1') and module._grad_weight1 is not None:
                module.weight1.grad = module._grad_weight1
            if hasattr(module, '_grad_weight2') and module._grad_weight2 is not None:
                module.weight2.grad = module._grad_weight2

    def sync_gradient_offload(self):
        """Synchronize gradient offload stream before reading CPU gradients.

        CRITICAL: Must be called before _attach_expert_gradients() to ensure
        all D2H gradient transfers are complete.

        Gradients are offloaded asynchronously on _grad_offload_stream.
        Without this sync, optimizer may read incomplete/partial gradients.
        """
        if self._grad_offload_stream is not None:
            self._grad_offload_stream.synchronize()

    def record_cpu_update_done(self):
        """Record Event signaling CPU optimizer update completion.

        After optimizer.step() on CPU, record an Event on default stream.
        The next iteration's prefetch on _load_stream will wait on this Event,
        ensuring CPU weights are updated before H2D copy starts.

        Implementation:
        - Touch CPU memory to ensure cache coherence
        - Record Event on default stream
        - This Event becomes a barrier that prefetch stream can wait on
        """
        if self.cpu_optimizer is not None and self.expert_modules:
            # Touch ALL expert modules' CPU memory to ensure complete cache coherence
            # CRITICAL: Must flush ALL cache lines, not just the first element!
            # DMA engine reads from RAM, not CPU cache. Without flushing all caches,
            # DMA may read stale weight values from previous iteration, causing NaN.
            # bfloat16: 2 bytes/elem, cache line 64 bytes → 32 elems/line
            # Using [::32] stride ensures we touch every cache line
            for module in self.expert_modules:
                if module.weight1 is not None:
                    _ = module.weight1.data.reshape(-1)[::32].sum().item()
                if module.weight2 is not None:
                    _ = module.weight2.data.reshape(-1)[::32].sum().item()

            # Record Event on the current (main) stream
            self._cpu_update_done_event.record(torch.cuda.current_stream())

    def get_cpu_update_event(self) -> torch.cuda.Event:
        """Get the CPU update completion Event for prefetch synchronization.

        Returns:
            CUDA Event that signals CPU weight update completion
        """
        return self._cpu_update_done_event

    def zero_grad(self, set_to_none: bool = False):
        """Zero all gradients.

        CRITICAL: Must sync gradient offload stream before zeroing CPU buffers.
        Gradients from previous iteration's backward are offloaded asynchronously
        on _grad_offload_stream. If we zero the CPU buffers before the offload
        completes, we corrupt the gradient data.

        Args:
            set_to_none: Whether to set gradients to None instead of zero
        """
        nvtx.range_push("DeepSpeedCPUOptimizer::zero_grad")

        # CRITICAL: Sync gradient offload stream BEFORE zeroing CPU buffers
        # This ensures previous iteration's D2H transfers are complete
        nvtx.range_push("sync_gradient_offload")
        self.sync_gradient_offload()
        nvtx.range_pop()

        # Zero GPU param gradients
        if self.gpu_optimizer is not None:
            self.gpu_optimizer.zero_grad(set_to_none=set_to_none)

        # Zero expert gradients (they're on CPU, now safe to zero)
        for module in self.expert_modules:
            if hasattr(module, 'zero_grad'):
                module.zero_grad(set_to_none=set_to_none)

        nvtx.range_pop()  # End DeepSpeedCPUOptimizer::zero_grad

    def _cpu_optimizer_step_thread(self):
        """Run CPU optimizer step in a background thread.

        This method is called in a separate thread to allow CPU optimizer
        to run concurrently with GPU optimizer, achieving CPU-GPU overlap.

        The thread:
        1. Acquires lock to protect optimizer state
        2. Runs cpu_optimizer.step()
        3. Touches CPU memory to ensure cache coherence
        4. Signals completion via _cpu_step_done_event
        """
        if self.cpu_optimizer is not None:
            nvtx.range_push("cpu_optimizer_step_thread")
            with self._cpu_step_lock:
                self.cpu_optimizer.step()
                # Touch CPU memory to ensure cache coherence
                # This ensures updated weights are visible to DMA engine for next H2D copy
                for module in self.expert_modules:
                    if module.weight1 is not None:
                        _ = module.weight1.data[0, 0, 0].item()
                    if module.weight2 is not None:
                        _ = module.weight2.data[0, 0, 0].item()
            nvtx.range_pop()
        # Signal completion to main thread
        self._cpu_step_done_event.set()

    def step(self):
        """Update parameters with CPU-GPU optimizer overlap.

        Flow:
        1. Sync gradient offload stream (wait for D2H transfers)
        2. Apply warmup learning rate
        3. Attach expert gradients to CPU parameters
        4. Gradient clipping
        5. Start CPU optimizer step in background thread (async)
        6. Allreduce GPU gradients + Step GPU optimizer (concurrent with CPU)
        7. Wait for CPU optimizer thread to complete
        8. Barrier across EP ranks
        9. Record Event for next iteration's prefetch

        CPU optimizer runs in a background thread while GPU optimizer runs on
        the main thread, achieving true CPU-GPU overlap for better performance.
        """
        nvtx.range_push("DeepSpeedCPUOptimizer::step")

        # CRITICAL: Sync gradient offload stream BEFORE attaching gradients
        # Gradients are offloaded asynchronously on _grad_offload_stream
        # Without this sync, we may read incomplete/partial gradients
        nvtx.range_push("sync_gradient_offload")
        self.sync_gradient_offload()
        nvtx.range_pop()

        # Apply warmup learning rate before optimizer step
        current_lr = self.get_lr()
        self.set_lr(current_lr)

        # Attach expert gradients (now safe - D2H transfers complete)
        self._attach_expert_gradients()

        # Gradient clipping
        if self.clip_grad > 0:
            nvtx.range_push("gradient_clipping")
            # Clip GPU param gradients
            gpu_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.gpu_params, self.clip_grad
            )

            # Clip expert gradients (CPU tensors)
            total_expert_grad_norm = 0.0
            for module in self.expert_modules:
                if hasattr(module, '_grad_weight1') and module._grad_weight1 is not None:
                    grad_norm = module._grad_weight1.norm().item()
                    total_expert_grad_norm += grad_norm ** 2
                if hasattr(module, '_grad_weight2') and module._grad_weight2 is not None:
                    grad_norm = module._grad_weight2.norm().item()
                    total_expert_grad_norm += grad_norm ** 2

            total_expert_grad_norm = total_expert_grad_norm ** 0.5

            if total_expert_grad_norm > self.clip_grad:
                scale = self.clip_grad / (total_expert_grad_norm + 1e-8)
                for module in self.expert_modules:
                    if hasattr(module, '_grad_weight1') and module._grad_weight1 is not None:
                        module._grad_weight1.mul_(scale)
                    if hasattr(module, '_grad_weight2') and module._grad_weight2 is not None:
                        module._grad_weight2.mul_(scale)
                # Re-attach clipped gradients
                self._attach_expert_gradients()

            nvtx.range_pop()

        # === CPU-GPU Optimizer Parallel Execution ===
        # CPU optimizer runs in background thread while GPU optimizer runs on main thread
        # This achieves true CPU-GPU overlap for better performance

        # Start CPU optimizer step in background thread (async with GPU optimizer)
        if self.cpu_optimizer is not None:
            self._cpu_step_done_event.clear()
            self._cpu_optimizer_thread = threading.Thread(
                target=self._cpu_optimizer_step_thread,
                daemon=True
            )
            self._cpu_optimizer_thread.start()

        # GPU optimizer step (runs concurrently with CPU optimizer)
        if self.gpu_optimizer is not None:
            nvtx.range_push("gpu_optimizer_step")
            # Allreduce GPU param gradients across EP ranks
            # In EP-only mode (TP=1, PP=1), GPU params are replicated across EP ranks
            if self._ep_size > 1 and dist.is_initialized():
                for param in self.gpu_params:
                    if param.grad is not None:
                        dist.all_reduce(
                            param.grad,
                            op=dist.ReduceOp.SUM,
                            group=self._ep_group,
                        )
                        param.grad.div_(self._ep_size)

            self.gpu_optimizer.step()

            nvtx.range_pop()

        # Wait for CPU optimizer thread to complete
        if self.cpu_optimizer is not None:
            self._cpu_step_done_event.wait()  # Block until CPU step finishes

        # Barrier: ensure rank 0 finishes CPU weight update AND cache flush before all ranks proceed
        # Shared memory weights are updated by rank 0, all ranks need to wait
        if self._ep_size > 1 and dist.is_initialized():
            dist.barrier(group=self._ep_group)
        # CRITICAL: Record Event signaling CPU update completion
        # This ensures next iteration's prefetch waits for CPU weights to be ready
        nvtx.range_push("record_cpu_update_event")
        self.record_cpu_update_done()
        nvtx.range_pop()

        self.step_count += 1

        nvtx.range_pop()  # End DeepSpeedCPUOptimizer::step

    def get_lr(self) -> float:
        """Get current learning rate with warmup support.

        During warmup phase, learning rate linearly increases from 0 to base_lr.

        Returns:
            Current learning rate
        """
        if self.step_count < self.warmup_steps:
            return self.base_lr * (self.step_count + 1) / self.warmup_steps
        return self.base_lr

    def set_lr(self, lr: float):
        """Set learning rate.

        Args:
            lr: New learning rate
        """
        self.lr = lr
        if self.gpu_optimizer is not None:
            for param_group in self.gpu_optimizer.param_groups:
                param_group['lr'] = lr
        if self.cpu_optimizer is not None:
            for param_group in self.cpu_optimizer.param_groups:
                param_group['lr'] = lr

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict for checkpointing.

        Returns:
            Dictionary containing optimizer state
        """
        state = {
            'step_count': self.step_count,
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
        }

        if self.gpu_optimizer is not None:
            state['gpu_optimizer'] = self.gpu_optimizer.state_dict()

        if self.cpu_optimizer is not None:
            state['cpu_optimizer'] = self.cpu_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict from checkpoint.

        Args:
            state_dict: Dictionary containing optimizer state
        """
        self.step_count = state_dict['step_count']
        self.lr = state_dict.get('lr', self.lr)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)

        if self.gpu_optimizer is not None and 'gpu_optimizer' in state_dict:
            self.gpu_optimizer.load_state_dict(state_dict['gpu_optimizer'])

        if self.cpu_optimizer is not None and 'cpu_optimizer' in state_dict:
            self.cpu_optimizer.load_state_dict(state_dict['cpu_optimizer'])