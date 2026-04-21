# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""OffloadOptimizer: Optimizer that offloads expert optimizer states to CPU.

This optimizer is designed for MoE models with expert weights stored on CPU.
It separates:
- GPU parameters (attention, embedding, router): standard AdamW on GPU
- Expert parameters (weights on CPU): optimizer states on CPU, update on CPU

Key features:
1. Expert optimizer states (exp_avg, exp_avg_sq) stored on CPU pinned memory
2. Prefetch optimizer states along with weights during backward
3. Update expert parameters immediately after gradient computation (on CPU)
4. No GPU memory for expert optimizer states

Usage:
    model = Qwen3MoEModel(config, ...)
    optimizer = OffloadOptimizer(model, lr=1e-4)

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()

        # Expert params updated during backward
        # Only need to step GPU params
        optimizer.step()
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.transformer.moe.experts import FusedDispatcherCacheGroupedMLP


class ExpertOptimizerState:
    """Optimizer states for expert parameters stored on CPU.

    For AdamW optimizer, stores:
    - exp_avg: First moment estimate (moving average of gradients)
    - exp_avg_sq: Second moment estimate (moving average of squared gradients)

    States are stored on CPU pinned memory for efficient async transfers.
    GPU workspace is allocated lazily when prefetch is needed.

    NOTE: Optimizer states are stored in FP32 by default for numerical stability,
    even when parameters are in BF16. This follows standard PyTorch AdamW practice.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        param_dtype: torch.dtype,
        state_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
        pin_memory: bool = True,
    ):
        """Initialize optimizer states.

        Args:
            shape: Shape of the parameter tensor
            param_dtype: Data type of the parameter (typically bf16)
            state_dtype: Data type for optimizer states (typically fp32 for stability)
            device: Device for states (default: CPU)
            pin_memory: Whether to pin CPU memory for faster transfers
        """
        # AdamW states on CPU (FP32 for numerical stability)
        self.param_dtype = param_dtype
        self.state_dtype = state_dtype
        self.exp_avg = torch.zeros(shape, dtype=state_dtype, device=device, pin_memory=pin_memory)
        self.exp_avg_sq = torch.zeros(shape, dtype=state_dtype, device=device, pin_memory=pin_memory)

        # GPU workspace for computation (lazy allocated)
        self._exp_avg_gpu: Optional[torch.Tensor] = None
        self._exp_avg_sq_gpu: Optional[torch.Tensor] = None

        # Current step count (for bias correction)
        self.step = 0

    def get_gpu_workspace(
        self,
        device: torch.device,
        num_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get GPU workspace for optimizer state computation.

        Creates workspace on first call, reuses on subsequent calls.
        The workspace is sized for `num_experts` experts and uses state_dtype (FP32).

        Args:
            device: Target GPU device
            num_experts: Number of experts in current set

        Returns:
            Tuple of (exp_avg_gpu, exp_avg_sq_gpu) workspace tensors in FP32
        """
        if self._exp_avg_gpu is None or self._exp_avg_gpu.shape[0] < num_experts:
            # Allocate workspace sized for max experts per set
            # Shape: [num_experts, ...]
            # Use state_dtype (FP32) for numerical stability
            workspace_shape = (num_experts,) + self.exp_avg.shape[1:]
            self._exp_avg_gpu = torch.empty(
                workspace_shape, dtype=self.state_dtype, device=device
            )
            self._exp_avg_sq_gpu = torch.empty(
                workspace_shape, dtype=self.state_dtype, device=device
            )

        return self._exp_avg_gpu[:num_experts], self._exp_avg_sq_gpu[:num_experts]

    def prefetch_to_gpu(
        self,
        expert_ids: List[int],
        device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prefetch optimizer states for specific experts to GPU.

        Args:
            expert_ids: List of expert indices to prefetch
            device: Target GPU device
            stream: Optional CUDA stream for async transfer

        Returns:
            Tuple of (exp_avg_gpu, exp_avg_sq_gpu) on GPU
        """
        num_experts = len(expert_ids)
        exp_avg_gpu, exp_avg_sq_gpu = self.get_gpu_workspace(device, num_experts)

        # Get CPU views for the specific experts
        cpu_exp_avg = self.exp_avg[expert_ids]
        cpu_exp_avg_sq = self.exp_avg_sq[expert_ids]

        if stream is not None:
            with torch.cuda.stream(stream):
                exp_avg_gpu.copy_(cpu_exp_avg, non_blocking=True)
                exp_avg_sq_gpu.copy_(cpu_exp_avg_sq, non_blocking=True)
        else:
            exp_avg_gpu.copy_(cpu_exp_avg, non_blocking=True)
            exp_avg_sq_gpu.copy_(cpu_exp_avg_sq, non_blocking=True)

        return exp_avg_gpu, exp_avg_sq_gpu

    def sync_back_to_cpu(
        self,
        expert_ids: List[int],
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """Sync updated optimizer states back to CPU.

        Args:
            expert_ids: List of expert indices that were updated
            stream: Optional CUDA stream for async transfer
        """
        if self._exp_avg_gpu is None:
            return

        num_experts = len(expert_ids)
        exp_avg_gpu = self._exp_avg_gpu[:num_experts]
        exp_avg_sq_gpu = self._exp_avg_sq_gpu[:num_experts]

        if stream is not None:
            with torch.cuda.stream(stream):
                self.exp_avg[expert_ids].copy_(exp_avg_gpu, non_blocking=True)
                self.exp_avg_sq[expert_ids].copy_(exp_avg_sq_gpu, non_blocking=True)
        else:
            self.exp_avg[expert_ids].copy_(exp_avg_gpu, non_blocking=True)
            self.exp_avg_sq[expert_ids].copy_(exp_avg_sq_gpu, non_blocking=True)


class OffloadOptimizer:
    """Custom optimizer for MoE models with CPU-offloaded expert optimizer states.

    Separates parameters into:
    1. GPU parameters (attention, embedding, router): standard AdamW on GPU
    2. Expert parameters: weights/gradients/optimizer-states all on CPU
       - Prefetch optimizer states with weights during backward
       - Update parameters on CPU immediately after gradient computation

    This reduces GPU memory usage significantly for MoE models where
    expert optimizer states would otherwise occupy O(experts) memory.

    Example:
        >>> model = Qwen3MoEModel(config, pg_collection, expert_sets)
        >>> optimizer = OffloadOptimizer(model, lr=1e-4)
        >>>
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()  # Only steps GPU params, expert params updated in backward
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,  # Conservative learning rate (warmup will gradually increase)
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-7,  # Larger epsilon for bf16 stability (1e-8 is too small)
        weight_decay: float = 0.01,
        dtype: torch.dtype = torch.bfloat16,
        warmup_steps: int = 100,  # Linear warmup for first N steps
    ):
        """Initialize OffloadOptimizer.

        Args:
            model: The MoE model to optimize
            lr: Learning rate (default 1e-3, higher than typical for bf16 precision)
            betas: AdamW beta coefficients
            eps: Epsilon for numerical stability (recommend 1e-7 for bf16, not 1e-8)
            weight_decay: Weight decay coefficient
            dtype: Data type for parameters (optimizer states use fp32 for stability)
            warmup_steps: Number of warmup steps for learning rate (default 100)
        """
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.dtype = dtype
        self.step_count = 0
        self.warmup_steps = warmup_steps

        # Separate parameters
        self.gpu_params: List[nn.Parameter] = []
        self.expert_modules: List[FusedDispatcherCacheGroupedMLP] = []

        # Collect parameters from model
        self._collect_parameters(model)

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

        # Expert optimizer states (CPU pinned memory)
        # Key: module id -> {'weight1': ExpertOptimizerState, 'weight2': ExpertOptimizerState}
        self._expert_states: Dict[int, Dict[str, ExpertOptimizerState]] = {}
        self._device = torch.device("cuda")

        # In-backward optimizer step mode
        self._in_backward_optimizer_step = False
        self._prefetched_opt_states: Dict[Tuple[int, Tuple[int, ...]], Tuple] = {}

        # EP group for gradient allreduce (GPU params are replicated across EP ranks)
        # In EP-only mode (TP=1, PP=1), GPU params need gradient averaging
        self._ep_group = parallel_state.get_expert_model_parallel_group()
        self._ep_size = dist.get_world_size(self._ep_group) if dist.is_initialized() else 1
        self._ep_rank = dist.get_rank(self._ep_group) if dist.is_initialized() else 0

        # Register optimizer with expert modules
        self._register_with_expert_modules()

    def _collect_parameters(self, model: nn.Module):
        """Collect and separate GPU params from expert params."""
        expert_param_ids = set()

        # Find all expert modules
        for name, module in model.named_modules():
            if isinstance(module, FusedDispatcherCacheGroupedMLP):
                self.expert_modules.append(module)
                # Mark expert parameters
                if hasattr(module, 'weight1') and module.weight1 is not None:
                    expert_param_ids.add(id(module.weight1))
                if hasattr(module, 'weight2') and module.weight2 is not None:
                    expert_param_ids.add(id(module.weight2))

        # Collect non-expert parameters (GPU params)
        for param in model.parameters():
            if id(param) not in expert_param_ids and param.requires_grad:
                self.gpu_params.append(param)

        # Log parameter counts
        total_gpu_params = sum(p.numel() for p in self.gpu_params)
        if self.expert_modules:
            expert = self.expert_modules[0]
            expert_params = expert.weight1.numel() + expert.weight2.numel()
            total_expert_params = expert_params * len(self.expert_modules)
        else:
            total_expert_params = 0

        print(
            f"[OffloadOptimizer] GPU params: {len(self.gpu_params)} tensors, "
            f"{total_gpu_params / 1e6:.2f}M params"
        )
        print(
            f"[OffloadOptimizer] Expert modules: {len(self.expert_modules)}, "
            f"~{total_expert_params / 1e6:.2f}M expert params on CPU"
        )

    def _register_with_expert_modules(self):
        """Register this optimizer with expert modules for prefetch callbacks."""
        for module in self.expert_modules:
            # Set callback for prefetching optimizer states
            if hasattr(module, '_set_optimizer_callback'):
                module._set_optimizer_callback(self._prefetch_callback)

    def _prefetch_callback(
        self,
        module: FusedDispatcherCacheGroupedMLP,
        expert_ids: List[int],
        stream: Optional[torch.cuda.Stream],
    ):
        """Callback for prefetching optimizer states during backward.

        Called by the expert module when weights are being prefetched.

        Args:
            module: The expert module requesting prefetch
            expert_ids: Expert IDs being prefetched
            stream: CUDA stream for async transfer
        """
        states = self._get_or_create_expert_state(module)
        if states:
            m1_gpu, v1_gpu = states['weight1'].prefetch_to_gpu(expert_ids, self._device, stream)
            m2_gpu, v2_gpu = states['weight2'].prefetch_to_gpu(expert_ids, self._device, stream)

            # Store prefetched states for in-backward optimizer step
            # Key: (module_id, tuple(expert_ids)) -> (m1, v1, m2, v2)
            if self._in_backward_optimizer_step:
                cache_key = (id(module), tuple(expert_ids))
                self._prefetched_opt_states[cache_key] = (m1_gpu, v1_gpu, m2_gpu, v2_gpu)

    def _get_or_create_expert_state(
        self,
        expert_module: FusedDispatcherCacheGroupedMLP,
    ) -> Dict[str, ExpertOptimizerState]:
        """Get or create optimizer states for an expert module.

        Optimizer states are stored in FP32 for numerical stability,
        even when weights are in BF16. This follows standard PyTorch AdamW practice.
        """
        module_id = id(expert_module)
        if module_id not in self._expert_states:
            states = {}

            if hasattr(expert_module, 'weight1') and expert_module.weight1 is not None:
                states['weight1'] = ExpertOptimizerState(
                    expert_module.weight1.shape,
                    param_dtype=expert_module.weight1.dtype,  # BF16 for weights
                    state_dtype=torch.float32,  # FP32 for optimizer states (numerical stability)
                    device=expert_module.weight1.device,
                    pin_memory=True,
                )

            if hasattr(expert_module, 'weight2') and expert_module.weight2 is not None:
                states['weight2'] = ExpertOptimizerState(
                    expert_module.weight2.shape,
                    param_dtype=expert_module.weight2.dtype,  # BF16 for weights
                    state_dtype=torch.float32,  # FP32 for optimizer states (numerical stability)
                    device=expert_module.weight2.device,
                    pin_memory=True,
                )

            self._expert_states[module_id] = states

        return self._expert_states[module_id]

    def zero_grad(self, set_to_none: bool = False):
        """Zero all gradients."""
        # Zero GPU param gradients
        if self.gpu_optimizer is not None:
            self.gpu_optimizer.zero_grad(set_to_none=set_to_none)

        # Zero expert gradients (they're on CPU)
        for module in self.expert_modules:
            if hasattr(module, 'zero_grad'):
                module.zero_grad(set_to_none=set_to_none)

    def step_experts(self, expert_ids: Optional[List[int]] = None):
        """Update expert parameters on CPU.

        This performs the AdamW update for expert weights:
        1. Get weights and gradients from CPU
        2. Get optimizer states from CPU
        3. Perform AdamW update (vectorized across all experts)
        4. Store updated weights and states back to CPU

        Should be called after backward pass, before optimizer.step().

        Args:
            expert_ids: Optional list of expert indices to update.
                       If None, updates all experts.
        """
        beta1, beta2 = self.betas

        for module in self.expert_modules:
            states = self._get_or_create_expert_state(module)
            if not states:
                continue

            # Get CPU tensor references
            weight1 = module.weight1.data  # CPU tensor [num_experts, hidden, ffn]
            weight2 = module.weight2.data  # CPU tensor [num_experts, ffn, hidden]
            grad1 = module._grad_weight1   # CPU tensor
            grad2 = module._grad_weight2   # CPU tensor

            if grad1 is None or grad2 is None:
                continue

            # Get optimizer states
            state1 = states['weight1']
            state2 = states['weight2']

            # Determine which experts to update
            num_experts = weight1.shape[0]
            if expert_ids is not None:
                idx = torch.tensor(expert_ids, device=weight1.device)
                w1 = weight1[idx]
                w2 = weight2[idx]
                g1 = grad1[idx]
                g2 = grad2[idx]
                m1 = state1.exp_avg[idx]
                v1 = state1.exp_avg_sq[idx]
                m2 = state2.exp_avg[idx]
                v2 = state2.exp_avg_sq[idx]
            else:
                # Update all experts - vectorized operation
                w1 = weight1
                w2 = weight2
                g1 = grad1
                g2 = grad2
                m1 = state1.exp_avg
                v1 = state1.exp_avg_sq
                m2 = state2.exp_avg
                v2 = state2.exp_avg_sq

            # Bias correction
            bias_correction1 = 1 - beta1 ** (self.step_count + 1)
            bias_correction2 = 1 - beta2 ** (self.step_count + 1)
            current_lr = self.get_lr()  # Use warmup learning rate
            step_size = current_lr / bias_correction1

            # Vectorized AdamW update on CPU (much faster than per-expert loop)
            # Use FP32 for numerical stability even with BF16 weights
            with torch.no_grad():
                # Convert weights and gradients to FP32 for stable computation
                w1_fp32 = w1.float()
                w2_fp32 = w2.float()
                g1_fp32 = g1.float()
                g2_fp32 = g2.float()

                # m1, v1, m2, v2 are already FP32 (from ExpertOptimizerState)

                # Update first and second moments (FP32 operations)
                m1.mul_(beta1).add_(g1_fp32, alpha=1 - beta1)
                v1.mul_(beta2).addcmul_(g1_fp32, g1_fp32, value=1 - beta2)

                m2.mul_(beta1).add_(g2_fp32, alpha=1 - beta1)
                v2.mul_(beta2).addcmul_(g2_fp32, g2_fp32, value=1 - beta2)

                # Compute denominator: sqrt(v) / sqrt(bias_correction2) + eps
                # Use larger eps for numerical stability
                eps_fp32 = max(self.eps, 1e-7)
                denom1 = (v1.sqrt() / math.sqrt(bias_correction2)).add_(eps_fp32)
                denom2 = (v2.sqrt() / math.sqrt(bias_correction2)).add_(eps_fp32)

                # Update parameters in FP32
                w1_fp32.addcdiv_(m1, denom1, value=-step_size)
                w2_fp32.addcdiv_(m2, denom2, value=-step_size)

                # Weight decay (AdamW style) in FP32
                if self.weight_decay > 0:
                    w1_fp32.add_(w1_fp32, alpha=-current_lr * self.weight_decay)
                    w2_fp32.add_(w2_fp32, alpha=-current_lr * self.weight_decay)

                # Convert updated weights back to original dtype (BF16)
                w1.copy_(w1_fp32.to(w1.dtype))
                w2.copy_(w2_fp32.to(w2.dtype))

            # Zero the gradients after update
            if hasattr(module, 'zero_grad'):
                module.zero_grad()

        # Increment step count after all expert updates
        self.step_count += 1

    def step(self):
        """Update non-expert parameters on GPU with gradient allreduce.

        In EP-only mode (TP=1, PP=1, EP>1), GPU params (attention, embedding, router)
        are replicated across all EP ranks. Gradients must be averaged before update
        to ensure all ranks converge to the same parameters.
        """
        if self.gpu_optimizer is not None:
            # Allreduce GPU param gradients across EP ranks
            if self._ep_size > 1 and dist.is_initialized():
                for param in self.gpu_params:
                    if param.grad is not None:
                        dist.all_reduce(
                            param.grad,
                            op=dist.ReduceOp.SUM,
                            group=self._ep_group,
                        )
                        # Average gradients
                        param.grad.div_(self._ep_size)

            self.gpu_optimizer.step()

    def step_all(self):
        """Update all parameters.

        If in-backward optimizer step is enabled, expert params are already updated
        during backward, so only GPU params need to be updated here.
        Otherwise, update expert params first, then GPU params.
        """
        # CRITICAL: Global sync point for in-backward optimizer step
        # Ensure all D2H copies from backward complete before next forward iteration
        if self._in_backward_optimizer_step:
            torch.cuda.synchronize()
            print(f"[R{self._ep_rank}] [step_all] torch.cuda.synchronize() completed - CPU weights updated")

        if not self._in_backward_optimizer_step:
            self.step_experts()
        self.step()

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict for checkpointing."""
        state = {
            'step_count': self.step_count,
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
        }

        # GPU optimizer state
        if self.gpu_optimizer is not None:
            state['gpu_optimizer'] = self.gpu_optimizer.state_dict()

        # Expert optimizer states
        expert_states = {}
        for module_id, states in self._expert_states.items():
            expert_states[str(module_id)] = {
                'weight1_exp_avg': states['weight1'].exp_avg,
                'weight1_exp_avg_sq': states['weight1'].exp_avg_sq,
                'weight1_step': states['weight1'].step,
                'weight2_exp_avg': states['weight2'].exp_avg,
                'weight2_exp_avg_sq': states['weight2'].exp_avg_sq,
                'weight2_step': states['weight2'].step,
            }
        state['expert_states'] = expert_states

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict from checkpoint."""
        self.step_count = state_dict['step_count']

        # GPU optimizer state
        if self.gpu_optimizer is not None and 'gpu_optimizer' in state_dict:
            self.gpu_optimizer.load_state_dict(state_dict['gpu_optimizer'])

        # Expert optimizer states are loaded lazily when modules are created
        # Store the state dict for later loading
        self._pending_expert_states = state_dict.get('expert_states', {})

    def get_lr(self) -> float:
        """Get current learning rate with warmup support.

        During warmup phase, learning rate linearly increases from 0 to self.lr.
        """
        if self.step_count < self.warmup_steps:
            # Linear warmup: 0 -> base_lr
            return self.lr * (self.step_count + 1) / self.warmup_steps
        return self.lr

    def set_lr(self, lr: float):
        """Set learning rate."""
        self.lr = lr
        if self.gpu_optimizer is not None:
            for param_group in self.gpu_optimizer.param_groups:
                param_group['lr'] = lr

    def enable_in_backward_optimizer_step(self, enabled: bool = True):
        """Enable/disable in-backward optimizer step mode.

        When enabled, AdamW update is performed on GPU immediately after
        gradient computation during backward, instead of waiting for
        step_experts() on CPU. This significantly reduces optimizer time.

        Args:
            enabled: Whether to enable in-backward optimizer step
        """
        self._in_backward_optimizer_step = enabled
        for module in self.expert_modules:
            if hasattr(module, '_set_in_backward_optimizer_step'):
                module._set_in_backward_optimizer_step(
                    enabled,
                    self._in_backward_step_callback if enabled else None
                )
        if enabled:
            print("[OffloadOptimizer] In-backward optimizer step ENABLED")
        else:
            print("[OffloadOptimizer] In-backward optimizer step DISABLED")

    def _in_backward_step_callback(
        self,
        module: FusedDispatcherCacheGroupedMLP,
        expert_ids: List[int],
        grad_w1: torch.Tensor,
        grad_w2: torch.Tensor,
        w1_gpu: torch.Tensor,
        w2_gpu: torch.Tensor,
        stream: torch.cuda.Stream,
    ):
        """Callback for performing AdamW update on GPU during backward.

        This is called immediately after gradient computation, with gradients
        and optimizer states already on GPU. Performs AdamW update and syncs
        updated weights/states back to CPU.

        Args:
            module: Expert module
            expert_ids: List of expert indices being updated
            grad_w1, grad_w2: Gradient tensors on GPU
            w1_gpu, w2_gpu: Weight tensors on GPU
            stream: CUDA stream for async CPU sync
        """
        # Get prefetched optimizer states
        cache_key = (id(module), tuple(expert_ids))
        opt_states = self._prefetched_opt_states.pop(cache_key, None)

        # DEBUG: Print callback info with layer_number
        layer_number = getattr(module, 'layer_number', 'unknown')
        # Save CPU tensor BEFORE update (clone to get actual values)
        cpu_w1_before = module.weight1.data[expert_ids[0]].clone()
        cpu_w1_data_ptr = module.weight1.data[expert_ids[0]].data_ptr()
        print(f"[R{self._ep_rank}] _in_backward_step_callback: module={id(module)}, layer={layer_number}, expert_ids={expert_ids}, step_count={self.step_count}")
        print(f"[R{self._ep_rank}]   cpu_w1.data_ptr()={cpu_w1_data_ptr}, mean={cpu_w1_before.mean().item():.6e}")
        print(f"[R{self._ep_rank}]   w1_gpu.mean={w1_gpu.mean().item():.6e}")

        if opt_states is None:
            # States not prefetched, create them now
            states = self._get_or_create_expert_state(module)
            if states is None:
                return
            m1_gpu = states['weight1']._exp_avg_gpu[:len(expert_ids)] if states['weight1']._exp_avg_gpu is not None else None
            v1_gpu = states['weight1']._exp_avg_sq_gpu[:len(expert_ids)] if states['weight1']._exp_avg_sq_gpu is not None else None
            m2_gpu = states['weight2']._exp_avg_gpu[:len(expert_ids)] if states['weight2']._exp_avg_gpu is not None else None
            v2_gpu = states['weight2']._exp_avg_sq_gpu[:len(expert_ids)] if states['weight2']._exp_avg_sq_gpu is not None else None
            if m1_gpu is None:
                # Allocate GPU workspace
                m1_gpu, v1_gpu = states['weight1'].get_gpu_workspace(self._device, len(expert_ids))
                m2_gpu, v2_gpu = states['weight2'].get_gpu_workspace(self._device, len(expert_ids))
                # Copy from CPU
                m1_gpu.copy_(states['weight1'].exp_avg[expert_ids])
                v1_gpu.copy_(states['weight1'].exp_avg_sq[expert_ids])
                m2_gpu.copy_(states['weight2'].exp_avg[expert_ids])
                v2_gpu.copy_(states['weight2'].exp_avg_sq[expert_ids])
        else:
            m1_gpu, v1_gpu, m2_gpu, v2_gpu = opt_states

        # Gradient norm logging
        with torch.no_grad():
            grad_w1_has_nan = torch.isnan(grad_w1).any().item() or torch.isinf(grad_w1).any().item()
            grad_w2_has_nan = torch.isnan(grad_w2).any().item() or torch.isinf(grad_w2).any().item()

            grad_norm = torch.sqrt(grad_w1.norm() ** 2 + grad_w2.norm() ** 2)
            layer_num = getattr(module, 'layer_number', '?')
            print(f"[R{self._ep_rank}] layer={layer_num}, set_idx={self.step_count % 4}, grad_norm={grad_norm.item():.6e}")

        beta1, beta2 = self.betas
        bias_correction1 = 1 - beta1 ** (self.step_count + 1)
        bias_correction2 = 1 - beta2 ** (self.step_count + 1)
        current_lr = self.get_lr()  # Use warmup learning rate
        step_size = current_lr / bias_correction1

        # Log warmup progress
        if self.step_count < self.warmup_steps:
            print(f"[R{self._ep_rank}] Warmup step {self.step_count}/{self.warmup_steps}, lr={current_lr:.6e}")

        # DEBUG: Check for nan in gradients and weights
        grad_w1_has_nan = torch.isnan(grad_w1).any().item()
        grad_w2_has_nan = torch.isnan(grad_w2).any().item()
        w1_gpu_has_nan = torch.isnan(w1_gpu).any().item()
        w2_gpu_has_nan = torch.isnan(w2_gpu).any().item()

        if grad_w1_has_nan or grad_w2_has_nan or w1_gpu_has_nan or w2_gpu_has_nan:
            print(f"[WARNING] NaN detected! Skipping AdamW update for this expert set")
            print(f"[WARNING] layer_number={layer_number}, expert_ids={expert_ids}, step_count={self.step_count}")
            print(f"[WARNING] grad_w1: mean={grad_w1.mean().item() if not grad_w1_has_nan else 'nan'}, "
                  f"std={grad_w1.std().item() if not grad_w1_has_nan else 'nan'}, "
                  f"abs_max={grad_w1.abs().max().item() if not grad_w1_has_nan else 'nan'}")
            print(f"[WARNING] grad_w2: mean={grad_w2.mean().item() if not grad_w2_has_nan else 'nan'}, "
                  f"std={grad_w2.std().item() if not grad_w2_has_nan else 'nan'}, "
                  f"abs_max={grad_w2.abs().max().item() if not grad_w2_has_nan else 'nan'}")
            # Skip the AdamW update when NaN is detected to prevent propagation
            # Still increment step count and record sync event to maintain consistency
            self.step_count += 1
            # Record event on current stream (not sync stream since no sync happened)
            sync_done_event = torch.cuda.Event()
            sync_done_event.record(torch.cuda.current_stream())
            module._last_sync_done_event = sync_done_event
            return

        # AdamW update on GPU (vectorized, very fast)
        # Use FP32 for numerical stability even with BF16 weights
        with torch.no_grad():
            # Convert gradients to FP32 for stable moment updates
            grad_w1_fp32 = grad_w1.float()
            grad_w2_fp32 = grad_w2.float()

            # Convert weights to FP32 for stable weight updates
            w1_gpu_fp32 = w1_gpu.float()
            w2_gpu_fp32 = w2_gpu.float()

            # Update first and second moments (FP32 operations)
            # m1_gpu, v1_gpu, m2_gpu, v2_gpu are already FP32 (from ExpertOptimizerState)
            m1_gpu.mul_(beta1).add_(grad_w1_fp32, alpha=1 - beta1)
            v1_gpu.mul_(beta2).addcmul_(grad_w1_fp32, grad_w1_fp32, value=1 - beta2)
            m2_gpu.mul_(beta1).add_(grad_w2_fp32, alpha=1 - beta1)
            v2_gpu.mul_(beta2).addcmul_(grad_w2_fp32, grad_w2_fp32, value=1 - beta2)

            # Compute denominator and update weights (FP32 operations)
            # Use larger eps for numerical stability (1e-7 for bf16-compatible)
            eps_fp32 = max(self.eps, 1e-7)  # Ensure eps is large enough for stable division
            denom1 = (v1_gpu.sqrt() / math.sqrt(bias_correction2)).add_(eps_fp32)
            denom2 = (v2_gpu.sqrt() / math.sqrt(bias_correction2)).add_(eps_fp32)
            w1_gpu_fp32.addcdiv_(m1_gpu, denom1, value=-step_size)
            w2_gpu_fp32.addcdiv_(m2_gpu, denom2, value=-step_size)

            # Weight decay (AdamW style) in FP32
            if self.weight_decay > 0:
                w1_gpu_fp32.add_(w1_gpu_fp32, alpha=-current_lr * self.weight_decay)
                w2_gpu_fp32.add_(w2_gpu_fp32, alpha=-current_lr * self.weight_decay)

            # Convert updated weights back to BF16 for storage
            w1_gpu.copy_(w1_gpu_fp32.bfloat16())
            w2_gpu.copy_(w2_gpu_fp32.bfloat16())

        # CRITICAL: Ensure AdamW update completes before sync
        # The AdamW update runs on the default stream, but we sync to CPU
        # on a separate stream. Use event-based synchronization for correctness.
        # Record an event on the default stream AFTER AdamW update completes.
        adamw_done_event = torch.cuda.Event()
        adamw_done_event.record(torch.cuda.current_stream())

        # D2H copy on dedicated stream
        with torch.cuda.stream(stream):
            # Wait for AdamW update to complete on default stream
            stream.wait_event(adamw_done_event)

            # Copy weights to CPU (blocking to ensure correctness)
            for i, exp_id in enumerate(expert_ids):
                module.weight1.data[exp_id].copy_(w1_gpu[i], non_blocking=False)
                module.weight2.data[exp_id].copy_(w2_gpu[i], non_blocking=False)

            # Sync optimizer states
            states = self._expert_states.get(id(module))
            if states is not None:
                for i, exp_id in enumerate(expert_ids):
                    states['weight1'].exp_avg[exp_id].copy_(m1_gpu[i], non_blocking=False)
                    states['weight1'].exp_avg_sq[exp_id].copy_(v1_gpu[i], non_blocking=False)
                    states['weight2'].exp_avg[exp_id].copy_(m2_gpu[i], non_blocking=False)
                    states['weight2'].exp_avg_sq[exp_id].copy_(v2_gpu[i], non_blocking=False)

        # CRITICAL: Wait for stream to complete before returning
        # This ensures CPU weights are updated before any subsequent operations
        stream.synchronize()

        # Increment step count after update
        self.step_count += 1

        # DEBUG: Verify D2H copy using max absolute difference (more sensitive than mean)
        cpu_w1_after = module.weight1.data[expert_ids[0]]
        w1_diff_max = torch.abs(cpu_w1_after - cpu_w1_before).max().item()
        w1_diff_mean = torch.abs(cpu_w1_after - cpu_w1_before).mean().item()
        print(f"[R{self._ep_rank}] D2H copy verification for module={id(module)}, step_count={self.step_count}")
        print(f"[R{self._ep_rank}]   cpu_w1_after.data_ptr()={cpu_w1_after.data_ptr()}, mean={cpu_w1_after.mean().item():.6e}")
        print(f"[R{self._ep_rank}]   max_abs_diff={w1_diff_max:.6e}, mean_abs_diff={w1_diff_mean:.6e}")
        if w1_diff_max < 1e-10:
            print(f"[R{self._ep_rank}] [ERROR] D2H copy FAILED! CPU tensor unchanged (max_diff={w1_diff_max:.6e})")
        else:
            print(f"[R{self._ep_rank}] [OK] D2H copy succeeded, CPU weights updated")

    def get_prefetched_optimizer_states(
        self,
        module: FusedDispatcherCacheGroupedMLP,
        expert_ids: List[int],
    ):
        """Get prefetched optimizer states for in-backward step.

        Args:
            module: Expert module
            expert_ids: List of expert indices

        Returns:
            Tuple of (m1_gpu, v1_gpu, m2_gpu, v2_gpu) or None
        """
        cache_key = (id(module), tuple(expert_ids))
        states = self._prefetched_opt_states.get(cache_key)
        if states is not None:
            # Remove from cache after use
            del self._prefetched_opt_states[cache_key]
        return states