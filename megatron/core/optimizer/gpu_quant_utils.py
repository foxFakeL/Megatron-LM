"""GPU-side quantization utilities for MoE expert weights.

This module provides GPU kernels for:
1. Dequantization: INT4/INT8 → BF16
2. LSQ delta/z gradient computation and update

CPU sends quantized weights + delta/z to GPU.
GPU dequantizes to BF16 and uses for computation.
GPU computes LSQ delta/z updates and returns new values to CPU.
"""

import torch
from typing import Optional, Tuple


def dequantize_int8(quant_weight: torch.Tensor, delta: torch.Tensor,
                    z: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize INT8 weights to BF16 on GPU.

    Args:
        quant_weight: uint8 tensor containing quantized weights
        delta: float32 per-group scale parameter
        z: float32 per-group zero point parameter
        group_size: Number of elements per quantization group

    Returns:
        BF16 dequantized weight tensor
    """
    # Ensure tensors are on GPU
    device = quant_weight.device
    quant_weight = quant_weight.to(device)
    delta = delta.to(device)
    z = z.to(device)

    num_elements = quant_weight.numel()
    num_groups = num_elements // group_size

    # Expand delta and z to match weight shape
    # delta[z] are per-group, need to expand to per-element
    delta_expanded = delta.repeat_interleave(group_size)
    z_expanded = z.repeat_interleave(group_size)

    # Dequantize: weight = quant_weight * delta + z
    # quant_weight is uint8, need to convert to float first
    weight_float = quant_weight.float()
    weight_bf16 = weight_float * delta_expanded + z_expanded

    return weight_bf16.to(torch.bfloat16)


def dequantize_int4(quant_weight: torch.Tensor, delta: torch.Tensor,
                    z: torch.Tensor, group_size: int,
                    original_size: Optional[int] = None) -> torch.Tensor:
    """Dequantize INT4 packed weights to BF16 on GPU.

    INT4 packing: 2 elements per byte (high nibble = first, low nibble = second)

    Args:
        quant_weight: uint8 tensor with packed INT4 weights
        delta: float32 per-group scale parameter
        z: float32 per-group zero point parameter
        group_size: Number of elements per quantization group
        original_size: Original number of elements (if odd, last element needs special handling)

    Returns:
        BF16 dequantized weight tensor
    """
    device = quant_weight.device
    quant_weight = quant_weight.to(device)
    delta = delta.to(device)
    z = z.to(device)

    # Unpack INT4: each byte contains 2 elements
    # High nibble (bits 4-7) = first element
    # Low nibble (bits 0-3) = second element
    high_nibbles = (quant_weight >> 4) & 0x0F
    low_nibbles = quant_weight & 0x0F

    # Interleave to get unpacked tensor
    # Shape: [num_bytes * 2]
    unpacked = torch.empty(quant_weight.numel() * 2, dtype=torch.float32, device=device)
    unpacked[0::2] = high_nibbles.float()
    unpacked[1::2] = low_nibbles.float()

    if original_size is not None:
        unpacked = unpacked[:original_size]

    num_elements = unpacked.numel()
    num_groups = num_elements // group_size

    # Expand delta and z
    delta_expanded = delta.repeat_interleave(group_size)
    z_expanded = z.repeat_interleave(group_size)

    # Dequantize
    weight_bf16 = unpacked * delta_expanded + z_expanded

    return weight_bf16.to(torch.bfloat16)


def compute_lsq_gradients(grad_output: torch.Tensor,
                          quant_weight: torch.Tensor,
                          delta: torch.Tensor,
                          z: torch.Tensor,
                          group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute LSQ gradients for delta and z on GPU.

    LSQ gradient formulas:
    - delta_grad = sum(grad_output * (weight_quant - z/delta)) per group
    - z_grad = sum(grad_output) per group

    Args:
        grad_output: Gradient from upstream (BF16)
        quant_weight: Quantized weight (uint8 for INT8, float32 for unpacked INT4)
        delta: Current delta (float32)
        z: Current z (float32)
        group_size: Number of elements per group

    Returns:
        Tuple of (delta_grad, z_grad) per group
    """
    device = grad_output.device
    grad_output = grad_output.to(device).float()
    quant_weight = quant_weight.to(device).float() if quant_weight.dtype == torch.uint8 else quant_weight.to(device)

    num_elements = grad_output.numel()
    num_groups = num_elements // group_size

    # Reshape to groups
    grad_groups = grad_output.view(num_groups, group_size)
    quant_groups = quant_weight.view(num_groups, group_size)

    # Compute delta gradient
    # delta_grad = sum(grad * (quant - z/delta))
    # Simplified: sum(grad * quant) / group_size - sum(grad) * z / delta / group_size
    delta_grad = (grad_groups * quant_groups).sum(dim=1)

    # Compute z gradient
    # z_grad = sum(grad)
    z_grad = grad_groups.sum(dim=1)

    return delta_grad, z_grad


def update_lsq_params(delta: torch.Tensor, z: torch.Tensor,
                      delta_grad: torch.Tensor, z_grad: torch.Tensor,
                      lr_quant: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update delta and z using LSQ learning rule on GPU.

    Update formulas:
    - delta_new = delta - lr * delta_grad
    - z_new = z - lr * z_grad

    Args:
        delta: Current delta (float32)
        z: Current z (float32)
        delta_grad: Gradient for delta (float32)
        z_grad: Gradient for z (float32)
        lr_quant: Learning rate for quantization parameters

    Returns:
        Tuple of (delta_new, z_new)
    """
    delta_new = delta - lr_quant * delta_grad
    z_new = z - lr_quant * z_grad

    # Ensure delta is positive (scale parameter must be positive)
    delta_new = delta_new.clamp(min=1e-6)

    return delta_new, z_new


class QuantizedWeightHandler:
    """Handler for quantized expert weights on GPU.

    Manages:
    - Dequantization of INT4/INT8 weights to BF16
    - LSQ delta/z gradient computation
    - Returning updated delta/z to CPU

    Usage:
        handler = QuantizedWeightHandler(expert_id, optimizer)
        w1_bf16, w2_bf16 = handler.dequantize_weights()
        # ... use w1_bf16, w2_bf16 for computation ...
        handler.compute_gradients(grad_w1, grad_w2)
        delta_new, z_new = handler.get_updated_params()
    """

    def __init__(self, expert_id: int, group_size: int = 128, lr_quant: float = 1e-4):
        """Initialize handler for an expert.

        Args:
            expert_id: Expert ID
            group_size: Quantization group size
            lr_quant: Learning rate for delta/z updates
        """
        self.expert_id = expert_id
        self.group_size = group_size
        self.lr_quant = lr_quant

        # Quantization state (set by load_quantized_weights)
        self.precision = 16  # Default BF16
        self.quant_w1 = None
        self.quant_w2 = None
        self.delta_w1 = None
        self.z_w1 = None
        self.delta_w2 = None
        self.z_w2 = None

        # GPU workspace for dequantized weights
        self.w1_bf16 = None
        self.w2_bf16 = None

        # Updated params to return to CPU
        self.delta_w1_new = None
        self.z_w1_new = None
        self.delta_w2_new = None
        self.z_w2_new = None

    def load_quantized_weights(self, quant_w1: Optional[torch.Tensor],
                               quant_w2: Optional[torch.Tensor],
                               delta_w1: Optional[torch.Tensor],
                               z_w1: Optional[torch.Tensor],
                               delta_w2: Optional[torch.Tensor],
                               z_w2: Optional[torch.Tensor],
                               precision: int,
                               w1_shape: Tuple[int, ...],
                               w2_shape: Tuple[int, ...]):
        """Load quantized weights and params from CPU to GPU.

        Args:
            quant_w1: Quantized weight1 (uint8) or None for BF16
            quant_w2: Quantized weight2 (uint8) or None for BF16
            delta_w1: Delta for weight1 (float32) or None for BF16
            z_w1: Z for weight1 (float32) or None for BF16
            delta_w2: Delta for weight2 (float32) or None for BF16
            z_w2: Z for weight2 (float32) or None for BF16
            precision: Quantization precision (16/8/4)
            w1_shape: Shape of weight1
            w2_shape: Shape of weight2
        """
        self.precision = precision
        device = torch.cuda.current_device()

        if precision == 16:
            # BF16: No quantization, params are None
            self.quant_w1 = None
            self.quant_w2 = None
            self.delta_w1 = None
            self.z_w1 = None
            self.delta_w2 = None
            self.z_w2 = None
        else:
            # INT8 or INT4: Copy to GPU
            if quant_w1 is not None:
                self.quant_w1 = quant_w1.to(device, non_blocking=True)
            if quant_w2 is not None:
                self.quant_w2 = quant_w2.to(device, non_blocking=True)
            if delta_w1 is not None:
                self.delta_w1 = delta_w1.to(device, non_blocking=True)
            if z_w1 is not None:
                self.z_w1 = z_w1.to(device, non_blocking=True)
            if delta_w2 is not None:
                self.delta_w2 = delta_w2.to(device, non_blocking=True)
            if z_w2 is not None:
                self.z_w2 = z_w2.to(device, non_blocking=True)

    def dequantize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize weights to BF16 for computation.

        Returns:
            Tuple of (w1_bf16, w2_bf16)
        """
        device = torch.cuda.current_device()

        if self.precision == 16:
            # BF16: Return original weights (handled separately)
            return None, None

        if self.precision == 8:
            # INT8 dequantization
            self.w1_bf16 = dequantize_int8(
                self.quant_w1, self.delta_w1, self.z_w1, self.group_size
            )
            self.w2_bf16 = dequantize_int8(
                self.quant_w2, self.delta_w2, self.z_w2, self.group_size
            )
        else:  # INT4
            # INT4 dequantization
            self.w1_bf16 = dequantize_int4(
                self.quant_w1, self.delta_w1, self.z_w1, self.group_size
            )
            self.w2_bf16 = dequantize_int4(
                self.quant_w2, self.delta_w2, self.z_w2, self.group_size
            )

        return self.w1_bf16, self.w2_bf16

    def compute_lsq_updates(self, grad_w1: torch.Tensor, grad_w2: torch.Tensor):
        """Compute LSQ delta/z updates from gradients.

        Args:
            grad_w1: Gradient for weight1
            grad_w2: Gradient for weight2
        """
        if self.precision == 16:
            # BF16: No quantization params to update
            self.delta_w1_new = None
            self.z_w1_new = None
            self.delta_w2_new = None
            self.z_w2_new = None
            return

        if self.precision == 8:
            # INT8 LSQ gradients
            quant_w1_float = self.quant_w1.float()
            quant_w2_float = self.quant_w2.float()

            delta_grad_w1, z_grad_w1 = compute_lsq_gradients(
                grad_w1, quant_w1_float, self.delta_w1, self.z_w1, self.group_size
            )
            delta_grad_w2, z_grad_w2 = compute_lsq_gradients(
                grad_w2, quant_w2_float, self.delta_w2, self.z_w2, self.group_size
            )
        else:  # INT4
            # INT4 LSQ gradients (need to unpack first)
            high_w1 = (self.quant_w1 >> 4) & 0x0F
            low_w1 = self.quant_w1 & 0x0F
            unpacked_w1 = torch.empty(self.quant_w1.numel() * 2, dtype=torch.float32, device=grad_w1.device)
            unpacked_w1[0::2] = high_w1.float()
            unpacked_w1[1::2] = low_w1.float()
            unpacked_w1 = unpacked_w1[:grad_w1.numel()]

            high_w2 = (self.quant_w2 >> 4) & 0x0F
            low_w2 = self.quant_w2 & 0x0F
            unpacked_w2 = torch.empty(self.quant_w2.numel() * 2, dtype=torch.float32, device=grad_w2.device)
            unpacked_w2[0::2] = high_w2.float()
            unpacked_w2[1::2] = low_w2.float()
            unpacked_w2 = unpacked_w2[:grad_w2.numel()]

            delta_grad_w1, z_grad_w1 = compute_lsq_gradients(
                grad_w1, unpacked_w1, self.delta_w1, self.z_w1, self.group_size
            )
            delta_grad_w2, z_grad_w2 = compute_lsq_gradients(
                grad_w2, unpacked_w2, self.delta_w2, self.z_w2, self.group_size
            )

        # Update params
        self.delta_w1_new, self.z_w1_new = update_lsq_params(
            self.delta_w1, self.z_w1, delta_grad_w1, z_grad_w1, self.lr_quant
        )
        self.delta_w2_new, self.z_w2_new = update_lsq_params(
            self.delta_w2, self.z_w2, delta_grad_w2, z_grad_w2, self.lr_quant
        )

    def get_updated_params(self) -> Tuple[Optional[torch.Tensor], ...]:
        """Get updated delta and z to return to CPU.

        Returns:
            Tuple of (delta_w1_new, z_w1_new, delta_w2_new, z_w2_new)
        """
        return (
            self.delta_w1_new,
            self.z_w1_new,
            self.delta_w2_new,
            self.z_w2_new,
        )