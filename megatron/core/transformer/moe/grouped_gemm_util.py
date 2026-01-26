# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

import torch

class FallbackOps:
    def gmm(self, x, weight, tokens_per_expert, trans_b=False):
        if x.shape[0] == 0:
            out_features = weight.shape[-1] if not trans_b else weight.shape[-2]
            return torch.empty(0, out_features, device=x.device, dtype=x.dtype)
        
        # x: [N, H], weight: [E, H, F], tokens_per_expert: [E]
        out_features = weight.shape[-1] if not trans_b else weight.shape[-2]
        output = torch.empty((x.shape[0], out_features), device=x.device, dtype=x.dtype)
        
        start = 0
        for i in range(weight.shape[0]):
            num_tokens = tokens_per_expert[i].item()
            if num_tokens > 0:
                chunk = x[start : start + num_tokens]
                w = weight[i]
                if trans_b:
                    torch.matmul(chunk, w.t(), out=output[start : start + num_tokens])
                else:
                    torch.matmul(chunk, w, out=output[start : start + num_tokens])
                start += num_tokens
        return output

def grouped_gemm_is_available():
    """Check if grouped_gemm is available."""
    return grouped_gemm is not None


def assert_grouped_gemm_is_available():
    """Assert that grouped_gemm is available."""
    if not grouped_gemm_is_available():
        import warnings
        warnings.warn(
            "Grouped GEMM is not available. Using slow fallback implementation. "
            "Please run `pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4` "
            "for performance."
        )


ops = grouped_gemm.ops if grouped_gemm_is_available() else FallbackOps()
