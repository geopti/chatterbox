"""
Root Mean Square Layer Normalization

Used in LLaMA models instead of LayerNorm for better training stability.
Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not re-center the activations,
    only re-scales them. This is more efficient and works well for transformers.

    Args:
        dim: The dimension to normalize over
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
