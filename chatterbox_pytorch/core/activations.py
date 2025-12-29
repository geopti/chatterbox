"""
Activation functions used in Chatterbox-TTS.

- SwiGLU: Used in LLaMA transformer FFN layers
- Snake: Sine-based periodic activation for HiFi-GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function used in LLaMA.

    SwiGLU(x) = SiLU(W1 @ x) * (W3 @ x)

    This combines a gated linear unit with the SiLU (Swish) activation.
    Reference: https://arxiv.org/abs/2002.05202

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4 * dim for standard, or custom for LLaMA)
        bias: Whether to use bias in linear layers
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Output tensor of shape (..., dim)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Snake(nn.Module):
    """
    Snake activation function for HiFi-GAN.

    Snake(x) = x + (1/a) * sin^2(a * x)

    A sine-based periodic activation that helps with audio generation.
    Reference: https://arxiv.org/abs/2006.08195

    Args:
        channels: Number of channels
        alpha: Initial alpha value (frequency parameter)
        alpha_trainable: Whether alpha should be learnable
        alpha_logscale: Whether to use log-scale for alpha
    """

    def __init__(
        self,
        channels: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale

        if alpha_logscale:
            # Log scale: alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(channels) * alpha)
        else:
            # Linear scale: alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(channels) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Snake activation.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of same shape
        """
        # Align alpha with x: (C,) -> (1, C, 1)
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)

        if self.alpha_logscale:
            alpha = torch.exp(alpha)

        # Snake activation: x + (1/a) * sin^2(a * x)
        x = x + (1.0 / (alpha + self.eps)) * torch.pow(torch.sin(x * alpha), 2)

        return x
