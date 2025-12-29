"""
Rotary Position Embedding (RoPE) for LLaMA3.

LLaMA3 uses a modified version of RoPE with scaling factors.
Reference: https://arxiv.org/abs/2104.09864
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class LLaMA3RotaryEmbedding(nn.Module):
    """
    LLaMA3-style Rotary Position Embedding with scaling.

    This implements the RoPE variant used in LLaMA3 with:
    - Base frequency scaling (rope_theta)
    - Dynamic frequency scaling for different position ranges

    Args:
        dim: Dimension of the embedding (should be head_dim)
        max_position_embeddings: Maximum sequence length
        base: Base for the sinusoidal frequencies (rope_theta)
        factor: Scaling factor for extended context
        low_freq_factor: Factor for low frequency components
        high_freq_factor: Factor for high frequency components
        original_max_position_embeddings: Original context length before scaling
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 500000.0,
        factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings

        # Compute the inverse frequencies with LLaMA3 scaling
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies with LLaMA3 scaling."""
        # Base inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

        # Apply LLaMA3 frequency scaling
        low_freq_wavelen = self.original_max_position_embeddings / self.low_freq_factor
        high_freq_wavelen = self.original_max_position_embeddings / self.high_freq_factor

        wavelen = 2 * math.pi / inv_freq

        # Smooth interpolation between scaled and original frequencies
        smooth = (self.original_max_position_embeddings / wavelen - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smooth = torch.clamp(smooth, 0.0, 1.0)

        # Scale frequencies based on wavelen
        scaled_inv_freq = inv_freq / self.factor

        # Interpolate: use scaled for long wavelengths, original for short
        inv_freq = torch.where(
            wavelen > low_freq_wavelen,
            scaled_inv_freq,
            torch.where(
                wavelen < high_freq_wavelen,
                inv_freq,
                (1 - smooth) * scaled_inv_freq + smooth * inv_freq,
            ),
        )

        return inv_freq

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cos/sin cache if needed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=torch.float32)

            # Compute freqs: (seq_len,) x (dim/2,) -> (seq_len, dim/2)
            freqs = torch.outer(t, self.inv_freq.to(device))

            # Duplicate for sin and cos: (seq_len, dim)
            emb = torch.cat((freqs, freqs), dim=-1)

            # Cache cos and sin values
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, n_heads, head_dim)
            position_ids: Optional position indices of shape (batch, seq_len)

        Returns:
            Tuple of (cos, sin) tensors for the given positions
        """
        seq_len = x.shape[1]

        if position_ids is None:
            # Update cache if needed
            self._update_cache(seq_len, x.device, x.dtype)
            cos = self._cos_cached[:seq_len]
            sin = self._sin_cached[:seq_len]
        else:
            # Update cache to cover all positions we need
            max_pos = int(position_ids.max().item()) + 1
            self._update_cache(max_pos, x.device, x.dtype)
            # Index into cache using position_ids
            # position_ids: (batch, seq_len) -> cos: (batch, seq_len, dim)
            cos = self._cos_cached[position_ids]
            sin = self._sin_cached[position_ids]

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the hidden dims.

    Takes the last dimension and rotates the first half with the second half:
    [x1, x2] -> [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, n_heads, head_dim)
        cos: Cosine embeddings of shape (seq_len, head_dim) or (batch, seq_len, head_dim)
        sin: Sine embeddings of shape (seq_len, head_dim) or (batch, seq_len, head_dim)
        unsqueeze_dim: Not used (kept for API compatibility)

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied
    """
    # Reshape cos/sin to broadcast with q/k
    if cos.dim() == 2:
        # (seq_len, head_dim) -> (1, seq_len, 1, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        # (batch, seq_len, head_dim) -> (batch, seq_len, 1, head_dim)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
