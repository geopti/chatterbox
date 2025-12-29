"""
Conditional Flow Matching (CFM) decoder for S3Gen.

U-Net style architecture for mel-spectrogram generation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.linear1(x))
        x = self.linear2(x)
        return x


class CausalConv1d(nn.Module):
    """Causal 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, : -self.padding]
        return x


class ResnetBlock1D(nn.Module):
    """Residual block for 1D convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Norm and conv 1
        self.norm1 = nn.GroupNorm(8, in_channels)
        if causal:
            self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size)
        else:
            self.conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=kernel_size // 2
            )

        # Time embedding projection
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        # Norm and conv 2
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        if causal:
            self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size)
        else:
            self.conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size, padding=kernel_size // 2
            )

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_emb_proj(F.silu(time_emb))[:, :, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class TransformerBlock(nn.Module):
    """Transformer block for decoder."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        inner_dim = n_heads * head_dim

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, dim)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class ConditionalDecoder(nn.Module):
    """
    U-Net style decoder for Conditional Flow Matching.

    Takes noised mel-spectrograms, encoder output (mu), timestep,
    and speaker embedding to predict the velocity field.
    """

    def __init__(
        self,
        in_channels: int = 320,  # x + mu + spk + cond
        out_channels: int = 80,
        channels: list = [256],
        n_blocks: int = 4,
        num_mid_blocks: int = 12,
        num_heads: int = 8,
        attention_head_dim: int = 64,
        dropout: float = 0.0,
        causal: bool = True,
        act_fn: str = "gelu",
        meanflow: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.meanflow = meanflow

        # Compute internal dimension
        time_embed_dim = channels[0] * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(channels[0]),
            TimestepEmbedding(channels[0], time_embed_dim),
        )

        # Ratio embedding for meanflow
        if meanflow:
            self.ratio_embed = nn.Sequential(
                SinusoidalPosEmb(channels[0]),
                TimestepEmbedding(channels[0], time_embed_dim),
            )

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, channels[0], 1)

        # Down blocks
        self.down_blocks = nn.ModuleList()
        ch = channels[0]
        for _ in range(n_blocks):
            self.down_blocks.append(
                ResnetBlock1D(ch, ch, time_embed_dim, dropout=dropout, causal=causal)
            )

        # Mid blocks (transformer)
        self.mid_blocks = nn.ModuleList()
        for _ in range(num_mid_blocks):
            self.mid_blocks.append(
                nn.ModuleList([
                    ResnetBlock1D(ch, ch, time_embed_dim, dropout=dropout, causal=causal),
                    TransformerBlock(ch, num_heads, attention_head_dim, dropout),
                ])
            )

        # Up blocks
        self.up_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.up_blocks.append(
                ResnetBlock1D(ch * 2, ch, time_embed_dim, dropout=dropout, causal=causal)
            )

        # Output projection
        self.output_norm = nn.GroupNorm(8, ch)
        self.output_proj = nn.Conv1d(ch, out_channels, 1)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noised mel-spectrogram of shape (batch, n_mels, time)
            mask: Mask of shape (batch, 1, time)
            mu: Encoder output of shape (batch, n_mels, time)
            t: Timestep of shape (batch,)
            spks: Speaker embedding of shape (batch, spk_dim)
            cond: Conditioning mel of shape (batch, n_mels, time)
            r: Ratio for meanflow (optional)

        Returns:
            Predicted velocity of shape (batch, n_mels, time)
        """
        # Time embedding
        t_emb = self.time_embed(t)

        if self.meanflow and r is not None:
            r_emb = self.ratio_embed(r)
            t_emb = t_emb + r_emb

        # Expand speaker embedding to time dimension
        spks_expanded = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])

        # Concatenate inputs: [x, mu, spks, cond]
        h = torch.cat([x, mu, spks_expanded, cond], dim=1)

        # Input projection
        h = self.input_proj(h)

        # Down path
        skip_connections = []
        for block in self.down_blocks:
            h = block(h, t_emb)
            skip_connections.append(h)

        # Mid blocks
        for resnet, transformer in self.mid_blocks:
            h = resnet(h, t_emb)
            # Transformer operates on (batch, time, channels)
            h = h.transpose(1, 2)
            h = transformer(h)
            h = h.transpose(1, 2)

        # Up path
        for block in self.up_blocks:
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_proj(h)

        return h * mask
