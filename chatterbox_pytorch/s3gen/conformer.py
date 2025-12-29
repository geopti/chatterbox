"""
Conformer encoder with upsampling for S3Gen.

The encoder processes speech tokens and upsamples them 2x for mel prediction.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConformerConvModule(nn.Module):
    """Convolution module for Conformer."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise conv expansion
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # Depthwise conv
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=padding, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Pointwise conv reduction
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, time, d_model)

        Returns:
            Output of same shape
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, d_model, time)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x.transpose(1, 2)  # (batch, time, d_model)


class ConformerAttention(nn.Module):
    """Multi-head self-attention for Conformer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.layer_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        x = self.layer_norm(x)

        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class ConformerFeedForward(nn.Module):
    """Feed-forward module for Conformer."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = F.silu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """Single Conformer block."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Feed-forward module 1 (half-step)
        self.ff1 = ConformerFeedForward(d_model, d_ff, dropout)

        # Multi-head self-attention
        self.attn = ConformerAttention(d_model, n_heads, dropout)

        # Convolution module
        self.conv = ConformerConvModule(d_model, conv_kernel_size, dropout)

        # Feed-forward module 2 (half-step)
        self.ff2 = ConformerFeedForward(d_model, d_ff, dropout)

        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # FF1 (half-step residual)
        x = x + 0.5 * self.ff1(x)

        # Attention
        x = x + self.attn(x, mask)

        # Convolution
        x = x + self.conv(x)

        # FF2 (half-step residual)
        x = x + 0.5 * self.ff2(x)

        # Final layer norm
        x = self.layer_norm(x)

        return x


class UpsampleConformerEncoder(nn.Module):
    """
    Conformer encoder with 2x upsampling.

    Takes speech tokens and produces continuous representations
    at 2x temporal resolution for mel-spectrogram prediction.
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Input embedding
        self.input_layer = nn.Linear(input_size, output_size)

        # Positional encoding
        self.pos_enc = PositionalEncoding(output_size, dropout=positional_dropout_rate)

        # Conformer blocks (before upsample)
        self.encoders = nn.ModuleList([
            ConformerBlock(
                d_model=output_size,
                n_heads=attention_heads,
                d_ff=linear_units,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout_rate,
            )
            for _ in range(num_blocks)
        ])

        # 2x Upsample
        self.upsample = nn.ConvTranspose1d(
            output_size,
            output_size,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Conformer blocks (after upsample)
        num_up_blocks = 4
        self.up_encoders = nn.ModuleList([
            ConformerBlock(
                d_model=output_size,
                n_heads=attention_heads,
                d_ff=linear_units,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout_rate,
            )
            for _ in range(num_up_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        x_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input embeddings of shape (batch, time, input_size)
            x_lens: Lengths of input sequences

        Returns:
            Tuple of (output, output_lens)
        """
        # Input projection
        x = self.input_layer(x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Conformer blocks (before upsample)
        for encoder in self.encoders:
            x = encoder(x)

        # 2x Upsample: (batch, time, dim) -> (batch, 2*time, dim)
        x = x.transpose(1, 2)  # (batch, dim, time)
        x = self.upsample(x)
        x = x.transpose(1, 2)  # (batch, 2*time, dim)

        # Conformer blocks (after upsample)
        for encoder in self.up_encoders:
            x = encoder(x)

        # Update lengths
        if x_lens is not None:
            x_lens = x_lens * 2

        return x, x_lens
