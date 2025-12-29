"""
Perceiver resampler for compressing speech prompt tokens.

Based on the Perceiver architecture from DeepMind.
Reference: https://arxiv.org/abs/2103.03206
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Attention block for Perceiver.

    Supports both self-attention and cross-attention.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Query tensor of shape (B, T1, dim)
            context: Key/Value tensor of shape (B, T2, dim). If None, self-attention.

        Returns:
            Output tensor of shape (B, T1, dim)
        """
        # Pre-norm
        x_norm = self.norm(x)

        if context is None:
            context = x_norm
        else:
            context = self.norm(context)

        batch_size = x.shape[0]

        # Compute Q, K, V
        q = self.to_q(x_norm).view(batch_size, -1, self.n_heads, self.head_dim)
        k = self.to_k(context).view(batch_size, -1, self.n_heads, self.head_dim)
        v = self.to_v(context).view(batch_size, -1, self.n_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # (B, heads, T1, head_dim)
        k = k.transpose(1, 2)  # (B, heads, T2, head_dim)
        v = v.transpose(1, 2)  # (B, heads, T2, head_dim)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out = self.proj_out(out)

        # Residual connection
        return x + out


class Perceiver(nn.Module):
    """
    Perceiver resampler for compressing variable-length input to fixed-length output.

    Uses learnable query tokens that cross-attend to the input, followed by
    self-attention to refine the output. Uses a SINGLE shared attention block
    for both operations (matching original checkpoint structure).

    Args:
        n_queries: Number of output query tokens
        query_dim: Dimension of each query token
        n_heads: Number of attention heads
    """

    def __init__(
        self,
        n_queries: int = 32,
        query_dim: int = 1024,
        n_heads: int = 4,
    ):
        super().__init__()

        # Learnable query tokens (named to match checkpoint)
        self.pre_attention_query = nn.Parameter(torch.empty(1, n_queries, query_dim))

        # Initialize queries
        variance = math.sqrt(3.0) * math.sqrt(2.0 / (n_queries + n_queries))
        self.pre_attention_query.data.uniform_(-variance, variance)

        # Single shared attention block (used for both cross and self attention)
        self.attn = AttentionBlock(query_dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input sequence to fixed-length output.

        Args:
            x: Input tensor of shape (B, T, dim)

        Returns:
            Output tensor of shape (B, n_queries, dim)
        """
        batch_size = x.shape[0]

        # Expand queries to batch size
        queries = self.pre_attention_query.expand(batch_size, -1, -1)

        # Cross-attention: queries attend to input
        out = self.attn(queries, x)

        # Self-attention: refine output (same block, different inputs)
        out = self.attn(out, out)

        return out
