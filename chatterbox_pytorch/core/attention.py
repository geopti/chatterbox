"""
Multi-Head Attention implementations.

Provides clean, from-scratch attention mechanisms used in the model.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with optional RoPE support.

    This is a clean implementation of multi-head attention that can be used
    with or without rotary position embeddings.

    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of key/value heads (for GQA). If None, uses n_heads.
        head_dim: Dimension per head. If None, computed as dim // n_heads.
        dropout: Dropout probability
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.dropout = dropout

        # Compute total dimensions
        self.n_rep = self.n_heads // self.n_kv_heads  # For GQA

        # Projection layers
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)

        self.scale = self.head_dim ** -0.5

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat key/value heads for Grouped Query Attention.

        Args:
            x: Tensor of shape (batch, seq_len, n_kv_heads, head_dim)

        Returns:
            Tensor of shape (batch, seq_len, n_heads, head_dim)
        """
        if self.n_rep == 1:
            return x

        batch, seq_len, n_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, self.n_rep, head_dim)
        return x.reshape(batch, seq_len, self.n_heads, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cos: Cosine for RoPE of shape (seq_len, head_dim)
            sin: Sine for RoPE of shape (seq_len, head_dim)
            attention_mask: Optional attention mask
            past_key_value: Optional cached key/value tensors
            use_cache: Whether to return cached key/value

        Returns:
            Tuple of (output, cache) where cache is None if use_cache=False
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary position embeddings if provided
        if cos is not None and sin is not None:
            from .rope import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        if use_cache:
            cache = (k, v)
        else:
            cache = None

        # Repeat K, V for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Compute output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch, seq_len, -1)

        # Output projection
        output = self.wo(attn_output)

        return output, cache


class RelativePositionMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Relative Position Encoding.

    Used in Conformer encoder layers for the S3Gen model.
    Based on the Shaw et al. relative position representation.

    Args:
        n_heads: Number of attention heads
        d_model: Model dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Projections
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        # Relative position
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        self.scale = self.head_dim ** -0.5

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative position shift for attention scores."""
        batch, n_heads, seq_len, _ = x.shape

        # Pad and reshape to shift positions
        x = F.pad(x, (1, 0))
        x = x.view(batch, n_heads, -1, seq_len)
        x = x[:, :, 1:, :]

        return x

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with relative position encoding.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            pos_emb: Position embeddings of shape (1, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.linear_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.linear_k(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.linear_v(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Position embedding projection
        p = self.linear_pos(pos_emb).view(1, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        p = p.transpose(1, 2)

        # Content-based attention (with bias u)
        q_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        ac = torch.matmul(q_u, k.transpose(-2, -1))

        # Position-based attention (with bias v)
        q_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        bd = torch.matmul(q_v, p.transpose(-2, -1))
        bd = self._rel_shift(bd)

        # Combine and scale
        attn_weights = (ac + bd) * self.scale

        # Apply mask
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Compute output
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.d_model)
        output = self.linear_out(output)

        return output
