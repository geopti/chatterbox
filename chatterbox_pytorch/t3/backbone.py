"""
LLaMA-style Transformer backbone for T3.

This is a from-scratch implementation of the LLaMA transformer architecture
without using HuggingFace transformers.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.rmsnorm import RMSNorm
from ..core.attention import MultiHeadAttention
from ..core.rope import LLaMA3RotaryEmbedding, apply_rotary_pos_emb


class LLaMAMLP(nn.Module):
    """
    LLaMA MLP block with SwiGLU activation.

    Architecture:
        out = down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)  # gate_proj
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)  # down_proj
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=bias)  # up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLaMAAttention(nn.Module):
    """
    LLaMA-style attention with RoPE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float = 500000.0,
        rope_scaling_factor: float = 8.0,
        rope_high_freq_factor: float = 4.0,
        rope_low_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int = 8192,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim ** -0.5
        self.attention_dropout = attention_dropout

        # Projections
        self.wq = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.wk = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.wv = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.wo = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        # RoPE
        self.rotary_emb = LLaMA3RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            factor=rope_scaling_factor,
            high_freq_factor=rope_high_freq_factor,
            low_freq_factor=rope_low_freq_factor,
            original_max_position_embeddings=rope_original_max_position_embeddings,
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat K/V heads for grouped query attention."""
        if self.num_kv_groups == 1:
            return x
        batch, seq_len, num_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :].expand(batch, seq_len, num_kv_heads, self.num_kv_groups, head_dim)
        return x.reshape(batch, seq_len, self.num_heads, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.wq(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.wv(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Get RoPE embeddings
        cos, sin = self.rotary_emb(q, position_ids)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        cache = (k, v) if use_cache else None

        # Repeat K, V for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)

        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.wo(attn_output)

        return output, cache


class LLaMABlock(nn.Module):
    """
    Single LLaMA transformer block.

    Architecture:
        x = x + attention(norm1(x))
        x = x + mlp(norm2(x))
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        rope_scaling_factor: float = 8.0,
        rope_high_freq_factor: float = 4.0,
        rope_low_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int = 8192,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        mlp_bias: bool = False,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = LLaMAAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            rope_high_freq_factor=rope_high_freq_factor,
            rope_low_freq_factor=rope_low_freq_factor,
            rope_original_max_position_embeddings=rope_original_max_position_embeddings,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
        )

        self.ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn = LLaMAMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=mlp_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, cache = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache


class LLaMABackbone(nn.Module):
    """
    LLaMA transformer backbone for T3.

    This is the core transformer that processes the concatenated
    conditioning, text, and speech embeddings.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 64,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        rope_scaling_factor: float = 8.0,
        rope_high_freq_factor: float = 4.0,
        rope_low_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int = 8192,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        mlp_bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers

        # Transformer layers
        self.layers = nn.ModuleList([
            LLaMABlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                rope_scaling_factor=rope_scaling_factor,
                rope_high_freq_factor=rope_high_freq_factor,
                rope_low_freq_factor=rope_low_freq_factor,
                rope_original_max_position_embeddings=rope_original_max_position_embeddings,
                attention_dropout=attention_dropout,
                attention_bias=attention_bias,
                mlp_bias=mlp_bias,
            )
            for _ in range(num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through the transformer.

        Args:
            inputs_embeds: Input embeddings of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_values: Cached key/value pairs for each layer
            use_cache: Whether to return cached key/value pairs

        Returns:
            Tuple of (hidden_states, new_cache)
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        hidden_states = inputs_embeds

        # Determine the total sequence length (including cached positions)
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[1]  # (batch, past_len, heads, head_dim)
        total_len = past_len + seq_len

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                past_len, total_len, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

        # Create causal mask if not provided
        if attention_mask is None:
            # For cached generation, we need a mask that allows attending to all past + current positions
            # Shape: (seq_len, total_len) - each new position can attend to all previous positions
            attention_mask = torch.triu(
                torch.full((seq_len, total_len), float("-inf"), device=hidden_states.device),
                diagonal=past_len + 1,  # Allow attending to past_len positions + current diagonal
            )

        new_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                new_cache.append(cache)

        hidden_states = self.norm(hidden_states)

        return hidden_states, new_cache
