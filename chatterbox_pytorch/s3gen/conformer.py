"""
Clean PyTorch implementation of UpsampleConformerEncoder for S3Gen.
No external dependencies beyond PyTorch.

Weight structure matches:
    flow.encoder.embed.out.{0,1}
    flow.encoder.encoders.{0-5}.{self_attn, feed_forward, norm_*}
    flow.encoder.pre_lookahead_layer.{conv1, conv2}
    flow.encoder.up_layer.conv
    flow.encoder.up_embed.out.{0,1}
    flow.encoder.up_encoders.{0-3}.*
    flow.encoder.after_norm
"""
import math
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Utility Functions
# ============================================================================

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Create mask tensor where True indicates padded positions."""
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    return seq_range_expand >= seq_length_expand


# ============================================================================
# Positional Encoding (ESPnet style relative position)
# ============================================================================

class EspnetRelPositionalEncoding(nn.Module):
    """Relative positional encoding (ESPnet style)."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe: Optional[torch.Tensor] = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1: self.pe.size(1) // 2 + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


# ============================================================================
# Attention with Relative Position
# ============================================================================

class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-Head Attention with relative position encoding.

    Weight keys: linear_q, linear_k, linear_v, linear_out, linear_pos, pos_bias_u, pos_bias_v
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        self.dropout = nn.Dropout(p=dropout_rate)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, :x.size(-1) // 2 + 1]
        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor, pos_emb: torch.Tensor,
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch = query.size(0)

        # Compute Q, K, V
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        q = q.transpose(1, 2)  # Keep for later use with bias

        # Handle cache
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        # Position encoding
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)

        # Compute attention with position bias
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        q_with_bias_u = (q + self.pos_bias_u.to(q.device)).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v.to(q.device)).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        # Apply mask
        if mask.size(2) > 0:
            mask = mask.unsqueeze(1).eq(0)
            mask = mask[:, :, :, :scores.size(-1)]
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x), new_cache


# ============================================================================
# Feed Forward
# ============================================================================

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward with SiLU activation.

    Weight keys: w_1, w_2
    """

    def __init__(self, idim: int, hidden_units: int, dropout_rate: float, activation: nn.Module):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


# ============================================================================
# Conformer Encoder Layer
# ============================================================================

class ConformerEncoderLayer(nn.Module):
    """Conformer Encoder Layer.

    Weight keys: self_attn.*, feed_forward.*, norm_ff.*, norm_mha.*
    """

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module,
                 feed_forward_macaron: Optional[nn.Module] = None,
                 conv_module: Optional[nn.Module] = None,
                 dropout_rate: float = 0.1, normalize_before: bool = True):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)

        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)
            self.norm_final = nn.LayerNorm(size, eps=1e-12)

        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor,
                mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
                cnn_cache: torch.Tensor = torch.zeros((0, 0, 0))) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Macaron FFN (not used in this model but kept for compatibility)
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # Multi-head self-attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Convolution module (not used in this model)
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        # Feed forward
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


# ============================================================================
# Subsampling / Embedding
# ============================================================================

class LinearNoSubsampling(nn.Module):
    """Linear embedding without subsampling.

    Weight keys: out.0 (Linear), out.1 (LayerNorm)
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc: nn.Module):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


# ============================================================================
# Upsample and Pre-Lookahead Layers
# ============================================================================

class Upsample1D(nn.Module):
    """1D upsampling layer.

    Weight keys: conv
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv1d(channels, out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    """Pre-lookahead layer.

    Weight keys: conv1, conv2
    """

    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        return outputs + inputs


# ============================================================================
# Chunk Masking
# ============================================================================

def subsequent_chunk_mask(size: int, chunk_size: int, num_left_chunks: int = -1,
                          device: torch.device = torch.device("cpu")) -> torch.Tensor:
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
    return pos_idx.unsqueeze(0) < block_value.unsqueeze(1)


def add_optional_chunk_mask(xs: torch.Tensor, masks: torch.Tensor,
                            use_dynamic_chunk: bool, use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int, static_chunk_size: int,
                            num_decoding_left_chunks: int) -> torch.Tensor:
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
        else:
            chunk_size = torch.randint(1, max_len, (1,)).item()
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size, -1, xs.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks
    elif static_chunk_size > 0:
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size, -1, xs.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks
    else:
        chunk_masks = masks
    return chunk_masks


# ============================================================================
# UpsampleConformerEncoder
# ============================================================================

class UpsampleConformerEncoder(nn.Module):
    """Upsample Conformer Encoder for S3Gen.

    Architecture:
        - embed: LinearNoSubsampling
        - pre_lookahead_layer: PreLookaheadLayer
        - encoders: 6 ConformerEncoderLayers
        - up_layer: Upsample1D (2x)
        - up_embed: LinearNoSubsampling
        - up_encoders: 4 ConformerEncoderLayers
        - after_norm: LayerNorm
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
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        macaron_style: bool = False,
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        key_bias: bool = True,
        # These are ignored but kept for compatibility with original API
        input_layer: str = "linear",
        pos_enc_layer_type: str = "rel_pos_espnet",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        cnn_module_norm: str = "batch_norm",
        positionwise_conv_kernel_size: int = 1,
        global_cmvn: nn.Module = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self._output_size = output_size
        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.global_cmvn = global_cmvn

        # Activation
        activation = nn.SiLU()

        # Embedding layer
        pos_enc = EspnetRelPositionalEncoding(output_size, positional_dropout_rate)
        self.embed = LinearNoSubsampling(input_size, output_size, dropout_rate, pos_enc)

        # After norm
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

        # Build encoder layers
        self.pre_lookahead_layer = PreLookaheadLayer(channels=512, pre_lookahead_len=3)

        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(attention_heads, output_size, attention_dropout_rate, key_bias),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate, activation),
                None,  # No macaron FFN
                None,  # No conv module
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ])

        # Upsample layer
        self.up_layer = Upsample1D(channels=512, out_channels=512, stride=2)

        # Upsample embedding
        up_pos_enc = EspnetRelPositionalEncoding(output_size, positional_dropout_rate)
        self.up_embed = LinearNoSubsampling(input_size, output_size, dropout_rate, up_pos_enc)

        # Upsample encoder layers
        self.up_encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(attention_heads, output_size, attention_dropout_rate, key_bias),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate, activation),
                None,
                None,
                dropout_rate,
                normalize_before,
            ) for _ in range(4)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor,
                decoding_chunk_size: int = 0,
                num_decoding_left_chunks: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            xs: Input tensor (B, T, D)
            xs_lens: Input lengths (B,)

        Returns:
            Output tensor (B, T', D) and masks (B, 1, T')
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk,
            decoding_chunk_size, self.static_chunk_size, num_decoding_left_chunks
        )

        # Lookahead + conformer encoder
        xs = self.pre_lookahead_layer(xs)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)

        # Upsample + conformer encoder
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()

        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(
            xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk,
            decoding_chunk_size, self.static_chunk_size * self.up_layer.stride, num_decoding_left_chunks
        )

        for layer in self.up_encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks
