"""
Clean PyTorch implementation of ConditionalDecoder for S3Gen.
No external dependencies beyond PyTorch.

Weight structure matches:
    flow.decoder.estimator.time_embeddings.*
    flow.decoder.estimator.time_mlp.*
    flow.decoder.estimator.down_blocks.0.{0,1,2}.*
    flow.decoder.estimator.mid_blocks.{0-11}.{0,1}.*
    flow.decoder.estimator.up_blocks.0.{0,1,2}.*
    flow.decoder.estimator.final_block.*
    flow.decoder.estimator.final_proj.*
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Time Embedding
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding MLP.

    Weight keys: linear_1, linear_2
    """

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.Mish()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


# ============================================================================
# Causal Convolution
# ============================================================================

class CausalConv1d(nn.Conv1d):
    """Causal 1D convolution with left-only padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias)
        assert stride == 1
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.causal_padding)
        return super().forward(x)


# ============================================================================
# Building Blocks
# ============================================================================

class CausalBlock1D(nn.Module):
    """Causal block: CausalConv1d -> LayerNorm -> Mish.

    Weight keys: block.0 (CausalConv1d), block.2 (LayerNorm)
    """

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.block = nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.block(x * mask)
        return output * mask


class Transpose(nn.Module):
    """Transpose dimensions."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class CausalResnetBlock1D(nn.Module):
    """Causal ResNet block.

    Weight keys: block1.*, block2.*, mlp.*, res_conv.*
    """

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, dim_out),
        )
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.mlp(t).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class Downsample1D(nn.Module):
    """Downsample by stride 2."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsample by factor 2 using transposed conv."""

    def __init__(self, channels: int, use_conv_transpose: bool = True):
        super().__init__()
        self.use_conv_transpose = use_conv_transpose
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        else:
            self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv_transpose:
            return self.conv(x)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ============================================================================
# Attention
# ============================================================================

class Attention(nn.Module):
    """Self-attention layer.

    Weight keys: to_q, to_k, to_v, to_out.0
    """

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )
        self.scale = dim_head ** -0.5

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.to_q(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.heads * self.dim_head)
        return self.to_out(out)


# ============================================================================
# Feed Forward
# ============================================================================

class GELU(nn.Module):
    """GELU activation with linear projection.

    Weight keys: proj
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):
    """Feed-forward network.

    Weight keys: net.0.proj, net.2
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            GELU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Transformer Block
# ============================================================================

class BasicTransformerBlock(nn.Module):
    """Basic transformer block.

    Weight keys: norm1, attn1.*, norm3, ff.*
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 dropout: float = 0.0, activation_fn: str = "gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, num_attention_heads, attention_head_dim, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, attention_mask)
        hidden_states = attn_output + hidden_states

        # Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


# ============================================================================
# Mask Utilities
# ============================================================================

def subsequent_chunk_mask(size: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    """Create chunk-based causal mask."""
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
    return pos_idx.unsqueeze(0) < block_value.unsqueeze(1)


def add_optional_chunk_mask(x: torch.Tensor, mask: torch.Tensor, use_dynamic_chunk: bool,
                             use_dynamic_left_chunk: bool, decoding_chunk_size: int,
                             static_chunk_size: int, num_decoding_left_chunks: int) -> torch.Tensor:
    """Create chunk mask for causal attention."""
    if static_chunk_size > 0:
        chunk_masks = subsequent_chunk_mask(x.size(1), static_chunk_size, x.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = mask & chunk_masks
    else:
        chunk_masks = mask
    return chunk_masks


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool mask to attention bias.

    Input mask: (B, 1, T) where True = valid position
    Output: (B, 1, 1, T) bias to add to attention scores
    """
    assert mask.dtype == torch.bool
    # Expand to (B, 1, 1, T) for broadcasting with (B, heads, T, T)
    if mask.ndim == 3:
        mask = mask.unsqueeze(2)  # (B, 1, 1, T)
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e+10
    return mask


# ============================================================================
# ConditionalDecoder
# ============================================================================

class ConditionalDecoder(nn.Module):
    """U-Net style decoder for Conditional Flow Matching.

    Weight structure matches checkpoint exactly with causal convolutions.
    """

    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 80,
        causal: bool = True,
        channels: list = [256],
        dropout: float = 0.0,
        attention_head_dim: int = 64,
        n_blocks: int = 4,
        num_mid_blocks: int = 12,
        num_heads: int = 8,
        act_fn: str = "gelu",
        meanflow: bool = False,
    ):
        super().__init__()
        channels = tuple(channels)
        self.meanflow = meanflow
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.static_chunk_size = 0

        # Time embedding
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # Initialize blocks
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # Down blocks
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1

            resnet = CausalResnetBlock1D(input_channel, output_channel, time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                ) for _ in range(n_blocks)
            ])
            downsample = Downsample1D(output_channel) if not is_last else CausalConv1d(output_channel, output_channel, 3)
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        # Mid blocks
        for _ in range(num_mid_blocks):
            resnet = CausalResnetBlock1D(channels[-1], channels[-1], time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=channels[-1],
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                ) for _ in range(n_blocks)
            ])
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        # Up blocks
        channels_rev = channels[::-1] + (channels[0],)
        for i in range(len(channels_rev) - 1):
            input_channel = channels_rev[i] * 2  # Skip connection doubles channels
            output_channel = channels_rev[i + 1]
            is_last = i == len(channels_rev) - 2

            resnet = CausalResnetBlock1D(input_channel, output_channel, time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                ) for _ in range(n_blocks)
            ])
            upsample = Upsample1D(output_channel, use_conv_transpose=True) if not is_last else CausalConv1d(output_channel, output_channel, 3)
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        # Final layers
        self.final_block = CausalBlock1D(channels_rev[-1], channels_rev[-1])
        self.final_proj = nn.Conv1d(channels_rev[-1], self.out_channels, 1)

        self.initialize_weights()

        # Meanflow time embedding mixer (not used in standard model)
        self.time_embed_mixer = None

    @property
    def dtype(self):
        return self.final_proj.weight.dtype

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor,
                t: torch.Tensor, spks: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, r: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 80, T) - noised mel
            mask: (B, 1, T) - output mask
            mu: (B, 80, T) - encoder output
            t: (B,) - timestep
            spks: (B, 80) - speaker embedding
            cond: (B, 80, T) - conditioning mel
            r: (B,) - end time for meanflow (optional)
        """
        # Time embedding
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        # Concatenate inputs: x, mu -> (B, 160, T)
        x = torch.cat([x, mu], dim=1)

        # Add speaker embedding if provided
        if spks is not None:
            spks = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, spks], dim=1)

        # Add conditioning if provided
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        # Down blocks
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)

            # Transformer blocks
            x = x.transpose(1, 2).contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = x.transpose(1, 2).contiguous()

            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # Mid blocks
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)

            x = x.transpose(1, 2).contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = x.transpose(1, 2).contiguous()

        # Up blocks
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = torch.cat([x[:, :, :skip.shape[-1]], skip], dim=1)
            x = resnet(x, mask_up, t)

            x = x.transpose(1, 2).contiguous()
            attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = x.transpose(1, 2).contiguous()

            x = upsample(x * mask_up)

        # Final layers
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
