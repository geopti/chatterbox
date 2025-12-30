"""
S3Tokenizer: WhisperVQ-style speech tokenizer (V2).

Pure PyTorch implementation matching the s3tokenizer package architecture.
Converts waveforms to discrete speech tokens using FSQ quantization.
"""

import math
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Constants
S3_SR = 16_000          # Sampling rate for tokenizer
S3_HOP = 160            # STFT hop size (100 frames/sec)
S3_TOKEN_HOP = 640      # Token hop size (25 tokens/sec)
S3_TOKEN_RATE = 25      # Tokens per second
SPEECH_VOCAB_SIZE = 6561  # Vocabulary size = 3^8 (ternary quantization)


# =============================================================================
# Utility functions
# =============================================================================

def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths: Batch of lengths (B,)
        max_len: Maximum length (optional)

    Returns:
        Mask tensor (B, max_T)
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for attention masking.

    Args:
        mask: Boolean mask tensor
        dtype: Target dtype

    Returns:
        Float mask with -1e10 for masked positions
    """
    assert mask.dtype == torch.bool
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e+10
    return mask


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute rotary position embeddings (RoPE).

    Args:
        dim: Dimension per head
        end: Maximum sequence length
        theta: Base for frequency computation

    Returns:
        Complex tensor of shape (end, dim) for RoPE
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return torch.cat((freqs_cis, freqs_cis), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        xq: Query tensor (B, T, n_heads, head_dim)
        xk: Key tensor (B, T, n_heads, head_dim)
        freqs_cis: Precomputed RoPE frequencies

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    real = torch.view_as_real(freqs_cis)
    cos, sin = real[:, :, 0], real[:, :, 1]
    cos = cos.unsqueeze(0).unsqueeze(2).to(xq.dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(xq.dtype)

    D = xq.shape[-1]
    half_l, half_r = xq[:, :, :, :D // 2], xq[:, :, :, D // 2:]
    xq_r = torch.cat((-half_r, half_l), dim=-1)

    D = xk.shape[-1]
    half_l, half_r = xk[:, :, :, :D // 2], xk[:, :, :, D // 2:]
    xk_r = torch.cat((-half_r, half_l), dim=-1)

    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


# =============================================================================
# Custom nn.Module subclasses with dtype handling
# =============================================================================

class LayerNorm(nn.LayerNorm):
    """LayerNorm that casts to float for computation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).type(x.dtype)


class Linear(nn.Linear):
    """Linear layer with dtype-aware computation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    """Conv1d with dtype-aware computation."""

    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


# =============================================================================
# Attention modules
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Base multi-head attention module."""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class FSMNMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with FSMN (Feedforward Sequential Memory Network) memory block."""

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
    ):
        super().__init__(n_state, n_head)

        # FSMN memory block - depthwise convolution
        self.fsmn_block = nn.Conv1d(
            n_state, n_state, kernel_size,
            stride=1, padding=0, groups=n_state, bias=False
        )
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = nn.ConstantPad1d((self.left_padding, self.right_padding), 0.0)

        # Override key to have no bias (matches original)
        self.key = Linear(n_state, n_state, bias=False)

    def forward_fsmn(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply FSMN memory block.

        Args:
            inputs: Input tensor (B, T, n_heads, head_dim)
            mask: Padding mask (B, T, 1)

        Returns:
            Memory output (B, T, D)
        """
        b, t, _, _ = inputs.size()
        inputs = inputs.view(b, t, -1)
        if mask is not None and mask.size(2) > 0:
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x = x + inputs
        return x * mask

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        # Apply RoPE
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # FSMN memory from value
        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.permute(0, 2, 1, 3) * scale
        v = v.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1) * scale

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach(), fsm_memory

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk, fsm_memory = self.qkv_attention(q, k, v, mask, mask_pad, freqs_cis)
        return self.out(wv) + fsm_memory, qk


class ResidualAttentionBlock(nn.Module):
    """Transformer block with FSMN attention."""

    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
    ):
        super().__init__()

        self.attn = FSMNMultiHeadAttention(n_state, n_head, kernel_size)
        self.attn_ln = LayerNorm(n_state, eps=1e-5)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp),
            nn.GELU(),
            Linear(n_mlp, n_state),
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), mask=mask, mask_pad=mask_pad, freqs_cis=freqs_cis)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


# =============================================================================
# Encoder
# =============================================================================

class AudioEncoderV2(nn.Module):
    """Audio encoder with RoPE positional encoding and FSMN attention."""

    def __init__(
        self,
        n_mels: int = 128,
        n_state: int = 1280,
        n_head: int = 20,
        n_layer: int = 6,
        stride: int = 2,
    ):
        super().__init__()
        self.stride = stride

        # Convolutional frontend
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        # RoPE frequencies (head_dim=64 for 1280/20=64)
        self.freqs_cis = precompute_freqs_cis(64, 1024 * 2)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head)
            for _ in range(n_layer)
        ])

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Mel spectrogram (B, n_mels, T)
            x_len: Lengths of each mel spectrogram (B,)

        Returns:
            Tuple of (encoded_features, output_lengths)
        """
        T = x.shape[-1]

        # Conv1 with masking
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)
        x = F.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        x_slen = (T + 2 - 1 * (3 - 1) - 1) // self.stride + 1

        # Conv2 with masking
        mask = make_non_pad_mask(x_len, x_slen).unsqueeze(1)
        x = F.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        x_slen = (x_slen + 2 - 1 * (3 - 1) - 1) // self.stride + 1

        # Prepare for transformer
        mask = make_non_pad_mask(x_len, x_slen).unsqueeze(1)
        x = x.permute(0, 2, 1)  # (B, T // 4, n_state)

        freqs_cis = self.freqs_cis.to(x.device)
        mask_pad = mask.transpose(1, 2)
        mask = mask_to_bias(mask, x.dtype)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask.unsqueeze(1), mask_pad, freqs_cis[:x.size(1)])

        return x, x_len


# =============================================================================
# Quantizer
# =============================================================================

class FSQCodebook(nn.Module):
    """
    Finite Scalar Quantization (FSQ) codebook.

    Projects input to 8 dimensions and quantizes each to ternary values {0, 1, 2}.
    Final token index = sum(value * 3^position) for each dimension.

    With 8 dimensions and 3 values each: 3^8 = 6561 possible codes.
    """

    def __init__(self, dim: int = 1280, level: int = 3):
        super().__init__()
        self.project_down = nn.Linear(dim, 8)
        self.level = level
        self.embed = None  # Not used in inference

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to token indices.

        Args:
            x: Input tensor (B, T, D)

        Returns:
            Token indices (B, T)
        """
        x_shape = x.shape

        # Flatten for processing
        x = x.reshape(-1, x.shape[-1])

        # Project to low dimension
        h = self.project_down(x).float()

        # FSQ quantization: tanh -> scale -> round -> shift to {0, 1, 2}
        h = h.tanh()
        h = h * 0.9990000128746033  # Scale to avoid exact ±1
        h = h.round() + 1  # Now in {0, 1, 2}

        # Compute indices using base-3
        powers = torch.pow(
            self.level,
            torch.arange(2 ** self.level, device=x.device, dtype=h.dtype)
        )
        mu = torch.sum(h * powers.unsqueeze(0), dim=-1)

        # Reshape back
        ind = mu.reshape(x_shape[0], x_shape[1]).int()
        return ind


class FSQVectorQuantization(nn.Module):
    """Vector quantization wrapper using FSQ codebook."""

    def __init__(self, dim: int = 1280, codebook_size: int = SPEECH_VOCAB_SIZE):
        super().__init__()
        assert 3 ** 8 == codebook_size
        self._codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to token indices."""
        return self._codebook.encode(x)


# =============================================================================
# Main S3Tokenizer
# =============================================================================

class S3Tokenizer(nn.Module):
    """
    S3 Speech Tokenizer V2.

    Pure PyTorch implementation matching the s3tokenizer package.
    Converts audio waveforms to discrete speech tokens using:
    1. Mel spectrogram extraction (128 mels)
    2. AudioEncoderV2 with RoPE and FSMN attention
    3. FSQ quantization (3^8 = 6561 codes)

    Args:
        n_mels: Number of mel bands (128)
        n_audio_state: Encoder dimension (1280)
        n_audio_head: Number of attention heads (20)
        n_audio_layer: Number of encoder blocks (6)
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        n_mels: int = 128,
        n_audio_state: int = 1280,
        n_audio_head: int = 20,
        n_audio_layer: int = 6,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.n_fft = 400
        self.hop_length = S3_HOP

        # Mel filter banks
        mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=n_mels,
        )
        self.register_buffer("_mel_filters", torch.from_numpy(mel_filters).float())

        # STFT window
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # Encoder
        self.encoder = AudioEncoderV2(
            n_mels=n_mels,
            n_state=n_audio_state,
            n_head=n_audio_head,
            n_layer=n_audio_layer,
            stride=2,
        )

        # Quantizer
        self.quantizer = FSQVectorQuantization(
            dim=n_audio_state,
            codebook_size=SPEECH_VOCAB_SIZE,
        )

    @property
    def device(self) -> torch.device:
        return self._mel_filters.device

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ) -> torch.Tensor:
        """
        Compute log-mel spectrogram.

        Args:
            audio: Audio waveform (B, T) or (T,)
            padding: Zero-padding to add

        Returns:
            Log-mel spectrogram (B, n_mels, T)
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))

        # STFT
        stft = torch.stft(
            audio,
            self.n_fft,
            S3_HOP,
            window=self.window.to(audio.device),
            return_complex=True,
        )

        # Power spectrogram
        magnitudes = stft[..., :-1].abs() ** 2

        # Mel spectrogram
        mel_spec = self._mel_filters.to(audio.device) @ magnitudes

        # Log scale with clamping
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def pad_audio(self, wavs: List[torch.Tensor], sr: int = S3_SR) -> List[torch.Tensor]:
        """
        Pad audio to token-aligned lengths.

        Args:
            wavs: List of audio tensors
            sr: Sample rate

        Returns:
            List of padded audio tensors
        """
        processed = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            # Compute required length
            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = math.ceil(n_tokens)
            intended_len = int(n_tokens * (sr / S3_TOKEN_RATE))

            # Pad
            if wav.shape[1] < intended_len:
                wav = F.pad(wav, (0, intended_len - wav.shape[1]))

            processed.append(wav)

        return processed

    def quantize(
        self,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize mel spectrograms to tokens.

        Args:
            mels: Mel spectrograms (B, n_mels, T)
            mel_lens: Mel lengths (B,)

        Returns:
            Tuple of (tokens, token_lens)
        """
        # Encode
        encoded, code_lens = self.encoder(mels, mel_lens)

        # Quantize
        indices = self.quantizer.encode(encoded)

        return indices, code_lens

    @torch.no_grad()
    def forward(
        self,
        wavs: List[torch.Tensor],
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize audio waveforms.

        Args:
            wavs: List of audio waveforms at 16kHz
            max_len: Maximum token length to return

        Returns:
            Tuple of (speech_tokens, token_lengths)
        """
        # Process each waveform
        mels = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav).float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)

            if max_len is not None:
                mel = mel[..., :max_len * 4]  # 4 mel frames per token

            mels.append(mel.squeeze(0))

        # Pad mels to same length
        max_mel_len = max(m.shape[1] for m in mels)
        padded_mels = []
        mel_lens = []

        for mel in mels:
            mel_len = mel.shape[1]
            mel_lens.append(mel_len)

            if mel_len < max_mel_len:
                mel = F.pad(mel, (0, max_mel_len - mel_len))

            padded_mels.append(mel)

        # Stack
        mels = torch.stack(padded_mels, dim=0)
        mel_lens = torch.tensor(mel_lens, device=self.device)

        # Quantize
        tokens, token_lens = self.quantize(mels, mel_lens)

        return tokens.long(), token_lens.long()


def drop_invalid_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """
    Drop tokens outside valid vocabulary range.

    Args:
        tokens: Token tensor

    Returns:
        Filtered tokens
    """
    return tokens[tokens < SPEECH_VOCAB_SIZE]
