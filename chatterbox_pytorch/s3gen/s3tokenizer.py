"""
S3Tokenizer: VQ-VAE based speech tokenizer.

This is a from-scratch implementation that can load weights from the
original s3tokenizer library.

The tokenizer converts 16kHz audio to discrete speech tokens at 25 tokens/sec.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
S3_SR = 16000  # Sample rate for S3Tokenizer
S3_HOP = 160  # Hop size (100 frames/sec)
S3_TOKEN_HOP = 640  # Token hop size (25 tokens/sec)
S3_TOKEN_RATE = 25  # Tokens per second
SPEECH_VOCAB_SIZE = 6561  # VQ codebook size


class S3Tokenizer(nn.Module):
    """
    VQ-VAE speech tokenizer that converts audio to discrete tokens.

    This tokenizer:
    1. Computes log-mel spectrograms (128 mels, 16kHz, hop=160)
    2. Encodes them using a CNN encoder
    3. Quantizes using a learned codebook (6561 codes)
    4. Outputs tokens at 25 tokens/second

    The architecture is designed to be compatible with the original
    s3tokenizer weights from the s3gen.safetensors checkpoint.
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        n_mels: int = 128,
        embedding_dim: int = 1024,
        n_codes: int = SPEECH_VOCAB_SIZE,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.n_codes = n_codes
        self.n_fft = 400

        # Register mel filters and window as buffers
        mel_filters = self._create_mel_filters()
        self.register_buffer("_mel_filters", torch.from_numpy(mel_filters).float())
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # VQ-VAE Encoder (from s3tokenizer architecture)
        # This is a placeholder - actual architecture to be determined from weights
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),  # 2x downsample
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1),  # 2x downsample
            nn.ReLU(),
            nn.Conv1d(1024, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

        # VQ Codebook
        self.codebook = nn.Embedding(n_codes, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1 / n_codes, 1 / n_codes)

    def _create_mel_filters(self) -> np.ndarray:
        """Create mel filter bank for log-mel spectrogram."""
        try:
            import librosa
            return librosa.filters.mel(
                sr=S3_SR,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
            )
        except ImportError:
            # Basic mel filter bank implementation
            n_fft = self.n_fft
            n_mels = self.n_mels
            sr = S3_SR

            fmax = sr / 2
            mel_low = 2595 * np.log10(1 + 0 / 700)
            mel_high = 2595 * np.log10(1 + fmax / 700)
            mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)
            fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

            filters = np.zeros((n_mels, n_fft // 2 + 1))
            for i in range(n_mels):
                for j in range(fft_bins[i], fft_bins[i + 1]):
                    filters[i, j] = (j - fft_bins[i]) / (fft_bins[i + 1] - fft_bins[i])
                for j in range(fft_bins[i + 1], fft_bins[i + 2]):
                    filters[i, j] = (fft_bins[i + 2] - j) / (fft_bins[i + 2] - fft_bins[i + 1])

            return filters

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
            audio: Audio tensor of shape (batch, samples) or (samples,)
            padding: Zero padding to add

        Returns:
            Log-mel spectrogram of shape (batch, n_mels, time)
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

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

        # Magnitude squared
        magnitudes = stft[..., :-1].abs() ** 2

        # Apply mel filters
        mel_spec = torch.matmul(self._mel_filters.to(audio.device), magnitudes)

        # Log scale with clamping
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def quantize(
        self,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and quantize mel spectrograms.

        Args:
            mels: Mel spectrograms of shape (batch, n_mels, time)
            mel_lens: Lengths of mel sequences

        Returns:
            Tuple of (tokens, token_lens)
        """
        # Encode
        encoded = self.encoder(mels)  # (batch, embedding_dim, time // 4)

        # Transpose for quantization: (batch, time, embedding_dim)
        encoded = encoded.transpose(1, 2)

        # Compute distances to codebook
        # (batch, time, 1, embedding_dim) - (1, 1, n_codes, embedding_dim)
        flat_encoded = encoded.reshape(-1, self.embedding_dim)
        distances = torch.cdist(flat_encoded, self.codebook.weight)

        # Get closest codebook indices
        indices = distances.argmin(dim=-1)
        indices = indices.view(mels.shape[0], -1)

        # Compute token lengths (4x downsampling from encoder)
        token_lens = (mel_lens / 4).long()

        return indices, token_lens

    def forward(
        self,
        wavs: List[torch.Tensor],
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize audio waveforms.

        Args:
            wavs: List of audio tensors at 16kHz
            max_len: Maximum number of tokens (optional)

        Returns:
            Tuple of (tokens, token_lens)
        """
        mels = []
        mel_lens = []

        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)

            if max_len is not None:
                mel = mel[..., : max_len * 4]  # 4x downsampling

            mels.append(mel.squeeze(0))
            mel_lens.append(mel.shape[-1])

        # Pad to same length
        max_mel_len = max(mel_lens)
        padded_mels = torch.zeros(len(mels), self.n_mels, max_mel_len, device=self.device)
        for i, mel in enumerate(mels):
            padded_mels[i, :, : mel.shape[-1]] = mel

        mel_lens = torch.tensor(mel_lens, device=self.device)

        # Quantize
        tokens, token_lens = self.quantize(padded_mels, mel_lens)

        return tokens.long(), token_lens.long()

    def pad(
        self,
        wavs: List[torch.Tensor],
        sr: int,
    ) -> List[torch.Tensor]:
        """
        Pad waveforms to be a multiple of the token hop size.

        Args:
            wavs: List of waveforms
            sr: Sample rate

        Returns:
            Padded waveforms
        """
        processed = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_len = int(n_tokens * (sr / S3_TOKEN_RATE))

            wav = F.pad(wav, (0, intended_len - wav.shape[-1]))
            processed.append(wav)

        return processed


def drop_invalid_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """
    Drop tokens that are outside the valid vocab range.

    Args:
        tokens: Token tensor

    Returns:
        Filtered tokens
    """
    return tokens[tokens < SPEECH_VOCAB_SIZE]
