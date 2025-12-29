"""
CAMPPlus X-vector speaker encoder.

Extracts 192-dimensional speaker embeddings from audio.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    """Time-Delay Neural Network layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class CAMDenseBlock(nn.Module):
    """Channel Attention Module with Dense connections."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_in = in_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(layer_in, growth_rate, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(growth_rate),
                    nn.ReLU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class TransitLayer(nn.Module):
    """Transition layer for dimension reduction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class StatsPool(nn.Module):
    """Statistics pooling layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)


class CAMPPlus(nn.Module):
    """
    CAMPPlus speaker encoder.

    Extracts 192-dimensional speaker embeddings from 80-dim fbank features.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        memory_efficient: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.memory_efficient = memory_efficient

        # Initial projection
        self.layer1 = nn.Sequential(
            nn.Conv1d(feat_dim, 512, 5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        # TDNN layers
        self.layer2 = TDNN(512, 512, kernel_size=3, dilation=2)
        self.layer3 = TDNN(512, 512, kernel_size=3, dilation=3)
        self.layer4 = TDNN(512, 512, kernel_size=1)
        self.layer5 = TDNN(512, 1500, kernel_size=1)

        # Stats pooling
        self.stats_pool = StatsPool()

        # Output layers
        self.fc = nn.Linear(3000, embedding_size)
        self.bn_fc = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, time, feat_dim) or (batch, feat_dim, time)

        Returns:
            Speaker embeddings of shape (batch, embedding_size)
        """
        # Handle input format
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if x.shape[-1] == self.feat_dim:
            # (batch, time, feat_dim) -> (batch, feat_dim, time)
            x = x.transpose(1, 2)

        # Forward through layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Stats pooling
        x = self.stats_pool(x)

        # Output projection
        x = self.fc(x)
        x = self.bn_fc(x)

        return x

    @torch.inference_mode()
    def inference(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from waveform.

        Args:
            wav: Audio waveform at 16kHz

        Returns:
            Speaker embedding of shape (1, embedding_size)
        """
        # Compute fbank features
        fbank = self._compute_fbank(wav)

        # Get embedding
        embedding = self(fbank)

        return embedding

    def _compute_fbank(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Compute 80-dimensional fbank features.

        Args:
            wav: Audio at 16kHz

        Returns:
            Fbank features of shape (batch, time, 80)
        """
        try:
            import torchaudio
            import torchaudio.compliance.kaldi as kaldi

            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            fbank = kaldi.fbank(
                wav,
                num_mel_bins=80,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )

            # Normalize
            fbank = fbank - fbank.mean(dim=0, keepdim=True)

            return fbank.unsqueeze(0)

        except ImportError:
            # Fallback using librosa
            import numpy as np

            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            if wav.ndim == 2:
                wav = wav.squeeze(0)

            import librosa

            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=16000,
                n_fft=512,
                hop_length=160,
                n_mels=80,
            )
            fbank = librosa.power_to_db(mel, ref=np.max).T

            # Normalize
            fbank = fbank - fbank.mean(axis=0, keepdims=True)

            return torch.from_numpy(fbank).float().unsqueeze(0)
