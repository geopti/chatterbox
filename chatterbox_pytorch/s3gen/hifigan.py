"""
HiFi-GAN vocoder with source-filter synthesis.

Converts mel-spectrograms to waveforms using neural source-filter modeling.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

from ..core.activations import Snake


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for same output size."""
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    """Residual block with Snake activation."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
    ):
        super().__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.activations1 = nn.ModuleList()
        self.activations2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
            )
            self.activations1.append(Snake(channels))
            self.activations2.append(Snake(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.convs1)):
            xt = self.activations1[i](x)
            xt = self.convs1[i](xt)
            xt = self.activations2[i](xt)
            xt = self.convs2[i](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


class SineGen(nn.Module):
    """
    Sine waveform generator for source-filter synthesis.

    Generates harmonic sine waves based on F0 (fundamental frequency).
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def _f0_to_uv(self, f0: torch.Tensor) -> torch.Tensor:
        """Generate voiced/unvoiced decision from F0."""
        return (f0 > self.voiced_threshold).float()

    @torch.no_grad()
    def forward(self, f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate sine waveforms from F0.

        Args:
            f0: Fundamental frequency of shape (batch, 1, time)

        Returns:
            Tuple of (sine_waves, uv, noise)
        """
        batch_size = f0.size(0)
        device = f0.device

        # Create frequency matrix for harmonics
        F_mat = torch.zeros(batch_size, self.harmonic_num + 1, f0.size(-1), device=device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i : i + 1, :] = f0 * (i + 1) / self.sample_rate

        # Phase accumulation
        theta_mat = 2 * np.pi * torch.cumsum(F_mat, dim=-1) % (2 * np.pi)

        # Random initial phase
        phase_vec = torch.rand(batch_size, self.harmonic_num + 1, 1, device=device) * 2 * np.pi
        phase_vec[:, 0, :] = 0  # No random phase for fundamental

        # Generate sine waves
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)

        # Voiced/unvoiced mask
        uv = self._f0_to_uv(f0)

        # Noise for unvoiced regions
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # Apply UV mask
        sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise


class SourceModule(nn.Module):
    """Source module for harmonic + noise synthesis."""

    def __init__(
        self,
        sample_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 10,
    ):
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = noise_std

        self.sine_gen = SineGen(
            sample_rate,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold,
        )

        # Merge harmonics
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate source signal.

        Args:
            f0: F0 of shape (batch, time, 1)

        Returns:
            Tuple of (source, noise, uv)
        """
        with torch.no_grad():
            sine_waves, uv, _ = self.sine_gen(f0.transpose(1, 2))
            sine_waves = sine_waves.transpose(1, 2)
            uv = uv.transpose(1, 2)

        # Merge harmonics
        source = self.tanh(self.l_linear(sine_waves))

        # Noise for unvoiced
        noise = torch.randn_like(uv) * self.sine_amp / 3

        return source, noise, uv


class ConvRNNF0Predictor(nn.Module):
    """
    F0 predictor using Conv + RNN.

    Predicts fundamental frequency from mel-spectrogram.
    """

    def __init__(
        self,
        in_channels: int = 80,
        hidden_channels: int = 512,
        out_channels: int = 1,
        num_layers: int = 5,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )

        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict F0 from mel-spectrogram.

        Args:
            x: Mel-spectrogram of shape (batch, n_mels, time)

        Returns:
            F0 of shape (batch, 1, time)
        """
        for conv in self.convs:
            x = F.elu(conv(x))

        # Global average pooling per frame
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)

        # Ensure positive F0
        x = F.softplus(x) * 100  # Scale to reasonable F0 range

        return x


class HiFTGenerator(nn.Module):
    """
    HiFi-GAN with source-filter synthesis.

    Neural vocoder that generates waveforms from mel-spectrograms
    using neural source-filter modeling.
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 5, 3],
        upsample_kernel_sizes: List[int] = [16, 11, 7],
        istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Source module
        upsample_scale = int(np.prod(upsample_rates) * istft_params["hop_len"])
        self.m_source = SourceModule(
            sample_rate=sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold,
        )

        # F0 upsampling
        self.f0_upsamp = nn.Upsample(scale_factor=upsample_scale)

        # Input convolution
        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, base_channels, 7, padding=3)
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        base_channels // (2 ** i),
                        base_channels // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Source downsampling
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum = np.cumprod(downsample_rates)

        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum[::-1],
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes,
            )
        ):
            if u == 1:
                self.source_downs.append(
                    nn.Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1)
                )
            else:
                self.source_downs.append(
                    nn.Conv1d(
                        istft_params["n_fft"] + 2,
                        base_channels // (2 ** (i + 1)),
                        int(u * 2),
                        int(u),
                        padding=int(u // 2),
                    )
                )
            self.source_resblocks.append(ResBlock(base_channels // (2 ** (i + 1)), k, d))

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        # Output convolution
        self.conv_post = weight_norm(
            nn.Conv1d(ch, istft_params["n_fft"] + 2, 7, padding=3)
        )

        # Reflection pad and STFT window
        self.reflection_pad = nn.ReflectionPad1d((1, 0))

        from scipy.signal import get_window
        stft_window = get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32)
        self.register_buffer("stft_window", torch.from_numpy(stft_window))

        # F0 predictor
        if f0_predictor is None:
            f0_predictor = ConvRNNF0Predictor()
        self.f0_predictor = f0_predictor

    def _stft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT."""
        spec = torch.stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(x.device),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compute inverse STFT."""
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        inverse = torch.istft(
            torch.complex(real, imag),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(magnitude.device),
        )
        return inverse

    def decode(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Decode mel-spectrogram to waveform.

        Args:
            x: Mel-spectrogram of shape (batch, n_mels, time)
            s: Source signal of shape (batch, 1, samples)

        Returns:
            Waveform of shape (batch, samples)
        """
        # Source STFT
        s_real, s_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_real, s_imag], dim=1)

        # Process mel
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # Fuse source
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)

        # Split into magnitude and phase
        n_fft_half = self.istft_params["n_fft"] // 2 + 1
        magnitude = torch.exp(x[:, :n_fft_half, :])
        phase = torch.sin(x[:, n_fft_half:, :])

        # Inverse STFT
        audio = self._istft(magnitude, phase)
        audio = torch.clamp(audio, -self.audio_limit, self.audio_limit)

        return audio

    def forward(self, batch: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        speech_feat = batch["speech_feat"].transpose(1, 2).to(device)

        # Predict F0
        f0 = self.f0_predictor(speech_feat)

        # Generate source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)

        # Decode
        audio = self.decode(speech_feat, s)

        return audio, f0

    @torch.inference_mode()
    def inference(
        self,
        speech_feat: torch.Tensor,
        cache_source: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate waveform from mel-spectrogram.

        Args:
            speech_feat: Mel-spectrogram of shape (batch, time, n_mels)
            cache_source: Cached source for streaming

        Returns:
            Tuple of (waveform, source)
        """
        if speech_feat.dim() == 2:
            speech_feat = speech_feat.unsqueeze(0)

        # Transpose if needed
        if speech_feat.shape[-1] == 80:
            speech_feat = speech_feat.transpose(1, 2)

        # Predict F0 (output: batch, 1, time_mel)
        f0 = self.f0_predictor(speech_feat)

        # Upsample F0 to waveform sample rate
        # f0 is (batch, 1, time_mel), f0_upsamp expects (batch, channels, time)
        s = self.f0_upsamp(f0)  # (batch, 1, time_wav)
        # m_source expects (batch, time, 1), so transpose
        s = s.transpose(1, 2)  # (batch, time_wav, 1)
        s, _, _ = self.m_source(s)  # returns (batch, time, 1)
        # Transpose back to (batch, 1, time) for decode
        s = s.transpose(1, 2)  # (batch, 1, time)

        # Use cached source if provided
        if cache_source is not None and cache_source.shape[2] > 0:
            s[:, :, : cache_source.shape[2]] = cache_source

        # Decode expects s of shape (batch, 1, samples)
        audio = self.decode(speech_feat, s)

        return audio, s
