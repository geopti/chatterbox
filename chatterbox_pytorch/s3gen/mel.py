"""
Mel spectrogram utilities for S3Gen.
"""

import torch
import torch.nn.functional as F
import numpy as np

# S3Gen operates at 24kHz
S3GEN_SR = 24000

# Mel spectrogram parameters
MEL_PARAMS = {
    "n_fft": 1920,
    "hop_length": 480,
    "win_length": 1920,
    "n_mels": 80,
    "fmin": 0,
    "fmax": 8000,
    "sample_rate": 24000,
}


def get_mel_filters(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float = None,
) -> np.ndarray:
    """
    Create mel filter bank.

    Args:
        sr: Sample rate
        n_fft: FFT size
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel filter bank of shape (n_mels, n_fft // 2 + 1)
    """
    try:
        import librosa
        return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    except ImportError:
        # Fall back to manual implementation
        fmax = fmax or sr / 2

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel points
        mel_low = hz_to_mel(fmin)
        mel_high = hz_to_mel(fmax)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # FFT bins
        fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        # Create filter bank
        filters = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            left = fft_bins[i]
            center = fft_bins[i + 1]
            right = fft_bins[i + 2]

            for j in range(left, center):
                filters[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filters[i, j] = (right - j) / (right - center)

        return filters


def mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 1920,
    hop_length: int = 480,
    win_length: int = 1920,
    n_mels: int = 80,
    sample_rate: int = 24000,
    fmin: float = 0,
    fmax: float = 8000,
) -> torch.Tensor:
    """
    Compute mel spectrogram.

    Args:
        audio: Audio tensor of shape (batch, samples) or (samples,)
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel bands
        sample_rate: Sample rate
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel spectrogram of shape (batch, n_mels, time)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    device = audio.device

    # Create window
    window = torch.hann_window(win_length, device=device)

    # STFT
    stft = torch.stft(
        audio,
        n_fft,
        hop_length,
        win_length,
        window=window,
        return_complex=True,
        pad_mode="reflect",
    )

    # Magnitude
    magnitude = stft.abs()

    # Mel filter bank
    mel_filters = get_mel_filters(sample_rate, n_fft, n_mels, fmin, fmax)
    mel_filters = torch.from_numpy(mel_filters).float().to(device)

    # Apply mel filters
    mel = torch.matmul(mel_filters, magnitude)

    # Log scale
    mel = torch.clamp(mel, min=1e-5)
    mel = torch.log(mel)

    return mel
