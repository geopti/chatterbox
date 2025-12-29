"""
Audio utilities for loading and saving audio files.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch


def load_wav(
    path: Union[str, Path],
    sr: int = 24000,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.

    Args:
        path: Path to audio file
        sr: Target sample rate (resamples if different)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for audio loading")

    audio, sample_rate = librosa.load(str(path), sr=sr)
    return audio, sample_rate


def save_wav(
    path: Union[str, Path],
    audio: Union[np.ndarray, torch.Tensor],
    sr: int = 24000,
) -> None:
    """
    Save audio to a WAV file.

    Args:
        path: Output path
        audio: Audio data as numpy array or torch tensor
        sr: Sample rate
    """
    import scipy.io.wavfile as wavfile

    # Convert to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    # Squeeze if batched
    if audio.ndim == 2:
        audio = audio.squeeze(0)

    # Ensure audio is in [-1, 1] range
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    # Save
    wavfile.write(str(path), sr, audio_int16)


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to a different sample rate.

    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for resampling")

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
