"""
Voice Encoder for speaker embedding extraction.

LSTM-based model that extracts 256-dimensional speaker embeddings from audio.
Based on the speaker verification architecture from Real-Time-Voice-Cloning.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class VoiceEncoderConfig:
    """Configuration for VoiceEncoder."""

    # Audio parameters
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    num_mels: int = 40
    fmin: float = 0.0
    fmax: float = 8000.0

    # Model parameters
    ve_hidden_size: int = 256
    speaker_embed_size: int = 256
    ve_partial_frames: int = 160

    # Inference parameters
    ve_final_relu: bool = True
    normalized_mels: bool = True
    flatten_lstm_params: bool = True


def melspectrogram(
    wav: np.ndarray,
    config: VoiceEncoderConfig,
) -> np.ndarray:
    """
    Compute mel-spectrogram from waveform.

    Args:
        wav: Audio waveform as numpy array
        config: VoiceEncoder configuration

    Returns:
        Mel-spectrogram as numpy array of shape (num_mels, T)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for mel-spectrogram computation")

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.num_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )

    # Convert to log scale and normalize
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

    # Normalize to [0, 1] range
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    return mel


class VoiceEncoder(nn.Module):
    """
    LSTM-based voice encoder for speaker embedding extraction.

    Takes mel-spectrograms and produces L2-normalized speaker embeddings
    that can be used for speaker verification or voice conditioning.

    Args:
        config: VoiceEncoder configuration
    """

    def __init__(self, config: Optional[VoiceEncoderConfig] = None):
        super().__init__()
        if config is None:
            config = VoiceEncoderConfig()

        self.config = config

        # 3-layer LSTM
        self.lstm = nn.LSTM(
            input_size=config.num_mels,
            hidden_size=config.ve_hidden_size,
            num_layers=3,
            batch_first=True,
        )

        if config.flatten_lstm_params:
            self.lstm.flatten_parameters()

        # Projection to embedding space
        self.proj = nn.Linear(config.ve_hidden_size, config.speaker_embed_size)

        # Cosine similarity parameters (for speaker verification)
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """
        Compute speaker embeddings from mel-spectrograms.

        Args:
            mels: Mel-spectrograms of shape (batch, time, num_mels)
                  Time should be config.ve_partial_frames for partials.

        Returns:
            L2-normalized embeddings of shape (batch, speaker_embed_size)
        """
        if self.config.normalized_mels:
            if mels.min() < 0 or mels.max() > 1:
                raise ValueError(
                    f"Mels outside [0, 1] range: min={mels.min()}, max={mels.max()}"
                )

        # Pass through LSTM
        _, (hidden, _) = self.lstm(mels)

        # Take the last layer's hidden state
        raw_embeds = self.proj(hidden[-1])

        if self.config.ve_final_relu:
            raw_embeds = F.relu(raw_embeds)

        # L2 normalize
        embeds = F.normalize(raw_embeds, p=2, dim=1)

        return embeds

    def embeds_from_wavs(
        self,
        wavs: List[np.ndarray],
        sample_rate: int,
        as_speaker: bool = False,
        batch_size: int = 32,
        overlap: float = 0.5,
        min_coverage: float = 0.8,
    ) -> np.ndarray:
        """
        Compute speaker embeddings from audio waveforms.

        Args:
            wavs: List of audio waveforms as numpy arrays
            sample_rate: Sample rate of the audio
            as_speaker: If True, return averaged speaker embedding
            batch_size: Batch size for processing partials
            overlap: Overlap between partial utterances
            min_coverage: Minimum coverage for partial utterances

        Returns:
            Embeddings as numpy array of shape (n_utterances, embed_size)
            or (embed_size,) if as_speaker=True
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa is required for audio processing")

        # Resample if needed
        if sample_rate != self.config.sample_rate:
            wavs = [
                librosa.resample(
                    wav,
                    orig_sr=sample_rate,
                    target_sr=self.config.sample_rate,
                    res_type="kaiser_fast",
                )
                for wav in wavs
            ]

        # Compute mel-spectrograms
        mels = [melspectrogram(wav, self.config).T for wav in wavs]

        return self.embeds_from_mels(
            mels,
            as_speaker=as_speaker,
            batch_size=batch_size,
            overlap=overlap,
            min_coverage=min_coverage,
        )

    def embeds_from_mels(
        self,
        mels: List[np.ndarray],
        as_speaker: bool = False,
        batch_size: int = 32,
        overlap: float = 0.5,
        min_coverage: float = 0.8,
    ) -> np.ndarray:
        """
        Compute speaker embeddings from mel-spectrograms.

        Args:
            mels: List of mel-spectrograms of shape (T, num_mels)
            as_speaker: If True, return averaged speaker embedding
            batch_size: Batch size for processing
            overlap: Overlap between partial utterances
            min_coverage: Minimum coverage for final partial

        Returns:
            Embeddings array
        """
        # Compute frame step from overlap
        frame_step = int(self.config.ve_partial_frames * (1 - overlap))

        # Extract partials and their utterance indices
        all_partials = []
        utt_counts = []

        for mel in mels:
            n_frames = mel.shape[0]
            n_partials = max(1, (n_frames - self.config.ve_partial_frames) // frame_step + 1)

            # Check if last partial has enough coverage
            remainder = (n_frames - self.config.ve_partial_frames) % frame_step
            if remainder / self.config.ve_partial_frames >= min_coverage:
                n_partials += 1

            # Pad if needed
            target_len = self.config.ve_partial_frames + frame_step * (n_partials - 1)
            if target_len > n_frames:
                mel = np.pad(mel, ((0, target_len - n_frames), (0, 0)))

            # Extract partials
            for i in range(n_partials):
                start = i * frame_step
                end = start + self.config.ve_partial_frames
                all_partials.append(mel[start:end])

            utt_counts.append(n_partials)

        # Stack and convert to tensor
        partials_tensor = torch.from_numpy(np.stack(all_partials)).float()

        # Process in batches
        with torch.inference_mode():
            partial_embeds = []
            for i in range(0, len(partials_tensor), batch_size):
                batch = partials_tensor[i : i + batch_size].to(self.device)
                embeds = self(batch)
                partial_embeds.append(embeds.cpu())

            partial_embeds = torch.cat(partial_embeds, dim=0).numpy()

        # Average partials within each utterance
        utt_embeds = []
        idx = 0
        for count in utt_counts:
            utt_embed = partial_embeds[idx : idx + count].mean(axis=0)
            utt_embed = utt_embed / np.linalg.norm(utt_embed)  # Re-normalize
            utt_embeds.append(utt_embed)
            idx += count

        utt_embeds = np.stack(utt_embeds)

        if as_speaker:
            # Average all utterance embeddings
            speaker_embed = utt_embeds.mean(axis=0)
            speaker_embed = speaker_embed / np.linalg.norm(speaker_embed)
            return speaker_embed

        return utt_embeds

    @staticmethod
    def voice_similarity(embed_a: np.ndarray, embed_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embed_a: First embedding
            embed_b: Second embedding

        Returns:
            Cosine similarity score
        """
        return float(np.dot(embed_a, embed_b))
