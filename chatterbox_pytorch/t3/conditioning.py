"""
Conditioning encoder for T3 model.

Handles speaker embeddings, emotion control, and speech prompt conditioning.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import T3Config
from .perceiver import Perceiver


@dataclass
class T3Cond:
    """
    Container for T3 conditioning data.

    Attributes:
        speaker_emb: Speaker embedding from VoiceEncoder, shape (B, 256)
        clap_emb: CLAP embedding for audio conditioning (unused)
        cond_prompt_speech_tokens: Speech tokens from reference audio, shape (B, T)
        cond_prompt_speech_emb: Embedded speech tokens, shape (B, T, dim)
        emotion_adv: Emotion exaggeration factor, 0.0-1.0
    """

    speaker_emb: Tensor
    clap_emb: Optional[Tensor] = None
    cond_prompt_speech_tokens: Optional[Tensor] = None
    cond_prompt_speech_emb: Optional[Tensor] = None
    emotion_adv: Optional[Tensor] = None

    def to(self, device=None, dtype=None) -> "T3Cond":
        """Move all tensors to device and dtype."""
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                # Don't cast dtype for integer tensors
                is_float = value.dtype.is_floating_point
                new_dtype = dtype if is_float else None
                setattr(self, key, value.to(device=device, dtype=new_dtype))
        return self


class T3CondEnc(nn.Module):
    """
    Conditioning encoder for T3.

    Processes all non-text conditioning inputs:
    - Speaker embedding (from VoiceEncoder)
    - Speech prompt tokens (from reference audio)
    - Emotion/exaggeration control

    Args:
        config: T3 configuration
    """

    def __init__(self, config: T3Config):
        super().__init__()
        self.config = config

        # Speaker embedding projection
        self.spkr_enc = nn.Linear(config.speaker_embed_size, config.hidden_size)

        # Emotion/exaggeration projection (optional)
        self.emotion_adv_fc = None
        if config.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, config.hidden_size, bias=False)

        # Perceiver resampler for speech prompt tokens (optional)
        self.perceiver = None
        if config.use_perceiver_resampler:
            self.perceiver = Perceiver(
                n_queries=config.perceiver_n_queries,
                query_dim=config.hidden_size,
                n_heads=config.perceiver_n_heads,
            )

    def forward(self, cond: T3Cond) -> Tensor:
        """
        Encode conditioning data.

        Args:
            cond: T3Cond container with conditioning data

        Returns:
            Conditioning embeddings of shape (B, cond_len, hidden_size)
        """
        # Project speaker embedding: (B, 256) -> (B, 1, dim)
        speaker_cond = self.spkr_enc(cond.speaker_emb.view(-1, self.config.speaker_embed_size))
        speaker_cond = speaker_cond.unsqueeze(1)  # (B, 1, dim)

        # Empty tensor for unused conditions
        B, dim = speaker_cond.shape[0], speaker_cond.shape[2]
        empty = torch.zeros(B, 0, dim, device=speaker_cond.device, dtype=speaker_cond.dtype)

        # CLAP embedding (not implemented)
        clap_cond = empty

        # Speech prompt conditioning
        if cond.cond_prompt_speech_emb is not None:
            speech_prompt_cond = cond.cond_prompt_speech_emb
            if self.perceiver is not None:
                # Compress speech tokens using perceiver
                speech_prompt_cond = self.perceiver(speech_prompt_cond)
        else:
            speech_prompt_cond = empty

        # Emotion conditioning
        if self.emotion_adv_fc is not None and cond.emotion_adv is not None:
            emotion_cond = self.emotion_adv_fc(cond.emotion_adv.view(-1, 1, 1))
        else:
            emotion_cond = empty

        # Concatenate all conditions
        cond_embeds = torch.cat([
            speaker_cond,
            clap_cond,
            speech_prompt_cond,
            emotion_cond,
        ], dim=1)

        return cond_embeds
