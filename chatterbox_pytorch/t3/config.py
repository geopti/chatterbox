"""
Configuration for T3 model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class T3Config:
    """
    Configuration for T3 (Token-to-Token) TTS model.

    This configuration defines all hyperparameters for the T3 model,
    including text/speech token vocabularies, transformer architecture,
    and conditioning settings.
    """

    # Text token settings
    start_text_token: int = 255
    stop_text_token: int = 0
    text_tokens_dict_size: int = 704
    max_text_tokens: int = 2048

    # Speech token settings
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    speech_tokens_dict_size: int = 8194
    max_speech_tokens: int = 4096

    # Transformer architecture (LLaMA 520M)
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    rms_norm_eps: float = 1e-5

    # RoPE settings (LLaMA3 style)
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 8.0
    rope_high_freq_factor: float = 4.0
    rope_low_freq_factor: float = 1.0
    rope_original_max_position_embeddings: int = 8192
    max_position_embeddings: int = 131072

    # Position embedding
    input_pos_emb: str = "learned"  # "learned" or None

    # Conditioning
    encoder_type: str = "voice_encoder"
    speaker_embed_size: int = 256
    speech_cond_prompt_len: int = 150
    use_perceiver_resampler: bool = True
    emotion_adv: bool = True

    # Perceiver settings
    perceiver_n_queries: int = 32
    perceiver_n_heads: int = 4

    @property
    def n_channels(self) -> int:
        """Model dimension (alias for hidden_size)."""
        return self.hidden_size

    @property
    def is_multilingual(self) -> bool:
        """Check if this is a multilingual model."""
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls) -> "T3Config":
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)

    @classmethod
    def multilingual(cls) -> "T3Config":
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=2454)
