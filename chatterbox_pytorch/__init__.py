"""
Chatterbox-TTS: Pure PyTorch From-Scratch Implementation

A clean, readable re-implementation of the Chatterbox-TTS model
that uses no HuggingFace transformers library.
"""

from .tts import ChatterboxTTS
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

__version__ = "0.1.0"
__all__ = ["ChatterboxTTS", "ChatterboxMultilingualTTS", "SUPPORTED_LANGUAGES"]
