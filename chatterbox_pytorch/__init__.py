"""
Chatterbox-TTS: Pure PyTorch From-Scratch Implementation

A clean, readable re-implementation of the Chatterbox-TTS model
that uses no HuggingFace transformers library.
"""

from .tts import ChatterboxTTS

__version__ = "0.1.0"
__all__ = ["ChatterboxTTS"]
