"""
S3Gen: Speech token to waveform generation.

This module combines:
- S3Tokenizer: VQ-VAE speech tokenizer
- CAMPPlus: X-vector speaker encoder
- UpsampleConformerEncoder: Token encoder with 2x upsampling
- CausalConditionalCFM: Flow matching for mel generation
- HiFTGenerator: Vocoder for waveform synthesis
"""

import logging
from functools import lru_cache
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from .s3tokenizer import S3Tokenizer, S3_SR, SPEECH_VOCAB_SIZE
from .xvector import CAMPPlus
from .conformer import UpsampleConformerEncoder
from .cfm import CausalConditionalCFM
from .decoder import ConditionalDecoder
from .hifigan import HiFTGenerator, ConvRNNF0Predictor
from .mel import mel_spectrogram, S3GEN_SR
from .flow import CausalMaskedDiffWithXvec

logger = logging.getLogger(__name__)


@lru_cache(100)
def get_resampler(src_sr: int, dst_sr: int, device: torch.device):
    """Get cached resampler."""
    return torchaudio.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(nn.Module):
    """
    Token-to-mel module using flow matching.

    Converts S3 speech tokens to mel-spectrograms using:
    - Speaker encoder (CAMPPlus) for voice cloning
    - Conformer encoder for token processing
    - CFM decoder for mel generation
    """

    def __init__(self, meanflow: bool = False):
        super().__init__()

        self.meanflow = meanflow

        # Speech tokenizer (for reference encoding)
        self.tokenizer = S3Tokenizer()

        # Speaker encoder
        self.speaker_encoder = CAMPPlus(memory_efficient=False)

        # Build encoder
        encoder = UpsampleConformerEncoder(
            input_size=512,
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
        )

        # Build CFM decoder using original architecture
        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
        )

        decoder = CausalConditionalCFM(
            in_channels=240,
            n_spks=1,
            spk_emb_dim=80,
            estimator=estimator,
        )

        # Flow module
        self.flow = CausalMaskedDiffWithXvec(
            input_size=512,
            output_size=80,
            spk_embed_dim=192,
            vocab_size=SPEECH_VOCAB_SIZE,
            token_mel_ratio=2,
            pre_lookahead_len=3,
            encoder=encoder,
            decoder=decoder,
        )

    @property
    def device(self) -> torch.device:
        return next(self.tokenizer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.flow.parameters()).dtype

    def embed_ref(
        self,
        ref_wav: Union[torch.Tensor, np.ndarray],
        ref_sr: int,
        device: str = "auto",
        ref_fade_out: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reference embeddings for voice cloning.

        Args:
            ref_wav: Reference waveform
            ref_sr: Reference sample rate
            device: Target device
            ref_fade_out: Whether to fade out reference

        Returns:
            Dictionary with:
            - prompt_token: Reference speech tokens
            - prompt_token_len: Token lengths
            - prompt_feat: Reference mel features
            - prompt_feat_len: Feature lengths
            - embedding: Speaker embedding
        """
        device = self.device if device == "auto" else torch.device(device)

        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)

        if ref_wav.dim() == 1:
            ref_wav = ref_wav.unsqueeze(0)

        if ref_wav.size(1) > 10 * ref_sr:
            logger.warning("S3Gen received ref longer than 10s")

        # Resample to 24kHz for mel extraction
        ref_wav_24 = ref_wav
        if ref_sr != S3GEN_SR:
            ref_wav_24 = get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)
        ref_wav_24 = ref_wav_24.to(device=device, dtype=self.dtype)

        # Extract mel spectrogram
        ref_mels_24 = mel_spectrogram(ref_wav_24).transpose(1, 2).to(dtype=self.dtype)
        ref_mels_24_len = None

        # Resample to 16kHz for tokenization and speaker embedding
        ref_wav_16 = ref_wav
        if ref_sr != S3_SR:
            ref_wav_16 = get_resampler(ref_sr, S3_SR, device)(ref_wav)

        # Speaker embedding
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16.to(dtype=self.dtype))

        # Tokenize reference
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16.float())

        # Ensure mel_len = 2 * token_len
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logger.warning("Reference mel length != 2 * reference token length")
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return {
            "prompt_token": ref_speech_tokens.to(device),
            "prompt_token_len": ref_speech_token_lens,
            "prompt_feat": ref_mels_24,
            "prompt_feat_len": ref_mels_24_len,
            "embedding": ref_x_vector,
        }

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        ref_dict: Optional[Dict[str, torch.Tensor]] = None,
        n_cfm_timesteps: Optional[int] = None,
        finalize: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
        noised_mels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate mel-spectrogram from speech tokens.

        Args:
            speech_tokens: Speech tokens of shape (batch, n_tokens)
            ref_wav: Reference waveform (optional if ref_dict provided)
            ref_sr: Reference sample rate
            ref_dict: Pre-computed reference embeddings
            n_cfm_timesteps: Number of CFM steps
            finalize: Whether streaming is finished
            speech_token_lens: Token lengths
            noised_mels: Pre-noised mels for meanflow

        Returns:
            Mel-spectrogram of shape (batch, n_mels, time)
        """
        assert (ref_wav is None) ^ (ref_dict is None), \
            "Must provide exactly one of ref_wav or ref_dict"

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            # Type/device casting
            for key in list(ref_dict):
                if isinstance(ref_dict[key], np.ndarray):
                    ref_dict[key] = torch.from_numpy(ref_dict[key])
                if torch.is_tensor(ref_dict[key]):
                    ref_dict[key] = ref_dict[key].to(device=self.device, dtype=self.dtype)

        speech_tokens = torch.atleast_2d(speech_tokens)

        if speech_token_lens is None:
            speech_token_lens = torch.LongTensor([st.size(-1) for st in speech_tokens]).to(self.device)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            noised_mels=noised_mels,
            n_timesteps=n_cfm_timesteps or (2 if self.meanflow else 10),
            meanflow=self.meanflow,
            **ref_dict,
        )

        return output_mels


class S3Gen(S3Token2Mel):
    """
    Full S3Gen model: Speech tokens to waveform.

    Combines token-to-mel (CFM) with mel-to-waveform (HiFi-GAN).
    """

    ignore_state_dict_missing = ("tokenizer._mel_filters", "tokenizer.window")

    def __init__(self, meanflow: bool = False):
        super().__init__(meanflow)

        # F0 predictor and vocoder
        f0_predictor = ConvRNNF0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # Fade-in to reduce artifacts
        n_trim = S3GEN_SR // 50  # 20ms
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)

        self.estimator_dtype = "fp32"

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        ref_dict: Optional[Dict[str, torch.Tensor]] = None,
        finalize: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
        skip_vocoder: bool = False,
        n_cfm_timesteps: Optional[int] = None,
        noised_mels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate waveform from speech tokens.

        Args:
            speech_tokens: Speech tokens
            ref_wav: Reference waveform
            ref_sr: Reference sample rate
            ref_dict: Pre-computed reference embeddings
            finalize: Whether streaming is finished
            speech_token_lens: Token lengths
            skip_vocoder: Return mel instead of waveform
            n_cfm_timesteps: Number of CFM steps
            noised_mels: Pre-noised mels for meanflow

        Returns:
            Waveform of shape (batch, samples) or mel if skip_vocoder
        """
        output_mels = super().forward(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            finalize=finalize,
            n_cfm_timesteps=n_cfm_timesteps,
            noised_mels=noised_mels,
        )

        if skip_vocoder:
            return output_mels

        # Run vocoder
        hift_cache_source = torch.zeros(1, 1, 0).to(self.device)
        output_wavs, *_ = self.mel2wav.inference(
            speech_feat=output_mels,
            cache_source=hift_cache_source,
        )

        if not self.training:
            # Fade in to reduce spillover from reference
            output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens: torch.LongTensor,
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        ref_dict: Optional[Dict[str, torch.Tensor]] = None,
        n_cfm_timesteps: Optional[int] = None,
        finalize: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate mel-spectrogram only."""
        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)
        noise = None
        if self.meanflow:
            noise = torch.randn(1, 80, speech_tokens.size(-1) * 2, dtype=self.dtype, device=self.device)

        output_mels = S3Token2Mel.forward(
            self,
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=finalize,
            noised_mels=noise,
        )
        return output_mels

    @torch.inference_mode()
    def hift_inference(
        self,
        speech_feat: torch.Tensor,
        cache_source: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run vocoder on mel-spectrogram."""
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(device=self.device, dtype=self.dtype)
        return self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens: torch.LongTensor,
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        ref_dict: Optional[Dict[str, torch.Tensor]] = None,
        drop_invalid_tokens: bool = True,
        n_cfm_timesteps: Optional[int] = None,
        speech_token_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full inference: tokens to waveform.

        Args:
            speech_tokens: Speech tokens
            ref_wav: Reference waveform
            ref_sr: Reference sample rate
            ref_dict: Pre-computed reference embeddings
            drop_invalid_tokens: Filter out-of-range tokens
            n_cfm_timesteps: Number of CFM steps
            speech_token_lens: Token lengths

        Returns:
            Tuple of (waveform, source_signal)
        """
        output_mels = self.flow_inference(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
        )

        output_mels = output_mels.to(dtype=self.dtype)
        output_wavs, output_sources = self.hift_inference(output_mels, None)

        # Fade in to reduce artifacts
        output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs, output_sources
