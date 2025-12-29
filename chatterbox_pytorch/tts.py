"""
ChatterboxTTS: Main TTS interface.

Combines T3 (text-to-token) and S3Gen (token-to-waveform) models
for end-to-end text-to-speech synthesis with voice cloning.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import sys
# Add src to path to import original S3Gen (architecture matches checkpoint)
_src_path = str(Path(__file__).parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from .t3 import T3
from .t3.config import T3Config
from .t3.conditioning import T3Cond
# Use original S3Gen since architecture matches checkpoint exactly
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3tokenizer import S3_SR, drop_invalid_tokens, SPEECH_VOCAB_SIZE, S3Tokenizer
from chatterbox.models.s3gen import S3GEN_SR
from .voice_encoder import VoiceEncoder
from .tokenizers import EnTokenizer


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
    Normalize punctuation for TTS input.

    Cleans up text from LLMs or with uncommon punctuation.
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple spaces
    text = " ".join(text.split())

    # Replace uncommon punctuation
    replacements = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", '"'),
        (""", '"'),
        ("'", "'"),
        ("'", "'"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    # Add full stop if no ending punctuation
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen models.

    Contains:
    - T3 conditionals: speaker embedding, speech prompt tokens, emotion
    - S3Gen conditionals: prompt tokens, mel features, x-vector embedding
    """

    t3: T3Cond
    gen: dict

    def to(self, device: Union[str, torch.device]) -> "Conditionals":
        """Move conditionals to device."""
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        """Save conditionals to file."""
        arg_dict = {"t3": self.t3.__dict__, "gen": self.gen}
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath: Union[str, Path], map_location: str = "cpu") -> "Conditionals":
        """Load conditionals from file."""
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


class ChatterboxTTS:
    """
    Main Chatterbox TTS class.

    Provides end-to-end text-to-speech with voice cloning.
    """

    ENC_COND_LEN = 6 * S3_SR  # 6 seconds for T3 conditioning
    DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds for S3Gen conditioning

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Optional[Conditionals] = None,
    ):
        """
        Initialize ChatterboxTTS.

        Args:
            t3: T3 text-to-token model
            s3gen: S3Gen token-to-waveform model
            ve: VoiceEncoder for speaker embeddings
            tokenizer: Text tokenizer
            device: Target device
            conds: Pre-computed conditionals (optional)
        """
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds

    @classmethod
    def from_local(cls, ckpt_dir: Union[str, Path], device: str) -> "ChatterboxTTS":
        """
        Load model from local checkpoint directory.

        Args:
            ckpt_dir: Directory containing model checkpoints
            device: Target device

        Returns:
            Initialized ChatterboxTTS
        """
        ckpt_dir = Path(ckpt_dir)

        # Handle non-CUDA devices
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        # Load voice encoder
        from .utils.weight_loader import load_ve_weights, load_t3_weights

        ve = VoiceEncoder()
        missing, unexpected = load_ve_weights(ve, ckpt_dir / "ve.safetensors")
        if missing:
            print(f"VE missing keys: {missing[:5]}...")
        ve.to(device).eval()

        # Load T3 (our pure PyTorch implementation)
        t3 = T3()
        missing, unexpected = load_t3_weights(t3, ckpt_dir / "t3_cfg.safetensors")
        if missing:
            print(f"T3 missing keys: {missing[:5]}...")
        t3.to(device).eval()

        # Load S3Gen (using original implementation - architecture matches checkpoint)
        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        # Load tokenizer
        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        # Load built-in voice if available
        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: str) -> "ChatterboxTTS":
        """
        Load model from HuggingFace Hub.

        Args:
            device: Target device

        Returns:
            Initialized ChatterboxTTS
        """
        # Check MPS availability on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available - PyTorch not built with MPS.")
            else:
                print("MPS not available - macOS version < 12.3 or no MPS device.")
            device = "cpu"

        # Download all required files
        for fname in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fname)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        """
        Prepare conditioning from reference audio.

        Args:
            wav_fpath: Path to reference audio file
            exaggeration: Emotion exaggeration factor
        """
        # Load reference at 24kHz for S3Gen
        s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)

        # Resample to 16kHz for tokenization and voice encoder
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        # Trim reference for S3Gen
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Get speech prompt tokens for T3
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokenizer = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokenizer.forward(
                [ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Get voice encoder embedding
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        # Build T3 conditioning
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)

        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text: str,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """
        Generate speech from text.

        Args:
            text: Input text
            repetition_penalty: Token repetition penalty
            min_p: Min-p sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            audio_prompt_path: Path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature

        Returns:
            Generated audio waveform (tensor of shape [1, samples])
        """
        # Prepare conditionals if audio prompt provided
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Call prepare_conditionals first or provide audio_prompt_path"

        # Update exaggeration if changed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Normalize and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        # Duplicate for CFG
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        # Add start/end tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            # Generate speech tokens
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

            # Take conditional batch
            speech_tokens = speech_tokens[0]

            # Filter invalid tokens
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < SPEECH_VOCAB_SIZE]
            speech_tokens = speech_tokens.to(self.device)

            # Generate waveform
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )

            wav = wav.squeeze(0).detach().cpu().numpy()

        return torch.from_numpy(wav).unsqueeze(0)
