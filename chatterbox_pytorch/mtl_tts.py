"""
ChatterboxMultilingualTTS: Multilingual TTS interface.

Combines T3 (text-to-token, multilingual config) and S3Gen (token-to-waveform)
for end-to-end multilingual text-to-speech synthesis with voice cloning.
Supports 23 languages.
"""

import re
from pathlib import Path
from typing import List, Optional, Union

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .t3 import T3
from .t3.config import T3Config
from .t3.conditioning import T3Cond
from .s3gen.model import S3Gen
from .s3gen.s3tokenizer import S3_SR, drop_invalid_tokens, SPEECH_VOCAB_SIZE
from .s3gen.mel import S3GEN_SR
from .voice_encoder import VoiceEncoder
from .tokenizers.mtl_tokenizer import MTLTokenizer
from .tts import Conditionals


REPO_ID = "ResembleAI/chatterbox"

SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """
    Normalize punctuation for multilingual TTS input.

    Includes CJK sentence enders in addition to standard punctuation.
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
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2018", "'"),
        ("\u2019", "'"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    # Add full stop if no ending punctuation (includes CJK enders)
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "\u3001", "\uff0c", "\u3002", "\uff1f", "\uff01"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


class ChatterboxMultilingualTTS:
    """
    Multilingual Chatterbox TTS class.

    Provides end-to-end multilingual text-to-speech with voice cloning.
    Supports 23 languages.
    """

    ENC_COND_LEN = 6 * S3_SR  # 6 seconds for T3 conditioning
    DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds for S3Gen conditioning

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Optional[Conditionals] = None,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir: Union[str, Path], device: str) -> "ChatterboxMultilingualTTS":
        """
        Load multilingual model from local checkpoint directory.

        Args:
            ckpt_dir: Directory containing model checkpoints
            device: Target device

        Returns:
            Initialized ChatterboxMultilingualTTS
        """
        ckpt_dir = Path(ckpt_dir)

        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        from .utils.weight_loader import load_ve_weights, load_t3_weights

        # Load voice encoder (same as English)
        ve = VoiceEncoder()
        missing, unexpected = load_ve_weights(ve, ckpt_dir / "ve.safetensors")
        if missing:
            print(f"VE missing keys: {missing[:5]}...")
        ve.to(device).eval()

        # Load T3 with multilingual config
        t3 = T3(T3Config.multilingual())
        missing, unexpected = load_t3_weights(t3, ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if missing:
            print(f"T3 missing keys: {missing[:5]}...")
        t3.to(device).eval()

        # Load S3Gen (same as English)
        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        # Load multilingual tokenizer
        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        # Load built-in voice if available
        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: str) -> "ChatterboxMultilingualTTS":
        """
        Load multilingual model from HuggingFace Hub.

        Args:
            device: Target device

        Returns:
            Initialized ChatterboxMultilingualTTS
        """
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available - PyTorch not built with MPS.")
            else:
                print("MPS not available - macOS version < 12.3 or no MPS device.")
            device = "cpu"

        # Download all required files
        for fname in [
            "ve.safetensors",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.safetensors",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt",
        ]:
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
        language_id: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        token_repetition_threshold: int = 3,
        trim_buffer: int = 25,
    ) -> torch.Tensor:
        """
        Generate speech from text in the specified language.

        Args:
            text: Input text
            language_id: Language code (e.g., "en", "fr", "ja")
            audio_prompt_path: Path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Token repetition penalty
            min_p: Min-p sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            token_repetition_threshold: Consecutive identical tokens to trigger forced EOS
            trim_buffer: Frames to keep after text completion when trimming garbage tail

        Returns:
            Generated audio waveform (tensor of shape [1, samples])
        """
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Prepare conditionals if audio prompt provided
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Call prepare_conditionals first or provide audio_prompt_path"

        # Update exaggeration if changed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Normalize and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)

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
                use_alignment_analyzer=True,
                token_repetition_threshold=token_repetition_threshold,
                trim_buffer=trim_buffer,
            )

            # Take conditional batch
            speech_tokens = speech_tokens[0]

            # Filter invalid tokens
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < SPEECH_VOCAB_SIZE]
            speech_tokens = speech_tokens.to(self.device)

            # Check minimum token count to avoid vocoder crash
            if len(speech_tokens) < 3:
                raise RuntimeError(
                    "T3 generated too few speech tokens. "
                    "Try shorter text or use generate_long() for multiple sentences."
                )

            # Track alignment analyzer state for retry logic
            self._last_forced_eos = getattr(self.t3, '_last_forced_eos', False)
            self._last_text_complete = getattr(self.t3, '_last_text_complete', True)

            # Generate waveform
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )

            wav = wav.squeeze(0).detach().cpu().numpy()

        return torch.from_numpy(wav).unsqueeze(0)

    def generate_long(
        self,
        text: str,
        language_id: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        pause_duration: float = 0.3,
        max_words: int = 60,
        token_repetition_threshold: int = 3,
        trim_buffer: int = 25,
    ) -> torch.Tensor:
        """
        Generate speech from long text by splitting into sentences.

        Each sentence is generated separately and concatenated with pauses.

        Args:
            text: Input text (can be multiple sentences)
            language_id: Language code (e.g., "en", "fr", "ja")
            audio_prompt_path: Path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Token repetition penalty
            min_p: Min-p sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            pause_duration: Silence between sentences in seconds
            max_words: Max words per chunk

        Returns:
            Generated audio waveform (tensor of shape [1, samples])
        """
        sentences = self._split_sentences(text, max_words=max_words)

        if len(sentences) == 0:
            raise ValueError("No sentences found in text")

        if len(sentences) == 1:
            return self.generate(
                sentences[0], language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration, cfg_weight=cfg_weight,
                temperature=temperature, repetition_penalty=repetition_penalty,
                min_p=min_p, top_p=top_p,
                token_repetition_threshold=token_repetition_threshold,
                trim_buffer=trim_buffer,
            )

        # Prepare conditionals once
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)

        pause_samples = int(pause_duration * self.sr)
        pause = torch.zeros(1, pause_samples)

        max_retries = 2
        chunks = []
        for i, sentence in enumerate(sentences):
            print(f"Generating sentence {i + 1}/{len(sentences)}: {sentence[:60]}...")
            wav = None
            for attempt in range(1 + max_retries):
                try:
                    wav = self.generate(
                        sentence, language_id,
                        exaggeration=exaggeration, cfg_weight=cfg_weight,
                        temperature=temperature, repetition_penalty=repetition_penalty,
                        min_p=min_p, top_p=top_p,
                        token_repetition_threshold=token_repetition_threshold,
                        trim_buffer=trim_buffer,
                    )
                    # Retry only if forced EOS cut the sentence short (text not fully spoken)
                    if self._last_forced_eos and not self._last_text_complete and attempt < max_retries:
                        print(f"  Sentence cut short by forced EOS, retrying sentence {i + 1} (attempt {attempt + 2}/{1 + max_retries})...")
                        continue
                    break
                except RuntimeError as e:
                    if attempt < max_retries:
                        print(f"  Error, retrying sentence {i + 1} (attempt {attempt + 2}/{1 + max_retries}): {e}")
                        continue
                    print(f"Warning: skipping sentence {i + 1} ({e})")
                    break

            if wav is not None:
                chunks.append(wav)
                if i < len(sentences) - 1:
                    chunks.append(pause)

        if not chunks:
            raise RuntimeError("Failed to generate any audio chunks")

        return torch.cat(chunks, dim=1)

    @staticmethod
    def _split_sentences(text: str, max_words: int = 60) -> List[str]:
        """Split text into chunks of up to max_words by grouping consecutive sentences."""
        # Split on standard and CJK sentence enders
        parts = re.split(r'(?<=[.!?\u3002\uff01\uff1f])\s*', text)
        sentences = [s.strip() for s in parts if s.strip()]

        if not sentences:
            return [text.strip()] if text.strip() else []

        # Group sentences into chunks that don't exceed max_words
        chunks = []
        current_chunk = []
        current_words = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current_chunk and current_words + word_count > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_words = word_count
            else:
                current_chunk.append(sentence)
                current_words += word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
