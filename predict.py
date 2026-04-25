import tempfile
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
from cog import BasePredictor, Input, Path

from chatterbox_pytorch.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

LANG_CHOICES = list(SUPPORTED_LANGUAGES.keys())


class Predictor(BasePredictor):
    def setup(self):
        self.tts = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

    def predict(
        self,
        text: str = Input(
            description="The text you want spoken. Can be a single sentence or a long paragraph — long inputs are automatically split into chunks.",
        ),
        language: str = Input(
            description="Language of the text. Use the two-letter code (en=English, fr=French, de=German, es=Spanish, ja=Japanese, zh=Chinese, ar=Arabic, el=Greek, etc.).",
            choices=LANG_CHOICES,
            default="en",
        ),
        audio_prompt: Optional[Path] = Input(
            description="Optional reference voice clip (.wav/.mp3). The output will mimic this voice. If left empty, a default voice is used.",
            default=None,
        ),
        cfg_weight: float = Input(
            description="How closely the speech follows the text. Higher = sticks to the text more strictly. Lower = more freedom (but can hallucinate or get stuck).",
            default=0.5, ge=0.0, le=1.0,
        ),
        exaggeration: float = Input(
            description="How expressive the voice is. Higher = more emotional / dramatic. Lower = more flat / neutral.",
            default=0.5, ge=0.0, le=1.0,
        ),
        temperature: float = Input(
            description="Randomness of the voice. Higher = more variation between runs. Lower = more consistent / robotic.",
            default=0.8, ge=0.0, le=2.0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeating the same sounds. Higher = less repetition.",
            default=2.0, ge=1.0, le=5.0,
        ),
        top_p: float = Input(
            description="Top-p (nucleus) sampling. Restricts the model to the most likely tokens. 1.0 = no restriction.",
            default=1.0, ge=0.0, le=1.0,
        ),
        pause_between_sentences: float = Input(
            description="Length of the silence (in seconds) inserted between sentences.",
            default=0.1, ge=0.0, le=5.0,
        ),
        max_words_per_chunk: int = Input(
            description="Long texts are split into chunks before generation. This is the max number of words per chunk. Smaller = safer for tricky languages, but slower.",
            default=60, ge=10, le=200,
        ),
        repeated_token_threshold: int = Input(
            description="If the model repeats the same sound this many times in a row, the chunk is cut off (prevents the model from getting stuck looping). Raise this if too much real speech is being cut.",
            default=3, ge=2, le=10,
        ),
        garbage_trim_buffer: int = Input(
            description="Number of audio frames kept after the model finishes saying the sentence (each frame = ~40ms). Lower = trims garbage tails more aggressively but may cut off the last syllable.",
            default=25, ge=0, le=200,
        ),
    ) -> Path:
        gen_kwargs = dict(
            language_id=language,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            token_repetition_threshold=repeated_token_threshold,
            trim_buffer=garbage_trim_buffer,
        )
        if audio_prompt is not None:
            gen_kwargs["audio_prompt_path"] = str(audio_prompt)

        gen_kwargs["pause_duration"] = pause_between_sentences
        gen_kwargs["max_words"] = max_words_per_chunk
        wav = self.tts.generate_long(text, **gen_kwargs)

        out = Path(tempfile.mkdtemp()) / "output.wav"
        data = wav.squeeze().cpu().numpy()
        data_int16 = (data * 32767).astype(np.int16)
        wavfile.write(str(out), self.tts.sr, data_int16)
        return out
