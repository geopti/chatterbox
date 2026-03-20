"""
Clean PyTorch implementation of Chatterbox Multilingual TTS.

Supports 23 languages using the pure PyTorch implementation.

Usage:
    python from_scratch_multilingual.py --language fr --text "Bonjour le monde"
    python from_scratch_multilingual.py --language ja --text "こんにちは世界"
    python from_scratch_multilingual.py --language en --text "Hello world" --ref path/to/reference.wav
    python from_scratch_multilingual.py --list-languages
"""

import argparse
import sys

import numpy as np
import scipy.io.wavfile as wavfile
import torch

from chatterbox_pytorch.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


def main():
    parser = argparse.ArgumentParser(description="Clean PyTorch Chatterbox Multilingual TTS")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to synthesize")
    parser.add_argument("--language", type=str, default=None,
                        help="Language code (e.g., en, fr, ja, zh)")
    parser.add_argument("--list-languages", action="store_true",
                        help="List supported languages and exit")
    parser.add_argument("--ref", type=str, default=None,
                        help="Path to reference audio for voice cloning (optional)")
    parser.add_argument("--output", type=str, default="output_multilingual.wav",
                        help="Output wav file path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run on")

    # Generation parameters
    parser.add_argument("--cfg", type=float, default=0.5,
                        help="Classifier-free guidance weight for T3 (0.0-1.0, default: 0.5)")
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="Emotion exaggeration factor (0.0-1.0, default: 0.5)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--repetition_penalty", type=float, default=2.0,
                        help="Token repetition penalty (default: 2.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling parameter (default: 1.0)")
    parser.add_argument("--long", action="store_true",
                        help="Split text into sentences and generate each separately (for longer texts)")
    parser.add_argument("--pause", type=float, default=0.3,
                        help="Pause duration between sentences in seconds when using --long (default: 0.3)")
    parser.add_argument("--max-words", type=int, default=60,
                        help="Max words per chunk when using --long (default: 60)")

    args = parser.parse_args()

    # Handle --list-languages
    if args.list_languages:
        print("Supported languages:")
        for code, name in SUPPORTED_LANGUAGES.items():
            print(f"  {code}: {name}")
        sys.exit(0)

    # Validate required args
    if args.language is None:
        parser.error("--language is required (use --list-languages to see options)")
    if args.text is None:
        parser.error("--text is required")

    print("=" * 60)
    print("Clean PyTorch Chatterbox Multilingual TTS")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Language: {args.language} ({SUPPORTED_LANGUAGES.get(args.language.lower(), 'unknown')})")
    print(f"Text: {args.text}")
    if args.ref:
        print(f"Reference: {args.ref}")
    print()
    print("Generation parameters:")
    print(f"  CFG weight: {args.cfg}")
    print(f"  Exaggeration: {args.exaggeration}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Repetition penalty: {args.repetition_penalty}")
    print(f"  Top-p: {args.top_p}")
    print()

    # Load model
    print("Loading multilingual model...")
    tts = ChatterboxMultilingualTTS.from_pretrained(device=args.device)
    print("Model loaded successfully!")
    print()

    # Generate speech
    gen_func = tts.generate_long if args.long else tts.generate
    gen_kwargs = dict(
        language_id=args.language,
        cfg_weight=args.cfg,
        exaggeration=args.exaggeration,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
    )
    if args.ref:
        gen_kwargs["audio_prompt_path"] = args.ref
    if args.long:
        gen_kwargs["pause_duration"] = args.pause
        gen_kwargs["max_words"] = args.max_words

    print("Generating speech..." + (" (long mode)" if args.long else ""))
    wav = gen_func(args.text, **gen_kwargs)

    # Convert to int16 for wav file
    wav_data = wav.squeeze().cpu().numpy()
    wav_int16 = (wav_data * 32767).astype(np.int16)

    # Save output
    wavfile.write(args.output, tts.sr, wav_int16)
    print()
    print(f"Audio saved to: {args.output}")
    print(f"Duration: {len(wav_int16) / tts.sr:.2f} seconds")
    print(f"Sample rate: {tts.sr} Hz")


if __name__ == "__main__":
    main()
