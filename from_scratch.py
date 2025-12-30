"""
Clean PyTorch implementation of Chatterbox TTS.

This script demonstrates the pure PyTorch implementation of S3Gen
without any external dependencies like diffusers or conformer packages.

Usage:
    python from_scratch.py
    python from_scratch.py --text "Your text here"
    python from_scratch.py --text "Your text here" --ref path/to/reference.wav
    python from_scratch.py --text "Your text" --cfg 0.7 --exaggeration 0.8
"""

import argparse
import numpy as np
import scipy.io.wavfile as wavfile
import torch

from chatterbox_pytorch.tts import ChatterboxTTS


def main():
    parser = argparse.ArgumentParser(description="Clean PyTorch Chatterbox TTS")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the clean PyTorch implementation.",
                        help="Text to synthesize")
    parser.add_argument("--ref", type=str, default=None,
                        help="Path to reference audio for voice cloning (optional)")
    parser.add_argument("--output", type=str, default="output.wav",
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
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Token repetition penalty (default: 1.2)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling parameter (default: 1.0)")

    args = parser.parse_args()

    print("=" * 60)
    print("Clean PyTorch Chatterbox TTS")
    print("=" * 60)
    print(f"Device: {args.device}")
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
    print("Loading model...")
    tts = ChatterboxTTS.from_pretrained(device=args.device)
    print("Model loaded successfully!")
    print()

    # Generate speech
    print("Generating speech...")
    if args.ref:
        wav = tts.generate(
            args.text,
            audio_prompt_path=args.ref,
            cfg_weight=args.cfg,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_p=args.top_p,
        )
    else:
        wav = tts.generate(
            args.text,
            cfg_weight=args.cfg,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_p=args.top_p,
        )

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
