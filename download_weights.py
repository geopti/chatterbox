#!/usr/bin/env python3
"""
Download Chatterbox TTS weights from HuggingFace.

Usage:
    python download_weights.py

Or with a token:
    HF_TOKEN=your_token python download_weights.py
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "ResembleAI/chatterbox"
FILES = [
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
]

def main():
    token = os.environ.get("HF_TOKEN")

    print("Downloading Chatterbox TTS weights from HuggingFace...")
    print(f"Repository: {REPO_ID}")
    print(f"Token: {'provided' if token else 'not provided'}\n")

    download_dir = None

    for fname in FILES:
        print(f"Downloading {fname}...")
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                token=token,
            )
            print(f"  -> {path}")
            download_dir = Path(path).parent
        except Exception as e:
            print(f"  ERROR: {e}")
            return 1

    print(f"\nAll files downloaded to: {download_dir}")
    print("\nTo run the test:")
    print(f"  python test_chatterbox_pytorch.py")

    return 0

if __name__ == "__main__":
    exit(main())
