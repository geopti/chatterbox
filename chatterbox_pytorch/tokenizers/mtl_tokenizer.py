"""
Multilingual text tokenizer for Chatterbox TTS.

Supports 23 languages with BPE tokenization and language ID prefixing.
"""

from typing import List, Optional
from unicodedata import normalize

import torch

from .text_tokenizer import SOT, EOT, UNK, SPACE


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose Hangul syllables into Jamo."""

    def decompose_hangul(char):
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        return initial + medial + final

    return ''.join(decompose_hangul(char) for char in text).strip()


class MTLTokenizer:
    """
    Multilingual text tokenizer.

    Uses BPE tokenization with language ID prefixing.
    Supports Korean Hangul decomposition (pure Python, no external deps).
    """

    def __init__(self, vocab_file_path: str):
        """
        Initialize tokenizer from multilingual vocab file.

        Args:
            vocab_file_path: Path to grapheme_mtl_merged_expanded_v1.json
        """
        from tokenizers import Tokenizer as HFTokenizer

        self.tokenizer: HFTokenizer = HFTokenizer.from_file(vocab_file_path)
        self._verify_special_tokens()

    def _verify_special_tokens(self):
        """Verify that required special tokens are in vocabulary."""
        vocab = self.tokenizer.get_vocab()
        assert SOT in vocab, f"Missing {SOT} in vocabulary"
        assert EOT in vocab, f"Missing {EOT} in vocabulary"

    def preprocess_text(
        self, raw_text: str, lowercase: bool = True, nfkd_normalize: bool = True
    ) -> str:
        """
        Preprocess text with lowercase and NFKD normalization.

        Args:
            raw_text: Input text
            lowercase: Whether to lowercase
            nfkd_normalize: Whether to apply NFKD normalization

        Returns:
            Preprocessed text
        """
        text = raw_text
        if lowercase:
            text = text.lower()
        if nfkd_normalize:
            text = normalize("NFKD", text)
        return text

    def text_to_tokens(
        self,
        text: str,
        language_id: str = None,
        lowercase: bool = True,
        nfkd_normalize: bool = True,
    ) -> torch.Tensor:
        """
        Convert text to token tensor.

        Args:
            text: Input text
            language_id: Language code (e.g., "en", "fr", "ja")
            lowercase: Whether to lowercase
            nfkd_normalize: Whether to apply NFKD normalization

        Returns:
            Token tensor of shape (1, seq_len)
        """
        token_ids = self.encode(
            text,
            language_id=language_id,
            lowercase=lowercase,
            nfkd_normalize=nfkd_normalize,
        )
        return torch.IntTensor(token_ids).unsqueeze(0)

    def encode(
        self,
        txt: str,
        language_id: str = None,
        lowercase: bool = True,
        nfkd_normalize: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Pipeline: preprocess -> korean normalize (if ko) -> prepend [lang_id] -> replace spaces -> BPE encode

        Args:
            txt: Input text
            language_id: Language code
            lowercase: Whether to lowercase
            nfkd_normalize: Whether to apply NFKD normalization

        Returns:
            List of token IDs
        """
        txt = self.preprocess_text(txt, lowercase=lowercase, nfkd_normalize=nfkd_normalize)

        # Language-specific processing (pure Python only)
        if language_id == "ko":
            txt = korean_normalize(txt)

        # Prepend language token
        if language_id:
            txt = f"[{language_id.lower()}]{txt}"

        txt = txt.replace(" ", SPACE)
        return self.tokenizer.encode(txt).ids

    def decode(self, token_ids) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token tensor or list of IDs

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy().tolist()

        txt = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        txt = txt.replace(" ", "").replace(SPACE, " ").replace(EOT, "").replace(UNK, "")
        return txt

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_special_token_id(self, token: str) -> Optional[int]:
        """Get ID for special token."""
        vocab = self.tokenizer.get_vocab()
        return vocab.get(token)

    @property
    def sot_id(self) -> int:
        """Get start-of-text token ID."""
        return self.get_special_token_id(SOT)

    @property
    def eot_id(self) -> int:
        """Get end-of-text token ID."""
        return self.get_special_token_id(EOT)
