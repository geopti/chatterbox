"""
Text tokenizer for Chatterbox TTS.

Wraps the HuggingFace tokenizers library for text tokenization.
"""

from typing import List, Optional

import torch

# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]


class EnTokenizer:
    """
    English text tokenizer.

    Uses the tokenizers library for BPE tokenization.
    """

    def __init__(self, vocab_file_path: str):
        """
        Initialize tokenizer from vocab file.

        Args:
            vocab_file_path: Path to tokenizer.json file
        """
        from tokenizers import Tokenizer as HFTokenizer

        self.tokenizer: HFTokenizer = HFTokenizer.from_file(vocab_file_path)
        self._verify_special_tokens()

    def _verify_special_tokens(self):
        """Verify that required special tokens are in vocabulary."""
        vocab = self.tokenizer.get_vocab()
        assert SOT in vocab, f"Missing {SOT} in vocabulary"
        assert EOT in vocab, f"Missing {EOT} in vocabulary"

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """
        Convert text to token tensor.

        Args:
            text: Input text

        Returns:
            Token tensor of shape (1, seq_len)
        """
        token_ids = self.encode(text)
        return torch.IntTensor(token_ids).unsqueeze(0)

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        # Replace spaces with special token
        text = text.replace(" ", SPACE)
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token tensor

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy().tolist()

        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        text = text.replace(" ", "")
        text = text.replace(SPACE, " ")
        text = text.replace(EOT, "")
        text = text.replace(UNK, "")
        return text

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
