"""
AlignmentStreamAnalyzer: Hallucination detector for multilingual TTS.

Monitors attention patterns during autoregressive generation to detect:
- False starts (gibberish at the beginning)
- Long tails (noise after text completion)
- Repetition errors (looping over phrases)

Based on the observation that certain attention heads implicitly learn
text-speech alignment. Uses heuristics on these attention maps to
detect anomalies and force early stopping when needed.

Original authors: John Meade, Jeremy Hsu (Resemble AI)
Adapted for pure PyTorch backbone (reads _last_attn_weights directly).
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Layer-head pairs that implicitly learn text-speech alignment.
# Format: (layer_idx, head_idx) — empirically selected from the 30x16 LLaMA model.
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


class AlignmentStreamAnalyzer:
    """
    Analyzes attention alignment during speech generation to detect hallucinations.

    Reads attention weights directly from backbone layers via _last_attn_weights
    (stored by LLaMAAttention.forward()). No hooks needed.
    """

    def __init__(self, backbone, text_tokens_slice, eos_idx):
        """
        Args:
            backbone: LLaMABackbone instance (to read _last_attn_weights from layers)
            text_tokens_slice: Tuple (start, end) of text token positions in the sequence
            eos_idx: Token ID for end-of-speech (stop_speech_token)
        """
        self.backbone = backbone
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx

        self.alignment = torch.zeros(0, j - i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None

        self.generated_tokens = []

    def _read_aligned_attentions(self):
        """Read attention weights from the 3 aligned heads directly from backbone layers."""
        attns = []
        for layer_idx, head_idx in LLAMA_ALIGNED_HEADS:
            attn = self.backbone.layers[layer_idx].attn._last_attn_weights
            attns.append(attn[0, head_idx].cpu())  # (seq_len, kv_len)
        return attns

    def step(self, logits, next_token=None):
        """
        Analyze alignment for current step and potentially modify logits.

        Called once per generation step, after CFG but before sampling.
        May suppress EOS (to prevent early termination) or force EOS
        (to stop hallucinations).

        Args:
            logits: Current step logits, shape (1, vocab_size)
            next_token: Last generated token (for repetition tracking)

        Returns:
            Modified logits
        """
        # Read and average attention weights from the 3 aligned heads
        aligned_attns = self._read_aligned_attentions()
        aligned_attn = torch.stack(aligned_attns).mean(dim=0)

        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # First chunk: full context including conditioning, text, and BOS
            A_chunk = aligned_attn[j:, i:j].clone().cpu()
        else:
            # Subsequent chunks: 1 row due to KV-caching
            A_chunk = aligned_attn[:, i:j].clone().cpu()

        # Monotonic masking: zero out future text positions
        A_chunk[:, self.curr_frame_pos + 1:] = 0

        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # Update text position tracking
        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        # False start detection: hallucinations at the start show up as
        # activations far off-diagonal (bottom of attention map)
        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Completion detection: text position reached near-end
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # Long tail detection: final tokens still getting activations after completion
        long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 5)

        # Alignment repetition: activations shift back to earlier tokens after completion
        alignment_repetition = self.complete and (A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)

        # Track generated tokens for repetition detection
        if next_token is not None:
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        # Token repetition: last 2 tokens identical
        token_repetition = (
            len(self.generated_tokens) >= 3
            and len(set(self.generated_tokens[-2:])) == 1
        )

        if token_repetition:
            logger.warning(f"Detected 2x repetition of token {self.generated_tokens[-1]}")

        # Suppress EOS to prevent early termination (before text is fully spoken)
        if cur_text_posn < S - 3 and S > 5:
            logits[..., self.eos_idx] = -(2**15)

        # Force EOS if a bad pattern is detected
        if long_tail or alignment_repetition or token_repetition:
            logger.warning(f"Forcing EOS: {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits
