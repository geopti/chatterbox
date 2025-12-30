"""
Flow matching module combining encoder and CFM decoder.

This module wraps the conformer encoder and CFM decoder for speech generation.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Create padding mask from lengths.

    Args:
        lengths: Lengths tensor of shape (batch,)
        max_len: Maximum length (optional)

    Returns:
        Mask of shape (batch, max_len) where True indicates padding
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    else:
        max_len = int(max_len)

    batch_size = int(lengths.size(0))
    seq_range = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)
    seq_range = seq_range.expand(batch_size, max_len)
    lengths = lengths.unsqueeze(1).expand(batch_size, max_len)

    return seq_range >= lengths


def _repeat_batch_dim(tensor: Optional[torch.Tensor], batch_size: int, ndim: int) -> Optional[torch.Tensor]:
    """Repeat batch dimension if it's equal to 1."""
    if tensor is None:
        return None

    # Add missing batch dim if needed
    while tensor.ndim < ndim:
        tensor = tensor.unsqueeze(0)

    # Repeat batch dim as needed
    if batch_size > 1 and tensor.size(0) == 1:
        tensor = tensor.repeat(batch_size, *([1] * (ndim - 1)))

    assert tensor.ndim == ndim, f"Expected {ndim=}, got {tensor.ndim=}"
    return tensor


class CausalMaskedDiffWithXvec(nn.Module):
    """
    Flow matching module with x-vector conditioning.

    Combines:
    - Token embedding
    - Conformer encoder (with upsampling)
    - CFM decoder
    - X-vector speaker conditioning
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 6561,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

        # Token embedding
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        # Speaker embedding projection
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)

        # Encoder (UpsampleConformerEncoder)
        self.encoder = encoder

        # Encoder output projection
        if encoder is not None:
            # Handle both method and property style output_size
            enc_out_size = encoder.output_size() if callable(encoder.output_size) else encoder.output_size
            self.encoder_proj = nn.Linear(enc_out_size, output_size)
        else:
            self.encoder_proj = nn.Linear(input_size, output_size)

        # Decoder (CausalConditionalCFM)
        self.decoder = decoder

    @torch.inference_mode()
    def inference(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: Optional[torch.Tensor],
        embedding: torch.Tensor,
        finalize: bool,
        n_timesteps: int = 10,
        noised_mels: Optional[torch.Tensor] = None,
        meanflow: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        """
        Generate mel-spectrogram from speech tokens.

        Args:
            token: Speech tokens of shape (batch, n_tokens)
            token_len: Token lengths
            prompt_token: Prompt tokens
            prompt_token_len: Prompt token lengths
            prompt_feat: Prompt mel features of shape (batch, time, n_mels)
            prompt_feat_len: Prompt feature lengths
            embedding: Speaker embedding of shape (batch, spk_dim)
            finalize: Whether this is the final generation
            n_timesteps: Number of CFM steps
            noised_mels: Pre-noised mels for meanflow
            meanflow: Whether to use meanflow mode

        Returns:
            Tuple of (mel_spectrogram, None)
        """
        batch_size = token.size(0)

        # X-vector projection
        embedding = torch.atleast_2d(embedding)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Adjust batch dimensions
        prompt_token = _repeat_batch_dim(prompt_token, batch_size, ndim=2)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, batch_size, ndim=1)
        prompt_feat = _repeat_batch_dim(prompt_feat, batch_size, ndim=3)
        prompt_feat_len = _repeat_batch_dim(prompt_feat_len, batch_size, ndim=1)
        embedding = _repeat_batch_dim(embedding, batch_size, ndim=2)

        # Concatenate prompt and target tokens
        token = torch.cat([prompt_token, token], dim=1)
        token_len = prompt_token_len + token_len

        # Create mask
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding.dtype)

        # Check for out-of-range tokens
        if (token >= self.vocab_size).any():
            logger.error(f"Out-of-range tokens found: {token.max()} >= {self.vocab_size}")

        # Embed tokens
        token_emb = self.input_embedding(token.long()) * mask

        # Encode
        h, h_masks = self.encoder(token_emb, token_len)

        if not finalize:
            h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]

        # h_masks is (batch, 1, time), sum across time dim to get lengths
        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)  # (batch,)
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1
        h = self.encoder_proj(h)

        # Prepare conditions
        conds = torch.zeros([batch_size, mel_len1 + mel_len2, self.output_size], device=token.device, dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        # Create mask for decoder
        mask = (~make_pad_mask(h_lengths)).unsqueeze(1).to(h.dtype)
        if mask.shape[0] != batch_size:
            mask = mask.repeat(batch_size, 1, 1)

        # Run CFM decoder
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            noised_mels=noised_mels,
            meanflow=meanflow,
        )

        # Remove prompt portion
        feat = feat[:, :, mel_len1:]

        assert feat.shape[2] == mel_len2, f"Shape mismatch: {feat.shape[2]} != {mel_len2}"

        return feat, None
