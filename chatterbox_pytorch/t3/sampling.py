"""
Sampling utilities for T3 token generation.

Implements temperature, top-p, top-k, min-p, and repetition penalty.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits.

    Args:
        logits: Logits tensor of shape (..., vocab_size)
        temperature: Temperature value (higher = more random)

    Returns:
        Scaled logits
    """
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply top-k filtering to logits.

    Keeps only the top k logits, sets others to -inf.

    Args:
        logits: Logits tensor of shape (..., vocab_size)
        k: Number of top logits to keep

    Returns:
        Filtered logits
    """
    if k <= 0:
        return logits

    vocab_size = logits.shape[-1]
    k = min(k, vocab_size)

    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[..., -1:]

    return torch.where(logits < threshold, float("-inf"), logits)


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) filtering to logits.

    Keeps the smallest set of logits with cumulative probability >= p.

    Args:
        logits: Logits tensor of shape (..., vocab_size)
        p: Cumulative probability threshold

    Returns:
        Filtered logits
    """
    if p >= 1.0:
        return logits

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find cutoff index
    sorted_mask = cumulative_probs > p

    # Keep at least one token
    sorted_mask[..., 0] = False

    # Shift mask to include the token that crosses threshold
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # Set filtered logits to -inf
    sorted_logits = torch.where(sorted_mask, float("-inf"), sorted_logits)

    # Unsort to original order
    original_order = sorted_indices.argsort(dim=-1)
    logits = sorted_logits.gather(-1, original_order)

    return logits


def apply_min_p(
    logits: torch.Tensor,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Apply min-p filtering to logits.

    Filters out tokens with probability less than min_p * max_probability.

    Args:
        logits: Logits tensor of shape (..., vocab_size)
        min_p: Minimum probability ratio threshold
        min_tokens_to_keep: Minimum number of tokens to keep

    Returns:
        Filtered logits
    """
    if min_p <= 0.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * max_prob

    # Create mask for tokens to filter
    mask = probs < threshold

    # Keep at least min_tokens_to_keep tokens
    if min_tokens_to_keep > 0:
        # Find top-k indices
        top_k_values, top_k_indices = torch.topk(probs, min_tokens_to_keep, dim=-1)
        # Don't filter these indices
        mask.scatter_(-1, top_k_indices, False)

    return torch.where(mask, float("-inf"), logits)


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Reduces probability of tokens that have already appeared.

    Args:
        logits: Logits tensor of shape (batch, vocab_size)
        input_ids: Previously generated token IDs of shape (batch, seq_len)
        penalty: Penalty factor (> 1 reduces repetition)

    Returns:
        Penalized logits
    """
    if penalty == 1.0:
        return logits

    # Get unique tokens from input_ids for each batch
    for batch_idx in range(input_ids.shape[0]):
        unique_tokens = input_ids[batch_idx].unique()

        for token in unique_tokens:
            if logits[batch_idx, token] > 0:
                logits[batch_idx, token] /= penalty
            else:
                logits[batch_idx, token] *= penalty

    return logits


def sample_token(
    logits: torch.Tensor,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    Sample a token from logits.

    Args:
        logits: Logits tensor of shape (batch, vocab_size)
        do_sample: If True, sample from distribution. If False, take argmax.

    Returns:
        Sampled token IDs of shape (batch, 1)
    """
    if do_sample:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    else:
        return logits.argmax(dim=-1, keepdim=True)


def process_logits(
    logits: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Apply all logit processing steps.

    Args:
        logits: Raw logits from model
        input_ids: Previously generated tokens for repetition penalty
        temperature: Temperature for scaling
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        min_p: Min-p filtering parameter
        repetition_penalty: Repetition penalty factor

    Returns:
        Processed logits
    """
    # Repetition penalty first (on raw logits)
    if input_ids is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)

    # Temperature scaling
    logits = apply_temperature(logits, temperature)

    # Filtering (order matters!)
    logits = apply_min_p(logits, min_p)
    logits = apply_top_p(logits, top_p)
    logits = apply_top_k(logits, top_k)

    return logits
