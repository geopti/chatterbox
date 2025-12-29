"""
Weight loading utilities for Chatterbox-PyTorch.

Handles loading weights from HuggingFace checkpoints and mapping them
to our from-scratch implementation.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


def download_weights(repo_id: str = "ResembleAI/chatterbox") -> Path:
    """
    Download weights from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Path to the downloaded checkpoint directory
    """
    from huggingface_hub import hf_hub_download

    files = [
        "ve.safetensors",
        "t3_cfg.safetensors",
        "s3gen.safetensors",
        "tokenizer.json",
        "conds.pt",
    ]

    local_path = None
    for fname in files:
        local_path = hf_hub_download(repo_id=repo_id, filename=fname)

    return Path(local_path).parent


def load_safetensors(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load a safetensors file.

    Args:
        path: Path to the safetensors file

    Returns:
        Dictionary of tensor name to tensor
    """
    from safetensors.torch import load_file

    return load_file(str(path))


def map_t3_key(old_key: str) -> Optional[str]:
    """
    Map T3 weight key from HuggingFace format to our implementation.

    Args:
        old_key: Original key from HuggingFace checkpoint

    Returns:
        Mapped key for our implementation, or None if key should be skipped
    """
    # Handle wrapped model keys
    if old_key.startswith("model."):
        old_key = old_key[6:]

    # Direct mappings for embeddings and heads
    direct_mappings = {
        "text_emb.weight": "text_emb.weight",
        "speech_emb.weight": "speech_emb.weight",
        "text_head.weight": "text_head.weight",
        "speech_head.weight": "speech_head.weight",
        "text_pos_emb.emb.weight": "text_pos_emb.weight",
        "speech_pos_emb.emb.weight": "speech_pos_emb.weight",
    }

    if old_key in direct_mappings:
        return direct_mappings[old_key]

    # Conditioning encoder mappings
    cond_mappings = {
        "cond_enc.spkr_enc.weight": "cond_enc.spkr_enc.weight",
        "cond_enc.spkr_enc.bias": "cond_enc.spkr_enc.bias",
        "cond_enc.emotion_adv_fc.weight": "cond_enc.emotion_adv_fc.weight",
    }

    if old_key in cond_mappings:
        return cond_mappings[old_key]

    # Perceiver mappings - direct mapping since we now use same structure
    if old_key.startswith("cond_enc.perceiver."):
        return old_key  # Direct mapping - structure matches

    # Transformer layer mappings
    # tfmr.layers.{i}.self_attn.{q,k,v,o}_proj.weight -> backbone.layers.{i}.attn.w{q,k,v,o}.weight
    layer_match = re.match(r"tfmr\.layers\.(\d+)\.(.+)", old_key)
    if layer_match:
        layer_idx = layer_match.group(1)
        rest = layer_match.group(2)

        # Attention projections
        attn_map = {
            "self_attn.q_proj.weight": "attn.wq.weight",
            "self_attn.k_proj.weight": "attn.wk.weight",
            "self_attn.v_proj.weight": "attn.wv.weight",
            "self_attn.o_proj.weight": "attn.wo.weight",
        }

        if rest in attn_map:
            return f"backbone.layers.{layer_idx}.{attn_map[rest]}"

        # MLP projections
        mlp_map = {
            "mlp.gate_proj.weight": "ffn.w1.weight",
            "mlp.up_proj.weight": "ffn.w3.weight",
            "mlp.down_proj.weight": "ffn.w2.weight",
        }

        if rest in mlp_map:
            return f"backbone.layers.{layer_idx}.{mlp_map[rest]}"

        # Layer norms
        norm_map = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
        }

        if rest in norm_map:
            return f"backbone.layers.{layer_idx}.{norm_map[rest]}"

    # Final layer norm
    if old_key == "tfmr.norm.weight":
        return "backbone.norm.weight"

    # Skip embed_tokens as we use our own embeddings
    if "embed_tokens" in old_key:
        return None

    # Log unknown keys
    print(f"Unknown T3 key: {old_key}")
    return None


def load_ve_weights(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
) -> Tuple[List[str], List[str]]:
    """
    Load VoiceEncoder weights.

    Args:
        model: VoiceEncoder model
        checkpoint_path: Path to ve.safetensors

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    state_dict = load_safetensors(checkpoint_path)

    # VoiceEncoder has direct key mapping
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    return missing, unexpected


def load_t3_weights(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
) -> Tuple[List[str], List[str]]:
    """
    Load T3 model weights with key mapping.

    Args:
        model: T3 model
        checkpoint_path: Path to t3_cfg.safetensors

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    state_dict = load_safetensors(checkpoint_path)

    # Handle wrapped state dict
    if "model" in state_dict:
        state_dict = state_dict["model"][0]

    # Map keys
    mapped_state_dict = {}
    unexpected = []

    for old_key, value in state_dict.items():
        new_key = map_t3_key(old_key)
        if new_key is not None:
            mapped_state_dict[new_key] = value
        else:
            unexpected.append(old_key)

    # Load weights
    result = model.load_state_dict(mapped_state_dict, strict=False)
    missing = result.missing_keys

    return missing, unexpected


def load_s3gen_weights(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
) -> Tuple[List[str], List[str]]:
    """
    Load S3Gen model weights.

    Args:
        model: S3Gen model
        checkpoint_path: Path to s3gen.safetensors

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    state_dict = load_safetensors(checkpoint_path)

    # S3Gen mostly uses direct mapping, but some buffers may be missing
    # (mel_filters, window) which are computed at runtime
    ignore_missing = {"tokenizer._mel_filters", "tokenizer.window"}

    result = model.load_state_dict(state_dict, strict=False)

    missing = [k for k in result.missing_keys if k not in ignore_missing]
    unexpected = result.unexpected_keys

    return missing, unexpected


def load_weights(
    models: Dict[str, nn.Module],
    checkpoint_dir: Union[str, Path],
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Load all model weights from a checkpoint directory.

    Args:
        models: Dictionary mapping model name to model instance
            Expected keys: "ve", "t3", "s3gen"
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Dictionary mapping model name to (missing_keys, unexpected_keys)
    """
    checkpoint_dir = Path(checkpoint_dir)
    results = {}

    if "ve" in models:
        results["ve"] = load_ve_weights(
            models["ve"],
            checkpoint_dir / "ve.safetensors",
        )

    if "t3" in models:
        results["t3"] = load_t3_weights(
            models["t3"],
            checkpoint_dir / "t3_cfg.safetensors",
        )

    if "s3gen" in models:
        results["s3gen"] = load_s3gen_weights(
            models["s3gen"],
            checkpoint_dir / "s3gen.safetensors",
        )

    return results
