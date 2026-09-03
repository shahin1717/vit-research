"""
Layer-wise Shannon Attention Entropy
=====================================
Quantifies whether a layer's attention distributions stay "spread out"
(healthy information routing) or collapse into a low-entropy spike on a
single key token (an attention sink).

    H_i^(l,h) = - sum_j A_ij^(l,h) * log2( A_ij^(l,h) + eps )

High entropy  -> queries attend broadly across many tokens.
Low entropy   -> queries attend almost entirely to one token (a sink).
"""

import torch
from typing import Dict


def compute_shannon_entropy(attention_tensor: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes Shannon entropy across the key dimension.

    Args:
        attention_tensor: [B, H, S, S] softmax attention weights, where the
            last dimension sums to 1 (it's a probability distribution over
            key tokens for each query token).
        eps: numerical stability constant. Without it, any A_ij that is
            exactly 0.0 sends log2(0) -> -inf, which then poisons the
            batch mean into NaN. Adding eps keeps log2 finite everywhere.

    Returns:
        Scalar mean entropy in bits, averaged over batch, heads, and
        query positions.
    """
    # -sum(p * log2(p + eps)) along the key dimension (last dim)
    entropy = -torch.sum(
        attention_tensor * torch.log2(attention_tensor + eps), dim=-1
    )  # -> [B, H, S] : one entropy value per query token
    return float(entropy.mean().item())


def compute_layerwise_entropy(attention_dict: Dict[int, torch.Tensor]) -> Dict[int, float]:
    """
    Takes a dict of layer_idx -> attention_tensor [B, H, S, S] (e.g. the
    `attention_maps` produced by ViTAttentionHookManager) and returns a
    dict of layer_idx -> mean_entropy_value, so you can plot entropy vs.
    depth to see whether/where it collapses.
    """
    results = {}
    for layer_idx, attn in attention_dict.items():
        results[layer_idx] = compute_shannon_entropy(attn)
    return results
