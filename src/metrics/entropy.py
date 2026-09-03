"""
Layer-wise Shannon Attention Entropy
=====================================
Quantifies whether a layer's attention distribution remains healthily dispersed
across tokens (information routing) or collapses into an unnatural low-entropy
spike on a single key token (an attention sink / artifact).

Mathematical Formulation:
    H_i^(l, h) = - sum_{j=1}^S A_{ij}^(l, h) * log2( A_{ij}^(l, h) + eps )
    bar{H}^(l) = (1 / (B * H * S)) * sum_{b, h, i} H_i^(l, h)

Public Interface:
- `compute_shannon_entropy`: Computes scalar Shannon entropy in bits for an attention tensor.
- `compute_layerwise_entropy`: Computes per-layer entropy across all transformer blocks.
"""

from typing import Dict
import torch


def compute_shannon_entropy(attention_tensor: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes Shannon entropy across the key token dimension (last axis).

    :param attention_tensor: Attention weight probabilities of shape [..., S, S],
                             where values along the last axis sum to 1.0.
    :param eps: Numerical stability constant added before log2 to prevent log2(0) -> NaN.
    :return: Scalar mean entropy in bits (log2), averaged across all batch/head/query axes.
    :raises ValueError: If attention_tensor is empty or lacks spatial axes.
    """
    if attention_tensor.numel() == 0:
        raise ValueError("Cannot compute entropy on an empty attention tensor.")

    if attention_tensor.ndim < 2:
        raise ValueError(
            f"Expected attention_tensor to have at least 2 dimensions, got shape {attention_tensor.shape}."
        )

    # Entropy along the key dimension (dim=-1): - sum(p * log2(p + eps))
    log_probs = torch.log2(attention_tensor + eps)
    entropy = -torch.sum(attention_tensor * log_probs, dim=-1)

    return float(entropy.mean().item())


def compute_layerwise_entropy(attention_dict: Dict[int, torch.Tensor], eps: float = 1e-12) -> Dict[int, float]:
    """
    Computes mean Shannon entropy for each layer in a collection of attention tensors.

    :param attention_dict: Mapping of layer_idx -> attention tensor [B, H, S, S].
    :param eps: Epsilon constant for numerical stability.
    :return: Dictionary mapping layer_idx -> mean entropy in bits.
    """
    results: Dict[int, float] = {}
    for layer_idx, attn in attention_dict.items():
        results[layer_idx] = compute_shannon_entropy(attn, eps=eps)
    return results
