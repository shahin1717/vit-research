"""
Patch-Norm Outlier Rate (3-Sigma Bound)
========================================
Darcet et al. (2023) show that background patch tokens in register-free
Vision Transformers develop abnormally high-norm activations ("attention sinks"),
while register-equipped ViTs route unneeded information into register tokens,
keeping spatial patch activation norms well-behaved.

Mathematical Formulation:
    n_i^(l)       = || x_i^(l) ||_2   (for each spatial patch token i in 1..N)
    mu_l, sigma_l = mean and std of n_i^(l) across spatial patches
    theta_l       = mu_l + 3 * sigma_l (3-sigma statistical bound)
    OutlierRate   = (1 / N) * sum_{i=1}^N  1[ n_i^(l) > theta_l ]

Public Interface:
- `compute_patch_outlier_rate`: Computes outlier fraction for a single layer activation tensor.
- `compute_layerwise_outlier_rate`: Computes outlier rate mapping across all network depths.
"""

from typing import Dict
import torch


def compute_patch_outlier_rate(patch_activations: torch.Tensor, k_registers: int = 0) -> float:
    """
    Computes the fraction of spatial patch tokens whose L2 norm exceeds
    mu + 3*sigma for that layer.

    Sequence Structure:
        Index 0:                 [CLS]
        Index 1 .. K:            [Register Tokens]
        Index (1+K) .. end:      [Spatial Patch Tokens] (N = 196 for 14x14)

    :param patch_activations: [B, S, d] or [S, d] intermediate activations at layer l.
    :param k_registers: Number of register tokens prepended after [CLS] (e.g. 0, 1, 4, 8).
    :return: Scalar outlier rate in [0.0, 1.0], averaged across the batch.
    :raises ValueError: If k_registers is negative or sequence length is insufficient.
    """
    if k_registers < 0:
        raise ValueError(f"k_registers cannot be negative, got {k_registers}.")

    if patch_activations.ndim == 2:
        patch_activations = patch_activations.unsqueeze(0)  # Convert [S, d] -> [1, S, d]

    if patch_activations.ndim != 3:
        raise ValueError(
            f"Expected patch_activations of shape [B, S, d] or [S, d], got {patch_activations.shape}."
        )

    _, seq_len, _ = patch_activations.shape
    start_idx = 1 + k_registers

    if seq_len <= start_idx:
        raise ValueError(
            f"Sequence length S={seq_len} is too short for 1 [CLS] + {k_registers} registers. "
            f"At least {start_idx + 1} tokens are required to extract spatial patches."
        )

    # Slice spatial image patches only, omitting [CLS] and all register tokens
    spatial_tokens = patch_activations[:, start_idx:, :]  # [B, N, d]
    norms = torch.norm(spatial_tokens, p=2, dim=-1)  # [B, N]

    # Compute mean and std per-sample across the N spatial patch tokens
    mu = norms.mean(dim=-1, keepdim=True)  # [B, 1]
    sigma = norms.std(dim=-1, keepdim=True)  # [B, 1]
    threshold = mu + 3.0 * sigma  # [B, 1]

    outliers = (norms > threshold).float()  # [B, N]
    return float(outliers.mean().item())


def compute_layerwise_outlier_rate(
    activations_dict: Dict[int, torch.Tensor], k_registers: int = 0
) -> Dict[int, float]:
    """
    Computes patch outlier rates across all layers.

    :param activations_dict: Mapping of layer_idx -> activations tensor [B, S, d].
    :param k_registers: Number of register tokens in the model architecture.
    :return: Dictionary mapping layer_idx -> outlier rate float.
    """
    results: Dict[int, float] = {}
    for layer_idx, acts in activations_dict.items():
        results[layer_idx] = compute_patch_outlier_rate(acts, k_registers=k_registers)
    return results
