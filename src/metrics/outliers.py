"""
Patch-Norm Outlier Rate (3-sigma bound)
=========================================
Darcet et al. (2023) show that background patch tokens in register-free
ViTs develop abnormally HIGH-NORM activations ("attention sinks"), while
register-equipped ViTs push that role onto the register tokens instead,
leaving spatial patch norms well-behaved.

This metric quantifies that directly: for each layer, what fraction of
spatial patch tokens have an L2 activation norm more than 3 standard
deviations above the layer's mean norm?

    n_i^(l)      = || x_i^(l) ||_2
    mu_l, sigma_l = mean / std of n_i^(l) over the N spatial patches
    theta_l      = mu_l + 3 * sigma_l
    OutlierRate  = (1/N) * sum_i  1[ n_i^(l) > theta_l ]
"""

import torch
from typing import Dict


def compute_patch_outlier_rate(patch_activations: torch.Tensor, k_registers: int = 0) -> float:
    """
    Computes the fraction of spatial patch tokens whose L2 norm exceeds
    mu + 3*sigma for that layer.

    Args:
        patch_activations: [B, S, d] hidden states at some layer, where
            S = 1 ([CLS]) + k_registers + 196 (spatial patches). The
            token ORDER in timm/DeiT-style ViTs is [CLS, registers...,
            patch_0, ..., patch_195] -- CLS and registers come first.
        k_registers: number of register tokens prepended after [CLS].
            Must match the model config (0, 1, 4, or 8 in this project).
            Getting this wrong silently corrupts the metric, because it
            shifts which tokens are treated as "spatial patches" -- see
            the Token Index Offset gotcha below.

    Returns:
        Scalar outlier rate in [0, 1], averaged over the batch.
    """
    # Never hardcode `[:, 1:]` -- with registers present, that slice would
    # incorrectly include register tokens as if they were spatial patches.
    # Always skip exactly (1 + k_registers) leading tokens.
    spatial_tokens = patch_activations[:, (1 + k_registers):, :]  # [B, N, d]

    norms = torch.norm(spatial_tokens, p=2, dim=-1)  # [B, N]

    # mean/std computed PER SAMPLE, per layer (over the N patch tokens),
    # matching the formulation: mu_l = (1/N) * sum_i n_i^(l)
    mu = norms.mean(dim=-1, keepdim=True)     # [B, 1]
    sigma = norms.std(dim=-1, keepdim=True)   # [B, 1]
    threshold = mu + 3.0 * sigma              # [B, 1]

    outliers = (norms > threshold).float()    # [B, N]
    return float(outliers.mean().item())


def compute_layerwise_outlier_rate(
    activations_dict: Dict[int, torch.Tensor], k_registers: int = 0
) -> Dict[int, float]:
    """
    Takes a dict of layer_idx -> activations [B, S, d] (e.g. the
    `intermediate_activations` produced by ViTAttentionHookManager) and
    returns layer_idx -> outlier_rate, so you can plot it vs. depth and
    compare K=0 (baseline) against K>=1 (register) models.
    """
    results = {}
    for layer_idx, acts in activations_dict.items():
        results[layer_idx] = compute_patch_outlier_rate(acts, k_registers=k_registers)
    return results
