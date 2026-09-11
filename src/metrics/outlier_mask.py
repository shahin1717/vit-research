"""
Patch-Norm Outlier Mask Utility (3-Sigma Bound)
==============================================
Provides token-level boolean outlier masking for spatial patches in
Vision Transformers, following Darcet et al. (ICLR 2024) and test-time
activation redirection mechanisms (Jiang et al., NeurIPS 2025).

Mathematical Formulation:
    n_i^(l)   = || x_i^(l) ||_2        (spatial patch i in 1..N)
    mu_l      = mean(n_i^(l))          across spatial patches
    sigma_l   = std(n_i^(l))           across spatial patches
    Mask_i    = (n_i^(l) > mu_l + 3 * sigma_l)

Public Interface:
- `get_outlier_mask`: Returns boolean tensor [B, N] marking outlier tokens.
"""

from typing import Union
import torch


def get_outlier_mask(
    patch_activations: torch.Tensor,
    k_registers: int = 0,
) -> torch.Tensor:
    """
    Computes a per-token boolean outlier mask over spatial image patch tokens
    based on the 3-sigma bound: ||x_i||_2 > mu + 3 * sigma.

    Sequence Layout:
        Index 0:                 [CLS]
        Index 1 .. K:            [Register Tokens / Slots]
        Index (1+K) .. end:      [Spatial Patch Tokens] (N tokens)

    :param patch_activations: [B, S, d] or [S, d] intermediate activations at a layer.
    :param k_registers: Number of register tokens prepended after [CLS] (e.g. 0, 1, 4, 8).
    :return: Boolean tensor of shape [B, N] where True indicates an outlier patch token.
    :raises ValueError: If k_registers is negative or sequence length is insufficient.
    """
    if k_registers < 0:
        raise ValueError(f"k_registers cannot be negative, got {k_registers}.")

    if patch_activations.ndim == 2:
        patch_activations = patch_activations.unsqueeze(0)

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

    # Slice spatial image patches only, omitting [CLS] and register slots
    spatial_tokens = patch_activations[:, start_idx:, :]  # [B, N, d]
    norms = torch.norm(spatial_tokens, p=2, dim=-1)  # [B, N]

    mu = norms.mean(dim=-1, keepdim=True)  # [B, 1]
    sigma = norms.std(dim=-1, keepdim=True)  # [B, 1]
    threshold = mu + 3.0 * sigma  # [B, 1]

    return norms > threshold  # [B, N] boolean tensor
