"""
Neuron Identification for Test-Time Register Allocation
======================================================
Identifies candidate "register neurons" in a target transformer block's MLP
whose activations are disproportionately concentrated on outlier-flagged
spatial patch tokens, following Stage 2.2 (Jiang et al., NeurIPS 2025).

Public Interface:
- `identify_register_neurons`: Discovers top-N candidate neurons at a given block.
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from src.metrics.outlier_mask import get_outlier_mask

logger = logging.getLogger(__name__)


def identify_register_neurons(
    model: nn.Module,
    calibration_images: torch.Tensor,
    target_block: int = 9,
    top_n: int = 4,
    device: Optional[torch.device] = None,
    k_registers: int = 0,
) -> List[int]:
    """
    Identifies candidate register neurons at target_block by measuring the
    activation gap between outlier patch tokens and normal patch tokens.

    :param model: Baseline K=0 model (or K=1 model with inactive register).
    :param calibration_images: Image tensor [B, 3, H, W] from validation or test split.
    :param target_block: Block index (0-indexed, e.g. 1, 9, 11, 12).
    :param top_n: Number of candidate neurons to select (default: 4).
    :param device: Compute device.
    :param k_registers: Number of register tokens present in the model sequence layout.
    :return: List of top_n neuron indices within [0, hidden_dim - 1].
    :raises IndexError: If target_block exceeds model depth.
    """
    if device is None:
        device = next(model.parameters()).device

    blocks = getattr(model, "blocks", None)
    if blocks is None and hasattr(model, "base_model"):
        blocks = model.base_model.blocks
    if blocks is None:
        raise AttributeError("Could not find .blocks in model.")

    if not (0 <= target_block < len(blocks)):
        raise IndexError(f"target_block {target_block} out of range [0, {len(blocks) - 1}].")

    block = blocks[target_block]
    if not hasattr(block, "mlp") or not hasattr(block.mlp, "act"):
        raise AttributeError(f"Block {target_block} does not have expected .mlp.act submodule.")

    captured: Dict[str, torch.Tensor] = {}

    def capture_mlp_act(module, inp, out):
        captured["mlp_act"] = out.detach()

    def capture_block_out(module, inp, out):
        if isinstance(out, torch.Tensor):
            captured["block_out"] = out.detach()
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            captured["block_out"] = out[0].detach()

    h_mlp = block.mlp.act.register_forward_hook(capture_mlp_act)
    h_block = block.register_forward_hook(capture_block_out)

    model.eval()
    with torch.no_grad():
        calibration_images = calibration_images.to(device)
        _ = model(calibration_images)

    h_mlp.remove()
    h_block.remove()

    mlp_act = captured["mlp_act"]      # [B, S, hidden_dim]
    block_out = captured["block_out"]  # [B, S, d]

    # Compute outlier mask on the residual stream
    outlier_mask = get_outlier_mask(block_out, k_registers=k_registers)  # [B, N] boolean

    # Spatial patch activations from MLP activation output
    start_idx = 1 + k_registers
    patch_acts = mlp_act[:, start_idx:, :]  # [B, N, hidden_dim]
    hidden_dim = patch_acts.shape[-1]

    num_outliers = int(outlier_mask.sum().item())
    total_patches = outlier_mask.numel()
    logger.info(
        "Block %d calibration: %d / %d outlier tokens flagged (%.2f%%).",
        target_block,
        num_outliers,
        total_patches,
        (num_outliers / total_patches) * 100.0 if total_patches > 0 else 0.0,
    )

    if num_outliers == 0:
        # Graceful fallback: take top 1% highest norm tokens as pseudo-outliers
        norms = torch.norm(patch_acts, p=2, dim=-1)  # [B, N]
        top_k_count = max(1, int(total_patches * 0.01))
        flat_norms = norms.flatten()
        threshold = flat_norms.topk(top_k_count).values[-1]
        outlier_mask = norms >= threshold
        logger.warning(
            "No 3-sigma outliers detected in calibration batch. "
            "Falling back to top %d (1%%) highest-norm patch tokens.",
            top_k_count,
        )

    outlier_tokens = patch_acts[outlier_mask]    # [num_outliers, hidden_dim]
    normal_tokens = patch_acts[~outlier_mask]   # [num_normal, hidden_dim]

    outlier_mean = outlier_tokens.mean(dim=0)    # [hidden_dim]
    normal_mean = normal_tokens.mean(dim=0)      # [hidden_dim]

    gap = (outlier_mean - normal_mean).abs()     # [hidden_dim]
    top_n = min(top_n, hidden_dim)
    register_neuron_ids = gap.topk(top_n).indices.tolist()

    logger.info(
        "Identified top %d candidate register neurons at block %d: %s (gaps: %s)",
        top_n,
        target_block,
        register_neuron_ids,
        [round(gap[idx].item(), 3) for idx in register_neuron_ids],
    )
    return register_neuron_ids
