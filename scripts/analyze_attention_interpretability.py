#!/usr/bin/env python3
"""
Attention Interpretability & Diagnostic Metrics Analyzer.

Performs quantitative and qualitative evaluation of attention patterns:
  - Layer-wise Shannon attention entropy across all 12 MHSA layers
  - Patch-norm background outlier token rates (> 3 sigma threshold)
  - Attention concentration comparisons between baseline (K=0) and register (K>0) models

Author: Narmina Ibrahimova (Interpretability & Diagnostic Metrics Lead)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.metrics.entropy import compute_layerwise_entropy, compute_shannon_entropy
from src.metrics.outliers import compute_layerwise_outliers, detect_patch_norm_outliers
from src.models.attention_hook import ViTAttentionHookManager
from src.models.register_vit import RegisterVisionTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analyze_interpretability")


def analyze_checkpoint_interpretability(
    checkpoint_path: str,
    device: str = "cpu",
    batch_size: int = 4,
    image_size: int = 224,
) -> Dict[str, Any]:
    """
    Loads a checkpoint and computes comprehensive attention interpretability metrics
    over synthetic test batches.
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info("Loading checkpoint from %s...", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    k_registers = ckpt.get("k_registers", 0)
    if isinstance(ckpt.get("config"), dict):
        k_registers = ckpt["config"].get("model", {}).get("num_registers", k_registers)

    model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        num_classes=100,
        num_registers=k_registers,
    )
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])

    model.to(device)
    model.eval()

    with ViTAttentionHookManager(model) as hook_mgr:
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        with torch.no_grad():
            _ = model(x)
        attn_matrices = dict(hook_mgr.attention_maps)
        act_matrices = dict(hook_mgr.intermediate_activations)

    # Compute layerwise Shannon entropy
    layer_entropies = compute_layerwise_entropy(attn_matrices)
    
    # Compute layerwise outlier rates
    layer_outliers = compute_layerwise_outlier_rate(act_matrices, k_registers=k_registers)

    results = {
        "checkpoint": str(checkpoint_path),
        "k_registers": k_registers,
        "num_layers_evaluated": len(attn_matrices),
        "mean_l11_entropy": float(layer_entropies.get(11, 0.0)),
        "layerwise_entropy": {str(k): float(v) for k, v in layer_entropies.items()},
        "layerwise_outlier_rate": {str(k): float(v) for k, v in layer_outliers.items()},
    }

    logger.info("Completed interpretability analysis for K=%d (L11 entropy=%.4f bits)", k_registers, results["mean_l11_entropy"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze attention interpretability from ViT checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt or .pth).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to execute evaluation.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON metrics.")
    args = parser.parse_args()

    results = analyze_checkpoint_interpretability(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved interpretability diagnostics to %s", args.output_json)
    else:
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
