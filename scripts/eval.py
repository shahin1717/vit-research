"""
Model Evaluation & Interpretability Analysis Harness
=====================================================
Evaluates trained Vision Transformer checkpoints with or without register tokens
on CIFAR-100 validation or test splits. Extracts Top-1/Top-5 accuracy, classification
loss, layer-wise attention entropy, and patch-norm outlier rates.

Public Interface & CLI:
- Run directly via command line:
    python scripts/eval.py --checkpoint checkpoints/rViT_K4_seed42/best_model.pt --k_registers 4
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct CLI execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

def get_autocast(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=enabled)
    return torch.cuda.amp.autocast(dtype=torch.float16, enabled=enabled)

from src.data.cifar100_subset import get_cifar100_loaders
from src.metrics import (
    compute_layerwise_entropy,
    compute_layerwise_outlier_rate,
)
from src.models import RegisterVisionTransformer, ViTAttentionHookManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    """Parses evaluation command line options."""
    parser = argparse.ArgumentParser(description="Evaluate Vision Transformer with Registers")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt or .pth)")
    parser.add_argument(
        "--k_registers",
        "--num_registers",
        dest="k_registers",
        type=int,
        default=None,
        help="Override register count K (0, 1, 4, 8)",
    )
    parser.add_argument("--backbone", type=str, default="vit_tiny_patch16_224", help="ViT backbone architecture")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset root directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save evaluation JSON results")
    return parser.parse_args()


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, 5)) -> List[float]:
    """Computes Top-K accuracy percentage."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float((correct_k / batch_size).item() * 100.0))
        return res


def main() -> None:
    """Main evaluation execution entrypoint."""
    args = parse_args()
    device = torch.device(args.device)

    # 1. Load Checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    logger.info("Loading checkpoint from: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Detect register count and model settings from checkpoint or CLI
    img_size = 224
    backbone = args.backbone
    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        model_cfg = checkpoint["config"].get("model", {})
        img_size = model_cfg.get("img_size", 224)
        if args.backbone == "vit_tiny_patch16_224" and "backbone" in model_cfg:
            backbone = model_cfg["backbone"]

    if args.k_registers is not None:
        k_registers = args.k_registers
    elif "k_registers" in checkpoint:
        k_registers = checkpoint["k_registers"]
    elif "config" in checkpoint and "model" in checkpoint["config"]:
        k_registers = checkpoint["config"]["model"].get("num_registers", 0)
    else:
        k_registers = 0

    logger.info("Evaluating architecture %s with K=%d register tokens on device: %s.", backbone, k_registers, device)

    # 2. Build Model and Load Weights
    model = RegisterVisionTransformer(
        model_name=backbone,
        num_classes=100,
        num_registers=k_registers,
        pretrained=False,  # Weights loaded from checkpoint
        img_size=img_size,
    ).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. Load Dataloader
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=img_size,
        download=True,
    )
    eval_loader = test_loader if args.split == "test" else val_loader
    logger.info("Evaluating split '%s' with %d samples across %d batches.", args.split, len(eval_loader.dataset), len(eval_loader))

    # 4. Evaluation Loop with Attention Hooks (Disarmed after batch 0 to prevent OOM/D2H thrash)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    layerwise_entropy: Dict[int, float] = {}
    layerwise_outliers: Dict[int, float] = {}

    hook_mgr: Optional[ViTAttentionHookManager] = ViTAttentionHookManager(model)
    try:
        with torch.no_grad():
            for step, (images, targets) in enumerate(eval_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                batch_size = images.size(0)

                with get_autocast(device.type, enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                top1, top5 = accuracy(outputs, targets, topk=(1, 5))
                total_loss += loss.item() * batch_size
                total_top1 += top1 * batch_size
                total_top5 += top5 * batch_size
                total_samples += batch_size

                # Intercept attention matrices & activations on batch 0 and immediately disarm
                if hook_mgr is not None and step == 0:
                    layerwise_entropy = compute_layerwise_entropy(hook_mgr.attention_maps)
                    layerwise_outliers = compute_layerwise_outlier_rate(
                        hook_mgr.intermediate_activations, k_registers=k_registers
                    )
                    hook_mgr.remove()
                    hook_mgr.clear()
                    hook_mgr = None
    finally:
        if hook_mgr is not None:
            hook_mgr.remove()
            hook_mgr.clear()

    final_loss = total_loss / total_samples
    final_top1 = total_top1 / total_samples
    final_top5 = total_top5 / total_samples

    logger.info("=== Evaluation Results (%s Split) ===", args.split.upper())
    logger.info("Loss: %.4f | Top-1 Accuracy: %.2f%% | Top-5 Accuracy: %.2f%%", final_loss, final_top1, final_top5)
    logger.info("Layerwise Attention Entropy (bits): %s", {k: round(v, 3) for k, v in layerwise_entropy.items()})
    logger.info("Layerwise Patch Outlier Rates: %s", {k: f"{v*100:.2f}%" for k, v in layerwise_outliers.items()})

    # 5. Export JSON Output
    results = {
        "checkpoint": str(ckpt_path),
        "k_registers": k_registers,
        "split": args.split,
        "metrics": {
            "loss": final_loss,
            "top1_accuracy": final_top1,
            "top5_accuracy": final_top5,
            "layerwise_entropy": layerwise_entropy,
            "layerwise_outliers": layerwise_outliers,
        },
    }

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Results successfully saved to: %s", out_path)


if __name__ == "__main__":
    main()
