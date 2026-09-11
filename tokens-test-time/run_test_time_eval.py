"""
CLI Runner for Test-Time Register Ablations
==========================================
Executes Stage 1 (passive empty slot) and Stage 2 (neuron identification +
activation redirection) across single checkpoints or all multi-seed K=0 models.

Usage:
    # Run on default exp01 checkpoint (Seed 42)
    python tokens-test-time/run_test_time_eval.py

    # Run across all 3 random seeds (42, 1337, 3407) with statistical aggregation
    python tokens-test-time/run_test_time_eval.py --all_seeds --target_blocks 9,11

    # Run specific stage on custom checkpoint
    python tokens-test-time/run_test_time_eval.py --checkpoint outputs/exp01_k0_s42/best_model.pth --target_blocks 9,11,1,12
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar100_subset import get_cifar100_loaders
from src.models import RegisterVisionTransformer
from tokens_test_time.model_prep import build_stage1_model
from tokens_test_time.neuron_selection import identify_register_neurons
from tokens_test_time.redirection import TestTimeRegisterManager
from tokens_test_time.eval_harness import evaluate_test_time_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tokens-test-time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test-Time Registers Evaluation Runner")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/exp01_k0_s42/best_model.pth",
        help="Path to K=0 model checkpoint (.pth)",
    )
    parser.add_argument(
        "--all_seeds",
        action="store_true",
        help="Evaluate across all 3 trained K=0 seeds (42, 1337, 3407) and report mean ± std",
    )
    parser.add_argument(
        "--target_blocks",
        type=str,
        default="9,11",
        help="Comma-separated block indices for Stage 2 redirection (e.g. '9,11' or '1,9,11,12')",
    )
    parser.add_argument(
        "--top_n_neurons",
        type=int,
        default=4,
        help="Number of candidate neurons per target block to redirect (default: 4)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root path to CIFAR-100 dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Split to evaluate on (test or val)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokens-test-time/results",
        help="Directory to save output JSON results and logs",
    )
    return parser.parse_args()


def load_k0_baseline_model(
    ckpt_path: Path,
    device: torch.device,
    backbone: str = "vit_tiny_patch16_224",
    num_classes: int = 100,
) -> RegisterVisionTransformer:
    """Builds and loads the untouched K=0 baseline model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RegisterVisionTransformer(
        model_name=backbone,
        num_classes=num_classes,
        num_registers=0,
        pretrained=False,
    ).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_evaluation_for_checkpoint(
    ckpt_path: Path,
    eval_loader: Any,
    calib_images: torch.Tensor,
    device: torch.device,
    target_blocks: List[int],
    top_n_neurons: int,
) -> Dict[str, Any]:
    """Runs Baseline K=0, Stage 1 (passive slot), and Stage 2 (redirection) for a single checkpoint."""
    logger.info("================================================================")
    logger.info("Evaluating Checkpoint: %s", ckpt_path.name)
    logger.info("================================================================")

    ckpt_results: Dict[str, Any] = {"checkpoint": str(ckpt_path)}

    # -------------------------------------------------------------
    # 1. Baseline K=0 (Untouched)
    # -------------------------------------------------------------
    logger.info("--- [Arm 1] Baseline K=0 (Untouched Control) ---")
    k0_model = load_k0_baseline_model(ckpt_path, device)
    k0_metrics = evaluate_test_time_model(
        k0_model,
        eval_loader,
        device=device,
        k_registers=0,
        measure_interpretability=True,
    )
    logger.info(
        "K=0 Baseline -> Top-1: %.2f%% | Loss: %.4f | L11 Entropy: %.3f bits | L11 Outliers: %.2f%%",
        k0_metrics["top1_accuracy"],
        k0_metrics["loss"],
        k0_metrics["layerwise_entropy"].get(11, 0.0),
        k0_metrics["layerwise_outliers"].get(11, 0.0) * 100.0,
    )
    ckpt_results["baseline_k0"] = k0_metrics
    del k0_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # -------------------------------------------------------------
    # 2. Stage 1: Passive Empty Slot (Untrained K=1 Register, Zeroed)
    # -------------------------------------------------------------
    logger.info("--- [Arm 2] Stage 1: Passive Empty Slot (Untrained Zeroed K=1 Slot) ---")
    s1_model, missing, unexpected = build_stage1_model(
        ckpt_path,
        device=device,
        num_registers=1,
    )
    s1_metrics = evaluate_test_time_model(
        s1_model,
        eval_loader,
        device=device,
        k_registers=1,
        measure_interpretability=True,
    )
    logger.info(
        "Stage 1 (Passive K=1) -> Top-1: %.2f%% | Loss: %.4f | L11 Entropy: %.3f bits | L11 Outliers: %.2f%%",
        s1_metrics["top1_accuracy"],
        s1_metrics["loss"],
        s1_metrics["layerwise_entropy"].get(11, 0.0),
        s1_metrics["layerwise_outliers"].get(11, 0.0) * 100.0,
    )
    ckpt_results["stage1_passive_slot"] = s1_metrics

    # -------------------------------------------------------------
    # 3. Stage 2: Active Neuron Identification + Redirection
    # -------------------------------------------------------------
    ckpt_results["stage2_redirection"] = {}
    for block_idx in target_blocks:
        logger.info("--- [Arm 3] Stage 2: Active Redirection at Block %d ---", block_idx)

        # 3.1 Identify neurons on baseline calibration batch
        # Fresh K=0 model to identify neurons on natural unperturbed activations
        temp_k0 = load_k0_baseline_model(ckpt_path, device)
        candidate_neurons = identify_register_neurons(
            model=temp_k0,
            calibration_images=calib_images,
            target_block=block_idx,
            top_n=top_n_neurons,
            device=device,
            k_registers=0,
        )
        del temp_k0
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # 3.2 Mount redirection hook onto Stage 1 model and evaluate
        with TestTimeRegisterManager(
            s1_model,
            target_block=block_idx,
            neuron_ids=candidate_neurons,
            k_registers=1,
        ):
            s2_metrics = evaluate_test_time_model(
                s1_model,
                eval_loader,
                device=device,
                k_registers=1,
                measure_interpretability=True,
            )

        logger.info(
            "Stage 2 (Block %d Redirection, neurons=%s) -> Top-1: %.2f%% | Loss: %.4f | L11 Entropy: %.3f | L%d Outliers: %.2f%%",
            block_idx,
            candidate_neurons,
            s2_metrics["top1_accuracy"],
            s2_metrics["loss"],
            s2_metrics["layerwise_entropy"].get(11, 0.0),
            block_idx,
            s2_metrics["layerwise_outliers"].get(block_idx, 0.0) * 100.0,
        )
        ckpt_results["stage2_redirection"][f"block_{block_idx}"] = {
            "target_block": block_idx,
            "neuron_ids": candidate_neurons,
            "metrics": s2_metrics,
        }

    del s1_model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return ckpt_results


def aggregate_seed_results(results_list: List[Dict[str, Any]], target_blocks: List[int]) -> Dict[str, Any]:
    """Computes mean ± std for each treatment arm across seeds."""
    summary: Dict[str, Any] = {}

    # Extract Baseline K=0
    k0_accs = [r["baseline_k0"]["top1_accuracy"] for r in results_list]
    k0_losses = [r["baseline_k0"]["loss"] for r in results_list]
    summary["baseline_k0"] = {
        "top1_mean": float(np.mean(k0_accs)),
        "top1_std": float(np.std(k0_accs, ddof=1)) if len(k0_accs) > 1 else 0.0,
        "loss_mean": float(np.mean(k0_losses)),
        "loss_std": float(np.std(k0_losses, ddof=1)) if len(k0_losses) > 1 else 0.0,
    }

    # Extract Stage 1 Passive Slot
    s1_accs = [r["stage1_passive_slot"]["top1_accuracy"] for r in results_list]
    s1_losses = [r["stage1_passive_slot"]["loss"] for r in results_list]
    summary["stage1_passive_slot"] = {
        "top1_mean": float(np.mean(s1_accs)),
        "top1_std": float(np.std(s1_accs, ddof=1)) if len(s1_accs) > 1 else 0.0,
        "loss_mean": float(np.mean(s1_losses)),
        "loss_std": float(np.std(s1_losses, ddof=1)) if len(s1_losses) > 1 else 0.0,
    }

    # Extract Stage 2 for each block
    summary["stage2_redirection"] = {}
    for block_idx in target_blocks:
        key = f"block_{block_idx}"
        s2_accs = [r["stage2_redirection"][key]["metrics"]["top1_accuracy"] for r in results_list]
        s2_losses = [r["stage2_redirection"][key]["metrics"]["loss"] for r in results_list]
        summary["stage2_redirection"][key] = {
            "target_block": block_idx,
            "top1_mean": float(np.mean(s2_accs)),
            "top1_std": float(np.std(s2_accs, ddof=1)) if len(s2_accs) > 1 else 0.0,
            "loss_mean": float(np.mean(s2_losses)),
            "loss_std": float(np.std(s2_losses, ddof=1)) if len(s2_losses) > 1 else 0.0,
        }

    return summary


def print_comparison_markdown_table(summary: Dict[str, Any], target_blocks: List[int]) -> None:
    """Prints a clear Markdown table comparing all test-time results with Table I."""
    print("\n" + "=" * 80)
    print("🎯 TEST-TIME REGISTERS ABLATION SUMMARY (vs. Table I Main Results)")
    print("=" * 80)
    print("| Configuration | Intervention Type | Test Top-1 Accuracy (%) | Test Loss | Source / Status |")
    print("|---|---|---|---|---|")
    print(f"| **Control Baseline ($K=0$)** | Standard Trained ViT-Tiny | **{summary['baseline_k0']['top1_mean']:.2f} ± {summary['baseline_k0']['top1_std']:.2f}** | {summary['baseline_k0']['loss_mean']:.4f} | Table I (Main Study) |")
    print("| **Trained Registers ($K=1$)** | Fine-tuned 1 Register | 74.66 ± 0.42 | 0.985 ± 0.015 | Table I (Main Study) |")
    print(f"| **Stage 1: Passive Slot ($K=1$)** | Untrained, Zeroed Slot (No Redir) | {summary['stage1_passive_slot']['top1_mean']:.2f} ± {summary['stage1_passive_slot']['top1_std']:.2f} | {summary['stage1_passive_slot']['loss_mean']:.4f} | **New (Jiang et al. Stage 1)** |")

    for block_idx in target_blocks:
        s2_info = summary["stage2_redirection"][f"block_{block_idx}"]
        print(
            f"| **Stage 2: Redirection (Block {block_idx})** | Top-4 Neurons $\\to$ Reg Slot | "
            f"{s2_info['top1_mean']:.2f} ± {s2_info['top1_std']:.2f} | {s2_info['loss_mean']:.4f} | **New (Jiang et al. Stage 2)** |"
        )
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    target_blocks_raw = [int(b.strip()) for b in args.target_blocks.split(",") if b.strip()]
    target_blocks: List[int] = []
    for b in target_blocks_raw:
        if b == 12:
            logger.info("Block 12 requested: mapping to 0-indexed block 11 (final layer of 12-block ViT).")
            target_blocks.append(11)
        elif 0 <= b <= 11:
            target_blocks.append(b)
        else:
            raise ValueError(f"Block {b} is out of bounds for 12-block ViT (valid: 0..11 or 12).")

    # Remove duplicates preserving order
    seen = set()
    target_blocks = [b for b in target_blocks if not (b in seen or seen.add(b))]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Test-Time Registers Ablation...")
    logger.info("Device: %s | Target Blocks: %s | Top N Neurons: %d", device, target_blocks, args.top_n_neurons)

    # 1. Load Data
    _, val_loader, test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
        image_size=224,
        download=True,
    )
    eval_loader = test_loader if args.split == "test" else val_loader
    logger.info("Loaded %s split with %d samples.", args.split, len(eval_loader.dataset))

    # 2. Extract Calibration Batch
    calib_images, _ = next(iter(eval_loader))
    calib_images = calib_images[:32].to(device)  # 32 images for clean calibration
    logger.info("Extracted calibration batch of shape: %s", calib_images.shape)

    # 3. Determine Checkpoints
    if args.all_seeds:
        checkpoints = [
            Path("outputs/exp01_k0_s42/best_model.pth"),
            Path("outputs/exp02_k0_s1337/best_model.pth"),
            Path("outputs/exp03_k0_s3407/best_model.pth"),
        ]
        # Filter existing
        checkpoints = [p for p in checkpoints if p.is_file()]
        if not checkpoints:
            raise FileNotFoundError("Could not locate multi-seed K=0 checkpoints in outputs/exp0[1-3]_k0_*/best_model.pth")
    else:
        checkpoints = [Path(args.checkpoint)]

    # 4. Run Evaluations
    all_results = []
    for ckpt in checkpoints:
        res = run_evaluation_for_checkpoint(
            ckpt_path=ckpt,
            eval_loader=eval_loader,
            calib_images=calib_images,
            device=device,
            target_blocks=target_blocks,
            top_n_neurons=args.top_n_neurons,
        )
        all_results.append(res)

    # 5. Statistical Aggregation
    summary = aggregate_seed_results(all_results, target_blocks)
    print_comparison_markdown_table(summary, target_blocks)

    # 6. Save JSON Deliverables
    results_payload = {
        "args": vars(args),
        "target_blocks": target_blocks,
        "summary": summary,
        "runs": all_results,
    }
    out_file = output_dir / "test_time_registers_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    logger.info("Saved full results JSON to: %s", out_file)


if __name__ == "__main__":
    main()
