"""
Unified ViT Training Engine
============================
Production-ready PyTorch training pipeline for Vision Transformers with registers
under data scarcity (Track 1 research).

Features:
- Full support for `RegisterVisionTransformer` with K in {0, 1, 4, 8} register tokens.
- Automatic Mixed Precision (AMP) with FP16 and unscaled gradient clipping.
- AdamW optimizer + linear warmup and cosine annealing learning rate scheduler.
- Label-smoothed CrossEntropyLoss (0.1).
- Integration with `ViTAttentionHookManager` for layerwise entropy & outlier extraction.
- Generalization gap tracking (Delta_L = L_val - L_train, Delta_Acc = Acc_train - Acc_val).
- Checkpoint serialization (`best_model.pt`, `latest_checkpoint.pt`) and JSON/CSV logging.
- Configuration loading from YAML with CLI overrides.
"""

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path for direct CLI execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
import yaml

# Forward-compatible AMP helpers for PyTorch >= 2.0
def get_autocast(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=enabled)
    return torch.cuda.amp.autocast(dtype=torch.float16, enabled=enabled)

def get_grad_scaler(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device=device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)

from src.data.cifar100_subset import get_cifar100_loaders
from src.metrics import (
    compute_accuracy_gap,
    compute_generalization_gap,
    compute_layerwise_entropy,
    compute_layerwise_outlier_rate,
    GeneralizationGapTracker,
)
from src.models import RegisterVisionTransformer, ViTAttentionHookManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Deterministic seed set to %d.", seed)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments and optional YAML config."""
    parser = argparse.ArgumentParser(description="Train Vision Transformer with Registers on CIFAR-100")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML experiment config")
    parser.add_argument(
        "--k_registers",
        "--num_registers",
        dest="k_registers",
        type=int,
        default=None,
        help="Number of register tokens (0, 1, 4, 8)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate")
    parser.add_argument("--min_lr", type=float, default=None, help="Minimum cosine LR")
    parser.add_argument("--weight_decay", type=float, default=None, help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="Linear warmup epochs")
    parser.add_argument("--gradient_clip", type=float, default=None, help="Max gradient norm")
    parser.add_argument("--amp", action="store_true", default=None, help="Enable AMP mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="Force disable AMP")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to metrics output directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to model checkpoints directory")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader worker threads")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', or auto)")
    parser.add_argument("--eval_freq", type=int, default=None, help="Validation frequency in epochs")
    parser.add_argument("--backbone", type=str, default=None, help="ViT model architecture name")
    parser.add_argument("--pretrained", action="store_true", default=None, help="Use ImageNet pretrained backbone")
    parser.add_argument("--no_pretrained", action="store_true", help="Train backbone from scratch")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads default configuration, merges YAML config, and applies CLI overrides."""
    config: Dict[str, Any] = {
        "experiment": {
            "name": "vit_tiny_rViT",
            "seed": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "model": {
            "backbone": "vit_tiny_patch16_224",
            "pretrained": True,
            "num_classes": 100,
            "num_registers": 0,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "img_size": 224,
        },
        "data": {
            "data_dir": "./data",
            "batch_size": 64,
            "num_workers": 2,
            "samples_per_class": 100,
            "val_split": 0.1,
            "image_size": 224,
        },
        "training": {
            "epochs": 50,
            "lr": 5e-4,
            "min_lr": 1e-5,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "amp": True,
            "gradient_clip": 1.0,
            "label_smoothing": 0.1,
        },
        "logging": {
            "output_dir": "./outputs",
            "checkpoint_dir": "./checkpoints",
            "eval_freq": 1,
            "save_attention_maps": True,
        },
    }

    # Merge YAML file if provided
    yaml_cfg: Dict[str, Any] = {}
    if args.config and os.path.isfile(args.config):
        logger.info("Loading config file: %s", args.config)
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
            for section, values in yaml_cfg.items():
                if section in config and isinstance(values, dict):
                    config[section].update(values)
                else:
                    config[section] = values

    # Normalize YAML synonym keys for logging and data
    log_sec = config.get("logging", {})
    if "save_dir" in log_sec and "output_dir" not in yaml_cfg.get("logging", {}):
        config["logging"]["output_dir"] = log_sec["save_dir"]
    if "eval_freq_epochs" in log_sec and "eval_freq" not in yaml_cfg.get("logging", {}):
        config["logging"]["eval_freq"] = log_sec["eval_freq_epochs"]

    data_sec = config.get("data", {})
    if "image_size" in data_sec:
        config["model"]["img_size"] = data_sec["image_size"]

    # Apply CLI overrides
    if args.k_registers is not None:
        config["model"]["num_registers"] = args.k_registers
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.min_lr is not None:
        config["training"]["min_lr"] = args.min_lr
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        config["training"]["warmup_epochs"] = args.warmup_epochs
    if args.gradient_clip is not None:
        config["training"]["gradient_clip"] = args.gradient_clip
    if args.amp is not None:
        config["training"]["amp"] = True
    if args.no_amp:
        config["training"]["amp"] = False
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir
    if args.output_dir is not None:
        config["logging"]["output_dir"] = args.output_dir
    if args.checkpoint_dir is not None:
        config["logging"]["checkpoint_dir"] = args.checkpoint_dir
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.device is not None:
        config["experiment"]["device"] = args.device
    if args.eval_freq is not None:
        config["logging"]["eval_freq"] = args.eval_freq
    if args.backbone is not None:
        config["model"]["backbone"] = args.backbone
    if args.pretrained is not None:
        config["model"]["pretrained"] = True
    if args.no_pretrained:
        config["model"]["pretrained"] = False

    # Standardize experiment name
    k_val = config["model"]["num_registers"]
    seed_val = config["experiment"]["seed"]
    if "name" not in config["experiment"] or config["experiment"]["name"] in (
        "vit_tiny_rViT", "vit_tiny_baseline_k0", "vit_tiny_k1", "vit_tiny_k4", "vit_tiny_k8", "default"
    ):
        config["experiment"]["name"] = f"rViT_K{k_val}_seed{seed_val}"

    return config


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, 5)) -> List[float]:
    """Computes Top-K precision for specified values of k."""
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


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    gradient_clip: float,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Executes a single epoch of training."""
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        with get_autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Scale gradients and step optimizer
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        # Compute accuracy metrics
        top1, top5 = accuracy(outputs, targets, topk=(1, 5))
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_top1 = total_top1 / total_samples
    epoch_top5 = total_top5 / total_samples
    return epoch_loss, epoch_top1, epoch_top5


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    k_registers: int,
    extract_attention_metrics: bool = True,
) -> Tuple[float, float, float, Dict[int, float], Dict[int, float]]:
    """Evaluates the model on validation data and computes interpretability metrics."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    layerwise_entropy: Dict[int, float] = {}
    layerwise_outliers: Dict[int, float] = {}

    # Attach hook manager if diagnostic metrics are requested
    hook_mgr: Optional[ViTAttentionHookManager] = None
    if extract_attention_metrics:
        hook_mgr = ViTAttentionHookManager(model)

    try:
        for step, (images, targets) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            with get_autocast(device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            top1, top5 = accuracy(outputs, targets, topk=(1, 5))
            total_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size

            # Intercept attention matrices from the first batch
            # Intercept attention matrices from the first batch and immediately disarm hooks
            # to eliminate device-to-host memory transfer overhead on all subsequent batches
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

    eval_loss = total_loss / total_samples
    eval_top1 = total_top1 / total_samples
    eval_top5 = total_top5 / total_samples
    return eval_loss, eval_top1, eval_top5, layerwise_entropy, layerwise_outliers


def main() -> None:
    """Main training execution entrypoint."""
    args = parse_args()
    cfg = load_config(args)

    exp_name = cfg["experiment"]["name"]
    seed = cfg["experiment"]["seed"]
    device_str = cfg["experiment"]["device"]
    device = torch.device(device_str if (torch.cuda.is_available() and device_str == "cuda") else "cpu")
    use_amp = cfg["training"]["amp"] and (device.type == "cuda")

    set_seed(seed)

    # Prepare file paths
    base_out = Path(cfg["logging"]["output_dir"])
    if args.output_dir is not None and base_out.name != "outputs":
        output_dir = base_out
    elif base_out.name in ("outputs", "."):
        output_dir = base_out / exp_name
    else:
        output_dir = base_out

    base_ckpt = Path(cfg["logging"]["checkpoint_dir"])
    if args.checkpoint_dir is not None and base_ckpt.name != "checkpoints":
        checkpoint_dir = base_ckpt
    elif base_ckpt.name in ("checkpoints", "."):
        checkpoint_dir = base_ckpt / exp_name
    else:
        checkpoint_dir = base_ckpt

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Experiment: %s ===", exp_name)
    logger.info("Device: %s | AMP: %s | Registers K=%d | Seed: %d", device, use_amp, cfg["model"]["num_registers"], seed)

    # 1. Prepare Dataloaders (Translate val_split -> train_ratio and supply image_size)
    img_size = cfg["model"].get("img_size", cfg["data"].get("image_size", 224))
    val_split = float(cfg["data"].get("val_split", 0.1))
    train_ratio = 1.0 - val_split

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        samples_per_class=cfg["data"]["samples_per_class"],
        train_ratio=train_ratio,
        image_size=img_size,
        seed=seed,
        download=True,
    )
    logger.info("Dataloaders ready. Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

    # 2. Build Model
    k_registers = cfg["model"]["num_registers"]
    model = RegisterVisionTransformer(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        num_registers=k_registers,
        pretrained=cfg["model"]["pretrained"],
        drop_rate=cfg["model"]["drop_rate"],
        attn_drop_rate=cfg["model"]["attn_drop_rate"],
        img_size=cfg["model"]["img_size"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model built. Total Parameters: %d (Trainable: %d)", total_params, trainable_params)

    # 3. Setup Loss, Optimizer, Scaler, and Schedulers
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(0.9, 0.999),
    )
    scaler = get_grad_scaler(device.type, enabled=use_amp)

    epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]

    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_sched = LambdaLR(optimizer, lr_lambda=lambda ep: float(ep + 1) / float(warmup_epochs))
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=cfg["training"]["min_lr"]
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["training"]["min_lr"])

    # 4. Initialize Metric Trackers
    gap_tracker = GeneralizationGapTracker()
    history: List[Dict[str, Any]] = []
    best_val_acc = -1.0
    best_epoch = 0

    csv_file = output_dir / "summary.csv"
    train_history_file = output_dir / "train_history.csv"
    for path_csv in (csv_file, train_history_file):
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_top1", "val_loss", "val_top1", "gen_gap", "acc_gap", "lr"])

    # 5. Epoch Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_top1, train_top5 = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            gradient_clip=cfg["training"]["gradient_clip"],
            use_amp=use_amp,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Evaluate at eval_freq or last epoch
        val_loss, val_top1, val_top5 = 0.0, 0.0, 0.0
        gen_gap, acc_gap = 0.0, 0.0
        entropy_dict: Dict[int, float] = {}
        outliers_dict: Dict[int, float] = {}

        if epoch % cfg["logging"]["eval_freq"] == 0 or epoch == epochs:
            val_loss, val_top1, val_top5, entropy_dict, outliers_dict = evaluate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                k_registers=k_registers,
                extract_attention_metrics=cfg["logging"]["save_attention_maps"],
            )
            gen_gap = gap_tracker.update(epoch, train_loss, val_loss)
            acc_gap = compute_accuracy_gap(train_top1, val_top1)

        epoch_duration = time.time() - t0

        logger.info(
            "Epoch [%02d/%02d] (%.1fs) | Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%% | Gap: %.4f | LR: %.6f",
            epoch,
            epochs,
            epoch_duration,
            train_loss,
            train_top1,
            val_loss,
            val_top1,
            gen_gap,
            current_lr,
        )

        # Log to both summary.csv and train_history.csv for complete contractual compliance
        csv_row = [epoch, f"{train_loss:.4f}", f"{train_top1:.2f}", f"{val_loss:.4f}", f"{val_top1:.2f}", f"{gen_gap:.4f}", f"{acc_gap:.2f}", f"{current_lr:.6f}"]
        for path_csv in (csv_file, train_history_file):
            with open(path_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)

        # Record structured epoch history
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "train_top5": train_top5,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "generalization_gap": gen_gap,
            "accuracy_gap": acc_gap,
            "lr": current_lr,
            "duration_sec": epoch_duration,
            "layerwise_entropy": entropy_dict,
            "layerwise_outliers": outliers_dict,
        }
        history.append(epoch_record)

        # Save Checkpoint if best validation accuracy (or first epoch)
        is_best = (val_top1 > best_val_acc) or (epoch == 1 and best_val_acc < 0.0)
        if is_best:
            best_val_acc = val_top1
            best_epoch = epoch
            best_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top1": val_top1,
                "val_loss": val_loss,
                "k_registers": k_registers,
                "seed": seed,
                "config": cfg,
            }
            torch.save(best_payload, checkpoint_dir / "best_model.pt")
            torch.save(best_payload, checkpoint_dir / "best_model.pth")
            if output_dir != checkpoint_dir:
                torch.save(best_payload, output_dir / "best_model.pth")
                torch.save(best_payload, output_dir / "best_model.pt")
            logger.info("Saved new best model checkpoint (Val Acc: %.2f%%) at epoch %d.", best_val_acc, epoch)

        # Save latest checkpoint
        latest_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_top1": val_top1,
            "val_loss": val_loss,
            "k_registers": k_registers,
            "seed": seed,
        }
        torch.save(latest_payload, checkpoint_dir / "latest_checkpoint.pt")
        torch.save(latest_payload, checkpoint_dir / "last_model.pth")
        if output_dir != checkpoint_dir:
            torch.save(latest_payload, output_dir / "last_model.pth")
            torch.save(latest_payload, output_dir / "latest_checkpoint.pt")

    total_training_time = time.time() - start_time
    logger.info("Training complete in %.2f minutes. Best Val Acc: %.2f%% at epoch %d.", total_training_time / 60.0, best_val_acc, best_epoch)

    # 6. Final Evaluation on Untouched Test Set using Best Model
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    if not best_ckpt_path.exists():
        best_ckpt_path = output_dir / "best_model.pth"
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded best model from epoch %d for final test set evaluation.", checkpoint["epoch"])

    test_loss, test_top1, test_top5, test_entropy, test_outliers = evaluate(
        model=model,
        val_loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        k_registers=k_registers,
        extract_attention_metrics=True,
    )

    logger.info("=== Final Test Results === Top-1: %.2f%% | Top-5: %.2f%% | Test Loss: %.4f", test_top1, test_top5, test_loss)

    # 7. Serialize Metrics JSON
    metrics_export = {
        "experiment": exp_name,
        "k_registers": k_registers,
        "seed": seed,
        "config": cfg,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_training_time_sec": total_training_time,
        "best_val_top1": best_val_acc,
        "best_epoch": best_epoch,
        "test_results": {
            "test_loss": test_loss,
            "test_top1": test_top1,
            "test_top5": test_top5,
            "layerwise_entropy": test_entropy,
            "layerwise_outliers": test_outliers,
        },
        "history": history,
    }

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    logger.info("Metrics successfully exported to %s", metrics_file)


if __name__ == "__main__":
    main()
