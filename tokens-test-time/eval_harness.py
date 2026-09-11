"""
Evaluation Harness for Test-Time Register Ablations
==================================================
Runs pure inference loops on CIFAR-100 splits, collecting:
- Top-1 and Top-5 accuracy
- CrossEntropy loss
- Layerwise attention entropy (bits)
- Layerwise patch outlier rates (%)
- Mechanism diagnostics before vs. after redirection

Reuses the standardized evaluation metrics from `src.metrics` and
hook instrumentation from `src.models.ViTAttentionHookManager`.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import (
    compute_layerwise_entropy,
    compute_layerwise_outlier_rate,
)
from src.models import ViTAttentionHookManager

logger = logging.getLogger(__name__)


def compute_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    """Computes Top-K accuracy percentages."""
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


def evaluate_test_time_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_registers: int = 1,
    measure_interpretability: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates a model (with or without active redirection hooks) over a dataloader split.

    :param model: The PyTorch model in eval mode.
    :param dataloader: DataLoader over test or validation images.
    :param device: Compute device (CUDA / CPU).
    :param k_registers: Number of register tokens in model sequence.
    :param measure_interpretability: If True, captures attention entropy and outlier rates on batch 0.
    :return: Dictionary containing loss, top1_accuracy, top5_accuracy, entropy, outliers.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    layerwise_entropy: Dict[int, float] = {}
    layerwise_outliers: Dict[int, float] = {}

    hook_mgr: Optional[ViTAttentionHookManager] = None
    if measure_interpretability:
        hook_mgr = ViTAttentionHookManager(model)

    try:
        with torch.no_grad():
            for step, (images, targets) in enumerate(dataloader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                batch_size = images.size(0)

                # Execute forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)

                top1, top5 = compute_accuracy(outputs, targets, topk=(1, 5))
                total_loss += loss.item() * batch_size
                total_top1 += top1 * batch_size
                total_top5 += top5 * batch_size
                total_samples += batch_size

                # Intercept attention matrices & activations on batch 0, then disarm to save VRAM
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

    final_loss = total_loss / total_samples if total_samples > 0 else 0.0
    final_top1 = total_top1 / total_samples if total_samples > 0 else 0.0
    final_top5 = total_top5 / total_samples if total_samples > 0 else 0.0

    return {
        "loss": final_loss,
        "top1_accuracy": final_top1,
        "top5_accuracy": final_top5,
        "total_samples": total_samples,
        "layerwise_entropy": layerwise_entropy,
        "layerwise_outliers": layerwise_outliers,
    }
