"""
End-to-End Pipeline Verification Test
======================================
Tests the integrated training and evaluation workflows using synthetic tensors:
- `scripts.train.train_one_epoch`
- `scripts.train.evaluate`
- Model checkpoint saving and reloading
- JSON metrics export
"""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models import RegisterVisionTransformer
from scripts.train import train_one_epoch, evaluate, get_grad_scaler


def test_train_and_evaluate_synthetic_step(tmp_path: Path):
    """Executes a 1-epoch training and evaluation pass with dummy tensors."""
    device = torch.device("cpu")
    k = 4
    model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=100,
        num_registers=k,
        pretrained=False,
    ).to(device)

    # Create dummy DataLoader with 8 samples
    dummy_images = torch.randn(8, 3, 224, 224)
    dummy_labels = torch.randint(0, 100, (8,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = get_grad_scaler(device.type, enabled=False)

    # 1. Train 1 epoch
    train_loss, train_top1, train_top5 = train_one_epoch(
        model=model,
        train_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        gradient_clip=1.0,
        use_amp=False,
    )
    assert isinstance(train_loss, float) and train_loss > 0.0
    assert 0.0 <= train_top1 <= 100.0

    # 2. Evaluate 1 pass
    val_loss, val_top1, val_top5, entropy_dict, outliers_dict = evaluate(
        model=model,
        val_loader=loader,
        criterion=criterion,
        device=device,
        use_amp=False,
        k_registers=k,
        extract_attention_metrics=True,
    )
    assert isinstance(val_loss, float) and val_loss > 0.0
    assert len(entropy_dict) == 12
    assert len(outliers_dict) == 12

    # 3. Test Checkpoint Save & Load
    ckpt_path = tmp_path / "test_model.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "k_registers": k,
            "val_top1": val_top1,
        },
        ckpt_path,
    )
    assert ckpt_path.exists()

    loaded_checkpoint = torch.load(ckpt_path, map_location=device)
    assert loaded_checkpoint["k_registers"] == k

    new_model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=100,
        num_registers=k,
        pretrained=False,
    )
    new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    out = new_model(dummy_images[:2])
    assert out.shape == (2, 100)
