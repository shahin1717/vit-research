"""
Model Preparation for Test-Time Registers (Stage 1 & Stage 2)
=============================================================
Provides loading routines to convert a standard trained K=0 Vision Transformer
into an architecture with K>=1 untrained register slots, following Stage 1
conventions (Jiang et al., NeurIPS 2025).

Public Interface:
- `build_stage1_model`: Loads trained K=0 weights into K=1 wrapper with zeroed registers.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from src.models.register_vit import RegisterVisionTransformer

logger = logging.getLogger(__name__)


def build_stage1_model(
    checkpoint_path: Union[str, Path],
    backbone: str = "vit_tiny_patch16_224",
    num_classes: int = 100,
    device: Optional[torch.device] = None,
    num_registers: int = 1,
    img_size: int = 224,
) -> Tuple[RegisterVisionTransformer, List[str], List[str]]:
    """
    Loads a trained K=0 Vision Transformer checkpoint into a RegisterVisionTransformer
    with num_registers >= 1 empty slots. Per Jiang et al. (2025), the register slot
    is zero-initialized (never trained) and remains inert unless explicitly redirected.

    :param checkpoint_path: Path to the trained K=0 model checkpoint (.pth or .pt).
    :param backbone: ViT architecture name (default: 'vit_tiny_patch16_224').
    :param num_classes: Number of classification targets (default: 100).
    :param device: Target compute device (CUDA / CPU).
    :param num_registers: Number of register slots to prepend (default: 1).
    :param img_size: Spatial image input resolution in pixels (default: 224).
    :return: Tuple of (model, missing_keys, unexpected_keys).
    :raises FileNotFoundError: If checkpoint path does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        # Check fallback in checkpoints/ or outputs/ if relative path given
        alt_paths = [
            Path("checkpoints") / ckpt_path.name,
            Path("outputs") / ckpt_path.parent.name / ckpt_path.name,
        ]
        found = False
        for alt in alt_paths:
            if alt.is_file():
                ckpt_path = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info("Loading Stage 1 base checkpoint from: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Auto-detect config if stored in checkpoint
    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        model_cfg = checkpoint["config"].get("model", {})
        img_size = model_cfg.get("img_size", img_size)
        if backbone == "vit_tiny_patch16_224" and "backbone" in model_cfg:
            backbone = model_cfg["backbone"]

    # Instantiate model with num_registers slot(s)
    model = RegisterVisionTransformer(
        model_name=backbone,
        num_classes=num_classes,
        num_registers=num_registers,
        pretrained=False,
        img_size=img_size,
    ).to(device)

    # Extract state_dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Load with strict=False: registers key will be missing because K=0 checkpoint has no registers
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Model loaded. Missing keys: %s | Unexpected keys: %s", missing, unexpected)

    # Conform strictly to Jiang et al.'s zero-initialized register convention
    if model.registers is not None:
        with torch.no_grad():
            model.registers.zero_()
        logger.info("Zero-initialized %d register slot(s) at index 1..%d.", num_registers, num_registers)

    model.eval()
    return model, missing, unexpected
