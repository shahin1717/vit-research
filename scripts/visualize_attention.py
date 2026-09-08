"""
[CLS] Spatial Attention Extraction and Overlay
==============================================
Builds the qualitative mechanism figure: the ``[CLS]`` query's attention over
image patches, at four depths, for a control checkpoint and a register
checkpoint, rendered on the same input image with one shared colour scale.

The tensor work is the part that has to be exactly right. Three details decide
whether the picture means anything:

1. **Register offset.** The token sequence is
   ``[CLS], r_1 ... r_K, p_1 ... p_196``. The spatial row of the attention
   matrix therefore starts at index ``1 + K``. Slicing from index 1 instead
   would fold the register columns into the map and still produce a plausible
   looking heatmap, so the slice width is asserted against the patch count.
2. **Fused attention.** Recent timm builds run scaled-dot-product attention,
   which never materialises the ``[B, H, S, S]`` matrix.
   :class:`~src.models.attention_hook.ViTAttentionHookManager` reconstructs the
   softmax weights from the Q/K projections, and this script relies on that.
3. **Shared normalisation.** Both rows are drawn with a single colour
   normalisation computed over every panel of the figure. Per-panel
   autoscaling would make two different distributions look identical.

The comparison also uses one fixed input image for every panel, selected by
index from the untouched CIFAR-100 test split, so the two rows differ only in
the checkpoint that produced them.

Public Interface
----------------
- :func:`load_checkpoint_model`
- :func:`extract_cls_attention`
- :func:`render_comparison`
- CLI: ``python scripts/visualize_attention.py --baseline ... --registers ...``
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import CIFAR100_MEAN, CIFAR100_STD, get_cifar100_transforms
from src.models.attention_hook import ViTAttentionHookManager
from src.models.register_vit import RegisterVisionTransformer

DEFAULT_LAYERS: Sequence[int] = (1, 4, 8, 12)


def _resolve_registers(checkpoint: Dict[str, object], override: Optional[int]) -> int:
    """
    Determines how many register tokens a checkpoint was trained with.

    The count is read from the checkpoint rather than assumed, because loading
    weights into a model built with the wrong ``K`` either raises on the
    register parameter or silently mis-aligns the token sequence.

    :param checkpoint: A loaded checkpoint payload.
    :param override: Explicit count from the command line, or ``None``.
    :return: The register count.
    :raises KeyError: If the checkpoint records no register count and none is given.
    """
    if override is not None:
        return override
    if "k_registers" in checkpoint:
        return int(checkpoint["k_registers"])
    config = checkpoint.get("config")
    if isinstance(config, dict) and isinstance(config.get("model"), dict):
        return int(config["model"].get("num_registers", 0))
    raise KeyError(
        "Checkpoint records no register count; pass --baseline_k / --registers_k explicitly."
    )


def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device,
    num_registers: Optional[int] = None,
    backbone: str = "vit_tiny_patch16_224",
) -> Tuple[RegisterVisionTransformer, int]:
    """
    Rebuilds the trained architecture and loads its weights.

    :param checkpoint_path: Path to a ``best_model.pth`` produced by ``scripts/train.py``.
    :param device: Device to place the model on.
    :param num_registers: Overrides the register count recorded in the checkpoint.
    :param backbone: timm architecture name.
    :return: The evaluation-mode model and its register count.
    :raises FileNotFoundError: If the checkpoint does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    k_registers = _resolve_registers(checkpoint, num_registers)

    img_size = 224
    config = checkpoint.get("config")
    if isinstance(config, dict) and isinstance(config.get("model"), dict):
        img_size = int(config["model"].get("img_size", 224))
        backbone = config["model"].get("backbone", backbone)

    model = RegisterVisionTransformer(
        model_name=backbone,
        num_classes=100,
        num_registers=k_registers,
        pretrained=False,
        img_size=img_size,
    ).to(device)

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, k_registers


def load_test_image(data_dir: str, index: int, image_size: int = 224) -> Tuple[torch.Tensor, int]:
    """
    Loads one image from the untouched CIFAR-100 test split.

    The evaluation transform is used, so the tensor the model sees is exactly
    the tensor the reported test metrics were computed on.

    :param data_dir: CIFAR-100 root directory.
    :param index: Index into the test split.
    :param image_size: Input resolution.
    :return: A ``[1, 3, H, W]`` tensor and the image's class label.
    """
    import torchvision

    _, eval_transform = get_cifar100_transforms(image_size)
    dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=None)
    image, label = dataset[index]
    return eval_transform(image).unsqueeze(0), int(label)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Undoes the CIFAR-100 normalisation for display.

    :param tensor: A ``[1, 3, H, W]`` normalised image tensor.
    :return: An ``[H, W, 3]`` array clipped to ``[0, 1]``.
    """
    mean = torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)
    image = (tensor.detach().cpu() * std + mean).clamp(0.0, 1.0)
    return image.squeeze(0).permute(1, 2, 0).numpy()


def extract_cls_attention(
    model: RegisterVisionTransformer,
    image: torch.Tensor,
    k_registers: int,
    layers: Sequence[int],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """
    Extracts the ``[CLS]`` attention over spatial patches at the requested depths.

    :param model: A loaded evaluation-mode model.
    :param image: A ``[1, 3, H, W]`` input tensor.
    :param k_registers: The model's register count.
    :param layers: One-based transformer block indices.
    :param device: Device to run the forward pass on.
    :return: Mapping ``{layer: [grid, grid] attention map}``.
    :raises RuntimeError: If a requested block produced no attention matrix, or
        if the spatial slice is not a square patch grid.
    """
    with ViTAttentionHookManager(model) as hooks:
        with torch.no_grad():
            model(image.to(device))
        captured = dict(hooks.attention_maps)

    maps: Dict[int, np.ndarray] = {}
    start = 1 + k_registers
    for layer in layers:
        index = layer - 1
        if index not in captured:
            raise RuntimeError(f"No attention captured for block {layer}.")

        attention = captured[index]
        spatial = attention[0, :, 0, start:].mean(dim=0)

        grid = int(math.isqrt(spatial.numel()))
        if grid * grid != spatial.numel():
            raise RuntimeError(
                f"Block {layer}: spatial slice has {spatial.numel()} entries, which is not a "
                f"square patch grid. Check that K={k_registers} matches the checkpoint."
            )

        upsampled = F.interpolate(
            spatial.reshape(1, 1, grid, grid).float(),
            size=(image.shape[-2], image.shape[-1]),
            mode="bicubic",
            align_corners=False,
        )
        maps[layer] = upsampled.squeeze().numpy()
    return maps


def render_comparison(
    background: np.ndarray,
    rows: Sequence[Tuple[str, Dict[int, np.ndarray]]],
    layers: Sequence[int],
    pdf_path: str,
    alpha: float = 0.6,
) -> str:
    """
    Composes the comparative overlay grid under one shared colour scale.

    :param background: The ``[H, W, 3]`` source image.
    :param rows: ``(row label, {layer: attention map})`` pairs, in draw order.
    :param layers: The block indices forming the columns.
    :param pdf_path: Destination PDF path.
    :param alpha: Overlay opacity.
    :return: The path written.
    """
    all_values = np.concatenate([maps[layer].ravel() for _, maps in rows for layer in layers])
    vmin, vmax = float(all_values.min()), float(all_values.max())

    figure, axes = plt.subplots(
        len(rows), len(layers), figsize=(2.1 * len(layers), 2.25 * len(rows)), squeeze=False
    )

    image = None
    for row_index, (label, maps) in enumerate(rows):
        for column, layer in enumerate(layers):
            axis = axes[row_index][column]
            axis.imshow(background)
            image = axis.imshow(maps[layer], cmap="inferno", alpha=alpha, vmin=vmin, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"Block {layer}", fontsize=10)
            if column == 0:
                axis.set_ylabel(label, fontsize=10)

    figure.suptitle("[CLS] attention over image patches", fontsize=11)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    colourbar = figure.colorbar(image, ax=axes, orientation="horizontal", fraction=0.04, pad=0.05)
    colourbar.set_label("Attention weight (shared scale)", fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return pdf_path


def parse_args() -> argparse.Namespace:
    """Parses command line options."""
    parser = argparse.ArgumentParser(description="Render the comparative [CLS] attention figure")
    parser.add_argument("--baseline", type=str, default="checkpoints/exp01_k0_s42/best_model.pth",
                        help="Checkpoint of the K=0 control run")
    parser.add_argument("--registers", type=str, default="checkpoints/exp07_k4_s42/best_model.pth",
                        help="Checkpoint of the register run to compare against")
    parser.add_argument("--baseline_k", type=int, default=None,
                        help="Overrides the register count read from the control checkpoint")
    parser.add_argument("--registers_k", type=int, default=None,
                        help="Overrides the register count read from the register checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data", help="CIFAR-100 root directory")
    parser.add_argument("--image_index", type=int, default=0,
                        help="Index into the CIFAR-100 test split; the same image is used for both rows")
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS),
                        help="One-based transformer blocks to render as columns")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the forward passes on")
    parser.add_argument("--output", type=str, default="paper/figures/attention_maps_comparison.pdf",
                        help="Destination PDF path")
    return parser.parse_args()


def main() -> int:
    """
    Command line entrypoint.

    :return: Process exit status; ``1`` if a checkpoint is missing or a slice
        fails its consistency check.
    """
    args = parse_args()
    device = torch.device(args.device)

    try:
        image, label = load_test_image(args.data_dir, args.image_index)
        rows: List[Tuple[str, Dict[int, np.ndarray]]] = []
        for path, override, template in (
            (args.baseline, args.baseline_k, "Control ($K={k}$)"),
            (args.registers, args.registers_k, "Registers ($K={k}$)"),
        ):
            model, k_registers = load_checkpoint_model(path, device, override)
            maps = extract_cls_attention(model, image, k_registers, args.layers, device)
            rows.append((template.format(k=k_registers), maps))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    written = render_comparison(denormalize(image), rows, args.layers, args.output)
    print(f"wrote {written} (test image {args.image_index}, class {label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
