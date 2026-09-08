"""
Quantitative Evidence Figures
=============================
Renders the four figures the manuscript uses to argue its hypotheses. Every
number is read from ``outputs/sweep_summary.json`` and the per-run
``train_history.csv`` files; nothing is entered by hand, so re-running the sweep
regenerates the figures without editing this script.

Figures
-------
``entropy_vs_layer.pdf``
    Layer-resolved Shannon attention entropy with :math:`\\pm 1\\sigma` bands
    across seeds, one curve per register arm, beside the paired per-seed
    difference against the control arm. The absolute curves nearly coincide, so
    the difference panel is what carries the evidence for H3.

``gen_gap_vs_registers.pdf``
    Generalization gap :math:`\\Delta\\mathcal{L}` per arm with seed dispersion,
    paired with the held-out test loss so the reader sees immediately whether a
    gap reduction transfers off the validation split. The dotted line marks the
    control arm. Evidence for H1.

``loss_curves.pdf``
    Seed-averaged training and validation loss trajectories per arm.

``accuracy_vs_registers.pdf``
    Validation and test Top-1 accuracy against :math:`K`. The two panels use a
    shared accuracy range so the difference in seed dispersion between the
    1,000-image validation split and the 10,000-image test split is visible
    rather than hidden by independent autoscaling. Evidence for H2.

Public Interface
----------------
- :func:`plot_entropy_vs_layer`
- :func:`plot_gen_gap`
- :func:`plot_loss_curves`
- :func:`plot_accuracy`
- CLI: ``python scripts/plot_metrics.py``
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REGISTER_ARMS: Sequence[int] = (0, 1, 4, 8)
NUM_LAYERS = 12

RUN_DIR_PATTERN = re.compile(r"^exp(?P<idx>\d+)_k(?P<k>\d+)_s(?P<seed>\d+)$")

ARM_COLOURS: Dict[int, str] = {0: "#d1495b", 1: "#3d6f9e", 4: "#2a9d8f", 8: "#e07a3f"}
ARM_MARKERS: Dict[int, str] = {0: "o", 1: "s", 4: "^", 8: "D"}

# Matplotlib stamps a creation timestamp into every PDF, which makes two runs
# over identical data produce different bytes. Suppressing it lets the committed
# figures be diffed against a fresh render as a reproducibility check.
PDF_METADATA: Dict[str, Any] = {"CreationDate": None}


def _arm_label(k: int) -> str:
    """Returns the legend label for a register arm."""
    return "Baseline ($K=0$)" if k == 0 else f"$K={k}$"


def _apply_style() -> None:
    """Sets the shared figure style so every export looks the same."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def load_summary(summary_path: str) -> Dict[str, Any]:
    """
    Loads the aggregated sweep summary.

    :param summary_path: Path to ``sweep_summary.json``.
    :return: The parsed summary.
    :raises FileNotFoundError: If the summary has not been generated yet.
    """
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"{summary_path} not found. Run: python src/utils/logger.py --output_dir outputs/"
        )
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_histories(outputs_dir: str) -> Dict[int, List[Dict[str, List[float]]]]:
    """
    Reads the per-epoch loss trajectories of every completed run.

    :param outputs_dir: Directory holding the ``expXX_kY_sZ`` run folders.
    :return: Mapping ``{K: [{"train_loss": [...], "val_loss": [...]}, ...]}``.
    """
    histories: Dict[int, List[Dict[str, List[float]]]] = {k: [] for k in REGISTER_ARMS}
    for entry in sorted(os.listdir(outputs_dir)):
        match = RUN_DIR_PATTERN.match(entry)
        csv_path = os.path.join(outputs_dir, entry, "train_history.csv")
        if match is None or not os.path.isfile(csv_path):
            continue

        k = int(match.group("k"))
        if k not in histories:
            continue

        train: List[float] = []
        val: List[float] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                train.append(float(row["train_loss"]))
                val.append(float(row["val_loss"]))
        if train:
            histories[k].append({"train_loss": train, "val_loss": val})
    return histories


def load_layerwise_runs(outputs_dir: str) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Reads the per-run layer-wise attention entropy vectors.

    The paired difference panel needs the individual runs rather than the arm
    means, because a difference formed after averaging carries no dispersion
    estimate.

    :param outputs_dir: Directory holding the ``expXX_kY_sZ`` run folders.
    :return: Mapping ``{K: {seed: entropy vector of length NUM_LAYERS}}``.
    """
    per_run: Dict[int, Dict[int, np.ndarray]] = {}
    for entry in sorted(os.listdir(outputs_dir)):
        match = RUN_DIR_PATTERN.match(entry)
        metrics_path = os.path.join(outputs_dir, entry, "metrics.json")
        if match is None or not os.path.isfile(metrics_path):
            continue

        with open(metrics_path, "r", encoding="utf-8") as handle:
            layers = json.load(handle).get("test_results", {}).get("layerwise_entropy", {})
        if len(layers) != NUM_LAYERS:
            continue

        vector = np.array([float(layers[str(layer)]) for layer in range(NUM_LAYERS)])
        per_run.setdefault(int(match.group("k")), {})[int(match.group("seed"))] = vector
    return per_run


def _layerwise(summary: Dict[str, Any], k: int, field: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a layer-resolved aggregate as mean and standard-deviation vectors.

    :param summary: Parsed sweep summary.
    :param k: Register arm.
    :param field: Either ``layerwise_entropy`` or ``layerwise_outliers``.
    :return: Two arrays of length :data:`NUM_LAYERS`.
    """
    block = summary[f"k_{k}"][field]
    means = np.array([block[str(layer)]["mean"] for layer in range(NUM_LAYERS)])
    stds = np.array([block[str(layer)]["std"] for layer in range(NUM_LAYERS)])
    return means, stds


def plot_entropy_vs_layer(
    summary: Dict[str, Any],
    per_run: Dict[int, Dict[int, np.ndarray]],
    pdf_path: str,
) -> str:
    """
    Draws layer-resolved attention entropy and the paired difference from the
    control arm.

    :param summary: Parsed sweep summary.
    :param per_run: Per-run entropy vectors from :func:`load_layerwise_runs`.
    :param pdf_path: Destination PDF path.
    :return: The path written.
    """
    layers = np.arange(1, NUM_LAYERS + 1)
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))

    for k in REGISTER_ARMS:
        means, stds = _layerwise(summary, k, "layerwise_entropy")
        axes[0].plot(
            layers, means, marker=ARM_MARKERS[k], markersize=4, linewidth=1.6,
            color=ARM_COLOURS[k], label=_arm_label(k),
        )
        axes[0].fill_between(layers, means - stds, means + stds,
                             color=ARM_COLOURS[k], alpha=0.18, linewidth=0)

    axes[0].set_xlabel("Transformer block $l$")
    axes[0].set_ylabel(r"Attention entropy $\bar{H}^{(l)}$ (bits)")
    axes[0].set_title("Absolute entropy", fontsize=10)
    axes[0].set_xticks(layers[::2])

    control = per_run.get(0, {})
    for k in REGISTER_ARMS[1:]:
        shared = sorted(seed for seed in per_run.get(k, {}) if seed in control)
        if not shared:
            continue
        deltas = np.array([per_run[k][seed] - control[seed] for seed in shared])
        mean = deltas.mean(axis=0)
        std = deltas.std(axis=0)
        axes[1].plot(layers, mean, marker=ARM_MARKERS[k], markersize=4, linewidth=1.6,
                     color=ARM_COLOURS[k], label=_arm_label(k))
        axes[1].fill_between(layers, mean - std, mean + std,
                             color=ARM_COLOURS[k], alpha=0.18, linewidth=0)

    axes[1].axhline(0.0, color=ARM_COLOURS[0], linewidth=1.2, linestyle=":")
    axes[1].set_xlabel("Transformer block $l$")
    axes[1].set_ylabel(r"$\bar{H}^{(l)}_{K} - \bar{H}^{(l)}_{K=0}$ (bits)")
    axes[1].set_title("Paired difference from the control arm", fontsize=10)
    axes[1].set_xticks(layers[::2])

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    figure.tight_layout()
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)
    return pdf_path


def plot_gen_gap(summary: Dict[str, Any], pdf_path: str) -> str:
    """
    Draws the generalization gap beside the held-out test loss.

    The two panels share nothing but the x-axis on purpose: the left panel is
    the quantity H1 predicts should fall, the right panel is the quantity that
    decides whether that fall means anything outside the validation split.

    :param summary: Parsed sweep summary.
    :param pdf_path: Destination PDF path.
    :return: The path written.
    """
    positions = np.arange(len(REGISTER_ARMS))

    gap_mean = np.array([summary[f"k_{k}"]["gen_gap_mean"] for k in REGISTER_ARMS])
    gap_std = np.array([summary[f"k_{k}"]["gen_gap_std"] for k in REGISTER_ARMS])
    test_mean = np.array([summary[f"k_{k}"]["test_loss_mean"] for k in REGISTER_ARMS])
    test_std = np.array([summary[f"k_{k}"]["test_loss_std"] for k in REGISTER_ARMS])

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))

    for axis, mean, std, title, ylabel, colour in (
        (axes[0], gap_mean, gap_std, "Generalization gap (validation)",
         r"$\Delta\mathcal{L}=\mathcal{L}_{\mathrm{val}}-\mathcal{L}_{\mathrm{train}}$", "#3d6f9e"),
        (axes[1], test_mean, test_std, "Held-out test loss", r"$\mathcal{L}_{\mathrm{test}}$", "#2a9d8f"),
    ):
        axis.errorbar(positions, mean, yerr=std, marker="o", markersize=6, linewidth=1.6,
                      capsize=5, color=colour)
        axis.axhline(mean[0], color="#d1495b", linewidth=1.0, linestyle=":", zorder=0)
        axis.set_xticks(positions)
        axis.set_xticklabels([str(k) for k in REGISTER_ARMS])
        axis.set_xlim(-0.4, len(REGISTER_ARMS) - 0.6)
        axis.set_xlabel("Register tokens $K$")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10)

    figure.tight_layout()
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)
    return pdf_path


def plot_loss_curves(histories: Dict[int, List[Dict[str, List[float]]]], pdf_path: str) -> str:
    """
    Draws seed-averaged training and validation loss trajectories.

    :param histories: Per-arm trajectories from :func:`load_histories`.
    :param pdf_path: Destination PDF path.
    :return: The path written.
    """
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True)

    for split, axis, title in (("train_loss", axes[0], "Training loss"),
                               ("val_loss", axes[1], "Validation loss")):
        for k in REGISTER_ARMS:
            runs = histories.get(k, [])
            if not runs:
                continue
            length = min(len(run[split]) for run in runs)
            stacked = np.array([run[split][:length] for run in runs])
            epochs = np.arange(1, length + 1)
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            axis.plot(epochs, mean, linewidth=1.5, color=ARM_COLOURS[k], label=_arm_label(k))
            axis.fill_between(epochs, mean - std, mean + std, color=ARM_COLOURS[k], alpha=0.15, linewidth=0)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss")
        axis.set_title(title, fontsize=10)

    axes[0].legend(ncol=2)
    figure.tight_layout()
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)
    return pdf_path


def plot_accuracy(summary: Dict[str, Any], pdf_path: str) -> str:
    """
    Draws validation and test Top-1 accuracy against the register count.

    :param summary: Parsed sweep summary.
    :param pdf_path: Destination PDF path.
    :return: The path written.
    """
    arms = np.array(REGISTER_ARMS, dtype=float)
    val_mean = np.array([summary[f"k_{k}"]["val_top1_mean"] for k in REGISTER_ARMS])
    val_std = np.array([summary[f"k_{k}"]["val_top1_std"] for k in REGISTER_ARMS])
    test_mean = np.array([summary[f"k_{k}"]["test_top1_mean"] for k in REGISTER_ARMS])
    test_std = np.array([summary[f"k_{k}"]["test_top1_std"] for k in REGISTER_ARMS])

    low = float(min((val_mean - val_std).min(), (test_mean - test_std).min())) - 0.6
    high = float(max((val_mean + val_std).max(), (test_mean + test_std).max())) + 0.6

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)

    for axis, mean, std, title, colour in (
        (axes[0], val_mean, val_std, "Validation split (1,000 images)", "#3d6f9e"),
        (axes[1], test_mean, test_std, "Test split (10,000 images)", "#2a9d8f"),
    ):
        axis.errorbar(
            np.arange(len(arms)), mean, yerr=std, marker="o", markersize=6,
            linewidth=1.6, capsize=5, color=colour,
        )
        axis.set_xticks(np.arange(len(arms)))
        axis.set_xticklabels([str(int(k)) for k in arms])
        axis.set_xlabel("Register tokens $K$")
        axis.set_title(title, fontsize=10)
        axis.set_ylim(low, high)

    axes[0].set_ylabel("Top-1 accuracy (\\%)")
    figure.tight_layout()
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)
    return pdf_path


def parse_args() -> argparse.Namespace:
    """Parses command line options."""
    parser = argparse.ArgumentParser(description="Render the manuscript evidence figures")
    parser.add_argument("--summary", type=str, default="outputs/sweep_summary.json",
                        help="Aggregated sweep summary produced by src/utils/logger.py")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Directory holding the per-run expXX_kY_sZ folders")
    parser.add_argument("--figures_dir", type=str, default="paper/figures",
                        help="Directory the vector PDFs are written into")
    return parser.parse_args()


def main() -> int:
    """
    Command line entrypoint.

    :return: Process exit status; ``1`` if the summary or the run directories
        are missing.
    """
    args = parse_args()
    _apply_style()

    try:
        summary = load_summary(args.summary)
    except FileNotFoundError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    if not os.path.isdir(args.output_dir):
        print(f"[error] outputs directory does not exist: {args.output_dir}", file=sys.stderr)
        return 1

    os.makedirs(args.figures_dir, exist_ok=True)
    histories = load_histories(args.output_dir)
    per_run = load_layerwise_runs(args.output_dir)

    written = [
        plot_entropy_vs_layer(summary, per_run, os.path.join(args.figures_dir, "entropy_vs_layer.pdf")),
        plot_gen_gap(summary, os.path.join(args.figures_dir, "gen_gap_vs_registers.pdf")),
        plot_loss_curves(histories, os.path.join(args.figures_dir, "loss_curves.pdf")),
        plot_accuracy(summary, os.path.join(args.figures_dir, "accuracy_vs_registers.pdf")),
    ]
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
