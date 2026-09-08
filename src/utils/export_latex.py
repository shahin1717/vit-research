"""
Automated LaTeX Results Table Generator
========================================
Converts the aggregated sweep summary into the LaTeX tables used by the
manuscript, so that no number in the paper is ever transcribed by hand.

Two tables are produced:

``results_table.tex``
    The main per-arm results, each reported as ``mean +/- std`` across the three
    seeds. The held-out test split is reported first because it is the only
    split no decision was taken on: the checkpoint selected for evaluation is
    the one that minimised loss on the validation split, so a validation number
    is a selected extremum and a test number is not. Validation loss and the
    generalization gap follow, so the reader can see the two together.

``paired_analysis.tex``
    The paired per-seed comparison against the K = 0 control. With only three
    seeds an unpaired comparison of arm means is dominated by seed variance, so
    the effect of registers is measured within each seed and then reduced. This
    table reports the mean paired difference, its standard deviation, how many
    of the three seeds moved in the hypothesised direction, and a paired
    t-statistic with its two-sided p-value. Test and validation quantities are
    both listed, because the contrast between them is the study's main result.

Both tables read only ``outputs/sweep_summary.json`` and the per-run
``metrics.json`` files; nothing is hard-coded.

Public Interface
----------------
- :func:`export_results_table`: Writes the main results table.
- :func:`export_paired_table`: Writes the paired comparison table.
- CLI: ``python src/utils/export_latex.py``
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REGISTER_ARMS: Sequence[int] = (0, 1, 4, 8)
SEEDS: Sequence[int] = (42, 1337, 3407)
FINAL_BLOCK = 11

RUN_DIR_PATTERN = re.compile(r"^exp(?P<idx>\d+)_k(?P<k>\d+)_s(?P<seed>\d+)$")


def _t_two_sided_p(t_stat: float, df: int) -> float:
    """
    Two-sided p-value of a t-statistic.

    Uses SciPy when available and otherwise falls back to the exact closed form
    for the small degrees of freedom this study produces (df = 2 with three
    seeds), so the table can be regenerated in a bare environment.

    :param t_stat: The t-statistic.
    :param df: Degrees of freedom.
    :return: Two-sided p-value.
    """
    if df <= 0:
        return float("nan")
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_stat), df))
    except ImportError:
        if df == 2:
            # Exact survival function of Student's t with 2 degrees of freedom.
            x = abs(t_stat)
            return float(1.0 - x / np.sqrt(2.0 + x * x))
        raise


def load_runs(outputs_dir: str) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Reads the per-seed quantities the paired analysis needs from every run.

    :param outputs_dir: Directory holding the ``expXX_kY_sZ`` run folders.
    :return: Nested mapping ``{K: {seed: {metric: value}}}``.
    :raises FileNotFoundError: If ``outputs_dir`` does not exist.
    """
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_dir}")

    runs: Dict[int, Dict[int, Dict[str, float]]] = {}
    for entry in sorted(os.listdir(outputs_dir)):
        match = RUN_DIR_PATTERN.match(entry)
        metrics_path = os.path.join(outputs_dir, entry, "metrics.json")
        if match is None or not os.path.isfile(metrics_path):
            continue

        with open(metrics_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        history = data.get("history", [])
        best = next((h for h in history if h.get("epoch") == data.get("best_epoch")), None)
        test = data.get("test_results", {})
        entropy = {int(layer): float(value) for layer, value in test.get("layerwise_entropy", {}).items()}

        if best is None or not history or not entropy:
            continue

        runs.setdefault(int(match.group("k")), {})[int(match.group("seed"))] = {
            "test_top1": float(test["test_top1"]),
            "test_top5": float(test["test_top5"]),
            "test_loss": float(test["test_loss"]),
            "val_top1": float(best["val_top1"]),
            "val_loss": float(best["val_loss"]),
            "train_loss": float(history[-1]["train_loss"]),
            "gen_gap": float(best["val_loss"]) - float(history[-1]["train_loss"]),
            "entropy_final": entropy[FINAL_BLOCK],
        }
    return runs


def _fmt(mean: float, std: float, digits: int) -> str:
    """Formats a mean and standard deviation as a LaTeX ``a $\\pm$ b`` cell."""
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _arm_label(k: int) -> str:
    """Returns the manuscript label for a register arm."""
    return "Baseline ($K=0$)" if k == 0 else f"Registers ($K={k}$)"


def _assert_sweep_is_complete(summary: Dict[str, Any], runs: Dict[int, Dict[int, Dict[str, float]]]) -> None:
    """
    Refuses to emit a table when the sweep behind it is incomplete.

    Without this guard an empty ``outputs/`` produces a zero-filled summary,
    every statistic reduces to ``0.0``, and the comparison against that summary
    passes because both sides are zero. The table would then be written full of
    zeros and NaNs and would silently replace a correct one.

    :param summary: Contents of ``sweep_summary.json``.
    :param runs: Per-seed records from :func:`load_runs`.
    :raises ValueError: If any arm has no completed seed.
    """
    empty = [k for k in REGISTER_ARMS if not runs.get(k)]
    if empty:
        arms = ", ".join(f"K={k}" for k in empty)
        raise ValueError(
            f"No completed runs found for {arms}. Populate outputs/ with the run "
            f"directories and re-run src/utils/logger.py before generating tables."
        )

    missing = summary.get("meta", {}).get("missing_runs", [])
    if missing:
        raise ValueError(
            "The sweep summary reports missing runs: " + ", ".join(missing)
        )


def _assert_matches_summary(
    summary: Dict[str, Any],
    stats: Dict[int, Dict[str, Tuple[float, float]]],
    tolerance: float = 1e-6,
) -> None:
    """
    Cross-checks the table's own reduction against the aggregation engine.

    This table recomputes its statistics from the per-run records rather than
    copying ``sweep_summary.json``, which means the two could silently drift
    apart and the manuscript would then disagree with the artifact it claims to
    be generated from. Every quantity that exists in both places is compared
    here, and a disagreement is raised rather than printed.

    :param summary: Contents of ``sweep_summary.json``.
    :param stats: Per-arm ``{field: (mean, std)}`` computed from the run records.
    :param tolerance: Absolute tolerance for the comparison.
    :raises ValueError: If any shared quantity disagrees.
    """
    shared = {
        "test_top1": ("test_top1_mean", "test_top1_std"),
        "test_loss": ("test_loss_mean", "test_loss_std"),
        "val_loss": ("val_loss_mean", "val_loss_std"),
        "gen_gap": ("gen_gap_mean", "gen_gap_std"),
    }

    mismatches: List[str] = []
    for k, fields in stats.items():
        arm = summary.get(f"k_{k}", {})
        for field, (mean_key, std_key) in shared.items():
            if field not in fields or mean_key not in arm:
                continue
            for value, key in zip(fields[field], (mean_key, std_key)):
                if abs(value - float(arm[key])) > tolerance:
                    mismatches.append(f"k_{k}.{key}: table {value!r} vs summary {arm[key]!r}")

    if mismatches:
        raise ValueError(
            "Table statistics disagree with outputs/sweep_summary.json:\n  "
            + "\n  ".join(mismatches)
        )


def export_results_table(
    summary: Dict[str, Any],
    runs: Dict[int, Dict[int, Dict[str, float]]],
    tex_path: str,
) -> str:
    """
    Writes the main per-arm results table.

    The best value in each column is bolded. Because the columns disagree —
    the control arm leads on the held-out split while a register arm leads on
    the validation-derived measures — the bolding is what makes the study's
    central tension visible in a single glance.

    :param summary: Contents of ``sweep_summary.json``.
    :param runs: Per-seed records from :func:`load_runs`.
    :param tex_path: Destination ``.tex`` path.
    :return: The LaTeX source that was written.
    """
    columns = (
        ("test_top1", r"\textbf{Test Top-1 (\%)}", 2, max),
        ("test_loss", r"\textbf{Test Loss}", 4, min),
        ("val_loss", r"\textbf{Val Loss}", 4, min),
        ("gen_gap", r"\textbf{Gap $\Delta\mathcal{L}$}", 4, min),
        ("entropy_final", r"\textbf{$\bar{H}^{(12)}$ (bits)}", 3, max),
    )

    stats: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for k in REGISTER_ARMS:
        seeds = [runs[k][s] for s in SEEDS if s in runs.get(k, {})]
        stats[k] = {
            field: (
                float(np.array([r[field] for r in seeds]).mean()),
                float(np.array([r[field] for r in seeds]).std()),
            )
            for field, _, _, _ in columns
        }

    _assert_sweep_is_complete(summary, runs)
    _assert_matches_summary(summary, stats)

    best = {
        field: chooser(stats[k][field][0] for k in REGISTER_ARMS)
        for field, _, _, chooser in columns
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Register ablation on low-data CIFAR-100 (100 images/class, ViT-Tiny, "
        r"50 epochs). Mean $\pm$ population standard deviation over the three seeds "
        r"\texttt{42}, \texttt{1337} and \texttt{3407}; best value per column in bold. "
        r"Test metrics come from a single pass over the untouched 10{,}000-image test split. "
        r"Validation loss is the value at the selected epoch and is therefore a selected "
        r"minimum, not an unbiased estimate. The gap "
        r"$\Delta\mathcal{L}=\mathcal{L}_{\mathrm{val}}-\mathcal{L}_{\mathrm{train}}$ is "
        r"formed within each seed before averaging. Losses are label-smoothed "
        r"($\varepsilon=0.1$) cross-entropy, so they carry a constant positive floor and are "
        r"comparable across arms but not against unsmoothed values.}",
        r"\label{tab:main_results}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"\toprule",
        r"\textbf{Configuration} & " + " & ".join(header for _, header, _, _ in columns) + r" \\",
        r"\midrule",
    ]

    for k in REGISTER_ARMS:
        cells = [_arm_label(k)]
        for field, _, digits, _ in columns:
            mean, std = stats[k][field]
            cell = _fmt(mean, std, digits)
            if mean == best[field]:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    tex = "\n".join(lines)

    os.makedirs(os.path.dirname(os.path.abspath(tex_path)), exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    return tex


def export_paired_table(runs: Dict[int, Dict[int, Dict[str, float]]], tex_path: str) -> str:
    """
    Writes the paired per-seed comparison against the control arm.

    For every register arm and every metric the difference is taken within a
    seed, which removes the seed-to-seed variance that otherwise swamps the
    treatment effect at this sample size.

    :param runs: Per-seed records from :func:`load_runs`.
    :param tex_path: Destination ``.tex`` path.
    :return: The LaTeX source that was written.
    """
    metrics = (
        ("test_top1", r"Test Top-1 (pp)", "increase", 2),
        ("test_loss", r"Test loss", "decrease", 4),
        ("val_loss", r"Validation loss", "decrease", 4),
        ("gen_gap", r"Gen.\ gap $\Delta\mathcal{L}$", "decrease", 4),
        ("entropy_final", r"$\bar{H}^{(12)}$ (bits)", "increase", 4),
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired per-seed effect of register tokens relative to the $K=0$ control. "
        r"Each difference is taken within a seed and then reduced over the three seeds, so "
        r"seed-to-seed variance cancels. ``Dir.'' counts how many of the three seeds moved in "
        r"the direction the corresponding hypothesis predicts. $t$ is a paired $t$-statistic "
        r"with $\mathrm{df}=2$ and $p$ is its two-sided value; at $n=3$ these are reported as "
        r"effect-size evidence, and no correction for the fifteen comparisons is applied, so "
        r"a single small $p$ should not be read as a confirmatory result.}",
        r"\label{tab:paired_effects}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\footnotesize",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Arm} & \textbf{Mean $\Delta$} & \textbf{SD} & "
        r"\textbf{Dir.} & \textbf{$t$} & \textbf{$p$} \\",
        r"\midrule",
    ]

    for field, label, wanted, digits in metrics:
        for position, k in enumerate((1, 4, 8)):
            deltas = np.array(
                [runs[k][s][field] - runs[0][s][field] for s in SEEDS if s in runs.get(k, {})]
            )
            mean = float(deltas.mean())
            sd = float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0
            hits = int(sum(1 for d in deltas if (d > 0 if wanted == "increase" else d < 0)))
            t_stat = mean / (sd / np.sqrt(deltas.size)) if sd > 0 else float("nan")
            p_value = _t_two_sided_p(t_stat, deltas.size - 1) if sd > 0 else float("nan")

            metric_cell = label if position == 0 else ""
            lines.append(
                f"{metric_cell} & $K={k}$ & {mean:+.{digits}f} & {sd:.{digits}f} & "
                f"{hits}/{deltas.size} & {t_stat:.2f} & {p_value:.3f} \\\\"
            )
        if field != metrics[-1][0]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tex = "\n".join(lines)

    os.makedirs(os.path.dirname(os.path.abspath(tex_path)), exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    return tex


def parse_args() -> argparse.Namespace:
    """Parses command line options."""
    parser = argparse.ArgumentParser(description="Generate the manuscript LaTeX tables")
    parser.add_argument("--summary", type=str, default="outputs/sweep_summary.json",
                        help="Aggregated sweep summary produced by src/utils/logger.py")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Directory holding the per-run expXX_kY_sZ folders")
    parser.add_argument("--tables_dir", type=str, default="paper/tables",
                        help="Directory the .tex tables are written into")
    return parser.parse_args()


def main() -> int:
    """
    Command line entrypoint.

    :return: Process exit status.
    """
    args = parse_args()
    with open(args.summary, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    runs = load_runs(args.output_dir)

    results_path = os.path.join(args.tables_dir, "results_table.tex")
    paired_path = os.path.join(args.tables_dir, "paired_analysis.tex")

    export_results_table(summary, runs, results_path)
    export_paired_table(runs, paired_path)

    print(f"Results table written to {results_path}")
    print(f"Paired analysis table written to {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
