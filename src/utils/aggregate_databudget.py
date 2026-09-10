#!/usr/bin/env python3
"""
300 Images/Class Data Budget Multi-Seed Aggregator
==================================================
File Purpose:
    Aggregates experimental outputs from the 300 images/class extension sweep
    (K in {0, 1, 4, 8} across random seeds {42, 1337, 3407}). Parses individual
    run directories matching `expNN_kK_pc300_sSEED`, extracts test/val/generalization
    metrics, computes multi-seed statistical reductions (Mean +/- Std), and outputs
    a consolidated JSON summary.

Architecture & System Context:
    Belongs to the results and analysis layer (`src/utils/`). Kept deliberately separate
    from `src/utils/logger.py` to preserve the immutability of the baseline 100/class
    12-run artifacts (`outputs/sweep_summary.json`).

Public Functions & CLI:
    - `parse_run(run_dir: str) -> Optional[Dict[str, Any]]`: Parses a single run directory.
    - `aggregate_sweep(output_dir: str) -> Dict[str, Any]`: Aggregates all matching runs.
    - CLI entrypoint via `main()`.

Dependencies:
    - Standard library: `argparse`, `json`, `os`, `re`, `collections`, `typing`
    - Third-party: `numpy`

Maintenance Invariants:
    - Strictly additive: only operates on the designated `--output_dir` (default: `outputs_databudget`).
    - Handles missing, incomplete, or crashed runs gracefully without throwing unhandled exceptions.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

# Canonical directory regex for 300/class extension runs
RUN_DIR_PATTERN = re.compile(r"^exp(?P<idx>\d+)_k(?P<k>\d+)_pc(?P<pc>\d+)_s(?P<seed>\d+)$")


def parse_run(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Parses a single experiment directory and extracts core evaluation metrics.

    :param run_dir: Absolute or relative path to an experiment output directory.
    :return: Dictionary of extracted scalar metrics, or None if metrics.json is missing.
    """
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return None

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read {metrics_path}: {e}")
        return None

    test = data.get("test_results", {})
    best_epoch = data.get("best_epoch")
    history = data.get("history", [])

    best_record = next((r for r in history if r.get("epoch") == best_epoch), {})
    final_record = history[-1] if history else {}

    val_loss = best_record.get("val_loss")
    train_loss = final_record.get("train_loss")
    gen_gap = (val_loss - train_loss) if (val_loss is not None and train_loss is not None) else None

    return {
        "test_top1": test.get("test_top1"),
        "test_loss": test.get("test_loss"),
        "test_top5": test.get("test_top5"),
        "val_loss": val_loss,
        "train_loss": train_loss,
        "gen_gap": gen_gap,
    }


def aggregate_sweep(output_dir: str = "outputs_databudget") -> Dict[str, Any]:
    """
    Scans output_dir, groups completed runs by register count K, and computes
    mean and standard deviation across seeds.

    :param output_dir: Root directory containing individual run directories.
    :return: Aggregated summary dictionary mapping arm keys (e.g. 'k0_pc300') to statistics.
    """
    if not os.path.isdir(output_dir):
        return {}

    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for name in sorted(os.listdir(output_dir)):
        match = RUN_DIR_PATTERN.match(name)
        if not match:
            continue
        run_dir = os.path.join(output_dir, name)
        record = parse_run(run_dir)
        if record is None:
            continue
        k = int(match.group("k"))
        groups[k].append(record)

    summary: Dict[str, Any] = {}
    for k, records in sorted(groups.items()):
        entry: Dict[str, Any] = {
            "num_registers": k,
            "samples_per_class": 300,
            "n_seeds": len(records),
            "records": records,
        }
        for metric in ("test_top1", "test_loss", "test_top5", "val_loss", "train_loss", "gen_gap"):
            values = [r[metric] for r in records if r.get(metric) is not None]
            if not values:
                continue
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
        summary[f"k{k}_pc300"] = entry

    return summary


def main():
    """CLI execution harness."""
    parser = argparse.ArgumentParser(
        description="Aggregate 300 images/class multi-seed ablation sweep results."
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_databudget",
        help="Path to output directory containing expNN_kK_pc300_sSEED folders.",
    )
    args = parser.parse_args()

    summary = aggregate_sweep(args.output_dir)
    if not summary:
        print(f"No completed runs found in {args.output_dir}.")
        return

    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Successfully generated aggregated summary at: {out_path}")
    print("=" * 60)
    print(f"{'Arm':<10}{'Seeds':<7}{'Test Top-1 (%)':<20}{'Gen Gap':<12}{'Val Loss':<12}")
    print("-" * 60)
    for key, e in summary.items():
        top1 = e.get("test_top1_mean")
        top1_sd = e.get("test_top1_std")
        gap = e.get("gen_gap_mean")
        val_loss = e.get("val_loss_mean")
        top1_str = f"{top1:.2f} ± {top1_sd:.2f}" if top1 is not None and top1_sd is not None else "—"
        gap_str = f"{gap:.4f}" if gap is not None else "—"
        val_str = f"{val_loss:.4f}" if val_loss is not None else "—"
        print(f"K={e['num_registers']:<8}{e['n_seeds']:<7}{top1_str:<20}{gap_str:<12}{val_str:<12}")
    print("=" * 60)


if __name__ == "__main__":
    main()
