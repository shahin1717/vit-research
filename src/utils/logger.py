"""
Multi-Seed Sweep Metrics Aggregator
====================================
Crawls the per-run experiment directories produced by ``scripts/train.py``,
parses every ``metrics.json``, and reduces the 12-run ablation matrix
(K in {0, 1, 4, 8} x seeds {42, 1337, 3407}) into a single treatment-arm
summary written to ``outputs/sweep_summary.json``.

Statistical Aggregation
-----------------------
For every register arm K, each scalar metric m is collected across the S seeds
that completed and reduced to::

    mean(m) = (1 / S) * sum_{s=1..S} m_s
    std(m)  = sqrt( (1 / S) * sum_{s=1..S} (m_s - mean(m))^2 )

The standard deviation is the population form (``numpy.std`` default,
``ddof=0``), matching the aggregation contract agreed with the analysis lead so
that the reported ``mean +/- std`` in the LaTeX results table is reproducible.

The generalization gap is aggregated per seed before averaging, i.e.

    gen_gap_mean = mean_s( L_val,s - L_train,s )

which is equivalent to the difference of the means but also yields a meaningful
per-arm standard deviation.

Run Directory Contract
----------------------
Directories are matched by the sweep naming convention ``expXX_kY_sZ``::

    outputs/exp07_k4_s42/metrics.json   ->  K = 4, seed = 42

Source Fields
-------------
``scripts/train.py`` serializes a nested record. The aggregator reads flat
top-level keys when present and otherwise derives them from the run history so
that it stays correct regardless of which schema revision produced the file:

===========================  ====================================================
Aggregated quantity          Source of truth
===========================  ====================================================
``best_val_top1``            top-level key
``best_val_loss``            top-level key, else ``history[best_epoch]``
``final_train_loss``         top-level key, else last epoch of ``history``
``mean_layerwise_entropy``   top-level key, else mean over the layer-wise
                             entropy dictionary of the final test evaluation
``mean_layerwise_outliers``  mean over the layer-wise patch-norm outlier rates
===========================  ====================================================

Public Interface
----------------
- :func:`parse_run_directory`: Extracts a flat metric record from one run.
- :func:`collect_runs`: Discovers and parses every run directory.
- :func:`aggregate_sweep_results`: Full aggregation entrypoint, writes JSON.
- CLI: ``python src/utils/logger.py --output_dir outputs/``
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REGISTER_ARMS: Sequence[int] = (0, 1, 4, 8)
SEEDS: Sequence[int] = (42, 1337, 3407)

RUN_DIR_PATTERN = re.compile(r"^exp(?P<idx>\d+)_k(?P<k>\d+)_s(?P<seed>\d+)$")

SUMMARY_FILENAME = "sweep_summary.json"
METRICS_FILENAME = "metrics.json"


def _mean_of_layer_dict(layer_dict: Any) -> Optional[float]:
    """
    Averages a ``{layer_index: value}`` mapping into a single scalar.

    JSON round-tripping turns the integer layer keys emitted by the metric
    functions into strings, so the keys are ignored and only the values are
    reduced.

    :param layer_dict: Mapping of layer index to metric value, or any other type.
    :return: Arithmetic mean over the layers, or ``None`` if unusable/empty.
    """
    if not isinstance(layer_dict, dict) or not layer_dict:
        return None
    values = [float(v) for v in layer_dict.values() if isinstance(v, (int, float))]
    if not values:
        return None
    return float(np.mean(values))


def _layer_vector(layer_dict: Any) -> Optional[Dict[int, float]]:
    """
    Normalizes a ``{layer_index: value}`` mapping to integer keys.

    :param layer_dict: Mapping of layer index (int or str) to metric value.
    :return: Mapping with integer keys, or ``None`` if unusable/empty.
    """
    if not isinstance(layer_dict, dict) or not layer_dict:
        return None
    vector: Dict[int, float] = {}
    for key, value in layer_dict.items():
        if not isinstance(value, (int, float)):
            continue
        try:
            vector[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return vector or None


def _epoch_record(history: Sequence[Dict[str, Any]], epoch: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    Retrieves the history entry for a 1-indexed epoch number.

    :param history: Per-epoch record list exported by the trainer.
    :param epoch: Target epoch number, or ``None``.
    :return: The matching epoch record, or ``None`` if it cannot be located.
    """
    if not history or epoch is None:
        return None
    for record in history:
        if record.get("epoch") == epoch:
            return record
    return None


def parse_run_directory(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Parses a single ``outputs/expXX_kY_sZ/metrics.json`` into a flat record.

    Missing quantities are derived from the epoch history whenever the trainer
    did not serialize them as top-level keys, so the aggregator never silently
    substitutes zeros for values that are recoverable.

    :param run_dir: Path to one experiment output directory.
    :return: Flat metric record, or ``None`` if the run has no readable
             ``metrics.json``.
    """
    metrics_path = os.path.join(run_dir, METRICS_FILENAME)
    if not os.path.isfile(metrics_path):
        return None

    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None

    history: List[Dict[str, Any]] = [
        record for record in data.get("history", []) if isinstance(record, dict)
    ]
    best_epoch = data.get("best_epoch")
    best_record = _epoch_record(history, best_epoch)
    last_record = history[-1] if history else None
    test_results = data.get("test_results") if isinstance(data.get("test_results"), dict) else {}

    # Top-1 accuracy at the best validation epoch.
    best_val_top1 = data.get("best_val_top1")
    if best_val_top1 is None and best_record is not None:
        best_val_top1 = best_record.get("val_top1")

    # Validation loss at the best validation epoch.
    best_val_loss = data.get("best_val_loss")
    if best_val_loss is None and best_record is not None:
        best_val_loss = best_record.get("val_loss")
    if best_val_loss is None and history:
        evaluated = [r.get("val_loss") for r in history if r.get("val_loss")]
        best_val_loss = min(evaluated) if evaluated else None

    # Training loss at the final epoch.
    final_train_loss = data.get("final_train_loss")
    if final_train_loss is None and last_record is not None:
        final_train_loss = last_record.get("train_loss")

    # Layer-wise diagnostics, preferring the final held-out test evaluation.
    entropy_vector = _layer_vector(test_results.get("layerwise_entropy"))
    if entropy_vector is None and best_record is not None:
        entropy_vector = _layer_vector(best_record.get("layerwise_entropy"))

    outlier_vector = _layer_vector(test_results.get("layerwise_outliers"))
    if outlier_vector is None and best_record is not None:
        outlier_vector = _layer_vector(best_record.get("layerwise_outliers"))

    mean_entropy = data.get("mean_layerwise_entropy")
    if mean_entropy is None:
        mean_entropy = _mean_of_layer_dict(entropy_vector)

    mean_outliers = data.get("mean_layerwise_outlier_rate")
    if mean_outliers is None:
        mean_outliers = _mean_of_layer_dict(outlier_vector)

    return {
        "run_dir": os.path.basename(os.path.normpath(run_dir)),
        "experiment": data.get("experiment"),
        "k_registers": data.get("k_registers"),
        "seed": data.get("seed"),
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "mean_layerwise_entropy": mean_entropy,
        "mean_layerwise_outlier_rate": mean_outliers,
        "layerwise_entropy": entropy_vector,
        "layerwise_outliers": outlier_vector,
        "test_top1": test_results.get("test_top1"),
        "test_top5": test_results.get("test_top5"),
        "test_loss": test_results.get("test_loss"),
        "total_training_time_sec": data.get("total_training_time_sec"),
    }


def collect_runs(outputs_dir: str = "outputs/") -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Discovers every ``expXX_kY_sZ`` directory and parses its metrics.

    :param outputs_dir: Root directory holding the per-run output folders.
    :return: Nested mapping ``{K: {seed: run_record}}``.
    :raises FileNotFoundError: If ``outputs_dir`` does not exist.
    """
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_dir}")

    runs: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for entry in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        match = RUN_DIR_PATTERN.match(entry)
        if match is None:
            continue

        record = parse_run_directory(run_dir)
        if record is None:
            continue

        k_value = int(match.group("k"))
        seed_value = int(match.group("seed"))
        # The directory name is authoritative: it is what the sweep runner
        # guarantees, whereas the JSON body reflects whatever CLI overrides the
        # run received.
        record["k_registers"] = k_value
        record["seed"] = seed_value
        runs.setdefault(k_value, {})[seed_value] = record

    return runs


def _reduce(values: Sequence[float]) -> Dict[str, float]:
    """
    Reduces a per-seed sample to mean and population standard deviation.

    :param values: Metric values, one per completed seed.
    :return: Dictionary with ``mean``, ``std`` and ``n`` entries.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std()), "n": int(array.size)}


def _aggregate_layerwise(records: Sequence[Dict[str, Any]], field: str) -> Dict[str, Dict[str, float]]:
    """
    Aggregates a per-layer diagnostic across seeds, layer by layer.

    Feeds the layer-index versus entropy figure, whose +/- 1 sigma error band
    requires per-layer dispersion rather than a single scalar per arm.

    :param records: Per-seed run records for one register arm.
    :param field: Record field holding the ``{layer: value}`` mapping.
    :return: Mapping of layer index (as string) to ``mean``/``std``/``n``.
    """
    per_layer: Dict[int, List[float]] = {}
    for record in records:
        vector = record.get(field)
        if not isinstance(vector, dict):
            continue
        for layer_idx, value in vector.items():
            per_layer.setdefault(int(layer_idx), []).append(float(value))

    return {str(layer): _reduce(values) for layer, values in sorted(per_layer.items())}


def aggregate_sweep_results(
    outputs_dir: str = "outputs/",
    register_arms: Sequence[int] = REGISTER_ARMS,
    seeds: Sequence[int] = SEEDS,
    summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregates the full sweep matrix and writes ``outputs/sweep_summary.json``.

    For every register arm the mean and standard deviation across seeds are
    computed for Top-1 accuracy, validation loss, layer-wise attention entropy,
    patch-norm outlier rate and the generalization gap. Arms with no completed
    run are emitted with zeroed statistics (and ``n = 0``) so that downstream
    table and figure generation never fails on a partially finished sweep.

    :param outputs_dir: Root directory holding the per-run output folders.
    :param register_arms: Register counts K that make up the treatment arms.
    :param seeds: Random seeds replicated within each arm.
    :param summary_path: Destination JSON path. Defaults to
                         ``<outputs_dir>/sweep_summary.json``.
    :return: The summary dictionary that was written to disk.
    :raises FileNotFoundError: If ``outputs_dir`` does not exist.
    """
    runs = collect_runs(outputs_dir)

    summary: Dict[str, Any] = {}
    missing_runs: List[str] = []
    total_found = 0

    for k in register_arms:
        arm_runs = runs.get(k, {})
        records = [arm_runs[seed] for seed in seeds if seed in arm_runs]
        total_found += len(records)
        missing_runs.extend(f"k{k}_s{seed}" for seed in seeds if seed not in arm_runs)

        def column(field: str) -> List[float]:
            """Collects one metric field across the arm's completed seeds."""
            return [
                float(record[field])
                for record in records
                if isinstance(record.get(field), (int, float))
            ]

        accs = column("best_val_top1")
        val_losses = column("best_val_loss")
        train_losses = column("final_train_loss")
        entropies = column("mean_layerwise_entropy")
        outlier_rates = column("mean_layerwise_outlier_rate")
        test_accs = column("test_top1")

        # Pair validation and training loss per seed so the gap keeps its
        # per-seed identity and yields a valid standard deviation.
        gaps = [
            float(record["best_val_loss"]) - float(record["final_train_loss"])
            for record in records
            if isinstance(record.get("best_val_loss"), (int, float))
            and isinstance(record.get("final_train_loss"), (int, float))
        ]

        acc_stats = _reduce(accs)
        val_loss_stats = _reduce(val_losses)
        train_loss_stats = _reduce(train_losses)
        entropy_stats = _reduce(entropies)
        outlier_stats = _reduce(outlier_rates)
        test_acc_stats = _reduce(test_accs)
        gap_stats = _reduce(gaps)

        summary[f"k_{k}"] = {
            "num_registers": k,
            "num_seeds": len(records),
            "seeds": [record["seed"] for record in records],
            "run_dirs": [record["run_dir"] for record in records],
            "val_top1_mean": acc_stats["mean"],
            "val_top1_std": acc_stats["std"],
            "val_loss_mean": val_loss_stats["mean"],
            "val_loss_std": val_loss_stats["std"],
            "train_loss_mean": train_loss_stats["mean"],
            "train_loss_std": train_loss_stats["std"],
            "entropy_mean": entropy_stats["mean"],
            "entropy_std": entropy_stats["std"],
            "outlier_rate_mean": outlier_stats["mean"],
            "outlier_rate_std": outlier_stats["std"],
            "test_top1_mean": test_acc_stats["mean"],
            "test_top1_std": test_acc_stats["std"],
            "gen_gap_mean": gap_stats["mean"],
            "gen_gap_std": gap_stats["std"],
            "layerwise_entropy": _aggregate_layerwise(records, "layerwise_entropy"),
            "layerwise_outliers": _aggregate_layerwise(records, "layerwise_outliers"),
        }

    summary["meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outputs_dir": os.path.abspath(outputs_dir),
        "register_arms": list(register_arms),
        "seeds": list(seeds),
        "expected_runs": len(register_arms) * len(seeds),
        "completed_runs": total_found,
        "missing_runs": missing_runs,
        "std_convention": "population (numpy.std, ddof=0)",
    }

    destination = summary_path or os.path.join(outputs_dir, SUMMARY_FILENAME)
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def format_summary_table(summary: Dict[str, Any]) -> str:
    """
    Renders the aggregated summary as a fixed-width console table.

    :param summary: Output of :func:`aggregate_sweep_results`.
    :return: Multi-line printable table string.
    """
    header = (
        f"{'Arm':>6} {'Seeds':>6} {'Val Top-1 (%)':>20} {'Val Loss':>18} "
        f"{'Gen Gap':>18} {'Entropy (bits)':>20}"
    )
    lines = [header, "-" * len(header)]

    for key, arm in summary.items():
        if key == "meta" or not isinstance(arm, dict):
            continue
        lines.append(
            f"{'K=' + str(arm['num_registers']):>6} "
            f"{arm['num_seeds']:>6} "
            f"{arm['val_top1_mean']:>11.2f} +/- {arm['val_top1_std']:<5.2f} "
            f"{arm['val_loss_mean']:>10.3f} +/- {arm['val_loss_std']:<4.3f} "
            f"{arm['gen_gap_mean']:>10.3f} +/- {arm['gen_gap_std']:<4.3f} "
            f"{arm['entropy_mean']:>11.3f} +/- {arm['entropy_std']:<5.3f}"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parses aggregator command line options."""
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed sweep metrics into outputs/sweep_summary.json"
    )
    parser.add_argument(
        "--output_dir",
        "--outputs_dir",
        dest="output_dir",
        type=str,
        default="outputs/",
        help="Directory containing the expXX_kY_sZ run folders (default: outputs/)",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Destination JSON path (default: <output_dir>/sweep_summary.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the console summary table",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any of the expected runs are missing",
    )
    return parser.parse_args()


def main() -> int:
    """
    Command line entrypoint for the aggregator.

    :return: Process exit status (0 on success, 1 under ``--strict`` when runs
             are missing).
    """
    args = parse_args()
    summary = aggregate_sweep_results(
        outputs_dir=args.output_dir,
        summary_path=args.summary_path,
    )

    destination = args.summary_path or os.path.join(args.output_dir, SUMMARY_FILENAME)
    meta = summary["meta"]

    if not args.quiet:
        print(format_summary_table(summary))
        print()
    print(
        f"Aggregated {meta['completed_runs']}/{meta['expected_runs']} runs "
        f"-> summary written to {destination}"
    )
    if meta["missing_runs"]:
        print(f"Missing runs: {', '.join(meta['missing_runs'])}")
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
