"""
Sweep Aggregation Verification Tests
=====================================
Validates that `src/utils/logger.py` correctly reduces the per-run artifacts
written by `scripts/train.py` into `outputs/sweep_summary.json`.

Coverage:
- Run directory discovery and the `expXX_kY_sZ` naming convention.
- Derivation of `best_val_loss`, `final_train_loss` and `mean_layerwise_entropy`
  from the nested trainer history when they are absent as top-level keys.
- Mean and population standard deviation across the three project seeds.
- The exact key contract consumed by `src/utils/export_latex.py`.
- Graceful zero-filled handling of treatment arms with no completed run.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from src.utils.logger import (
    REGISTER_ARMS,
    SEEDS,
    aggregate_sweep_results,
    collect_runs,
    format_summary_table,
    parse_run_directory,
)

# Key contract required by the LaTeX results table generator.
LATEX_TABLE_KEYS = (
    "val_top1_mean",
    "val_top1_std",
    "val_loss_mean",
    "val_loss_std",
    "gen_gap_mean",
    "entropy_mean",
    "entropy_std",
)

NUM_LAYERS = 12


def _metrics_payload(
    k_registers: int,
    seed: int,
    best_val_top1: float,
    best_val_loss: float,
    final_train_loss: float,
    entropy_base: float,
    epochs: int = 3,
) -> Dict:
    """
    Builds a metrics.json payload with the exact structure emitted by
    `scripts/train.py`, including the nested per-epoch history and the final
    test evaluation block.

    :param k_registers: Register count K of the run.
    :param seed: Random seed of the run.
    :param best_val_top1: Top-1 accuracy at the best validation epoch.
    :param best_val_loss: Validation loss at the best validation epoch.
    :param final_train_loss: Training loss at the last epoch.
    :param entropy_base: Layer-0 entropy; later layers decay linearly from it.
    :param epochs: Number of epochs to synthesize.
    :return: Nested metrics dictionary.
    """
    best_epoch = epochs - 1
    layerwise_entropy = {str(layer): entropy_base - 0.1 * layer for layer in range(NUM_LAYERS)}
    layerwise_outliers = {str(layer): 0.01 + 0.001 * layer for layer in range(NUM_LAYERS)}

    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        is_best = epoch == best_epoch
        is_last = epoch == epochs
        history.append(
            {
                "epoch": epoch,
                "train_loss": final_train_loss if is_last else final_train_loss + 0.5,
                "train_top1": 40.0 + epoch,
                "train_top5": 70.0 + epoch,
                "val_loss": best_val_loss if is_best else best_val_loss + 0.4,
                "val_top1": best_val_top1 if is_best else best_val_top1 - 3.0,
                "val_top5": 60.0 + epoch,
                "generalization_gap": 0.0,
                "accuracy_gap": 0.0,
                "lr": 0.0005,
                "duration_sec": 8.4,
                "layerwise_entropy": layerwise_entropy,
                "layerwise_outliers": layerwise_outliers,
            }
        )

    return {
        "experiment": f"rViT_K{k_registers}_seed{seed}",
        "k_registers": k_registers,
        "seed": seed,
        "config": {"model": {"num_registers": k_registers}},
        "total_params": 5_710_000,
        "trainable_params": 5_710_000,
        "total_training_time_sec": 480.0,
        "best_val_top1": best_val_top1,
        "best_epoch": best_epoch,
        "test_results": {
            "test_loss": best_val_loss + 0.05,
            "test_top1": best_val_top1 - 1.5,
            "test_top5": 78.0,
            "layerwise_entropy": layerwise_entropy,
            "layerwise_outliers": layerwise_outliers,
        },
        "history": history,
    }


def _write_run(
    outputs_dir: Path,
    exp_idx: int,
    k_registers: int,
    seed: int,
    payload: Optional[Dict],
) -> Path:
    """
    Materializes one `outputs/expXX_kY_sZ/` directory with the four contract
    artifacts. Passing ``payload=None`` creates the directory without a
    metrics.json, simulating a crashed run.

    :param outputs_dir: Root outputs directory.
    :param exp_idx: Global experiment index (1-based).
    :param k_registers: Register count K.
    :param seed: Random seed.
    :param payload: metrics.json content, or ``None`` for a failed run.
    :return: The created run directory path.
    """
    run_dir = outputs_dir / f"exp{exp_idx:02d}_k{k_registers}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "train_history.csv").write_text("epoch,train_loss\n1,4.0\n", encoding="utf-8")
    (run_dir / "best_model.pth").write_bytes(b"")
    (run_dir / "last_model.pth").write_bytes(b"")

    if payload is not None:
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return run_dir


@pytest.fixture()
def full_sweep_outputs(tmp_path: Path) -> Path:
    """
    Builds a complete synthetic 12-run sweep tree with deterministic metrics.

    Accuracy, loss and entropy are offset per arm and per seed so every
    aggregate is analytically predictable.

    :param tmp_path: pytest temporary directory.
    :return: Path to the populated outputs directory.
    """
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    exp_idx = 0
    for arm_position, k in enumerate(REGISTER_ARMS):
        for seed_position, seed in enumerate(SEEDS):
            exp_idx += 1
            _write_run(
                outputs_dir,
                exp_idx,
                k,
                seed,
                _metrics_payload(
                    k_registers=k,
                    seed=seed,
                    best_val_top1=50.0 + arm_position + seed_position,
                    best_val_loss=2.0 + 0.1 * arm_position,
                    final_train_loss=1.0 + 0.1 * arm_position,
                    entropy_base=5.0 + 0.5 * arm_position,
                ),
            )

    return outputs_dir


def test_run_discovery_finds_the_full_matrix(full_sweep_outputs: Path):
    """All twelve canonical run directories are discovered and keyed correctly."""
    runs = collect_runs(str(full_sweep_outputs))

    assert sorted(runs.keys()) == sorted(REGISTER_ARMS)
    for k in REGISTER_ARMS:
        assert sorted(runs[k].keys()) == sorted(SEEDS)
    assert sum(len(seeds) for seeds in runs.values()) == 12


def test_run_discovery_ignores_foreign_directories(full_sweep_outputs: Path):
    """Directories outside the sweep naming convention are not aggregated."""
    (full_sweep_outputs / "scratch_notes").mkdir()
    (full_sweep_outputs / "baseline_k0").mkdir()
    (full_sweep_outputs / "sweep_summary.json").write_text("{}", encoding="utf-8")

    runs = collect_runs(str(full_sweep_outputs))
    assert sum(len(seeds) for seeds in runs.values()) == 12


def test_parse_run_directory_derives_missing_top_level_keys(tmp_path: Path):
    """
    best_val_loss, final_train_loss and mean_layerwise_entropy are recovered
    from the nested history when the trainer does not export them flat.
    """
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    payload = _metrics_payload(
        k_registers=4,
        seed=3407,
        best_val_top1=61.0,
        best_val_loss=1.75,
        final_train_loss=0.90,
        entropy_base=4.8,
    )
    run_dir = _write_run(outputs_dir, 9, 4, 3407, payload)

    record = parse_run_directory(str(run_dir))

    assert record is not None
    assert record["best_val_top1"] == pytest.approx(61.0)
    assert record["best_val_loss"] == pytest.approx(1.75)
    assert record["final_train_loss"] == pytest.approx(0.90)
    # Mean over 12 layers of an arithmetic sequence starting at 4.8 with step -0.1.
    expected_entropy = float(np.mean([4.8 - 0.1 * layer for layer in range(NUM_LAYERS)]))
    assert record["mean_layerwise_entropy"] == pytest.approx(expected_entropy)
    assert set(record["layerwise_entropy"].keys()) == set(range(NUM_LAYERS))


def test_parse_run_directory_returns_none_for_crashed_run(tmp_path: Path):
    """A run directory without metrics.json is reported as unparseable."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    run_dir = _write_run(outputs_dir, 1, 0, 3407, payload=None)

    assert parse_run_directory(str(run_dir)) is None


def test_aggregate_writes_summary_with_latex_contract(full_sweep_outputs: Path):
    """The written summary exposes every key the LaTeX table generator reads."""
    summary = aggregate_sweep_results(outputs_dir=str(full_sweep_outputs))

    summary_path = full_sweep_outputs / "sweep_summary.json"
    assert summary_path.is_file()

    on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk.keys() == summary.keys()

    for k in REGISTER_ARMS:
        arm = on_disk[f"k_{k}"]
        for key in LATEX_TABLE_KEYS:
            assert key in arm, f"k_{k} is missing the required key '{key}'"
            assert isinstance(arm[key], float)


def test_aggregate_statistics_match_numpy_reference(full_sweep_outputs: Path):
    """Mean and population std reproduce the analytic values for each arm."""
    summary = aggregate_sweep_results(outputs_dir=str(full_sweep_outputs))

    for arm_position, k in enumerate(REGISTER_ARMS):
        arm = summary[f"k_{k}"]
        expected_accs = [50.0 + arm_position + seed_position for seed_position in range(len(SEEDS))]
        expected_entropy = float(
            np.mean([5.0 + 0.5 * arm_position - 0.1 * layer for layer in range(NUM_LAYERS)])
        )

        assert arm["num_seeds"] == len(SEEDS)
        assert arm["seeds"] == list(SEEDS)
        assert arm["val_top1_mean"] == pytest.approx(float(np.mean(expected_accs)))
        assert arm["val_top1_std"] == pytest.approx(float(np.std(expected_accs)))
        assert arm["val_loss_mean"] == pytest.approx(2.0 + 0.1 * arm_position)
        assert arm["val_loss_std"] == pytest.approx(0.0)
        assert arm["entropy_mean"] == pytest.approx(expected_entropy)
        # Identical validation and training losses across seeds -> constant gap.
        assert arm["gen_gap_mean"] == pytest.approx(1.0)
        assert arm["gen_gap_std"] == pytest.approx(0.0)


def test_aggregate_reports_layerwise_error_bands(full_sweep_outputs: Path):
    """Per-layer entropy statistics are emitted for the entropy-versus-layer figure."""
    summary = aggregate_sweep_results(outputs_dir=str(full_sweep_outputs))

    layerwise = summary["k_4"]["layerwise_entropy"]
    assert len(layerwise) == NUM_LAYERS
    for layer in range(NUM_LAYERS):
        entry = layerwise[str(layer)]
        assert entry["n"] == len(SEEDS)
        assert entry["std"] == pytest.approx(0.0)
    assert layerwise["0"]["mean"] > layerwise["11"]["mean"]


def test_aggregate_tolerates_incomplete_sweep(tmp_path: Path):
    """
    A partially completed sweep aggregates without raising: present arms are
    reduced normally and absent arms are zero-filled with num_seeds = 0.
    """
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    _write_run(
        outputs_dir,
        3,
        0,
        3407,
        _metrics_payload(0, 3407, best_val_top1=48.0, best_val_loss=2.2,
                         final_train_loss=1.1, entropy_base=5.0),
    )
    # EXP-01 crashed before writing metrics.json.
    _write_run(outputs_dir, 1, 0, 42, payload=None)

    summary = aggregate_sweep_results(outputs_dir=str(outputs_dir))

    assert summary["k_0"]["num_seeds"] == 1
    assert summary["k_0"]["val_top1_mean"] == pytest.approx(48.0)
    assert summary["k_0"]["val_top1_std"] == pytest.approx(0.0)

    for k in (1, 4, 8):
        arm = summary[f"k_{k}"]
        assert arm["num_seeds"] == 0
        assert arm["val_top1_mean"] == 0.0
        assert arm["gen_gap_mean"] == 0.0

    meta = summary["meta"]
    assert meta["expected_runs"] == 12
    assert meta["completed_runs"] == 1
    assert "k0_s42" in meta["missing_runs"]
    assert len(meta["missing_runs"]) == 11


def test_aggregate_rejects_missing_outputs_directory(tmp_path: Path):
    """A non-existent outputs directory raises rather than writing an empty summary."""
    with pytest.raises(FileNotFoundError):
        aggregate_sweep_results(outputs_dir=str(tmp_path / "does_not_exist"))


def test_format_summary_table_renders_every_arm(full_sweep_outputs: Path):
    """The console table renders one row per treatment arm and skips metadata."""
    summary = aggregate_sweep_results(outputs_dir=str(full_sweep_outputs))
    table = format_summary_table(summary)

    lines = table.splitlines()
    assert len(lines) == 2 + len(REGISTER_ARMS)
    for k in REGISTER_ARMS:
        assert f"K={k}" in table
    assert "meta" not in table
