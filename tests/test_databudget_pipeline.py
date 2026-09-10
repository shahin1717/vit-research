"""
Unit tests for 300 Images/Class Data Budget Extension Pipeline
==============================================================
Validates:
1. All 4 new YAML configs for 300 images/class.
2. Naming convention and regex contract for databudget sweep directories.
3. Multi-seed metrics parsing and aggregation logic in `src/utils/aggregate_databudget.py`.
4. Stratified sampler split arithmetic for 300 samples/class (270 train / 30 val).
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from src.utils.aggregate_databudget import (
    RUN_DIR_PATTERN,
    aggregate_sweep,
    parse_run,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


class TestDataBudgetConfigs:
    """Validates the 4 new configuration files."""

    @pytest.mark.parametrize(
        "config_rel_path,expected_k",
        [
            ("configs/baseline_k0_300pc.yaml", 0),
            ("configs/vit_tiny_k1_300pc.yaml", 1),
            ("configs/vit_tiny_k4_300pc.yaml", 4),
            ("configs/vit_tiny_k8_300pc.yaml", 8),
        ],
    )
    def test_config_structure_and_parameters(self, config_rel_path: str, expected_k: int):
        config_path = WORKSPACE_ROOT / config_rel_path
        assert config_path.is_file(), f"Missing config file: {config_rel_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "experiment" in cfg
        assert "model" in cfg
        assert "data" in cfg
        assert "training" in cfg
        assert "logging" in cfg

        # Verify model parameters
        assert cfg["model"]["num_registers"] == expected_k
        assert cfg["model"]["backbone"] == "vit_tiny_patch16_224"
        assert cfg["model"]["num_classes"] == 100

        # Verify data budget configuration
        assert cfg["data"]["samples_per_class"] == 300, (
            f"Expected samples_per_class == 300, got {cfg['data'].get('samples_per_class')}"
        )
        assert cfg["data"]["val_split"] == 0.1
        assert cfg["data"]["image_size"] == 224

        # Verify output isolation (must point to *_databudget)
        save_dir = cfg["logging"]["save_dir"]
        ckpt_dir = cfg["logging"]["checkpoint_dir"]
        assert "outputs_databudget" in save_dir, f"save_dir not isolated: {save_dir}"
        assert "checkpoints_databudget" in ckpt_dir, f"checkpoint_dir not isolated: {ckpt_dir}"


class TestDataBudgetRegexAndParsing:
    """Validates regex matching and directory structure contracts."""

    def test_run_dir_pattern_matches_valid_directories(self):
        valid_dirs = [
            "exp13_k0_pc300_s42",
            "exp14_k0_pc300_s1337",
            "exp19_k4_pc300_s42",
            "exp24_k8_pc300_s3407",
        ]
        for name in valid_dirs:
            match = RUN_DIR_PATTERN.match(name)
            assert match is not None, f"Failed to match valid directory: {name}"
            assert int(match.group("pc")) == 300

    def test_run_dir_pattern_rejects_legacy_directories(self):
        legacy_dirs = [
            "exp01_k0_s42",
            "exp07_k4_s42",
            "baseline_k0",
            "random_folder",
        ]
        for name in legacy_dirs:
            match = RUN_DIR_PATTERN.match(name)
            assert match is None, f"Legacy or invalid directory should not match: {name}"

    def test_parse_run_extracts_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_file = Path(tmp_dir) / "metrics.json"
            mock_data = {
                "best_epoch": 5,
                "test_results": {
                    "test_top1": 78.5,
                    "test_loss": 1.45,
                    "test_top5": 94.2,
                },
                "history": [
                    {"epoch": 1, "train_loss": 2.5, "val_loss": 2.2},
                    {"epoch": 5, "train_loss": 0.75, "val_loss": 1.40},
                ],
            }
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(mock_data, f)

            record = parse_run(tmp_dir)
            assert record is not None
            assert record["test_top1"] == 78.5
            assert record["test_loss"] == 1.45
            assert record["test_top5"] == 94.2
            assert record["val_loss"] == 1.40
            assert record["train_loss"] == 0.75
            assert pytest.approx(record["gen_gap"], 1e-5) == (1.40 - 0.75)


class TestDataBudgetAggregationEngine:
    """Validates multi-seed reduction in aggregate_databudget.py."""

    def test_aggregate_sweep_multi_seed_reduction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create synthetic runs: K=0 with 2 seeds, K=4 with 2 seeds
            runs = [
                ("exp13_k0_pc300_s42", 76.0, 1.50, 0.70),
                ("exp14_k0_pc300_s1337", 77.0, 1.40, 0.80),
                ("exp19_k4_pc300_s42", 78.0, 1.35, 0.65),
                ("exp20_k4_pc300_s1337", 78.5, 1.30, 0.60),
            ]
            for dir_name, top1, val_loss, gen_gap in runs:
                run_path = Path(tmp_dir) / dir_name
                run_path.mkdir(parents=True, exist_ok=True)
                mock_metrics = {
                    "best_epoch": 2,
                    "test_results": {"test_top1": top1, "test_loss": val_loss + 0.05, "test_top5": 95.0},
                    "history": [
                        {"epoch": 1, "train_loss": 2.0, "val_loss": 2.1},
                        {"epoch": 2, "train_loss": val_loss - gen_gap, "val_loss": val_loss},
                    ],
                }
                with open(run_path / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(mock_metrics, f)

            summary = aggregate_sweep(tmp_dir)
            assert "k0_pc300" in summary
            assert "k4_pc300" in summary

            # K=0 checks
            k0_entry = summary["k0_pc300"]
            assert k0_entry["num_registers"] == 0
            assert k0_entry["n_seeds"] == 2
            assert pytest.approx(k0_entry["test_top1_mean"], 1e-4) == 76.5
            assert pytest.approx(k0_entry["test_top1_std"], 1e-4) == 0.5
            assert pytest.approx(k0_entry["val_loss_mean"], 1e-4) == 1.45
            assert pytest.approx(k0_entry["gen_gap_mean"], 1e-4) == 0.75

            # K=4 checks
            k4_entry = summary["k4_pc300"]
            assert k4_entry["num_registers"] == 4
            assert k4_entry["n_seeds"] == 2
            assert pytest.approx(k4_entry["test_top1_mean"], 1e-4) == 78.25
            assert pytest.approx(k4_entry["test_top1_std"], 1e-4) == 0.25


class TestStratifiedDataBudgetSplitArithmetic:
    """Validates split proportions for samples_per_class == 300."""

    def test_split_proportions_at_300_samples_per_class(self):
        samples_per_class = 300
        train_ratio = 0.9
        num_classes = 100

        n_train_per_class = int(round(samples_per_class * train_ratio))
        n_val_per_class = samples_per_class - n_train_per_class

        assert n_train_per_class == 270, f"Expected 270 train/class, got {n_train_per_class}"
        assert n_val_per_class == 30, f"Expected 30 val/class, got {n_val_per_class}"
        assert (n_train_per_class + n_val_per_class) == samples_per_class

        total_train = n_train_per_class * num_classes
        total_val = n_val_per_class * num_classes
        assert total_train == 27000
        assert total_val == 3000
        assert total_train + total_val == 30000
