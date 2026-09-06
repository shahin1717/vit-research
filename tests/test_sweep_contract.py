"""
Sweep Contract Verification Tests
==================================
Verifies that `scripts/run_sweep.sh` drives the real `scripts/train.py` exactly as
the 12-run experimental matrix specifies. Where the previous suite exercises the
aggregator in isolation, this suite closes the seam between the orchestrator and
the trainer, which is the interface no single-owner test covers.

Method
------
The runner is executed in plan mode (`--plan`), which prints the exact command
line it would invoke for each experiment without training anything. Those command
lines are then fed through `scripts.train.parse_args` and `scripts.train.load_config`
- the real implementations, not copies - and the resulting configuration is
asserted against the specification.

The output/checkpoint directory resolution lives inline in `scripts.train.main`,
so the corresponding statements are extracted from the module's AST and executed
verbatim. This keeps the assertion bound to the shipped code: if that block is
ever edited, the test re-executes the edited version rather than a stale copy.

Coverage
--------
- The runner emits exactly the twelve `EXP-01 .. EXP-12` invocations, in order.
- Config file, register count and seed of every run match the matrix table.
- `load_config` resolves each invocation to the intended K, seed and epoch count.
- Output and checkpoint directories resolve to canonical, mutually isolated paths.
- The four-artifact contract is the one the runner enforces.
- The register arms and seeds in `configs/sweep_config.yaml` match the matrix.
"""

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SWEEP = REPO_ROOT / "scripts" / "run_sweep.sh"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"

# The specification's 12-run matrix: (exp index, config file, K, seed).
EXPECTED_MATRIX = [
    (1, "configs/baseline_k0.yaml", 0, 42),
    (2, "configs/baseline_k0.yaml", 0, 1337),
    (3, "configs/baseline_k0.yaml", 0, 3407),
    (4, "configs/vit_tiny_k1.yaml", 1, 42),
    (5, "configs/vit_tiny_k1.yaml", 1, 1337),
    (6, "configs/vit_tiny_k1.yaml", 1, 3407),
    (7, "configs/vit_tiny_k4.yaml", 4, 42),
    (8, "configs/vit_tiny_k4.yaml", 4, 1337),
    (9, "configs/vit_tiny_k4.yaml", 4, 3407),
    (10, "configs/vit_tiny_k8.yaml", 8, 42),
    (11, "configs/vit_tiny_k8.yaml", 8, 1337),
    (12, "configs/vit_tiny_k8.yaml", 8, 3407),
]

REQUIRED_ARTIFACTS = ("metrics.json", "best_model.pth", "last_model.pth", "train_history.csv")

# The interpreter path may contain spaces (a Windows install under Program Files),
# so the command portion is matched non-greedily up to the script name.
PLAN_LINE = re.compile(r"\[plan\]\s+(?P<command>.+?)\s+scripts/train\.py\s+(?P<args>.+)$")


def _find_bash() -> Optional[str]:
    """
    Locates a bash interpreter.

    On Windows the Git for Windows bash is frequently installed but absent from
    ``PATH``, so the standard installation locations are probed before giving up.

    :return: Path to a usable bash executable, or ``None`` if none was found.
    """
    found = shutil.which("bash")
    if found:
        return found

    candidates = [
        os.environ.get("BASH", ""),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\bin\bash.exe"),
        "/bin/bash",
        "/usr/bin/bash",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


BASH = _find_bash()

requires_bash = pytest.mark.skipif(
    BASH is None,
    reason="bash is required to execute scripts/run_sweep.sh",
)


def _run_plan(tmp_path: Path, extra: List[str] = ()) -> List[List[str]]:
    """
    Executes the sweep runner in plan mode and returns the argument list of every
    `scripts/train.py` invocation it would perform.

    Output and checkpoint roots are redirected into a temporary directory so the
    test never touches the repository's real `outputs/` or `checkpoints/` trees.

    :param tmp_path: pytest temporary directory used as the artifact root.
    :param extra: Additional flags to pass to the runner.
    :return: One token list per planned experiment, in execution order.
    :raises AssertionError: If the runner exits non-zero.
    """
    # Paths are handed to bash in POSIX form: a Windows backslash path would be
    # read as escape sequences, and the script is addressed relative to the
    # working directory so no drive-letter translation is needed.
    command = [
        BASH,
        "scripts/run_sweep.sh",
        "--plan",
        "--python",
        Path(sys.executable).as_posix(),
        "--output-root",
        (tmp_path / "outputs").as_posix(),
        "--checkpoint-root",
        (tmp_path / "checkpoints").as_posix(),
        *extra,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert completed.returncode == 0, (
        f"run_sweep.sh --plan exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )

    plans: List[List[str]] = []
    for line in completed.stdout.splitlines():
        match = PLAN_LINE.search(line)
        if match:
            plans.append(match.group("args").split())
    return plans


def _argument_value(tokens: List[str], flag: str) -> str:
    """
    Reads the value that follows a flag in a planned command line.

    :param tokens: Tokenized `scripts/train.py` arguments.
    :param flag: Flag whose value is wanted, e.g. ``--seed``.
    :return: The value token.
    :raises AssertionError: If the flag is absent or has no value.
    """
    assert flag in tokens, f"{flag} missing from planned command: {' '.join(tokens)}"
    index = tokens.index(flag)
    assert index + 1 < len(tokens), f"{flag} has no value in: {' '.join(tokens)}"
    return tokens[index + 1]


def _resolve_directories(cfg: Dict[str, Any], args: Any, exp_name: str) -> Dict[str, Path]:
    """
    Executes the output/checkpoint path resolution statements taken verbatim from
    `scripts.train.main`.

    The statements are located in the module AST between the assignment of
    ``base_out`` and the final branch assigning ``checkpoint_dir``, compiled on
    their own and executed against the supplied configuration, so the assertion
    always reflects the shipped implementation.

    :param cfg: Configuration dictionary produced by `load_config`.
    :param args: Parsed argument namespace.
    :param exp_name: Experiment name from the configuration.
    :return: Mapping with the resolved ``output_dir`` and ``checkpoint_dir``.
    :raises AssertionError: If the block cannot be located in the module source.
    """
    tree = ast.parse(TRAIN_SCRIPT.read_text(encoding="utf-8"), filename=str(TRAIN_SCRIPT))
    main_fn = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )
    assert main_fn is not None, "scripts/train.py no longer defines main()"

    def assigns(node: ast.AST, name: str) -> bool:
        """Reports whether a statement (or any nested statement) assigns `name`."""
        return any(
            isinstance(target, ast.Name) and target.id == name
            for sub in ast.walk(node)
            if isinstance(sub, ast.Assign)
            for target in sub.targets
        )

    start = next((i for i, node in enumerate(main_fn.body) if assigns(node, "base_out")), None)
    end = next(
        (i for i in range(len(main_fn.body) - 1, -1, -1) if assigns(main_fn.body[i], "checkpoint_dir")),
        None,
    )
    assert start is not None and end is not None and end >= start, (
        "Could not locate the output/checkpoint path resolution block in scripts/train.py"
    )

    block = ast.Module(body=main_fn.body[start : end + 1], type_ignores=[])
    ast.fix_missing_locations(block)

    namespace: Dict[str, Any] = {"Path": Path, "cfg": cfg, "args": args, "exp_name": exp_name}
    exec(compile(block, filename=str(TRAIN_SCRIPT), mode="exec"), namespace)

    return {"output_dir": namespace["output_dir"], "checkpoint_dir": namespace["checkpoint_dir"]}


@requires_bash
def test_runner_plans_exactly_the_twelve_run_matrix(tmp_path: Path):
    """The runner emits the twelve specified experiments, in specification order."""
    plans = _run_plan(tmp_path)

    assert len(plans) == 12, f"Expected 12 planned runs, got {len(plans)}"

    for (exp_idx, config, k, seed), tokens in zip(EXPECTED_MATRIX, plans):
        exp_id = f"exp{exp_idx:02d}_k{k}_s{seed}"
        assert _argument_value(tokens, "--config") == config
        assert _argument_value(tokens, "--num_registers") == str(k)
        assert _argument_value(tokens, "--seed") == str(seed)
        assert _argument_value(tokens, "--output_dir").endswith(exp_id)
        assert _argument_value(tokens, "--checkpoint_dir").endswith(exp_id)


@requires_bash
def test_runner_isolates_checkpoints_per_run(tmp_path: Path):
    """Every run receives its own checkpoint directory, so seeds cannot collide."""
    plans = _run_plan(tmp_path)

    checkpoint_dirs = [_argument_value(tokens, "--checkpoint_dir") for tokens in plans]
    output_dirs = [_argument_value(tokens, "--output_dir") for tokens in plans]

    assert len(set(checkpoint_dirs)) == 12
    assert len(set(output_dirs)) == 12
    assert all(out != ckpt for out, ckpt in zip(output_dirs, checkpoint_dirs))


@requires_bash
def test_subset_run_keeps_canonical_experiment_indices(tmp_path: Path):
    """
    A subset run numbers experiments by their position in the full matrix, so a
    Kaggle slice writes directories that merge into the main tree unchanged.
    """
    plans = _run_plan(tmp_path, extra=["--registers", "4"])

    assert len(plans) == 3
    resolved = [os.path.basename(_argument_value(tokens, "--output_dir")) for tokens in plans]
    assert resolved == ["exp07_k4_s42", "exp08_k4_s1337", "exp09_k4_s3407"]


@requires_bash
def test_epoch_override_reaches_the_trainer(tmp_path: Path):
    """The Definition-of-Done smoke sweep forwards --epochs to every run."""
    plans = _run_plan(tmp_path, extra=["--epochs", "1"])

    assert len(plans) == 12
    assert all(_argument_value(tokens, "--epochs") == "1" for tokens in plans)


@requires_bash
def test_planned_command_lines_load_into_the_intended_config(tmp_path: Path, monkeypatch):
    """
    Each planned command line is parsed and merged by the real
    `scripts.train.parse_args` / `load_config`, and must resolve to the register
    count, seed, epoch count and canonical directories of its matrix row.
    """
    from scripts.train import load_config, parse_args

    plans = _run_plan(tmp_path, extra=["--epochs", "1"])

    for (exp_idx, config, k, seed), tokens in zip(EXPECTED_MATRIX, plans):
        exp_id = f"exp{exp_idx:02d}_k{k}_s{seed}"
        monkeypatch.setattr(sys, "argv", ["train.py", *tokens])
        args = parse_args()
        cfg = load_config(args)

        assert cfg["model"]["num_registers"] == k
        assert cfg["experiment"]["seed"] == seed
        assert cfg["training"]["epochs"] == 1
        assert cfg["model"]["backbone"] == "vit_tiny_patch16_224"
        assert cfg["data"]["samples_per_class"] == 100
        # The YAML uses `image_size`; the model must be built at the same size.
        assert cfg["model"]["img_size"] == cfg["data"]["image_size"]

        directories = _resolve_directories(cfg, args, cfg["experiment"]["name"])
        assert directories["output_dir"].name == exp_id
        assert directories["checkpoint_dir"].name == exp_id
        assert directories["output_dir"] != directories["checkpoint_dir"]


def test_cli_register_override_wins_over_the_config_file(monkeypatch):
    """
    The runner passes `--num_registers` on every invocation, so the override must
    take precedence over the arm config. The matrix alone cannot prove this,
    because there the two always agree; this case makes them disagree.
    """
    from scripts.train import load_config, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(REPO_ROOT / "configs" / "baseline_k0.yaml"), "--num_registers", "8"],
    )
    cfg = load_config(parse_args())

    assert cfg["model"]["num_registers"] == 8, "CLI --num_registers did not override the config"


def test_runner_enforces_the_four_artifact_contract():
    """The runner verifies exactly the four artifacts the specification requires."""
    source = RUN_SWEEP.read_text(encoding="utf-8")
    match = re.search(r"REQUIRED_ARTIFACTS=\((?P<items>[^)]*)\)", source)

    assert match is not None, "run_sweep.sh no longer declares REQUIRED_ARTIFACTS"
    declared = tuple(item.strip('"') for item in match.group("items").split())
    assert declared == REQUIRED_ARTIFACTS


def test_runner_records_failures_and_does_not_abort_on_one():
    """
    The runner must not use `set -e`, which would abort the remaining runs on the
    first failure, and must append failures to outputs/failures.log.
    """
    source = RUN_SWEEP.read_text(encoding="utf-8")

    assert not re.search(r"^set -e\b", source, flags=re.MULTILINE)
    assert "FAILURE_LOG=" in source
    assert "failures.log" in source
    assert "empty_cache" in source


def test_sweep_config_matches_the_specified_matrix():
    """configs/sweep_config.yaml declares the four arms and the three project seeds."""
    with open(REPO_ROOT / "configs" / "sweep_config.yaml", "r", encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle)

    assert sweep["parameters"]["num_registers"]["values"] == [0, 1, 4, 8]
    assert sweep["parameters"]["seed"]["values"] == [42, 1337, 3407]
    assert sweep["parameters"]["backbone"]["values"] == ["vit_tiny_patch16_224"]
    assert sweep["parameters"]["samples_per_class"]["value"] == 100


@pytest.mark.parametrize(
    "config_name,expected_k",
    [
        ("baseline_k0.yaml", 0),
        ("vit_tiny_k1.yaml", 1),
        ("vit_tiny_k4.yaml", 4),
        ("vit_tiny_k8.yaml", 8),
    ],
)
def test_arm_configs_declare_the_correct_register_count(config_name: str, expected_k: int):
    """Each treatment arm config pins its register count and the shared hyperparameters."""
    with open(REPO_ROOT / "configs" / config_name, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    assert cfg["model"]["num_registers"] == expected_k
    assert cfg["model"]["backbone"] == "vit_tiny_patch16_224"
    assert cfg["model"]["num_classes"] == 100
    assert cfg["data"]["samples_per_class"] == 100
    assert cfg["data"]["val_split"] == 0.1
    assert cfg["training"]["epochs"] == 50
    assert cfg["training"]["amp"] is True
