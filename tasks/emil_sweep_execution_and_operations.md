# 📦 Team Task Package: Emil — Sweep Execution & Experiment Operations Lead

**Assignee:** Emil
**Role:** Experiment Configuration, Automation & GPU Operations Lead
**Target Code Files:**
* `configs/baseline_k0.yaml`, `configs/vit_tiny_k1.yaml`, `configs/vit_tiny_k4.yaml`, `configs/vit_tiny_k8.yaml`, `configs/sweep_config.yaml`
* `scripts/run_sweep.sh`
* `notebooks/kaggle_sweep_runner.ipynb`
* `tests/test_sweep_contract.py`
* `docs/sweep_runbook.md`

**Target Delivery:** Wednesday, 2 September 2026
**Cross-References:** [Team Task Division](../team_task_division.md) | [Results Engine & Analysis Package](rufet_results_engine_and_analysis.md) | [Compute Budget](../compute_request_and_budget.md) | [Integration Blockers](INTEGRATION_BLOCKERS.md)

---

## 🎯 1. Mission & Scientific Context

Statistical validity requires every treatment arm to be replicated across three
independent random seeds (`42`, `1337`, `3407`), giving
$4 \text{ arms} \times 3 \text{ seeds} = 12$ complete training runs.

Twelve runs launched by hand is twelve chances to mistype a seed, overwrite a
checkpoint, or lose a crashed run without noticing. Your mission is to make the
matrix reproducible and unattended:

1. **Configuration Templates (`configs/*.yaml`):** Structured, reproducible YAML
   defining model hyperparameters, optimizer settings, dataset paths and the seed
   list, one file per treatment arm plus the sweep definition.
2. **Automated Sweep Runner (`scripts/run_sweep.sh`):** A robust unattended bash
   runner executing all twelve experiments sequentially on the assigned A100 MIG
   instance with zero manual intervention.
3. **Kaggle Parallel Runner (`notebooks/kaggle_sweep_runner.ipynb`):** A portable
   notebook letting team members execute subsets of the twelve runs on Kaggle GPU
   instances in parallel and merge the slices back.
4. **Execution Guarantees (`tests/test_sweep_contract.py`, `docs/sweep_runbook.md`):**
   An automated proof that the runner drives the trainer exactly as this matrix
   specifies, and an operations runbook for the GPU slot.

**Interface with the analysis side:** you own everything up to and including the
per-run artifacts in `outputs/expXX_kY_sZ/`. The results engine and analysis package
(Rufet) consumes them. You do not compute statistics; they do not launch runs.

---

## 📊 2. The 12-Run Experimental Matrix

| Exp ID | Config File | Backbone | Registers ($K$) | Seed | Output Directory |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `EXP-01` | `configs/baseline_k0.yaml` | `vit_tiny_patch16_224` | 0 (Control) | 42 | `outputs/exp01_k0_s42/` |
| `EXP-02` | `configs/baseline_k0.yaml` | `vit_tiny_patch16_224` | 0 (Control) | 1337 | `outputs/exp02_k0_s1337/` |
| `EXP-03` | `configs/baseline_k0.yaml` | `vit_tiny_patch16_224` | 0 (Control) | 3407 | `outputs/exp03_k0_s3407/` |
| `EXP-04` | `configs/vit_tiny_k1.yaml` | `vit_tiny_patch16_224` | 1 | 42 | `outputs/exp04_k1_s42/` |
| `EXP-05` | `configs/vit_tiny_k1.yaml` | `vit_tiny_patch16_224` | 1 | 1337 | `outputs/exp05_k1_s1337/` |
| `EXP-06` | `configs/vit_tiny_k1.yaml` | `vit_tiny_patch16_224` | 1 | 3407 | `outputs/exp06_k1_s3407/` |
| `EXP-07` | `configs/vit_tiny_k4.yaml` | `vit_tiny_patch16_224` | 4 | 42 | `outputs/exp07_k4_s42/` |
| `EXP-08` | `configs/vit_tiny_k4.yaml` | `vit_tiny_patch16_224` | 4 | 1337 | `outputs/exp08_k4_s1337/` |
| `EXP-09` | `configs/vit_tiny_k4.yaml` | `vit_tiny_patch16_224` | 4 | 3407 | `outputs/exp09_k4_s3407/` |
| `EXP-10` | `configs/vit_tiny_k8.yaml` | `vit_tiny_patch16_224` | 8 | 42 | `outputs/exp10_k8_s42/` |
| `EXP-11` | `configs/vit_tiny_k8.yaml` | `vit_tiny_patch16_224` | 8 | 1337 | `outputs/exp11_k8_s1337/` |
| `EXP-12` | `configs/vit_tiny_k8.yaml` | `vit_tiny_patch16_224` | 8 | 3407 | `outputs/exp12_k8_s3407/` |

The experiment index is fixed by the position in the **full** matrix. A subset run —
one arm on Kaggle, or a re-run of a single failed seed — therefore still writes the
same canonical directory names and merges back without renaming.

---

## ⚙️ 3. Configuration Templates

`configs/sweep_config.yaml` is the single source of truth for the matrix:

```yaml
parameters:
  num_registers:
    values: [0, 1, 4, 8]
  seed:
    values: [42, 1337, 3407]
  backbone:
    values: ["vit_tiny_patch16_224"]
  samples_per_class:
    value: 100
```

The runner reads the arms and seeds from this file so the executed matrix and the
documented matrix cannot drift apart; the hard-coded matrix remains as a fallback.

The four arm configs are identical except for `model.num_registers` and the
per-arm output paths: `vit_tiny_patch16_224`, 100 classes, 100 samples/class,
`val_split: 0.1`, batch 64, AdamW at `lr=5e-4` with `weight_decay=0.05`, cosine
schedule with 5 warmup epochs down to `min_lr=1e-5`, 50 epochs, AMP on, gradient
clipping at 1.0.

---

## 💻 4. Runner Requirements (`scripts/run_sweep.sh`)

1. **Sequential execution** of the twelve experiments, driving `scripts/train.py`
   with `--config`, `--num_registers`, `--seed`, `--output_dir`, `--checkpoint_dir`.
2. **Per-run checkpoint isolation.** Each arm config pins a single
   `logging.checkpoint_dir`; without an override the three seeds of an arm
   overwrite one another's `best_model.pt`. The runner must pass
   `--checkpoint_dir checkpoints/<EXP_ID>` so every run owns its own tree.
3. **Failure resilience.** A crashing run must be recorded in
   `outputs/failures.log` (timestamp, experiment, exit code, log path) and the
   remaining runs must continue. `set -e` is therefore prohibited at the top level.
4. **VRAM hygiene.** `torch.cuda.empty_cache()` between runs, on top of the
   implicit release at process exit.
5. **Artifact verification.** After each run, confirm the four contract artifacts
   exist; a run that exits `0` but leaves them incomplete is a failure.
6. **Logging.** Per-run `outputs/<EXP_ID>.log`, plus a cumulative `outputs/sweep.log`.
7. **Subset selection** (`--registers`, `--seeds`) so Kaggle sessions can execute
   slices, and `--epochs` for the smoke sweep.
8. **Resume** (`--skip-existing`) so a partially failed matrix can be completed
   without recomputing finished runs.
9. **Closing aggregation** — invoke `python src/utils/logger.py --output_dir outputs/`.

---

## ⚠️ 5. Operations & Hardware Safety Directives

1. **Subprocess failure resilience:** one broken experiment must not end the sweep.
2. **GPU memory leakage:** sequential subprocesses free VRAM at exit, but the
   explicit cache flush guarantees zero fragmentation before the next run starts.
3. **Structured metrics standard:** every run directory `outputs/expXX_kY_sZ/`
   must contain exactly:
   * `metrics.json` — Top-1 accuracy, losses, entropy
   * `best_model.pth` — weights at the best validation epoch
   * `last_model.pth` — final-epoch checkpoint
   * `train_history.csv` — per-epoch training and validation loss curves
4. **Line endings:** `run_sweep.sh` must be stored with LF endings. A CRLF copy
   committed from Windows dies on the Linux A100 with `$'\r': command not found`.
5. **Dataloader workers:** `num_workers=4` on Linux/A100, `2` on Kaggle and
   Windows, to avoid shared-memory deadlocks.

---

## 🛰️ 6. Kaggle Parallel Execution

`notebooks/kaggle_sweep_runner.ipynb` executes a slice of the matrix on a Kaggle
GPU session. Suggested split: one arm per session (`REGISTERS = [0]`, `[1]`, `[4]`,
`[8]`), three runs each. The notebook obtains the repository, installs missing
dependencies, runs the preflight check, invokes the runner for its slice,
aggregates, and packages `outputs/expXX_kY_sZ/` for merge-back. Because directory
names are globally unique, slices unpack into one tree without collisions.

---

## ✅ 7. Definition of Done & Verification Test

Your deliverable is complete when:

1. `scripts/run_sweep.sh` has executable permissions. On a Windows clone
   (`core.filemode = false`) the bit is not recorded by a plain `git add`; set it
   explicitly with `git update-index --add --chmod=+x scripts/run_sweep.sh`.
2. A dry-run with `--epochs 1` completes for all 12 experiments without error, and
   each of the twelve directories carries all four contract artifacts.
3. `python src/utils/logger.py` correctly parses those runs and generates
   `outputs/sweep_summary.json`.
4. `pytest tests/test_sweep_contract.py` passes, proving the runner emits exactly
   the twelve specified invocations, that `scripts/train.py` resolves each one to
   the intended register count, seed, epoch count and canonical directories, and
   that the artifact contract and failure-resilience rules are enforced.
5. `python scripts/preflight_check.py` exits `0` before any GPU time is consumed.

**Operational deliverable:** the twelve completed run directories in `outputs/`,
handed to the results engine and analysis package. Budgeted wall-clock on the
assigned A100 MIG 3g.20gb slice is ~8 minutes per run, ~96 minutes for the matrix.
