# 🛠️ Sweep Operations Runbook

**Owner:** Emil — Sweep Orchestration & Operations Lead
**Scope:** `scripts/run_sweep.sh`, `configs/*.yaml`, `src/utils/logger.py`, `notebooks/kaggle_sweep_runner.ipynb`
**Cross-references:** [Task package](../tasks/emil_sweep_execution_and_operations.md) · [Integration blockers](../tasks/INTEGRATION_BLOCKERS.md) · [Compute budget](../compute_request_and_budget.md)

---

## 1. The experiment matrix

Twelve runs: four treatment arms `K ∈ {0, 1, 4, 8}` replicated over three seeds `{42, 1337, 3407}`.

| Exp | Config | K | Seed | Output directory | Checkpoint directory |
| :--- | :--- | :---: | :---: | :--- | :--- |
| EXP-01 | `configs/baseline_k0.yaml` | 0 | 42 | `outputs/exp01_k0_s42/` | `checkpoints/exp01_k0_s42/` |
| EXP-02 | `configs/baseline_k0.yaml` | 0 | 1337 | `outputs/exp02_k0_s1337/` | `checkpoints/exp02_k0_s1337/` |
| EXP-03 | `configs/baseline_k0.yaml` | 0 | 3407 | `outputs/exp03_k0_s3407/` | `checkpoints/exp03_k0_s3407/` |
| EXP-04 | `configs/vit_tiny_k1.yaml` | 1 | 42 | `outputs/exp04_k1_s42/` | `checkpoints/exp04_k1_s42/` |
| EXP-05 | `configs/vit_tiny_k1.yaml` | 1 | 1337 | `outputs/exp05_k1_s1337/` | `checkpoints/exp05_k1_s1337/` |
| EXP-06 | `configs/vit_tiny_k1.yaml` | 1 | 3407 | `outputs/exp06_k1_s3407/` | `checkpoints/exp06_k1_s3407/` |
| EXP-07 | `configs/vit_tiny_k4.yaml` | 4 | 42 | `outputs/exp07_k4_s42/` | `checkpoints/exp07_k4_s42/` |
| EXP-08 | `configs/vit_tiny_k4.yaml` | 4 | 1337 | `outputs/exp08_k4_s1337/` | `checkpoints/exp08_k4_s1337/` |
| EXP-09 | `configs/vit_tiny_k4.yaml` | 4 | 3407 | `outputs/exp09_k4_s3407/` | `checkpoints/exp09_k4_s3407/` |
| EXP-10 | `configs/vit_tiny_k8.yaml` | 8 | 42 | `outputs/exp10_k8_s42/` | `checkpoints/exp10_k8_s42/` |
| EXP-11 | `configs/vit_tiny_k8.yaml` | 8 | 1337 | `outputs/exp11_k8_s1337/` | `checkpoints/exp11_k8_s1337/` |
| EXP-12 | `configs/vit_tiny_k8.yaml` | 8 | 3407 | `outputs/exp12_k8_s3407/` | `checkpoints/exp12_k8_s3407/` |

The experiment index is a function of the position in the **full** matrix, so a partial
run (a single arm on Kaggle, a re-run of one failed seed) still writes the same
canonical directory names and merges without renaming.

---

## 2. Standard operating procedure

```bash
# 0. Environment
conda activate aiac-res
pip install -r requirements.txt
chmod +x scripts/run_sweep.sh

# 1. Preflight — never burn GPU hours on a broken tree
python scripts/preflight_check.py

# 2. Smoke sweep — one epoch per run, exercises every arm end to end
bash scripts/run_sweep.sh --epochs 1

# 3. Reset the smoke artifacts, then launch the real sweep unattended
rm -rf outputs/exp* checkpoints/exp*
nohup bash scripts/run_sweep.sh > outputs/nohup_sweep.log 2>&1 &

# 4. Monitor
tail -f outputs/sweep.log

# 5. Verify the matrix is complete (non-zero exit while any run is missing)
python src/utils/logger.py --output_dir outputs/ --strict
```

Budgeted wall-clock on the assigned A100 MIG 3g.20gb slice: ~8 min per run,
**~96 min for the full matrix** (see the compute budget document).

---

## 3. `scripts/run_sweep.sh` options

| Flag | Purpose |
| :--- | :--- |
| `-e, --epochs N` | Override the 50-epoch schedule; `1` for smoke sweeps |
| `-r, --registers "0 4"` | Run a subset of treatment arms |
| `-s, --seeds "42 1337"` | Run a subset of seeds |
| `-d, --data-dir PATH` | Dataset root forwarded to `scripts/train.py` |
| `-o, --output-root PATH` | Alternative root for run directories |
| `-c, --checkpoint-root PATH` | Alternative root for checkpoint directories |
| `-p, --python BIN` | Interpreter to use (Kaggle passes `sys.executable`) |
| `--extra-args "..."` | Raw arguments forwarded to the trainer, e.g. `"--num_workers 2"` |
| `--skip-existing` | Skip runs whose `metrics.json` already exists (resume) |
| `--no-aggregate` | Skip the closing aggregation step |
| `--plan` | Print the execution plan without training (not the §5 smoke sweep, which is `--epochs 1` and does train) |

Exit status: `0` all runs succeeded · `1` at least one run failed · `2` unknown option ·
`3` the Python interpreter was not found (the runner refuses to start rather than
failing all twelve runs identically; if `python` is absent but `python3` is present
it switches automatically).

The register arms and seeds are read from `configs/sweep_config.yaml`
(`parameters.num_registers.values` and `parameters.seed.values`) so the runner and the
documented sweep definition cannot drift apart; the built-in matrix is the fallback.

---

## 4. Per-run artifact contract

Every completed run directory `outputs/expXX_kY_sZ/` must contain:

| Artifact | Content |
| :--- | :--- |
| `metrics.json` | Top-1/Top-5, losses, generalization gap, layer-wise entropy and outlier rates, full epoch history |
| `best_model.pth` | Weights at the best validation epoch |
| `last_model.pth` | Final-epoch checkpoint |
| `train_history.csv` | Per-epoch train/val loss and accuracy curves |

The runner verifies all four after each run. A run that exits `0` but leaves the
contract incomplete is recorded as a failure in `outputs/failures.log`, which is how a
silently truncated run is caught before the analysis phase depends on it.

Additional files written by the trainer for compatibility with `scripts/eval.py` and
`scripts/visualize_attention.py`: `best_model.pt`, `latest_checkpoint.pt`, `summary.csv`
in the run directory, plus the mirrored checkpoint tree under `checkpoints/expXX_kY_sZ/`.

---

## 5. Logs

| File | Content |
| :--- | :--- |
| `outputs/sweep.log` | Full cumulative stdout/stderr of the sweep (truncated at each launch) |
| `outputs/expXX_kY_sZ.log` | Per-run stdout/stderr |
| `outputs/failures.log` | Append-only failure ledger: timestamp, experiment, exit code, log path |

`outputs/failures.log` is never truncated — it is the audit trail across re-runs.

---

## 6. Failure handling

The runner never aborts the matrix on a single bad run.

1. Read `outputs/failures.log` to identify the failed experiment and its exit code.
2. Read the per-run log named in that entry for the traceback.
3. Fix, then re-run only what is missing:

   ```bash
   bash scripts/run_sweep.sh --registers 1 --seeds 1337
   ```

   or replay everything that has no `metrics.json` yet:

   ```bash
   bash scripts/run_sweep.sh --skip-existing
   ```

4. Re-aggregate and confirm: `python src/utils/logger.py --output_dir outputs/ --strict`.

Common causes:

| Symptom | Cause | Action |
| :--- | :--- | :--- |
| `CUDA out of memory` | MIG slice shared with a peer group | Re-run the single experiment, or lower `data.batch_size` in the arm config |
| `Dataloader worker killed` | Shared-memory limit (Kaggle, Docker) | `--extra-args "--num_workers 2"` |
| Run exits `0`, artifacts incomplete | Disk full or an interrupted write | Free space, delete the run directory, re-run that experiment |
| Aggregator reports `Missing runs` | Not every run finished | Re-run the listed experiments before generating figures |

---

## 7. Aggregation

```bash
python src/utils/logger.py --output_dir outputs/
```

Writes `outputs/sweep_summary.json`, keyed `k_0`, `k_1`, `k_4`, `k_8`, each carrying
mean and population standard deviation (`numpy.std`, `ddof=0`) across the seeds that
completed:

* `val_top1_mean` / `val_top1_std` — Top-1 accuracy at the best validation epoch
* `val_loss_mean` / `val_loss_std` — validation loss at that epoch
* `train_loss_mean` / `train_loss_std` — final-epoch training loss
* `gen_gap_mean` / `gen_gap_std` — per-seed `L_val − L_train`, then reduced
* `entropy_mean` / `entropy_std` — mean layer-wise Shannon attention entropy
* `outlier_rate_mean` / `outlier_rate_std` — mean layer-wise patch-norm outlier rate
* `test_top1_mean` / `test_top1_std` — accuracy on the untouched 10 000-image test split
* `layerwise_entropy` / `layerwise_outliers` — per-layer mean/std for the ±1σ error bands in the entropy-versus-layer figure

The `meta` block records how many of the 12 runs were found and names the missing ones.
Arms with no completed run are zero-filled with `num_seeds: 0` so figure and table
generation never crashes on a partially finished sweep.

This file is the sole input to `src/utils/export_latex.py` and `scripts/plot_metrics.py`.

---

## 8. Kaggle parallel execution

`notebooks/kaggle_sweep_runner.ipynb` runs a slice of the matrix on a Kaggle GPU
session. Suggested split: one arm per session (`REGISTERS = [0]`, `[1]`, `[4]`, `[8]`),
three runs each.

Session requirements: *Accelerator → GPU* and *Internet → On* (CIFAR-100 download and
the `timm` install). Keep `NUM_WORKERS = 2` — four dataloader workers deadlock on
Kaggle's shared-memory limit.

Each session exports `sweep_k<arms>.zip` containing only its `outputs/expXX_kY_sZ/`
directories. Because the directory names are globally unique, the slices unpack into a
single tree with no collisions; aggregate once over the merged tree with `--strict` to
confirm all 12 runs are present.

---

## 9. Pre-handover checklist

- [ ] `python scripts/preflight_check.py` exits `0`
- [ ] `pytest` passes (`tests/test_sweep_contract.py` verifies the runner drives `scripts/train.py` per the matrix)
- [ ] `bash scripts/run_sweep.sh --epochs 1` completes with `Failed: 0`
- [ ] Twelve `outputs/expXX_kY_sZ/` directories, each with all four contract artifacts
- [ ] `outputs/failures.log` is empty or every entry has been re-run successfully
- [ ] `python src/utils/logger.py --output_dir outputs/ --strict` exits `0`
- [ ] `outputs/sweep_summary.json` handed to the analysis lead
