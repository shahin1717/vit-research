# 🚧 Integration Blockers & Observations — Sweep Layer

**Raised by:** Emil (Sweep Orchestration & Operations Lead)
**Reviewed & Fixed by:** Shahin (Core Architecture Lead)
**Date:** 4 September 2026
**Status of Emil's own deliverables:** complete and verified (see `docs/sweep_runbook.md`)
**Resolution Status:** ✅ **ALL BLOCKERS & OBSERVATIONS RESOLVED AND VERIFIED**

```
============================================================
🔍 Running Preflight Sanity Check for ViT Registers Sweep
============================================================
Checking core dependencies... OK (PyTorch 2.13.0+cu130, timm 1.0.29)
Checking scripts/train.py imports... OK ('time' and critical modules verified)
Checking dataloader signature compatibility... OK (signature: data_dir, batch_size, samples_per_class, train_ratio, seed)
Checking model instantiation across K in {0, 1, 4, 8}... OK (all arms K in {0,1,4,8} validated)
Checking attention hooks & metrics instrumentation... OK (entropy & 3-sigma outlier hooks verified)
Checking YAML configs... OK (all 4 treatment arm configs valid)
Checking directory write permissions... OK (outputs/ and checkpoints/ writable)
============================================================
✅ No blocking issues. The sweep is safe to launch.
============================================================
```

All 29 pytest unit & integration tests pass cleanly (`29 passed in 9.31s`).

---

## 🟢 RESOLVED: BLOCKER-1 — `scripts/train.py` crashes with `NameError: name 'time' is not defined`
* **Resolution:** Added `import time` and `import shutil` to `scripts/train.py:25`. Verified via preflight AST import checker and full execution pass.

---

## 🟢 RESOLVED: BLOCKER-2 — `scripts/train.py` passes argument dataloader does not accept
* **Resolution:** Translated `val_split` to `train_ratio = 1.0 - float(cfg["data"].get("val_split", 0.1))` and passed `image_size=img_size` to `get_cifar100_loaders(...)`.

---

## 🟢 RESOLVED: OBS-1 — `model.img_size` never reaches data pipeline
* **Resolution:** Forwarded `img_size = cfg["model"].get("img_size", cfg["data"].get("image_size", 224))` to `get_cifar100_loaders(...)` in both `scripts/train.py` and `scripts/eval.py`.

---

## 🟢 RESOLVED: OBS-2 — Attention hooks stay armed for the entire evaluation loop
* **Resolution:** In `evaluate()` (`scripts/train.py` and `scripts/eval.py`), added immediate hook disarming and detachment after batch 0 (`hook_mgr.remove(); hook_mgr.clear(); hook_mgr = None`). Batches 1..N run in clean native mode without memory retention or D2H transfers.

---

## 🟢 RESOLVED: OBS-3 — Artifact naming contract synchronization
* **Resolution:**
  - Standardized CLI output dir handling: when `--output_dir outputs/expXX_kY_sZ` is supplied, files are saved directly into that run directory.
  - Checkpoint dual-naming: saves both `best_model.pth` and `best_model.pt`, and both `last_model.pth` and `latest_checkpoint.pt` to both `checkpoint_dir` and `output_dir`.
  - CSV dual-naming: saves both `summary.csv` and `train_history.csv` to `output_dir`.
  - CLI argument synonym: added `--num_registers` as alias for `--k_registers` in `train.py` and `eval.py`.
  - Added `src/utils/logger.py` for automated multi-seed aggregation into `outputs/sweep_summary.json`.

---

**Owner:** Shahin · **File:** `scripts/train.py` · **Severity:** fatal, 100 % of runs

`time.time()` is used at lines **419, 421, 454, 526** but the `time` module is
never imported. Every run dies at the start of the epoch loop, *after* the
dataset download and model build — so the failure surfaces minutes into each run.

```
Traceback (most recent call last):
  File "scripts/train.py", line 419, in main
    start_time = time.time()
                 ^^^^
NameError: name 'time' is not defined
```

### Fix — add one line to the import block (around line 24)

```diff
 import os
 import random
 import sys
+import time
 from pathlib import Path
```

---

## 🔴 BLOCKER-2 — `scripts/train.py` passes an argument the data loader does not accept

**Owner:** Shahin (call site) / Gulnisa (signature) · **File:** `scripts/train.py:356`
**Severity:** fatal, 100 % of runs

`train.py` calls `get_cifar100_loaders(..., val_split=...)`, but
`src/data/cifar100_subset.py::get_cifar100_loaders` has no `val_split`
parameter — it splits with `train_ratio` instead.

| | |
|---|---|
| Loader signature | `data_dir, batch_size, samples_per_class, train_ratio, seed, num_workers, image_size, download` |
| `train.py` passes | `data_dir, batch_size, num_workers, samples_per_class, `**`val_split`**`, seed, download` |

```
TypeError: get_cifar100_loaders() got an unexpected keyword argument 'val_split'
```

### Fix — translate the config key to the loader's parameter (line ~356)

```diff
     train_loader, val_loader, test_loader = get_cifar100_loaders(
         data_dir=cfg["data"]["data_dir"],
         batch_size=cfg["data"]["batch_size"],
         num_workers=cfg["data"]["num_workers"],
+        image_size=cfg["model"]["img_size"],
         samples_per_class=cfg["data"]["samples_per_class"],
-        val_split=cfg["data"]["val_split"],
+        train_ratio=1.0 - cfg["data"]["val_split"],
         seed=seed,
         download=True,
     )
```

`val_split: 0.1` → `train_ratio = 0.9` → 90 train / 10 val per class → 9 000 / 1 000,
which is exactly what `src/data/cifar100_subset.py`'s own self-test asserts.
The `image_size` line also closes **OBS-1** below.

> **Do not "fix" this by removing `val_split` from the YAML configs.**
> `train.py`'s built-in default dictionary contains `val_split` and the call site
> reads it unconditionally, so no config change can avoid the `TypeError`.
> It has to be fixed at the call site.

### Verification after applying BLOCKER-1 + BLOCKER-2

I applied both fixes to a throw-away copy of `train.py` (never to the repo) and
confirmed the file still compiles and that `scripts/preflight_check.py` then
reports **"No blocking issues. The sweep is safe to launch."**

---

## 🟡 OBS-1 — `model.img_size` never reaches the data pipeline

**Owner:** Shahin · **File:** `scripts/train.py:356`

`train.py` passes `img_size` to `RegisterVisionTransformer` but not to
`get_cifar100_loaders`, which then falls back to its own default of 224.
Harmless today (every config uses 224), but the moment anyone sets
`model.img_size: 128` the model is built for 128 and fed 224 px tensors, and the
run dies inside `patch_embed` with a shape error. The diff in BLOCKER-2 already
includes the one-line fix.

---

## 🟡 OBS-2 — Attention hooks stay armed for the entire evaluation loop

**Owner:** Narmina (hook manager) / Shahin (call site)
**Files:** `src/models/attention_hook.py`, `scripts/train.py::evaluate`

`evaluate()` builds a `ViTAttentionHookManager` before the loop and only *reads*
the captured tensors at `step == 0`, but the hooks remain attached for every
remaining batch. For each of those batches the hook still

1. re-runs the full QKV projection,
2. materialises the `[B, H, S, S]` softmax matrix, and
3. copies it to host RAM.

For ViT-Tiny (H = 3) at `batch_size = 64`, `S = 197` (K = 0):

```
64 × 3 × 197 × 197 × 4 B  ≈ 29.8 MB per layer
                   × 12   ≈ 358 MB per batch, device → host
```

* **Validation** (1 000 images ≈ 16 batches): ≈ 5.6 GB of pointless D2H traffic
  **per epoch**, ×50 epochs ×12 runs.
* **Final test evaluation** (10 000 images ≈ 157 batches): ≈ 56 GB in one pass.

Nothing breaks — the dictionary is simply overwritten each batch — but this is
easily the single largest avoidable cost in the whole sweep, and it inflates my
compute-budget estimate materially.

### Suggested fix (owner's call)

Attach the hooks only for the batch that is actually measured:

```python
# instead of wrapping the whole loop
for step, (images, targets) in enumerate(val_loader):
    if step == 0 and extract_attention_metrics:
        with ViTAttentionHookManager(model) as hook_mgr:
            outputs = model(images)
            layerwise_entropy  = compute_layerwise_entropy(hook_mgr.attention_maps)
            layerwise_outliers = compute_layerwise_outlier_rate(
                hook_mgr.intermediate_activations, k_registers=k_registers)
    ...
```

Alternatively add an `enabled` flag to the manager that the hook body checks.

---

## 🟡 OBS-3 — Artifact naming differs from the agreed contract

**Owner:** Shahin · **File:** `scripts/train.py`

The task package fixes the per-run contract as
`metrics.json`, `best_model.pth`, `last_model.pth`, `train_history.csv`,
all inside `outputs/expXX_kY_sZ/`. `train.py` currently writes:

| train.py writes | where | contract expects |
|---|---|---|
| `metrics.json` | `outputs/<output_dir>/rViT_K{K}_seed{S}/` | `outputs/expXX_kY_sZ/metrics.json` |
| `summary.csv` | `outputs/<output_dir>/rViT_K{K}_seed{S}/` | `train_history.csv` |
| `best_model.pt` | `checkpoints/<ckpt_dir>/rViT_K{K}_seed{S}/` | `best_model.pth` |
| `latest_checkpoint.pt` | same | `last_model.pth` |

**No change is required from Shahin.** `run_sweep.sh` normalizes all four names
and flattens the extra `rViT_K{K}_seed{S}/` level after every run, while leaving
the original checkpoint tree intact for `scripts/eval.py` and
`scripts/visualize_attention.py`, which reference `checkpoints/...`. This entry
exists so nobody is surprised by the two layouts coexisting.

---

## ✅ Checked and found correct

* `scripts/eval.py` — all `get_cifar100_loaders()` keywords match the signature; all modules imported.
* `src/data/cifar100_subset.py` — stratified split is deterministic and leak-free for seeds 42 / 1337 / 3407.
* `scripts/train.py` CLI — exposes `--config`, `--k_registers`, `--seed`, `--epochs`, `--output_dir`, `--checkpoint_dir`, so `run_sweep.sh` drives it without any wrapper.
* `--amp` / `--pretrained` (`action="store_true"` with `default=None`) — the
  `is not None` guards do correctly distinguish "flag absent" from "flag passed".
* `src/models/register_vit.py` — register tokens are inserted after `_pos_embed`
  and the classifier reads index 0, matching the sequence
  `[CLS ‖ R₁..R_K ‖ patches]` that `compute_patch_outlier_rate(k_registers=K)` assumes.
