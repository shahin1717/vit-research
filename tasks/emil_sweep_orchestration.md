# 📦 Team Task Package: Emil — Sweep Orchestration & Operations Lead

**Assignee:** Emil  
**Role:** Automation & Experiment Operations Lead  
**Target Code Files:**
* `scripts/run_sweep.sh`
* `configs/baseline_k0.yaml`, `configs/vit_tiny_k1.yaml`, `configs/vit_tiny_k4.yaml`, `configs/vit_tiny_k8.yaml`, `configs/sweep_config.yaml`
* `src/utils/logger.py`
* `notebooks/kaggle_sweep_runner.ipynb`  
**Target Delivery:** Wednesday, 2 September 2026  
**Cross-References:** [Team Task Division](file:///home/shahin/aiac-res/team_task_division.md) | [Compute Budget](file:///home/shahin/aiac-res/compute_request_and_budget.md) | [Roadmap](file:///home/shahin/aiac-res/roadmap.md)  

---

## 🎯 1. Mission & Scientific Context

To ensure the statistical validity of our research findings, every treatment arm must be replicated across **3 independent random seeds (`42`, `1337`, `3407`)**. This produces a total of **12 complete training runs ($4 \text{ arms} \times 3 \text{ seeds}$)**.

Your mission is to automate the execution, monitoring, and result aggregation of the entire experimental matrix:
1. **Config Templates (`configs/*.yaml`):** Structured, reproducible YAML configuration files defining model hyperparameters, optimizer settings, dataset paths, and seed lists.
2. **Automated Sweep Runner (`scripts/run_sweep.sh`):** A robust, unattended bash runner that executes all 12 experiments sequentially on the assigned A100 MIG instance with zero manual intervention.
3. **Kaggle Parallel Runner (`notebooks/kaggle_sweep_runner.ipynb`):** A portable notebook allowing team members to execute subsets of the 12 runs on Kaggle GPU instances in parallel.
4. **Metrics Aggregator (`src/utils/logger.py`):** An automated parser that crawls `outputs/`, aggregates JSON logs across seeds, computes $\text{Mean} \pm \text{Std}$ for all metrics, and outputs `outputs/sweep_summary.json`.

---

## 📊 2. The 12-Run Experimental Matrix

| Exp ID | Config File | Model Backbone | Register Count ($K$) | Random Seed | Output Directory |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`EXP-01`** | `configs/baseline_k0.yaml` | `vit_tiny_patch16_224` | 0 (Control) | 42 | `outputs/exp01_k0_s42/` |
| **`EXP-02`** | `configs/baseline_k0.yaml` | `vit_tiny_patch16_224` | 0 (Control) | 1337 | `outputs/exp02_k0_s1337/` |
| **`EXP-03`** | `configs/baseline_k0.yaml` | `vit_tiny_patch16_224` | 0 (Control) | 3407 | `outputs/exp03_k0_s3407/` |
| **`EXP-04`** | `configs/vit_tiny_k1.yaml` | `vit_tiny_patch16_224` | 1 | 42 | `outputs/exp04_k1_s42/` |
| **`EXP-05`** | `configs/vit_tiny_k1.yaml` | `vit_tiny_patch16_224` | 1 | 1337 | `outputs/exp05_k1_s1337/` |
| **`EXP-06`** | `configs/vit_tiny_k1.yaml` | `vit_tiny_patch16_224` | 1 | 3407 | `outputs/exp06_k1_s3407/` |
| **`EXP-07`** | `configs/vit_tiny_k4.yaml` | `vit_tiny_patch16_224` | 4 | 42 | `outputs/exp07_k4_s42/` |
| **`EXP-08`** | `configs/vit_tiny_k4.yaml` | `vit_tiny_patch16_224` | 4 | 1337 | `outputs/exp08_k4_s1337/` |
| **`EXP-09`** | `configs/vit_tiny_k4.yaml` | `vit_tiny_patch16_224` | 4 | 3407 | `outputs/exp09_k4_s3407/` |
| **`EXP-10`** | `configs/vit_tiny_k8.yaml` | `vit_tiny_patch16_224` | 8 | 42 | `outputs/exp10_k8_s42/` |
| **`EXP-11`** | `configs/vit_tiny_k8.yaml` | `vit_tiny_patch16_224` | 8 | 1337 | `outputs/exp11_k8_s1337/` |
| **`EXP-12`** | `configs/vit_tiny_k8.yaml` | `vit_tiny_patch16_224` | 8 | 3407 | `outputs/exp12_k8_s3407/` |

---

## 💻 3. Implementation Blueprint

### 1. Automated Sweep Script (`scripts/run_sweep.sh`)

```bash
#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "🚀 Starting 12-Run ViT Register Ablation Sweep Matrix"
echo "=========================================================="

SEEDS=(42 1337 3407)
REGISTERS=(0 1 4 8)
CONFIGS=("configs/baseline_k0.yaml" "configs/vit_tiny_k1.yaml" "configs/vit_tiny_k4.yaml" "configs/vit_tiny_k8.yaml")

EXP_IDX=1

for idx in "${!REGISTERS[@]}"; do
  K="${REGISTERS[$idx]}"
  CONFIG="${CONFIGS[$idx]}"
  
  for SEED in "${SEEDS[@]}"; do
    EXP_ID=$(printf "exp%02d_k%d_s%d" $EXP_IDX $K $SEED)
    echo ""
    echo "----------------------------------------------------------"
    echo "▶️ [${EXP_IDX}/12] Executing ${EXP_ID} (K=${K}, Seed=${SEED})..."
    echo "----------------------------------------------------------"
    
    python scripts/train.py \
      --config "$CONFIG" \
      --num_registers "$K" \
      --seed "$SEED" \
      --output_dir "outputs/${EXP_ID}" \
      2>&1 | tee "outputs/${EXP_ID}.log"
      
    EXP_IDX=$((EXP_IDX + 1))
    
    # Clean GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
  done
done

echo ""
echo "=========================================================="
echo "✅ All 12 Experiments Completed! Aggregating metrics..."
python src/utils/logger.py --output_dir outputs/
echo "=========================================================="
```

### 2. Metrics Aggregator (`src/utils/logger.py`)

```python
import os
import json
import numpy as np
from typing import Dict, Any

def aggregate_sweep_results(outputs_dir: str = "outputs/") -> Dict[str, Any]:
    """
    Crawls all experiment subdirectories in outputs/, parses metrics.json,
    computes mean and std across seeds for K in {0, 1, 4, 8},
    and writes outputs/sweep_summary.json.
    """
    summary = {}
    for k in [0, 1, 4, 8]:
        accs, val_losses, train_losses, entropies = [], [], [], []
        for seed in [42, 1337, 3407]:
            for folder in os.listdir(outputs_dir):
                if f"k{k}_s{seed}" in folder:
                    metrics_path = os.path.join(outputs_dir, folder, "metrics.json")
                    if os.path.exists(metrics_path):
                        with open(metrics_path, "r") as f:
                            data = json.load(f)
                            accs.append(data.get("best_val_top1", 0.0))
                            val_losses.append(data.get("best_val_loss", 0.0))
                            train_losses.append(data.get("final_train_loss", 0.0))
                            entropies.append(data.get("mean_layerwise_entropy", 0.0))
        
        summary[f"k_{k}"] = {
            "val_top1_mean": float(np.mean(accs)) if accs else 0.0,
            "val_top1_std": float(np.std(accs)) if accs else 0.0,
            "val_loss_mean": float(np.mean(val_losses)) if val_losses else 0.0,
            "val_loss_std": float(np.std(val_losses)) if val_losses else 0.0,
            "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
            "entropy_std": float(np.std(entropies)) if entropies else 0.0,
            "gen_gap_mean": float(np.mean(np.array(val_losses) - np.array(train_losses))) if val_losses else 0.0,
        }
    
    summary_path = os.path.join(outputs_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregated summary written to {summary_path}")
    return summary
```

---

## ⚠️ 4. Operations & Hardware Safety Directives

1. **Subprocess Failure Resilience:** If one experiment encounters a bug, do not crash the entire 12-run runner. Use error trapping in bash or python to log failures to `outputs/failures.log` and continue.
2. **GPU Memory Leakage:** Python subshells launched sequentially automatically free VRAM upon process exit, but always include `torch.cuda.empty_cache()` to guarantee zero memory fragmentation.
3. **Structured Metrics Standard:** Every run directory `outputs/expXX_kY_sZ/` must output exactly:
   * `metrics.json` (Structured dictionary of Top-1 acc, loss, entropy)
   * `best_model.pth` (Model weights at best validation epoch)
   * `last_model.pth` (Final epoch checkpoint)
   * `train_history.csv` (Per-epoch training and validation loss curves)

---

## ✅ 5. Definition of Done & Verification Test

Your deliverable is complete when:
1. `scripts/run_sweep.sh` has executable permissions (`chmod +x scripts/run_sweep.sh`).
2. A dry-run with `--epochs 1` completes for all 12 experiments without error.
3. Running `python src/utils/logger.py` correctly parses and generates `outputs/sweep_summary.json`.
