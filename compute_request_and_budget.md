# 🖥️ GPU Compute Resource Request & Training Budget Documentation

**Course:** DLE-AI-202 (Deep Learning), Cohort I 2026  
**Track:** Track 1 — Pure Research  
**Project Title:** Do Register Tokens Regularize Vision Transformers Under Data Scarcity?  
**Team Members:** Shahin, Gulnisa, Narmina, Emil, Rufet  
**Cluster Setup:** 12 Groups sharing 2x NVIDIA A100 (40 GB) -> 4x MIG Instances (20 GB VRAM each, 3 Groups per 20 GB Slice)  
**Date:** August 31, 2026  
**Target Submission:** Monday, 7 September 2026, 23:59  

---

## 🔬 1. Research Setup & Workload Sizing

Our study investigates whether learnable **register tokens** (K in {0, 1, 4, 8}) act as an inductive regularizer against severe overfitting when a compact Vision Transformer is trained under extreme data starvation.

### Controlled Independent Variables:
1. **Model Backbone:** ViT-Tiny (timm: `vit_tiny_patch16_224`), 12 transformer layers, 3 attention heads, embedding dimension d = 192, **5.71 Million parameters**.
2. **Register Token Allocations (4 treatment arms):**
   * Arm 1: K = 0 (Control Baseline — register-free ViT-Tiny)
   * Arm 2: K = 1 (Minimal register injection)
   * Arm 3: K = 4 (Standard register allocation)
   * Arm 4: K = 8 (Capacity dilution stress test)
3. **Statistical Confidence (3 independent random seeds per arm):**
   * Seed 42, Seed 1337, Seed 3407
4. **Total Experiments:** 4 treatment arms * 3 seeds = **12 full training runs**.

### Dataset & Low-Resource Sizing:
* **Dataset:** CIFAR-100 stratified low-data subset.
* **Training Set:** Exactly 100 images per class across 100 classes = **10,000 images total**.
* **Internal Split:** 9,000 training images (90/class) and 1,000 validation images (10/class).
* **Test Set:** Standard 10,000 test images (evaluated once at the end of training).

---

## ⏱️ 2. Step-by-Step Training Time Calculations on Shared A100 MIG (20 GB)

The A100 MIG instance (MIG 3g.20gb profile) delivers high-speed throughput with 42 SMs and 168 Ampere Tensor Cores with Automatic Mixed Precision (AMP).

### A. Per-Run Step & Epoch Calculations
* **Batch Size:** 64 images
* **Training Steps per Epoch:** 9,000 images / 64 = **141 steps per epoch**
* **Validation Steps per Epoch:** 1,000 images / 64 = **16 steps per epoch**
* **Total Epochs per Run:** 50 epochs
* **Total Training Steps per Run:** 141 steps * 50 epochs = **7,050 optimizer steps**

### B. Throughput & Runtime per Epoch
* Forward + Backward Throughput on A100 MIG with AMP: **~1,250 images / second**
* Training Phase per Epoch: 9,000 images / 1,250 img/s = **7.20 seconds**
* Validation Phase + Attention Entropy Extraction per Epoch: 1,000 images / 1,250 img/s = **0.80 seconds**
* Metric Logging & Checkpoint IO Overhead: **~0.40 seconds**
* **Total Time per Epoch:** 7.20 + 0.80 + 0.40 = **8.40 seconds**

### C. Runtime per Single Experiment (50 Epochs)
* 50 epochs * 8.40 seconds = 420 seconds = **7.0 minutes**
* Adding safety margin for checkpoint saving and attention matrix serialization:
* **Estimated Runtime per Run = 8.0 minutes (0.133 GPU Hours)**

### D. Full 12-Run Matrix Calculation on A100 MIG (20 GB)

| Experiment ID | Configuration | Register Count (K) | Random Seed | Epochs | Estimated Time on MIG Slice |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Baseline (Control) | K = 0 | Seed 42 | 50 | 8.0 min (0.13 h) |
| **EXP-02** | Baseline (Control) | K = 0 | Seed 1337 | 50 | 8.0 min (0.13 h) |
| **EXP-03** | Baseline (Control) | K = 0 | Seed 3407 | 50 | 8.0 min (0.13 h) |
| **EXP-04** | Register Arm 1 | K = 1 | Seed 42 | 50 | 8.0 min (0.13 h) |
| **EXP-05** | Register Arm 1 | K = 1 | Seed 1337 | 50 | 8.0 min (0.13 h) |
| **EXP-06** | Register Arm 1 | K = 1 | Seed 3407 | 50 | 8.0 min (0.13 h) |
| **EXP-07** | Register Arm 2 | K = 4 | Seed 42 | 50 | 8.0 min (0.13 h) |
| **EXP-08** | Register Arm 2 | K = 4 | Seed 1337 | 50 | 8.0 min (0.13 h) |
| **EXP-09** | Register Arm 2 | K = 4 | Seed 3407 | 50 | 8.0 min (0.13 h) |
| **EXP-10** | Register Arm 3 | K = 8 | Seed 42 | 50 | 8.0 min (0.13 h) |
| **EXP-11** | Register Arm 3 | K = 8 | Seed 1337 | 50 | 8.0 min (0.13 h) |
| **EXP-12** | Register Arm 3 | K = 8 | Seed 3407 | 50 | 8.0 min (0.13 h) |
| **TOTALS** | **12 Complete Runs** | **4 Treatment Arms** | **3 Seeds Each** | **600 Total Epochs** | **96 min (1.60 GPU Hours)** |

---

## 📊 3. Group Time Budget & Fair-Share Breakdown

In a 48-hour scheduled window, each 20 GB MIG slice has 48 hours shared across 3 groups (**16.0 GPU Hours fair-share per group**):

```text
Our Team's Compute Footprint within Group Allocation (16.0h Fair-Share):
├── Core 12-Run Sweeps (12 runs * 8.0 min)     : 1.60 GPU Hours (10.0% of fair-share)
├── Pilot Sanity & Pipeline Validation Passes : 0.40 GPU Hours  (2.5% of fair-share)
├── Full Evaluation, Heatmaps & Profiling     : 1.00 GPU Hours  (6.25% of fair-share)
├── Contingency Safety Buffer                 : 1.00 GPU Hours  (6.25% of fair-share)
└── Unused Headroom Left for Peer Groups      : 12.00 GPU Hours (75.0% FREE for Peers)
```

Our team only requires **4.0 consecutive hours**, leaving **12.0 hours** of our fair share (plus the other 32 hours of the slice) completely open for the other 2 groups sharing our instance.

---

## 💾 4. Peak Memory (VRAM) & Storage Safety Verification

### A. VRAM Consumption Breakdown on A100 MIG (20 GB)

| Memory Component | Calculation / Formula | VRAM Footprint |
| :--- | :--- | :--- |
| **Model Parameters** | 5.71M parameters in FP16 (2 bytes/param) | ~11.4 MB |
| **Gradients** | 5.71M parameters in FP16 | ~11.4 MB |
| **Optimizer States (AdamW)** | FP32 master weights (4 bytes) + Momentum (4 bytes) + Variance (4 bytes) | ~68.5 MB |
| **Batch Activations (Batch 64)** | 64 batch * 205 sequence tokens * 192 dim * 12 layers * 2 bytes (with AMP) | ~1,850 MB (~1.85 GB) |
| **Attention Hooks Matrix Storage** | 64 batch * 12 layers * 3 heads * 205 * 205 * 2 bytes | ~193 MB |
| **PyTorch & CUDA Runtime Context** | A100 CUDA driver, cuDNN kernels, workspace memory | ~750 MB |
| **TOTAL PEAK VRAM OCCUPANCY** | Sum of all runtime components | **~2.88 GB** |
| **DEDICATED MIG INSTANCE VRAM** | 20 GB HBM2 memory slice | **20.00 GB** |
| **FREE VRAM HEADROOM** | Available buffer | **17.12 GB (85.6% Free)** |

> Peak memory is under 3.0 GB, utilizing only 14.4% of the 20 GB allocation.

---

### B. Disk & Storage Budget

| Storage Item | Description & Quantity | Disk Footprint |
| :--- | :--- | :--- |
| **CIFAR-100 Dataset** | Downloaded raw binary archives and unpacked cache | ~170 MB |
| **Model Checkpoints** | 12 runs * 2 checkpoints (best_acc.pth, last.pth) * ~23 MB each | ~552 MB |
| **Metrics & Log Files** | JSON logs, step histories, CSV summaries across 12 runs | ~40 MB |
| **Attention Heatmaps & Plots** | Vector PDF plots and PNG image overlays | ~80 MB |
| **Codebase & Virtual Env** | Project scripts, configs, and Python dependencies | ~1,200 MB (~1.2 GB) |
| **TOTAL DISK OCCUPANCY** | Sum of all files | **~2.04 GB** (Leaves ~6 GB for peer groups) |

---

## 🗓️ 5. Suggested Time Slot Scheduling

To ensure smooth scheduling across the 12 groups, we propose any **single 4-hour window** on our assigned 20 GB MIG instance between **Tuesday, 1 September and Thursday, 3 September 2026**:

```text
Proposed 4-Hour Time Slot (Example: 10:00 – 14:00):
├── 10:00 – 10:30 (0.5h) : Pilot test & AMP validation (K=0 and K=4 pilot)
├── 10:30 – 12:15 (1.75h): Unattended 12-Run Sweeps execution (`bash scripts/run_sweep.sh`)
├── 12:15 – 13:15 (1.0h) : Attention map generation & entropy curves evaluation
└── 13:15 – 14:00 (0.75h): Final result verification -> Release GPU immediately to peer group
```

---

## 📋 6. Summary for TA Scheduling

1. **Cohort Sharing Alignment:** 12 course groups, 4x 20 GB MIG slices (3 groups per slice).
2. **Exact Quota Requested:** **A single 4.0-hour time slot** on our assigned 20 GB MIG instance.
3. **Fairness Guarantee:** Uses only **25%** of our group's 16-hour fair-share window (leaves 75% for our 2 peer groups).
4. **Execution Guarantee:** Fully automated batch execution (`bash scripts/run_sweep.sh`) with zero interactive supervisor overhead.
5. **Memory Safety:** Sub-3.0 GB VRAM requirement ensures zero OOM risk.
