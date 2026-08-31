# 🖥️ GPU Compute Resource Request & Training Budget Documentation

**Course:** DLE-AI-202 (Deep Learning), Cohort I 2026  
**Track:** Track 1 — Pure Research  
**Project Title:** Do Register Tokens Regularize Vision Transformers Under Data Scarcity?  
**Team Members:** Shahin, Gulnisa, Narmina, Emil, Rufet  
**Cluster Architecture:** 2x NVIDIA A100 (40 GB) partitioned into 4x MIG Instances (20 GB VRAM each, MIG 3g.20gb profile)  
**Date:** August 31, 2026  
**Target Submission:** Monday, 7 September 2026, 23:59  

---

## 📌 1. Executive Summary & Resource Request (For TA)

This document details the compute allocation request, training benchmarks, and execution plan for our Deep Learning research project tailored to the class cluster: **2x NVIDIA A100 (40 GB) partitioned into 4x MIG slices of 20 GB VRAM each (MIG 3g.20gb)**.

### Quick Resource Summary Table

| Parameter | Cluster Hardware Specification | Project Consumption | Headroom / Margin | Compliance Status |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster Hardware** | **2x NVIDIA A100 (40 GB)** | 4x MIG Instances (MIG 3g.20gb profile) | 2 slices per physical A100 | Fully Supported |
| **Assigned Instance** | **1x or 2x A100 MIG Instance (20 GB VRAM)** | Peak VRAM: **~2.88 GB – 3.20 GB** | **16.80 GB Free (84.0% Free VRAM)** | Zero Risk of OOM |
| **Total Requested Compute** | **~2.0 Days (48 Hours allocated window)** | **4.0 MIG GPU Hours total** (12 runs + buffers) | **44.0 Hours Buffer (91.7% under ceiling)** | Highly Efficient |
| **Wall-Clock Time (Single Instance)** | Sequential sweep execution | **96 minutes total** (12 runs * 8.0 min) | Finishes in under 2 hours | Ultra Fast |
| **Wall-Clock Time (2x Parallel MIG)** | Parallel execution (2 instances) | **48 minutes total** (6 runs per instance) | Finishes in under 1 hour | Maximum Efficiency |
| **Storage / NVMe Scratch** | **25.0 GB Quota** | **~2.04 GB** (Data cache + 24 weights + logs) | **22.96 GB Free Storage** | Fully Compliant |
| **Precision Acceleration** | Ampere 3rd Gen Tensor Cores | Automatic Mixed Precision (AMP: FP16 / BF16) | Peak Tensor Throughput | Fully Enabled |
| **Execution Mode** | Single batch runner (`bash scripts/run_sweep.sh`) | Fully non-interactive | Unattended automated runs | Zero TA Overhead |

---

## 🔬 2. Research Setup & Workload Sizing

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

## ⏱️ 3. Step-by-Step Training Time Calculations on A100 MIG (20 GB)

Each A100 MIG instance (MIG 3g.20gb profile) provides 3 GPU slices (42 Streaming Multiprocessors, 168 Tensor Cores, 20 GB HBM2 memory with ~800 GB/s bandwidth).

### A. Per-Run Step & Epoch Calculations
* **Batch Size:** 64 images
* **Training Steps per Epoch:** 9,000 images / 64 = **141 steps per epoch**
* **Validation Steps per Epoch:** 1,000 images / 64 = **16 steps per epoch**
* **Total Epochs per Run:** 50 epochs
* **Total Training Steps per Run:** 141 steps * 50 epochs = **7,050 optimizer steps**

### B. Throughput & Runtime per Epoch on A100 MIG (20 GB)
* Forward + Backward Throughput (ViT-Tiny on 224x224 with AMP): **~1,250 images / second**
* Training Phase per Epoch: 9,000 images / 1,250 img/s = **7.20 seconds**
* Validation Phase + Attention Entropy Extraction per Epoch: 1,000 images / 1,250 img/s = **0.80 seconds**
* Metric Logging & Checkpoint IO Overhead: **~0.40 seconds**
* **Total Time per Epoch:** 7.20 + 0.80 + 0.40 = **8.40 seconds**

### C. Runtime per Single Experiment (50 Epochs)
* 50 epochs * 8.40 seconds = 420 seconds = **7.0 minutes**
* Adding safety margin for checkpoint saving and attention matrix serialization:
* **Estimated Runtime per Run = 8.0 minutes (0.133 GPU Hours)**

### D. Full 12-Run Matrix Calculation on A100 MIG (20 GB)

| Experiment ID | Configuration | Register Count (K) | Random Seed | Epochs | Estimated Time (1 Instance) |
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

## 📊 4. Total Quota Allocation (4.0 MIG GPU Hours)

```text
Total Requested A100 MIG Quota: 4.0 GPU Hours
├── Core Experimental Sweeps (12 runs * 8.0 min) : 1.60 GPU Hours (40.0%)
├── Pilot Sanity & Pipeline Validation Passes   : 0.40 GPU Hours (10.0%)
├── Full Evaluation, Heatmaps & Profiling       : 1.00 GPU Hours (25.0%)
└── Hardware Contingency & Re-run Safety Buffer : 1.00 GPU Hours (25.0%)
```

### Scheduling Options for TA:
* **Option A (Single 20 GB MIG Instance):** Allocate 1 instance for **4.0 consecutive hours**. Sweeps finish in 96 minutes; remaining time covers evaluations and buffer.
* **Option B (Two 20 GB MIG Instances in Parallel):** Allocate 2 instances for **2.0 consecutive hours**. Runs EXP-01 through EXP-06 on Instance 1 and EXP-07 through EXP-12 on Instance 2. Wall-clock training finishes in **48 minutes**!

---

## 💾 5. Peak Memory (VRAM) & Storage Safety Verification

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

> **Conclusion on VRAM:** Peak VRAM is **under 3.0 GB**, which uses only **14.4%** of the 20 GB MIG slice, guaranteeing 100% immunity to Out-Of-Memory (OOM) failures.

---

### B. Disk & Storage Budget

| Storage Item | Description & Quantity | Disk Footprint |
| :--- | :--- | :--- |
| **CIFAR-100 Dataset** | Downloaded raw binary archives and unpacked cache | ~170 MB |
| **Model Checkpoints** | 12 runs * 2 checkpoints (best_acc.pth, last.pth) * ~23 MB each | ~552 MB |
| **Metrics & Log Files** | JSON logs, step histories, CSV summaries across 12 runs | ~40 MB |
| **Attention Heatmaps & Plots** | Vector PDF plots and PNG image overlays | ~80 MB |
| **Codebase & Virtual Env** | Project scripts, configs, and Python dependencies | ~1,200 MB (~1.2 GB) |
| **TOTAL DISK OCCUPANCY** | Sum of all files | **~2.04 GB** (Well below 25 GB limit) |

---

## 🗓️ 6. Compute Scheduling & Execution Timeline

We request that our quota be scheduled between **Tuesday, 1 September and Thursday, 3 September 2026**:

```text
Execution Milestones on A100 MIG:
├── Slot 1 (10:00 – 10:30) : Pilot run validation (0.4 GPU Hours) -> Verify AMP & dataset loader
├── Slot 2 (10:30 – 12:15) : Core 12-Run Sweeps (1.60 GPU Hours) -> bash scripts/run_sweep.sh
├── Slot 3 (12:15 – 13:15) : Attention heatmap extraction & metrics evaluation (1.00 GPU Hours)
└── Slot 4 (13:15 – 14:15) : Safety buffer / reruns if needed (1.00 GPU Hours)
```

---

## 📋 7. Summary for Course Teaching Assistants (TA)

1. **Exact Hardware Request:** **1x or 2x NVIDIA A100 MIG Instance (20 GB VRAM, MIG 3g.20gb profile)** from the 2x A100 40GB cluster pool.
2. **Exact Quota Needed:** **4.0 MIG GPU Hours** (sweep training finishes in **96 min on 1 instance**, or **48 min on 2 instances**).
3. **VRAM Safety Margin:** Occupies only **~2.88 GB** of the 20 GB allocation (**85.6% free headroom**).
4. **Execution Guarantee:** Fully automated single-command runner (`bash scripts/run_sweep.sh`) with zero interactive supervisor overhead.
5. **Deliverable Integration:** Checkpoint metrics and attention entropy matrices will directly populate the results tables in our paper manuscript (`paper/main.tex`) for final submission on Mon 7 Sep 2026.
