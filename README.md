# 🔬 Do Register Tokens Regularize Vision Transformers Under Data Scarcity?

[![Course](https://img.shields.io/badge/Course-DLE--AI--202-blue.svg)](https://github.com/)
[![Track](https://img.shields.io/badge/Track-1%20Pure%20Research-green.svg)](https://github.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-0.9+-orange.svg)](https://github.com/huggingface/pytorch-image-models)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)

Official experimental codebase and reproduction pipeline for the research project:
> **"Do Register Tokens Regularize Vision Transformers Under Data Scarcity?"**  
> *Course: DLE-AI-202: Deep Learning (Cohort I 2026)*  
> *Track: 1 — Pure Research*  
> *Obsidian Knowledge Vault: `C:\Vaults\aiac-res` (`/mnt/c/Vaults/aiac-res`)*

---

## 🧭 Project Motivation & Research Hypothesis

Darcet et al. (2023), *"Vision Transformers Need Registers,"* revealed that standard ViTs consistently convert uniform background patch tokens into high-norm **attention sinks** to store non-semantic context. While learnable register tokens successfully eliminate these artifacts in foundation-scale models (DINOv2, OpenCLIP), their role under severe **data scarcity** remains unexplored.

In this project, we investigate:
1. **Structural Regularization:** Do register tokens act as an inductive regularizer, reducing the generalization gap $\Delta(\mathcal{L}_{\text{train}}, \mathcal{L}_{\text{val}})$ in data-starved ViTs?
2. **Capacity Dilution Hypothesis:** In small embedding dimensions ($d=192$ in ViT-Tiny), does allocating token capacity to registers dilute representation power when $K$ becomes large ($K \ge 8$)?
3. **Quantitative Entropy Diagnostics:** Beyond visual inspection, does register injection systematically prevent layer-wise Shannon attention entropy collapse?

---

## 🗂️ Repository Structure

```text
aiac-res/
├── configs/                  # Experiment configuration files (YAML)
│   ├── baseline_k0.yaml      # Register-free control baseline (K = 0)
│   ├── vit_tiny_k1.yaml      # Treatment arm K = 1
│   ├── vit_tiny_k4.yaml      # Treatment arm K = 4
│   ├── vit_tiny_k8.yaml      # Treatment arm K = 8
│   └── sweep_config.yaml     # Matrix sweep definition across seeds & registers
│
├── src/                      # Core modular Python package
│   ├── __init__.py
│   ├── models/               # Model architectures & register wrappers
│   │   ├── __init__.py
│   │   ├── register_vit.py   # Register token injection into timm ViTs
│   │   └── attention_hook.py # Softmax attention matrix extraction hooks
│   ├── data/                 # Dataset loaders & samplers
│   │   ├── __init__.py
│   │   ├── cifar100_subset.py# Stratified 10k low-data subset generator
│   │   └── transforms.py     # Training & evaluation augmentations
│   ├── metrics/              # Quantitative evaluation functions
│   │   ├── __init__.py
│   │   ├── entropy.py        # Layer-wise Shannon attention entropy H(A_l)
│   │   ├── outliers.py       # Patch-norm outlier rate (mu + 3*sigma)
│   │   └── generalization.py # Train-val loss gap tracker
│   └── utils/                # Utility and helper functions
│       ├── __init__.py
│       ├── seed.py           # Multi-seed determinism (torch, cuda, numpy)
│       ├── logger.py         # Structured JSON & console metric logger
│       └── checkpoint.py     # Checkpoint saving and resumption
│
├── scripts/                  # Executable pipeline entry points
│   ├── train.py              # Main training script (with AMP & metrics)
│   ├── eval.py               # Checkpoint evaluation & attention metrics
│   ├── run_sweep.sh          # Orchestrates K in {0, 1, 4, 8} x 3 seeds
│   └── visualize_attention.py# Generates side-by-side attention heatmaps
│
├── notebooks/                # Jupyter exploration & visualization
│   └── 01_attention_entropy_eda.ipynb # Interactive inspection of attention maps
│
├── data/                     # Local dataset storage (git-ignored, cached)
├── checkpoints/              # Model weights and saved checkpoints (git-ignored)
├── outputs/                  # Training logs, CSV metrics, and figures (git-ignored)
├── requirements.txt          # Pinned dependency specifications
├── .gitignore                # Comprehensive exclusions for data & weights
├── aiac-res.code-workspace   # Multi-root workspace linking codebase & Obsidian vault
├── dl_final_project_task_breakdown.md # Course task breakdown reference
├── vit_registers_proposal.md # Formal course research proposal
└── README.md                 # This documentation
```

---

## ⚡ Quickstart

### 1. Conda Environment Setup
```bash
conda create -n aiac-res python=3.11 -y
conda activate aiac-res
pip install -r requirements.txt
```

### 2. Run Single Baseline Training ($K=0$)
```bash
python scripts/train.py --config configs/baseline_k0.yaml
```

### 3. Run Full Sweep Matrix ($K \in \{0, 1, 4, 8\} 	imes 3	ext{ Seeds}$)
```bash
bash scripts/run_sweep.sh
```

### 4. Evaluate & Extract Attention Metrics
```bash
python scripts/eval.py --checkpoint checkpoints/baseline_k0/best.pth --config configs/baseline_k0.yaml
```

### 5. Generate Publication Figures
```bash
python scripts/visualize_attention.py --checkpoint-baseline checkpoints/baseline_k0/best.pth --checkpoint-reg checkpoints/vit_tiny_k4/best.pth
```

---

## 📊 Evaluation & Logged Metrics

1. **Top-1 Accuracy:** Standard classification accuracy on stratified validation and test sets.
2. **Generalization Gap:**
   $$\Delta\mathcal{L} = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$$
3. **Layer-wise Shannon Attention Entropy:**
   $$H_i^{(l, h)} = -\sum_{j=1}^S A_{i,j}^{(l, h)} \log\big(A_{i,j}^{(l, h)} + \epsilon\big)$$
4. **Patch Norm Outlier Rate:**
   $$\text{Outliers} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\big(\|x_i^{(l)}\|_2 > \mu_l + 3\sigma_l\big)$$

---

## 🔗 Obsidian Knowledge Base Integration

This project is connected via multi-root workspace (`aiac-res.code-workspace`) to the **Obsidian Vault** located at `C:\Vaults\aiac-res` (`/mnt/c/Vaults/aiac-res`), organized under the **ACE (Atlas, Calendar, Efforts) + AIOS** framework:

* 🏛️ **`Atlas/`**: Theoretical architecture notes, mathematical proofs, and literature references.
* 📅 **`Calendar/`**: Daily experiment logs and progress milestones.
* ⚡ **`Efforts/`**: Active experiment matrices, task checklists, and sweep plans.
* 📄 **`paper/`**: LaTeX source manuscript (`main.tex`, `sections/`, `references.bib`).
* 🎤 **`presentation/`**: Slide decks for final defense (`slides.tex`, `slides.md`).
