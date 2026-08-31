# 📦 Team Task Package: Gulnisa — Data Engineering & Low-Data Pipeline

**Assignee:** Gulnisa  
**Role:** Lead Data Engineer  
**Target Code Files:**
* `src/data/cifar100_subset.py`
* `src/data/__init__.py`  
**Target Delivery:** Tuesday, 1 September 2026  
**Cross-References:** [Team Task Division](file:///home/shahin/aiac-res/team_task_division.md) | [Architecture Note](file:///mnt/c/Vaults/aiac-res/Atlas/ViT%20Registers%20Architecture.md)  

---

## 🎯 1. Mission & Scientific Context

Our research investigates how Vision Transformers (ViTs) perform under severe data scarcity. While standard CIFAR-100 contains 50,000 training images (500 images/class), our experimental design creates an extreme low-data regime with **exactly 100 images per class (10,000 images total)**.

Your mission is to build an exact, deterministic, stratified data pipeline that guarantees:
1. **Zero Data Leakage:** Training, validation, and test subsets must remain strictly disjoint across all random seeds.
2. **Perfect Class Balance:** Exactly 90 training samples and 10 validation samples per class across all 100 classes.
3. **Reproducibility:** Seeded sampling (`42`, `1337`, `3407`) ensures 100% reproducible experimental runs.
4. **Resolution Compatibility:** Upsampling and preprocessing $32 \times 32$ CIFAR images to $224 \times 224$ with bicubic interpolation and standard CIFAR-100 normalization for Vision Transformers.

---

## 📐 2. Mathematical & Algorithmic Specification

### A. Stratified Partitioning Logic
Let the full training set of CIFAR-100 be denoted as:
$$\mathcal{D}_{\text{train\_full}} = \{(x_i, y_i)\}_{i=1}^{50000}, \quad y_i \in \{0, 1, \dots, 99\}$$

For each class $c \in \{0, \dots, 99\}$:
1. Extract all indices: $\mathcal{I}_c = \{i \mid y_i = c\}$. Note that $|\mathcal{I}_c| = 500$.
2. Seed the pseudorandom generator: $\text{RNG} = \text{np.random.RandomState}(\text{seed})$.
3. Sample without replacement: $\mathcal{S}_c = \text{RNG.choice}(\mathcal{I}_c, \text{size}=100, \text{replace}=\text{False})$.
4. Partition $\mathcal{S}_c$ into:
   * **Training Indices:** $\mathcal{I}_{c, \text{train}} = \mathcal{S}_c[:90]$ ($90$ samples/class $\times 100 = \mathbf{9{,}000\text{ images}}$).
   * **Validation Indices:** $\mathcal{I}_{c, \text{val}} = \mathcal{S}_c[90:]$ ($10$ samples/class $\times 100 = \mathbf{1{,}000\text{ images}}$).
5. The full evaluation test set uses the standard CIFAR-100 test split ($|\mathcal{D}_{\text{test}}| = \mathbf{10{,}000\text{ images}}$).

### B. Normalization Statistics
Standard CIFAR-100 channel mean and standard deviation:
* **Mean:** $\mu = [0.5071, 0.4867, 0.4408]$
* **Std:** $\sigma = [0.2675, 0.2565, 0.2761]$

---

## 💻 3. Implementation Blueprint (`src/data/cifar100_subset.py`)

### Required Function Signatures:

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from typing import Tuple, Optional

# 1. Transform Definition
def get_cifar100_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform).
    Train Transform:
      - RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC)
      - RandomHorizontalFlip(p=0.5)
      - AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
      - ToTensor()
      - Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    Eval Transform:
      - Resize(224, interpolation=InterpolationMode.BICUBIC)
      - CenterCrop(224)
      - ToTensor()
      - Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    """

# 2. Stratified Subsetting
def build_stratified_subsets(
    dataset: torchvision.datasets.CIFAR100,
    samples_per_class: int = 100,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Extracts stratified train and val Subsets with exact class balancing.
    """

# 3. Main Loader Entrypoint
def get_cifar100_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    samples_per_class: int = 100,
    train_ratio: float = 0.9,
    seed: int = 42,
    num_workers: int = 4,
    image_size: int = 224,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    Pin memory enabled if torch.cuda.is_available().
    """
```

---

## ⚠️ 4. Potential Pitfalls & How to Prevent Them

1. **Transform Overwrite Bug:** `torchvision.datasets.CIFAR100` applies transforms at the dataset level. If you slice with `Subset`, applying transforms naively can cause validation data to receive random augmentations.
   * *Fix:* Use a `TransformDataset` wrapper class that wraps `Subset` and applies specific transforms per split.
2. **Data Leakage Across Seeds:** Different seeds (`42`, `1337`, `3407`) must generate distinct stratified samples, but within each seed, train and val splits must never overlap.
   * *Fix:* Verify `set(train_indices).intersection(set(val_indices)) == set()`.
3. **Bicubic Interpolation Requirement:** Bilinear interpolation on $32 \times 32 \to 224 \times 224$ causes blurry patch boundaries for Vision Transformers. Always use `InterpolationMode.BICUBIC`.
4. **Worker Deadlocks on Multi-Processing:** Set `num_workers=4` for Linux/A100 and `num_workers=2` for Kaggle/Windows to prevent shared-memory deadlocks.

---

## ✅ 5. Definition of Done & Verification Test

Your deliverable is complete when running the standalone test passes with zero errors:

```bash
python src/data/cifar100_subset.py
```

### Expected Console Output:
```text
============================================================
Testing Stratified CIFAR-100 Low-Data Pipeline
============================================================
Train batches: 141 (9,000 samples)
Val batches:   16 (1,000 samples)
Test batches:  157 (10,000 samples)
Sample Batch Image Tensor Shape: torch.Size([64, 3, 224, 224])
Sample Batch Label Tensor Shape: torch.Size([64])
Image Tensor Min: -1.9821, Max: 2.1245
============================================================
Data Loader Verification Passed Successfully!
```
