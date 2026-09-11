"""
Stratified low-data CIFAR-100 pipeline.

Builds exact, deterministic, class-balanced subsets of CIFAR-100
across configurable data budget regimes:
  - Primary low-data benchmark: 100 images/class (9,000 train / 1,000 val)
  - Data budget scaling ablation: 300 images/class (27,000 train / 3,000 val)
for the "Do Register Tokens Regularize Vision Transformers Under Data Scarcity?" project.

Author: Gulnisa Abdurahmanli (Lead Data Engineer)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.transforms.functional import InterpolationMode
from typing import Tuple, Optional

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


# ---------------------------------------------------------------------------
# 1. Transforms
# ---------------------------------------------------------------------------
def get_cifar100_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform) for CIFAR-100 upsampled to
    `image_size` x `image_size`, suitable for ViT input.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])

    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# Helper: wraps a Subset with its own transform, avoiding the "transform
# overwrite" bug where train/val share the underlying dataset's transform.
# ---------------------------------------------------------------------------
class TransformDataset(Dataset):
    """Wraps a Subset (or any Dataset) and applies its own transform,
    ignoring whatever transform is set on the underlying dataset."""

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        # Access the underlying PIL image directly, bypassing the base
        # dataset's own `.transform` (which may be None or set to something
        # else) so each split's transform is applied exactly once.
        base_dataset = self.subset.dataset
        real_idx = self.subset.indices[idx]

        # torchvision.datasets.CIFAR100 stores raw numpy arrays in .data
        # and integer labels in .targets. We fetch the raw PIL image via
        # the dataset's own image-decoding path but without its transform,
        # by temporarily reading from .data/.targets directly.
        img = base_dataset.data[real_idx]
        target = base_dataset.targets[real_idx]

        from PIL import Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# ---------------------------------------------------------------------------
# 2. Stratified Subsetting
# ---------------------------------------------------------------------------
def build_stratified_subsets(
    dataset: torchvision.datasets.CIFAR100,
    samples_per_class: int = 100,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Extracts stratified train and val Subsets with exact class balancing.

    For each class c:
      1. Gather all indices with label == c.
      2. Seed RNG = np.random.RandomState(seed).
      3. Sample `samples_per_class` indices without replacement.
      4. Split into train (first `train_ratio` fraction) / val (remainder).

    Guarantees:
      - Exactly samples_per_class * train_ratio train samples / class
      - Exactly samples_per_class * (1 - train_ratio) val samples / class
      - No overlap between train and val indices, for a given seed.
    """
    targets = np.array(dataset.targets)
    num_classes = int(targets.max()) + 1

    n_train_per_class = int(round(samples_per_class * train_ratio))
    n_val_per_class = samples_per_class - n_train_per_class

    train_indices = []
    val_indices = []

    rng = np.random.RandomState(seed)

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        assert len(class_indices) >= samples_per_class, (
            f"Class {c} has only {len(class_indices)} samples, "
            f"need at least {samples_per_class}."
        )

        sampled = rng.choice(class_indices, size=samples_per_class, replace=False)

        train_indices.extend(sampled[:n_train_per_class].tolist())
        val_indices.extend(sampled[n_train_per_class:].tolist())

    # Sanity check: zero data leakage across the split.
    assert set(train_indices).isdisjoint(set(val_indices)), (
        "Data leakage detected: train/val indices overlap."
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


# ---------------------------------------------------------------------------
# 3. Main Loader Entrypoint
# ---------------------------------------------------------------------------
def get_cifar100_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    samples_per_class: int = 100,
    train_ratio: float = 0.9,
    seed: int = 42,
    num_workers: int = 4,
    image_size: int = 224,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds the full stratified low-data pipeline and returns
    (train_loader, val_loader, test_loader).

    - train/val: stratified subset of the CIFAR-100 train split
      (samples_per_class per class, split by train_ratio).
    - test: full, untouched CIFAR-100 test split (10,000 images).
    """
    train_transform, eval_transform = get_cifar100_transforms(image_size)

    # Base datasets loaded WITHOUT a transform - transforms are applied
    # exclusively via TransformDataset, per split.
    full_train = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=download, transform=None
    )
    full_test = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=download, transform=None
    )

    train_subset, val_subset = build_stratified_subsets(
        full_train,
        samples_per_class=samples_per_class,
        train_ratio=train_ratio,
        seed=seed,
    )

    train_dataset = TransformDataset(train_subset, train_transform)
    val_dataset = TransformDataset(val_subset, eval_transform)
    test_dataset = TransformDataset(
        Subset(full_test, list(range(len(full_test)))), eval_transform
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 4. Split and Budget Verification Utilities
# ---------------------------------------------------------------------------
def verify_data_budget_split(
    dataset: torchvision.datasets.CIFAR100,
    samples_per_class: int = 100,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> dict:
    """
    Validates stratification, class distribution balance, and zero-leakage
    across a configured data budget split.
    """
    train_subset, val_subset = build_stratified_subsets(
        dataset,
        samples_per_class=samples_per_class,
        train_ratio=train_ratio,
        seed=seed,
    )
    targets = np.array(dataset.targets)
    train_targets = targets[train_subset.indices]
    val_targets = targets[val_subset.indices]

    num_classes = int(targets.max()) + 1
    expected_train = int(round(samples_per_class * train_ratio))
    expected_val = samples_per_class - expected_train

    train_counts = np.bincount(train_targets, minlength=num_classes)
    val_counts = np.bincount(val_targets, minlength=num_classes)

    assert (train_counts == expected_train).all(), f"Train split imbalance for budget {samples_per_class}"
    assert (val_counts == expected_val).all(), f"Val split imbalance for budget {samples_per_class}"
    assert set(train_subset.indices).isdisjoint(set(val_subset.indices)), "Overlap detected between train and val"

    return {
        "samples_per_class": samples_per_class,
        "train_samples": len(train_subset),
        "val_samples": len(val_subset),
        "train_per_class": expected_train,
        "val_per_class": expected_val,
        "num_classes": num_classes,
        "is_leakage_free": True,
        "class_balance_verified": True,
    }


# ---------------------------------------------------------------------------
# 5. Standalone verification test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Stratified CIFAR-100 Low-Data Pipeline")
    print("=" * 60)

    # 1. Test standard 100 images/class pipeline (Phase 1-5 baseline)
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        data_dir="./data",
        batch_size=64,
        samples_per_class=100,
        train_ratio=0.9,
        seed=42,
        num_workers=2,
        image_size=224,
        download=True,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)

    print(f"[100pc] Train: {n_train:,} samples | Val: {n_val:,} samples | Test: {n_test:,} samples")
    assert n_train == 9000, f"Expected 9,000 train samples, got {n_train}"
    assert n_val == 1000, f"Expected 1,000 val samples, got {n_val}"
    assert n_test == 10000, f"Expected 10,000 test samples, got {n_test}"

    # 2. Test 300 images/class pipeline (Phase 6 data scaling ablation)
    full_train = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=False, transform=None
    )
    stats_300 = verify_data_budget_split(full_train, samples_per_class=300, seed=42)
    print(f"[300pc] Train: {stats_300['train_samples']:,} samples | Val: {stats_300['val_samples']:,} samples | Balanced: {stats_300['class_balance_verified']}")
    assert stats_300["train_samples"] == 27000
    assert stats_300["val_samples"] == 3000

    # Cross-seed leakage sanity check across all three project seeds.
    for seed in (42, 1337, 3407):
        for budget in (100, 300):
            tr, va = build_stratified_subsets(full_train, samples_per_class=budget, seed=seed)
            overlap = set(tr.indices).intersection(set(va.indices))
            assert not overlap, f"Leakage found for budget {budget}, seed {seed}"
    print("Cross-seed and cross-budget leakage checks passed for seeds 42, 1337, 3407.")

    print("=" * 60)
    print("Data Loader Verification Passed Successfully!")
    print("=" * 60)
