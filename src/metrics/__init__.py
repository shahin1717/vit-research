"""
Vision Transformer Evaluation & Interpretability Metrics
=========================================================
Diagnostic and evaluation metrics for Vision Transformers with registers
under data scarcity experiments (Track 1).

Public Exports:
- `compute_shannon_entropy`: Scalar Shannon entropy calculation over attention matrices.
- `compute_layerwise_entropy`: Layerwise Shannon attention entropy.
- `compute_patch_outlier_rate`: Patch activation outlier rate using 3-sigma bound.
- `compute_layerwise_outlier_rate`: Layerwise outlier rate across all transformer blocks.
- `compute_generalization_gap`: Difference between validation and training loss.
- `compute_accuracy_gap`: Difference between training and validation accuracy.
- `GeneralizationGapTracker`: Tracker class for multi-epoch generalization curves.
"""

from .entropy import compute_shannon_entropy, compute_layerwise_entropy
from .outliers import compute_patch_outlier_rate, compute_layerwise_outlier_rate
from .generalization import (
    compute_generalization_gap,
    compute_accuracy_gap,
    GeneralizationGapTracker,
)

__all__ = [
    "compute_shannon_entropy",
    "compute_layerwise_entropy",
    "compute_patch_outlier_rate",
    "compute_layerwise_outlier_rate",
    "compute_generalization_gap",
    "compute_accuracy_gap",
    "GeneralizationGapTracker",
]