"""
src/metrics
===========
Evaluation and diagnostic metrics for Vision Transformers with registers.
"""

from .entropy import compute_shannon_entropy, compute_layerwise_entropy
from .outliers import compute_patch_outlier_rate, compute_layerwise_outlier_rate
from .generalization import compute_generalization_gap, GeneralizationGapTracker

__all__ = [
    "compute_shannon_entropy",
    "compute_layerwise_entropy",
    "compute_patch_outlier_rate",
    "compute_layerwise_outlier_rate",
    "compute_generalization_gap",
    "GeneralizationGapTracker",
]