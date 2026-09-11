"""
Test-Time Registers Package (Jiang et al., NeurIPS 2025 Reimplementation)
========================================================================
Provides test-time activation redirection and register slot interventions
for Vision Transformers without requiring retraining or gradient updates.

Exports:
- `build_stage1_model`: Loads K=0 weights into K=1 wrapper with zeroed registers.
- `identify_register_neurons`: Discovers candidate outlier-concentrated neurons.
- `make_redirect_hook`: Builds activation redirection forward hook.
- `TestTimeRegisterManager`: Context manager to safely mount/unmount redirection hook.
- `evaluate_test_time_model`: Runs test set evaluation and collects diagnostics.
"""

from .model_prep import build_stage1_model
from .neuron_selection import identify_register_neurons
from .redirection import make_redirect_hook, TestTimeRegisterManager
from .eval_harness import evaluate_test_time_model

__all__ = [
    "build_stage1_model",
    "identify_register_neurons",
    "make_redirect_hook",
    "TestTimeRegisterManager",
    "evaluate_test_time_model",
]
