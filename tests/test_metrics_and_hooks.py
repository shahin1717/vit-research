"""
Unit & Regression Test Suite: Attention Hooks & Metrics
========================================================
Comprehensive validation of:
- `src.models.attention_hook.ViTAttentionHookManager`
- `src.metrics.entropy`
- `src.metrics.outliers`
- `src.metrics.generalization`
"""

import math
import pytest
import timm
import torch
import torch.nn as nn

from src.models import ViTAttentionHookManager
from src.metrics import (
    compute_shannon_entropy,
    compute_layerwise_entropy,
    compute_patch_outlier_rate,
    compute_layerwise_outlier_rate,
    compute_generalization_gap,
    compute_accuracy_gap,
    GeneralizationGapTracker,
)


class DummyViTWrapper(nn.Module):
    """Simulates a model wrapper (like RegisterVisionTransformer)."""

    def __init__(self, vit_backbone: nn.Module):
        super().__init__()
        self.vit = vit_backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


# ==============================================================================
# 1. Attention Hook Manager Tests
# ==============================================================================

def test_hook_manager_standard_vit():
    """Verifies that hooks attach to all 12 blocks and extract attention + activations."""
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)

    with ViTAttentionHookManager(model) as hook_mgr:
        _ = model(dummy_input)

        # 12 layers in ViT-Tiny
        assert len(hook_mgr.attention_maps) == 12
        assert len(hook_mgr.intermediate_activations) == 12

        # Check attention shape: [B=2, Heads=3, SeqLen=197, SeqLen=197]
        attn_0 = hook_mgr.attention_maps[0]
        assert attn_0.shape == (2, 3, 197, 197)
        # Softmax rows should sum to 1.0
        assert torch.allclose(attn_0.sum(dim=-1), torch.ones(2, 3, 197), atol=1e-5)

        # Check activations shape: [B=2, SeqLen=197, Dim=192]
        acts_0 = hook_mgr.intermediate_activations[0]
        assert acts_0.shape == (2, 197, 192)

        # Verify real residual stream variance (NOT squashed by LayerNorm to ~1e-6)
        spatial_norms = torch.norm(acts_0[:, 1:, :], dim=-1)
        assert spatial_norms.std().item() > 0.1, "Activation norms should exhibit non-zero residual variance!"


def test_hook_manager_wrapped_model():
    """Verifies that hook manager discovers .blocks inside nested model wrappers."""
    backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    wrapped_model = DummyViTWrapper(backbone)
    dummy_input = torch.randn(1, 3, 224, 224)

    with ViTAttentionHookManager(wrapped_model) as hook_mgr:
        _ = wrapped_model(dummy_input)
        assert len(hook_mgr.attention_maps) == 12
        assert len(hook_mgr.intermediate_activations) == 12


def test_hook_manager_cleanup():
    """Verifies that remove() cleans up hooks and clear() empties RAM."""
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)

    hook_mgr = ViTAttentionHookManager(model)
    _ = model(dummy_input)
    assert len(hook_mgr.attention_maps) == 12

    hook_mgr.clear()
    assert len(hook_mgr.attention_maps) == 0
    assert len(hook_mgr.intermediate_activations) == 0

    hook_mgr.remove()
    _ = model(dummy_input)
    # No new maps should be stored since hooks were removed
    assert len(hook_mgr.attention_maps) == 0


# ==============================================================================
# 2. Shannon Attention Entropy Tests
# ==============================================================================

def test_shannon_entropy_uniform_distribution():
    """For uniform attention over S tokens, entropy must equal log2(S)."""
    S = 197
    uniform_attn = torch.full((1, 3, S, S), 1.0 / S)
    entropy = compute_shannon_entropy(uniform_attn)
    expected_entropy = math.log2(S)
    assert pytest.approx(entropy, rel=1e-4) == expected_entropy


def test_shannon_entropy_one_hot_distribution():
    """For completely collapsed (sink) attention, entropy must equal 0.0 bits."""
    S = 197
    one_hot = torch.zeros((1, 3, S, S))
    one_hot[..., 0] = 1.0  # All attention concentrated on first token
    entropy = compute_shannon_entropy(one_hot)
    assert pytest.approx(entropy, abs=1e-5) == 0.0


def test_layerwise_entropy():
    """Verifies layerwise entropy dictionary aggregation."""
    S = 64
    attn_dict = {
        0: torch.full((1, 1, S, S), 1.0 / S),
        1: torch.full((1, 1, S, S), 1.0 / S),
    }
    res = compute_layerwise_entropy(attn_dict)
    assert len(res) == 2
    assert pytest.approx(res[0], rel=1e-4) == math.log2(S)


def test_entropy_error_handling():
    """Verifies appropriate exceptions on empty or invalid inputs."""
    with pytest.raises(ValueError):
        compute_shannon_entropy(torch.empty(0))

    with pytest.raises(ValueError):
        compute_shannon_entropy(torch.tensor(1.0))


# ==============================================================================
# 3. Patch Outlier Rate Tests
# ==============================================================================

def test_outlier_rate_no_outliers():
    """Uniform normal features have virtually 0 outliers at 3-sigma (Gaussian < 0.3%)."""
    # 100 samples, S = 1 (CLS) + 196 (patches)
    acts = torch.randn(100, 197, 192)
    rate = compute_patch_outlier_rate(acts, k_registers=0)
    assert 0.0 <= rate < 0.01  # Should be well below 1% for standard Gaussian


def test_outlier_rate_synthetic_sink_detection():
    """Verifies that an artificial high-norm attention sink is cleanly detected."""
    # S = 1 (CLS) + 4 (registers) + 196 (spatial) = 201
    acts = torch.randn(4, 201, 192)
    # Inject an extreme 50x outlier into a spatial background patch (token index 100)
    acts[:, 100, :] *= 50.0

    rate = compute_patch_outlier_rate(acts, k_registers=4)
    # Exactly 1 patch out of 196 is an outlier -> expected fraction ~ 1/196 = 0.0051 (0.51%)
    assert rate > 0.004
    assert rate < 0.02


def test_outlier_rate_2d_input():
    """Verifies that unbatched [S, d] input is automatically handled."""
    acts = torch.randn(197, 192)
    rate = compute_patch_outlier_rate(acts, k_registers=0)
    assert isinstance(rate, float)


def test_outlier_rate_k_registers_offset():
    """Verifies that register tokens are correctly skipped from spatial analysis."""
    # When K=4, tokens 1..4 are registers. If a register has a huge norm,
    # it should NOT count as a SPATIAL patch outlier!
    acts = torch.randn(2, 201, 192)
    acts[:, 2, :] *= 100.0  # Token index 2 is a REGISTER token, not spatial!

    rate = compute_patch_outlier_rate(acts, k_registers=4)
    # Since the outlier is on a register token, spatial outlier rate should remain ~0!
    assert rate < 0.005


def test_outlier_rate_error_handling():
    """Verifies validation on negative K or insufficient tokens."""
    with pytest.raises(ValueError):
        compute_patch_outlier_rate(torch.randn(2, 10, 16), k_registers=-1)

    # Sequence length 5 is too short for CLS(1) + K(4) + spatial(>=1)
    with pytest.raises(ValueError):
        compute_patch_outlier_rate(torch.randn(2, 5, 16), k_registers=4)


# ==============================================================================
# 4. Generalization Gap Tests
# ==============================================================================

def test_generalization_gap_calculations():
    """Verifies loss and accuracy gap calculations."""
    loss_gap = compute_generalization_gap(train_loss=1.20, val_loss=1.45)
    assert pytest.approx(loss_gap, abs=1e-5) == 0.25

    acc_gap = compute_accuracy_gap(train_acc=85.0, val_acc=78.5)
    assert pytest.approx(acc_gap, abs=1e-5) == 6.5


def test_generalization_gap_tracker():
    """Verifies history tracking, dict-of-lists export, and final_gap retrieval."""
    tracker = GeneralizationGapTracker()
    assert tracker.as_dict_of_lists() == {"epoch": [], "train_loss": [], "val_loss": [], "gap": []}

    g1 = tracker.update(epoch=1, train_loss=2.0, val_loss=2.2)
    g2 = tracker.update(epoch=2, train_loss=1.5, val_loss=1.8)

    assert pytest.approx(g1, abs=1e-5) == 0.2
    assert pytest.approx(g2, abs=1e-5) == 0.3
    assert pytest.approx(tracker.final_gap(), abs=1e-5) == 0.3

    history = tracker.as_dict_of_lists()
    assert history["epoch"] == [1.0, 2.0]
    assert history["gap"] == [pytest.approx(0.2), pytest.approx(0.3)]
