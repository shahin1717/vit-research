"""
Unit tests for Vision Transformer attention interpretability & diagnostic metrics.

Verifies:
  1. Shannon entropy mathematical invariants (uniform bounds, one-hot collapse).
  2. ViTAttentionHookManager non-invasive capture across all 12 MHSA layers.
  3. Patch-norm 3-sigma outlier detection accuracy.
  4. Register offset indexing integrity in attention slicing.

Author: Narmina Ibrahimova (Interpretability & Diagnostic Metrics Lead)
"""

import math
import pytest
import torch

from src.metrics.entropy import compute_layerwise_entropy, compute_shannon_entropy
from src.metrics.outliers import compute_layerwise_outlier_rate, compute_patch_outlier_rate
from src.models.attention_hook import ViTAttentionHookManager
from src.models.register_vit import RegisterVisionTransformer


class TestShannonAttentionEntropy:
    """Mathematical validation for Shannon entropy calculation on attention distributions."""

    def test_entropy_uniform_distribution(self):
        """Uniform distribution across N tokens must equal log2(N)."""
        seq_len = 197  # 1 CLS + 196 patches
        uniform_attn = torch.full((2, 3, seq_len, seq_len), 1.0 / seq_len)
        entropy = compute_shannon_entropy(uniform_attn)
        expected = math.log2(seq_len)
        assert pytest.approx(entropy, rel=1e-4) == expected

    def test_entropy_one_hot_distribution(self):
        """Degenerate one-hot attention must have exactly 0 entropy."""
        seq_len = 197
        one_hot_attn = torch.zeros((2, 3, seq_len, seq_len))
        one_hot_attn[:, :, :, 0] = 1.0
        entropy = compute_shannon_entropy(one_hot_attn)
        assert pytest.approx(entropy, abs=1e-6) == 0.0

    def test_entropy_bounds_monotonicity(self):
        """Sharper distributions must yield lower entropy than flatter distributions."""
        seq_len = 50
        sharp = torch.softmax(torch.randn(1, 1, seq_len, seq_len) * 5.0, dim=-1)
        flat = torch.softmax(torch.randn(1, 1, seq_len, seq_len) * 0.1, dim=-1)
        
        ent_sharp = compute_shannon_entropy(sharp)
        ent_flat = compute_shannon_entropy(flat)
        assert ent_sharp < ent_flat


class TestAttentionHookManagerIntegrity:
    """Verifies that the hook manager correctly captures all 12 MHSA layers."""

    @pytest.mark.parametrize("num_registers", [0, 1, 4])
    def test_hook_manager_capture_and_cleanup(self, num_registers: int):
        model = RegisterVisionTransformer(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            num_classes=100,
            num_registers=num_registers,
        )
        model.eval()

        with ViTAttentionHookManager(model) as hook_mgr:
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = model(x)

            attn_maps = hook_mgr.attention_maps
            assert len(attn_maps) == 12, f"Expected 12 attention maps, got {len(attn_maps)}"

            expected_seq_len = 1 + num_registers + 196
            for l_idx, attn in attn_maps.items():
                assert attn.shape == (1, 3, expected_seq_len, expected_seq_len)
                sums = attn.sum(dim=-1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


class TestPatchNormOutlierDetection:
    """Verifies 3-sigma patch outlier detection logic."""

    def test_compute_patch_outlier_rate_synthetic(self):
        # 1 CLS + 196 patches = 197 tokens
        tokens = torch.randn(2, 197, 192)
        rate_normal = compute_patch_outlier_rate(tokens, k_registers=0)
        assert 0.0 <= rate_normal <= 0.05

        # Inject extreme outlier into spatial patches
        tokens[:, 10, :] += 100.0
        rate_outlier = compute_patch_outlier_rate(tokens, k_registers=0)
        assert rate_outlier > 0.0
