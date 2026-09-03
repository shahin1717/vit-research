"""
Unit & Integration Test Suite: Register Vision Transformer (rViT)
==================================================================
Validates:
- `src.models.register_vit.RegisterVisionTransformer` across K in {0, 1, 4, 8}
- Token sequence geometry: [CLS (1) || Registers (K) || Patches (196)]
- Gradient flow to register parameters
- Token partitioning methods (CLS, registers, spatial patches)
- Full compatibility with `ViTAttentionHookManager` and diagnostic metrics
"""

import pytest
import torch
import torch.nn as nn

from src.models import RegisterVisionTransformer, ViTAttentionHookManager
from src.metrics import compute_layerwise_entropy, compute_layerwise_outlier_rate


@pytest.mark.parametrize("k", [0, 1, 4, 8])
def test_register_vit_forward_shapes(k: int):
    """Verifies that forward_features and forward output exact shapes for all K."""
    model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=100,
        num_registers=k,
        pretrained=False,
    )
    B = 2
    dummy_x = torch.randn(B, 3, 224, 224)

    # Features: [B, 1 (CLS) + K (Registers) + 196 (Patches), 192 (Dim)]
    feats = model.forward_features(dummy_x)
    expected_seq_len = 1 + k + 196
    assert feats.shape == (B, expected_seq_len, 192), f"Failed feature shape for K={k}: {feats.shape}"

    # Output Logits: [B, 100]
    logits = model(dummy_x)
    assert logits.shape == (B, 100), f"Failed logits shape for K={k}: {logits.shape}"


def test_register_parameter_gradient_flow():
    """Verifies that register tokens are learnable parameters receiving non-zero gradients."""
    k = 4
    model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=100,
        num_registers=k,
        pretrained=False,
    )

    assert model.registers is not None
    assert isinstance(model.registers, nn.Parameter)
    assert model.registers.shape == (1, k, 192)
    assert model.registers.requires_grad is True

    # Run dummy backward pass
    dummy_x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([5, 42])
    criterion = nn.CrossEntropyLoss()

    output = model(dummy_x)
    loss = criterion(output, target)
    loss.backward()

    # Verify gradients flowed into registers
    assert model.registers.grad is not None
    assert torch.norm(model.registers.grad).item() > 0.0, "Registers should receive non-zero gradients!"


def test_k0_has_no_register_parameter():
    """Verifies that baseline K=0 does not allocate register parameters."""
    model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=100,
        num_registers=0,
        pretrained=False,
    )
    assert model.registers is None


def test_token_partition_utilities():
    """Verifies get_cls_token, get_register_tokens, and get_spatial_tokens."""
    k = 4
    model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=100,
        num_registers=k,
        pretrained=False,
    )
    dummy_x = torch.randn(2, 3, 224, 224)
    feats = model.forward_features(dummy_x)

    cls_tok = model.get_cls_token(feats)
    assert cls_tok.shape == (2, 192)

    reg_tok = model.get_register_tokens(feats)
    assert reg_tok is not None
    assert reg_tok.shape == (2, k, 192)

    spatial_tok = model.get_spatial_tokens(feats)
    assert spatial_tok.shape == (2, 196, 192)


def test_hook_manager_integration_with_registers():
    """Verifies that ViTAttentionHookManager seamlessly extracts attention & activations with registers."""
    k = 4
    model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=100,
        num_registers=k,
        pretrained=False,
    )
    dummy_x = torch.randn(2, 3, 224, 224)

    with ViTAttentionHookManager(model) as hook_mgr:
        _ = model(dummy_x)

        # Verify 12 layers captured
        assert len(hook_mgr.attention_maps) == 12
        assert len(hook_mgr.intermediate_activations) == 12

        # Attention shape: [B=2, Heads=3, SeqLen=201, SeqLen=201]
        attn_0 = hook_mgr.attention_maps[0]
        assert attn_0.shape == (2, 3, 201, 201)

        # Activations shape: [B=2, SeqLen=201, Dim=192]
        acts_0 = hook_mgr.intermediate_activations[0]
        assert acts_0.shape == (2, 201, 192)

        # Test diagnostic metrics calculation
        entropy_dict = compute_layerwise_entropy(hook_mgr.attention_maps)
        assert len(entropy_dict) == 12
        assert all(isinstance(v, float) and v > 0 for v in entropy_dict.values())

        outliers_dict = compute_layerwise_outlier_rate(hook_mgr.intermediate_activations, k_registers=k)
        assert len(outliers_dict) == 12
        assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in outliers_dict.values())


def test_invalid_num_registers():
    """Verifies that negative register count raises ValueError."""
    with pytest.raises(ValueError):
        RegisterVisionTransformer(num_registers=-1)
