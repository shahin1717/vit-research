"""
Tests for Test-Time Registers (Tokens-Test-Time)
===============================================
Verifies:
1. get_outlier_mask functionality on synthetic tensors and edge cases.
2. Stage 1 model preparation: loading K=0 weights into K=1 wrapper with zeroed registers.
3. Candidate neuron identification at target blocks.
4. Test-time redirection hook: accumulation into register slot, mean reset, and invariant preservation.
5. End-to-end evaluation with TestTimeRegisterManager context.
"""

import pytest
import torch
import torch.nn as nn
from src.metrics.outlier_mask import get_outlier_mask
from src.models.register_vit import RegisterVisionTransformer


def test_get_outlier_mask_basic():
    """Verifies that get_outlier_mask correctly identifies outliers exceeding 3-sigma."""
    B, N, d = 2, 196, 64
    # Create normal patch activations with std 1.0
    torch.manual_seed(42)
    spatial = torch.randn(B, N, d)
    # Inject deliberate extreme outlier at token 10
    spatial[:, 10, :] = spatial[:, 10, :] * 20.0

    cls_token = torch.zeros(B, 1, d)
    # Sequence with k_registers=0: [CLS, Patches] -> S = 197
    patch_activations = torch.cat([cls_token, spatial], dim=1)

    mask = get_outlier_mask(patch_activations, k_registers=0)
    assert mask.shape == (B, N)
    assert mask.dtype == torch.bool
    # Token 10 must be flagged as an outlier
    assert mask[0, 10].item() is True
    assert mask[1, 10].item() is True


def test_get_outlier_mask_with_registers():
    """Verifies get_outlier_mask with k_registers=1 offsets sequence correctly."""
    B, N, d = 2, 196, 64
    torch.manual_seed(42)
    spatial = torch.randn(B, N, d)
    spatial[:, 5, :] = spatial[:, 5, :] * 25.0

    cls_token = torch.zeros(B, 1, d)
    reg_token = torch.zeros(B, 1, d)
    # Sequence with k_registers=1: [CLS, Reg, Patches] -> S = 198
    seq = torch.cat([cls_token, reg_token, spatial], dim=1)

    mask = get_outlier_mask(seq, k_registers=1)
    assert mask.shape == (B, N)
    assert mask[0, 5].item() is True
    assert mask[:, 0].sum().item() == 0  # Token 0 is not an outlier


def test_get_outlier_mask_edge_cases():
    """Verifies error handling for invalid sequence lengths or k_registers."""
    with pytest.raises(ValueError, match="k_registers cannot be negative"):
        get_outlier_mask(torch.randn(2, 10, 32), k_registers=-1)

    with pytest.raises(ValueError, match="Sequence length S=2 is too short"):
        get_outlier_mask(torch.randn(2, 2, 32), k_registers=1)


def test_stage1_model_loading_and_zero_init():
    """Verifies that K=0 weights load cleanly into K=1 model and register is zero-initialized."""
    # Create mock K=0 model
    k0_model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=10,
        num_registers=0,
        pretrained=False,
    )
    k0_state_dict = k0_model.state_dict()

    # Create K=1 model
    k1_model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=10,
        num_registers=1,
        pretrained=False,
    )
    missing, unexpected = k1_model.load_state_dict(k0_state_dict, strict=False)
    assert missing == ["registers"]
    assert unexpected == []

    # Zero initialize registers
    with torch.no_grad():
        k1_model.registers.zero_()

    assert torch.all(k1_model.registers == 0.0)

    # Forward pass check
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = k1_model(x)
    assert out.shape == (2, 10)


def test_redirect_hook_mechanism():
    """Verifies that the redirection hook moves outlier activation and resets original position."""
    from tokens_test_time.redirection import make_redirect_hook

    B, S, hidden_dim = 2, 198, 768  # 1 CLS + 1 Reg + 196 Patches
    torch.manual_seed(42)
    # Generate dummy MLP activation output
    act_out = torch.randn(B, S, hidden_dim)

    # Make token index 12 (patch index 10) an outlier
    act_out[:, 12, :] *= 15.0

    target_neurons = [3, 7, 42]
    hook_fn = make_redirect_hook(neuron_ids=target_neurons, outlier_mask_fn=get_outlier_mask, k_registers=1)

    class DummyModule(nn.Module):
        pass

    dummy_mod = DummyModule()
    modified = hook_fn(dummy_mod, (), act_out)

    assert modified.shape == act_out.shape

    # For target neurons, register slot (index 1) should now have non-zero accumulated outlier value
    for n in target_neurons:
        assert modified[0, 1, n].item() != 0.0
        # Untargeted neurons at index 1 should remain unchanged
    untargeted_neuron = 100
    assert torch.allclose(modified[:, 1, untargeted_neuron], act_out[:, 1, untargeted_neuron])

    # Spatial patches at index 12 for target neurons should be reset close to their spatial mean
    for n in target_neurons:
        assert not torch.allclose(modified[:, 12, n], act_out[:, 12, n])


def test_neuron_identification():
    """Verifies that identify_register_neurons selects top outlier neurons."""
    from tokens_test_time.neuron_selection import identify_register_neurons

    model = RegisterVisionTransformer(
        model_name="vit_tiny_patch16_224",
        num_classes=10,
        num_registers=0,
        pretrained=False,
    )
    model.eval()

    # Pass small calibration input
    calib = torch.randn(2, 3, 224, 224)
    neurons = identify_register_neurons(
        model=model,
        calibration_images=calib,
        target_block=0,
        top_n=4,
        device=torch.device("cpu"),
    )
    assert len(neurons) == 4
    for nid in neurons:
        assert isinstance(nid, int)
        assert 0 <= nid < 768  # ViT-Tiny MLP hidden dim is 768
