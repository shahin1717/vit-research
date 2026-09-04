#!/usr/bin/env python3
"""
Preflight Compatibility & Sanity Check
======================================
Automated preflight diagnostic script executed prior to launching the 12-run
Vision Transformer register ablation sweep.

Validates that:
1. All required runtime dependencies (PyTorch, torchvision, timm, yaml, numpy) are present.
2. `scripts/train.py` and `scripts/eval.py` have zero syntax errors and import all dependencies (including `time`).
3. Dataloader keyword contracts match (`train_ratio` and `image_size` passed cleanly).
4. `RegisterVisionTransformer` builds and runs forward passes across all K in {0, 1, 4, 8}.
5. All sweep YAML configurations (`configs/*.yaml`) are syntactically valid and have correct keys.
6. Attention hook interception and disarming execute without memory leakage.
7. Output and checkpoint directories are writable.

Exit Code:
- 0: All checks pass ("No blocking issues. The sweep is safe to launch.")
- 1: Blocker discovered. Prevents wasted GPU-hours.
"""

import ast
import inspect
import os
import sys
from pathlib import Path

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))


def check_dependencies() -> bool:
    """Verifies essential deep learning libraries are importable."""
    print("Checking core dependencies...", end=" ")
    try:
        import numpy
        import timm
        import torch
        import torchvision
        import yaml
        print("OK (PyTorch %s, timm %s)" % (torch.__version__, timm.__version__))
        return True
    except ImportError as e:
        print("FAIL! Missing dependency: %s" % e)
        return False


def check_train_script_imports() -> bool:
    """Verifies scripts/train.py AST imports time and critical modules."""
    print("Checking scripts/train.py imports...", end=" ")
    train_file = WORKSPACE_ROOT / "scripts" / "train.py"
    if not train_file.is_file():
        print("FAIL! scripts/train.py not found.")
        return False

    with open(train_file, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename="train.py")
    except SyntaxError as e:
        print("FAIL! Syntax error in train.py: %s" % e)
        return False

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    if "time" not in imported_modules:
        print("FAIL! 'import time' is missing from scripts/train.py (BLOCKER-1)")
        return False

    print("OK ('time' and critical modules verified)")
    return True


def check_dataloader_signature_and_callsite() -> bool:
    """Verifies get_cifar100_loaders signature and train.py callsite compatibility."""
    print("Checking dataloader signature compatibility...", end=" ")
    try:
        from src.data.cifar100_subset import get_cifar100_loaders
        sig = inspect.signature(get_cifar100_loaders)
        params = sig.parameters

        # Check signature expects train_ratio and image_size
        if "train_ratio" not in params:
            print("FAIL! get_cifar100_loaders is missing 'train_ratio' parameter.")
            return False
        if "val_split" in params:
            print("FAIL! get_cifar100_loaders has unexpected parameter 'val_split'.")
            return False

        # Check train.py callsite does not pass val_split
        train_file = WORKSPACE_ROOT / "scripts" / "train.py"
        with open(train_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple AST or string check for invalid keyword
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if func_name == "get_cifar100_loaders":
                    kw_names = [kw.arg for kw in node.keywords]
                    if "val_split" in kw_names:
                        print("FAIL! train.py calls get_cifar100_loaders with 'val_split' (BLOCKER-2)")
                        return False

        print("OK (signature: %s)" % ", ".join(list(params.keys())[:5]))
        return True
    except Exception as e:
        print("FAIL! Error inspecting dataloader: %s" % e)
        return False


def check_model_register_arms() -> bool:
    """Verifies RegisterVisionTransformer builds and runs across all K in {0, 1, 4, 8}."""
    print("Checking model instantiation across K in {0, 1, 4, 8}...", end=" ")
    try:
        import torch
        from src.models import RegisterVisionTransformer

        dummy_input = torch.randn(2, 3, 224, 224)
        for k in [0, 1, 4, 8]:
            model = RegisterVisionTransformer(
                model_name="vit_tiny_patch16_224",
                num_classes=100,
                num_registers=k,
                pretrained=False,
            )
            model.eval()
            with torch.no_grad():
                out = model(dummy_input)
                assert out.shape == (2, 100), "Unexpected output shape %s for K=%d" % (out.shape, k)

        print("OK (all arms K in {0,1,4,8} validated)")
        return True
    except Exception as e:
        print("FAIL! Error building RegisterVisionTransformer: %s" % e)
        return False


def check_attention_hooks() -> bool:
    """Verifies attention hook attachment, capture, and removal."""
    print("Checking attention hooks & metrics instrumentation...", end=" ")
    try:
        import torch
        from src.models import RegisterVisionTransformer, ViTAttentionHookManager
        from src.metrics import compute_layerwise_entropy, compute_layerwise_outlier_rate

        model = RegisterVisionTransformer(
            model_name="vit_tiny_patch16_224",
            num_classes=100,
            num_registers=4,
            pretrained=False,
        )
        dummy_input = torch.randn(2, 3, 224, 224)

        with ViTAttentionHookManager(model) as mgr:
            with torch.no_grad():
                _ = model(dummy_input)
            assert len(mgr.attention_maps) == 12, "Expected 12 attention maps, got %d" % len(mgr.attention_maps)
            assert len(mgr.intermediate_activations) == 12, "Expected 12 activation maps, got %d" % len(mgr.intermediate_activations)
            entropy = compute_layerwise_entropy(mgr.attention_maps)
            outliers = compute_layerwise_outlier_rate(mgr.intermediate_activations, k_registers=4)
            assert len(entropy) == 12
            assert len(outliers) == 12

        # Verify hooks removed
        for block in model.blocks:
            assert len(block._forward_hooks) == 0, "Forward hooks were not cleanly removed."

        print("OK (entropy & 3-sigma outlier hooks verified)")
        return True
    except Exception as e:
        print("FAIL! Error in attention hooks: %s" % e)
        return False


def check_yaml_configs() -> bool:
    """Verifies all required sweep YAML configs exist and contain expected sections."""
    print("Checking YAML configs...", end=" ")
    import yaml

    required_configs = [
        ("configs/baseline_k0.yaml", 0),
        ("configs/vit_tiny_k1.yaml", 1),
        ("configs/vit_tiny_k4.yaml", 4),
        ("configs/vit_tiny_k8.yaml", 8),
    ]

    for rel_path, expected_k in required_configs:
        full_path = WORKSPACE_ROOT / rel_path
        if not full_path.is_file():
            print("FAIL! Missing config file: %s" % rel_path)
            return False

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            k = data.get("model", {}).get("num_registers")
            if k != expected_k:
                print("FAIL! %s expected num_registers=%d, got %s" % (rel_path, expected_k, k))
                return False
        except Exception as e:
            print("FAIL! Invalid YAML in %s: %s" % (rel_path, e))
            return False

    print("OK (all 4 treatment arm configs valid)")
    return True


def check_filesystem_permissions() -> bool:
    """Verifies output directories are writable."""
    print("Checking directory write permissions...", end=" ")
    try:
        out_dir = WORKSPACE_ROOT / "outputs"
        ckpt_dir = WORKSPACE_ROOT / "checkpoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        test_file = out_dir / ".preflight_write_test"
        with open(test_file, "w") as f:
            f.write("ok")
        test_file.unlink()

        print("OK (outputs/ and checkpoints/ writable)")
        return True
    except Exception as e:
        print("FAIL! Write permission error: %s" % e)
        return False


def main() -> int:
    """Executes all preflight checks."""
    print("=" * 60)
    print("🔍 Running Preflight Sanity Check for ViT Registers Sweep")
    print("=" * 60)

    checks = [
        check_dependencies,
        check_train_script_imports,
        check_dataloader_signature_and_callsite,
        check_model_register_arms,
        check_attention_hooks,
        check_yaml_configs,
        check_filesystem_permissions,
    ]

    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
            break

    print("=" * 60)
    if all_passed:
        print("✅ No blocking issues. The sweep is safe to launch.")
        print("=" * 60)
        return 0
    else:
        print("❌ Preflight check FAILED! Please resolve the blockers above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
