# 🧪 Bonus Experiment: Test-Time Registers (No Training Required)

**Branch:** `tokens-test-time`  
**Folder:** `tokens-test-time/`  
**Reference:** Jiang et al., *"Test-Time Registers for Vision Transformers"*, NeurIPS 2025  
**Codebase:** [shahin1717/vit-research](https://github.com/shahin1717/vit-research)

---

## 🔬 Motivation & Scientific Question

In our primary 24-run controlled study, we established that **fine-tuned, learned register tokens** (Darcet et al., ICLR 2024) do not improve held-out test accuracy under low-data regimes on CIFAR-100 (100 and 300 images/class).

Recently, Jiang et al. (NeurIPS 2025) proposed a fundamentally different paradigm: **Test-Time Registers**. Rather than spending compute learning register tokens during training, they observed on large foundation models (DINOv2, OpenCLIP) that:
1. High-norm outlier artifacts are driven by a sparse subset of specific MLP neurons.
2. At **inference time only**, one can identify these "register neurons" and redirect their outlier activity into an empty, untrained register token slot, resetting the spatial patch values to their spatial mean.
3. This intervention requires **zero training, zero gradients, and zero backward passes**.

### Scope & Adapter Note
The official `test-time-registers` repository only ships adapters for CLIP and DINOv2 backbones. Here, we implemented the **core mechanism directly against our own `RegisterVisionTransformer` wrapper** in a clean, self-contained module for `timm` Vision Transformers.

---

## 🛠️ Implementation Architecture

```
tokens-test-time/
├── __init__.py                # Package exports
├── model_prep.py              # Stage 1: Load K=0 weights into K=1 wrapper (zero-initialized)
├── neuron_selection.py        # Stage 2.2: Identify candidate register neurons on calibration batch
├── redirection.py             # Stage 2.3: PyTorch forward hook to redirect outlier activations
├── eval_harness.py            # Standardized inference loop gathering Top-1/Top-5, loss, entropy, outliers
├── run_test_time_eval.py      # CLI runner supporting single checkpoints and 3-seed statistical aggregations
└── results/
    └── test_time_registers_results.json  # Full multi-seed experimental JSON log
```

### Stage 1: Passive Empty Slot (Untrained Zeroed Slot)
Loads the trained $K=0$ backbone weights (`outputs/exp0[1-3]_k0_*/best_model.pth`) into a $K=1$ architecture (`RegisterVisionTransformer(num_classes=100, num_registers=1)`). The register slot at index 1 is zero-initialized (`model.registers.zero_()`), matching Jiang et al.'s convention.

### Stage 2: Neuron Identification & Activation Redirection
1. **Outlier Detection (`src/metrics/outlier_mask.py`):**
   Applies the exact 3-sigma threshold:
   $$\text{Outlier Mask}_i = \left( \|x_i\|_2 > \mu + 3\sigma \right)$$
2. **Neuron Selection:**
   On a calibration batch, hooks `model.blocks[TARGET_BLOCK].mlp.act` and residual output. Computes the activation gap between outlier and normal patch tokens for each hidden neuron ($d_{\text{mlp}} = 768$):
   $$\Delta_j = \left| \mathbb{E}_{\text{outliers}}[a_{i, j}] - \mathbb{E}_{\text{normal}}[a_{i, j}] \right|$$
   Extracts top $N=4$ neurons with the highest concentration on outlier tokens.
3. **Intervention Hook:**
   Attaches a forward hook to `block.mlp.act` during test set inference:
   - Accumulates outlier values of selected neurons into the register slot (index 1).
   - Resets the original spatial patch positions to the per-sample spatial mean for that neuron.

---

## 📊 Empirical Results (3-Seed Statistical Aggregation on CIFAR-100 Test Set)

Evaluated across all 3 trained $K=0$ baseline checkpoints (random seeds 42, 1337, 3407) on the full 10,000-image untouched CIFAR-100 test set:

| Configuration | Intervention Mechanism | Test Top-1 Acc (%) | Test Loss | Source / Status |
|---|---|---|---|---|
| **Control Baseline ($K=0$)** | Standard Fine-Tuned ViT-Tiny | **$75.18 \pm 0.30$** | $1.1110 \pm 0.0037$ | Table I (Main Study) |
| **Trained Registers ($K=1$)** | Fine-Tuned 1 Register | $74.66 \pm 0.42$ | $0.985 \pm 0.015$ | Table I (Main Study) |
| **Stage 1: Passive Slot ($K=1$)** | Untrained Zeroed Slot (No Redir) | **$75.25 \pm 0.24$** | $1.1103 \pm 0.0036$ | **New (Jiang et al. Stage 1)** |
| **Stage 2: Redirection (Block 1)** | Top-4 Neurons $\to$ Reg Slot | **$75.25 \pm 0.26$** | $1.1104 \pm 0.0038$ | **New (Jiang et al. Stage 2)** |
| **Stage 2: Redirection (Block 9)** | Top-4 Neurons $\to$ Reg Slot | **$75.24 \pm 0.25$** | $1.1103 \pm 0.0036$ | **New (Jiang et al. Stage 2)** |
| **Stage 2: Redirection (Block 11)** | Top-4 Neurons $\to$ Reg Slot | **$75.25 \pm 0.24$** | $1.1103 \pm 0.0036$ | **New (Jiang et al. Stage 2)** |

---

## 🔍 Key Findings & Discussion Takeaways

1. **Passive Extra Slot is Harmless ($75.25\%$ vs. $75.18\%$):**
   Pre-pending an empty, zero-initialized register slot to an already-trained $K=0$ backbone produces negligible variation ($+0.07\%$, well within $1\sigma$). Unlike training with registers (which caused a $-0.52\%$ dip due to optimization drift under scarce data), an inert slot does not perturb the existing spatial self-attention routing.
2. **Neuron Redirection Does Not Impact Classification Accuracy:**
   Redirecting the top outlier neurons at early (Block 1), middle (Block 9), or late (Block 11) transformer layers into the register slot maintains identical accuracy ($75.24\% - 75.25\%$).
3. **Scale Invariance & Foundation vs. Compact Models:**
   While Jiang et al. observed qualitative improvements in dense downstream tasks (segmentation masks, depth maps) on massive foundation models (DINOv2-Large, OpenCLIP), in compact classification models ($d=192, L=12$), test-time redirection neither helps nor harms classification generalization.
4. **Paper Recommendation:**
   This experiment provides excellent supplementary context for an **extended discussion paragraph** or **Appendix note**, emphasizing that the absence of register regularization under data scarcity holds whether registers are learned during training or redirected at inference time.

---

## 🚀 Reproduction Command

To reproduce these exact results on any workstation or GPU:

```bash
# Evaluate across all 3 random seeds on CIFAR-100 test set
python tokens-test-time/run_test_time_eval.py --all_seeds --target_blocks 1,9,11

# Run unit and integration tests
pytest tests/test_tokens_test_time.py
```
