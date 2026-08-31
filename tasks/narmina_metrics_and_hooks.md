# 📦 Team Task Package: Narmina — Attention Hooks & Diagnostic Metrics Suite

**Assignee:** Narmina  
**Role:** Lead Metrics & Interpretability Engineer  
**Target Code Files:**
* `src/models/attention_hook.py`
* `src/metrics/entropy.py`
* `src/metrics/outliers.py`
* `src/metrics/generalization.py`
* `src/metrics/__init__.py`  
**Target Delivery:** Tuesday, 1 September 2026  
**Cross-References:** [Team Task Division](file:///home/shahin/aiac-res/team_task_division.md) | [Architecture Note](file:///mnt/c/Vaults/aiac-res/Atlas/ViT%20Registers%20Architecture.md) | [Deep Research Note](file:///mnt/c/Vaults/aiac-res/Atlas/ViT_Registers_Deep_Research_and_Self_Grilling.md)  

---

## 🎯 1. Mission & Scientific Context

In a Pure Research paper (Track 1), visual inspection of attention heatmaps is subjective and scientifically insufficient. We need **exact mathematical metrics** to quantify the attention sink phenomenon and prove whether register tokens eliminate background artifacts and stabilize attention entropy under data scarcity.

Your mission is to build:
1. **Attention Hook Manager (`src/models/attention_hook.py`):** Non-invasive forward hooks intercepting the attention weight matrices $A^{(l, h)}$ across all 12 Multi-Head Attention blocks in ViT-Tiny without breaking backpropagation.
2. **Layer-wise Shannon Attention Entropy (`src/metrics/entropy.py`):** Measuring whether attention distributions collapse into low-entropy spikes (sinks) or maintain healthy information routing.
3. **Patch-Norm Outlier Rate (`src/metrics/outliers.py`):** Quantifying the fraction of spatial patch tokens whose $L_2$ activation norm exceeds the $\mu_l + 3\sigma_l$ artifact threshold.
4. **Generalization Gap Tracker (`src/metrics/generalization.py`):** Computing $\Delta\mathcal{L} = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$ across epochs.

---

## 📐 2. Mathematical Formulations

### A. Attention Weight Interception
In ViT-Tiny ($L=12$ layers, $H_{\text{heads}}=3$), the sequence length is:
$$S = 1 + K + N$$
where $1$ is `[CLS]`, $K \in \{0, 1, 4, 8\}$ is the register count, and $N = 196$ (14x14 patches).

For each layer $l \in \{1, \dots, 12\}$ and head $h \in \{1, 2, 3\}$, scaled dot-product attention computes:
$$A^{(l, h)} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times H_{\text{heads}} \times S \times S}$$

### B. Layer-wise Shannon Attention Entropy
The Shannon entropy for query token $i$ in layer $l$, head $h$ across key tokens $j \in \{1, \dots, S\}$ is:
$$H_i^{(l, h)} = -\sum_{j=1}^{S} A_{i, j}^{(l, h)} \log_2 \left( A_{i, j}^{(l, h)} + \epsilon \right), \quad \epsilon = 10^{-12}$$

The mean layer-wise entropy $\bar{H}^{(l)}$ across all queries and heads in a batch is:
$$\bar{H}^{(l)} = \frac{1}{B \cdot H_{\text{heads}} \cdot S} \sum_{b=1}^{B} \sum_{h=1}^{H_{\text{heads}}} \sum_{i=1}^{S} H_i^{(l, h)}$$

* **Hypothesis:** Baseline models ($K=0$) experience an unnatural entropy drop in deep layers ($l \ge 8$) as background queries collapse into single-token sinks. Register models ($K \ge 1$) maintain balanced entropy.

### C. Patch-Norm Outlier Rate ($3\sigma$ Bound)
Let $X_{\text{patch}}^{(l)} \in \mathbb{R}^{N \times d}$ be the intermediate spatial activations at layer $l$ (excluding `[CLS]` and registers).
1. Compute the $L_2$ norm of each patch token:
   $$n_i^{(l)} = \|x_i^{(l)}\|_2 = \sqrt{\sum_{k=1}^d (x_{i, k}^{(l)})^2}, \quad \forall i \in \{1, \dots, N\}$$
2. Compute the layer mean $\mu_l = \frac{1}{N} \sum_{i=1}^N n_i^{(l)}$ and standard deviation $\sigma_l = \sqrt{\frac{1}{N}\sum_{i=1}^N (n_i^{(l)} - \mu_l)^2}$.
3. Define the outlier threshold: $\theta_l = \mu_l + 3\sigma_l$.
4. Outlier rate:
   $$\text{OutlierRate}(l) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( n_i^{(l)} > \theta_l \right)$$

---

## 💻 3. Implementation Blueprint

### 1. Attention Hook Manager (`src/models/attention_hook.py`)

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class ViTAttentionHookManager:
    """
    Attaches forward hooks to all multi-head attention layers in a timm Vision Transformer.
    Stores detached attention matrices and clears them on demand to prevent CUDA OOM.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self.intermediate_activations: Dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        # In timm ViT: model.blocks[i].attn
        for layer_idx, block in enumerate(self.model.blocks):
            # Hook the attention module
            hook = block.attn.register_forward_hook(self._make_attn_hook(layer_idx))
            self.hooks.append(hook)

    def _make_attn_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # In timm, attn module computes weights internally;
            # Intercept or calculate softmax(Q @ K.T / sqrt(d))
            # Detach immediately!
            pass
        return hook_fn

    def clear(self):
        """Clears stored matrices from memory."""
        self.attention_maps.clear()
        self.intermediate_activations.clear()

    def remove(self):
        """Removes all PyTorch forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```

### 2. Shannon Entropy Function (`src/metrics/entropy.py`)

```python
import torch
from typing import Dict

def compute_shannon_entropy(attention_tensor: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes Shannon entropy across the key dimension:
    attention_tensor shape: [B, H, S, S]
    Returns scalar mean entropy in bits (log2).
    """
    # - sum(p * log2(p + eps)) along last dim
    entropy = -torch.sum(attention_tensor * torch.log2(attention_tensor + eps), dim=-1)
    return float(entropy.mean().item())

def compute_layerwise_entropy(attention_dict: Dict[int, torch.Tensor]) -> Dict[int, float]:
    """
    Takes dictionary of layer_idx -> attention_tensor [B, H, S, S]
    Returns dictionary of layer_idx -> mean_entropy_value.
    """
    results = {}
    for layer_idx, attn in attention_dict.items():
        results[layer_idx] = compute_shannon_entropy(attn)
    return results
```

### 3. Outlier Rate Function (`src/metrics/outliers.py`)

```python
import torch
from typing import Dict

def compute_patch_outlier_rate(patch_activations: torch.Tensor, k_registers: int = 0) -> float:
    """
    Computes fraction of patch tokens with L2 norm > mu + 3*sigma.
    patch_activations: [B, S, d] where S = 1 (cls) + K (registers) + 196 (patches).
    Extracts only spatial patches: patch_activations[:, (1 + k_registers):, :]
    """
    spatial_tokens = patch_activations[:, (1 + k_registers):, :] # [B, 196, d]
    norms = torch.norm(spatial_tokens, p=2, dim=-1) # [B, 196]
    
    mu = norms.mean()
    sigma = norms.std()
    threshold = mu + 3.0 * sigma
    
    outliers = (norms > threshold).float()
    return float(outliers.mean().item())
```

---

## ⚠️ 4. Critical Memory & Computational Gotchas

1. **CUDA Out-of-Memory (OOM) via Gradient Retention:**
   * If you store attention tensors inside the hook without calling `.detach().cpu()`, PyTorch retains the entire autograd computation graph across all 12 layers!
   * *Rule:* Always call `attn_matrix.detach().cpu()` inside the hook.
2. **Hook Accumulation Leak:** If you instantiate `ViTAttentionHookManager(model)` repeatedly in training loops without calling `.remove()`, multiple duplicate hooks run simultaneously, multiplying VRAM usage.
   * *Rule:* Initialize the hook manager once or always call `hook_manager.remove()` in `finally:` blocks.
3. **$\log(0)$ Numerical Instability:** Direct `torch.log2(0)` produces `-inf`, which propagates `NaN` across batch reductions. Always add $\epsilon = 10^{-12}$.
4. **Token Index Offset:** ViT sequence shape changes when registers are present ($S = 1 + K + 196$). Never hardcode patch slice indices like `[:, 1:]`. Always use `[:, (1 + num_registers):]`.

---

## ✅ 5. Definition of Done & Verification Test

Your deliverable is complete when the standalone metric test passes cleanly:

```bash
python -c "
import torch
from src.metrics.entropy import compute_shannon_entropy
from src.metrics.outliers import compute_patch_outlier_rate

dummy_attn = torch.softmax(torch.randn(4, 3, 201, 201), dim=-1)
entropy = compute_shannon_entropy(dummy_attn)
print(f'Test Shannon Entropy: {entropy:.4f} bits (Expected ~7.6 bits)')

dummy_acts = torch.randn(4, 201, 192)
outlier_rate = compute_patch_outlier_rate(dummy_acts, k_registers=4)
print(f'Test Outlier Rate: {outlier_rate * 100:.2f}% (Expected < 1.0%)')
"
```
