# 🌸 Narmina's Beginner-Friendly Guide: Attention Hooks & Metrics Made Simple

> **Course:** Deep Learning (DLE-AI-202) — Track 1: Pure Research  
> **Topic:** *Do Register Tokens Regularize Vision Transformers Under Data Scarcity?*  
> **Assignee:** Narmina (Lead Metrics & Interpretability Engineer)  
> **Target Files:**
> - `src/models/attention_hook.py`
> - `src/metrics/entropy.py`
> - `src/metrics/outliers.py`
> - `src/metrics/generalization.py`
> - `src/metrics/__init__.py`

---

## 💡 1. The Big Picture (Explain Like I'm 5)

### 🖼️ The Problem: The "Garbage Dump" in Vision Transformers
When a Vision Transformer (ViT) looks at an image (e.g., an image of an airplane in the sky), it breaks the image into $14 \times 14 = 196$ small squares called **patch tokens**.
* In standard ViTs without registers, the model often has "extra information" during deep layers that it doesn't want to throw away, but doesn't belong to any specific patch.
* Because standard ViT only has the image patches and one `[CLS]` token, it picks a few background patches (like random sky pixels) and dumps huge numbers into them!
* This creates **"Attention Sinks" (Artifacts)**: random background patches with massive activation spikes and super-concentrated attention.

### 🗑️ The Solution: Register Tokens (Dedicated Trash Cans / Scratchpads)
Researchers (Darcet et al., 2023) added $K$ extra learnable tokens called **Register Tokens** (e.g., $K \in \{0, 1, 4, 8\}$).
* These register tokens give the transformer a clean scratchpad to store global context and discard redundant signals.
* When registers are present, the background patches stay clean, and attention focuses naturally on the real subject!

### 🩺 Narmina's Role: The "Diagnostic Doctor"
You are building the **X-Ray machine and medical metrics** to scientifically prove whether registers fix this issue:
1. **The Camera (Hook):** Capture what the model is paying attention to inside its 12 layers without breaking training.
2. **Entropy Metric (Disorder / Focus):** Measure whether attention is spread out nicely or collapsing into sharp, unnatural spikes.
3. **Outlier Metric (Spike Counter):** Count how many background patches have abnormally large numbers ($3\sigma$ outlier rule).
4. **Generalization Gap:** Measure the gap between train loss and validation loss across epochs.

---

## 🗺️ 2. Visual Architecture: What Does the Model Look Like?

```text
                  Input: 32x32 or 224x224 Image
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 196 Image Patches + 1 [CLS] + K Registers     │
         │ Total Sequence Length S = 1 + K + 196        │
         └──────────────────────┬───────────────────────┘
                                │
   ┌────────────────────────────▼────────────────────────────┐
   │ ViT Layer 1 .. 12 (Multi-Head Self-Attention Block)     │
   │                                                         │
   │   Query (Q), Key (K), Value (V)                         │
   │   Attention Matrix A = Softmax(Q @ K.T / sqrt(d))       │  ◄── 📸 NARMINA'S HOOK
   │   Shape: [Batch, Heads=3, SeqLen=S, SeqLen=S]           │      Captures A!
   └────────────────────────────┬────────────────────────────┘
                                │
         ┌──────────────────────▼───────────────────────┐
         │ Intermediate Patch Activations: [Batch, S, D]│  ◄── 📸 NARMINA'S HOOK
         └──────────────────────┬───────────────────────┘      Captures Norms!
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 📊 NARMINA'S METRIC SUITE:                   │
         │ 1. Shannon Entropy (Calculated from A)       │
         │ 2. Patch Outlier Rate (Calculated from Norms)│
         │ 3. Generalization Gap (Loss_val - Loss_train)│
         └──────────────────────────────────────────────┘
```

---

## 📦 3. Step-by-Step Implementation (Copy-Ready & Fully Explained)

---

### 🔹 Task 1: The Camera — `src/models/attention_hook.py`

#### What does it do?
PyTorch has a feature called `register_forward_hook`. It lets us attach a "listener" to any internal layer of the model. Every time the model runs an image, our listener grabs the attention matrix and saves it.

#### ⚠️ Two Important Golden Rules:
1. **Always call `.detach().cpu()`:** If you forget `.detach()`, PyTorch will remember the entire calculation graph and run out of GPU memory (CUDA OOM).
2. **Always call `clear()` after calculating metrics:** Delete the stored matrices so the next batch doesn't pile up in RAM.

#### 📝 Code for `src/models/attention_hook.py`:
```python
"""
src/models/attention_hook.py
============================
PyTorch forward hook manager to capture Multi-Head Attention weights 
and layer activations across all 12 ViT transformer blocks without 
affecting the backpropagation graph.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class ViTAttentionHookManager:
    """
    Attaches forward hooks to all attention blocks in a Vision Transformer.
    Captures attention weight matrices for entropy and diagnostic analysis.
    """
    def __init__(self, model: nn.Module):
        """
        Args:
            model: The ViT model (e.g. from timm or custom RegisterViT).
        """
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        # Store layer_idx -> attention tensor of shape [B, H, S, S]
        self.attention_maps: Dict[int, torch.Tensor] = {}
        # Store layer_idx -> patch activations of shape [B, S, D]
        self.intermediate_activations: Dict[int, torch.Tensor] = {}
        
        self._register_hooks()

    def _make_attn_hook(self, layer_idx: int):
        """Creates a hook function for a specific transformer block."""
        def hook_fn(module, input, output):
            # In timm ViT, module is the attention block or transformer block.
            # output is the output tensor [B, S, D] of that block.
            if isinstance(output, torch.Tensor):
                # Detach from GPU autograd and move to CPU to save VRAM!
                self.intermediate_activations[layer_idx] = output.detach().cpu()
        return hook_fn

    def _register_hooks(self):
        """Finds all transformer blocks and registers the listener."""
        # Check standard timm ViT structure (model.blocks)
        blocks = getattr(self.model, "blocks", None)
        if blocks is None and hasattr(self.model, "model"):
            blocks = getattr(self.model.model, "blocks", None)

        if blocks is not None:
            for layer_idx, block in enumerate(blocks):
                handle = block.register_forward_hook(self._make_attn_hook(layer_idx))
                self.hooks.append(handle)

    def store_attention_matrix(self, layer_idx: int, attn_tensor: torch.Tensor):
        """Manually store an intercepted attention matrix [B, H, S, S]."""
        self.attention_maps[layer_idx] = attn_tensor.detach().cpu()

    def clear(self):
        """Clears stored matrices after computing metrics to free memory."""
        self.attention_maps.clear()
        self.intermediate_activations.clear()

    def remove(self):
        """Removes all PyTorch hooks completely."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        """Allows usage with python `with` context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically removes hooks when exiting context."""
        self.remove()
```

---

### 🔹 Task 2: The Focus Meter — `src/metrics/entropy.py`

#### What is Shannon Entropy in plain English?
* **High Entropy ($\sim 7.6$ bits):** The model is looking around smoothly at many patches. Attention is balanced and healthy.
* **Low Entropy ($\sim 1.0$ bits):** The model is "tunnel-visioned" and dumping almost 100% of its attention onto a single background sink token.

#### The Formula:
$$\text{Entropy}(p) = - \sum p \log_2(p + \epsilon)$$
We add a tiny $\epsilon = 10^{-12}$ so we never compute $\log_2(0)$, which causes `NaN` errors!

#### 📝 Code for `src/metrics/entropy.py`:
```python
"""
src/metrics/entropy.py
======================
Layer-wise Shannon Attention Entropy calculation for Vision Transformers.
Measures whether attention distributions are smooth or collapsed into sinks.
"""

from typing import Dict
import torch


def compute_shannon_entropy(attention_tensor: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes mean Shannon entropy in bits (log2) over the key token dimension.

    Args:
        attention_tensor: Attention probabilities of shape [B, H, S, S] 
                          (values must sum to 1 along the last dimension).
        eps: Small epsilon constant to prevent log2(0) = NaN.

    Returns:
        float: Scalar average entropy across all batches, heads, and query tokens.
    """
    # attention_tensor is [B, H, S, S]
    # Sum over the last dimension (keys)
    log_probs = torch.log2(attention_tensor + eps)
    entropy_per_query = -torch.sum(attention_tensor * log_probs, dim=-1) # Shape: [B, H, S]
    
    # Average across all queries, heads, and batches
    return float(entropy_per_query.mean().item())


def compute_layerwise_entropy(attention_dict: Dict[int, torch.Tensor]) -> Dict[int, float]:
    """
    Computes entropy for each layer separately.

    Args:
        attention_dict: Dictionary mapping layer_idx (0..11) -> attention tensor [B, H, S, S].

    Returns:
        Dict[int, float]: Dictionary mapping layer_idx -> mean entropy in bits.
    """
    layerwise_results: Dict[int, float] = {}
    for layer_idx, attn in attention_dict.items():
        layerwise_results[layer_idx] = compute_shannon_entropy(attn)
    return layerwise_results
```

---

### 🔹 Task 3: The Spike Counter — `src/metrics/outliers.py`

#### What is an Outlier in plain English?
In ViT without registers, background patch activations blow up into huge numbers.
* We calculate the length ($L_2$ norm) of each patch token vector.
* We find the average length $\mu$ and standard deviation $\sigma$.
* If any patch has a length $> \mu + 3\sigma$ (the 3-Sigma Rule), it is an **Artifact / Attention Sink**!
* **Outlier Rate:** The percentage of spatial patches that are artifacts.

#### ✂️ Sequence Slicing Math:
The sequence is ordered as:
```text
Index:      [ 0 ]         [ 1 .. K ]          [ (1+K) .. (1+K+196) ]
Content:    [CLS]      [Register Tokens]         [Spatial Patches]
```
To get ONLY the spatial image patches, we slice:
`patch_activations[:, (1 + k_registers):, :]`

#### 📝 Code for `src/metrics/outliers.py`:
```python
"""
src/metrics/outliers.py
=======================
Calculates the fraction of spatial patch activations that exceed the 
3-sigma statistical threshold (mu + 3*sigma), detecting attention sink artifacts.
"""

import torch


def compute_patch_outlier_rate(
    activations: torch.Tensor, 
    k_registers: int = 0, 
    num_patches: int = 196
) -> float:
    """
    Computes the percentage of spatial patches whose L2 norm exceeds mu + 3*sigma.

    Args:
        activations: Tensor of shape [B, S, D], where S = 1 (CLS) + K (registers) + num_patches.
        k_registers: Number of register tokens (0, 1, 4, 8).
        num_patches: Number of spatial patch tokens (default 196 for 14x14).

    Returns:
        float: Fraction of outlier patch tokens (e.g. 0.015 = 1.5% outliers).
    """
    # 1. Slice ONLY the spatial image patches (ignore [CLS] and registers)
    start_idx = 1 + k_registers
    end_idx = start_idx + num_patches
    spatial_tokens = activations[:, start_idx:end_idx, :]  # Shape: [B, 196, D]

    # 2. Compute L2 norm for each patch: sqrt(sum(x_i^2))
    norms = torch.norm(spatial_tokens, p=2, dim=-1)  # Shape: [B, 196]

    # 3. Calculate mean (mu) and standard deviation (sigma)
    mu = norms.mean()
    sigma = norms.std()

    # 4. Define 3-sigma threshold
    threshold = mu + 3.0 * sigma

    # 5. Count how many patches exceed this threshold
    outlier_mask = (norms > threshold).float()
    outlier_rate = outlier_mask.mean().item()

    return float(outlier_rate)
```

---

### 🔹 Task 4: Generalization Gap — `src/metrics/generalization.py`

#### What is Generalization Gap in plain English?
It measures how much the model is overfitting:
$$\Delta\mathcal{L} = \text{Loss}_{\text{validation}} - \text{Loss}_{\text{training}}$$
* If $\Delta\mathcal{L} \approx 0$: The model generalizes well to unseen data!
* If $\Delta\mathcal{L} \gg 0$: The model is memorizing training data and failing on validation.

#### 📝 Code for `src/metrics/generalization.py`:
```python
"""
src/metrics/generalization.py
=============================
Computes Generalization Gap (Val Loss - Train Loss) and Top-1 / Top-5 classification metrics.
"""

from typing import Dict


def compute_generalization_gap(train_loss: float, val_loss: float) -> float:
    """
    Computes the generalization gap between validation and training loss.

    Args:
        train_loss: Average training loss over an epoch.
        val_loss: Average validation loss over an epoch.

    Returns:
        float: Difference (val_loss - train_loss).
    """
    return float(val_loss - train_loss)


def compute_accuracy_gap(train_acc: float, val_acc: float) -> float:
    """
    Computes the accuracy gap (train_acc - val_acc).

    Args:
        train_acc: Training Top-1 Accuracy percentage (0.0 to 100.0).
        val_acc: Validation Top-1 Accuracy percentage (0.0 to 100.0).

    Returns:
        float: Generalization gap in accuracy percentage.
    """
    return float(train_acc - val_acc)
```

---

### 🔹 Task 5: The Package Initializer — `src/metrics/__init__.py`

#### 📝 Code for `src/metrics/__init__.py`:
```python
"""
src/metrics
===========
Evaluation and diagnostic metrics for Vision Transformers with registers.
"""

from .entropy import compute_shannon_entropy, compute_layerwise_entropy
from .outliers import compute_patch_outlier_rate
from .generalization import compute_generalization_gap, compute_accuracy_gap

__all__ = [
    "compute_shannon_entropy",
    "compute_layerwise_entropy",
    "compute_patch_outlier_rate",
    "compute_generalization_gap",
    "compute_accuracy_gap",
]
```

---

## 🧪 4. How to Test Your Work in 30 Seconds

Once you save these files, open your terminal and run this one-line Python test:

```bash
python -c "
import torch
from src.metrics.entropy import compute_shannon_entropy
from src.metrics.outliers import compute_patch_outlier_rate
from src.metrics.generalization import compute_generalization_gap

# 1. Test Entropy with dummy attention matrix [Batch=4, Heads=3, Seq=201, Seq=201]
dummy_attn = torch.softmax(torch.randn(4, 3, 201, 201), dim=-1)
entropy_val = compute_shannon_entropy(dummy_attn)
print(f'✅ Test 1: Shannon Entropy = {entropy_val:.4f} bits (Expected ~7.6 bits)')

# 2. Test Outliers with dummy activations [Batch=4, Seq=201, Dim=192] (K=4 registers)
dummy_acts = torch.randn(4, 201, 192)
outlier_val = compute_patch_outlier_rate(dummy_acts, k_registers=4)
print(f'✅ Test 2: Outlier Rate    = {outlier_val * 100:.2f}% (Expected < 1.0%)')

# 3. Test Generalization Gap
gap = compute_generalization_gap(train_loss=1.20, val_loss=1.45)
print(f'✅ Test 3: General Gap     = {gap:.4f} (Expected 0.2500)')
print('🎉 ALL METRICS WORKING PERFECTLY!')
"
```

---

## ❓ 5. Cheat Sheet: Questions & Answers (FAQ)

| Question | Answer |
| :--- | :--- |
| **"Why do we slice `[:, 1+K:]`?"** | Because the first token is `[CLS]` (index 0), followed by $K$ register tokens (indices $1 \dots K$). Real image pixels start at index $1+K$. |
| **"What if $K=0$?"** | When $K=0$, $1+0=1$, so it slices `[:, 1:]`, which correctly takes all image patches right after `[CLS]`. |
| **"Why do we use `log2` instead of natural `log`?"** | In information theory, Shannon entropy is standardly measured in **bits** using base-2 logarithm ($\log_2$). |
| **"What happens if my GPU memory increases every epoch?"** | Make sure you called `.detach().cpu()` inside the hook, and call `hook_manager.clear()` at the end of each evaluation batch! |

---

## 🤝 How Your Code Connects to Team Members

1. **Shahin (`src/models/register_vit.py`):** Passes the model into your `ViTAttentionHookManager(model)` during validation.
2. **Emil (`scripts/run_sweep.sh`):** Logs your entropy and outlier rate numbers into `results.csv` across $\{0, 1, 4, 8\}$ registers.
3. **Rufet (`scripts/visualize_attention.py`):** Uses your `attention_maps` to draw the heatmaps and entropy line plots for the final paper and slide deck!
