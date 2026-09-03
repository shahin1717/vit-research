# 🧠 Vision Transformer Models & Register Token Architecture (`src/models`)

This package houses the core architectural components for **Track 1 (Pure Research)**:  
> **"Do Register Tokens Regularize Vision Transformers Under Data Scarcity?"**  
> *DLE-AI-202 (Deep Learning Course, Cohort I 2026)*

---

## 📌 Table of Contents
1. [Theoretical Motivation & The Attention Sink Problem](#1-theoretical-motivation--the-attention-sink-problem)
2. [How `register_vit.py` Works](#2-how-register_vitpy-works)
   - [Sequence Layout & Formatting](#sequence-layout--formatting)
   - [Why Registers Do NOT Receive Positional Embeddings](#why-registers-do-not-receive-positional-embeddings)
   - [Pretrained Backbone Loading Strategy](#pretrained-backbone-loading-strategy)
   - [Truncated Normal Parameter Initialization](#truncated-normal-parameter-initialization)
   - [Forward Pass & Register Discard](#forward-pass--register-discard)
3. [Public API & Token Partition Utilities](#3-public-api--token-partition-utilities)
4. [Interaction with `ViTAttentionHookManager`](#4-interaction-with-vitattentionhookmanager)
5. [Parameter Footprint & Model Configurations](#5-parameter-footprint--model-configurations)
6. [Quickstart Usage Example](#6-quickstart-usage-example)

---

## 1. Theoretical Motivation & The Attention Sink Problem

Standard Vision Transformers (Dosovitskiy et al., 2020) process an image by partitioning it into non-overlapping spatial patches (e.g., $14 \times 14 = 196$ patches for a $224 \times 224$ image with patch size $16 \times 16$).

### The Artifact:
In low-data training regimes or deep layers ($l \ge 8$), background patch tokens that contain little task-discriminative information develop **abnormally high $L_2$ activation norms** and sharp attention concentration. Because Softmax rows must sum to $1.0$, the transformer repurposes these background patches as **temporary memory sinks**, severely degrading feature maps and attention interpretability.

### The Remedy (Darcet et al., ICLR 2024):
By prepending $K \in \{1, 4, 8\}$ learnable **Register Tokens** alongside the `[CLS]` token, the model is provided with dedicated non-spatial storage scratchpads. The registers absorb redundant global signals, leaving the spatial patch tokens clean and structurally well-behaved.

---

## 2. How `register_vit.py` Works

[`src/models/register_vit.py`](register_vit.py) defines the **`RegisterVisionTransformer`** wrapper module around a standard `timm` Vision Transformer backbone (`vit_tiny_patch16_224`, $d=192$, 12 layers, 3 attention heads).

```
                             Input Image [B, 3, 224, 224]
                                          │
                                          ▼
                      Patch Embedding: 196 Spatial Tokens
                                          │
                                          ▼
             Add Learned Positional Embeddings: [CLS (1) + Patches (196)]
                                          │
                                          ▼
             ┌─────────────────────────────────────────────────────────┐
             │ 🪄 INJECT K REGISTERS:                                  │
             │ Sequence: [ CLS (0) || Regs (1..K) || Patches (1+K..) ] │
             │ Sequence Length S = 1 + K + 196                         │
             └────────────────────────────┬────────────────────────────┘
                                          │
                                          ▼
             12x Multi-Head Self-Attention + MLP Blocks (Residual Stream)
                                          │
                                          ▼
             ┌─────────────────────────────────────────────────────────┐
             │ 🗑️ DISCARD REGISTERS & SPATIAL PATCHES:                  │
             │ Extract solely the [CLS] token: x[:, 0]                 │
             └────────────────────────────┬────────────────────────────┘
                                          │
                                          ▼
                         Classification Head (100 Classes)
```

### Sequence Layout & Formatting
For a batch of images $x \in \mathbb{R}^{B \times 3 \times 224 \times 224}$, the token sequence entering transformer block 1 is:

$$\mathbf{X}_0 = \Big[ \mathbf{x}_{\text{cls}} \;\Big\|\; \mathbf{r}_1, \dots, \mathbf{r}_K \;\Big\|\; \mathbf{x}_1, \dots, \mathbf{x}_{196} \Big] \in \mathbb{R}^{B \times (1 + K + 196) \times 192}$$

| Token Index | Token Role | Count | Receives Spatial Pos Embed? | Sent to Final Head? |
|---|---|---|---|---|
| `0` | Class Token (`[CLS]`) | 1 | Yes (`pos_embed[:, :1]`) | **Yes** (Pooled representation) |
| `1 .. K` | **Register Tokens** | $K \in \{0, 1, 4, 8\}$ | **No** (Non-spatial) | **No** (Discarded post-block 12) |
| `1+K .. 1+K+195` | Spatial Patch Tokens | 196 | Yes (`pos_embed[:, 1:]`) | **No** (Discarded post-block 12) |

---

### Why Registers Do NOT Receive Positional Embeddings
A critical architectural pitfall is applying positional embeddings *after* concatenating registers. Doing so would shift the 2D spatial grid coordinates assigned to patch tokens by $K$ positions, corrupting the spatial geometry learned by the model.

`RegisterVisionTransformer` strictly executes:
```python
# 1. Patch projection + official ImageNet positional embeddings
x = self.base_model.patch_embed(x)
x = self.base_model._pos_embed(x)  # Shape: [B, 1 (CLS) + 196 (Patches), d]

# 2. Squeeze registers between [CLS] and Patches
if self.num_registers > 0:
    cls_token = x[:, :1, :]
    patch_tokens = x[:, 1:, :]
    reg_tokens = self.registers.expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, reg_tokens, patch_tokens], dim=1)  # Shape: [B, 1 + K + 196, d]
```
Registers remain strictly translation-invariant and coordinate-free.

---

### Pretrained Backbone Loading Strategy
If `timm.create_model(..., reg_tokens=4)` is called with `pretrained=True`, `timm` attempts to interpolate the pretrained positional embedding tensor to $201$ tokens, which crashes with a 2D spatial grid mismatch error (`shape [1, 13, 13, -1] is invalid for input of size 36864`).

`RegisterVisionTransformer` avoids this entirely:
1. Instantiates `base_model = timm.create_model(..., pretrained=True)` with standard $K=0$.
2. Pretrained ImageNet weights load with 100% fidelity without dimension conflicts.
3. Injects the custom learnable register parameter separately.

---

### Truncated Normal Parameter Initialization
Register tokens must break symmetry upon initialization. Zero-initialization (`torch.zeros`) results in zero-gradient flatlines during initial forward passes through Multi-Head Self-Attention.

Registers are initialized with a **truncated normal distribution**:
```python
self.registers = nn.Parameter(torch.zeros(1, num_registers, self.embed_dim))
nn.init.trunc_normal_(self.registers, std=0.02)
```
When $K=0$, `self.register_parameter("registers", None)` ensures zero extra memory or parameters are allocated.

---

### Forward Pass & Register Discard
After passing through all 12 transformer blocks and final LayerNorm, the representation is:
$$\mathbf{X}_{12} \in \mathbb{R}^{B \times (1 + K + 196) \times d}$$

Because the classification task predicts image labels from global context, **registers and spatial patch tokens are discarded**:
```python
cls_out = x[:, 0]  # Extract ONLY the [CLS] token at index 0
logits = self.base_model.head(cls_out)  # [B, 100]
```

---

## 3. Public API & Token Partition Utilities

In addition to `forward(x)` and `forward_features(x)`, the wrapper provides dedicated partition helpers for visualization, attention mapping, and diagnostic analysis:

```python
# Extract only the class token vector [B, d]
cls_vec = model.get_cls_token(features)

# Extract only the register tokens [B, K, d] (or None if K=0)
reg_tokens = model.get_register_tokens(features)

# Extract only the 196 spatial image patch tokens [B, 196, d]
patch_tokens = model.get_spatial_tokens(features)
```

---

## 4. Interaction with `ViTAttentionHookManager`

[`src/models/attention_hook.py`](attention_hook.py) seamlessly attaches forward hooks to `RegisterVisionTransformer`:
1. **Direct Attribute Exposing:** `RegisterVisionTransformer` exposes `self.blocks = self.base_model.blocks` and `self.head = self.base_model.head`.
2. **Attention Maps ($A^{(l, h)}$):** Hooks each block's attention module (`block.attn`) to reconstruct the unfused $[B, H, S, S]$ softmax attention matrices.
3. **Residual Stream Activations ($X^{(l)}$):** Hooks each transformer block output (`block(x)`) to capture unnormalized hidden states for the 3-sigma patch outlier rate metric.

---

## 5. Parameter Footprint & Model Configurations

Backbone: `vit_tiny_patch16_224` ($d=192$, $L=12$, $H=3$, $\text{classes}=100$).

| Registers ($K$) | Register Param Count ($K \times 192$) | Total Trainable Parameters | Sequence Length ($S$) |
|---|---|---|---|
| **$K = 0$** (Baseline) | $0$ | $5{,}544{,}676$ | $197$ ($1 + 196$) |
| **$K = 1$** | $+192$ | $5{,}544{,}868$ | $198$ ($1 + 1 + 196$) |
| **$K = 4$** | $+768$ | $5{,}545{,}444$ | $201$ ($1 + 4 + 196$) |
| **$K = 8$** | $+1{,}536$ | $5{,}546{,}212$ | $205$ ($1 + 8 + 196$) |

*Note: Adding 4 registers increases total parameter count by only $0.013\%$, ensuring any observed regularization effect is structural rather than an increase in model capacity.*

---

## 6. Quickstart Usage Example

```python
import torch
from src.models import RegisterVisionTransformer, ViTAttentionHookManager
from src.metrics import compute_layerwise_entropy, compute_layerwise_outlier_rate

# 1. Instantiate ViT-Tiny with K=4 register tokens
model = RegisterVisionTransformer(
    model_name="vit_tiny_patch16_224",
    num_classes=100,
    num_registers=4,
    pretrained=True
)

dummy_images = torch.randn(2, 3, 224, 224)

# 2. Forward pass with attention hooks attached
with ViTAttentionHookManager(model) as hook_mgr:
    logits = model(dummy_images)
    print("Logits shape:", logits.shape)  # torch.Size([2, 100])

    # 3. Compute diagnostic metrics
    entropy_by_layer = compute_layerwise_entropy(hook_mgr.attention_maps)
    outliers_by_layer = compute_layerwise_outlier_rate(
        hook_mgr.intermediate_activations, k_registers=4
    )

print("Layer 11 Entropy:", round(entropy_by_layer[11], 3), "bits")
print("Layer 11 Outlier Rate:", round(outliers_by_layer[11] * 100, 2), "%")
```
