# 📦 Team Task Package: Rufet — Visualizations, LaTeX Paper & Oral Defense

**Assignee:** Rufet  
**Role:** Lead Analyst & Academic Paper / Defense Deck Lead  
**Target Code & Document Files:**
* `scripts/visualize_attention.py`
* `scripts/plot_metrics.py`
* `src/utils/export_latex.py`
* `paper/sections/04_experimental_setup.tex` & `paper/sections/05_results_and_analysis.tex`
* `presentation/slides.tex` & `presentation/slides.md`  
**Target Delivery:** Thursday, 4 September 2026  
**Cross-References:** [Academic Paper Draft](file:///mnt/c/Vaults/aiac-res/Atlas/Academic%20Paper%20Draft.md) | [Oral Defense Outline](file:///mnt/c/Vaults/aiac-res/Atlas/Oral%20Defense%20Presentation%20Outline.md) | [Paper Directory](file:///mnt/c/Vaults/aiac-res/paper/README.md)  

---

## 🎯 1. Mission & Scientific Context

In a peer-reviewed research project, world-class empirical results are only as impactful as their visual presentation and narrative clarity.

Your mission is to translate raw experiment outputs, attention tensors, and checkpoint weights into:
1. **Publication-Grade Attention Heatmaps (`scripts/visualize_attention.py`):** Visual comparison of attention maps from `[CLS]` across layers $l \in \{1, 4, 8, 12\}$, demonstrating that registers eliminate spurious background spikes.
2. **Quantitative Metric Figures (`scripts/plot_metrics.py`):**
   * *Figure 1:* Layer Index ($1 \to 12$) vs. Shannon Entropy $\bar{H}^{(l)}$ across $K \in \{0, 1, 4, 8\}$.
   * *Figure 2:* Register Count $K$ vs. Generalization Gap ($\Delta\mathcal{L} = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$).
   * *Figure 3:* Training & Validation loss curves over 50 epochs.
3. **Automated LaTeX Tables (`src/utils/export_latex.py`):** Converting `outputs/sweep_summary.json` into `paper/tables/results_table.tex`.
4. **LaTeX Manuscript & Presentation Deck:** Authoring the experimental results sections in LaTeX and finalizing the 8-slide oral defense deck.

---

## 🎨 2. Visualizations Specification & Guidelines

### Figure 1: Attention Map Spatial Heatmap Overlay
* **Extraction:** Query attention weights from the `[CLS]` token to all spatial patch keys: $A_{\text{cls}, 1+K:}^{(l)} \in \mathbb{R}^{196}$.
* **Reshaping:** Reshape the 196 vector into a $14 \times 14$ 2D spatial grid.
* **Upsampling:** Bicubic upsampling to $224 \times 224$ resolution.
* **Overlay:** Blend the heat map (using `matplotlib.cm.inferno` or `jet`, $\alpha=0.6$) over the original test image.
* **Layout:** Side-by-side comparative grid:
  * Top Row: Baseline ($K=0$) across Layers 1, 4, 8, 12.
  * Bottom Row: Register ViT ($K=4$) across Layers 1, 4, 8, 12.
* **Export:** `paper/figures/attention_maps_comparison.pdf` (vector format for LaTeX).

### Figure 2: Layer-wise Shannon Entropy Curves
* **X-Axis:** Layer index $l \in \{1, 2, \dots, 12\}$.
* **Y-Axis:** Mean Shannon Entropy $\bar{H}^{(l)}$ in bits.
* **Curves:** 4 distinct lines for $K=0$, $K=1$, $K=4$, $K=8$ with error bands showing standard deviation ($\pm 1\sigma$) across seeds.
* **Styling:** Seaborn theme, bold fonts, gridlines, exported as `paper/figures/entropy_vs_layer.pdf`.

---

## 💻 3. Implementation Blueprint

### 1. Attention Visualizer Script (`scripts/visualize_attention.py`)

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

def generate_attention_heatmap(
    model, 
    image_tensor: torch.Tensor, 
    original_pil: Image.Image,
    layers: list = [1, 4, 8, 12],
    k_registers: int = 0,
    save_path: str = "paper/figures/attention_comparison.pdf"
):
    """
    Extracts [CLS] attention across specified layers, upsamples to 224x224,
    and saves side-by-side comparative overlay figure.
    """
    # 1. Forward pass with hooks
    # 2. Extract cls -> patch attention: attn[0, :, 0, (1 + k_registers):]
    # 3. Mean over heads -> shape [196] -> reshape [14, 14]
    # 4. Upsample to [224, 224] with TF.resize(..., interpolation=InterpolationMode.BICUBIC)
    # 5. Plot grid with original image overlay
```

### 2. LaTeX Table Generator (`src/utils/export_latex.py`)

```python
import json

def export_latex_table(json_path: str = "outputs/sweep_summary.json", tex_path: str = "paper/tables/results_table.tex"):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    tex = r"""\begin{table}[t]
\centering
\caption{\textbf{Main Results on Low-Data CIFAR-100 (100 samples/class)}. Mean and standard deviation across 3 random seeds (\texttt{42}, \texttt{1337}, \texttt{3407}).}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Top-1 Accuracy (\%)} & \textbf{Val Loss} & \textbf{Gen Gap ($\Delta\mathcal{L}$)} & \textbf{Layer 12 Entropy (bits)} \\
\midrule
"""
    for k in [0, 1, 4, 8]:
        item = data[f"k_{k}"]
        label = f"ViT-Tiny ($K={k}$)" if k > 0 else "ViT-Tiny (Baseline, $K=0$)"
        acc = f"{item['val_top1_mean']:.2f} $\\pm$ {item['val_top1_std']:.2f}"
        val_l = f"{item['val_loss_mean']:.3f} $\\pm$ {item['val_loss_std']:.3f}"
        gap = f"{item['gen_gap_mean']:.3f}"
        ent = f"{item['entropy_mean']:.2f} $\\pm$ {item['entropy_std']:.2f}"
        tex += f"{label} & {acc} & {val_l} & {gap} & {ent} \\\\\n"
        
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"LaTeX table generated at {tex_path}")
```

---

## 📑 4. LaTeX Paper & Slide Deck Deliverables

1. **`paper/sections/04_experimental_setup.tex`:** Detail dataset stratification, model architecture, register initialization, and training hyperparameters.
2. **`paper/sections/05_results_and_analysis.tex`:** Analyze the quantitative results, reference Table 1 and Figures 1–3, and discuss whether the Capacity Dilution hypothesis holds for $K=8$.
3. **`presentation/slides.tex`:** Prepare the 8-slide Beamer presentation for the 10–12 minute defense:
   * Slide 1: Title & Team
   * Slide 2: Motivation (The Softmax Attention Sink Problem)
   * Slide 3: Research Question & Hypotheses
   * Slide 4: Proposed Register Mechanism ($K \in \{0, 1, 4, 8\}$)
   * Slide 5: Experimental Setup (Low-Data Stratification)
   * Slide 6: Quantitative Results & Accuracy Comparisons
   * Slide 7: Attention Heatmaps & Layer-wise Entropy Proofs
   * Slide 8: Conclusion & Key Takeaways

---

## ✅ 5. Definition of Done & Verification Test

Your deliverable is complete when:
1. `python scripts/plot_metrics.py` generates high-resolution PDF vector figures in `paper/figures/`.
2. `python src/utils/export_latex.py` generates valid LaTeX table in `paper/tables/results_table.tex`.
3. Running `pdflatex paper/main.tex` compiles with **zero errors and zero missing citations**.
4. Running `pdflatex presentation/slides.tex` produces the final defense slide deck.
