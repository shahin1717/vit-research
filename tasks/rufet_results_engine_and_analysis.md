# 📦 Team Task Package: Rufet — Results Engine & Statistical Analysis Lead

**Assignee:** Rufet
**Role:** Statistical Inference Layer, Attention Extraction & Evidence Generation
**Target Code Files:**
* `src/utils/logger.py` — multi-seed statistical reduction engine
* `src/utils/__init__.py` — package exports
* `tests/test_sweep_aggregation.py` — reduction-engine test suite
* `scripts/visualize_attention.py` — attention tensor extraction & spatial reconstruction
* `scripts/plot_metrics.py` — quantitative evidence figures
* `src/utils/export_latex.py` — automated results-table generator
* `.gitattributes` — repository line-ending policy

**Target Delivery:** Friday, 4 September 2026
**Cross-References:** [Team Task Division](../team_task_division.md) | [Sweep Execution Package](emil_sweep_execution_and_operations.md) | [Roadmap](../roadmap.md)

---

## 🎯 1. Mission & Scientific Context

The sweep produces twelve directories of raw JSON. None of them is a result. A
result is an *inference*: a treatment effect with a dispersion estimate that
survives seed noise, and a mechanism claim backed by the attention tensors that
produced it. Constructing both is this package's engineering problem.

Three hypotheses from the proposal must be decided by code you write:

| Hypothesis | Decided by |
| :--- | :--- |
| **H1 — Structural regularization**: registers shrink $\Delta\mathcal{L} = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$ | Per-seed gap reduction in the aggregation engine |
| **H2 — Capacity dilution**: at $d=192$, large $K$ dilutes representation power | Non-monotonic accuracy across $K \in \{0,1,4,8\}$ with $\sigma$ across seeds |
| **H3 — Entropy collapse prevention**: registers stop layer-wise attention entropy collapse | Layer-resolved $\bar{H}^{(l)}$ with $\pm 1\sigma$ bands, plus the attention overlays |

**Interface with the execution side:** the sweep package (Emil) owns everything up
to and including the per-run artifacts. This package owns everything downstream.
`outputs/sweep_summary.json` is the single boundary object — once it exists, no
figure, table or manuscript claim may read a raw run directory again.

---

## 📥 2. Input Contract and the Schema Recovery Problem

Every completed run directory `outputs/expXX_kY_sZ/` provides:

| Artifact | Consumed for |
| :--- | :--- |
| `metrics.json` | `best_val_top1`, `best_epoch`, per-epoch `history`, `test_results` carrying `layerwise_entropy` and `layerwise_outliers` |
| `train_history.csv` | Per-epoch train/validation loss and accuracy curves |
| `best_model.pth` | Weights at the best validation epoch, for attention extraction |
| `last_model.pth` | Final-epoch checkpoint |

### 2.1 The recovery problem

`scripts/train.py` serialises a **nested** record. The three quantities the results
table needs are not top-level keys:

| Required quantity | Where it actually lives |
| :--- | :--- |
| `best_val_loss` | `history[best_epoch]["val_loss"]` |
| `final_train_loss` | `history[-1]["train_loss"]` |
| `mean_layerwise_entropy` | mean over `test_results["layerwise_entropy"].values()` |

A reader written as `data.get("best_val_loss", 0.0)` therefore returns `0.0` — and
does so **silently**, filling validation loss, generalization gap and entropy with
zeros. That is three of the four columns of the main results table, and nothing
raises. The engine must prefer a flat key when present and otherwise reconstruct
the value from the nested record, and the test suite must pin that behaviour.

JSON also stringifies the integer layer keys emitted by the metric functions, so
layer indices must be re-normalised to `int` on load.

---

## 🧮 3. Statistical Reduction Engine (`src/utils/logger.py`)

### 3.1 Reduction

For each arm $K \in \{0, 1, 4, 8\}$, over the $S$ seeds that completed:

$$\mu(m) = \frac{1}{S}\sum_{s=1}^{S} m_s
\qquad
\sigma(m) = \sqrt{\frac{1}{S}\sum_{s=1}^{S}\big(m_s - \mu(m)\big)^2}$$

Population form (`numpy.std`, `ddof=0`) is the agreed convention; the engine must
record it in the output so the manuscript can state the estimator it used.

### 3.2 Why the gap is paired

The generalization gap must be formed **per seed, then averaged**:

$$\overline{\Delta\mathcal{L}} = \frac{1}{S}\sum_{s=1}^{S}\big(\mathcal{L}_{\text{val},s} - \mathcal{L}_{\text{train},s}\big)$$

Averaging the two losses separately and subtracting gives the same mean but
destroys the pairing, and with it any valid $\sigma$ for the gap. H1 is judged on
that dispersion, so the pairing is not cosmetic.

### 3.3 Layer-resolved aggregation

Entropy and patch-norm outlier rate are vectors over 12 layers, not scalars. The
engine reduces them **per layer across seeds**, producing $\mu$ and $\sigma$ for
each depth. Without this, the entropy figure has no error band and H3 cannot be
stated with a confidence claim.

### 3.4 Output schema

`outputs/sweep_summary.json`, keyed `k_0`, `k_1`, `k_4`, `k_8`. Each arm carries at
minimum the seven fields the table generator reads — `val_top1_mean/std`,
`val_loss_mean/std`, `entropy_mean/std`, `gen_gap_mean` — plus `gen_gap_std`, the
per-layer aggregates, and a `meta` block recording which of the twelve runs were
found and which are missing.

### 3.5 Robustness requirements

* Discovery matches the canonical `expXX_kY_sZ` name; foreign directories are ignored.
* An arm with no completed run is emitted zero-filled with `num_seeds: 0` rather
  than raising, so figures can be drafted mid-sweep.
* A missing or malformed `metrics.json` skips that run instead of aborting.
* CLI `python src/utils/logger.py --output_dir outputs/`, with `--strict` exiting
  non-zero while any of the twelve runs is absent — this is what makes a merge of
  parallel Kaggle slices verifiable rather than assumed.

---

## 🔬 4. Attention Extraction Engine (`scripts/visualize_attention.py`)

The mechanism claim is not a plot; it is a tensor operation that has to be exactly
right, because an off-by-$K$ slice silently produces a plausible but wrong picture.

1. **Load** a checkpoint into `RegisterVisionTransformer` with the run's own $K$
   (read from the checkpoint, not assumed).
2. **Capture** with `ViTAttentionHookManager`. Modern timm runs fused
   scaled-dot-product attention, so the $[B, H, S, S]$ matrix is never materialised
   — the hook reconstructs the softmax weights from the Q/K projections.
3. **Slice** the `[CLS]` query row against spatial keys only:
   $A^{(l)}_{\text{cls},\,1+K:} \in \mathbb{R}^{196}$. Index 0 is `[CLS]`, indices
   $1 \dots K$ are registers; the slice must start at $1+K$ or the register columns
   contaminate the map.
4. **Reduce** over heads, reshape $196 \to 14 \times 14$, bicubic upsample to
   $224 \times 224$.
5. **Overlay** on the source image (`inferno`, $\alpha = 0.6$).
6. **Compose** the comparative grid: top row $K=0$, bottom row $K=4$, columns
   layers 1, 4, 8, 12. Export `paper/figures/attention_maps_comparison.pdf`.

Both rows must use the same image and the same colour normalisation, otherwise the
comparison is not a comparison.

---

## 📊 5. Evidence Figures (`scripts/plot_metrics.py`)

| Figure | Content | Hypothesis |
| :--- | :--- | :---: |
| **F2** | Layer $1 \dots 12$ versus $\bar{H}^{(l)}$ (bits), four curves, $\pm 1\sigma$ bands | H3 |
| **F3** | $K$ versus $\Delta\mathcal{L}$ with error bars from `gen_gap_std` | H1 |
| **F4** | Train/validation loss curves, mean across seeds per arm, from `train_history.csv` | H1 |
| **F5** | $K$ versus Top-1 accuracy with $\sigma$ — the direct dilution test | H2 |

All exports are vector PDF into `paper/figures/`.

---

## 📐 6. Automated Results Table (`src/utils/export_latex.py`)

Reads `outputs/sweep_summary.json`, writes `paper/tables/results_table.tex` with
one row per arm: configuration, Top-1 accuracy, validation loss, generalization
gap, layer-12 entropy — each rendered `mean $\pm$ std`. Every number in the
manuscript's main table originates here; none is typed by hand, so re-running the
sweep regenerates the paper's numbers without manual transcription.

The generated table and figures are then wired into the experimental-setup and
results sections, which must state an explicit verdict on H2 at $K=8$.

---

## ✅ 7. Definition of Done & Verification Test

Your deliverable is complete when:

1. `python src/utils/logger.py --output_dir outputs/` writes
   `outputs/sweep_summary.json` and prints the per-arm summary table.
2. `pytest tests/test_sweep_aggregation.py` passes, covering run discovery, the
   reconstruction of `best_val_loss` / `final_train_loss` / `mean_layerwise_entropy`
   from nested records, mean and standard deviation against a NumPy reference, the
   layer-resolved aggregates, the seven-key table contract, and the zero-filled
   handling of an incomplete sweep.
3. `python src/utils/logger.py --output_dir outputs/ --strict` exits `0` once all
   twelve runs are present.
4. `python scripts/visualize_attention.py` writes the comparative overlay figure,
   with the spatial slice verified against a $K=0$ and a $K=4$ checkpoint.
5. `python scripts/plot_metrics.py` writes F2–F5 as vector PDFs.
6. `python src/utils/export_latex.py` writes `paper/tables/results_table.tex`, and
   its numbers match `sweep_summary.json` exactly.
