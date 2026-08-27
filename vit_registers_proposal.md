# Do Register Tokens Regularize Vision Transformers Under Data Scarcity?

**Track:** 1 — Pure Research
**Course:** DLE-AI-202, Cohort I 2026
**Team:** _[names here]_

---

## 1. Research Question

Darcet et al. (2023), *"Vision Transformers Need Registers,"* showed that adding learnable register tokens removes high-norm attention artifacts in **large-scale** models (DINOv2, OpenCLIP) trained on millions of images. The mechanism was demonstrated, but never tested under the opposite regime: **a small model with too little data to comfortably fit.**

We ask: **when a ViT is starved of data, do register tokens still just clean up attention — or do they also act as a structural regularizer that reduces overfitting?**

## 2. Why This Is Still Open

Three existing works sit close to this question, and we've checked each one against our plan:

| Prior work | What it does | Why it doesn't cover our question |
|---|---|---|
| Darcet et al., 2023 (original) | Introduces registers, shows clean attention maps qualitatively, at scale | Large models, large data, no overfitting/regularization framing |
| *Registers in Small ViTs* (reproducibility study, 2025) | Re-tests registers on a small ViT (DeiT-III Small) | Asks whether artifacts *appear* at small scale — does not test low-data regimes or generalization gap |
| Jiang et al., 2025 (NeurIPS), *"ViTs Don't Need Trained Registers"* | Compares trained vs. training-free registers, head-to-head | Compared on DINOv2/OpenCLIP at full scale — not small models under data scarcity |

No prior work frames registers as a **regularizer** or quantifies their effect with **attention entropy under a strict low-data budget**. That combination is our contribution.

## 3. What We Will Do Differently

- **Scale shift:** ViT-Tiny / ViT-Small trained on a strict low-data subset (~10k images, CIFAR-100), not a foundation-scale pretraining run.
- **Reframing the mechanism:** test registers as a *regularizer against overfitting* under data scarcity, not just an attention-cleanup trick.
- **Downstream framing:** measure whether registers speed up convergence inside a tight, fixed 2-day training window.

## 4. What We Will Quantitatively Prove

1. **Quantifying the artifacts.** The original paper showed clean attention maps qualitatively. We compute an **attention entropy** and/or **sparsity index** per layer, and test whether background-token spikes drop *systematically and measurably* when registers are added — not just visually.
2. **The capacity threshold.** Sweep register count ∈ {0, 1, 4, 8}. Map the point where more registers stop helping and start hurting — the hypothesis being that in a small model, too many registers dilute the token dimension's limited capacity.
3. **Generalization gap.** Track train/val loss delta across the register-count sweep. Test whether registers narrow this gap — i.e., whether they function as a regularizer, not just an interpretability fix.

## 5. Method

- **Model:** ViT-Tiny or ViT-Small (timm), initialized from a small pretrained checkpoint where reasonable, or trained from scratch given the model is small enough to fit the compute budget either way — decide based on early pilot runs.
- **Baseline:** identical architecture, register count = 0, standard training recipe.
- **Treatment arms:** register count ∈ {1, 4, 8}, all else held constant.
- **Seeds:** ≥3 per arm, for statistical confidence on entropy/accuracy differences.
- **Metrics logged:** top-1 accuracy, train/val loss curves, per-layer attention entropy, patch-norm outlier rate.

## 6. Dataset

- **Dataset:** CIFAR-100 subset, ~10,000 images (deliberately below the course's ≥10k image floor as a *low-resource study* — explicitly justified here per §3 of the brief, since data scarcity is the independent variable, not an oversight).
- **License/source:** CIFAR-100 is publicly available for research use (Krizhevsky, 2009).
- **Splits:** fixed train/val/test, stratified by class; test set touched once for final numbers only.

## 7. Baseline & Reproducibility

- **Baseline to beat/compare:** register-free ViT-Tiny/Small trained identically — this isn't about beating SOTA, it's the controlled comparison the track requires.
- **Reproducibility:** fixed seeds, pinned `requirements.txt`, single `run_all` entry point reproducing headline numbers, checkpoints logged to NVMe scratch per §2.1 of the brief.

## 8. Compute Plan (fits budget)

- ViT-Tiny/Small at CIFAR resolution (32×32, or upscaled modestly) is far below the ~10–12 GB ceiling — plenty of headroom for running the full {0,1,4,8} × ≥3-seed sweep within the ~2-day window.
- Mixed precision (AMP) used regardless, to leave margin for larger batch sizes and faster sweeps.
- Pipeline correctness verified on a small subset **before** the booked GPU window is used for the full sweep, per course guidance.

## 9. Timeline

| Date | Milestone |
|---|---|
| Sun 16 Aug | Proposal submitted (this document) |
| Mon 24 Aug | Data pipeline running end-to-end on subset; register-free baseline trained |
| Fri 28 Aug | Full register-count × seed sweep complete; entropy/sparsity metrics computed |
| Mon 31 Aug | Generalization-gap analysis + capacity-threshold plot done; first draft figures |
| Sep 1–6 | Paper writing, slides, ablation write-up, contribution report |
| Mon 7 Sep, 23:59 | Final submission — code, paper, slides |
| Week of 8 Sep | Oral defense |

## 10. Risks & Mitigations

- **Risk:** effect size may be small or noisy at this scale. **Mitigation:** the track explicitly accepts a well-documented negative result — a clean "registers don't meaningfully regularize small ViTs" finding is a valid, honest outcome if that's what the data shows.
- **Risk:** entropy/sparsity metric choice affects interpretation. **Mitigation:** report both an entropy-based and a norm-based (outlier rate) metric, so the claim isn't hostage to one measurement choice.

## 11. Deliverables Checklist (cross-track requirements)

- [ ] Baseline (register-free) vs. treatment arms, clearly compared
- [ ] Fixed train/val/test split, no test-set leakage
- [ ] Quantitative result table (accuracy × register count × seed) + matching plots
- [ ] Ablation isolating register count as the sole changed variable
- [ ] Fixed seeds, pinned environment, one-command reproduction
