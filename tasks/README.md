# 👥 Team Task Packages & Execution Guides

**Project:** *Do Register Tokens Regularize Vision Transformers Under Data Scarcity?*  
**Course:** DLE-AI-202 (Deep Learning), Cohort I 2026 — Track 1: Pure Research  
**Submission Deadline:** Monday, 7 September 2026, 23:59 | Oral Defense: Week of Sep 8, 2026  
**Repository Root:** `/home/shahin/aiac-res/` | **Vault Root:** `C:/Vaults/aiac-res/`  

---

## 🧭 Team Responsibility Index

Each team member has a dedicated, deep-dive task specification markdown file containing mathematical context, exact API contracts, step-by-step implementation blueprints, failure modes, and verification tests:

| Member | Role & Ownership | Dedicated Task File | Key Deliverables & Code |
| :--- | :--- | :--- | :--- |
| **Gulnisa** | **Lead Data Engineer** | [`tasks/gulnisa_data_pipeline.md`](file:///home/shahin/aiac-res/tasks/gulnisa_data_pipeline.md) | `src/data/cifar100_subset.py`<br>`src/data/__init__.py` |
| **Narmina** | **Lead Metrics & Hooks Engineer** | [`tasks/narmina_metrics_and_hooks.md`](file:///home/shahin/aiac-res/tasks/narmina_metrics_and_hooks.md) | `src/models/attention_hook.py`<br>`src/metrics/*.py` |
| **Shahin** | **Core Architecture & Training Lead** | [`tasks/shahin_core_architecture_and_training.md`](file:///home/shahin/aiac-res/tasks/shahin_core_architecture_and_training.md) | `src/models/register_vit.py`<br>`scripts/train.py`, `scripts/eval.py` |
| **Emil** | **Sweep Orchestration & Operations** | [`tasks/emil_sweep_orchestration.md`](file:///home/shahin/aiac-res/tasks/emil_sweep_orchestration.md) | `scripts/run_sweep.sh`<br>`configs/*.yaml`, `src/utils/logger.py` |
| **Rufet** | **Visualizations & LaTeX Paper Lead** | [`tasks/rufet_visualization_paper_presentation.md`](file:///home/shahin/aiac-res/tasks/rufet_visualization_paper_presentation.md) | `scripts/visualize_attention.py`<br>`paper/`, `presentation/` |

---

## 📅 Milestones & Integration Timeline

```text
Team Timeline (Aug 31 - Sep 7):
├── Phase 1 (Aug 31 - Sep 1) : Gulnisa (Data) & Narmina (Metrics) & Shahin (Model) complete unit tests
├── Phase 2 (Sep 1 - Sep 3)  : Emil runs 12-experiment sweep matrix (A100 MIG / Kaggle)
├── Phase 3 (Sep 3 - Sep 5)  : Rufet generates attention heatmaps, entropy curves, results tables
├── Phase 4 (Sep 5 - Sep 7)  : Team completes LaTeX paper draft & oral defense slide deck
└── Phase 5 (Mon 7 Sep 23:59): Final Submission Due
```
