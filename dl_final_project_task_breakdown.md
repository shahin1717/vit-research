# DL Final Project — Task Breakdown, All Tracks

Compute budget reminder for every idea below: **~2 days scheduled training, ~10–12 GB peak GPU memory.** Use pretrained backbones + fine-tuning / LoRA everywhere — none of these should be trained from scratch.

Proposal is due **Sun 16 Aug** — pick one, confirm the dataset actually exists in the size you need, and lock scope before then.

---

## TRACK 1 — Pure Research

### 1. Do register tokens help small ViTs trained on limited data?
- **Data:** small image classification dataset (e.g. CIFAR-100 or a benchmark subset), ~10k+ images
- **Baseline:** small ViT trained from a pretrained checkpoint, standard fine-tune
- **Method:** add learned register tokens (per "Vision Transformers Need Registers"), compare attention-map artifacts with/without
- **Ablation:** vary number of register tokens (0, 1, 4, 8); measure both accuracy and attention-map quality (qualitative + a sparsity/entropy metric)
- **Deliverable:** attention-map visualizations pre/post, accuracy table, honest discussion if effect is small/absent

### 2. LoRA rank vs. catastrophic forgetting
- **Data:** 2–3 small sequential fine-tuning tasks (e.g. sentiment → NLI → topic classification), existing HF datasets
- **Baseline:** full fine-tune sequential training (expected to forget a lot)
- **Method:** LoRA fine-tune at multiple ranks (2, 8, 32, 128) through the same task sequence
- **Ablation:** rank sweep × ≥3 seeds; measure accuracy drop on task 1 after training on task 2/3
- **Deliverable:** rank-vs-forgetting plot, results table, discussion of where returns diminish

### 3. Curriculum ordering in low-data fine-tuning
- **Data:** small labeled dataset (few hundred–few thousand examples), any modality
- **Baseline:** random-order fine-tuning
- **Method:** define a difficulty heuristic (loss under a reference model, length, or human-labeled difficulty); train easy→hard vs. hard→easy vs. random
- **Ablation:** vary curriculum steepness/pacing; multiple seeds for significance
- **Deliverable:** learning curves per ordering, final accuracy table, discussion of when curriculum helps/hurts

### 4. Positional encoding extrapolation
- **Data:** synthetic or small existing sequence task with controllable length
- **Baseline:** small transformer with learned absolute positional encoding
- **Method:** compare RoPE vs. ALiBi vs. learned PE, train on short sequences only
- **Ablation:** test generalization to sequences 2x–4x longer than training length
- **Deliverable:** length-vs-performance plot per PE scheme, clear write-up of which extrapolates best and why

### 5. Induction heads in a synthetic-language toy model
- **Data:** synthetic sequence task designed to require in-context copying/pattern completion
- **Baseline:** none needed in the usual sense — this is interpretability, so the "baseline" is the untrained/early-training model
- **Method:** train a tiny transformer from scratch (small enough to fit budget) on the synthetic task
- **Ablation/analysis:** probe attention heads for induction-head behavior at different training checkpoints; correlate emergence with a loss "phase transition" if one appears
- **Deliverable:** head-by-head visualization, training-dynamics plot, discussion tying to existing interpretability literature

---

## TRACK 2 — Applied / Domain Research

### 1. Mugham dastgah (mode) classification from audio ⭐ (no existing DL baseline found)
- **Data:** mugham recordings labeled by dastgah/mode — source from state archives, khanende performer channels, Azerbaijan State Radio; verify licensing; segment into clips; split by *performer* to avoid leakage
- **Baseline:** MFCC features + classical classifier (SVM/RF), or a small CNN on spectrograms trained from scratch
- **Method:** fine-tune a pretrained audio model (Wav2Vec2, AST, or similar small variant)
- **Evaluation:** per-mode precision/recall, confusion matrix (which modes get confused and why — tie back to musicological similarity), class imbalance handling
- **Deliverable:** confusion matrix + discussion of practical implications (archiving, education, preservation)
- **Risk flag:** confirm real data volume before the proposal deadline; have a fallback (fewer modes / explicit low-resource framing per §3) ready

### 2. Azerbaijani carpet weaving-school classification ⭐ (only generic/old carpet CV work found)
- **Data:** carpet images labeled by region/school (Shirvan, Karabakh, Ganja, Guba, etc.) — museum archives, UNESCO materials, Carpet Museum resources; check licensing carefully
- **Baseline:** classical ML on color/texture features (matches what little prior work exists, e.g. KNN/SVM on ~200 images) — beat this cleanly
- **Method:** fine-tune a pretrained CNN or ViT backbone
- **Evaluation:** per-region confusion matrix, not just top-1 accuracy; check for regional class imbalance
- **Deliverable:** region confusion analysis, discussion of visually similar/confusable schools

### 3. Register/dialect classifier for Azerbaijani (text or speech) ⭐ (unstudied)
- **Data:** text — formal (news, official docs) vs. colloquial (social media, forum) sources; or speech — regional accent samples if available
- **Baseline:** bag-of-words/TF-IDF + classical classifier
- **Method:** fine-tune a small multilingual/Turkic-aware pretrained LM (text) or audio model (speech)
- **Evaluation:** per-class metrics, error analysis on borderline/mixed-register examples
- **Deliverable:** discussion of linguistic implications, practical use cases (content moderation, education tools)

### 4. Historical Azerbaijani Cyrillic → Latin OCR/handwriting recognition
- **Data:** ≥2k annotated scanned document/image samples (detection/seg. floor per §3); check archival access and rights
- **Baseline:** existing generic OCR tool (Tesseract or similar) run as-is
- **Method:** fine-tune a pretrained OCR/vision-language model on the historical script
- **Evaluation:** character/word error rate vs. baseline; per-document-quality breakdown (clean vs. degraded scans)
- **Deliverable:** error-rate table, qualitative examples of failure cases, discussion of archival value

### 5. Crop-disease detection on regional crops (pomegranate, hazelnut, saffron)
- **Data:** ≥10k images across ≥5 classes if using an established benchmark subset, or a smaller justified regional dataset (explicit justification required in proposal if below floor)
- **Baseline:** generic pretrained image classifier (e.g. ImageNet backbone, no fine-tuning) as a naive baseline
- **Method:** fine-tune a pretrained CNN/ViT on regional crop images
- **Evaluation:** per-disease and per-crop metrics, calibration (this matters for a real deployment use case — false negatives on disease are costly)
- **Deliverable:** cost-sensitive error discussion, practical implications for farmers/extension services

### 6. Azerbaijani NER / text classification with a modern transformer
- **Data:** existing Azerbaijani NER corpora (thin but present — see awesome-azerbaijani-nlp resources) or the 2024 English-Azerbaijani parallel corpus adapted for a classification task
- **Baseline:** existing morphological analyzer / classical CRF-based NER if available, or majority-class baseline
- **Method:** fine-tune a pretrained multilingual transformer (e.g. XLM-R) on the Azerbaijani NER task
- **Evaluation:** entity-level F1, per-entity-type breakdown
- **Deliverable:** honest note that base NER itself is less novel now — strengthen novelty by adding a harder angle (cross-domain generalization, or few-shot entity types)
- **Note:** weakest novelty of the Track 2 options since NER work already exists — only pick this if you add a genuinely new angle

---

## TRACK 3 — Industry Product

### 1. Azerbaijani AI-generated-text detector ⭐ (no existing tool found)
- **Data:** human-written Azerbaijani text (news, forums, student writing) + AI-generated text from multiple models/prompts as the positive class; split by source domain to test generalization
- **Baseline:** perplexity-based detection or classical stylometric features + classifier
- **Model:** fine-tune a small transformer classifier
- **Product build:** functioning demo — paste text, get a score, live at defense
- **Deployment:** Dockerfile or runnable API; benchmark inference latency and memory on modest hardware (CPU-only target)
- **Evaluation:** precision/recall, and critically — false-positive rate on human-written text (a detector that flags real students' work is a real product risk, discuss this)
- **Deliverable:** deployment note (latency, model size, memory/cost) as required by the brief for Track 3

### 2. Deepfake / voice-clone audio detector
- **Data:** real speech samples + AI-cloned/synthesized speech (multiple TTS/voice-clone tools) as positive class
- **Baseline:** simple spectral-feature classifier
- **Model:** fine-tune a pretrained audio classifier (Wav2Vec2 or similar) for real-vs-cloned detection
- **Product build:** live demo — play a clip, get a verdict, very strong defense moment
- **Deployment:** package for CPU inference, benchmark latency
- **Evaluation:** precision/recall/EER (equal error rate — standard for this task), robustness check against unseen cloning tools if time allows
- **Deliverable:** deployment note, discussion of generalization limits (cloning tools evolve fast)

### 3. OCR + structuring assistant for scanned Azerbaijani official forms
- **Data:** ≥2k annotated scanned form/document images (detection/seg. floor); real or synthetically generated forms if real ones are hard to source
- **Baseline:** raw OCR output with no structuring
- **Model:** fine-tune an OCR/document-understanding pretrained model; add a structuring step (regex/rules or a small seq2seq for field extraction)
- **Product build:** upload a scan, get structured fields out — functioning demo end to end on new input
- **Deployment:** Dockerfile, latency benchmark on CPU
- **Evaluation:** field-level extraction accuracy, error breakdown by scan quality
- **Deliverable:** deployment note; discuss real user (admin staff) and workflow fit

### 4. On-device mushroom/plant safety identifier
- **Data:** ≥10k images across ≥5 classes (edible vs. toxic look-alikes common in the region) — check for an existing benchmark subset (e.g. iNaturalist-derived) to reduce collection burden
- **Baseline:** generic pretrained classifier with no fine-tuning
- **Model:** fine-tune a small, efficient backbone (MobileNet/EfficientNet-lite class) explicitly for on-device size
- **Product build:** simple UI (mobile-style or web) that classifies a photo and gives a clear safety warning
- **Deployment:** must run on CPU/consumer GPU — this is the whole point; report model size and latency prominently
- **Evaluation:** per-class accuracy with a strong emphasis on toxic-class recall (false negatives on "toxic" are the dangerous failure mode — discuss explicitly)
- **Deliverable:** deployment note, clear discussion of safety limitations (this is not a medical/safety-certified tool — say so)

### 5. Semantic search over a niche corpus (e.g. legal codes or lecture transcripts)
- **Data:** a specific corpus nobody else has indexed (pick one with real access — university course transcripts, a specific legal code, etc.)
- **Baseline:** keyword/TF-IDF search
- **Model:** pretrained embedding model (fine-tune or use off-the-shelf) + vector search index
- **Product build:** functioning search API/UI that returns relevant passages for a query
- **Deployment:** Dockerfile or runnable API, latency benchmark
- **Evaluation:** retrieval quality (precision@k, or human-judged relevance on a query set you construct)
- **Deliverable:** deployment note; discuss real user and use case clearly since novelty matters less than usefulness on this track

---

## Quick comparison for decision-making

| Idea | Novelty confirmed | Data risk | Demo strength | Overall effort |
|---|---|---|---|---|
| T1: LoRA rank forgetting | Medium | Low | Low | Low |
| T1: Register tokens | Medium | Low | Low | Low-Med |
| T2: Mugham classification | **High** | **High** | Medium | Med-High |
| T2: Carpet classification | **High** | Medium | Medium | Medium |
| T2: Register/dialect classifier | **High** | Medium | Low | Medium |
| T3: AI-text detector | **High** | Low | **High** | Medium |
| T3: Deepfake audio detector | Medium-High | Medium | **High** | Medium |
| T3: Mushroom identifier | Medium | Low | High | Medium |

Everything marked ⭐ above was checked against current search results and had no or minimal existing deep-learning work — genuinely open ground as of today.
