# 🎙️ Oral Defense Presentation Script (Balanced Multi-Turn Flow)
**Course:** DLE-AI-202 (Deep Learning), Cohort I 2026 — Track 1: Pure Research  
**Project:** *Do Register Tokens Regularize Vision Transformers Under Data Scarcity? A Controlled 24-Run Empirical Ablation on Low-Data CIFAR-100*  
**Deck Source:** [`presentation/slides.tex`](file:///home/shahin/aiac-res/presentation/slides.tex) | **Target Duration:** 13–15 Minutes Total  

---

### 📋 Overview of Balanced Speaker Roles

| Part | Speaker | Official Role | Slides Covered | Target Time |
|:---:|---|---|:---:|:---:|
| **1** | **Shahin Alakparov** *(Starts)* | Core Architecture & Training Lead | **Slides 1 – 4** | ~3:00 |
| **2** | **Gulnisa Abdurahmanli** | Data Engineering & Multi-Budget Lead | **Slides 5 – 6** | ~2:00 |
| **3** | **Narmina Ibrahimova** | Metrics & Interpretability Lead | **Slides 7 – 8** | ~2:15 |
| **4** | **Emil Ahmedli** | Ablation Sweeps & Hardware Lead | **Slides 9 – 10** | ~2:30 |
| **5** | **Rufet Dosteliyev** | Ablation Analytics, Figures & Paper Lead | **Slide 11** | ~1:30 |
| **6** | **Narmina Ibrahimova** | Metrics & Interpretability Lead | **Slide 12** | ~1:15 |
| **7** | **Gulnisa Abdurahmanli** | Data Engineering & Multi-Budget Lead | **Slide 13** | ~1:15 |
| **8** | **Shahin Alakparov** *(Closes)* | Core Architecture & Training Lead | **Slides 14 – 16** | ~2:30 |
| **—** | **All Team Members** | Committee Examination | **Slide 17 (Q&A)** | Open |

---

## 👤 Turn 1: Shahin Alakparov (Opening, Context & Hypotheses)
**Slides Covered:** 1, 2, 3, 4 | **Allocated Time:** ~3:00  

---

### [SLIDE 1: Title Page]
*(Wait for committee attention. Speak with calm, authoritative confidence.)*

> "Good morning, members of the evaluation committee and colleagues. 
> 
> Today, our team is presenting our empirical research study: **'Do Register Tokens Regularize Vision Transformers Under Data Scarcity? A Controlled 24-Run Empirical Ablation on Low-Data CIFAR-100.'**
> 
> Over the past several weeks, our group investigated an open question in transformer representation learning: whether register tokens—originally proposed by Darcet and colleagues at ICLR 2024 to clean up spatial attention maps in foundation models—function as an architectural regularizer when models are starved of training data."

---

### [ADVANCE TO SLIDE 2: Team Task Distribution & Course §6 Ownership]

> "Before we unpack the machine learning problem, I want to briefly outline our team's division of responsibility under Course Section 6.
> 
> Our research was executed across five tightly integrated, modular engineering layers, with equal effort across all members:
> - I led the core PyTorch model architecture, the zero-register invariant wrapper, and the AMP training harness.
> - **Gulnisa Abdurahmanli** engineered our multi-budget stratified data samplers and cross-split isolation protocols.
> - **Narmina Ibrahimova** authored our non-invasive attention hooks and diagnostic interpretability metrics.
> - **Emil Ahmedli** configured our remote NVIDIA A100 GPU cluster environment and orchestrated all 24 sweep runs.
> - And **Rufet Dosteliyev** led our statistical analytics engine, publication figure pipelines, and LaTeX manuscript authoring.
> 
> We will each present our respective technical domains during this defense."

---

### [ADVANCE TO SLIDE 3: 01. The Question]

> "Let us begin with the core observation that motivated our research.
> 
> In standard Vision Transformers, attention heads quietly vandalize their own spatial representations. In ICLR 2024, Darcet and colleagues demonstrated that in models like DINOv2 and DeiT, a small handful of background patch tokens consistently acquire massive activation norms. When you inspect the self-attention maps, the `[CLS]` token routes an enormous portion of its attention budget directly to these empty background patches.
> 
> Why does this happen? The network is essentially repurposing uninformative image regions as scratch space to store and compute global context. Darcet's proposed fix was remarkably simple: prepending dedicated, learnable scratchpad tokens—called **register tokens**—to the sequence. The model offloads its scratchpad computations to the registers, and the spatial attention artifacts disappear.
> 
> Here is the critical gap in that narrative:
> Darcet et al. demonstrated this phenomenon exclusively in massive foundation models trained on hundreds of millions or billions of images. But if register tokens absorb representational capacity that the model would otherwise waste on memorizing artifacts, they possess the structural properties of a **regularizer**.
> 
> And in machine learning, a regularizer only proves its worth when data is scarce and severe overfitting bites.
> 
> This led directly to our research question: **When a Vision Transformer is starved of data, do register tokens regularize the model, shrink the generalization gap, and produce superior held-out test performance?**"

---

### [ADVANCE TO SLIDE 4: Three Hypotheses, Each Falsifiable]

> "To answer this question rigorously, we formulated three concrete, falsifiable hypotheses:
> 
> 1. **Hypothesis 1: The Regularization Hypothesis.** We hypothesized that register tokens would act as an architectural bottleneck, shrinking the generalization gap—defined as validation loss minus training loss ($\Delta\mathcal{L} = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$)—and lowering validation loss.
> 2. **Hypothesis 2: The Capacity Dilution Hypothesis.** In a compact model like ViT-Tiny with embedding dimension $d=192$, allocating attention capacity to multiple register tokens might dilute the Softmax distribution over true visual tokens. We hypothesized that for large $K$, accuracy would plateau or degrade.
> 3. **Hypothesis 3: The Entropy Collapse Hypothesis.** We hypothesized that without registers, severe data starvation would cause catastrophic entropy collapse in deep attention layers and a surge in high-norm patch outliers—both of which registers would systematically prevent.
> 
> Nobody has tested registers as a regularizer under strict data scarcity. A clean, controlled empirical answer is valuable whichever way it falls.
> 
> To explain how we designed our data environment to isolate this effect with zero leakage, I will now pass the floor to Gulnisa."

---

## 👤 Turn 2: Gulnisa Abdurahmanli (Data Environment & Control)
**Slides Covered:** 5, 6 | **Allocated Time:** ~2:00  

---

### [SLIDE 5: 02. Starving the Model]
*(Acknowledge Shahin with a nod; speak clearly about experimental discipline.)*

> "Thank you, Shahin. I am Gulnisa Abdurahmanli, and I led the data engineering, multi-budget sampling pipelines, and experimental control protocols.
> 
> To test whether register tokens act as a regularizer, we had to deliberately starve the model of training data so that overfitting would actively manifest.
> 
> We constructed two controlled low-data regimes on CIFAR-100:
> - First, our **primary benchmark of 100 images per class**, providing 10,000 total images partitioned into a strict 9,000 training, 1,000 validation, and the standard 10,000 held-out test set. At 9,000 images, a Vision Transformer suffers intense data scarcity.
> - Second, an **extension budget of 300 images per class**—27,000 training images—which I will discuss later to verify scale invariance.
> 
> Every image is upsampled via Bicubic interpolation to $224 \times 224$ pixels. This ensures the model operates on the standard $14 \times 14$ patch grid with patch size 16, yielding exactly 196 spatial tokens.
> 
> For training, we apply RandomResizedCrop, horizontal flipping, and AutoAugment CIFAR-10 policy. For validation and testing, we use deterministic resizing and center cropping."

---

### [ADVANCE TO SLIDE 6: One Thing Moves, and Only One]

> "The central methodological challenge in small-sample deep learning is random noise.
> 
> Across three random seeds, the between-seed variation in validation accuracy was **2.17 percentage points**. In a subtle ablation study, 2.17 pp of random noise would completely drown out any genuine architectural effect.
> 
> To overcome this, we enforced strict **within-seed pairing**:
> 
> Within each random seed:
> - All four register arms—$K \in \{0, 1, 4, 8\}$—see the exact same subset of training images.
> - They see them in the exact same mini-batch order.
> - They use the exact same 50-epoch schedule, AdamW optimizer ($lr = 5 \times 10^{-4}$), linear warmup, cosine decay, and batch size 64.
> 
> The *only* element that moves in our study is $K$, the number of register tokens.
> 
> Because the arms are strictly matched within each seed, we can compute paired differences:
> $$\bar{\delta} = X_K - X_0$$
> This allows us to subtract the 2.17 pp between-seed noise completely out of the equation and evaluate the pure marginal effect of register tokens with high statistical power.
> 
> I will now hand over to Narmina to explain our model architecture and diagnostic hooks."

---

## 👤 Turn 3: Narmina Ibrahimova (Architecture & Diagnostic Hooks)
**Slides Covered:** 7, 8 | **Allocated Time:** ~2:15  

---

### [SLIDE 7: 03. Registers & Instrumentation]
*(Step forward confidently; transition smoothly into model internals.)*

> "Thank you, Gulnisa. I am Narmina Ibrahimova, and I led the model instrumentation, attention hooks, and diagnostic metrics.
> 
> Let us look at how register tokens are sequenced inside our `RegisterVisionTransformer`.
> 
> As shown in the diagram on Slide 7, the input sequence is formed as:
> $$\text{[CLS]} \parallel [r_1, \dots, r_K] \parallel [p_1, \dots, p_{196}]$$
> 
> The $K$ register tokens are learnable vectors in $\mathbb{R}^{K \times d}$. Following Darcet, they receive **no positional embeddings**. This ensures they have no spatial bias and can move freely in sequence space.
> 
> Registers attend to and are attended by all tokens across all 12 multi-head self-attention blocks. Prior to the linear classification head, they are completely discarded—only the CLS token is passed forward.
> 
> At $K=8$, the parameter cost is tiny: just 1,536 parameters out of 5.54 million, a relative increase of less than 0.03%.
> 
> However, this introduces a dangerous engineering trap: **the off-by-$K$ spatial slicing error**.
> If an engineer slices spatial patch features from index 1 instead of $1+K$, register tokens leak into the spatial feature map. The code will execute without crashing and produce plausible-looking heatmaps. To prevent this, our test suite contains explicit shape-assertion tests verifying slice boundaries across all 12 layers."

---

### [ADVANCE TO SLIDE 8: Two Diagnostic Mechanism Readouts]

> "To understand what happens inside the network, accuracy alone is insufficient; we needed to inspect the internal self-attention mechanics.
> 
> We engineered two non-invasive diagnostic readouts:
> 
> 1. **Layer-wise Shannon Attention Entropy** $\bar{H}^{(l)}$:
>    $$\bar{H}^{(l)} = -\frac{1}{BHS} \sum_{b,h,i,j} A_{b,h,i,j}^{(l)} \log_2 A_{b,h,i,j}^{(l)}$$
>    In modern PyTorch, attention is executed via fused C++ kernels that do not store intermediate attention weights. Our `ViTAttentionHookManager` intercepts the Query and Key projections, reconstructs the full $[B, H, S, S]$ softmax attention matrix, computes the Shannon entropy across all heads, and immediately clears the tensor to prevent memory leaks. If attention sinks emerge, entropy collapses.
> 
> 2. **Patch-Norm Outlier Rate** $\rho^{(l)}$:
>    We compute the fraction of spatial patch tokens whose $L_2$ norm exceeds the layer mean plus three standard deviations ($\mu + 3\sigma$) on the residual stream, evaluated before LayerNorm.
> 
> To guarantee total parity, both readouts are evaluated across all checkpoints using the exact same frozen probe of 64 test images.
> 
> I will now pass the floor to Emil to discuss our cluster infrastructure, sweep execution, and our primary empirical findings."

---

## 👤 Turn 4: Emil Ahmedli (Cluster Execution & Primary Sweep Results)
**Slides Covered:** 9, 10 | **Allocated Time:** ~2:30  

---

### [SLIDE 9: 04. Sweep Execution on NVIDIA A100]
*(Deliver with an operational, systems-oriented authority.)*

> "Thank you, Narmina. I am Emil Ahmedli, and I led the hardware operations, cluster configuration, and sweep execution.
> 
> Executing a controlled empirical ablation across multiple seeds and data budgets requires high operational rigor.
> 
> Our study comprises a full **24-run experimental matrix**:
> - 12 runs on the primary 100 images/class budget across all 4 register arms ($K \in \{0, 1, 4, 8\}$) and 3 random seeds (42, 1337, and 3407).
> - 12 runs on the 300 images/class scaling extension under the identical 4 arms and 3 seeds.
> 
> To execute this, we configured a secure WireGuard VPN tunnel (`team1.conf`) connecting into our remote NVIDIA A100-SXM4 GPU cluster, operating within an isolated 20GB MIG compute slice.
> 
> I authored automated execution harnesses—`scripts/run_sweep.sh` and `scripts/run_databudget_sweep.sh`—incorporating dynamic GPU memory clearing, automated checkpoint serialization, and per-epoch telemetry logging.
> 
> The cluster completed all 24 runs—representing over 8 hours of continuous GPU compute—with **100% execution success, zero out-of-memory errors, and zero crashes**."

---

### [ADVANCE TO SLIDE 10: 05. Main Results: 100 Images/Class]

> "Now, let us examine the core empirical results generated by our 24 sweep runs on Slide 10.
> 
> Immediately, a remarkable pattern appears right down the middle of the table:
> - Looking at the first two columns—**held-out Test Accuracy and Test Loss**—the baseline model with **zero registers ($K=0$) takes first place**, achieving $75.19\%$ accuracy and a test loss of $1.6357$.
> - But looking at the last two columns—**Validation Loss and the Generalization Gap $\Delta\mathcal{L}$**—the $K=4$ configuration appears to win, showing the lowest validation loss of $1.6492$ and the smallest generalization gap of $0.7680$.
> 
> When we examine the paired comparisons across seeds, $K=1$ and $K=8$ lose test accuracy in **3 out of 3 seeds**. $K=4$ edges ahead in only 1 of 3 seeds, with a mean delta of $-0.13$ pp and a non-significant $p$-value of $0.466$.
> 
> On held-out test data, registers cost between zero and half a point of accuracy; they gain nothing.
> 
> To explain why validation loss dropped while test loss stayed flat, Rufet will present our statistical analysis of the generalization gap."

---

## 👤 Turn 5: Rufet Dosteliyev (The Generalization Gap Discovery)
**Slides Covered:** 11 | **Allocated Time:** ~1:30  

---

### [SLIDE 11: The Finding: A Regularizer on One Split Only]
*(Speak with analytical precision; guide the committee through the statistical finding.)*

> "Thank you, Emil. I am Rufet Dosteliyev, and I led the statistical analytics, visualization pipelines, and the writing of our paper.
> 
> Emil just showed that validation loss fell while test loss remained flat. This brings us to the central scientific discovery of our research: **registers act as a regularizer on one split only**.
> 
> Look at the figure on the left and the statistical breakdown on the right of Slide 11:
> When comparing $K=4$ against control across all three seeds:
> - On the validation split, validation loss falls by $-0.0337$, consistent across all three seeds, yielding a statistically significant $p = 0.024$. The generalization gap shrinks with $p = 0.051$.
> - But on the held-out test set, that effect completely vanishes: $\bar{\delta} = +0.0018$, with a non-significant $p = 0.823$.
> 
> Why does this regularization signal collapse?
> Look at the final training losses across arms: $0.8782$ for $K=0$, $0.8780$ for $K=1$, $0.8812$ for $K=4$, and $0.8829$ for $K=8$. That is a total spread of just 0.005.
> 
> In machine learning, a true regularizer trades away training fit in exchange for superior held-out generalization. Here, training loss never moved. **Nothing was traded.**
> 
> The apparent regularization on validation was simply an artifact of checkpoint selection picking the minimum over 50 noisy epochs on a small 1,000-image split.
> 
> Now, Narmina will take us through the internal mechanism checks on entropy and patch outliers."

---

## 👤 Turn 6: Narmina Ibrahimova (Mechanism Checks)
**Slides Covered:** 12 | **Allocated Time:** ~1:15  

---

### [SLIDE 12: Mechanism Checks: H2 & H3]
*(Step in with diagnostic authority; interpret the curves.)*

> "Thank you, Rufet. Let us now examine whether the internal attention mechanics match our initial hypotheses on Slide 12.
> 
> First, **Hypothesis 2 (Capacity Dilution) is strongly supported.**
> Prepending 8 registers ($K=8$) produced the only statistically significant held-out degradation in the entire study: test loss increased by $+0.0203$ in **3 out of 3 seeds** with $p = 0.014$. Furthermore, final-block attention entropy dropped across all 3 seeds. In a compact 192-dimensional model, forcing attention to allocate probability mass across 8 content-free tokens measurably taxes model capacity.
> 
> Second, **Hypothesis 3 (Entropy Collapse) is not supported.**
> As shown in the left figure, the layer-wise entropy curves lie directly on top of each other ($\Delta H \approx 0.001$ bits). Furthermore, the outlier rate actually rises slightly from $1.227\%$ to $1.254\%$. Registers simply do not absorb artifacts in this regime.
> 
> To test whether this null result was merely an artifact of extreme starvation at 100 images per class, Gulnisa will now present our 300 images per class scaling extension."

---

## 👤 Turn 7: Gulnisa Abdurahmanli (Scale Invariance Extension)
**Slides Covered:** 13 | **Allocated Time:** ~1:15  

---

### [SLIDE 13: 06. Scale Invariance: 100 vs. 300 Images/Class]
*(Present the scaling results confidently.)*

> "Thank you, Narmina. A natural question from any reviewer is: could this null result be an artifact of extreme data starvation at 100 images per class?
> 
> To resolve this, we tripled the data budget to **300 images per class**—27,000 training images.
> 
> As you can see in the vector plot on Slide 13, tripling the training data caused baseline accuracy to surge by **+6.14 percentage points**, from $75.19\%$ to $81.33\%$.
> 
> Yet, under this expanded data budget, registers still provided **zero advantage**:
> - Control ($K=0$) achieved $\mathbf{81.33\%}$.
> - $K=1$ and $K=4$ reached $81.23\%$.
> - $K=8$ reached $81.02\%$.
> 
> All pairwise differences remain statistically non-significant ($p > 0.50$).
> This proves that our null result is **scale-invariant** across low-data budgets.
> 
> Finally, Shahin will present our bonus test-time experiment and conclude the presentation."

---

## 👤 Turn 8: Shahin Alakparov (Bonus Experiment, Interpretation & Closing)
**Slides Covered:** 14, 15, 16, 17 | **Allocated Time:** ~2:30  

---

### [SLIDE 14: 07. Bonus Experiment: Registers Without Training]

> "Thank you, Gulnisa. We also conducted an exploratory bonus experiment testing a cutting-edge question: **do registers even need to be trained?**
> 
> In NeurIPS 2025, Jiang et al. showed that on large foundation models, one can identify outlier-producing neurons and redirect their activation into an empty, untrained token slot at test time with zero training.
> 
> We implemented Jiang et al.'s redirection mechanism against our $K=0$ checkpoints:
> - Baseline accuracy is $75.18\%$.
> - Adding a passive, empty slot yields $75.25\%$ ($+0.07$ pp).
> - Actively redirecting outlier neurons in Block 1 or Block 9 yields $75.25\%$ and $75.24\%$.
> 
> Active redirection does not outperform the passive inert slot. This confirms that attention sinks do not bottleneck classification accuracy in compact Vision Transformers."

---

### [ADVANCE TO SLIDE 15: Interpretation: A Scope Result, Not a Refutation]

> "We want to be very precise about how to interpret our findings.
> 
> This is a **scope result, not a refutation of Darcet et al.**
> 
> In Darcet's work, registers are pre-trained from scratch across hundreds of millions of images. In our setting, registers are introduced to an ImageNet-pretrained backbone during fine-tuning. Over 50 epochs on low data, the global attention routing never reorganizes.
> 
> Heavy-tailed outliers are genuinely present—our control outlier rate of $1.23\%$ is 9 times higher than a Gaussian baseline—but fine-tuned registers simply do not absorb them. And at $d=192$, dedicating tokens to registers imposes a real representational penalty."

---

### [ADVANCE TO SLIDE 16: Conclusion & Defense Takeaways]

> "To conclude our presentation, our study establishes four clear takeaways:
> 
> 1. **No held-out benefit:** Across 24 controlled runs, registers never outperform the register-free baseline on held-out test data.
> 2. **The validation illusion:** The apparent regularization benefit on validation ($p=0.024$) is an artifact of checkpoint selection; training fit was never traded.
> 3. **Scale invariance:** Tripling the data budget adds +6.14 pp to accuracy but preserves the exact null result, and test-time registers confirm this finding.
> 4. **Transferable methodological lesson:** Never infer generalization from the split used to perform model selection.
> 
> Our complete codebase, configuration files, and automated report generators are fully open-source and reproducible on GitHub."

---

### [ADVANCE TO SLIDE 17: Thank You & Q&A]
*(Gesture to the entire team and face the committee.)*

> "Thank you for your time and attention. Our team is now delighted to open the floor and take your questions."

---

## 🎯 Quick-Reference Q&A Cheat Sheet (For Committee Questions)

| Question Area | Primary Responder | Core 20-Second Answer |
|---|---|---|
| **Why did Val loss drop ($p=0.024$) while Test loss was flat ($p=0.823$)?** | **Rufet** | On a 1,000-image validation split, seed variance is 9× higher than on the 10,000-image test set. Val loss is used to pick the minimum checkpoint over 50 noisy epochs. Training loss was identical across all arms (0.878 vs 0.883), meaning no regularization trade occurred. |
| **Could the null result just be data starvation at 100 images/class?** | **Gulnisa / Rufet** | We directly disproved that in Phase 6 by tripling data to 300 pc (27,000 images). Baseline accuracy surged by +6.14 pp to 81.33%, yet registers still provided 0.00 pp benefit (all $p > 0.50$), proving scale invariance. |
| **How did you prevent data leakage and indexing bugs?** | **Narmina / Shahin** | Stratified subsets used fixed generator seeds. For the model, spatial tokens start strictly at index $1+K$, backed by 72 unit tests checking tensor dimensions across all 12 blocks. |
| **How was cluster execution verified?** | **Emil** | Automated bash runners executed all 24 runs over WireGuard on an NVIDIA A100 GPU cluster with dynamic VRAM cache flushing, achieving 100% completion with zero OOMs and independent JSON logging. |
| **How does Jiang et al. (NeurIPS 2025) test-time registers compare?** | **Shahin** | We tested passive slots and neuron redirection at test-time with zero training. Passive slot gave +0.07 pp, and active redirection gave +0.06 pp, confirming redirection does not beat an inert slot in compact classification models. |
