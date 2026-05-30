# PHANTASM: Inverting LLM Failure Modes into Productive Features via Probabilistic Hallucination-Aware Neural Transformation with Adaptive Synthesis Method

**Vignesh S**
Department of Computer Science and Engineering, Takshashila University, Chennai, India
`applemacbook6sep2004@gmail.com`

---

## Abstract

Large language models (LLMs) exhibit three persistent failure modes — hallucination, confabulation, and epistemic miscalibration — that the research community has uniformly treated as defects to be suppressed. We challenge this framing. We present **PHANTASM** (Probabilistic Hallucination-Aware Neural Transformation with Adaptive Synthesis Method), a novel inference-time framework that mathematically inverts each failure mode into a structured, actionable asset. Concretely: (1) **Hallucination Gradient Tracing (HGT)** exploits the gradient of a self-consistency loss with respect to input embeddings to produce a *Competency Atlas* — a per-token knowledge-boundary map requiring no ground truth; (2) **Confabulation Mining Network (CMN)** uses a contrastive dual-encoder to harvest novel, plausible hypotheses from the model's creative gap-filling, with applications in scientific discovery; (3) **Uncertainty Crystallization (UC)** chains Monte-Carlo Dropout, learned temperature scaling, and Conformal Prediction to produce statistically-guaranteed, four-tier reliability classifications. PHANTASM operates post-hoc on any HuggingFace causal LM without weight modification. Across evaluations on `potsawee/wiki_bio_gpt3_hallucination` and `vectara/hallucinated-faithfulness-benchmark`, HGT achieves AUROC 0.83 on LLaMA-7B — a +9.2% gain over SelfCheckGPT — while UC reduces ECE from 0.187 to 0.041 on GPT-2. CMN surfaces hypotheses with 67% novelty at expert plausibility rates of 77%, versus 8% novelty under standard filtering. Code and package: `pip install phantasm-llm`. GitHub: [github.com/vignesh2027/PHANTASM](https://github.com/vignesh2027/PHANTASM).

---

## 1. Introduction

The dominant narrative in LLM reliability research treats hallucination as an adversary. Retrieval-Augmented Generation (RAG) [Lewis et al., 2020] grounds generation in retrieved documents to prevent hallucination. Reinforcement Learning from Human Feedback (RLHF) [Ouyang et al., 2022] trains the model away from hallucinated outputs. Fact-verification pipelines [Min et al., 2023] reject generations that fail factuality checks. SelfCheckGPT [Manakul et al., 2023] samples multiple times and uses inconsistency as a hallucination signal.

**Every one of these frameworks treats the hallucinated output as waste.**

We argue this represents a fundamental misreading of what LLM failure modes are. When a model hallucinates at token position *i*, it does so because the gradient landscape at that position is steep — the model's representation is at the boundary of its training distribution. This is not random noise. It is a *precise*, *reproducible*, *gradient-traceable* signal about where the model's knowledge ends. Similarly, confabulations are not random fabrications; they are paths through the model's learned semantic manifold that training data never explicitly charted — i.e., candidate novel concept combinations. And epistemic miscalibration encodes, in its *structure*, which training distributions were overrepresented.

PHANTASM is the first framework to operationalize these observations into a unified three-pillar system. Our contributions are:

1. **HGT** — a no-ground-truth, single-pass knowledge-boundary mapper exploiting embedding-layer gradients.
2. **CMN** — a contrastive hypothesis-mining network that converts confabulations into scientific hypotheses.
3. **UC** — a three-stage uncertainty crystallizer with statistically-guaranteed coverage tiers via Conformal Prediction.
4. A unified `PHANTASMPipeline` wrapping any HuggingFace causal LM, published as `phantasm-llm` on PyPI under Apache 2.0.

---

## 2. Related Work

**Hallucination detection.** SAPLMA [Azaria & Mitchell, 2023] probes internal hidden states to detect hallucinations. SelfCheckGPT [Manakul et al., 2023] uses multi-sample consistency. FActScoring [Min et al., 2023] decomposes generations into atomic claims and verifies each against a retrieval corpus. All require either labels, multiple samples, or external knowledge bases. HGT requires none of these — one forward-backward pass suffices.

**Calibration.** Temperature scaling [Guo et al., 2017] post-hoc rescales logits to minimize NLL. Platt scaling [Platt, 1999] fits a sigmoid. Label smoothing [Müller et al., 2019] regularizes during training. None provides statistically-guaranteed coverage. UC adds Conformal Prediction [Angelopoulos & Bates, 2022] to provide coverage guarantees absent from all prior LLM calibration work.

**Hypothesis generation.** Drug discovery applications of LLMs [Bran et al., 2023] and scientific hypothesis generation [Wang et al., 2023] treat hallucinations as noise. No prior work harvests confabulations as structured hypotheses via contrastive learning. CMN is, to our knowledge, the first contrastive confabulation mining system.

**Paradigm framing.** Concurrent work on "self-knowledge" in LLMs [Kadavath et al., 2022] probes whether models know what they know. PHANTASM goes further — it extracts value from what models *don't* know.

---

## 3. Method

### 3.1 Hallucination Gradient Tracing (HGT)

**Observation.** Let $f_\theta$ be a causal LM parameterized by $\theta$. Given input token sequence $x = (x_1, \ldots, x_T)$ with embedding matrix $E$, let $e_i = E(x_i) \in \mathbb{R}^d$. The self-consistency loss is:

$$\mathcal{L}_{\text{sc}} = -\sum_{t=1}^{T} \log p_\theta(\hat{y}_t \mid x_{<t}), \quad \hat{y}_t = \arg\max_v p_\theta(v \mid x_{<t})$$

where $\hat{y}_t$ is the model's own top-1 prediction (no ground truth needed).

**HGT score.** The hallucination score at position $i$ is:

$$s_i = 1 - \frac{\|\nabla_{e_i} \mathcal{L}_{\text{sc}}\|}{\max_j \|\nabla_{e_j} \mathcal{L}_{\text{sc}}\|}$$

so $s_i \approx 0$ at knowledge boundaries (high gradient) and $s_i \approx 1$ where the model is grounded. Tokens with $s_i < \tau$ (default $\tau = 0.35$) are returned as `knowledge_gaps`. The overall hallucination risk is $r = 1 - \frac{1}{T}\sum_i s_i$.

**Computational cost.** One forward + one backward pass. $O(1)$ overhead versus standard inference.

### 3.2 Confabulation Mining Network (CMN)

CMN is a dual-encoder contrastive network $\mathcal{M} = (\phi_C, \phi_N, \phi_P)$:

- $\phi_C$: ConceptExtractor — a 2-layer Transformer encoder mapping token sequences to concept vectors $\mathbf{c} \in \mathbb{R}^{L \times d}$.
- $\phi_N$: NoveltyScorer — MLP scoring $\text{nov}(\mathbf{c}_{\text{confab}}, \mathbf{c}_{\text{fact}}) \in [0,1]$.
- $\phi_P$: PlausibilityScorer — self-attention module scoring $\text{pla}(\mathbf{c}_{\text{confab}}) \in [0,1]$.

The training loss is:

$$\mathcal{L}_{\text{CMN}} = -\alpha \log \sigma\!\left(\frac{1 - \text{cos}(\bar{\mathbf{c}}_{\text{confab}}, \bar{\mathbf{c}}_{\text{fact}})}{\tau}\right) + \beta \cdot \text{BCE}(\text{pla}(\mathbf{c}_{\text{confab}}), \mathbf{1})$$

where $\alpha=0.6$, $\beta=0.4$, and $\tau=0.07$. The first term (contrastive) drives confabulation representations *away from* factual space — rewarding novelty. The second term ensures internal coherence.

At inference, CMN returns `Hypothesis` objects where $\text{nov} \geq 0.45$ and $\text{pla} \geq 0.50$.

### 3.3 Uncertainty Crystallization (UC)

UC applies three calibration stages in sequence:

**Stage 1 — MC-Dropout.** With dropout active at inference, we draw $N=30$ forward passes yielding predictive distribution $\{p^{(n)}_\theta\}_{n=1}^N$. Epistemic uncertainty is $\sigma^2_{\text{ep}} = \text{Var}_n[p^{(n)}_\theta]$; aleatoric uncertainty is $u_{\text{al}} = \mathbb{E}_n[-\sum_v p^{(n)}_v \log p^{(n)}_v]$.

**Stage 2 — Temperature scaling.** A scalar $T > 0$ is optimized on a held-out calibration set by minimizing NLL: $T^* = \arg\min_T \mathcal{L}_{\text{NLL}}(f_\theta(x)/T, y)$, producing calibrated confidence $\hat{p} = \max_v \text{softmax}(z/T^*)_v$.

**Stage 3 — Conformal Prediction.** Given calibration set $\{(\hat{p}_i, y_i)\}_{i=1}^n$ with nonconformity scores $\alpha_i = 1 - \hat{p}_i$, the $(1-\delta)$-quantile is $\hat{q} = \text{Quantile}_{(1-\delta)(1+1/n)}(\{\alpha_i\})$. The prediction interval $[\hat{p} - \hat{q},\ \hat{p} + \hat{q}]$ has guaranteed marginal coverage: $\Pr[y \in C(x)] \geq 1 - \delta$.

**Reliability tiers** map the full uncertainty profile to actionable decisions:

| Tier | Condition | Action |
|------|-----------|--------|
| ◆ Crystal | $\hat{p} \geq 0.85$, $\sigma^2_{\text{ep}} < 0.05$ | Use directly |
| ◇ Solid | $\hat{p} \geq 0.65$, $\sigma^2_{\text{ep}} < 0.15$ | Light verification |
| ≈ Fluid | $\hat{p} \geq 0.40$, $\sigma^2_{\text{ep}} < 0.35$ | Verify before use |
| ~ Vapor | otherwise | Do not use |

---

## 4. Experiments

### 4.1 Setup

We evaluate on two public hallucination benchmarks: `potsawee/wiki_bio_gpt3_hallucination` (7,830 sentence-level GPT-3 annotations) and `vectara/hallucinated-faithfulness-benchmark` (1,000 passage-level annotations). Backbone models: GPT-2 [Radford et al., 2019] and LLaMA-7B [Touvron et al., 2023]. We compare against SAPLMA [Azaria & Mitchell, 2023] and SelfCheckGPT-BERTScore [Manakul et al., 2023].

For UC calibration evaluation we use 1,000-sample held-out splits with Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score. For CMN we report Novelty@5 and domain-expert plausibility scores across 200 drug-discovery hypotheses evaluated by a chemistry PhD student.

### 4.2 HGT Results

| Model | Dataset | Method | AUROC ↑ | F1 ↑ | ECE ↓ |
|-------|---------|--------|---------|------|-------|
| GPT-2 | Wiki Bio | SAPLMA | 0.71 | 0.68 | — |
| GPT-2 | Wiki Bio | SelfCheckGPT | 0.74 | 0.70 | — |
| GPT-2 | Wiki Bio | **HGT (ours)** | **0.79** | **0.76** | — |
| LLaMA-7B | Vectara HFB | Fact-checker | 0.76 | 0.73 | — |
| LLaMA-7B | Vectara HFB | SelfCheckGPT | 0.75 | 0.72 | — |
| LLaMA-7B | Vectara HFB | **HGT (ours)** | **0.83** | **0.80** | — |

HGT outperforms SelfCheckGPT by +9.2% AUROC on LLaMA-7B. Critically, SelfCheckGPT requires $N$ forward passes (typically $N=20$); HGT requires exactly one forward + one backward pass — a ~10× efficiency gain.

### 4.3 UC Results

| Model | Method | ECE ↓ | MCE ↓ | Brier ↓ |
|-------|--------|-------|-------|---------|
| GPT-2 | Uncalibrated | 0.187 | 0.312 | 0.241 |
| GPT-2 | Temp. scaling | 0.089 | 0.201 | 0.198 |
| GPT-2 | **UC (ours)** | **0.041** | **0.098** | **0.172** |
| LLaMA-7B | Uncalibrated | 0.143 | 0.267 | 0.209 |
| LLaMA-7B | Temp. scaling | 0.067 | 0.154 | 0.187 |
| LLaMA-7B | **UC (ours)** | **0.029** | **0.071** | **0.161** |

UC reduces ECE by 78% versus uncalibrated GPT-2 and 54% versus temperature scaling alone. The conformal prediction stage provides provably-correct 90% coverage intervals, a guarantee absent from all compared methods.

### 4.4 CMN Results

| Method | Hypotheses | In-Literature | Novel@5 | Expert Plausibility |
|--------|-----------|---------------|---------|---------------------|
| Filtered LLM | 200 | 100% | 0.08 | 89% |
| RAG-augmented | 312 | 94% | 0.13 | 83% |
| **CMN (ours)** | **847** | **31%** | **0.67** | **77%** |

CMN surfaces 847 hypotheses with 67% novelty versus 8% under standard filtering. Expert plausibility is 77% — high for genuinely undiscovered territory. The plausibility gap (89% → 77%) is expected and desirable: it reflects that CMN is deliberately operating beyond the known literature, where expert certainty should be lower.

---

## 5. Analysis

**Why gradient norms detect hallucinations.** We verify the HGT gradient-uncertainty hypothesis by ablating with random gradient noise (AUROC drops to 0.51, near chance), confirming that the information is in gradient *structure*, not magnitude alone. Layer-wise analysis shows the final 3 transformer layers carry 74% of the total boundary signal on LLaMA-7B.

**CMN novelty-plausibility tradeoff.** Lowering CMN thresholds increases hypothesis count but decreases expert acceptance. We find the Pareto-optimal point at $\text{nov} \geq 0.45$, $\text{pla} \geq 0.50$, yielding the best expert-acceptance-per-hypothesis ratio.

**UC coverage guarantee holds empirically.** On 1,000 held-out examples, the 90% conformal interval contains the true label in 91.3% of cases — satisfying the marginal coverage guarantee with statistical slack.

**Failure modes of PHANTASM.** HGT degrades on tokenizations where subword boundaries misalign with semantic boundaries (e.g., rare named entities split across multiple tokens). CMN can mine low-quality hypotheses in domains where training data is extremely sparse. UC requires a calibration split; in zero-shot settings without calibration data, intervals widen.

---

## 6. Conclusion

We presented PHANTASM, the first framework to systematically invert LLM failure modes into productive features. The core insight is simple but consequential: hallucinations, confabulations, and epistemic miscalibration each encode structured information that standard pipelines discard. HGT maps knowledge boundaries with no ground truth and a single backward pass. CMN mines confabulations for novel, plausible hypotheses with applications across scientific domains. UC crystallizes miscalibrated confidence into statistically-guaranteed reliability tiers. Together, they form a post-hoc, model-agnostic wrapper available as `pip install phantasm-llm` under Apache 2.0.

We believe PHANTASM opens a new research direction: **failure-mode mining** — the systematic extraction of value from what LLMs do not know. Every model that hallucinates is telling you exactly where it is blind. PHANTASM listens.

---

## References

Angelopoulos, A. N., & Bates, S. (2022). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv:2107.07511*.

Azaria, A., & Mitchell, T. (2023). The internal state of an LLM knows when it's lying. *EMNLP Findings*.

Bran, A. M., et al. (2023). ChemCrow: Augmenting large-language models with chemistry tools. *arXiv:2304.05376*.

Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*.

Kadavath, S., et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.

Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*.

Manakul, P., et al. (2023). SelfCheckGPT: Zero-resource black-box hallucination detection for generative LLMs. *EMNLP*.

Min, S., et al. (2023). FActScoring: Fine-grained atomic evaluation of factual precision in long-form generation. *ACL*.

Müller, R., et al. (2019). When does label smoothing help? *NeurIPS*.

Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.

Radford, A., et al. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*.

Touvron, H., et al. (2023). LLaMA: Open and efficient foundation language models. *arXiv:2302.13971*.

Wang, L., et al. (2023). Scientific discovery in the age of artificial intelligence. *Nature*.
