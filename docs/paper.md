# Research Paper

<div style="text-align: center; padding: 2rem 0;">
  <h1 style="font-size: 1.4rem; font-weight: bold; line-height: 1.5;">
    PHANTASM: Inverting LLM Failure Modes into Productive Features via<br>
    Probabilistic Hallucination-Aware Neural Transformation<br>
    with Adaptive Synthesis Method
  </h1>
  <p style="margin-top: 1rem; font-size: 1rem;">
    <strong>Vignesh S</strong><br>
    Department of Computer Science and Engineering, Takshashila University, Chennai, India<br>
    <code>applemacbook6sep2004@gmail.com</code>
  </p>
</div>

---

## Abstract

Large language models (LLMs) exhibit three persistent failure modes — hallucination, confabulation, and epistemic miscalibration — that the research community has uniformly treated as defects to be suppressed. We challenge this framing. We present **PHANTASM** (Probabilistic Hallucination-Aware Neural Transformation with Adaptive Synthesis Method), a novel inference-time framework that mathematically inverts each failure mode into a structured, actionable asset. Concretely: **(1) Hallucination Gradient Tracing (HGT)** exploits the gradient of a self-consistency loss with respect to input embeddings to produce a *Competency Atlas* — a per-token knowledge-boundary map requiring no ground truth; **(2) Confabulation Mining Network (CMN)** uses a contrastive dual-encoder to harvest novel, plausible hypotheses from the model's creative gap-filling, with applications in scientific discovery; **(3) Uncertainty Crystallization (UC)** chains Monte-Carlo Dropout, learned temperature scaling, and Conformal Prediction to produce statistically-guaranteed, four-tier reliability classifications. PHANTASM operates post-hoc on any HuggingFace causal LM without weight modification. HGT achieves AUROC 0.83 on LLaMA-7B (+9.2% over SelfCheckGPT). UC reduces ECE from 0.187 to 0.041 on GPT-2. CMN surfaces hypotheses with 67% novelty at 77% expert plausibility versus 8% novelty under standard filtering.

**Code:** `pip install phantasm-llm` · **GitHub:** [github.com/vignesh2027/PHANTASM](https://github.com/vignesh2027/PHANTASM)

---

## 1. Introduction

The dominant narrative in LLM reliability research treats hallucination as an adversary. Retrieval-Augmented Generation [Lewis et al., 2020] grounds generation in retrieved documents. RLHF [Ouyang et al., 2022] trains models away from hallucinated outputs. Fact-verification pipelines [Min et al., 2023] reject non-factual generations. SelfCheckGPT [Manakul et al., 2023] uses multi-sample inconsistency as a hallucination signal. **Every one of these frameworks treats the hallucinated output as waste.**

We argue this is a fundamental misreading. When a model hallucinates at position *i*, it does so because the gradient landscape at that position is steep — the model's representation is at the boundary of its training distribution. This is not random noise. It is a *precise*, *reproducible*, *gradient-traceable* signal about where the model's knowledge ends. Similarly, confabulations are not random fabrications; they are paths through the model's learned semantic manifold that training data never explicitly charted — candidate novel concept combinations. And epistemic miscalibration encodes, in its structure, which training distributions were overrepresented.

PHANTASM is the first framework to operationalize these observations into a unified three-pillar system.

---

## 2. Related Work

**Hallucination detection.** SAPLMA [Azaria & Mitchell, 2023] probes internal hidden states. SelfCheckGPT [Manakul et al., 2023] uses multi-sample consistency. FActScoring [Min et al., 2023] decomposes claims and verifies via retrieval. All require labels, multiple samples, or external knowledge bases. HGT requires none — one forward-backward pass suffices.

**Calibration.** Temperature scaling [Guo et al., 2017] minimizes NLL post-hoc. No prior LLM calibration work provides statistically-guaranteed coverage. UC adds Conformal Prediction [Angelopoulos & Bates, 2022] to provide such guarantees.

**Hypothesis generation.** Drug discovery [Bran et al., 2023] and scientific hypothesis generation [Wang et al., 2023] applications treat confabulations as noise. CMN is the first contrastive confabulation mining system.

---

## 3. Method

### 3.1 Hallucination Gradient Tracing (HGT)

Let $f_\theta$ be a causal LM. Given input embeddings $e_i \in \mathbb{R}^d$, the self-consistency loss is:

$$\mathcal{L}_{\text{sc}} = -\sum_{t=1}^{T} \log p_\theta(\hat{y}_t \mid x_{<t}), \quad \hat{y}_t = \arg\max_v p_\theta(v \mid x_{<t})$$

The HGT knowledge-boundary score at position $i$:

$$s_i = 1 - \frac{\|\nabla_{e_i} \mathcal{L}_{\text{sc}}\|}{\max_j \|\nabla_{e_j} \mathcal{L}_{\text{sc}}\|}$$

$s_i \approx 0$ at knowledge boundaries (high gradient); $s_i \approx 1$ where grounded. Overall hallucination risk: $r = 1 - \frac{1}{T}\sum_i s_i$. Computational cost: one forward + one backward pass.

### 3.2 Confabulation Mining Network (CMN)

CMN is a dual-encoder $(\phi_C, \phi_N, \phi_P)$:

- **ConceptExtractor** $\phi_C$: 2-layer Transformer encoder → concept vectors $\mathbf{c} \in \mathbb{R}^{L \times d}$
- **NoveltyScorer** $\phi_N$: MLP scoring distance from factual space
- **PlausibilityScorer** $\phi_P$: self-attention coherence scoring

Training loss:

$$\mathcal{L}_{\text{CMN}} = -0.6\log\sigma\!\left(\frac{1 - \cos(\bar{\mathbf{c}}_{\text{confab}}, \bar{\mathbf{c}}_{\text{fact}})}{\tau}\right) + 0.4\cdot\text{BCE}(\text{pla}(\mathbf{c}_{\text{confab}}), \mathbf{1})$$

The contrastive term rewards novelty (confabulation ≠ fact); the BCE term enforces coherence. Inference threshold: $\text{nov} \geq 0.45$, $\text{pla} \geq 0.50$.

### 3.3 Uncertainty Crystallization (UC)

Three sequential stages:

1. **MC-Dropout** ($N=30$ passes): $\sigma^2_{\text{ep}} = \text{Var}_n[p^{(n)}_\theta]$, $u_{\text{al}} = \mathbb{E}_n[H(p^{(n)}_\theta)]$
2. **Temperature Scaling**: $T^* = \arg\min_T \mathcal{L}_{\text{NLL}}(z/T, y)$ → calibrated confidence $\hat{p}$
3. **Conformal Prediction**: nonconformity score $\alpha_i = 1-\hat{p}_i$; interval $[\hat{p}-\hat{q},\, \hat{p}+\hat{q}]$ with $\Pr[y \in C(x)] \geq 0.90$

**Reliability tiers:**

| Tier | Condition | Action |
|------|-----------|--------|
| ◆ Crystal | $\hat{p} \geq 0.85$, $\sigma^2_{\text{ep}} < 0.05$ | Use directly |
| ◇ Solid | $\hat{p} \geq 0.65$, $\sigma^2_{\text{ep}} < 0.15$ | Light verification |
| ≈ Fluid | $\hat{p} \geq 0.40$, $\sigma^2_{\text{ep}} < 0.35$ | Verify before use |
| ~ Vapor | otherwise | Do not use |

---

## 4. Experiments

### 4.1 HGT — Hallucination Detection

| Model | Dataset | Method | AUROC ↑ | F1 ↑ |
|-------|---------|--------|---------|------|
| GPT-2 | Wiki Bio | SAPLMA | 0.71 | 0.68 |
| GPT-2 | Wiki Bio | SelfCheckGPT | 0.74 | 0.70 |
| GPT-2 | Wiki Bio | **HGT** | **0.79** | **0.76** |
| LLaMA-7B | Vectara HFB | SelfCheckGPT | 0.75 | 0.72 |
| LLaMA-7B | Vectara HFB | **HGT** | **0.83** | **0.80** |

HGT surpasses SelfCheckGPT by +9.2% AUROC on LLaMA-7B while requiring ~10× fewer forward passes (1 vs. 20).

### 4.2 UC — Calibration

| Model | Method | ECE ↓ | MCE ↓ | Brier ↓ |
|-------|--------|-------|-------|---------|
| GPT-2 | Uncalibrated | 0.187 | 0.312 | 0.241 |
| GPT-2 | Temp. scaling | 0.089 | 0.201 | 0.198 |
| GPT-2 | **UC** | **0.041** | **0.098** | **0.172** |
| LLaMA-7B | Uncalibrated | 0.143 | 0.267 | 0.209 |
| LLaMA-7B | **UC** | **0.029** | **0.071** | **0.161** |

78% ECE reduction on GPT-2. Conformal 90% interval achieves 91.3% empirical coverage, satisfying the theoretical guarantee.

### 4.3 CMN — Hypothesis Mining

| Method | Hypotheses | Novel@5 | Expert Plausibility |
|--------|-----------|---------|---------------------|
| Filtered LLM | 200 | 0.08 | 89% |
| RAG-augmented | 312 | 0.13 | 83% |
| **CMN** | **847** | **0.67** | **77%** |

CMN surfaces 4.2× more hypotheses with 8.4× greater novelty, at 77% expert plausibility — high for genuinely undiscovered territory.

---

## 5. Conclusion

PHANTASM establishes a new paradigm: **failure-mode mining**. Hallucination traces knowledge boundaries. Confabulation seeds hypothesis generation. Miscalibration enables precision uncertainty oracles. All three pillars operate post-hoc, model-agnostic, on any causal LM without weight modification. PHANTASM is available at `pip install phantasm-llm` under Apache 2.0. Every model that hallucinates is telling you exactly where it is blind. PHANTASM listens.

---

## References

Angelopoulos & Bates (2022). A gentle introduction to conformal prediction. *arXiv:2107.07511*. · Azaria & Mitchell (2023). The internal state of an LLM knows when it's lying. *EMNLP Findings*. · Bran et al. (2023). ChemCrow. *arXiv:2304.05376*. · Guo et al. (2017). On calibration of modern neural networks. *ICML*. · Kadavath et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*. · Lewis et al. (2020). RAG for knowledge-intensive NLP. *NeurIPS*. · Manakul et al. (2023). SelfCheckGPT. *EMNLP*. · Min et al. (2023). FActScoring. *ACL*. · Müller et al. (2019). When does label smoothing help? *NeurIPS*. · Ouyang et al. (2022). InstructGPT. *NeurIPS*. · Radford et al. (2019). GPT-2. *OpenAI Blog*. · Touvron et al. (2023). LLaMA. *arXiv:2302.13971*. · Wang et al. (2023). Scientific discovery in the age of AI. *Nature*.
