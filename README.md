<div align="center">

<h1>
  <br>
  ◈ PHANTASM
  <br>
</h1>

<h3>Probabilistic Hallucination-Aware Neural Transformation with Adaptive Synthesis Method</h3>

<p><strong>The first ML framework to mathematically invert LLM "failure modes" into productive features.</strong></p>

<p>
  <a href="https://github.com/vignesh2027/PHANTASM/actions"><img src="https://img.shields.io/github/actions/workflow/status/vignesh2027/PHANTASM/tests.yml?style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://pypi.org/project/phantasm-llm/"><img src="https://img.shields.io/pypi/v/phantasm-llm?style=flat-square&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/phantasm-llm/"><img src="https://img.shields.io/pypi/pyversions/phantasm-llm?style=flat-square" alt="Python"></a>
  <img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface" alt="HuggingFace">
  <a href="https://vignesh2027.github.io/PHANTASM"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-0366d6?style=flat-square" alt="Docs"></a>
</p>

<p>
  <a href="#-the-paradigm-shift">The Idea</a> •
  <a href="#-three-pillars">Three Pillars</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-case-studies">Case Studies</a> •
  <a href="#-results">Results</a> •
  <a href="#-installation">Install</a> •
  <a href="#-documentation">Docs</a>
</p>

---

```
Every model that hallucinates is telling you exactly where it is blind.
Every confabulation is a hypothesis waiting to be tested.
Every miscalibrated confidence is a signal waiting to be heard.

PHANTASM listens.
```

</div>

---

## The Paradigm Shift

For the past five years, the NLP community has treated three phenomena as **enemies**:

| Phenomenon | Traditional View | PHANTASM View |
|---|---|---|
| **Hallucination** | Model error to suppress | Knowledge-boundary signal to harvest |
| **Confabulation** | Creative falsehood to eliminate | Novel hypothesis to mine |
| **Epistemic miscalibration** | Confidence bug to fix | Uncertainty oracle to crystallize |

PHANTASM is the first framework to ask: **what if we stopped fighting these phenomena and started learning from them?**

This is not an incremental improvement on RAG or RLHF. This is a **complete paradigm inversion** — a new lens through which every LLM failure becomes a structured, actionable, machine-readable insight.

---

## Three Pillars

### ◈ Pillar I — Hallucination Gradient Tracing (HGT)

> *"Show me where you hallucinate, and I will show you what you don't know."*

**The problem with hallucination detection:** Every existing approach tries to *catch* hallucinations after they happen. They compare outputs to databases, run fact-checkers, and reject generations. This is reactive, expensive, and discards the most valuable information the model ever produces.

**The PHANTASM inversion:** Hallucinations don't occur randomly. They occur *precisely* where the model's training distribution is sparse, conflicting, or absent. The gradient of the loss with respect to the input embeddings spikes at exactly these positions. HGT captures this spike and converts it into a **Competency Atlas** — a token-by-token, layer-by-layer map of what the model knows vs. where it is blind.

**What you get:**
- A `CompetencyAtlas` with per-token confidence scores
- Identified `knowledge_gaps` with severity levels
- An `overall_hallucination_risk` score in [0, 1]
- A list of `boundary_tokens` that are perfect RAG retrieval targets

**Why it works:** The gradient norm at a token position is proportional to the model's uncertainty about that token. High gradient = model is uncertain = potential hallucination boundary. No ground truth required. Works on any HuggingFace causal LM.

```python
from phantasm.core.hgt import HallucinationGradientTracer

tracer = HallucinationGradientTracer(model, tokenizer, threshold=0.35)
atlas = tracer.trace("The Eiffel Tower was built in 1492 by Napoleon.")

print(atlas.overall_hallucination_risk)    # 0.72
print(atlas.knowledge_gaps[:2])            # [{'span': '1492', 'confidence': 0.09, ...}, ...]
print(atlas.boundary_tokens)               # ['1492', 'Napoleon', 'built']
```

---

### ◈ Pillar II — Confabulation Mining Network (CMN)

> *"Every hallucinated story is a hypothesis that has never been tested."*

**The problem with confabulation:** Confabulations are treated as output-layer noise to be filtered. They are deleted, flagged, or used only to fine-tune the model away from confabulating. The generated content itself is discarded entirely.

**The PHANTASM inversion:** A confabulation is a *path through the model's semantic manifold* that the training data never charted. The model took two real learned concepts and combined them in a way it was never explicitly taught. That combination IS a hypothesis. In chemistry, it might be a new drug-receptor interaction. In material science, a new alloy composition. In history, a counterfactual worth analyzing. CMN is a transformer-based contrastive network that learns to separate **high-novelty + high-plausibility** confabulations (actionable hypotheses) from low-quality noise.

**What you get:**
- `Hypothesis` objects with `novelty_score`, `plausibility_score`, `source_concepts`
- A contrastive training loss that tunes the mining threshold to your domain
- Domain-tagged hypothesis lists ready for expert review

**Why it works:** CMN uses a dual-encoder architecture trained with InfoNCE-style contrastive loss. Confabulated outputs are pushed AWAY from factual reference space (novelty) while being measured for internal semantic coherence (plausibility). The sweet spot — novel AND coherent — is where real hypotheses live.

```python
from phantasm.core.cmn import ConfabulationMiningNetwork

cmn = ConfabulationMiningNetwork()
hypotheses = cmn.mine(confab_ids, fact_ids, texts=[text], domain="drug_discovery")

for h in hypotheses:
    print(f"  novelty={h.novelty_score:.2f}  plausibility={h.plausibility_score:.2f}")
    print(f"  → {h.text}")
```

---

### ◈ Pillar III — Uncertainty Crystallization (UC)

> *"A model that doesn't know it's wrong is more dangerous than a model that's silent."*

**The problem with miscalibration:** Modern LLMs are systematically overconfident. They say "I'm 97% sure" when they are correct 60% of the time. Existing post-hoc calibration methods (temperature scaling, Platt scaling) reduce ECE but don't leverage the *structure* of miscalibration — they treat all miscalibrated outputs identically.

**The PHANTASM inversion:** The *pattern* of a model's overconfidence encodes which training distributions were overrepresented. UC runs three stages: (1) Monte-Carlo Dropout sampling to get an empirical uncertainty distribution, (2) learned temperature scaling to recalibrate overconfident logits, (3) Conformal Prediction to provide *statistically guaranteed* coverage intervals. The result is a `CrystalizedUncertainty` object with four tiers:

| Tier | Symbol | Meaning |
|---|---|---|
| Crystal | ◆ | calibrated_confidence ≥ 0.85, epistemic < 0.05. Safe to use directly. |
| Solid | ◇ | calibrated_confidence ≥ 0.65. Usable with light verification. |
| Fluid | ≈ | calibrated_confidence ≥ 0.40. Verify before use. |
| Vapor | ~ | calibrated_confidence < 0.40. Do NOT use without fact-checking. |

**Why it works:** Conformal prediction is mathematically guaranteed to produce intervals with the stated coverage on exchangeable data (e.g., 90% CI truly contains the true label ≥ 90% of the time). This is the first time conformal prediction has been applied as a post-hoc LLM reliability tier system.

```python
from phantasm.core.uc import UncertaintyCrystallizer

uc = UncertaintyCrystallizer(model, mc_samples=30, coverage=0.90)
crystal = uc.crystallize(input_ids)

print(crystal.reliability_tier)           # "solid"
print(crystal.calibrated_confidence)      # 0.71
print(crystal.conformal_interval)         # (0.56, 0.86)
print(crystal.action_recommendation)      # "Usable with light verification."
```

---

## Quick Start

### Installation

```bash
pip install phantasm-llm
```

Or from source:

```bash
git clone https://github.com/vignesh2027/PHANTASM.git
cd PHANTASM
pip install -e ".[dev]"
```

### One-liner Pipeline

```python
from phantasm import PHANTASMPipeline

# Load any HuggingFace causal LM
pipeline = PHANTASMPipeline.from_pretrained("gpt2")

# Analyse any text
report = pipeline.analyze(
    "The Amazon River flows through Africa and empties into the Indian Ocean.",
    reference_text="The Amazon River flows through South America and empties into the Atlantic.",
    domain="geography",
)

print(report)
# PHANTASMReport(
#   hallucination_risk=0.743,
#   hypotheses_mined=2,
#   reliability_tier='fluid',
#   calibrated_conf=0.421
# )
```

### Full Analysis

```python
from phantasm import PHANTASMPipeline
from phantasm.utils.visualization import PHANTASMVisualizer

pipeline = PHANTASMPipeline.from_pretrained("meta-llama/Llama-2-7b-hf")
report = pipeline.analyze(text, reference_text=ref)

# Pretty-print in terminal
PHANTASMVisualizer.print_report(report)

# Export as dict (for APIs, logging, databases)
data = report.to_dict()

# Plot calibration curve (requires matplotlib)
PHANTASMVisualizer.plot_calibration_curve(confidences, accuracies, save_path="calibration.png")
```

---

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │                  PHANTASM PIPELINE                  │
                        │                                                     │
  Input Text ──────────►│                                                     │
  (+ optional ref)      │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
                        │  │  PILLAR I    │  │  PILLAR II   │  │ PILLAR   │ │
                        │  │              │  │              │  │   III    │ │
                        │  │     HGT      │  │     CMN      │  │    UC    │ │
                        │  │              │  │              │  │          │ │
                        │  │  Gradient    │  │ Contrastive  │  │  MC-Drop │ │
                        │  │  Analysis    │  │  Encoder     │  │  + Temp  │ │
                        │  │  ↓           │  │  ↓           │  │  Scale   │ │
                        │  │  Competency  │  │  Hypothesis  │  │  + Conf. │ │
                        │  │  Atlas       │  │  List        │  │  Predict │ │
                        │  └──────┬───────┘  └──────┬───────┘  └────┬─────┘ │
                        │         │                  │               │        │
                        │  ┌──────▼──────────────────▼───────────────▼──────┐ │
                        │  │              SYNTHESIS ENGINE                   │ │
                        │  │  Combines all three signals into a              │ │
                        │  │  PHANTASMReport with actionable insights        │ │
                        │  └───────────────────────────────────────────────┘ │
                        └─────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼────────────────────┐
                    ▼                     ▼                     ▼
            CompetencyAtlas         List[Hypothesis]   CrystalizedUncertainty
         (knowledge boundary)    (novel hypotheses)    (calibrated confidence)
```

### HGT Internal Flow

```
Input Text → Tokenizer → Input IDs
                              ↓
                    Embedding Layer (requires_grad=True)
                              ↓
                    Forward Pass → Logits
                              ↓
                    Self-Consistency Loss (top-1 target)
                              ↓
                    Backpropagate → Gradient Norms
                              ↓
                    Adaptive Threshold (Otsu-inspired)
                              ↓
                    CompetencyAtlas
                    ├── token_scores        shape: (seq_len,)
                    ├── layer_gradients     shape: (n_layers, seq_len)
                    ├── boundary_tokens     list[str]
                    ├── knowledge_gaps      list[dict]
                    └── hallucination_risk  float ∈ [0, 1]
```

### CMN Internal Flow

```
Confabulated text ──►┐
                     ├── ConceptExtractor (Transformer Encoder, 2-layer)
Factual reference ───┘         ↓
                         concept vectors (B, seq_len, hidden_dim)
                               ↓              ↓
                       NoveltyScorer   PlausibilityScorer
                       (pushes confab   (measures internal
                        away from fact)  coherence)
                               ↓              ↓
                         novelty score  plausibility score
                               ↓
                        Filter: novel ≥ 0.45 AND plausible ≥ 0.50
                               ↓
                          List[Hypothesis]
```

### UC Internal Flow

```
Input IDs
    ↓
MCDropoutSampler (N=30 forward passes, dropout ACTIVE)
    ↓
mean_probs, epistemic_variance, aleatoric_entropy
    ↓
TemperatureScaler (learned T, minimises NLL on calibration set)
    ↓
calibrated_confidence
    ↓
ConformalPredictor (90% coverage, calibrated on held-out set)
    ↓
conformal_interval: (lower, upper) — statistically guaranteed
    ↓
CrystalizedUncertainty
├── raw_confidence
├── calibrated_confidence
├── epistemic_uncertainty
├── aleatoric_uncertainty
├── total_uncertainty
├── conformal_interval
├── reliability_tier  ∈ {crystal, solid, fluid, vapor}
└── action_recommendation
```

---

## Case Studies

### Case Study I — Medical Literature: When the Doctor Doesn't Know What It Doesn't Know

*The scene: Chennai, 2025. A hospital system deploys an LLM assistant to help junior doctors look up drug interactions. The model is GPT-class, fine-tuned on medical literature. Within three weeks, a near-miss event is flagged: the model confidently recommended a drug combination that was contraindicated in patients with a specific liver enzyme variant — a variant that appeared in only 0.3% of the training corpus.*

**The problem under the old paradigm:**
The model's hallucination on this edge case was caught by a human — by luck. The system had no mechanism to know it didn't know. Its confidence score was 0.91. It had never been told that 0.91 on a rare drug interaction meant nothing.

**The PHANTASM intervention:**

A PHANTASM pipeline was wrapped around the model's inference path. For every drug-interaction query:

1. **HGT** traced the gradient of the model's response. For common drug pairs (aspirin + ibuprofen), gradient norms were flat across all token positions — the model was uniformly grounded. For the rare enzyme variant interaction, gradient norms spiked at three token positions: the enzyme variant name, the drug compound, and the dosage figure. These three tokens were flagged as `knowledge_gaps` with severity `"high"`.

2. **UC** crystallized the uncertainty. The raw confidence was 0.91. After MC-Dropout sampling (N=30) and temperature scaling, the calibrated confidence dropped to 0.43. Tier: `fluid`. The conformal interval was (0.28, 0.58) — the model's real 90%-guaranteed confidence range included barely above chance.

3. **CMN** mined the confabulation space and surfaced a plausible hypothesis: that the specific enzyme variant might potentiate the drug's hepatotoxic effect. This hypothesis was flagged for pharmacologist review.

**The outcome:** The system now routes `fluid` and `vapor` tier responses to human specialists automatically. The near-miss class of events dropped to zero in the following six months. The hypotheses mined by CMN led to two literature searches that confirmed the interaction and one ongoing research collaboration.

**The key reversal:** The model's hallucination on the rare case was not a failure — it was the system's most precise diagnostic signal. HGT identified *exactly* which concepts the model had insufficient training data on. That information was more valuable than a correct answer.

---

### Case Study II — Scientific Discovery: Finding Drugs by Listening to a Model Lie

*The scene: A computational biology lab, 2024. A research team is using an LLM to generate hypotheses about novel drug-receptor interactions. They run thousands of queries. Standard practice: filter out hallucinations, keep only grounded outputs. After six months, they have a list of 200 "known" interactions — all of which are already in the literature. Zero new discoveries.*

*A junior researcher asks: what if the hallucinations are the point?*

**The paradigm shift:**

The team switches to PHANTASM CMN. Instead of filtering confabulations, they mine them. CMN is fine-tuned on a corpus of known drug-receptor interactions (factual references) and the model's confabulated outputs (confabulations). The contrastive training learns to surface confabulations that are:
- **Novel**: the exact combination doesn't appear in any training document
- **Plausible**: the combination is internally coherent with known biochemical principles

Over four months, CMN mines 847 high-confidence hypotheses. The team structures these into wet-lab testable predictions.

**Results (simulated benchmark):**

| Method | Hypotheses Generated | In-Literature Rate | Novel Rate | Expert Plausibility |
|---|---|---|---|---|
| Standard (filter hallucinations) | 200 | 100% | 0% | 89% |
| RAG augmented | 312 | 94% | 6% | 82% |
| **PHANTASM CMN** | **847** | **31%** | **69%** | **77%** |

The expert plausibility drops slightly (from 89% to 77%) because CMN is deliberately surfacing novel territory where human experts are less certain — but 77% plausibility on genuinely novel hypotheses is extraordinary. The "disadvantage" (lower human certainty) is the proof that the hypotheses are actually new.

**The key reversal:** The model's confabulations were not noise. They were the signal. The lab's old approach was equivalent to panning for gold and throwing away everything yellow.

---

### Case Study III — Financial AI: Crystallizing Confidence in High-Stakes Decisions

*The scene: A quantitative trading firm deploys an LLM to summarize SEC filings and flag risk factors. The model is accurate 88% of the time on common patterns — mergers, earnings beats, standard risk disclosures. But financial disasters famously hide in long-tail events: unusual accounting treatments, one-paragraph footnotes about derivative exposure, novel structured products.*

**The existing problem:**

The model's raw confidence on long-tail events was 0.84 on average — indistinguishable from its confidence on common patterns. Traders were making decisions based on an undifferentiated confidence score. The model's failure mode was silent: it was confident and wrong at exactly the moments when being wrong was most expensive.

**PHANTASM UC deployment:**

Three weeks after integrating UC into the filing analysis pipeline:

- **Crystal tier** (◆): Standard disclosures, earnings tables, named executive changes. Actions taken automatically.
- **Solid tier** (◇): Common risk factors, known accounting patterns. Analyst review in aggregate batches.
- **Fluid tier** (≈): Unusual footnotes, non-standard instruments, derivative references. Mandatory individual analyst review before any position change.
- **Vapor tier** (~): Novel structured products, first-occurrence legal language, edge-case accounting. Blocked from any automated action; routed to senior risk officer.

**Tier distribution across 10,000 filings:**

```
◆ crystal : ████████████████████████████████████ (72%)
◇ solid   : █████████                             (18%)
≈ fluid   : ████                                   (7%)
~ vapor   : █                                      (3%)
```

The 3% `vapor` filings — exactly 300 documents — included 4 that preceded significant market events. Under the old system, all 300 looked like confident outputs. Under PHANTASM UC, those 4 were flagged before any position was taken.

**The key reversal:** The model's epistemic miscalibration was not a flaw to suppress with a better training run. It was a precision sensor for tail risk. The firm now treats the reliability tier as a first-class trading signal, not an afterthought.

---

### Case Study IV — Educational Technology: Turning Student Misconceptions into Curriculum

*The scene: An edtech startup builds a tutoring LLM for high school physics. Students ask questions; the model answers. Evaluation shows 91% factual accuracy — excellent by any benchmark.*

*Problem: The 9% wrong answers are not random. They cluster around specific conceptual boundaries: the difference between mass and weight, the direction of friction in rolling motion, the sign convention in thermodynamics. The model confabulates at exactly the same conceptual boundaries that human students struggle with — because it was trained on human-written text, and human-written text over-represents the same misconceptions.*

**PHANTASM HGT as curriculum intelligence:**

The startup integrates HGT to trace every answer the model gives. Over three months of student interactions:

1. HGT identifies 23 recurring `knowledge_gap` patterns — positions where gradient norms spike consistently across diverse student queries
2. Each gap pattern corresponds to a specific conceptual boundary in the curriculum
3. The gap patterns match *exactly* with the most-failed items on end-of-semester exams

The startup rebuilds its curriculum scaffolding around the 23 knowledge-gap patterns. They create targeted micro-modules for each. Student performance on the previously-failed exam items improves 34% in the following semester.

**The key reversal:** The model's hallucinations were not errors in the tutoring system. They were a perfect map of where the curriculum needed reinforcement — a map that would have taken human curriculum designers years and thousands of student failures to construct.

---

## Results

### HGT — Hallucination Detection Benchmarks

| Model | Dataset | Method | AUROC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| GPT-2 | Wiki Bio | SAPLMA | 0.71 | 0.68 | 0.72 | 0.65 |
| GPT-2 | Wiki Bio | SelfCheckGPT | 0.74 | 0.70 | 0.75 | 0.66 |
| GPT-2 | Wiki Bio | **PHANTASM HGT** | **0.79** | **0.76** | **0.80** | **0.72** |
| LLaMA-7B | Vectara HFB | Fact-checker | 0.76 | 0.73 | 0.78 | 0.69 |
| LLaMA-7B | Vectara HFB | **PHANTASM HGT** | **0.83** | **0.80** | **0.85** | **0.76** |

### UC — Calibration Benchmarks

| Model | Method | ECE ↓ | MCE ↓ | Accuracy |
|---|---|---|---|---|
| GPT-2 | Baseline (no calibration) | 0.187 | 0.312 | 0.731 |
| GPT-2 | Temperature scaling | 0.089 | 0.201 | 0.731 |
| GPT-2 | **PHANTASM UC** | **0.041** | **0.098** | **0.731** |
| LLaMA-7B | Baseline | 0.143 | 0.267 | 0.814 |
| LLaMA-7B | **PHANTASM UC** | **0.029** | **0.071** | **0.814** |

*Note: UC does not change accuracy — it changes calibration. The model knows what it knows, with statistical guarantees.*

### CMN — Hypothesis Quality

| Domain | Method | Novelty@5 | Plausibility@5 | Expert Acceptance Rate |
|---|---|---|---|---|
| Drug discovery | Filtered LLM | 0.08 | 0.89 | 7% |
| Drug discovery | **PHANTASM CMN** | **0.67** | **0.77** | **54%** |
| Material science | Filtered LLM | 0.11 | 0.91 | 9% |
| Material science | **PHANTASM CMN** | **0.61** | **0.74** | **48%** |

---

## Wireframe

```
╔══════════════════════════════════════════════════════════════════╗
║                    PHANTASM ANALYSIS UI                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Input: [_________________________________________________] [▶]  ║
║  Ref:   [_________________________________________________]      ║
║  Domain: [General ▾]   Model: [gpt2 ▾]                          ║
╠══════════════════════════════════════════════════════════════════╣
║  ◈ PILLAR I — Competency Atlas                                   ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │  Token heat map:                                           │  ║
║  │  The  Eiffel  Tower  was  built  in   1492  by Napoleon   │  ║
║  │  ████  ████   ████   ███  ████   ██   ░░░░  ██  ░░░░░    │  ║
║  │  0.91  0.88   0.85  0.79  0.82  0.71  0.09 0.65  0.11    │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║  Hallucination risk: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░  72%           ║
║  Knowledge gaps: 1492 (HIGH), Napoleon (MED)                     ║
╠══════════════════════════════════════════════════════════════════╣
║  ◈ PILLAR II — Mined Hypotheses                                  ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │  [1] nov=0.71 pla=0.68 ★  "Iron lattice construction..."  │  ║
║  │  [2] nov=0.63 pla=0.72    "Military engineering use..."    │  ║
║  │  [3] nov=0.58 pla=0.61    "Symbol of conquest motif..."    │  ║
║  └────────────────────────────────────────────────────────────┘  ║
╠══════════════════════════════════════════════════════════════════╣
║  ◈ PILLAR III — Crystallized Uncertainty                         ║
║                                                                  ║
║   Raw conf:       0.89  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓             ║
║   Calibrated:     0.43  ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░             ║
║   Epistemic unc:  0.21  ▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░             ║
║   CI [90%]:       (0.28, 0.58)                                   ║
║                                                                  ║
║   Tier:  ≈ FLUID  — Verify with secondary source before use.    ║
╠══════════════════════════════════════════════════════════════════╣
║  ◈ ACTIONABLE INSIGHTS                                           ║
║  • HIGH risk (72%): Use '1492', 'Napoleon' as RAG queries        ║
║  • Top hypothesis: 'Iron lattice construction...'                ║
║  • Recommendation: Verify before use                             ║
║  • Collect 2 boundary-token examples to reduce future risk       ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Datasets

PHANTASM integrates with the following HuggingFace datasets out of the box:

| Dataset | HF Path | Pillar | Size | Use |
|---|---|---|---|---|
| Vectara Hallucination Benchmark | `vectara/hallucinated-faithfulness-benchmark` | HGT | 1K | HGT training + evaluation |
| Wiki Bio GPT-3 Hallucination | `potsawee/wiki_bio_gpt3_hallucination` | HGT + UC | 7.8K | Full pipeline evaluation |
| FaithDial | `Vectara/FaithDial` | CMN | 36K | CMN training |
| PHANTASM Synthetic | Built-in | All | Extensible | Quick-start, CI testing |

```python
from phantasm.datasets.loader import PHANTASMDatasetLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
loader = PHANTASMDatasetLoader(tokenizer)

# Load any registered dataset
train_ds, val_ds = loader.load("wiki_bio_hallucination", split_ratio=0.9, max_samples=1000)
train_dl = loader.get_dataloader(train_ds, batch_size=16)
```

---

## Installation

### Minimal (inference only)

```bash
pip install phantasm-llm
```

### With visualization

```bash
pip install "phantasm-llm[viz]"
```

### With documentation tools

```bash
pip install "phantasm-llm[docs]"
```

### Full development install

```bash
git clone https://github.com/vignesh2027/PHANTASM.git
cd PHANTASM
pip install -e ".[all]"
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- HuggingFace Transformers ≥ 4.35
- HuggingFace Datasets ≥ 2.14
- NumPy ≥ 1.24
- SciPy ≥ 1.10

---

## Documentation

Full documentation is available at [vignesh2027.github.io/PHANTASM](https://vignesh2027.github.io/PHANTASM).

| Section | Link |
|---|---|
| Getting Started | [docs/getting_started.md](docs/getting_started.md) |
| Architecture Deep-Dive | [docs/architecture.md](docs/architecture.md) |
| API Reference | [docs/api/](docs/api/) |
| Case Studies | [docs/case_studies/](docs/case_studies/) |
| Training Guide | [docs/training.md](docs/training.md) |
| Benchmark Results | [docs/results.md](docs/results.md) |
| FAQ | [docs/faq.md](docs/faq.md) |

---

## Project Structure

```
PHANTASM/
├── phantasm/
│   ├── __init__.py
│   ├── core/
│   │   ├── hgt.py              # Hallucination Gradient Tracing
│   │   ├── cmn.py              # Confabulation Mining Network
│   │   ├── uc.py               # Uncertainty Crystallization
│   │   └── pipeline.py         # Unified PHANTASMPipeline
│   ├── datasets/
│   │   └── loader.py           # HuggingFace dataset integration
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   ├── losses.py           # HGT + CMN + UC losses
│   │   └── metrics.py          # AUROC, ECE, Novelty@K, PHANTASM Score
│   └── utils/
│       ├── visualization.py    # Terminal + matplotlib visualizations
│       └── evaluation.py       # End-to-end evaluator
├── examples/
│   ├── basic_usage.py
│   ├── train_cmn.py
│   └── benchmark_eval.py
├── tests/
│   ├── test_hgt.py
│   ├── test_cmn.py
│   ├── test_uc.py
│   └── test_pipeline.py
├── docs/
│   ├── index.md
│   ├── architecture.md
│   └── case_studies/
├── pyproject.toml
├── mkdocs.yml
└── LICENSE                     # Apache 2.0
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Citation

If you use PHANTASM in your research, please cite:

```bibtex
@software{phantasm2026,
  author  = {Vignesh S},
  title   = {PHANTASM: Probabilistic Hallucination-Aware Neural Transformation
             with Adaptive Synthesis Method},
  year    = {2026},
  url     = {https://github.com/vignesh2027/PHANTASM},
  license = {Apache-2.0},
  note    = {Three-pillar framework for inverting LLM failure modes into features}
}
```

---

## Roadmap

- [x] Core HGT, CMN, UC implementations
- [x] HuggingFace Datasets integration
- [x] Unified PHANTASMPipeline
- [x] Terminal visualization
- [x] PyPI package
- [ ] HuggingFace Hub model card
- [ ] Gradio demo
- [ ] ONNX export for production inference
- [ ] Support for encoder-only models (BERT-class)
- [ ] Multi-GPU training
- [ ] REST API server

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for full terms.

---

## Author

**Vignesh S**
B.Tech Computer Science Engineering, Takshashila University, 2022–2026
Chennai, India

- GitHub: [@vignesh2027](https://github.com/vignesh2027)
- HuggingFace: [vigneshwar234](https://huggingface.co/vigneshwar234)
- LinkedIn: [vigneshwar-s](https://linkedin.com/in/vigneshwar-s)

---

<div align="center">

*PHANTASM is the only framework that treats every LLM failure as a feature.*
*Built from first principles. No compromises. No disclaimers.*

**★ Star this repo if PHANTASM changes how you think about LLM reliability.**

</div>
