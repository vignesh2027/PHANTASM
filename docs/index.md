# PHANTASM

**Probabilistic Hallucination-Aware Neural Transformation with Adaptive Synthesis Method**

> *The first ML framework to mathematically invert LLM "failure modes" into productive features.*

---

## What is PHANTASM?

PHANTASM is a **paradigm-inverting PyTorch framework** that takes the three most reviled failure modes of large language models — hallucination, confabulation, and epistemic miscalibration — and converts each one into a structured, actionable, machine-readable asset.

Instead of suppressing failures, PHANTASM harvests them.

---

## Three Pillars at a Glance

=== "Pillar I: HGT"

    **Hallucination Gradient Tracing**

    Uses gradient analysis to map exactly WHERE the model's knowledge ends.
    Produces a `CompetencyAtlas` — the model's blind spots, on a silver platter.

    ```python
    atlas = tracer.trace("The Eiffel Tower was built in 1492.")
    # atlas.overall_hallucination_risk = 0.72
    # atlas.boundary_tokens = ['1492', 'Napoleon']
    ```

=== "Pillar II: CMN"

    **Confabulation Mining Network**

    Mines the model's creative falsehoods for novel, plausible hypotheses.
    Turns the "garbage" into gold — literally.

    ```python
    hypotheses = cmn.mine(confab_ids, fact_ids, texts=[text])
    # Hypothesis(novelty=0.71, plausibility=0.68, ...)
    ```

=== "Pillar III: UC"

    **Uncertainty Crystallization**

    Converts overconfident, miscalibrated outputs into statistically-guaranteed
    confidence tiers using MC-Dropout + Temperature Scaling + Conformal Prediction.

    ```python
    crystal = uc.crystallize(input_ids)
    # CrystalizedUncertainty(tier='fluid', calibrated=0.43, ci=(0.28, 0.58))
    ```

---

## Quick Start

```bash
pip install phantasm-llm
```

```python
from phantasm import PHANTASMPipeline

pipeline = PHANTASMPipeline.from_pretrained("gpt2")
report = pipeline.analyze("The Amazon flows through Africa.", domain="geography")
print(report)
```

---

## Why PHANTASM is Different

| Framework | What it does with hallucinations |
|---|---|
| RAG | Tries to prevent them |
| RLHF | Trains away from them |
| Fact-checkers | Catches them after generation |
| SelfCheckGPT | Detects them via sampling |
| **PHANTASM** | **Converts them into knowledge maps, hypotheses, and confidence oracles** |

---

## Author

**Vignesh S** — B.Tech CSE, Takshashila University, 2022–2026.
GitHub: [vignesh2027](https://github.com/vignesh2027)
HuggingFace: [vigneshwar234](https://huggingface.co/vigneshwar234)

Apache 2.0 License.
