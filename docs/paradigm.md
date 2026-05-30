# The Paradigm Shift

## What everyone else does

Every LLM reliability framework in existence operates on the same premise: **failure modes are bugs**. You suppress them, detect them, train away from them, or build guardrails around them.

- **RAG** prevents hallucinations by grounding generation in retrieved documents.
- **RLHF** trains the model away from hallucinating by penalizing it.
- **Fact-checkers** catch hallucinations after they occur and flag them.
- **SelfCheckGPT** samples multiple times and checks consistency.

All of these approaches treat the hallucination as a **waste product** — something to throw away.

## What PHANTASM does

PHANTASM starts from a different premise: **failure modes are signals**.

When a model hallucinates at a specific token position, it is not producing random noise. It is producing a precise, reproducible, mathematically characterizable signal about the boundary of its training distribution. When it confabulates a creative combination of concepts, it is exploring a region of its learned semantic manifold that no training document explicitly charted. When it is overconfident on a wrong answer, it is telling you exactly which training distribution was overrepresented.

**PHANTASM does not discard these signals. It harvests them.**

## The three inversions

| Failure | Old view | PHANTASM inversion |
|---|---|---|
| Hallucination | Error to suppress | Knowledge-boundary map |
| Confabulation | False output to discard | Hypothesis to mine |
| Miscalibration | Confidence bug | Uncertainty oracle |

## Why this hasn't been done before

The standard training pipeline optimizes **against** hallucination at the output level. This means the gradient signal from hallucinated outputs is used to reduce future hallucination — discarding the positional and structural information about WHERE and WHY the model hallucinated.

PHANTASM operates **without touching training**. It is a post-hoc, inference-time framework that wraps any existing model. No fine-tuning required. No dataset curation required. The failures your current model produces TODAY are already a rich dataset — PHANTASM reads them.

## The result

A model that hallucinates is not a broken model. It is a model that has not yet been read correctly.

PHANTASM reads it.
