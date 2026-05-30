# Pillar I — Hallucination Gradient Tracing (HGT)

## Core idea

Hallucinations occur at the edges of the model's training distribution. The gradient of the loss with respect to input embeddings spikes at exactly those positions — where the model is uncertain. HGT captures this spike and converts it into a **Competency Atlas**.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from phantasm.core.hgt import HallucinationGradientTracer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tracer = HallucinationGradientTracer(model, tokenizer, threshold=0.35)
atlas = tracer.trace("The Eiffel Tower was built in 1492 by Napoleon.")

print(atlas.overall_hallucination_risk)   # float in [0, 1]
print(atlas.boundary_tokens)              # ['1492', 'Napoleon']
print(atlas.knowledge_gaps)              # [{'span': '1492', 'confidence': 0.09, ...}]
```

## CompetencyAtlas fields

| Field | Type | Description |
|---|---|---|
| `token_scores` | Tensor (seq_len,) | Per-token confidence. 1=grounded, 0=boundary. |
| `layer_gradients` | Tensor (n_layers, seq_len) | Raw gradient norms per layer. |
| `boundary_tokens` | list[str] | Tokens below the confidence threshold. |
| `knowledge_gaps` | list[dict] | Structured gap info with severity. |
| `overall_hallucination_risk` | float | Aggregate risk for the full text. |

## Offline risk scoring (no model required)

```python
from phantasm.core.hgt import score_hallucination_risk

# Overlap-based (needs reference)
risk = score_hallucination_risk(generation, reference=ref, method="overlap")

# Entropy-based (no reference needed)
risk = score_hallucination_risk(generation, method="entropy")
```
