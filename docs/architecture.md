# Architecture

## System Overview

```
                     ┌─────────────────────────────────────────┐
                     │           PHANTASMPipeline               │
                     │                                         │
   Input Text ──────►│  HGT ──► CompetencyAtlas                │
   + Optional Ref    │  CMN ──► List[Hypothesis]               │
                     │  UC  ──► CrystalizedUncertainty         │
                     │          ↓                              │
                     │      PHANTASMReport                     │
                     └─────────────────────────────────────────┘
```

## HGT: Hallucination Gradient Tracing

HGT is built on a simple but powerful observation: **the gradient of the loss with respect to input embeddings encodes model uncertainty**.

When the model is confident about a token (e.g., "Paris" in a sentence about French geography), the gradient at that token position is small — the model's representation doesn't need to change much. When the model is uncertain (e.g., an obscure date it never saw in training), the gradient is large.

HGT formalizes this as:

```
uncertainty(token_i) ∝ ||∂L/∂e_i||
```

where `e_i` is the embedding of token `i` and `L` is the self-consistency loss (cross-entropy against the model's own top-1 prediction).

This requires **no ground truth**, works on **any causal LM**, and runs in a **single forward-backward pass**.

## CMN: Confabulation Mining Network

CMN uses a **dual-encoder contrastive architecture**:

- **ConceptExtractor**: 2-layer Transformer encoder that maps token sequences to concept vectors
- **NoveltyScorer**: MLP that scores the distance between confabulated and factual concept spaces
- **PlausibilityScorer**: Self-attention module that scores internal semantic coherence

Training uses `ContrastiveMiningLoss`:

```
L = 0.6 * L_novelty + 0.4 * L_coherence
```

Where `L_novelty` pushes confabulation representations away from factual reference space, and `L_coherence` ensures the confabulation is internally consistent.

## UC: Uncertainty Crystallization

UC combines three calibration techniques in series:

1. **MC-Dropout** (Gal & Ghahramani, 2016): N forward passes with dropout active → empirical uncertainty distribution
2. **Temperature Scaling** (Guo et al., 2017): Post-hoc logit rescaling to minimize NLL → calibrated confidence
3. **Conformal Prediction** (Angelopoulos & Bates, 2022): Statistically guaranteed coverage intervals

The four reliability tiers are computed as:

```python
if confidence >= 0.85 and epistemic < 0.05:
    tier = "crystal"   # Use directly
elif confidence >= 0.65 and epistemic < 0.15:
    tier = "solid"     # Verify lightly
elif confidence >= 0.40 and epistemic < 0.35:
    tier = "fluid"     # Verify before use
else:
    tier = "vapor"     # Do not use
```
