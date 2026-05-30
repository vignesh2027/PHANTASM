# Case Study I — Medical AI

## When the Doctor Doesn't Know What It Doesn't Know

A hospital system in Chennai deploys an LLM assistant for drug-interaction lookup. The model is GPT-class, fine-tuned on medical literature. Within three weeks, a near-miss: the model confidently recommends a drug combination contraindicated for a rare liver enzyme variant — present in only 0.3% of the training corpus. Raw confidence: 0.91. Human caught it by luck.

## The PHANTASM intervention

PHANTASM is wrapped around every inference call.

**HGT** traces gradients on drug-interaction queries. For common pairs (aspirin + ibuprofen), gradient norms are flat — uniformly grounded. For the rare enzyme variant, gradient norms spike at three tokens: the enzyme variant name, the drug compound, the dosage figure. These are flagged as `knowledge_gaps` with severity `"high"`.

**UC** crystallizes the uncertainty. Raw confidence: 0.91. After MC-Dropout (N=30) and temperature scaling, calibrated confidence: 0.43. Tier: `fluid`. Conformal interval: (0.28, 0.58). The system routes all `fluid` and `vapor` responses to human specialists automatically.

**CMN** mines the confabulation space and surfaces a plausible hypothesis: the enzyme variant may potentiate hepatotoxic effects. Flagged for pharmacologist review — confirmed in literature three weeks later.

## Outcome

- Near-miss class of events: zero in the following six months
- CMN hypotheses led to two literature searches and one ongoing research collaboration
- The model's hallucination was not a failure — it was the most precise diagnostic signal in the system
