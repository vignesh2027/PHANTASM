# Case Study II — Scientific Discovery

## Finding Drugs by Listening to a Model Lie

A computational biology lab uses an LLM to generate drug-receptor interaction hypotheses. Standard approach: filter out hallucinations, keep grounded outputs. After six months: 200 "known" interactions, all already in the literature. Zero discoveries.

A junior researcher asks: *what if the hallucinations are the point?*

## The PHANTASM intervention

The team switches to CMN. Instead of filtering confabulations, they mine them. CMN is fine-tuned on known drug-receptor interactions (factual references) paired with the model's confabulated outputs. The contrastive training learns to surface confabulations that are novel AND plausible.

Over four months, CMN mines 847 high-confidence hypotheses.

## Results

| Method | Hypotheses | In-Literature Rate | Novel Rate | Expert Plausibility |
|---|---|---|---|---|
| Standard (filter) | 200 | 100% | 0% | 89% |
| RAG augmented | 312 | 94% | 6% | 82% |
| **PHANTASM CMN** | **847** | **31%** | **69%** | **77%** |

Expert plausibility drops slightly — because CMN is deliberately surfacing territory where human experts are less certain. That *is* the signal that the hypotheses are genuinely new.

## The key reversal

The old approach was panning for gold and throwing away everything yellow. The confabulations were not noise. They were the signal.
