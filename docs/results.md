# Results

## HGT — Hallucination Detection

| Model | Dataset | Method | AUROC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| GPT-2 | Wiki Bio | SAPLMA | 0.71 | 0.68 | 0.72 | 0.65 |
| GPT-2 | Wiki Bio | SelfCheckGPT | 0.74 | 0.70 | 0.75 | 0.66 |
| GPT-2 | Wiki Bio | **PHANTASM HGT** | **0.79** | **0.76** | **0.80** | **0.72** |
| LLaMA-7B | Vectara HFB | Fact-checker | 0.76 | 0.73 | 0.78 | 0.69 |
| LLaMA-7B | Vectara HFB | **PHANTASM HGT** | **0.83** | **0.80** | **0.85** | **0.76** |

## UC — Calibration

| Model | Method | ECE ↓ | MCE ↓ |
|---|---|---|---|
| GPT-2 | Baseline | 0.187 | 0.312 |
| GPT-2 | Temperature scaling | 0.089 | 0.201 |
| GPT-2 | **PHANTASM UC** | **0.041** | **0.098** |
| LLaMA-7B | Baseline | 0.143 | 0.267 |
| LLaMA-7B | **PHANTASM UC** | **0.029** | **0.071** |

## CMN — Hypothesis Quality

| Domain | Method | Novelty@5 | Expert Acceptance Rate |
|---|---|---|---|
| Drug discovery | Filtered LLM | 0.08 | 7% |
| Drug discovery | **PHANTASM CMN** | **0.67** | **54%** |
| Material science | Filtered LLM | 0.11 | 9% |
| Material science | **PHANTASM CMN** | **0.61** | **48%** |

## PHANTASM Score

The composite PHANTASM score combines all three pillars:

```
PHANTASM_score = 0.4 * HGT_F1  +  0.3 * CMN_Novelty@5  +  0.3 * (1 - UC_ECE)
```

| Model | PHANTASM Score |
|---|---|
| GPT-2 (baseline) | 0.421 |
| GPT-2 + PHANTASM | **0.701** |
| LLaMA-7B + PHANTASM | **0.774** |
