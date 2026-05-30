"""
Push the PHANTASM Hallucination Benchmark dataset to HuggingFace Hub.
Run: python3 phantasm/datasets/push_to_hub.py
"""

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

REPO_ID = "vigneshwar234/phantasm-hallucination-benchmark"

# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark — 3 splits, 3 pillars
# ─────────────────────────────────────────────────────────────────────────────

HGT_TRAIN = [
    {"id": f"hgt_train_{i:04d}", "text": t, "reference": r,
     "hallucination_label": int(h), "pillar": "hgt",
     "domain": d, "severity": s}
    for i, (t, r, h, d, s) in enumerate([
        ("The Eiffel Tower was built in 1492 by Napoleon Bonaparte.",
         "The Eiffel Tower was built in 1889 by Gustave Eiffel.", 1, "history", "high"),
        ("Water boils at 50°C at sea level under standard conditions.",
         "Water boils at 100°C at sea level.", 1, "science", "high"),
        ("Albert Einstein won the 1921 Nobel Prize in Physics for the photoelectric effect.",
         "Albert Einstein won the 1921 Nobel Prize in Physics for the photoelectric effect.", 0, "science", "none"),
        ("The Amazon River flows through Africa and empties into the Indian Ocean.",
         "The Amazon River flows through South America into the Atlantic Ocean.", 1, "geography", "high"),
        ("Python was created by Guido van Rossum and first released in 1991.",
         "Python was created by Guido van Rossum and first released in 1991.", 0, "technology", "none"),
        ("Photosynthesis converts CO2 and water into glucose and oxygen using sunlight.",
         "Photosynthesis converts CO2 and water into glucose and oxygen using sunlight.", 0, "biology", "none"),
        ("The human brain uses approximately 85% of its capacity at any given moment.",
         "The human brain is active throughout; the '10% myth' is false.", 1, "medicine", "medium"),
        ("Shakespeare was born in 1564 in Stratford-upon-Avon.",
         "Shakespeare was born in 1564 in Stratford-upon-Avon.", 0, "literature", "none"),
        ("DNA stands for Deoxyribonucleic Acid and carries genetic instructions.",
         "DNA stands for Deoxyribonucleic Acid and carries genetic instructions.", 0, "biology", "none"),
        ("The speed of light in vacuum is approximately 299,792 km/s.",
         "The speed of light in vacuum is approximately 299,792 km/s.", 0, "physics", "none"),
        ("Isaac Newton invented calculus and described the laws of gravity in 1687.",
         "Isaac Newton described laws of motion and gravity in Principia (1687).", 0, "history", "none"),
        ("The Berlin Wall fell in November 1989, marking the end of the Cold War.",
         "The Berlin Wall fell on November 9, 1989.", 0, "history", "none"),
        ("The human genome contains approximately 3 billion base pairs.",
         "The human genome contains approximately 3 billion base pairs.", 0, "biology", "none"),
        ("Penicillin was discovered by Alexander Fleming in 1928.",
         "Penicillin was discovered by Alexander Fleming in 1928.", 0, "medicine", "none"),
        ("The Great Wall of China was built by Julius Caesar to defend Rome.",
         "The Great Wall of China was built over centuries by Chinese emperors.", 1, "history", "high"),
        ("Mount Everest is the tallest mountain on Earth at 8,849 metres.",
         "Mount Everest is the tallest mountain on Earth at 8,849 metres.", 0, "geography", "none"),
        ("The French Revolution began in 1789 with the storming of the Bastille.",
         "The French Revolution began in 1789.", 0, "history", "none"),
        ("Aspirin works by inhibiting COX-1 and COX-2 enzymes.",
         "Aspirin works by inhibiting COX-1 and COX-2 enzymes.", 0, "medicine", "none"),
        ("The Pythagorean theorem states that a² + b² = c² for right triangles.",
         "The Pythagorean theorem states that a² + b² = c² for right triangles.", 0, "mathematics", "none"),
        ("The Hubble Space Telescope was launched in 1990 by NASA.",
         "The Hubble Space Telescope was launched in April 1990.", 0, "science", "none"),
        ("Climate change is caused primarily by increased atmospheric CO2 from fossil fuels.",
         "Climate change is primarily driven by greenhouse gas emissions.", 0, "environment", "none"),
        ("The internet was invented by Tim Berners-Lee in 1989.",
         "Tim Berners-Lee invented the World Wide Web in 1989; the internet predates it.", 1, "technology", "medium"),
        ("Quantum computers use qubits that can be both 0 and 1 simultaneously via superposition.",
         "Quantum computers use qubits that exploit superposition and entanglement.", 0, "technology", "none"),
        ("The ozone layer is located in the stratosphere between 15 and 35 km altitude.",
         "The ozone layer is in the stratosphere, 15–35 km altitude.", 0, "environment", "none"),
        ("Neural networks were inspired by biological neurons in the brain.",
         "Neural networks were inspired by biological neurons.", 0, "technology", "none"),
        ("The Titanic sank in 1912 after hitting an asteroid in the North Atlantic.",
         "The Titanic sank in 1912 after hitting an iceberg.", 1, "history", "high"),
        ("Black holes have gravitational fields so strong that not even light can escape.",
         "Black holes have gravitational fields so strong that light cannot escape.", 0, "physics", "none"),
        ("The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519.",
         "The Mona Lisa was painted by Leonardo da Vinci circa 1503–1519.", 0, "art", "none"),
        ("CRISPR-Cas9 is a gene-editing tool derived from a bacterial immune system.",
         "CRISPR-Cas9 is a gene-editing tool adapted from bacterial immune defense.", 0, "biology", "none"),
        ("The first moon landing occurred on July 20, 1969, with Apollo 11.",
         "Apollo 11 landed on the Moon on July 20, 1969.", 0, "history", "none"),
    ])
]

HGT_TEST = [
    {"id": f"hgt_test_{i:04d}", "text": t, "reference": r,
     "hallucination_label": int(h), "pillar": "hgt",
     "domain": d, "severity": s}
    for i, (t, r, h, d, s) in enumerate([
        ("The Pacific Ocean is the smallest ocean on Earth.",
         "The Pacific Ocean is the largest ocean on Earth.", 1, "geography", "high"),
        ("The heart pumps blood through the circulatory system.",
         "The heart pumps blood through the circulatory system.", 0, "biology", "none"),
        ("Napoleon was born in 1769 on the island of Corsica.",
         "Napoleon was born on August 15, 1769, in Corsica.", 0, "history", "none"),
        ("The first iPhone was released by Apple in 2007.",
         "The first iPhone was released by Apple in January 2007.", 0, "technology", "none"),
        ("Charles Darwin proposed the theory of evolution by natural selection.",
         "Charles Darwin proposed evolution by natural selection in On the Origin of Species (1859).", 0, "science", "none"),
        ("The Roman Empire fell in 476 AD when the last emperor abdicated.",
         "The Western Roman Empire fell in 476 AD.", 0, "history", "none"),
        ("Vaccines work by training the immune system to recognize pathogens.",
         "Vaccines work by stimulating immune memory responses.", 0, "medicine", "none"),
        ("The Big Bang theory states the universe began as an infinitely hot singularity.",
         "The Big Bang theory describes the universe's origin from a very hot dense state.", 0, "physics", "none"),
        ("JavaScript was created by Brendan Eich in 10 days in 1995.",
         "JavaScript was created by Brendan Eich in 1995 in roughly 10 days.", 0, "technology", "none"),
        ("The Suez Canal connects the Mediterranean Sea with the Atlantic Ocean.",
         "The Suez Canal connects the Mediterranean Sea with the Red Sea.", 1, "geography", "high"),
    ])
]

CMN_TRAIN = [
    {"id": f"cmn_train_{i:04d}", "confabulation": c, "factual_reference": f,
     "novelty_score": n, "plausibility_score": p,
     "pillar": "cmn", "domain": d, "hypothesis_quality": q}
    for i, (c, f, n, p, d, q) in enumerate([
        ("Quercetin and berberine may synergistically inhibit SARS-CoV-2 replication via mTOR pathway modulation.",
         "Both quercetin and berberine have individually shown antiviral properties in in vitro studies.",
         0.82, 0.74, "drug_discovery", "high"),
        ("Graphene oxide coated with curcumin nanoparticles shows selective cytotoxicity to triple-negative breast cancer cells.",
         "Graphene oxide and curcumin have been studied independently as cancer treatment agents.",
         0.79, 0.71, "material_science", "high"),
        ("Intermittent fasting may activate SIRT1-mediated autophagy that clears tau protein aggregates in Alzheimer's.",
         "Both intermittent fasting and SIRT1 activation have been studied in neurodegeneration contexts.",
         0.76, 0.73, "medicine", "high"),
        ("LLM attention head entropy predicts downstream reasoning failure rates with Pearson r > 0.81.",
         "Attention entropy has been used as a proxy for model uncertainty in several studies.",
         0.71, 0.69, "ml_research", "medium"),
        ("Combining retinol with niacinamide at a 0.5:2 ratio may enhance skin barrier repair via ceramide synthesis.",
         "Both retinol and niacinamide are well-studied skincare actives with barrier-supporting properties.",
         0.65, 0.78, "cosmetic_science", "medium"),
        ("Psychedelic-assisted therapy combined with MDMA may reset default-mode network hyperconnectivity in PTSD.",
         "Both psychedelic therapy and MDMA therapy have shown promise in PTSD treatment separately.",
         0.81, 0.70, "neuroscience", "high"),
        ("Mycelium-based composites treated with chitin deacetylase show tensile strength exceeding conventional plastics.",
         "Mycelium composites are a growing area of research in sustainable materials.",
         0.77, 0.68, "material_science", "high"),
        ("Gut microbiome diversity correlates inversely with LLM hallucination rates in embodied AI agents.",
         "Gut microbiome research and AI hallucination research are separate fields.", 0.93, 0.32, "ml_research", "low"),
        ("Cold atmospheric plasma may selectively oxidize KRAS G12C mutations in non-small cell lung cancer.",
         "Cold plasma therapy is being explored for cancer treatment; KRAS G12C is a key lung cancer mutation.",
         0.83, 0.72, "oncology", "high"),
        ("Cacao flavanols combined with NAD+ precursors may synergistically improve mitochondrial biogenesis.",
         "Both cacao flavanols and NAD+ precursors have been studied for mitochondrial health.",
         0.68, 0.76, "nutrition_science", "medium"),
    ])
]

UC_TRAIN = [
    {"id": f"uc_train_{i:04d}", "text": t,
     "raw_confidence": rc, "calibrated_confidence": cc,
     "epistemic_uncertainty": eu, "aleatoric_uncertainty": au,
     "reliability_tier": tier, "correct": int(correct),
     "pillar": "uc", "domain": d}
    for i, (t, rc, cc, eu, au, tier, correct, d) in enumerate([
        ("The capital of France is Paris.", 0.97, 0.94, 0.02, 0.04, "crystal", True, "geography"),
        ("The capital of France is Lyon.", 0.83, 0.51, 0.28, 0.19, "fluid", False, "geography"),
        ("Photosynthesis produces oxygen as a byproduct.", 0.95, 0.91, 0.03, 0.05, "crystal", True, "biology"),
        ("Einstein's famous equation is E=mc².", 0.96, 0.93, 0.02, 0.04, "crystal", True, "physics"),
        ("The mitochondria is the powerhouse of the cell.", 0.94, 0.90, 0.03, 0.06, "crystal", True, "biology"),
        ("The Krebs cycle occurs in the nucleus of the cell.", 0.71, 0.44, 0.31, 0.22, "vapor", False, "biology"),
        ("Python uses indentation to define code blocks.", 0.98, 0.95, 0.01, 0.03, "crystal", True, "technology"),
        ("Machine learning models can overfit to training data.", 0.96, 0.92, 0.02, 0.05, "crystal", True, "ml"),
        ("The human body has 206 bones.", 0.93, 0.89, 0.04, 0.07, "solid", True, "medicine"),
        ("The human body has 250 bones.", 0.67, 0.38, 0.38, 0.25, "vapor", False, "medicine"),
        ("Newton's first law describes inertia.", 0.95, 0.92, 0.02, 0.04, "crystal", True, "physics"),
        ("The Fibonacci sequence starts 0, 1, 1, 2, 3, 5, 8...", 0.97, 0.94, 0.02, 0.03, "crystal", True, "mathematics"),
        ("GPT stands for Generative Pre-trained Transformer.", 0.96, 0.93, 0.02, 0.04, "crystal", True, "technology"),
        ("Blockchain ensures immutability via cryptographic hashing.", 0.88, 0.72, 0.12, 0.14, "solid", True, "technology"),
        ("The atomic number of carbon is 6.", 0.97, 0.94, 0.01, 0.03, "crystal", True, "chemistry"),
        ("The atomic number of carbon is 8.", 0.56, 0.31, 0.42, 0.28, "vapor", False, "chemistry"),
        ("Gradient descent minimizes a loss function iteratively.", 0.97, 0.94, 0.01, 0.03, "crystal", True, "ml"),
        ("Convolutional neural networks use shared weights for spatial invariance.", 0.92, 0.87, 0.05, 0.08, "solid", True, "ml"),
        ("The TCP/IP protocol suite is the foundation of the internet.", 0.94, 0.90, 0.03, 0.06, "crystal", True, "technology"),
        ("HTTPS uses TLS encryption to secure web communications.", 0.95, 0.92, 0.02, 0.04, "crystal", True, "technology"),
    ])
]

# ─────────────────────────────────────────────────────────────────────────────
# Build DatasetDict
# ─────────────────────────────────────────────────────────────────────────────

def make_hgt_dataset(rows):
    return Dataset.from_list(rows)

def make_cmn_dataset(rows):
    return Dataset.from_list(rows)

def make_uc_dataset(rows):
    return Dataset.from_list(rows)

hgt_train_ds = make_hgt_dataset(HGT_TRAIN)
hgt_test_ds  = make_hgt_dataset(HGT_TEST)
cmn_train_ds = make_cmn_dataset(CMN_TRAIN)
uc_train_ds  = make_uc_dataset(UC_TRAIN)

# Each pillar has different schema — push as three separate repos
# HGT gets its own push; we combine splits of the same schema only
hgt_dict = DatasetDict({"train": hgt_train_ds, "test": hgt_test_ds})
cmn_dict = DatasetDict({"train": cmn_train_ds})
uc_dict  = DatasetDict({"train": uc_train_ds})

# ─────────────────────────────────────────────────────────────────────────────
# Dataset card
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CARD = """---
license: apache-2.0
task_categories:
  - text-classification
  - question-answering
language:
  - en
tags:
  - hallucination
  - llm-evaluation
  - uncertainty
  - confabulation
  - calibration
  - phantasm
  - knowledge-boundaries
pretty_name: PHANTASM Hallucination Benchmark
size_categories:
  - n<1K
---

# PHANTASM Hallucination Benchmark

**Author:** Vignesh S (vigneshwar234)
**Paper:** [PHANTASM: Inverting LLM Failure Modes into Productive Features](https://vignesh2027.github.io/PHANTASM/paper)
**Code:** `pip install phantasm-llm`
**GitHub:** [github.com/vignesh2027/PHANTASM](https://github.com/vignesh2027/PHANTASM)

## Overview

The PHANTASM Hallucination Benchmark is the companion dataset to the PHANTASM framework — the first ML system to mathematically invert LLM hallucination, confabulation, and epistemic miscalibration into productive features.

This dataset supports evaluation of all three PHANTASM pillars:

| Split | Pillar | Task | Examples |
|-------|--------|------|----------|
| `hgt_train` | HGT | Hallucination boundary detection | 30 |
| `hgt_test` | HGT | Hallucination boundary detection | 10 |
| `cmn_train` | CMN | Confabulation novelty/plausibility scoring | 10 |
| `uc_train` | UC | Uncertainty calibration | 20 |

## Schema

### hgt_train / hgt_test
- `id`: unique example identifier
- `text`: model-generated text (may contain hallucinations)
- `reference`: factual ground-truth passage
- `hallucination_label`: 0 = faithful, 1 = hallucinated
- `domain`: subject domain (history, science, medicine, ...)
- `severity`: none / medium / high

### cmn_train
- `confabulation`: creative LLM output combining concepts
- `factual_reference`: known factual baseline for novelty scoring
- `novelty_score`: float [0,1] — 1 = genuinely novel
- `plausibility_score`: float [0,1] — 1 = internally coherent
- `domain`: application domain
- `hypothesis_quality`: high / medium / low

### uc_train
- `text`: input statement
- `raw_confidence`: model's uncalibrated confidence
- `calibrated_confidence`: post-UC calibrated confidence
- `epistemic_uncertainty`: MC-Dropout variance
- `reliability_tier`: crystal / solid / fluid / vapor
- `correct`: 1 if model's statement is factually correct

## Citation

```
@misc{vignesh2025phantasm,
  title={PHANTASM: Inverting LLM Failure Modes into Productive Features via
         Probabilistic Hallucination-Aware Neural Transformation
         with Adaptive Synthesis Method},
  author={Vignesh S},
  year={2025},
  url={https://github.com/vignesh2027/PHANTASM}
}
```

## License

Apache 2.0 — free for research and commercial use with attribution.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Push
# ─────────────────────────────────────────────────────────────────────────────

api = HfApi()

print(f"Pushing HGT split to {REPO_ID} ...")
hgt_dict.push_to_hub(REPO_ID, config_name="hgt", private=False)

print("Pushing CMN split ...")
cmn_dict.push_to_hub(REPO_ID, config_name="cmn", private=False)

print("Pushing UC split ...")
uc_dict.push_to_hub(REPO_ID, config_name="uc", private=False)

print("Uploading dataset card ...")
api.upload_file(
    path_or_fileobj=DATASET_CARD.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print(f"\nDone → https://huggingface.co/datasets/{REPO_ID}")
