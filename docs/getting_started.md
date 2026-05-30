# Getting Started

## Installation

```bash
pip install phantasm-llm
```

For visualization support:

```bash
pip install "phantasm-llm[viz]"
```

Full development install:

```bash
git clone https://github.com/vignesh2027/PHANTASM.git
cd PHANTASM
pip install -e ".[all]"
```

## Your first analysis

```python
from phantasm import PHANTASMPipeline

# Load any HuggingFace causal LM
pipeline = PHANTASMPipeline.from_pretrained("gpt2")

# Analyse text — no reference needed
report = pipeline.analyze("The Eiffel Tower was built in 1492 by Napoleon.")
print(report)
```

## With a reference text

```python
report = pipeline.analyze(
    "The Amazon River flows through Africa.",
    reference_text="The Amazon River flows through South America.",
    domain="geography",
)
```

## Reading the report

```python
# Hallucination risk (0 = grounded, 1 = highly likely hallucinated)
print(report.competency_atlas.overall_hallucination_risk)

# Specific knowledge gaps
for gap in report.competency_atlas.knowledge_gaps:
    print(gap["span"], gap["confidence"], gap["severity"])

# Mined hypotheses
for h in report.mined_hypotheses:
    print(h.novelty_score, h.plausibility_score, h.text)

# Calibrated confidence + tier
uc = report.uncertainty
print(uc.reliability_tier)        # "crystal" | "solid" | "fluid" | "vapor"
print(uc.calibrated_confidence)   # float in [0, 1]
print(uc.conformal_interval)      # (lower, upper) with 90% statistical guarantee

# Human-readable synthesis
print(report.synthesis_summary)
for insight in report.actionable_insights:
    print("•", insight)

# Export as dict (for APIs, databases, logging)
data = report.to_dict()
```

## Visualization

```python
from phantasm.utils.visualization import PHANTASMVisualizer

# Terminal pretty-print
PHANTASMVisualizer.print_report(report)

# Matplotlib calibration curve (requires phantasm-llm[viz])
PHANTASMVisualizer.plot_calibration_curve(confidences, accuracies, save_path="cal.png")

# Hypothesis scatter plot
PHANTASMVisualizer.plot_hypothesis_scatter(report.mined_hypotheses, save_path="hyp.png")
```

## Using individual pillars

You can use HGT, CMN, and UC independently without the full pipeline:

```python
from phantasm.core.hgt import HallucinationGradientTracer
from phantasm.core.cmn import ConfabulationMiningNetwork
from phantasm.core.uc import UncertaintyCrystallizer
```

See the [Pillars documentation](pillars/hgt.md) for details.
