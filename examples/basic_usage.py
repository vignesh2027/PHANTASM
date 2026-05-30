"""
PHANTASM — Basic Usage Example
================================
Demonstrates all three pillars on a small GPT-2 model.
Run:
    python examples/basic_usage.py
"""

import torch
from phantasm import PHANTASMPipeline
from phantasm.utils.visualization import PHANTASMVisualizer

print("Loading PHANTASM pipeline (GPT-2)...")
pipeline = PHANTASMPipeline.from_pretrained("gpt2", device="cpu")

# --- Single analysis ---
text = (
    "The Eiffel Tower was constructed in 1492 by Napoleon Bonaparte "
    "as a symbol of French imperialism, standing 1200 metres tall in the "
    "heart of London, England."
)
reference = (
    "The Eiffel Tower was built between 1887 and 1889 by Gustave Eiffel "
    "for the 1889 World's Fair in Paris, France. It stands 330 metres tall."
)

print(f"\nAnalysing: '{text[:80]}...'")
report = pipeline.analyze(text, reference_text=reference, domain="history")

# Pretty print
PHANTASMVisualizer.print_report(report)

# Dict export
data = report.to_dict()
print("Exported report keys:", list(data.keys()))

# --- Batch usage ---
texts = [
    "Water boils at 50°C at sea level.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "The human brain uses exactly 10% of its capacity at any time.",
]

print("\n--- Batch Analysis ---")
for t in texts:
    r = pipeline.analyze(t)
    tier_sym = {"crystal": "◆", "solid": "◇", "fluid": "≈", "vapor": "~"}.get(
        r.uncertainty.reliability_tier, "?"
    )
    print(
        f"  {tier_sym} risk={r.competency_atlas.overall_hallucination_risk:.2f}"
        f"  conf={r.uncertainty.calibrated_confidence:.2f}"
        f"  hyp={len(r.mined_hypotheses)}"
        f"  | '{t[:55]}...'"
    )

print("\n✓ PHANTASM example complete.")
