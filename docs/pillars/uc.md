# Pillar III — Uncertainty Crystallization (UC)

## Core idea

Epistemic miscalibration — where a model is wrong but confident — encodes structural information about overrepresented training distributions. UC converts this into a calibrated, statistically guaranteed confidence oracle using three stages: MC-Dropout, Temperature Scaling, and Conformal Prediction.

## Usage

```python
from phantasm.core.uc import UncertaintyCrystallizer

uc = UncertaintyCrystallizer(model, mc_samples=30, coverage=0.90)
crystal = uc.crystallize(input_ids)

print(crystal.reliability_tier)           # "crystal" | "solid" | "fluid" | "vapor"
print(crystal.calibrated_confidence)      # float in [0, 1]
print(crystal.conformal_interval)         # (lower, upper) — 90% statistical guarantee
print(crystal.action_recommendation)      # human-readable next step
```

## Reliability tiers

| Tier | Symbol | Condition | Action |
|---|---|---|---|
| Crystal | ◆ | conf ≥ 0.85, epistemic < 0.05 | Use directly |
| Solid | ◇ | conf ≥ 0.65, epistemic < 0.15 | Light verification |
| Fluid | ≈ | conf ≥ 0.40, epistemic < 0.35 | Verify before use |
| Vapor | ~ | conf < 0.40 | Do not use unverified |

## CrystalizedUncertainty fields

| Field | Type | Description |
|---|---|---|
| `raw_confidence` | float | Original uncalibrated confidence. |
| `calibrated_confidence` | float | After temperature scaling. |
| `epistemic_uncertainty` | float | MC-Dropout variance. |
| `aleatoric_uncertainty` | float | Expected entropy (irreducible). |
| `total_uncertainty` | float | Epistemic + aleatoric. |
| `conformal_interval` | tuple | (lower, upper) with guaranteed coverage. |
| `reliability_tier` | str | One of four tiers. |
| `action_recommendation` | str | What to do with this output. |
