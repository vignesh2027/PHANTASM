# Pillar II — Confabulation Mining Network (CMN)

## Core idea

Confabulations are not random noise — they are paths through the model's semantic manifold that training data never explicitly charted. CMN mines these paths for novel, plausible hypotheses using a contrastive dual-encoder architecture.

## Usage

```python
from phantasm.core.cmn import ConfabulationMiningNetwork

cmn = ConfabulationMiningNetwork(
    vocab_size=50257,
    hidden_dim=256,
    novelty_threshold=0.45,
    plausibility_threshold=0.50,
)

hypotheses = cmn.mine(confab_ids, fact_ids, texts=[text], domain="drug_discovery")
for h in hypotheses:
    print(f"novelty={h.novelty_score:.2f}  plausibility={h.plausibility_score:.2f}")
    print(h.text)
```

## Hypothesis fields

| Field | Type | Description |
|---|---|---|
| `text` | str | The full confabulated text. |
| `source_concepts` | list[str] | Key concepts extracted from the text. |
| `novelty_score` | float | Distance from factual reference space [0,1]. |
| `plausibility_score` | float | Internal semantic coherence [0,1]. |
| `domain` | str | Application domain label. |

## Training CMN

```python
from phantasm.training.trainer import PHANTASMTrainer, TrainingConfig

config = TrainingConfig(epochs=10, learning_rate=1e-4)
trainer = PHANTASMTrainer(cmn, train_dl, val_dl, config)
history = trainer.train()
```
