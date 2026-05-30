# Training

## Training CMN

The Confabulation Mining Network can be trained on any (confabulation, reference) pair dataset.

```python
from transformers import AutoTokenizer
from phantasm.core.cmn import ConfabulationMiningNetwork
from phantasm.datasets.loader import PHANTASMDatasetLoader
from phantasm.training.trainer import PHANTASMTrainer, TrainingConfig

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

loader = PHANTASMDatasetLoader(tokenizer, max_length=128)
train_ds, val_ds = loader.load("faith_dial")
train_dl = loader.get_dataloader(train_ds, batch_size=16)
val_dl = loader.get_dataloader(val_ds, batch_size=16, shuffle=False)

model = ConfabulationMiningNetwork(vocab_size=tokenizer.vocab_size)
config = TrainingConfig(
    epochs=10,
    learning_rate=2e-5,
    device="cuda",
    save_dir="./checkpoints",
)

trainer = PHANTASMTrainer(model, train_dl, val_dl, config)
history = trainer.train()
```

## TrainingConfig options

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 10 | Number of training epochs |
| `batch_size` | 16 | Batch size |
| `learning_rate` | 2e-5 | AdamW learning rate |
| `weight_decay` | 0.01 | L2 regularization |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `device` | "cpu" | "cpu" / "cuda" / "mps" |
| `save_dir` | "./phantasm_checkpoints" | Checkpoint directory |
| `alpha` | 0.4 | HGT loss weight |
| `beta` | 0.3 | CMN loss weight |
| `gamma` | 0.3 | UC loss weight |

## Loss functions

PHANTASM uses a unified multi-task loss:

```
L_total = α * L_HGT  +  β * L_CMN  +  γ * L_UC
```

- **L_HGT**: Binary cross-entropy on per-token hallucination labels + gradient regularizer
- **L_CMN**: InfoNCE-style contrastive loss (novelty) + BCE coherence loss
- **L_UC**: NLL + Expected Calibration Error minimization
