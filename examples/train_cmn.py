"""
Train the Confabulation Mining Network on PHANTASM synthetic data.
Run:
    python examples/train_cmn.py
"""

import torch
from transformers import AutoTokenizer

from phantasm.core.cmn import ConfabulationMiningNetwork
from phantasm.datasets.loader import PHANTASMDatasetLoader
from phantasm.training.trainer import PHANTASMTrainer, TrainingConfig

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

loader = PHANTASMDatasetLoader(tokenizer, max_length=128)
train_ds, val_ds = loader.load("phantasm_synthetic")
train_dl = loader.get_dataloader(train_ds, batch_size=4, shuffle=True)
val_dl = loader.get_dataloader(val_ds, batch_size=4, shuffle=False)

model = ConfabulationMiningNetwork(vocab_size=tokenizer.vocab_size)
config = TrainingConfig(epochs=3, learning_rate=1e-4, save_dir="./checkpoints_cmn")

trainer = PHANTASMTrainer(model, train_dl, val_dl, config)
history = trainer.train()

print("\nTraining complete.")
print("Final train loss:", history["train_loss"][-1])
print("Final val loss  :", history["val_loss"][-1])
trainer.save_history("cmn_history.json")
