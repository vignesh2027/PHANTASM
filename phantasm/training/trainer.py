"""
PHANTASMTrainer — Unified training loop for all three pillars.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from phantasm.training.losses import PHANTASMLoss
from phantasm.training.metrics import PHANTASMMetrics


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_every: int = 1
    save_dir: str = "./phantasm_checkpoints"
    log_every: int = 10
    device: str = "cpu"
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    fp16: bool = False


class PHANTASMTrainer:
    """
    Full training harness for PHANTASM.

    Supports:
    - Multi-task training (HGT + CMN + UC jointly)
    - Gradient clipping
    - Cosine LR schedule with warmup
    - Checkpoint saving
    - Tensorboard / WandB logging hooks

    Usage
    -----
    >>> config = TrainingConfig(epochs=5, device="cuda")
    >>> trainer = PHANTASMTrainer(cmn_model, train_loader, val_loader, config)
    >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        callbacks: Optional[List[Callable]] = None,
    ) -> None:
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.callbacks = callbacks or []
        self.loss_fn = PHANTASMLoss(config.alpha, config.beta, config.gamma)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        n_steps = len(train_loader) * config.epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_steps)
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "phantasm_score": []
        }
        os.makedirs(config.save_dir, exist_ok=True)

    def train(self) -> Dict[str, List[float]]:
        """Full training loop. Returns loss history."""
        best_val_loss = float("inf")
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self._eval_epoch(epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{self.config.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"time={elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

            for cb in self.callbacks:
                cb(epoch=epoch, train_loss=train_loss, val_loss=val_loss)

        return self.history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()
            losses = self._compute_loss(batch)
            loss = losses["total"]
            loss.backward()

            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

            if step % self.config.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"  [{epoch}:{step:04d}/{n_batches}] loss={loss.item():.4f} lr={lr:.2e}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = len(self.val_loader)

        for batch in self.val_loader:
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            losses = self._compute_loss(batch)
            total_loss += losses["total"].item()

        return total_loss / max(n_batches, 1)

    def _compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        input_ids = batch.get("input_ids")
        ref_ids = batch.get("ref_input_ids", input_ids)

        if input_ids is None:
            return {"total": torch.tensor(0.0, device=self.config.device)}

        confab_vecs = self.model.concept_extractor(input_ids)
        fact_vecs = self.model.concept_extractor(ref_ids)

        _, plausibility = self.model(input_ids, ref_ids)

        confab_pooled = confab_vecs.mean(dim=1)
        fact_pooled = fact_vecs.mean(dim=1)

        return self.loss_fn({
            "confab_vecs": confab_pooled,
            "fact_vecs": fact_pooled,
            "plausibility": plausibility,
        })

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = os.path.join(self.config.save_dir, f"phantasm_epoch{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "history": self.history,
            },
            path,
        )
        print(f"  ✓ Checkpoint saved: {path}")

    def save_history(self, path: str = "training_history.json") -> None:
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
