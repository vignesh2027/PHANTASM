"""
PHANTASM unified loss functions for joint HGT + CMN + UC training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class HGTLoss(nn.Module):
    """
    Hallucination Gradient Tracing loss.
    Supervised binary cross-entropy on per-token hallucination labels,
    combined with a gradient-magnitude regulariser.
    """

    def __init__(self, lambda_reg: float = 0.01) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        token_scores: torch.Tensor,
        labels: torch.Tensor,
        gradient_norms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy(token_scores, labels.float())
        reg = torch.zeros(1, device=token_scores.device)
        if gradient_norms is not None:
            reg = self.lambda_reg * gradient_norms.mean()
        return bce + reg


class CMNLoss(nn.Module):
    """
    Confabulation Mining Network training loss.
    InfoNCE-style contrastive loss to push confabulations away from
    factual space while keeping them internally coherent.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        confab_vecs: torch.Tensor,
        fact_vecs: torch.Tensor,
        plausibility_scores: torch.Tensor,
    ) -> torch.Tensor:
        sim = F.cosine_similarity(confab_vecs, fact_vecs, dim=-1)
        novelty_loss = -torch.log(torch.sigmoid((1.0 - sim) / self.temperature)).mean()
        coherence_loss = F.binary_cross_entropy(
            plausibility_scores,
            torch.ones_like(plausibility_scores),
        )
        return 0.6 * novelty_loss + 0.4 * coherence_loss


class UCLoss(nn.Module):
    """
    Uncertainty Crystallization calibration loss.
    Expected Calibration Error (ECE) minimisation + NLL.
    """

    def __init__(self, n_bins: int = 10) -> None:
        super().__init__()
        self.n_bins = n_bins

    def forward(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        nll = F.cross_entropy(probs, labels)
        ece = self._compute_ece(probs, labels)
        return nll + 0.5 * ece

    def _compute_ece(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=probs.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=probs.device)
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = accuracies[mask].float().mean()
                ece += (mask.sum().float() / len(probs)) * (bin_conf - bin_acc).abs()
        return ece


class PHANTASMLoss(nn.Module):
    """
    Unified multi-task loss combining all three PHANTASM pillars.

    L_total = α * L_HGT  +  β * L_CMN  +  γ * L_UC
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hgt_loss = HGTLoss()
        self.cmn_loss = CMNLoss()
        self.uc_loss = UCLoss()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        if "token_scores" in batch and "hgt_labels" in batch:
            losses["hgt"] = self.hgt_loss(batch["token_scores"], batch["hgt_labels"])
        else:
            losses["hgt"] = torch.tensor(0.0)

        if "confab_vecs" in batch and "fact_vecs" in batch and "plausibility" in batch:
            losses["cmn"] = self.cmn_loss(
                batch["confab_vecs"], batch["fact_vecs"], batch["plausibility"]
            )
        else:
            losses["cmn"] = torch.tensor(0.0)

        if "probs" in batch and "uc_labels" in batch:
            losses["uc"] = self.uc_loss(batch["probs"], batch["uc_labels"])
        else:
            losses["uc"] = torch.tensor(0.0)

        losses["total"] = (
            self.alpha * losses["hgt"]
            + self.beta * losses["cmn"]
            + self.gamma * losses["uc"]
        )
        return losses
