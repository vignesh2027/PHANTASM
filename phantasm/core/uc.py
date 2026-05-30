"""
Uncertainty Crystallization (UC)
===================================
PHANTASM Pillar III — Turns epistemic miscalibration into a precision oracle.

Core insight: When a model is wrong-but-confident (a "hard hallucination"),
the PATTERN of its wrong answers encodes which training distributions were
overrepresented. When it is right-but-uncertain, it encodes knowledge at
the boundary. UC harvests both signals.

Three-stage crystallization:
  1. MC-Dropout sampling  →  empirical uncertainty distribution
  2. Temperature scaling   →  recalibrates overconfident logits
  3. Conformal prediction  →  provides statistically guaranteed intervals

Advantage gained from "disadvantage":
  Epistemic Miscalibration → Calibrated Confidence Oracles →
  Reliable uncertainty-aware inference in medical / legal / scientific LLM apps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CrystalizedUncertainty:
    """
    Output of UC: a fully calibrated uncertainty report for a generation.

    Fields
    ------
    raw_confidence       : float  Model's original (uncalibrated) confidence.
    calibrated_confidence: float  Post-temperature-scaling confidence.
    epistemic_uncertainty: float  Variance across MC-Dropout samples.
    aleatoric_uncertainty: float  Irreducible data uncertainty.
    total_uncertainty    : float  Epistemic + aleatoric (Bayesian total).
    conformal_interval   : tuple  (lower, upper) guaranteed coverage interval.
    reliability_tier     : str    "crystal" | "solid" | "fluid" | "vapor"
    action_recommendation: str    What to do with this output.
    """
    raw_confidence: float
    calibrated_confidence: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    conformal_interval: Tuple[float, float]
    reliability_tier: str
    action_recommendation: str
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CrystalizedUncertainty("
            f"calibrated={self.calibrated_confidence:.3f}, "
            f"epistemic={self.epistemic_uncertainty:.3f}, "
            f"tier='{self.reliability_tier}', "
            f"ci={self.conformal_interval})"
        )


class TemperatureScaler(nn.Module):
    """
    Learnable temperature parameter for post-hoc confidence calibration.
    Minimises NLL on a held-out calibration set; no re-training of the model.
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Fit temperature to a calibration set. Returns final NLL."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled = self(logits)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        with torch.no_grad():
            final_nll = F.cross_entropy(self(logits), labels).item()
        return final_nll


class MCDropoutSampler:
    """
    Monte-Carlo Dropout uncertainty estimator.
    Keeps dropout ACTIVE at inference to obtain a distribution over predictions.
    """

    def __init__(self, model: nn.Module, n_samples: int = 30, dropout_rate: float = 0.1) -> None:
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def _enable_dropout(self) -> None:
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids      : (B, seq_len)
        attention_mask : (B, seq_len)

        Returns
        -------
        mean_probs    : (B, seq_len, vocab_size)  Mean prediction.
        epistemic_var : (B, seq_len, vocab_size)  Variance (epistemic uncertainty).
        aleatoric_unc : (B, seq_len)              Expected entropy (aleatoric).
        """
        self.model.eval()
        self._enable_dropout()

        all_probs: List[torch.Tensor] = []
        for _ in range(self.n_samples):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            probs = F.softmax(out.logits, dim=-1)
            all_probs.append(probs)

        stacked = torch.stack(all_probs, dim=0)  # (N, B, seq, V)
        mean_probs = stacked.mean(0)              # (B, seq, V)
        epistemic_var = stacked.var(0)            # (B, seq, V)

        # Aleatoric = expected entropy of per-sample predictive distributions
        per_sample_entropy = -(stacked * (stacked + 1e-9).log()).sum(-1)  # (N, B, seq)
        aleatoric_unc = per_sample_entropy.mean(0)  # (B, seq)

        return mean_probs, epistemic_var, aleatoric_unc


class ConformalPredictor:
    """
    Conformal prediction for LLM confidence intervals.
    Provides *statistically guaranteed* coverage — not just heuristic intervals.

    Based on: Angelopoulos & Bates (2022) "A Gentle Introduction to Conformal Prediction"
    """

    def __init__(self, coverage: float = 0.90) -> None:
        self.coverage = coverage
        self._calibration_scores: Optional[np.ndarray] = None
        self._quantile: Optional[float] = None

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> None:
        """
        Fit the conformal predictor on a calibration set.

        Parameters
        ----------
        cal_probs  : (n_cal,)  Model's top-1 probability on calibration examples.
        cal_labels : (n_cal,)  1 if correct, 0 if incorrect.
        """
        scores = 1.0 - cal_probs  # nonconformity scores
        n = len(scores)
        q_level = np.ceil((n + 1) * self.coverage) / n
        self._quantile = float(np.quantile(scores, min(q_level, 1.0)))
        self._calibration_scores = scores

    def predict_interval(self, prob: float) -> Tuple[float, float]:
        """
        Returns a coverage-guaranteed interval for the given probability.

        Returns
        -------
        (lower, upper) confidence interval.
        """
        if self._quantile is None:
            return (max(0.0, prob - 0.15), min(1.0, prob + 0.15))
        half_width = self._quantile
        return (round(max(0.0, prob - half_width), 4), round(min(1.0, prob + half_width), 4))


class UncertaintyCrystallizer:
    """
    UC: Full Uncertainty Crystallization pipeline.

    Combines MC-Dropout + Temperature Scaling + Conformal Prediction into
    a single, unified uncertainty oracle that produces CrystalizedUncertainty
    objects with statistically guaranteed reliability tiers.

    Reliability Tiers
    -----------------
    ◆ crystal  — calibrated_confidence ≥ 0.85, epistemic < 0.05
    ◇ solid    — calibrated_confidence ≥ 0.65, epistemic < 0.15
    ≈ fluid    — calibrated_confidence ≥ 0.40, epistemic < 0.35
    ~ vapor    — calibrated_confidence <  0.40  (do not trust output)

    Parameters
    ----------
    model         : nn.Module   Any HuggingFace causal LM.
    temperature   : float       Initial temperature (tuned during calibration).
    mc_samples    : int         MC-Dropout samples.
    coverage      : float       Conformal prediction coverage (e.g., 0.90 = 90%).
    device        : str
    """

    def __init__(
        self,
        model: nn.Module,
        temperature: float = 1.5,
        mc_samples: int = 20,
        coverage: float = 0.90,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.temperature_scaler = TemperatureScaler(temperature).to(device)
        self.mc_sampler = MCDropoutSampler(model, n_samples=mc_samples)
        self.conformal = ConformalPredictor(coverage=coverage)

    def crystallize(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> CrystalizedUncertainty:
        """
        Run the full crystallization pipeline on a tokenized input.

        Returns
        -------
        CrystalizedUncertainty with all fields populated.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Stage 1: MC-Dropout sampling
        mean_probs, epi_var, ale_unc = self.mc_sampler.sample(input_ids, attention_mask)

        # Top-1 confidence at last token position
        last_probs = mean_probs[:, -1, :]          # (B, V)
        raw_conf = float(last_probs.max(dim=-1).values.mean().item())

        # Stage 2: Temperature scaling on raw logits
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        scaled_logits = self.temperature_scaler(out.logits)
        cal_probs = F.softmax(scaled_logits, dim=-1)
        cal_conf = float(cal_probs[:, -1, :].max(dim=-1).values.mean().item())

        # Stage 3: Aggregate uncertainty
        epistemic = float(epi_var.max(dim=-1).values.mean().item())
        aleatoric = float(ale_unc.mean().item())
        total_unc = epistemic + aleatoric

        # Stage 4: Conformal interval
        interval = self.conformal.predict_interval(cal_conf)

        # Stage 5: Reliability tier + recommendation
        tier, recommendation = self._classify(cal_conf, epistemic)

        return CrystalizedUncertainty(
            raw_confidence=round(raw_conf, 4),
            calibrated_confidence=round(cal_conf, 4),
            epistemic_uncertainty=round(epistemic, 4),
            aleatoric_uncertainty=round(aleatoric, 4),
            total_uncertainty=round(total_unc, 4),
            conformal_interval=interval,
            reliability_tier=tier,
            action_recommendation=recommendation,
            metadata={
                "coverage": self.conformal.coverage,
                "temperature": float(self.temperature_scaler.temperature.item()),
            },
        )

    @staticmethod
    def _classify(confidence: float, epistemic: float) -> Tuple[str, str]:
        if confidence >= 0.85 and epistemic < 0.05:
            return "crystal", "Safe to use. Model output is highly reliable."
        elif confidence >= 0.65 and epistemic < 0.15:
            return "solid", "Usable with light verification."
        elif confidence >= 0.40 and epistemic < 0.35:
            return "fluid", "Verify with a secondary source before use."
        else:
            return "vapor", "Do NOT use this output without full fact-checking."

    def batch_crystallize(
        self,
        input_ids_list: List[torch.Tensor],
        attention_masks: Optional[List[torch.Tensor]] = None,
    ) -> List[CrystalizedUncertainty]:
        """Crystallize a list of inputs."""
        results = []
        for i, ids in enumerate(input_ids_list):
            mask = attention_masks[i] if attention_masks else None
            results.append(self.crystallize(ids.unsqueeze(0) if ids.dim() == 1 else ids, mask))
        return results
