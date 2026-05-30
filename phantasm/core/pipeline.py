"""
PHANTASMPipeline — Unified entry point combining HGT + CMN + UC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from phantasm.core.hgt import HallucinationGradientTracer, CompetencyAtlas
from phantasm.core.cmn import ConfabulationMiningNetwork, Hypothesis
from phantasm.core.uc import UncertaintyCrystallizer, CrystalizedUncertainty


@dataclass
class PHANTASMReport:
    """
    Full PHANTASM analysis report for a single text input.

    Fields
    ------
    competency_atlas       : HGT output — knowledge-boundary map.
    mined_hypotheses       : CMN output — novel hypotheses extracted.
    uncertainty            : UC  output — calibrated confidence.
    synthesis_summary      : Human-readable paragraph summarising all three.
    actionable_insights    : Bullet-point list of what to do next.
    """
    competency_atlas: CompetencyAtlas
    mined_hypotheses: List[Hypothesis]
    uncertainty: CrystalizedUncertainty
    synthesis_summary: str
    actionable_insights: List[str]
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PHANTASMReport(\n"
            f"  hallucination_risk={self.competency_atlas.overall_hallucination_risk:.3f},\n"
            f"  hypotheses_mined={len(self.mined_hypotheses)},\n"
            f"  reliability_tier='{self.uncertainty.reliability_tier}',\n"
            f"  calibrated_conf={self.uncertainty.calibrated_confidence:.3f}\n"
            f")"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hallucination_risk": self.competency_atlas.overall_hallucination_risk,
            "boundary_tokens": self.competency_atlas.boundary_tokens,
            "knowledge_gaps": self.competency_atlas.knowledge_gaps,
            "hypotheses": [
                {
                    "text": h.text[:200],
                    "novelty": h.novelty_score,
                    "plausibility": h.plausibility_score,
                    "domain": h.domain,
                }
                for h in self.mined_hypotheses
            ],
            "uncertainty": {
                "raw_confidence": self.uncertainty.raw_confidence,
                "calibrated_confidence": self.uncertainty.calibrated_confidence,
                "epistemic": self.uncertainty.epistemic_uncertainty,
                "aleatoric": self.uncertainty.aleatoric_uncertainty,
                "tier": self.uncertainty.reliability_tier,
                "interval": self.uncertainty.conformal_interval,
                "recommendation": self.uncertainty.action_recommendation,
            },
            "synthesis": self.synthesis_summary,
            "insights": self.actionable_insights,
        }


class PHANTASMPipeline:
    """
    Unified PHANTASM pipeline — one call, full analysis.

    Usage
    -----
    >>> pipeline = PHANTASMPipeline.from_pretrained("gpt2")
    >>> report = pipeline.analyze("The Eiffel Tower was built in 1492 by Napoleon.")
    >>> print(report)

    Parameters
    ----------
    model_name_or_path : str   HuggingFace model ID or local path.
    device             : str   "cpu" | "cuda" | "mps"
    hgt_threshold      : float Knowledge-boundary threshold for HGT.
    mc_samples         : int   MC-Dropout samples for UC.
    coverage           : float Conformal prediction coverage.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cpu",
        hgt_threshold: float = 0.35,
        mc_samples: int = 20,
        coverage: float = 0.90,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.hgt = HallucinationGradientTracer(
            model, tokenizer, threshold=hgt_threshold, device=device
        )
        self.cmn = ConfabulationMiningNetwork().to(device)
        self.uc = UncertaintyCrystallizer(
            model, mc_samples=mc_samples, coverage=coverage, device=device
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        device: str = "cpu",
        **kwargs,
    ) -> "PHANTASMPipeline":
        """Load a HuggingFace model and wrap it in the PHANTASM pipeline."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return cls(model, tokenizer, device=device, **kwargs)

    def analyze(
        self,
        text: str,
        reference_text: Optional[str] = None,
        domain: str = "general",
    ) -> PHANTASMReport:
        """
        Full three-pillar PHANTASM analysis.

        Parameters
        ----------
        text           : str   Text to analyse (model output or prompt).
        reference_text : str   Optional factual reference for CMN.
        domain         : str   Application domain label (for CMN hypotheses).

        Returns
        -------
        PHANTASMReport with all three pillars populated.
        """
        # --- Pillar I: HGT ---
        atlas = self.hgt.trace(text)

        # --- Pillar II: CMN ---
        ref = reference_text or text
        confab_enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256, padding="max_length"
        )
        fact_enc = self.tokenizer(
            ref, return_tensors="pt", truncation=True, max_length=256, padding="max_length"
        )
        confab_ids = confab_enc["input_ids"].to(self.device)
        fact_ids = fact_enc["input_ids"].to(self.device)

        hypotheses = self.cmn.mine(
            confab_ids, fact_ids, texts=[text], domain=domain
        )

        # --- Pillar III: UC ---
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        uc_result = self.uc.crystallize(
            enc["input_ids"].to(self.device),
            enc.get("attention_mask", None),
        )

        # --- Synthesis ---
        summary, insights = self._synthesize(atlas, hypotheses, uc_result)

        return PHANTASMReport(
            competency_atlas=atlas,
            mined_hypotheses=hypotheses,
            uncertainty=uc_result,
            synthesis_summary=summary,
            actionable_insights=insights,
            metadata={"text_length": len(text), "domain": domain},
        )

    def _synthesize(
        self,
        atlas: CompetencyAtlas,
        hypotheses: List[Hypothesis],
        uc: CrystalizedUncertainty,
    ) -> tuple:
        risk = atlas.overall_hallucination_risk
        tier = uc.reliability_tier
        n_hyp = len(hypotheses)

        summary = (
            f"PHANTASM analysis complete. "
            f"Hallucination risk: {risk:.1%}. "
            f"Reliability tier: {tier.upper()} (calibrated confidence {uc.calibrated_confidence:.1%}). "
            f"{n_hyp} novel hypotheses extracted from confabulation patterns. "
            f"Epistemic uncertainty: {uc.epistemic_uncertainty:.4f}. "
            f"90% conformal interval: {uc.conformal_interval}."
        )

        insights: List[str] = []

        if risk > 0.6:
            insights.append(
                f"HIGH hallucination risk ({risk:.1%}): "
                f"Target {len(atlas.boundary_tokens)} boundary tokens for RAG retrieval."
            )
        elif risk > 0.3:
            insights.append(
                f"MODERATE hallucination risk ({risk:.1%}): "
                f"Verify {len(atlas.knowledge_gaps)} knowledge gaps with secondary sources."
            )
        else:
            insights.append(f"LOW hallucination risk ({risk:.1%}): Output is broadly grounded.")

        if n_hyp > 0:
            best = max(hypotheses, key=lambda h: h.novelty_score * h.plausibility_score)
            insights.append(
                f"Top hypothesis (novelty={best.novelty_score:.2f}, "
                f"plausibility={best.plausibility_score:.2f}): '{best.text[:100]}...'"
            )

        insights.append(f"Uncertainty recommendation: {uc.action_recommendation}")
        insights.append(
            f"Fine-tuning target: collect {len(atlas.boundary_tokens)} "
            f"boundary-token examples to reduce future hallucination."
        )

        return summary, insights
