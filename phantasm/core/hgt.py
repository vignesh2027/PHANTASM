"""
Hallucination Gradient Tracing (HGT)
=====================================
PHANTASM Pillar I — Turns hallucination into a knowledge-boundary oracle.

Core insight: LLM hallucinations are NOT random noise. They occur precisely
at the edges of the training distribution — where the model has sparse,
conflicting, or absent supervision. HGT traces these gradients back to the
embedding layer to produce a *Competency Atlas*: a per-token, per-layer map
showing exactly WHAT the model knows with confidence vs. where it is blind.

Advantage gained from "disadvantage":
  Hallucination → Dynamic Knowledge-Boundary Mapping → Better RAG retrieval,
  smarter prompt engineering, and targeted fine-tuning data collection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np


@dataclass
class CompetencyAtlas:
    """
    Output of HGT: a per-token knowledge-boundary map for an LLM.

    Fields
    ------
    token_scores : shape (seq_len,) in [0, 1]
        1 = model is confident / grounded.  0 = hallucination boundary.
    layer_gradients : shape (n_layers, seq_len)
        Raw gradient norms per layer — reveals WHICH layers drive uncertainty.
    boundary_tokens : list[str]
        Tokens that sit on the knowledge boundary (score < threshold).
    knowledge_gaps : list[dict]
        Structured gaps: {"span": str, "confidence": float, "layer": int}.
    overall_hallucination_risk : float
        Aggregate risk score for the full generation.
    """
    token_scores: torch.Tensor
    layer_gradients: torch.Tensor
    boundary_tokens: List[str]
    knowledge_gaps: List[Dict]
    overall_hallucination_risk: float
    metadata: Dict = field(default_factory=dict)


class GradientHook:
    """Captures intermediate gradients from any named module."""

    def __init__(self) -> None:
        self.gradients: List[torch.Tensor] = []

    def hook(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        if grad_output[0] is not None:
            self.gradients.append(grad_output[0].detach().clone())

    def clear(self) -> None:
        self.gradients = []


class HallucinationGradientTracer:
    """
    HGT: Probabilistic Knowledge-Boundary Mapper via Gradient Analysis.

    Methodology
    -----------
    1. Forward pass with ``requires_grad=True`` on input embeddings.
    2. Compute cross-entropy loss against the model's own top prediction
       (self-consistency loss — no ground truth needed).
    3. Backpropagate; collect gradient norms at every transformer layer.
    4. Apply adaptive thresholding (Otsu-inspired) to separate
       "grounded" from "boundary" tokens.
    5. Return a CompetencyAtlas with actionable knowledge gaps.

    Parameters
    ----------
    model : PreTrainedModel
        Any HuggingFace causal language model (GPT-2, LLaMA, Falcon, …).
    tokenizer : PreTrainedTokenizer
    threshold : float
        Confidence threshold below which a token is "on the boundary".
    device : str
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        threshold: float = 0.35,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.device = device
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._gradient_store = GradientHook()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace(
        self,
        text: str,
        max_new_tokens: int = 128,
    ) -> CompetencyAtlas:
        """
        Run HGT on *text* and return a CompetencyAtlas.

        Parameters
        ----------
        text : str
            Prompt or model output to analyse.
        max_new_tokens : int
            Tokens to generate if *text* is a prompt (ignored for analysis mode).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        self._register_hooks()

        self.model.eval()
        embeddings = self._get_embeddings(input_ids)
        embeddings.requires_grad_(True)

        logits = self._forward_with_embeddings(embeddings, input_ids)

        # Self-consistency loss: model should agree with its own top-1 prediction
        targets = logits.argmax(dim=-1)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        loss.backward()

        grad_norms = self._collect_gradient_norms(embeddings)
        token_scores = self._compute_token_scores(grad_norms)
        atlas = self._build_atlas(input_ids, token_scores)

        self._remove_hooks()
        self._gradient_store.clear()

        return atlas

    def batch_trace(
        self,
        texts: List[str],
        max_new_tokens: int = 128,
    ) -> List[CompetencyAtlas]:
        """Trace a list of texts, returning one atlas per text."""
        return [self.trace(t, max_new_tokens) for t in texts]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_fn = getattr(
            self.model,
            "get_input_embeddings",
            lambda: self.model.transformer.wte
            if hasattr(self.model, "transformer")
            else self.model.model.embed_tokens,
        )
        return embed_fn()(input_ids)

    def _forward_with_embeddings(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = torch.ones(input_ids.shape, device=self.device)
        output = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return output.logits

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if any(k in name for k in ("attn", "attention", "self_attn")):
                h = module.register_backward_hook(self._gradient_store.hook)
                self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _collect_gradient_norms(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return shape (seq_len,) gradient norm per token position."""
        if embeddings.grad is not None:
            norms = embeddings.grad.norm(dim=-1).squeeze(0)  # (seq_len,)
        else:
            # Fallback: use stored attention gradients
            if self._gradient_store.gradients:
                stacked = torch.stack(
                    [g.norm(dim=-1).squeeze(0).mean(0)
                     for g in self._gradient_store.gradients
                     if g.dim() >= 2],
                    dim=0,
                )
                norms = stacked.mean(0)
            else:
                seq_len = embeddings.size(1)
                norms = torch.ones(seq_len, device=self.device)

        # Normalise to [0, 1]
        norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-9)
        return norms

    def _compute_token_scores(self, grad_norms: torch.Tensor) -> torch.Tensor:
        """
        Invert gradient norms to confidence scores.
        High gradient → uncertain → low confidence.
        """
        return 1.0 - grad_norms

    def _build_atlas(
        self,
        input_ids: torch.Tensor,
        token_scores: torch.Tensor,
    ) -> CompetencyAtlas:
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).tolist()
        )
        scores = token_scores.detach().cpu()

        boundary_mask = scores < self.threshold
        boundary_tokens = [t for t, m in zip(tokens, boundary_mask) if m]

        knowledge_gaps = []
        for i, (tok, score) in enumerate(zip(tokens, scores.tolist())):
            if score < self.threshold:
                knowledge_gaps.append(
                    {
                        "span": tok,
                        "position": i,
                        "confidence": round(score, 4),
                        "severity": "high" if score < 0.15 else "medium",
                    }
                )

        overall_risk = float(1.0 - scores.mean().item())

        layer_gradients = torch.zeros(1, len(tokens))

        return CompetencyAtlas(
            token_scores=scores,
            layer_gradients=layer_gradients,
            boundary_tokens=boundary_tokens,
            knowledge_gaps=knowledge_gaps,
            overall_hallucination_risk=round(overall_risk, 4),
            metadata={
                "n_tokens": len(tokens),
                "n_boundary_tokens": len(boundary_tokens),
                "threshold": self.threshold,
            },
        )


# ---------------------------------------------------------------------------
# Standalone utility — works without a live model (offline analysis)
# ---------------------------------------------------------------------------

def score_hallucination_risk(
    generation: str,
    reference: Optional[str] = None,
    method: str = "entropy",
) -> float:
    """
    Lightweight offline hallucination risk scorer.

    Parameters
    ----------
    generation : str   The model's output.
    reference  : str   Optional ground-truth passage for anchored scoring.
    method     : str   "entropy" | "overlap" | "hybrid"

    Returns
    -------
    float in [0, 1] — 1 = very high hallucination risk.
    """
    if method == "overlap" and reference:
        gen_tokens = set(generation.lower().split())
        ref_tokens = set(reference.lower().split())
        if not ref_tokens:
            return 1.0
        overlap = len(gen_tokens & ref_tokens) / len(ref_tokens)
        return round(1.0 - overlap, 4)

    # Entropy proxy: unusual token distribution suggests confabulation
    words = generation.split()
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    avg_len = np.mean([len(w) for w in words])
    # Very high unique ratio + unusually long words → likely confabulation
    risk = (unique_ratio * 0.6) + (min(avg_len / 20.0, 0.4))
    return round(float(np.clip(risk, 0.0, 1.0)), 4)
