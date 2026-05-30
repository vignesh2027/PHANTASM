"""
Confabulation Mining Network (CMN)
====================================
PHANTASM Pillar II — Turns confabulation into a hypothesis-generation engine.

Core insight: When an LLM "confabulates" (creatively fills knowledge gaps),
it is not producing random noise. It is *recombining real, learned concepts*
in novel ways.  CMN harvests these recombinations as structured hypotheses.

Scientific principle: LLMs learn a dense semantic manifold. Confabulations
are trajectories *through* this manifold that the training data never
explicitly charted — i.e., potentially undiscovered connections.

Advantage gained from "disadvantage":
  Confabulation → Creative Hypothesis Mining → Scientific Discovery,
  Drug Interaction Prediction, Novel Material Design, Story Generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Hypothesis:
    """A structured hypothesis mined from a confabulated output."""
    text: str
    source_concepts: List[str]
    novelty_score: float        # 0 = known fact  →  1 = genuinely novel
    plausibility_score: float   # 0 = implausible →  1 = highly plausible
    domain: str
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Hypothesis(novelty={self.novelty_score:.2f}, "
            f"plausibility={self.plausibility_score:.2f}, "
            f"domain='{self.domain}')\n  {self.text[:120]}"
        )


class ConceptExtractor(nn.Module):
    """
    Lightweight transformer encoder that extracts concept vectors from text.
    Used to detect WHICH concepts the model combined in a confabulation.
    """

    def __init__(self, vocab_size: int = 30522, hidden_dim: int = 256, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.concept_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns per-token concept vectors of shape (B, seq_len, hidden_dim)."""
        x = self.embedding(input_ids)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.concept_proj(x)


class NoveltyScorer(nn.Module):
    """
    Scores how novel a confabulation is relative to a factual reference corpus.
    Uses contrastive learning: confabulated ↔ factual pairs.
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, confab_vec: torch.Tensor, fact_vec: torch.Tensor) -> torch.Tensor:
        """Returns novelty score in [0, 1]."""
        combined = torch.cat([confab_vec, fact_vec], dim=-1)
        return self.scorer(combined).squeeze(-1)


class PlausibilityScorer(nn.Module):
    """
    Scores the semantic plausibility of a confabulation using internal
    consistency checks — does it form a coherent logical chain?
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, concept_vectors: torch.Tensor) -> torch.Tensor:
        """Returns plausibility score in [0, 1] for each item in batch."""
        attn_out, _ = self.attention(concept_vectors, concept_vectors, concept_vectors)
        pooled = attn_out.mean(dim=1)
        return self.scorer(pooled).squeeze(-1)


class ConfabulationMiningNetwork(nn.Module):
    """
    CMN: Full confabulation mining pipeline.

    Architecture
    ------------
    Input text  →  ConceptExtractor  →  concept vectors
                                          ↓            ↓
                                   NoveltyScorer  PlausibilityScorer
                                          ↓            ↓
                                     novelty       plausibility
                                          ↓            ↓
                                     Hypothesis object

    Training
    --------
    Train with (confabulation, factual_reference) pairs using the
    ContrastiveMiningLoss below. Factual references can be Wikipedia
    passages or domain corpora. No labels needed — unsupervised.

    Parameters
    ----------
    vocab_size   : int   Vocabulary size (default: GPT-2 vocab).
    hidden_dim   : int   Internal representation dimension.
    novelty_threshold  : float   Min novelty for a hypothesis to be returned.
    plausibility_threshold : float   Min plausibility for a hypothesis.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 256,
        novelty_threshold: float = 0.45,
        plausibility_threshold: float = 0.50,
    ) -> None:
        super().__init__()
        self.concept_extractor = ConceptExtractor(vocab_size, hidden_dim)
        self.novelty_scorer = NoveltyScorer(hidden_dim)
        self.plausibility_scorer = PlausibilityScorer(hidden_dim)
        self.novelty_threshold = novelty_threshold
        self.plausibility_threshold = plausibility_threshold

    def forward(
        self,
        confab_ids: torch.Tensor,
        fact_ids: torch.Tensor,
        confab_mask: Optional[torch.Tensor] = None,
        fact_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        confab_ids : (B, seq_len_c)  Token IDs of confabulated text.
        fact_ids   : (B, seq_len_f)  Token IDs of factual reference text.

        Returns
        -------
        novelty_scores      : (B,)
        plausibility_scores : (B,)
        """
        confab_vecs = self.concept_extractor(confab_ids, confab_mask)
        fact_vecs = self.concept_extractor(fact_ids, fact_mask)

        confab_pooled = confab_vecs.mean(dim=1)
        fact_pooled = fact_vecs.mean(dim=1)

        novelty = self.novelty_scorer(confab_pooled, fact_pooled)
        plausibility = self.plausibility_scorer(confab_vecs)

        return novelty, plausibility

    @torch.no_grad()
    def mine(
        self,
        confab_ids: torch.Tensor,
        fact_ids: torch.Tensor,
        texts: List[str],
        domain: str = "general",
        confab_mask: Optional[torch.Tensor] = None,
        fact_mask: Optional[torch.Tensor] = None,
    ) -> List[Hypothesis]:
        """
        Mine hypotheses from confabulated outputs.

        Parameters
        ----------
        confab_ids : (B, seq_len_c)
        fact_ids   : (B, seq_len_f)
        texts      : list of raw confabulated strings (length B)
        domain     : application domain label

        Returns
        -------
        List of Hypothesis objects that pass novelty + plausibility thresholds.
        """
        self.eval()
        novelty_scores, plausibility_scores = self(
            confab_ids, fact_ids, confab_mask, fact_mask
        )

        hypotheses: List[Hypothesis] = []
        for i, (text, nov, pla) in enumerate(
            zip(texts, novelty_scores.tolist(), plausibility_scores.tolist())
        ):
            if nov >= self.novelty_threshold and pla >= self.plausibility_threshold:
                source_concepts = self._extract_concepts(text)
                hypotheses.append(
                    Hypothesis(
                        text=text,
                        source_concepts=source_concepts,
                        novelty_score=round(nov, 4),
                        plausibility_score=round(pla, 4),
                        domain=domain,
                        metadata={"batch_index": i},
                    )
                )
        return hypotheses

    def _extract_concepts(self, text: str) -> List[str]:
        """Heuristic concept extraction (replace with NER in production)."""
        words = text.split()
        # Return words > 4 chars as candidate concepts
        return [w.strip(".,;:!?") for w in words if len(w) > 4][:10]


class ContrastiveMiningLoss(nn.Module):
    """
    Training loss for CMN.

    Pushes confabulation representations AWAY from their factual
    counterparts (novelty) while keeping internal coherence (plausibility).

    L_total = α * L_contrastive + β * L_coherence
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.6, beta: float = 0.4) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        confab_vecs: torch.Tensor,
        fact_vecs: torch.Tensor,
        novelty_scores: torch.Tensor,
        plausibility_scores: torch.Tensor,
    ) -> torch.Tensor:
        # Contrastive: confab should be novel (far from fact baseline)
        similarity = F.cosine_similarity(confab_vecs, fact_vecs, dim=-1)
        l_contrastive = -torch.log(
            torch.exp((1.0 - similarity) / self.temperature) /
            (torch.exp((1.0 - similarity) / self.temperature) + 1e-9 + 1.0)
        ).mean()

        # Coherence: plausibility should be high (internal consistency)
        l_coherence = F.binary_cross_entropy(
            plausibility_scores,
            torch.ones_like(plausibility_scores),
        )

        return self.alpha * l_contrastive + self.beta * l_coherence
