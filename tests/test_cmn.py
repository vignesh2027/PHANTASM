"""Tests for Confabulation Mining Network (CMN)."""

import pytest
import torch

from phantasm.core.cmn import (
    ConfabulationMiningNetwork,
    ConceptExtractor,
    NoveltyScorer,
    PlausibilityScorer,
    ContrastiveMiningLoss,
    Hypothesis,
)


@pytest.fixture
def cmn():
    return ConfabulationMiningNetwork(vocab_size=1000, hidden_dim=64)


def test_concept_extractor_shape():
    model = ConceptExtractor(vocab_size=1000, hidden_dim=64, n_heads=2, n_layers=1)
    ids = torch.randint(0, 1000, (2, 16))
    out = model(ids)
    assert out.shape == (2, 16, 64)


def test_novelty_scorer():
    scorer = NoveltyScorer(hidden_dim=64)
    confab = torch.randn(4, 64)
    fact = torch.randn(4, 64)
    scores = scorer(confab, fact)
    assert scores.shape == (4,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_plausibility_scorer():
    scorer = PlausibilityScorer(hidden_dim=64)
    vecs = torch.randn(4, 16, 64)
    scores = scorer(vecs)
    assert scores.shape == (4,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_cmn_forward(cmn):
    confab_ids = torch.randint(0, 1000, (2, 32))
    fact_ids = torch.randint(0, 1000, (2, 32))
    novelty, plausibility = cmn(confab_ids, fact_ids)
    assert novelty.shape == (2,)
    assert plausibility.shape == (2,)


def test_cmn_mine_returns_hypotheses(cmn):
    confab_ids = torch.randint(0, 1000, (1, 32))
    fact_ids = torch.randint(0, 1000, (1, 32))
    # Set thresholds low to guarantee at least some hypotheses
    cmn.novelty_threshold = 0.0
    cmn.plausibility_threshold = 0.0
    hyps = cmn.mine(confab_ids, fact_ids, texts=["The compound interacts with receptor X"], domain="chemistry")
    assert isinstance(hyps, list)
    for h in hyps:
        assert isinstance(h, Hypothesis)
        assert 0.0 <= h.novelty_score <= 1.0
        assert 0.0 <= h.plausibility_score <= 1.0


def test_contrastive_loss():
    loss_fn = ContrastiveMiningLoss()
    confab = torch.randn(4, 64)
    fact = torch.randn(4, 64)
    pla = torch.sigmoid(torch.randn(4))
    loss = loss_fn(confab, fact, torch.randn(4), pla)
    assert loss.item() >= 0
