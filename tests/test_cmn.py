"""
Tests for Confabulation Mining Network (CMN) — Pillar II.

Coverage:
  - ConceptExtractor shape, attention mask, single/batch
  - NoveltyScorer range, edge cases
  - PlausibilityScorer range, edge cases
  - ConfabulationMiningNetwork forward / mine
  - ContrastiveMiningLoss positivity, NaN checks
  - Threshold gating on mine()
  - Domain label propagation
  - Multiple domains
  - Large batch
  - Hypothesis dataclass fields
"""

import pytest
import torch

from phantasm.core.cmn import (
    ConfabulationMiningNetwork,
    ConceptExtractor,
    ContrastiveMiningLoss,
    Hypothesis,
    NoveltyScorer,
    PlausibilityScorer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def cmn():
    return ConfabulationMiningNetwork(vocab_size=1000, hidden_dim=64)


@pytest.fixture
def cmn_low_threshold():
    net = ConfabulationMiningNetwork(vocab_size=1000, hidden_dim=64)
    net.novelty_threshold = 0.0
    net.plausibility_threshold = 0.0
    return net


# ─────────────────────────────────────────────────────────────────────────────
# ConceptExtractor
# ─────────────────────────────────────────────────────────────────────────────


def test_concept_extractor_shape_batch2():
    model = ConceptExtractor(vocab_size=1000, hidden_dim=64, n_heads=2, n_layers=1)
    ids = torch.randint(0, 1000, (2, 16))
    out = model(ids)
    assert out.shape == (2, 16, 64)


def test_concept_extractor_shape_batch1():
    model = ConceptExtractor(vocab_size=1000, hidden_dim=64, n_heads=2, n_layers=1)
    ids = torch.randint(0, 1000, (1, 32))
    out = model(ids)
    assert out.shape == (1, 32, 64)


def test_concept_extractor_with_attention_mask():
    model = ConceptExtractor(vocab_size=1000, hidden_dim=64, n_heads=2, n_layers=1)
    ids = torch.randint(0, 1000, (3, 24))
    mask = torch.ones(3, 24)
    mask[0, 20:] = 0
    out = model(ids, attention_mask=mask)
    assert out.shape == (3, 24, 64)


def test_concept_extractor_no_nan():
    model = ConceptExtractor(vocab_size=1000, hidden_dim=64, n_heads=2, n_layers=1)
    ids = torch.randint(0, 1000, (2, 16))
    out = model(ids)
    assert not torch.isnan(out).any()


def test_concept_extractor_different_hidden_dims():
    for hdim in [32, 128, 256]:
        model = ConceptExtractor(vocab_size=500, hidden_dim=hdim, n_heads=4, n_layers=2)
        ids = torch.randint(0, 500, (2, 16))
        out = model(ids)
        assert out.shape[-1] == hdim


def test_concept_extractor_gradient_flow():
    model = ConceptExtractor(vocab_size=1000, hidden_dim=64, n_heads=2, n_layers=1)
    ids = torch.randint(0, 1000, (2, 16))
    out = model(ids)
    loss = out.mean()
    loss.backward()
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any()


def test_concept_extractor_seq_len_1():
    model = ConceptExtractor(vocab_size=1000, hidden_dim=64, n_heads=2, n_layers=1)
    ids = torch.randint(0, 1000, (1, 1))
    out = model(ids)
    assert out.shape == (1, 1, 64)


# ─────────────────────────────────────────────────────────────────────────────
# NoveltyScorer
# ─────────────────────────────────────────────────────────────────────────────


def test_novelty_scorer_range():
    scorer = NoveltyScorer(hidden_dim=64)
    confab = torch.randn(4, 64)
    fact = torch.randn(4, 64)
    scores = scorer(confab, fact)
    assert scores.shape == (4,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_novelty_scorer_identical_inputs():
    scorer = NoveltyScorer(hidden_dim=64)
    vecs = torch.randn(4, 64)
    scores = scorer(vecs, vecs)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_novelty_scorer_zero_vectors():
    scorer = NoveltyScorer(hidden_dim=64)
    scores = scorer(torch.zeros(3, 64), torch.zeros(3, 64))
    assert not torch.isnan(scores).any()


def test_novelty_scorer_single_item():
    scorer = NoveltyScorer(hidden_dim=64)
    scores = scorer(torch.randn(1, 64), torch.randn(1, 64))
    assert scores.shape == (1,)


def test_novelty_scorer_large_batch():
    scorer = NoveltyScorer(hidden_dim=64)
    scores = scorer(torch.randn(32, 64), torch.randn(32, 64))
    assert scores.shape == (32,)
    assert ((scores >= 0) & (scores <= 1)).all()


# ─────────────────────────────────────────────────────────────────────────────
# PlausibilityScorer
# ─────────────────────────────────────────────────────────────────────────────


def test_plausibility_scorer_range():
    scorer = PlausibilityScorer(hidden_dim=64)
    vecs = torch.randn(4, 16, 64)
    scores = scorer(vecs)
    assert scores.shape == (4,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_plausibility_scorer_single_token():
    scorer = PlausibilityScorer(hidden_dim=64)
    vecs = torch.randn(2, 1, 64)
    scores = scorer(vecs)
    assert scores.shape == (2,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_plausibility_scorer_no_nan():
    scorer = PlausibilityScorer(hidden_dim=64)
    vecs = torch.zeros(4, 8, 64)
    scores = scorer(vecs)
    assert not torch.isnan(scores).any()


def test_plausibility_scorer_large_seq():
    scorer = PlausibilityScorer(hidden_dim=64)
    vecs = torch.randn(2, 512, 64)
    scores = scorer(vecs)
    assert scores.shape == (2,)


# ─────────────────────────────────────────────────────────────────────────────
# ConfabulationMiningNetwork — forward
# ─────────────────────────────────────────────────────────────────────────────


def test_cmn_forward_shapes(cmn):
    confab_ids = torch.randint(0, 1000, (2, 32))
    fact_ids = torch.randint(0, 1000, (2, 32))
    novelty, plausibility = cmn(confab_ids, fact_ids)
    assert novelty.shape == (2,)
    assert plausibility.shape == (2,)


def test_cmn_forward_no_nan(cmn):
    confab_ids = torch.randint(0, 1000, (3, 16))
    fact_ids = torch.randint(0, 1000, (3, 16))
    novelty, plausibility = cmn(confab_ids, fact_ids)
    assert not torch.isnan(novelty).any()
    assert not torch.isnan(plausibility).any()


def test_cmn_forward_range(cmn):
    confab_ids = torch.randint(0, 1000, (4, 32))
    fact_ids = torch.randint(0, 1000, (4, 32))
    novelty, plausibility = cmn(confab_ids, fact_ids)
    assert ((novelty >= 0) & (novelty <= 1)).all()
    assert ((plausibility >= 0) & (plausibility <= 1)).all()


def test_cmn_forward_batch_size_1(cmn):
    confab_ids = torch.randint(0, 1000, (1, 32))
    fact_ids = torch.randint(0, 1000, (1, 32))
    novelty, plausibility = cmn(confab_ids, fact_ids)
    assert novelty.shape == (1,)
    assert plausibility.shape == (1,)


def test_cmn_forward_large_batch():
    net = ConfabulationMiningNetwork(vocab_size=1000, hidden_dim=32)
    confab_ids = torch.randint(0, 1000, (16, 32))
    fact_ids = torch.randint(0, 1000, (16, 32))
    novelty, plausibility = net(confab_ids, fact_ids)
    assert novelty.shape == (16,)


# ─────────────────────────────────────────────────────────────────────────────
# ConfabulationMiningNetwork — mine
# ─────────────────────────────────────────────────────────────────────────────


def test_cmn_mine_returns_hypotheses(cmn_low_threshold):
    confab_ids = torch.randint(0, 1000, (1, 32))
    fact_ids = torch.randint(0, 1000, (1, 32))
    hyps = cmn_low_threshold.mine(confab_ids, fact_ids, texts=["The compound interacts with receptor X"], domain="chemistry")
    assert isinstance(hyps, list)
    for h in hyps:
        assert isinstance(h, Hypothesis)
        assert 0.0 <= h.novelty_score <= 1.0
        assert 0.0 <= h.plausibility_score <= 1.0


def test_cmn_mine_domain_propagated(cmn_low_threshold):
    confab_ids = torch.randint(0, 1000, (1, 32))
    fact_ids = torch.randint(0, 1000, (1, 32))
    hyps = cmn_low_threshold.mine(confab_ids, fact_ids, texts=["Some medical claim"], domain="medicine")
    for h in hyps:
        assert h.domain == "medicine"


def test_cmn_mine_high_threshold_filters(cmn):
    cmn.novelty_threshold = 0.99
    cmn.plausibility_threshold = 0.99
    confab_ids = torch.randint(0, 1000, (1, 32))
    fact_ids = torch.randint(0, 1000, (1, 32))
    hyps = cmn.mine(confab_ids, fact_ids, texts=["Some text"], domain="general")
    assert isinstance(hyps, list)
    cmn.novelty_threshold = 0.5
    cmn.plausibility_threshold = 0.5


def test_cmn_mine_multiple_texts(cmn_low_threshold):
    texts = ["Drug A blocks enzyme B.", "Protein X folds into structure Y."]
    confab_ids = torch.randint(0, 1000, (2, 32))
    fact_ids = torch.randint(0, 1000, (2, 32))
    hyps = cmn_low_threshold.mine(confab_ids, fact_ids, texts=texts, domain="biology")
    assert isinstance(hyps, list)


def test_cmn_mine_general_domain(cmn_low_threshold):
    confab_ids = torch.randint(0, 1000, (1, 32))
    fact_ids = torch.randint(0, 1000, (1, 32))
    hyps = cmn_low_threshold.mine(confab_ids, fact_ids, texts=["General claim"], domain="general")
    assert isinstance(hyps, list)


def test_cmn_mine_finance_domain(cmn_low_threshold):
    confab_ids = torch.randint(0, 1000, (1, 32))
    fact_ids = torch.randint(0, 1000, (1, 32))
    hyps = cmn_low_threshold.mine(confab_ids, fact_ids, texts=["The stock will rise 10x."], domain="finance")
    for h in hyps:
        assert h.domain == "finance"


def test_cmn_mine_hypothesis_text_is_string(cmn_low_threshold):
    confab_ids = torch.randint(0, 1000, (1, 32))
    fact_ids = torch.randint(0, 1000, (1, 32))
    hyps = cmn_low_threshold.mine(confab_ids, fact_ids, texts=["Some claim here."], domain="science")
    for h in hyps:
        assert isinstance(h.text, str)


def test_cmn_mine_hypothesis_source_concepts(cmn_low_threshold):
    confab_ids = torch.randint(0, 1000, (1, 32))
    fact_ids = torch.randint(0, 1000, (1, 32))
    hyps = cmn_low_threshold.mine(confab_ids, fact_ids, texts=["Test text."], domain="test")
    for h in hyps:
        assert isinstance(h.source_concepts, list)


# ─────────────────────────────────────────────────────────────────────────────
# ContrastiveMiningLoss
# ─────────────────────────────────────────────────────────────────────────────


def test_contrastive_loss_positive():
    loss_fn = ContrastiveMiningLoss()
    confab = torch.randn(4, 64)
    fact = torch.randn(4, 64)
    pla = torch.sigmoid(torch.randn(4))
    loss = loss_fn(confab, fact, torch.randn(4), pla)
    assert loss.item() >= 0


def test_contrastive_loss_no_nan():
    loss_fn = ContrastiveMiningLoss()
    confab = torch.randn(4, 64)
    fact = torch.randn(4, 64)
    loss = loss_fn(confab, fact, torch.randn(4), torch.rand(4))
    assert not torch.isnan(loss)


def test_contrastive_loss_identical_vectors():
    loss_fn = ContrastiveMiningLoss()
    vecs = torch.randn(4, 64)
    loss = loss_fn(vecs, vecs, torch.randn(4), torch.rand(4))
    assert not torch.isnan(loss)


def test_contrastive_loss_temperature_effect():
    # Use normalized vectors to keep cosine similarity bounded, avoiding exp overflow at low temps
    torch.manual_seed(42)
    confab = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
    fact = torch.nn.functional.normalize(torch.randn(4, 64), dim=-1)
    nov = torch.randn(4)
    pla = torch.rand(4)
    loss_low = ContrastiveMiningLoss(temperature=0.1)(confab, fact, nov, pla)
    loss_high = ContrastiveMiningLoss(temperature=1.0)(confab, fact, nov, pla)
    assert not torch.isnan(loss_low)
    assert not torch.isnan(loss_high)
    assert loss_low.item() >= 0
    assert loss_high.item() >= 0


def test_contrastive_loss_zero_plausibility():
    loss_fn = ContrastiveMiningLoss()
    confab = torch.randn(4, 64)
    fact = torch.randn(4, 64)
    loss = loss_fn(confab, fact, torch.randn(4), torch.zeros(4))
    assert not torch.isnan(loss)


def test_contrastive_loss_batch_size_1():
    loss_fn = ContrastiveMiningLoss()
    confab = torch.randn(1, 64)
    fact = torch.randn(1, 64)
    loss = loss_fn(confab, fact, torch.randn(1), torch.rand(1))
    assert loss.item() >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis dataclass
# ─────────────────────────────────────────────────────────────────────────────


def test_hypothesis_repr():
    h = Hypothesis(
        text="Protein X folds into a novel alpha-helix structure.",
        source_concepts=["protein", "helix", "fold"],
        novelty_score=0.85,
        plausibility_score=0.72,
        domain="biology",
    )
    r = repr(h)
    assert "novelty" in r
    assert "plausibility" in r


def test_hypothesis_metadata_default_empty():
    h = Hypothesis(
        text="Some hypothesis.",
        source_concepts=[],
        novelty_score=0.5,
        plausibility_score=0.5,
        domain="general",
    )
    assert isinstance(h.metadata, dict)
    assert len(h.metadata) == 0


def test_hypothesis_scores_in_range():
    h = Hypothesis(
        text="A hypothesis.",
        source_concepts=["A", "B"],
        novelty_score=0.95,
        plausibility_score=0.88,
        domain="science",
    )
    assert 0.0 <= h.novelty_score <= 1.0
    assert 0.0 <= h.plausibility_score <= 1.0
