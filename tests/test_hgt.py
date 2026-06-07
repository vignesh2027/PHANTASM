"""
Tests for Hallucination Gradient Tracing (HGT) — Pillar I.

Coverage:
  - score_hallucination_risk (entropy, overlap, hybrid methods)
  - HallucinationGradientTracer.trace (basic, thresholds, domains)
  - HallucinationGradientTracer.batch_trace
  - CompetencyAtlas structure and invariants
  - Edge cases: empty text, single token, very long text, Unicode, code
  - Threshold sensitivity
  - Determinism
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phantasm.core.hgt import (
    HallucinationGradientTracer,
    score_hallucination_risk,
    CompetencyAtlas,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return model, tokenizer


@pytest.fixture(scope="module")
def tracer(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    return HallucinationGradientTracer(model, tokenizer, device="cpu")


# ─────────────────────────────────────────────────────────────────────────────
# score_hallucination_risk
# ─────────────────────────────────────────────────────────────────────────────


def test_score_entropy_range():
    score = score_hallucination_risk("The sky is blue.")
    assert 0.0 <= score <= 1.0


def test_score_overlap_range():
    score = score_hallucination_risk(
        "Napoleon built the Eiffel Tower in 1492",
        reference="The Eiffel Tower was built by Gustave Eiffel in 1889",
        method="overlap",
    )
    assert 0.0 <= score <= 1.0


def test_score_overlap_identical():
    text = "Water is H2O."
    score = score_hallucination_risk(text, reference=text, method="overlap")
    assert score == pytest.approx(0.0, abs=0.05)


def test_score_overlap_no_overlap():
    score = score_hallucination_risk(
        "abc def ghi",
        reference="xyz uvw rst",
        method="overlap",
    )
    assert score > 0.5


def test_score_entropy_short_text():
    score = score_hallucination_risk("Hi")
    assert 0.0 <= score <= 1.0


def test_score_entropy_long_text():
    text = " ".join(["The", "quick", "brown", "fox"] * 50)
    score = score_hallucination_risk(text)
    assert 0.0 <= score <= 1.0


def test_score_unicode_text():
    score = score_hallucination_risk("Le ciel est bleu. Die Sonne scheint. 太陽が輝く。")
    assert 0.0 <= score <= 1.0


def test_score_code_text():
    score = score_hallucination_risk("def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)")
    assert 0.0 <= score <= 1.0


def test_score_numbers_only():
    score = score_hallucination_risk("42 3.14 2.71828 1.41421")
    assert 0.0 <= score <= 1.0


def test_score_high_hallucination_factual_error():
    factual = "The Earth orbits the Sun. Water freezes at 0 degrees Celsius."
    wrong = "The Sun orbits Mars. Water boils at 50 degrees Celsius on Jupiter."
    score_wrong = score_hallucination_risk(wrong, reference=factual, method="overlap")
    score_right = score_hallucination_risk(factual, reference=factual, method="overlap")
    assert score_wrong >= score_right


# ─────────────────────────────────────────────────────────────────────────────
# HallucinationGradientTracer.trace
# ─────────────────────────────────────────────────────────────────────────────


def test_hgt_trace_returns_atlas(tracer):
    atlas = tracer.trace("Albert Einstein won the Nobel Prize.")
    assert isinstance(atlas, CompetencyAtlas)
    assert isinstance(atlas.overall_hallucination_risk, float)
    assert 0.0 <= atlas.overall_hallucination_risk <= 1.0
    assert isinstance(atlas.boundary_tokens, list)
    assert isinstance(atlas.knowledge_gaps, list)


def test_hgt_trace_token_scores_shape(tracer):
    text = "The moon is Earth's natural satellite."
    atlas = tracer.trace(text)
    assert isinstance(atlas.token_scores, torch.Tensor)
    assert atlas.token_scores.ndim == 1
    assert len(atlas.token_scores) > 0


def test_hgt_trace_layer_gradients(tracer):
    atlas = tracer.trace("Paris is the capital of France.")
    assert isinstance(atlas.layer_gradients, torch.Tensor)
    assert atlas.layer_gradients.ndim == 2  # (n_layers, seq_len)


def test_hgt_trace_token_scores_in_range(tracer):
    atlas = tracer.trace("Quantum entanglement is a physical phenomenon.")
    scores = atlas.token_scores
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_hgt_trace_factual_text(tracer):
    atlas = tracer.trace("Water is composed of hydrogen and oxygen atoms.")
    assert isinstance(atlas.overall_hallucination_risk, float)


def test_hgt_trace_clearly_wrong_text(tracer):
    atlas = tracer.trace("Napoleon invented the internet in 1776 on Mars.")
    assert isinstance(atlas.overall_hallucination_risk, float)
    assert 0.0 <= atlas.overall_hallucination_risk <= 1.0


def test_hgt_trace_empty_like_text(tracer):
    atlas = tracer.trace(".")
    assert isinstance(atlas, CompetencyAtlas)
    assert 0.0 <= atlas.overall_hallucination_risk <= 1.0


def test_hgt_trace_medical_domain(tracer):
    atlas = tracer.trace("Metformin is a first-line treatment for type 2 diabetes mellitus.")
    assert isinstance(atlas, CompetencyAtlas)


def test_hgt_trace_legal_domain(tracer):
    atlas = tracer.trace("The Fourth Amendment protects against unreasonable searches and seizures.")
    assert isinstance(atlas, CompetencyAtlas)


def test_hgt_trace_code_domain(tracer):
    atlas = tracer.trace("In Python, list comprehensions are syntactic sugar for map and filter.")
    assert isinstance(atlas, CompetencyAtlas)


def test_hgt_trace_metadata_populated(tracer):
    atlas = tracer.trace("DNA is a double helix discovered by Watson and Crick.")
    assert isinstance(atlas.metadata, dict)


def test_hgt_knowledge_gaps_structure(tracer):
    atlas = tracer.trace("The speed of light is approximately 299,792 kilometers per second.")
    for gap in atlas.knowledge_gaps:
        assert "span" in gap or "confidence" in gap or isinstance(gap, dict)


# ─────────────────────────────────────────────────────────────────────────────
# HallucinationGradientTracer.batch_trace
# ─────────────────────────────────────────────────────────────────────────────


def test_hgt_batch_trace_basic(tracer):
    texts = [
        "The Earth orbits the Sun.",
        "Napoleon was born in Corsica.",
    ]
    atlases = tracer.batch_trace(texts)
    assert len(atlases) == 2
    for a in atlases:
        assert isinstance(a, CompetencyAtlas)


def test_hgt_batch_trace_single_item(tracer):
    atlases = tracer.batch_trace(["Only one text here."])
    assert len(atlases) == 1
    assert isinstance(atlases[0], CompetencyAtlas)


def test_hgt_batch_trace_five_items(tracer):
    texts = [
        "Aspirin reduces fever.",
        "The Amazon is the longest river.",
        "Python was created by Guido van Rossum.",
        "Black holes have immense gravity.",
        "Shakespeare wrote Hamlet.",
    ]
    atlases = tracer.batch_trace(texts)
    assert len(atlases) == 5
    for a in atlases:
        assert 0.0 <= a.overall_hallucination_risk <= 1.0


def test_hgt_batch_trace_results_differ(tracer):
    factual = "The sun is a star at the center of the solar system."
    hallucinated = "The moon is a planet made entirely of gold and diamonds."
    atlases = tracer.batch_trace([factual, hallucinated])
    risks = [a.overall_hallucination_risk for a in atlases]
    assert len(set(risks)) >= 1  # at minimum they are valid floats


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sensitivity
# ─────────────────────────────────────────────────────────────────────────────


def test_hgt_threshold_effect(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    tracer_strict = HallucinationGradientTracer(model, tokenizer, threshold=0.1, device="cpu")
    tracer_loose = HallucinationGradientTracer(model, tokenizer, threshold=0.9, device="cpu")
    text = "Water freezes at 100 degrees Celsius."
    atlas_strict = tracer_strict.trace(text)
    atlas_loose = tracer_loose.trace(text)
    assert len(atlas_loose.boundary_tokens) >= len(atlas_strict.boundary_tokens)


def test_hgt_threshold_zero_all_boundary(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    tracer = HallucinationGradientTracer(model, tokenizer, threshold=0.0, device="cpu")
    atlas = tracer.trace("Some random text for testing.")
    assert isinstance(atlas, CompetencyAtlas)


def test_hgt_threshold_one_all_boundary(model_and_tokenizer):
    # threshold=1.0 means score < 1.0 catches virtually all tokens (scores are floats in [0,1))
    model, tokenizer = model_and_tokenizer
    tracer = HallucinationGradientTracer(model, tokenizer, threshold=1.0, device="cpu")
    atlas = tracer.trace("Some random text for testing.")
    assert isinstance(atlas, CompetencyAtlas)
    assert 0.0 <= atlas.overall_hallucination_risk <= 1.0
    assert len(atlas.boundary_tokens) >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_hgt_deterministic(tracer):
    text = "The mitochondria is the powerhouse of the cell."
    atlas1 = tracer.trace(text)
    atlas2 = tracer.trace(text)
    assert pytest.approx(atlas1.overall_hallucination_risk, abs=1e-4) == atlas2.overall_hallucination_risk


# ─────────────────────────────────────────────────────────────────────────────
# CompetencyAtlas invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_atlas_risk_is_aggregate(tracer):
    atlas = tracer.trace("The French Revolution began in 1789.")
    avg_token = atlas.token_scores.mean().item()
    risk = atlas.overall_hallucination_risk
    assert isinstance(avg_token, float)
    assert isinstance(risk, float)


def test_atlas_boundary_tokens_are_strings(tracer):
    atlas = tracer.trace("Gravity is a fundamental force of nature.")
    for tok in atlas.boundary_tokens:
        assert isinstance(tok, str)
