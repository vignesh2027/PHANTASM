"""Tests for Hallucination Gradient Tracing (HGT)."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phantasm.core.hgt import HallucinationGradientTracer, score_hallucination_risk, CompetencyAtlas


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return model, tokenizer


def test_score_hallucination_risk_entropy():
    score = score_hallucination_risk("The sky is blue and beautiful today.")
    assert 0.0 <= score <= 1.0


def test_score_hallucination_risk_overlap():
    gen = "Napoleon built the Eiffel Tower in 1492"
    ref = "The Eiffel Tower was built by Gustave Eiffel in 1889"
    score = score_hallucination_risk(gen, reference=ref, method="overlap")
    assert 0.0 <= score <= 1.0


def test_hgt_trace_returns_atlas(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    tracer = HallucinationGradientTracer(model, tokenizer, device="cpu")
    atlas = tracer.trace("Albert Einstein won the Nobel Prize.")
    assert isinstance(atlas, CompetencyAtlas)
    assert isinstance(atlas.overall_hallucination_risk, float)
    assert 0.0 <= atlas.overall_hallucination_risk <= 1.0
    assert isinstance(atlas.boundary_tokens, list)
    assert isinstance(atlas.knowledge_gaps, list)


def test_hgt_batch_trace(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    tracer = HallucinationGradientTracer(model, tokenizer, device="cpu")
    atlases = tracer.batch_trace([
        "The Earth orbits the Sun.",
        "Napoleon was born in France.",
    ])
    assert len(atlases) == 2
    for a in atlases:
        assert isinstance(a, CompetencyAtlas)


def test_hgt_threshold_effect(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    tracer_strict = HallucinationGradientTracer(model, tokenizer, threshold=0.1, device="cpu")
    tracer_loose = HallucinationGradientTracer(model, tokenizer, threshold=0.9, device="cpu")
    text = "Water freezes at 100 degrees Celsius."
    atlas_strict = tracer_strict.trace(text)
    atlas_loose = tracer_loose.trace(text)
    assert len(atlas_loose.boundary_tokens) >= len(atlas_strict.boundary_tokens)
