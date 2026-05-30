"""Tests for Uncertainty Crystallization (UC)."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phantasm.core.uc import (
    UncertaintyCrystallizer,
    TemperatureScaler,
    ConformalPredictor,
    CrystalizedUncertainty,
)


def test_temperature_scaler_forward():
    scaler = TemperatureScaler(init_temperature=2.0)
    logits = torch.randn(4, 10)
    scaled = scaler(logits)
    assert scaled.shape == logits.shape


def test_temperature_scaler_calibrate():
    scaler = TemperatureScaler()
    logits = torch.randn(20, 5)
    labels = torch.randint(0, 5, (20,))
    nll = scaler.calibrate(logits, labels, max_iter=5)
    assert isinstance(nll, float)


def test_conformal_predictor():
    pred = ConformalPredictor(coverage=0.90)
    cal_probs = torch.rand(100).numpy()
    cal_labels = (torch.rand(100) > 0.5).numpy().astype(int)
    pred.calibrate(cal_probs, cal_labels)
    lo, hi = pred.predict_interval(0.7)
    assert 0.0 <= lo <= hi <= 1.0


def test_conformal_predictor_uncalibrated():
    pred = ConformalPredictor()
    lo, hi = pred.predict_interval(0.5)
    assert 0.0 <= lo <= hi <= 1.0


@pytest.fixture(scope="module")
def small_uc():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return UncertaintyCrystallizer(model, mc_samples=3, device="cpu"), tokenizer


def test_uc_crystallize_returns_report(small_uc):
    uc, tokenizer = small_uc
    enc = tokenizer("The sun rises in the east.", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    assert isinstance(result, CrystalizedUncertainty)
    assert result.reliability_tier in ("crystal", "solid", "fluid", "vapor")
    assert 0.0 <= result.raw_confidence <= 1.0
    assert 0.0 <= result.calibrated_confidence <= 1.0
    lo, hi = result.conformal_interval
    assert lo <= hi


def test_uc_tiers_cover_all():
    from phantasm.core.uc import UncertaintyCrystallizer as UC
    tiers = set()
    for conf, epi in [(0.9, 0.01), (0.7, 0.1), (0.5, 0.2), (0.2, 0.5)]:
        tier, _ = UC._classify(conf, epi)
        tiers.add(tier)
    assert tiers == {"crystal", "solid", "fluid", "vapor"}
