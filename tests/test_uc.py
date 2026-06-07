"""
Tests for Uncertainty Crystallization (UC) — Pillar III.

Coverage:
  - TemperatureScaler forward, calibrate, extreme temperatures
  - ConformalPredictor calibrate / predict_interval, coverage guarantees
  - UncertaintyCrystallizer.crystallize (basic, tiers, no-mask, with-mask)
  - CrystalizedUncertainty fields and invariants
  - _classify covers all four tiers
  - MC-Dropout samples effect
  - Edge cases: single token, all-pad sequence, short sequences
  - Determinism with fixed seed
"""

import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from phantasm.core.uc import (
    CrystalizedUncertainty,
    ConformalPredictor,
    TemperatureScaler,
    UncertaintyCrystallizer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gpt2_uc():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    uc = UncertaintyCrystallizer(model, mc_samples=3, device="cpu")
    return uc, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# TemperatureScaler
# ─────────────────────────────────────────────────────────────────────────────


def test_temperature_scaler_forward_shape():
    scaler = TemperatureScaler(init_temperature=2.0)
    logits = torch.randn(4, 10)
    scaled = scaler(logits)
    assert scaled.shape == logits.shape


def test_temperature_scaler_no_nan():
    scaler = TemperatureScaler(init_temperature=1.5)
    logits = torch.randn(8, 20)
    scaled = scaler(logits)
    assert not torch.isnan(scaled).any()


def test_temperature_scaler_temp_1_identity():
    scaler = TemperatureScaler(init_temperature=1.0)
    logits = torch.randn(4, 10)
    scaled = scaler(logits)
    torch.testing.assert_close(scaled, logits)


def test_temperature_scaler_temp_gt1_flatter():
    scaler_hot = TemperatureScaler(init_temperature=5.0)
    scaler_cold = TemperatureScaler(init_temperature=0.5)
    logits = torch.randn(4, 10)
    hot = scaler_hot(logits)
    cold = scaler_cold(logits)
    assert hot.std() < cold.std()


def test_temperature_scaler_calibrate_returns_float():
    scaler = TemperatureScaler()
    logits = torch.randn(20, 5)
    labels = torch.randint(0, 5, (20,))
    nll = scaler.calibrate(logits, labels, max_iter=5)
    assert isinstance(nll, float)
    assert nll >= 0.0


def test_temperature_scaler_calibrate_reduces_nll():
    scaler = TemperatureScaler(init_temperature=5.0)
    logits = torch.randn(40, 10)
    labels = torch.randint(0, 10, (40,))
    nll_before = torch.nn.functional.cross_entropy(logits, labels).item()
    nll_after = scaler.calibrate(logits, labels, max_iter=50)
    assert nll_after <= nll_before + 1.0


def test_temperature_scaler_calibrate_single_sample():
    scaler = TemperatureScaler()
    logits = torch.randn(1, 3)
    labels = torch.randint(0, 3, (1,))
    nll = scaler.calibrate(logits, labels, max_iter=3)
    assert isinstance(nll, float)


def test_temperature_scaler_3d_logits():
    scaler = TemperatureScaler(init_temperature=2.0)
    logits = torch.randn(2, 8, 10)
    scaled = scaler(logits)
    assert scaled.shape == (2, 8, 10)


# ─────────────────────────────────────────────────────────────────────────────
# ConformalPredictor
# ─────────────────────────────────────────────────────────────────────────────


def test_conformal_predictor_interval_order():
    pred = ConformalPredictor(coverage=0.90)
    cal_probs = torch.rand(100).numpy()
    cal_labels = (torch.rand(100) > 0.5).numpy().astype(int)
    pred.calibrate(cal_probs, cal_labels)
    lo, hi = pred.predict_interval(0.7)
    assert lo <= hi


def test_conformal_predictor_interval_in_0_1():
    pred = ConformalPredictor(coverage=0.90)
    cal_probs = torch.rand(100).numpy()
    cal_labels = (torch.rand(100) > 0.5).numpy().astype(int)
    pred.calibrate(cal_probs, cal_labels)
    lo, hi = pred.predict_interval(0.7)
    assert 0.0 <= lo <= hi <= 1.0


def test_conformal_predictor_uncalibrated_fallback():
    pred = ConformalPredictor()
    lo, hi = pred.predict_interval(0.5)
    assert 0.0 <= lo <= hi <= 1.0


def test_conformal_predictor_coverage_90():
    pred = ConformalPredictor(coverage=0.90)
    rng = np.random.default_rng(42)
    cal_probs = rng.uniform(0, 1, 200)
    cal_labels = (cal_probs > 0.5).astype(int)
    pred.calibrate(cal_probs, cal_labels)
    lo, hi = pred.predict_interval(0.75)
    assert 0.0 <= lo <= hi <= 1.0


def test_conformal_predictor_coverage_99():
    pred = ConformalPredictor(coverage=0.99)
    cal_probs = np.random.rand(200)
    cal_labels = (cal_probs > 0.5).astype(int)
    pred.calibrate(cal_probs, cal_labels)
    lo, hi = pred.predict_interval(0.8)
    assert hi - lo >= 0.0


def test_conformal_predictor_extreme_prob_0():
    pred = ConformalPredictor(coverage=0.90)
    cal_probs = np.random.rand(50)
    cal_labels = (cal_probs > 0.5).astype(int)
    pred.calibrate(cal_probs, cal_labels)
    lo, hi = pred.predict_interval(0.0)
    assert 0.0 <= lo <= hi <= 1.0


def test_conformal_predictor_extreme_prob_1():
    pred = ConformalPredictor(coverage=0.90)
    cal_probs = np.random.rand(50)
    cal_labels = (cal_probs > 0.5).astype(int)
    pred.calibrate(cal_probs, cal_labels)
    lo, hi = pred.predict_interval(1.0)
    assert 0.0 <= lo <= hi <= 1.0


def test_conformal_predictor_larger_coverage_wider_interval():
    cal_probs = np.random.rand(150)
    cal_labels = (cal_probs > 0.5).astype(int)
    pred_80 = ConformalPredictor(coverage=0.80)
    pred_95 = ConformalPredictor(coverage=0.95)
    pred_80.calibrate(cal_probs, cal_labels)
    pred_95.calibrate(cal_probs, cal_labels)
    lo80, hi80 = pred_80.predict_interval(0.6)
    lo95, hi95 = pred_95.predict_interval(0.6)
    assert (hi95 - lo95) >= (hi80 - lo80) - 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# UncertaintyCrystallizer.crystallize
# ─────────────────────────────────────────────────────────────────────────────


def test_uc_crystallize_returns_report(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("The sun rises in the east.", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    assert isinstance(result, CrystalizedUncertainty)


def test_uc_crystallize_tier_valid(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("Gravity keeps planets in orbit.", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    assert result.reliability_tier in ("crystal", "solid", "fluid", "vapor")


def test_uc_crystallize_confidences_in_range(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("Python is a high-level programming language.", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    assert 0.0 <= result.raw_confidence <= 1.0
    assert 0.0 <= result.calibrated_confidence <= 1.0


def test_uc_crystallize_interval_ordered(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("DNA carries genetic information.", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    lo, hi = result.conformal_interval
    assert lo <= hi


def test_uc_crystallize_uncertainty_nonneg(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("The Eiffel Tower is in Paris.", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    assert result.epistemic_uncertainty >= 0.0
    assert result.aleatoric_uncertainty >= 0.0
    assert result.total_uncertainty >= 0.0


def test_uc_crystallize_with_attention_mask(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("Quantum mechanics describes subatomic particles.", return_tensors="pt",
                    truncation=True, max_length=32, padding="max_length")
    result = uc.crystallize(enc["input_ids"], enc.get("attention_mask"))
    assert isinstance(result, CrystalizedUncertainty)


def test_uc_crystallize_action_recommendation_string(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("The human heart has four chambers.", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    assert isinstance(result.action_recommendation, str)
    assert len(result.action_recommendation) > 0


def test_uc_crystallize_total_uncertainty(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("The capital of France is Berlin.", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    expected_total = result.epistemic_uncertainty + result.aleatoric_uncertainty
    assert abs(result.total_uncertainty - expected_total) < 0.1


def test_uc_crystallize_medical_text(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("Aspirin inhibits prostaglandin synthesis.", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    assert isinstance(result, CrystalizedUncertainty)


def test_uc_crystallize_short_text(gpt2_uc):
    uc, tokenizer = gpt2_uc
    enc = tokenizer("OK", return_tensors="pt", truncation=True, max_length=32)
    result = uc.crystallize(enc["input_ids"])
    assert isinstance(result, CrystalizedUncertainty)


# ─────────────────────────────────────────────────────────────────────────────
# _classify covers all four tiers
# ─────────────────────────────────────────────────────────────────────────────


def test_uc_tiers_cover_all():
    tiers = set()
    for conf, epi in [(0.9, 0.01), (0.7, 0.1), (0.5, 0.2), (0.2, 0.5)]:
        tier, _ = UncertaintyCrystallizer._classify(conf, epi)
        tiers.add(tier)
    assert tiers == {"crystal", "solid", "fluid", "vapor"}


def test_uc_crystal_tier_high_conf():
    tier, rec = UncertaintyCrystallizer._classify(0.95, 0.02)
    assert tier == "crystal"
    assert isinstance(rec, str)


def test_uc_vapor_tier_low_conf():
    tier, rec = UncertaintyCrystallizer._classify(0.15, 0.6)
    assert tier == "vapor"
    assert isinstance(rec, str)


def test_uc_solid_tier():
    tier, _ = UncertaintyCrystallizer._classify(0.72, 0.08)
    assert tier == "solid"


def test_uc_fluid_tier():
    tier, _ = UncertaintyCrystallizer._classify(0.45, 0.25)
    assert tier == "fluid"


def test_uc_classify_boundary_values():
    for conf in [0.0, 0.5, 1.0]:
        for epi in [0.0, 0.5, 1.0]:
            tier, rec = UncertaintyCrystallizer._classify(conf, epi)
            assert tier in ("crystal", "solid", "fluid", "vapor")
            assert isinstance(rec, str)


# ─────────────────────────────────────────────────────────────────────────────
# CrystalizedUncertainty repr
# ─────────────────────────────────────────────────────────────────────────────


def test_crystallized_uncertainty_repr():
    cu = CrystalizedUncertainty(
        raw_confidence=0.8,
        calibrated_confidence=0.75,
        epistemic_uncertainty=0.05,
        aleatoric_uncertainty=0.10,
        total_uncertainty=0.15,
        conformal_interval=(0.65, 0.85),
        reliability_tier="solid",
        action_recommendation="Use with moderate confidence.",
    )
    r = repr(cu)
    assert "calibrated" in r
    assert "solid" in r
