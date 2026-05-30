"""
test_metrics_losses.py — tests for PHANTASMMetrics and all loss functions.
"""
import pytest
import torch

from phantasm.training.losses import CMNLoss, HGTLoss, PHANTASMLoss
from phantasm.training.metrics import PHANTASMMetrics


# ── HGTLoss ───────────────────────────────────────────────────────────────────

def test_hgt_loss_perfect_prediction():
    loss_fn = HGTLoss()
    scores = torch.tensor([0.99, 0.01, 0.99])
    labels = torch.tensor([1.0, 0.0, 1.0])
    loss = loss_fn(scores, labels)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_hgt_loss_with_gradient_reg():
    loss_fn = HGTLoss(lambda_reg=0.1)
    scores = torch.rand(4)
    labels = torch.randint(0, 2, (4,)).float()
    grad_norms = torch.rand(4)
    loss = loss_fn(scores, labels, grad_norms)
    loss_no_reg = loss_fn(scores, labels)
    assert loss.item() > loss_no_reg.item()


def test_hgt_loss_zero_reg_when_no_norms():
    loss_fn = HGTLoss(lambda_reg=0.5)
    scores = torch.tensor([0.5, 0.5])
    labels = torch.tensor([1.0, 0.0])
    loss_with = loss_fn(scores, labels, torch.zeros(2))
    loss_without = loss_fn(scores, labels, None)
    assert abs(loss_with.item() - loss_without.item()) < 1e-5


# ── CMNLoss ───────────────────────────────────────────────────────────────────

def test_cmn_loss_positive():
    loss_fn = CMNLoss(temperature=0.07)
    confab = torch.randn(4, 64)
    fact = torch.randn(4, 64)
    plausibility = torch.rand(4)
    loss = loss_fn(confab, fact, plausibility)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_cmn_loss_identical_vecs():
    loss_fn = CMNLoss()
    vecs = torch.randn(2, 32)
    loss = loss_fn(vecs, vecs, torch.rand(2))
    assert not torch.isnan(loss)


# ── PHANTASMLoss ──────────────────────────────────────────────────────────────

def test_phantasm_loss_combined():
    loss_fn = PHANTASMLoss()
    batch = {
        "token_scores": torch.rand(4),
        "hgt_labels": torch.randint(0, 2, (4,)).float(),
        "confab_vecs": torch.randn(4, 64),
        "fact_vecs": torch.randn(4, 64),
        "plausibility": torch.rand(4),
    }
    losses = loss_fn(batch)
    assert losses["total"].item() >= 0
    assert not torch.isnan(losses["total"])
    assert "hgt" in losses
    assert "cmn" in losses


def test_phantasm_loss_empty_batch():
    loss_fn = PHANTASMLoss()
    losses = loss_fn({})
    assert losses["total"].item() == pytest.approx(0.0, abs=1e-6)


def test_phantasm_loss_partial_batch():
    loss_fn = PHANTASMLoss()
    batch = {
        "token_scores": torch.rand(2),
        "hgt_labels": torch.tensor([1.0, 0.0]),
    }
    losses = loss_fn(batch)
    assert not torch.isnan(losses["total"])


# ── PHANTASMMetrics.hgt_metrics ───────────────────────────────────────────────

def test_hgt_metrics_perfect():
    y_true = [1.0, 0.0, 1.0, 0.0]
    y_pred = [0.9, 0.1, 0.9, 0.1]
    m = PHANTASMMetrics.hgt_metrics(y_true, y_pred)
    assert m["precision"] == pytest.approx(1.0, abs=1e-3)
    assert m["recall"] == pytest.approx(1.0, abs=1e-3)
    assert m["f1"] == pytest.approx(1.0, abs=1e-3)


def test_hgt_metrics_all_positive():
    y_true = [1.0, 1.0, 1.0]
    y_pred = [0.9, 0.8, 0.7]
    m = PHANTASMMetrics.hgt_metrics(y_true, y_pred)
    assert m["recall"] == pytest.approx(1.0, abs=1e-3)


def test_hgt_metrics_all_wrong():
    y_true = [1.0, 1.0]
    y_pred = [0.1, 0.1]
    m = PHANTASMMetrics.hgt_metrics(y_true, y_pred)
    assert m["precision"] == pytest.approx(0.0, abs=1e-3)


def test_hgt_metrics_auroc_range():
    y_true = [1.0, 0.0, 1.0, 0.0, 1.0]
    y_pred = [0.8, 0.2, 0.7, 0.3, 0.6]
    m = PHANTASMMetrics.hgt_metrics(y_true, y_pred)
    assert 0.0 <= m["auroc"] <= 1.0


# ── PHANTASMMetrics.cmn_metrics ───────────────────────────────────────────────

def test_cmn_metrics_empty():
    m = PHANTASMMetrics.cmn_metrics([])
    assert m["novelty@5"] == pytest.approx(0.0, abs=1e-3)


def test_cmn_metrics_with_hypotheses():
    hyp_scores = [[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    m = PHANTASMMetrics.cmn_metrics(hyp_scores, k=3)
    assert "novelty@3" in m
    assert "coverage@3" in m
    assert 0.0 <= m["novelty@3"] <= 1.0


def test_cmn_metrics_empty_inner_list():
    m = PHANTASMMetrics.cmn_metrics([[]])
    assert m["novelty@5"] == pytest.approx(0.0, abs=1e-3)


# ── PHANTASMMetrics.uc_metrics ────────────────────────────────────────────────

def test_uc_metrics_perfect_calibration():
    confidences = [0.9, 0.9, 0.1, 0.1]
    correct = [1, 1, 0, 0]
    m = PHANTASMMetrics.uc_metrics(confidences, correct)
    assert m["ece"] >= 0.0
    assert m["mce"] >= 0.0
    assert 0.0 <= m["mean_accuracy"] <= 1.0


def test_uc_metrics_returns_all_keys():
    m = PHANTASMMetrics.uc_metrics([0.7, 0.3], [1, 0])
    for k in ("ece", "mce", "mean_confidence", "mean_accuracy"):
        assert k in m


# ── PHANTASMMetrics.phantasm_score ────────────────────────────────────────────

def test_phantasm_score_range():
    score = PHANTASMMetrics.phantasm_score(hgt_f1=0.8, cmn_novelty=0.7, uc_ece=0.1)
    assert 0.0 <= score <= 1.0


def test_phantasm_score_better_f1_higher_score():
    low = PHANTASMMetrics.phantasm_score(hgt_f1=0.3, cmn_novelty=0.5, uc_ece=0.2)
    high = PHANTASMMetrics.phantasm_score(hgt_f1=0.9, cmn_novelty=0.5, uc_ece=0.2)
    assert high > low


def test_phantasm_score_lower_ece_higher_score():
    bad_calib = PHANTASMMetrics.phantasm_score(hgt_f1=0.7, cmn_novelty=0.5, uc_ece=0.5)
    good_calib = PHANTASMMetrics.phantasm_score(hgt_f1=0.7, cmn_novelty=0.5, uc_ece=0.05)
    assert good_calib > bad_calib


# ── _auroc ────────────────────────────────────────────────────────────────────

def test_auroc_all_positive():
    score = PHANTASMMetrics._auroc([1.0, 1.0], [0.9, 0.8])
    assert score == pytest.approx(0.5, abs=0.1)


def test_auroc_all_negative():
    score = PHANTASMMetrics._auroc([0.0, 0.0], [0.9, 0.8])
    assert score == pytest.approx(0.5, abs=0.1)


def test_auroc_perfect():
    score = PHANTASMMetrics._auroc([0.0, 0.0, 1.0, 1.0], [0.1, 0.2, 0.8, 0.9])
    assert score == pytest.approx(1.0, abs=1e-4)
