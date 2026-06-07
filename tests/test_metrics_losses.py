"""
Tests for PHANTASMMetrics and all loss functions.

Coverage:
  - HGTLoss: basic, gradient regularization, zero-reg equivalence, NaN safety
  - CMNLoss: basic, temperature, identical vecs, single-item, NaN safety
  - PHANTASMLoss: combined, partial batch, empty batch, weights, no-nan
  - PHANTASMMetrics.hgt_metrics: perfect, all-positive, all-wrong, AUROC range
  - PHANTASMMetrics.cmn_metrics: empty, with hypotheses, different k values
  - PHANTASMMetrics.uc_metrics: perfect calibration, all keys, bins
  - PHANTASMMetrics.phantasm_score: range, monotonicity, ECE effect
  - PHANTASMMetrics._auroc: edge cases
"""

import pytest
import torch

from phantasm.training.losses import CMNLoss, HGTLoss, PHANTASMLoss
from phantasm.training.metrics import PHANTASMMetrics


# ─────────────────────────────────────────────────────────────────────────────
# HGTLoss
# ─────────────────────────────────────────────────────────────────────────────


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


def test_hgt_loss_no_nan_random():
    loss_fn = HGTLoss()
    for _ in range(5):
        scores = torch.rand(8)
        labels = torch.randint(0, 2, (8,)).float()
        loss = loss_fn(scores, labels)
        assert not torch.isnan(loss)


def test_hgt_loss_nonneg():
    loss_fn = HGTLoss()
    scores = torch.rand(6)
    labels = torch.randint(0, 2, (6,)).float()
    assert loss_fn(scores, labels).item() >= 0


def test_hgt_loss_all_ones_labels():
    loss_fn = HGTLoss()
    scores = torch.rand(4)
    labels = torch.ones(4)
    loss = loss_fn(scores, labels)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


def test_hgt_loss_all_zeros_labels():
    loss_fn = HGTLoss()
    scores = torch.rand(4)
    labels = torch.zeros(4)
    loss = loss_fn(scores, labels)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


def test_hgt_loss_lambda_zero_equals_no_reg():
    loss_fn_0 = HGTLoss(lambda_reg=0.0)
    loss_fn_none = HGTLoss()
    scores = torch.rand(4)
    labels = torch.randint(0, 2, (4,)).float()
    grad_norms = torch.rand(4)
    assert abs(loss_fn_0(scores, labels, grad_norms).item() -
               loss_fn_none(scores, labels).item()) < 1e-5


def test_hgt_loss_single_element():
    loss_fn = HGTLoss()
    loss = loss_fn(torch.tensor([0.7]), torch.tensor([1.0]))
    assert not torch.isnan(loss)


# ─────────────────────────────────────────────────────────────────────────────
# CMNLoss
# ─────────────────────────────────────────────────────────────────────────────


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


def test_cmn_loss_single_item():
    loss_fn = CMNLoss()
    loss = loss_fn(torch.randn(1, 64), torch.randn(1, 64), torch.rand(1))
    assert not torch.isnan(loss)


def test_cmn_loss_temperature_effect():
    confab = torch.randn(4, 64)
    fact = torch.randn(4, 64)
    pla = torch.rand(4)
    loss_low = CMNLoss(temperature=0.01)(confab, fact, pla)
    loss_high = CMNLoss(temperature=1.0)(confab, fact, pla)
    assert loss_low.item() >= 0
    assert loss_high.item() >= 0


def test_cmn_loss_zero_plausibility():
    loss_fn = CMNLoss()
    confab = torch.randn(4, 64)
    fact = torch.randn(4, 64)
    loss = loss_fn(confab, fact, torch.zeros(4))
    assert not torch.isnan(loss)


def test_cmn_loss_large_batch():
    loss_fn = CMNLoss()
    confab = torch.randn(32, 128)
    fact = torch.randn(32, 128)
    loss = loss_fn(confab, fact, torch.rand(32))
    assert loss.item() >= 0


def test_cmn_loss_gradient_flows():
    loss_fn = CMNLoss()
    confab = torch.randn(4, 64, requires_grad=True)
    fact = torch.randn(4, 64)
    loss = loss_fn(confab, fact, torch.rand(4))
    loss.backward()
    assert confab.grad is not None
    assert not torch.isnan(confab.grad).any()


# ─────────────────────────────────────────────────────────────────────────────
# PHANTASMLoss
# ─────────────────────────────────────────────────────────────────────────────


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


def test_phantasm_loss_partial_batch_hgt_only():
    loss_fn = PHANTASMLoss()
    batch = {
        "token_scores": torch.rand(2),
        "hgt_labels": torch.tensor([1.0, 0.0]),
    }
    losses = loss_fn(batch)
    assert not torch.isnan(losses["total"])
    assert losses["total"].item() >= 0


def test_phantasm_loss_partial_batch_cmn_only():
    loss_fn = PHANTASMLoss()
    batch = {
        "confab_vecs": torch.randn(4, 64),
        "fact_vecs": torch.randn(4, 64),
        "plausibility": torch.rand(4),
    }
    losses = loss_fn(batch)
    assert not torch.isnan(losses["total"])


def test_phantasm_loss_total_equals_sum():
    # alpha=1, beta=1, gamma=0 → total = 1*hgt + 1*cmn + 0*uc = hgt + cmn
    loss_fn = PHANTASMLoss(alpha=1.0, beta=1.0, gamma=0.0)
    batch = {
        "token_scores": torch.rand(4),
        "hgt_labels": torch.randint(0, 2, (4,)).float(),
        "confab_vecs": torch.randn(4, 64),
        "fact_vecs": torch.randn(4, 64),
        "plausibility": torch.rand(4),
    }
    losses = loss_fn(batch)
    expected = losses.get("hgt", torch.tensor(0.0)).item() + losses.get("cmn", torch.tensor(0.0)).item()
    assert abs(losses["total"].item() - expected) < 1e-4


def test_phantasm_loss_weights_scale_total():
    # Higher alpha → larger hgt contribution → larger total (with hgt > 0)
    batch = {
        "token_scores": torch.rand(4),
        "hgt_labels": torch.randint(0, 2, (4,)).float(),
    }
    losses_1x = PHANTASMLoss(alpha=0.4)(batch)
    losses_2x = PHANTASMLoss(alpha=0.8)(batch)
    assert losses_2x["total"].item() >= losses_1x["total"].item() - 1e-6


def test_phantasm_loss_no_nan_random():
    loss_fn = PHANTASMLoss()
    for _ in range(5):
        batch = {
            "token_scores": torch.rand(6),
            "hgt_labels": torch.randint(0, 2, (6,)).float(),
            "confab_vecs": torch.randn(6, 64),
            "fact_vecs": torch.randn(6, 64),
            "plausibility": torch.rand(6),
        }
        losses = loss_fn(batch)
        assert not torch.isnan(losses["total"])


# ─────────────────────────────────────────────────────────────────────────────
# PHANTASMMetrics.hgt_metrics
# ─────────────────────────────────────────────────────────────────────────────


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


def test_hgt_metrics_returns_all_keys():
    m = PHANTASMMetrics.hgt_metrics([1.0, 0.0], [0.9, 0.1])
    for key in ("precision", "recall", "f1", "auroc"):
        assert key in m


def test_hgt_metrics_large_batch():
    import random
    random.seed(0)
    y_true = [float(random.randint(0, 1)) for _ in range(100)]
    y_pred = [random.random() for _ in range(100)]
    m = PHANTASMMetrics.hgt_metrics(y_true, y_pred)
    assert 0.0 <= m["f1"] <= 1.0
    assert 0.0 <= m["auroc"] <= 1.0


def test_hgt_metrics_threshold_50():
    y_true = [1.0, 0.0, 1.0, 0.0]
    y_pred = [0.6, 0.4, 0.7, 0.3]
    m = PHANTASMMetrics.hgt_metrics(y_true, y_pred)
    assert m["precision"] >= 0.0
    assert m["recall"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PHANTASMMetrics.cmn_metrics
# ─────────────────────────────────────────────────────────────────────────────


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


def test_cmn_metrics_k1():
    hyp_scores = [[0.9], [0.8], [0.7]]
    m = PHANTASMMetrics.cmn_metrics(hyp_scores, k=1)
    assert "novelty@1" in m
    assert 0.0 <= m["novelty@1"] <= 1.0


def test_cmn_metrics_k10():
    hyp_scores = [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]]
    m = PHANTASMMetrics.cmn_metrics(hyp_scores, k=10)
    assert "novelty@10" in m
    assert 0.0 <= m["novelty@10"] <= 1.0


def test_cmn_metrics_multiple_examples():
    hyp_scores = [
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.85, 0.75, 0.65, 0.55, 0.45],
        [0.7, 0.6, 0.5, 0.4, 0.3],
    ]
    m = PHANTASMMetrics.cmn_metrics(hyp_scores, k=5)
    assert 0.0 <= m["novelty@5"] <= 1.0
    assert 0.0 <= m["coverage@5"] <= 1.0


def test_cmn_metrics_single_high_score():
    hyp_scores = [[1.0]]
    m = PHANTASMMetrics.cmn_metrics(hyp_scores, k=1)
    assert m["novelty@1"] == pytest.approx(1.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# PHANTASMMetrics.uc_metrics
# ─────────────────────────────────────────────────────────────────────────────


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


def test_uc_metrics_mean_confidence_range():
    confidences = [0.6, 0.7, 0.8, 0.9]
    correct = [1, 1, 0, 1]
    m = PHANTASMMetrics.uc_metrics(confidences, correct)
    assert 0.0 <= m["mean_confidence"] <= 1.0


def test_uc_metrics_all_correct():
    confidences = [0.9, 0.85, 0.95]
    correct = [1, 1, 1]
    m = PHANTASMMetrics.uc_metrics(confidences, correct)
    assert m["mean_accuracy"] == pytest.approx(1.0, abs=1e-3)


def test_uc_metrics_all_wrong():
    confidences = [0.9, 0.85, 0.95]
    correct = [0, 0, 0]
    m = PHANTASMMetrics.uc_metrics(confidences, correct)
    assert m["mean_accuracy"] == pytest.approx(0.0, abs=1e-3)


def test_uc_metrics_single_sample():
    m = PHANTASMMetrics.uc_metrics([0.8], [1])
    assert "ece" in m


def test_uc_metrics_large_batch():
    import random
    random.seed(42)
    confs = [random.random() for _ in range(200)]
    correct = [random.randint(0, 1) for _ in range(200)]
    m = PHANTASMMetrics.uc_metrics(confs, correct)
    assert m["ece"] >= 0.0
    assert 0.0 <= m["mean_accuracy"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PHANTASMMetrics.phantasm_score
# ─────────────────────────────────────────────────────────────────────────────


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


def test_phantasm_score_higher_novelty_higher_score():
    low = PHANTASMMetrics.phantasm_score(hgt_f1=0.7, cmn_novelty=0.2, uc_ece=0.1)
    high = PHANTASMMetrics.phantasm_score(hgt_f1=0.7, cmn_novelty=0.9, uc_ece=0.1)
    assert high > low


def test_phantasm_score_perfect():
    score = PHANTASMMetrics.phantasm_score(hgt_f1=1.0, cmn_novelty=1.0, uc_ece=0.0)
    assert score == pytest.approx(1.0, abs=0.05)


def test_phantasm_score_worst():
    score = PHANTASMMetrics.phantasm_score(hgt_f1=0.0, cmn_novelty=0.0, uc_ece=1.0)
    assert score == pytest.approx(0.0, abs=0.05)


def test_phantasm_score_monotone_f1():
    scores = [
        PHANTASMMetrics.phantasm_score(hgt_f1=f, cmn_novelty=0.5, uc_ece=0.1)
        for f in [0.0, 0.25, 0.5, 0.75, 1.0]
    ]
    assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))


def test_phantasm_score_monotone_ece():
    scores = [
        PHANTASMMetrics.phantasm_score(hgt_f1=0.7, cmn_novelty=0.6, uc_ece=e)
        for e in [0.0, 0.1, 0.2, 0.4, 0.8]
    ]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# PHANTASMMetrics._auroc
# ─────────────────────────────────────────────────────────────────────────────


def test_auroc_all_positive():
    score = PHANTASMMetrics._auroc([1.0, 1.0], [0.9, 0.8])
    assert score == pytest.approx(0.5, abs=0.1)


def test_auroc_all_negative():
    score = PHANTASMMetrics._auroc([0.0, 0.0], [0.9, 0.8])
    assert score == pytest.approx(0.5, abs=0.1)


def test_auroc_perfect():
    score = PHANTASMMetrics._auroc([0.0, 0.0, 1.0, 1.0], [0.1, 0.2, 0.8, 0.9])
    assert score == pytest.approx(1.0, abs=1e-4)


def test_auroc_random_better_than_chance():
    y_true = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    y_pred = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    score = PHANTASMMetrics._auroc(y_true, y_pred)
    assert score > 0.5


def test_auroc_range():
    import random
    random.seed(7)
    y_true = [float(random.randint(0, 1)) for _ in range(50)]
    y_pred = [random.random() for _ in range(50)]
    score = PHANTASMMetrics._auroc(y_true, y_pred)
    assert 0.0 <= score <= 1.0
