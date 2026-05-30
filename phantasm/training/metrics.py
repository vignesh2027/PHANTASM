"""
PHANTASM evaluation metrics for all three pillars.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class PHANTASMMetrics:
    """
    Computes and aggregates metrics for HGT, CMN, and UC.

    Metrics
    -------
    HGT : AUROC, F1, Precision, Recall for hallucination detection.
    CMN : Novelty@K, Plausibility@K, Hypothesis Coverage.
    UC  : ECE (Expected Calibration Error), MCE, Reliability curve AUC.
    """

    @staticmethod
    def hgt_metrics(
        y_true: List[float],
        y_pred: List[float],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Binary classification metrics for HGT hallucination detection."""
        tp = fp = tn = fn = 0
        for true, pred in zip(y_true, y_pred):
            p = 1 if pred >= threshold else 0
            t = 1 if true >= threshold else 0
            if p == 1 and t == 1:
                tp += 1
            elif p == 1 and t == 0:
                fp += 1
            elif p == 0 and t == 0:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-9)

        auroc = PHANTASMMetrics._auroc(y_true, y_pred)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "auroc": round(auroc, 4),
        }

    @staticmethod
    def cmn_metrics(
        hypotheses_per_query: List[List[float]],
        k: int = 5,
    ) -> Dict[str, float]:
        """
        CMN metrics: mean novelty and plausibility across top-K hypotheses.

        Parameters
        ----------
        hypotheses_per_query : list of lists, each inner list = [novelty * plausibility scores]
        k                    : top-K to consider
        """
        novelty_at_k: List[float] = []
        coverage: List[float] = []
        for hyp_scores in hypotheses_per_query:
            top_k = sorted(hyp_scores, reverse=True)[:k]
            novelty_at_k.append(np.mean(top_k) if top_k else 0.0)
            coverage.append(min(len(hyp_scores) / k, 1.0))

        return {
            f"novelty@{k}": round(float(np.mean(novelty_at_k)), 4),
            f"coverage@{k}": round(float(np.mean(coverage)), 4),
        }

    @staticmethod
    def uc_metrics(
        confidences: List[float],
        correct: List[int],
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Calibration metrics: ECE (Expected Calibration Error) and MCE.

        Parameters
        ----------
        confidences : model's max-class probability for each example.
        correct     : 1 if prediction was correct, 0 otherwise.
        n_bins      : number of confidence bins.
        """
        confidences = np.array(confidences)
        correct = np.array(correct, dtype=float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        n = len(confidences)

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_conf = confidences[mask].mean()
            bin_acc = correct[mask].mean()
            gap = abs(bin_conf - bin_acc)
            ece += (mask.sum() / n) * gap
            mce = max(mce, gap)

        return {
            "ece": round(float(ece), 4),
            "mce": round(float(mce), 4),
            "mean_confidence": round(float(confidences.mean()), 4),
            "mean_accuracy": round(float(correct.mean()), 4),
        }

    @staticmethod
    def _auroc(y_true: List[float], y_score: List[float]) -> float:
        """Trapezoidal AUROC implementation (no sklearn dependency)."""
        pairs = sorted(zip(y_score, y_true), reverse=True)
        pos = sum(y_true)
        neg = len(y_true) - pos
        if pos == 0 or neg == 0:
            return 0.5

        tp = fp = 0
        auc = 0.0
        prev_fp = 0

        for _, label in pairs:
            if label >= 0.5:
                tp += 1
            else:
                fp += 1
                auc += tp  # area under step

        auc /= pos * neg
        return float(np.clip(auc, 0.0, 1.0))

    @staticmethod
    def phantasm_score(
        hgt_f1: float,
        cmn_novelty: float,
        uc_ece: float,
        w_hgt: float = 0.4,
        w_cmn: float = 0.3,
        w_uc: float = 0.3,
    ) -> float:
        """
        Composite PHANTASM score: single number for paper comparisons.

        Higher = better. ECE is inverted (lower ECE = better calibration).
        """
        uc_calibration_score = 1.0 - min(uc_ece, 1.0)
        score = w_hgt * hgt_f1 + w_cmn * cmn_novelty + w_uc * uc_calibration_score
        return round(float(score), 4)
