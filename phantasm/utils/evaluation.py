"""
PHANTASMEvaluator — End-to-end evaluation suite.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from phantasm.training.metrics import PHANTASMMetrics


class PHANTASMEvaluator:
    """
    Evaluates a PHANTASMPipeline on a list of (text, reference, label) triples.

    Usage
    -----
    >>> ev = PHANTASMEvaluator(pipeline)
    >>> results = ev.evaluate(samples)
    >>> ev.print_summary(results)
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def evaluate(
        self,
        samples: List[Dict],
        domain: str = "general",
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        samples : list of dicts with keys:
            - "text"      (str, required)
            - "reference" (str, optional)
            - "label"     (float, optional — 1=hallucinated, 0=grounded)
        """
        y_true: List[float] = []
        y_pred: List[float] = []
        confidences: List[float] = []
        correct: List[int] = []
        reports: List[Any] = []

        for s in samples:
            text = s["text"]
            ref = s.get("reference", None)
            label = s.get("label", None)

            report = self.pipeline.analyze(text, reference_text=ref, domain=domain)
            reports.append(report)

            risk = report.competency_atlas.overall_hallucination_risk
            y_pred.append(risk)
            if label is not None:
                y_true.append(float(label))

            cal_conf = report.uncertainty.calibrated_confidence
            confidences.append(cal_conf)
            if label is not None:
                pred_binary = 1 if risk >= 0.5 else 0
                correct.append(1 if pred_binary == int(label >= 0.5) else 0)

        metrics: Dict[str, Any] = {}

        if y_true and len(y_true) == len(y_pred):
            metrics["hgt"] = PHANTASMMetrics.hgt_metrics(y_true, y_pred)

        if confidences and correct and len(confidences) == len(correct):
            metrics["uc"] = PHANTASMMetrics.uc_metrics(confidences, correct)

        hyp_scores_per_query = [
            [h.novelty_score * h.plausibility_score for h in r.mined_hypotheses]
            for r in reports
        ]
        metrics["cmn"] = PHANTASMMetrics.cmn_metrics(hyp_scores_per_query)

        if "hgt" in metrics and "cmn" in metrics and "uc" in metrics:
            metrics["phantasm_score"] = PHANTASMMetrics.phantasm_score(
                hgt_f1=metrics["hgt"].get("f1", 0.0),
                cmn_novelty=metrics["cmn"].get("novelty@5", 0.0),
                uc_ece=metrics["uc"].get("ece", 0.0),
            )

        metrics["n_samples"] = len(samples)
        metrics["reports"] = reports
        return metrics

    @staticmethod
    def print_summary(metrics: Dict[str, Any]) -> None:
        print("\n" + "═" * 55)
        print("  PHANTASM EVALUATION SUMMARY")
        print("═" * 55)
        print(f"  Samples: {metrics.get('n_samples', '?')}")

        if "hgt" in metrics:
            h = metrics["hgt"]
            print(f"\n  HGT (Hallucination Detection)")
            print(f"  ├─ AUROC     : {h.get('auroc', 'N/A')}")
            print(f"  ├─ F1        : {h.get('f1', 'N/A')}")
            print(f"  ├─ Precision : {h.get('precision', 'N/A')}")
            print(f"  └─ Recall    : {h.get('recall', 'N/A')}")

        if "cmn" in metrics:
            c = metrics["cmn"]
            print(f"\n  CMN (Hypothesis Mining)")
            for k, v in c.items():
                print(f"  └─ {k}: {v}")

        if "uc" in metrics:
            u = metrics["uc"]
            print(f"\n  UC (Uncertainty Calibration)")
            print(f"  ├─ ECE  : {u.get('ece', 'N/A')}")
            print(f"  ├─ MCE  : {u.get('mce', 'N/A')}")
            print(f"  └─ Acc  : {u.get('mean_accuracy', 'N/A')}")

        if "phantasm_score" in metrics:
            print(f"\n  ★ PHANTASM Score: {metrics['phantasm_score']:.4f}")
        print("═" * 55 + "\n")
