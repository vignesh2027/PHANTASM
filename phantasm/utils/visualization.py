"""
PHANTASM Visualization utilities.
Produces ASCII and Matplotlib visualizations of all three pillars.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


TIER_SYMBOLS = {
    "crystal": "◆",
    "solid": "◇",
    "fluid": "≈",
    "vapor": "~",
}

TIER_COLORS_ANSI = {
    "crystal": "\033[92m",   # green
    "solid": "\033[94m",     # blue
    "fluid": "\033[93m",     # yellow
    "vapor": "\033[91m",     # red
}
RESET = "\033[0m"


class PHANTASMVisualizer:
    """
    Terminal-first, matplotlib-optional visualizer for PHANTASM reports.
    """

    @staticmethod
    def print_report(report: Any) -> None:
        """Pretty-print a PHANTASMReport to the terminal."""
        atlas = report.competency_atlas
        uc = report.uncertainty
        tier = uc.reliability_tier
        color = TIER_COLORS_ANSI.get(tier, "")
        sym = TIER_SYMBOLS.get(tier, "?")

        print("\n" + "═" * 65)
        print("  PHANTASM ANALYSIS REPORT")
        print("═" * 65)

        # HGT section
        print(f"\n  ◈ PILLAR I — Hallucination Gradient Tracing")
        print(f"  ├─ Overall risk   : {atlas.overall_hallucination_risk:.2%}")
        bar_len = int(atlas.overall_hallucination_risk * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  ├─ Risk bar       : [{bar}]")
        print(f"  ├─ Boundary tokens: {len(atlas.boundary_tokens)}")
        if atlas.knowledge_gaps:
            print(f"  └─ Top gap        : '{atlas.knowledge_gaps[0]['span']}' "
                  f"(conf={atlas.knowledge_gaps[0]['confidence']:.3f})")

        # CMN section
        print(f"\n  ◈ PILLAR II — Confabulation Mining Network")
        print(f"  └─ Hypotheses mined: {len(report.mined_hypotheses)}")
        for i, h in enumerate(report.mined_hypotheses[:3]):
            print(f"     [{i+1}] novelty={h.novelty_score:.2f} pla={h.plausibility_score:.2f}"
                  f"  '{h.text[:60]}...'")

        # UC section
        print(f"\n  ◈ PILLAR III — Uncertainty Crystallization")
        print(f"  ├─ Raw confidence     : {uc.raw_confidence:.2%}")
        print(f"  ├─ Calibrated conf.   : {uc.calibrated_confidence:.2%}")
        print(f"  ├─ Epistemic uncert.  : {uc.epistemic_uncertainty:.4f}")
        print(f"  ├─ Aleatoric uncert.  : {uc.aleatoric_uncertainty:.4f}")
        print(f"  ├─ 90% CI             : {uc.conformal_interval}")
        print(f"  ├─ Tier               : {color}{sym} {tier.upper()}{RESET}")
        print(f"  └─ Recommendation     : {uc.action_recommendation}")

        # Synthesis
        print(f"\n  ◈ SYNTHESIS")
        print(f"  {report.synthesis_summary}")
        print(f"\n  ◈ ACTIONABLE INSIGHTS")
        for insight in report.actionable_insights:
            print(f"  • {insight}")

        print("\n" + "═" * 65 + "\n")

    @staticmethod
    def ascii_competency_map(
        tokens: List[str],
        scores: List[float],
        width: int = 60,
    ) -> str:
        """Render a token-level competency map as an ASCII heatmap."""
        lines = ["Competency Map (█ = grounded, ░ = boundary):", ""]
        row = ""
        for tok, score in zip(tokens, scores):
            char = "█" if score >= 0.5 else ("▓" if score >= 0.3 else "░")
            row += char
            if len(row) >= width:
                lines.append(row)
                row = ""
        if row:
            lines.append(row)
        return "\n".join(lines)

    @staticmethod
    def plot_calibration_curve(
        confidences: List[float],
        accuracies: List[float],
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """Reliability diagram (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib not installed — run: pip install matplotlib")
            return

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accs = []
        bin_confs = []
        gaps = []

        conf_arr = np.array(confidences)
        acc_arr = np.array(accuracies)

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf_arr > lo) & (conf_arr <= hi)
            if mask.sum() > 0:
                bin_accs.append(acc_arr[mask].mean())
                bin_confs.append(conf_arr[mask].mean())
                gaps.append(abs(conf_arr[mask].mean() - acc_arr[mask].mean()))
            else:
                bin_accs.append(0.0)
                bin_confs.append((lo + hi) / 2)
                gaps.append(0.0)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1.5)
        ax.bar(
            bin_centers, bin_accs, width=0.09, alpha=0.7,
            label="Calibrated (PHANTASM UC)", color="#2196F3", edgecolor="white"
        )
        ax.bar(
            bin_centers, gaps, width=0.09, alpha=0.3,
            bottom=bin_accs, label="Calibration gap", color="#F44336"
        )
        ax.set_xlabel("Confidence", fontsize=13)
        ax.set_ylabel("Accuracy", fontsize=13)
        ax.set_title("PHANTASM UC — Reliability Diagram", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()

    @staticmethod
    def plot_hypothesis_scatter(
        hypotheses: List[Any],
        save_path: Optional[str] = None,
    ) -> None:
        """Scatter plot of CMN hypotheses in novelty × plausibility space."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — run: pip install matplotlib")
            return

        if not hypotheses:
            print("No hypotheses to plot.")
            return

        novelties = [h.novelty_score for h in hypotheses]
        plausibilities = [h.plausibility_score for h in hypotheses]
        labels = [h.text[:30] for h in hypotheses]

        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = ax.scatter(
            novelties, plausibilities,
            c=[n * p for n, p in zip(novelties, plausibilities)],
            cmap="plasma", s=120, alpha=0.8, edgecolors="black", linewidths=0.5
        )
        ax.axvline(x=0.45, color="gray", linestyle="--", lw=1, label="Novelty threshold")
        ax.axhline(y=0.50, color="gray", linestyle=":", lw=1, label="Plausibility threshold")
        plt.colorbar(scatter, label="Combined score (novelty × plausibility)")
        ax.set_xlabel("Novelty Score", fontsize=13)
        ax.set_ylabel("Plausibility Score", fontsize=13)
        ax.set_title("PHANTASM CMN — Hypothesis Space", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        for i, (x, y, lbl) in enumerate(zip(novelties, plausibilities, labels)):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, alpha=0.8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
