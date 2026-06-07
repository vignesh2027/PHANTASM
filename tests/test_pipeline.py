"""
Integration tests for PHANTASMPipeline — end-to-end three-pillar analysis.

Coverage:
  - from_pretrained loading
  - analyze() — basic, with reference, domain labels, various text types
  - PHANTASMReport fields, repr, to_dict structure
  - Batch-style sequential calls
  - Edge cases: single word, very long text, code, numbers, multilingual-adjacent
  - Risk-tier consistency
  - Actionable insights structure
  - Metadata propagation
  - Synthesis summary content
"""

import pytest

from phantasm import PHANTASMPipeline
from phantasm.core.pipeline import PHANTASMReport


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline():
    return PHANTASMPipeline.from_pretrained("gpt2", device="cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Basic smoke tests
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_analyze_returns_report(pipeline):
    report = pipeline.analyze("The moon is made of green cheese.")
    assert isinstance(report, PHANTASMReport)


def test_pipeline_analyze_synthesis_is_str(pipeline):
    report = pipeline.analyze("The moon is made of green cheese.")
    assert isinstance(report.synthesis_summary, str)
    assert len(report.synthesis_summary) > 0


def test_pipeline_analyze_insights_list(pipeline):
    report = pipeline.analyze("The moon is made of green cheese.")
    assert isinstance(report.actionable_insights, list)
    assert len(report.actionable_insights) > 0


def test_pipeline_analyze_insights_are_strings(pipeline):
    report = pipeline.analyze("The sun rises in the west.")
    for insight in report.actionable_insights:
        assert isinstance(insight, str)


def test_pipeline_analyze_hallucination_risk_range(pipeline):
    report = pipeline.analyze("Napoleon invented the telephone in 1492.")
    assert 0.0 <= report.competency_atlas.overall_hallucination_risk <= 1.0


def test_pipeline_analyze_tier_valid(pipeline):
    report = pipeline.analyze("Photosynthesis converts light into chemical energy.")
    assert report.uncertainty.reliability_tier in ("crystal", "solid", "fluid", "vapor")


# ─────────────────────────────────────────────────────────────────────────────
# PHANTASMReport.to_dict
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_to_dict_keys(pipeline):
    report = pipeline.analyze("Python is a programming language.")
    d = report.to_dict()
    assert "hallucination_risk" in d
    assert "uncertainty" in d
    assert "insights" in d
    assert "synthesis" in d
    assert "hypotheses" in d


def test_pipeline_to_dict_uncertainty_subkeys(pipeline):
    report = pipeline.analyze("The Earth is flat.")
    d = report.to_dict()
    uc = d["uncertainty"]
    assert "raw_confidence" in uc
    assert "calibrated_confidence" in uc
    assert "epistemic" in uc
    assert "aleatoric" in uc
    assert "tier" in uc
    assert "interval" in uc
    assert "recommendation" in uc


def test_pipeline_to_dict_hypotheses_structure(pipeline):
    report = pipeline.analyze("Compound X cures all known diseases instantly.")
    d = report.to_dict()
    for h in d["hypotheses"]:
        assert "text" in h
        assert "novelty" in h
        assert "plausibility" in h
        assert "domain" in h


def test_pipeline_to_dict_risk_float(pipeline):
    report = pipeline.analyze("Water boils at 50 degrees Celsius at sea level.")
    d = report.to_dict()
    assert isinstance(d["hallucination_risk"], float)
    assert 0.0 <= d["hallucination_risk"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PHANTASMReport repr
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_report_repr(pipeline):
    report = pipeline.analyze("The human genome has 46 chromosomes.")
    r = repr(report)
    assert "PHANTASMReport" in r
    assert "hallucination_risk" in r
    assert "reliability_tier" in r


# ─────────────────────────────────────────────────────────────────────────────
# Reference text
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_with_reference(pipeline):
    report = pipeline.analyze(
        "Einstein won the Nobel Prize for relativity.",
        reference_text="Einstein won the Nobel Prize for the photoelectric effect.",
    )
    assert isinstance(report, PHANTASMReport)


def test_pipeline_with_matching_reference(pipeline):
    text = "Water is composed of hydrogen and oxygen."
    report = pipeline.analyze(text, reference_text=text)
    assert isinstance(report, PHANTASMReport)


def test_pipeline_with_contradicting_reference(pipeline):
    report = pipeline.analyze(
        "The boiling point of water is 50°C.",
        reference_text="Water boils at 100°C at standard atmospheric pressure.",
    )
    assert isinstance(report, PHANTASMReport)


# ─────────────────────────────────────────────────────────────────────────────
# Domain labels
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_domain_medicine(pipeline):
    report = pipeline.analyze("Aspirin inhibits COX-1 and COX-2 enzymes.", domain="medicine")
    assert report.metadata["domain"] == "medicine"


def test_pipeline_domain_finance(pipeline):
    report = pipeline.analyze("The stock price will triple next quarter.", domain="finance")
    assert report.metadata["domain"] == "finance"


def test_pipeline_domain_science(pipeline):
    report = pipeline.analyze("The Higgs boson was discovered at CERN in 2012.", domain="science")
    assert report.metadata["domain"] == "science"


def test_pipeline_domain_legal(pipeline):
    report = pipeline.analyze("The defendant has the right to remain silent.", domain="legal")
    assert report.metadata["domain"] == "legal"


def test_pipeline_domain_education(pipeline):
    report = pipeline.analyze("Photosynthesis occurs in the chloroplast.", domain="education")
    assert report.metadata["domain"] == "education"


def test_pipeline_domain_general(pipeline):
    report = pipeline.analyze("The sky is blue on a clear day.")
    assert report.metadata["domain"] == "general"


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_metadata_text_length(pipeline):
    text = "Short text."
    report = pipeline.analyze(text)
    assert report.metadata["text_length"] == len(text)


def test_pipeline_metadata_domain_key(pipeline):
    report = pipeline.analyze("Some text here.", domain="custom")
    assert "domain" in report.metadata
    assert "text_length" in report.metadata


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_single_word(pipeline):
    report = pipeline.analyze("Hello")
    assert isinstance(report, PHANTASMReport)
    assert 0.0 <= report.competency_atlas.overall_hallucination_risk <= 1.0


def test_pipeline_numbers_only(pipeline):
    report = pipeline.analyze("42 3.14 2.71828 1.618")
    assert isinstance(report, PHANTASMReport)


def test_pipeline_code_snippet(pipeline):
    report = pipeline.analyze(
        "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
        domain="code",
    )
    assert isinstance(report, PHANTASMReport)


def test_pipeline_punctuation_heavy(pipeline):
    report = pipeline.analyze("Yes! No? Maybe... Definitely! Never.")
    assert isinstance(report, PHANTASMReport)


def test_pipeline_long_text(pipeline):
    text = ("The history of artificial intelligence is long and fascinating. " * 10).strip()
    report = pipeline.analyze(text)
    assert isinstance(report, PHANTASMReport)
    assert 0.0 <= report.competency_atlas.overall_hallucination_risk <= 1.0


def test_pipeline_repeated_calls_consistent(pipeline):
    text = "The mitochondria is the powerhouse of the cell."
    report1 = pipeline.analyze(text)
    report2 = pipeline.analyze(text)
    assert abs(
        report1.competency_atlas.overall_hallucination_risk -
        report2.competency_atlas.overall_hallucination_risk
    ) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Sequential multi-domain pipeline calls
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_sequential_domains(pipeline):
    texts = [
        ("Penicillin was discovered by Alexander Fleming.", "medicine"),
        ("Bitcoin uses a proof-of-work consensus mechanism.", "finance"),
        ("The speed of light is approximately 3×10⁸ m/s.", "science"),
        ("The Treaty of Versailles ended World War I.", "history"),
    ]
    for text, domain in texts:
        report = pipeline.analyze(text, domain=domain)
        assert isinstance(report, PHANTASMReport)
        assert report.metadata["domain"] == domain


def test_pipeline_five_different_texts(pipeline):
    texts = [
        "Gravity is described by Einstein's general relativity.",
        "The Amazon River is the longest river in the world.",
        "Python was created by Guido van Rossum in 1991.",
        "The human body has 206 bones.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    for text in texts:
        report = pipeline.analyze(text)
        assert 0.0 <= report.competency_atlas.overall_hallucination_risk <= 1.0
        assert report.uncertainty.reliability_tier in ("crystal", "solid", "fluid", "vapor")


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis content checks
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_synthesis_mentions_risk(pipeline):
    report = pipeline.analyze("The dinosaurs went extinct 65 million years ago.")
    assert "hallucination" in report.synthesis_summary.lower() or "risk" in report.synthesis_summary.lower()


def test_pipeline_synthesis_mentions_tier(pipeline):
    report = pipeline.analyze("Gravity is a fundamental force.")
    tier = report.uncertainty.reliability_tier
    assert tier in report.synthesis_summary.lower() or "reliability" in report.synthesis_summary.lower()


def test_pipeline_insights_contain_recommendation(pipeline):
    report = pipeline.analyze("RNA carries genetic code from DNA to ribosomes.")
    found = any("recommend" in i.lower() or "confidence" in i.lower() or "uncertainty" in i.lower()
                for i in report.actionable_insights)
    assert found
