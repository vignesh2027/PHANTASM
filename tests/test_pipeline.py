"""Integration test for PHANTASMPipeline."""

import pytest
from phantasm import PHANTASMPipeline
from phantasm.core.pipeline import PHANTASMReport


@pytest.fixture(scope="module")
def pipeline():
    return PHANTASMPipeline.from_pretrained("gpt2", device="cpu")


def test_pipeline_analyze_returns_report(pipeline):
    report = pipeline.analyze("The moon is made of green cheese.")
    assert isinstance(report, PHANTASMReport)
    assert isinstance(report.synthesis_summary, str)
    assert isinstance(report.actionable_insights, list)
    assert len(report.actionable_insights) > 0


def test_pipeline_to_dict(pipeline):
    report = pipeline.analyze("Python is a programming language.")
    d = report.to_dict()
    assert "hallucination_risk" in d
    assert "uncertainty" in d
    assert "insights" in d


def test_pipeline_with_reference(pipeline):
    report = pipeline.analyze(
        "Einstein won the Nobel Prize for relativity.",
        reference_text="Einstein won the Nobel Prize for the photoelectric effect.",
    )
    assert isinstance(report, PHANTASMReport)


def test_pipeline_domain_label(pipeline):
    report = pipeline.analyze("Aspirin inhibits COX-1 and COX-2 enzymes.", domain="medicine")
    assert report.metadata["domain"] == "medicine"
