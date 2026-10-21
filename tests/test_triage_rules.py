"""
Tests for the deterministic triage rules.

These use plain pytest asserts, so they run in CI without any ECG data
or trained models.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.triage_rules import ClinicalTriageEngine  # noqa: E402

engine = ClinicalTriageEngine()


def make_result(risk=10.0, hr=70.0, qrs=95.0):
    """Build the minimal input the triage engine reads."""
    return {
        "risk_score_pct": risk,
        "clinical_biomarkers": {
            "heart_rate_bpm": hr,
            "qrs_duration_ms": qrs,
            "qrs_plausible": True,
        },
        "st_by_lead": {},
        "patient": {"age": None, "sex": None},
    }


def rule_ids(result):
    return [finding["rule_id"] for finding in result["findings"]]


def test_normal_ecg_low_risk_is_routine():
    result = engine.evaluate(make_result(risk=10.0))
    assert result["final_tier"] == ClinicalTriageEngine.TIER_ROUTINE
    assert result["override_active"] is False


def test_severe_tachycardia_escalates():
    result = engine.evaluate(make_result(hr=140.0))
    assert result["final_tier"] == ClinicalTriageEngine.TIER_ESCALATION
    assert result["highest_severity"] == "ESCALATE"
    assert "G2_SEVERE_TACHYCARDIA" in rule_ids(result)


def test_severe_bradycardia_escalates():
    result = engine.evaluate(make_result(hr=35.0))
    assert result["final_tier"] == ClinicalTriageEngine.TIER_ESCALATION
    assert "G2_SEVERE_BRADYCARDIA" in rule_ids(result)


def test_wide_qrs_low_risk_goes_to_review_not_routine():
    result = engine.evaluate(make_result(risk=10.0, qrs=150.0))
    assert result["final_tier"] == ClinicalTriageEngine.TIER_REVIEW
    assert result["highest_severity"] == "REVIEW"
    assert "G3_WIDE_QRS" in rule_ids(result)


def test_review_finding_never_lowers_a_high_statistical_risk():
    result = engine.evaluate(make_result(risk=80.0, qrs=150.0))
    assert result["final_tier"] == ClinicalTriageEngine.TIER_ESCALATION