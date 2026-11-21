
"""Verify that quality metadata reaches triage without changing routing."""

import copy

from src.triage.triage_rules import ClinicalTriageEngine


def make_payload(
    *,
    v2_status="not_validated",
    v3_status="not_validated",
    risk=10.0,
):
    return {
        "status": "success",
        "risk_score_pct": risk,
        "patient": {"age": 45, "sex": "MALE"},
        "st_by_lead": {
            "V2": 0.30,
            "V3": 0.32,
        },
        "clinical_biomarkers": {
            "heart_rate_bpm": 75.0,
            "qrs_duration_ms": 90.0,
            "qrs_plausible": True,
        },
        "quality_contract": {
            "quality_policy_version": "feature_quality_v2",
            "review_required_features": [
                f"st_{lead}"
                for lead, status in (
                    ("V2", v2_status),
                    ("V3", v3_status),
                )
                if status == "review_required"
            ],
            "unavailable_features": [],
            "feature_quality": {
                "st_V2": {
                    "status": v2_status,
                    "reasons": (
                        ["baseline_drift_in_relevant_leads"]
                        if v2_status == "review_required"
                        else []
                    ),
                },
                "st_V3": {
                    "status": v3_status,
                    "reasons": (
                        ["baseline_drift_in_relevant_leads"]
                        if v3_status == "review_required"
                        else []
                    ),
                },
            },
        },
        "st_provenance": {},
    }


def test_clean_st_pair_preserves_escalation():
    engine = ClinicalTriageEngine()
    result = engine.evaluate(make_payload())

    assert result["final_tier"] == engine.TIER_ESCALATION

    quality = result["quality_assessment"]

    assert quality["status"] == "assessed"
    assert quality["st_finding_quality_concern"] is False
    assert quality["routing_modified"] is False


def test_flagged_st_pair_preserves_escalation():
    engine = ClinicalTriageEngine()

    clean = engine.evaluate(make_payload())
    flagged = engine.evaluate(
        make_payload(v3_status="review_required")
    )

    assert flagged["final_tier"] == clean["final_tier"]
    assert flagged["effective_risk_pct"] == clean["effective_risk_pct"]
    assert flagged["findings"] == clean["findings"]

    assessment = flagged["quality_assessment"]

    assert assessment["st_finding_quality_concern"] is True
    assert assessment["routing_modified"] is False
    assert assessment["st_finding_assessments"]

    assert any(
        "V3" in item["flagged_leads"]
        for item in assessment["st_finding_assessments"]
    )


def test_unavailable_st_source_is_reported():
    engine = ClinicalTriageEngine()

    result = engine.evaluate(
        make_payload(v2_status="unavailable")
    )

    assessment = result["quality_assessment"]

    assert assessment["st_finding_quality_concern"] is True
    assert result["final_tier"] == engine.TIER_ESCALATION


def test_missing_quality_contract_does_not_fabricate_validation():
    engine = ClinicalTriageEngine()
    payload = make_payload()
    payload.pop("quality_contract")
    payload.pop("st_provenance")

    result = engine.evaluate(payload)

    assessment = result["quality_assessment"]

    assert assessment["status"] == "not_available"
    assert assessment["measurement_validation"] == "not_established"
    assert assessment["routing_modified"] is False


def test_no_st_finding_does_not_claim_st_quality_problem():
    engine = ClinicalTriageEngine()

    payload = make_payload(v3_status="review_required")
    payload["st_by_lead"] = {
        "V2": 0.02,
        "V3": 0.03,
    }

    result = engine.evaluate(payload)

    assert result["quality_assessment"][
        "st_finding_assessments"
    ] == []

    assert result["quality_assessment"][
        "st_finding_quality_concern"
    ] is False


def test_metadata_does_not_mutate_original_payload():
    engine = ClinicalTriageEngine()
    payload = make_payload(v2_status="review_required")
    original = copy.deepcopy(payload)

    engine.evaluate(payload)

    assert payload == original


def test_quality_warning_does_not_raise_statistical_risk():
    engine = ClinicalTriageEngine()

    clean = make_payload(risk=72.0)
    flagged = make_payload(
        risk=72.0,
        v3_status="review_required",
    )

    baseline = engine.evaluate(clean)
    assessed = engine.evaluate(flagged)

    assert assessed["baseline_ensemble_risk_pct"] == 72.0
    assert assessed["effective_risk_pct"] == 72.0
    assert assessed["final_tier"] == baseline["final_tier"]
