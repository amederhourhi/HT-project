
"""Regression tests for research report accuracy."""

from src.triage.report_generator import generate_clinical_report


def make_payload(st_by_lead=None):
    """Build a minimal synthetic ensemble result."""
    return {
        "patient": {"age": 45, "sex": "MALE"},
        "st_by_lead": st_by_lead or {},
        "clinical_biomarkers": {
            "overall_sqi": 0.85,
            "heart_rate_bpm": 72.0,
            "qrs_duration_ms": 95.0,
            "max_st_elevation_mv": 0.15,
        },
        "engine_probabilities": {
            "interpretable_rf": 0.20,
            "deep_cnn": 0.30,
            "fused_composite": 0.25,
        },
        "territory_breakdown": {},
    }


def make_triage(override=False):
    return {
        "final_tier": "TIER_1_ROUTINE",
        "triage_action": "Routine follow-up",
        "effective_risk_pct": 25.0,
        "override_active": override,
        "override_reasons": [],
        "findings": [],
        "quality_assessment": {
            "status": "not_available",
            "measurement_validation": "not_established",
        },
    }


def test_partial_st_coverage_is_reported():
    report = generate_clinical_report(
        "synthetic",
        make_payload({"V2": 0.15, "V3": 0.18}),
        make_triage(),
    )

    assert "2/12 available" in report
    assert "12/12 leads active" not in report
    assert "ST Readings Missing" in report


def test_missing_st_values_are_not_counted():
    report = generate_clinical_report(
        "synthetic",
        make_payload({
            "V2": None,
            "V3": float("nan"),
            "II": 0.0,
        }),
        make_triage(),
    )

    # A genuine zero is available.
    # None and NaN are unavailable.
    assert "1/12 available" in report


def test_no_guardrail_does_not_imply_normal_ecg():
    report = generate_clinical_report(
        "synthetic",
        make_payload({"II": 0.0}),
        make_triage(),
    )

    assert "Within Normal Limits" not in report
    assert "ECG normality not established" in report


def test_missing_quality_validation_is_disclosed():
    report = generate_clinical_report(
        "synthetic",
        make_payload({"II": 0.0}),
        make_triage(),
    )

    assert "not_available" in report
    assert "Not independently established" in report
    assert "not clinical validation" in report


def test_flagged_st_pair_warning_appears():
    """Quality concerns must be visible in the report."""

    triage = make_triage(override=True)

    triage["override_reasons"] = [
        "G1_CONTIGUOUS_STE"
    ]

    triage["quality_assessment"] = {
        "status": "assessed",
        "review_required_features": ["st_V2"],
        "unavailable_features": ["st_V5"],
        "st_finding_quality_concern": True,
        "st_finding_assessments": [
            {
                "leads": ["V2", "V3"],
                "known_quality_concern": True,
                "flagged_leads": ["V2"],
                "lead_quality": {
                    "V2": {
                        "status": "review_required",
                        "reasons": [
                            "aggregate_provenance_mismatch"
                        ],
                    },
                },
            },
        ],
    }

    report = generate_clinical_report(
        "synthetic",
        make_payload({
            "V2": 0.30,
            "V3": 0.32,
        }),
        triage,
    )

    assert "Review Required" in report
    assert "st_V2" in report
    assert "st_V5" in report
    assert "ST pair V2/V3" in report
    assert "aggregate_provenance_mismatch" in report
    assert "Known quality concern" in report


def test_missing_quality_metadata_is_disclosed():
    """Missing metadata must never imply reliability."""

    triage = make_triage()

    triage["quality_assessment"] = {
        "status": "not_available",
        "measurement_validation": "not_established",
    }

    report = generate_clinical_report(
        "synthetic",
        make_payload({"II": 0.0}),
        triage,
    )

    assert "feature-quality metadata missing" in report
    assert "Measurement reliability has not been established" in report

def test_no_flagged_quality_issue_is_not_validation():
    """No known warning is not proof of clinical accuracy."""

    triage = make_triage()

    triage["quality_assessment"] = {
        "status": "assessed",
        "review_required_features": [],
        "unavailable_features": [],
        "st_finding_quality_concern": False,
        "st_finding_assessments": [],
    }

    report = generate_clinical_report(
        "synthetic",
        make_payload({"II": 0.0}),
        triage,
    )

    assert "None flagged by available heuristics" in report
    assert "clinical validity not established" in report
    assert "Within Normal Limits" not in report