
"""Ensure API and report agree on ST measurement availability."""

from src.triage.report_generator import generate_clinical_report
from src.triage.st_measurement_view import build_st_measurement_view


def make_ensemble():
    return {
        "patient": {"age": 45, "sex": "MALE"},
        "st_by_lead": {
            "II": 0.0,
            "V2": 0.30,
            "V3": 0.32,
            "V5": 0.50,
        },
        "clinical_biomarkers": {
            "overall_sqi": 0.85,
            "heart_rate_bpm": 75.0,
            "qrs_duration_ms": 90.0,
            "max_st_elevation_mv": 0.50,
        },
        "engine_probabilities": {
            "interpretable_rf": 0.20,
            "deep_cnn": 0.30,
            "fused_composite": 0.25,
        },
        "quality_contract": {
            "feature_quality": {
                "st_V5": {
                    "status": "unavailable",
                    "reasons": ["no_contributing_st_leads"],
                },
            },
        },
        "territory_breakdown": {
            "Lateral (LCx)": {
                "max_st_mv": 0.0,
                "leads": ["I", "AVL", "V5", "V6"],
            },
        },
    }


def make_triage():
    return {
        "final_tier": "TIER_1_ROUTINE",
        "triage_action": "Research follow-up",
        "effective_risk_pct": 25.0,
        "override_active": False,
        "override_reasons": [],
        "findings": [],
        "quality_assessment": {
            "status": "not_available",
        },
    }


def test_report_matches_measurement_view():
    ensemble = make_ensemble()

    view = build_st_measurement_view(ensemble)

    report = generate_clinical_report(
        "synthetic",
        ensemble,
        make_triage(),
    )

    assert view["available_lead_count"] == 3
    assert "3/12 available" in report

    assert view["peak_st"]["max_st_mv"] == 0.32
    assert "Peak ST Shift     : +0.320 mV" in report

    assert (
        view["territories"]["Lateral (LCx)"]["max_st_mv"]
        is None
    )
    assert "Lateral (LCx): Unavailable" in report


def test_actual_zero_remains_reportable():
    ensemble = make_ensemble()

    ensemble["st_by_lead"] = {"II": 0.0}
    ensemble["quality_contract"] = {}

    view = build_st_measurement_view(ensemble)

    report = generate_clinical_report(
        "synthetic",
        ensemble,
        make_triage(),
    )

    assert view["per_lead"]["II"]["st_mv"] == 0.0
    assert view["available_lead_count"] == 1

    assert "Peak ST Shift     : +0.000 mV" in report
    assert "Inferior (RCA): +0.000 mV" in report


def test_excluded_provenance_matches_report():
    ensemble = make_ensemble()

    ensemble["st_provenance"] = {
        "st_V2": {
            "available": False,
            "source_leads": [],
            "value_mv": None,
        },
    }

    view = build_st_measurement_view(ensemble)

    report = generate_clinical_report(
        "synthetic",
        ensemble,
        make_triage(),
    )

    assert view["available_lead_count"] == 2
    assert "2/12 available" in report
    assert "Peak ST Shift     : +0.320 mV" in report