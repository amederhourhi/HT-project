
"""Test report-only ST measurement availability.

No downloaded ECGs or model checkpoints needed.
"""

from src.triage.report_generator import (
    generate_clinical_report,
)


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


def make_ensemble(readings):
    return {
        "patient": {
            "age": 45,
            "sex": "MALE",
        },
        "st_by_lead": readings,
        "clinical_biomarkers": {
            "heart_rate_bpm": 75.0,
            "qrs_duration_ms": 95.0,
            "overall_sqi": 0.85,

            # Historical aggregate value.
            # The report must not blindly trust it.
            "max_st_elevation_mv": 0.0,
        },
        "engine_probabilities": {
            "interpretable_rf": 0.20,
            "deep_cnn": 0.30,
            "fused_composite": 0.25,
        },

        # Deliberately contains a misleading fallback.
        "territory_breakdown": {
            "Inferior (RCA)": {
                "leads": ["II", "III", "AVF"],
                "max_st_mv": 0.0,
            },
        },
    }


def test_unavailable_territory_not_reported_as_zero():
    ensemble = make_ensemble({
        "V2": 0.30,
        "V3": 0.32,
    })

    report = generate_clinical_report(
        "synthetic",
        ensemble,
        make_triage(),
    )

    assert "2/12 available" in report
    assert "Inferior (RCA): Unavailable" in report
    assert "Lateral (LCx): Unavailable" in report
    assert "Peak ST Shift     : +0.320 mV" in report


def test_no_st_readings_means_no_peak():
    ensemble = make_ensemble({})

    report = generate_clinical_report(
        "synthetic",
        ensemble,
        make_triage(),
    )

    assert "0/12 available" in report
    assert "Peak ST Shift     : Unavailable" in report


def test_genuine_zero_remains_a_measurement():
    ensemble = make_ensemble({
        "II": 0.0,
    })

    report = generate_clinical_report(
        "synthetic",
        ensemble,
        make_triage(),
    )

    assert "1/12 available" in report
    assert "Peak ST Shift     : +0.000 mV" in report
    assert "Inferior (RCA): +0.000 mV" in report


def test_nonfinite_values_are_not_reported():
    ensemble = make_ensemble({
        "II": float("nan"),
        "V2": None,
        "V3": float("inf"),
    })

    report = generate_clinical_report(
        "synthetic",
        ensemble,
        make_triage(),
    )

    assert "0/12 available" in report
    assert "Peak ST Shift     : Unavailable" in report


def test_unavailable_provenance_excludes_lead():
    ensemble = make_ensemble({
        "V2": 0.30,
        "V3": 0.32,
    })

    ensemble["st_provenance"] = {
        "st_V2": {
            "available": False,
            "source_leads": [],
            "value_mv": None,
        },
    }

    report = generate_clinical_report(
        "synthetic",
        ensemble,
        make_triage(),
    )

    assert "1/12 available" in report
    assert "Peak ST Shift     : +0.320 mV" in report


def test_provenance_mismatch_is_disclosed():
    ensemble = make_ensemble({
        "V2": 0.30,
    })

    ensemble["provenance_consistency"] = {
        "max_st_elevation_mv": {
            "consistent": False,
            "reason": "aggregate_value_mismatch",
        },
    }

    report = generate_clinical_report(
        "synthetic",
        ensemble,
        make_triage(),
    )

    assert "aggregate_value_mismatch" in report
    assert "legacy feature value not displayed" in report
    assert "Peak ST Shift     : +0.300 mV" in report