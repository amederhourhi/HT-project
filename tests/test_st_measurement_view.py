
"""Tests for the ST measurement presentation contract."""

import copy

from src.triage.st_measurement_view import (
    build_st_measurement_view,
)


def test_empty_measurements_are_unavailable():
    result = build_st_measurement_view({
        "st_by_lead": {},
    })

    assert result["available_lead_count"] == 0
    assert result["peak_st"]["max_st_mv"] is None

    for territory in result["territories"].values():
        assert territory["max_st_mv"] is None
        assert territory["coverage"] == "unavailable"


def test_real_zero_is_preserved():
    result = build_st_measurement_view({
        "st_by_lead": {"II": 0.0},
    })

    assert result["available_lead_count"] == 1
    assert result["per_lead"]["II"]["st_mv"] == 0.0
    assert result["peak_st"]["max_st_mv"] == 0.0


def test_nonfinite_values_are_excluded():
    result = build_st_measurement_view({
        "st_by_lead": {
            "II": float("nan"),
            "III": float("inf"),
            "V2": None,
        },
    })

    assert result["available_lead_count"] == 0
    assert result["per_lead"]["II"]["st_mv"] is None
    assert result["per_lead"]["III"]["available"] is False


def test_unavailable_quality_excludes_numeric_value():
    result = build_st_measurement_view({
        "st_by_lead": {
            "V2": 0.30,
            "V3": 0.32,
        },
        "quality_contract": {
            "feature_quality": {
                "st_V2": {"status": "unavailable"},
            },
        },
    })

    assert result["available_lead_count"] == 1
    assert result["per_lead"]["V2"]["st_mv"] is None
    assert result["peak_st"]["max_st_mv"] == 0.32


def test_unavailable_provenance_excludes_value():
    result = build_st_measurement_view({
        "st_by_lead": {"V2": 0.30},
        "st_provenance": {
            "st_V2": {"available": False},
        },
    })

    assert result["per_lead"]["V2"]["st_mv"] is None
    assert result["available_lead_count"] == 0


def test_review_required_remains_visible():
    result = build_st_measurement_view({
        "st_by_lead": {"V2": 0.30},
        "quality_contract": {
            "feature_quality": {
                "st_V2": {
                    "status": "review_required",
                    "reasons": [
                        "aggregate_provenance_mismatch"
                    ],
                },
            },
        },
    })

    lead = result["per_lead"]["V2"]

    assert lead["available"] is True
    assert lead["st_mv"] == 0.30
    assert lead["quality_status"] == "review_required"
    assert "aggregate_provenance_mismatch" in lead["reasons"]
    assert result["clinical_validation"] == "not_established"


def test_partial_territory_uses_real_measurement():
    result = build_st_measurement_view({
        "st_by_lead": {
            "V2": 0.30,
            "V3": 0.32,
        },
        "territory_breakdown": {
            # Historical fallback must be ignored.
            "Inferior (RCA)": {
                "max_st_mv": 0.0,
            },
        },
    })

    anterior = result["territories"][
        "Anterior / Septal (LAD)"
    ]

    inferior = result["territories"]["Inferior (RCA)"]

    assert anterior["coverage"] == "partial"
    assert anterior["max_st_mv"] == 0.32

    assert inferior["coverage"] == "unavailable"
    assert inferior["max_st_mv"] is None
    assert result["legacy_classifier_aggregates_used"] is False


def test_tied_maximum_tracks_both_leads():
    result = build_st_measurement_view({
        "st_by_lead": {
            "V2": 0.20,
            "V3": 0.20,
        },
    })

    assert result["peak_st"]["source_leads"] == [
        "V2", "V3"
    ]


def test_provenance_mismatch_is_disclosed():
    result = build_st_measurement_view({
        "st_by_lead": {"V2": 0.30},
        "provenance_consistency": {
            "max_st_elevation_mv": {
                "consistent": False,
                "reason": "aggregate_value_mismatch",
            },
        },
    })

    assert result["aggregate_provenance_concerns"] == [
        {
            "feature": "max_st_elevation_mv",
            "reason": "aggregate_value_mismatch",
        },
    ]


def test_input_is_not_modified():
    payload = {
        "st_by_lead": {
            "V2": 0.30,
            "V3": 0.32,
        },
    }

    original = copy.deepcopy(payload)

    build_st_measurement_view(payload)

    assert payload == original
