
from src.features.st_provenance import build_st_provenance
from src.features.extract_features import FEATURE_COLUMNS
from src.features.quality_contract import (
    build_feature_quality_contract,
)


def test_aggregate_tracks_actual_source():
    values = {
        "V2": 0.30,
        "V3": 0.12,
        "V5": 0.08,
    }

    result = build_st_provenance(values, values.keys())

    assert result["st_anterior_max"]["source_leads"] == ["V2"]
    assert result["max_st_elevation_mv"]["source_leads"] == ["V2"]


def test_tied_maxima_preserve_both_leads():
    result = build_st_provenance(
        {"V2": 0.20, "V3": 0.20},
        ["V2", "V3"],
    )

    assert result["st_anterior_max"]["source_leads"] == [
        "V2", "V3"
    ]


def test_untrusted_lead_cannot_contribute():
    result = build_st_provenance(
        {"V2": 0.50, "V3": 0.10},
        ["V3"],
    )

    assert result["st_anterior_max"]["source_leads"] == ["V3"]


def test_empty_territory_is_unavailable():
    result = build_st_provenance(
        {"V2": 0.10},
        ["V2"],
    )

    assert result["st_lateral_max"]["available"] is False
    assert result["st_lateral_max"]["value_mv"] is None


def test_global_depression_tracks_minimum():
    result = build_st_provenance(
        {"V1": -0.04, "V6": -0.17},
        ["V1", "V6"],
    )

    assert result["max_st_depression_mv"]["source_leads"] == ["V6"]


def test_irrelevant_drift_does_not_flag_global_max():
    values = {"V2": 0.30, "V5": 0.05}
    provenance = build_st_provenance(values, values.keys())

    features = {name: 0.0 for name in FEATURE_COLUMNS}
    features["leads_trusted"] = "V2,V5"

    contract = build_feature_quality_contract(
        features,
        {
            "baseline_drift_review_leads": ["V5"],
            "technical_failure_leads": [],
        },
        sqi_version="v3",
        feature_columns=FEATURE_COLUMNS,
        st_provenance=provenance,
    )

    quality = contract["feature_quality"]

    assert quality["st_V5"]["status"] == "review_required"
    assert quality["max_st_elevation_mv"]["status"] == "not_validated"
    assert quality["st_lateral_max"]["status"] == "review_required"


def test_tied_max_with_drift_requires_review():
    values = {"V2": 0.20, "V3": 0.20}
    provenance = build_st_provenance(values, values.keys())

    features = {name: 0.0 for name in FEATURE_COLUMNS}
    features["leads_trusted"] = "V2,V3"

    contract = build_feature_quality_contract(
        features,
        {
            "baseline_drift_review_leads": ["V3"],
            "technical_failure_leads": [],
        },
        sqi_version="v3",
        feature_columns=FEATURE_COLUMNS,
        st_provenance=provenance,
    )

    assert (
        contract["feature_quality"]["st_anterior_max"]["status"]
        == "review_required"
    )
