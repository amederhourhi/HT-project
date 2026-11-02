
import pytest

from src.features.extract_features import FEATURE_COLUMNS
from src.features.quality_contract import (
    build_feature_quality_contract,
    QUALITY_POLICY_VERSION,
)


def make_features(trusted=None):
    if trusted is None:
        trusted = (
            "I,II,III,AVR,AVL,AVF,"
            "V1,V2,V3,V4,V5,V6"
        )

    features = {
        name: 0.0 for name in FEATURE_COLUMNS
    }
    features["leads_trusted"] = trusted
    return features


def make_contract(
    *,
    drift=None,
    trusted=None,
    version="v3",
):
    features = make_features(trusted)

    contract = build_feature_quality_contract(
        features,
        {
            "baseline_drift_review_leads": drift or [],
            "technical_failure_leads": [],
        },
        sqi_version=version,
        feature_columns=FEATURE_COLUMNS,
    )

    return features, contract


def test_contract_version():
    _, result = make_contract()

    assert (
        result["quality_policy_version"]
        == QUALITY_POLICY_VERSION
    )


def test_clean_ecg_does_not_claim_validation():
    _, result = make_contract()

    assert result["feature_quality"]["st_V5"]["status"] == (
        "not_validated"
    )


def test_baseline_drift_marks_affected_st_lead():
    _, result = make_contract(drift=["V5"])

    assert result["feature_quality"]["st_V5"]["status"] == (
        "review_required"
    )
    assert result["feature_quality"]["st_V2"]["status"] == (
        "not_validated"
    )


def test_territory_review_is_localized():
    _, result = make_contract(drift=["V5"])

    quality = result["feature_quality"]

    assert quality["st_lateral_max"]["status"] == (
        "review_required"
    )
    assert quality["st_anterior_max"]["status"] == (
        "not_validated"
    )
    assert quality["st_inferior_max"]["status"] == (
        "not_validated"
    )


def test_global_st_max_requires_review():
    _, result = make_contract(drift=["V5"])

    quality = result["feature_quality"]

    assert (
        quality["max_st_elevation_mv"]["status"]
        == "review_required"
    )
    assert (
        quality["max_st_depression_mv"]["status"]
        == "review_required"
    )


def test_untrusted_lead_is_unavailable():
    _, result = make_contract(
        trusted="I,II,V1,V2"
    )

    assert (
        result["feature_quality"]["st_V5"]["status"]
        == "unavailable"
    )


def test_missing_territory_is_unavailable():
    _, result = make_contract(
        trusted="I,II,V1,V2"
    )

    assert (
        result["feature_quality"]["st_inferior_max"]["status"]
        == "not_validated"
    )
    assert (
        result["feature_quality"]["st_lateral_max"]["status"]
        == "not_validated"
    )

    _, result = make_contract(
        trusted="V1,V2"
    )

    assert (
        result["feature_quality"]["st_inferior_max"]["status"]
        == "unavailable"
    )


def test_v1_does_not_inherit_v3_flags():
    _, result = make_contract(
        version="v1",
        drift=["V5"],
    )

    assert (
        result["feature_quality"]["st_V5"]["status"]
        == "not_validated"
    )


def test_classifier_features_are_unchanged():
    features = make_features()
    original = features.copy()

    build_feature_quality_contract(
        features,
        {"baseline_drift_review_leads": ["V5"]},
        sqi_version="v3",
        feature_columns=FEATURE_COLUMNS,
    )

    assert features == original
    assert len(FEATURE_COLUMNS) == 23


def test_missing_classifier_feature_fails():
    features = make_features()
    del features["overall_sqi"]

    with pytest.raises(ValueError):
        build_feature_quality_contract(
            features,
            {},
            sqi_version="v3",
            feature_columns=FEATURE_COLUMNS,
        )


def test_invalid_sqi_version_fails():
    with pytest.raises(ValueError):
        build_feature_quality_contract(
            make_features(),
            {},
            sqi_version="unknown",
            feature_columns=FEATURE_COLUMNS,
        )
