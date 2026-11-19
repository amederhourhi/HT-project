
"""Tests for fail-closed SQI V3 research inference decisions."""

from copy import deepcopy

import pytest

from src.features.extract_features import FEATURE_COLUMNS
from src.features.quality_contract import build_feature_quality_contract
from src.triage.inference_quality import evaluate_research_readiness


ALL_LEADS = (
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)


def make_result(*, trusted=ALL_LEADS, drift=()):
    """Create a realistic feature-quality result for testing."""

    features = {name: 0.0 for name in FEATURE_COLUMNS}
    features["leads_trusted"] = ",".join(trusted)

    sqi = {
        "algorithm": "experimental_sqi_v3_separated",
        "is_acceptable": True,
        "ready_for_automated_inference": True,
        "requires_review": bool(drift),
        "st_measurement_review_required": bool(drift),
        "technical_failure_leads": [],
        "baseline_drift_review_leads": list(drift),
    }

    contract = build_feature_quality_contract(
        features,
        sqi,
        sqi_version="v3",
        feature_columns=FEATURE_COLUMNS,
    )

    return {
        "status": "success",
        "features": features,
        "sqi_details": sqi,
        "quality_contract": contract,
    }


def test_clean_record_is_eligible_for_research_only():
    decision = evaluate_research_readiness(make_result())

    assert decision["decision"] == "eligible"
    assert decision["allow_research_scoring"] is True
    assert decision["clinical_validation"] == "not_established"


def test_baseline_drift_blocks_scoring():
    decision = evaluate_research_readiness(
        make_result(drift=["V5"])
    )

    assert decision["decision"] == "abstain"
    assert "st_V5" in decision["review_required_features"]
    assert "st_lateral_max" in decision["review_required_features"]


def test_unavailable_lead_blocks_scoring():
    trusted = [lead for lead in ALL_LEADS if lead != "V5"]

    decision = evaluate_research_readiness(
        make_result(trusted=trusted)
    )

    assert decision["allow_research_scoring"] is False
    assert "st_V5" in decision["unavailable_features"]


def test_morphology_review_blocks_even_if_technically_ready():
    result = make_result()
    result["sqi_details"]["requires_review"] = True

    decision = evaluate_research_readiness(result)

    assert decision["decision"] == "abstain"


def test_technical_failure_blocks_scoring():
    result = make_result()
    result["sqi_details"]["technical_failure_leads"] = ["V3"]

    assert not evaluate_research_readiness(result)[
        "allow_research_scoring"
    ]


def test_sqi_not_ready_blocks_scoring():
    result = make_result()
    result["sqi_details"]["ready_for_automated_inference"] = False

    assert not evaluate_research_readiness(result)[
        "allow_research_scoring"
    ]


def test_missing_quality_contract_blocks_scoring():
    result = make_result()
    del result["quality_contract"]

    assert evaluate_research_readiness(result)["decision"] == "abstain"


def test_wrong_sqi_version_blocks_scoring():
    result = make_result()
    result["quality_contract"]["sqi_version"] = "v1"

    assert evaluate_research_readiness(result)["decision"] == "abstain"


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), "0.1"])
def test_invalid_feature_values_block_scoring(invalid):
    result = make_result()
    result["features"]["st_V2"] = invalid

    assert evaluate_research_readiness(result)["decision"] == "abstain"


def test_inconsistent_quality_summary_blocks_scoring():
    result = make_result()

    # Simulate corrupted quality metadata.
    result["quality_contract"]["review_required_features"] = ["st_V5"]

    assert evaluate_research_readiness(result)["decision"] == "abstain"


def test_rejected_extraction_cannot_be_scored():
    result = make_result()
    result["status"] = "rejected"

    assert evaluate_research_readiness(result)["decision"] == "abstain"


def test_no_input_is_rejected():
    assert evaluate_research_readiness(None)["decision"] == "abstain"
