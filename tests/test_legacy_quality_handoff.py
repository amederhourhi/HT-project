
"""
Verify V1 quality metadata reaches clinical triage.

No trained checkpoints or downloaded ECGs required.

These tests preserve the existing model predictions
and triage routing while verifying quality propagation.
"""

import numpy as np
import pytest

import src.triage.risk_fusion as fusion

from src.dsp.input_validation import STANDARD_LEADS
from src.triage.triage_rules import ClinicalTriageEngine


def make_model(monkeypatch, quality_contract):
    """Create a synthetic legacy ensemble without checkpoints."""

    # Do not call __init__, which loads model files.
    model = object.__new__(
        fusion.CardioSenseEnsemble
    )

    # Synthetic measurements sufficient for V1 inference.
    features = {
        "leads_trusted": "V2,V3",
        "st_anterior_max": 0.32,
        "st_inferior_max": 0.0,
        "st_lateral_max": 0.0,
        "heart_rate_bpm": 75.0,
        "qrs_duration_ms": 90.0,
        "qrs_plausible": True,
        "max_st_elevation_mv": 0.32,
        "overall_sqi": 0.85,
        "beats_used": 10,
    }

    # Two contiguous leads meet the research ST rule.
    features["st_V2"] = 0.30
    features["st_V3"] = 0.32

    # Fake extraction: only model inference is mocked.
    # Real triage and quality handoff will execute.
    def fake_extract(
        signals,
        *,
        fs,
        lead_names,
        sqi_version,
    ):
        assert sqi_version == "v1"
        assert fs == 100.0
        assert list(lead_names) == list(STANDARD_LEADS)

        return {
            "status": "success",
            "features": features,
            "quality_contract": quality_contract,
            "st_provenance": {
                "st_V2": {
                    "available": True,
                    "source_leads": ["V2"],
                    "value_mv": 0.30,
                },
                "st_V3": {
                    "available": True,
                    "source_leads": ["V3"],
                    "value_mv": 0.32,
                },
            },
            "sqi_details": {
                "overall_sqi": 0.85,
                "is_acceptable": True,
            },
        }

    monkeypatch.setattr(
        fusion,
        "extract_clinical_features",
        fake_extract,
    )

    # Fixed probabilities: RF 20%, CNN 30%.
    # The fused research score must remain 25%.
    monkeypatch.setattr(
        model,
        "_predict_rf",
        lambda features: 0.20,
    )

    monkeypatch.setattr(
        model,
        "_predict_cnn",
        lambda signals, fs: 0.30,
    )

    # Execute the REAL clinical triage engine.
    model.triage_engine = ClinicalTriageEngine()

    return model


def make_contract(flagged):
    """Produce synthetic V1 quality metadata."""

    status = (
        "review_required"
        if flagged
        else "not_validated"
    )

    reasons = (
        ["aggregate_provenance_mismatch"]
        if flagged
        else ["st_quality_not_assessed_by_v1"]
    )

    return {
        "quality_policy_version": "feature_quality_v2",
        "sqi_version": "v1",
        "review_required_features": (
            ["st_V2"] if flagged else []
        ),
        "unavailable_features": [],
        "feature_quality": {
            "st_V2": {
                "status": status,
                "reasons": reasons,
            },
            "st_V3": {
                "status": "not_validated",
                "reasons": [
                    "st_quality_not_assessed_by_v1"
                ],
            },
        },
    }


def predict(model):
    """Run one synthetic standardized ECG."""

    signals = np.zeros((1000, 12))

    return model.predict(
        signals,
        fs=100.0,
        lead_names=list(STANDARD_LEADS),
        age=45,
        sex="MALE",
        units="mV",
    )


@pytest.mark.parametrize("flagged", [False, True])
def test_quality_metadata_reaches_triage(
    monkeypatch,
    flagged,
):
    contract = make_contract(flagged)

    model = make_model(
        monkeypatch,
        contract,
    )

    result = predict(model)

    # The extraction contract reaches the ensemble.
    assert result["quality_contract"] == contract
    assert result["st_provenance"]["st_V2"][
        "source_leads"
    ] == ["V2"]

    # The real triage engine receives the metadata.
    assessment = result["clinical_triage"][
        "quality_assessment"
    ]

    assert assessment["status"] == "assessed"
    assert assessment["st_finding_quality_concern"] is flagged
    assert assessment["routing_modified"] is False

    # Measurement concerns must not change probabilities.
    assert result["risk_score_pct"] == 25.0

    # The contiguous ST finding still determines routing.
    assert result["risk_tier"] == "TIER_3_ESCALATION"

    if flagged:
        assert any(
            "V2" in item["flagged_leads"]
            for item in assessment["st_finding_assessments"]
        )


def test_missing_contract_does_not_claim_validation(
    monkeypatch,
):
    model = make_model(
        monkeypatch,
        quality_contract=None,
    )

    result = predict(model)

    assessment = result["clinical_triage"][
        "quality_assessment"
    ]

    assert assessment["status"] == "not_available"
    assert assessment["measurement_validation"] == (
        "not_established"
    )
    assert result["risk_score_pct"] == 25.0
