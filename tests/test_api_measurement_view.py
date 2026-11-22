
"""
Tests for quality-aware ST measurements in the API.

Uses synthetic model outputs. No real model predictions
or downloaded ECGs are required.
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.api import main as api


@pytest.fixture
def evaluation_env(monkeypatch):
    """Replace waveform loading and model inference."""

    ensemble = {
        "status": "success",
        "risk_score_pct": 25.0,

        "patient": {
            "age": 45,
            "sex": "MALE",
        },

        "st_by_lead": {
            "II": 0.0,
            "V2": 0.30,
            "V3": 0.32,
            "V5": 0.50,
        },

        "clinical_biomarkers": {
            "heart_rate_bpm": 75.0,
            "qrs_duration_ms": 90.0,
            "qrs_plausible": True,
            "overall_sqi": 0.85,
            "max_st_elevation_mv": 0.50,
            "beats_used": 10,
        },

        "engine_probabilities": {
            "interpretable_rf": 0.20,
            "deep_cnn": 0.30,
            "fused_composite": 0.25,
        },

        # Legacy aggregate: must not control presentation.
        "territory_breakdown": {
            "Lateral (LCx)": {
                "leads": ["I", "AVL", "V5", "V6"],
                "max_st_mv": 0.0,
            },
        },

        "quality_contract": {
            "quality_policy_version": "feature_quality_v2",
            "sqi_version": "v1",
            "review_required_features": ["st_V2"],
            "unavailable_features": ["st_V5"],
            "feature_quality": {
                "st_V2": {
                    "status": "review_required",
                    "reasons": [
                        "aggregate_provenance_mismatch",
                    ],
                },
                "st_V3": {
                    "status": "not_validated",
                    "reasons": [],
                },
                "st_V5": {
                    "status": "unavailable",
                    "reasons": [
                        "no_contributing_st_leads",
                    ],
                },
            },
        },

        "st_provenance": {
            "st_V5": {
                "available": False,
                "source_leads": [],
                "value_mv": None,
            },
        },
    }

    original = deepcopy(ensemble)

    monkeypatch.setattr(
        api,
        "_load_record",
        lambda record_id: (
            np.zeros((1000, 12)),
            100.0,
            [
                "I", "II", "III", "AVR", "AVL", "AVF",
                "V1", "V2", "V3", "V4", "V5", "V6",
            ],
        ),
    )

    monkeypatch.setattr(
        api,
        "ensemble_pipeline",
        SimpleNamespace(
            predict=lambda *args, **kwargs: ensemble
        ),
    )

    # Only report formatting is mocked.
    # The real measurement view and triage engine execute.
    monkeypatch.setattr(
        api,
        "generate_clinical_report",
        lambda *args: "Synthetic research report",
    )

    return ensemble, original


def evaluate():
    """Evaluate the synthetic ECG through API code."""

    return api.evaluate_record(
        api.EvaluationRequest(
            record_id="synthetic",
            age=45,
            sex="MALE",
        )
    )


def test_api_returns_quality_aware_view(evaluation_env):
    result = evaluate()

    view = result["measurement_view"]

    assert result["status"] == "success"
    assert view["clinical_validation"] == "not_established"

    assert view["available_lead_count"] == 3
    assert view["per_lead"]["V5"]["st_mv"] is None
    assert view["per_lead"]["II"]["st_mv"] == 0.0

    assert (
        view["territories"]["Lateral (LCx)"]["max_st_mv"]
        is None
    )


def test_api_lead_table_excludes_unavailable_value(
    evaluation_env,
):
    result = evaluate()

    rows = {
        row["lead"]: row
        for row in result["lead_st_table"]
    }

    assert rows["V5"]["available"] is False
    assert rows["V5"]["st_mv"] is None
    assert rows["V5"]["cleared"] is None

    assert rows["II"]["available"] is True
    assert rows["II"]["st_mv"] == 0.0
    assert rows["II"]["cleared"] is False


def test_api_preserves_review_required_measurement(
    evaluation_env,
):
    result = evaluate()

    rows = {
        row["lead"]: row
        for row in result["lead_st_table"]
    }

    assert rows["V2"]["st_mv"] == 0.30
    assert rows["V2"]["quality_status"] == (
        "review_required"
    )

    assert "aggregate_provenance_mismatch" in (
        rows["V2"]["quality_reasons"]
    )


def test_api_preserves_legacy_ensemble_output(
    evaluation_env,
):
    ensemble, original = evaluation_env

    result = evaluate()

    assert ensemble == original
    assert result["ensemble"] == original

    assert result["ensemble"][
        "territory_breakdown"
    ]["Lateral (LCx)"]["max_st_mv"] == 0.0

    assert result["measurement_view"][
        "territories"
    ]["Lateral (LCx)"]["max_st_mv"] is None


def test_quality_does_not_modify_triage(
    evaluation_env,
):
    result = evaluate()

    # Research ST criteria still trigger on V2 and V3.
    assert result["triage"]["final_tier"] == (
        "TIER_3_ESCALATION"
    )

    # The existing probability remains unchanged.
    assert result["triage"][
        "effective_risk_pct"
    ] == 25.0

    assert result["measurement_presentation"][
        "triage_routing_modified"
    ] is False