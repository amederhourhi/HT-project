
"""Test the V3 quality handoff without loading RF or CNN checkpoints."""

import numpy as np
import pytest

import src.triage.risk_fusion as fusion
from src.features.extract_features import FEATURE_COLUMNS
from src.features.quality_contract import build_feature_quality_contract


def make_extraction(*, drift=(), trusted=None):
    """Build a valid V3 extraction result with real quality metadata."""

    leads = fusion.DEFAULT_LEADS

    if trusted is None:
        trusted = leads

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


@pytest.fixture
def record(monkeypatch):
    """Provide a standardized ECG without reading disk files."""

    signals = np.zeros((1000, 12), dtype=float)

    def fake_loader(path):
        assert path == "test_record"
        return signals, 100.0, fusion.DEFAULT_LEADS.copy()

    monkeypatch.setattr(
        fusion,
        "load_standardized_ecg",
        fake_loader,
    )

    return signals


def test_eligible_ecg_is_not_scored(monkeypatch, record):
    """Passing quality does not authorize using old checkpoints."""

    def fake_extract(signals, *, fs, lead_names, sqi_version):
        assert signals is record
        assert fs == 100.0
        assert lead_names == fusion.DEFAULT_LEADS
        assert sqi_version == "v3"

        return make_extraction()

    monkeypatch.setattr(
        fusion, "extract_clinical_features", fake_extract
    )

    result = fusion.CardioSenseEnsemble.assess_v3_record(
        "test_record"
    )

    assert result["status"] == "quality_eligible_unscored"
    assert result["quality_decision"]["allow_research_scoring"]
    assert result["model_scoring_performed"] is False
    assert result["risk_score_pct"] is None
    assert result["clinical_triage"] is None
    assert result["model_compatibility"] == "not_verified"


def test_review_required_abstains(monkeypatch, record):
    monkeypatch.setattr(
        fusion,
        "extract_clinical_features",
        lambda *args, **kwargs: make_extraction(drift=["V5"]),
    )

    result = fusion.CardioSenseEnsemble.assess_v3_record(
        "test_record"
    )

    assert result["status"] == "abstained"
    assert result["risk_score_pct"] is None
    assert "st_V5" in (
        result["quality_decision"]["review_required_features"]
    )


def test_unavailable_measurement_abstains(monkeypatch, record):
    trusted = [
        lead for lead in fusion.DEFAULT_LEADS if lead != "V5"
    ]

    monkeypatch.setattr(
        fusion,
        "extract_clinical_features",
        lambda *args, **kwargs: make_extraction(trusted=trusted),
    )

    result = fusion.CardioSenseEnsemble.assess_v3_record(
        "test_record"
    )

    assert result["status"] == "abstained"
    assert "st_V5" in (
        result["quality_decision"]["unavailable_features"]
    )


def test_extraction_rejection_abstains(monkeypatch, record):
    monkeypatch.setattr(
        fusion,
        "extract_clinical_features",
        lambda *args, **kwargs: {
            "status": "rejected",
            "reason": "Technical signal failure in V3",
            "sqi_details": {"technical_failure_leads": ["V3"]},
            "features": None,
        },
    )

    result = fusion.CardioSenseEnsemble.assess_v3_record(
        "test_record"
    )

    assert result["status"] == "abstained"
    assert "Technical signal failure" in result["reason"]
    assert result["model_scoring_performed"] is False


def test_missing_quality_contract_abstains(monkeypatch, record):
    extraction = make_extraction()
    del extraction["quality_contract"]

    monkeypatch.setattr(
        fusion,
        "extract_clinical_features",
        lambda *args, **kwargs: extraction,
    )

    result = fusion.CardioSenseEnsemble.assess_v3_record(
        "test_record"
    )

    assert result["status"] == "abstained"
    assert result["quality_decision"]["allow_research_scoring"] is False


def test_unexpected_extraction_status_raises(monkeypatch, record):
    monkeypatch.setattr(
        fusion,
        "extract_clinical_features",
        lambda *args, **kwargs: {"status": "unknown"},
    )

    with pytest.raises(ValueError, match="Unexpected"):
        fusion.CardioSenseEnsemble.assess_v3_record("test_record")
