
"""Tests for SQI version selection in feature extraction."""

import numpy as np
import pytest

import src.features.extract_features as features


LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def test_unknown_version_rejected_before_processing():
    with pytest.raises(ValueError, match="sqi_version"):
        features.extract_clinical_features(
            np.zeros((1000, 12)),
            sqi_version="v4",
        )


def test_v1_preserves_default_behavior(monkeypatch):
    calls = []

    monkeypatch.setattr(
        features,
        "filter_ecg",
        lambda x, **kwargs: x,
    )

    def mock_v1(signal, **kwargs):
        calls.append("v1")
        return {
            "is_acceptable": False,
            "overall_sqi": 0.2,
            "passed_leads": "0/12",
        }

    monkeypatch.setattr(
        features,
        "assess_ecg_quality",
        mock_v1,
    )

    def unexpected_v3(*args, **kwargs):
        pytest.fail("V3 should not run by default")

    monkeypatch.setattr(
        features,
        "assess_ecg_quality_v3",
        unexpected_v3,
    )

    result = features.extract_clinical_features(
        np.zeros((1000, 12)),
        lead_names=LEADS,
    )

    assert result["status"] == "rejected"
    assert calls == ["v1"]


def test_v3_receives_raw_signal(monkeypatch):
    raw = np.ones((1000, 12))
    cleaned = np.full_like(raw, 2.0)

    monkeypatch.setattr(
        features,
        "filter_ecg",
        lambda x, **kwargs: cleaned,
    )

    def mock_v3(filtered, **kwargs):
        assert filtered is cleaned
        assert kwargs["raw_signals"] is raw
        assert kwargs["fs"] == 100.0

        return {
            "is_acceptable": True,
            "technical_failure_leads": ["V3"],
            "overall_sqi": 0.99,
        }

    monkeypatch.setattr(
        features,
        "assess_ecg_quality_v3",
        mock_v3,
    )

    def unexpected_peaks(*args, **kwargs):
        pytest.fail(
            "R-peak detection must not run after "
            "technical failure"
        )

    monkeypatch.setattr(
        features,
        "detect_r_peaks",
        unexpected_peaks,
    )

    result = features.extract_clinical_features(
        raw,
        lead_names=LEADS,
        sqi_version="v3",
    )

    assert result["status"] == "rejected"
    assert "V3" in result["reason"]
    assert result["features"] is None


def test_v3_rejects_unacceptable_quality(monkeypatch):
    monkeypatch.setattr(
        features,
        "filter_ecg",
        lambda x, **kwargs: x,
    )

    monkeypatch.setattr(
        features,
        "assess_ecg_quality_v3",
        lambda *args, **kwargs: {
            "is_acceptable": False,
            "technical_failure_leads": [],
            "overall_sqi": 0.4,
            "passed_leads": "5/12",
        },
    )

    result = features.extract_clinical_features(
        np.zeros((1000, 12)),
        lead_names=LEADS,
        sqi_version="v3",
    )

    assert result["status"] == "rejected"
    assert result["features"] is None
