
"""Tests for standardized ECG input in the legacy ensemble.

These tests do not load trained model checkpoints.
They verify what reaches feature extraction and the CNN.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import src.triage.risk_fusion as fusion
from src.dsp.input_validation import STANDARD_LEADS


@pytest.fixture
def ensemble(monkeypatch):
    """Create an ensemble with fake models for input testing."""

    # Avoid loading actual RF and CNN checkpoints.
    model = object.__new__(fusion.CardioSenseEnsemble)

    # Record which signals reach each model component.
    observed = {
        "features": [],
        "cnn": [],
    }

    # Minimal valid feature result for the legacy pipeline.
    features = {
        "leads_trusted": ",".join(STANDARD_LEADS),
        "st_anterior_max": 0.0,
        "st_inferior_max": 0.0,
        "st_lateral_max": 0.0,
        "heart_rate_bpm": 70.0,
        "qrs_duration_ms": 90.0,
        "qrs_plausible": True,
        "max_st_elevation_mv": 0.0,
        "overall_sqi": 1.0,
        "beats_used": 10,
    }

    for lead in STANDARD_LEADS:
        features[f"st_{lead}"] = 0.0

    def fake_extract(signals, *, fs, lead_names, sqi_version):
        """Capture the waveform received by feature extraction."""
        assert fs == 100.0
        assert list(lead_names) == list(STANDARD_LEADS)
        assert sqi_version == "v1"

        observed["features"].append(signals.copy())

        return {
            "status": "success",
            "features": features.copy(),
        }

    def fake_cnn(signals, fs):
        """Capture the waveform received by the CNN."""
        assert fs == 100.0
        observed["cnn"].append(signals.copy())
        return 0.30

    # Replace inference only. Keep real input validation.
    monkeypatch.setattr(
        fusion, "extract_clinical_features", fake_extract
    )
    monkeypatch.setattr(model, "_predict_cnn", fake_cnn)
    monkeypatch.setattr(
        model, "_predict_rf", lambda features: 0.20
    )

    # Minimal fake triage response.
    model.triage_engine = SimpleNamespace(
        evaluate=lambda result: {
            "final_tier": result["risk_tier"],
            "triage_action": result["triage_action"],
        }
    )

    return model, observed


def make_ecg():
    """Generate a synthetic ECG with distinct lead waveforms."""
    time = np.arange(1000) / 100.0

    return np.column_stack([
        0.1 * np.sin(
            2 * np.pi * (1.0 + i * 0.2) * time
        ) + 0.01 * i
        for i in range(12)
    ])


def test_shuffled_leads_and_units_produce_same_inputs(ensemble):
    """Equivalent ECGs must be identical after standardization."""

    model, observed = ensemble

    # Original ECG: standard lead order, all values in mV.
    original = make_ecg()

    result_original = model.predict(
        original,
        fs=100.0,
        lead_names=list(STANDARD_LEADS),
        units="mV",
    )

    # Represent the SAME ECG using shuffled channels.
    order = [11, 5, 0, 3, 8, 10, 1, 6, 9, 4, 7, 2]

    shuffled = original[:, order].copy()
    shuffled_names = [STANDARD_LEADS[i] for i in order]

    # Some channels use microvolts instead of millivolts.
    units = [
        "uV" if i % 2 else "mV"
        for i in range(12)
    ]

    for i, unit in enumerate(units):
        if unit == "uV":
            shuffled[:, i] *= 1000.0

    result_shuffled = model.predict(
        shuffled,
        fs=100.0,
        lead_names=shuffled_names,
        units=units,
    )

    # Feature extraction must receive the correct signals.
    np.testing.assert_allclose(
        observed["features"][0],
        observed["features"][1],
        rtol=0,
        atol=1e-12,
    )

    # The CNN must receive the same standardized signals.
    np.testing.assert_allclose(
        observed["cnn"][0],
        observed["cnn"][1],
        rtol=0,
        atol=1e-12,
    )

    # RF features and CNN input must share the same waveform.
    np.testing.assert_allclose(
        observed["features"][0],
        observed["cnn"][0],
    )

    # The final research risk score must also match.
    assert (
        result_original["risk_score_pct"]
        == result_shuffled["risk_score_pct"]
        == 25.0
    )


@pytest.mark.parametrize("problem", [
    "missing_lead_names",
    "duplicate_leads",
    "invalid_units",
    "wrong_sampling_rate",
    "wrong_sample_count",
    "nan_samples",
])
def test_invalid_ecg_is_rejected(ensemble, problem):
    """Invalid ECGs must fail before reaching either model."""

    model, observed = ensemble
    signals = make_ecg()

    kwargs = {
        "fs": 100.0,
        "lead_names": list(STANDARD_LEADS),
        "units": "mV",
    }

    if problem == "missing_lead_names":
        kwargs["lead_names"] = None

    elif problem == "duplicate_leads":
        kwargs["lead_names"] = ["I"] * 12

    elif problem == "invalid_units":
        kwargs["units"] = "unknown"

    elif problem == "wrong_sampling_rate":
        kwargs["fs"] = 500.0

    elif problem == "wrong_sample_count":
        signals = signals[:500]

    elif problem == "nan_samples":
        signals[0, 0] = np.nan

    with pytest.raises(ValueError):
        model.predict(signals, **kwargs)

    # Invalid data must never reach inference.
    assert observed["features"] == []
    assert observed["cnn"] == []
