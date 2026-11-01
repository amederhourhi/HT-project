import numpy as np
import pytest
from src.dsp.sqi_v3 import assess_ecg_quality_v3


def test_stable_repeating_signal_no_sustained_flag():
    t = np.arange(1000) / 100
    x = 0.3 * np.sin(2 * np.pi * 3 * t)
    result = assess_ecg_quality_v3(x[:, None], lead_names=["I"], fs=100)
    assert result["sustained_instability_leads"] == []
    assert result["requires_review"] is False


def test_sustained_amplitude_change_requests_review():
    t = np.arange(1000) / 100
    x = 0.3 * np.sin(2 * np.pi * 3 * t)
    x[400:] *= 2.5
    result = assess_ecg_quality_v3(x[:, None], lead_names=["I"], fs=100)
    assert result["sustained_instability_leads"] == ["I"]
    assert result["requires_review"] is True
    assert result["ready_for_automated_inference"] is True
    assert result["artifact_suspected_leads"] == ["I"]


def test_transient_change_is_flagged_without_sustained_review():
    t = np.arange(1000) / 100
    x = 0.3 * np.sin(2 * np.pi * 3 * t)
    x[400:600] *= 2.5
    result = assess_ecg_quality_v3(x[:, None], lead_names=["I"], fs=100)
    lead = result["lead_scores"]["I"]
    assert lead["transient_amplitude_change"] is True
    assert lead["sustained_amplitude_instability"] is False


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        assess_ecg_quality_v3(np.ones((1000, 1)), amplitude_ratio_threshold=1.0)
