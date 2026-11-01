import numpy as np
import pytest
from src.dsp.sqi_v3 import assess_ecg_quality_v3

FS = 100
T = np.arange(1000) / FS

def base():
    return 0.3 * np.sin(2*np.pi*3*T)

def evaluate(channels):
    return assess_ecg_quality_v3(np.column_stack(channels), lead_names=[f"L{i}" for i in range(len(channels))], fs=FS)

def test_stable_is_not_flagged():
    r = evaluate([base(), base() * .7])
    assert not r["requires_review"]
    assert r["technical_failure_leads"] == []

def test_synchronized_change_is_morphology_review_not_technical_failure():
    a, b = base(), base()*.7
    a[400:] *= 2.5
    b[400:] *= 2.5
    r = evaluate([a,b])
    assert set(r["morphology_review_leads"]) == {"L0", "L1"}
    assert r["technical_failure_leads"] == []
    assert r["ready_for_automated_inference"] is True

def test_lead_local_change_is_suspected_not_proven_artifact():
    a,b=base(),base()*.7
    a[400:] *= 2.5
    r=evaluate([a,b])
    assert r["artifact_suspected_leads"] == ["L0"]
    assert r["morphology_review_leads"] == []
    assert r["ready_for_automated_inference"] is True

def test_flatline_is_hard_failure():
    a,b=base(),base()*.7
    a[400:600] = 0
    r=evaluate([a,b])
    assert "L0" in r["technical_failure_leads"]
    assert r["ready_for_automated_inference"] is False

def test_transient_change_is_reviewed_without_sustained_classification():
    a=base()
    a[400:600] *= 2.5
    r=evaluate([a])
    assert r["lead_scores"]["L0"]["transient_amplitude_change"] is True
    assert not r["requires_review"]

def test_invalid_threshold():
    with pytest.raises(ValueError):
        assess_ecg_quality_v3(base()[:,None], amplitude_ratio_threshold=1)
