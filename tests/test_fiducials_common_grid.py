"""Regression tests for experimental common-grid delineation."""
import numpy as np
import pytest
from src.features.fiducials import analyze_12lead_features as original
from src.features.fiducials_common_grid import analyze_12lead_features as common

NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def synthetic(fs):
    t = np.arange(int(fs*10)) / fs
    peaks = np.array([int(round(p * fs)) for p in np.arange(0.7, 9.0, 0.8)])
    x = np.zeros((len(t), 12))
    for j in range(12):
        for p in peaks / fs:
            x[:, j] += (0.8 + 0.02*j) * np.exp(-0.5*((t-p)/0.014)**2)
            x[:, j] -= 0.14 * np.exp(-0.5*((t-(p+0.028))/0.013)**2)
    return x, peaks


def test_500hz_same_as_original_for_core_measurements():
    x, peaks = synthetic(500)
    a = original(x, peaks, NAMES, fs=500)
    b = common(x, peaks, NAMES, fs=500)
    assert a['status'] == b['status'] == 'success'
    assert a['qrs_duration_ms'] == b['qrs_duration_ms']
    assert a['st_by_lead'] == b['st_by_lead']
    assert a['beats_used'] == b['beats_used']


def test_100hz_keeps_a_500hz_analysis_grid():
    x, peaks = synthetic(100)
    result = common(x, peaks, NAMES, fs=100)
    assert result['status'] == 'success'
    assert result['analysis_fs'] == 500
    assert result['qrs_duration_ms'] > 0
    assert np.isfinite(list(result['st_by_lead'].values())).all()
    assert result['qrs_duration_ms'] == pytest.approx(
        result['j_point_ms_relative_to_r'] - result['qrs_onset_ms_relative_to_r']
    )


def test_reject_nonfinite_samples():
    x, peaks = synthetic(100)
    x[25, 3] = np.nan
    assert common(x, peaks, NAMES, fs=100)['status'] == 'rejected'


def test_unsupported_rate_fails_explicitly():
    x, peaks = synthetic(100)
    with pytest.raises(ValueError, match='100 Hz and 500 Hz'):
        common(x, peaks, NAMES, fs=200)
