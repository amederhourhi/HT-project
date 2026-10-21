import numpy as np
import pytest

from src.dsp.filters import filter_ecg, get_bandpass_filter, get_notch_filter


def test_bandpass_filter_coefficients_are_finite():
    b, a = get_bandpass_filter(fs=100.0)
    assert np.isfinite(b).all()
    assert np.isfinite(a).all()


def test_notch_filter_coefficients_are_finite():
    b, a = get_notch_filter(notch_freq=50.0, fs=250.0)
    assert np.isfinite(b).all()
    assert np.isfinite(a).all()


def test_filter_preserves_signal_shape_and_finite_values():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(500, 12))
    cleaned = filter_ecg(raw, fs=100.0)

    assert cleaned.shape == raw.shape
    assert np.isfinite(cleaned).all()


def test_filter_rejects_non_finite_input():
    raw = np.ones((100, 2))
    raw[10, 0] = np.nan

    with pytest.raises(ValueError):
        filter_ecg(raw, fs=100.0)


def test_filter_rejects_too_short_signal():
    raw = np.ones((10, 2))

    with pytest.raises(ValueError):
        filter_ecg(raw, fs=100.0)
