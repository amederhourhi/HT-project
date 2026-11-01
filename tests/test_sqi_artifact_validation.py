
"""Controlled artifact tests for CardioSense SQI V3."""

import numpy as np
import pytest

from src.dsp.sqi_v3 import assess_ecg_quality_v3


FS = 100
N = 1000
T = np.arange(N) / FS


def ecg_like():
    signal = 0.018 * np.sin(2 * np.pi * 0.25 * T)

    for second in np.arange(0.5, 10, 1.0):
        signal += 1.1 * np.exp(
            -0.5 * ((T - second) / 0.025) ** 2
        )
        signal += 0.20 * np.exp(
            -0.5 * ((T - (second + 0.22)) / 0.09) ** 2
        )

    return signal


def evaluate(channels, *, adc_rails_mv=None):
    signals = np.column_stack(channels)

    return assess_ecg_quality_v3(
        signals,
        raw_signals=signals,
        lead_names=[
            f"L{i}" for i in range(len(channels))
        ],
        fs=FS,
        adc_rails_mv=adc_rails_mv,
    )


def technical_alert(result, lead):
    return (
        lead in result["technical_failure_leads"]
        or not result["lead_scores"][lead]["lead_usable"]
    )


def test_clean_reference_not_a_technical_failure():
    base = ecg_like()
    result = evaluate([base, base * 0.8, base * 1.1])

    assert result["technical_failure_leads"] == []


def test_flatline_window_flagged():
    base = ecg_like()
    modified = base.copy()
    modified[400:600] = 0

    result = evaluate([modified, base * 0.8])

    assert technical_alert(result, "L0")
    assert not technical_alert(result, "L1")


def test_isolated_large_beat_preserved():
    base = ecg_like()
    modified = base.copy()

    modified += 2.0 * np.exp(
        -0.5 * ((T - 5.7) / 0.055) ** 2
    )

    result = evaluate([modified, base * 0.8])

    assert not technical_alert(result, "L0")


def test_adc_clipping_is_detected():
    base = ecg_like()
    clipped = np.clip(base, -0.4, 0.4)

    result = evaluate(
        [clipped, base],
        adc_rails_mv=(-0.4, 0.4),
    )

    assert "L0" in result["technical_failure_leads"]
    assert result["lead_scores"]["L0"]["clipping_confirmed_windows"]
    assert not result["is_acceptable"]


def test_sustained_high_frequency_noise_is_detected():
    base = ecg_like()
    noisy = base.copy()

    rng = np.random.default_rng(20261031)
    noisy[400:800] += 0.6 * rng.standard_normal(400)

    result = evaluate([noisy, base])

    assert technical_alert(result, "L0")


def test_large_baseline_drift_is_detected():
    base = ecg_like()
    drift = base + 1.3 * np.sin(
        2 * np.pi * 0.18 * T
    )

    result = evaluate([drift, base])

    assert technical_alert(result, "L0")


def test_nonfinite_input_rejected():
    base = ecg_like()
    bad = base.copy()
    bad[40] = np.nan

    with pytest.raises(ValueError):
        evaluate([bad, base])


def test_clean_peak_not_misclassified_as_clipping():
    base = ecg_like()
    result = evaluate([base, base * 0.8])

    assert (
        result["lead_scores"]["L0"]["clipping_suspected_windows"]
        == []
    )


def test_clipping_isolated_to_one_lead():
    base = ecg_like()
    clipped = np.clip(base, -0.4, 0.4)

    result = evaluate(
        [clipped, base],
        adc_rails_mv=(-0.4, 0.4),
    )

    assert "L0" in result["technical_failure_leads"]
    assert "L1" not in result["technical_failure_leads"]
    assert not result["ready_for_automated_inference"]


def test_bad_raw_shape_rejected():
    base = ecg_like()

    with pytest.raises(ValueError):
        assess_ecg_quality_v3(
            np.column_stack([base, base]),
            fs=FS,
            raw_signals=base,
        )


def test_short_noise_burst_is_review_only():
    base = ecg_like()
    noisy = base.copy()

    rng = np.random.default_rng(21)
    noisy[400:600] += 0.6 * rng.standard_normal(200)

    result = evaluate([noisy, base])

    assert "L0" not in result["technical_failure_leads"]
    assert (
        "high_frequency_noise_review"
        in result["lead_scores"]["L0"]["windows"][2]["flags"]
    )


def test_sustained_noise_is_localized():
    base = ecg_like()
    noisy = base.copy()

    rng = np.random.default_rng(20261031)
    noisy[400:800] += 0.6 * rng.standard_normal(400)

    result = evaluate([noisy, base])

    assert (
        result["lead_scores"]["L0"]["high_frequency_noise_windows"]
        == [2, 3]
    )
    assert "L0" in result["technical_failure_leads"]
    assert "L1" not in result["technical_failure_leads"]


def test_mild_baseline_wander_not_hard_failure():
    base = ecg_like()
    mild = base + 0.12 * np.sin(
        2 * np.pi * 0.18 * T
    )

    result = evaluate([mild, base])

    assert result["lead_scores"]["L0"]["baseline_drift_windows"] == []


def test_baseline_drift_is_localized():
    base = ecg_like()
    drift = base + 1.3 * np.sin(
        2 * np.pi * 0.18 * T
    )

    result = evaluate([drift, base])

    assert result["lead_scores"]["L0"]["baseline_drift_windows"]
    assert "L0" in result["technical_failure_leads"]
    assert "L1" not in result["technical_failure_leads"]
