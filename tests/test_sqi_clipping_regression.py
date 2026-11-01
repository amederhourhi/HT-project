
"""Regression tests for local plateaus and ADC saturation."""

import numpy as np
import pytest

from src.dsp.sqi_v3 import _rail_plateau_windows


def make_windows():
    return [{
        "start_sec": 0.0,
        "end_sec": 2.0,
        "flags": [],
        "usable": True,
    }]


def test_local_plateau_is_warning_only():
    signal = np.linspace(-0.5, 0.5, 200)
    signal[50:55] = -0.6

    suspected, confirmed = _rail_plateau_windows(
        signal,
        fs=100,
        windows=make_windows(),
    )

    assert suspected == [0]
    assert confirmed == []


def test_known_adc_rail_detects_saturation():
    signal = np.linspace(-0.5, 0.5, 200)
    signal[50:60] = 1.0

    suspected, confirmed = _rail_plateau_windows(
        signal,
        fs=100,
        windows=make_windows(),
        adc_rails_mv=(-1.0, 1.0),
    )

    assert suspected == []
    assert confirmed == [0]


def test_nonrail_plateau_not_confirmed():
    signal = np.linspace(-0.5, 0.5, 200)
    signal[50:60] = -0.175

    suspected, confirmed = _rail_plateau_windows(
        signal,
        fs=100,
        windows=make_windows(),
        adc_rails_mv=(-1.0, 1.0),
    )

    assert suspected == []
    assert confirmed == []


def test_normal_signal_has_no_warning():
    signal = np.linspace(-0.5, 0.5, 200)

    suspected, confirmed = _rail_plateau_windows(
        signal,
        fs=100,
        windows=make_windows(),
    )

    assert suspected == []
    assert confirmed == []


def test_flatline_is_not_clipping():
    signal = np.full(200, -0.175)

    suspected, confirmed = _rail_plateau_windows(
        signal,
        fs=100,
        windows=make_windows(),
    )

    assert suspected == []
    assert confirmed == []


@pytest.mark.parametrize(
    "rails",
    [
        (-1.0, 1.0),
        (-2.0, 2.0),
    ],
)
def test_plateau_within_rails_is_not_saturation(rails):
    signal = np.linspace(-0.5, 0.5, 200)
    signal[40:50] = -0.175

    suspected, confirmed = _rail_plateau_windows(
        signal,
        fs=100,
        windows=make_windows(),
        adc_rails_mv=rails,
    )

    assert suspected == []
    assert confirmed == []
