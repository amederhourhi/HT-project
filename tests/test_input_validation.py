
"""Input validation tests; no patient data or model checkpoints required."""

import numpy as np
import pytest

from src.dsp.input_validation import (
    STANDARD_LEADS,
    validate_ecg_input,
)


@pytest.fixture
def ecg():
    """A structurally valid synthetic 10-second, 100 Hz ECG."""
    return np.zeros((1000, 12), dtype=float)


def check(signals, *, fs=100, leads=STANDARD_LEADS, units="mV"):
    """Short helper to keep the test cases readable."""
    return validate_ecg_input(signals, fs, leads, units)


def test_normal_input_is_not_modified(ecg):
    ecg[:, 0] = 2.5
    original = ecg.copy()

    result, names = check(ecg)

    assert result is not ecg
    assert result.shape == (1000, 12)
    assert names == list(STANDARD_LEADS)

    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(ecg, original)


def test_reversed_lead_order_reorders_actual_waveforms():
    # Give every lead a unique numeric marker.
    source = np.tile(
        np.arange(12, dtype=float), (1000, 1)
    )

    reversed_source = source[:, ::-1]
    reversed_labels = list(STANDARD_LEADS)[::-1]

    result, names = check(
        reversed_source, leads=reversed_labels
    )

    np.testing.assert_array_equal(result, source)
    assert names == list(STANDARD_LEADS)


def test_wfdb_units_per_channel_are_converted(ecg):
    ecg[:, 0] = 1000.0  # 1000 microvolts
    ecg[:, 1] = 2.0     # 2 millivolts

    units = ["uV", "mV"] + ["mV"] * 10

    result, _ = check(ecg, units=units)

    np.testing.assert_allclose(result[:, 0], 1.0)
    np.testing.assert_allclose(result[:, 1], 2.0)


@pytest.mark.parametrize("unit,value", [
    ("mV", 1.0),
    ("uV", 1000.0),
    ("µV", 1000.0),
    ("μV", 1000.0),
    ("V", 0.001),
])
def test_supported_units_convert_to_mv(unit, value):
    source = np.full((1000, 12), value)

    result, _ = check(source, units=unit)

    np.testing.assert_allclose(result, 1.0)


@pytest.mark.parametrize("fs", [
    0, -100, 80, np.nan, np.inf, "100", True,
])
def test_reject_bad_sample_rate(ecg, fs):
    with pytest.raises(ValueError, match="fs"):
        check(ecg, fs=fs)


@pytest.mark.parametrize("shape", [
    (1000,),
    (1000, 11),
    (1000, 12, 1),
    (1, 12),
])
def test_reject_bad_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        check(np.zeros(shape))


def test_reject_missing_or_duplicate_leads(ecg):
    with pytest.raises(ValueError, match="lead_names"):
        check(ecg, leads=None)

    names = list(STANDARD_LEADS)
    names[0] = "II"

    with pytest.raises(ValueError, match="Duplicate"):
        check(ecg, leads=names)


def test_reject_wrong_lead_set(ecg):
    names = list(STANDARD_LEADS)
    names[-1] = "V7"

    with pytest.raises(ValueError, match="missing"):
        check(ecg, leads=names)


def test_reject_nan_and_infinity(ecg):
    for value in [np.nan, np.inf, -np.inf]:
        corrupted = ecg.copy()
        corrupted[0, 0] = value

        with pytest.raises(ValueError, match="NaN or infinity"):
            check(corrupted)


def test_reject_unknown_units(ecg):
    with pytest.raises(ValueError, match="units"):
        check(ecg, units="counts")


def test_reject_non_numeric_data():
    with pytest.raises(ValueError, match="real numbers"):
        check(np.full((1000, 12), "1"))

    with pytest.raises(ValueError, match="real numbers"):
        check(np.full((1000, 12), 1 + 2j))


def test_units_follow_original_channels_when_reordered():
    # The first original channel is in microvolts, while the others
    # are in millivolts. Conversion must happen before reordering.
    original = np.ones((1000, 12), dtype=float)
    original[:, 0] = 1000.0

    labels = list(STANDARD_LEADS)[::-1]
    units = ["uV"] + ["mV"] * 11

    result, _ = check(
        original, leads=labels, units=units
    )

    np.testing.assert_allclose(result, 1.0)


def test_common_avr_capitalization_is_supported(ecg):
    labels = list(STANDARD_LEADS)
    labels[3:6] = ["aVR", "aVL", "aVF"]

    _, normalized = check(ecg, leads=labels)

    assert normalized == list(STANDARD_LEADS)
