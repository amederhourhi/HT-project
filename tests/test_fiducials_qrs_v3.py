
import numpy as np
import pytest

from src.features.fiducials_qrs_v3 import (
    analyze_12lead_features,
)


def test_reject_nonfinite():
    a = np.zeros((1000, 12))
    a[2, 0] = np.nan

    assert analyze_12lead_features(
        a, [300, 700], ["x"] * 12, 100
    )["status"] == "rejected"


def test_reject_duplicate_peaks():
    a = np.zeros((1000, 12))

    assert analyze_12lead_features(
        a, [400, 400], ["x"] * 12, 100
    )["status"] == "rejected"


def test_unsupported_rate():
    with pytest.raises(ValueError):
        analyze_12lead_features(
            np.zeros((1000, 12)),
            [100, 500],
            ["x"] * 12,
            250,
        )


def test_zero_signal_reject():
    result = analyze_12lead_features(
        np.zeros((1000, 12)),
        [300, 700],
        ["x"] * 12,
        100,
    )

    assert result["status"] == "rejected"
