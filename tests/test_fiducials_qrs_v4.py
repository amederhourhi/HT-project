
import numpy as np
import pytest

from src.features.fiducials_qrs_v4 import (
    choose_offset,
    analyze_12lead_features,
)


def test_isolated_late_offset_is_excluded():
    offsets = list(enumerate([
        220, 221, 222, 220, 221, 251
    ]))

    selected, lead, gap = choose_offset(offsets)

    assert selected == 222
    assert lead == 5
    assert gap == 58.0


def test_supported_late_offset_kept():
    offsets = list(enumerate([
        220, 221, 222, 220, 250, 251
    ]))

    selected, lead, _ = choose_offset(offsets)

    assert selected == 251
    assert lead is None


def test_too_few_leads_do_not_trim():
    selected, lead, _ = choose_offset([
        (0, 200),
        (1, 250),
    ])

    assert selected == 250
    assert lead is None


def test_reject_nan():
    a = np.zeros((1000, 12))
    a[0, 0] = np.nan

    result = analyze_12lead_features(
        a, [300, 700], ["x"] * 12, 100
    )

    assert result["status"] == "rejected"


def test_reject_duplicate_peaks():
    result = analyze_12lead_features(
        np.zeros((1000, 12)),
        [300, 300],
        ["x"] * 12,
        100,
    )

    assert result["status"] == "rejected"


def test_unsupported_rate():
    with pytest.raises(ValueError):
        analyze_12lead_features(
            np.zeros((1000, 12)),
            [300, 700],
            ["x"] * 12,
            250,
        )
