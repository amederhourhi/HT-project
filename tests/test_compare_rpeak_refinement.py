
"""Offline tests for experimental R-peak refinement."""

import numpy as np
import pytest

from src.features.r_peaks import detect_r_peaks

from scripts.compare_rpeak_refinement import (
    snap_candidate,
    detect_snap_variant,
    score_annotated_window,
    eval_record,
)


def test_asymmetric_window_excludes_later_s_trough():
    wave = np.zeros(220)

    wave[100] = 0.4
    wave[160] = -0.45

    # Symmetric window selects the stronger S trough.
    assert snap_candidate(
        wave,
        115,
        1000,
        back_ms=50,
        forward_ms=50,
    ) == 160

    # Asymmetric window retains the earlier R peak.
    assert snap_candidate(
        wave,
        115,
        1000,
        back_ms=50,
        forward_ms=20,
    ) == 100


def test_asymmetric_window_can_retain_negative_peak():
    wave = np.zeros(220)

    wave[100] = -0.5
    wave[160] = 0.6

    assert snap_candidate(
        wave,
        115,
        1000,
        back_ms=50,
        forward_ms=20,
    ) == 100


def test_symmetric_variant_reproduces_baseline():
    wave = np.zeros(1000)

    for index in (200, 550, 850):
        wave[index] = 1.0
        wave[index + 20] = -1.1

    for fs in (100, 500):
        expected = detect_r_peaks(
            wave,
            fs=fs,
        )

        reproduced = detect_snap_variant(
            wave,
            fs,
            back_ms=50,
            forward_ms=50,
        )

        np.testing.assert_array_equal(
            expected,
            reproduced,
        )


def test_scoring_excludes_unannotated_edges():
    result = score_annotated_window(
        [100, 200],
        [0, 100, 200, 300],
        100,
        100,
    )

    assert (
        result["tp"],
        result["fp"],
        result["fn"],
        result["unscored_outside_window"],
    ) == (2, 0, 0, 2)


def test_strict_scoring_counts_wrong_localization():
    result = score_annotated_window(
        [100],
        [107],
        100,
        50,
    )

    assert (
        result["tp"],
        result["fp"],
        result["fn"],
        result["unscored_outside_window"],
    ) == (0, 1, 1, 0)


def test_duplicate_inside_window_counts_as_fp():
    result = score_annotated_window(
        [100],
        [95, 100],
        100,
        50,
    )

    assert (
        result["tp"],
        result["fp"],
        result["fn"],
    ) == (1, 1, 0)


def test_reserved_records_rejected():
    with pytest.raises(
        ValueError,
        match="development records 1-30",
    ):
        eval_record(31)


def test_invalid_signal_rejected():
    with pytest.raises(
        ValueError,
        match="finite 1D",
    ):
        detect_snap_variant(
            [0.0, float("nan"), 1.0],
            100,
        )