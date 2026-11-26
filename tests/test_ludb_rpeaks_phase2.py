
"""Offline deterministic tests for expert R-peak one-to-one matching."""

import pytest

from scripts.benchmark_ludb_rpeaks_phase2 import (
    match_peaks,
    summarize,
    evaluate_record,
)


def test_perfect_matches():

    r = match_peaks(
        [100, 200, 300],
        [100, 200, 300],
        fs=100,
        tolerance_ms=50,
    )

    assert (
        r["tp"],
        r["fp"],
        r["fn"],
        r["f1"],
    ) == (3, 0, 0, 1)

    assert r["matched_timing_errors_ms"] == [
        0.0,
        0.0,
        0.0,
    ]


def test_extra_detection_counts_false_positive():

    r = match_peaks(
        [100, 300],
        [100, 200, 300],
        fs=100,
        tolerance_ms=50,
    )

    assert (
        r["tp"],
        r["fp"],
        r["fn"],
    ) == (2, 1, 0)


def test_unmatched_reference_counts_false_negative():

    r = match_peaks(
        [100, 200, 300],
        [100, 300],
        fs=100,
        tolerance_ms=50,
    )

    assert (
        r["tp"],
        r["fp"],
        r["fn"],
    ) == (2, 0, 1)


def test_empty_detections_are_not_excluded():

    r = match_peaks(
        [100, 200],
        [],
        fs=100,
        tolerance_ms=50,
    )

    assert (
        r["tp"],
        r["fp"],
        r["fn"],
        r["recall"],
    ) == (0, 0, 2, 0)


def test_chooses_nearest_one_to_one_assignment():

    r = match_peaks(
        [100, 110],
        [104, 106],
        fs=100,
        tolerance_ms=100,
    )

    assert r["tp"] == 2

    assert r["matched_timing_errors_ms"] == [
        40.0,
        -40.0,
    ]


def test_prevents_double_match():

    r = match_peaks(
        [100],
        [99, 101],
        fs=100,
        tolerance_ms=50,
    )

    assert (
        r["tp"],
        r["fp"],
        r["fn"],
    ) == (1, 1, 0)


def test_tolerance_boundary_inclusive():

    r = match_peaks(
        [100],
        [105],
        fs=100,
        tolerance_ms=50,
    )

    assert r["tp"] == 1


def test_tolerance_boundary_exclusive_beyond_limit():

    r = match_peaks(
        [100],
        [106],
        fs=100,
        tolerance_ms=50,
    )

    assert (
        r["tp"],
        r["fp"],
        r["fn"],
    ) == (0, 1, 1)


@pytest.mark.parametrize(
    "values",
    [
        [2, 1],
        [1, 1],
        [-1, 10],
        [float("nan")],
    ],
)
def test_invalid_reference_rejected(values):

    with pytest.raises(ValueError):

        match_peaks(
            values,
            [1],
            fs=100,
            tolerance_ms=50,
        )


def test_invalid_sampling_frequency_rejected():

    with pytest.raises(ValueError):

        match_peaks(
            [100],
            [100],
            fs=0,
            tolerance_ms=50,
        )


def test_aggregate_micro_average_includes_failed_record():

    a = match_peaks(
        [100, 200],
        [100, 200],
        fs=100,
        tolerance_ms=50,
    )

    b = match_peaks(
        [100, 200],
        [],
        fs=100,
        tolerance_ms=50,
    )

    result = summarize([a, b])

    assert (
        result["tp"],
        result["fp"],
        result["fn"],
    ) == (2, 0, 2)

    assert result["recall"] == 0.5

    assert (
        result["records_with_missed_beats"]
        == 1
    )


def test_empty_summary_rejected():

    with pytest.raises(ValueError):
        summarize([])


def test_reserved_record_rejected_before_loading():

    with pytest.raises(
        ValueError,
        match="Only LUDB development records 1-30",
    ):
        evaluate_record(31)