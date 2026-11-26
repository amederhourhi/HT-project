
"""Data-free tests for Phase II paired evaluation protocol."""

from copy import deepcopy

import pytest

from scripts.compare_rpeak_refinement import (
    score_annotated_window,
)

from scripts.audit_rpeak_pairwise_protocol import (
    METHODS,
    RATES,
    TOLERANCES_MS,
    paired_f1_bootstrap,
    summarize_cohort,
    validate_cohort,
)


def make_rows():
    """Build paired synthetic scores for two records."""

    result = []

    for rid in (1, 2):
        for fs in RATES:

            scale = fs // 100

            reference = [
                100 * scale,
                200 * scale,
            ]

            for method in METHODS:

                if method == "current" and rid == 1:

                    # One baseline peak is 70 ms late.
                    detected = [
                        0,
                        100 * scale,
                        207 * scale,
                        300 * scale,
                    ]

                else:
                    detected = [
                        0,
                        100 * scale,
                        201 * scale,
                        300 * scale,
                    ]

                for tolerance in TOLERANCES_MS:

                    score = score_annotated_window(
                        reference,
                        detected,
                        fs,
                        tolerance,
                    )

                    result.append({
                        "record_id": rid,
                        "fs_hz": fs,
                        "tolerance_ms": tolerance,
                        "method": method,
                        **score,
                    })

    return result


def test_complete_paired_cohort():

    report = summarize_cohort(
        make_rows(),
        (1, 2),
        replicates=50,
    )

    assert len(report) == 4

    result = next(
        row for row in report
        if (
            row["fs_hz"] == 100
            and row["tolerance_ms"] == 50.0
        )
    )

    assert result["baseline"]["tp"] == 3
    assert result["candidate"]["tp"] == 4

    assert result["baseline"]["unscored"] == 4
    assert result["candidate"]["unscored"] == 4

    assert result["delta_f1"] > 0
    assert result["regression_records"] == []


def test_large_tolerance_recovers_mislocalized_peak():

    report = summarize_cohort(
        make_rows(),
        (1, 2),
        replicates=10,
    )

    result = next(
        row for row in report
        if (
            row["fs_hz"] == 100
            and row["tolerance_ms"] == 100.0
        )
    )

    assert result["baseline"]["tp"] == 4
    assert result["candidate"]["tp"] == 4
    assert result["delta_f1"] == pytest.approx(0.0)


def test_missing_record_rejected():

    rows = [
        row for row in make_rows()
        if row["record_id"] != 2
    ]

    with pytest.raises(
        ValueError,
        match="Missing score rows",
    ):
        validate_cohort(rows, (1, 2))


def test_duplicate_pair_rejected():

    rows = make_rows()

    with pytest.raises(
        ValueError,
        match="Duplicate score row",
    ):
        validate_cohort(
            rows + [rows[0]],
            (1, 2),
        )


def test_unexpected_record_rejected():

    rows = make_rows()

    rows[0] = {
        **rows[0],
        "record_id": 101,
    }

    with pytest.raises(
        ValueError,
        match="Unexpected score row",
    ):
        validate_cohort(rows, (1, 2))


def test_bad_denominator_rejected():

    rows = deepcopy(make_rows())

    rows[0]["fn"] += 1

    with pytest.raises(
        ValueError,
        match="Inconsistent event counts",
    ):
        validate_cohort(rows, (1, 2))


def test_unscored_edges_must_agree_between_tolerances():

    rows = deepcopy(make_rows())

    bad = next(
        row for row in rows
        if (
            row["record_id"],
            row["fs_hz"],
            row["tolerance_ms"],
            row["method"],
        ) == (1, 100, 100.0, "current")
    )

    bad["unscored_outside_window"] += 1

    with pytest.raises(
        ValueError,
        match="Tolerance inconsistency",
    ):
        validate_cohort(rows, (1, 2))


def test_bootstrap_is_seeded_and_record_paired():

    ids = (1, 2)

    indexed = validate_cohort(
        make_rows(),
        ids,
    )

    first = paired_f1_bootstrap(
        indexed,
        ids,
        100,
        50.0,
        replicates=200,
        seed=7,
    )

    second = paired_f1_bootstrap(
        indexed,
        ids,
        100,
        50.0,
        replicates=200,
        seed=7,
    )

    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_invalid_bootstrap_count_rejected():

    indexed = validate_cohort(
        make_rows(),
        (1, 2),
    )

    with pytest.raises(
        ValueError,
        match="positive bootstrap",
    ):
        paired_f1_bootstrap(
            indexed,
            (1, 2),
            100,
            50.0,
            replicates=0,
        )


def test_record_specific_regression_is_reported():

    rows = make_rows()

    # Reverse the two algorithms for record 1.
    # The candidate should then contain the strict miss.
    for fs in RATES:
        for tolerance in TOLERANCES_MS:

            before = next(
                row for row in rows
                if (
                    row["record_id"],
                    row["fs_hz"],
                    row["tolerance_ms"],
                    row["method"],
                ) == (1, fs, tolerance, "current")
            )

            after = next(
                row for row in rows
                if (
                    row["record_id"],
                    row["fs_hz"],
                    row["tolerance_ms"],
                    row["method"],
                ) == (
                    1,
                    fs,
                    tolerance,
                    "asymmetric_50_20",
                )
            )

            fields = (
                "tp",
                "fp",
                "fn",
                "matched_timing_errors_ms",
                "precision",
                "recall",
                "f1",
            )

            for field in fields:
                before[field], after[field] = (
                    after[field],
                    before[field],
                )

    report = summarize_cohort(
        rows,
        (1, 2),
        replicates=10,
    )

    strict = next(
        row for row in report
        if (
            row["fs_hz"] == 100
            and row["tolerance_ms"] == 50.0
        )
    )

    assert strict["regression_records"] == [1]