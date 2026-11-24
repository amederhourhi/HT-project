
"""Synthetic offline tests for Milestone 9 evidence reconciliation."""

import numpy as np
import pandas as pd
import pytest

from scripts.audit_m09_readiness import analyze_failure_cases


@pytest.fixture
def source():
    """Artificial values illustrating the saved boundary-error arithmetic."""

    vals = {
        31: (
            (-8, -18, -10),
            (26, -18, -44),
        ),
        72: (
            (-6, -8, -2),
            (-36, -8, 28),
        ),
        78: (
            (12, 18, 6),
            (12, -6, -18),
        ),
        85: (
            (4, -8, -12),
            (-26, -8, 18),
        ),
    }

    main = []
    offset = []

    for rid, (current, v4) in vals.items():

        for method, errors in (
            ("current", current),
            ("qrs_v4", v4),
        ):
            main.append({
                "record_id": rid,
                "method": method,
                "status": "success",
                "reason": "",
                "onset_error_ms": errors[0],
                "offset_error_ms": errors[1],
                "duration_error_ms": errors[2],
            })

        offset.append({
            "record_id": rid,
            "method": "offset_only_counterfactual",
            "onset_error_ms": current[0],
            "offset_error_ms": v4[1],
            "duration_error_ms": (
                v4[1] - current[0]
            ),
        })

    notes = pd.DataFrame(
        [
            (31, -54, 8, -76, -42, 34, 9),
            (72, -58, 2.5, -68, -98, -30, 8),
            (85, -37, 5.5, -36, -66, -30, 10),

            # Development metadata, not an evaluated
            # holdout outcome.
            (20, -59, 2.5, -50, -62, -12, 8),
        ],

        columns=[
            "record_id",
            "expert_lead_ii_median_onset_ms",
            "expert_lead_ii_onset_iqr_ms",
            "original_global_onset_ms",
            "v4_global_onset_ms",
            "v4_minus_original_ms",
            "expert_beats",
        ],
    )

    return (
        pd.DataFrame(main),
        pd.DataFrame(offset),
        notes,
    )


def audit(source):
    return analyze_failure_cases(
        *source,
        expected_records=(31, 72, 78, 85),
        focus=(31, 72, 78, 85),
        draws=20,
        seed=42,
    )


def test_case_mechanisms(source):
    report = audit(source)

    cases = {
        c["record_id"]: c
        for c in report["focus_cases"]
    }

    assert cases[31][
        "observed_mechanism"
    ] == "onset_shift_only"

    assert cases[72][
        "observed_mechanism"
    ] == "onset_shift_only"

    assert cases[78][
        "observed_mechanism"
    ] == "offset_shift_only"

    assert cases[31][
        "absolute_duration_error_increase_ms"
    ] == 34

    assert cases[72][
        "absolute_duration_error_increase_ms"
    ] == 26

    assert cases[78][
        "absolute_duration_error_increase_ms"
    ] == 12

    assert cases[31][
        "lead_ii_reference"
    ]["reference_dependent_ranking"] is True

    assert cases[78][
        "lead_ii_reference"
    ] is None

    assert report[
        "development_only_onset_note_ids"
    ] == [20]

    assert report[
        "research_readiness"
    ]["qrs_v4_promotion"] == "blocked"

    assert report[
        "reserved_data_read_by_this_audit"
    ]["ludb_101_200"] is False


def test_deterministic(source):
    assert audit(source) == audit(source)


def test_no_mutation(source):
    originals = [
        x.copy(deep=True)
        for x in source
    ]

    audit(source)

    for old, new in zip(originals, source):
        pd.testing.assert_frame_equal(
            old,
            new,
        )


def test_bad_counterfactual_is_rejected(source):
    source[1].loc[
        0,
        "onset_error_ms",
    ] += 1

    with pytest.raises(
        ValueError,
        match="Counterfactual: inconsistent onset",
    ):
        audit(source)


def test_missing_counterfactual_record_is_rejected(source):
    source[1].drop(
        index=0,
        inplace=True,
    )

    with pytest.raises(
        ValueError,
        match="Counterfactual: missing or reserved",
    ):
        audit(source)


def test_duplicate_counterfactual_record_is_rejected(source):
    source[1].loc[
        1,
        "record_id",
    ] = 31

    with pytest.raises(
        ValueError,
        match="Counterfactual: duplicate",
    ):
        audit(source)


def test_bad_onset_shift_is_rejected(source):
    source[2].loc[
        0,
        "v4_minus_original_ms",
    ] = 1

    with pytest.raises(
        ValueError,
        match="inconsistent onset displacement",
    ):
        audit(source)


def test_bad_reference_is_rejected(source):
    source[2].loc[
        0,
        "original_global_onset_ms",
    ] += 2

    source[2].loc[
        0,
        "v4_minus_original_ms",
    ] -= 2

    with pytest.raises(
        ValueError,
        match="reference mismatch",
    ):
        audit(source)


def test_reserved_note_is_rejected(source):
    source[2].loc[
        0,
        "record_id",
    ] = 101

    with pytest.raises(
        ValueError,
        match="unexpected or reserved",
    ):
        audit(source)


def test_duplicate_note_is_rejected(source):
    source[2].loc[
        1,
        "record_id",
    ] = 31

    with pytest.raises(
        ValueError,
        match="duplicate record IDs",
    ):
        audit(source)


def test_negative_iqr_is_rejected(source):
    source[2].loc[
        0,
        "expert_lead_ii_onset_iqr_ms",
    ] = -1

    with pytest.raises(
        ValueError,
        match="negative onset IQR",
    ):
        audit(source)


def test_nonfinite_holdout_error_is_rejected(source):
    source[0].loc[
        0,
        "onset_error_ms",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="Nonfinite",
    ):
        audit(source)


def test_nonfinite_counterfactual_is_rejected(source):
    source[1].loc[
        0,
        "duration_error_ms",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="nonfinite duration_error_ms",
    ):
        audit(source)


def test_invalid_focus_is_rejected(source):
    with pytest.raises(
        ValueError,
        match="Invalid focused",
    ):
        analyze_failure_cases(
            *source,
            expected_records=(31, 72, 78, 85),
            focus=(31, 101),
            draws=20,
        )