
"""Offline tests for the saved LUDB expert QRS reference audit."""

import numpy as np
import pandas as pd
import pytest

from scripts.audit_ludb_qrs_accuracy import (
    make_report,
    verify_holdout,
)


@pytest.fixture
def sample():
    """Artificial three-record data, not real LUDB measurements."""

    rows = []

    for rid, old_on, new_on in (
        (31, 10, 5),
        (32, -5, -10),
        (33, 8, 8),
    ):

        for method, onset in (
            ("current", old_on),
            ("qrs_v4", new_on),
        ):

            offset = -5

            rows.append({
                "record_id": rid,
                "method": method,
                "status": "success",
                "reason": "",
                "onset_error_ms": onset,
                "offset_error_ms": offset,
                "duration_error_ms": (
                    offset - onset
                ),
            })

    return pd.DataFrame(rows)


def audit(sample):
    return make_report(
        sample,
        expected_records=(31, 32, 33),
        draws=100,
        seed=42,
    )


def test_valid_paired_audit(sample):
    result = audit(sample)

    assert result["evaluation_records"] == 3

    paired = result["metrics"][
        "onset_error_ms"
    ]["paired"]

    assert (
        paired["improved"],
        paired["worsened"],
        paired["unchanged"],
    ) == (1, 1, 1)

    assert result[
        "reserved_ludb_101_to_200_evaluated"
    ] is False

    assert result[
        "ptbxl_fold10_evaluated"
    ] is False


def test_read_only(sample):
    original = sample.copy(deep=True)

    audit(sample)

    pd.testing.assert_frame_equal(
        sample,
        original,
    )


def test_bootstrap_deterministic(sample):
    assert audit(sample) == audit(sample)


def test_reject_duplicate_pair(sample):
    sample = pd.concat(
        [
            sample,
            sample.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        audit(sample)


def test_reject_missing_method(sample):
    sample = sample.drop(index=0)

    with pytest.raises(
        ValueError,
        match="Incomplete method",
    ):
        audit(sample)


def test_reject_unexpected_record(sample):
    sample.loc[0, "record_id"] = 101

    with pytest.raises(
        ValueError,
        match="Unexpected or missing",
    ):
        audit(sample)


def test_reject_unknown_method(sample):
    sample.loc[0, "method"] = "experimental"

    with pytest.raises(
        ValueError,
        match="Unexpected QRS method",
    ):
        audit(sample)


def test_reject_failed_record(sample):
    sample.loc[0, "status"] = "failed"

    with pytest.raises(
        ValueError,
        match="Incomplete held-out",
    ):
        audit(sample)


def test_reject_nonfinite(sample):
    sample.loc[0, "onset_error_ms"] = np.nan

    with pytest.raises(
        ValueError,
        match="Nonfinite",
    ):
        audit(sample)


def test_reject_inconsistent_duration(sample):
    sample.loc[0, "duration_error_ms"] += 5

    with pytest.raises(
        ValueError,
        match="identity mismatch",
    ):
        audit(sample)


def test_reject_non_integer_record(sample):
    sample["record_id"] = (
        sample["record_id"].astype(float)
    )

    sample.loc[0, "record_id"] = 31.5

    with pytest.raises(
        ValueError,
        match="Invalid LUDB record IDs",
    ):
        audit(sample)


def test_reject_nonpositive_draws(sample):
    with pytest.raises(
        ValueError,
        match="Bootstrap draws",
    ):
        make_report(
            sample,
            expected_records=(31, 32, 33),
            draws=0,
        )


def test_reject_missing_column(sample):
    with pytest.raises(
        ValueError,
        match="Missing required",
    ):
        verify_holdout(
            sample.drop(
                columns=["offset_error_ms"]
            ),
            expected_records=(31, 32, 33),
        )