
"""Tests for matched V1/V3 validation and coverage accounting."""

import pandas as pd
import pytest

from scripts.compare_v1_v3_rf import (
    paired_validation,
    validation_coverage,
)


def make_frame(ids, labels, patients=None):
    if patients is None:
        patients = ids

    return pd.DataFrame({
        "ecg_id": ids,
        "patient_id": patients,
        "target_label": labels,
    })


def test_pairing_aligns_ecg_ids():
    v1 = make_frame([3, 1, 2], [0, 0, 1])
    v3 = make_frame([2, 3, 4, 1], [1, 0, 1, 0])

    left, right = paired_validation(v1, v3)

    assert list(left.index) == [1, 2, 3]
    assert list(right.index) == [1, 2, 3]


def test_pairing_rejects_label_mismatch():
    v1 = make_frame([1, 2], [0, 1])
    v3 = make_frame([1, 2], [1, 0])

    with pytest.raises(ValueError, match="labels differ"):
        paired_validation(v1, v3)


def test_pairing_rejects_patient_mismatch():
    v1 = make_frame([1, 2], [0, 1], [10, 20])
    v3 = make_frame([1, 2], [0, 1], [10, 99])

    with pytest.raises(ValueError, match="patient IDs differ"):
        paired_validation(v1, v3)


def test_pairing_rejects_duplicate_ids():
    v1 = make_frame([1, 1, 2], [0, 0, 1])
    v3 = make_frame([1, 2], [0, 1])

    with pytest.raises(ValueError, match="IDs are invalid"):
        paired_validation(v1, v3)


def test_pairing_rejects_no_overlap():
    v1 = make_frame([1, 2], [0, 1])
    v3 = make_frame([3, 4], [0, 1])

    with pytest.raises(ValueError, match="No shared"):
        paired_validation(v1, v3)


def test_pairing_requires_both_classes():
    v1 = make_frame([1, 2, 3], [0, 0, 1])
    v3 = make_frame([1, 2, 4], [0, 0, 1])

    with pytest.raises(ValueError, match="both classes"):
        paired_validation(v1, v3)


def test_coverage_counts_selection_by_label():
    manifest = pd.DataFrame({
        "ecg_id": [1, 2, 3, 4, 5, 6],
        "strat_fold": [9, 9, 9, 9, 9, 10],
        "target_label": [0, 0, 1, 1, 1, 1],
    })

    v1 = make_frame([1, 3, 4], [0, 1, 1])
    v3 = make_frame([1, 2, 3, 5], [0, 0, 1, 1])

    coverage = validation_coverage(manifest, v1, v3)

    assert coverage[0]["manifest_records"] == 2
    assert coverage[0]["v1_accepted"] == 1
    assert coverage[0]["v3_eligible"] == 2

    assert coverage[1]["manifest_records"] == 3
    assert coverage[1]["paired_records"] == 1


def test_coverage_rejects_unknown_selection():
    manifest = pd.DataFrame({
        "ecg_id": [1, 2],
        "strat_fold": [9, 9],
        "target_label": [0, 1],
    })

    v1 = make_frame([1, 99], [0, 1])
    v3 = make_frame([1, 2], [0, 1])

    with pytest.raises(ValueError, match="inconsistent"):
        validation_coverage(manifest, v1, v3)
