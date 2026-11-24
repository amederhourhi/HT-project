
"""Synthetic regression tests for fold-9 selection and coverage."""

from copy import deepcopy

import pandas as pd
import pytest

from scripts.audit_validation_coverage import (
    summarize_validation_coverage,
)


@pytest.fixture
def sample():
    """Provide artificial train, validation and test records."""

    # Seven records belong to fold 9.
    # Fold 10 and training records must not enter evaluation.
    pin = pd.DataFrame({
        "ecg_id": [
            11, 12, 13, 14, 15, 16, 19, 17, 18,
        ],
        "strat_fold": [
            9, 9, 9, 9, 9, 9, 9, 10, 1,
        ],
        "target_label": [
            0, 1, 0, 1, 0, 1, 1, 1, 0,
        ],
    })

    v1 = pin.loc[
        pin.ecg_id.isin(
            [11, 12, 13, 14, 17, 18]
        )
    ].copy()

    v3 = pin.loc[
        pin.ecg_id.isin(
            [11, 12, 15, 16, 17, 18]
        )
    ].copy()

    pairs = pd.DataFrame({
        "ecg_id": [11, 12],
        "target_label": [0, 1],
        "v1_prediction": [0, 0],
        "v3_prediction": [0, 1],
    })

    report = {
        "evaluation_fold": 9,
        "test_fold_evaluated": False,
        "coverage_by_label": [
            {
                "target_label": 0,
                "manifest_records": 3,
                "v1_accepted": 2,
                "v3_eligible": 2,
                "paired_records": 1,
                "v1_coverage": 2 / 3,
                "v3_coverage": 2 / 3,
            },
            {
                "target_label": 1,
                "manifest_records": 4,
                "v1_accepted": 2,
                "v3_eligible": 2,
                "paired_records": 1,
                "v1_coverage": 0.5,
                "v3_coverage": 0.5,
            },
        ],
    }

    return [pin, v1, v3, pairs, report]


def run(sample):
    return summarize_validation_coverage(*sample)


def test_expected_coverage_and_matched_accuracy(sample):
    result = run(sample)

    assert result["manifest_records"] == 7

    assert result["availability_overlap"] == {
        "both": 2,
        "v1_only": 2,
        "v3_only": 2,
        "neither": 1,
        "paired_fraction_of_fold": 2 / 7,
    }

    assert result["feature_availability"][
        "v3_eligible_rows"
    ] == 4

    assert result["matched_predictions_only"][
        "v1_correct"
    ] == 1

    assert result["matched_predictions_only"][
        "v3_correct"
    ] == 2

    assert result["test_fold_evaluated"] is False


def test_input_is_not_modified(sample):
    original = deepcopy(sample)

    run(sample)

    for actual, expected in zip(
        sample[:4],
        original[:4],
    ):
        pd.testing.assert_frame_equal(
            actual,
            expected,
        )

    assert sample[4] == original[4]


def test_frozen_coverage_mismatch_blocks(sample):
    sample[4]["coverage_by_label"][0][
        "v3_eligible"
    ] = 1

    with pytest.raises(
        ValueError,
        match="Frozen coverage mismatch",
    ):
        run(sample)


def test_duplicate_cohort_id_blocks(sample):
    sample[0].loc[1, "ecg_id"] = 11

    with pytest.raises(
        ValueError,
        match="duplicate ECG identifiers",
    ):
        run(sample)


def test_unknown_v3_record_blocks(sample):
    sample[2].loc[0, "ecg_id"] = 999

    with pytest.raises(
        ValueError,
        match="outside pinned cohort",
    ):
        run(sample)


def test_v3_label_mismatch_blocks(sample):
    sample[2].loc[
        sample[2].ecg_id == 12,
        "target_label",
    ] = 0

    with pytest.raises(
        ValueError,
        match="target_label disagrees",
    ):
        run(sample)


def test_v1_fold_mismatch_blocks(sample):
    sample[1].loc[
        sample[1].ecg_id == 12,
        "strat_fold",
    ] = 10

    with pytest.raises(
        ValueError,
        match="strat_fold disagrees",
    ):
        run(sample)


def test_paired_intersection_is_required(sample):
    sample[3] = sample[3].iloc[:1].copy()

    with pytest.raises(
        ValueError,
        match="Paired prediction IDs differ",
    ):
        run(sample)


def test_fold10_cannot_enter_predictions(sample):
    sample[3].loc[len(sample[3])] = [
        17, 1, 1, 1,
    ]

    with pytest.raises(
        ValueError,
        match="Paired prediction IDs differ",
    ):
        run(sample)


def test_paired_labels_must_match_pin(sample):
    sample[3].loc[0, "target_label"] = 1

    with pytest.raises(
        ValueError,
        match="Paired labels disagree",
    ):
        run(sample)


def test_saved_predictions_must_be_binary(sample):
    sample[3].loc[0, "v3_prediction"] = 2

    with pytest.raises(
        ValueError,
        match="Invalid saved predicted class",
    ):
        run(sample)


def test_fold10_evaluation_flag_blocks(sample):
    sample[4]["test_fold_evaluated"] = True

    with pytest.raises(
        ValueError,
        match="Fold-10 reservation",
    ):
        run(sample)