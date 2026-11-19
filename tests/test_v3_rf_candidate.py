
"""Tests for V3 RF training and evaluation safeguards."""

import numpy as np
import pandas as pd
import pytest

from scripts.train_v3_rf_candidate import (
    FEATURE_COLUMNS,
    calculate_metrics,
    make_calibration_folds,
    split_name,
)


def test_partition_mapping():
    assert split_name(1) == "train"
    assert split_name(8) == "train"
    assert split_name(9) == "validation"
    assert split_name(10) == "test"


def test_invalid_partition_fails():
    with pytest.raises(ValueError):
        split_name(11)


def test_grouped_calibration_prevents_patient_leakage():
    # Two ECGs per patient, with balanced patient-level labels.
    patients = np.repeat(np.arange(60), 2)
    labels = np.repeat(np.arange(60) % 2, 2)

    features = pd.DataFrame(
        np.zeros((120, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    splits = make_calibration_folds(
        features,
        labels,
        patients,
    )

    assert len(splits) == 5

    for training, calibration in splits:
        assert not (
            set(patients[training]) & set(patients[calibration])
        )


def test_metrics_perfect_predictions():
    metrics = calculate_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_metrics_wrong_predictions():
    metrics = calculate_metrics(
        [0, 0, 1, 1],
        [0.8, 0.9, 0.1, 0.2],
    )

    assert metrics["accuracy"] == 0.0
    assert metrics["sensitivity"] == 0.0
    assert metrics["specificity"] == 0.0


def test_metrics_reject_single_class():
    with pytest.raises(ValueError):
        calculate_metrics([0, 0], [0.1, 0.2])


@pytest.mark.parametrize("invalid", [
    float("nan"),
    float("inf"),
    -0.1,
    1.1,
])
def test_metrics_reject_invalid_probabilities(invalid):
    with pytest.raises(ValueError):
        calculate_metrics([0, 1], [0.2, invalid])
