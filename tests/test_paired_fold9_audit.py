
"""Offline synthetic tests for the paired fold-9 audit."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.audit_paired_fold9 import (
    load_verified_pairs,
    paired_summary,
)
from scripts.train_v3_rf_candidate import (
    calculate_metrics,
    sha256_file,
)
from src.features.extract_features import FEATURE_COLUMNS


@pytest.fixture
def files(tmp_path):
    """Create artificial frozen artifacts for regression testing."""

    pin = pd.DataFrame({
        "ecg_id": [11, 12, 13, 14, 15, 16],
        "strat_fold": [9, 9, 9, 9, 1, 10],
        "target_label": [0, 1, 0, 1, 0, 1],
    })

    v1 = pin.copy()
    v3 = pin.copy()

    v3["patient_id"] = [
        101, 102, 103, 104, 105, 106,
    ]

    pairs = pd.DataFrame({
        "ecg_id": [11, 12, 13, 14],
        "patient_id": [101, 102, 103, 104],
        "target_label": [0, 1, 0, 1],
        "v1_probability": [0.1, 0.2, 0.6, 0.9],
        "v3_probability": [0.05, 0.8, 0.7, 0.7],
        "v1_prediction": [0, 0, 1, 1],
        "v3_prediction": [0, 1, 1, 1],
    })

    paths = {
        key: tmp_path / f"{key}.csv"
        for key in ("pairs", "pin", "v1", "v3")
    }

    for key, data in (
        ("pairs", pairs),
        ("pin", pin),
        ("v1", v1),
        ("v3", v3),
    ):
        data.to_csv(paths[key], index=False)

    report = {
        "contract_version": "v1_v3_matched_fold9_v1",
        "evaluation_fold": 9,
        "test_fold_evaluated": False,
        "feature_order": FEATURE_COLUMNS,

        "artifacts": {
            "paired_predictions_sha256": sha256_file(
                paths["pairs"]
            ),
            "v1_dataset_sha256": sha256_file(
                paths["v1"]
            ),
            "v3_dataset_sha256": sha256_file(
                paths["v3"]
            ),
        },

        "validation_matched": {
            name: calculate_metrics(
                pairs["target_label"],
                pairs[f"{name}_probability"],
            )
            for name in ("v1", "v3")
        },
    }

    paths["report"] = tmp_path / "report.json"

    paths["report"].write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    return paths


def verify(files):
    """Invoke the complete audit using synthetic files."""

    return load_verified_pairs(
        files["pairs"],
        files["report"],
        files["pin"],
        files["v1"],
        files["v3"],
    )


def change_cell(
    files,
    key,
    row,
    column,
    value,
    *,
    refresh_hash=True,
):
    """Modify a synthetic fixture to simulate corruption."""

    frame = pd.read_csv(files[key])
    frame.loc[row, column] = value
    frame.to_csv(files[key], index=False)

    if refresh_hash and key != "pin":
        report = json.loads(
            files["report"].read_text(encoding="utf-8")
        )

        hash_key = {
            "pairs": "paired_predictions_sha256",
            "v1": "v1_dataset_sha256",
            "v3": "v3_dataset_sha256",
        }[key]

        report["artifacts"][hash_key] = sha256_file(
            files[key]
        )

        files["report"].write_text(
            json.dumps(report),
            encoding="utf-8",
        )


def test_clean_pairs_reproduce_frozen_metrics(files):
    pairs, _ = verify(files)

    assert len(pairs) == 4

    result = paired_summary(
        pairs,
        draws=30,
        seed=42,
    )

    assert result["matched_ecgs"] == 4
    assert result["paired_decisions"][
        "v3_correct_v1_wrong"
    ] == 1
    assert result["paired_decisions"][
        "v1_correct_v3_wrong"
    ] == 0
    assert result["test_fold_evaluated"] is False


def test_predictions_hash_tamper_blocks(files):
    change_cell(
        files,
        "pairs",
        0,
        "v3_probability",
        0.9,
        refresh_hash=False,
    )

    with pytest.raises(
        ValueError,
        match="Provenance hash mismatch",
    ):
        verify(files)


def test_frozen_test_fold_must_remain_unevaluated(files):
    report = json.loads(
        files["report"].read_text(encoding="utf-8")
    )

    report["test_fold_evaluated"] = True

    files["report"].write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="unsafe paired evaluation",
    ):
        verify(files)


def test_incorrect_patient_id_blocks(files):
    change_cell(
        files,
        "pairs",
        0,
        "patient_id",
        999,
    )

    with pytest.raises(
        ValueError,
        match="Patient IDs differ",
    ):
        verify(files)


def test_falsified_predicted_label_blocks(files):
    change_cell(
        files,
        "pairs",
        0,
        "v3_prediction",
        1,
    )

    with pytest.raises(
        ValueError,
        match="Predicted classes",
    ):
        verify(files)


def test_non_fold9_record_blocks(files):
    change_cell(
        files,
        "pin",
        0,
        "strat_fold",
        10,
    )

    with pytest.raises(
        ValueError,
        match="Non-fold-9",
    ):
        verify(files)


def test_v3_patient_partition_leakage_blocks(files):
    change_cell(
        files,
        "v3",
        4,
        "patient_id",
        101,
    )

    with pytest.raises(
        ValueError,
        match="patients overlap",
    ):
        verify(files)


def test_false_published_metric_blocks(files):
    report = json.loads(
        files["report"].read_text(encoding="utf-8")
    )

    report["validation_matched"]["v3"][
        "roc_auc"
    ] = 0.01

    files["report"].write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="metric mismatch",
    ):
        verify(files)


def test_nonfinite_probability_blocks(files):
    change_cell(
        files,
        "pairs",
        1,
        "v1_probability",
        np.nan,
    )

    with pytest.raises(
        ValueError,
        match="Invalid probabilities",
    ):
        verify(files)