
"""Tests for paired-decision and patient-bootstrap analysis."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from scripts.sqi_ablation_uncertainty import (
    load_verified_inputs,
    paired_decisions,
    patient_bootstrap,
    sha256,
)


def test_paired_decisions_count_changes():
    result = paired_decisions(
        [0, 0, 1, 1],
        [0.8, 0.2, 0.4, 0.7],
        [0.1, 0.9, 0.6, 0.7],
    )

    assert result["v3_correct_v1_wrong"] == 2
    assert result["v1_correct_v3_wrong"] == 1
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 0
    assert result["changed_predicted_classes"] == 3


def test_identical_decisions_have_p_one():
    result = paired_decisions(
        [0, 1],
        [0.1, 0.8],
        [0.2, 0.9],
    )

    assert result["mcnemar_exact_two_sided_p"] == 1.0
    assert result["changed_predicted_classes"] == 0


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), -0.1, 1.1],
)
def test_bad_probabilities_rejected(bad):
    with pytest.raises(ValueError, match="Probabilities"):
        paired_decisions(
            [0, 1],
            [0.1, bad],
            [0.2, 0.7],
        )


def test_patient_bootstrap_deterministic():
    data = pd.DataFrame({
        "patient_id": [1, 1, 2, 3, 4, 5],
        "target_label": [0, 0, 1, 0, 1, 1],
        "v1_with_sqi": [0.2, 0.6, 0.2, 0.2, 0.8, 0.4],
        "v3_with_sqi": [0.2, 0.4, 0.7, 0.8, 0.8, 0.6],
    })

    first = patient_bootstrap(
        data,
        draws=100,
        seed=42,
    )

    second = patient_bootstrap(
        data,
        draws=100,
        seed=42,
    )

    assert first == second
    assert first["draws"] == 100
    assert first["auc_draws_with_both_classes"] > 0


def test_missing_patient_is_rejected():
    data = pd.DataFrame({
        "patient_id": [np.nan],
        "target_label": [0],
        "v1_with_sqi": [0.1],
        "v3_with_sqi": [0.2],
    })

    with pytest.raises(ValueError, match="Missing patients"):
        patient_bootstrap(data, draws=10)


def test_load_verifies_hashes_and_labels(tmp_path):
    predictions = tmp_path / "predictions.csv"
    manifest = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.json"

    frame = pd.DataFrame({
        "ecg_id": [1, 2, 3, 4],
        "target_label": [0, 1, 0, 1],
        "v1_with_sqi": [0.1, 0.8, 0.2, 0.9],
        "v3_with_sqi": [0.1, 0.7, 0.2, 0.9],
        "shared_without_sqi": [0.2, 0.7, 0.1, 0.8],
    })

    frame.to_csv(predictions, index=False)

    pd.DataFrame({
        "ecg_id": [1, 2, 3, 4],
        "patient_id": [11, 22, 33, 44],
        "strat_fold": [9, 9, 9, 9],
        "target_label": [0, 1, 0, 1],
    }).to_csv(manifest, index=False)

    report = {
        "experiment": "sqi_rf_ablation_v1",
        "test_fold_evaluated": False,
        "predictions_sha256": sha256(predictions),
        "input_sha256": {
            "cohort_manifest": sha256(manifest)
        },
        "validation_ecg_ids": [1, 2, 3, 4],
        "validation_metrics": {
            name: {
                "roc_auc": roc_auc_score(
                    frame["target_label"],
                    frame[name],
                )
            }
            for name in (
                "v1_with_sqi",
                "v3_with_sqi",
                "shared_without_sqi",
            )
        },
    }

    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    checked, _ = load_verified_inputs(
        predictions,
        report_path,
        manifest,
    )

    assert len(checked) == 4
    assert checked["patient_id"].nunique() == 4

    # Tampering with predictions must invalidate the audit.
    frame.loc[0, "v3_with_sqi"] = 0.9
    frame.to_csv(predictions, index=False)

    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_inputs(
            predictions,
            report_path,
            manifest,
        )
