
"""Isolate the effect of SQI values on CardioSense RF performance.

All variants use identical ECGs, labels, grouped calibration
folds, and RF settings. Fold 10 is never evaluated.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from scripts.compare_v1_v3_rf import load_v1_partitions
from scripts.train_v3_rf_candidate import (
    ROOT,
    DATASET,
    MANIFEST,
    calculate_metrics,
    make_calibration_folds,
    verify_contract,
    verify_dataset,
)
from src.features.extract_features import FEATURE_COLUMNS


OUTPUT_DIR = ROOT / "docs/model_contracts"
REPORT = OUTPUT_DIR / "sqi_rf_ablation_fold9.json"
PREDICTIONS = OUTPUT_DIR / "sqi_rf_ablation_fold9_predictions.csv"

SHARED_FEATURES = [
    name for name in FEATURE_COLUMNS
    if name != "overall_sqi"
]


def sha256(path):
    """Fingerprint an experiment input."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def pair_partition(v1, v3):
    """Align the same ECGs and verify their metadata."""

    if not v1["ecg_id"].is_unique or not v3["ecg_id"].is_unique:
        raise ValueError("Duplicate ECG identifiers")

    common = sorted(
        set(v1["ecg_id"]) & set(v3["ecg_id"])
    )

    if not common:
        raise ValueError("No shared ECGs")

    left = v1.set_index("ecg_id").loc[common].copy()
    right = v3.set_index("ecg_id").loc[common].copy()

    for name in ("target_label", "strat_fold"):
        if not np.array_equal(
            left[name].to_numpy(),
            right[name].to_numpy(),
        ):
            raise ValueError(f"Mismatched {name}")

    if not np.array_equal(
        left["patient_id"].astype(str).to_numpy(),
        right["patient_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Mismatched patient IDs")

    return left, right


def verify_shared_features(v1, v3):
    """Require all features except SQI to match numerically."""

    left = v1[SHARED_FEATURES].to_numpy(dtype=float)
    right = v3[SHARED_FEATURES].to_numpy(dtype=float)

    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite shared features")

    matches = np.isclose(
        left,
        right,
        rtol=1e-9,
        atol=1e-9,
    )

    if not matches.all():
        row, column = np.argwhere(~matches)[0]
        raise ValueError(
            "Unexpected non-SQI feature difference: "
            f"ECG {v1.index[row]}, {SHARED_FEATURES[column]}"
        )

    return True


def train_variant(train, feature_names, folds):
    """Train one RF with the same patient-aware calibration splits."""

    model = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        method="sigmoid",
        cv=folds,
        ensemble=True,
    )

    model.fit(
        train[feature_names],
        train["target_label"],
    )

    if list(model.classes_) != [0, 1]:
        raise ValueError("Invalid classifier class order")

    if list(model.feature_names_in_) != feature_names:
        raise ValueError("Invalid trained feature order")

    return model


def score_variant(model, validation, feature_names):
    """Compute fold-9 probabilities and classification metrics."""

    probabilities = model.predict_proba(
        validation[feature_names]
    )[:, 1]

    metrics = calculate_metrics(
        validation["target_label"],
        probabilities,
    )

    return probabilities, metrics


def sqi_difference_count(v1, v3):
    """Count ECGs whose overall SQI changed."""

    same = np.isclose(
        v1["overall_sqi"].to_numpy(dtype=float),
        v3["overall_sqi"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )

    return int((~same).sum())


def main():
    """Run the three-arm controlled ablation."""

    if REPORT.exists() or PREDICTIONS.exists():
        raise FileExistsError(
            "Ablation results already exist; refusing to overwrite"
        )

    # Verify the existing V3 dataset contract first.
    contract = verify_contract()

    v1 = load_v1_partitions()
    v3 = verify_dataset(
        pd.read_csv(DATASET),
        contract,
    )

    train_v1, train_v3 = pair_partition(
        v1["train"],
        v3["train"],
    )

    val_v1, val_v3 = pair_partition(
        v1["validation"],
        v3["validation"],
    )

    # Earlier audits found only SQI differences on paired ECGs.
    verify_shared_features(train_v1, train_v3)
    verify_shared_features(val_v1, val_v3)

    if (
        set(train_v1["patient_id"])
        & set(val_v1["patient_id"])
    ):
        raise ValueError("Patient leakage between training and validation")

    for name, frame in (
        ("train", train_v1),
        ("validation", val_v1),
    ):
        if frame["target_label"].nunique() != 2:
            raise ValueError(f"{name} lacks a diagnostic class")

    # Reuse exactly the same calibration indices in every variant.
    folds = make_calibration_folds(
        train_v1[FEATURE_COLUMNS],
        train_v1["target_label"],
        train_v1["patient_id"],
    )

    variants = {
        "v1_with_sqi": {
            "train": train_v1,
            "validation": val_v1,
            "features": FEATURE_COLUMNS,
        },
        "v3_with_sqi": {
            "train": train_v3,
            "validation": val_v3,
            "features": FEATURE_COLUMNS,
        },
        "shared_without_sqi": {
            "train": train_v1,
            "validation": val_v1,
            "features": SHARED_FEATURES,
        },
    }

    print("\nCardioSense | SQI Random Forest Ablation")
    print(f"Matched training ECGs   : {len(train_v1)}")
    print(f"Matched validation ECGs : {len(val_v1)}")
    print(f"SQI changes in training : {sqi_difference_count(train_v1, train_v3)}")
    print(f"SQI changes in validation: {sqi_difference_count(val_v1, val_v3)}")
    print("Patient-aware CV       : 5 folds")
    print("Test fold evaluated    : False")

    outcomes = {}
    probabilities = {}

    for name, config in variants.items():
        print(f"\nTraining: {name}")

        model = train_variant(
            config["train"],
            config["features"],
            folds,
        )

        probs, metrics = score_variant(
            model,
            config["validation"],
            config["features"],
        )

        probabilities[name] = probs
        outcomes[name] = metrics

    predictions = pd.DataFrame({
        "ecg_id": val_v1.index.to_numpy(),
        "target_label": val_v1["target_label"].to_numpy(),
        **{
            name: probs
            for name, probs in probabilities.items()
        },
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temp_predictions = PREDICTIONS.with_name(
        PREDICTIONS.name + ".tmp"
    )
    predictions.to_csv(temp_predictions, index=False)
    temp_predictions.replace(PREDICTIONS)

    report = {
        "experiment": "sqi_rf_ablation_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "train_records": len(train_v1),
        "validation_records": len(val_v1),
        "train_ecg_ids": [int(x) for x in train_v1.index],
        "validation_ecg_ids": [int(x) for x in val_v1.index],
        "sqi_changes_train": sqi_difference_count(
            train_v1, train_v3
        ),
        "sqi_changes_validation": sqi_difference_count(
            val_v1, val_v3
        ),
        "feature_order_with_sqi": FEATURE_COLUMNS,
        "feature_order_without_sqi": SHARED_FEATURES,
        "calibration": "5-fold patient-aware sigmoid",
        "validation_metrics": outcomes,
        "test_fold_evaluated": False,
        "input_sha256": {
            "v1_dataset": sha256(
                ROOT / "data/processed/clinical_features_validated.csv"
            ),
            "v3_dataset": sha256(DATASET),
            "cohort_manifest": sha256(MANIFEST),
        },
        "predictions_sha256": sha256(PREDICTIONS),
        "limitations": [
            "Small matched validation cohort",
            "Fold 9 has been used in prior experiments",
            "Training restricted to the common V1/V3 population",
            "Quality gate is experimental",
            "No independent external validation",
        ],
    }

    temp_report = REPORT.with_name(REPORT.name + ".tmp")
    temp_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_report.replace(REPORT)

    print("\nMatched Fold-9 Metrics")
    print(
        f"{'Metric':15s} "
        f"{'V1 + SQI':>12s} "
        f"{'V3 + SQI':>12s} "
        f"{'No SQI':>12s}"
    )

    for metric in (
        "accuracy",
        "sensitivity",
        "specificity",
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
    ):
        values = [
            outcomes[name][metric]
            for name in variants
        ]

        print(
            f"{metric:15s} "
            + " ".join(f"{value:12.4f}" for value in values)
        )

    print(f"\nReport      : {REPORT}")
    print(f"Predictions : {PREDICTIONS}")
    print("Model checkpoints: unchanged")


if __name__ == "__main__":
    main()
