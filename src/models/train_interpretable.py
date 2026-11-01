"""Train and evaluate CardioSense's calibrated Random Forest baseline.

PTB-XL fold protocol: 1-8 training, 9 validation, 10 final test.
This script does not overwrite the deployed model checkpoint.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)

from src.features.extract_features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "clinical_features.csv"
MODEL_PATH = ROOT / "models" / "checkpoints" / "interpretable_model_fold10_candidate.joblib"
REPORT_PATH = ROOT / "docs" / "rf_fold10_evaluation.json"


def load_data(path=DATA):
    frame = pd.read_csv(path)
    required = [*FEATURE_COLUMNS, "strat_fold", "target_label"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["strat_fold"].isna().any() or not frame["strat_fold"].isin(range(1, 11)).all():
        raise ValueError("strat_fold must contain integers 1 through 10")
    if not frame["target_label"].isin([0, 1]).all():
        raise ValueError("target_label must be binary (0 or 1)")
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns contain missing values; resolve upstream before training")
    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Non-numeric feature: {column}")
    if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain infinite values")

    subsets = {
        "train": frame.loc[frame["strat_fold"].between(1, 8)].copy(),
        "validation": frame.loc[frame["strat_fold"] == 9].copy(),
        "test": frame.loc[frame["strat_fold"] == 10].copy(),
    }
    for name, subset in subsets.items():
        if subset.empty or subset["target_label"].nunique() != 2:
            raise ValueError(f"{name} must contain both label classes")

    # Identifiers are optional in exported feature tables, but when available
    # they must not cross fold boundaries.
    for identifier in ("ecg_id", "patient_id"):
        if identifier in frame.columns:
            for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
                a = set(subsets[first][identifier].dropna())
                b = set(subsets[second][identifier].dropna())
                if a & b:
                    raise ValueError(f"{identifier} overlaps between {first} and {second}")

    return subsets


def metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    predicted = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "records": int(len(labels)),
        "positives": int(sum(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "log_loss": float(log_loss(labels, probs, labels=[0, 1])),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    partitions = load_data()
    train = partitions["train"]
    val = partitions["validation"]
    test = partitions["test"]
    print(f"Records: train={len(train)}, val={len(val)}, test={len(test)}")

    base = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    # Cross-validated sigmoid calibration uses training records only.
    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    calibrated.fit(train[FEATURE_COLUMNS], train["target_label"])

    val_probs = calibrated.predict_proba(val[FEATURE_COLUMNS])[:, 1]
    val_metrics = metrics(val["target_label"], val_probs)
    print("\nValidation fold-9 results (threshold 0.50):")
    for key, value in val_metrics.items():
        print(f"  {key}: {value}")

    # Frozen model and default threshold are evaluated exactly once here.
    test_probs = calibrated.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test_metrics = metrics(test["target_label"], test_probs)
    print("\nIndependent fold-10 results (threshold 0.50):")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, MODEL_PATH)
    report = {
        "protocol": "PTB-XL folds 1-8 train / 9 validation / 10 test",
        "candidate_checkpoint": str(MODEL_PATH),
        "features": list(FEATURE_COLUMNS),
        "model": "CalibratedClassifierCV(RandomForestClassifier, sigmoid, cv=5)",
        "validation": val_metrics,
        "test": test_metrics,
        "limitations": ["Small test cohort", "No external validation", "0.5 threshold not clinically validated"],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved candidate: {MODEL_PATH}")
    print(f"Saved evaluation: {REPORT_PATH}")


if __name__ == "__main__":
    main()
