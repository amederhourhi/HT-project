
"""Train a separate CardioSense SQI V3 Random Forest candidate.

Verifies dataset provenance, uses patient-aware calibration folds,
and evaluates on fold 9 only. Fold 10 is reserved.

Deployed checkpoints are never modified.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.features.extract_features import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data/processed/clinical_features_v3_candidate.csv"
CONTRACT = ROOT / "docs/sqi_audit/v3_dataset_contract.json"
MANIFEST = ROOT / "data/cohort/cohort_manifest.csv"
QUALITY_AUDIT = ROOT / "docs/sqi_audit/v1_v3_feature_shift_records.csv"

MODEL_PATH = (
    ROOT / "models/checkpoints/interpretable_model_v3_research_candidate.joblib"
)
REPORT_PATH = (
    ROOT / "docs/model_contracts/interpretable_model_v3_research_candidate.json"
)

SEED = 42
CALIBRATION_FOLDS = 5
THRESHOLD = 0.5


def sha256_file(path):
    """Calculate a reproducible file fingerprint."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit():
    """Identify the local source revision."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )

    return result.stdout.strip()


def split_name(fold):
    """Map official PTB-XL folds to research partitions."""

    if 1 <= fold <= 8:
        return "train"
    if fold == 9:
        return "validation"
    if fold == 10:
        return "test"

    raise ValueError(f"Invalid fold: {fold}")


def verify_contract():
    """Verify the frozen dataset and upstream processing evidence."""

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    if contract.get("contract_version") != "v3_research_dataset_v1":
        raise ValueError("Unsupported V3 dataset contract")

    if contract.get("sqi_version") != "v3":
        raise ValueError("Dataset does not declare SQI V3")

    if contract.get("quality_policy") != "fail_closed_research_gate":
        raise ValueError("Unexpected quality selection policy")

    if contract.get("feature_order") != FEATURE_COLUMNS:
        raise ValueError("Classifier feature order changed")

    if contract["dataset"]["sha256"] != sha256_file(DATASET):
        raise ValueError("V3 dataset fingerprint mismatch")

    if contract["inputs"]["manifest_sha256"] != sha256_file(MANIFEST):
        raise ValueError("Cohort manifest fingerprint mismatch")

    if contract["inputs"]["quality_audit_sha256"] != sha256_file(
        QUALITY_AUDIT
    ):
        raise ValueError("Quality audit fingerprint mismatch")

    # Reject changes in preprocessing or quality-gate code.
    for name, expected_hash in contract["source_sha256"].items():
        path = ROOT / name

        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Source changed since extraction: {name}")

    return contract


def verify_dataset(frame, contract):
    """Check the frozen cohort, feature values, and patient separation."""

    required = {
        "ecg_id", "patient_id", "target_label",
        "strat_fold", *FEATURE_COLUMNS,
    }

    if not required.issubset(frame.columns):
        raise ValueError("V3 dataset is missing required columns")

    if len(frame) != contract["record_count"]:
        raise ValueError("V3 dataset record count mismatch")

    if frame["ecg_id"].isna().any() or not frame["ecg_id"].is_unique:
        raise ValueError("Missing or duplicate ECG identifiers")

    if frame["patient_id"].isna().any():
        raise ValueError("Missing patient identifiers")

    if not frame["target_label"].isin([0, 1]).all():
        raise ValueError("Invalid target labels")

    if not frame["strat_fold"].isin(range(1, 11)).all():
        raise ValueError("Invalid stratified folds")

    for name in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"Non-numeric feature: {name}")

    if not np.isfinite(
        frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError("Missing or non-finite classifier values")

    # Verify that the candidate contains exactly the audited ECGs.
    audit = pd.read_csv(QUALITY_AUDIT)

    if not audit["ecg_id"].is_unique:
        raise ValueError("Duplicate quality-audit ECG IDs")

    eligible = audit.loc[
        audit["quality_decision"].eq("eligible"),
        "ecg_id",
    ]

    if set(frame["ecg_id"]) != set(eligible):
        raise ValueError("Candidate differs from audited eligible records")

    # Verify that demographics and fold assignments were not changed.
    manifest = pd.read_csv(MANIFEST)

    if not manifest["ecg_id"].is_unique:
        raise ValueError("Duplicate manifest ECG IDs")

    aligned = manifest.set_index("ecg_id").loc[
        frame["ecg_id"]
    ].reset_index()

    for name in ("patient_id", "target_label", "strat_fold"):
        if not np.array_equal(
            frame[name].to_numpy(),
            aligned[name].to_numpy(),
        ):
            raise ValueError(f"Manifest metadata mismatch: {name}")

    work = frame.copy()
    work["split"] = work["strat_fold"].map(split_name)

    # A patient must belong to exactly one partition.
    overlap = work.groupby("patient_id")["split"].nunique()

    if overlap.gt(1).any():
        raise ValueError("Patient leakage between partitions")

    partitions = {
        name: work.loc[work["split"].eq(name)].copy()
        for name in ("train", "validation", "test")
    }

    for name, subset in partitions.items():
        if subset.empty or subset["target_label"].nunique() != 2:
            raise ValueError(f"Invalid class coverage in {name}")

    expected_counts = {
        (item["split"], int(item["target_label"])): int(item["count"])
        for item in contract["split_label_counts"]
    }

    actual_counts = {
        (str(split), int(label)): int(count)
        for (split, label), count in work.groupby(
            ["split", "target_label"]
        ).size().items()
    }

    if expected_counts != actual_counts:
        raise ValueError("Dataset split counts differ from contract")

    return partitions


def make_calibration_folds(features, labels, patients):
    """Create five internal calibration folds without patient overlap."""

    splitter = StratifiedGroupKFold(
        n_splits=CALIBRATION_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    splits = list(splitter.split(features, labels, patients))

    y = np.asarray(labels)
    groups = np.asarray(patients)

    for train_idx, calibration_idx in splits:
        training_patients = set(groups[train_idx])
        calibration_patients = set(groups[calibration_idx])

        if training_patients & calibration_patients:
            raise ValueError("Patient leakage in calibration folds")

        if len(np.unique(y[train_idx])) != 2:
            raise ValueError("Calibration training fold lacks a class")

        if len(np.unique(y[calibration_idx])) != 2:
            raise ValueError("Calibration holdout fold lacks a class")

    return splits


def calculate_metrics(labels, probabilities):
    """Evaluate binary classification at the frozen 0.5 threshold."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    if y.size == 0 or y.shape != p.shape:
        raise ValueError("Invalid evaluation arrays")

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation requires both classes")

    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid predicted probabilities")

    predicted = (p >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predicted, labels=[0, 1]
    ).ravel()

    return {
        "records": int(len(y)),
        "positives": int(y.sum()),
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y, predicted)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def train_candidate(partitions):
    """Train using folds 1–8 and evaluate on fold 9 only."""

    train = partitions["train"]
    validation = partitions["validation"]

    x_train = train[FEATURE_COLUMNS]
    y_train = train["target_label"]

    folds = make_calibration_folds(
        x_train,
        y_train,
        train["patient_id"],
    )

    base = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    model = CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=folds,
        ensemble=True,
    )

    model.fit(x_train, y_train)

    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected classifier class order")

    if list(model.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("Trained model feature order mismatch")

    probabilities = model.predict_proba(
        validation[FEATURE_COLUMNS]
    )[:, 1]

    metrics = calculate_metrics(
        validation["target_label"],
        probabilities,
    )

    return model, metrics


def main():
    """Train, evaluate, and save a versioned research candidate."""

    # Never silently overwrite an existing experiment.
    if MODEL_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError(
            "V3 candidate model or report already exists"
        )

    contract = verify_contract()
    frame = pd.read_csv(DATASET)
    partitions = verify_dataset(frame, contract)

    print("\nCardioSense | V3 Random Forest Candidate")
    print(f"Train      : {len(partitions['train'])}")
    print(f"Validation : {len(partitions['validation'])}")
    print(f"Reserved   : {len(partitions['test'])}")
    print(f"Features   : {len(FEATURE_COLUMNS)}")
    print("Calibration: patient-aware, 5 folds")

    model, metrics = train_candidate(partitions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_model = MODEL_PATH.with_name(
        MODEL_PATH.name + ".tmp"
    )

    joblib.dump(model, temporary_model)
    temporary_model.replace(MODEL_PATH)

    report = {
        "contract_version": "rf_v3_research_candidate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "status": "research_candidate",
        "sqi_version": "v3",
        "quality_policy": "fail_closed_research_gate",
        "feature_order": FEATURE_COLUMNS,
        "class_order": [0, 1],
        "training": {
            "folds": list(range(1, 9)),
            "records": len(partitions["train"]),
            "random_seed": SEED,
            "n_estimators": 100,
            "max_depth": 5,
            "class_weight": "balanced",
            "calibration_method": "sigmoid",
            "calibration_folds": CALIBRATION_FOLDS,
            "calibration_grouping": "patient_id",
            "calibration_ensemble": True,
        },
        "validation": {
            "fold": 9,
            "metrics": metrics,
        },
        "test": {
            "fold": 10,
            "records": len(partitions["test"]),
            "evaluated": False,
        },
        "artifacts": {
            "candidate_model": MODEL_PATH.relative_to(ROOT).as_posix(),
            "candidate_model_sha256": sha256_file(MODEL_PATH),
            "training_dataset_sha256": sha256_file(DATASET),
            "dataset_contract_sha256": sha256_file(CONTRACT),
            "quality_audit_sha256": sha256_file(QUALITY_AUDIT),
            "training_script_sha256": sha256_file(Path(__file__)),
        },
        "environment": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "joblib")
        },
        "deployed_model_replaced": False,
        "clinical_validation": "not_established",
        "limitations": [
            "Small, quality-selected research cohort",
            "No external validation",
            "Fold 10 not evaluated in this training run",
            "Existing V1 checkpoint lineage remains unverified",
            "SQI V3 compatibility is specific to this research pipeline",
        ],
    }

    temporary_report = REPORT_PATH.with_name(
        REPORT_PATH.name + ".tmp"
    )

    temporary_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(REPORT_PATH)

    print("\nFold 9 Validation Results")

    for name, value in metrics.items():
        print(f"{name:17s}: {value}")

    print("\nSaved Candidate")
    print(f"Model    : {MODEL_PATH}")
    print(f"Contract : {REPORT_PATH}")
    print("Fold 10  : not evaluated")
    print("Deployed checkpoints: unchanged")


if __name__ == "__main__":
    main()
