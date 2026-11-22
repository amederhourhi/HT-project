
"""Audit CardioSense model structure and artifact provenance.

Checks checkpoint structure, feature order, file hashes, dataset
fingerprints, and dependency versions without running predictions.

Structural compatibility does not prove training lineage,
SQI V3 compatibility, or clinical validity.
"""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.features.extract_features import FEATURE_COLUMNS
from src.models.ecg_cnn import CardioNet1D


ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = (
    "src/dsp/filters.py",
    "src/dsp/input_validation.py",
    "src/dsp/sqi.py",
    "src/dsp/sqi_v3.py",
    "src/data/ecg_loader.py",
    "src/features/extract_features.py",
    "src/features/quality_contract.py",
    "src/models/ecg_cnn.py",
    "src/models/ecg_dataset.py",
    "src/models/train_cnn.py",
    "src/models/train_interpretable.py",
)

DEPENDENCIES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "torch",
    "wfdb",
    "joblib",
)


def sha256_file(path):
    """Calculate a file's SHA-256 fingerprint."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def file_record(path):
    """Record a file's location, size, and fingerprint."""

    path = Path(path)

    result = {
        "path": path.relative_to(ROOT).as_posix(),
        "exists": path.is_file(),
    }

    if path.is_file():
        result["bytes"] = path.stat().st_size
        result["sha256"] = sha256_file(path)

    return result


def git_commit():
    """Retrieve the current local Git commit."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        return result.stdout.strip()

    except (OSError, subprocess.SubprocessError):
        return None


def dependency_versions():
    """Record the currently installed package versions."""

    installed = {
        "python": platform.python_version()
    }

    for package in DEPENDENCIES:
        try:
            installed[package] = version(package)

        except PackageNotFoundError:
            installed[package] = None

    return installed


def inspect_rf_schema(model):
    """Verify Random Forest input features and class ordering."""

    names = getattr(model, "feature_names_in_", None)
    count = getattr(model, "n_features_in_", None)
    classes = getattr(model, "classes_", None)

    # Convert NumPy values into JSON-compatible Python types.
    names = (
        [str(name) for name in names]
        if names is not None
        else None
    )

    labels = (
        [
            value.item() if hasattr(value, "item") else value
            for value in classes
        ]
        if classes is not None
        else None
    )

    names_match = (
        names == FEATURE_COLUMNS
        if names is not None
        else None
    )

    count_match = (
        int(count) == len(FEATURE_COLUMNS)
        if count is not None
        else None
    )

    classes_match = (
        labels == [0, 1]
        if labels is not None
        else None
    )

    checks = (
        names_match,
        count_match,
        classes_match,
    )

    if False in checks:
        status = "fail"
    elif None in checks:
        status = "incomplete"
    else:
        status = "pass"

    return {
        "model_type": (
            f"{type(model).__module__}."
            f"{type(model).__name__}"
        ),
        "feature_names": names,
        "feature_count": (
            int(count) if count is not None else None
        ),
        "classes": labels,
        "expected_feature_names": list(FEATURE_COLUMNS),
        "feature_order_matches": names_match,
        "feature_count_matches": count_match,
        "class_order_matches": classes_match,
        "structural_check": status,
    }


def inspect_cnn_weights(weights, expected):
    """Check checkpoint tensors against the CNN architecture."""

    actual_keys = set(weights)
    expected_keys = set(expected)

    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)

    shape_mismatches = []
    invalid_tensors = []

    for name in sorted(actual_keys & expected_keys):
        value = weights[name]

        if not isinstance(value, torch.Tensor):
            invalid_tensors.append(name)
            continue

        if value.shape != expected[name].shape:
            shape_mismatches.append({
                "name": name,
                "expected": list(expected[name].shape),
                "actual": list(value.shape),
            })

        if not bool(torch.isfinite(value).all()):
            invalid_tensors.append(name)

    compatible = not any((
        missing,
        unexpected,
        shape_mismatches,
        invalid_tensors,
    ))

    return {
        "structural_check": (
            "pass" if compatible else "fail"
        ),
        "expected_state_keys": len(expected),
        "checkpoint_state_keys": len(weights),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "invalid_tensors": invalid_tensors,
    }


def audit_rf(path):
    """Inspect the deployed RF without generating predictions."""

    result = {
        "artifact": file_record(path),
        "training_lineage": "unverified",
        "sqi_v3_compatibility": "unverified",
    }

    if not path.is_file():
        result["structural_check"] = "missing"
        return result

    try:
        # joblib uses pickle: only load trusted project files.
        model = joblib.load(path)

        result.update(
            inspect_rf_schema(model)
        )

    except Exception as exc:
        result["structural_check"] = "error"
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def audit_cnn(path):
    """Verify CNN checkpoint structure without model inference."""

    result = {
        "artifact": file_record(path),
        "training_lineage": "unverified",
        "sqi_v3_compatibility": "unverified",
    }

    if not path.is_file():
        result["structural_check"] = "missing"
        return result

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        # Accept a plain state dictionary or wrapped checkpoint.
        weights = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
            else checkpoint
        )

        if not isinstance(weights, dict):
            raise TypeError(
                "CNN checkpoint is not a state dictionary"
            )

        # Compare against the current CardioNet1D architecture.
        expected = CardioNet1D(
            in_channels=12,
            num_classes=2,
        ).state_dict()

        result.update(
            inspect_cnn_weights(weights, expected)
        )

    except Exception as exc:
        result["structural_check"] = "error"
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def dataset_record(path):
    """Fingerprint a dataset and verify its feature-column order."""

    result = file_record(path)

    if not path.is_file():
        return result

    try:
        # Read only CSV headers; no training data is modified.
        columns = pd.read_csv(
            path,
            nrows=0,
        ).columns.tolist()

        present = [
            name for name in columns
            if name in FEATURE_COLUMNS
        ]

        result["feature_order_matches"] = (
            present == FEATURE_COLUMNS
        )

        result["missing_features"] = [
            name for name in FEATURE_COLUMNS
            if name not in columns
        ]

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def build_report():
    """Collect model, dataset, code, and environment evidence."""

    checkpoints = ROOT / "models" / "checkpoints"
    processed = ROOT / "data" / "processed"

    source_hashes = {
        name: file_record(ROOT / name)
        for name in SOURCE_FILES
    }

    datasets = {
        name: dataset_record(processed / name)
        for name in (
            "clinical_features.csv",
            "clinical_features_validated.csv",
        )
    }

    return {
        "audit_version": "model_contract_audit_v1",
        "generated_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "git_commit": git_commit(),
        "environment": dependency_versions(),
        "source_files": source_hashes,
        "datasets": datasets,
        "deployed_models": {
            "random_forest": audit_rf(
                checkpoints / "interpretable_model.joblib"
            ),
            "cnn": audit_cnn(
                checkpoints / "ecg_cnn.pt"
            ),
        },
        "candidate_artifacts": {
            "random_forest": file_record(
                checkpoints
                / "interpretable_model_fold10_candidate.joblib"
            ),
            "cnn": file_record(
                checkpoints
                / "ecg_cnn_fold10_candidate.pt"
            ),
        },
        "conclusion": {
            "structural_checks_are_not_training_proof": True,
            "deployed_checkpoint_lineage_verified": False,
            "sqi_v3_model_compatibility_verified": False,
            "clinical_validation_established": False,
        },
    }


def main():
    """Generate the audit report and print key results."""

    report = build_report()

    output = ROOT / "docs" / "model_contract_audit.json"
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    rf = report["deployed_models"]["random_forest"]
    cnn = report["deployed_models"]["cnn"]

    print("\nCardioSense Model Contract Audit")
    print(f"Git commit          : {report['git_commit']}")
    print(f"RF structure        : {rf['structural_check']}")
    print(f"CNN structure       : {cnn['structural_check']}")
    print(
        "RF feature order    : "
        f"{rf.get('feature_order_matches', 'unknown')}"
    )
    print(
        "RF class order      : "
        f"{rf.get('class_order_matches', 'unknown')}"
    )
    print("Training lineage    : unverified")
    print("SQI V3 compatibility: unverified")
    print(f"Audit written to    : {output}")


if __name__ == "__main__":
    main()
