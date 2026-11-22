
"""Build an auditable SQI V3 research feature dataset.

Only quality-eligible ECGs enter the dataset. Selection must
match the existing cohort audit. No model training is performed.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.ecg_loader import load_standardized_ecg
from src.features.extract_features import (
    FEATURE_COLUMNS,
    extract_clinical_features,
)
from src.triage.inference_quality import evaluate_research_readiness


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/cohort"
MANIFEST = COHORT / "cohort_manifest.csv"
AUDIT = ROOT / "docs/sqi_audit/v1_v3_feature_shift_records.csv"

OUTPUT = ROOT / "data/processed/clinical_features_v3_candidate.csv"
REPORT = ROOT / "docs/sqi_audit/v3_dataset_contract.json"

META_COLUMNS = [
    "ecg_id", "patient_id", "target_label",
    "strat_fold", "age", "sex",
]


def sha256(path):
    """Compute an artifact's SHA-256 fingerprint."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit():
    """Get the source revision used for extraction."""
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
    """Preserve the official PTB-XL partition assignment."""
    if 1 <= fold <= 8:
        return "train"
    if fold == 9:
        return "validation"
    if fold == 10:
        return "test"
    raise ValueError(f"Invalid fold: {fold}")


def verify_dataset(frame):
    """Validate feature values, class coverage, and patient separation."""
    if frame.empty:
        raise ValueError("V3 candidate dataset is empty")

    if frame["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ECG IDs")

    if frame["patient_id"].isna().any():
        raise ValueError("Missing patient IDs")

    if not frame["target_label"].isin([0, 1]).all():
        raise ValueError("Invalid target labels")

    if not frame["strat_fold"].isin(range(1, 11)).all():
        raise ValueError("Invalid stratified folds")

    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Missing classifier features")

    values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite classifier features")

    work = frame.copy()
    work["split"] = work["strat_fold"].map(split_name)

    if work.groupby("patient_id")["split"].nunique().gt(1).any():
        raise ValueError("Patient leakage across data partitions")

    for name in ("train", "validation", "test"):
        subset = work[work["split"] == name]

        if subset.empty or subset["target_label"].nunique() != 2:
            raise ValueError(
                f"{name} is empty or lacks a diagnostic class"
            )

    return work.groupby(
        ["split", "target_label"]
    ).size().to_dict()


def build_dataset():
    """Regenerate V3 features and verify the frozen quality selection."""
    manifest = pd.read_csv(MANIFEST)
    audit = pd.read_csv(AUDIT)

    required = {
        *META_COLUMNS, "filename_lr",
    }
    if not required.issubset(manifest.columns):
        raise ValueError("Cohort manifest is missing required fields")

    if manifest["ecg_id"].duplicated().any():
        raise ValueError("Duplicate manifest ECG IDs")

    if audit["ecg_id"].duplicated().any():
        raise ValueError("Duplicate audit ECG IDs")

    manifest = manifest.set_index("ecg_id")
    audit = audit.set_index("ecg_id")

    if set(manifest.index) != set(audit.index):
        raise ValueError("Manifest and quality audit contain different ECGs")

    for field in ("target_label", "strat_fold"):
        if not manifest[field].sort_index().equals(
            audit[field].sort_index()
        ):
            raise ValueError(f"Metadata mismatch: {field}")

    if not audit["quality_decision"].isin(
        ["eligible", "abstain"]
    ).all():
        raise ValueError("Invalid quality decisions in audit")

    # The previous audit is a frozen selection contract.
    expected_ids = set(
        audit.index[audit["quality_decision"] == "eligible"]
    )

    rows = []

    for ecg_id, record in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Building V3 dataset",
    ):
        ecg_id = int(ecg_id)
        record_path = COHORT / str(record["filename_lr"])

        if not all(
            record_path.with_suffix(ext).is_file()
            for ext in (".hea", ".dat")
        ):
            status = "missing"
            decision = "abstain"
            features = None
        else:
            signals, fs, leads = load_standardized_ecg(record_path)

            extraction = extract_clinical_features(
                signals,
                fs=fs,
                lead_names=leads,
                sqi_version="v3",
            )

            status = extraction["status"]
            quality = evaluate_research_readiness(extraction)
            decision = quality["decision"]
            features = extraction.get("features")

        # Reject changes in extraction or eligibility.
        expected = audit.loc[ecg_id]

        if (
            status != expected["v3_status"]
            or decision != expected["quality_decision"]
        ):
            raise RuntimeError(
                f"ECG {ecg_id}: V3 result changed since audit"
            )

        if decision != "eligible":
            continue

        if not isinstance(features, dict):
            raise RuntimeError(
                f"ECG {ecg_id}: eligible without features"
            )

        age = pd.to_numeric(
            pd.Series([record.get("age")]),
            errors="coerce",
        ).iloc[0]

        # Match the historical cohort age policy.
        if pd.notna(age) and not 0 <= age <= 110:
            age = float("nan")

        entry = {
            "ecg_id": ecg_id,
            "patient_id": record["patient_id"],
            "target_label": int(record["target_label"]),
            "strat_fold": int(record["strat_fold"]),
            "age": age,
            "sex": record.get("sex"),
        }

        for name in FEATURE_COLUMNS:
            if name not in features:
                raise ValueError(
                    f"ECG {ecg_id} missing feature: {name}"
                )
            entry[name] = features[name]

        rows.append(entry)

    frame = pd.DataFrame(
        rows,
        columns=META_COLUMNS + FEATURE_COLUMNS,
    ).sort_values("ecg_id").reset_index(drop=True)

    actual_ids = set(frame["ecg_id"])

    if actual_ids != expected_ids:
        raise RuntimeError(
            "V3 dataset does not reproduce the frozen selection"
        )

    counts = verify_dataset(frame)

    return frame, counts


def main():
    """Save the candidate dataset and its provenance contract."""
    frame, counts = build_dataset()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to silently replace an existing candidate.
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError(
            "V3 candidate output already exists. "
            "Review the existing artifacts before rebuilding."
        )

    temp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(OUTPUT)

    sources = [
        "src/data/ecg_loader.py",
        "src/dsp/input_validation.py",
        "src/dsp/filters.py",
        "src/dsp/sqi_v3.py",
        "src/features/extract_features.py",
        "src/features/quality_contract.py",
        "src/triage/inference_quality.py",
    ]

    contract = {
        "contract_version": "v3_research_dataset_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "sqi_version": "v3",
        "quality_policy": "fail_closed_research_gate",
        "model_training_performed": False,
        "model_compatibility_verified": False,
        "feature_order": FEATURE_COLUMNS,
        "record_count": len(frame),
        "split_label_counts": [
            {
                "split": split,
                "target_label": int(label),
                "count": int(count),
            }
            for (split, label), count in sorted(counts.items())
        ],
        "inputs": {
            "manifest_sha256": sha256(MANIFEST),
            "quality_audit_sha256": sha256(AUDIT),
        },
        "dataset": {
            "path": OUTPUT.relative_to(ROOT).as_posix(),
            "sha256": sha256(OUTPUT),
        },
        "source_sha256": {
            path: sha256(ROOT / path)
            for path in sources
        },
        "limitations": [
            "SQI V3 quality flags are experimental",
            "Training lineage of deployed checkpoints is unverified",
            "No clinical validation has been established",
        ],
    }

    temp_report = REPORT.with_name(REPORT.name + ".tmp")
    temp_report.write_text(
        json.dumps(contract, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_report.replace(REPORT)

    print("\nCardioSense V3 Research Dataset")
    print(f"Eligible ECGs : {len(frame)}")
    print(f"Features      : {len(FEATURE_COLUMNS)}")
    print("Patient leaks : 0")
    print(f"Dataset       : {OUTPUT}")
    print(f"Contract      : {REPORT}")
    print("Model training: not performed")


if __name__ == "__main__":
    main()
