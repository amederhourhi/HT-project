
"""CardioSense batch feature extraction.

Loads a pinned ECG cohort, validates recordings, extracts features,
and records the outcome of every ECG for reproducibility.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.ecg_loader import load_standardized_ecg
from src.features.extract_features import extract_clinical_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = {
    "ecg_id",
    "filename_lr",
    "target_label",
    "strat_fold",
}


def load_cohort_manifest(cohort_dir):
    """Load and validate the pinned PTB-XL cohort metadata."""

    manifest_path = cohort_dir / "cohort_manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cohort manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    if manifest.empty:
        raise ValueError("Cohort manifest is empty")

    missing = REQUIRED_COLUMNS - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Missing manifest columns: {sorted(missing)}"
        )

    if manifest[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(
            "Required manifest fields contain missing values"
        )

    if manifest["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ECG IDs in cohort manifest")

    if not manifest["target_label"].isin([0, 1]).all():
        raise ValueError("Invalid target labels; expected 0 or 1")

    if not manifest["strat_fold"].isin(range(1, 11)).all():
        raise ValueError("Invalid PTB-XL fold; expected 1 through 10")

    if manifest["filename_lr"].astype(str).str.strip().eq("").any():
        raise ValueError("Manifest contains empty recording paths")

    # Match the age policy in src/data/download_ptbxl.py.
    # PTB-XL uses 300 for privacy-censored ages, not a literal age.
    if "age" in manifest.columns:
        manifest["age"] = pd.to_numeric(
            manifest["age"],
            errors="coerce",
        )

        invalid_age = (
            manifest["age"].notna()
            & ~manifest["age"].between(0, 110)
        )

        if invalid_age.any():
            count = int(invalid_age.sum())
            manifest.loc[invalid_age, "age"] = float("nan")

            print(
                f"[Metadata] Normalized {count} "
                "out-of-range age value(s) to NaN"
            )

    return manifest


def save_csv(frame, path):
    """Write a CSV through a temporary file to avoid partial output."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")

    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def build_clinical_features_dataset(cohort_dir, output_csv):
    """Extract clinical features and save a record-level audit.

    Only successful recordings enter the feature dataset.
    Every manifest record receives exactly one audit entry.

    The existing SQI V1 default and classifier features are preserved.
    """

    cohort_dir = Path(cohort_dir)
    output_csv = Path(output_csv)

    audit_csv = output_csv.with_name(
        f"{output_csv.stem}_audit.csv"
    )

    manifest = load_cohort_manifest(cohort_dir)

    feature_rows = []
    audit_rows = []

    print("\nCardioSense | Batch Feature Extraction")
    print(f"Manifest records : {len(manifest)}")
    print(f"SQI policy       : V1 (existing default)")
    print(f"Output           : {output_csv}")

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Extracting ECG features",
    ):
        ecg_id = int(row["ecg_id"])
        record_base = cohort_dir / str(row["filename_lr"])

        status = "error"
        stage = "filesystem"
        reason = ""

        header = record_base.with_suffix(".hea")
        signal_file = record_base.with_suffix(".dat")

        # Missing recordings must be distinguished from SQI rejection.
        if not header.is_file() or not signal_file.is_file():
            status = "missing"
            reason = "WFDB header or signal file not found"

        else:
            try:
                # Load standardized 12-lead samples in millivolts.
                signals, fs, lead_names = load_standardized_ecg(
                    record_base
                )

            except ValueError as exc:
                status = "invalid_input"
                stage = "input_validation"
                reason = str(exc)

            except Exception as exc:
                status = "error"
                stage = "loading"
                reason = f"{type(exc).__name__}: {exc}"

            else:
                stage = "feature_extraction"

                try:
                    # Preserve existing extraction and SQI behavior.
                    result = extract_clinical_features(
                        signals,
                        fs=fs,
                        lead_names=lead_names,
                    )

                    if result["status"] == "success":
                        features = result["features"]

                        if not isinstance(features, dict):
                            raise ValueError(
                                "Missing feature dictionary"
                            )

                        entry = {
                            "ecg_id": ecg_id,
                            "target_label": int(row["target_label"]),
                            "strat_fold": int(row["strat_fold"]),
                            "age": row.get("age"),
                            "sex": row.get("sex"),
                        }

                        entry.update(features)
                        feature_rows.append(entry)

                        status = "success"
                        stage = "complete"
                        reason = "OK"

                    elif result["status"] == "rejected":
                        status = "rejected"
                        reason = str(
                            result.get(
                                "reason",
                                "Unspecified extraction rejection",
                            )
                        )

                    else:
                        raise ValueError(
                            f"Unexpected status: {result['status']}"
                        )

                except Exception as exc:
                    status = "error"
                    reason = f"{type(exc).__name__}: {exc}"

        # Keep an audit trail for every manifest entry.
        audit_rows.append({
            "ecg_id": ecg_id,
            "filename_lr": str(row["filename_lr"]),
            "target_label": int(row["target_label"]),
            "strat_fold": int(row["strat_fold"]),
            "status": status,
            "stage": stage,
            "reason": reason,
        })

    # Check that no records disappeared during processing.
    if len(audit_rows) != len(manifest):
        raise RuntimeError("Incomplete extraction audit")

    success_count = sum(
        item["status"] == "success"
        for item in audit_rows
    )

    if success_count != len(feature_rows):
        raise RuntimeError("Audit and feature counts disagree")

    # Save audit results even if every extraction failed.
    audit = pd.DataFrame(audit_rows)
    save_csv(audit, audit_csv)

    if not feature_rows:
        raise RuntimeError(
            f"No successful extractions. Check {audit_csv}"
        )

    features = pd.DataFrame(feature_rows)
    save_csv(features, output_csv)

    # Summarize outcomes without printing every SQI rejection.
    counts = Counter(audit["status"])

    print("\nExtraction Results")
    print(f"Manifest records : {len(manifest)}")
    print(f"Successful       : {counts['success']}")
    print(f"SQI rejected     : {counts['rejected']}")
    print(f"Invalid inputs   : {counts['invalid_input']}")
    print(f"Missing files    : {counts['missing']}")
    print(f"Errors           : {counts['error']}")

    print(
        f"Coverage         : "
        f"{success_count / len(manifest):.2%}"
    )

    print("\nClass Distribution")
    print(
        f"NORM             : "
        f"{int((features['target_label'] == 0).sum())}"
    )
    print(
        f"MI label family  : "
        f"{int((features['target_label'] == 1).sum())}"
    )

    print("\nSaved Files")
    print(f"Features : {output_csv}")
    print(f"Audit    : {audit_csv}")

    return features


if __name__ == "__main__":
    cohort_dir = PROJECT_ROOT / "data" / "cohort"

    output_csv = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "clinical_features_validated.csv"
    )

    build_clinical_features_dataset(
        cohort_dir=cohort_dir,
        output_csv=output_csv,
    )
