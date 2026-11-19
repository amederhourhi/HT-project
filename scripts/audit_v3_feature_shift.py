
"""Compare CardioSense SQI V1 and V3 feature distributions.

Uses the validated V1 dataset as the baseline and runs V3 on the
same pinned cohort. No models are loaded or modified.
"""

import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.ecg_loader import load_standardized_ecg
from src.features.extract_features import (
    FEATURE_COLUMNS,
    extract_clinical_features,
)
from src.triage.inference_quality import evaluate_research_readiness


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/cohort/cohort_manifest.csv"
BASELINE = ROOT / "data/processed/clinical_features_validated.csv"
COHORT = ROOT / "data/cohort"
OUTPUT_DIR = ROOT / "docs/sqi_audit"

TOLERANCE = 1e-9


def sha256(path):
    """Fingerprint an input or source file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def git_commit():
    """Identify the current code revision."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def load_inputs():
    """Verify that the V1 baseline matches the pinned cohort."""

    manifest = pd.read_csv(MANIFEST)
    baseline = pd.read_csv(BASELINE)

    required_manifest = {
        "ecg_id", "filename_lr", "strat_fold", "target_label",
    }
    required_baseline = {
        "ecg_id", "strat_fold", "target_label", *FEATURE_COLUMNS,
    }

    if not required_manifest.issubset(manifest.columns):
        raise ValueError("Manifest is missing required columns")

    if not required_baseline.issubset(baseline.columns):
        raise ValueError("V1 baseline is missing required columns")

    for name, frame in (
        ("manifest", manifest),
        ("baseline", baseline),
    ):
        if frame["ecg_id"].isna().any():
            raise ValueError(f"{name} contains missing ECG IDs")

        if frame["ecg_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate ECG IDs")

    manifest = manifest.set_index("ecg_id")
    baseline = baseline.set_index("ecg_id")

    if not baseline.index.isin(manifest.index).all():
        raise ValueError("V1 baseline contains unrecognized ECG IDs")

    # A paired comparison is invalid if labels or folds changed.
    paired_metadata = manifest.loc[baseline.index]

    for column in ("target_label", "strat_fold"):
        if not (
            paired_metadata[column].to_numpy()
            == baseline[column].to_numpy()
        ).all():
            raise ValueError(
                f"V1 and manifest metadata differ: {column}"
            )

    return manifest, baseline


def compare_features(v1, v3, ecg_id):
    """Return measurements changed beyond numerical tolerance."""

    differences = []

    for name in FEATURE_COLUMNS:
        if name not in v1 or name not in v3:
            raise ValueError(f"Missing classifier feature: {name}")

        old = float(v1[name])
        new = float(v3[name])

        if not math.isfinite(old) or not math.isfinite(new):
            raise ValueError(f"Non-finite feature: {name}")

        if not math.isclose(
            old,
            new,
            rel_tol=TOLERANCE,
            abs_tol=TOLERANCE,
        ):
            differences.append({
                "ecg_id": int(ecg_id),
                "feature": name,
                "v1_value": old,
                "v3_value": new,
                "absolute_difference": abs(new - old),
            })

    return differences


def evaluate_record(record_path):
    """Run V3 extraction and the existing quality gate."""

    signals, fs, leads = load_standardized_ecg(record_path)

    extraction = extract_clinical_features(
        signals,
        fs=fs,
        lead_names=leads,
        sqi_version="v3",
    )

    if extraction.get("status") not in {"success", "rejected"}:
        raise ValueError("Unexpected V3 extraction status")

    quality = evaluate_research_readiness(extraction)

    return extraction, quality


def save_csv(rows, columns, path):
    """Write a CSV safely without changing existing study outputs."""

    table = pd.DataFrame(rows, columns=columns)
    temp = path.with_name(path.name + ".tmp")

    table.to_csv(temp, index=False)
    temp.replace(path)


def main():
    """Measure coverage, research eligibility and paired feature drift."""

    manifest, baseline = load_inputs()
    records = []
    differences = []

    print("\nCardioSense | V1-V3 Feature Compatibility")
    print(f"Manifest records : {len(manifest)}")
    print(f"V1 baseline      : {len(baseline)}")

    for ecg_id, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Assessing SQI V3",
    ):
        ecg_id = int(ecg_id)
        record_path = COHORT / str(row["filename_lr"])

        record = {
            "ecg_id": ecg_id,
            "target_label": int(row["target_label"]),
            "strat_fold": int(row["strat_fold"]),
            "v1_success": ecg_id in baseline.index,
            "v3_status": "missing",
            "quality_decision": "abstain",
            "reason": "",
            "review_feature_count": 0,
            "unavailable_feature_count": 0,
            "paired_success": False,
            "changed_feature_count": 0,
        }

        if not all(
            record_path.with_suffix(ext).is_file()
            for ext in (".hea", ".dat")
        ):
            record["reason"] = "Recording files missing"

        else:
            try:
                extraction, quality = evaluate_record(record_path)

            except Exception as exc:
                record["v3_status"] = "error"
                record["reason"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            else:
                record["v3_status"] = extraction["status"]
                record["quality_decision"] = quality["decision"]
                record["review_feature_count"] = len(
                    quality["review_required_features"]
                )
                record["unavailable_feature_count"] = len(
                    quality["unavailable_features"]
                )
                record["reason"] = (
                    extraction.get("reason", "")
                    if extraction["status"] == "rejected"
                    else quality["reason"]
                )

                # Only compare records accepted by both versions.
                if (
                    record["v1_success"]
                    and extraction["status"] == "success"
                ):
                    changes = compare_features(
                        baseline.loc[ecg_id],
                        extraction["features"],
                        ecg_id,
                    )

                    record["paired_success"] = True
                    record["changed_feature_count"] = len(changes)

                    differences.extend(changes)

        records.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records_path = (
        OUTPUT_DIR / "v1_v3_feature_shift_records.csv"
    )
    differences_path = (
        OUTPUT_DIR / "v1_v3_feature_differences.csv"
    )
    summary_path = (
        OUTPUT_DIR / "v1_v3_feature_shift_summary.json"
    )

    record_columns = [
        "ecg_id", "target_label", "strat_fold", "v1_success",
        "v3_status", "quality_decision", "reason",
        "review_feature_count", "unavailable_feature_count",
        "paired_success", "changed_feature_count",
    ]

    difference_columns = [
        "ecg_id", "feature", "v1_value",
        "v3_value", "absolute_difference",
    ]

    save_csv(records, record_columns, records_path)
    save_csv(differences, difference_columns, differences_path)

    record_frame = pd.DataFrame(records)
    delta_frame = pd.DataFrame(
        differences,
        columns=difference_columns,
    )

    # Compare coverage and decisions across the pinned cohort.
    status_counts = Counter(record_frame["v3_status"])
    decisions = Counter(record_frame["quality_decision"])

    paired = record_frame["paired_success"]
    changed = record_frame["changed_feature_count"] > 0

    feature_changes = (
        delta_frame.groupby("feature")
        .agg(
            changed_values=("ecg_id", "count"),
            max_absolute_difference=(
                "absolute_difference", "max"
            ),
        )
        .sort_values("changed_values", ascending=False)
        .reset_index()
        .to_dict(orient="records")
        if not delta_frame.empty
        else []
    )

    summary = {
        "audit_version": "v1_v3_feature_shift_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "manifest_sha256": sha256(MANIFEST),
        "v1_baseline_sha256": sha256(BASELINE),
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "src/dsp/sqi_v3.py",
                "src/features/extract_features.py",
                "src/features/quality_contract.py",
                "src/triage/inference_quality.py",
            )
        },
        "comparison_tolerance": TOLERANCE,
        "manifest_records": len(manifest),
        "v1_successes": len(baseline),
        "v3_status_counts": dict(status_counts),
        "v3_quality_decisions": dict(decisions),
        "paired_successes": int(paired.sum()),
        "paired_records_with_changes": int(
            (paired & changed).sum()
        ),
        "changed_feature_values": len(differences),
        "feature_changes": feature_changes,
        "by_label": (
            record_frame.groupby("target_label")
            .agg(
                records=("ecg_id", "count"),
                v3_successes=(
                    "v3_status",
                    lambda s: int(s.eq("success").sum()),
                ),
                eligible=(
                    "quality_decision",
                    lambda s: int(s.eq("eligible").sum()),
                ),
            )
            .reset_index()
            .to_dict(orient="records")
        ),
        "by_fold": (
            record_frame.groupby("strat_fold")
            .agg(
                records=("ecg_id", "count"),
                v3_successes=(
                    "v3_status",
                    lambda s: int(s.eq("success").sum()),
                ),
                eligible=(
                    "quality_decision",
                    lambda s: int(s.eq("eligible").sum()),
                ),
            )
            .reset_index()
            .to_dict(orient="records")
        ),
        "limitations": [
            "No model inference was performed",
            "Feature drift does not establish model compatibility",
            "SQI quality flags are experimental",
            "No clinical validation was performed",
        ],
    }

    temp_json = summary_path.with_name(
        summary_path.name + ".tmp"
    )
    temp_json.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_json.replace(summary_path)

    print("\nResults")
    print(f"V1 successes         : {len(baseline)}")
    print(f"V3 successes         : {status_counts['success']}")
    print(f"V3 rejected          : {status_counts['rejected']}")
    print(f"Missing              : {status_counts['missing']}")
    print(f"Errors               : {status_counts['error']}")
    print(f"V3 quality eligible  : {decisions['eligible']}")
    print(f"Paired V1/V3 successes: {int(paired.sum())}")
    print(f"Paired ECGs changed  : {int((paired & changed).sum())}")
    print(f"Changed feature values: {len(differences)}")

    print("\nMost frequently changed features:")
    for item in feature_changes[:10]:
        print(
            f"  {item['feature']:25s} "
            f"{item['changed_values']:4d} changed"
        )

    print("\nSaved:")
    print(records_path)
    print(differences_path)
    print(summary_path)


if __name__ == "__main__":
    main()
