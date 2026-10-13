"""
CardioSense - Batch Feature Extraction Pipeline
Runs the pinned clinical cohort through the quality gates and delineator, and
compiles a clean tabular dataset for machine learning and guardrail auditing.

Column contract (see src/features/extract_features.py):
  - meta columns           : ecg_id, target_label, strat_fold, age, sex
  - FEATURE_COLUMNS        : classifier inputs, ST measured at the J point
  - LEGACY_COLUMNS         : per-lead ST at J+60, for convention comparison only
  - DIAGNOSTIC_COLUMNS     : beats_used, qrs_plausible, leads_trusted
"""

import sys
from collections import Counter
from pathlib import Path

# Add project root to Python search path so it finds 'src'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import pandas as pd
import wfdb
from tqdm import tqdm

from src.features.extract_features import extract_clinical_features


def build_clinical_features_dataset(cohort_dir, output_csv):
    """
    Iterates the cohort, extracts one clinical feature vector per record, and
    saves a labelled CSV carrying the official PTB-XL stratified folds.

    Every record that does not make it into the output is counted under a reason,
    rather than into one anonymous bucket: "the SQI gate rejected 49 records" and
    "49 records raised an exception we swallowed" are very different facts, and
    the old code could not tell them apart.
    """
    manifest_path = cohort_dir / "cohort_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cohort manifest not found at {manifest_path}. "
            f"Run: python src/data/download_ptbxl.py --cohort"
        )

    df_manifest = pd.read_csv(manifest_path, index_col="ecg_id")
    print(f"\nProcessing {len(df_manifest)} records through CardioSense Feature Pipeline...")

    dataset_rows = []
    drop_out = Counter()

    for ecg_id, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="Extracting Features"):
        record_base = cohort_dir / row["filename_lr"]

        if not (record_base.with_suffix(".hea").exists() and record_base.with_suffix(".dat").exists()):
            drop_out["record files missing locally"] += 1
            continue

        try:
            signals, fields = wfdb.rdsamp(str(record_base))
            result = extract_clinical_features(
                signals, fs=fields["fs"], lead_names=fields["sig_name"]
            )
        except Exception as exc:  # a crash is a bug, not a quality rejection
            drop_out[f"unhandled {type(exc).__name__}"] += 1
            continue

        if result["status"] != "success":
            # Bucket by the leading words of the reason so the tally stays readable.
            reason = str(result["reason"]).split("(")[0].strip()
            drop_out[reason] += 1
            continue

        entry = {
            "ecg_id": int(ecg_id),
            "target_label": int(row["target_label"]),  # 0 = NORM, 1 = infarction family
            "strat_fold": int(row["strat_fold"]),      # 1-8: train, 9-10: test
            "age": row.get("age"),
            "sex": row.get("sex"),
        }
        entry.update(result["features"])
        dataset_rows.append(entry)

    df_features = pd.DataFrame(dataset_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print(f"  - Extracted  : {len(df_features)} records")
    print(f"  - Dropped    : {sum(drop_out.values())} records")
    for reason, count in drop_out.most_common():
        print(f"      {count:4d}  {reason}")
    if len(df_features):
        print(f"  - NORM reference (label 0) : {int((df_features.target_label == 0).sum())}")
        print(f"  - Infarction   (label 1) : {int((df_features.target_label == 1).sum())}")
    print(f"  - Saved to: {output_csv.relative_to(project_root)}")
    print("=" * 60)


if __name__ == "__main__":
    cohort_directory = project_root / "data" / "cohort"
    output_dataset_path = project_root / "data" / "processed" / "clinical_features.csv"

    build_clinical_features_dataset(cohort_directory, output_dataset_path)