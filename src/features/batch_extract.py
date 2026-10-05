"""
CardioSense - Batch Feature Extraction Pipeline
Processes all 297 clinical cohort ECGs, runs quality gates,
and compiles a clean tabular dataset for Machine Learning.
"""

import sys
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
    Iterates through the downloaded cohort, extracts 19 clinical biomarkers
    per patient, and saves a labeled CSV with official PTB-XL stratified folds.
    """
    manifest_path = cohort_dir / "cohort_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cohort manifest not found at {manifest_path}")

    df_manifest = pd.read_csv(manifest_path, index_col="ecg_id")
    print(f"\nProcessing {len(df_manifest)} records through CardioSense Feature Pipeline...")

    dataset_rows = []
    rejected_count = 0

    for ecg_id, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="Extracting Features"):
        rel_path = row["filename_lr"]
        record_base = cohort_dir / rel_path

        # Check if record files exist
        if not record_base.with_suffix(".hea").exists():
            continue

        try:
            # Read 12-lead raw signals
            signals, fields = wfdb.rdsamp(str(record_base))
            fs = fields["fs"]
            lead_names = fields["sig_name"]

            # Run full feature extraction and SQI gatekeeper
            result = extract_clinical_features(signals, fs=fs, lead_names=lead_names)

            if result["status"] == "success":
                entry = {
                    "ecg_id": ecg_id,
                    "target_label": row["target_label"],  # 0 = NORM, 1 = MI
                    "strat_fold": row["strat_fold"],      # 1-8: Train, 9-10: Test
                    "age": row.get("age", None),
                    "sex": row.get("sex", None),
                }
                entry.update(result["features"])
                dataset_rows.append(entry)
            else:
                rejected_count += 1
        except Exception:
            rejected_count += 1

    # Convert to DataFrame
    df_features = pd.DataFrame(dataset_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print(f"Extraction Complete!")
    print(f"  - Successfully Extracted: {len(df_features)} records")
    print(f"  - Rejected by SQI Gate  : {rejected_count} records")
    print(f"  - Normal (NORM) Cases   : {sum(df_features['target_label'] == 0)}")
    print(f"  - Infarction (MI) Cases : {sum(df_features['target_label'] == 1)}")
    print(f"  - Saved Clean Dataset to: {output_csv.relative_to(project_root)}")
    print("=" * 60)


if __name__ == "__main__":
    cohort_directory = project_root / "data" / "cohort"
    output_dataset_path = project_root / "data" / "processed" / "clinical_features.csv"

    build_clinical_features_dataset(cohort_directory, output_dataset_path)