
"""Verify our proposed cohort against official PTB-XL metadata."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the same diagnostic-code definitions as the project.
from src.data.download_ptbxl import INFARCTION_SUBCLASSES, _scp_code_set

PIN = ROOT / "data/processed/cohort_ecg_ids.csv"
DATABASE = ROOT / "data/raw/ptbxl_database.csv"


def main():
    if not DATABASE.exists():
        raise FileNotFoundError(f"Missing official metadata: {DATABASE}")

    pin = pd.read_csv(PIN)
    db = pd.read_csv(DATABASE, index_col="ecg_id")

    # 1. Check that all candidate ECG IDs exist officially.
    missing = sorted(set(pin.ecg_id) - set(db.index))

    if missing:
        raise ValueError(f"ECG IDs missing from PTB-XL: {missing}")

    if db.index.has_duplicates:
        raise ValueError("Duplicate ECG IDs in official metadata")

    # Preserve the candidate's order when selecting official records.
    official = db.loc[pin.ecg_id].reset_index()

    # 2. Independently derive our research labels from SCP codes.
    codes = official["scp_codes"].map(_scp_code_set)

    is_mi = codes.map(lambda c: bool(c & INFARCTION_SUBCLASSES))
    is_norm = codes.map(lambda c: "NORM" in c)

    derived_labels = is_mi.astype(int)

    # Neither category, or both categories, is ambiguous.
    ambiguous = (is_mi & is_norm) | ~(is_mi | is_norm)

    label_mismatch = pin.target_label.ne(derived_labels)

    # 3. Verify official stratified folds.
    fold_mismatch = pin.strat_fold.ne(official.strat_fold)

    # 4. Check patient-level separation.
    patient_folds = official.groupby("patient_id").strat_fold.nunique()
    overlapping_patients = set(patient_folds[patient_folds > 1].index)

    patient_overlap = official.patient_id.isin(overlapping_patients)
    missing_patient = official.patient_id.isna()

    # Report problems without modifying either dataset.
    problems = pd.DataFrame({
        "ecg_id": pin.ecg_id,
        "ambiguous_label": ambiguous,
        "label_mismatch": label_mismatch,
        "fold_mismatch": fold_mismatch,
        "patient_overlap": patient_overlap,
        "missing_patient_id": missing_patient,
    })

    failures = problems.drop(columns="ecg_id").any(axis=1)
    failed = problems.loc[failures]

    print("\n--- PTB-XL Cohort Verification ---")
    print(f"Candidate records: {len(pin)}")
    print(f"Missing official IDs: {len(missing)}")
    print(f"Ambiguous diagnostic labels: {ambiguous.sum()}")
    print(f"Label mismatches: {label_mismatch.sum()}")
    print(f"Fold mismatches: {fold_mismatch.sum()}")
    print(f"Patients crossing folds: {len(overlapping_patients)}")
    print(f"Records with missing patient IDs: {missing_patient.sum()}")

    if not failed.empty:
        print("\nRecords requiring investigation:")
        print(failed.to_string(index=False))
        raise SystemExit("\nVERIFICATION FAILED")

    print("\nVERIFICATION PASSED")


if __name__ == "__main__":
    main()