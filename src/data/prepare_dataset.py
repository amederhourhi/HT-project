"""
CardioSense - Balanced Cohort Downloader
Downloads a balanced clinical dataset (NORM vs MI) across official PTB-XL
stratified folds (Folds 1-8: Train, Folds 9-10: Test).
"""

import os
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"


# ---------------------------------------------------------
# Block 1: Robust File Downloader
# ---------------------------------------------------------
def download_file(url, destination):
    """Downloads a single file from PhysioNet if it does not already exist."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return True

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(destination, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False


# ---------------------------------------------------------
# Block 2: Cohort Selection (Balanced NORM vs MI)
# ---------------------------------------------------------
def select_balanced_cohort(metadata_csv, n_per_class=150):
    """
    Selects an equal number of Normal ('NORM') and Heart Attack ('MI') records
    while preserving the original 'strat_fold' column for training/testing.
    """
    df = pd.read_csv(metadata_csv, index_col="ecg_id")

    # Filter Normal records (strictly NORM with high diagnostic certainty)
    norm_records = df[df["scp_codes"].str.contains("'NORM': 100", na=False)].copy()
    norm_records["target_label"] = 0  # 0 = Normal

    # Filter Myocardial Infarction records (AMI, IMI, ASMI, etc.)
    mi_records = df[df["scp_codes"].str.contains("MI", na=False) & ~df["scp_codes"].str.contains("'NORM'", na=False)].copy()
    mi_records["target_label"] = 1   # 1 = Myocardial Infarction

    # Direct stratified sampling (150 of each class)
    sampled_norm = norm_records.sample(n=n_per_class, random_state=42)
    sampled_mi = mi_records.sample(n=n_per_class, random_state=42)

    # Combine into a single cohort DataFrame
    cohort = pd.concat([sampled_norm, sampled_mi])
    return cohort


# ---------------------------------------------------------
# Block 3: Batch Download Selected Waveforms
# ---------------------------------------------------------
def download_cohort_waveforms(cohort_df, destination_dir):
    """Downloads the .hea and .dat files for every selected patient."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {len(cohort_df)} clinical 100 Hz ECG records from PhysioNet...")

    success_count = 0
    for ecg_id, row in tqdm(cohort_df.iterrows(), total=len(cohort_df), desc="Fetching ECGs"):
        rel_path = row["filename_lr"]
        
        # Download both header (.hea) and signal data (.dat)
        hea_url = f"{BASE_URL}{rel_path.replace(os.sep, '/')}.hea"
        dat_url = f"{BASE_URL}{rel_path.replace(os.sep, '/')}.dat"
        
        hea_dest = destination_dir / f"{rel_path}.hea"
        dat_dest = destination_dir / f"{rel_path}.dat"

        if download_file(hea_url, hea_dest) and download_file(dat_url, dat_dest):
            success_count += 1

    print(f"\nSuccessfully saved {success_count}/{len(cohort_df)} records to {destination_dir}")
    
    # Save manifest CSV for training
    manifest_path = destination_dir / "cohort_manifest.csv"
    cohort_df.to_csv(manifest_path)
    print(f"Cohort manifest saved to: {manifest_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    meta_path = project_root / "data" / "raw" / "ptbxl_database.csv"
    cohort_dir = project_root / "data" / "cohort"

    cohort = select_balanced_cohort(meta_path, n_per_class=150)
    print(f"Selected Balanced Cohort: {len(cohort)} records")
    print(f"  - Normal (NORM) : {sum(cohort['target_label'] == 0)}")
    print(f"  - Infarction (MI): {sum(cohort['target_label'] == 1)}")
    print(f"  - Fold Split    : Train (Folds 1-8): {sum(cohort['strat_fold'] <= 8)} | Test (Folds 9-10): {sum(cohort['strat_fold'] > 8)}")

    download_cohort_waveforms(cohort, cohort_dir)