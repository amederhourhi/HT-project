"""
CardioSense - PTB-XL Dataset Downloader
Fetches metadata and 100 Hz ECG records from PhysioNet.
"""

import os
import argparse
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

def download_file(url: str, destination: Path) -> bool:
    """Downloads a file from a URL with a visual progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  [Already exists] {destination.name}")
        return True

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(destination, "wb") as file, tqdm(
            desc=destination.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)
        return True
    except Exception as e:
        print(f"  [Error downloading {url}]: {e}")
        if destination.exists():
            destination.unlink()
        return False

def download_metadata(data_dir: Path) -> tuple[Path, Path]:
    """Downloads ptbxl_database.csv and scp_statements.csv."""
    print("\n--- Downloading PTB-XL Metadata ---")
    data_dir.mkdir(parents=True, exist_ok=True)

    db_csv = data_dir / "ptbxl_database.csv"
    scp_csv = data_dir / "scp_statements.csv"

    download_file(f"{BASE_URL}ptbxl_database.csv", db_csv)
    download_file(f"{BASE_URL}scp_statements.csv", scp_csv)
    return db_csv, scp_csv

def download_sample_records(data_dir: Path, samples_dir: Path, n_per_class: int = 2):
    """
    Downloads a small representative subset of 100 Hz ECG records
    covering key clinical categories: NORM, MI, and STTC.
    """
    print("\n--- Downloading Sample ECG Records (100 Hz) ---")
    samples_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "ptbxl_database.csv"
    if not db_path.exists():
        download_metadata(data_dir)

    df = pd.read_csv(db_path, index_col="ecg_id")

    # In PTB-XL, labels are stored inside the 'scp_codes' column
    # e.g., {'NORM': 100.0} or {'AMI': 100.0} (Anterior Myocardial Infarction)
    class_patterns = {
        "NORM": "'NORM'",
        "MI": "MI",      # Matches AMI (Anterior), IMI (Inferior), ASMI (Anteroseptal), etc.
        "STTC": "STTC",  # Matches ST/T changes and early ischemia
    }

    sample_ids = []
    for diagnostic_class, pattern in class_patterns.items():
        matches = df[df["scp_codes"].str.contains(pattern, na=False)].index
        selected = matches[:n_per_class].tolist()
        sample_ids.extend(selected)
        print(f"Selected {len(selected)} samples for class '{diagnostic_class}': {selected}")

    # PhysioNet WFDB format uses two files per recording: .hea (header) and .dat (binary signal)
    downloaded_records = []
    for ecg_id in sample_ids:
        relative_path_lr = df.loc[ecg_id, "filename_lr"]
        
        for ext in [".hea", ".dat"]:
            # Ensure URL uses forward slashes
            remote_url = f"{BASE_URL}{relative_path_lr.replace(os.sep, '/')}{ext}"
            local_path = samples_dir / f"{relative_path_lr}{ext}"
            download_file(remote_url, local_path)
            
        record_base = str(samples_dir / relative_path_lr)
        downloaded_records.append(record_base)

    print(f"\nSuccessfully downloaded {len(sample_ids)} sample records to {samples_dir}")
    return downloaded_records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PTB-XL metadata and samples")
    parser.add_argument("--samples_only", action="store_true", default=True, help="Download only metadata and small sample set")
    parser.add_argument("--full", action="store_true", help="Download the full 100 Hz dataset (~1.7 GB)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    samples_dir = project_root / "data" / "samples"

    # Fetch metadata (will skip if already downloaded)
    download_metadata(raw_dir)

    if args.full:
        print("\n[Full Download Requested] We will implement the bulk downloader in Phase 3 during model training.")
    else:
        download_sample_records(raw_dir, samples_dir, n_per_class=2)