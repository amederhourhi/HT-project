"""
CardioSense - PTB-XL Dataset Downloader
Fetches metadata and 100 Hz ECG records from PhysioNet, and materialises the
pinned clinical cohort (manifest + waveforms) used by feature extraction,
model training and the triage regression gate.
"""

import argparse
import ast
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"


def download_file(url: str, destination: Path) -> bool:
    """Downloads a file from a URL with a visual progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
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


# ---------------------------------------------------------
# Pinned clinical cohort
# ---------------------------------------------------------
# The cohort is deliberately pinned rather than re-sampled: every downstream
# number (the P0-1 guardrail audit, the trained models, the dashboard library)
# must refer to one frozen set of records. Re-sampling here would silently
# move the denominator under those results.
INFARCTION_SUBCLASSES = {
    "AMI", "IMI", "LMI", "PMI",
    "ASMI", "ALMI", "ILMI", "IPMI", "IPLMI",
    "INJAS", "INJAL", "INJIL", "INJIN", "INJLA",
}

# Ages outside this window are entry errors, not patients. PTB-XL carries at
# least one record with age = 300, which would otherwise drive the sex- and
# age-specific STEMI cut-points and any age-valued model input.
AGE_PLAUSIBLE_RANGE = (0, 110)


def _scp_code_set(raw_scp_codes: str) -> set:
    """Parses the stringified {'CODE': confidence, ...} cell from ptbxl_database."""
    text = str(raw_scp_codes)
    if not text.startswith("{"):
        return set()
    try:
        return set(ast.literal_eval(text))
    except (ValueError, SyntaxError):
        return set()


def build_cohort_manifest(
    pinned_ids_csv: Path,
    database_csv: Path,
    cohort_dir: Path,
    download: bool = True,
) -> pd.DataFrame:
    """
    Writes data/cohort/cohort_manifest.csv for the pinned ecg_ids, enriched with
    authoritative PTB-XL metadata, then fetches the matching WFDB records.

    Args:
        pinned_ids_csv: committed CSV holding the frozen 'ecg_id' column.
        database_csv: official ptbxl_database.csv (source of truth for metadata).
        cohort_dir: destination directory for the manifest and waveforms.
        download: when False, only the manifest is written.

    Returns:
        The manifest as a DataFrame, indexed by ecg_id.
    """
    pinned = pd.read_csv(pinned_ids_csv, usecols=["ecg_id"])["ecg_id"].astype(int)
    db = pd.read_csv(database_csv).set_index("ecg_id")

    missing = sorted(set(pinned) - set(db.index))
    if missing:
        raise KeyError(f"Pinned ecg_ids absent from ptbxl_database.csv: {missing[:10]}")

    manifest = db.loc[pinned].copy()

    # Age sanity: out-of-window ages become NaN (unknown) instead of being
    # carried downstream as if they were real patients.
    too_young, too_old = AGE_PLAUSIBLE_RANGE
    implausible_age = manifest["age"].notna() & (
        (manifest["age"] < too_young) | (manifest["age"] > too_old)
    )
    if implausible_age.any():
        affected = [int(x) for x in manifest.index[implausible_age]]
        print(f"[warn] {len(affected)} records carry an implausible age {affected}; "
              f"age set to unknown (NaN).")
        manifest.loc[implausible_age, "age"] = float("nan")

    # target_label: 1 = infarction-family SCP code present, 0 = NORM reference.
    # Derived here rather than copied so the definition lives in one place.
    codes = manifest["scp_codes"].map(_scp_code_set)
    manifest["target_label"] = codes.map(
        lambda c: 1 if c & INFARCTION_SUBCLASSES else 0
    ).astype(int)

    unlabelled = codes.map(lambda c: "NORM" not in c and not (c & INFARCTION_SUBCLASSES))
    if unlabelled.any():
        print(f"[warn] {int(unlabelled.sum())} records carry neither NORM nor an "
              f"infarction code; they are kept but will default to label 0.")

    keep = [
        "ecg_id", "patient_id", "filename_lr", "target_label", "strat_fold",
        "age", "sex", "scp_codes", "infarction_stadium1", "validated_by_human",
    ]
    manifest = manifest.reset_index()[keep]

    cohort_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cohort_dir / "cohort_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nCohort manifest written: {manifest_path} ({len(manifest)} records)")
    print(f"  label 0 (NORM reference)        : {int((manifest.target_label == 0).sum())}")
    print(f"  label 1 (infarction family)     : {int((manifest.target_label == 1).sum())}")
    print(f"  folds 1-8 (train) / 9-10 (test) : "
          f"{int((manifest.strat_fold <= 8).sum())} / {int((manifest.strat_fold > 8).sum())}")

    if download:
        print("\n--- Fetching cohort waveforms (100 Hz) ---")
        incomplete = []
        for ecg_id, rel_path in tqdm(
            zip(manifest["ecg_id"], manifest["filename_lr"]), total=len(manifest),
            desc="Records", unit="rec",
        ):
            local = Path(str(cohort_dir / rel_path))
            # Both extensions are attempted unconditionally: short-circuiting on
            # the header would leave a half-written record behind and hide the
            # data file that has to be retried alongside it.
            for ext in (".hea", ".dat"):
                download_file(f"{BASE_URL}{rel_path.replace(os.sep, '/')}{ext}", local.with_suffix(ext))
            if not (local.with_suffix(".hea").exists() and local.with_suffix(".dat").exists()):
                incomplete.append(int(ecg_id))

        complete = len(manifest) - len(incomplete)
        print(f"Cohort records available locally: {complete}/{len(manifest)}")
        if incomplete:
            print(f"[retry needed] ecg_ids missing a file: {incomplete}")

    return manifest


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    parser = argparse.ArgumentParser(description="Download PTB-XL metadata, samples or the pinned cohort")
    parser.add_argument("--samples_only", action="store_true", help="Download metadata and a small sample set")
    parser.add_argument("--cohort", action="store_true",
                        help="Materialise the pinned cohort (manifest + waveforms) under data/cohort/")
    parser.add_argument("--manifest_only", action="store_true",
                        help="With --cohort: write the manifest but skip the waveform download")
    args = parser.parse_args()

    raw_dir = project_root / "data" / "raw"
    samples_dir = project_root / "data" / "samples"
    cohort_dir = project_root / "data" / "cohort"

    download_metadata(raw_dir)

    if args.cohort:
        build_cohort_manifest(
            # The pin lives in its own tracked file. Reading it back out of
            # clinical_features.csv would be circular: that file is this
            # pipeline's output, so one extra gate rejection would quietly
            # shrink the cohort and move the denominator under every result
            # computed from it.
            pinned_ids_csv=project_root / "data" / "processed" / "cohort_ecg_ids.csv",
            database_csv=raw_dir / "ptbxl_database.csv",
            cohort_dir=cohort_dir,
            download=not args.manifest_only,
        )
    else:
        download_sample_records(raw_dir, samples_dir, n_per_class=2)
