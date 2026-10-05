"""
CardioSense - Master Feature Extractor Verification
Batch processes all downloaded sample records through the full pipeline
and prints the resulting clinical feature matrix.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import pandas as pd
import wfdb
from src.features.extract_features import extract_clinical_features

# ---------------------------------------------------------
# Block 1: Locate sample records and metadata
# ---------------------------------------------------------
samples_dir = project_root / "data" / "samples"
metadata_path = project_root / "data" / "raw" / "ptbxl_database.csv"

header_files = sorted(list(samples_dir.glob("**/*.hea")))
if not header_files:
    raise FileNotFoundError("No sample files found in data/samples.")

df_meta = pd.read_csv(metadata_path, index_col="ecg_id")

# ---------------------------------------------------------
# Block 2: Process each record through the master pipeline
# ---------------------------------------------------------
records_data = []

print(f"\nProcessing {len(header_files)} sample ECG records through CardioSense pipeline...\n")

for h in header_files:
    record_path = str(h.with_suffix(""))
    ecg_id = int(h.stem.split("_")[0])
    diagnosis = df_meta.loc[ecg_id, "scp_codes"]

    # Load raw signal
    signals, fields = wfdb.rdsamp(record_path)
    fs = fields["fs"]
    lead_names = fields["sig_name"]

    # Run through full aggregator pipeline
    result = extract_clinical_features(signals, fs=fs, lead_names=lead_names)

    if result["status"] == "success":
        row = {"ecg_id": ecg_id, "diagnosis": diagnosis}
        row.update(result["features"])
        records_data.append(row)
        print(f"  [PASSED] ECG ID: {ecg_id:5d} | SQI: {result['features']['overall_sqi']} | HR: {result['features']['heart_rate_bpm']:4.1f} BPM | Max ST: {result['features']['max_st_elevation_mv']:+.3f} mV")
    else:
        print(f"  [REJECTED] ECG ID: {ecg_id:5d} | Reason: {result['reason']}")

# ---------------------------------------------------------
# Block 3: Display structured clinical DataFrame
# ---------------------------------------------------------
df_features = pd.DataFrame(records_data)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("\n" + "=" * 80)
print("EXTRACTED CLINICAL FEATURE TABLE (First 6 columns + ST V1-V3):")
print("=" * 80)

summary_cols = ["ecg_id", "heart_rate_bpm", "sdnn_ms", "qrs_duration_ms", "max_st_elevation_mv", "st_V1", "st_V2", "st_V3", "overall_sqi"]
print(df_features[summary_cols].to_string(index=False))
print("=" * 80)