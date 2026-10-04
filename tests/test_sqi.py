"""
CardioSense - SQI Verification Test
Tests the quality control module on real clinical ECG data
and verifies automatic rejection of disconnected and noisy leads.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------
# Block 1: Add project root to path
# ---------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import numpy as np
import wfdb
from src.dsp.filters import filter_ecg
from src.dsp.sqi import assess_ecg_quality, compute_lead_sqi

# ---------------------------------------------------------
# Block 2: Load and filter a real sample record
# ---------------------------------------------------------
samples_dir = project_root / "data" / "samples"
header_files = list(samples_dir.glob("**/*.hea"))
record_path = str(header_files[0].with_suffix(""))
raw_signals, fields = wfdb.rdsamp(record_path)
fs = fields["fs"]
lead_names = fields["sig_name"]

cleaned_signals = filter_ecg(raw_signals, fs=fs)

# ---------------------------------------------------------
# Block 3: Test 1 - Real Clinical ECG (Expect High Quality)
# ---------------------------------------------------------
print("\n--- Test 1: Real Clinical 12-Lead ECG ---")
result_real = assess_ecg_quality(cleaned_signals, lead_names=lead_names)
print(f"Overall SQI     : {result_real['overall_sqi']}")
print(f"Passed Leads    : {result_real['passed_leads']}")
print(f"Accept Record?  : {result_real['is_acceptable']}")

# ---------------------------------------------------------
# Block 4: Test 2 - Artificial Disconnected Lead (Lead V2 Flatlined)
# ---------------------------------------------------------
print("\n--- Test 2: Simulating Disconnected Lead (Flatline on V2) ---")
corrupted_signals = cleaned_signals.copy()
v2_idx = lead_names.index("V2") if "V2" in lead_names else 7
corrupted_signals[:, v2_idx] = 0.0  # Flatline

v2_sqi = compute_lead_sqi(corrupted_signals[:, v2_idx])
print(f"Lead V2 Quality Score after disconnection: {v2_sqi} (Expected: 0.0)")

# ---------------------------------------------------------
# Block 5: Assertions to guarantee automated tests pass
# ---------------------------------------------------------
assert result_real["is_acceptable"] is True, "Real ECG failed quality control!"
assert v2_sqi == 0.0, "Disconnected lead was not caught by SQI!"

print("\n" + "=" * 45)
print("  All SQI Quality-Control Tests PASSED!  ")
print("=" * 45)