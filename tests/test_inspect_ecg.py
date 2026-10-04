"""
CardioSense - Visual ECG Inspection Test
Loads a real 12-lead ECG sample from PTB-XL and plots all 12 leads.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

# ---------------------------------------------------------
# Block 1: Define paths to our samples and metadata
# ---------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
samples_dir = project_root / "data" / "samples"
metadata_path = project_root / "data" / "raw" / "ptbxl_database.csv"

# ---------------------------------------------------------
# Block 2: Locate the downloaded .hea (header) files
# ---------------------------------------------------------
header_files = list(samples_dir.glob("**/*.hea"))
if not header_files:
    raise FileNotFoundError("No ECG sample files found in data/samples. Run download_ptbxl.py first.")

# Pick the first sample record (strip the .hea extension for wfdb)
first_header = header_files[0]
record_base_path = str(first_header.with_suffix(""))
ecg_id = int(first_header.stem.split("_")[0])  # Extract numeric ECG ID

# ---------------------------------------------------------
# Block 3: Read the binary waveform using the WFDB library
# ---------------------------------------------------------
# signals: numpy array of shape (samples, 12 leads) in millivolts (mV)
# fields: dictionary containing sampling frequency (fs), lead names, etc.
signals, fields = wfdb.rdsamp(record_base_path)

sampling_rate = fields["fs"]          # Sampling frequency: 100 Hz
lead_names = fields["sig_name"]       # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
num_samples, num_leads = signals.shape
time_axis = np.arange(num_samples) / sampling_rate  # Time in seconds (0 to 10s)

# ---------------------------------------------------------
# Block 4: Look up the clinical diagnosis from the metadata CSV
# ---------------------------------------------------------
df_meta = pd.read_csv(metadata_path, index_col="ecg_id")
patient_age = df_meta.loc[ecg_id, "age"]
patient_sex = "Male" if df_meta.loc[ecg_id, "sex"] == 0 else "Female"
scp_codes = df_meta.loc[ecg_id, "scp_codes"]

print("=" * 55)
print(f"Loaded ECG Record ID : {ecg_id}")
print(f"Patient Demographics : {patient_age} years old, {patient_sex}")
print(f"Cardiologist Labels  : {scp_codes}")
print(f"Matrix Shape         : {num_samples} samples x {num_leads} leads")
print(f"Duration             : {num_samples / sampling_rate:.1f} seconds at {sampling_rate} Hz")
print("=" * 55)

# ---------------------------------------------------------
# Block 5: Plot all 12 leads in a clean 4-row x 3-column clinical grid
# ---------------------------------------------------------
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 8), sharex=True)
axes = axes.flatten()  # Flatten 2D grid into 1D list of 12 subplots

for i, lead in enumerate(lead_names):
    ax = axes[i]
    # Plot voltage (mV) against time (seconds)
    ax.plot(time_axis, signals[:, i], color="navy", linewidth=1.0)
    ax.set_title(f"Lead {lead}", fontsize=11, fontweight="bold")
    ax.set_ylabel("mV", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

# Set common X-axis label on bottom plots
for ax in axes[-3:]:
    ax.set_xlabel("Time (seconds)", fontsize=10)

fig.suptitle(f"CardioSense - Raw 12-Lead ECG (ID: {ecg_id} | Diagnosis: {scp_codes})", fontsize=14, fontweight="bold")
plt.tight_layout()

# Display the interactive window
plt.show()