"""
CardioSense - Visual ECG Inspection Test
Loads a real 12-lead ECG sample from PTB-XL and plots all 12 leads.
Saves the figure to docs/figures/01_raw_12lead_ecg.png.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

# ---------------------------------------------------------
# Block 1: Define paths to samples, metadata, and figures
# What it does: Sets up absolute path references relative to project root.
# How it does it: Uses pathlib.Path to construct cross-platform directory paths.
# ---------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
samples_dir = project_root / "data" / "samples"
metadata_path = project_root / "data" / "raw" / "ptbxl_database.csv"
figures_dir = project_root / "docs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Block 2: Locate the downloaded sample header files
# What it does: Finds the first available .hea record in data/samples.
# How it does it: Uses glob pattern matching on the samples directory.
# ---------------------------------------------------------
header_files = list(samples_dir.glob("**/*.hea"))
if not header_files:
    raise FileNotFoundError("No ECG sample files found in data/samples. Run download_ptbxl.py first.")

first_header = header_files[0]
record_base_path = str(first_header.with_suffix(""))
ecg_id = int(first_header.stem.split("_")[0])

# ---------------------------------------------------------
# Block 3: Read binary waveform data using WFDB
# What it does: Loads the raw voltage arrays and sampling parameters.
# How it does it: Calls wfdb.rdsamp() and computes a time vector in seconds.
# ---------------------------------------------------------
signals, fields = wfdb.rdsamp(record_base_path)
sampling_rate = fields["fs"]
lead_names = fields["sig_name"]
num_samples, num_leads = signals.shape
time_axis = np.arange(num_samples) / sampling_rate

# ---------------------------------------------------------
# Block 4: Look up clinical diagnosis (with cold-start fallback)
# What it does: Reads age, sex, and diagnostic codes if the metadata CSV exists.
# How it does it: Checks if ptbxl_database.csv exists before reading; falls back
#                 to placeholder strings on fresh clones where raw data is absent.
# ---------------------------------------------------------
if metadata_path.exists():
    df_meta = pd.read_csv(metadata_path, index_col="ecg_id")
    if ecg_id in df_meta.index:
        patient_age = df_meta.loc[ecg_id, "age"]
        patient_sex = "Male" if df_meta.loc[ecg_id, "sex"] == 0 else "Female"
        scp_codes = df_meta.loc[ecg_id, "scp_codes"]
    else:
        scp_codes = "Record ID not in metadata"
else:
    scp_codes = "Raw sample (metadata CSV not downloaded)"

# ---------------------------------------------------------
# Block 5: Plot all 12 leads in a standard 4x3 clinical grid
# What it does: Plots voltage over time for each lead and saves the image.
# How it does it: Iterates over lead axes in matplotlib, labels titles/axes,
#                 and exports the plot to docs/figures/01_raw_12lead_ecg.png.
# ---------------------------------------------------------
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 8), sharex=True)
axes = axes.flatten()

for i, lead in enumerate(lead_names):
    ax = axes[i]
    ax.plot(time_axis, signals[:, i], color="navy", linewidth=1.0)
    ax.set_title(f"Lead {lead}", fontsize=11, fontweight="bold")
    ax.set_ylabel("mV", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

for ax in axes[-3:]:
    ax.set_xlabel("Time (seconds)", fontsize=10)

fig.suptitle(f"CardioSense - Raw 12-Lead ECG (ID: {ecg_id} | Diagnosis: {scp_codes})", fontsize=13, fontweight="bold")
plt.tight_layout()

save_path = figures_dir / "01_raw_12lead_ecg.png"
plt.savefig(save_path, dpi=150)
print(f"[Saved figure to: {save_path.relative_to(project_root)}]")

plt.close()