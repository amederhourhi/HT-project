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
# Block 1: Define paths to our samples and metadata
# ---------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
samples_dir = project_root / "data" / "samples"
metadata_path = project_root / "data" / "raw" / "ptbxl_database.csv"
figures_dir = project_root / "docs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Block 2: Locate the downloaded .hea (header) files
# ---------------------------------------------------------
header_files = list(samples_dir.glob("**/*.hea"))
if not header_files:
    raise FileNotFoundError("No ECG sample files found in data/samples. Run download_ptbxl.py first.")

first_header = header_files[0]
record_base_path = str(first_header.with_suffix(""))
ecg_id = int(first_header.stem.split("_")[0])

# ---------------------------------------------------------
# Block 3: Read the binary waveform using WFDB
# ---------------------------------------------------------
signals, fields = wfdb.rdsamp(record_base_path)
sampling_rate = fields["fs"]
lead_names = fields["sig_name"]
num_samples, num_leads = signals.shape
time_axis = np.arange(num_samples) / sampling_rate

# ---------------------------------------------------------
# Block 4: Look up clinical diagnosis from metadata
# ---------------------------------------------------------
df_meta = pd.read_csv(metadata_path, index_col="ecg_id")
patient_age = df_meta.loc[ecg_id, "age"]
patient_sex = "Male" if df_meta.loc[ecg_id, "sex"] == 0 else "Female"
scp_codes = df_meta.loc[ecg_id, "scp_codes"]

# ---------------------------------------------------------
# Block 5: Plot all 12 leads in a clean 4x3 clinical grid
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

fig.suptitle(f"CardioSense - Raw 12-Lead ECG (ID: {ecg_id} | Diagnosis: {scp_codes})", fontsize=14, fontweight="bold")
plt.tight_layout()

# Save figure to docs/figures/
save_path = figures_dir / "01_raw_12lead_ecg.png"
plt.savefig(save_path, dpi=150)
print(f"[Saved figure to: {save_path.relative_to(project_root)}]")

plt.show()