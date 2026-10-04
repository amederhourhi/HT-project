"""
CardioSense - Filter Verification Test
Compares Raw ECG vs Cleaned ECG to visually verify baseline wander removal.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------
# Block 1: Add project root to Python search path so it finds 'src'
# ---------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import wfdb

# Now we can safely import our custom filter
from src.dsp.filters import filter_ecg

# ---------------------------------------------------------
# Block 2: Load a raw sample record from data/samples
# ---------------------------------------------------------
samples_dir = project_root / "data" / "samples"
header_files = list(samples_dir.glob("**/*.hea"))

if not header_files:
    raise FileNotFoundError("No samples found. Run download_ptbxl.py first.")

record_path = str(header_files[0].with_suffix(""))
raw_signals, fields = wfdb.rdsamp(record_path)
fs = fields["fs"]
lead_names = fields["sig_name"]
time_axis = np.arange(raw_signals.shape[0]) / fs

# ---------------------------------------------------------
# Block 3: Apply our zero-phase filter (0.5 - 40 Hz)
# ---------------------------------------------------------
cleaned_signals = filter_ecg(raw_signals, fs=fs, lowcut=0.5, highcut=40.0)

# ---------------------------------------------------------
# Block 4: Plot Raw vs Filtered comparison on Lead II and Lead III
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 7), sharex=True)

# Find Lead II and Lead III indexes
lead_ii_idx = lead_names.index("II") if "II" in lead_names else 1
lead_iii_idx = lead_names.index("III") if "III" in lead_names else 2

# Subplot 1: Lead II
ax1.plot(time_axis, raw_signals[:, lead_ii_idx], label="Raw Signal (Drifting)", color="crimson", alpha=0.5, linewidth=1.2)
ax1.plot(time_axis, cleaned_signals[:, lead_ii_idx], label="Cleaned Signal (0.5 - 40 Hz)", color="navy", linewidth=1.2)
ax1.set_title("Lead II: Raw vs Zero-Phase Filtered ECG", fontsize=12, fontweight="bold")
ax1.set_ylabel("mV", fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right")

# Subplot 2: Lead III
ax2.plot(time_axis, raw_signals[:, lead_iii_idx], label="Raw Signal (Drifting)", color="crimson", alpha=0.5, linewidth=1.2)
ax2.plot(time_axis, cleaned_signals[:, lead_iii_idx], label="Cleaned Signal (0.5 - 40 Hz)", color="navy", linewidth=1.2)
ax2.set_title("Lead III: Raw vs Zero-Phase Filtered ECG", fontsize=12, fontweight="bold")
ax2.set_ylabel("mV", fontsize=10)
ax2.set_xlabel("Time (seconds)", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()