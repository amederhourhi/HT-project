"""
CardioSense - Filter Verification Test
Compares Raw ECG vs Cleaned ECG to visually verify baseline wander removal.
Saves the figure to docs/figures/02_filter_comparison.png.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from src.dsp.filters import filter_ecg

# ---------------------------------------------------------
# Block 1: Load a raw sample record
# ---------------------------------------------------------
samples_dir = project_root / "data" / "samples"
figures_dir = project_root / "docs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

header_files = list(samples_dir.glob("**/*.hea"))
record_path = str(header_files[0].with_suffix(""))
raw_signals, fields = wfdb.rdsamp(record_path)
fs = fields["fs"]
lead_names = fields["sig_name"]
time_axis = np.arange(raw_signals.shape[0]) / fs

# ---------------------------------------------------------
# Block 2: Apply zero-phase filter (0.5 - 40 Hz)
# ---------------------------------------------------------
cleaned_signals = filter_ecg(raw_signals, fs=fs, lowcut=0.5, highcut=40.0)

# ---------------------------------------------------------
# Block 3: Plot Raw vs Filtered comparison on Lead II and Lead III
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 7), sharex=True)

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

# Save figure to docs/figures/
save_path = figures_dir / "02_filter_comparison.png"
plt.savefig(save_path, dpi=150)
print(f"[Saved figure to: {save_path.relative_to(project_root)}]")

plt.show()