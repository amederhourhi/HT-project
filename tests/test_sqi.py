"""
CardioSense - SQI Verification Test
Tests quality control and saves visual proof to docs/figures/03_sqi_assessment.png.
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
from src.dsp.sqi import assess_ecg_quality, compute_lead_sqi

# ---------------------------------------------------------
# Block 1: Load and filter real sample record
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

cleaned_signals = filter_ecg(raw_signals, fs=fs)

# ---------------------------------------------------------
# Block 2: Test Real ECG and Simulate Disconnected Lead
# ---------------------------------------------------------
result_real = assess_ecg_quality(cleaned_signals, lead_names=lead_names)

# Corrupt Lead V2 with flatline
corrupted_signals = cleaned_signals.copy()
v2_idx = lead_names.index("V2") if "V2" in lead_names else 7
corrupted_signals[:, v2_idx] = 0.0
v2_sqi_bad = compute_lead_sqi(corrupted_signals[:, v2_idx])

# ---------------------------------------------------------
# Block 3: Visual Quality Control Figure
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 8))

# Subplot 1: Per-Lead SQI Bar Chart
leads = list(result_real["lead_scores"].keys())
scores = [result_real["lead_scores"][l]["sqi"] for l in leads]
colors = ["forestgreen" if s >= 0.6 else "crimson" for s in scores]

ax1.bar(leads, scores, color=colors, edgecolor="black", width=0.6)
ax1.axhline(0.6, color="crimson", linestyle="--", linewidth=1.5, label="Rejection Threshold (0.6)")
ax1.set_ylim(0, 1.1)
ax1.set_title(f"12-Lead Quality Index (Overall SQI: {result_real['overall_sqi']} | Status: ACCEPTED)", fontsize=12, fontweight="bold")
ax1.set_ylabel("SQI Score (0 to 1)", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="lower right")

# Subplot 2: Normal vs Disconnected Flatline
ax2.plot(time_axis, cleaned_signals[:, v2_idx], color="navy", label=f"Normal Lead V2 (SQI = {result_real['lead_scores']['V2']['sqi']})", linewidth=1.2)
ax2.plot(time_axis, corrupted_signals[:, v2_idx], color="crimson", linestyle="--", label=f"Simulated Disconnected Lead V2 (SQI = {v2_sqi_bad} -> REJECTED)", linewidth=2.0)
ax2.set_title("Artifact Detection: Disconnected Electrode vs Valid Signal", fontsize=12, fontweight="bold")
ax2.set_xlabel("Time (seconds)", fontsize=10)
ax2.set_ylabel("mV", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right")

plt.tight_layout()

# Save figure
save_path = figures_dir / "04_sqi_assessment.png"
plt.savefig(save_path, dpi=150)
print(f"[Saved figure to: {save_path.relative_to(project_root)}]")

# Assertions
assert result_real["is_acceptable"] is True
assert v2_sqi_bad == 0.0

plt.show()