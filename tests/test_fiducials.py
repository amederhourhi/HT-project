"""
CardioSense - Fiducial Landmark Verification Test
Plots a zoomed-in heartbeat with PR-baseline, R, S, J-point, and ST-measurement point.
Saves the plot to docs/figures/ for clinical documentation.
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
from src.features.r_peaks import detect_r_peaks
from src.features.fiducials import delineate_beat_landmarks, analyze_12lead_features

# ---------------------------------------------------------
# Block 1: Load and filter sample ECG
# ---------------------------------------------------------
samples_dir = project_root / "data" / "samples"
header_files = list(samples_dir.glob("**/*.hea"))
record_path = str(header_files[0].with_suffix(""))

raw_signals, fields = wfdb.rdsamp(record_path)
fs = fields["fs"]
lead_names = fields["sig_name"]
time_axis = np.arange(raw_signals.shape[0]) / fs

cleaned_signals = filter_ecg(raw_signals, fs=fs)
lead_ii_idx = lead_names.index("II") if "II" in lead_names else 1
lead_ii = cleaned_signals[:, lead_ii_idx]

# ---------------------------------------------------------
# Block 2: Detect peaks and analyze all 12 leads
# ---------------------------------------------------------
r_peaks = detect_r_peaks(lead_ii, fs=fs)
analysis = analyze_12lead_features(cleaned_signals, r_peaks, lead_names, fs=fs)

print("=" * 50)
print(f"Estimated QRS Duration : {analysis['qrs_duration_ms']} ms")
print(f"Max ST Elevation       : {analysis['max_st_elevation_mv']} mV")
print(f"Max ST Depression      : {analysis['max_st_depression_mv']} mV")
print("\nST Deviation by Lead (mV):")
for lead, val in analysis["st_by_lead"].items():
    print(f"  {lead:5s}: {val:+.3f} mV")
print("=" * 50)

# ---------------------------------------------------------
# Block 3: Delineate landmarks for a single representative beat (Beat 2)
# ---------------------------------------------------------
target_beat_r = r_peaks[2]
lm = delineate_beat_landmarks(lead_ii, target_beat_r, fs=fs)

# Zoom in on window +/- 250 ms around the beat
start_sample = target_beat_r - int(0.25 * fs)
end_sample = target_beat_r + int(0.40 * fs)
beat_time = time_axis[start_sample:end_sample]
beat_signal = lead_ii[start_sample:end_sample]

# ---------------------------------------------------------
# Block 4: Plot Zoomed Heartbeat with Landmark Markers
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(beat_time, beat_signal, color="black", linewidth=2.0, label="ECG Beat")

# Landmark markers
plt.scatter(time_axis[lm["base_idx"]], lead_ii[lm["base_idx"]], color="blue", s=70, zorder=5, label="PR Baseline")
plt.scatter(time_axis[lm["r_peak_idx"]], lead_ii[lm["r_peak_idx"]], color="red", s=70, zorder=5, label="R-Peak")
plt.scatter(time_axis[lm["s_peak_idx"]], lead_ii[lm["s_peak_idx"]], color="purple", s=70, zorder=5, label="S-Peak")
plt.scatter(time_axis[lm["j_point_idx"]], lead_ii[lm["j_point_idx"]], color="darkorange", s=70, zorder=5, label="J-Point")
plt.scatter(time_axis[lm["st_measure_idx"]], lead_ii[lm["st_measure_idx"]], color="green", s=70, zorder=5, label="ST (J + 60ms)")

# Draw baseline reference line
plt.axhline(lead_ii[lm["base_idx"]], color="blue", linestyle=":", alpha=0.6, label="Isoelectric Level")

plt.title(f"CardioSense Landmark Detection (Beat 2) | ST Deviation: {lm['st_deviation_mv']:+.3f} mV", fontsize=12, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("mV", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

# Save to docs/figures/
figures_dir = project_root / "docs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
save_path = figures_dir / "05_beat_landmarks.png"
plt.savefig(save_path, dpi=150)
print(f"\n[Saved verification plot to: {save_path.relative_to(project_root)}]")

plt.show()