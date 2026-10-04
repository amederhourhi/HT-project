"""
CardioSense - R-Peak & HRV Verification Test
Locates heartbeats, calculates Heart Rate & HRV, and plots red markers on each R-peak.
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
from src.features.r_peaks import detect_r_peaks, compute_hrv_features

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

# Use Lead II (gold standard rhythm lead in cardiology)
lead_ii_idx = lead_names.index("II") if "II" in lead_names else 1
lead_ii = cleaned_signals[:, lead_ii_idx]

# ---------------------------------------------------------
# Block 2: Detect R-peaks and compute HRV
# ---------------------------------------------------------
r_peaks = detect_r_peaks(lead_ii, fs=fs)
hrv_metrics = compute_hrv_features(r_peaks, fs=fs)

print("=" * 45)
print(f"Detected Heartbeats : {hrv_metrics['num_beats']} beats")
print(f"Calculated Heart Rate: {hrv_metrics['heart_rate_bpm']} BPM")
print(f"Mean RR Interval    : {hrv_metrics['mean_rr_sec']} s")
print(f"SDNN (Variability)  : {hrv_metrics['sdnn_ms']} ms")
print(f"RMSSD (Vagal Tone)  : {hrv_metrics['rmssd_ms']} ms")
print("=" * 45)

# ---------------------------------------------------------
# Block 3: Plot Lead II with detected R-peaks highlighted
# ---------------------------------------------------------
plt.figure(figsize=(13, 5))
plt.plot(time_axis, lead_ii, label="Cleaned Lead II", color="navy", linewidth=1.2)
plt.scatter(
    time_axis[r_peaks],
    lead_ii[r_peaks],
    color="crimson",
    s=60,
    zorder=5,
    label=f"Detected R-Peaks ({len(r_peaks)} beats)",
)

plt.title(
    f"CardioSense R-Peak Detector: {hrv_metrics['heart_rate_bpm']} BPM | SDNN: {hrv_metrics['sdnn_ms']} ms",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("mV", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()