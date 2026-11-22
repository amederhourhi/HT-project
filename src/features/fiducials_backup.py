"""
CardioSense - R-Peak Detection and Heart Rate Variability (HRV)
Detects QRS complexes and calculates clinical metrics: Heart Rate, RR intervals, SDNN, and RMSSD.
"""

import numpy as np
from scipy import signal

#Energy Transformation (Derivative + Square)
def qrs_energy(lead_signal):
    """
    Amplifies steep QRS slopes and suppresses flat P/T waves.
    Computes first difference (derivative), then squares the amplitude.
    """
    diff = np.diff(lead_signal, prepend=lead_signal[0])
    squared = diff ** 2
    return squared

# Detect R-Peaks with Refractory Period
def detect_r_peaks(lead_signal, fs=100.0):
    """Detect candidate QRS peaks and enforce refractory spacing *after* refinement.

    Note: this repairs the spacing invariant; it is not a validated detector
    of all QRS morphologies and may still require a different lead/polarity.
    """
    lead_signal = np.asarray(lead_signal, dtype=float)
    if lead_signal.ndim != 1 or not np.all(np.isfinite(lead_signal)):
        raise ValueError("lead_signal must be a finite one-dimensional array")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if len(lead_signal) < 3:
        return np.array([], dtype=int)

    energy = qrs_energy(lead_signal)
    min_distance = max(1, int(np.ceil(0.25 * fs)))
    threshold = 0.3 * np.percentile(energy, 98)
    candidates, properties = signal.find_peaks(
        energy, height=threshold, distance=min_distance
    )
    search_radius = max(1, int(round(0.05 * fs)))

    refined = {}
    for p, strength in zip(candidates, properties["peak_heights"]):
        start = max(0, int(p) - search_radius)
        end = min(len(lead_signal), int(p) + search_radius + 1)
        peak = start + int(np.argmax(np.abs(lead_signal[start:end])))
        refined[peak] = max(float(strength), refined.get(peak, -np.inf))

    # Moving a candidate up to 50 ms can violate find_peaks' distance
    # constraint; restore that invariant with energy-priority suppression.
    selected = []
    for peak in sorted(refined, key=lambda p: (-refined[p], p)):
        if all(abs(peak - kept) >= min_distance for kept in selected):
            selected.append(peak)
    return np.array(sorted(selected), dtype=int)
#Extract Clinical Heart Rate & HRV Features
def compute_hrv_features(r_peaks, fs=100.0):
    """
    Computes standard clinical rhythm metrics from detected R-peaks:
    - Heart Rate (BPM)
    - Mean RR Interval (seconds)
    - SDNN: Standard deviation of RR intervals (ms)
    - RMSSD: Root mean square of successive differences (ms)
    """
    if len(r_peaks) < 2:
        return {
            "num_beats": len(r_peaks),
            "heart_rate_bpm": 0.0,
            "mean_rr_sec": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
        }

    # RR intervals in seconds and milliseconds
    rr_intervals_sec = np.diff(r_peaks) / fs
    rr_intervals_ms = rr_intervals_sec * 1000.0

    mean_rr = float(np.mean(rr_intervals_sec))
    heart_rate_bpm = float(60.0 / mean_rr) if mean_rr > 0 else 0.0
    sdnn_ms = float(np.std(rr_intervals_ms))
    successive_diffs = np.diff(rr_intervals_ms)
    rmssd_ms = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0

    return {
        "num_beats": len(r_peaks),
        "heart_rate_bpm": round(heart_rate_bpm, 1),
        "mean_rr_sec": round(mean_rr, 3),
        "sdnn_ms": round(sdnn_ms, 1),
        "rmssd_ms": round(rmssd_ms, 1),
    }