"""
CardioSense - R-Peak Detection and Heart Rate Variability (HRV)
Detects QRS complexes and calculates clinical metrics: Heart Rate, RR intervals, SDNN, and RMSSD.
"""

import numpy as np
from scipy import signal


# ---------------------------------------------------------
# Block 1: Energy Transformation (Derivative + Square)
# ---------------------------------------------------------
def qrs_energy(lead_signal):
    """
    Amplifies steep QRS slopes and suppresses flat P/T waves.
    Computes first difference (derivative), then squares the amplitude.
    """
    diff = np.diff(lead_signal, prepend=lead_signal[0])
    squared = diff ** 2
    return squared


# ---------------------------------------------------------
# Block 2: Detect R-Peaks with Refractory Period
# ---------------------------------------------------------
def detect_r_peaks(lead_signal, fs=100.0):
    """
    Locates R-peak sample indices using adaptive peak finding.
    Enforces a physiological refractory lockout of 250 ms (max ~240 BPM).
    """
    energy = qrs_energy(lead_signal)
    
    # Minimum distance between beats in samples (250 ms)
    min_distance = int(0.25 * fs)
    
    # Adaptive threshold: 30% of 98th percentile energy (resistant to outliers)
    threshold = 0.3 * np.percentile(energy, 98)
    
    # Find candidate peaks above threshold
    peaks, _ = signal.find_peaks(energy, height=threshold, distance=min_distance)
    
    # Snap to the exact local maximum in the actual voltage signal
    refined_peaks = []
    search_radius = int(0.05 * fs)  # Search +/- 50 ms
    
    for p in peaks:
        start = max(0, p - search_radius)
        end = min(len(lead_signal), p + search_radius + 1)
        actual_peak = start + np.argmax(np.abs(lead_signal[start:end]))
        refined_peaks.append(actual_peak)
        
    return np.array(sorted(list(set(refined_peaks))))


# ---------------------------------------------------------
# Block 3: Extract Clinical Heart Rate & HRV Features
# ---------------------------------------------------------
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
    
    # Successive differences between consecutive RR intervals
    successive_diffs = np.diff(rr_intervals_ms)
    rmssd_ms = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0

    return {
        "num_beats": len(r_peaks),
        "heart_rate_bpm": round(heart_rate_bpm, 1),
        "mean_rr_sec": round(mean_rr, 3),
        "sdnn_ms": round(sdnn_ms, 1),
        "rmssd_ms": round(rmssd_ms, 1),
    }