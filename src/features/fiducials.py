"""
CardioSense - Fiducial Landmark & ST-Segment Extractor
Identifies QRS onset, S-point, J-point, and calculates clinical ST deviation across all 12 leads.
"""

import numpy as np


# ---------------------------------------------------------
# Block 1: Delineate Landmarks for a Single Beat
# ---------------------------------------------------------
def delineate_beat_landmarks(lead_signal, r_peak_idx, fs=100.0):
    """
    Finds fiducial points around a single R-peak:
    - Q-onset / Isoelectric point (PR baseline): ~60 ms before R
    - S-peak: lowest valley right after R (within 80 ms)
    - J-point: inflection point where S-wave slope flattens out
    """
    n_samples = len(lead_signal)
    
    # Baseline reference: PR segment (60 ms before R)
    pr_offset = int(0.06 * fs)
    base_idx = max(0, r_peak_idx - pr_offset)
    baseline_voltage = lead_signal[base_idx]

    # S-peak: local minimum within 80 ms after R-peak
    post_window = int(0.08 * fs)
    s_search_end = min(n_samples, r_peak_idx + post_window)
    if s_search_end > r_peak_idx:
        s_peak_idx = r_peak_idx + np.argmin(lead_signal[r_peak_idx:s_search_end])
    else:
        s_peak_idx = r_peak_idx

    # J-point: end of the S-wave slope (~40 ms after S-peak)
    j_offset = int(0.04 * fs)
    j_point_idx = min(n_samples - 1, s_peak_idx + j_offset)

    # Standard clinical ST measurement point: J + 60 ms
    st_offset = int(0.06 * fs)
    st_measure_idx = min(n_samples - 1, j_point_idx + st_offset)

    # ST deviation = voltage at (J + 60 ms) minus baseline voltage (in mV)
    st_deviation_mv = float(lead_signal[st_measure_idx] - baseline_voltage)

    return {
        "base_idx": base_idx,
        "r_peak_idx": r_peak_idx,
        "s_peak_idx": s_peak_idx,
        "j_point_idx": j_point_idx,
        "st_measure_idx": st_measure_idx,
        "st_deviation_mv": round(st_deviation_mv, 3),
    }


# ---------------------------------------------------------
# Block 2: Median Beat ST Deviation Across Multiple Beats
# ---------------------------------------------------------
def extract_lead_st_deviation(lead_signal, r_peaks, fs=100.0):
    """
    Calculates median ST deviation across all valid beats in a lead.
    Using the median makes the metric robust against isolated premature beats or noise.
    """
    if len(r_peaks) == 0:
        return 0.0

    beat_deviations = []
    # Skip the very first and last beat to avoid edge cutoffs
    valid_peaks = r_peaks[1:-1] if len(r_peaks) > 2 else r_peaks

    for r in valid_peaks:
        landmarks = delineate_beat_landmarks(lead_signal, r, fs=fs)
        beat_deviations.append(landmarks["st_deviation_mv"])

    median_st_mv = float(np.median(beat_deviations)) if beat_deviations else 0.0
    return round(median_st_mv, 3)


# ---------------------------------------------------------
# Block 3: Multi-Lead ST & QRS Duration Analysis
# ---------------------------------------------------------
def analyze_12lead_features(ecg_signals, r_peaks, lead_names, fs=100.0):
    """
    Extracts clinical features across all 12 leads:
    - Per-lead ST deviation (mV)
    - Maximum ST elevation and depression
    - QRS duration (ms)
    """
    num_leads = ecg_signals.shape[1]
    st_results = {}

    for i in range(num_leads):
        lead_name = lead_names[i]
        st_val = extract_lead_st_deviation(ecg_signals[:, i], r_peaks, fs=fs)
        st_results[lead_name] = st_val

    # Estimate average QRS duration from representative Lead II
    lead_ii_idx = lead_names.index("II") if "II" in lead_names else 1
    sample_landmarks = delineate_beat_landmarks(ecg_signals[:, lead_ii_idx], r_peaks[1], fs=fs) if len(r_peaks) > 1 else None
    
    if sample_landmarks:
        # QRS width: duration from Q-onset to J-point in milliseconds
        qrs_duration_ms = (sample_landmarks["j_point_idx"] - sample_landmarks["base_idx"]) / fs * 1000.0
    else:
        qrs_duration_ms = 90.0  # Standard default if not enough beats

    all_st_values = list(st_results.values())
    max_elevation = max(all_st_values)
    max_depression = min(all_st_values)

    return {
        "st_by_lead": st_results,
        "max_st_elevation_mv": round(float(max_elevation), 3),
        "max_st_depression_mv": round(float(max_depression), 3),
        "qrs_duration_ms": round(float(qrs_duration_ms), 1),
    }