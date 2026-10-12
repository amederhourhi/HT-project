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
    Finds physiological fiducial landmarks around a single R-peak:
    - Base idx: Isoelectric PR reference (~60 ms before R)
    - Q-onset: Inflection / trough where QRS depolarization begins
    - S-peak: First local minimum following R-wave downstroke
    - J-point: End of S-wave upstroke returning toward baseline
    - ST point: Standard clinical measurement at J + 60 ms
    """
    n_samples = len(lead_signal)

    # 1. Isoelectric PR baseline reference: 60 ms before R-peak
    pr_offset = int(0.06 * fs)
    base_idx = max(0, r_peak_idx - pr_offset)
    baseline_voltage = lead_signal[base_idx]

    # 2. Dynamic Q-onset detection: scan backwards up to 60 ms before R
    pre_window = int(0.06 * fs)
    start_q = max(0, r_peak_idx - pre_window)
    segment_q = lead_signal[start_q : r_peak_idx + 1]

    # Locate Q-trough (minimum within search window)
    q_onset_rel = np.argmin(segment_q)
    q_onset_idx = start_q + q_onset_rel

    # If minimum is at the edge (flat PR segment), find steep upstroke onset
    if q_onset_idx == start_q and len(segment_q) > 2:
        diffs = np.diff(segment_q)
        max_diff = np.max(diffs)
        for i in range(len(segment_q) - 1, 0, -1):
            if diffs[i - 1] < 0.25 * max_diff:
                q_onset_idx = start_q + i
                break

    # 3. S-peak: first local minimum immediately following R downstroke (within 70 ms)
    post_window = int(0.07 * fs)
    s_search_end = min(n_samples, r_peak_idx + post_window)
    s_peak_idx = r_peak_idx

    for i in range(r_peak_idx + 1, s_search_end - 1):
        if lead_signal[i] <= lead_signal[i - 1] and lead_signal[i] <= lead_signal[i + 1]:
            s_peak_idx = i
            break

    # Fallback if no distinct trough found
    if s_peak_idx == r_peak_idx and s_search_end > r_peak_idx:
        s_peak_idx = r_peak_idx + np.argmin(lead_signal[r_peak_idx:s_search_end])

    # 4. J-point: end of S-wave upstroke (~30 ms after S-peak at 100 Hz)
    j_offset = int(0.03 * fs)
    j_point_idx = min(n_samples - 1, s_peak_idx + j_offset)

    # 5. Clinical ST measurement point: J + 60 ms
    st_offset = int(0.06 * fs)
    st_measure_idx = min(n_samples - 1, j_point_idx + st_offset)

    # ST deviation relative to PR isoelectric baseline (in mV)
    st_deviation_mv = float(lead_signal[st_measure_idx] - baseline_voltage)

    # QRS duration from Q-onset to J-point (in ms)
    qrs_duration_ms = (j_point_idx - q_onset_idx) / fs * 1000.0

    return {
        "base_idx": base_idx,
        "q_onset_idx": q_onset_idx,
        "r_peak_idx": r_peak_idx,
        "s_peak_idx": s_peak_idx,
        "j_point_idx": j_point_idx,
        "st_measure_idx": st_measure_idx,
        "st_deviation_mv": round(st_deviation_mv, 3),
        "qrs_duration_ms": round(float(qrs_duration_ms), 1),
    }


# ---------------------------------------------------------
# Block 2: Median Beat ST Deviation Across Multiple Beats
# ---------------------------------------------------------
def extract_lead_st_deviation(lead_signal, r_peaks, fs=100.0):
    """
    Calculates median ST deviation across all valid beats in a lead.
    The median provides noise-rejection against ectopic beats or baseline jitter.
    """
    if len(r_peaks) == 0:
        return 0.0

    beat_deviations = []
    # Skip edge beats
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
    - QRS duration (ms) from representative Lead II
    """
    num_leads = ecg_signals.shape[1]
    st_results = {}

    for i in range(num_leads):
        lead_name = lead_names[i]
        st_val = extract_lead_st_deviation(ecg_signals[:, i], r_peaks, fs=fs)
        st_results[lead_name] = st_val

    # Lead II representative beat for rhythm & conduction timing
    lead_ii_idx = lead_names.index("II") if "II" in lead_names else 1
    sample_landmarks = (
        delineate_beat_landmarks(ecg_signals[:, lead_ii_idx], r_peaks[1], fs=fs)
        if len(r_peaks) > 1
        else None
    )

    if sample_landmarks:
        qrs_duration_ms = sample_landmarks["qrs_duration_ms"]
    else:
        qrs_duration_ms = 90.0

    all_st_values = list(st_results.values())
    max_elevation = max(all_st_values)
    max_depression = min(all_st_values)

    return {
        "st_by_lead": st_results,
        "max_st_elevation_mv": round(float(max_elevation), 3),
        "max_st_depression_mv": round(float(max_depression), 3),
        "qrs_duration_ms": round(float(qrs_duration_ms), 1),
    }