"""
CardioSense - End-to-End Clinical Feature Aggregator
Integrates filtering, SQI quality control, rhythm detection, and 12-lead
fiducial landmarking into a single structured clinical feature vector.
"""

import numpy as np

# Import our modular DSP and feature extractors
from src.dsp.filters import filter_ecg
from src.dsp.sqi import assess_ecg_quality
from src.features.r_peaks import detect_r_peaks, compute_hrv_features
from src.features.fiducials import analyze_12lead_features


# ---------------------------------------------------------
# Block 1: Standardized Feature Column Names
# ---------------------------------------------------------
# The exact ordered list of clinical features our ML model will expect
FEATURE_COLUMNS = [
    # Rhythm & Autonomic features
    "heart_rate_bpm",
    "mean_rr_sec",
    "sdnn_ms",
    "rmssd_ms",
    # Ventricular conduction
    "qrs_duration_ms",
    # Global acute ischemia markers
    "max_st_elevation_mv",
    "max_st_depression_mv",
    # Per-lead ST deviation (12 leads)
    "st_I",
    "st_II",
    "st_III",
    "st_AVR",
    "st_AVL",
    "st_AVF",
    "st_V1",
    "st_V2",
    "st_V3",
    "st_V4",
    "st_V5",
    "st_V6",
    # Signal quality score
    "overall_sqi",
]


# ---------------------------------------------------------
# Block 2: Master Feature Extraction Function
# ---------------------------------------------------------
def extract_clinical_features(raw_signals, fs=100.0, lead_names=None):
    """
    Processes a raw 12-lead ECG matrix (samples x leads).
    Returns:
        dict: containing 'status' ('success' or 'rejected'), 'features' dict, and 'sqi_details'.
    """
    if lead_names is None:
        lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # 1. Zero-phase bandpass filter (0.5 - 40 Hz)
    cleaned_signals = filter_ecg(raw_signals, fs=fs, lowcut=0.5, highcut=40.0)

    # 2. Automated Signal Quality Gate (SQI)
    sqi_result = assess_ecg_quality(cleaned_signals, lead_names=lead_names, accept_threshold=0.60)
    
    if not sqi_result["is_acceptable"]:
        return {
            "status": "rejected",
            "reason": f"Signal quality unacceptable (SQI: {sqi_result['overall_sqi']}, passed: {sqi_result['passed_leads']})",
            "sqi_details": sqi_result,
            "features": None,
        }

    # 3. Detect R-peaks on Lead II (standard clinical rhythm lead)
    lead_ii_idx = lead_names.index("II") if "II" in lead_names else 1
    r_peaks = detect_r_peaks(cleaned_signals[:, lead_ii_idx], fs=fs)

    # If rhythm is too aberrant or no beats found
    if len(r_peaks) < 2:
        return {
            "status": "rejected",
            "reason": "Insufficient heartbeats detected for reliable rhythm analysis",
            "sqi_details": sqi_result,
            "features": None,
        }

    # 4. Extract Heart Rate & HRV metrics
    hrv = compute_hrv_features(r_peaks, fs=fs)

    # 5. Extract 12-lead fiducial landmarks (ST deviations & QRS duration)
    landmarks = analyze_12lead_features(cleaned_signals, r_peaks, lead_names=lead_names, fs=fs)

    # 6. Assemble into structured clinical feature dictionary
    features = {
        "heart_rate_bpm": hrv["heart_rate_bpm"],
        "mean_rr_sec": hrv["mean_rr_sec"],
        "sdnn_ms": hrv["sdnn_ms"],
        "rmssd_ms": hrv["rmssd_ms"],
        "qrs_duration_ms": landmarks["qrs_duration_ms"],
        "max_st_elevation_mv": landmarks["max_st_elevation_mv"],
        "max_st_depression_mv": landmarks["max_st_depression_mv"],
        "overall_sqi": sqi_result["overall_sqi"],
    }

    # Add each lead's ST deviation with standard naming (e.g., 'st_V2')
    for lead in lead_names:
        key = f"st_{lead.upper()}"
        features[key] = landmarks["st_by_lead"].get(lead, 0.0)

    return {
        "status": "success",
        "reason": "OK",
        "sqi_details": sqi_result,
        "features": features,
    }