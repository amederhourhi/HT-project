"""
CardioSense - End-to-End Clinical Feature Aggregator
Integrates filtering, SQI quality control, rhythm detection, 12-lead fiducial
landmarking, and anatomical coronary territory groupings into a structured clinical feature vector.
"""

import numpy as np

# Import modular DSP and feature extractors
from src.dsp.filters import filter_ecg
from src.dsp.sqi import assess_ecg_quality
from src.features.r_peaks import detect_r_peaks, compute_hrv_features
from src.features.fiducials import analyze_12lead_features


# ---------------------------------------------------------
# Block 1: Standardized Feature Column Names
# ---------------------------------------------------------
# Ordered list of clinical biomarkers for the interpretable classifier (23 features)
#
# ST convention: every ST column below is measured at the J point against the
# isoelectric PR segment, which is how the AHA/ACC criteria this project applies
# define ST deviation. The legacy J+60 readings are still extracted, but only as
# the trailing LEGACY_COLUMNS - deliberately outside FEATURE_COLUMNS, so
# switching the measurement convention does not silently change the trained
# classifier's input contract part-way through the hardening sprint.
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
    # Coronary Artery Territory markers (Contiguous ischemic burden)
    "st_anterior_max",  # LAD territory (V1, V2, V3, V4)
    "st_inferior_max",  # RCA territory (II, III, aVF)
    "st_lateral_max",   # LCx territory (I, aVL, V5, V6)
    # Per-lead ST deviation at the J point (12 leads)
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

# Audit-only columns. Emitted in the dataset so the new convention can be
# compared against the old one and a suspect delineation can be traced, but they
# are not inputs to the interpretable classifier.
LEGACY_COLUMNS = [f"st_j60_{lead}" for lead in
                  ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]]
DIAGNOSTIC_COLUMNS = ["beats_used", "qrs_plausible"]


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

    # 2. Automated Signal Quality Gatekeeper (SQI)
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

    if landmarks["status"] == "rejected":
        return {
            "status": "rejected",
            "reason": f"Delineation failed: {landmarks['reason']}",
            "sqi_details": sqi_result,
            "features": None,
        }

    st_by_lead = landmarks["st_by_lead"]          # ST at the J point
    st_details = landmarks["st_details"]          # both conventions + plausibility

    # 6. Anatomical Coronary Artery Territory Maximums
    # Only leads the delineator vouches for can contribute. Without that filter a
    # single failed lead would report a plausible-looking territory maximum, and
    # the deterministic guardrails would escalate on a measurement artefact.
    def territory_max(leads):
        trusted = [st_by_lead[ld] for ld in leads
                   if ld in st_by_lead and st_details.get(ld, {}).get("plausible", False)]
        return max(trusted) if trusted else 0.0

    # LAD (Left Anterior Descending) territory: V1-V4
    st_anterior_max = territory_max(["V1", "V2", "V3", "V4"])
    # RCA (Right Coronary Artery) territory: II, III, aVF
    st_inferior_max = territory_max(["II", "III", "AVF"])
    # LCx (Left Circumflex) territory: I, aVL, V5, V6
    st_lateral_max = territory_max(["I", "AVL", "V5", "V6"])

    # 7. Assemble into structured clinical feature dictionary
    features = {
        "heart_rate_bpm": hrv["heart_rate_bpm"],
        "mean_rr_sec": hrv["mean_rr_sec"],
        "sdnn_ms": hrv["sdnn_ms"],
        "rmssd_ms": hrv["rmssd_ms"],
        "qrs_duration_ms": landmarks["qrs_duration_ms"],
        "max_st_elevation_mv": landmarks["max_st_elevation_mv"],
        "max_st_depression_mv": landmarks["max_st_depression_mv"],
        "st_anterior_max": round(float(st_anterior_max), 3),
        "st_inferior_max": round(float(st_inferior_max), 3),
        "st_lateral_max": round(float(st_lateral_max), 3),
        "overall_sqi": sqi_result["overall_sqi"],
        # Delineation diagnostics
        "beats_used": landmarks["beats_used"],
        "qrs_plausible": landmarks["qrs_plausible"],
    }

    # Per-lead ST deviation at the J point (the guideline convention)
    for lead in lead_names:
        features[f"st_{lead.upper()}"] = st_by_lead.get(lead, 0.0)

    # Legacy J+60 readings, kept for convention comparison only
    for lead in lead_names:
        reading = st_details.get(lead) or {}
        features[f"st_j60_{lead.upper()}"] = reading.get("st_at_j60_mv")

    return {
        "status": "success",
        "reason": "OK",
        "sqi_details": sqi_result,
        "st_details": st_details,
        "features": features,
    }