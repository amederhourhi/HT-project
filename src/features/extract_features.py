
"""
CardioSense - Clinical Feature Extraction.

Combines ECG filtering, signal quality assessment, rhythm analysis,
fiducial measurements, and anatomical feature aggregation.
"""

from src.dsp.filters import filter_ecg
from src.dsp.sqi import assess_ecg_quality
from src.dsp.sqi_v3 import assess_ecg_quality_v3
from src.features.r_peaks import detect_r_peaks, compute_hrv_features
from src.features.fiducials import analyze_12lead_features


DEFAULT_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Keep this ordering unchanged until the RF model is retrained.
FEATURE_COLUMNS = [
    "heart_rate_bpm",
    "mean_rr_sec",
    "sdnn_ms",
    "rmssd_ms",
    "qrs_duration_ms",
    "max_st_elevation_mv",
    "max_st_depression_mv",
    "st_anterior_max",
    "st_inferior_max",
    "st_lateral_max",
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
    "overall_sqi",
]

LEGACY_COLUMNS = [
    f"st_j60_{lead}" for lead in DEFAULT_LEADS
]

DIAGNOSTIC_COLUMNS = [
    "beats_used",
    "qrs_plausible",
    "leads_trusted",
]


def extract_clinical_features(
    raw_signals,
    fs=100.0,
    lead_names=None,
    *,
    sqi_version="v1",
):
    """
    Extract clinical features from a 12-lead ECG.

    SQI V1 remains the default for backward compatibility.
    SQI V3 receives both filtered and raw ECG samples.
    """

    if lead_names is None:
        lead_names = DEFAULT_LEADS.copy()

    if sqi_version not in ("v1", "v3"):
        raise ValueError(
            "sqi_version must be 'v1' or 'v3'"
        )

    cleaned_signals = filter_ecg(
        raw_signals,
        fs=fs,
        lowcut=0.5,
        highcut=40.0,
    )

    # Select the signal quality evaluator.
    if sqi_version == "v1":
        sqi_result = assess_ecg_quality(
            cleaned_signals,
            lead_names=lead_names,
            accept_threshold=0.60,
        )

    else:
        sqi_result = assess_ecg_quality_v3(
            cleaned_signals,
            lead_names=lead_names,
            fs=fs,
            raw_signals=raw_signals,
            accept_threshold=0.60,
        )

        technical_failures = sqi_result[
            "technical_failure_leads"
        ]

        # Do not extract clinical measurements from a recording
        # with detected technical failures.
        if technical_failures:
            return {
                "status": "rejected",
                "reason": (
                    "Technical signal failure in leads: "
                    + ", ".join(technical_failures)
                ),
                "sqi_details": sqi_result,
                "features": None,
            }

    if not sqi_result["is_acceptable"]:
        return {
            "status": "rejected",
            "reason": (
                "Signal quality unacceptable "
                f"(SQI: {sqi_result['overall_sqi']}, "
                f"passed: {sqi_result['passed_leads']})"
            ),
            "sqi_details": sqi_result,
            "features": None,
        }

    # Rhythm analysis.
    lead_ii_idx = (
        lead_names.index("II")
        if "II" in lead_names
        else 1
    )

    r_peaks = detect_r_peaks(
        cleaned_signals[:, lead_ii_idx],
        fs=fs,
    )

    if len(r_peaks) < 2:
        return {
            "status": "rejected",
            "reason": (
                "Insufficient heartbeats detected "
                "for reliable rhythm analysis"
            ),
            "sqi_details": sqi_result,
            "features": None,
        }

    hrv = compute_hrv_features(
        r_peaks,
        fs=fs,
    )

    # Fiducial and ST-segment measurements.
    landmarks = analyze_12lead_features(
        cleaned_signals,
        r_peaks,
        lead_names=lead_names,
        fs=fs,
    )

    if landmarks["status"] == "rejected":
        return {
            "status": "rejected",
            "reason": (
                "Delineation failed: "
                f"{landmarks['reason']}"
            ),
            "sqi_details": sqi_result,
            "features": None,
        }

    st_by_lead = landmarks["st_by_lead"]
    st_details = landmarks["st_details"]

    trusted_leads = [
        lead
        for lead in lead_names
        if (
            lead in st_by_lead
            and st_details.get(lead, {}).get(
                "plausible", False
            )
        )
    ]

    def territory_max(leads):
        readings = [
            st_by_lead[lead]
            for lead in leads
            if lead in trusted_leads
        ]

        return max(readings) if readings else 0.0

    st_anterior_max = territory_max(
        ["V1", "V2", "V3", "V4"]
    )

    st_inferior_max = territory_max(
        ["II", "III", "AVF"]
    )

    st_lateral_max = territory_max(
        ["I", "AVL", "V5", "V6"]
    )

    # Preserve the existing RF feature schema.
    features = {
        "heart_rate_bpm": hrv["heart_rate_bpm"],
        "mean_rr_sec": hrv["mean_rr_sec"],
        "sdnn_ms": hrv["sdnn_ms"],
        "rmssd_ms": hrv["rmssd_ms"],
        "qrs_duration_ms": landmarks["qrs_duration_ms"],
        "max_st_elevation_mv": landmarks[
            "max_st_elevation_mv"
        ],
        "max_st_depression_mv": landmarks[
            "max_st_depression_mv"
        ],
        "st_anterior_max": round(
            float(st_anterior_max), 3
        ),
        "st_inferior_max": round(
            float(st_inferior_max), 3
        ),
        "st_lateral_max": round(
            float(st_lateral_max), 3
        ),
        "overall_sqi": sqi_result["overall_sqi"],
        "beats_used": landmarks["beats_used"],
        "qrs_plausible": landmarks["qrs_plausible"],
        "leads_trusted": ",".join(trusted_leads),
    }

    # J-point ST deviation.
    for lead in lead_names:
        features[f"st_{lead.upper()}"] = (
            st_by_lead.get(lead, 0.0)
        )

    # Legacy J+60 values, excluded from classifier inputs.
    for lead in lead_names:
        reading = st_details.get(lead) or {}

        features[f"st_j60_{lead.upper()}"] = (
            reading.get("st_at_j60_mv")
        )

    return {
        "status": "success",
        "reason": "OK",
        "sqi_details": sqi_result,
        "st_details": st_details,
        "features": features,
    }
