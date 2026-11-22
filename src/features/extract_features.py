
"""
CardioSense - Clinical Feature Extraction.

Pipeline:
    Raw 12-lead ECG
      -> digital filtering
      -> signal quality assessment
      -> R-peak and HRV analysis
      -> fiducial landmarking
      -> ST feature aggregation
      -> ST measurement provenance
      -> feature quality contract

The original 23-feature classifier schema remains unchanged.

SQI V1 remains the default. SQI V3 is experimental and
must be selected explicitly.

Measurement provenance and quality warnings are metadata.
They do not currently alter risk predictions or clinical rules.
"""

import math

from src.dsp.filters import filter_ecg
from src.dsp.sqi import assess_ecg_quality
from src.dsp.sqi_v3 import assess_ecg_quality_v3

from src.features.r_peaks import (
    detect_r_peaks,
    compute_hrv_features,
)
from src.features.fiducials import analyze_12lead_features
from src.features.quality_contract import (
    build_feature_quality_contract,
)
from src.features.st_provenance import (
    build_st_provenance,
)


# ------------------------------------------------------------
# Lead definitions
# ------------------------------------------------------------

DEFAULT_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


# ------------------------------------------------------------
# Frozen Random Forest feature schema
# ------------------------------------------------------------
# IMPORTANT:
# Do not change this order or add quality metadata here.
# Existing checkpoints expect these exact 23 inputs.
# ------------------------------------------------------------

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


# Historical J+60 measurements.
# These are diagnostic columns, NOT RF inputs.

LEGACY_COLUMNS = [
    f"st_j60_{lead}"
    for lead in DEFAULT_LEADS
]


DIAGNOSTIC_COLUMNS = [
    "beats_used",
    "qrs_plausible",
    "leads_trusted",
]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _rejected(reason, sqi_result):
    """Return a consistent rejected-extraction payload."""

    return {
        "status": "rejected",
        "reason": reason,
        "sqi_details": sqi_result,
        "features": None,
    }


def _territory_max(st_by_lead, trusted_leads, leads):
    """
    Return the largest ST value among trusted territory leads.

    The 0.0 fallback is preserved for compatibility with
    historical classifier inputs.

    IMPORTANT:
    This fallback must not be interpreted as a genuine
    measured zero. Provenance reports its unavailability.
    """

    readings = [
        st_by_lead[lead]
        for lead in leads
        if lead in trusted_leads
    ]

    return max(readings) if readings else 0.0


def _check_provenance_consistency(features, provenance):
    """
    Compare aggregate provenance with the numeric feature values.

    The extractor may calculate some aggregates internally
    while the fiducial analyzer supplies global extrema.

    We must not claim that a lead contributed to an aggregate
    if provenance cannot reproduce the reported value.

    A mismatch is recorded separately and never modifies
    the historical classifier inputs.
    """

    aggregate_names = (
        "max_st_elevation_mv",
        "max_st_depression_mv",
        "st_anterior_max",
        "st_inferior_max",
        "st_lateral_max",
    )

    result = {}

    for name in aggregate_names:
        entry = provenance[name]

        if not entry["available"]:
            result[name] = {
                "consistent": False,
                "reason": "no_contributing_leads",
            }
            continue

        feature_value = features.get(name)
        provenance_value = entry.get("value_mv")

        try:
            reported = float(feature_value)
            calculated = float(provenance_value)

            consistent = (
                math.isfinite(reported)
                and math.isfinite(calculated)
                and math.isclose(
                    reported,
                    calculated,
                    rel_tol=0.0,
                    abs_tol=0.0011,
                )
            )

        except (TypeError, ValueError, OverflowError):
            consistent = False

        result[name] = {
            "consistent": consistent,
            "reason": (
                "matched"
                if consistent
                else "aggregate_value_mismatch"
            ),
        }

    return result


# ------------------------------------------------------------
# Main extraction pipeline
# ------------------------------------------------------------

def extract_clinical_features(
    raw_signals,
    fs=100.0,
    lead_names=None,
    *,
    sqi_version="v1",
):
    """
    Extract ECG biomarkers and quality metadata.

    Parameters
    ----------
    raw_signals:
        ECG array with shape (samples, leads).

    fs:
        Sampling frequency in Hz.

    lead_names:
        Lead labels corresponding to ECG columns.
        Defaults to the standard 12-lead ordering.

    sqi_version:
        "v1" - historical quality gate (default)
        "v3" - experimental artifact-aware quality gate

    Returns
    -------
    dict:
        Successful results include:

            status
            reason
            sqi_details
            st_details
            features
            st_provenance
            provenance_consistency
            quality_contract

        Rejected results preserve the historical rejection
        structure with features=None.

    Notes
    -----
    All classifier feature values retain their historical
    definitions.

    Quality metadata is advisory and is not yet consumed
    by the clinical triage engine.
    """

    if lead_names is None:
        lead_names = DEFAULT_LEADS.copy()

    if sqi_version not in ("v1", "v3"):
        raise ValueError(
            "sqi_version must be 'v1' or 'v3'"
        )

    # --------------------------------------------------------
    # Stage 1 - Digital signal processing
    # --------------------------------------------------------

    cleaned_signals = filter_ecg(
        raw_signals,
        fs=fs,
        lowcut=0.5,
        highcut=40.0,
    )

    # --------------------------------------------------------
    # Stage 2 - Signal Quality Index
    # --------------------------------------------------------

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

        # Confirmed technical failures still block
        # the full-record extraction pathway.
        #
        # Baseline drift alone is review-only under the
        # current experimental V3 policy.

        if technical_failures:

            return _rejected(
                (
                    "Technical signal failure in leads: "
                    + ", ".join(technical_failures)
                ),
                sqi_result,
            )

    if not sqi_result["is_acceptable"]:

        return _rejected(
            (
                "Signal quality unacceptable "
                f"(SQI: {sqi_result['overall_sqi']}, "
                f"passed: {sqi_result['passed_leads']})"
            ),
            sqi_result,
        )

    # --------------------------------------------------------
    # Stage 3 - R-peak detection and HRV
    # --------------------------------------------------------

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

        return _rejected(
            (
                "Insufficient heartbeats detected "
                "for reliable rhythm analysis"
            ),
            sqi_result,
        )

    hrv = compute_hrv_features(
        r_peaks,
        fs=fs,
    )

    # --------------------------------------------------------
    # Stage 4 - Fiducial landmark analysis
    # --------------------------------------------------------

    landmarks = analyze_12lead_features(
        cleaned_signals,
        r_peaks,
        lead_names=lead_names,
        fs=fs,
    )

    if landmarks["status"] == "rejected":

        return _rejected(
            (
                "Delineation failed: "
                f"{landmarks['reason']}"
            ),
            sqi_result,
        )

    st_by_lead = landmarks["st_by_lead"]
    st_details = landmarks["st_details"]

    # --------------------------------------------------------
    # Stage 5 - Identify trusted fiducial measurements
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Stage 6 - Anatomical territory aggregation
    # --------------------------------------------------------

    st_anterior_max = _territory_max(
        st_by_lead,
        trusted_leads,
        ["V1", "V2", "V3", "V4"],
    )

    st_inferior_max = _territory_max(
        st_by_lead,
        trusted_leads,
        ["II", "III", "AVF"],
    )

    st_lateral_max = _territory_max(
        st_by_lead,
        trusted_leads,
        ["I", "AVL", "V5", "V6"],
    )

    # --------------------------------------------------------
    # Stage 7 - Build the original feature vector
    # --------------------------------------------------------
    # Do not alter this layout before deliberately migrating
    # the training and inference schemas.
    # --------------------------------------------------------

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

        # Diagnostic information; excluded from FEATURE_COLUMNS.
        "beats_used": landmarks["beats_used"],
        "qrs_plausible": landmarks["qrs_plausible"],
        "leads_trusted": ",".join(trusted_leads),
    }

    # Per-lead J-point ST measurements.
    for lead in lead_names:

        features[f"st_{lead.upper()}"] = (
            st_by_lead.get(lead, 0.0)
        )

    # Legacy J+60 measurements for comparisons.
    # These remain excluded from classifier inputs.

    for lead in lead_names:

        reading = st_details.get(lead) or {}

        features[f"st_j60_{lead.upper()}"] = (
            reading.get("st_at_j60_mv")
        )

    # --------------------------------------------------------
    # Stage 8 - ST measurement provenance
    # --------------------------------------------------------
    # This metadata identifies the actual source lead(s)
    # of each ST maximum and minimum.
    #
    # Tied extrema preserve all tied source leads.
    #
    # No existing classifier value is changed.
    # --------------------------------------------------------

    st_provenance = build_st_provenance(
        st_by_lead,
        trusted_leads,
    )

    provenance_consistency = (
        _check_provenance_consistency(
            features,
            st_provenance,
        )
    )

    # --------------------------------------------------------
    # Stage 9 - Feature quality contract
    # --------------------------------------------------------

    quality_contract = build_feature_quality_contract(
        features,
        sqi_result,
        sqi_version=sqi_version,
        feature_columns=FEATURE_COLUMNS,
        st_provenance=st_provenance,
        provenance_consistency=provenance_consistency,
    )

    # --------------------------------------------------------
    # Stage 10 - Successful extraction
    # --------------------------------------------------------

    return {
        "status": "success",
        "reason": "OK",
        "sqi_details": sqi_result,
        "st_details": st_details,
        "features": features,
        "st_provenance": st_provenance,
        "provenance_consistency": provenance_consistency,
        "quality_contract": quality_contract,
    }
