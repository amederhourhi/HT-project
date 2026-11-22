
"""EXPERIMENTAL ablation: V2 corroborated morphological onset + V1 energy offset.

Not clinically validated. Separates V2 onset benefit from its offset overshoot.
Only 100/500 Hz input. No changes to live production pipeline.
"""

import numpy as np
from scipy.signal import resample_poly

from src.features import fiducials as old
from src.features.fiducials_qrs_v2 import (
    _candidate_boundaries,
    ANALYSIS_FS,
)


def analyze_12lead_features(ecg_signals, r_peaks, lead_names, fs=500.0):
    signals = np.asarray(ecg_signals, dtype=float)

    if signals.ndim != 2 or signals.shape[1] != len(lead_names):
        raise ValueError(
            "Expected 2-D samples x leads with matching names"
        )

    if fs not in (100, 500):
        raise ValueError("Only 100 Hz and 500 Hz are supported")

    if not np.isfinite(signals).all():
        return {"status": "rejected", "reason": "Non-finite input"}

    peaks = np.asarray(r_peaks, dtype=int)

    if (
        peaks.ndim != 1
        or len(peaks) < 2
        or np.any(np.diff(peaks) <= 0)
    ):
        return {"status": "rejected", "reason": "Invalid R peaks"}

    median = []
    counts = []
    r_idx = None

    for col in range(signals.shape[1]):
        beat, idx, n = old.build_median_beat(
            signals[:, col],
            peaks,
            fs=fs,
        )

        if beat is None:
            return {
                "status": "rejected",
                "reason": "Median beat unavailable",
            }

        median.append(beat)
        counts.append(n)

        if r_idx is None:
            r_idx = idx

    median = np.stack(median)

    if fs == 100:
        median = resample_poly(
            median,
            5,
            1,
            axis=1,
        )
        r_idx *= 5

    # V2 onset + original energy-based offset.
    entries, v2_bounds = _candidate_boundaries(
        median,
        r_idx,
        ANALYSIS_FS,
    )

    old_on, old_off = old.find_global_qrs_bounds(
        median,
        r_idx,
        fs=ANALYSIS_FS,
    )

    if not v2_bounds or old_on is None:
        return {
            "status": "rejected",
            "reason": "Boundary delineation failed",
        }

    onset = int(v2_bounds[0])
    offset = int(old_off)

    if not (0 <= onset < offset < median.shape[1]):
        return {
            "status": "rejected",
            "reason": "Inconsistent hybrid boundaries",
        }

    duration = (
        (offset - onset) * 1000.0 / ANALYSIS_FS
    )

    st = {}
    details = {}

    for name, beat in zip(lead_names, median):
        baseline, base_idx = old.estimate_isoelectric_baseline(
            beat,
            onset,
            fs=ANALYSIS_FS,
        )

        reading = old.measure_lead_st(
            beat,
            baseline,
            offset,
            fs=ANALYSIS_FS,
        )

        value = float(reading["st_at_j_mv"])

        plausible = (
            old.ST_PLAUSIBLE_RANGE_MV[0]
            <= value
            <= old.ST_PLAUSIBLE_RANGE_MV[1]
        )

        st[name] = round(value, 3)

        details[name] = {
            "st_at_j_mv": round(value, 3),
            "st_at_j60_mv": (
                None
                if reading["st_at_j60_mv"] is None
                else round(reading["st_at_j60_mv"], 3)
            ),
            "baseline_mv": round(baseline, 3),
            "plausible": bool(plausible),
            "baseline_position_ms_relative_to_r": (
                base_idx - r_idx
            ) * 2.0,
        }

    vals = [
        v["st_at_j_mv"]
        for v in details.values()
        if v["plausible"]
    ]

    return {
        "status": "success",
        "st_by_lead": st,
        "st_details": details,
        "qrs_onset_idx": onset,
        "qrs_offset_idx": offset,
        "analysis_fs": ANALYSIS_FS,
        "qrs_onset_ms_relative_to_r": (
            onset - r_idx
        ) * 2.0,
        "j_point_ms_relative_to_r": (
            offset - r_idx
        ) * 2.0,
        "qrs_duration_ms": duration,
        "qrs_plausible": (
            old.QRS_DURATION_RANGE_MS[0]
            <= duration
            <= old.QRS_DURATION_RANGE_MS[1]
        ),
        "beats_used": min(counts),
        "max_st_elevation_mv": max(vals) if vals else 0.0,
        "max_st_depression_mv": min(vals) if vals else 0.0,
        "lead_boundaries": entries,
        "offset_source": "original_energy_detector",
        "onset_source": "v2_corrob_morphology",
    }
