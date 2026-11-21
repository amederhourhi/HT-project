
"""CardioSense QRS V4 — experimental corroborated offset detection.

Uses V2 onset and original energy-derived offset, with isolation
handling for a single unusually late offset.

NOT clinically validated. Do not connect to production triage.
"""

import numpy as np
from scipy.signal import resample_poly

from src.features import fiducials as old
from src.features.fiducials_qrs_v2 import (
    _candidate_boundaries,
    ANALYSIS_FS,
)

OFFSET_ISOLATION_MS = 20.0
MIN_ELIGIBLE_LEADS = 6


def choose_offset(lead_offsets, fs=500):
    """Return selected offset, excluded lead index, and separation in ms."""
    if not lead_offsets:
        return None, None, None

    ranked = sorted(lead_offsets, key=lambda item: item[1])

    if len(ranked) < MIN_ELIGIBLE_LEADS:
        return int(ranked[-1][1]), None, None

    gap_ms = (
        (ranked[-1][1] - ranked[-2][1])
        * 1000.0 / fs
    )

    if gap_ms > OFFSET_ISOLATION_MS:
        return (
            int(ranked[-2][1]),
            int(ranked[-1][0]),
            float(gap_ms),
        )

    return int(ranked[-1][1]), None, float(gap_ms)


def energy_offset_candidates(median, r_idx, fs=500):
    """Recover each eligible lead's original energy-derived QRS offset."""
    start = max(
        0,
        r_idx - int(old.QRS_SEARCH_BACK_MS * fs / 1000),
    )
    end = min(
        median.shape[1] - 1,
        r_idx + int(old.QRS_SEARCH_FWD_MS * fs / 1000),
    )

    envelopes = [
        old._lead_energy_envelope(beat, start, end, fs)
        for beat in median
    ]

    powers = [
        float(np.max(env)) for env, _ in envelopes
    ]

    loudest = max(powers)

    if loudest <= 0:
        return []

    hold = max(2, envelopes[0][1])
    offsets = []

    for i, ((env, _), power) in enumerate(
        zip(envelopes, powers)
    ):
        if power < old.LOUDEST_LEAD_QUALITY_SHARE * loudest:
            continue

        onset, offset = old._bounds_from_envelope(
            env,
            int(np.argmax(env)),
            old.QRS_ENERGY_FRACTION * power,
            hold,
        )

        if offset > onset:
            offsets.append((i, int(start + offset)))

    return offsets


def analyze_12lead_features(
    ecg_signals,
    r_peaks,
    lead_names,
    fs=500.0,
):
    signals = np.asarray(ecg_signals, dtype=float)

    if (
        signals.ndim != 2
        or signals.shape[1] != len(lead_names)
    ):
        raise ValueError(
            "Expected samples x leads with matching names"
        )

    if fs not in (100, 500):
        raise ValueError(
            "Only 100 Hz and 500 Hz are supported"
        )

    if not np.isfinite(signals).all():
        return {
            "status": "rejected",
            "reason": "Non-finite input",
        }

    peaks = np.asarray(r_peaks, dtype=int)

    if (
        peaks.ndim != 1
        or len(peaks) < 2
        or np.any(np.diff(peaks) <= 0)
    ):
        return {
            "status": "rejected",
            "reason": "Invalid R peaks",
        }

    median = []
    counts = []
    r_idx = None

    for col in range(signals.shape[1]):
        beat, idx, count = old.build_median_beat(
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
        counts.append(count)

        if r_idx is None:
            r_idx = idx

    median = np.stack(median)

    if fs == 100:
        median = resample_poly(
            median, 5, 1, axis=1
        )
        r_idx *= 5

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

    candidates = energy_offset_candidates(
        median,
        r_idx,
        fs=ANALYSIS_FS,
    )

    offset, isolated_index, gap_ms = choose_offset(
        candidates,
        fs=ANALYSIS_FS,
    )

    if (
        offset is None
        or not (0 <= onset < offset < median.shape[1])
    ):
        return {
            "status": "rejected",
            "reason": "Inconsistent V4 boundaries",
        }

    duration = (
        (offset - onset) * 1000.0 / ANALYSIS_FS
    )

    st = {}
    details = {}

    for name, beat in zip(lead_names, median):
        baseline, base_idx = (
            old.estimate_isoelectric_baseline(
                beat,
                onset,
                fs=ANALYSIS_FS,
            )
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

    valid_st = [
        item["st_at_j_mv"]
        for item in details.values()
        if item["plausible"]
    ]

    isolated_lead = (
        None if isolated_index is None
        else str(lead_names[isolated_index])
    )

    return {
        "status": "success",
        "st_by_lead": st,
        "st_details": details,
        "qrs_onset_idx": onset,
        "qrs_offset_idx": int(offset),
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
        "max_st_elevation_mv": (
            max(valid_st) if valid_st else 0.0
        ),
        "max_st_depression_mv": (
            min(valid_st) if valid_st else 0.0
        ),
        "lead_boundaries": entries,
        "isolated_energy_offset_lead": isolated_lead,
        "offset_isolation_gap_ms": gap_ms,
        "offset_source": (
            "energy_corrob_experimental"
            if isolated_lead is not None
            else "original_energy"
        ),
        "onset_source": "v2_corrob_morphology",
    }
