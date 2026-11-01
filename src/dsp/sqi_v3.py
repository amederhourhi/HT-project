
"""CardioSense SQI V3 - Experimental signal quality assessment.

Separates technical failures from changes in ECG morphology.
The thresholds are research heuristics, not clinically validated.
"""

from copy import deepcopy

import numpy as np
from scipy.signal import medfilt

from src.dsp.sqi_v2 import assess_ecg_quality_v2


def _longest_run(flags):
    longest = current = 0

    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)

    return longest


def _plateau_runs(mask):
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return (edges[1::2] - edges[::2]).tolist()


def _rail_plateau_windows(
    signal,
    fs,
    windows,
    min_run=3,
    min_fraction=0.005,
    adc_rails_mv=None,
):
    """Separate local plateaus from saturation at known ADC rails."""
    suspected = []
    confirmed = []

    for index, window in enumerate(windows):
        start = int(round(window["start_sec"] * fs))
        stop = int(round(window["end_sec"] * fs))
        segment = signal[start:stop]

        if len(segment) < min_run or np.ptp(segment) < 1e-10:
            continue

        bounds = (
            (np.min(segment), np.max(segment))
            if adc_rails_mv is None
            else adc_rails_mv
        )

        for bound in set(bounds):
            mask = segment == bound
            runs = _plateau_runs(mask)

            if (
                runs
                and max(runs) >= min_run
                and np.mean(mask) >= min_fraction
            ):
                if adc_rails_mv is None:
                    suspected.append(index)
                else:
                    confirmed.append(index)
                break

    return suspected, confirmed


def _high_frequency_noise_windows(
    signal,
    fs,
    windows,
    *,
    residual_limit_mv=0.12,
    affected_fraction=0.15,
    min_windows=2,
):
    """Detect sustained high-frequency contamination."""
    kernel = max(3, int(round(0.05 * fs)) | 1)
    suspect = []

    for index, window in enumerate(windows):
        start = int(round(window["start_sec"] * fs))
        stop = int(round(window["end_sec"] * fs))
        segment = signal[start:stop]

        if len(segment) < kernel or kernel >= len(segment):
            continue

        baseline = medfilt(segment, kernel_size=kernel)
        residual = segment - baseline

        rms = float(np.sqrt(np.mean(residual ** 2)))
        large_fraction = float(
            np.mean(np.abs(residual) > residual_limit_mv)
        )

        window["hf_residual_rms_mv"] = round(rms, 5)
        window["hf_large_residual_fraction"] = round(
            large_fraction, 4
        )

        if (
            rms > residual_limit_mv
            and large_fraction > affected_fraction
        ):
            suspect.append(index)
            if "high_frequency_noise_review" not in window["flags"]:
                window["flags"].append("high_frequency_noise_review")

    sustained = []
    current = []

    for index in suspect:
        if current and index != current[-1] + 1:
            if len(current) >= min_windows:
                sustained.extend(current)
            current = []

        current.append(index)

    if len(current) >= min_windows:
        sustained.extend(current)

    return sustained


def _baseline_drift_windows(
    signal,
    fs,
    windows,
    *,
    drift_span_limit_mv=0.9,
    local_excursion_mv=0.45,
):
    """Detect substantial slow baseline movement in raw ECG."""
    kernel = max(3, int(round(0.8 * fs)) | 1)

    if kernel >= len(signal):
        return []

    baseline = medfilt(signal, kernel_size=kernel)
    margin = kernel // 2
    core = baseline[margin:-margin] if margin else baseline

    if len(core) < 5:
        return []

    span = float(
        np.percentile(core, 95) - np.percentile(core, 5)
    )

    if span <= drift_span_limit_mv:
        return []

    reference = float(np.median(core))
    suspect = []

    for index, window in enumerate(windows):
        start = max(
            margin, int(round(window["start_sec"] * fs))
        )
        stop = min(
            len(signal) - margin,
            int(round(window["end_sec"] * fs)),
        )

        if stop <= start:
            continue

        segment = baseline[start:stop]
        local_level = float(np.median(segment))
        local_span = float(
            np.percentile(segment, 90)
            - np.percentile(segment, 10)
        )

        window["baseline_level_mv"] = round(local_level, 4)
        window["baseline_local_span_mv"] = round(local_span, 4)

        if (
            abs(local_level - reference) > local_excursion_mv
            or local_span > local_excursion_mv
        ):
            if "baseline_drift_suspected" not in window["flags"]:
                window["flags"].append("baseline_drift_suspected")
            suspect.append(index)

    return suspect


def assess_ecg_quality_v3(
    ecg_signals,
    lead_names=None,
    fs=100.0,
    *,
    amplitude_ratio_threshold=1.5,
    sustained_windows=3,
    raw_signals=None,
    adc_rails_mv=None,
    **v2_options,
):
    """Return technical quality and separate morphology warnings."""
    if (
        not np.isfinite(amplitude_ratio_threshold)
        or amplitude_ratio_threshold <= 1
    ):
        raise ValueError(
            "amplitude_ratio_threshold must be finite and > 1"
        )

    if (
        isinstance(sustained_windows, bool)
        or not isinstance(sustained_windows, int)
        or sustained_windows < 2
    ):
        raise ValueError(
            "sustained_windows must be an integer >= 2"
        )

    if adc_rails_mv is not None:
        rails = np.asarray(adc_rails_mv, dtype=float)

        if (
            rails.shape != (2,)
            or not np.isfinite(rails).all()
            or rails[0] >= rails[1]
        ):
            raise ValueError(
                "adc_rails_mv must contain finite (min, max) limits"
            )

        if raw_signals is None:
            raise ValueError(
                "raw_signals required when adc_rails_mv is supplied"
            )

        adc_rails_mv = tuple(rails)

    result = deepcopy(
        assess_ecg_quality_v2(
            ecg_signals,
            lead_names=lead_names,
            fs=fs,
            **v2_options,
        )
    )

    filtered = np.asarray(ecg_signals, dtype=float)

    if raw_signals is not None:
        raw = np.asarray(raw_signals, dtype=float)

        if (
            raw.shape != filtered.shape
            or not np.isfinite(raw).all()
        ):
            raise ValueError(
                "raw_signals must be finite and match ECG shape"
            )
    else:
        raw = filtered

    lead_data = result["lead_scores"]
    amplitude_flags = {}
    technical_failures = []
    sustained_leads = []
    baseline_review_leads = []

    for index, (name, info) in enumerate(lead_data.items()):
        windows = info["windows"]

        amplitudes = np.array(
            [w["peak_to_peak_mv"] for w in windows],
            dtype=float,
        )

        reference = float(
            np.median(np.sort(amplitudes)[:min(2, len(amplitudes))])
        )

        ratios = amplitudes / max(reference, 1e-8)
        flags = ratios > amplitude_ratio_threshold
        amplitude_flags[name] = flags

        run = _longest_run(flags)

        flatline_windows = [
            i for i, w in enumerate(windows)
            if "low_variance" in w["flags"]
        ]

        suspected_clipping, confirmed_clipping = (
            _rail_plateau_windows(
                raw[:, index],
                fs,
                windows,
                adc_rails_mv=adc_rails_mv,
            )
        )

        noise_windows = _high_frequency_noise_windows(
            filtered[:, index], fs, windows
        )

        drift_windows = _baseline_drift_windows(
            raw[:, index], fs, windows
        )

        for i in noise_windows:
            if "high_frequency_noise_suspected" not in windows[i]["flags"]:
                windows[i]["flags"].append(
                    "high_frequency_noise_suspected"
                )
            windows[i]["usable"] = False


        # Unknown rails: retain warnings, not hard failures.
        for i in suspected_clipping:
            if "clipping_suspected" not in windows[i]["flags"]:
                windows[i]["flags"].append("clipping_suspected")

        # Known rails: saturation can invalidate the affected window.
        for i in confirmed_clipping:
            if "clipping_at_known_rail" not in windows[i]["flags"]:
                windows[i]["flags"].append("clipping_at_known_rail")
            windows[i]["usable"] = False

        info["high_frequency_noise_windows"] = noise_windows
        info["baseline_drift_windows"] = drift_windows
        info["baseline_drift_review"] = bool(drift_windows)
        if drift_windows:
            baseline_review_leads.append(name)
        info["clipping_suspected_windows"] = suspected_clipping
        info["clipping_confirmed_windows"] = confirmed_clipping

        failure_windows = sorted(
            set(
                flatline_windows
                + confirmed_clipping
                + noise_windows
            )
        )

        info["technical_failure_windows"] = failure_windows
        info["technical_quality"] = (
            "unusable"
            if failure_windows
            else "no_detected_hard_failure"
        )

        info["amplitude_reference_mv"] = round(reference, 5)
        info["longest_instability_run"] = run
        info["sustained_amplitude_instability"] = (
            run >= sustained_windows
        )
        info["transient_amplitude_change"] = bool(
            np.any(flags) and run < sustained_windows
        )

        for window, flagged, ratio in zip(windows, flags, ratios):
            window["amplitude_ratio"] = round(float(ratio), 3)

            if flagged and "amplitude_instability_review" not in window["flags"]:
                window["flags"].append(
                    "amplitude_instability_review"
                )

        if failure_windows:
            technical_failures.append(name)

        if run >= sustained_windows:
            sustained_leads.append(name)

    morphology_leads = []
    localized_leads = []

    for name in sustained_leads:
        flags = amplitude_flags[name]

        coincident = any(
            other != name and np.any(flags & other_flags)
            for other, other_flags in amplitude_flags.items()
        )

        if coincident:
            morphology_leads.append(name)
        else:
            localized_leads.append(name)

    for name, info in lead_data.items():
        info["morphology_review"] = name in morphology_leads
        info["artifact_suspected"] = name in localized_leads

        info["lead_usable"] = bool(
            info["acceptable"]
            and not info["technical_failure_windows"]
        )

        info["review_flags"] = sorted(
            {
                flag
                for window in info["windows"]
                for flag in window["flags"]
                if (
                    "review" in flag
                    or flag == "clipping_suspected"
                )
            }
        )

    acceptable = bool(
        result["is_acceptable"]
        and not technical_failures
    )

    result.update({
        "algorithm": "experimental_sqi_v3_separated",
        "technical_failure_leads": technical_failures,
        "sustained_instability_leads": sustained_leads,
        "morphology_review_leads": morphology_leads,
        "artifact_suspected_leads": localized_leads,
        "baseline_drift_review_leads": baseline_review_leads,
        "st_measurement_review_required": bool(baseline_review_leads),
        "requires_review": bool(sustained_leads or baseline_review_leads),
        "technical_quality": (
            "unusable"
            if technical_failures
            else "no_detected_hard_failure"
        ),
        "is_acceptable": acceptable,
        "ready_for_automated_inference": acceptable,
        "warnings": [
            "Morphology and artifact flags are hypotheses.",
            "Local plateaus are warnings without known ADC limits.",
            "Artifact thresholds are experimental.",
            "Automated inference readiness is not clinically validated.",
            "Prefer unfiltered raw_signals for artifact assessment.",
        ],
    })

    return result
