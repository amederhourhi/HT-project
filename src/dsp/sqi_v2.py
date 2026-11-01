"""Experimental CardioSense SQI v2. Not validated for clinical use.

Keeps the legacy SQI contract, adds independent window-level artifact indicators,
and avoids treating global ECG kurtosis as proof of signal corruption.
Input is a filtered ECG (samples, leads), measured in mV.
"""

import numpy as np
from scipy.signal import medfilt


def _validate(signals, lead_names, fs, window_seconds):
    x = np.asarray(signals, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1 or x.shape[0] < 20:
        raise ValueError("ECG must be a 2D array with at least 20 samples")
    if not np.isfinite(x).all():
        raise ValueError("ECG contains NaN or infinity")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if window_seconds <= 0 or not np.isfinite(window_seconds):
        raise ValueError("window_seconds must be positive and finite")
    names = list(lead_names) if lead_names is not None else [f"Lead_{i+1}" for i in range(x.shape[1])]
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError("Lead names must be unique and match ECG channels")
    return x, names


def _odd_kernel(n, preferred):
    k = min(n if n % 2 else n - 1, preferred if preferred % 2 else preferred + 1)
    return max(3, k) if n >= 3 else 1


def _window_metrics(samples, fs):
    """Diagnostic proxies, not clinically validated noise estimates."""
    x = np.asarray(samples, dtype=float)
    p2p = float(np.ptp(x))
    variance = float(np.var(x))
    # Equal-to-extreme counts are more meaningful on raw digitized ECG;
    # this filtered-data proxy is only a supporting indicator.
    clipped_ratio = float(np.mean((x == x.min()) | (x == x.max())))
    # A 5-point median tracks a sharp QRS better than a slow baseline estimator.
    # Residual is an artifact proxy; clean, narrow QRS can also increase it.
    kernel = _odd_kernel(len(x), 5)
    baseline = medfilt(x, kernel_size=kernel) if kernel > 1 else x
    residual = x - baseline
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    rms = float(np.sqrt(np.mean((x - np.median(x))**2)))
    roughness_ratio = residual_rms / max(rms, 1e-6)
    return {
        "peak_to_peak_mv": round(p2p, 5),
        "variance_mv2": round(variance, 8),
        "extreme_sample_ratio": round(clipped_ratio, 4),
        "roughness_ratio": round(float(roughness_ratio), 4),
    }


def assess_ecg_quality_v2(
    ecg_signals,
    lead_names=None,
    fs=100.0,
    accept_threshold=0.6,
    window_seconds=2.0,
    min_good_fraction=0.8,
    min_good_leads=None,
    flatline_variance=1e-4,
    max_roughness_ratio=0.40,
):
    """Experimental quality assessment with local suspect-window reporting.

    In keeping with v1, returns is_acceptable/overall_sqi/passed_leads/lead_scores.
    Scores indicate heuristic usability, not probability of a clean ECG.
    Does not silently mask poor leads or reclassify physiological abnormalities.
    """
    x, names = _validate(ecg_signals, lead_names, fs, window_seconds)
    if not (0 <= accept_threshold <= 1 and 0 < min_good_fraction <= 1):
        raise ValueError("Invalid acceptance parameters")
    samples_per_window = max(20, int(round(window_seconds * fs)))
    # Non-overlapping windows; a final short fragment is merged into the previous one.
    boundaries = list(range(0, x.shape[0], samples_per_window))
    if len(boundaries) > 1 and x.shape[0] - boundaries[-1] < 20:
        boundaries.pop()
    boundaries.append(x.shape[0])
    lead_scores = {}
    for index, name in enumerate(names):
        lead = x[:, index]
        windows = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            metrics = _window_metrics(lead[start:stop], fs)
            reasons = []
            if metrics["variance_mv2"] < flatline_variance:
                reasons.append("low_variance")
            # Roughness is reported as suspect, not hard failure: sharp QRS can cause it.
            if metrics["roughness_ratio"] > max_roughness_ratio:
                reasons.append("high_roughness_review")
            # Strong non-flatline failures only: unvalidated as hardware rail detection.
            valid = "low_variance" not in reasons
            windows.append({
                "start_sec": round(start / fs, 3),
                "end_sec": round(stop / fs, 3),
                "usable": valid,
                "flags": reasons,
                **metrics,
            })
        usable_fraction = sum(w["usable"] for w in windows) / len(windows)
        flatline = np.var(lead) < flatline_variance
        # A high localized amplitude excursion vs other windows can indicate artifact;
        # report as a warning, not automatic rejection (could be ectopy).
        ranges = np.array([w["peak_to_peak_mv"] for w in windows])
        reference = float(np.median(ranges))
        for w in windows:
            if reference > 0 and w["peak_to_peak_mv"] > 3.0 * reference:
                w["flags"].append("amplitude_outlier_review")
        # Do not reward implausible amplitudes; do not treat high kurtosis as noise.
        amplitude_ok = 0.1 <= float(np.ptp(lead)) <= 6.0
        score = round(float(usable_fraction if amplitude_ok else 0.4 * usable_fraction), 3)
        if flatline:
            score = 0.0
        lead_scores[name] = {
            "sqi": score,
            "acceptable": bool(score >= accept_threshold and usable_fraction >= min_good_fraction),
            "usable_window_fraction": round(usable_fraction, 3),
            "windows": windows,
            "review_flags": sorted(set(flag for w in windows for flag in w["flags"] if "review" in flag)),
        }
    passed = sum(v["acceptable"] for v in lead_scores.values())
    total = len(names)
    required = max(1, total - 2) if min_good_leads is None else int(min_good_leads)
    if not 1 <= required <= total:
        raise ValueError("min_good_leads must be between 1 and number of leads")
    overall = round(float(np.mean([v["sqi"] for v in lead_scores.values()])), 3)
    return {
        "is_acceptable": bool(passed >= required and overall >= accept_threshold),
        "overall_sqi": overall,
        "passed_leads": f"{passed}/{total}",
        "lead_scores": lead_scores,
        "algorithm": "experimental_sqi_v2",
        "warnings": ["Heuristic scores have not been validated against expert quality labels"],
    }
