"""
CardioSense - Signal Quality Index (SQI)
Evaluates lead integrity, detects disconnected electrodes, clipping,
and motion artifacts to reject unreliable ECG recordings before AI analysis.
"""

import numpy as np
from scipy.stats import kurtosis


# ---------------------------------------------------------
# Block 1: Detect Disconnected Leads (Flatlines)
# ---------------------------------------------------------
def check_flatline(lead_signal, min_variance=1e-4):
    """
    Returns True if the lead is flatlined or disconnected.
    An unplugged lead has near-zero variance across the 10-second window.
    """
    variance = np.var(lead_signal)
    return bool(variance < min_variance)


# ---------------------------------------------------------
# Block 2: Detect ADC Saturation (Clipping)
# ---------------------------------------------------------
def check_clipping(lead_signal, max_percent_clipped=0.05):
    """
    Returns True if the signal is saturated/clipping at hardware rails.
    Detects if more than 5% of samples are stuck at the exact min or max amplitude.
    """
    min_val, max_val = np.min(lead_signal), np.max(lead_signal)
    if min_val == max_val:
        return True  # Zero dynamic range

    # Count samples touching the extreme thresholds
    clipped_samples = np.sum((lead_signal == min_val) | (lead_signal == max_val))
    ratio = clipped_samples / len(lead_signal)
    return bool(ratio > max_percent_clipped)


# ---------------------------------------------------------
# Block 3: Calculate Lead-Level SQI Score (0.0 to 1.0)
# ---------------------------------------------------------
def compute_lead_sqi(lead_signal):
    """
    Computes a normalized Signal Quality Index (SQI) score [0.0 - 1.0].
    - 0.0: Disconnected, clipped, or pure artifact (Reject).
    - 0.6 - 0.79: Marginal quality (Warning).
    - 0.8 - 1.0: Excellent quality (Accept).
    """
    # Immediate rejection if flatlined or heavily clipped
    if check_flatline(lead_signal) or check_clipping(lead_signal):
        return 0.0

    # Kurtosis check: normal QRS spikes have positive kurtosis (> 3.0 in standard definition)
    # Excessive kurtosis (> 25) means large violent motion spikes
    k = kurtosis(lead_signal, fisher=False)  # Normal Gaussian = 3.0

    if k < 2.0:
        # Too flat, lacks typical QRS peaks
        kurtosis_score = 0.3
    elif 2.0 <= k <= 15.0:
        # Ideal physiological ECG morphology
        kurtosis_score = 1.0
    else:
        # Large erratic spikes (motion artifacts)
        kurtosis_score = max(0.1, 1.0 - (k - 15.0) / 20.0)

    # Dynamic range sanity check: healthy ECG is typically 0.2 mV to 5.0 mV peak-to-peak
    peak_to_peak = np.ptp(lead_signal)
    if 0.1 <= peak_to_peak <= 6.0:
        amplitude_score = 1.0
    else:
        amplitude_score = 0.4

    # Weighted composite score
    sqi = 0.6 * kurtosis_score + 0.4 * amplitude_score
    return round(float(np.clip(sqi, 0.0, 1.0)), 3)


# ---------------------------------------------------------
# Block 4: Multi-Lead Quality Assessment
# ---------------------------------------------------------
def assess_ecg_quality(ecg_signals, lead_names=None, accept_threshold=0.6):
    """
    Evaluates all 12 leads of an ECG matrix (samples x leads).
    Returns an overall pass/fail boolean, average SQI, and per-lead breakdown.
    """
    num_leads = ecg_signals.shape[1]
    if lead_names is None:
        lead_names = [f"Lead_{i+1}" for i in range(num_leads)]

    lead_scores = {}
    passed_leads = 0

    for i in range(num_leads):
        score = compute_lead_sqi(ecg_signals[:, i])
        lead_name = lead_names[i]
        lead_scores[lead_name] = {
            "sqi": score,
            "acceptable": bool(score >= accept_threshold),
        }
        if score >= accept_threshold:
            passed_leads += 1

    overall_sqi = round(float(np.mean([item["sqi"] for item in lead_scores.values()])), 3)
    
    # ECG is accepted if at least 10 of the 12 leads pass quality checks
    is_acceptable = bool(passed_leads >= (num_leads - 2) and overall_sqi >= accept_threshold)

    return {
        "is_acceptable": is_acceptable,
        "overall_sqi": overall_sqi,
        "passed_leads": f"{passed_leads}/{num_leads}",
        "lead_scores": lead_scores,
    }