"""
CardioSense - Guideline-Conformant Fiducial Delineation
Builds a noise-robust median beat per lead, locates the QRS complex globally,
and measures ST deviation at the J point and at J+60 ms against a per-lead
isoelectric baseline.

Why this module exists
----------------------
The clinical criteria this project applies (Fourth Universal Definition of MI)
define ST elevation at the *J point* - the end of the QRS - referenced to the
isoelectric PR segment. The previous implementation measured at J+60 ms, using
a J point approximated as "S trough + 30 ms" and a baseline taken from a single
sample 60 ms before the R peak. On normal hearts with an upsloping ST segment
that combination reads 2-3.5 mm of elevation where none exists, which is what
made the deterministic guardrails fire on healthy patients.

Both conventions are still reported here so the statistical engine can keep its
existing inputs while the safety rules move to the guideline one.
"""

import numpy as np

# ---------------------------------------------------------
# Block 1: Timing constants (milliseconds)
# ---------------------------------------------------------
# Every window is declared in ms and converted through `fs`, so the module
# behaves correctly at the 100 Hz and 500 Hz PTB-XL samplings alike.
BEAT_PRE_MS = 250          # PR segment and P wave sit before the R peak
BEAT_POST_MS = 450         # covers the ST segment and most of the T wave
QRS_SEARCH_BACK_MS = 120   # widest physiological QRS onset before R
QRS_SEARCH_FWD_MS = 160    # QRS offset can extend well past R in bundle branch block
ENERGY_SMOOTH_MS = 12      # smoothing width for the QRS energy envelope
# QRS boundary threshold, as a share of the envelope's peak above its floor.
#
# A *relative* threshold scales with the R-wave amplitude, so a tall R hides a
# slow terminal S deflection and the boundary lands inside the QRS. The value
# below is calibrated on the cohort rather than guessed:
#   fraction  QRS mean   NORM limb-lead ST   NORM escalation   infarction kept
#   0.060     81.6 ms    -0.015 mV (V4/V5)   1.6%              15.7%
#   0.015     97.4 ms    within 0.003 mV     1.6%              18.9%
#   0.008    105.0 ms    within 0.002 mV     2.4%              20.5%
# 0.015 is the only setting that puts QRS duration on the published normal
# (~90-100 ms) and flattens the healthy ST profile without spending either a
# healthy false positive or an infarction. 0.008 over-reads the QRS and starts
# picking up the T wave.
QRS_ENERGY_FRACTION = 0.015
BASELINE_MIN_WIDTH_MS = 30  # isoelectric window we average over
BASELINE_SCAN_MS = 60       # how far back from QRS onset we look for that window
BASELINE_GUARD_MS = 10      # gap kept clear of QRS onset so no window touches the QRS
ST_OFFSET_MS = 60          # the secondary "J+60" convention
ECTOPIC_RR_TOLERANCE = 0.20  # beat rejected when RR deviates this much from median RR

# A lead must reach this share of the loudest lead's QRS energy before it is
# allowed to vote on the global QRS boundary. Leads like aVL or III are often
# near-isoelectric, and on those leads the threshold crossing is noise.
LOUDEST_LEAD_QUALITY_SHARE = 0.05

# Plausibility envelope. Anything outside these bounds is a delineation failure,
# not a clinical finding, and must never reach a hard safety rule.
QRS_DURATION_RANGE_MS = (50.0, 220.0)
ST_PLAUSIBLE_RANGE_MV = (-1.0, 1.0)


# ---------------------------------------------------------
# Block 2: Median Beat Construction
# ---------------------------------------------------------
def build_median_beat(lead_signal, r_peaks, fs=100.0):
    """
    Aligns beats on their R peak and takes the sample-wise median.

    Beats whose preceding RR interval deviates from the median RR by more than
    ECTOPIC_RR_TOLERANCE are dropped first: an ectopic beat has a genuinely
    different repolarisation shape, and averaging it in would smear the very
    ST segment we are trying to measure.

    Returns:
        (median_beat, r_index_in_beat, beats_used) or (None, None, 0) when the
        record does not contain enough clean beats to delineate.
    """
    if len(r_peaks) < 2:
        return None, None, 0

    pre = int(round(BEAT_PRE_MS * fs / 1000.0))
    post = int(round(BEAT_POST_MS * fs / 1000.0))

    rr = np.diff(np.asarray(r_peaks, dtype=float))
    median_rr = float(np.median(rr)) if rr.size else float("inf")

    # Keep beats that are well inside the record and not preceded by an ectopic coupling.
    usable = []
    for i, r in enumerate(r_peaks):
        if r - pre < 0 or r + post >= len(lead_signal):
            continue
        if i > 0 and rr[i - 1] > 0 and abs(rr[i - 1] - median_rr) / median_rr > ECTOPIC_RR_TOLERANCE:
            continue
        usable.append(r)

    if not usable:
        # Fall back to the edge-valid beats rather than returning nothing; a
        # single noisy RR outlier should not cost us the whole record.
        usable = [r for r in r_peaks if r - pre >= 0 and r + post < len(lead_signal)]
    if not usable:
        return None, None, 0

    stack = np.stack([lead_signal[r - pre: r + post] for r in usable])
    return np.median(stack, axis=0), pre, len(usable)


# ---------------------------------------------------------
# Block 3: Global QRS Bounds
# ---------------------------------------------------------
def _lead_energy_envelope(lead_beat, search_start, search_end, fs):
    """
    Smoothed squared-derivative energy of one lead across the QRS search window.

    Squaring the derivative collapses the steep QRS upstrokes into a single
    broad bump; smoothing over ENERGY_SMOOTH_MS stops one noisy sample from
    moving a boundary. Returns (envelope, smoothing_width).
    """
    segment = lead_beat[search_start:search_end + 1]
    energy = np.diff(segment, prepend=segment[0]) ** 2
    width = max(1, int(round(ENERGY_SMOOTH_MS * fs / 1000.0)))
    return np.convolve(energy, np.ones(width) / width, mode="same"), width


def _bounds_from_envelope(envelope, peak_idx, level, hold):
    """
    Walks outwards from an energy peak until it stays below `level`.

    The sustained requirement is what makes this stable. The envelope dips below
    its threshold several times - once on the genuine Q/T junction, and again in
    the noise flutter either side of it - so accepting only a crossing that holds
    for a full smoothing width picks the junction rather than the flutter.
    """
    onset, offset = peak_idx, peak_idx

    i = peak_idx
    while i >= hold:
        if np.all(envelope[i - hold:i] < level):
            onset = i
            break
        i -= 1

    i = peak_idx
    while i <= len(envelope) - hold - 1:
        if np.all(envelope[i + 1:i + 1 + hold] < level):
            offset = i + 1
            break
        i += 1

    return max(0, onset), min(len(envelope) - 1, offset)


def find_global_qrs_bounds(median_beats, r_index, fs=100.0):
    """
    Locates QRS onset and offset as the union of the twelve per-lead boundaries.

    The QRS is one electrical event, but it neither starts nor finishes at the
    same sample in every lead: a slurred terminal S wave in V6 can outlast a
    sharp septal vector in V2 by tens of milliseconds. Guideline convention
    measures QRS duration from the earliest onset to the *latest* offset across
    the leads, so the union is the correct answer. Summing lead energy instead -
    the tempting shortcut, and the first version of this function - approximates
    the earliest common end and silently truncates the slurred portion, which
    then reads back as a large false ST elevation at the J point.

    Leads much quieter than the loudest are excluded. On a near-isoelectric lead
    the threshold crossing is noise, and because we take a union a single
    noise-driven crossing widens the QRS for all twelve leads and displaces the
    baseline window for all twelve too.

    Args:
        median_beats: array of shape (n_leads, beat_length).
        r_index: index of the R peak within the beat window.

    Returns:
        (qrs_onset_idx, qrs_offset_idx) in samples, or (None, None) on failure.
    """
    n = median_beats.shape[1]
    search_start = max(0, r_index - int(QRS_SEARCH_BACK_MS * fs / 1000.0))
    search_end = min(n - 1, r_index + int(QRS_SEARCH_FWD_MS * fs / 1000.0))
    if search_end - search_start < 4:
        return None, None

    lead_envelopes = [_lead_energy_envelope(np.asarray(lead, dtype=float), search_start, search_end, fs)
                      for lead in median_beats]
    envelopes = [env for env, _ in lead_envelopes]
    hold = max(2, lead_envelopes[0][1])

    peaks = [float(np.max(env)) for env in envelopes]
    loudest = max(peaks)
    if loudest <= 0:
        return None, None

    onsets, offsets = [], []
    for env, peak in zip(envelopes, peaks):
        if peak < LOUDEST_LEAD_QUALITY_SHARE * loudest:
            continue  # too quiet to place a boundary more reliably than noise
        onset, offset = _bounds_from_envelope(env, int(np.argmax(env)), QRS_ENERGY_FRACTION * peak, hold)
        if offset > onset:
            onsets.append(onset)
            offsets.append(offset)

    if not onsets:
        return None, None

    onset = search_start + min(onsets)
    offset = search_start + max(offsets)
    if not (0 <= onset < offset < n):
        return None, None
    return onset, offset


# ---------------------------------------------------------
# Block 4: Isoelectric Baseline
# ---------------------------------------------------------
def estimate_isoelectric_baseline(lead_beat, qrs_onset_idx, fs=100.0):
    """
    Finds the flattest stretch of PR segment just before the QRS and calls it
    the isoelectric line.

    Two properties matter here, and both were wrong in the previous design. The
    window must stay clear of the QRS: a candidate allowed to start one sample
    before onset happily swallows the septal Q wave and drags the baseline down
    with it. The window must also stay clear of the *preceding* beat's T wave -
    hence the short scan span - because a flat sample mid-T-downslope is flat but
    emphatically not isoelectric, and using it as the reference silently offsets
    every ST measurement on that lead.

    Within those bounds we still prefer the flattest window rather than a fixed
    one, so the estimate does not depend on P-wave delineation.

    Returns:
        (baseline_mv, baseline_centre_idx). The index is reported so the
        waveform inspector can show *where* the reference was taken - a
        measurement is only auditable if its baseline is visible.
    """
    width = max(2, int(round(BASELINE_MIN_WIDTH_MS * fs / 1000.0)))
    guard = max(1, int(round(BASELINE_GUARD_MS * fs / 1000.0)))
    scan = int(round(BASELINE_SCAN_MS * fs / 1000.0))

    # Rightmost sample the window may end on, then the earliest it may start.
    stop = max(width + guard, qrs_onset_idx - guard)
    start = max(0, stop - max(scan, width))
    fallback = max(0, qrs_onset_idx - guard)
    if stop - start < width:
        return float(lead_beat[fallback]), fallback

    windows = [(float(np.ptp(lead_beat[s:s + width])), s) for s in range(start, stop - width + 1)]
    if not windows:
        return float(lead_beat[fallback]), fallback

    _, best = min(windows, key=lambda w: (w[0], w[1]))
    centre = best + width // 2
    return float(np.median(lead_beat[best:best + width])), centre


# ---------------------------------------------------------
# Block 5: Per-Lead ST Measurement
# ---------------------------------------------------------
def measure_lead_st(lead_beat, baseline_mv, j_point_idx, fs=100.0):
    """
    Reads ST deviation at the J point and at J+60 ms for one lead.

    Returns:
        dict with both conventions in mV plus 'st_at_j60_idx', the sample the
        J+60 reading was taken on. The J-point reading needs no index - it is
        j_point_idx by definition. Both J+60 fields are None when the beat
        window is too short to reach J+60 (a very slow heart rate in a short
        window); reported as unknown rather than silently guessed.
    """
    st_at_j = float(lead_beat[j_point_idx] - baseline_mv)

    offset = int(round(ST_OFFSET_MS * fs / 1000.0))
    st_idx = j_point_idx + offset
    if st_idx >= len(lead_beat):
        return {"st_at_j_mv": st_at_j, "st_at_j60_mv": None, "st_at_j60_idx": None}

    return {"st_at_j_mv": st_at_j,
            "st_at_j60_mv": float(lead_beat[st_idx] - baseline_mv),
            "st_at_j60_idx": int(st_idx)}


# ---------------------------------------------------------
# Block 6: Twelve-Lead Analysis
# ---------------------------------------------------------
def analyze_12lead_features(ecg_signals, r_peaks, lead_names, fs=100.0):
    """
    Delineates a 12-lead record and returns guideline-conformant measurements.

    Args:
        ecg_signals: filtered signal matrix, shape (n_samples, n_leads), in mV.
        r_peaks: R-peak indices detected on the rhythm lead.
        lead_names: lead labels matching the columns of ecg_signals.
        fs: sampling frequency in Hz.

    Returns:
        dict with per-lead ST at both conventions, the shared QRS timing, the
        measured beat count, and per-lead plausibility flags. 'status' is
        'rejected' when the record cannot be delineated reliably at all.
    """
    signals = np.asarray(ecg_signals, dtype=float)
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    # One median beat per lead, all aligned on the same R peak.
    beats, r_index, used = [], None, 0
    for col in range(signals.shape[1]):
        beat, idx, count = build_median_beat(signals[:, col], r_peaks, fs=fs)
        if beat is None:
            return {"status": "rejected", "reason": "Too few clean beats to build a median beat"}
        beats.append(beat)
        r_index = idx if r_index is None else r_index
        used = max(used, count)

    median_beats = np.stack(beats)  # (n_leads, beat_length)

    qrs_onset, qrs_offset = find_global_qrs_bounds(median_beats, r_index, fs=fs)
    if qrs_onset is None or not (qrs_offset > qrs_onset):
        return {"status": "rejected", "reason": "QRS boundaries could not be located"}

    qrs_duration_ms = float((qrs_offset - qrs_onset) / fs * 1000.0)
    qrs_plausible = QRS_DURATION_RANGE_MS[0] <= qrs_duration_ms <= QRS_DURATION_RANGE_MS[1]

    st_by_lead, flags = {}, {}
    for i, name in enumerate(lead_names):
        baseline, _ = estimate_isoelectric_baseline(median_beats[i], qrs_onset, fs=fs)
        reading = measure_lead_st(median_beats[i], baseline, qrs_offset, fs=fs)

        st_at_j = reading["st_at_j_mv"]
        # A reading outside the plausible band means the delineation failed on
        # this lead; keep the number for audit but mark it so rules can ignore it.
        plausible = st_at_j is not None and ST_PLAUSIBLE_RANGE_MV[0] <= st_at_j <= ST_PLAUSIBLE_RANGE_MV[1]
        st_by_lead[name] = round(st_at_j, 3) if st_at_j is not None else 0.0
        flags[name] = {
            "st_at_j_mv": round(st_at_j, 3) if st_at_j is not None else None,
            "st_at_j60_mv": round(reading["st_at_j60_mv"], 3) if reading["st_at_j60_mv"] is not None else None,
            "baseline_mv": round(baseline, 3),
            "plausible": bool(plausible),
        }

    measured = [v["st_at_j_mv"] for v in flags.values() if v["st_at_j_mv"] is not None and v["plausible"]]
    return {
        "status": "success",
        "st_by_lead": st_by_lead,
        "st_details": flags,
        "max_st_elevation_mv": round(max(measured), 3) if measured else 0.0,
        "max_st_depression_mv": round(min(measured), 3) if measured else 0.0,
        "qrs_duration_ms": round(qrs_duration_ms, 1),
        "qrs_onset_idx": int(qrs_onset),
        "qrs_offset_idx": int(qrs_offset),
        "qrs_plausible": bool(qrs_plausible),
        "beats_used": int(used),
    }


# ---------------------------------------------------------
# Block 7: Single-Beat View (kept for the dashboard inspector)
# ---------------------------------------------------------
def delineate_beat_landmarks(lead_signal, r_peak_idx, fs=100.0):
    """
    Delineates one beat in place of a median beat, for the waveform inspector.

    This is a convenience overlay: the plot shows where the algorithm believes
    the QRS boundaries and ST measurement point are on a *single* beat, so a
    clinician can eyeball the delineation. Triage numbers come from
    analyze_12lead_features(), never from here.
    """
    n = len(lead_signal)
    pre = int(round(BEAT_PRE_MS * fs / 1000.0))
    post = int(round(BEAT_POST_MS * fs / 1000.0))
    if r_peak_idx - pre < 0 or r_peak_idx + post >= n:
        return None

    beat = lead_signal[r_peak_idx - pre: r_peak_idx + post]
    onset, offset = find_global_qrs_bounds(beat.reshape(1, -1), pre, fs=fs)
    if onset is None or offset <= onset:
        return None

    baseline, baseline_centre = estimate_isoelectric_baseline(beat, onset, fs=fs)
    reading = measure_lead_st(beat, baseline, offset, fs=fs)

    # Landmark indices are absolute (into the caller's full signal), matching
    # how the inspector plots them. st_deviation_mv is read at j_point_idx;
    # st_at_j60_idx only locates the legacy J+60 reading for comparison.
    shift = r_peak_idx - pre
    j60 = reading["st_at_j60_idx"]
    return {
        "beat": beat,
        "r_idx_in_beat": pre,
        "q_onset_idx": shift + onset,
        "j_point_idx": shift + offset,
        "st_at_j60_idx": (shift + j60) if j60 is not None else None,
        "baseline_idx": shift + baseline_centre,
        "baseline_voltage": round(baseline, 3),
        "st_deviation_mv": round(reading["st_at_j_mv"], 3),
        "st_at_j60_mv": round(reading["st_at_j60_mv"], 3) if reading["st_at_j60_mv"] is not None else None,
        "qrs_duration_ms": round((offset - onset) / fs * 1000.0, 1),
    }
