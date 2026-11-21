"""EXPERIMENTAL QRS v2. Not for clinical triage.

Uses the established median-beat construction and ST/baseline code, but
compares independent per-lead energy boundaries with a conservative
voltage-morphology extension and a corroboration test for isolated extremes.
The evidence for either change MUST come from independent annotated landmarks.
"""
import numpy as np
from scipy.signal import resample_poly
from src.features import fiducials as old

ANALYSIS_FS = 500


def _candidate_boundaries(beats, r_index, fs=500):
    """Return candidate bounds and audit data in the analysis-grid coordinates."""
    beats = np.asarray(beats, dtype=float)
    start = max(0, r_index - round(old.QRS_SEARCH_BACK_MS * fs / 1000))
    stop = min(beats.shape[1] - 1, r_index + round(old.QRS_SEARCH_FWD_MS * fs / 1000))
    envelopes = [old._lead_energy_envelope(b, start, stop, fs)[0] for b in beats]
    maxima = np.array([float(np.max(e)) for e in envelopes])
    loudest = float(max(maxima))
    if loudest <= 0:
        return [], []
    hold = max(2, int(round(old.ENERGY_SMOOTH_MS * fs / 1000)))
    entries = []
    for i, (beat, env, power) in enumerate(zip(beats, envelopes, maxima)):
        if power < old.LOUDEST_LEAD_QUALITY_SHARE * loudest:
            continue
        a, b = old._bounds_from_envelope(env, int(np.argmax(env)), old.QRS_ENERGY_FRACTION * power, hold)
        if b <= a:
            continue
        a, b = start + a, start + b
        # Independent voltage morphology check. Estimate quiet-region noise
        # robustly and extend only while sustained displacement is present.
        # This is a testable hypothesis, not clinically established logic.
        quiet = beat[max(0, r_index - round(.23*fs)):max(1, r_index - round(.15*fs))]
        if quiet.size >= 8:
            baseline = float(np.median(quiet))
            mad = float(np.median(np.abs(quiet - baseline))) * 1.4826
            # No claim of accurate noise floor in arrhythmic/short-PR beats.
            threshold = max(0.012, 3.0 * mad, 0.025 * float(np.max(np.abs(beat[a:b] - baseline))))
            # Require ~8 ms contiguous displacement, not a lone sample.
            support = max(3, round(.008 * fs))
            # Limit morphology extensions to 30 ms on either side.
            for idx in range(a - 1, max(start, a - round(.030 * fs)) - 1, -1):
                if idx - support + 1 < start or not np.all(np.abs(beat[idx-support+1:idx+1] - baseline) > threshold):
                    break
                a = idx
            for idx in range(b + 1, min(stop, b + round(.030*fs)) + 1):
                if idx + support > len(beat) or not np.all(np.abs(beat[idx:idx+support] - baseline) > threshold):
                    break
                b = idx
        entries.append({'lead_index':i, 'onset':int(a), 'offset':int(b), 'energy_peak':float(power)})
    if not entries:
        return [], []
    # Isolated extremes are *flagged*, not removed: rejection could eliminate
    # legitimate single-lead activity. Provide a second, corroborated estimate.
    starts = np.array([e['onset'] for e in entries])
    ends = np.array([e['offset'] for e in entries])
    tolerance = round(.020 * fs)
    for e in entries:
        e['onset_supported'] = int(np.sum(np.abs(starts - e['onset']) <= tolerance)) >= 2
        e['offset_supported'] = int(np.sum(np.abs(ends - e['offset']) <= tolerance)) >= 2
    onset_candidates = [e['onset'] for e in entries if e['onset_supported']]
    offset_candidates = [e['offset'] for e in entries if e['offset_supported']]
    if not onset_candidates or not offset_candidates:
        return entries, []
    return entries, [min(onset_candidates), max(offset_candidates)]


def analyze_12lead_features(ecg_signals, r_peaks, lead_names, fs=500.0):
    signals = np.asarray(ecg_signals, dtype=float)
    if signals.ndim != 2 or signals.shape[1] != len(lead_names):
        raise ValueError('Expected 2-D samples x leads with matching names')
    if fs not in (100, 500):
        raise ValueError('Only 100 Hz and 500 Hz are supported')
    if not np.isfinite(signals).all():
        return {'status':'rejected','reason':'Non-finite input'}
    peaks = np.asarray(r_peaks, dtype=int)
    if peaks.ndim != 1 or len(peaks) < 2 or np.any(np.diff(peaks) <= 0):
        return {'status':'rejected','reason':'Invalid R peaks'}
    median = []
    counts = []
    for col in range(signals.shape[1]):
        beat, r_index, n = old.build_median_beat(signals[:,col], peaks, fs=fs)
        if beat is None:
            return {'status':'rejected','reason':'Median beat unavailable'}
        median.append(beat)
        counts.append(n)
    median = np.stack(median)
    if fs == 100:
        median = resample_poly(median, 5, 1, axis=1)
        r_index *= 5
    entries, bound = _candidate_boundaries(median, r_index, ANALYSIS_FS)
    if not bound or bound[1] <= bound[0]:
        return {'status':'rejected','reason':'No corroborated QRS boundaries','lead_boundaries':entries}
    onset, offset = bound
    duration = (offset - onset) * 2.0
    details = {}
    st = {}
    for name, beat in zip(lead_names, median):
        baseline, base_idx = old.estimate_isoelectric_baseline(beat, onset, fs=500)
        reading = old.measure_lead_st(beat, baseline, offset, fs=500)
        value = float(reading['st_at_j_mv'])
        plausible = old.ST_PLAUSIBLE_RANGE_MV[0] <= value <= old.ST_PLAUSIBLE_RANGE_MV[1]
        st[name] = round(value, 3)
        details[name] = {'st_at_j_mv':round(value,3), 'st_at_j60_mv':None if reading['st_at_j60_mv'] is None else round(reading['st_at_j60_mv'],3), 'baseline_mv':round(baseline,3), 'plausible':bool(plausible), 'baseline_position_ms_relative_to_r':(base_idx-r_index)*2.0}
    vals = [v['st_at_j_mv'] for v in details.values() if v['plausible']]
    return {'status':'success','st_by_lead':st,'st_details':details,'qrs_onset_idx':int(onset),'qrs_offset_idx':int(offset),'analysis_fs':500,'qrs_onset_ms_relative_to_r':(onset-r_index)*2.0,'j_point_ms_relative_to_r':(offset-r_index)*2.0,'qrs_duration_ms':duration,'qrs_plausible':old.QRS_DURATION_RANGE_MS[0] <= duration <= old.QRS_DURATION_RANGE_MS[1], 'beats_used':min(counts),'max_st_elevation_mv':max(vals) if vals else 0.0,'max_st_depression_mv':min(vals) if vals else 0.0,'lead_boundaries':entries,'isolated_boundaries':[e for e in entries if not(e['onset_supported'] and e['offset_supported'])]}
