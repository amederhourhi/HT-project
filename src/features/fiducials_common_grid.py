"""EXPERIMENTAL 500 Hz analysis-grid fiducial estimator.

Input: filtered ECG in mV, R peaks in input sample coordinates.
Does not change existing production fiducials or clinical rules.
Upsampling increases numerical grid resolution, NOT physical signal information.
"""
import numpy as np
from scipy.signal import resample_poly

from src.features import fiducials as legacy

ANALYSIS_FS = 500


def analyze_12lead_features(ecg_signals, r_peaks, lead_names, fs=100.0):
    signals = np.asarray(ecg_signals, dtype=float)
    if signals.ndim != 2 or signals.shape[1] != len(lead_names):
        raise ValueError('Expected (samples, leads) signal and matching lead_names')
    if not np.isfinite(signals).all():
        return {'status': 'rejected', 'reason': 'Nonfinite signal samples'}
    if fs not in (100, 500):
        raise ValueError('Candidate supports original 100 Hz and 500 Hz signals only')
    peaks = np.asarray(r_peaks, dtype=int)
    if peaks.ndim != 1 or len(peaks) < 2 or np.any(np.diff(peaks) <= 0):
        return {'status': 'rejected', 'reason': 'Missing or nonmonotonic R peaks'}

    beats = []
    used_counts = []
    r_index = None
    for column in range(signals.shape[1]):
        beat, idx, used = legacy.build_median_beat(signals[:, column], peaks, fs=fs)
        if beat is None:
            return {'status': 'rejected', 'reason': 'Too few clean beats to build a median beat'}
        beats.append(beat)
        used_counts.append(used)
        r_index = idx if r_index is None else r_index
    median_beats = np.stack(beats)

    factor = ANALYSIS_FS // int(fs)
    if factor != 1:
        # Upsample the whole median beat: preserve sub-sample position for
        # onset, baseline and ST, rather than rounding J back to 100 Hz.
        median_beats = resample_poly(median_beats, factor, 1, axis=1)
        r_index *= factor

    onset, offset = legacy.find_global_qrs_bounds(median_beats, r_index, fs=ANALYSIS_FS)
    if onset is None or offset <= onset:
        return {'status': 'rejected', 'reason': 'QRS boundaries could not be located'}

    duration_ms = (offset - onset) * 1000.0 / ANALYSIS_FS
    st_by_lead, details = {}, {}
    for name, beat in zip(lead_names, median_beats):
        baseline, baseline_idx = legacy.estimate_isoelectric_baseline(beat, onset, fs=ANALYSIS_FS)
        readings = legacy.measure_lead_st(beat, baseline, offset, fs=ANALYSIS_FS)
        value = readings['st_at_j_mv']
        plausible = legacy.ST_PLAUSIBLE_RANGE_MV[0] <= value <= legacy.ST_PLAUSIBLE_RANGE_MV[1]
        st_by_lead[name] = round(value, 3)
        details[name] = {
            'st_at_j_mv': round(value, 3),
            'st_at_j60_mv': round(readings['st_at_j60_mv'], 3) if readings['st_at_j60_mv'] is not None else None,
            'baseline_mv': round(baseline, 3),
            'plausible': bool(plausible),
            'baseline_position_ms_relative_to_r': round((baseline_idx-r_index)*1000/ANALYSIS_FS, 1),
        }
    trusted_values = [d['st_at_j_mv'] for d in details.values() if d['plausible']]
    return {
        'status': 'success',
        'st_by_lead': st_by_lead,
        'st_details': details,
        'max_st_elevation_mv': round(max(trusted_values), 3) if trusted_values else 0.0,
        'max_st_depression_mv': round(min(trusted_values), 3) if trusted_values else 0.0,
        'qrs_duration_ms': round(duration_ms, 1),
        # Index fields refer to the analysis grid, NOT the input grid.
        'qrs_onset_idx': int(onset),
        'qrs_offset_idx': int(offset),
        'analysis_fs': ANALYSIS_FS,
        'qrs_onset_ms_relative_to_r': round((onset-r_index)*1000/ANALYSIS_FS, 1),
        'j_point_ms_relative_to_r': round((offset-r_index)*1000/ANALYSIS_FS, 1),
        'qrs_plausible': bool(legacy.QRS_DURATION_RANGE_MS[0] <= duration_ms <= legacy.QRS_DURATION_RANGE_MS[1]),
        'beats_used': int(min(used_counts)),
    }
