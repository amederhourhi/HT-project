"""Read-only, paired comparison of current and candidate fiducials.

Uses locally available PTB-XL 500 Hz records and their controlled 100 Hz
resamplings, matched R peaks, and identical 500 Hz filtering. This measures
sampling-rate agreement, NOT accuracy against expert annotations.

Run: python -m scripts.benchmark_common_grid_cohort
"""
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from src.dsp.filters import filter_ecg
from src.features.r_peaks import detect_r_peaks
from src.features.fiducials import analyze_12lead_features as current
from src.features.fiducials_common_grid import analyze_12lead_features as candidate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'cohort'
OUTPUT = ROOT / 'docs' / 'clinical_validation'
LEADS = ('I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


def one_record(ecg_id, path):
    raw, fields = wfdb.rdsamp(str(path))
    if fields['fs'] != 500 or raw.shape != (5000, 12):
        raise ValueError(f'Unexpected 500 Hz waveform: fs={fields["fs"]}, shape={raw.shape}')
    names = fields['sig_name']
    if len({x.upper() for x in names}) != 12:
        raise ValueError('Expected 12 unique lead names')
    filtered500 = filter_ecg(raw, fs=500)
    filtered100 = resample_poly(filtered500, 1, 5, axis=0)
    ii = [x.upper() for x in names].index('II')
    peaks500 = np.asarray(detect_r_peaks(filtered500[:, ii], fs=500), dtype=int)
    peaks100 = np.unique(np.rint(peaks500 / 5).astype(int))
    if len(peaks500) < 2 or len(peaks100) != len(peaks500):
        raise ValueError('Too few peaks or mapping collisions')
    entries = []
    for method, fn in [('current', current), ('common_grid', candidate)]:
        outputs = {}
        for fs, signal, peaks in ((500, filtered500, peaks500), (100, filtered100, peaks100)):
            result = fn(signal, peaks, names, fs=fs)
            if result['status'] != 'success':
                raise ValueError(f'{method} at {fs} Hz: {result.get("reason", result["status"])}')
            outputs[fs] = result
        high, low = outputs[500], outputs[100]
        entries.append({
            'ecg_id': int(ecg_id), 'method': method,
            'peaks': len(peaks500), 'beats_500': high['beats_used'],
            'beats_100': low['beats_used'],
            'qrs_500_ms': high['qrs_duration_ms'],
            'qrs_100_ms': low['qrs_duration_ms'],
            'abs_qrs_difference_ms': abs(high['qrs_duration_ms'] - low['qrs_duration_ms']),
            **{f'abs_st_difference_{lead}_uv': 1000 * abs(
                high['st_by_lead'][names[[x.upper() for x in names].index(lead)]] -
                low['st_by_lead'][names[[x.upper() for x in names].index(lead)]]
            ) for lead in LEADS},
            **{f'st_500_{lead}_mv': high['st_by_lead'][names[[x.upper() for x in names].index(lead)]] for lead in LEADS},
        })
    return entries


def main():
    manifest = pd.read_csv(DATA / 'cohort_manifest.csv').set_index('ecg_id')
    rows, errors, unavailable = [], [], []
    for ecg_id, record in manifest.iterrows():
        base = DATA / str(record['filename_hr'])
        if not base.with_suffix('.hea').is_file() or not base.with_suffix('.dat').is_file():
            unavailable.append(int(ecg_id))
            continue
        try:
            rows.extend(one_record(ecg_id, base))
            print(f'ECG {ecg_id}: analyzed')
        except Exception as exc:
            errors.append({'ecg_id': int(ecg_id), 'error': f'{type(exc).__name__}: {exc}'})
            print(f'ECG {ecg_id}: ERROR {exc}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print('No complete local 500 Hz comparisons. Download additional 500 Hz records first.')
        return
    df.to_csv(OUTPUT / 'common_grid_cohort_pairs.csv', index=False)
    if errors:
        pd.DataFrame(errors).to_csv(OUTPUT / 'common_grid_cohort_errors.csv', index=False)
    print(f'\nCompleted {len(df)//2} ECGs, {len(unavailable)} unavailable, {len(errors)} failed')
    paired = df.set_index(['ecg_id', 'method']).sort_index()
    counts = {'improved': 0, 'worsened': 0, 'unchanged': 0}
    metrics = ['abs_qrs_difference_ms'] + [f'abs_st_difference_{l}_uv' for l in LEADS]
    print('\nMetric | current median | candidate median | improved | worsened | unchanged | current p95 | candidate p95')
    for metric in metrics:
        cur = paired.xs('current', level='method')[metric]
        new = paired.xs('common_grid', level='method')[metric]
        diff = (new - cur).round(6)
        imp, worse, same = int((diff < 0).sum()), int((diff > 0).sum()), int((diff == 0).sum())
        print(f'{metric} | {cur.median():.2f} | {new.median():.2f} | {imp} | {worse} | {same} | {cur.quantile(.95):.2f} | {new.quantile(.95):.2f}')
    print('\n500 Hz preservation (candidate vs current):')
    for lead in LEADS:
        metric = f'st_500_{lead}_mv'
        cur = paired.xs('current', level='method')[metric]
        new = paired.xs('common_grid', level='method')[metric]
        changes = (cur != new).sum()
        if changes:
            print(f'  {lead}: {changes} unexpected differences')
    print(f'\nCSV: {OUTPUT / "common_grid_cohort_pairs.csv"}')
    print('NOTE: Agreement with 500 Hz is not medical ground truth; do not change clinical triage from this benchmark.')


if __name__ == '__main__':
    main()
