"""Read-only evaluation of baseline estimators, using matched peaks and common grid.

Does not modify CardioSense production code or clinical decisions.
Requires original PTB-XL 500 Hz files, cohort manifest, and local CardioSense modules.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from src.dsp.filters import filter_ecg
from src.features.r_peaks import detect_r_peaks
from src.features.fiducials import (
    build_median_beat, find_global_qrs_bounds, estimate_isoelectric_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'data' / 'cohort'
OUT = ROOT / 'docs' / 'clinical_validation' / 'baseline_strategy_audit.csv'
FS = 500
# Explicit experimental fixed window: [QRS onset - 55 ms, QRS onset - 25 ms).
# NOT validated PR delineation. Excludes immediate QRS; may overlap P-wave.
FIXED_GAP_MS = 25
FIXED_WIDTH_MS = 30
LEADS = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
DEV_IDS = {1918,14639}


def median_beats(signal, peaks, fs):
    beats = []
    reference_index = None
    counts = []
    for c in range(signal.shape[1]):
        beat, idx, count = build_median_beat(signal[:,c], peaks, fs=fs)
        if beat is None:
            return None
        beats.append(beat)
        counts.append(count)
        if reference_index is None:
            reference_index = idx
    arr = np.stack(beats)
    if fs == 100:
        arr = resample_poly(arr, 5, 1, axis=1)
        reference_index *= 5
    return arr, reference_index, min(counts)


def fixed_baseline(beat, onset, fs=FS):
    gap = round(FIXED_GAP_MS * fs / 1000)
    width = round(FIXED_WIDTH_MS * fs / 1000)
    end = int(onset) - gap
    start = end - width
    if start < 0 or end > len(beat) or end <= start:
        return None, None
    return float(np.median(beat[start:end])), (start+end-1)/2


def evaluate(signals, fs, peaks, lead_names):
    result = median_beats(signals, peaks, fs)
    if result is None:
        return None
    beats, r_idx, n_used = result
    onset, offset = find_global_qrs_bounds(beats, r_idx, fs=FS)
    if onset is None:
        return None
    values = {}
    for lead in LEADS:
        index = [str(n).upper() for n in lead_names].index(lead)
        beat = beats[index]
        current, current_pos = estimate_isoelectric_baseline(beat, onset, fs=FS)
        fixed, fixed_pos = fixed_baseline(beat, onset)
        if fixed is None:
            return None
        j_v = float(beat[offset])
        values[lead] = {
            'j_v': j_v, 'current_baseline': current,
            'current_pos': current_pos, 'fixed_baseline': fixed,
            'fixed_pos': fixed_pos,
            'current_st': j_v-current, 'fixed_st': j_v-fixed,
        }
    return {'onset_ms': (onset-r_idx)*2, 'j_ms': (offset-r_idx)*2,
            'qrs_ms': (offset-onset)*2, 'beats':n_used, 'leads':values}


def main():
    manifest = pd.read_csv(COHORT / 'cohort_manifest.csv').set_index('ecg_id')
    rows = []
    skipped = []
    for ecg_id, record in manifest.iterrows():
        hr = COHORT / str(record['filename_hr'])
        if not hr.with_suffix('.dat').exists() or not hr.with_suffix('.hea').exists():
            continue
        try:
            raw, fields = wfdb.rdsamp(str(hr))
            if fields['fs'] != 500 or raw.shape[1] != 12:
                raise ValueError('Expected 500 Hz 12-lead recording')
            filtered_500 = filter_ecg(raw, fs=500)
            filtered_100 = resample_poly(filtered_500, 1, 5, axis=0)
            lead_names = fields['sig_name']
            lead_ii = [str(s).upper() for s in lead_names].index('II')
            peaks_500 = np.asarray(detect_r_peaks(filtered_500[:,lead_ii], fs=500), dtype=int)
            peaks_100 = np.unique(np.rint(peaks_500/5).astype(int))
            a = evaluate(filtered_500,500,peaks_500,lead_names)
            b = evaluate(filtered_100,100,peaks_100,lead_names)
            if a is None or b is None:
                raise ValueError('Fiducial extraction failed')
            for lead in LEADS:
                x, y = a['leads'][lead],b['leads'][lead]
                rows.append({
                    'ecg_id': int(ecg_id), 'group':'development' if int(ecg_id) in DEV_IDS else 'validation',
                    'lead':lead, 'beats_500':a['beats'],'beats_100':b['beats'],
                    'qrs_500_ms':a['qrs_ms'],'qrs_100_ms':b['qrs_ms'],
                    'j_500_ms':a['j_ms'],'j_100_ms':b['j_ms'],
                    'current_st_500_mv':x['current_st'],'current_st_100_mv':y['current_st'],
                    'fixed_st_500_mv':x['fixed_st'],'fixed_st_100_mv':y['fixed_st'],
                    'current_abs_difference_uv':abs(x['current_st']-y['current_st'])*1000,
                    'fixed_abs_difference_uv':abs(x['fixed_st']-y['fixed_st'])*1000,
                    'current_baseline_difference_uv':(y['current_baseline']-x['current_baseline'])*1000,
                    'fixed_baseline_difference_uv':(y['fixed_baseline']-x['fixed_baseline'])*1000,
                    'j_voltage_difference_uv':(y['j_v']-x['j_v'])*1000,
                    'current_baseline_time_500_ms':(x['current_pos']-125)*2,
                    'current_baseline_time_100_ms':(y['current_pos']-125)*2,
                })
            print(f'ECG {ecg_id}: analyzed')
        except Exception as exc:
            skipped.append((ecg_id,str(exc)))
            print(f'ECG {ecg_id}: FAILED {type(exc).__name__}: {exc}')
    if not rows:
        raise SystemExit('No paired 500 Hz recordings processed')
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUT,index=False)
    print('\nSaved:',OUT)
    valid = df[df.group=='validation'].copy()
    print(f'Validation ECGs analyzed: {valid.ecg_id.nunique()}, failures: {len(skipped)}')
    if valid.empty:
        return
    grouped = valid.groupby('lead')
    summary=[]
    for lead, part in grouped:
        old = part.current_abs_difference_uv
        new = part.fixed_abs_difference_uv
        summary.append({'lead':lead, 'current_median_uv':round(old.median(),1),
                        'fixed_median_uv':round(new.median(),1),
                        'current_p95_uv':round(old.quantile(.95),1),
                        'fixed_p95_uv':round(new.quantile(.95),1),
                        'improved':int((new<old).sum()),'worsened':int((new>old).sum())})
    print('\nPER-LEAD BASELINE STRATEGY COMPARISON')
    print(pd.DataFrame(summary).to_string(index=False))
    valid['change_uv']=valid.fixed_abs_difference_uv-valid.current_abs_difference_uv
    print('\nWORST REGRESSIONS (fixed minus current, microvolts)')
    print(valid.nlargest(12,'change_uv')[['ecg_id','lead','current_abs_difference_uv','fixed_abs_difference_uv','change_uv','j_voltage_difference_uv']].round(1).to_string(index=False))
    print('\nFOCUS CASES')
    focus=valid[valid.ecg_id.isin([13379,16507,11047,10231]) & valid.lead.isin(['I','II','AVF','V1','V4'])]
    print(focus[['ecg_id','lead','current_abs_difference_uv','fixed_abs_difference_uv','current_baseline_difference_uv','fixed_baseline_difference_uv','j_voltage_difference_uv']].round(1).to_string(index=False))
    print('\nNOTE: Fixed PR window is an experiment, NOT a validated clinical baseline method.')

if __name__ == '__main__':
    main()
