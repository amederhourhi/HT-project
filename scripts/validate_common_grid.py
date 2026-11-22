"""Run paired comparison; produces CSV; does not change clinical decisions."""
from pathlib import Path
import pandas as pd
import numpy as np
import wfdb
from scipy.signal import resample_poly
from src.dsp.filters import filter_ecg
from src.features.r_peaks import detect_r_peaks
from src.features.fiducials import analyze_12lead_features as original
from src.features.fiducials_common_grid import analyze_12lead_features as candidate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'cohort'


def main():
    manifest = pd.read_csv(DATA / 'cohort_manifest.csv').set_index('ecg_id')
    rows = []
    for ecg_id in (1918, 14639):
        item = manifest.loc[ecg_id]
        raw, fields = wfdb.rdsamp(str(DATA / item['filename_hr']))
        names = fields['sig_name']
        high = filter_ecg(raw, fs=500)
        low = resample_poly(high, 1, 5, axis=0)
        lead_ii = [n.upper() for n in names].index('II')
        peaks_500 = np.asarray(detect_r_peaks(high[:, lead_ii], fs=500), dtype=int)
        peaks_100 = np.unique(np.rint(peaks_500 / 5).astype(int))
        for method, fn in [('current', original), ('common_grid', candidate)]:
            for fs, signal, peaks in [(500, high, peaks_500), (100, low, peaks_100)]:
                result = fn(signal, peaks, names, fs=fs)
                row = {'ecg_id': ecg_id, 'method': method, 'fs': fs,
                       'status': result['status']}
                if result['status'] == 'success':
                    row.update(qrs_ms=result['qrs_duration_ms'], beats=result['beats_used'])
                    row.update({lead: result['st_by_lead'][lead] for lead in ('V1','V2','V3','V4')})
                else:
                    row['reason'] = result.get('reason')
                rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = ROOT / 'docs' / 'clinical_validation' / 'common_grid_candidate_comparison.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('\nSaved:', out)
    print('\nDo not route candidate outputs to triage without independent validation.')


if __name__ == '__main__':
    main()
