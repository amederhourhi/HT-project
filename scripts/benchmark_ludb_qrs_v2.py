"""Compare current QRS, common-grid, and QRS-v2 against downloaded LUDB 1-30.
No downloading, no production changes, no test-set access.
Run: python -m scripts.benchmark_ludb_qrs_v2
"""
from pathlib import Path
import csv
import numpy as np
import wfdb
from src.dsp.filters import filter_ecg
from src.features import fiducials as old
from src.features import fiducials_common_grid as common
from src.features import fiducials_qrs_v2 as new
from scripts.benchmark_ludb_landmarks import qrs_triplets, expert_global_beat_references

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/ludb_reference/data'
OUTPUT = ROOT / 'docs/clinical_validation/ludb_qrs_v2_development.csv'


def evaluate(record_id):
    path = DATA / str(record_id)
    signal, fields = wfdb.rdsamp(str(path))
    if fields['fs'] != 500 or signal.shape[1] != 12:
        raise ValueError('Expected 12-lead 500 Hz LUDB waveform')
    peaks = np.array([x[1] for x in qrs_triplets(wfdb.rdann(str(path), 'ii'))], dtype=int)
    refs = expert_global_beat_references(path, peaks)
    lookup = {r['anchor']: r for r in refs}
    pre, post = round(old.BEAT_PRE_MS*.5), round(old.BEAT_POST_MS*.5)
    rr = np.diff(peaks)
    median_rr = float(np.median(rr))
    selected = []
    for i, peak in enumerate(peaks):
        if peak-pre < 0 or peak+post >= len(signal) or peak not in lookup:
            continue
        if i and abs(rr[i-1]-median_rr)/median_rr > old.ECTOPIC_RR_TOLERANCE:
            continue
        selected.append(lookup[peak])
    if len(selected) < 3:
        raise ValueError('Insufficient matched annotated beats')
    reference_on = float(np.median([(r['onset']-r['anchor'])*2 for r in selected]))
    reference_off = float(np.median([(r['offset']-r['anchor'])*2 for r in selected]))
    filtered = filter_ecg(signal, fs=500)
    rows=[]
    for method, analyzer in [('current',old.analyze_12lead_features),('common_grid',common.analyze_12lead_features),('qrs_v2',new.analyze_12lead_features)]:
        result = analyzer(filtered, peaks, lead_names=fields['sig_name'], fs=500)
        if result['status'] != 'success':
            rows.append({'record_id':record_id,'method':method,'status':result['status'],'reason':result.get('reason','')})
            continue
        onset = (result['qrs_onset_idx'] - pre)*2
        offset = (result['qrs_offset_idx'] - pre)*2
        rows.append({'record_id':record_id,'method':method,'status':'success','onset_error_ms':onset-reference_on,'offset_error_ms':offset-reference_off,'duration_error_ms':(offset-onset)-(reference_off-reference_on),'isolated_leads':len(result.get('isolated_boundaries',[]))})
    return rows


def main():
    rows=[]
    for rid in range(1,31):
        try:
            rows.extend(evaluate(rid))
            print(f'LUDB {rid:03d}: evaluated')
        except Exception as exc:
            print(f'LUDB {rid:03d}: FAILED: {type(exc).__name__}: {exc}')
            for method in ('current','common_grid','qrs_v2'):
                rows.append({'record_id':rid,'method':method,'status':'error','reason':str(exc)})
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    columns=['record_id','method','status','reason','onset_error_ms','offset_error_ms','duration_error_ms','isolated_leads']
    with OUTPUT.open('w', newline='', encoding='utf8') as file:
        writer=csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows: writer.writerow(row)
    print('\nEXPERT LANDMARK ERRORS (lower absolute error is better)')
    for method in ('current','common_grid','qrs_v2'):
        subset=[r for r in rows if r['method']==method and r['status']=='success']
        print(f'\n{method}: {len(subset)}/30 successful')
        if not subset: continue
        for col in ('onset_error_ms','offset_error_ms','duration_error_ms'):
            v=np.array([r[col] for r in subset],float)
            print(f'{col}: signed_median={np.median(v):+.1f} ms, median_abs={np.median(abs(v)):.1f} ms, p95_abs={np.percentile(abs(v),95):.1f} ms')
    baseline={r['record_id']:r for r in rows if r['method']=='current' and r['status']=='success'}
    modified={r['record_id']:r for r in rows if r['method']=='qrs_v2' and r['status']=='success'}
    improved=worsened=tied=0
    print('\nRECORD-LEVEL QRS DURATION REGRESSIONS')
    for rid in sorted(baseline.keys() & modified.keys()):
        x=abs(baseline[rid]['duration_error_ms']); y=abs(modified[rid]['duration_error_ms'])
        if y<x: improved+=1
        elif y>x: worsened+=1; print(f'LUDB {rid}: {x:.1f} -> {y:.1f} ms')
        else: tied+=1
    print(f'Improved {improved}, worsened {worsened}, unchanged {tied}')
    print('Saved:',OUTPUT)
    print('Experimental only: do not connect qrs_v2 to triage.')

if __name__ == '__main__': main()
