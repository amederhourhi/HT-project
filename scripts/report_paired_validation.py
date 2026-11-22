"""Report independent-of-development engineering consistency results from paired benchmark.
Run after python -m scripts.benchmark_common_grid_cohort.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'clinical_validation'
LEADS = ('I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


def main():
    roster = pd.read_csv(OUT / 'common_grid_validation_subset_v1.csv')
    benchmark = pd.read_csv(OUT / 'common_grid_cohort_pairs.csv')
    intended = set(roster.ecg_id.astype(int))
    benchmark = benchmark[benchmark.ecg_id.isin(intended)]
    current = benchmark[benchmark.method == 'current'].set_index('ecg_id')
    candidate = benchmark[benchmark.method == 'common_grid'].set_index('ecg_id')
    complete = sorted(set(current.index) & set(candidate.index))
    missing = sorted(intended - set(complete))
    print(f'Primary validation set: {len(complete)}/{len(intended)} paired ECGs; missing {len(missing)}')
    if missing:
        print('Missing ECG IDs:', missing)
    if not complete:
        raise SystemExit('No matched cases; run benchmark first.')

    metrics = ['abs_qrs_difference_ms'] + [f'abs_st_difference_{lead}_uv' for lead in LEADS]
    rows = []
    for metric in metrics:
        before = current.loc[complete, metric]
        after = candidate.loc[complete, metric]
        delta = (after - before).round(6)
        rows.append({
            'metric': metric,
            'n': len(complete),
            'current_median': before.median(),
            'candidate_median': after.median(),
            'current_p95': before.quantile(.95),
            'candidate_p95': after.quantile(.95),
            'improved': int((delta < 0).sum()),
            'worsened': int((delta > 0).sum()),
            'unchanged': int((delta == 0).sum()),
            'max_candidate_difference': after.max(),
        })
    summary = pd.DataFrame(rows)
    target = OUT / 'common_grid_validation_summary_v1.csv'
    summary.to_csv(target, index=False)
    print(summary.round(2).to_string(index=False))
    print('Saved:', target)
    print('CAUTION: 100/500 Hz agreement does not establish anatomical or clinical correctness.')
    if missing:
        raise SystemExit('Validation cohort incomplete; do not make a go/no-go decision.')


if __name__ == '__main__':
    main()
