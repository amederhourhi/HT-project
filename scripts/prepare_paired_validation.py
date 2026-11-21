"""Select reproducible PTB-XL paired-validation subset and fetch ONLY needed 500 Hz files.

Usage:
  python -m scripts.prepare_paired_validation --dry-run
  python -m scripts.prepare_paired_validation
  python -m scripts.benchmark_common_grid_cohort

Uses the existing data/cohort/cohort_manifest.csv; does not overwrite healthy files.
This is an engineering consistency sample, NOT independent clinical ground truth.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'cohort'
OUT = ROOT / 'docs' / 'clinical_validation'
BASE_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
DEV_IDS = {1918, 14639}
SEED = 20261102


def find_column(columns, options):
    return next((name for name in options if name in columns), None)


def select_records(manifest, count):
    data = manifest.copy()
    data['ecg_id'] = pd.to_numeric(data['ecg_id'], errors='raise').astype(int)
    data = data.drop_duplicates(subset='ecg_id')
    data = data[~data['ecg_id'].isin(DEV_IDS)]
    data = data[data['filename_hr'].notna()].copy()
    if len(data) < count:
        raise ValueError(f'Only {len(data)} eligible records, requested {count}')

    fold = find_column(data.columns, ('strat_fold', 'fold', 'patient_fold'))
    label = find_column(data.columns, ('target_label', 'label', 'mi_label', 'is_mi', 'target'))
    strat_cols = [x for x in (fold, label) if x]
    if not strat_cols:
        data['_stratum'] = 'all'
    else:
        data['_stratum'] = data[strat_cols].fillna('missing').astype(str).agg('|'.join, axis=1)

    rng = np.random.default_rng(SEED)
    groups = {}
    for stratum, part in data.groupby('_stratum', sort=True):
        ids = part['ecg_id'].to_numpy().copy()
        rng.shuffle(ids)
        groups[stratum] = list(map(int, ids))

    chosen = []
    while len(chosen) < count:
        progress = False
        for key in sorted(groups):
            if len(chosen) >= count:
                break
            if groups[key]:
                chosen.append(groups[key].pop())
                progress = True
        if not progress:
            break

    selected = data.set_index('ecg_id').loc[chosen].reset_index()
    selected['selection_seed'] = SEED
    selected['subset_role'] = 'prospective_engineering_validation'
    return selected.drop(columns=['_stratum']), strat_cols


def fetch_file(rel_path: str, suffix: str, dry_run: bool):
    relative = Path(rel_path.replace('\\', '/'))
    dest = (DATA / relative).with_suffix(suffix)
    if dest.is_file() and dest.stat().st_size > 0:
        return 'exists'
    if dry_run:
        return 'would_download'
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote = BASE_URL + relative.as_posix() + suffix
    temporary = dest.with_name(dest.name + '.part')
    for attempt in range(3):
        try:
            req = Request(remote, headers={'User-Agent': 'CardioSense-validation/1.0'})
            with urlopen(req, timeout=60) as source, temporary.open('wb') as target:
                import shutil
                shutil.copyfileobj(source, target)
            if temporary.stat().st_size <= 0:
                raise IOError('Downloaded file is empty')
            temporary.replace(dest)
            return 'downloaded'
        except (HTTPError, URLError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                return f'ERROR: {exc}'
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=60, help='New ECGs, excludes the 2 development ECGs')
    parser.add_argument('--dry-run', action='store_true', help='Select and list files without downloading')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')

    manifest_file = DATA / 'cohort_manifest.csv'
    manifest = pd.read_csv(manifest_file)
    if not {'ecg_id', 'filename_hr'}.issubset(manifest.columns):
        raise ValueError(f'Manifest requires ecg_id, filename_hr. Found: {list(manifest.columns)}')
    selected, strata = select_records(manifest, args.count)
    OUT.mkdir(parents=True, exist_ok=True)
    roster = OUT / 'common_grid_validation_subset_v1.csv'
    if roster.exists():
        old = pd.read_csv(roster)
        if set(old['ecg_id'].astype(int)) != set(selected['ecg_id'].astype(int)):
            raise RuntimeError(f'Existing selection differs: {roster}. Do not silently change the validation set.')
    else:
        selected.to_csv(roster, index=False)
    print(f'Selected {len(selected)} ECGs; excluded development records {sorted(DEV_IDS)}')
    print('Stratification columns:', strata if strata else '(none present; random selection)')
    print('Selection recorded:', roster)
    print('Record IDs:', ', '.join(map(str, selected['ecg_id'].to_list())))

    errors = []
    downloaded = 0
    for i, rec in enumerate(selected.itertuples(index=False), 1):
        relative = str(rec.filename_hr)
        statuses = [fetch_file(relative, ext, args.dry_run) for ext in ('.hea', '.dat')]
        downloaded += statuses.count('downloaded')
        print(f'[{i}/{len(selected)}] ECG {rec.ecg_id}: {statuses[0]}, {statuses[1]}')
        if any(status.startswith('ERROR') for status in statuses):
            errors.append({'ecg_id': rec.ecg_id, 'problem': str(statuses)})
            continue
        if not args.dry_run:
            try:
                base = DATA / relative
                signals, fields = wfdb.rdsamp(str(base))
                if fields['fs'] != 500 or signals.shape != (5000, 12):
                    raise ValueError(f'Unexpected ECG: {fields["fs"]} Hz, shape={signals.shape}')
            except Exception as exc:
                errors.append({'ecg_id': rec.ecg_id, 'problem': f'Invalid waveform: {exc}'})

    if errors:
        pd.DataFrame(errors).to_csv(OUT / 'common_grid_download_errors.csv', index=False)
        raise SystemExit(f'{len(errors)} download/validation failures. See common_grid_download_errors.csv')
    print(f'Completed; {downloaded} files downloaded. Dry run: {args.dry_run}')
    print('Next: python -m scripts.benchmark_common_grid_cohort')
    print('The benchmark will include the two development records; exclude their IDs in validation summaries.')


if __name__ == '__main__':
    main()
