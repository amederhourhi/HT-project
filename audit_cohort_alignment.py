"""Compare usable waveform records and extracted clinical features by ECG ID.

Read-only diagnostic. Does not train models or modify project files.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
# Works in the project root or in a scripts/ directory.
if not (ROOT / 'data').exists() and (ROOT.parent / 'data').exists():
    ROOT = ROOT.parent

MANIFEST = ROOT / 'data/cohort/cohort_manifest.csv'
FEATURES = ROOT / 'data/processed/clinical_features.csv'
COHORT_DIR = ROOT / 'data/cohort'


def normalize_ids(series):
    return series.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def main():
    for path in (MANIFEST, FEATURES):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')

    manifest = pd.read_csv(MANIFEST)
    features = pd.read_csv(FEATURES)
    required_manifest = {'ecg_id', 'strat_fold', 'target_label', 'filename_lr'}
    required_features = {'ecg_id', 'strat_fold', 'target_label'}
    for name, df, required in [('manifest', manifest, required_manifest), ('features', features, required_features)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'{name}: missing required columns {sorted(missing)}')
        df['_id'] = normalize_ids(df['ecg_id'])
        if df['_id'].isna().any() or (df['_id'] == '').any():
            raise ValueError(f'{name}: blank ECG IDs')
        if df['_id'].duplicated().any():
            duplicated = df.loc[df['_id'].duplicated(False), 'ecg_id'].tolist()
            raise ValueError(f'{name}: duplicate ECG IDs, examples: {duplicated[:12]}')

    manifest['_waveform_exists'] = manifest['filename_lr'].map(
        lambda rel: all((COHORT_DIR / str(rel)).with_suffix(ext).is_file() for ext in ('.hea', '.dat'))
    )
    usable = manifest[manifest['_waveform_exists']].copy()
    merged = usable[['_id', 'ecg_id', 'strat_fold', 'target_label']].merge(
        features[['_id', 'strat_fold', 'target_label']], on='_id', how='outer',
        suffixes=('_cnn', '_rf'), indicator=True, validate='one_to_one'
    )

    print(f'Manifest rows: {len(manifest)}; usable waveform rows: {len(usable)}; feature rows: {len(features)}')
    print(f'Missing waveform files: {len(manifest) - len(usable)}')
    print('\nPer-fold counts (usable waveforms vs feature rows):')
    counts = pd.concat([
        usable.groupby('strat_fold').size().rename('waveforms'),
        features.groupby('strat_fold').size().rename('features')
    ], axis=1).fillna(0).astype(int)
    print(counts.to_string())

    print('\nID alignment among usable waveforms and features:')
    print(merged['_merge'].value_counts().to_string())
    both = merged[merged['_merge'] == 'both']
    fold_mismatch = both[both['strat_fold_cnn'] != both['strat_fold_rf']]
    label_mismatch = both[both['target_label_cnn'] != both['target_label_rf']]
    print(f'Fold mismatches on shared IDs: {len(fold_mismatch)}')
    print(f'Label mismatches on shared IDs: {len(label_mismatch)}')
    if len(fold_mismatch):
        print(fold_mismatch.head(15).to_string(index=False))
    if len(label_mismatch):
        print(label_mismatch.head(15).to_string(index=False))

    for title, indicator in [('Waveform only (no feature row)', 'left_only'),
                             ('Feature only (no usable waveform)', 'right_only')]:
        missing = merged[merged['_merge'] == indicator]
        print(f'\n{title}: {len(missing)}')
        if len(missing):
            columns = ['_id', 'strat_fold_cnn', 'strat_fold_rf', 'target_label_cnn', 'target_label_rf']
            print(missing[columns].head(30).to_string(index=False))
    if len(fold_mismatch) or len(label_mismatch):
        print('\nACTION: Stop ensemble development until IDs, labels and folds are reconciled.')
    else:
        print('\nShared-record labels and folds match. Investigate unmatched IDs before fusion.')


if __name__ == '__main__':
    main()
