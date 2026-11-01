"""Read-only audit of which ECG records are absent from the clinical feature table."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'data/cohort/cohort_manifest.csv'
FEATURES = ROOT / 'data/processed/clinical_features.csv'
WAVEFORM_ROOT = ROOT / 'data/cohort'


def ids(series):
    return series.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def check_unique(df, name):
    if df['_id'].isna().any() or df['_id'].eq('').any() or df['_id'].duplicated().any():
        raise ValueError(f'{name}: blank or duplicate ECG IDs found')


def main():
    manifest = pd.read_csv(MANIFEST)
    features = pd.read_csv(FEATURES)
    for name, df, fields in (
        ('manifest', manifest, {'ecg_id', 'strat_fold', 'target_label', 'filename_lr'}),
        ('features', features, {'ecg_id', 'strat_fold', 'target_label'}),
    ):
        missing = fields - set(df.columns)
        if missing:
            raise ValueError(f'{name}: missing columns: {sorted(missing)}')
        df['_id'] = ids(df['ecg_id'])
        check_unique(df, name)

    def waveform_exists(rel):
        if pd.isna(rel):
            return False
        base = WAVEFORM_ROOT / str(rel)
        return base.with_suffix('.hea').exists() and base.with_suffix('.dat').exists()

    manifest['waveform_exists'] = manifest['filename_lr'].map(waveform_exists)
    usable = manifest.loc[manifest['waveform_exists']].copy()
    joined = usable.merge(
        features[['_id']].assign(has_features=True), on='_id', how='left', validate='one_to_one'
    )
    joined['has_features'] = joined['has_features'].fillna(False).astype(bool)
    joined['outcome'] = joined['has_features'].map({True: 'included', False: 'missing_features'})

    print(f'Usable waveforms: {len(joined)}')
    print(f'Features present: {int(joined.has_features.sum())}')
    print(f'Features missing: {int((~joined.has_features).sum())}')

    table = pd.crosstab(joined['target_label'], joined['outcome'])
    table = table.reindex(columns=['included','missing_features'], fill_value=0)
    table['total'] = table.sum(axis=1)
    table['missing_pct'] = (100 * table['missing_features'] / table['total']).round(1)
    print('\nMissingness by target label:\n' + table.to_string())

    fold = pd.crosstab(joined['strat_fold'], joined['outcome'])
    fold = fold.reindex(columns=['included','missing_features'], fill_value=0)
    fold['total'] = fold.sum(axis=1)
    fold['missing_pct'] = (100 * fold['missing_features'] / fold['total']).round(1)
    print('\nMissingness by fold:\n' + fold.to_string())

    print('\nMissing waveform records (not among the 46 feature omissions):')
    print(manifest.loc[~manifest.waveform_exists, ['ecg_id','strat_fold','target_label']].to_string(index=False))

    print('\nRecords missing features (all 46):')
    print(joined.loc[~joined.has_features, ['ecg_id','strat_fold','target_label','filename_lr']].sort_values(['strat_fold','ecg_id']).to_string(index=False))

    print('\nInterpretation: Missingness statistics identify patterns, not the extraction failure causes.')
    print('Next: inspect the batch feature extractor and its failure/rejection logs.')

if __name__ == '__main__':
    main()
