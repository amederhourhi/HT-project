
# CardioSense — Cohort Protocol

## 1. Dataset

- Source: PTB-XL version 1.0.3
- Sampling rate: 100 Hz
- Research cohort: 300 ECG recordings
- Normal-reference labels: 150
- Infarction-family labels: 150

This is a deliberately balanced experimental cohort,
not a clinically representative patient population.

## 2. Canonical cohort pin

File: `data/processed/cohort_ecg_ids.csv`

Each row contains:

- `ecg_id`: Official PTB-XL recording identifier
- `strat_fold`: Original PTB-XL fold assignment
- `target_label`: Binary research label (0 or 1)

The cohort includes all 300 records regardless of
waveform availability, signal quality or extraction success.

The canonical pin must never be regenerated from a
feature-extraction output.

## 3. Recorded extraction outcomes

| Metric | Result |
|---|---:|
| Pinned ECG records | 300 |
| Waveforms available in recorded experiment | 297 |
| Missing waveform files | 3 |
| Successful SQI V1 feature extractions | 251 |
| Successful SQI V3 feature extractions | 295 |

These are recorded experimental outcomes, not claims
of diagnostic accuracy.

Source:
`docs/sqi_audit/full_pipeline_v1_vs_v3.csv`

## 4. Dataset partitions

PTB-XL's recommended partitioning:

- Folds 1–8: Training
- Fold 9: Validation
- Fold 10: Testing

Patient IDs must not cross fold boundaries.

Model selection and signal-processing decisions made
using a fold can compromise its independence as a
held-out evaluation set.

## 5. Reproducibility requirements

The dataset reconstruction process must verify:

1. Every pinned ECG ID exists in official PTB-XL metadata.
2. ECG IDs are unique.
3. Diagnostic labels match the pinned research labels.
4. Stratified folds match the original assignments.
5. Patient identities do not cross fold boundaries.
6. Ambiguous or unrecognized diagnostic labels are rejected.

Invalid metadata must cause an explicit error.

## 6. Verification commands

Run from the project root:

```powershell
python scripts/verify_cohort_contract.py
python -m pytest tests/test_cohort_contract.py -v
```

To reconstruct the manifest without downloading waveforms:

```powershell
python src/data/download_ptbxl.py --cohort --manifest_only
```

The resulting manifest must contain the same 300 ECG IDs,
research labels and fold assignments.

## 7. Research limitations

The binary infarction-family label is not confirmation
of an acute coronary occlusion or an emergency diagnosis.

This cohort is small and intentionally balanced.

The existing SQI comparisons and model-development
experiments are not independent clinical validation.
