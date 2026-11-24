
# Milestone 8 — QRS Expert-Reference Accuracy Audit

## Scope

Evaluate previously saved QRS delineation results
against LUDB expert annotations.

- Dataset: LUDB
- Records: 31–100
- Paired recordings: 70
- Algorithms: current and experimental QRS V4
- Reference: expert-annotated global QRS boundaries
- R-peaks: expert-supplied
- Analysis: read-only, recording-level
- Bootstrap: paired recording resampling, 3,000 draws

This is not a new independent holdout evaluation.
Records 31–100 have already been examined.

## QRS Onset Accuracy

Current algorithm:
- Median absolute error: 16 ms
- Mean absolute error: 15.34 ms
- 95th-percentile absolute error: 26 ms

QRS V4:
- Median absolute error: 16 ms
- Mean absolute error: 15.03 ms
- 95th-percentile absolute error: 27.55 ms

Paired V4 minus current mean absolute error:
- Difference: -0.31 ms
- Descriptive bootstrap 95% interval: [-2.14, 1.69] ms
- Improved recordings: 17
- Worsened recordings: 11
- Unchanged recordings: 42

Interpretation:
The observed average improvement is small and uncertain.
Some individual onset errors become substantially worse.

## QRS Offset Accuracy

Current algorithm:
- Median absolute error: 8.5 ms
- Mean absolute error: 11.41 ms
- 95th-percentile absolute error: 28 ms

QRS V4:
- Median absolute error: 8 ms
- Mean absolute error: 10.93 ms
- 95th-percentile absolute error: 27.55 ms

Paired V4 minus current mean absolute error:
- Difference: -0.49 ms
- Descriptive bootstrap 95% interval: [-1.29, 0.00] ms
- Improved recordings: 2
- Worsened recordings: 0
- Unchanged recordings: 68

Interpretation:
Offset accuracy is nearly unchanged across the sample.

## QRS Duration Accuracy

Current algorithm:
- Median absolute error: 22.5 ms
- Mean absolute error: 24.16 ms
- 95th-percentile absolute error: 46 ms

QRS V4:
- Median absolute error: 18.5 ms
- Mean absolute error: 21.13 ms
- 95th-percentile absolute error: 45.10 ms

Paired V4 minus current mean absolute error:
- Difference: -3.03 ms
- Descriptive bootstrap 95% interval: [-5.34, -0.71] ms
- Improved recordings: 24
- Worsened recordings: 6
- Unchanged recordings: 40

Interpretation:
V4 reduces average QRS duration error on this
previously evaluated dataset. The interval is
descriptive, not evidence of general clinical validity.

## Largest Duration Regressions

| LUDB record | Current error | V4 error | Increase |
|---|---:|---:|---:|
| 31 | 10 ms | 44 ms | 34 ms |
| 72 | 2 ms | 28 ms | 26 ms |
| 78 | 6 ms | 18 ms | 12 ms |
| 85 | 12 ms | 18 ms | 6 ms |
| 45 | 6 ms | 10 ms | 4 ms |

Records 31, 72 and 78 merit targeted investigation.
This investigation must not be used to claim that the
same recordings remain untouched evaluation data.

## Engineering Decision

Do not promote QRS V4 to the deployed pipeline.

Reasons:
1. The average benefit is modest.
2. Several individual regressions are substantial.
3. Automated R-peak detection was not evaluated.
4. ST amplitude accuracy was not validated here.
5. LUDB 31–100 has already been evaluated.
6. No clinical or external deployment validation exists.

The deployed QRS algorithm and model checkpoints remain
unchanged.

## Reproduction

From the project root:

    python -m pytest tests/test_ludb_qrs_accuracy.py -v
    python -m scripts.audit_ludb_qrs_accuracy

Inputs:
- docs/clinical_validation/ludb_qrs_v4_holdout.csv

Scripts:
- scripts/audit_ludb_qrs_accuracy.py
- tests/test_ludb_qrs_accuracy.py

## Reserved Data

- LUDB 101–200: not evaluated by Milestone 8
- PTB-XL fold 10: not evaluated by Milestone 8

Preserve these reservations.

## Next Research Priority

Investigate why onset boundaries regress on specific
recordings. Evaluate reference-definition consistency,
lead selection, waveform morphology and sampling rate.

Any modified algorithm must undergo a separate,
predefined evaluation rather than being judged only
against records used to develop it.

This milestone establishes measured research evidence,
not diagnostic or clinical reliability.
