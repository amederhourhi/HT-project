# CardioSense — Phase II R-Peak Refinement Study

**Study:** P2-RPEAK-01
**Status:** Development evaluation completed; independent evaluation pending
**Candidate:** Asymmetric −50/+20 ms refinement

## Objective

Determine whether shortening the forward peak-refinement window improves CardioSense's R-peak timing accuracy without changing its energy-based candidate detector.

The production detector remains unchanged.

## Development Dataset

LUDB records 1–30, previously used during algorithm development.

- Reference: lead-II expert `N` peak annotations
- Annotated peaks: 253
- Sampling: native 500 Hz and derived 100 Hz
- Matching: one-to-one event alignment
- Tolerances: ±50 ms and ±100 ms
- Scoring region: first-to-last annotated QRS peak, with a fixed 100 ms margin

Detections outside the scoring region are reported as unscored, not presumed correct.

The 100 Hz waveforms were resampled from 500 Hz and do not represent independently acquired device measurements.

## Candidate Implementation

Current refinement:

`back_ms=50`, `forward_ms=50`

Experimental refinement:

`back_ms=50`, `forward_ms=20`

The energy threshold, candidate detection procedure, refractory period, and absolute-voltage refinement rule are unchanged.

The refinement window was modified following investigation of LUDB 023. Consequently, the development results cannot serve as independent confirmation.

## Results

| Metric | Current | Candidate |
|---|---:|---:|
| 500 Hz F1, ±50 ms | 0.9881 | 0.9960 |
| 100 Hz F1, ±50 ms | 0.9842 | 1.0000 |
| 500 Hz F1, ±100 ms | 1.0000 | 1.0000 |
| 100 Hz F1, ±100 ms | 1.0000 | 1.0000 |
| 500 Hz mean absolute timing error | 4.97 ms | 4.23 ms |
| 100 Hz mean absolute timing error | 5.73 ms | 4.58 ms |
| 500 Hz maximum absolute error | 66 ms | 56 ms |
| 100 Hz maximum absolute error | 70 ms | 40 ms |

### Paired timing changes

500 Hz: 10 beats improved, 1 worsened, 242 unchanged.

100 Hz: 7 beats improved, 0 worsened, 246 unchanged.

The only observed beat-level regression was LUDB 013 beat 3, increasing from 0 ms to 2 ms absolute error.

The only remaining strict-tolerance failure was LUDB 017 at 500 Hz, with one false-positive/false-negative pair at ±50 ms.

### Unscored detections

500 Hz: 63 detections outside the annotated scoring region.

100 Hz: 65 detections outside the annotated scoring region.

Their true physiological identities remain unverified.

## Failure Mechanism

The investigation of LUDB 023 demonstrated that the current refinement sometimes moves a correctly localized derivative-energy candidate approximately 50 ms forward to a larger-magnitude negative S-wave trough.

Reducing the forward search window corrected the observed LUDB 023 localization errors in the development experiment.

This finding does not establish that the same change improves performance on other waveform morphologies.

## Independent Evaluation Protocol — Freeze Before Use

The candidate implementation and matching procedure must be committed before any new independent evaluation.

Primary comparison: paired baseline-versus-candidate performance at ±50 ms.

Secondary comparisons: ±100 ms matching, mean and percentile absolute timing errors, maximum error, and record-level regressions.

The evaluation must report complete denominators, missing or failed recordings, and detections outside annotated intervals.

The preliminary advancement criteria are:

1. No reduction in aggregate ±50 ms F1 relative to baseline at either sampling rate.
2. No new missed reference peaks at ±50 ms without explicit investigation.
3. No increase in aggregate mean or 95th-percentile absolute timing error.
4. All record-level regressions and unscored detections disclosed.

These are research advancement criteria, not clinical performance standards.

LUDB 31–100 has already been examined in prior QRS research and is not to be described as an untouched project-wide evaluation set.

LUDB 101–200 remains reserved for a separately planned evaluation. Do not inspect or use these records for further parameter tuning.

PTB-XL fold 10 also remains reserved from model evaluation.

## Decision

Freeze the experimental asymmetric 50/20 ms refinement for independent investigation.

Do not replace the production R-peak detector.

Do not change model checkpoints, clinical rules, or deployed ECG processing on the basis of this experiment.

Independent performance, edge-region detection correctness, alternative QRS morphologies, and downstream fiducial accuracy remain unresolved.

**Conclusion:** Promising development result; no clinical or deployment approval.