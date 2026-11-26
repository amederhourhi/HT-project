# CardioSense — Phase II R-Peak Evaluation Protocol

**Study ID:** P2-RPEAK-02
**Status:** Development protocol verified; independent evaluation pending
**Methods:** Frozen production baseline versus asymmetric −50/+20 ms research candidate

## 1. Objective

Evaluate whether the frozen asymmetric peak-refinement candidate improves R-peak localization while preserving detection reliability.

The experiment does not validate clinical diagnosis, ST-segment measurement accuracy, or deployment readiness.

## 2. Development Protocol Verification

The evaluation harness was tested on LUDB development records 1–30.

- 30 records and 253 lead-II annotated N peaks
- Original sampling frequency: 500 Hz
- Derived sampling frequency: 100 Hz using resampling
- Matching: monotone, one-to-one, maximum match count followed by minimum total absolute timing error
- Primary matching tolerance: ±50 ms
- Secondary tolerance: ±100 ms
- Scoring window: first to last annotated QRS peak, with a fixed 100 ms boundary margin
- Excluded boundary detections: reported separately as unscored

Neither algorithm uses expert annotations during detection.

## 3. Frozen Comparison

**Baseline:** Symmetric −50/+50 ms absolute-amplitude refinement.

**Candidate:** Asymmetric −50/+20 ms absolute-amplitude refinement.

Energy detection, thresholding, refractory period, preprocessing, and candidate selection remain unchanged.

## 4. Development Verification Results

| Measurement | Baseline | Candidate |
|---|---:|---:|
| 500 Hz F1, ±50 ms | 0.9881 | 0.9960 |
| 500 Hz F1, ±100 ms | 1.0000 | 1.0000 |
| 100 Hz F1, ±50 ms | 0.9842 | 1.0000 |
| 100 Hz F1, ±100 ms | 1.0000 | 1.0000 |
| 500 Hz mean absolute timing error, ±100 ms matching | 4.97 ms | 4.23 ms |
| 100 Hz mean absolute timing error, ±100 ms matching | 5.73 ms | 4.58 ms |

### Record-level paired bootstrap

The evaluation uses 2,000 bootstrap resamples, with complete recordings as sampling units and seed `20261126`.

**500 Hz, ±50 ms:**

- F1 improvement: +0.0079
- Percentile bootstrap 95% interval: [0.0000, 0.0244]

**100 Hz, ±50 ms:**

- F1 improvement: +0.0158
- Percentile bootstrap 95% interval: [0.0000, 0.0488]

Both intervals include zero. These results do not establish statistically significant superiority.

The improvement is concentrated in the LUDB 023 localization errors that motivated the experimental change. Development performance is therefore subject to selection and tuning bias.

## 5. Evaluation Integrity

The evaluation must:

1. Require a complete predefined recording cohort.
2. Reject duplicate or unexpected record-method combinations.
3. Preserve paired baseline-versus-candidate comparisons.
4. Verify consistent reference denominators.
5. Verify event-count identities and matching consistency.
6. Report all missing, rejected, and unscored events.
7. Report record-level regressions, not only pooled performance.
8. Bootstrap complete records, rather than treating individual heartbeats as independent observations.
9. Keep the 500 Hz and resampled 100 Hz comparisons separate.

Mean and percentile errors calculated only from matched detections are conditional on successful matching. Timing comparisons should also be assessed using a common matched reference-beat population.

## 6. Independent Evaluation Plan

The intended independent cohort is LUDB 101–200, reserved from earlier project evaluation.

Before accessing this cohort, the executable runner, fixed analysis configuration, test coverage, acceptance rules, output schema, and evaluation code version must be frozen.

The independent evaluation must:

- Compare the frozen implementations without tuning either algorithm.
- Process each predefined recording using the same signal-processing protocol.
- Record all missing and failed cases.
- Report precision, recall, F1, timing errors, uncertainty intervals, unscored detections, and record-level regressions.
- Distinguish recording-level and beat-level denominators.
- Avoid selecting favorable subsets after examining results.
- Preserve negative results without modifying the frozen candidate.

Any later algorithm change must constitute a new experimental version and cannot reuse this independent cohort as an untouched holdout.

LUDB 31–100 was previously evaluated for QRS research and must not be represented as a completely untouched project-wide cohort.

## 7. Advancement Criteria

A candidate may proceed to further research review only if:

1. Aggregate F1 at ±50 ms does not decline relative to baseline at either sampling rate.
2. New missed reference beats and major record-level regressions are explicitly investigated.
3. Aggregate mean and 95th-percentile absolute timing errors do not increase.
4. Missing records, processing failures, and unscored boundary detections are reported.
5. Confidence intervals and uncertainty in improvement are disclosed.

These criteria determine eligibility for further research, not clinical deployment.

## 8. Current Decision

The paired evaluation protocol is suitable for freezing after deterministic tests and CI validation.

The asymmetric candidate remains research-only.

No independent LUDB 101–200 analysis, production detector replacement, model retraining, or clinical promotion has occurred under this milestone.