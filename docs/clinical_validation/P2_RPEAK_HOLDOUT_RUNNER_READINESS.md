# CardioSense — Phase II Independent R-Peak Runner Readiness

**Study:** P2-RPEAK-03

**Status:** Runner development parity verified; independent evaluation not executed.

## Objective

Implement a reproducible, protected evaluation runner for comparing the frozen CardioSense baseline R-peak detector against the asymmetric −50/+20 ms research candidate.

No production detector changes are included.

## Development Verification

The runner was executed against LUDB development records 1–30 and reproduced the previously frozen evaluation results on all 30 recordings.

| Measurement | Baseline | Candidate |
|---|---:|---:|
| 500 Hz F1 at ±50 ms | 0.9881 | 0.9960 |
| 500 Hz F1 at ±100 ms | 1.0000 | 1.0000 |
| 100 Hz F1 at ±50 ms | 0.9842 | 1.0000 |
| 100 Hz F1 at ±100 ms | 1.0000 | 1.0000 |

The paired record-bootstrap intervals were reproduced.

No reserved LUDB recordings were accessed during the development smoke test.

## Frozen Implementations

- Baseline: unchanged production R-peak detector
- Candidate: asymmetric −50/+20 ms refinement
- Signal processing: shared filtering and sampling-rate handling
- Matching: frozen one-to-one reference matching
- Scoring: annotation-window evaluation with a fixed 100 ms margin

## Reserved Cohort

The predefined independent evaluation cohort is LUDB records 101–200.

The runner requires the complete cohort and does not permit arbitrary subset selection.

LUDB 31–100 was previously used in research and is not considered a completely untouched project-wide cohort.

## Execution Safeguards

The protected runner requires:

1. Explicit one-time evaluation confirmation.
2. Execution from the `main` branch.
3. A clean Git working tree.
4. A matching 40-character Git commit SHA.
5. Complete locally available reference files.
6. An exclusive attempt marker.
7. Dataset file checksums.
8. Recording-level failure reporting.
9. Non-overwriting result output.

The local attempt marker is an operational safeguard, not a tamper-proof audit log. Its persistence must be verified before evaluation.

## Failure Handling

Incomplete evaluations must not be reported as complete or reduced to successful records.

Any interrupted attempt must be documented before deciding how to handle it.

The frozen candidate must not be modified based on independent evaluation results and then reassessed on the same cohort as though it were untouched.

## Independent Evaluation Status

Not executed.

The holdout runner must pass deterministic tests, CI integration, and code review before independent evaluation is authorized.

The resulting study will evaluate R-peak localization. It will not establish clinical diagnostic safety, downstream ST measurement accuracy, or deployment readiness.

## Decision

Retain the baseline as production.

Retain the asymmetric refinement as research-only.

Freeze and review the independent evaluation runner before accessing LUDB 101–200.