# CardioSense — Final Engineering and Research Readiness Review

**Milestone:** 9 of 9  
**Review date:** November 24, 2026  
**Status:** Engineering research baseline established; clinical readiness not established

## 1. Executive Summary

CardioSense has completed nine engineering milestones addressing cohort integrity, input standardization, automated regression coverage, measurement reporting, research-model isolation, evaluation uncertainty, selection coverage, expert-reference accuracy, and failure analysis.

These milestones substantially improve software auditability and the ability to reproduce research findings from committed artifacts.

They do not establish that CardioSense is clinically safe or diagnostically reliable.

**Final decision:** Keep the deployed V1 pipeline unchanged. Retain V3 and QRS V4 as experimental research candidates.

## 2. Milestone Outcomes

| Milestone | Area | Outcome |
|---|---|---|
| M1 | Cohort reproducibility | Canonical 300-record cohort and metadata checks |
| M2 | ECG input contracts | Standardized and validated ECG input handling |
| M3 | Automated tests | Expanded deterministic regression coverage |
| M4 | Measurement reporting | Improved representation of unavailable and quality-flagged measurements |
| M5 | V3 model integrity | Frozen artifact verification and isolated research inference |
| M6 | Paired evaluation | Reproduced V1/V3 comparisons and uncertainty |
| M7 | Selection coverage | Quantified excluded and overlapping validation ECGs |
| M8 | Expert-reference accuracy | Reproduced LUDB QRS landmark-error comparisons |
| M9 | Failure analysis | Reconciled boundary shifts, counterfactuals, and reference definitions |

Completion of an engineering milestone does not imply clinical validation.

## 3. Research Model Evaluation

### Matched V1/V3 comparison

Evaluation data: 25 matched ECGs from 25 patients in PTB-XL fold 9.

| Metric | V1 | V3 |
|---|---:|---:|
| Accuracy | 64% | 68% |
| Sensitivity | 56.25% | 62.50% |
| Specificity | 77.78% | 77.78% |
| ROC-AUC | 0.7083 | 0.6597 |
| PR-AUC | 0.8700 | 0.8291 |
| Brier score | 0.2219 | 0.2313 |

V3 corrected two V1 errors and introduced one new error.

The accuracy difference was +4 percentage points, with a descriptive patient-bootstrap 95% interval of approximately −8 to +20 percentage points.

The ROC-AUC difference was −0.0486, with an interval of approximately −0.1600 to +0.0441.

**Conclusion:** The matched development-fold data does not establish V3 superiority. Model promotion is not justified.

## 4. Evaluation Coverage

The full PTB-XL fold-9 cohort contains 31 ECGs.

| Category | Recordings |
|---|---:|
| Available in both V1 and V3 | 25 |
| V1 only | 3 |
| V3 only | 3 |
| Neither | 0 |

V1 and V3 each have 28 available feature records, corresponding to 90.32% of fold 9.

Their matched intersection covers 80.65%.

Coverage also varies across the two diagnostic labels.

**Limitation:** Accuracy calculated on 25 jointly available ECGs must not be represented as accuracy on all 31 validation ECGs.

Unavailable measurements must not be treated as negative diagnostic findings.

## 5. LUDB Expert-Reference Evaluation

Previously saved measurements from LUDB recordings 31–100 were audited for the current and experimental QRS V4 delineation algorithms.

Expert-supplied R-peaks were used in the original benchmark. The audit therefore concerns QRS boundary delineation, not fully automated ECG processing.

### QRS duration accuracy

| Metric | Current | V4 |
|---|---:|---:|
| Median absolute error | 22.5 ms | 18.5 ms |
| Mean absolute error | 24.16 ms | 21.13 ms |
| 95th-percentile absolute error | 46.0 ms | 45.1 ms |

The paired mean absolute error difference was −3.03 ms (V4 minus current).

The descriptive recording-bootstrap 95% interval was approximately −5.34 to −0.71 ms.

Of 70 recordings, V4 improved duration accuracy in 24, worsened it in 6, and left it unchanged in 40.

**Conclusion:** V4 has a modest measured average duration-error advantage on this previously evaluated dataset, alongside meaningful individual regressions.

## 6. Reconciled Failure Cases

| LUDB record | Changed boundary | Absolute duration-error regression |
|---|---|---:|
| 31 | Onset shifted 34 ms later | +34 ms |
| 72 | Onset shifted 30 ms earlier | +26 ms |
| 78 | Offset shifted 24 ms earlier | +12 ms |
| 85 | Onset shifted 30 ms earlier | +6 ms |

The saved offset-only counterfactual confirms that the regressions in records 31, 72, and 85 are attributable mathematically to onset changes; record 78 is an offset-change regression.

### Reference-definition concern

In record 31, the V4 onset is closer to the expert lead-II onset but farther from the annotated global QRS onset.

These references measure different definitions and are not interchangeable.

The observations identify which boundary moved and how the measured error changed.

They do not establish the algorithmic or physiological reason for selecting that boundary.

## 7. Unresolved Engineering and Scientific Risks

**Historical reproducibility:** The reconstructed cohort manifest aligns with the checked frozen metadata, but the historical manifest byte-level SHA-256 mismatch remains unresolved. Original V1 checkpoint training lineage also remains unverified.

**QRS delineation:** Substantial individual onset and duration regressions require investigation and predefined acceptance criteria before promoting any alternative algorithm.

**R-peak detection:** The saved LUDB comparison uses expert-supplied peaks. It cannot establish end-to-end R-peak detector accuracy.

**ST measurement:** Independent expert-reference validation of ST amplitude, J-point positioning, and baseline selection remains insufficient.

**Clinical evaluation:** The experimental cohort is deliberately balanced. Fold-9 development results do not establish population-level calibration, clinical sensitivity, generalization, or safety.

**Research-model uncertainty:** V3 was evaluated on a small, quality-selected development sample and was not demonstrated to outperform V1.

**Software validation limits:** Deterministic tests protect known contracts and regression behaviors but cannot substitute for independent medical reference evaluation.

## 8. Release and Promotion Decisions

| Component | Decision |
|---|---|
| Existing V1 application | Remain unchanged; not clinically validated |
| V3 Random Forest | Research candidate only |
| Experimental QRS V4 | Not approved for promotion |
| SQI V3 | Experimental quality-control mechanism |
| Clinical triage or diagnostic claims | Not established |
| LUDB 101–200 | Preserve for a separately predefined evaluation |
| PTB-XL fold 10 | No new model evaluation performed in these milestones |

No changes to production model checkpoints or deployed triage routing are authorized by this review.

## 9. Next Research Priorities

1. **Measurement accuracy:** Investigate LUDB onset regressions using waveform morphology, individual lead boundaries, expert reference definitions, and beat-level evidence.
2. **End-to-end performance:** Evaluate R-peak detection and downstream delineation together rather than relying solely on expert-supplied peaks.
3. **ST reference validation:** Establish independent annotations and predefined error tolerances for ST amplitude, J-point timing, and isoelectric baseline measurements.
4. **Reproducibility:** Resolve historical cohort-manifest identity and recover verifiable V1 training provenance.
5. **Independent evaluation:** Define acceptance criteria, failure handling, patient-level separation, and analysis procedures before evaluating any remaining reserved set.
6. **Clinical assessment:** Require appropriate independent clinical expertise and representative external validation before any diagnostic deployment claims.

Previously evaluated records must not be presented as untouched validation data following further algorithm development.

## 10. Reproduction Commands

From the CardioSense project root:

```powershell
python -m pytest tests/test_m09_readiness.py -v
python -m scripts.audit_m09_readiness

python -m pytest tests/test_ludb_qrs_accuracy.py -v
python -m scripts.audit_ludb_qrs_accuracy

python -m scripts.audit_validation_coverage
python -m scripts.audit_paired_fold9
```

The Milestone 9 audit reads previously saved files under `docs/clinical_validation`. Its reported input SHA-256 values identify the files observed at audit time; they do not establish original historical byte identity.

## 11. Final Assessment

**Software engineering and auditability:** Improved and supported by automated regression tests and repeatable saved-artifact audits.

**Research reproducibility:** Partially verified, with documented historical provenance exceptions.

**QRS measurement reliability:** Quantified on an existing LUDB dataset, with meaningful residual errors and regressions.

**Research-model superiority:** Not established for V3.

**Clinical reliability:** Not established.

CardioSense should continue as a controlled research prototype. The next phase must prioritize independent measurement accuracy and clinical evidence, not deployment or model promotion.

---

**Review conclusion:** Nine engineering milestones completed. Scientific and clinical validation remain separate, unresolved requirements.