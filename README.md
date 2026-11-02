# CardioSense

**Explainable ECG analysis and experimental clinical decision support**  
*Python · SciPy · PyTorch · scikit-learn · PTB-XL · Streamlit · React/Vite (prototype)*

> **Status (2 November 2026): research/engineering prototype — not a medical device.** CardioSense is not clinically validated, approved for clinical use, or suitable for autonomous diagnosis, cath-lab activation, or patient disposition. The system's current research data labels distinguish broad infarction-related records from normal ECGs; they do **not** establish acute coronary occlusion.

[Engineering & verification log](docs/ENGINEERING_LOG.md) · [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/)

## At a glance

CardioSense explores how **digital signal processing, measurable ECG biomarkers, machine learning, and auditable decision rules** can work together. It accepts 12-lead ECG waveforms, filters them, evaluates signal quality, identifies cardiac landmarks, produces interpretable features, and supports an experimental dual-model inference and triage pipeline.

**Current verified engineering results (300-record local manifest):**

| Measure | Legacy SQI V1 | Experimental SQI V3 |
|---|---:|---:|
| Available ECG waveforms | 297 | 297 |
| Full feature extraction succeeded | 251 | **295** |
| Full feature extraction rejected | 46 | **2** |
| Missing waveform files | 3 | 3 |
| Extraction success among available signals | 84.5% | **99.3%** |

The comparison ran the complete `extract_clinical_features()` path for both versions on the same 300 manifest entries. V3 **recovered 45** V1-rejected records and **newly rejected 1** V1-accepted record (net +44). The 99.3% figure measures **pipeline completion, not signal-quality accuracy, MI sensitivity, or clinical safety**. The change deliberately treats baseline drift as a review warning instead of rejecting the entire recording. [Methods and audit](docs/ENGINEERING_LOG.md#16-experimental-sqi-v3-and-technical-artifact-review--november-2026)

Latest focused local verification: **46/46 tests passed** across SQI and feature-quality contract tests. This is **not** a full-suite or clinical-validation result.

## What the system actually does

```text
PTB-XL 12-lead ECG (typically 10 seconds at 100 Hz)
         |
         v
Filtering (0.5–40 Hz) + signal-quality assessment
         |         └─ V1: legacy default; V3: experimental opt-in
         v
R-peaks / rhythm + QRS / J-point / per-lead ST measurement
         |
         +------> 23 numerical ML features
         |                 + feature-quality metadata (new, advisory)
         v
Experimental Random Forest + 12-channel 1D CNN
         |
         v
Model fusion + deterministic contiguous-lead rule engine
         |
         v
Tier / findings / clinician-facing research reader
```

The **quality contract currently reports warnings but does not yet control downstream model use or the clinical rule engine**. A successful extraction must not be interpreted as proof that ST measurements are reliable.

### Main components

| Area | Key paths | Purpose / maturity |
|---|---|---|
| Data ingestion | `src/data/`, `data/` | PTB-XL sample/cohort workflows; raw/cohort waveforms typically local and ignored by Git |
| DSP | `src/dsp/filters.py`, `sqi.py`, `sqi_v2.py`, `sqi_v3.py` | Bandpass filtering; legacy SQI and **experimental** morphology/artifact-aware quality policy |
| Clinical features | `src/features/fiducials.py`, `r_peaks.py`, `extract_features.py` | R-peaks, intervals, J-point ST deviation, territory summaries, 23 classifier inputs |
| Quality metadata | `src/features/quality_contract.py` | `feature_quality_v1`: per-feature `not_validated` / `review_required` / `unavailable` indicators; advisory only |
| ML | `src/models/train_interpretable.py`, `train_cnn.py` | Candidate Random Forest / CNN training using folds 1–8 train, 9 validation, 10 test |
| Rules + fusion | `src/triage/leads.py`, `triage_rules.py`, `risk_fusion.py` | Contiguous-lead criteria, patient context, deterministic findings, ensemble/guardrail integration |
| Evaluation | `src/evaluation/`, `scripts/`, `docs/sqi_audit/` | Guardrail audits, cohort alignment, experimental comparisons and diagnostic figures |
| Interface | `src/dashboard/app.py`, `web/` | Streamlit research reader; separate Vite/React frontend foundation (not a production deployment) |
| Tests | `tests/` | Unit, integration, regression, waveform and reader checks |

The ESP32 single-lead hardware path is a **planned prototype**; it should not be confused with the current 12-lead software evaluation.

## Why the project matters

An ECG classifier alone is difficult to audit. CardioSense instead exposes **what was measured**, **which leads were assessed**, **which clinical rule fired**, and **where signal quality may compromise interpretation**. This makes it a useful engineering research platform while also exposing hard problems: landmark uncertainty, noisy signals, selection bias, class-label mismatch, dataset leakage and agreement between the model and rule-based layers.

The rule engine has already been revised from single-lead territory maxima to contiguous-lead findings with demographic V2–V3 thresholds. An earlier audit showed **32/124 healthy-labeled records (25.8%)** escalated by older J+60-based rules; after correcting J-point measurement and rewriting rules, an earlier **251-record cohort** audit reported **3/124 (2.4%)**. These are **historical rule-behavior measurements on a selected cohort**, not prospective clinical specificity, and the current V3-expanded cohort has not received an equivalent end-to-end audit.

## Model results — read the denominators

The current **candidate** models were retrained/evaluated on a corrected fold protocol (1–8 train, 9 validation, 10 test), but **their test populations differ** due to feature availability:

| Candidate model | Fold-10 ECGs evaluated | ROC-AUC | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| 1D CNN | 31 | 0.9042 | 87.1% | 81.25% | 93.33% |
| Interpretable RF | 24 | 0.7357 | 66.7% | 57.14% | 80.0% |

**Do not compare these scores as a controlled head-to-head test.** They have different sample sizes and the candidate checkpoints are not automatically the active models. Older performance and ensemble figures are preserved with their original context in the [engineering log](docs/ENGINEERING_LOG.md). Because fold-10 ECGs were inspected during SQI development, it is **no longer an untouched validation set for the revised SQI policy**.

## Getting started

### Prerequisites

- Python **3.10+**; Windows PowerShell examples below (Linux/macOS commands vary).
- The PTB-XL data or supplied sample recordings for waveform-dependent demonstrations. Downloading the entire cohort is optional for running most pure unit tests.
- Model checkpoints for full ML/ensemble inference; these are not required for every test.

```powershell
# Clone and enter the repository
git clone https://github.com/amederhourhi/HT-project.git
cd HT-project

# Create and activate a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the project and development dependencies
python -m pip install -e ".[dev]"

# Run fast targeted tests (no full cohort benchmark required)
python -m pytest tests/test_sqi_v3.py tests/test_sqi_v3_separation.py tests/test_sqi_artifact_validation.py tests/test_sqi_clipping_regression.py tests/test_feature_sqi_selection.py tests/test_feature_quality_contract.py -q
```

Additional test coverage and dataset requirements are documented in the [engineering log](docs/ENGINEERING_LOG.md). Running `pytest` across every script may require local waveforms, checkpoints, or a GUI-compatible environment; review the individual tests.

### Running the research interface

```powershell
streamlit run src/dashboard/app.py
```

The separate `web/` directory contains a Vite/React frontend foundation. It should not be assumed to replicate all Streamlit reader features or have a production deployment.

### Reproduce the SQI comparison

With `data/cohort/cohort_manifest.csv` and its referenced waveform files available locally:

```powershell
python scripts/compare_sqi_v3.py
```

Output: `docs/sqi_audit/full_pipeline_v1_vs_v3.csv`. The script compares complete extraction outcomes and does **not** retrain or replace model checkpoints. The CSV represents a local experimental run, not a public validation benchmark.

## Key engineering decisions and present limits

- **ST is measured at the J point**, referenced to the PR isoelectric baseline, rather than using a J+60 legacy reading as the rule input. At 100 Hz, timing resolution remains limited.
- **SQI V3 remains opt-in** (`sqi_version="v3"`). Flatlines, known-rail clipping and sustained HF interference can fail the quality gate; baseline drift is tracked as a warning. The warning **does not make affected ST values safe for automated use**.
- **Feature Quality Contract v1** adds per-feature metadata without altering the existing 23 numerical classifier inputs. Some aggregate ST flags are conservative: they consider potentially contributing leads, not necessarily the actual lead that supplied the maximum.
- **Ground truth is limited**: a broad infarction-vs-normal target is not the same as angiographically verified acute MI/STEMI. Small samples and a selected cohort restrict generalization.
- **End-to-end quality-aware triage remains unfinished**: current warnings are not yet reliably enforced at the model/rule boundary; candidate model calibration, aligned ensemble evaluation, external validation and formal traceability remain open.
- **Clinical claims are deliberately limited**: this repository demonstrates research software and software tests, not adherence certification to IEC 62304/ISO 13485 or FDA/CE clearance.

## Roadmap

1. **ST feature provenance:** record the exact source lead and measurement validity for per-territory and global ST extrema.
2. **Quality-aware inference:** propagate warning/unavailable states through rule execution, API and reports; define safe abstention/clinician-review behavior.
3. **Reproducible V3 dataset:** version feature extraction, track case inclusion/exclusion, ensure no leakage, and retrain RF against stable inputs.
4. **Fair aligned evaluation:** evaluate candidate CNN/RF/fusion on the *same records*, with threshold calibration and honest uncertainty intervals; reserve a genuinely independent test/external set.
5. **Operational hardening:** traceable SBAR reports, checkpoint/version hashes, CI, robust interfaces, reproducible data provenance, and prospective clinical collaboration before any deployment claims.

For chronological experiments, failures, corrected assumptions, and verification commands, see **[ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md)**. The older numerical tables there are part of the historical record and are not all current benchmarks.

## Responsible use

**For research and education only.** CardioSense is not a licensed medical device and must not be used to diagnose, exclude myocardial infarction, make emergency-routing decisions, or substitute for clinical evaluation. Medical interpretations and model confidence require independent expert validation and regulatory work before patient use.
