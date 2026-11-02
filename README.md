<div align="center">

# CardioSense

### From raw ECG signals to explainable, quality-aware cardiac analysis

**An open-source research prototype combining biomedical signal processing, interpretable ECG biomarkers, deep learning, and auditable triage rules.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1D%20CNN-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-0C55A5?logo=scipy&logoColor=white)](https://scipy.org/) [![Dataset](https://img.shields.io/badge/ECG-PTB--XL-18794E)](https://physionet.org/content/ptb-xl/1.0.3/) ![Stage](https://img.shields.io/badge/Stage-Research%20Prototype-EA8C23)

[**Getting started**](#getting-started) · [**Architecture**](#how-cardiosense-works) · [**Results**](#experimental-results) · [**Engineering log**](docs/ENGINEERING_LOG.md) · [**Roadmap**](#roadmap)

</div>

> [!IMPORTANT]
> **Research and education only.** CardioSense is **not** a clinically validated medical device, a diagnostic service, or an autonomous emergency-triage system. Experimental predictions and rule findings must not be used for patient care. See [Safety, scientific validity, and limitations](#safety-scientific-validity-and-limitations).

---

## The project in 30 seconds

An electrocardiogram (ECG) records the electrical activity of the heart. It contains clinically meaningful patterns, but recordings can also include noise, drifting baselines, or incomplete lead information. A neural network may output a prediction without explaining which signal characteristics drove it; a traditional rule engine can be interpretable yet vulnerable to flawed measurements.

**CardioSense investigates what happens when those approaches are engineered together:**

1. **Inspect the recording.** Filter noise and assess signal quality without treating every unusual waveform as an artifact.
2. **Extract interpretable measurements.** Identify R-peaks, estimate rhythm statistics, delineate the QRS/J point, and measure ST deviation across 12 leads.
3. **Apply two complementary models.** Combine a feature-based Random Forest with a 12-channel, one-dimensional convolutional neural network (1D CNN).
4. **Make decisions auditable.** Keep statistical model output separate from deterministic rule findings and the final research triage tier.
5. **Expose uncertainty in the measurements.** Attach per-feature quality metadata so questionable ST values can be identified rather than silently treated as reliable.

The goal is a **transparent engineering platform**, not a claim that software can independently diagnose acute myocardial infarction.

### Snapshot · 2 November 2026

| Engineering indicator | Observed result | Interpretation |
|---|---:|---|
| ECG records in local cohort manifest | **300** | Small experimental, deliberately constructed cohort |
| Waveform files available | **297** | 3 files unavailable locally |
| Full feature extraction with legacy SQI V1 | **251 / 297 (84.5%)** | Original extraction coverage |
| Full feature extraction with experimental SQI V3 | **295 / 297 (99.3%)** | **+44 net successful extractions**, not diagnostic accuracy |
| Focused SQI + feature-quality tests | **46 passed** | Local targeted regression verification, not the entire suite |
| Classifier feature dimensions | **23** | Unchanged by the new quality metadata |

**What is new:** SQI V3 can warn on baseline drift without automatically rejecting the whole ECG; a versioned `feature_quality_v1` contract now records potential measurement concerns. Both remain **experimental**. [Read the full investigation →](docs/ENGINEERING_LOG.md#16-experimental-sqi-v3-and-technical-artifact-review--november-2026)

---

## How CardioSense works

```text
                12-lead ECG waveform (PTB-XL; 10 s, 100 Hz)
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │ Signal processing & quality      │
                  │ 0.5–40 Hz filtering               │
                  │ SQI V1 (default) / V3 (opt-in)    │
                  └─────────────────┬────────────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │ Physiological feature extraction │
                  │ R-peaks, RR / HRV, QRS, J-point   │
                  │ ST deviation across 12 leads     │
                  └────────────┬───────────┬─────────┘
                               │           │
              23 ML features ──┘           └── Quality metadata
                      │                       (advisory; not yet enforced)
                      ▼
              ┌──────────────────────────┐         ┌─────────────────────┐
              │ Interpretable RF model   │         │ 12-lead 1D CNN      │
              │ Feature-based inference  │         │ Waveform inference  │
              └─────────────┬────────────┘         └──────────┬──────────┘
                            └────────────┬───────────────────┘
                                         ▼
                             Experimental model fusion
                                         │
                                         ▼
                       Deterministic clinical-rule evaluation
                       (contiguous leads, patient context)
                                         │
                                         ▼
                     Research tier + findings + reader / report
```

**Read this as an architecture, not a single validated deployment.** The individual components exist, and the ensemble-to-guardrail handoff has been integrated. The end-to-end enforcement of feature-quality warnings at inference and reporting boundaries is **still under development**.

### The five technical layers

<details>
<summary><strong>1 · Digital signal processing (DSP) and signal quality</strong></summary>

The software works with simultaneous 12-lead recordings. A 0.5–40 Hz filtering stage reduces baseline wander and high-frequency interference before feature extraction.

- **SQI V1 (default):** original lead scoring and acceptance rules.
- **SQI V2:** research iteration addressing overly aggressive morphology rejection.
- **SQI V3 (opt-in):** adds technical failure checks (flatline, clipping when ADC rails are known, sustained high-frequency interference), while recording baseline drift as a review warning rather than automatically rejecting an otherwise analyzable recording.

A drift warning means a measurement *may be compromised*. It **does not** mean its ST segment has been independently validated.

**Code:** [`src/dsp/`](src/dsp/) · **Audit:** [`docs/sqi_audit/`](docs/sqi_audit/)

</details>

<details>
<summary><strong>2 · Explainable ECG biomarkers</strong></summary>

The feature extractor derives heart rate, RR-based variability, QRS duration, J-point ST measurements, and territory-level summaries. For automated classifier inputs, it uses **23 numerical feature columns**; additional diagnostics may be included in the returned feature dictionary.

ST deviation used by the rule engine is measured at the **J point relative to a PR baseline**. Historical J+60 measurements are retained for comparison, not silently substituted for J-point clinical cut-points. At 100 Hz, time resolution and delineation uncertainty remain limitations.

**Code:** [`src/features/`](src/features/) · **Historical correction:** [Engineering log, Section 12](docs/ENGINEERING_LOG.md#12-st-measurement-rebuilt-at-the-j-point-hardening-sprint-p0-3)

</details>

<details>
<summary><strong>3 · Complementary machine-learning engines</strong></summary>

- **Interpretable engine:** Random Forest trained on extracted features, with historical calibration experiments.
- **Waveform engine:** 1D CNN reading the 12-channel ECG tensor directly.
- **Fusion:** experimental combination of model probabilities. Deterministic findings can change the *final tier* independently of the original statistical score.

Candidate training scripts now use **folds 1–8 for training, fold 9 for validation, and fold 10 for testing**. The candidate artifacts are separate from any older active checkpoints. Evaluation on unequal record sets is not a fair head-to-head comparison.

**Code:** [`src/models/`](src/models/) · [`src/triage/risk_fusion.py`](src/triage/risk_fusion.py)

</details>

<details>
<summary><strong>4 · Traceable deterministic rules and reporting</strong></summary>

The rule engine evaluates contiguous ST-elevation lead pairs using lead- and demographic-specific cut-points where applicable, and distinguishes findings that warrant escalation from those that warrant clinician review. The project also contains research-facing reporting and a Streamlit reader.

A key historical fix was to stop making decisions using a J+60 value against criteria defined at the J point. Another was to replace single-lead territory maxima with contiguous-lead rules. These changes reduced spurious triggering in a **selected research cohort**; they do not establish clinical specificity.

**Code:** [`src/triage/`](src/triage/) · [`src/dashboard/`](src/dashboard/) · **Details:** [Engineering log, Sections 11–14](docs/ENGINEERING_LOG.md#11-deterministic-guardrail-layer-audit--system-level-verification-hardening-sprint-p0-1)

</details>

<details>
<summary><strong>5 · Feature-quality contract (new)</strong></summary>

Successful extraction now includes `quality_contract` metadata alongside the original numerical features. The initial policy, `feature_quality_v1`, reports:

| Status | Meaning |
|---|---|
| `review_required` | A detected quality concern may affect the feature |
| `unavailable` | The required measurement or contributing lead is unavailable |
| `not_validated` | No targeted issue established; **not** a declaration of reliability |

For example, baseline drift in **V5** can mark `st_V5` and relevant ST aggregates for review. The current aggregate policy is intentionally conservative because it does not yet track the **exact lead that produced each maximum**. The feature values remain unchanged, and downstream ML/rule execution does **not** yet enforce these warnings.

**Code:** [`src/features/quality_contract.py`](src/features/quality_contract.py) · **Details:** [Engineering log, Section 17](docs/ENGINEERING_LOG.md#17-feature-quality-contract-v1--november-2026)

</details>

---

## Experimental results

### 1. The SQI investigation: 251 → 295 successful ECGs

The original quality gate rejected waveforms partly because sharp or unusual cardiac morphology could be scored as poor quality. SQI V3 was developed to distinguish candidate acquisition failures from morphology that should not be automatically discarded.

| Full pipeline outcome | SQI V1 | Experimental SQI V3 |
|---|---:|---:|
| Manifest entries | 300 | 300 |
| Available waveform files | 297 | 297 |
| Successful feature extractions | **251** | **295** |
| Rejected after waveform loading | 46 | 2 |
| Missing waveform files | 3 | 3 |
| Extraction success / available waveforms | **84.5%** | **99.3%** |

Paired comparison on the 297 available recordings:

| Transition | ECGs |
|---|---:|
| Successful with both V1 and V3 | 250 |
| Rejected with V1; successful with V3 | **45** |
| Successful with V1; rejected with V3 | 1 |
| Rejected with both | 1 |

**Interpretation:** V3 produces **44 additional usable extraction outputs on net**. It does **not** demonstrate improved detection of myocardial infarction. Baseline drift is now treated as a review condition; ST-fidelity analysis is still needed before those measurements are trusted downstream. The same cohort helped guide detector changes, so it is **development evidence, not independent validation**.

Data: [`full_pipeline_v1_vs_v3.csv`](docs/sqi_audit/full_pipeline_v1_vs_v3.csv) · Method and error analysis: [Engineering log, Section 16](docs/ENGINEERING_LOG.md#16-experimental-sqi-v3-and-technical-artifact-review--november-2026)

### 2. Model evaluation: candidate checkpoints, different denominators

These are **candidate-model measurements**, not a claim about the final deployed system:

| Fold-10 candidate | 12-lead CNN | Interpretable Random Forest |
|---|---:|---:|
| ECGs evaluated | 31 | 24 |
| ROC-AUC | 0.9042 | 0.7357 |
| Accuracy | 87.1% | 66.7% |
| Sensitivity | 81.25% | 57.14% |
| Specificity | 93.33% | 80.0% |

**Why no winner is declared:** The RF evaluation used fewer records because earlier feature extraction omitted cases. Since the two engines did **not** face the same test set, the numbers are not a controlled model comparison. Older model/ensemble benchmarks used other protocols and are retained—clearly labeled as historical—in the [engineering log](docs/ENGINEERING_LOG.md#15-ensemble-handoff-and-corrected-model-evaluation--octobernovember-2026).

Fold 10 was also inspected during SQI development; it cannot now be treated as an untouched evaluation set for the tuned SQI policy. A genuinely independent set is required for credible final claims.

### 3. Deterministic-rule hardening

Earlier research audits found that rules using **J+60 values** and single-lead territory maxima could trigger on healthy-labeled ECGs. The correction moved the relevant ST input to **the J point** and rebuilt the rule criteria around **contiguous leads**.

| Historical implementation on selected 251-record cohort | Healthy-labeled emergency escalations |
|---|---:|
| Original rules with J+60 ST values | 32/124 (**25.8%**) |
| Original rules after J-point correction | 4/124 (**3.2%**) |
| Rebuilt contiguous-lead rules at J | 3/124 (**2.4%**) |

These figures describe **rule triggering**, not validated false-positive rates for acute MI. The broad PTB-XL label does not establish who has an acute occlusion, and the later V3-expanded population has not been independently evaluated end to end.

---

## Getting started

### Requirements

- Python **3.10+** and `pip`.
- Git; PowerShell commands below assume Windows 10/11.
- ECG waveform files for waveform-based demonstrations and cohort benchmarks.
- Appropriate model checkpoints for inference paths that load trained weights. The repository may not ship the latest **candidate** binaries.

### Install

```powershell
# 1. Clone
git clone https://github.com/amederhourhi/HT-project.git
cd HT-project

# 2. Create and activate an environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install project and development tools
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For Linux/macOS, activate with `source .venv/bin/activate` and use `python3 -m venv .venv` if needed.

### Run targeted verification

This is the focused suite used during the recent SQI/feature-quality work:

```powershell
python -m pytest tests/test_sqi_v3.py tests/test_sqi_v3_separation.py tests/test_sqi_artifact_validation.py tests/test_sqi_clipping_regression.py tests/test_feature_sqi_selection.py tests/test_feature_quality_contract.py -q
```

**Last reported local result:** `46 passed`. Some other tests require downloaded ECG records, trained checkpoints, or additional setup. Do not equate this targeted result with a full repository test pass.

To explore individual subsystems, see [`tests/`](tests/) and the exact reproduction notes in the [engineering log](docs/ENGINEERING_LOG.md).

### Run the research reader

```powershell
streamlit run src/dashboard/app.py
```

This opens the Streamlit-based research interface. Data and model requirements depend on the reader's workflow. The separate [`web/`](web/) Vite/React project is a **frontend foundation**, not a verified replacement for the reader.

### Reproduce the V1 versus V3 extraction comparison

With `data/cohort/cohort_manifest.csv` and referenced waveforms already available:

```powershell
python scripts/compare_sqi_v3.py
```

The script runs complete feature extraction under both SQI policies and writes [`docs/sqi_audit/full_pipeline_v1_vs_v3.csv`](docs/sqi_audit/full_pipeline_v1_vs_v3.csv). It does not overwrite original model checkpoints or retrain a model.

> **Dataset:** This project uses [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/). The 300-record manifest is an **experimental subset**, not the full PTB-XL dataset. Raw recordings and local cohort files are generally Git-ignored. Consult [`src/data/`](src/data/) and the [engineering log](docs/ENGINEERING_LOG.md) for dataset construction and caveats.

---

## Repository map

```text
HT-project/
├── src/
│   ├── data/              PTB-XL ingestion and cohort tooling
│   ├── dsp/               Filters and SQI V1 / V2 / V3
│   ├── features/          Beat detection, fiducials, feature extraction,
│   │                      quality_contract.py, batch processing
│   ├── models/            Random Forest and CNN definitions/training
│   ├── triage/            Lead rules, guardrails, ensemble fusion, reporting
│   ├── evaluation/        Audits and evaluation metrics
│   └── dashboard/         Streamlit research reader
├── web/                   Separate React/Vite frontend foundation
├── tests/                 Unit, regression, integration and visual checks
├── scripts/               SQI comparison and signal inspection tools
├── data/                  Dataset metadata, tracked examples and local files
├── models/checkpoints/    Model artifacts (availability varies)
├── docs/
│   ├── ENGINEERING_LOG.md Detailed chronology, experiments and corrections
│   ├── sqi_audit/         Comparison CSVs; some plots are local-only
│   └── figures/           Historical engineering plots
├── pyproject.toml         Packaging / development configuration
└── README.md
```

### Where to look if you are contributing

| If you want to understand… | Start here |
|---|---|
| Filtering and artifact detection | [`src/dsp/`](src/dsp/) and [`tests/test_sqi_artifact_validation.py`](tests/test_sqi_artifact_validation.py) |
| J-point and ST measurement | [`src/features/fiducials.py`](src/features/fiducials.py) and [`tests/test_fiducials.py`](tests/test_fiducials.py) |
| Why a feature was flagged | [`src/features/quality_contract.py`](src/features/quality_contract.py) |
| CNN / RF training protocols | [`src/models/`](src/models/) and [engineering log, Section 15](docs/ENGINEERING_LOG.md#15-ensemble-handoff-and-corrected-model-evaluation--octobernovember-2026) |
| Guardrail rules and explainability | [`src/triage/`](src/triage/) and [`tests/test_triage.py`](tests/test_triage.py) |
| What was measured, changed, or superseded | [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md) |

---

## Roadmap

The next work is **not another broad SQI rewrite**. The engineering priority is to make quality metadata operationally meaningful.

| Priority | Milestone | Why it matters |
|---|---|---|
| **Next** | **ST measurement provenance** | Record the actual source lead for each global and territory ST maximum; refine quality flags accordingly |
| **Next** | **Quality-aware rule and model handoff** | Carry review/unavailable states through triage and reports; define explicit review/abstention behavior |
| Following | Versioned feature dataset and RF retraining | Prevent mixed SQI policies, silent cohort changes, and incompatible feature distributions |
| Following | Paired CNN/RF/ensemble evaluation | Compare engines on the **same** patient records, with calibration and uncertainty estimates |
| Longer term | Independent data, 500 Hz analysis and clinical collaboration | Evaluate generalization, landmark precision, and measurement validity beyond the development cohort |
| Exploratory | Hardware and expanded frontend | Develop the proposed ESP32 acquisition path and mature the React interface without overstating readiness |

Milestones, unresolved defects and historical decisions are tracked in the [engineering log](docs/ENGINEERING_LOG.md#19-next-phase--measurement-provenance-and-quality-aware-inference).

---

## Safety, scientific validity, and limitations

CardioSense is an **educational and research software prototype**. It has not undergone prospective clinical validation, external multicenter evaluation, formal device verification/validation, or regulatory approval. It must **not** be used for real patient diagnosis, infarction exclusion, cath-lab activation, or autonomous treatment recommendations.

Specific limitations worth understanding:

- **Labels:** In this experiment, “infarction” is a broad dataset label; it is not a verified acute ST-elevation event. Infarction classification and urgent ischemia triage are different targets.
- **Sampling and selection:** The experimental manifest is small and not a representative clinical population. SQI policy changes also change which records appear in feature-based training and evaluation.
- **Signal fidelity:** A filtered ECG can preserve recognizable QRS complexes while its ST measurement remains unreliable. The new quality contract reports concerns but does not yet establish validity.
- **Model evaluation:** Candidate RF/CNN scores have different denominators. Their reported performance cannot establish a fair comparison, calibrated clinical risk, or patient safety.
- **Rules and interface:** Contiguous-lead rule execution is research logic. Quality warnings are not yet enforced throughout inference and reporting. Hardware and the React interface remain exploratory components.

The architecture draws on concepts relevant to medical software engineering, but **does not claim IEC 62304 / ISO 13485 compliance, CE marking, FDA authorization, or clinically certified biomarkers**.

---

## Project documentation

**[Engineering & Verification Log →](docs/ENGINEERING_LOG.md)** — the authoritative chronological record of designs, experiments, numerical results, corrected assumptions, implementation changes, known defects, and reproduction commands.

**[PTB-XL dataset →](https://physionet.org/content/ptb-xl/1.0.3/)** — original source of the ECG recordings used in the prototype; dataset authors and terms should be cited when using the data.

---

<div align="center">

**CardioSense — building an ECG analysis system that makes both predictions and measurement limitations inspectable.**

*Active research development · November 2026*

</div>
