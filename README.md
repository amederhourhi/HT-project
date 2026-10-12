# CardioSense 
### Explainable Clinical Decision-Support System for Early Myocardial Infarction & Ischemia

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Signal Processing](https://img.shields.io/badge/DSP-SciPy%20Zero--Phase-00599C.svg)](https://scipy.org/)
[![Clinical Data](https://img.shields.io/badge/Dataset-PTB--XL%20(PhysioNet)-green.svg)](https://physionet.org/content/ptb-xl/1.0.3/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange.svg)]()

> **CardioSense** is a prototype clinical decision-support system (CDSS) designed to detect early electrophysiological signs of **Myocardial Infarction (heart attacks)** and **acute myocardial ischemia** from standard 12-lead ECGs and portable single-lead hardware, assisting clinicians in rapid triage and patient routing.

---

## The Clinical Mission

Time is myocardium. When a coronary artery occludes during an acute heart attack, millions of heart muscle cells die every minute. Early, confident detection on an ECG is the single most critical factor in routing patients to emergency cardiac catheterization (PCI).

However, real-world ECG analysis faces two major hurdles:
1. **Noisy Signals**: Loose electrodes, patient shivering, and breathing drift produce artifacts that mimic heart attacks, causing false alarms and hospital overcrowding.
2. **The "Black-Box" AI Dilemma**: Pure deep-learning models lack clinical interpretability. Clinicians cannot verify *why* a neural network triggered an alarm.

**CardioSense bridges this gap** using a **dual-engine architecture**: pairing clinically certified cardiological measurements (ST elevation, J-point shifts, QRS duration, HRV) with raw deep waveform representations, gated by strict signal-quality checks.

---

## System Architecture

```text
                       [ Patient ECG Input ]
              (12-Lead PTB-XL Dataset or ESP32 Hardware)
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │        1. Preprocessing & Quality Gate          │
        │  • 0.5–40 Hz Zero-Phase Butterworth Bandpass    │
        │  • Signal Quality Index (SQI): Flatline,        │
        │    Clipping, and Kurtosis-based rejection       │
        └───────────────────────┬─────────────────────────┘
                                │ (If SQI >= 0.60)
                                ▼
         ┌───────────────────────────────────────────────┐
         │             Dual-Engine Inference             │
         ├───────────────────────┬───────────────────────┤
         │  Engine A: Classical  │  Engine B: Deep Wave  │
         │  • R-Peak & HRV       │  • Multichannel 1D    │
         │  • Fiducial Landmark  │    ResNet / CNN       │
         │  • ST Elevation @ J+60│  • Latent Repolar-    │
         │  • QRS Duration       │    ization Dynamics   │
         └───────────┬───────────┴───────────┬───────────┘
                     │                       │
                     └───────────┬───────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │            2. Calibrated Risk Fusion            │
        │  • Platt-Scaled Probability Calibration         │
        │  • Explainability Layer (Lead & Segment XAI)    │
        └───────────────────────┬─────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────┐
        │          3. Clinical Triage Interface           │
        │  • Triage Routing: Routine | Review | Emergent  │
        │  • Grounded LLM Clinical Summary (Guardrailed)  │
        │  • Clinician Dashboard (Streamlit / FastAPI)    │
        └─────────────────────────────────────────────────┘
```

---

## Core Capabilities

### 1. Robust Signal Quality Control (SQI)
- Automated gatekeeper evaluating lead integrity before inference.
- Rejects disconnected leads (variance $\sigma^2 < 10^{-4}$), amplifier saturation/clipping, and violent motion bursts.
- Tested on clinical datasets: **12/12 leads approved (0.914 SQI)**, with instantaneous **0.000 SQI rejection** on unattached channels.

### 2. Deterministic Cardiological Biomarkers
- **Zero-Phase Filtering (`filtfilt`)**: Eliminates respiration baseline wander with **0.0 ms phase delay**, preserving true PR and QT intervals.
- **R-Peak Detector**: Derivative-squared energy transformation with dynamic thresholding and physiological refractory lockout.
- **ST-Segment Deviation**: Pinpoints the **J-point** and quantifies microvolt shifts at **$J + 60\text{ ms}$** across all 12 leads relative to the isoelectric PR baseline (essential for STEMI criteria).

### 3. Dual-Model Decision Support
- Combines interpretable clinical rules (Random Forest with Platt calibration, **0.853 ROC-AUC**) with end-to-end 1D convolutional neural networks directly on raw multichannel voltages (**0.860 ROC-AUC**, **92.6% Specificity**).

### 4. Hardware Streaming Prototype *(Upcoming)*
- Live single-lead ECG capture using an ESP32 microcontroller and analog front-end (AFE), demonstrating edge-to-cloud clinical triage.

---

## Repository Structure

```text
HT-project/
├── data/
│   ├── raw/             # Local PTB-XL database metadata
│   ├── samples/         # Sample 100 Hz 12-lead records
│   ├── cohort/          # Balanced 300-patient training cohort
│   └── processed/       # Tabular extracted clinical features CSV
├── src/
│   ├── data/            # PTB-XL automated ingestion & cohort downloader
│   ├── dsp/             # Zero-phase filters & Signal Quality Index (SQI)
│   ├── features/        # R-peak detector, HRV, J-point ST landmarking, batch extractor
│   ├── models/          # Interpretable Random Forest & PyTorch 1D-CNN
│   └── triage/          # Risk fusion & explainability (XAI)
├── tests/               # Visual verification & unit tests
├── docs/
│   ├── ENGINEERING_LOG.md  # Detailed clinical lab journal & verification data
│   └── figures/            # Clinical verification plots (Figures 01 to 08)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Quickstart Guide

### 1. Clone & Set Up Environment
```bash
# Clone repository
git clone https://github.com/amederhourhi/HT-project.git
cd HT-project

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Verification Tests
```bash
# 1. Inspect raw 12-lead waveforms
python tests/test_inspect_ecg.py

# 2. Verify zero-phase baseline filter
python tests/test_filters.py

# 3. Test R-peak detector and HRV metrics
python tests/test_r_peaks.py

# 4. Run Signal Quality Index (SQI) checks
python tests/test_sqi.py

# 5. Run fiducial landmarking and ST analysis
python tests/test_fiducials.py
```

---

## 📊 Verification & Engineering Log

Detailed mathematical derivations, filter specs, clinical verification tables, and benchmark graphs are fully documented in our lab journal:

**[Read the Full Engineering & Verification Log](docs/ENGINEERING_LOG.md)**

---

## ⚠️ Medical Disclaimer

> [!WARNING]
> ### Medical Device & Clinical Disclaimer
> **This project was primarily built for learning and research purposes.**  
> It is a prototype clinical decision-support system intended solely for scientific exploration, algorithmic evaluation, and education. It is **not** an FDA-cleared, CE-marked, or clinically certified medical diagnostic device.  
> 
> CardioSense is designed to assist and support qualified clinicians—it must **never** be used as an autonomous diagnostic tool, nor should it ever replace the clinical judgment, physical examination, or diagnostic decisions of a licensed healthcare professional. Any suspected acute cardiac event or potentially life-threatening symptom requires immediate consultation with certified emergency medical services.