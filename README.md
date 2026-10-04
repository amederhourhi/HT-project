### Explainable Clinical Decision-Support System for Early Myocardial Infarction & Ischemia

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Signal Processing](https://img.shields.io/badge/DSP-SciPy%20Zero--Phase-00599C.svg)](https://scipy.org/)
[![Clinical Data](https://img.shields.io/badge/Dataset-PTB--XL%20(PhysioNet)-green.svg)](https://physionet.org/content/ptb-xl/1.0.3/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange.svg)]()

> **A prototype clinical decision-support system (CDSS) designed to detect early electrophysiological signs of **Myocardial Infarction (heart attacks)** and **acute myocardial ischemia** from standard 12-lead ECGs and portable single-lead hardware, assisting clinicians in rapid triage and patient routing.

---

##  The Clinical Mission

Time is myocardium. When a coronary artery occludes during an acute heart attack, millions of heart muscle cells die every minute. Early, confident detection on an ECG is the single most critical factor in routing patients to emergency cardiac catheterization (PCI).

However, real-world ECG analysis faces two major hurdles:
1. **Noisy Signals**: Loose electrodes, patient shivering, and breathing drift produce artifacts that mimic heart attacks, causing false alarms and hospital overcrowding.
2. **The "Black-Box" AI Dilemma**: Pure deep-learning models lack clinical interpretability. Clinicians cannot verify *why* a neural network triggered an alarm.

**CardioSense bridges this gap** using a **dual-engine architecture**: pairing clinically certified cardiological measurements (ST elevation, J-point shifts, QRS duration, HRV) with raw deep waveform representations, gated by strict signal-quality checks.

---

##  System Architecture
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


---

##  Core Capabilities

### 1. Robust Signal Quality Control (SQI)
- Automated gatekeeper evaluating lead integrity before inference.
- Rejects disconnected leads (variance $\sigma^2 < 10^{-4}$), amplifier saturation/clipping, and violent motion bursts.
- Tested on clinical datasets: **12/12 leads approved (0.914 SQI)**, with instantaneous **0.000 SQI rejection** on unattached channels.

### 2. Deterministic Cardiological Biomarkers
- **Zero-Phase Filtering (`filtfilt`)**: Eliminates respiration baseline wander with **0.0 ms phase delay**, preserving true PR and QT intervals.
- **R-Peak Detector**: Derivative-squared energy transformation with dynamic thresholding and physiological refractory lockout.
- **ST-Segment Deviation**: Pinpoints the **J-point** and quantifies microvolt shifts at **$J + 60\text{ ms}$** across all 12 leads relative to the isoelectric PR baseline (essential for STEMI criteria).

### 3. Dual-Model Decision Support *(In Development)*
- Combines interpretable clinical rules (gradient-boosted trees on fiducial markers) with end-to-end 1D convolutional neural networks directly on raw multichannel voltages.
- Calibrated risk scoring using isotonic regression.

### 4. Hardware Streaming Prototype *(Upcoming)*
- Live single-lead ECG capture using an ESP32 microcontroller and analog front-end (AFE), demonstrating edge-to-cloud clinical triage.

---

##  Repository Structure

```text
HT-project/
├── data/
│   ├── raw/             # Local PTB-XL database metadata
│   └── samples/         # Downloaded 100 Hz 12-lead sample records
├── src/
│   ├── data/            # PTB-XL automated ingestion & parsers
│   ├── dsp/             # Zero-phase filters & Signal Quality Index (SQI)
│   ├── features/        # R-peak detector, HRV, & J-point ST landmarking
│   ├── models/          # Interpretable ML & 1D-CNN architectures
│   └── triage/          # Calibrated risk scoring & clinical routing
├── tests/               # Visual verification & automated unit tests
├── docs/
│   ├── ENGINEERING_LOG.md  # Detailed lab journal, verification data & plots
│   └── figures/            # Clinical verification plots
├── requirements.txt     # Python dependencies
└── README.md
```

> [!WARNING]
> ### Medical Device & Clinical Disclaimer
> **This project was primarily built for learning purposes . it is a prototype clinical decision-support research system.**  
> It is intended solely for scientific research, algorithmic evaluation, and educational demonstration. It is **not** an FDA-cleared, CE-marked, or clinically certified medical diagnostic device.  
>It is designed to assist and support qualified clinicians—it must **never** be used as an autonomous diagnostic tool, nor should it ever replace the clinical judgment, physical examination, or diagnostic decisions of a licensed healthcare professional. Any suspected acute cardiac event or potentially life-threatening symptom requires immediate consultation with certified emergency medical services.