"""
CardioSense - Ensemble Risk Fusion & Explainability Verification Test
Evaluates Patient 1 (Normal) vs Patient 8 (Chronic/Healed Infarct) and
validates coronary territory ischemic attribution without dirtying git.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import wfdb

from src.triage.risk_fusion import CardioSenseEnsemble

# ---------------------------------------------------------
# Block 1: Parse Arguments & Setup Paths
# What it does: Configures CLI options and target output directories.
# How it does it: Uses argparse to make saving documentation figures opt-in.
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Test dual-engine risk fusion and coronary localization.")
parser.add_argument("--save-figure", action="store_true", help="Explicitly update docs/figures/09_risk_explainability.png")
args, _ = parser.parse_known_args()

project_root = Path(__file__).resolve().parents[1]
samples_dir = project_root / "data" / "samples" / "records100" / "00000"
figures_dir = project_root / "docs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

rec_norm = samples_dir / "00001_lr"
rec_mi = samples_dir / "00008_lr"

cases = [
    ("Patient 1 (Normal Control)", rec_norm),
    ("Patient 8 (Inferior Infarction / Healed)", rec_mi),
]

# ---------------------------------------------------------
# Block 2: Initialize Ensemble Pipeline (with Cold-Start Warning)
# What it does: Loads the trained Random Forest and 1D-CNN checkpoints.
# How it does it: Catches missing checkpoint files and alerts the user cleanly.
# ---------------------------------------------------------
print("Initializing CardioSense Ensemble Pipeline...")
try:
    ensemble = CardioSenseEnsemble()
except FileNotFoundError as err:
    print(f"\n[SKIP] Cannot run full ensemble test: {err}")
    print("Train checkpoints first using: python src/models/train_interpretable.py")
    sys.exit(0)

# ---------------------------------------------------------
# Block 3: Run End-to-End Prediction
# What it does: Runs both test cases through feature extraction and models.
# How it does it: Calls ensemble.predict() and prints key biomarkers & tiers.
# ---------------------------------------------------------
evaluations = []
for label, rec_path in cases:
    raw_sig, fields = wfdb.rdsamp(str(rec_path))
    fs = fields["fs"]
    lead_names = fields["sig_name"]

    res = ensemble.predict(raw_sig, fs=fs, lead_names=lead_names)
    evaluations.append((label, res))

    print("\n" + "=" * 65)
    print(f"  EVALUATION REPORT: {label.upper()}")
    print("=" * 65)
    print(f"Risk Score                  : {res['risk_score_pct']}%  [{res['risk_tier']}]")
    print(f"Clinical Recommendation     : {res['triage_action']}")
    print(f"Interpretable RF Engine     : {res['engine_probabilities']['interpretable_rf'] * 100:.1f}%")
    print(f"Deep Waveform CNN Engine    : {res['engine_probabilities']['deep_cnn'] * 100:.1f}%")
    print(f"Suspected Culprit Vessel    : {res['culprit_artery']}")
    print(f"Primary Ischemic Territory  : {res['primary_ischemic_territory']}")
    print("\nCoronary Territory Ischemic Burden:")
    for t_name, t_info in res["territory_breakdown"].items():
        print(f"  - {t_name:24s}: Max ST = {t_info['max_st_mv']:+.3f} mV")
    print(f"\nConduction: HR = {res['clinical_biomarkers']['heart_rate_bpm']} BPM | QRS = {res['clinical_biomarkers']['qrs_duration_ms']} ms")
    print("=" * 65)

# ---------------------------------------------------------
# Block 4: Plot & Conditionally Save Figure 09
# What it does: Plots dual-engine risk and regional ST territory comparisons.
# How it does it: Writes to docs/figures only if --save-figure flag was supplied.
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: Model probability breakdown
labels = ["Patient 1 (NORM)", "Patient 8 (MI)"]
x = np.arange(len(labels))
width = 0.25

rf_scores = [evaluations[0][1]["engine_probabilities"]["interpretable_rf"] * 100,
             evaluations[1][1]["engine_probabilities"]["interpretable_rf"] * 100]
cnn_scores = [evaluations[0][1]["engine_probabilities"]["deep_cnn"] * 100,
              evaluations[1][1]["engine_probabilities"]["deep_cnn"] * 100]
fused_scores = [evaluations[0][1]["risk_score_pct"],
                evaluations[1][1]["risk_score_pct"]]

ax1.bar(x - width, rf_scores, width, label="Engine A: Interpretable RF", color="steelblue")
ax1.bar(x, cnn_scores, width, label="Engine B: Deep 1D-CNN", color="darkorange")
ax1.bar(x + width, fused_scores, width, label="Fused Ensemble Score", color="crimson")
ax1.axhline(35, color="green", linestyle=":", label="Low Risk Cutoff (35%)")
ax1.axhline(65, color="red", linestyle=":", label="High Risk Cutoff (65%)")
ax1.set_ylabel("Cardiac Risk Score (%)", fontsize=10)
ax1.set_title("CardioSense Dual-Engine Probability Comparison", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylim(0, 100)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left")

# Subplot 2: Regional ischemic territory ST levels
territories = ["Anterior (LAD)", "Inferior (RCA)", "Lateral (LCx)"]
pt1_st = [evaluations[0][1]["territory_breakdown"]["Anterior / Septal (LAD)"]["max_st_mv"],
          evaluations[0][1]["territory_breakdown"]["Inferior (RCA)"]["max_st_mv"],
          evaluations[0][1]["territory_breakdown"]["Lateral (LCx)"]["max_st_mv"]]
pt8_st = [evaluations[1][1]["territory_breakdown"]["Anterior / Septal (LAD)"]["max_st_mv"],
          evaluations[1][1]["territory_breakdown"]["Inferior (RCA)"]["max_st_mv"],
          evaluations[1][1]["territory_breakdown"]["Lateral (LCx)"]["max_st_mv"]]

x2 = np.arange(len(territories))
ax2.bar(x2 - width / 2, pt1_st, width, label="Patient 1 (NORM)", color="cadetblue")
ax2.bar(x2 + width / 2, pt8_st, width, label="Patient 8 (MI)", color="firebrick")
ax2.axhline(0.10, color="black", linestyle="--", label="Clinical Ischemic Alarm (0.10 mV)")
ax2.set_ylabel("Max Regional ST Shift (mV)", fontsize=10)
ax2.set_title("Coronary Territory Ischemic Localization", fontsize=11, fontweight="bold")
ax2.set_xticks(x2)
ax2.set_xticklabels(territories, fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right")

plt.tight_layout()

if args.save_figure:
    fig9_path = figures_dir / "09_risk_explainability.png"
    plt.savefig(fig9_path, dpi=150)
    print(f"\n[Saved figure to: {fig9_path.relative_to(project_root)}]")

plt.close()