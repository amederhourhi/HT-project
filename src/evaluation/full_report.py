"""
CardioSense - Full Honest Evaluation Report (v2)
------------------------------------------------
Produces a clear, side-by-side performance summary for the current prototype.

What this script reports:
1. Exact dataset size after SQI filtering
2. Interpretable Random Forest performance (two operating points)
3. 1D-CNN performance on the same official test folds
4. Simple side-by-side comparison of the two engines
5. Deterministic guardrail escalation rates on the test folds

Design goals:
- Readable and easy to maintain
- Uses the same official folds everywhere
- No clever tricks, just clean sequential code
- Heavy comments so every block is understandable
"""

import sys
from pathlib import Path

# Add project root so "src" imports work
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.features.extract_features import FEATURE_COLUMNS
from src.models.ecg_cnn import CardioNet1D
from src.models.ecg_dataset import ECGDataset
from src.evaluation.score_guardrails import score_row, print_summary
from src.triage.triage_rules import ClinicalTriageEngine


# ---------------------------------------------------------
# Block 1: Paths
# ---------------------------------------------------------
FEATURES_CSV = project_root / "data" / "processed" / "clinical_features.csv"
MANIFEST_CSV = project_root / "data" / "cohort" / "cohort_manifest.csv"
COHORT_DIR = project_root / "data" / "cohort"
CHECKPOINT_DIR = project_root / "models" / "checkpoints"
RF_CHECKPOINT = CHECKPOINT_DIR / "interpretable_model.joblib"
CNN_CHECKPOINT = CHECKPOINT_DIR / "ecg_cnn.pt"


# ---------------------------------------------------------
# Block 2: Load feature data (for Random Forest)
# ---------------------------------------------------------
def load_feature_data():
    """
    What it does:
        Loads the processed feature CSV and splits it into
        official train / test folds.

    How it works:
        Folds 1-8 = train, folds 9-10 = test.
        Only keeps the columns the Random Forest was trained on.
    """
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    train_df = df[df["strat_fold"] <= 8].copy()
    test_df = df[df["strat_fold"] > 8].copy()

    X_test = test_df[feature_cols]
    y_test = test_df["target_label"]

    return df, train_df, test_df, X_test, y_test


# ---------------------------------------------------------
# Block 3: Print dataset summary
# ---------------------------------------------------------
def print_dataset_summary(df, train_df, test_df):
    """
    What it does:
        Shows exactly how many records we have and the class balance.
    """
    print("\n" + "=" * 70)
    print("  DATASET SUMMARY (after SQI filtering)")
    print("=" * 70)
    print(f"Total records available          : {len(df)}")
    print(f"  Normal (label 0)               : {(df['target_label'] == 0).sum()}")
    print(f"  Infarction (label 1)           : {(df['target_label'] == 1).sum()}")
    print("-" * 70)
    print(f"Train set (folds 1-8)            : {len(train_df)}")
    print(f"Test set  (folds 9-10)           : {len(test_df)}")
    print(f"  Test Normal                    : {(test_df['target_label'] == 0).sum()}")
    print(f"  Test Infarction                : {(test_df['target_label'] == 1).sum()}")
    print("=" * 70)


# ---------------------------------------------------------
# Block 4: Evaluate Random Forest
# ---------------------------------------------------------
def evaluate_random_forest(X_test, y_test):
    """
    What it does:
        Loads the calibrated Random Forest and reports
        performance at two clinical thresholds.

    Returns:
        Dictionary with AUC and the two operating-point metrics
        so we can reuse them in the side-by-side table.
    """
    if not RF_CHECKPOINT.exists():
        print("\n[WARNING] RF checkpoint not found. Skipping RF evaluation.")
        return None

    model = joblib.load(RF_CHECKPOINT)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    def metrics_at(thresh):
        y_pred = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred, pos_label=1)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "acc": acc,
            "sens": sens,
            "spec": spec,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }

    results = {
        "auc": auc,
        "std": metrics_at(0.50),
        "triage": metrics_at(0.35),
    }

    print("\n" + "=" * 70)
    print("  INTERPRETABLE RANDOM FOREST (Calibrated)")
    print("=" * 70)
    print(f"ROC-AUC                          : {auc:.3f}")
    print("-" * 70)
    print("Standard Diagnostic (threshold = 0.50)")
    m = results["std"]
    print(f"  Accuracy                       : {m['acc']*100:.1f}%")
    print(f"  Sensitivity                    : {m['sens']*100:.1f}%")
    print(f"  Specificity                    : {m['spec']*100:.1f}%")
    print(f"  Confusion Matrix               : TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
    print("-" * 70)
    print("Emergency Triage (threshold = 0.35)")
    m = results["triage"]
    print(f"  Accuracy                       : {m['acc']*100:.1f}%")
    print(f"  Sensitivity                    : {m['sens']*100:.1f}%")
    print(f"  Specificity                    : {m['spec']*100:.1f}%")
    print(f"  Confusion Matrix               : TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
    print("=" * 70)

    return results


# ---------------------------------------------------------
# Block 5: Evaluate 1D-CNN on the official test folds
# ---------------------------------------------------------
def evaluate_cnn():
    """
    What it does:
        Loads the trained 1D-CNN and evaluates it on the
        same official test folds (9 and 10) using the waveform files.

    How it works:
        Uses the existing ECGDataset class so the preprocessing
        (filter + z-score) is identical to training.
    """
    if not CNN_CHECKPOINT.exists():
        print("\n[WARNING] CNN checkpoint not found. Skipping CNN evaluation.")
        return None

    device = torch.device("cpu")
    model = CardioNet1D(in_channels=12, num_classes=2).to(device)

    # Load weights (handles both plain state_dict and checkpoint dict)
    ckpt = torch.load(CNN_CHECKPOINT, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Official test folds only
    test_dataset = ECGDataset(MANIFEST_CSV, COHORT_DIR, folds=[9, 10])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for waves, labels in test_loader:
            waves = waves.to(device)
            logits = model(waves)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    auc = roc_auc_score(y_true, y_prob)

    def metrics_at(thresh):
        y_pred = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred, pos_label=1)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "acc": acc,
            "sens": sens,
            "spec": spec,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }

    results = {
        "auc": auc,
        "std": metrics_at(0.50),
        "triage": metrics_at(0.35),
        "n": len(y_true),
    }

    print("\n" + "=" * 70)
    print("  1D-CNN (Deep Waveform Model)")
    print("=" * 70)
    print(f"Test records evaluated           : {results['n']}")
    print(f"ROC-AUC                          : {auc:.3f}")
    print("-" * 70)
    print("Standard Diagnostic (threshold = 0.50)")
    m = results["std"]
    print(f"  Accuracy                       : {m['acc']*100:.1f}%")
    print(f"  Sensitivity                    : {m['sens']*100:.1f}%")
    print(f"  Specificity                    : {m['spec']*100:.1f}%")
    print(f"  Confusion Matrix               : TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
    print("-" * 70)
    print("Emergency Triage (threshold = 0.35)")
    m = results["triage"]
    print(f"  Accuracy                       : {m['acc']*100:.1f}%")
    print(f"  Sensitivity                    : {m['sens']*100:.1f}%")
    print(f"  Specificity                    : {m['spec']*100:.1f}%")
    print(f"  Confusion Matrix               : TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
    print("=" * 70)

    return results


# ---------------------------------------------------------
# Block 6: Side-by-side comparison table
# ---------------------------------------------------------
def print_side_by_side(rf_results, cnn_results):
    """
    What it does:
        Prints a simple comparison of the two engines
        so the differences are easy to see at a glance.
    """
    print("\n" + "=" * 70)
    print("  SIDE-BY-SIDE COMPARISON (Test folds 9-10)")
    print("=" * 70)

    if rf_results is None and cnn_results is None:
        print("No model results available.")
        return

    print(f"{'Metric':<28} {'Random Forest':>15} {'1D-CNN':>12}")
    print("-" * 70)

    def fmt(val):
        return f"{val*100:.1f}%" if val is not None else "  -  "

    rf_auc = f"{rf_results['auc']:.3f}" if rf_results else "  -  "
    cnn_auc = f"{cnn_results['auc']:.3f}" if cnn_results else "  -  "
    print(f"{'ROC-AUC':<28} {rf_auc:>15} {cnn_auc:>12}")

    print("-" * 70)
    print("Standard threshold (0.50)")
    if rf_results:
        print(f"{'  Sensitivity':<28} {fmt(rf_results['std']['sens']):>15} {fmt(cnn_results['std']['sens']) if cnn_results else '  -  ':>12}")
        print(f"{'  Specificity':<28} {fmt(rf_results['std']['spec']):>15} {fmt(cnn_results['std']['spec']) if cnn_results else '  -  ':>12}")
    print("-" * 70)
    print("Triage threshold (0.35)")
    if rf_results:
        print(f"{'  Sensitivity':<28} {fmt(rf_results['triage']['sens']):>15} {fmt(cnn_results['triage']['sens']) if cnn_results else '  -  ':>12}")
        print(f"{'  Specificity':<28} {fmt(rf_results['triage']['spec']):>15} {fmt(cnn_results['triage']['spec']) if cnn_results else '  -  ':>12}")
    print("=" * 70)


# ---------------------------------------------------------
# Block 7: Guardrail evaluation
# ---------------------------------------------------------
def evaluate_guardrails(test_df):
    """
    What it does:
        Runs the deterministic rules on the test fold and
        reports false-escalation rate on healthy records.
    """
    print("\n" + "=" * 70)
    print("  DETERMINISTIC GUARDRAIL LAYER (Test Folds 9-10)")
    print("=" * 70)

    engine = ClinicalTriageEngine()
    outcomes = [score_row(engine, row) for _, row in test_df.iterrows()]
    print_summary("test", outcomes)


# ---------------------------------------------------------
# Block 8: Main
# ---------------------------------------------------------
def main():
    print("\nCardioSense – Full Honest Evaluation Report (v2)")
    print("Prototype – official test folds only (N≈52)")

    df, train_df, test_df, X_test, y_test = load_feature_data()

    print_dataset_summary(df, train_df, test_df)

    rf_results = evaluate_random_forest(X_test, y_test)
    cnn_results = evaluate_cnn()
    print_side_by_side(rf_results, cnn_results)
    evaluate_guardrails(test_df)

    print("\nReport finished.")
    print("Copy the key numbers into the Engineering Log under the existing")
    print("'Honest Evaluation Snapshot' section (or create an updated version).\n")


if __name__ == "__main__":
    main()