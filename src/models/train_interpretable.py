"""
CardioSense - Interpretable Baseline Classifier Training
Trains and calibrates a Random Forest on extracted clinical features
using official PTB-XL stratified train/test folds (Folds 1-8: Train, 9-10: Test).
Reports both standard diagnostic and emergency triage operating points.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.features.extract_features import FEATURE_COLUMNS


# ---------------------------------------------------------
# Block 1: Load Clinical Features & Official Folds
# ---------------------------------------------------------
def load_train_test_data(features_csv):
    """Splits data strictly using PTB-XL official benchmark folds."""
    df = pd.read_csv(features_csv)

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Official PTB-XL split: Folds 1-8 for training, Folds 9-10 for testing
    train_df = df[df["strat_fold"] <= 8]
    test_df = df[df["strat_fold"] > 8]

    X_train = train_df[feature_cols]
    y_train = train_df["target_label"]

    X_test = test_df[feature_cols]
    y_test = test_df["target_label"]

    return X_train, y_train, X_test, y_test, feature_cols


# ---------------------------------------------------------
# Block 2: Train Model & Calibrate Probabilities
# ---------------------------------------------------------
def train_and_calibrate(X_train, y_train):
    """
    Trains a constrained Random Forest (max_depth=5 for full interpretability)
    and applies Platt scaling (sigmoid calibration) so predicted probabilities
    represent true clinical probabilities.
    """
    base_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    base_rf.fit(X_train, y_train)

    # Probability calibration (Platt Scaling)
    calibrated_model = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=5)
    calibrated_model.fit(X_train, y_train)

    return base_rf, calibrated_model


# ---------------------------------------------------------
# Block 3: Clinical Performance Evaluation (Dual Operating Points)
# ---------------------------------------------------------
def evaluate_clinical_metrics(model, X_test, y_test, triage_threshold=0.35):
    """
    Calculates cardiology metrics at two operating thresholds:
    1. Standard Diagnostic (0.50)
    2. Emergency Triage (0.35, high-sensitivity rule-out)
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Standard Diagnostic Threshold (0.50)
    y_pred_std = (y_prob >= 0.50).astype(int)
    acc_std = accuracy_score(y_test, y_pred_std)
    sens_std = recall_score(y_test, y_pred_std, pos_label=1)
    cm_std = confusion_matrix(y_test, y_pred_std)
    tn_std, fp_std, fn_std, tp_std = cm_std.ravel()
    spec_std = tn_std / (tn_std + fp_std) if (tn_std + fp_std) > 0 else 0.0

    # 2. Emergency Triage Screening Threshold (0.35)
    y_pred_tri = (y_prob >= triage_threshold).astype(int)
    acc_tri = accuracy_score(y_test, y_pred_tri)
    sens_tri = recall_score(y_test, y_pred_tri, pos_label=1)
    cm_tri = confusion_matrix(y_test, y_pred_tri)
    tn_tri, fp_tri, fn_tri, tp_tri = cm_tri.ravel()
    spec_tri = tn_tri / (tn_tri + fp_tri) if (tn_tri + fp_tri) > 0 else 0.0

    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 65)
    print("      CARDIOSENSE INTERPRETABLE MODEL - TEST BENCHMARK      ")
    print("=" * 65)
    print(f"Test Cohort Size             : {len(y_test)} patients (Official Folds 9-10)")
    print(f"ROC-AUC Discrimination       : {auc:.3f}")
    print("-" * 65)
    print("STANDARD DIAGNOSTIC OPERATING POINT (Threshold = 0.50):")
    print(f"  Accuracy                   : {acc_std * 100:.1f}%")
    print(f"  Sensitivity (Catch MI)     : {sens_std * 100:.1f}%")
    print(f"  Specificity (Rule-out Norm): {spec_std * 100:.1f}%")
    print(f"  Confusion Matrix           : [TN: {tn_std:2d} | FP: {fp_std:2d}] (True Normal vs False Alarm)")
    print(f"                               [FN: {fn_std:2d} | TP: {tp_std:2d}] (Missed MI vs Detected MI)")
    print("-" * 65)
    print(f"ACUTE EMERGENCY TRIAGE OPERATING POINT (Threshold = {triage_threshold:.2f}):")
    print(f"  Accuracy                   : {acc_tri * 100:.1f}%")
    print(f"  Sensitivity (Catch MI)     : {sens_tri * 100:.1f}%  <-- Prioritized for Triage")
    print(f"  Specificity (Rule-out Norm): {spec_tri * 100:.1f}%")
    print(f"  Confusion Matrix           : [TN: {tn_tri:2d} | FP: {fp_tri:2d}]")
    print(f"                               [FN: {fn_tri:2d} | TP: {tp_tri:2d}]")
    print("=" * 65)

    return y_prob, auc


# ---------------------------------------------------------
# Block 4: Generate Verification Figures (06 & 07)
# ---------------------------------------------------------
def plot_and_save_figures(base_rf, feature_names, y_test, y_prob, auc_score, figures_dir):
    """Plots and saves Feature Importance and ROC Curve to docs/figures/."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 06: Feature Importance Bar Chart ---
    importances = base_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:10]  # Top 10 features

    plt.figure(figsize=(10, 5))
    plt.barh(
        [feature_names[i] for i in sorted_idx][::-1],
        importances[sorted_idx][::-1],
        color="steelblue",
        edgecolor="black",
    )
    plt.title("CardioSense - Top 10 Clinical Features Driving MI Detection", fontsize=12, fontweight="bold")
    plt.xlabel("Gini Feature Importance", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig6_path = figures_dir / "06_feature_importance.png"
    plt.savefig(fig6_path, dpi=150)
    print(f"[Saved Feature Importance figure to: {fig6_path.relative_to(project_root)}]")
    plt.close()

    # --- Figure 07: Calibrated ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="crimson", linewidth=2.0, label=f"CardioSense (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier (AUC = 0.500)")
    plt.title("CardioSense - Calibrated ROC Curve (Test Set)", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig7_path = figures_dir / "07_roc_curve.png"
    plt.savefig(fig7_path, dpi=150)
    print(f"[Saved ROC Curve figure to: {fig7_path.relative_to(project_root)}]")
    plt.close()


if __name__ == "__main__":
    dataset_file = project_root / "data" / "processed" / "clinical_features.csv"
    checkpoints_dir = project_root / "models" / "checkpoints"
    figures_directory = project_root / "docs" / "figures"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_tr, y_tr, X_te, y_te, feat_names = load_train_test_data(dataset_file)
    print(f"Loaded: {len(X_tr)} Training patients | {len(X_te)} Test patients")

    # 2. Train and calibrate
    rf_base, cal_model = train_and_calibrate(X_tr, y_tr)

    # 3. Evaluate dual operating points
    probs, auc_val = evaluate_clinical_metrics(cal_model, X_te, y_te, triage_threshold=0.35)

    # 4. Generate Figures 06 & 07
    plot_and_save_figures(rf_base, feat_names, y_te, probs, auc_val, figures_directory)

    # 5. Save model checkpoint
    model_save_path = checkpoints_dir / "interpretable_model.joblib"
    joblib.dump(cal_model, model_save_path)
    print(f"\nTrained model successfully saved to: {model_save_path.relative_to(project_root)}")