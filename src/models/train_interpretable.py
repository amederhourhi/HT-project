"""
CardioSense - Interpretable Baseline Classifier Training
Trains and calibrates a Random Forest on extracted clinical features
using official PTB-XL stratified train/test folds (Folds 1-8: Train, 9-10: Test).
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


# ---------------------------------------------------------
# Block 1: Load Clinical Features & Official Folds
# ---------------------------------------------------------
def load_train_test_data(features_csv):
    """Splits data strictly using PTB-XL official benchmark folds."""
    df = pd.read_csv(features_csv)

    feature_cols = [
        "heart_rate_bpm", "mean_rr_sec", "sdnn_ms", "rmssd_ms", "qrs_duration_ms",
        "max_st_elevation_mv", "max_st_depression_mv",
        "st_I", "st_II", "st_III", "st_AVR", "st_AVL", "st_AVF",
        "st_V1", "st_V2", "st_V3", "st_V4", "st_V5", "st_V6",
        "overall_sqi",
    ]

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
# Block 3: Clinical Performance Evaluation
# ---------------------------------------------------------
def evaluate_clinical_metrics(model, X_test, y_test):
    """Calculates cardiology-focused metrics: Sensitivity, Specificity, AUC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    # Sensitivity (Recall): Did we detect actual heart attacks?
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    
    # Specificity: True Negative Rate (avoiding false hospital alarms)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 55)
    print("  CARDIOSENSE INTERPRETABLE MODEL - TEST RESULTS  ")
    print("=" * 55)
    print(f"Test Cohort Size       : {len(y_test)} patients (Folds 9-10)")
    print(f"Accuracy               : {acc * 100:.1f}%")
    print(f"Sensitivity (Catch MI) : {sensitivity * 100:.1f}%  <-- Most Critical")
    print(f"Specificity (No False) : {specificity * 100:.1f}%")
    print(f"ROC-AUC Score          : {auc:.3f}")
    print("\nConfusion Matrix:")
    print(f"  [TN: {tn:2d} | FP: {fp:2d}] (True Normal vs False Alarm)")
    print(f"  [FN: {fn:2d} | TP: {tp:2d}] (Missed MI vs Detected MI)")
    print("=" * 55)

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

    # 3. Evaluate
    probs, auc_val = evaluate_clinical_metrics(cal_model, X_te, y_te)

    # 4. Generate Figures 06 & 07
    plot_and_save_figures(rf_base, feat_names, y_te, probs, auc_val, figures_directory)

    # 5. Save model checkpoint
    model_save_path = checkpoints_dir / "interpretable_model.joblib"
    joblib.dump(cal_model, model_save_path)
    print(f"\nTrained model successfully saved to: {model_save_path.relative_to(project_root)}")