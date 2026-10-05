"""
CardioSense - 1D-CNN Deep Waveform Training
Trains CardioNet1D on raw 12-lead ECG time-series using official PTB-XL folds
and plots training/validation convergence curves.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score

from src.models.ecg_dataset import ECGDataset
from src.models.ecg_cnn import CardioNet1D

# Set reproducible random seed
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# Block 1: Evaluation Helper Function
# ---------------------------------------------------------
def evaluate_model(model, dataloader, criterion, device):
    """Evaluates test loss, accuracy, sensitivity, specificity, and ROC-AUC."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for waves, labels in dataloader:
            waves, labels = waves.to(device), labels.to(device)
            logits = model(waves)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    sens = recall_score(all_labels, all_preds, pos_label=1)
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(all_labels, all_probs)

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "sensitivity": sens,
        "specificity": spec,
        "auc": auc,
        "cm": (tn, fp, fn, tp),
    }
    return metrics


# ---------------------------------------------------------
# Block 2: Training Loop
# ---------------------------------------------------------
def train_cardionet(epochs=20, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining CardioNet1D on device: {device}")

    # Dataset paths
    manifest_csv = project_root / "data" / "cohort" / "cohort_manifest.csv"
    cohort_dir = project_root / "data" / "cohort"
    checkpoints_dir = project_root / "models" / "checkpoints"
    figures_dir = project_root / "docs" / "figures"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Official PTB-XL Folds (Train: 1-8, Test: 9-10)
    train_dataset = ECGDataset(manifest_csv, cohort_dir, folds=list(range(1, 9)))
    test_dataset = ECGDataset(manifest_csv, cohort_dir, folds=[9, 10])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Cohort size: {len(train_dataset)} Train ECGs | {len(test_dataset)} Test ECGs")

    model = CardioNet1D(in_channels=12, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "test_loss": [], "test_acc": []}

    print("\nStarting Training Epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for waves, labels in train_loader:
            waves, labels = waves.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(waves)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch [{epoch:2d}/{epochs}] - Train Loss: {train_loss:.4f} | Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['acc']*100:.1f}%")

    # Final Clinical Evaluation
    final = evaluate_model(model, test_loader, criterion, device)
    tn, fp, fn, tp = final["cm"]

    print("\n" + "=" * 55)
    print("  CARDIOSENSE 1D-CNN (DEEP WAVEFORM) - TEST RESULTS  ")
    print("=" * 55)
    print(f"Accuracy               : {final['acc'] * 100:.1f}%")
    print(f"Sensitivity (Catch MI) : {final['sensitivity'] * 100:.1f}%")
    print(f"Specificity (No False) : {final['specificity'] * 100:.1f}%")
    print(f"ROC-AUC Score          : {final['auc']:.3f}")
    print("\nConfusion Matrix:")
    print(f"  [TN: {tn:2d} | FP: {fp:2d}] (True Normal vs False Alarm)")
    print(f"  [FN: {fn:2d} | TP: {tp:2d}] (Missed MI vs Detected MI)")
    print("=" * 55)

    # Plot Figure 08: Training Convergence Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="navy", linewidth=1.8)
    ax1.plot(range(1, epochs + 1), history["test_loss"], label="Test Loss", color="crimson", linewidth=1.8)
    ax1.set_title("Cross-Entropy Loss vs Epochs", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(range(1, epochs + 1), [a * 100 for a in history["test_acc"]], label="Test Accuracy", color="forestgreen", linewidth=1.8)
    ax2.set_title("Test Accuracy (%) on Unseen Folds", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    fig8_path = figures_dir / "08_cnn_training_curve.png"
    plt.savefig(fig8_path, dpi=150)
    print(f"\n[Saved CNN Training figure to: {fig8_path.relative_to(project_root)}]")
    plt.close()

    # Save model checkpoint
    save_path = checkpoints_dir / "ecg_cnn.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Deep learning checkpoint saved to: {save_path.relative_to(project_root)}")


if __name__ == "__main__":
    train_cardionet(epochs=20, batch_size=16, lr=1e-3)