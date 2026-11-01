"""Train CardioNet1D with separate train, validation, and test folds."""

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ecg_cnn import CardioNet1D
from src.models.ecg_dataset import ECGDataset

SEED = 42
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLDS = [9]
TEST_FOLDS = [10]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    labels, probabilities = [], []
    loss_sum = 0.0

    with torch.inference_mode():
        for waves, targets in loader:
            waves = waves.to(device)
            targets = targets.to(device)
            logits = model(waves)
            loss_sum += criterion(logits, targets).item() * len(targets)
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels.extend(targets.cpu().tolist())

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.size == 0:
        raise ValueError("Evaluation split contains no records.")

    predicted = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

    return {
        "loss": loss_sum / len(y),
        "accuracy": accuracy_score(y, predicted),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "brier": brier_score_loss(y, p),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_cardionet(epochs=20, batch_size=16, lr=1e-3, patience=5):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = ROOT / "data" / "cohort" / "cohort_manifest.csv"
    cohort_dir = ROOT / "data" / "cohort"
    checkpoint_dir = ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_data = ECGDataset(manifest, cohort_dir, folds=TRAIN_FOLDS)
    val_data = ECGDataset(manifest, cohort_dir, folds=VAL_FOLDS)
    test_data = ECGDataset(manifest, cohort_dir, folds=TEST_FOLDS)

    for name, dataset in (("train", train_data), ("validation", val_data), ("test", test_data)):
        if not len(dataset):
            raise ValueError(f"No usable ECG records in {name} split. Check cohort files.")

    # Guard against duplicated ECG identifiers across splits.
    ids = [{str(record["ecg_id"]) for record in ds.records}
           for ds in (train_data, val_data, test_data)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("ECG identifier overlap detected across folds.")

    print(f"Device: {device}")
    print(f"Records: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = CardioNet1D(in_channels=12, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    stagnant_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for waves, targets in train_loader:
            waves, targets = waves.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(waves), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)

        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)
        train_loss = running_loss / len(train_data)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, val_auc={val_metrics['roc_auc']:.4f}")

        if val_metrics["loss"] < best_loss - 1e-5:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_weights)
    model.eval()

    # Test fold is evaluated only after training and checkpoint selection.
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nBest epoch: {best_epoch} (validation loss={best_loss:.4f})")
    print("Independent fold-10 test results:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value}")

    # Keep the current production checkpoint untouched until results are reviewed.
    save_path = checkpoint_dir / "ecg_cnn_fold10_candidate.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint: {save_path}")
    return test_metrics


if __name__ == "__main__":
    train_cardionet()
