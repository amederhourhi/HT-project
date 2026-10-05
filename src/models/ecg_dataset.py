"""
CardioSense - PyTorch ECG Waveform Dataset
Loads 12-lead filtered signals, applies z-score normalization,
and returns (12, 1000) tensors for deep learning.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb

from src.dsp.filters import filter_ecg


class ECGDataset(Dataset):
    """
    PyTorch Dataset for 12-lead ECG waveforms.
    Each sample returns:
        - waveform tensor: shape (12 leads, 1000 samples)
        - target label: 0 (Normal) or 1 (Myocardial Infarction)
    """

    def __init__(self, manifest_csv, cohort_dir, folds=None):
        self.cohort_dir = Path(cohort_dir)
        df = pd.read_csv(manifest_csv)

        # Filter by official stratified folds if specified
        if folds is not None:
            df = df[df["strat_fold"].isin(folds)].reset_index(drop=True)

        self.records = []
        for _, row in df.iterrows():
            rel_path = row["filename_lr"]
            record_base = self.cohort_dir / rel_path
            
            # Ensure BOTH header (.hea) and binary signal (.dat) exist!
            if record_base.with_suffix(".hea").exists() and record_base.with_suffix(".dat").exists():
                self.records.append({
                    "path": str(record_base),
                    "label": int(row["target_label"]),
                    "ecg_id": row["ecg_id"],
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        
        # 1. Load raw 12-lead signal: shape (1000, 12)
        signals, fields = wfdb.rdsamp(item["path"])
        fs = fields["fs"]

        # 2. Filter baseline wander & high frequency noise (0.5 - 40 Hz)
        cleaned = filter_ecg(signals, fs=fs)

        # 3. Z-score normalize per lead: zero mean, unit variance
        mean = np.mean(cleaned, axis=0, keepdims=True)
        std = np.std(cleaned, axis=0, keepdims=True) + 1e-6
        normalized = (cleaned - mean) / std

        # 4. Transpose to PyTorch 1D-CNN convention: (channels=12, time=1000)
        tensor_wave = torch.tensor(normalized.T, dtype=torch.float32)
        tensor_label = torch.tensor(item["label"], dtype=torch.long)

        return tensor_wave, tensor_label