
"""
CardioSense — PyTorch ECG Dataset

Loads ECG recordings using the centralized input-validation pipeline.

Pipeline:
    WFDB recording
        -> Validate ECG structure, units, and lead order
        -> Convert voltage to millivolts
        -> Apply ECG filtering
        -> Z-score normalization
        -> Convert to PyTorch tensor

Each item returns:
    waveform: torch.float32 tensor, shape (12, 1000)
    label:    torch.long tensor, value 0 or 1

The original 100 Hz CNN preprocessing policy is preserved.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.ecg_loader import load_standardized_ecg
from src.dsp.filters import filter_ecg


class ECGDataset(Dataset):
    """Dataset of standardized 12-lead ECG recordings."""

    def __init__(self, manifest_csv, cohort_dir, folds=None):
        """
        Parameters
        ----------
        manifest_csv:
            Path to the cohort manifest.

        cohort_dir:
            Directory containing WFDB waveform files.

        folds:
            Optional PTB-XL fold numbers used to select records.
            Example: folds=[1, 2, 3] or folds=[10].
        """

        self.cohort_dir = Path(cohort_dir)

        # 1. Load the existing cohort manifest.
        df = pd.read_csv(manifest_csv)

        # 2. Select the requested official PTB-XL folds.
        if folds is not None:
            df = df[df["strat_fold"].isin(folds)].copy()

        # 3. Register records with both required WFDB files.
        # Missing waveforms are skipped, not removed from the
        # canonical 300-record cohort definition.
        self.records = []

        for _, row in df.iterrows():
            record_base = self.cohort_dir / row["filename_lr"]

            header_path = record_base.with_suffix(".hea")
            signal_path = record_base.with_suffix(".dat")

            if not (header_path.is_file() and signal_path.is_file()):
                continue

            self.records.append({
                "path": str(record_base),
                "label": int(row["target_label"]),
                "ecg_id": int(row["ecg_id"]),
            })

    def __len__(self):
        """Return the number of locally available recordings."""
        return len(self.records)

    def __getitem__(self, idx):
        """Load, preprocess, and return one ECG and its label."""

        item = self.records[idx]

        # -----------------------------------------------------
        # 1. Load and validate the ECG.
        # -----------------------------------------------------
        # The centralized loader guarantees:
        # - 12 recognized and uniquely identified ECG leads
        # - Standard lead order: I, II, III, AVR, AVL, AVF,
        #   V1, V2, V3, V4, V5, V6
        # - Voltage converted to millivolts
        # - Valid sampling frequency
        # - Finite, real-valued waveform samples
        #
        # Shape: (samples, 12)
        signals_mv, fs, _ = load_standardized_ecg(
            item["path"]
        )

        # -----------------------------------------------------
        # 2. Enforce the existing CNN input dimensions.
        # -----------------------------------------------------
        # Current CardioSense CNN training uses 10-second
        # PTB-XL recordings at 100 Hz (1000 samples per lead).
        #
        # Do not silently resample, pad, or truncate data.
        if fs != 100.0 or signals_mv.shape != (1000, 12):
            raise ValueError(
                f"ECG {item['ecg_id']}: incompatible CNN input. "
                f"Expected fs=100 Hz and shape=(1000, 12), "
                f"received fs={fs} and shape={signals_mv.shape}"
            )

        # -----------------------------------------------------
        # 3. Filter the standardized ECG.
        # -----------------------------------------------------
        # Preserve the project's existing 0.5–40 Hz filter.
        # Signals are already ordered and measured in mV.
        cleaned = filter_ecg(signals_mv, fs=fs)

        # -----------------------------------------------------
        # 4. Apply per-lead z-score normalization.
        # -----------------------------------------------------
        # Each lead is normalized independently:
        #
        #       z = (x - mean) / (std + epsilon)
        #
        # The epsilon prevents division by zero for flat
        # or nearly constant channels.
        mean = np.mean(
            cleaned,
            axis=0,
            keepdims=True,
        )

        std = np.std(
            cleaned,
            axis=0,
            keepdims=True,
        )

        normalized = (cleaned - mean) / (std + 1e-6)

        # -----------------------------------------------------
        # 5. Convert to the PyTorch CNN tensor format.
        # -----------------------------------------------------
        # Before: (1000 samples, 12 leads)
        # After:  (12 channels, 1000 samples)
        waveform = torch.tensor(
            normalized.T.copy(),
            dtype=torch.float32,
        )

        # Binary research label:
        # 0 = normal-reference family
        # 1 = infarction family
        label = torch.tensor(
            item["label"],
            dtype=torch.long,
        )

        return waveform, label
