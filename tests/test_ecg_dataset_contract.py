
"""Regression tests for the CNN's standardized ECG input."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dsp.input_validation import STANDARD_LEADS
from src.models.ecg_dataset import ECGDataset


def test_cnn_input_is_independent_of_lead_order_and_units(
    tmp_path, monkeypatch
):
    """Equivalent ECGs must produce equivalent CNN tensors."""

    fs = 100.0
    time = np.arange(1000) / fs

    # 1. Create a synthetic 12-lead ECG.
    # Every lead has a different frequency so a channel
    # ordering mistake cannot hide behind identical signals.
    original = np.column_stack([
        np.sin(2 * np.pi * (1.0 + i * 0.3) * time)
        for i in range(12)
    ])

    # 2. Shuffle both the waveform columns and their labels.
    order = [11, 5, 0, 3, 8, 10, 1, 6, 9, 4, 7, 2]

    shuffled = original[:, order].copy()
    shuffled_leads = [STANDARD_LEADS[i] for i in order]

    # Mix voltage units between channels.
    # The underlying physical signal remains identical.
    units = ["uV" if i % 2 else "mV" for i in range(12)]

    for i, unit in enumerate(units):
        if unit == "uV":
            shuffled[:, i] *= 1000

    # 3. Create a small temporary manifest.
    manifest = tmp_path / "manifest.csv"

    pd.DataFrame([
        {
            "ecg_id": 1,
            "filename_lr": "original",
            "target_label": 0,
            "strat_fold": 1,
        },
        {
            "ecg_id": 2,
            "filename_lr": "shuffled",
            "target_label": 0,
            "strat_fold": 1,
        },
    ]).to_csv(manifest, index=False)

    # ECGDataset checks for both WFDB files.
    # These are placeholders because the WFDB reader
    # will be replaced with our synthetic test reader.
    for name in ("original", "shuffled"):
        (tmp_path / f"{name}.hea").touch()
        (tmp_path / f"{name}.dat").touch()

    # 4. Simulate the data returned by wfdb.rdsamp().
    def fake_rdsamp(record_path):
        name = Path(record_path).name

        if name == "original":
            return original.copy(), {
                "fs": fs,
                "sig_name": list(STANDARD_LEADS),
                "units": ["mV"] * 12,
            }

        if name == "shuffled":
            return shuffled.copy(), {
                "fs": fs,
                "sig_name": shuffled_leads,
                "units": units,
            }

        raise ValueError(f"Unexpected test record: {name}")

    monkeypatch.setattr(
        "src.data.ecg_loader.wfdb.rdsamp",
        fake_rdsamp,
    )

    # 5. Execute the REAL ECGDataset preprocessing.
    dataset = ECGDataset(manifest, tmp_path)

    wave_a, label_a = dataset[0]
    wave_b, label_b = dataset[1]

    # 6. Both versions must produce the same CNN input.
    assert wave_a.shape == (12, 1000)
    assert wave_b.shape == (12, 1000)

    assert label_a.item() == label_b.item() == 0

    torch.testing.assert_close(
        wave_a,
        wave_b,
        rtol=1e-5,
        atol=1e-6,
    )
