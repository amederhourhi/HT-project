
"""Tests for the WFDB loading and batch-extraction boundary."""

import numpy as np
import pandas as pd
import pytest

import src.data.ecg_loader as loader
import src.features.batch_extract as batch
from src.dsp.input_validation import STANDARD_LEADS


def test_loader_standardizes_wfdb_record(monkeypatch):
    # Create a known 12-lead waveform in standard order.
    expected = np.tile(
        np.arange(1, 13, dtype=float), (1000, 1)
    )

    # Simulate WFDB delivering channels in reverse order,
    # with amplitudes expressed in microvolts.
    incoming = expected[:, ::-1] * 1000

    fields = {
        "fs": 100,
        "sig_name": list(STANDARD_LEADS)[::-1],
        "units": ["uV"] * 12,
    }

    monkeypatch.setattr(
        loader.wfdb,
        "rdsamp",
        lambda path: (incoming, fields),
    )

    signals, fs, leads = loader.load_standardized_ecg(
        "example_record"
    )

    np.testing.assert_allclose(signals, expected)
    assert fs == 100.0
    assert leads == list(STANDARD_LEADS)


def test_loader_rejects_missing_unit_metadata(monkeypatch):
    fields = {
        "fs": 100,
        "sig_name": list(STANDARD_LEADS),
    }

    monkeypatch.setattr(
        loader.wfdb,
        "rdsamp",
        lambda path: (np.zeros((1000, 12)), fields),
    )

    with pytest.raises(ValueError, match="units"):
        loader.load_standardized_ecg("example_record")


def test_batch_extraction_uses_standardized_loader(
    monkeypatch, tmp_path
):
    # Build a tiny temporary cohort without downloading ECG data.
    cohort = tmp_path / "data" / "cohort"
    record = cohort / "records100" / "00000" / "00001_lr"

    record.parent.mkdir(parents=True)

    # The batch extractor checks whether both files exist.
    record.with_suffix(".hea").touch()
    record.with_suffix(".dat").touch()

    pd.DataFrame([{
        "ecg_id": 1,
        "filename_lr": "records100/00000/00001_lr",
        "target_label": 0,
        "strat_fold": 10,
    }]).to_csv(cohort / "cohort_manifest.csv", index=False)

    waveform = np.zeros((1000, 12))
    calls = []

    def fake_load(path):
        assert path == record
        calls.append("load")
        return waveform, 100.0, list(STANDARD_LEADS)

    def fake_extract(signals, *, fs, lead_names):
        assert signals is waveform
        assert fs == 100.0
        assert lead_names == list(STANDARD_LEADS)
        calls.append("extract")
        return {
            "status": "success",
            "features": {"heart_rate_bpm": 72.0},
        }

    monkeypatch.setattr(
        batch, "load_standardized_ecg", fake_load
    )
    monkeypatch.setattr(
        batch, "extract_clinical_features", fake_extract
    )

    output = tmp_path / "data" / "processed" / "features.csv"

    batch.build_clinical_features_dataset(cohort, output)

    assert calls == ["load", "extract"]

    saved = pd.read_csv(output)
    assert len(saved) == 1
    assert saved.loc[0, "heart_rate_bpm"] == 72.0
