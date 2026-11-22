
"""Regression tests for the CardioSense REST API input contract.

Uses temporary WFDB placeholders and synthetic ECG samples.

The real centralized ECG validator is executed.
No actual waveform downloads or model predictions are needed.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from src.api import main as api
from src.dsp.input_validation import STANDARD_LEADS


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    """Configure an isolated API environment with one synthetic ECG."""

    # ---------------------------------------------------------
    # 1. Create temporary sample and cohort directories.
    # ---------------------------------------------------------
    samples_dir = tmp_path / "samples"
    cohort_dir = tmp_path / "cohort"

    record_dir = samples_dir / "records100" / "00000"
    record_dir.mkdir(parents=True)
    cohort_dir.mkdir()

    record_path = record_dir / "00001_lr"

    # The resolver checks whether both WFDB files exist.
    # Their contents are mocked by fake_rdsamp below.
    record_path.with_suffix(".hea").touch()
    record_path.with_suffix(".dat").touch()

    # Redirect the API to our temporary directories.
    monkeypatch.setattr(api, "samples_dir", samples_dir)
    monkeypatch.setattr(api, "cohort_dir", cohort_dir)
    monkeypatch.setattr(
        api,
        "manifest_path",
        cohort_dir / "cohort_manifest.csv",
    )

    # ---------------------------------------------------------
    # 2. Build a synthetic 12-lead recording in millivolts.
    # ---------------------------------------------------------
    fs = 100.0
    time = np.arange(1000) / fs

    original = np.column_stack([
        (0.1 + i * 0.01)
        * np.sin(2 * np.pi * (1.0 + i * 0.2) * time)
        + 0.003 * i
        for i in range(12)
    ])

    # ---------------------------------------------------------
    # 3. Shuffle lead order and mix voltage units.
    # ---------------------------------------------------------
    order = [11, 5, 0, 3, 8, 10, 1, 6, 9, 4, 7, 2]

    shuffled = original[:, order].copy()
    shuffled_names = [STANDARD_LEADS[i] for i in order]

    units = [
        "uV" if i % 2 else "mV"
        for i in range(12)
    ]

    for i, unit in enumerate(units):
        if unit == "uV":
            shuffled[:, i] *= 1000.0

    # ---------------------------------------------------------
    # 4. Replace only the WFDB disk reader.
    # ---------------------------------------------------------
    # The real load_standardized_ecg() and
    # validate_ecg_input() functions still execute.
    def fake_rdsamp(path):
        assert Path(path) == record_path

        return shuffled.copy(), {
            "fs": fs,
            "sig_name": shuffled_names,
            "units": units,
        }

    monkeypatch.setattr(
        "src.data.ecg_loader.wfdb.rdsamp",
        fake_rdsamp,
    )

    # ---------------------------------------------------------
    # 5. Return a real FastAPI test client.
    # ---------------------------------------------------------
    return {
        "client": TestClient(api.app),
        "original": original,
        "cohort_dir": cohort_dir,
    }


def test_record_catalogue(api_env):
    """The API should list the available sample record."""

    client = api_env["client"]

    response = client.get("/api/records")

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 1
    assert body["records"][0]["id"] == "00001_lr"
    assert body["records"][0]["category"] == "sample"


def test_signals_are_standardized(api_env):
    """The API must return canonical lead order and mV values."""

    client = api_env["client"]
    original = api_env["original"]

    response = client.get(
        "/api/records/00001_lr/signals"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["sampling_rate"] == 100.0
    assert body["lead_names"] == list(STANDARD_LEADS)

    # All twelve channels must be present.
    assert set(body["raw_signals"]) == set(STANDARD_LEADS)
    assert set(body["filtered_signals"]) == set(STANDARD_LEADS)

    # Check several samples from every channel.
    # The API rounds values to three decimal places.
    for lead_index, lead in enumerate(STANDARD_LEADS):
        actual = body["raw_signals"][lead]

        assert len(actual) == 1000

        for sample_index in (0, 100, 275, 500):
            expected = round(
                float(original[sample_index, lead_index]),
                3,
            )

            assert actual[sample_index] == pytest.approx(
                expected,
                abs=1e-6,
            )


def test_evaluation_receives_standardized_millivolts(
    api_env, monkeypatch
):
    """The ensemble must receive ordered mV data exactly once."""

    client = api_env["client"]
    original = api_env["original"]

    received = {}

    def fake_predict(
        signals,
        *,
        fs,
        lead_names,
        age,
        sex,
        units,
    ):
        # Capture the data passed from API to ensemble.
        received["signals"] = signals.copy()
        received["fs"] = fs
        received["leads"] = list(lead_names)
        received["units"] = units

        # End early so no real model prediction is run.
        return {
            "status": "rejected",
            "reason": "Synthetic test rejection",
            "sqi_details": {},
        }

    monkeypatch.setattr(
        api,
        "ensemble_pipeline",
        SimpleNamespace(predict=fake_predict),
    )

    response = client.post(
        "/api/triage/evaluate",
        json={
            "record_id": "00001_lr",
            "age": 55,
            "sex": "MALE",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    assert received["fs"] == 100.0
    assert received["units"] == "mV"
    assert received["leads"] == list(STANDARD_LEADS)

    np.testing.assert_allclose(
        received["signals"],
        original,
        rtol=0,
        atol=1e-12,
    )


def test_unknown_record_returns_404(api_env):
    """Unknown ECG IDs must never load another recording."""

    client = api_env["client"]

    response = client.get(
        "/api/records/99999/signals"
    )

    assert response.status_code == 404


def test_invalid_record_id_returns_400(api_env):
    """Reject wildcard identifiers instead of interpreting them."""

    client = api_env["client"]

    response = client.get(
        "/api/records/bad*name/signals"
    )

    assert response.status_code == 400


def test_missing_cohort_waveform_returns_404(api_env):
    """A pinned ECG with missing files must not be substituted."""

    client = api_env["client"]
    cohort_dir = api_env["cohort_dir"]

    pd.DataFrame([{
        "ecg_id": 77,
        "filename_lr": "records100/00000/00077_lr",
    }]).to_csv(
        cohort_dir / "cohort_manifest.csv",
        index=False,
    )

    response = client.get(
        "/api/records/77/signals"
    )

    assert response.status_code == 404
    assert "missing" in response.json()["detail"].lower()


def test_unsafe_manifest_path_is_rejected(api_env):
    """A manifest must not point outside the cohort directory."""

    client = api_env["client"]
    cohort_dir = api_env["cohort_dir"]

    pd.DataFrame([{
        "ecg_id": 88,
        "filename_lr": "../../outside_record",
    }]).to_csv(
        cohort_dir / "cohort_manifest.csv",
        index=False,
    )

    response = client.get(
        "/api/records/88/signals"
    )

    assert response.status_code == 500
    assert "unsafe path" in response.json()["detail"].lower()


def test_invalid_waveform_units_return_422(
    api_env, monkeypatch
):
    """Unsupported physical units must fail at the API boundary."""

    client = api_env["client"]

    def bad_rdsamp(path):
        return np.ones((1000, 12)), {
            "fs": 100.0,
            "sig_name": list(STANDARD_LEADS),
            "units": ["invalid"] * 12,
        }

    monkeypatch.setattr(
        "src.data.ecg_loader.wfdb.rdsamp",
        bad_rdsamp,
    )

    response = client.get(
        "/api/records/00001_lr/signals"
    )

    assert response.status_code == 422
    assert "Invalid ECG input" in response.json()["detail"]
