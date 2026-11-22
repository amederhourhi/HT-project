
"""Regression tests for the pinned PTB-XL cohort contract."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.download_ptbxl import (
    build_cohort_manifest,
    load_pinned_cohort,
)

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_cohort_matches_audit():
    """All 300 original records must remain pinned."""
    pin = load_pinned_cohort(
        ROOT / "data/processed/cohort_ecg_ids.csv"
    )
    audit = pd.read_csv(
        ROOT / "docs/sqi_audit/full_pipeline_v1_vs_v3.csv",
        usecols=["ecg_id", "strat_fold", "target_label"],
    )

    assert len(pin) == 300
    assert pin["ecg_id"].is_unique
    assert pin["target_label"].value_counts().to_dict() == {
        0: 150,
        1: 150,
    }

    pd.testing.assert_frame_equal(
        pin.reset_index(drop=True),
        audit.reset_index(drop=True),
    )


@pytest.mark.parametrize("bad_row", [
    "11,0,0",    # Fold below valid range
    "11,11,0",   # Fold above valid range
    "11,1,2",    # Invalid label
    "0,1,0",     # Invalid ECG ID
])
def test_invalid_pin_values_are_rejected(tmp_path, bad_row):
    """Malformed cohort definitions must fail immediately."""
    path = tmp_path / "bad_pin.csv"
    path.write_text(
        "ecg_id,strat_fold,target_label\n"
        + bad_row + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_pinned_cohort(path)


def test_duplicate_ecg_ids_are_rejected(tmp_path):
    """One ECG cannot appear twice."""
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "ecg_id,strat_fold,target_label\n"
        "11,1,0\n"
        "11,1,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_pinned_cohort(path)


@pytest.fixture
def synthetic_cohort(tmp_path):
    """Create tiny fake PTB-XL metadata without downloading anything."""
    pin = pd.DataFrame([
        {"ecg_id": 11, "strat_fold": 1, "target_label": 0},
        {"ecg_id": 22, "strat_fold": 9, "target_label": 1},
    ])

    metadata = pd.DataFrame([
        {
            "ecg_id": 11,
            "patient_id": 101,
            "strat_fold": 1,
            "scp_codes": "{'NORM': 100.0}",
            "age": 50,
            "sex": 0,
            "filename_lr": "records100/11_lr",
            "infarction_stadium1": "",
            "validated_by_human": True,
        },
        {
            "ecg_id": 22,
            "patient_id": 202,
            "strat_fold": 9,
            "scp_codes": "{'AMI': 100.0}",
            "age": 60,
            "sex": 1,
            "filename_lr": "records100/22_lr",
            "infarction_stadium1": "",
            "validated_by_human": True,
        },
    ])

    pin_path = tmp_path / "pin.csv"
    metadata_path = tmp_path / "metadata.csv"
    output_dir = tmp_path / "cohort"

    return pin, metadata, pin_path, metadata_path, output_dir


def build_from_synthetic(fixture, pin, metadata):
    """Write fake inputs and run the real manifest builder."""
    _, _, pin_path, metadata_path, output_dir = fixture

    pin.to_csv(pin_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    return build_cohort_manifest(
        pinned_ids_csv=pin_path,
        database_csv=metadata_path,
        cohort_dir=output_dir,
        download=False,
    )


def test_valid_manifest_is_generated(synthetic_cohort):
    """A valid pin must generate the expected manifest."""
    pin, metadata, *_ = synthetic_cohort

    result = build_from_synthetic(
        synthetic_cohort, pin, metadata
    )

    assert result["ecg_id"].tolist() == [11, 22]
    assert result["target_label"].tolist() == [0, 1]
    assert result["strat_fold"].tolist() == [1, 9]


@pytest.mark.parametrize("problem", [
    "wrong_label",
    "wrong_fold",
    "unknown_diagnosis",
    "patient_overlap",
    "missing_record",
])
def test_corrupted_metadata_is_rejected(
    synthetic_cohort, problem
):
    """Deliberately introduce one problem and expect rejection."""
    original_pin, original_metadata, *_ = synthetic_cohort

    pin = original_pin.copy()
    metadata = original_metadata.copy()

    if problem == "wrong_label":
        pin.loc[1, "target_label"] = 0

    elif problem == "wrong_fold":
        pin.loc[1, "strat_fold"] = 8

    elif problem == "unknown_diagnosis":
        metadata.loc[1, "scp_codes"] = "{'STTC': 100.0}"

    elif problem == "patient_overlap":
        metadata.loc[1, "patient_id"] = 101

    elif problem == "missing_record":
        metadata = metadata.iloc[:1]

    with pytest.raises((ValueError, KeyError)):
        build_from_synthetic(
            synthetic_cohort, pin, metadata
        )
