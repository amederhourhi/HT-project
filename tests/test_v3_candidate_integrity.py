"""Offline synthetic tests for the read-only V3 checkpoint gate."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np

from scripts import verify_v3_candidate as gate


def _write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _json(root, relative, value):
    return _write(root, relative, json.dumps(value).encode("utf-8"))


def make_fake_repo(tmp_path, monkeypatch):
    """Build tiny fake artifacts with internally consistent hashes."""
    root = tmp_path
    manifest = _write(root, gate.MANIFEST, b"manifest")
    quality = _write(root, gate.QUALITY_AUDIT, b"quality")
    dataset = _write(root, gate.DATASET, b"training data")
    trainer = _write(root, gate.TRAINER, b"trainer source")
    model = _write(root, gate.MODEL, b"trusted synthetic model")
    rf = _write(root, gate.BASELINE_RF, b"unchanged V1 RF")
    cnn = _write(root, gate.BASELINE_CNN, b"unchanged V1 CNN")
    source = _write(root, "src/dsp/filters.py", b"frozen filters")

    data_contract = {
        "contract_version": "v3_research_dataset_v1",
        "sqi_version": "v3",
        "quality_policy": "fail_closed_research_gate",
        "feature_order": list(gate.FEATURE_COLUMNS),
        "record_count": 4,
        "split_label_counts": [
            {"split": "train", "target_label": 0, "count": 1},
            {"split": "train", "target_label": 1, "count": 1},
            {"split": "validation", "target_label": 1, "count": 1},
            {"split": "test", "target_label": 0, "count": 1},
        ],
        "inputs": {"manifest_sha256": manifest, "quality_audit_sha256": quality},
        "dataset": {"sha256": dataset},
        "source_sha256": {"src/dsp/filters.py": source},
    }
    contract_hash = _json(root, gate.DATASET_CONTRACT, data_contract)
    report = {
        "contract_version": "rf_v3_research_candidate_v1",
        "status": "research_candidate",
        "sqi_version": "v3",
        "quality_policy": "fail_closed_research_gate",
        "feature_order": list(gate.FEATURE_COLUMNS),
        "class_order": [0, 1],
        "training": {"folds": list(range(1,9)), "records": 2},
        "validation": {"fold": 9, "metrics": {"records": 1}},
        "test": {"fold": 10, "records": 1, "evaluated": False},
        "deployed_model_replaced": False,
        "clinical_validation": "not_established",
        "artifacts": {
            "candidate_model": gate.MODEL,
            "candidate_model_sha256": model,
            "training_dataset_sha256": dataset,
            "dataset_contract_sha256": contract_hash,
            "quality_audit_sha256": quality,
            "training_script_sha256": trainer,
        },
    }
    _json(root, gate.REPORT, report)
    _json(root, gate.BASELINE_AUDIT, {"deployed_models": {
        "random_forest": {"artifact": {"sha256": rf}},
        "cnn": {"artifact": {"sha256": cnn}},
    }})
    def fake_load(path):
        assert path == root / gate.MODEL
        return SimpleNamespace(
            feature_names_in_=np.array(gate.FEATURE_COLUMNS),
            n_features_in_=len(gate.FEATURE_COLUMNS),
            classes_=np.array([0, 1]),
        )
    monkeypatch.setattr(gate.joblib, "load", fake_load)
    return root


def test_valid_artifact_passes(tmp_path, monkeypatch):
    root = make_fake_repo(tmp_path, monkeypatch)
    result = gate.audit_candidate(root)
    assert result["status"] == "integrity_verified", result["issues"]
    assert result["checks"]["candidate_rf_structure"]
    assert result["test_fold_evaluated_by_this_audit"] is False


def test_missing_local_checkpoint_fails(tmp_path, monkeypatch):
    root = make_fake_repo(tmp_path, monkeypatch)
    (root / gate.MODEL).unlink()
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert not result["checks"]["candidate_model_hash"]


def test_tampered_model_never_deserialized(tmp_path, monkeypatch):
    root = make_fake_repo(tmp_path, monkeypatch)
    (root / gate.MODEL).write_bytes(b"untrusted bytes")
    def fail_load(*args):
        raise AssertionError("Must not deserialize on hash mismatch")
    monkeypatch.setattr(gate.joblib, "load", fail_load)
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert "candidate_rf_structure" not in result["checks"]


def test_test_fold_opened_is_blocked(tmp_path, monkeypatch):
    root = make_fake_repo(tmp_path, monkeypatch)
    p = root / gate.REPORT
    report = json.loads(p.read_text(encoding="utf-8"))
    report["test"]["evaluated"] = True
    _json(root, gate.REPORT, report)
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert not result["checks"]["locked_partitions"]


def test_change_to_frozen_source_is_blocked(tmp_path, monkeypatch):
    root = make_fake_repo(tmp_path, monkeypatch)
    (root / "src/dsp/filters.py").write_bytes(b"changed filters")
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert not result["checks"]["source:src/dsp/filters.py"]


def test_deployed_v1_model_modification_is_blocked(tmp_path, monkeypatch):
    root = make_fake_repo(tmp_path, monkeypatch)
    (root / gate.BASELINE_RF).write_bytes(b"changed V1 model")
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert not result["checks"]["deployed_rf_unchanged"]


def test_wrong_model_schema_fails(tmp_path, monkeypatch):
    root = make_fake_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(gate.joblib, "load", lambda path: SimpleNamespace(
        feature_names_in_=np.array(list(reversed(gate.FEATURE_COLUMNS))),
        n_features_in_=len(gate.FEATURE_COLUMNS),
        classes_=np.array([0, 1]),
    ))
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert not result["checks"]["candidate_rf_structure"]

# -----------------------------------------------------------------------------
# Reconstructed-manifest exception regression tests
# -----------------------------------------------------------------------------

import pandas as pd

from src.data.download_ptbxl import build_cohort_manifest


def make_reconstructed_repo(tmp_path, monkeypatch):
    """Create a real pinned mini-cohort whose new bytes differ from old hash."""
    root = make_fake_repo(tmp_path, monkeypatch)

    rows = [
        (11, 101, 1, 0, 45, 0, "{'NORM': 100.0}"),
        (22, 202, 2, 1, 56, 1, "{'AMI': 100.0}"),
        (33, 303, 9, 1, 63, 1, "{'IMI': 100.0}"),
        (44, 404, 10, 0, 48, 0, "{'NORM': 100.0}"),
    ]
    pin = pd.DataFrame([
        {"ecg_id": x[0], "strat_fold": x[2], "target_label": x[3]}
        for x in rows
    ])
    official = pd.DataFrame([
        {
            "ecg_id": ecg_id, "patient_id": patient,
            "strat_fold": fold, "age": age, "sex": sex,
            "scp_codes": code, "filename_lr": f"records100/{ecg_id}_lr",
            "infarction_stadium1": "", "validated_by_human": True,
        }
        for ecg_id, patient, fold, label, age, sex, code in rows
    ])

    (root / gate.PIN).parent.mkdir(parents=True, exist_ok=True)
    (root / gate.OFFICIAL_METADATA).parent.mkdir(parents=True, exist_ok=True)
    pin.to_csv(root / gate.PIN, index=False)
    official.to_csv(root / gate.OFFICIAL_METADATA, index=False)

    build_cohort_manifest(
        pinned_ids_csv=root / gate.PIN,
        database_csv=root / gate.OFFICIAL_METADATA,
        cohort_dir=(root / "data/cohort"),
        download=False,
    )

    audit = pd.DataFrame([
        {
            "ecg_id": ecg_id, "target_label": label,
            "strat_fold": fold, "quality_decision": "eligible",
        }
        for ecg_id, patient, fold, label, age, sex, code in rows
    ])
    audit.to_csv(root / gate.QUALITY_AUDIT, index=False)

    manifest = pd.read_csv(root / gate.MANIFEST)
    data = manifest[list(gate.META_COLUMNS)].copy()
    for feature in gate.FEATURE_COLUMNS:
        data[feature] = 0.0
    data.to_csv(root / gate.DATASET, index=False)

    # Update every legitimate frozen hash, but preserve the original historical
    # manifest SHA from make_fake_repo() as a deliberate mismatch.
    contract_path = root / gate.DATASET_CONTRACT
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["dataset"]["sha256"] = gate.sha256_file(root / gate.DATASET)
    contract["inputs"]["quality_audit_sha256"] = gate.sha256_file(root / gate.QUALITY_AUDIT)
    assert contract["inputs"]["manifest_sha256"] != gate.sha256_file(root / gate.MANIFEST)
    _json(root, gate.DATASET_CONTRACT, contract)

    report_path = root / gate.REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"]["training_dataset_sha256"] = gate.sha256_file(root / gate.DATASET)
    report["artifacts"]["dataset_contract_sha256"] = gate.sha256_file(root / gate.DATASET_CONTRACT)
    report["artifacts"]["quality_audit_sha256"] = gate.sha256_file(root / gate.QUALITY_AUDIT)
    _json(root, gate.REPORT, report)
    return root


def test_reconstructed_manifest_is_separate_explicit_status(tmp_path, monkeypatch):
    root = make_reconstructed_repo(tmp_path, monkeypatch)
    result = gate.audit_candidate(root)
    assert result["status"] == "research_integrity_verified_with_manifest_exception", result["issues"]
    assert result["manifest_provenance"] == "reconstructed_metadata_aligned"
    assert result["checks"]["historical_manifest_hash_matches"] is False
    assert result["checks"]["manifest_reconstruction_verified"] is True
    assert result["warnings"]
    assert result["clinical_validation"] == "not_established"
    assert result["test_fold_evaluated_by_this_audit"] is False


def test_reconstruction_missing_metadata_blocks(tmp_path, monkeypatch):
    root = make_reconstructed_repo(tmp_path, monkeypatch)
    (root / gate.OFFICIAL_METADATA).unlink()
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert result["checks"]["manifest_reconstruction_verified"] is False


def test_current_manifest_not_reproducible_blocks(tmp_path, monkeypatch):
    root = make_reconstructed_repo(tmp_path, monkeypatch)
    with (root / gate.MANIFEST).open("ab") as stream:
        stream.write(b"\n")
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert result["checks"]["manifest_reconstruction_verified"] is False


def test_v3_age_mismatch_blocks_even_if_dataset_hashes_refreshed(tmp_path, monkeypatch):
    root = make_reconstructed_repo(tmp_path, monkeypatch)
    frame = pd.read_csv(root / gate.DATASET)
    frame.loc[0, "age"] = 99
    frame.to_csv(root / gate.DATASET, index=False)

    contract_path = root / gate.DATASET_CONTRACT
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["dataset"]["sha256"] = gate.sha256_file(root / gate.DATASET)
    _json(root, gate.DATASET_CONTRACT, contract)

    report_path = root / gate.REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"]["training_dataset_sha256"] = gate.sha256_file(root / gate.DATASET)
    report["artifacts"]["dataset_contract_sha256"] = gate.sha256_file(root / gate.DATASET_CONTRACT)
    _json(root, gate.REPORT, report)

    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert not result["checks"]["manifest_reconstruction_verified"]


def test_wrong_pinned_membership_blocks(tmp_path, monkeypatch):
    root = make_reconstructed_repo(tmp_path, monkeypatch)
    pin = pd.read_csv(root / gate.PIN)
    pin.loc[0, "ecg_id"] = 55
    pin.to_csv(root / gate.PIN, index=False)
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert not result["checks"]["manifest_reconstruction_verified"]


def test_bad_reconstruction_never_deserializes_model(tmp_path, monkeypatch):
    root = make_reconstructed_repo(tmp_path, monkeypatch)
    (root / gate.OFFICIAL_METADATA).unlink()
    def fail_load(*args):
        raise AssertionError("Never deserialize after a failed provenance gate")
    monkeypatch.setattr(gate.joblib, "load", fail_load)
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert "candidate_rf_structure" not in result["checks"]


def test_tampered_candidate_with_reconstructed_manifest_still_blocks(tmp_path, monkeypatch):
    root = make_reconstructed_repo(tmp_path, monkeypatch)
    (root / gate.MODEL).write_bytes(b"tampered checkpoint")
    result = gate.audit_candidate(root)
    assert result["status"] == "blocked"
    assert not result["checks"]["candidate_model_hash"]