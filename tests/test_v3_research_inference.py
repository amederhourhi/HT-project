
"""Offline, synthetic tests for isolated V3 research scoring."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.triage.v3_research_inference as scorer_module
from src.features.extract_features import FEATURE_COLUMNS


class FakeRF:
    def __init__(self):
        self.feature_names_in_ = list(FEATURE_COLUMNS)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.classes_ = [0, 1]
        self.calls = 0
        self.probabilities = np.array([[0.7, 0.3]])

    def predict_proba(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        assert frame.shape == (1, len(FEATURE_COLUMNS))
        self.calls += 1
        return self.probabilities


@pytest.fixture
def harness(tmp_path, monkeypatch):
    model_path = tmp_path / scorer_module.MODEL
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"synthetic checked artifact")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

    report_path = tmp_path / scorer_module.REPORT
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({
            "artifacts": {
                "candidate_model_sha256": digest,
            },
            "feature_order": FEATURE_COLUMNS,
            "class_order": [0, 1],
            "sqi_version": "v3",
            "test": {"evaluated": False},
        }),
        encoding="utf-8",
    )

    manifest_path = tmp_path / scorer_module.MANIFEST
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame([
        {
            "ecg_id": 101,
            "strat_fold": 7,
            "filename_lr": "records100/101_lr",
        },
        {
            "ecg_id": 102,
            "strat_fold": 9,
            "filename_lr": "records100/102_lr",
        },
        {
            "ecg_id": 103,
            "strat_fold": 10,
            "filename_lr": "records100/103_lr",
        },
        {
            "ecg_id": 104,
            "strat_fold": 8,
            "filename_lr": "records100/104_lr",
        },
    ]).to_csv(manifest_path, index=False)

    quality_path = tmp_path / scorer_module.QUALITY_AUDIT
    quality_path.parent.mkdir(parents=True)
    pd.DataFrame([
        {
            "ecg_id": 101,
            "quality_decision": "eligible",
        },
        {
            "ecg_id": 102,
            "quality_decision": "eligible",
        },
        {
            "ecg_id": 103,
            "quality_decision": "eligible",
        },
        {
            "ecg_id": 104,
            "quality_decision": "abstain",
        },
    ]).to_csv(quality_path, index=False)

    for record_id in (101, 102):
        base = (
            manifest_path.parent
            / f"records100/{record_id}_lr"
        )
        base.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        base.with_suffix(".hea").write_text(
            "synthetic",
            encoding="utf-8",
        )
        base.with_suffix(".dat").write_bytes(
            b"synthetic"
        )

    audit = {
        "status": (
            "research_integrity_verified_with_manifest_exception"
        ),
        "issues": [],
        "manifest_provenance": (
            "reconstructed_metadata_aligned"
        ),
    }

    fake = FakeRF()
    calls = {
        "load": 0,
        "extract": 0,
        "model_load": 0,
    }

    def fake_loader(path):
        calls["load"] += 1
        return (
            np.zeros((1000, 12)),
            100.0,
            list(scorer_module.STANDARD_LEADS),
        )

    def fake_extraction(*args, **kwargs):
        calls["extract"] += 1
        assert kwargs["sqi_version"] == "v3"
        return {
            "status": "success",
            "features": {
                name: 0.0
                for name in FEATURE_COLUMNS
            },
        }

    def fake_model_load(path):
        calls["model_load"] += 1
        return fake

    monkeypatch.setattr(
        scorer_module,
        "audit_candidate",
        lambda root: audit,
    )
    monkeypatch.setattr(
        scorer_module,
        "load_standardized_ecg",
        fake_loader,
    )
    monkeypatch.setattr(
        scorer_module,
        "extract_clinical_features",
        fake_extraction,
    )
    monkeypatch.setattr(
        scorer_module,
        "evaluate_research_readiness",
        lambda x: {
            "decision": "eligible",
            "allow_research_scoring": True,
            "reason": "Research quality eligible",
        },
    )
    monkeypatch.setattr(
        scorer_module.joblib,
        "load",
        fake_model_load,
    )

    return SimpleNamespace(
        root=tmp_path,
        audit=audit,
        model=fake,
        calls=calls,
        model_path=model_path,
    )


def test_training_record_scored_as_in_sample(harness):
    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )
    result = scorer.score_ecg_id(101)

    assert result["status"] == "scored_research_only"
    assert result["evaluation_context"] == (
        "in_sample_training_cohort"
    )
    assert result[
        "research_probability_infarction_family"
    ] == pytest.approx(0.3)
    assert result["model_scoring_performed"] is True
    assert result["clinical_triage"] is None
    assert result[
        "historical_manifest_identity_verified"
    ] is False
    assert harness.model.calls == 1


def test_fold9_is_identified_as_development_validation(
    harness,
):
    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )
    result = scorer.score_ecg_id(102)

    assert result["status"] == "scored_research_only"
    assert result["evaluation_context"] == (
        "development_validation_cohort"
    )


def test_fold10_is_never_loaded_or_scored(harness):
    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )
    result = scorer.score_ecg_id(103)

    assert result["status"] == "abstained"
    assert "Fold 10" in result["reason"]
    assert result[
        "research_probability_infarction_family"
    ] is None
    assert harness.calls["load"] == 0
    assert harness.model.calls == 0


def test_frozen_quality_audit_abstains_before_loading(
    harness,
):
    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )
    result = scorer.score_ecg_id(104)

    assert result["status"] == "abstained"
    assert harness.calls["load"] == 0
    assert harness.model.calls == 0


def test_unknown_ecg_id_abstains(harness):
    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )
    result = scorer.score_ecg_id(888)

    assert result["status"] == "abstained"
    assert harness.calls["load"] == 0


def test_invalid_ecg_id_is_rejected(harness):
    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )

    with pytest.raises(ValueError):
        scorer.score_ecg_id("101")


def test_current_quality_gate_abstains(
    harness,
    monkeypatch,
):
    monkeypatch.setattr(
        scorer_module,
        "evaluate_research_readiness",
        lambda x: {
            "decision": "abstain",
            "allow_research_scoring": False,
            "reason": "Measurement review required",
        },
    )

    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )
    result = scorer.score_ecg_id(101)

    assert result["status"] == "abstained"
    assert "Measurement review required" in result["reason"]
    assert harness.model.calls == 0


def test_extraction_failure_abstains(
    harness,
    monkeypatch,
):
    monkeypatch.setattr(
        scorer_module,
        "extract_clinical_features",
        lambda *args, **kwargs: {
            "status": "rejected",
            "reason": "technical failure",
        },
    )

    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )
    result = scorer.score_ecg_id(101)

    assert result["status"] == "abstained"
    assert harness.model.calls == 0


def test_bad_probability_fails_closed(harness):
    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )

    harness.model.probabilities = np.array([
        [np.nan, np.nan]
    ])

    with pytest.raises(
        ValueError,
        match="invalid probabilities",
    ):
        scorer.score_ecg_id(101)


def test_checkpoint_schema_mismatch_blocks_initialization(
    harness,
):
    harness.model.classes_ = [1, 0]

    with pytest.raises(
        RuntimeError,
        match="incompatible feature schema",
    ):
        scorer_module.V3ResearchScorer(
            root=harness.root
        )


def test_blocked_integrity_audit_does_not_load_model(
    harness,
):
    harness.audit["status"] = "blocked"
    harness.audit["issues"] = [
        "model hash mismatch"
    ]

    with pytest.raises(
        RuntimeError,
        match="integrity gate blocked",
    ):
        scorer_module.V3ResearchScorer(
            root=harness.root
        )

    assert harness.calls["model_load"] == 0


def test_unverified_manifest_exception_is_rejected(
    harness,
):
    harness.audit["manifest_provenance"] = "unverified"

    with pytest.raises(
        RuntimeError,
        match="manifest exception",
    ):
        scorer_module.V3ResearchScorer(
            root=harness.root
        )

    assert harness.calls["model_load"] == 0


def test_candidate_tamper_after_audit_blocks_loading(
    harness,
):
    harness.model_path.write_bytes(
        b"modified after audit"
    )

    with pytest.raises(
        RuntimeError,
        match="hash changed",
    ):
        scorer_module.V3ResearchScorer(
            root=harness.root
        )

    assert harness.calls["model_load"] == 0


def test_invalid_waveform_path_cannot_escape_cohort(
    harness,
):
    manifest_path = (
        harness.root / scorer_module.MANIFEST
    )

    frame = pd.read_csv(manifest_path)

    frame.loc[
        frame.ecg_id == 101,
        "filename_lr",
    ] = "../../escape"

    frame.to_csv(
        manifest_path,
        index=False,
    )

    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )

    with pytest.raises(
        RuntimeError,
        match="escapes the cohort",
    ):
        scorer.score_ecg_id(101)

    assert harness.calls["load"] == 0


def test_missing_waveform_abstains(harness):
    base = (
        harness.root
        / "data/cohort/records100/101_lr"
    )
    base.with_suffix(".dat").unlink()

    scorer = scorer_module.V3ResearchScorer(
        root=harness.root
    )
    result = scorer.score_ecg_id(101)

    assert result["status"] == "abstained"
    assert harness.calls["load"] == 0
