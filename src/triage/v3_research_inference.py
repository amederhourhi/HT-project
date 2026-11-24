
"""Isolated, fail-closed SQI V3 Random Forest research scoring.

This component is deliberately NOT used by the legacy V1 ensemble or API.
It accepts only ECG IDs from the pinned, verified cohort, and refuses fold 10.

Research scoring is permitted only when:
  * the existing candidate integrity audit passes, including its explicitly
    documented reconstructed-manifest exception if applicable;
  * the candidate checkpoint still matches its frozen SHA-256 contract;
  * the record is outside fold 10 and was eligible in the frozen quality audit;
  * current SQI V3 extraction passes evaluate_research_readiness(); and
  * the candidate produces a valid binary probability vector.

No clinical tier, diagnosis, disposition, or emergency recommendation is emitted.
This module does not train, persist or modify a model, or evaluate fold 10.
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.verify_v3_candidate import (
    MODEL,
    REPORT,
    MANIFEST,
    QUALITY_AUDIT,
    ROOT,
    audit_candidate,
    sha256_file,
)
from src.data.ecg_loader import load_standardized_ecg
from src.dsp.input_validation import STANDARD_LEADS
from src.features.extract_features import (
    FEATURE_COLUMNS,
    extract_clinical_features,
)
from src.triage.inference_quality import evaluate_research_readiness


ACCEPTED_AUDIT_STATUSES = frozenset({
    "integrity_verified",
    "research_integrity_verified_with_manifest_exception",
})


class V3ResearchScorer:
    """Load a verified research RF candidate without affecting deployed V1.

    This is a local, pinned-cohort development tool, not an API service.
    Its returned probability refers to the project's binary research label
    (infarction family), not to a validated acute-MI diagnosis.
    """

    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.audit = audit_candidate(self.root)
        status = self.audit.get("status")

        if (
            status not in ACCEPTED_AUDIT_STATUSES
            or self.audit.get("issues")
        ):
            raise RuntimeError(
                "V3 research checkpoint integrity gate blocked: "
                + "; ".join(
                    self.audit.get("issues") or [str(status)]
                )
            )

        if (
            status
            == "research_integrity_verified_with_manifest_exception"
            and self.audit.get("manifest_provenance")
            != "reconstructed_metadata_aligned"
        ):
            raise RuntimeError(
                "V3 manifest exception lacks verified provenance"
            )

        self.manifest_provenance = self.audit[
            "manifest_provenance"
        ]

        self.training_contract = json.loads(
            (self.root / REPORT).read_text(
                encoding="utf-8"
            )
        )

        model_path = self.root / MODEL

        expected_hash = self.training_contract[
            "artifacts"
        ]["candidate_model_sha256"]

        # Close the gap between audit verification
        # and the local model load.
        if sha256_file(model_path) != expected_hash:
            raise RuntimeError(
                "V3 checkpoint hash changed since integrity audit"
            )

        if (
            self.training_contract.get("feature_order")
            != FEATURE_COLUMNS
            or self.training_contract.get("class_order")
            != [0, 1]
            or self.training_contract.get("sqi_version")
            != "v3"
            or self.training_contract.get(
                "test", {}
            ).get("evaluated") is not False
        ):
            raise RuntimeError(
                "V3 research checkpoint contract mismatch"
            )

        # joblib uses pickle-based deserialization.
        # Load only after checking project-owned hashes.
        self.model = joblib.load(model_path)

        if (
            list(
                getattr(
                    self.model,
                    "feature_names_in_",
                    [],
                )
            ) != FEATURE_COLUMNS
            or int(
                getattr(
                    self.model,
                    "n_features_in_",
                    -1,
                )
            ) != len(FEATURE_COLUMNS)
            or list(
                getattr(
                    self.model,
                    "classes_",
                    [],
                )
            ) != [0, 1]
        ):
            raise RuntimeError(
                "V3 RF checkpoint has an incompatible feature schema"
            )

        manifest = pd.read_csv(
            self.root / MANIFEST
        )

        audit = pd.read_csv(
            self.root / QUALITY_AUDIT
        )

        required_manifest = {
            "ecg_id",
            "strat_fold",
            "filename_lr",
        }

        required_audit = {
            "ecg_id",
            "quality_decision",
        }

        if (
            not required_manifest.issubset(
                manifest.columns
            )
            or not required_audit.issubset(
                audit.columns
            )
            or manifest["ecg_id"].duplicated().any()
            or audit["ecg_id"].duplicated().any()
        ):
            raise RuntimeError(
                "Invalid frozen cohort manifest or quality audit"
            )

        # The integrity audit previously verified
        # their content and membership.
        self.manifest = manifest.set_index(
            "ecg_id",
            verify_integrity=True,
        )

        self.quality_audit = audit.set_index(
            "ecg_id",
            verify_integrity=True,
        )

        self.model_sha256 = expected_hash

    def _base_result(self, ecg_id, fold=None):
        """Build the non-clinical output shared by every outcome."""

        return {
            "ecg_id": ecg_id,
            "strat_fold": fold,
            "sqi_version": "v3",
            "model_type": "v3_rf_research_candidate",
            "candidate_model_sha256": self.model_sha256,
            "manifest_provenance": self.manifest_provenance,
            "historical_manifest_identity_verified": (
                self.manifest_provenance
                == "historical_hash_matched"
            ),
            "clinical_validation": "not_established",
            "clinical_triage": None,
            "triage_routing_modified": False,
            "v1_models_modified": False,
            "research_probability_infarction_family": None,
            "model_scoring_performed": False,
            "quality_decision": None,
        }

    def score_ecg_id(self, ecg_id):
        """Score one authorized development ECG; never fold 10.

        Fold 1-8 results are in-sample research predictions.
        Fold 9 is an already-used development validation fold.
        Neither represents new independent validation.
        """

        if (
            isinstance(ecg_id, bool)
            or not isinstance(
                ecg_id,
                (int, np.integer),
            )
            or ecg_id <= 0
        ):
            raise ValueError(
                "ecg_id must be a positive integer"
            )

        ecg_id = int(ecg_id)
        result = self._base_result(ecg_id)

        if (
            ecg_id not in self.manifest.index
            or ecg_id not in self.quality_audit.index
        ):
            return {
                **result,
                "status": "abstained",
                "reason": (
                    "ECG ID is not in the verified research cohort"
                ),
            }

        row = self.manifest.loc[ecg_id]
        fold = int(row["strat_fold"])
        result["strat_fold"] = fold

        # Block fold 10 BEFORE waveform loading
        # or model inference.
        if fold == 10:
            return {
                **result,
                "status": "abstained",
                "reason": (
                    "Fold 10 is reserved and cannot "
                    "be scored in development"
                ),
            }

        if fold not in range(1, 10):
            raise RuntimeError(
                "Invalid stratified fold in verified manifest"
            )

        result["evaluation_context"] = (
            "in_sample_training_cohort"
            if fold <= 8
            else "development_validation_cohort"
        )

        # Respect the frozen quality audit.
        if (
            self.quality_audit.loc[
                ecg_id,
                "quality_decision",
            ] != "eligible"
        ):
            return {
                **result,
                "status": "abstained",
                "reason": (
                    "Record was not eligible in the "
                    "frozen V3 quality audit"
                ),
            }

        # Only load waveform paths from the verified manifest.
        cohort_root = (
            self.root / "data" / "cohort"
        ).resolve()

        filename = row["filename_lr"]

        if (
            not isinstance(filename, str)
            or not filename.strip()
        ):
            raise RuntimeError(
                "Invalid manifest waveform filename"
            )

        record_path = (
            cohort_root / filename
        ).resolve()

        if not record_path.is_relative_to(
            cohort_root
        ):
            raise RuntimeError(
                "Manifest waveform path escapes the cohort directory"
            )

        if not all(
            record_path.with_suffix(ext).is_file()
            for ext in (".hea", ".dat")
        ):
            return {
                **result,
                "status": "abstained",
                "reason": (
                    "Waveform files are unavailable locally"
                ),
            }

        signals_mv, fs, lead_names = (
            load_standardized_ecg(record_path)
        )

        if (
            fs != 100.0
            or list(lead_names)
            != list(STANDARD_LEADS)
        ):
            raise ValueError(
                "Unexpected standardized ECG "
                "sampling rate or lead order"
            )

        if (
            signals_mv.ndim != 2
            or signals_mv.shape[1]
            != len(STANDARD_LEADS)
        ):
            raise ValueError(
                "Unexpected standardized ECG waveform shape"
            )

        if not np.isfinite(
            signals_mv
        ).all():
            raise ValueError(
                "Nonfinite standardized ECG samples"
            )

        # Run experimental V3 feature extraction.
        extraction = extract_clinical_features(
            signals_mv,
            fs=fs,
            lead_names=lead_names,
            sqi_version="v3",
        )

        quality = evaluate_research_readiness(
            extraction
        )

        result["quality_decision"] = quality

        # Fail closed when quality review is required.
        if (
            extraction.get("status") != "success"
            or quality.get("decision") != "eligible"
            or quality.get("allow_research_scoring")
            is not True
        ):
            return {
                **result,
                "status": "abstained",
                "reason": quality.get(
                    "reason",
                    "V3 quality gate did not pass",
                ),
            }

        features = extraction.get("features")

        if not isinstance(features, dict):
            raise ValueError(
                "Eligible extraction has no feature dictionary"
            )

        # Preserve the exact 23-feature model order.
        frame = pd.DataFrame(
            [features],
            columns=FEATURE_COLUMNS,
        )

        if (
            frame.isna().any().any()
            or not np.isfinite(
                frame.to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(
                "Eligible V3 features are missing or nonfinite"
            )

        probabilities = np.asarray(
            self.model.predict_proba(frame),
            dtype=float,
        )

        if (
            probabilities.shape != (1, 2)
            or not np.isfinite(
                probabilities
            ).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not math.isclose(
                float(probabilities.sum()),
                1.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                "V3 research model returned invalid probabilities"
            )

        return {
            **result,
            "status": "scored_research_only",
            "reason": (
                "Eligible for research RF scoring; "
                "not clinical validation"
            ),
            "model_scoring_performed": True,
            "research_probability_infarction_family": (
                float(probabilities[0, 1])
            ),
        }
