
"""CardioSense ensemble inference and quality assessment.

Legacy V1 inference remains available for regression testing.
V3 quality assessment never invokes unverified model checkpoints.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.ecg_loader import load_standardized_ecg
from src.dsp.filters import filter_ecg
from src.features.extract_features import (
    FEATURE_COLUMNS,
    extract_clinical_features,
)
from src.models.ecg_cnn import CardioNet1D
from src.triage.inference_quality import evaluate_research_readiness
from src.triage.triage_rules import ClinicalTriageEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


class CardioSenseEnsemble:
    """Provide legacy model inference and separate V3 quality assessment."""

    def __init__(self, checkpoints_dir=None):
        self.checkpoints_dir = (
            Path(checkpoints_dir)
            if checkpoints_dir is not None
            else PROJECT_ROOT / "models" / "checkpoints"
        )

        rf_path = self.checkpoints_dir / "interpretable_model.joblib"
        cnn_path = self.checkpoints_dir / "ecg_cnn.pt"

        missing = [
            str(path)
            for path in (rf_path, cnn_path)
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required model checkpoints:\n"
                + "\n".join(missing)
            )

        self.device = torch.device("cpu")
        self.rf_model = joblib.load(rf_path)

        self.cnn_model = CardioNet1D(
            in_channels=12,
            num_classes=2,
        ).to(self.device)

        checkpoint = torch.load(
            cnn_path,
            map_location=self.device,
            weights_only=True,
        )

        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
            else checkpoint
        )

        self.cnn_model.load_state_dict(state_dict)
        self.cnn_model.eval()

        self.triage_engine = ClinicalTriageEngine()

    @staticmethod
    def assess_v3_record(record_path):
        """Assess a WFDB record with V3; never run model inference.

        This method is static so that no RF/CNN checkpoints need to
        be loaded merely to inspect a recording's quality.
        """

        # The shared loader validates units, channels, and sampling rate.
        signals, fs, lead_names = load_standardized_ecg(record_path)

        extraction = extract_clinical_features(
            signals,
            fs=fs,
            lead_names=lead_names,
            sqi_version="v3",
        )

        status = extraction.get("status")

        if status not in ("success", "rejected"):
            raise ValueError(
                f"Unexpected feature extraction status: {status}"
            )

        # The quality gate checks V3 metadata and every RF feature.
        quality = evaluate_research_readiness(extraction)

        if status == "rejected":
            outcome = "abstained"
            reason = extraction.get("reason", quality["reason"])

        elif not quality["allow_research_scoring"]:
            outcome = "abstained"
            reason = quality["reason"]

        else:
            # Passing the quality gate is not sufficient to score:
            # the V3-compatible RF checkpoint is not yet verified.
            outcome = "quality_eligible_unscored"
            reason = (
                "Quality gate passed, but V3 model compatibility "
                "has not been established"
            )

        return {
            "status": outcome,
            "reason": reason,
            "sqi_version": "v3",
            "quality_decision": quality,
            "sqi_details": extraction.get("sqi_details"),
            "quality_contract": extraction.get("quality_contract"),
            "model_compatibility": "not_verified",
            "model_scoring_performed": False,
            "risk_score_pct": None,
            "clinical_triage": None,
        }

    @staticmethod
    def _validate_probability(value, model_name):
        """Require a finite model probability between zero and one."""

        probability = float(value)

        if not np.isfinite(probability):
            raise ValueError(
                f"{model_name} returned a non-finite probability."
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{model_name} returned a probability outside [0, 1]."
            )

        return probability

    def _predict_rf(self, features):
        """Run the existing calibrated tabular classifier."""

        frame = pd.DataFrame([features])[FEATURE_COLUMNS]
        probability = self.rf_model.predict_proba(frame)[0, 1]

        return self._validate_probability(
            probability, "Random Forest"
        )

    def _predict_cnn(self, raw_signals, fs):
        """Filter, normalize, and classify a 12-lead waveform."""

        cleaned = filter_ecg(raw_signals, fs=fs)

        mean = np.mean(cleaned, axis=0, keepdims=True)
        std = np.std(cleaned, axis=0, keepdims=True) + 1e-6

        normalized = (cleaned - mean) / std

        waveform = torch.as_tensor(
            normalized.T.copy(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.inference_mode():
            logits = self.cnn_model(waveform)
            probability = torch.softmax(logits, dim=1)[0, 1].item()

        return self._validate_probability(
            probability, "1D-CNN"
        )

    @staticmethod
    def _statistical_tier(risk_score_pct):
        """Apply the existing prototype probability bands."""

        if risk_score_pct < 35.0:
            return (
                "TIER_1_ROUTINE",
                "Routine Follow-up (Normal / Sub-acute findings)",
            )

        if risk_score_pct < 65.0:
            return (
                "TIER_2_REVIEW",
                "Urgent Clinician Review "
                "(Intermediate Risk / Borderline findings)",
            )

        return (
            "TIER_3_ESCALATION",
            "Immediate Emergency Escalation "
            "(High Risk Infarction Suspected)",
        )

    @staticmethod
    def _analyze_territories(features):
        """Preserve the existing territory summary calculation."""

        territories = {
            "Anterior / Septal (LAD)": {
                "artery": "Left Anterior Descending (LAD)",
                "leads": ["V1", "V2", "V3", "V4"],
                "max_st_mv": features["st_anterior_max"],
            },
            "Inferior (RCA)": {
                "artery": "Right Coronary Artery (RCA)",
                "leads": ["II", "III", "AVF"],
                "max_st_mv": features["st_inferior_max"],
            },
            "Lateral (LCx)": {
                "artery": "Left Circumflex (LCx)",
                "leads": ["I", "AVL", "V5", "V6"],
                "max_st_mv": features["st_lateral_max"],
            },
        }

        primary_name, primary_data = max(
            territories.items(),
            key=lambda item: item[1]["max_st_mv"],
        )

        culprit_artery = (
            primary_data["artery"]
            if primary_data["max_st_mv"] >= 0.10
            else "None (Sub-threshold ST deviation)"
        )

        return territories, primary_name, culprit_artery

    def predict(
        self,
        raw_signals,
        fs=100.0,
        lead_names=None,
        age=None,
        sex=None,
    ):
        """Run the existing V1-trained ensemble.

        Retained for compatibility with earlier experiments.
        This path does not use the V3 quality gate and is not
        a clinically validated decision-support system.
        """

        if lead_names is None:
            lead_names = DEFAULT_LEADS.copy()

        # Explicitly pin the legacy preprocessing policy.
        feat_result = extract_clinical_features(
            raw_signals,
            fs=fs,
            lead_names=lead_names,
            sqi_version="v1",
        )

        if feat_result["status"] == "rejected":
            return {
                "status": "rejected",
                "reason": feat_result["reason"],
                "sqi_details": feat_result.get("sqi_details"),
            }

        if feat_result["status"] != "success":
            raise ValueError(
                "Unexpected feature extraction status: "
                f"{feat_result['status']}"
            )

        features = feat_result["features"]

        trusted_leads = [
            lead.strip()
            for lead in features["leads_trusted"].split(",")
            if lead.strip()
        ]

        st_by_lead = {
            lead: float(features[f"st_{lead.upper()}"])
            for lead in trusted_leads
        }

        p_rf = self._predict_rf(features)
        p_cnn = self._predict_cnn(raw_signals, fs)

        p_composite = 0.5 * p_rf + 0.5 * p_cnn
        risk_score_pct = round(p_composite * 100.0, 1)

        statistical_tier, statistical_action = (
            self._statistical_tier(risk_score_pct)
        )

        territories, primary_territory, culprit_artery = (
            self._analyze_territories(features)
        )

        result = {
            "status": "success",
            "risk_score_pct": risk_score_pct,
            "statistical_tier": statistical_tier,
            "risk_tier": statistical_tier,
            "triage_action": statistical_action,
            "engine_probabilities": {
                "interpretable_rf": round(p_rf, 3),
                "deep_cnn": round(p_cnn, 3),
                "fused_composite": round(p_composite, 3),
            },
            "culprit_artery": culprit_artery,
            "primary_ischemic_territory": primary_territory,
            "territory_breakdown": territories,
            "st_by_lead": st_by_lead,
            "patient": {
                "age": age,
                "sex": sex,
            },
            "clinical_biomarkers": {
                "heart_rate_bpm": features["heart_rate_bpm"],
                "qrs_duration_ms": features["qrs_duration_ms"],
                "qrs_plausible": features["qrs_plausible"],
                "max_st_elevation_mv": features["max_st_elevation_mv"],
                "overall_sqi": features["overall_sqi"],
                "beats_used": features["beats_used"],
            },
        }

        clinical_triage = self.triage_engine.evaluate(result)

        result["clinical_triage"] = clinical_triage
        result["risk_tier"] = clinical_triage["final_tier"]
        result["triage_action"] = clinical_triage["triage_action"]

        return result
