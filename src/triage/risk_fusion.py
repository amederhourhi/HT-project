
"""
CardioSense - Ensemble Inference and Clinical Risk Fusion.

Combines Random Forest and 1D-CNN predictions with deterministic
clinical triage rules.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.dsp.filters import filter_ecg
from src.features.extract_features import (
    FEATURE_COLUMNS,
    extract_clinical_features,
)
from src.models.ecg_cnn import CardioNet1D
from src.triage.triage_rules import ClinicalTriageEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


class CardioSenseEnsemble:
    """Run ECG inference and apply clinical triage guardrails."""

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
                + "\nTrain the models before running inference."
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
    def _validate_probability(value, model_name):
        """Reject invalid classifier probability outputs."""
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
        """Predict using the calibrated tabular classifier."""
        feature_frame = pd.DataFrame([features])[FEATURE_COLUMNS]

        probability = self.rf_model.predict_proba(feature_frame)[0, 1]

        return self._validate_probability(
            probability,
            "Random Forest",
        )

    def _predict_cnn(self, raw_signals, fs):
        """Predict using normalized 12-lead waveforms."""
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
            probability,
            "1D-CNN",
        )

    @staticmethod
    def _statistical_tier(risk_score_pct):
        """Map ensemble score to the existing prototype thresholds."""
        if risk_score_pct < 35.0:
            return (
                "TIER_1_ROUTINE",
                "Routine Follow-up (Normal / Sub-acute findings)",
            )

        if risk_score_pct < 65.0:
            return (
                "TIER_2_REVIEW",
                "Urgent Clinician Review (Intermediate Risk / Borderline findings)",
            )

        return (
            "TIER_3_ESCALATION",
            "Immediate Emergency Escalation (High Risk Infarction Suspected)",
        )

    @staticmethod
    def _analyze_territories(features):
        """Summarize regional ST measurements."""
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
        """Run ECG analysis, model inference, and clinical triage."""
        if lead_names is None:
            lead_names = DEFAULT_LEADS.copy()

        # Signal quality must pass before classification.
        feat_result = extract_clinical_features(
            raw_signals,
            fs=fs,
            lead_names=lead_names,
        )

        if feat_result["status"] == "rejected":
            return {
                "status": "rejected",
                "reason": feat_result["reason"],
                "sqi_details": feat_result.get("sqi_details"),
            }

        if feat_result["status"] != "success":
            raise ValueError(
                f"Unexpected feature extraction status: "
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

        # Statistical inference.
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

        # Clinical rules can escalate, but never downgrade, ML risk.
        clinical_triage = self.triage_engine.evaluate(result)

        result["clinical_triage"] = clinical_triage
        result["risk_tier"] = clinical_triage["final_tier"]
        result["triage_action"] = clinical_triage["triage_action"]

        return result
