
"""
CardioSense — Ensemble Risk Fusion and Quality Assessment.

Two separate research pathways:

1. Legacy V1 inference:
   - Validate and standardize incoming ECG data
   - Extract V1 clinical features
   - Run Random Forest and 1D CNN
   - Combine probabilities
   - Apply research triage rules

2. Experimental V3 quality assessment:
   - Validate the WFDB recording
   - Extract V3 quality-aware features
   - Evaluate research readiness
   - Do NOT run unverified model checkpoints

This is a research prototype, not a clinically validated
diagnostic or emergency-triage system.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.ecg_loader import load_standardized_ecg
from src.dsp.input_validation import (
    STANDARD_LEADS,
    validate_ecg_input,
)
from src.dsp.filters import filter_ecg
from src.features.extract_features import (
    FEATURE_COLUMNS,
    extract_clinical_features,
)
from src.models.ecg_cnn import CardioNet1D
from src.triage.inference_quality import evaluate_research_readiness
from src.triage.triage_rules import ClinicalTriageEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Preserve the original list-based interface for existing tests.
DEFAULT_LEADS = list(STANDARD_LEADS)


class CardioSenseEnsemble:
    """Legacy V1 ensemble with separate experimental V3 quality assessment."""

    # ---------------------------------------------------------
    # 1. Initialize models
    # ---------------------------------------------------------
    def __init__(self, checkpoints_dir=None):
        """Load existing V1-compatible RF and CNN checkpoints."""

        self.checkpoints_dir = (
            Path(checkpoints_dir)
            if checkpoints_dir is not None
            else PROJECT_ROOT / "models" / "checkpoints"
        )

        rf_path = (
            self.checkpoints_dir / "interpretable_model.joblib"
        )
        cnn_path = (
            self.checkpoints_dir / "ecg_cnn.pt"
        )

        # Fail explicitly when required checkpoints are absent.
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

        # Load only trusted, project-generated model artifacts.
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

        # Support both direct state dictionaries and
        # checkpoints containing "model_state_dict".
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
            else checkpoint
        )

        self.cnn_model.load_state_dict(state_dict)
        self.cnn_model.eval()

        self.triage_engine = ClinicalTriageEngine()

    # ---------------------------------------------------------
    # 2. Experimental V3 quality assessment
    # ---------------------------------------------------------
    @staticmethod
    def assess_v3_record(record_path):
        """Assess WFDB quality without running RF/CNN inference.

        A successful quality assessment does not establish
        compatibility with the existing V1-trained checkpoints.
        """

        # Centralized WFDB loader:
        # validates units, sampling rate, and lead order.
        signals_mv, fs, lead_names = load_standardized_ecg(
            record_path
        )

        extraction = extract_clinical_features(
            signals_mv,
            fs=fs,
            lead_names=lead_names,
            sqi_version="v3",
        )

        status = extraction.get("status")

        if status not in ("success", "rejected"):
            raise ValueError(
                f"Unexpected feature extraction status: {status}"
            )

        # Check V3 measurement-quality metadata and the
        # readiness of every required RF feature.
        quality = evaluate_research_readiness(extraction)

        if status == "rejected":
            outcome = "abstained"
            reason = extraction.get(
                "reason",
                quality["reason"],
            )

        elif not quality["allow_research_scoring"]:
            outcome = "abstained"
            reason = quality["reason"]

        else:
            # Do not score with older V1 checkpoints.
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
            "quality_contract": extraction.get(
                "quality_contract"
            ),
            "model_compatibility": "not_verified",
            "model_scoring_performed": False,
            "risk_score_pct": None,
            "clinical_triage": None,
        }

    # ---------------------------------------------------------
    # 3. Probability validation
    # ---------------------------------------------------------
    @staticmethod
    def _validate_probability(value, model_name):
        """Require a finite probability between zero and one."""

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

    # ---------------------------------------------------------
    # 4. Random Forest inference
    # ---------------------------------------------------------
    def _predict_rf(self, features):
        """Run the existing feature-based Random Forest."""

        # Preserve the exact training feature order.
        frame = pd.DataFrame([features])[FEATURE_COLUMNS]

        probability = self.rf_model.predict_proba(
            frame
        )[0, 1]

        return self._validate_probability(
            probability,
            "Random Forest",
        )

    # ---------------------------------------------------------
    # 5. CNN inference
    # ---------------------------------------------------------
    def _predict_cnn(self, signals_mv, fs):
        """Classify an already standardized 12-lead ECG.

        IMPORTANT:
        The caller must supply signals in millivolts,
        with channels in STANDARD_LEADS order.
        """

        # Preserve the legacy preprocessing policy.
        cleaned = filter_ecg(
            signals_mv,
            fs=fs,
        )

        # Normalize independently for each lead.
        mean = np.mean(
            cleaned,
            axis=0,
            keepdims=True,
        )

        std = np.std(
            cleaned,
            axis=0,
            keepdims=True,
        ) + 1e-6

        normalized = (cleaned - mean) / std

        # CNN expects (batch, channels, samples).
        # (1000, 12) -> (1, 12, 1000)
        waveform = torch.as_tensor(
            normalized.T.copy(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.inference_mode():
            logits = self.cnn_model(waveform)

            probability = torch.softmax(
                logits,
                dim=1,
            )[0, 1].item()

        return self._validate_probability(
            probability,
            "1D-CNN",
        )

    # ---------------------------------------------------------
    # 6. Statistical risk tiers
    # ---------------------------------------------------------
    @staticmethod
    def _statistical_tier(risk_score_pct):
        """Apply the existing prototype probability bands.

        These are project-specific research thresholds,
        not clinically validated intervention cutoffs.
        """

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

    # ---------------------------------------------------------
    # 7. Anatomical territory summary
    # ---------------------------------------------------------
    @staticmethod
    def _analyze_territories(features):
        """Preserve the existing research territory summary."""

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

        # Preserve previous threshold behavior.
        culprit_artery = (
            primary_data["artery"]
            if primary_data["max_st_mv"] >= 0.10
            else "None (Sub-threshold ST deviation)"
        )

        return (
            territories,
            primary_name,
            culprit_artery,
        )

    # ---------------------------------------------------------
    # 8. Legacy V1 ensemble prediction
    # ---------------------------------------------------------
    def predict(
        self,
        raw_signals,
        fs=100.0,
        lead_names=None,
        age=None,
        sex=None,
        *,
        units="mV",
    ):
        """Run the existing V1-trained ensemble.

        Parameters
        ----------
        raw_signals:
            ECG matrix in its ORIGINAL channel order.

        fs:
            Sampling frequency in Hz.

        lead_names:
            Original channel names matching raw_signals.
            Explicit names are required.

        age, sex:
            Optional patient context for research rules.

        units:
            Original ECG voltage units, either one string
            or one unit per channel.

            Defaults to "mV" for legacy compatibility.
            New WFDB callers should pass the actual units.

        Notes
        -----
        This pathway uses V1-trained checkpoints.
        It does not apply V3 measurement-quality gates.
        """

        # -----------------------------------------------------
        # Step A: Validate and standardize the ECG.
        # -----------------------------------------------------
        # This is the key Milestone 2 correction.
        #
        # The validator:
        # - Checks lead names and uniqueness
        # - Reorders the actual waveform channels
        # - Converts physical units to millivolts
        # - Rejects malformed and non-finite samples
        # - Verifies the sampling frequency
        #
        # Both models will use this same standardized ECG.
        signals_mv, standard_leads = validate_ecg_input(
            raw_signals=raw_signals,
            fs=fs,
            lead_names=lead_names,
            units=units,
        )

        # -----------------------------------------------------
        # Step B: Enforce the legacy CNN input dimensions.
        # -----------------------------------------------------
        # Current CNN checkpoints expect 10-second,
        # 100 Hz, 12-lead ECG recordings.
        #
        # No implicit resampling, padding or truncation.
        if fs != 100.0 or signals_mv.shape != (1000, 12):
            raise ValueError(
                "Legacy ensemble requires fs=100 Hz "
                "and shape=(1000, 12). "
                f"Received fs={fs}, "
                f"shape={signals_mv.shape}"
            )

        # -----------------------------------------------------
        # Step C: Extract V1 clinical features.
        # -----------------------------------------------------
        feat_result = extract_clinical_features(
            signals_mv,
            fs=fs,
            lead_names=standard_leads,
            sqi_version="v1",
        )

        if feat_result["status"] == "rejected":
            return {
                "status": "rejected",
                "reason": feat_result["reason"],
                "sqi_details": feat_result.get(
                    "sqi_details"
                ),
            }

        if feat_result["status"] != "success":
            raise ValueError(
                "Unexpected feature extraction status: "
                f"{feat_result['status']}"
            )

        features = feat_result["features"]

        # -----------------------------------------------------
        # Step D: Collect trusted ST measurements.
        # -----------------------------------------------------
        trusted_leads = [
            lead.strip()
            for lead in features["leads_trusted"].split(",")
            if lead.strip()
        ]

        st_by_lead = {
            lead: float(features[f"st_{lead.upper()}"])
            for lead in trusted_leads
        }

        # -----------------------------------------------------
        # Step E: Run both models on the SAME recording.
        # -----------------------------------------------------
        # RF receives features extracted from signals_mv.
        # CNN receives signals_mv directly.
        #
        # This prevents disagreement caused solely by
        # different lead ordering or voltage units.
        p_rf = self._predict_rf(features)

        p_cnn = self._predict_cnn(
            signals_mv,
            fs,
        )

        # Preserve the existing 50/50 fusion policy.
        p_composite = 0.5 * p_rf + 0.5 * p_cnn

        risk_score_pct = round(
            p_composite * 100.0,
            1,
        )

        statistical_tier, statistical_action = (
            self._statistical_tier(risk_score_pct)
        )

        # -----------------------------------------------------
        # Step F: Summarize anatomical territories.
        # -----------------------------------------------------
        territories, primary_territory, culprit_artery = (
            self._analyze_territories(features)
        )

        # -----------------------------------------------------
        # Step G: Build the research inference result.
        # -----------------------------------------------------
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
                "max_st_elevation_mv": features[
                    "max_st_elevation_mv"
                ],
                "overall_sqi": features["overall_sqi"],
                "beats_used": features["beats_used"],
            },
        }

        # -----------------------------------------------------
        # Step H: Apply the existing research triage rules.
        # -----------------------------------------------------
        clinical_triage = self.triage_engine.evaluate(
            result
        )

        result["clinical_triage"] = clinical_triage
        result["risk_tier"] = clinical_triage["final_tier"]
        result["triage_action"] = clinical_triage[
            "triage_action"
        ]

        return result
