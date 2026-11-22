
"""
CardioSense — Ensemble Risk Fusion and Quality Assessment.

Two separate research pathways:

1. Legacy V1 inference:
   - Validate and standardize incoming ECG data
   - Extract V1 clinical features
   - Run Random Forest and 1D CNN
   - Combine probabilities
   - Preserve measurement-quality metadata
   - Apply research triage rules

2. Experimental V3 quality assessment:
   - Validate the WFDB recording
   - Extract V3 quality-aware features
   - Evaluate research readiness
   - Do NOT run unverified model checkpoints

Quality metadata is advisory. It does not modify model
probabilities or clinical triage routing.

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
from src.triage.inference_quality import (
    evaluate_research_readiness,
)
from src.triage.triage_rules import ClinicalTriageEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Preserve the original list-based interface.
DEFAULT_LEADS = list(STANDARD_LEADS)


class CardioSenseEnsemble:
    """
    Legacy V1 ensemble with a separate experimental
    V3 quality-assessment pathway.
    """

    # =====================================================
    # 1. Initialize trained models
    # =====================================================

    def __init__(self, checkpoints_dir=None):
        """Load existing V1 RF and CNN checkpoints."""

        self.checkpoints_dir = (
            Path(checkpoints_dir)
            if checkpoints_dir is not None
            else PROJECT_ROOT / "models" / "checkpoints"
        )

        rf_path = (
            self.checkpoints_dir
            / "interpretable_model.joblib"
        )

        cnn_path = (
            self.checkpoints_dir
            / "ecg_cnn.pt"
        )

        # Both checkpoints must exist.
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

        # Load only trusted project-generated artifacts.
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

        # Support both checkpoint layouts.
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
            else checkpoint
        )

        self.cnn_model.load_state_dict(state_dict)
        self.cnn_model.eval()

        self.triage_engine = ClinicalTriageEngine()

    # =====================================================
    # 2. Experimental V3 quality assessment
    # =====================================================

    @staticmethod
    def assess_v3_record(record_path):
        """
        Assess WFDB quality without model inference.

        A successful V3 assessment does not establish
        compatibility with V1-trained RF/CNN checkpoints.
        """

        # Validate waveform structure, lead ordering,
        # sampling rate, and physical voltage units.
        signals_mv, fs, lead_names = (
            load_standardized_ecg(record_path)
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
                "Unexpected feature extraction status: "
                f"{status}"
            )

        # Evaluate V3 technical and feature readiness.
        quality = evaluate_research_readiness(
            extraction
        )

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

            # Do not score using incompatible V1 models.
            outcome = "quality_eligible_unscored"

            reason = (
                "Quality gate passed, but V3 model "
                "compatibility has not been established"
            )

        return {
            "status": outcome,
            "reason": reason,
            "sqi_version": "v3",
            "quality_decision": quality,

            "sqi_details": extraction.get(
                "sqi_details"
            ),

            "quality_contract": extraction.get(
                "quality_contract"
            ),

            "model_compatibility": "not_verified",
            "model_scoring_performed": False,
            "risk_score_pct": None,
            "clinical_triage": None,
        }

    # =====================================================
    # 3. Probability validation
    # =====================================================

    @staticmethod
    def _validate_probability(value, model_name):
        """
        Require a finite model probability within [0, 1].
        """

        probability = float(value)

        if not np.isfinite(probability):
            raise ValueError(
                f"{model_name} returned a "
                "non-finite probability."
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{model_name} returned a probability "
                "outside [0, 1]."
            )

        return probability

    # =====================================================
    # 4. Random Forest inference
    # =====================================================

    def _predict_rf(self, features):
        """Predict using the existing V1 Random Forest."""

        # Preserve the exact feature order used in training.
        frame = pd.DataFrame([features])[FEATURE_COLUMNS]

        probability = self.rf_model.predict_proba(
            frame
        )[0, 1]

        return self._validate_probability(
            probability,
            "Random Forest",
        )

    # =====================================================
    # 5. CNN inference
    # =====================================================

    def _predict_cnn(self, signals_mv, fs):
        """
        Predict using a standardized 12-lead ECG.

        Expected:
            - Physical units: mV
            - Lead order: STANDARD_LEADS
            - Shape: (1000, 12)
            - Sampling frequency: 100 Hz
        """

        # Apply the existing ECG filtering policy.
        cleaned = filter_ecg(
            signals_mv,
            fs=fs,
        )

        # Per-lead normalization.
        mean = np.mean(
            cleaned,
            axis=0,
            keepdims=True,
        )

        std = (
            np.std(
                cleaned,
                axis=0,
                keepdims=True,
            )
            + 1e-6
        )

        normalized = (cleaned - mean) / std

        # Convert:
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

    # =====================================================
    # 6. Statistical risk classification
    # =====================================================

    @staticmethod
    def _statistical_tier(risk_score_pct):
        """
        Apply existing research probability thresholds.

        These are project-specific thresholds, not
        clinically validated intervention cutoffs.
        """

        if risk_score_pct < 35.0:
            return (
                "TIER_1_ROUTINE",
                "Routine Follow-up "
                "(Normal / Sub-acute findings)",
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

    # =====================================================
    # 7. Anatomical territory analysis
    # =====================================================

    @staticmethod
    def _analyze_territories(features):
        """Preserve existing territory aggregation."""

        territories = {
            "Anterior / Septal (LAD)": {
                "artery": (
                    "Left Anterior Descending (LAD)"
                ),
                "leads": ["V1", "V2", "V3", "V4"],
                "max_st_mv": features[
                    "st_anterior_max"
                ],
            },

            "Inferior (RCA)": {
                "artery": (
                    "Right Coronary Artery (RCA)"
                ),
                "leads": ["II", "III", "AVF"],
                "max_st_mv": features[
                    "st_inferior_max"
                ],
            },

            "Lateral (LCx)": {
                "artery": (
                    "Left Circumflex (LCx)"
                ),
                "leads": ["I", "AVL", "V5", "V6"],
                "max_st_mv": features[
                    "st_lateral_max"
                ],
            },
        }

        primary_name, primary_data = max(
            territories.items(),
            key=lambda item: item[1]["max_st_mv"],
        )

        # Preserve the historical threshold behavior.
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

    # =====================================================
    # 8. Legacy V1 ensemble prediction
    # =====================================================

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
        """
        Run the existing V1 RF/CNN ensemble.

        Parameters
        ----------
        raw_signals:
            Original ECG matrix (samples, channels).

        fs:
            Sampling frequency in Hz.

        lead_names:
            Names corresponding to original channels.

        age, sex:
            Optional demographic information.

        units:
            Original voltage units. May be one string
            or one unit per channel.

        Returns
        -------
        dict:
            Ensemble probabilities, clinical biomarkers,
            quality metadata, and research triage output.

        Notes
        -----
        This pathway uses V1-trained checkpoints.

        V3 quality gates are not applied to V1 inference.

        Forwarded quality metadata is advisory and does
        not modify inference or routing decisions.
        """

        # -------------------------------------------------
        # A. Validate and standardize incoming ECG
        # -------------------------------------------------

        signals_mv, standard_leads = (
            validate_ecg_input(
                raw_signals=raw_signals,
                fs=fs,
                lead_names=lead_names,
                units=units,
            )
        )

        # -------------------------------------------------
        # B. Enforce legacy model dimensions
        # -------------------------------------------------

        if (
            fs != 100.0
            or signals_mv.shape != (1000, 12)
        ):
            raise ValueError(
                "Legacy ensemble requires fs=100 Hz "
                "and shape=(1000, 12). "
                f"Received fs={fs}, "
                f"shape={signals_mv.shape}"
            )

        # -------------------------------------------------
        # C. Extract legacy V1 clinical features
        # -------------------------------------------------

        feat_result = extract_clinical_features(
            signals_mv,
            fs=fs,
            lead_names=standard_leads,
            sqi_version="v1",
        )

        # Do not score rejected recordings.
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

        # -------------------------------------------------
        # D. Collect trusted ST measurements
        # -------------------------------------------------

        trusted_leads = [
            lead.strip()
            for lead in features["leads_trusted"].split(",")
            if lead.strip()
        ]

        # Keep only ST readings from trusted leads.
        #
        # Untrusted leads are omitted rather than
        # represented by fabricated zero measurements.
        st_by_lead = {
            lead: float(
                features[f"st_{lead.upper()}"]
            )
            for lead in trusted_leads
        }

        # -------------------------------------------------
        # E. Run the two existing models
        # -------------------------------------------------

        # Both models use the same standardized ECG.
        p_rf = self._predict_rf(features)

        p_cnn = self._predict_cnn(
            signals_mv,
            fs,
        )

        # Preserve historical 50/50 fusion.
        p_composite = (
            0.5 * p_rf
            + 0.5 * p_cnn
        )

        risk_score_pct = round(
            p_composite * 100.0,
            1,
        )

        statistical_tier, statistical_action = (
            self._statistical_tier(
                risk_score_pct
            )
        )

        # -------------------------------------------------
        # F. Analyze anatomical territories
        # -------------------------------------------------

        (
            territories,
            primary_territory,
            culprit_artery,
        ) = self._analyze_territories(features)

        # -------------------------------------------------
        # G. Assemble the ensemble result
        # -------------------------------------------------

        result = {
            "status": "success",

            # Statistical prediction.
            "risk_score_pct": risk_score_pct,
            "statistical_tier": statistical_tier,
            "risk_tier": statistical_tier,
            "triage_action": statistical_action,

            # Preserve original engine probabilities.
            "engine_probabilities": {
                "interpretable_rf": round(
                    p_rf, 3
                ),
                "deep_cnn": round(
                    p_cnn, 3
                ),
                "fused_composite": round(
                    p_composite, 3
                ),
            },

            # Anatomical interpretation.
            "culprit_artery": culprit_artery,

            "primary_ischemic_territory": (
                primary_territory
            ),

            "territory_breakdown": territories,

            # Trusted ST measurements.
            "st_by_lead": st_by_lead,

            # =================================================
            # MILESTONE 4: Measurement-quality metadata handoff
            # =================================================
            #
            # These fields are produced during V1 feature
            # extraction but were previously discarded.
            #
            # Forwarding them allows the existing triage
            # quality-handoff module to inspect concerns.
            #
            # Missing metadata remains None.
            #
            # IMPORTANT:
            # These fields DO NOT:
            #   - Modify RF classifier features
            #   - Modify CNN input
            #   - Modify model probabilities
            #   - Change clinical guardrail thresholds
            #   - Change final triage routing
            #   - Establish clinical measurement validity

            "quality_contract": feat_result.get(
                "quality_contract"
            ),

            "st_provenance": feat_result.get(
                "st_provenance"
            ),

            "provenance_consistency": feat_result.get(
                "provenance_consistency"
            ),

            "sqi_details": feat_result.get(
                "sqi_details"
            ),

            # Patient demographics.
            "patient": {
                "age": age,
                "sex": sex,
            },

            # Original clinical biomarkers.
            "clinical_biomarkers": {
                "heart_rate_bpm": features[
                    "heart_rate_bpm"
                ],

                "qrs_duration_ms": features[
                    "qrs_duration_ms"
                ],

                "qrs_plausible": features[
                    "qrs_plausible"
                ],

                "max_st_elevation_mv": features[
                    "max_st_elevation_mv"
                ],

                "overall_sqi": features[
                    "overall_sqi"
                ],

                "beats_used": features[
                    "beats_used"
                ],
            },
        }

        # -------------------------------------------------
        # H. Apply the existing clinical triage engine
        # -------------------------------------------------

        # The engine now sees quality_contract and
        # st_provenance when those are available.
        #
        # Its quality assessment is metadata-only.
        # Existing findings and routing are preserved.
        clinical_triage = (
            self.triage_engine.evaluate(result)
        )

        result["clinical_triage"] = clinical_triage

        # Preserve existing final output fields.
        result["risk_tier"] = (
            clinical_triage["final_tier"]
        )

        result["triage_action"] = (
            clinical_triage["triage_action"]
        )

        return result
