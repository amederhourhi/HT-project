"""
CardioSense - Calibrated Multi-Model Risk Fusion & Explainability
Combines the interpretable biomarker classifier and deep 1D-CNN into a unified
clinical risk score (0-100%) and localizes the culprit coronary artery territory.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import joblib
import numpy as np
import pandas as pd
import torch

from src.dsp.filters import filter_ecg
from src.features.extract_features import extract_clinical_features, FEATURE_COLUMNS
from src.models.ecg_cnn import CardioNet1D


class CardioSenseEnsemble:
    """
    Dual-engine ensemble risk fusion and anatomical coronary localizer.
    """

    def __init__(self, checkpoints_dir=None):
        if checkpoints_dir is None:
            checkpoints_dir = project_root / "models" / "checkpoints"
        else:
            checkpoints_dir = Path(checkpoints_dir)

        rf_path = checkpoints_dir / "interpretable_model.joblib"
        cnn_path = checkpoints_dir / "ecg_cnn.pt"

        if not rf_path.exists() or not cnn_path.exists():
            raise FileNotFoundError(f"Checkpoints missing in {checkpoints_dir}. Ensure models are trained.")

        # 1. Load Interpretable Random Forest (Platt calibrated)
        self.rf_model = joblib.load(rf_path)

        # 2. Load Deep 1D-CNN (CardioNet1D)
        self.device = torch.device("cpu")
        self.cnn_model = CardioNet1D(in_channels=12, num_classes=2).to(self.device)
        ckpt = torch.load(cnn_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.cnn_model.load_state_dict(state_dict)
        self.cnn_model.eval()

    def predict(self, raw_signals, fs=100.0, lead_names=None, age=None, sex=None):
        """
        Executes end-to-end evaluation for a 12-lead ECG.

        Args:
            age / sex: patient demographics when known. The Fourth UDMI ST
                cut-points depend on both for leads V2-V3, so the guardrails need
                them. `None` is an accepted value rather than an error: the engine
                then holds those leads to the most sensitive threshold and records
                that it did.

        Returns:
            dict with composite risk score, tier, engine probabilities, per-lead
            trusted ST measurements, patient demographics, coronary territory
            attribution, and key biomarkers.
        """
        if lead_names is None:
            lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        # 1. Feature Extraction & SQI Gate
        feat_result = extract_clinical_features(raw_signals, fs=fs, lead_names=lead_names)
        if feat_result["status"] == "rejected":
            return {
                "status": "rejected",
                "reason": feat_result["reason"],
                "sqi_details": feat_result["sqi_details"],
            }

        features = feat_result["features"]

        # Per-lead ST, restricted to leads the delineator vouches for. The guardrails
        # judge contiguity one lead at a time, so a lead that failed plausibility
        # checking has to be ABSENT from this map: the extractor writes 0.0 for it
        # (the classifier expects all twelve columns), and a 0.0 read as "no
        # deviation" would turn a measurement failure into a clinical judgement.
        trusted_leads = [lead for lead in features["leads_trusted"].split(",") if lead]
        st_by_lead = {lead: float(features[f"st_{lead.upper()}"]) for lead in trusted_leads}

        # 2. Engine A: Interpretable Classifier Probability
        feat_df = pd.DataFrame([features])[FEATURE_COLUMNS]
        p_rf = float(self.rf_model.predict_proba(feat_df)[0, 1])

        # 3. Engine B: Deep 1D-CNN Probability
        cleaned = filter_ecg(raw_signals, fs=fs)
        mean = np.mean(cleaned, axis=0, keepdims=True)
        std = np.std(cleaned, axis=0, keepdims=True) + 1e-6
        normalized = (cleaned - mean) / std

        wave_tensor = torch.tensor(normalized.T, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn_model(wave_tensor)
            p_cnn = float(torch.softmax(logits, dim=1)[0, 1].item())

        # 4. Calibrated Ensemble Fusion (Equal 50/50 Weighting)
        p_composite = 0.5 * p_rf + 0.5 * p_cnn
        risk_score_pct = round(p_composite * 100.0, 1)

        # 5. Risk Stratification Tier
        if p_composite < 0.35:
            risk_tier = "LOW_RISK"
            triage_action = "Routine Follow-up (Normal / Sub-acute)"
        elif p_composite < 0.65:
            risk_tier = "MODERATE_RISK"
            triage_action = "Clinician Review (Borderline / Intermediate Suspicion)"
        else:
            risk_tier = "HIGH_RISK"
            triage_action = "Urgent Emergency Escalation (Acute Coronary Syndrome Suspected)"

        # 6. Anatomical Coronary Artery Territory Attribution
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

        # Identify primary culprit vessel based on maximum regional ST shift
        primary_territory = max(territories.items(), key=lambda t: t[1]["max_st_mv"])
        culprit_artery = (
            primary_territory[1]["artery"]
            if primary_territory[1]["max_st_mv"] >= 0.10
            else "None (Sub-threshold ST deviation)"
        )

        return {
            "status": "success",
            "risk_score_pct": risk_score_pct,
            "risk_tier": risk_tier,
            "triage_action": triage_action,
            "engine_probabilities": {
                "interpretable_rf": round(p_rf, 3),
                "deep_cnn": round(p_cnn, 3),
                "fused_composite": round(p_composite, 3),
            },
            "culprit_artery": culprit_artery,
            "primary_ischemic_territory": primary_territory[0],
            "territory_breakdown": territories,
            # Guardrail inputs. `st_by_lead` carries only trusted leads; `patient`
            # may hold None values, which the engine resolves to its conservative
            # demographic fallback.
            "st_by_lead": st_by_lead,
            "patient": {"age": age, "sex": sex},
            "clinical_biomarkers": {
                "heart_rate_bpm": features["heart_rate_bpm"],
                "qrs_duration_ms": features["qrs_duration_ms"],
                "qrs_plausible": features["qrs_plausible"],
                "max_st_elevation_mv": features["max_st_elevation_mv"],
                "overall_sqi": features["overall_sqi"],
            },
        }