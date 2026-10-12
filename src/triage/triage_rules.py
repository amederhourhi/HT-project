"""
CardioSense - Deterministic Clinical Safety Guardrails & Triage Engine
Applies strict evidence-based cardiology rules (AHA/ACC STEMI criteria,
extreme rhythm anomalies, conduction delays) to override statistical models
and prevent false-negative discharge errors.
"""


class ClinicalTriageEngine:
    """
    Deterministic safety guardrail gatekeeper that evaluates electrophysiological
    findings and assigns a patient to an immutable clinical triage tier.
    """

    # Clinical safety thresholds (AHA/ACC & ESC Fourth Universal Definition of MI)
    STEMI_PRECORDIAL_THRESH_MV = 0.200  # V1-V4: >= 0.20 mV (2.0 mm)
    STEMI_LIMB_THRESH_MV = 0.100        # II, III, aVF, I, aVL: >= 0.10 mV (1.0 mm)
    SEVERE_TACHYCARDIA_BPM = 130.0
    SEVERE_BRADYCARDIA_BPM = 40.0
    WIDE_QRS_THRESH_MS = 130.0

    def evaluate(self, ensemble_result):
        """
        Evaluates ensemble output against deterministic clinical safety rules.
        Returns:
            dict containing finalized triage tier, recommended action,
            effective risk score, and list of triggered safety overrides.
        """
        biomarkers = ensemble_result["clinical_biomarkers"]
        territories = ensemble_result["territory_breakdown"]

        ant_st = territories["Anterior / Septal (LAD)"]["max_st_mv"]
        inf_st = territories["Inferior (RCA)"]["max_st_mv"]
        lat_st = territories["Lateral (LCx)"]["max_st_mv"]

        overrides = []

        # -----------------------------------------------------
        # Guardrail 1: Hard STEMI Transmural Injury Criteria
        # -----------------------------------------------------
        if ant_st >= self.STEMI_PRECORDIAL_THRESH_MV:
            overrides.append(
                f"AHA/ACC STEMI Criteria Met: Anterior ST elevation is {ant_st:+.3f} mV "
                f"(threshold: >= +{self.STEMI_PRECORDIAL_THRESH_MV:.3f} mV). High suspicion of acute LAD occlusion."
            )

        if inf_st >= self.STEMI_LIMB_THRESH_MV:
            overrides.append(
                f"AHA/ACC STEMI Criteria Met: Inferior ST elevation is {inf_st:+.3f} mV "
                f"(threshold: >= +{self.STEMI_LIMB_THRESH_MV:.3f} mV). High suspicion of acute RCA occlusion."
            )

        if lat_st >= self.STEMI_LIMB_THRESH_MV:
            overrides.append(
                f"AHA/ACC STEMI Criteria Met: Lateral ST elevation is {lat_st:+.3f} mV "
                f"(threshold: >= +{self.STEMI_LIMB_THRESH_MV:.3f} mV). High suspicion of acute LCx occlusion."
            )

        # -----------------------------------------------------
        # Guardrail 2: Critical Hemodynamic / Rhythm Instability
        # -----------------------------------------------------
        hr = biomarkers["heart_rate_bpm"]
        if hr > self.SEVERE_TACHYCARDIA_BPM:
            overrides.append(
                f"Severe Tachycardia Alert: Heart rate is {hr:.1f} BPM "
                f"(exceeds physiological emergency threshold: > {self.SEVERE_TACHYCARDIA_BPM:.0f} BPM)."
            )
        elif hr < self.SEVERE_BRADYCARDIA_BPM:
            overrides.append(
                f"Severe Bradycardia Alert: Heart rate is {hr:.1f} BPM "
                f"(falls below emergency threshold: < {self.SEVERE_BRADYCARDIA_BPM:.0f} BPM)."
            )

        # -----------------------------------------------------
        # Guardrail 3: Severe Conduction Defect / Wide QRS
        # -----------------------------------------------------
        qrs = biomarkers["qrs_duration_ms"]
        if qrs >= self.WIDE_QRS_THRESH_MS:
            overrides.append(
                f"Marked Conduction Delay: QRS duration is {qrs:.1f} ms "
                f"(>= {self.WIDE_QRS_THRESH_MS:.0f} ms). Suspected Bundle Branch Block masking repolarization."
            )

        # -----------------------------------------------------
        # Final Triage Assignment
        # -----------------------------------------------------
        base_risk = ensemble_result["risk_score_pct"]

        if len(overrides) > 0:
            # Hard safety override triggered: Escalate to Tier 3 regardless of statistical probability
            final_tier = "TIER_3_ESCALATION"
            triage_action = "Immediate Emergency Escalation -- Stat Cardiology / Cath Lab Activation"
            effective_risk = max(base_risk, 85.0)
            override_active = True
        else:
            override_active = False
            effective_risk = base_risk
            if base_risk < 35.0:
                final_tier = "TIER_1_ROUTINE"
                triage_action = "Routine Follow-up (Normal / Sub-acute findings)"
            elif base_risk < 65.0:
                final_tier = "TIER_2_REVIEW"
                triage_action = "Urgent Clinician Review (Intermediate Risk / Borderline findings)"
            else:
                final_tier = "TIER_3_ESCALATION"
                triage_action = "Immediate Emergency Escalation (High Risk Infarction Suspected)"

        return {
            "final_tier": final_tier,
            "triage_action": triage_action,
            "effective_risk_pct": effective_risk,
            "baseline_ensemble_risk_pct": base_risk,
            "override_active": override_active,
            "override_reasons": overrides,
        }