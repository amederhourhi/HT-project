"""
CardioSense - Deterministic Clinical Safety Guardrails & Triage Engine
Applies strict evidence-based cardiology rules (AHA/ACC STEMI criteria, extreme
rhythm anomalies, conduction delays) to override statistical models and prevent
false-negative discharge errors.
"""

from collections import defaultdict
from typing import NamedTuple

from src.triage.leads import find_stemi_pairs, normalise_age, normalise_sex

# ---------------------------------------------------------
# Block 1: Finding Severity Constants & Data Structure
# What it does: Defines finding severities and the structured Finding type.
# How it does it: Uses immutable string constants and NamedTuple for typing.
# ---------------------------------------------------------
ESCALATE = "ESCALATE"  # Immediate emergency escalation (cath lab activation)
REVIEW = "REVIEW"      # Clinician must manually review before dismissal
NO_FINDING = "NONE"


class Finding(NamedTuple):
    """One triggered guardrail with associated measurement and territory."""
    rule_id: str
    severity: str
    detail: str
    territory: str = ""

    def as_reason(self) -> str:
        """Formatted line for clinician-facing reports."""
        return f"[{self.rule_id} ({self.severity})] {self.detail}"


class ClinicalTriageEngine:
    """
    Deterministic safety guardrail gatekeeper that evaluates electrophysiological
    findings and assigns a patient to an immutable clinical triage tier.
    """

    # ---------------------------------------------------------
    # Block 2: Clinical Thresholds & Tier Constants
    # What it does: Declares rhythm limits, tier strings, and probability bands.
    # How it does it: Uses class-level constants shared across the system.
    # ---------------------------------------------------------
    SEVERE_TACHYCARDIA_BPM = 130.0
    SEVERE_BRADYCARDIA_BPM = 40.0
    WIDE_QRS_THRESH_MS = 130.0

    TIER_ROUTINE = "TIER_1_ROUTINE"
    TIER_REVIEW = "TIER_2_REVIEW"
    TIER_ESCALATION = "TIER_3_ESCALATION"

    ROUTINE_MAX_RISK_PCT = 35.0
    REVIEW_MAX_RISK_PCT = 65.0

    # ---------------------------------------------------------
    # Block 3: Primary Evaluation Pipeline
    # What it does: Normalizes demographics and runs all safety checks.
    # How it does it: Calls ST elevation, rhythm, and conduction evaluators.
    # ---------------------------------------------------------
    def evaluate(self, ensemble_result: dict) -> dict:
        biomarkers = ensemble_result["clinical_biomarkers"]
        st_by_lead = ensemble_result["st_by_lead"]
        patient = ensemble_result.get("patient") or {}

        age = normalise_age(patient.get("age"))
        sex = normalise_sex(patient.get("sex"))

        findings = []
        findings.extend(self._check_st_elevation(st_by_lead, age, sex))
        findings.extend(self._check_rhythm(biomarkers))
        findings.extend(self._check_conduction(biomarkers))

        return self._assign_tier(ensemble_result["risk_score_pct"], findings)

    # ---------------------------------------------------------
    # Block 4: Guardrail 1 - Contiguous-Lead ST Elevation (Grouped)
    # What it does: Identifies contiguous pairs clearing Fourth UDMI cut-points.
    # How it does it: Groups hits by territory to prevent duplicate report lines.
    # ---------------------------------------------------------
    def _check_st_elevation(
        self, st_by_lead: dict[str, float], age: float | None, sex: str | None
    ) -> list[Finding]:
        stemi_pairs = find_stemi_pairs(st_by_lead, age, sex)
        if not stemi_pairs:
            return []

        # Group satisfied pairs by coronary territory
        by_territory = defaultdict(list)
        for pair in stemi_pairs:
            by_territory[pair.territory].append(pair)

        findings = []
        for territory, pairs in by_territory.items():
            culprit = territory.rsplit("(", 1)[-1].rstrip(")")
            pair_summaries = [
                f"{p.lead_a}/{p.lead_b} ({p.st_a_mv:+.3f}/{p.st_b_mv:+.3f} mV)"
                for p in pairs
            ]
            pairs_str = ", ".join(pair_summaries)

            findings.append(
                Finding(
                    rule_id="G1_CONTIGUOUS_STE",
                    severity=ESCALATE,
                    territory=territory,
                    detail=(
                        f"STEMI criteria met in contiguous leads: {pairs_str}. "
                        f"High suspicion of acute {culprit} occlusion."
                    ),
                )
            )
        return findings

    # ---------------------------------------------------------
    # Block 5: Guardrail 2 - Critical Rhythm Instability
    # What it does: Evaluates extreme tachycardia and bradycardia.
    # How it does it: Checks resting rate against hemodynamic emergency limits.
    # ---------------------------------------------------------
    def _check_rhythm(self, biomarkers: dict) -> list[Finding]:
        hr = biomarkers["heart_rate_bpm"]

        if hr > self.SEVERE_TACHYCARDIA_BPM:
            return [
                Finding(
                    rule_id="G2_SEVERE_TACHYCARDIA",
                    severity=ESCALATE,
                    detail=(
                        f"Heart rate {hr:.1f} bpm exceeds emergency threshold "
                        f"(> {self.SEVERE_TACHYCARDIA_BPM:.0f} bpm): demand ischemia risk."
                    ),
                )
            ]
        if hr < self.SEVERE_BRADYCARDIA_BPM:
            return [
                Finding(
                    rule_id="G2_SEVERE_BRADYCARDIA",
                    severity=ESCALATE,
                    detail=(
                        f"Heart rate {hr:.1f} bpm falls below emergency threshold "
                        f"(< {self.SEVERE_BRADYCARDIA_BPM:.0f} bpm): suspected high-grade AV block."
                    ),
                )
            ]
        return []

    # ---------------------------------------------------------
    # Block 6: Guardrail 3 - Conduction Delay / Wide QRS
    # What it does: Flags bundle branch blocks as requiring manual clinician review.
    # How it does it: Checks QRS duration >= 130 ms and emits REVIEW severity.
    # ---------------------------------------------------------
    def _check_conduction(self, biomarkers: dict) -> list[Finding]:
        qrs = biomarkers["qrs_duration_ms"]
        if qrs < self.WIDE_QRS_THRESH_MS:
            return []

        caveat = ""
        if not biomarkers.get("qrs_plausible", True):
            caveat = " Measured duration falls outside plausible band."

        return [
            Finding(
                rule_id="G3_WIDE_QRS",
                severity=REVIEW,
                detail=(
                    f"QRS duration {qrs:.1f} ms (>= {self.WIDE_QRS_THRESH_MS:.0f} ms): "
                    f"suspected bundle branch block. ST analysis requires manual interpretation.{caveat}"
                ),
            )
        ]

    # ---------------------------------------------------------
    # Block 7: Tier Assignment & Output Assembly
    # What it does: Maps findings and statistical risk to final triage tier.
    # How it does it: Safety-ordered precedence (ESCALATE -> REVIEW -> statistical score).
    #                 Honest probability reporting without cosmetic floors.
    # ---------------------------------------------------------
    def _assign_tier(self, base_risk: float, findings: list[Finding]) -> dict:
        severities = {finding.severity for finding in findings}
        effective_risk = base_risk

        if ESCALATE in severities:
            final_tier = self.TIER_ESCALATION
            triage_action = "Immediate Emergency Escalation -- Stat Cardiology / Cath Lab Activation"
        elif REVIEW in severities:
            if base_risk >= self.REVIEW_MAX_RISK_PCT:
                final_tier = self.TIER_ESCALATION
                triage_action = "Immediate Emergency Escalation (High Risk Infarction Suspected)"
            else:
                final_tier = self.TIER_REVIEW
                triage_action = "Mandatory Clinician Review (Conduction defect or intermediate risk)"
        else:
            if base_risk < self.ROUTINE_MAX_RISK_PCT:
                final_tier = self.TIER_ROUTINE
                triage_action = "Routine Follow-up (Normal / Sub-acute findings)"
            elif base_risk < self.REVIEW_MAX_RISK_PCT:
                final_tier = self.TIER_REVIEW
                triage_action = "Urgent Clinician Review (Intermediate Risk / Borderline findings)"
            else:
                final_tier = self.TIER_ESCALATION
                triage_action = "Immediate Emergency Escalation (High Risk Infarction Suspected)"

        return {
            "final_tier": final_tier,
            "triage_action": triage_action,
            "effective_risk_pct": effective_risk,
            "baseline_ensemble_risk_pct": base_risk,
            "override_active": bool(findings),
            "highest_severity": (
                ESCALATE if ESCALATE in severities
                else REVIEW if REVIEW in severities
                else NO_FINDING
            ),
            "findings": [finding._asdict() for finding in findings],
            "override_reasons": [finding.as_reason() for finding in findings],
        }