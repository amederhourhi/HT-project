"""
CardioSense - Deterministic Clinical Safety Guardrails & Triage Engine
Applies strict evidence-based cardiology rules (AHA/ACC STEMI criteria, extreme
rhythm anomalies, conduction delays) to override statistical models and prevent
false-negative discharge errors.

Two deliberate design choices define this file:

1. The ST thresholds are not here. They live in `src/triage/leads.py`, next to the
   guideline text they come from, so a reviewer can check a number against its
   source without reading evaluation logic. This module decides what a measurement
   *means*; `leads.py` records what the definition *says*.

2. Findings carry a severity. "A rule fired" is not the same question as "the cath
   lab should activate", and collapsing both into one boolean is how a screening
   tool ends up escalating healthy patients. ESCALATE is a demand for emergency
   care; REVIEW means the tracing cannot safely be read as normal.
"""

from typing import NamedTuple

from src.triage.leads import find_stemi_pairs, normalise_age, normalise_sex

# Severity of a triggered guardrail. The two values are not interchangeable, and
# the tier assignment below is the only place their meaning is defined.
ESCALATE = "ESCALATE"  # immediate emergency escalation (cath-lab activation)
REVIEW = "REVIEW"      # a clinician must read this tracing before it is dismissed
NO_FINDING = "NONE"


class Finding(NamedTuple):
    """
    One triggered guardrail, with the measurement that triggered it.

    `rule_id` is a stable identifier, not prose: the cohort audit counts fires per
    rule, a report can cite its own source, and rewording an explanatory sentence
    must never silently change what the evaluation thinks a rule is.
    """

    rule_id: str
    severity: str
    detail: str
    territory: str = ""

    def as_reason(self) -> str:
        """Clinician-facing line: rule, severity, then the measurement."""
        return f"[{self.rule_id} ({self.severity})] {self.detail}"


class ClinicalTriageEngine:
    """
    Deterministic safety guardrail gatekeeper that evaluates electrophysiological
    findings and assigns a patient to an immutable clinical triage tier.
    """

    # ---------------------------------------------------------
    # Thresholds owned by this module
    # ---------------------------------------------------------
    # ST elevation cut-points are NOT listed here - see leads.py. These three
    # govern rhythm and conduction, which the guidelines treat as states that are
    # emergencies in themselves, not as measurements needing demographic limits.
    SEVERE_TACHYCARDIA_BPM = 130.0
    SEVERE_BRADYCARDIA_BPM = 40.0
    WIDE_QRS_THRESH_MS = 130.0

    # Tier names are part of the payload contract shared with the dashboard, the
    # SBAR report and the cohort scorer. Defined once, referenced everywhere.
    TIER_ROUTINE = "TIER_1_ROUTINE"
    TIER_REVIEW = "TIER_2_REVIEW"
    TIER_ESCALATION = "TIER_3_ESCALATION"

    # Statistical banding of the ensemble score, in percent.
    ROUTINE_MAX_RISK_PCT = 35.0
    REVIEW_MAX_RISK_PCT = 65.0

    # Floor applied when an ESCALATE finding overrides a low statistical score.
    # The hard override of the *tier* is the defensible part; this number is the
    # cosmetic part, and it is scheduled for removal in the reporting cleanup.
    ESCALATION_RISK_FLOOR_PCT = 85.0

    def evaluate(self, ensemble_result: dict) -> dict:
        """
        Evaluate one ensemble output against the deterministic safety rules.

        Args:
            ensemble_result: payload from `CardioSenseEnsemble.predict()`. The
                guardrails read `risk_score_pct`, `clinical_biomarkers`,
                `st_by_lead` and `patient`. `st_by_lead` must contain ONLY leads
                the delineator vouched for: an unmeasured lead has to be absent
                rather than present as 0.0, because contiguity is judged lead by
                lead and a placeholder would silently fail the criterion instead
                of being reported as unknown.

        Returns:
            dict with the finalized tier, recommended action, effective risk, the
            structured findings and their prose rendering.
        """
        biomarkers = ensemble_result["clinical_biomarkers"]
        st_by_lead = ensemble_result["st_by_lead"]
        patient = ensemble_result.get("patient") or {}

        # Demographics are normalised at this boundary. The cohort CSV carries
        # PTB-XL's 0.0/1.0 sex codes and NaN for unusable ages; the dashboard holds
        # strings. Either way, unknown resolves to None and the cut-point helper
        # applies the most sensitive threshold instead of relaxing one.
        age = normalise_age(patient.get("age"))
        sex = normalise_sex(patient.get("sex"))

        findings = []
        findings.extend(self._check_st_elevation(st_by_lead, age, sex))
        findings.extend(self._check_rhythm(biomarkers))
        findings.extend(self._check_conduction(biomarkers))

        return self._assign_tier(ensemble_result["risk_score_pct"], findings)

    # ---------------------------------------------------------
    # Guardrail 1: contiguous-lead ST elevation
    # ---------------------------------------------------------
    def _check_st_elevation(
        self, st_by_lead: dict[str, float], age: float | None, sex: str | None
    ) -> list[Finding]:
        """
        Fourth UDMI ST elevation: TWO contiguous leads, each past its own cut-point.

        The previous version of this rule compared one regional maximum per
        territory against a single threshold, which is not the criterion. It
        escalated on a lone high-taking-off V2 - common in healthy young men - and
        it applied 0.20 mV across the precordium where the definition asks for
        0.10 mV outside V2-V3, so it was simultaneously too eager and too blind.
        """
        findings = []
        for pair in find_stemi_pairs(st_by_lead, age, sex):
            culprit = pair.territory.rsplit("(", 1)[-1].rstrip(")")
            findings.append(
                Finding(
                    rule_id="G1_CONTIGUOUS_STE",
                    severity=ESCALATE,
                    territory=pair.territory,
                    detail=(
                        f"STEMI criteria met in contiguous leads {pair.lead_a}/{pair.lead_b}: "
                        f"{pair.st_a_mv:+.3f} mV and {pair.st_b_mv:+.3f} mV "
                        f"(cut-points: {pair.basis_a}; {pair.basis_b}). "
                        f"High suspicion of acute {culprit} occlusion."
                    ),
                )
            )
        return findings

    # ---------------------------------------------------------
    # Guardrail 2: critical rhythm instability
    # ---------------------------------------------------------
    def _check_rhythm(self, biomarkers: dict) -> list[Finding]:
        """
        Extreme rate is a haemodynamic emergency independent of morphology.

        Sustained tachycardia above ~130 bpm compromises diastolic filling and
        raises supply/demand mismatch (type-2 ischaemia); a rate below ~40 bpm
        implies high-grade block or an escape rhythm. Neither requires the ST
        segment to be abnormal in order to matter.
        """
        hr = biomarkers["heart_rate_bpm"]

        if hr > self.SEVERE_TACHYCARDIA_BPM:
            return [
                Finding(
                    rule_id="G2_SEVERE_TACHYCARDIA",
                    severity=ESCALATE,
                    detail=(
                        f"Heart rate {hr:.1f} bpm exceeds the emergency threshold "
                        f"(> {self.SEVERE_TACHYCARDIA_BPM:.0f} bpm): impaired diastolic "
                        f"filling and demand-ischaemia risk."
                    ),
                )
            ]
        if hr < self.SEVERE_BRADYCARDIA_BPM:
            return [
                Finding(
                    rule_id="G2_SEVERE_BRADYCARDIA",
                    severity=ESCALATE,
                    detail=(
                        f"Heart rate {hr:.1f} bpm falls below the emergency threshold "
                        f"(< {self.SEVERE_BRADYCARDIA_BPM:.0f} bpm): suspected high-grade "
                        f"AV block or escape rhythm."
                    ),
                )
            ]
        return []

    # ---------------------------------------------------------
    # Guardrail 3: conduction delay
    # ---------------------------------------------------------
    def _check_conduction(self, biomarkers: dict) -> list[Finding]:
        """
        Wide QRS means the repolarisation analysis cannot be trusted. It does not
        by itself indicate coronary occlusion, which is why this rule reports to
        REVIEW and not to ESCALATE.

        Bundle branch block both masks ST elevation and produces secondary ST-T
        changes that mimic it, so a guideline cut-point applied to a blocked QRS is
        not measuring what the definition assumed. Escalating every blocked tracing
        to the cath lab would spend the false-positive budget on exactly the
        patients whose ST segment is least interpretable; leaving them in routine
        follow-up would discharge them. A clinician reads these before anyone does
        either.
        """
        qrs = biomarkers["qrs_duration_ms"]
        if qrs < self.WIDE_QRS_THRESH_MS:
            return []

        # The plausibility band is the delineator's own opinion of its measurement.
        # Reported inside the finding rather than suppressing it: a genuinely 250 ms
        # QRS is a real conduction defect, and an artefactual one still deserves
        # human eyes. Severity is REVIEW either way, so neither case activates a
        # cath lab.
        caveat = ""
        if not biomarkers.get("qrs_plausible", True):
            caveat = (
                " The measured duration also falls outside the plausible conduction "
                "band, so delineation is uncertain."
            )

        return [
            Finding(
                rule_id="G3_WIDE_QRS",
                severity=REVIEW,
                detail=(
                    f"QRS duration {qrs:.1f} ms (>= {self.WIDE_QRS_THRESH_MS:.0f} ms): "
                    f"suspected bundle branch block. ST criteria are unreliable in this "
                    f"configuration and require manual interpretation.{caveat}"
                ),
            )
        ]

    # ---------------------------------------------------------
    # Tier assignment
    # ---------------------------------------------------------
    def _assign_tier(self, base_risk: float, findings: list[Finding]) -> dict:
        """
        Fold guardrail findings and statistical risk into one tier.

        Precedence is safety-ordered: any ESCALATE finding wins outright; otherwise
        a REVIEW finding lifts the patient out of routine follow-up but not into
        emergency care; otherwise the tier is purely the statistical banding.
        """
        severities = {finding.severity for finding in findings}

        if ESCALATE in severities:
            final_tier = self.TIER_ESCALATION
            triage_action = "Immediate Emergency Escalation -- Stat Cardiology / Cath Lab Activation"
            # The floor exists so a rule-driven escalation does not print as a 4%
            # risk. It is presentation, not evidence; the tier is the decision.
            effective_risk = max(base_risk, self.ESCALATION_RISK_FLOOR_PCT)
        elif REVIEW in severities:
            effective_risk = base_risk
            if base_risk >= self.REVIEW_MAX_RISK_PCT:
                # A review finding never argues a high statistical score downward.
                final_tier = self.TIER_ESCALATION
                triage_action = "Immediate Emergency Escalation (High Risk Infarction Suspected)"
            elif base_risk >= self.ROUTINE_MAX_RISK_PCT:
                final_tier = self.TIER_REVIEW
                triage_action = "Urgent Clinician Review (Intermediate Risk / Borderline findings)"
            else:
                final_tier = self.TIER_REVIEW
                triage_action = (
                    "Mandatory Clinician Review - a guardrail finding makes automatic "
                    "discharge unsafe (ST interpretation unreliable)"
                )
        else:
            effective_risk = base_risk
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
            # True when any deterministic rule fired, at either severity. Consumers
            # that mean "sent to emergency care" must read final_tier instead.
            "override_active": bool(findings),
            "highest_severity": (
                ESCALATE if ESCALATE in severities
                else REVIEW if REVIEW in severities
                else NO_FINDING
            ),
            # Structured findings for machine consumers (audits, reports, tests).
            "findings": [finding._asdict() for finding in findings],
            # Prose rendering for humans, prefixed with the rule id so a sentence in
            # a report can be traced back to the rule that wrote it.
            "override_reasons": [finding.as_reason() for finding in findings],
        }
