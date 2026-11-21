
"""Advisory quality assessment for CardioSense clinical triage.

The assessment is metadata-only. It never changes model probabilities,
guardrail findings, final triage tiers, or emergency escalation decisions.

No warning is not the same as independent clinical validation.
"""

from src.triage.leads import (
    find_stemi_pairs,
    normalise_age,
    normalise_sex,
)


QUALITY_HANDOFF_VERSION = "triage_quality_handoff_v1"


def summarize_quality_handoff(payload, findings):
    """Assess measurement concerns relevant to clinical ST findings.

    The contiguous-lead rule is evaluated using the same input readings
    and demographic normalization as the clinical triage engine.

    This is NOT a second clinical-rule decision engine: it only
    identifies which already-detected ST pairs may have quality concerns.
    """

    contract = payload.get("quality_contract")
    provenance = payload.get("st_provenance")
    has_contract = isinstance(contract, dict)

    result = {
        "version": QUALITY_HANDOFF_VERSION,
        "status": "assessed" if has_contract else "not_available",
        "quality_policy_version": (
            contract.get("quality_policy_version")
            if has_contract else None
        ),
        "measurement_validation": "not_established",
        "routing_modified": False,
        "review_required_features": (
            list(contract.get("review_required_features", []))
            if has_contract else []
        ),
        "unavailable_features": (
            list(contract.get("unavailable_features", []))
            if has_contract else []
        ),
        "st_finding_assessments": [],
        "st_finding_quality_concern": False,
        "st_provenance_available": isinstance(provenance, dict),
    }

    if not has_contract:
        return result

    feature_quality = contract.get("feature_quality") or {}

    # Evaluate only measurement concerns associated with ST rules
    # that were actually triggered by the clinical engine.
    has_st_finding = any(
        finding.rule_id == "G1_CONTIGUOUS_STE"
        for finding in findings
    )

    if not has_st_finding:
        return result

    patient = payload.get("patient") or {}
    age = normalise_age(patient.get("age"))
    sex = normalise_sex(patient.get("sex"))

    pairs = find_stemi_pairs(
        payload["st_by_lead"], age, sex
    )

    for pair in pairs:
        lead_assessments = {}

        for lead in (pair.lead_a, pair.lead_b):
            feature_name = f"st_{lead.upper()}"
            quality = feature_quality.get(feature_name)

            if not isinstance(quality, dict):
                status = "unknown"
                reasons = ["feature_quality_missing"]
            else:
                status = quality.get("status", "unknown")
                reasons = list(quality.get("reasons", []))

            lead_assessments[lead] = {
                "feature": feature_name,
                "status": status,
                "reasons": reasons,
            }

        # "not_validated" is not an assurance of reliability.
        # Only explicit problems or missing metadata are flagged
        # as known concerns at this stage.
        flagged = [
            lead
            for lead, quality in lead_assessments.items()
            if quality["status"] in (
                "review_required",
                "unavailable",
                "unknown",
            )
        ]

        result["st_finding_assessments"].append({
            "rule_id": "G1_CONTIGUOUS_STE",
            "territory": pair.territory,
            "leads": [pair.lead_a, pair.lead_b],
            "lead_quality": lead_assessments,
            "known_quality_concern": bool(flagged),
            "flagged_leads": flagged,
        })

    result["st_finding_quality_concern"] = any(
        item["known_quality_concern"]
        for item in result["st_finding_assessments"]
    )

    return result
