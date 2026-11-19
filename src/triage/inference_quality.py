
"""Quality gate for experimental CardioSense research inference.

An eligible result may be used in a separately validated research
pipeline. Eligibility does not establish clinical reliability.
"""

from math import isfinite
from numbers import Real

from src.features.extract_features import FEATURE_COLUMNS
from src.features.quality_contract import QUALITY_POLICY_VERSION


V3_ALGORITHM = "experimental_sqi_v3_separated"

VALID_STATUSES = {
    "not_validated",
    "review_required",
    "unavailable",
}


def _decision(allowed, reason, review=None, unavailable=None):
    """Build a consistent decision for downstream consumers."""
    return {
        "decision": "eligible" if allowed else "abstain",
        "allow_research_scoring": allowed,
        "reason": reason,
        "review_required_features": review or [],
        "unavailable_features": unavailable or [],
        "clinical_validation": "not_established",
    }


def evaluate_research_readiness(extraction):
    """Check whether extracted ECG features are eligible for research scoring.

    Fail closed when SQI V3, feature values, or quality metadata
    are missing, inconsistent, or indicate a quality concern.
    """

    # Require a completed feature extraction.
    if not isinstance(extraction, dict):
        return _decision(False, "Invalid extraction result")

    if extraction.get("status") != "success":
        return _decision(False, "Feature extraction did not succeed")

    features = extraction.get("features")
    sqi = extraction.get("sqi_details")
    contract = extraction.get("quality_contract")

    if not all(isinstance(item, dict) for item in (features, sqi, contract)):
        return _decision(False, "Missing features or quality metadata")

    # Require the exact SQI and feature-quality policy versions.
    if (
        sqi.get("algorithm") != V3_ALGORITHM
        or contract.get("sqi_version") != "v3"
        or contract.get("quality_policy_version") != QUALITY_POLICY_VERSION
    ):
        return _decision(False, "Unsupported quality policy version")

    # SQI must confirm technical readiness.
    if (
        sqi.get("is_acceptable") is not True
        or sqi.get("ready_for_automated_inference") is not True
    ):
        return _decision(False, "SQI V3 did not establish technical readiness")

    failures = sqi.get("technical_failure_leads")

    if not isinstance(failures, list):
        return _decision(False, "Missing technical failure assessment")

    if failures:
        return _decision(False, "Technical signal failures detected")

    # All 23 classifier inputs must exist and contain finite numbers.
    for name in FEATURE_COLUMNS:
        value = features.get(name)

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
        ):
            return _decision(False, f"Invalid classifier feature: {name}")

    # Validate the complete per-feature quality contract.
    quality = contract.get("feature_quality")

    if not isinstance(quality, dict):
        return _decision(False, "Missing per-feature quality assessments")

    if set(quality) != set(FEATURE_COLUMNS):
        return _decision(False, "Incomplete feature-quality contract")

    review = []
    unavailable = []

    for name in FEATURE_COLUMNS:
        item = quality[name]

        if not isinstance(item, dict):
            return _decision(False, f"Malformed quality assessment: {name}")

        status = item.get("status")

        if status not in VALID_STATUSES:
            return _decision(False, f"Unknown quality status: {name}")

        if status == "review_required":
            review.append(name)
        elif status == "unavailable":
            unavailable.append(name)

    review.sort()
    unavailable.sort()

    # Verify the contract summaries against the individual features.
    if (
        contract.get("review_required_features") != review
        or contract.get("unavailable_features") != unavailable
    ):
        return _decision(False, "Inconsistent feature-quality summaries")

    # SQI V3 can permit technical processing while still requesting
    # manual review for possible morphology or baseline changes.
    if (
        not isinstance(sqi.get("requires_review"), bool)
        or not isinstance(sqi.get("st_measurement_review_required"), bool)
    ):
        return _decision(False, "Incomplete SQI V3 review metadata")

    if (
        sqi["requires_review"]
        or sqi["st_measurement_review_required"]
        or review
        or unavailable
    ):
        return _decision(
            False,
            "Automated scoring withheld pending measurement review",
            review,
            unavailable,
        )

    return _decision(
        True,
        "No flagged quality concerns; research scoring only",
    )
