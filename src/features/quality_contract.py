
"""Versioned feature-quality metadata for CardioSense.

This module does not modify classifier inputs, invalidate clinical
measurements, or claim that a measurement is clinically validated.

It records known quality concerns separately from numeric features.
"""

QUALITY_POLICY_VERSION = "feature_quality_v1"

TERRITORIES = {
    "anterior": {"V1", "V2", "V3", "V4"},
    "inferior": {"II", "III", "AVF"},
    "lateral": {"I", "AVL", "V5", "V6"},
}

GLOBAL_ST_FEATURES = {
    "max_st_elevation_mv",
    "max_st_depression_mv",
}

TERRITORY_FEATURES = {
    "st_anterior_max": TERRITORIES["anterior"],
    "st_inferior_max": TERRITORIES["inferior"],
    "st_lateral_max": TERRITORIES["lateral"],
}


def build_feature_quality_contract(
    features,
    sqi_details,
    *,
    sqi_version,
    feature_columns,
):
    """Describe measurement-quality concerns without altering features.

    Status meanings:
      review_required: a known quality concern affects the feature
      not_validated: no targeted concern, but reliability isn't proven
      unavailable: the required measurement is not available

    V1 has no V3 baseline-drift diagnostics, so its ST features
    remain unassessed rather than being called reliable.
    """

    if not isinstance(features, dict):
        raise TypeError("features must be a dictionary")

    if not isinstance(sqi_details, dict):
        raise TypeError("sqi_details must be a dictionary")

    if sqi_version not in ("v1", "v3"):
        raise ValueError("Unsupported SQI version")

    columns = list(feature_columns)

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate feature column names")

    missing_columns = [
        name for name in columns
        if name not in features
    ]
    if missing_columns:
        raise ValueError(
            f"Missing classifier features: {missing_columns}"
        )

    trusted = set(
        lead.strip().upper()
        for lead in features.get(
            "leads_trusted", ""
        ).split(",")
        if lead.strip()
    )

    drift_leads = (
        set(
            lead.upper()
            for lead in sqi_details.get(
                "baseline_drift_review_leads", []
            )
        )
        if sqi_version == "v3"
        else set()
    )

    technical_failures = set(
        lead.upper()
        for lead in sqi_details.get(
            "technical_failure_leads", []
        )
    )

    # A review warning does not prove a measurement is invalid.
    # Conversely, no warning does not establish clinical validity.
    result = {}

    all_st_leads = {
        "I", "II", "III", "AVR", "AVL", "AVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    }

    for feature_name in columns:
        reasons = []
        affected_leads = set()
        unavailable = False

        if feature_name.startswith("st_") and (
            feature_name not in TERRITORY_FEATURES
        ):
            lead = feature_name[3:].upper()

            if lead in all_st_leads:
                affected_leads = {lead}

                if lead not in trusted:
                    unavailable = True
                    reasons.append(
                        "lead_not_trusted_by_fiducial_analysis"
                    )

        elif feature_name in TERRITORY_FEATURES:
            territory = TERRITORY_FEATURES[feature_name]
            affected_leads = territory & trusted

            if not affected_leads:
                unavailable = True
                reasons.append(
                    "no_trusted_leads_in_territory"
                )

        elif feature_name in GLOBAL_ST_FEATURES:
            affected_leads = all_st_leads & trusted

            if not affected_leads:
                unavailable = True
                reasons.append(
                    "no_trusted_st_leads"
                )

        # Review only features whose ST measurements may depend
        # on leads showing baseline-drift warnings.
        if not unavailable and affected_leads:
            if sqi_version == "v1":
                reasons.append(
                    "st_quality_not_assessed_by_v1"
                )
            else:
                flagged = affected_leads & drift_leads
                if flagged:
                    reasons.append(
                        "baseline_drift_in_relevant_leads"
                    )

                failed = affected_leads & technical_failures
                if failed:
                    reasons.append(
                        "technical_failure_in_relevant_leads"
                    )

        if unavailable:
            status = "unavailable"
        elif (
            "baseline_drift_in_relevant_leads" in reasons
            or "technical_failure_in_relevant_leads" in reasons
        ):
            status = "review_required"
        else:
            status = "not_validated"

        result[feature_name] = {
            "status": status,
            "reasons": reasons,
            "affected_leads": sorted(
                affected_leads
            ),
        }

    review_features = sorted(
        key for key, item in result.items()
        if item["status"] == "review_required"
    )

    unavailable_features = sorted(
        key for key, item in result.items()
        if item["status"] == "unavailable"
    )

    return {
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "sqi_version": sqi_version,
        "feature_quality": result,
        "review_required_features": review_features,
        "unavailable_features": unavailable_features,
        "has_measurement_quality_concerns": bool(
            review_features or unavailable_features
        ),
        "clinical_measurement_validation": "not_established",
        "notes": [
            "Quality flags are experimental heuristics.",
            "Not validated does not mean reliable.",
            "Baseline drift can affect ST measurement fidelity.",
            "Classifier feature values are not modified.",
        ],
    }
