
"""
CardioSense - Versioned Feature Quality Contract.

Separates numerical clinical features from information about
their measurement quality, availability, and provenance.

This module does NOT:
    - change classifier feature values;
    - perform imputation;
    - validate a clinical diagnosis;
    - change triage routing;
    - prove that unflagged measurements are reliable.

It provides advisory metadata for future quality-aware inference.
"""

import math


# ------------------------------------------------------------
# Quality policy version
# ------------------------------------------------------------

QUALITY_POLICY_VERSION = "feature_quality_v2"


# ------------------------------------------------------------
# ECG lead definitions
# ------------------------------------------------------------

ALL_ST_LEADS = {
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
}


TERRITORIES = {
    "anterior": {"V1", "V2", "V3", "V4"},
    "inferior": {"II", "III", "AVF"},
    "lateral": {"I", "AVL", "V5", "V6"},
}


TERRITORY_FEATURES = {
    "st_anterior_max": TERRITORIES["anterior"],
    "st_inferior_max": TERRITORIES["inferior"],
    "st_lateral_max": TERRITORIES["lateral"],
}


GLOBAL_ST_FEATURES = {
    "max_st_elevation_mv",
    "max_st_depression_mv",
}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _normalize_leads(values):
    """Normalize a sequence of ECG lead names."""

    if values is None:
        return set()

    if isinstance(values, str):
        values = values.split(",")

    return {
        str(lead).strip().upper()
        for lead in values
        if str(lead).strip()
    }


def _get_trusted_leads(features):
    """Read the existing trusted-lead diagnostic field."""

    return _normalize_leads(
        features.get("leads_trusted", "")
    )


def _feature_leads_without_provenance(
    feature_name,
    trusted,
):
    """
    Legacy lead attribution.

    This is deliberately conservative because it cannot
    identify the specific lead that produced an extremum.
    """

    if feature_name in TERRITORY_FEATURES:

        territory = TERRITORY_FEATURES[feature_name]

        return (
            territory & trusted,
            bool(territory & trusted),
            "no_trusted_leads_in_territory",
        )

    if feature_name in GLOBAL_ST_FEATURES:

        leads = ALL_ST_LEADS & trusted

        return (
            leads,
            bool(leads),
            "no_trusted_st_leads",
        )

    if feature_name.startswith("st_"):

        lead = feature_name[3:].upper()

        if lead in ALL_ST_LEADS:

            return (
                {lead},
                lead in trusted,
                "lead_not_trusted_by_fiducial_analysis",
            )

    # Not an ST-derived feature.
    return set(), True, None


def _source_leads_from_provenance(
    feature_name,
    provenance,
):
    """
    Extract precise source-lead attribution when available.

    Returns None when the feature has no provenance record.
    """

    if (
        provenance is None
        or feature_name not in provenance
    ):
        return None

    record = provenance[feature_name]

    if not isinstance(record, dict):
        raise ValueError(
            f"Invalid provenance for {feature_name}"
        )

    available = record.get("available", False)

    if not isinstance(available, bool):
        raise ValueError(
            f"Invalid availability for {feature_name}"
        )

    leads = _normalize_leads(
        record.get("source_leads", [])
    )

    if not available:
        if leads:
            raise ValueError(
                f"Unavailable {feature_name} has source leads"
            )

        return set(), False

    if not leads:
        raise ValueError(
            f"Available {feature_name} has no source leads"
        )

    return leads, True


def _is_finite(value):
    """Check numeric availability without modifying the value."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


# ------------------------------------------------------------
# Public quality contract
# ------------------------------------------------------------

def build_feature_quality_contract(
    features,
    sqi_details,
    *,
    sqi_version,
    feature_columns,
    st_provenance=None,
    provenance_consistency=None,
):
    """
    Construct versioned feature-quality metadata.

    Parameters
    ----------
    features:
        Original classifier/diagnostic feature dictionary.

    sqi_details:
        Result returned by the selected SQI algorithm.

    sqi_version:
        "v1" or "v3".

    feature_columns:
        Frozen classifier input names.

    st_provenance:
        Optional mapping from ST feature name to its actual
        contributing lead(s).

    provenance_consistency:
        Optional mapping reporting whether an aggregate
        matches the value reconstructed from its source leads.

    Status meanings
    ---------------
    unavailable:
        No usable source measurement is established.

    review_required:
        A known quality concern affects the measurement, or
        the reported aggregate disagrees with its provenance.

    not_validated:
        No targeted issue was identified, but independent
        clinical measurement reliability is not established.

    Important
    ---------
    No classifier values are modified.

    For older callers without provenance, the contract retains
    conservative territory-wide quality attribution.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not isinstance(features, dict):
        raise TypeError(
            "features must be a dictionary"
        )

    if not isinstance(sqi_details, dict):
        raise TypeError(
            "sqi_details must be a dictionary"
        )

    if sqi_version not in ("v1", "v3"):
        raise ValueError(
            "Unsupported SQI version"
        )

    if (
        st_provenance is not None
        and not isinstance(st_provenance, dict)
    ):
        raise TypeError(
            "st_provenance must be a dictionary or None"
        )

    if (
        provenance_consistency is not None
        and not isinstance(provenance_consistency, dict)
    ):
        raise TypeError(
            "provenance_consistency must be a dictionary or None"
        )

    columns = list(feature_columns)

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Duplicate feature column names"
        )

    missing_columns = [
        name
        for name in columns
        if name not in features
    ]

    if missing_columns:
        raise ValueError(
            f"Missing classifier features: {missing_columns}"
        )

    # --------------------------------------------------------
    # Gather quality information
    # --------------------------------------------------------

    trusted = _get_trusted_leads(features)

    drift_leads = (
        _normalize_leads(
            sqi_details.get(
                "baseline_drift_review_leads", []
            )
        )
        if sqi_version == "v3"
        else set()
    )

    technical_failures = _normalize_leads(
        sqi_details.get(
            "technical_failure_leads", []
        )
    )

    # --------------------------------------------------------
    # Assess individual classifier features
    # --------------------------------------------------------

    feature_quality = {}

    for feature_name in columns:

        reasons = []
        affected_leads = set()
        available = True

        # ----------------------------------------------------
        # Determine measurement source
        # ----------------------------------------------------

        legacy_leads, legacy_available, missing_reason = (
            _feature_leads_without_provenance(
                feature_name,
                trusted,
            )
        )

        affected_leads = legacy_leads
        available = legacy_available

        is_st_feature = (
            feature_name in TERRITORY_FEATURES
            or feature_name in GLOBAL_ST_FEATURES
            or (
                feature_name.startswith("st_")
                and feature_name[3:].upper()
                in ALL_ST_LEADS
            )
        )

        provenance_used = False

        if is_st_feature:

            precise = _source_leads_from_provenance(
                feature_name,
                st_provenance,
            )

            if precise is not None:

                affected_leads, available = precise
                provenance_used = True

                # Source leads must also be recognized as
                # trusted by the fiducial analyzer.

                if available and not (
                    affected_leads <= trusted
                ):
                    available = False
                    reasons.append(
                        "provenance_source_not_trusted"
                    )

        # ----------------------------------------------------
        # Identify unavailable measurements
        # ----------------------------------------------------

        if not available:

            if provenance_used:
                if (
                    "provenance_source_not_trusted"
                    not in reasons
                ):
                    reasons.append(
                        "no_contributing_st_leads"
                    )

            elif missing_reason:
                reasons.append(missing_reason)

        # A numeric non-finite value is unavailable even
        # when source attribution exists.

        if (
            available
            and is_st_feature
            and not _is_finite(features[feature_name])
        ):
            available = False
            reasons.append(
                "nonfinite_st_feature_value"
            )

        # ----------------------------------------------------
        # Apply quality warnings
        # ----------------------------------------------------

        if available and is_st_feature:

            if sqi_version == "v1":

                # V1 has no targeted baseline-drift review
                # diagnostics, so avoid claiming assessment.
                reasons.append(
                    "st_quality_not_assessed_by_v1"
                )

            else:

                if affected_leads & drift_leads:

                    reasons.append(
                        "baseline_drift_in_relevant_leads"
                    )

                if affected_leads & technical_failures:

                    reasons.append(
                        "technical_failure_in_relevant_leads"
                    )

        # ----------------------------------------------------
        # Check provenance consistency
        # ----------------------------------------------------

        if (
            available
            and is_st_feature
            and provenance_consistency is not None
            and feature_name in provenance_consistency
        ):

            consistency = provenance_consistency[
                feature_name
            ]

            if not isinstance(consistency, dict):
                raise ValueError(
                    f"Invalid consistency data for {feature_name}"
                )

            if consistency.get("consistent") is False:

                reasons.append(
                    "aggregate_provenance_mismatch"
                )

        # ----------------------------------------------------
        # Determine final quality status
        # ----------------------------------------------------

        review_reasons = {
            "baseline_drift_in_relevant_leads",
            "technical_failure_in_relevant_leads",
            "aggregate_provenance_mismatch",
        }

        if not available:

            status = "unavailable"

        elif review_reasons.intersection(reasons):

            status = "review_required"

        else:

            status = "not_validated"

        feature_quality[feature_name] = {
            "status": status,
            "reasons": reasons,
            "affected_leads": sorted(
                affected_leads
            ),
        }

    # --------------------------------------------------------
    # Aggregate contract diagnostics
    # --------------------------------------------------------

    review_features = sorted(
        feature_name
        for feature_name, entry
        in feature_quality.items()
        if entry["status"] == "review_required"
    )

    unavailable_features = sorted(
        feature_name
        for feature_name, entry
        in feature_quality.items()
        if entry["status"] == "unavailable"
    )

    return {
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "sqi_version": sqi_version,
        "provenance_enabled": st_provenance is not None,
        "feature_quality": feature_quality,
        "review_required_features": review_features,
        "unavailable_features": unavailable_features,
        "has_measurement_quality_concerns": bool(
            review_features or unavailable_features
        ),
        "clinical_measurement_validation": (
            "not_established"
        ),
        "notes": [
            "Quality flags are experimental heuristics.",
            "Not validated does not mean reliable.",
            "Source leads identify contributors to ST features.",
            "Baseline drift may compromise ST measurement fidelity.",
            "An unavailable measurement may retain a historical numeric placeholder.",
            "Classifier feature values are not modified.",
            "Quality metadata does not yet control clinical triage.",
        ],
    }
