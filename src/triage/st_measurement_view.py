
"""
CardioSense - ST Measurement Presentation Contract.

Converts legacy ensemble ST results into a separate,
quality-aware presentation structure.

The input ensemble may include historical numerical
placeholders used by its frozen Random Forest schema.

These placeholders must not be represented as
genuine measured ECG values in API responses.

Rules:
    - Read individual ST values from st_by_lead.
    - Exclude absent or nonfinite readings.
    - Exclude explicitly unavailable features.
    - Exclude explicitly unavailable provenance sources.
    - Preserve genuine numeric zero measurements.
    - Retain review-required readings with warnings.
    - Calculate territory extrema from reportable
      readings, never legacy classifier aggregates.
    - Use None for unavailable summaries.
    - Never claim independent clinical validation.

This module does not change model scoring,
feature extraction or clinical triage decisions.
"""

import math

from src.triage.leads import LEADS, TERRITORIES


VIEW_VERSION = "st_measurement_view_v1"


def _finite_number(value):
    """Return a finite real-number representation or None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) else None


def _maximum_summary(readings, leads):
    """Build an extremum using only available readings."""

    selected = {
        lead: readings[lead]
        for lead in leads
        if lead in readings
    }

    if not selected:
        return {
            "available": False,
            "max_st_mv": None,
            "source_leads": [],
        }

    maximum = max(selected.values())

    sources = [
        lead
        for lead, value in selected.items()
        if math.isclose(
            value,
            maximum,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ]

    return {
        "available": True,
        "max_st_mv": maximum,
        "source_leads": sources,
    }


def build_st_measurement_view(ensemble_result):
    """Return JSON-compatible ST measurements for presentation.

    Reads the legacy ensemble without mutating it.

    Missing or contradictory availability metadata
    is handled conservatively.

    'review_required' measurements remain visible
    with their review status; they are not treated
    as independently validated observations.
    """

    if not isinstance(ensemble_result, dict):
        raise TypeError(
            "ensemble_result must be a dictionary"
        )

    # -----------------------------------------------------
    # 1. Read original measurements and metadata.
    # -----------------------------------------------------

    supplied = ensemble_result.get("st_by_lead") or {}

    contract = (
        ensemble_result.get("quality_contract") or {}
    )

    provenance = (
        ensemble_result.get("st_provenance") or {}
    )

    consistency = (
        ensemble_result.get(
            "provenance_consistency"
        ) or {}
    )

    if not isinstance(supplied, dict):
        raise TypeError(
            "st_by_lead must be a dictionary"
        )

    quality_by_feature = (
        contract.get("feature_quality") or {}
    )

    # -----------------------------------------------------
    # 2. Assess every standard ECG lead.
    # -----------------------------------------------------

    reportable = {}
    per_lead = {}

    for lead in LEADS:

        feature_name = f"st_{lead}"

        number = _finite_number(
            supplied.get(lead)
        )

        feature_quality = quality_by_feature.get(
            feature_name
        )

        source = provenance.get(feature_name)

        if not isinstance(feature_quality, dict):
            feature_quality = {}

        if not isinstance(source, dict):
            source = {}

        quality_status = feature_quality.get(
            "status",
            "unknown",
        )

        reasons = list(
            feature_quality.get("reasons") or []
        )

        # -------------------------------------------------
        # Determine whether a reading is reportable.
        # -------------------------------------------------

        available = number is not None

        if number is None:
            reasons.append(
                "missing_or_nonfinite_st_reading"
            )

        if quality_status == "unavailable":
            available = False

        if source.get("available") is False:
            available = False
            reasons.append(
                "st_provenance_unavailable"
            )

        if not available:
            quality_status = "unavailable"
            number = None
        else:
            reportable[lead] = number

        per_lead[lead] = {
            "available": available,
            "st_mv": number,
            "quality_status": quality_status,
            "reasons": list(dict.fromkeys(reasons)),
        }

    # -----------------------------------------------------
    # 3. Calculate a report-only global peak.
    # -----------------------------------------------------

    peak = _maximum_summary(
        reportable,
        LEADS,
    )

    # -----------------------------------------------------
    # 4. Calculate territory-specific measurements.
    # -----------------------------------------------------

    territories = {}

    for territory_name, territory_leads in (
        TERRITORIES.items()
    ):

        summary = _maximum_summary(
            reportable,
            territory_leads,
        )

        count = sum(
            lead in reportable
            for lead in territory_leads
        )

        total = len(territory_leads)

        if count == 0:
            coverage = "unavailable"
        elif count == total:
            coverage = "complete"
        else:
            coverage = "partial"

        territories[territory_name] = {
            **summary,
            "coverage": coverage,
            "available_leads": count,
            "total_leads": total,
        }

    # -----------------------------------------------------
    # 5. Identify aggregate-provenance warnings.
    # -----------------------------------------------------

    aggregate_concerns = []

    for name, item in consistency.items():

        if not isinstance(item, dict):
            continue

        if item.get("consistent") is False:

            aggregate_concerns.append({
                "feature": name,
                "reason": item.get(
                    "reason",
                    "unspecified",
                ),
            })

    # -----------------------------------------------------
    # 6. Return the presentation contract.
    # -----------------------------------------------------

    return {
        "version": VIEW_VERSION,
        "purpose": "research_measurement_presentation",
        "clinical_validation": "not_established",
        "routing_modified": False,

        # These are the 12 possible displayed leads,
        # including nulls for unavailable measurements.
        "per_lead": per_lead,

        "available_lead_count": len(reportable),

        "missing_leads": [
            lead
            for lead in LEADS
            if lead not in reportable
        ],

        "peak_st": peak,
        "territories": territories,

        "aggregate_provenance_concerns": (
            aggregate_concerns
        ),

        "legacy_classifier_aggregates_used": False,

        "notes": [
            "Values are algorithmic estimates.",
            "Availability does not establish accuracy.",
            "Review-required values are not independently validated.",
            "Historical classifier placeholders are excluded.",
            "The original model and triage results are unchanged.",
        ],
    }
