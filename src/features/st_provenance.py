
"""Trace ST-derived features to their contributing ECG leads.

Metadata only: this module does not modify classifier values.
"""

import math

LEADS = (
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

GROUPS = {
    "st_anterior_max": ("V1", "V2", "V3", "V4"),
    "st_inferior_max": ("II", "III", "AVF"),
    "st_lateral_max": ("I", "AVL", "V5", "V6"),
}


def build_st_provenance(st_by_lead, trusted_leads):
    """Identify every lead attaining an ST aggregate's extremum.

    Ties retain all contributing leads. Unavailable aggregates have
    no source leads rather than a fabricated zero-valued source.
    """

    trusted = {str(x).upper() for x in trusted_leads}

    readings = {}
    for lead, value in st_by_lead.items():
        name = str(lead).upper()

        if name not in LEADS or name not in trusted:
            continue

        if value is None:
            continue

        number = float(value)
        if math.isfinite(number):
            readings[name] = number

    result = {}

    def add_extremum(feature_name, candidates, operation):
        selected = {
            lead: readings[lead]
            for lead in candidates
            if lead in readings
        }

        if not selected:
            result[feature_name] = {
                "available": False,
                "source_leads": [],
                "value_mv": None,
            }
            return

        extreme = (
            max(selected.values())
            if operation == "max"
            else min(selected.values())
        )

        sources = sorted(
            lead for lead, value in selected.items()
            if math.isclose(value, extreme, abs_tol=1e-9)
        )

        result[feature_name] = {
            "available": True,
            "source_leads": sources,
            "value_mv": round(extreme, 3),
        }

    for lead in LEADS:
        result[f"st_{lead}"] = {
            "available": lead in readings,
            "source_leads": [lead] if lead in readings else [],
            "value_mv": readings.get(lead),
        }

    for name, group in GROUPS.items():
        add_extremum(name, group, "max")

    add_extremum(
        "max_st_elevation_mv", LEADS, "max"
    )
    add_extremum(
        "max_st_depression_mv", LEADS, "min"
    )

    return result
