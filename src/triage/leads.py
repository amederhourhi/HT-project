"""
CardioSense - ECG Lead Geometry and Guideline ST Cut-Points

Pure reference knowledge: no signal processing, no model, no file I/O. Everything
the triage rules need to know about where the leads sit and how high an ST
deviation has to be before it counts as acute injury lives here, so the rule
engine stays a short readable evaluator instead of a wall of cardiology numbers.

Provenance: Fourth Universal Definition of Myocardial Infarction (Thygesen et
al., 2018) for the cut-points, AHA/ACC/HRS standardisation of the ECG for the
contiguous-lead grouping.
"""

import math
from typing import NamedTuple, Sequence


# ---------------------------------------------------------
# Block 1: Lead geometry
# ---------------------------------------------------------

# Canonical spelling used by every artefact in this project. PTB-XL returns
# 'aVR'/'aVL'/'aVF' in a record's sig_name; the feature CSV columns and the
# territory keys both upper-case them. Normalising in one place means the
# guardrails never silently miss a lead because of a spelling difference.
LEADS: tuple[str, ...] = (
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

# Coronary territories, keyed with the exact strings the ensemble payload and the
# SBAR report already use, so attribution cannot drift from the rule that fired.
TERRITORIES: dict[str, tuple[str, ...]] = {
    "Anterior / Septal (LAD)": ("V1", "V2", "V3", "V4"),
    "Inferior (RCA)": ("II", "III", "AVF"),
    "Lateral (LCx)": ("I", "AVL", "V5", "V6"),
}

# Contiguous lead pairs. The definition of ST elevation requires TWO contiguous
# leads; a single lead crossing the cut-point is not a criterion. A pair is
# satisfied only when each lead clears its OWN cut-point, which is why V2 (high
# take-off is common and physiological) needs the demographic thresholds below
# while V1 does not.
CONTIGUOUS_PAIRS: tuple[tuple[str, str], ...] = (
    ("V1", "V2"), ("V2", "V3"), ("V3", "V4"),      # anterior / septal chain
    ("V4", "V5"), ("V5", "V6"),                    # lateral precordial chain
    ("I", "AVL"),                                  # high lateral
    ("II", "AVF"), ("III", "AVF"), ("II", "III"),  # inferior
)


# ---------------------------------------------------------
# Block 2: Guideline cut-points
# ---------------------------------------------------------

# ST elevation magnitudes in mV (1.0 mV = 10 mm at standard calibration).
STEMI_GENERIC_THRESH_MV = 0.10     # every lead except V2-V3
V23_WOMEN_THRESH_MV = 0.15         # women, any age
V23_MEN_40_PLUS_THRESH_MV = 0.20   # men aged >= 40
V23_MEN_UNDER_40_THRESH_MV = 0.25  # men aged < 40 (benign early repolarization)

# V2 and V3 are the only leads whose criterion depends on the patient.
DEMOGRAPHIC_LEADS: frozenset[str] = frozenset({"V2", "V3"})
AGE_ADULT_MALE_CUTOFF_YEARS = 40.0

MALE = "MALE"
FEMALE = "FEMALE"

# Applied when age or sex is missing for a demographic lead. Choosing the
# *lowest* published value means we never relax a threshold just because the
# paperwork is absent: the error we allow is an extra review, the error we
# refuse is a discharge. Findings that fall back record that they did.
UNKNOWN_DEMOGRAPHIC_THRESH_MV = V23_WOMEN_THRESH_MV


def normalise_sex(raw: object) -> str | None:
    """
    Map whatever the caller has to MALE / FEMALE, or None when unknown.

    PTB-XL stores sex as 0.0 (male) / 1.0 (female) and pandas reads that column
    back as float, with NaN for records the download stage could not attribute.
    Accepting the source encodings here keeps the CSV convention out of the rule
    engine.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().upper()
        if text.startswith("M"):
            return MALE
        if text.startswith("F"):
            return FEMALE
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if math.isnan(value):
            return None
        if value == 0.0:
            return MALE
        if value == 1.0:
            return FEMALE
        # Anything else is not a sex code. Treat it as unknown rather than
        # guessing, so the more sensitive demographic fallback applies.
        return None
    return None


def normalise_age(raw: object) -> float | None:
    """
    Return age in years, or None when it is missing or implausible.

    The download stage already NaNs out-of-window ages (PTB-XL carries a record
    with age = 300), so here we only need to recognise that None/NaN means
    unknown.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or value <= 0 or value > 110:
        return None
    return value


def cut_point_mv(lead: str, age: float | None, sex: str | None) -> tuple[float, str]:
    """
    Guideline threshold for ST elevation in this lead, with its provenance.

    Returns:
        (threshold_mv, basis). `basis` is a short human-readable clause naming
        which line of the definition was applied. It travels into the finding so
        a clinician reading the report can see the number the decision was made
        against, not only the decision.
    """
    lead = lead.upper()

    if lead not in DEMOGRAPHIC_LEADS:
        return STEMI_GENERIC_THRESH_MV, f"generic >= {STEMI_GENERIC_THRESH_MV:.2f} mV"

    if sex == MALE and age is not None:
        if age < AGE_ADULT_MALE_CUTOFF_YEARS:
            return (
                V23_MEN_UNDER_40_THRESH_MV,
                f"men < {AGE_ADULT_MALE_CUTOFF_YEARS:.0f} y in {lead}: "
                f">= {V23_MEN_UNDER_40_THRESH_MV:.2f} mV",
            )
        return (
            V23_MEN_40_PLUS_THRESH_MV,
            f"men >= {AGE_ADULT_MALE_CUTOFF_YEARS:.0f} y in {lead}: "
            f">= {V23_MEN_40_PLUS_THRESH_MV:.2f} mV",
        )

    if sex == FEMALE:
        return (
            V23_WOMEN_THRESH_MV,
            f"women in {lead}: >= {V23_WOMEN_THRESH_MV:.2f} mV",
        )

    # Demographics unusable for a threshold that depends on them.
    return (
        UNKNOWN_DEMOGRAPHIC_THRESH_MV,
        f"age/sex unknown: {lead} held to the most sensitive cut-point "
        f">= {UNKNOWN_DEMOGRAPHIC_THRESH_MV:.2f} mV",
    )


def territory_of(pair: Sequence[str]) -> str:
    """
    Name the territory a satisfied pair belongs to, for culprit attribution.

    V4 sits in both the anterior and lateral chains, so the answer is the first
    territory containing both leads. Ambiguity is acceptable in a report label;
    the measurement that drove it is carried alongside it.
    """
    pair_set = {lead.upper() for lead in pair}
    for name, leads in TERRITORIES.items():
        if pair_set.issubset(set(leads)):
            return name
    return "Unmapped"


class StePair(NamedTuple):
    """One contiguous pair meeting the ST elevation criterion, with evidence."""

    lead_a: str
    lead_b: str
    st_a_mv: float
    st_b_mv: float
    cut_a_mv: float
    cut_b_mv: float
    territory: str
    basis_a: str
    basis_b: str

    @property
    def max_st_mv(self) -> float:
        """Larger of the two deviations - what to rank culprit pairs by."""
        return max(self.st_a_mv, self.st_b_mv)

    def description(self) -> str:
        """Clinician-facing sentence naming lead, value and threshold."""
        return (
            f"{self.lead_a} {self.st_a_mv:+.3f} mV / {self.lead_b} "
            f"{self.st_b_mv:+.3f} mV (cut-points {self.cut_a_mv:.2f} / "
            f"{self.cut_b_mv:.2f} mV) - {self.territory}"
        )


def find_stemi_pairs(
    st_by_lead: dict[str, float],
    age: float | None,
    sex: str | None,
) -> list[StePair]:
    """
    Return every contiguous pair satisfying the STEMI criterion.

    Args:
        st_by_lead: ST deviation at the J point per lead, in mV. Must already be
            restricted to leads the delineator vouched for - a lead whose QRS
            boundary or baseline estimate failed has to be absent from this dict,
            not present with a placeholder value. Contiguity is judged on
            single-lead numbers, so a bad lead can no longer be outvoted by a
            territory maximum; leaving it in would let an artefact escalate.
        age: years, or None when unknown.
        sex: MALE / FEMALE from `normalise_sex`, or None when unknown.

    Returns:
        Matching pairs, most-deviated first. An empty list means the criterion is
        not met anywhere - not that the ECG is normal.
    """
    trusted = {lead.upper(): value for lead, value in st_by_lead.items()}
    hits: list[StePair] = []

    for lead_a, lead_b in CONTIGUOUS_PAIRS:
        # Both leads must be present and measured. A missing lead is unknown,
        # and unknown never satisfies a criterion.
        if lead_a not in trusted or lead_b not in trusted:
            continue

        cut_a, basis_a = cut_point_mv(lead_a, age, sex)
        cut_b, basis_b = cut_point_mv(lead_b, age, sex)

        st_a = float(trusted[lead_a])
        st_b = float(trusted[lead_b])

        if st_a >= cut_a and st_b >= cut_b:
            hits.append(
                StePair(
                    lead_a=lead_a,
                    lead_b=lead_b,
                    st_a_mv=st_a,
                    st_b_mv=st_b,
                    cut_a_mv=cut_a,
                    cut_b_mv=cut_b,
                    territory=territory_of((lead_a, lead_b)),
                    basis_a=basis_a,
                    basis_b=basis_b,
                )
            )

    hits.sort(key=lambda hit: hit.max_st_mv, reverse=True)
    return hits
