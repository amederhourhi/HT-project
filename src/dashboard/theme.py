"""
CardioSense - Triage Reader presentation layer

Two jobs, both purely visual:
  * inject `styles.css` into the page
  * map a triage tier or finding severity onto the headline and accent class
    the reader uses

The tier strings are imported from `ClinicalTriageEngine` rather than repeated
here. A view that invents its own tier names is how a second vocabulary starts,
and this project already has one of those to retire.
"""

from pathlib import Path

import streamlit as st

from src.triage.triage_rules import ESCALATE, REVIEW, ClinicalTriageEngine

CSS_FILE = Path(__file__).resolve().parent / "styles.css"

# tier -> (headline, accent class). The neutral class is the fallback for a
# tier the reader has never seen, which should be never, but not silently.
TIER_PRESENTATION = {
    ClinicalTriageEngine.TIER_ESCALATION: ("Tier 3 - Emergency escalation", "is-escalate"),
    ClinicalTriageEngine.TIER_REVIEW: ("Tier 2 - Clinician review", "is-review"),
    ClinicalTriageEngine.TIER_ROUTINE: ("Tier 1 - Routine follow-up", "is-routine"),
}

# severity -> chip class
SEVERITY_PRESENTATION = {
    ESCALATE: "sev-escalate",
    REVIEW: "sev-review",
}


@st.cache_data(show_spinner=False)
def _read_css() -> str:
    """Read the stylesheet once per server process, not once per rerun."""
    if not CSS_FILE.exists():
        return ""
    return CSS_FILE.read_text(encoding="utf-8")


def inject_css() -> None:
    """Push the stylesheet into the page. Called once, at the top of app.py."""
    st.markdown(f"<style>{_read_css()}</style>", unsafe_allow_html=True)


def tier_presentation(tier: str) -> tuple[str, str]:
    """Return the headline and accent class for a triage tier."""
    return TIER_PRESENTATION.get(tier, ("Triage tier unavailable", "is-neutral"))


def severity_class(severity: str) -> str:
    """Return the chip class for a finding severity."""
    return SEVERITY_PRESENTATION.get(severity, "rule")
