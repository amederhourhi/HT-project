"""
CardioSense - Theme & Style Loader
Loads styles.css and injects dynamic CSS variables for dark/light modes
and full-screen emergency alert atmospheres.
"""

from pathlib import Path
import streamlit as st


def apply_theme_and_atmosphere(theme: str, tier: str, is_results_page: bool = False):
    """Loads external styles.css and sets dynamic color variables."""
    css_file = Path(__file__).resolve().parent / "styles.css"
    if not css_file.exists():
        return

    css_content = css_file.read_text(encoding="utf-8")
    is_dark = theme == "dark"

    # Default palettes
    bg_main = "#0B0F19" if is_dark else "#F8FAFC"
    text_main = "#F8FAFC" if is_dark else "#0F172A"
    card_bg = "#161E2E" if is_dark else "#FFFFFF"
    card_border = "#2A374A" if is_dark else "#E2E8F0"

    # Full screen color takeover on results page
    if is_results_page:
        if tier == "TIER_3_ESCALATION":
            bg_main = "#450A0A" if is_dark else "#FEF2F2"
            card_bg = "#7F1D1D" if is_dark else "#FEE2E2"
            card_border = "#DC2626"
            text_main = "#FEE2E2" if is_dark else "#7F1D1D"
        elif tier == "TIER_1_ROUTINE":
            bg_main = "#052E16" if is_dark else "#F0FDF4"
            card_bg = "#064E3B" if is_dark else "#DCFCE7"
            card_border = "#16A34A"
            text_main = "#DCFCE7" if is_dark else "#064E3B"
        else:
            bg_main = "#1E293B" if is_dark else "#F1F5F9"
            card_bg = "#334155" if is_dark else "#E2E8F0"
            card_border = "#64748B"
            text_main = "#F1F5F9" if is_dark else "#1E293B"

    # Inject variables and external stylesheet
    st.markdown(
        f"""
        <style>
        :root {{
            --bg-main: {bg_main};
            --text-main: {text_main};
            --card-bg: {card_bg};
            --card-border: {card_border};
        }}
        {css_content}
        </style>
        """,
        unsafe_allow_html=True,
    )