"""
CardioSense - Clinical Triage Reader

One screen: choose a record, read the disposition, read the evidence that
produced it, read the tracing, copy the note.

What this replaced
------------------
The previous interface was a two-card launch hub, a light/dark toggle, a
full-screen colour wash driven by the triage tier, an animated red banner, and a
case library whose tabs asserted "Anterior STEMI (LAD)" over records that do not
meet the STEMI criterion. Its headline metric was a "Suspected Culprit" derived
from the maximum regional ST shift, which is exactly the single-lead reasoning
the guardrail sprint removed. None of it survived the rebuild.

What is here instead is what the old UI never showed: the per-lead ST
measurements the decision is built on, the cut-point each lead is held to, and
which contiguous pairs satisfied the definition. Those numbers come from
`src.triage.leads` - the same module the rule engine calls - so the page cannot
drift from the rule.

Ground-truth labels appear in the record picker on purpose. This is an
engineering and demonstration tool, not a blind reading station.
"""

import html
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import wfdb

from src.dashboard import theme
from src.dsp.filters import filter_ecg
from src.features.fiducials import delineate_beat_landmarks
from src.features.r_peaks import detect_r_peaks
from src.triage.leads import (
    CONTIGUOUS_PAIRS,
    LEADS,
    cut_point_mv,
    find_stemi_pairs,
    normalise_age,
    normalise_sex,
)
from src.triage.report_generator import generate_clinical_report
from src.triage.risk_fusion import CardioSenseEnsemble
from src.triage.triage_rules import ClinicalTriageEngine

st.set_page_config(
    page_title="CardioSense | Triage Reader",
    layout="wide",
    initial_sidebar_state="collapsed",
)

theme.inject_css()

# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------

COHORT_DIR = project_root / "data" / "cohort"
MANIFEST_CSV = COHORT_DIR / "cohort_manifest.csv"
FEATURES_CSV = project_root / "data" / "processed" / "clinical_features.csv"
CHECKPOINT_DIR = project_root / "models" / "checkpoints"

# The two tracked sample records, kept so the reader can be demonstrated on a
# record with no manifest entry. They carry no demographics, which makes them a
# live exercise of the unknown-demographic fallback rather than an accident.
SAMPLE_RECORDS = (
    ("sample 00001_lr", "data/samples/records100/00000/00001_lr"),
    ("sample 00008_lr", "data/samples/records100/00000/00008_lr"),
)


@st.cache_resource
def load_pipeline():
    """Load both engines once per server process; they are read-only afterwards."""
    return CardioSenseEnsemble(), ClinicalTriageEngine()


@st.cache_data(show_spinner=False)
def load_catalogue() -> list[dict]:
    """
    Build the record picker from the cohort manifest.

    Neither the manifest nor the waveforms are tracked - `data/cohort/` is
    ignored as a whole, and both are produced by
    `src/data/download_ptbxl.py --cohort`. A record is listed only when its
    WFDB files are on disk alongside the manifest, because a half-materialised
    cohort is the normal state on a machine where that download was
    interrupted, and a picker full of records that fail to open is worse than a
    short one.
    """
    records: list[dict] = []

    if MANIFEST_CSV.exists():
        manifest = pd.read_csv(MANIFEST_CSV)
        for row in manifest.itertuples(index=False):
            base = COHORT_DIR / row.filename_lr
            if not (base.with_suffix(".hea").exists() and base.with_suffix(".dat").exists()):
                continue
            age = int(row.age) if pd.notnull(row.age) else None
            sex = {0.0: "Male", 1.0: "Female"}.get(float(row.sex) if pd.notnull(row.sex) else None)
            label = "infarction" if row.target_label == 1 else "normal"
            records.append(
                {
                    "key": f"cohort {int(row.ecg_id):05d}",
                    "caption": f"#{int(row.ecg_id):05d}  {label}  "
                               f"{_demographic_text(age, sex)}",
                    "path": str(base),
                    "age": age,
                    "sex": sex,
                }
            )

    for key, relative in SAMPLE_RECORDS:
        base = project_root / relative
        if base.with_suffix(".hea").exists() and base.with_suffix(".dat").exists():
            records.append(
                {
                    "key": key,
                    "caption": f"{key}  no demographics on file",
                    "path": str(base),
                    "age": None,
                    "sex": None,
                }
            )

    return records


def _demographic_text(age, sex) -> str:
    """Render '67M', '41F', or 'age/sex unknown' for the picker line."""
    if age is None and sex is None:
        return "age/sex unknown"
    initial = {"Male": "M", "Female": "F"}.get(sex or "", "?")
    return f"{age if age is not None else '?'}{initial}"


@st.cache_data(show_spinner="Running the dual-engine analysis...")
def analyse(record_path: str, age, sex) -> dict:
    """
    Read the waveform and run it through the shipped ensemble and guardrails.

    This is the production path end to end: `wfdb` -> `CardioSenseEnsemble` ->
    `ClinicalTriageEngine`. Nothing here reimplements either step, which is the
    failure mode every other artefact in this repository was written to avoid.
    """
    ensemble, triage_engine = load_pipeline()
    raw_signals, fields = wfdb.rdsamp(record_path)

    ensemble_result = ensemble.predict(
        raw_signals,
        fs=fields["fs"],
        lead_names=fields["sig_name"],
        age=age,
        sex=sex,
    )

    if ensemble_result.get("status") != "success":
        return {
            "rejected": True,
            "reason": ensemble_result.get("reason", "signal quality gate rejected this record"),
        }

    return {
        "rejected": False,
        "ensemble": ensemble_result,
        "triage": triage_engine.evaluate(ensemble_result),
        "raw_signals": raw_signals,
        "fields": fields,
    }


@st.cache_resource(show_spinner=False)
def build_provenance() -> dict:
    """
    Describe the build the screen is showing.

    A triage decision that cannot be tied to a commit cannot be defended after
    the fact, and the SBAR note does not yet carry this information. Until it
    does, the reader states it on screen.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        commit = ""

    models = {}
    if CHECKPOINT_DIR.exists():
        for artefact in sorted(CHECKPOINT_DIR.glob("*")):
            if artefact.is_file():
                stamp = datetime.fromtimestamp(artefact.stat().st_mtime)
                models[artefact.name] = stamp.strftime("%Y-%m-%d %H:%M")

    return {
        "commit": commit or "unknown",
        "models": models,
        "cohort_rows": len(pd.read_csv(FEATURES_CSV)) if FEATURES_CSV.exists() else 0,
    }


# ---------------------------------------------------------
# HTML fragments
# ---------------------------------------------------------
# Every value interpolated below is escaped. The data is local and trusted, but
# a renderer that assumes its inputs are safe is how an injected record label
# becomes page markup.


def decision_card(caption: str, triage_result: dict) -> str:
    headline, accent = theme.tier_presentation(triage_result["final_tier"])
    return f"""
    <div class="cs-decision {accent}">
      <div>
        <div class="cs-tier">{html.escape(headline)}</div>
        <div class="cs-action">{html.escape(triage_result["triage_action"])}</div>
        <div class="cs-patient">{html.escape(caption)}</div>
      </div>
      <div class="cs-riskpair">
        <div>
          <div class="cs-figure">{triage_result["effective_risk_pct"]:.1f}%</div>
          <div class="cs-figure-label">Effective risk</div>
        </div>
        <div>
          <div class="cs-figure">{triage_result["baseline_ensemble_risk_pct"]:.1f}%</div>
          <div class="cs-figure-label">Model risk before rules</div>
        </div>
      </div>
    </div>
    """


def findings_panel(triage_result: dict) -> str:
    findings = triage_result["findings"]
    if not findings:
        items = '<li><span class="cs-chip rule">none</span> No guardrail fired. ' \
                'This is not a claim that the ECG is normal - only that no rule ' \
                'in <code>src/triage/triage_rules.py</code> is satisfied by the ' \
                'measurements above.</li>'
    else:
        items = "".join(
            f'<li><span class="cs-chip {theme.severity_class(f["severity"])}">'
            f'{html.escape(f["severity"])}</span>'
            f'<span class="cs-chip rule">{html.escape(f["rule_id"])}</span>'
            f'<span>{html.escape(f["detail"])}</span></li>'
            for f in findings
        )

    return f"""
    <div class="cs-panel">
      <p class="cs-panel-title">Guardrail findings</p>
      <p class="cs-panel-note">
        ESCALATE activates the emergency pathway. REVIEW means the tracing cannot
        be dismissed as normal and a clinician must read it.
      </p>
      <ul class="cs-findings">{items}</ul>
    </div>
    """


def st_evidence_panel(st_by_lead: dict, age, sex) -> str:
    """
    The per-lead measurements the criterion is evaluated on.

    Cut-points come from `cut_point_mv`, the same function the rule engine calls,
    so the number on screen cannot disagree with the number that decided the
    tier. A lead clearing its own cut-point is highlighted but explicitly not
    called a criterion: the definition needs two contiguous leads.
    """
    normalised_age = normalise_age(age)
    normalised_sex = normalise_sex(sex)

    rows = []
    for index, lead in enumerate(LEADS):
        cut, basis = cut_point_mv(lead, normalised_age, normalised_sex)
        measured = lead in st_by_lead
        value = st_by_lead[lead] if measured else None
        clears = measured and value >= cut

        classes = []
        if lead == "V1":
            classes.append("group-start")
        if clears:
            classes.append("meets")
        if not measured:
            classes.append("unmeasured")

        rows.append(
            f'<tr class="{" ".join(classes)}">'
            f'<td class="lead-name">{lead}</td>'
            f'<td class="num">{"-" if not measured else f"{value:+.3f}"}</td>'
            f'<td class="num">{cut:.2f}</td>'
            f'<td class="basis">{html.escape(basis)}</td>'
            f'<td class="num">{"not measured" if not measured else ("clears" if clears else "below")}</td>'
            f"</tr>"
        )

    return f"""
    <div class="cs-panel">
      <p class="cs-panel-title">ST deviation at the J point, per lead</p>
      <p class="cs-panel-note">
        Measured against the isoelectric PR baseline. A lead that failed
        plausibility checking is reported as not measured rather than as zero:
        an absent reading is not a normal reading. Clearing a cut-point in one
        lead is not the criterion - the definition requires two contiguous leads.
      </p>
      <table class="cs-data">
        <thead>
          <tr>
            <th>Lead</th><th class="num">ST at J (mV)</th><th class="num">Cut-point (mV)</th>
            <th>Threshold applied</th><th class="num">vs cut-point</th>
          </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
    """


def contiguity_panel(st_by_lead: dict, age, sex) -> str:
    """
    Which contiguous pairs were testable, and which satisfied the definition.

    Produced by calling `find_stemi_pairs` - the rule itself - rather than by
    restating its logic in the view.
    """
    normalised_age = normalise_age(age)
    normalised_sex = normalise_sex(sex)

    testable = sum(1 for a, b in CONTIGUOUS_PAIRS if a in st_by_lead and b in st_by_lead)
    pairs = find_stemi_pairs(st_by_lead, normalised_age, normalised_sex)

    # `body` is substituted into an indented block of raw HTML. It has to be
    # stripped: a leading or trailing newline lands a whitespace-only line inside
    # the <div>, a blank line closes the raw-HTML block, and everything after it
    # is parsed as markdown - which prints the table as literal source text.
    if not pairs:
        body = (
            f'<p class="cs-panel-note" style="margin:0">No contiguous pair meets the '
            f'statement in the fourth universal definition of myocardial infarction. '
            f'{testable} of {len(CONTIGUOUS_PAIRS)} pairs were testable from the leads '
            f'that were measured; the rest were not evaluable, which is not the same '
            f'as negative.</p>'
        )
    else:
        rows = "".join(
            f'<tr class="meets"><td class="lead-name">{html.escape(p.lead_a)} / {html.escape(p.lead_b)}</td>'
            f'<td class="num">{p.st_a_mv:+.3f} / {p.st_b_mv:+.3f}</td>'
            f'<td class="num">{p.cut_a_mv:.2f} / {p.cut_b_mv:.2f}</td>'
            f'<td>{html.escape(p.territory)}</td></tr>'
            for p in pairs
        )
        body = f"""
        <table class="cs-data">
          <thead><tr><th>Pair</th><th class="num">ST at J (mV)</th>
          <th class="num">Cut-points (mV)</th><th>Territory</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <p class="cs-panel-note" style="margin:.7rem 0 0 0">
          {testable} of {len(CONTIGUOUS_PAIRS)} pairs were testable; {len(pairs)} satisfied
          the criterion.
        </p>
        """.strip()

    return f"""
    <div class="cs-panel">
      <p class="cs-panel-title">Contiguous-lead criterion</p>
      {body}
    </div>
    """


def quality_strip(biomarkers: dict, st_by_lead: dict) -> str:
    def item(label, value, css_class=""):
        return (
            f'<div class="cs-strip-item"><div class="cs-strip-label">{label}</div>'
            f'<div class="cs-strip-value {css_class}">{value}</div></div>'
        )

    plausible = biomarkers.get("qrs_plausible", True)
    beats = biomarkers.get("beats_used", "-")
    return f"""
    <div class="cs-panel">
      <p class="cs-panel-title">Signal quality and measurement confidence</p>
      <div class="cs-strip">
        {item("SQI", f"{biomarkers['overall_sqi']:.3f}")}
        {item("Beats averaged", html.escape(str(beats)))}
        {item("Heart rate", f"{biomarkers['heart_rate_bpm']:.1f} bpm")}
        {item("QRS duration", f"{biomarkers['qrs_duration_ms']:.0f} ms",
             "" if plausible else "cs-flag-warn")}
        {item("QRS plausible", "yes" if plausible else "no - measurements unreliable",
              "" if plausible else "cs-flag-bad")}
        {item("Leads measured", f"{len(st_by_lead)} of {len(LEADS)}",
              "" if len(st_by_lead) == len(LEADS) else "cs-flag-warn")}
      </div>
    </div>
    """


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------

catalogue = load_catalogue()

st.markdown(
    """
    <div class="cs-masthead">
      <div class="cs-wordmark">CardioSense
        <small>Deterministic triage reader - adjunctive decision support, not a diagnosis</small>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not catalogue:
    st.error(
        "No records to read. The reader lists only records whose manifest entry "
        "and WFDB files are both on disk, and nothing matched. Materialise the "
        "cohort with `python src/data/download_ptbxl.py --cohort`."
    )
    st.stop()

col_pick, col_age, col_sex = st.columns([5, 1, 1])

with col_pick:
    choice = st.selectbox(
        "Record",
        options=range(len(catalogue)),
        format_func=lambda i: catalogue[i]["caption"],
        index=next(
            (i for i, r in enumerate(catalogue) if r["key"] == "cohort 00544"),
            0,
        ),
        help="Defaults to record 544: the clearest contiguous anterior elevation in the cohort.",
    )
with col_age:
    record = catalogue[choice]
    age_input = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=int(record["age"]) if record["age"] is not None else 0,
        step=1,
        key=f"age::{record['key']}",
        help="0 means unknown. The engine then holds V2-V3 to its most sensitive "
             "cut-point rather than relaxing them.",
    )
with col_sex:
    sex_input = st.selectbox(
        "Sex",
        options=["Unknown", "Male", "Female"],
        index={"Unknown": 0, "Male": 1, "Female": 2}[record["sex"] or "Unknown"],
        key=f"sex::{record['key']}",
    )

st.caption(
    "Demographics default to the values on file and can be overridden to see how "
    "the V2-V3 thresholds move."
)

result = analyse(record["path"], age_input or None, None if sex_input == "Unknown" else sex_input)

if result["rejected"]:
    st.error(f"Signal rejected before analysis: {result['reason']}")
    st.caption("No triage decision is produced from a record that failed the quality gate.")
    st.stop()

ens = result["ensemble"]
tri = result["triage"]
st_by_lead = ens["st_by_lead"]

st.markdown(decision_card(record["caption"], tri), unsafe_allow_html=True)
st.markdown(findings_panel(tri), unsafe_allow_html=True)
st.markdown(st_evidence_panel(st_by_lead, age_input or None, sex_input), unsafe_allow_html=True)
st.markdown(contiguity_panel(st_by_lead, age_input or None, sex_input), unsafe_allow_html=True)
st.markdown(quality_strip(ens["clinical_biomarkers"], st_by_lead), unsafe_allow_html=True)

# ---------------------------------------------------------
# Engines and regional maxima
# ---------------------------------------------------------

col_engine, col_region = st.columns(2)

# A panel that has to hold a matplotlib figure cannot be built out of HTML
# strings: `st.pyplot` renders into its own container, so a <div> opened in an
# earlier `st.markdown` call is long closed by then. These three panels use
# `st.container(border=True)` and the border is styled to match `.cs-panel`.
with col_engine:
    with st.container(border=True):
        st.markdown(
            '<p class="cs-panel-title">Statistical engines</p>'
            '<p class="cs-panel-note">The fused score is the unweighted mean of the two '
            'engine probabilities. No calibration is applied, so the two bars disagreeing '
            'by a wide margin is information, not noise to average away.</p>',
            unsafe_allow_html=True,
        )
        probs = ens["engine_probabilities"]
        figure, axis = plt.subplots(figsize=(5.6, 2.1))
        names = ["Interpretable RF", "Deep 1D-CNN", "Fused (mean)"]
        values = [probs["interpretable_rf"] * 100, probs["deep_cnn"] * 100, probs["fused_composite"] * 100]
        axis.barh(names[::-1], values[::-1], color=["#8a939f", "#8a939f", "#1f5c8b"][::-1])
        axis.set_xlim(0, 100)
        axis.set_xlabel("Probability of acute pathology (%)", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

with col_region:
    with st.container(border=True):
        st.markdown(
            '<p class="cs-panel-title">Regional ST maxima</p>'
            '<p class="cs-panel-note">Descriptive only. These maxima are what the old '
            'guardrail escalated on and what the "suspected culprit" label is still '
            'derived from - neither is a criterion, and neither should be read as one.</p>',
            unsafe_allow_html=True,
        )
        rows = "".join(
            f'<tr><td>{html.escape(name)}</td><td class="num">{info["max_st_mv"]:+.3f}</td>'
            f'<td class="basis">{html.escape(", ".join(info["leads"]))}</td></tr>'
            for name, info in ens["territory_breakdown"].items()
        )
        st.markdown(
            f'<table class="cs-data"><thead><tr><th>Territory</th>'
            f'<th class="num">Max ST (mV)</th><th>Leads</th></tr></thead><tbody>{rows}</tbody></table>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Tracing inspector
# ---------------------------------------------------------

with st.container(border=True):
    st.markdown(
        '<p class="cs-panel-title">Tracing inspector</p>',
        unsafe_allow_html=True,
    )

    raw_signals = result["raw_signals"]
    fields = result["fields"]
    lead_names = fields["sig_name"]
    fs = fields["fs"]

    # Open on the lead that drove the decision when there is one, so the first
    # thing a reader sees is the evidence rather than whichever lead happened to
    # be second.
    finding_leads = [lead for lead in LEADS if lead in st_by_lead and st_by_lead[lead] >= cut_point_mv(
        lead, normalise_age(age_input or None), normalise_sex(sex_input))[0]]
    default_lead = finding_leads[0] if finding_leads else ("II" if "II" in lead_names else lead_names[0])

    inspect_lead = st.selectbox(
        "Lead",
        options=lead_names,
        index=lead_names.index(default_lead) if default_lead in lead_names else 0,
    )

    cleaned = filter_ecg(raw_signals, fs=fs)
    lead_signal = cleaned[:, lead_names.index(inspect_lead)]
    time_axis = np.arange(len(lead_signal)) / fs
    r_peaks = detect_r_peaks(lead_signal, fs=fs)

    if len(r_peaks) >= 3:
        target_r = r_peaks[2]
        landmarks = delineate_beat_landmarks(lead_signal, target_r, fs=fs)

        if landmarks is None:
            st.info("This beat sits too close to the edge of the recording to delineate.")
        else:
            start = target_r - int(0.25 * fs)
            end = target_r + int(0.40 * fs)

            figure, axis = plt.subplots(figsize=(10.5, 3.2))
            axis.plot(
                time_axis[start:end],
                lead_signal[start:end],
                color="#1a1d21",
                linewidth=1.6,
            )
            markers = (
                ("PR baseline", landmarks["baseline_idx"], "#1f5c8b", landmarks["baseline_voltage"]),
                ("Q onset", landmarks["q_onset_idx"], "#7c3aed", None),
                ("R peak", target_r, "#b3261e", None),
                ("ST at J (measured)", landmarks["j_point_idx"], "#9a5b00", None),
            )
            for label, index, colour, voltage in markers:
                y = voltage if voltage is not None else lead_signal[index]
                axis.scatter([time_axis[index]], [y], color=colour, s=45, label=label, zorder=5)

            if landmarks.get("st_at_j60_idx") is not None:
                j60 = landmarks["st_at_j60_idx"]
                axis.scatter([time_axis[j60]], [lead_signal[j60]], color="#8a939f", s=32,
                             marker="x", label="J+60 (legacy, not used)", zorder=5)

            axis.axhline(landmarks["baseline_voltage"], color="#1f5c8b", linestyle=":", alpha=0.5)
            axis.set_title(
                f"Lead {inspect_lead}  |  QRS {landmarks['qrs_duration_ms']:.0f} ms  |  "
                f"ST at J {landmarks['st_deviation_mv']:+.3f} mV",
                fontsize=9.5,
                loc="left",
            )
            axis.legend(loc="upper right", ncol=5, fontsize=7.5, frameon=False)
            axis.grid(linestyle="--", alpha=0.22)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
    else:
        st.info("Fewer than three beats were detected, so there is nothing to delineate.")

    st.markdown(
        '<p class="cs-panel-note" style="margin:0">The J+60 marker is drawn for comparison '
        "only. Every measurement in this reader stops at the J point.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Note and provenance
# ---------------------------------------------------------

report_text = generate_clinical_report(record["key"], ens, tri)

with st.expander("SBAR note"):
    st.caption(
        "Generated by `src/triage/report_generator.py`. It carries no timestamp, "
        "model version or commit reference yet - that is the open P0-5 item."
    )
    st.code(report_text, language="text")

st.download_button(
    "Download SBAR note (.txt)",
    data=report_text,
    file_name=f"{record['key'].replace(' ', '_')}_note.txt",
)

provenance = build_provenance()
model_line = "  ".join(f"{name} {stamp}" for name, stamp in provenance["models"].items())
st.markdown(
    f"""
    <div class="cs-footer">
      commit <code>{html.escape(provenance["commit"])}</code> &nbsp;|&nbsp;
      rules <code>src/triage/triage_rules.py</code> + <code>src/triage/leads.py</code>
      &nbsp;|&nbsp; {provenance["cohort_rows"]} cohort records scored by
      <code>src/evaluation/score_guardrails.py</code>
      <br>{html.escape(model_line)}
    </div>
    """,
    unsafe_allow_html=True,
)
