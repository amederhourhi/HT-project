"""
CardioSense CDSS - Clinical Emergency Interface
Card-based intake, dual-engine risk fusion, and full-screen triage atmosphere.
Styles are cleanly decoupled in styles.css and theme.py.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import wfdb

from src.dashboard.theme import apply_theme_and_atmosphere
from src.dsp.filters import filter_ecg
from src.features.fiducials import delineate_beat_landmarks
from src.features.r_peaks import detect_r_peaks
from src.triage.report_generator import generate_clinical_report
from src.triage.risk_fusion import CardioSenseEnsemble
from src.triage.triage_rules import ClinicalTriageEngine

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioSense | Emergency Triage",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_cardiosense_pipeline():
    ensemble = CardioSenseEnsemble()
    triage_engine = ClinicalTriageEngine()
    return ensemble, triage_engine


@st.cache_data
def load_grouped_cohort():
    cohort_dir = project_root / "data" / "cohort"
    features_csv = project_root / "data" / "processed" / "clinical_features.csv"
    manifest_csv = cohort_dir / "cohort_manifest.csv"

    if not features_csv.exists() or not manifest_csv.exists():
        return {}, {}, {}

    df_feats = pd.read_csv(features_csv)
    df_man = pd.read_csv(manifest_csv)
    merged = df_feats.merge(df_man[["ecg_id", "filename_lr"]], on="ecg_id", how="left")

    def format_records(sub_df, category_name):
        records = []
        for _, r in sub_df.iterrows():
            record_base = cohort_dir / r["filename_lr"]
            if record_base.with_suffix(".hea").exists() and record_base.with_suffix(".dat").exists():
                records.append({
                    "ecg_id": f"{int(r['ecg_id']):05d}",
                    "path": str(record_base),
                    "age": int(r["age"]) if pd.notnull(r.get("age")) else 55,
                    "gender": "Male" if r.get("sex") == 0.0 else ("Female" if r.get("sex") == 1.0 else "N/A"),
                    "category": category_name,
                })
        return records

    ant_df = merged[(merged["target_label"] == 1) & (merged["st_anterior_max"] >= 0.15)]
    inf_df = merged[(merged["target_label"] == 1) & (merged["st_inferior_max"] >= 0.10)]
    norm_df = merged[merged["target_label"] == 0]

    return {
        "anterior": format_records(ant_df, "Anterior STEMI (LAD)"),
        "inferior": format_records(inf_df, "Inferior STEMI (RCA)"),
        "normal": format_records(norm_df, "Normal Control"),
    }


ensemble, triage_engine = load_cardiosense_pipeline()
grouped_cohort = load_grouped_cohort()

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
if "view" not in st.session_state:
    st.session_state["view"] = "welcome"
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "active_patient" not in st.session_state:
    st.session_state["active_patient"] = None
if "triage_tier" not in st.session_state:
    st.session_state["triage_tier"] = "NEUTRAL"

# Decoupled style injection
is_results = st.session_state["view"] == "results"
apply_theme_and_atmosphere(st.session_state["theme"], st.session_state["triage_tier"], is_results)


# ---------------------------------------------------------
# Header Bar (Home Navigation & Theme Toggle)
# ---------------------------------------------------------
col_brand, col_theme = st.columns([8, 2])

with col_brand:
    if st.session_state["view"] != "welcome":
        if st.button("⬅ Return to Home Hub", key="btn_home"):
            st.session_state["view"] = "welcome"
            st.session_state["triage_tier"] = "NEUTRAL"
            st.session_state["active_patient"] = None
            st.rerun()
    else:
        st.markdown("<h2 style='margin:0;'>🫀 CARDIOSENSE CDSS</h2>", unsafe_allow_html=True)

with col_theme:
    curr_theme = st.session_state["theme"]
    new_theme_label = "☀ Light Mode" if curr_theme == "dark" else "🌙 Dark Mode"
    if st.button(new_theme_label, use_container_width=True):
        st.session_state["theme"] = "light" if curr_theme == "dark" else "dark"
        st.rerun()

st.markdown("<hr style='margin-top:8px; margin-bottom:20px; opacity:0.2;'>", unsafe_allow_html=True)


# =========================================================
# VIEW 1: WELCOME LAUNCHING HUB
# =========================================================
if st.session_state["view"] == "welcome":
    st.markdown(
        """
        <div style='text-align: center; padding: 25px 10px 35px 10px;'>
            <h1 class='main-title'>Clinical Decision-Support Hub</h1>
            <p class='subtitle'>Autonomous 12-lead electrophysiology analysis, deterministic safety guardrails, and instant coronary territory localization.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class='hub-card'>
                <div style='font-size: 2.6rem; margin-bottom: 10px;'>🚑</div>
                <div class='hub-title'>Live Emergency Walk-In</div>
                <div class='hub-desc'>Intake a new patient arriving right now with <b>zero past hospital records</b>. Enter demographics, acquire live 10-second voltage, and run instant triage.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Emergency Intake ➡", key="btn_go_intake", use_container_width=True):
            st.session_state["view"] = "intake"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class='hub-card'>
                <div style='font-size: 2.6rem; margin-bottom: 10px;'>📂</div>
                <div class='hub-title'>Verified Clinical Library</div>
                <div class='hub-desc'>Explore 297 validated PTB-XL cases grouped by specific pathology: <b>Anterior STEMI (LAD)</b>, <b>Inferior STEMI (RCA)</b>, and <b>Normal Controls</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Browse Grouped Library ➡", key="btn_go_library", use_container_width=True):
            st.session_state["view"] = "library"
            st.rerun()


# =========================================================
# VIEW 2: NEW PATIENT EMERGENCY INTAKE
# =========================================================
elif st.session_state["view"] == "intake":
    st.markdown("## 🚑 Emergency Department Intake (First-Time Walk-In)")
    st.caption("Patient has zero prior medical history. All triage decisions are made from real-time 10-second electrophysiology.")

    st.markdown("#### 1. Patient Demographics")
    c_fn, c_ln, c_age, c_gen = st.columns(4)
    with c_fn:
        first_name = st.text_input("First Name", value="Alex")
    with c_ln:
        last_name = st.text_input("Last Name", value="Mercer")
    with c_age:
        age = st.number_input("Age", min_value=18, max_value=105, value=62, step=1)
    with c_gen:
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("#### 2. Bedside ECG Acquisition")
    c_scen1, c_scen2, c_scen3 = st.columns(3)

    with c_scen1:
        st.markdown(
            """
            <div class='hub-card' style='border-color: #EF4444;'>
                <div style='font-size: 1.8rem;'>🔴</div>
                <b>Crushing Chest Pain</b>
                <div style='font-size: 0.85rem; opacity: 0.8;'>Simulates acute transmural anterior injury (LAD)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Acquire & Analyze (LAD Case)", use_container_width=True):
            st.session_state["active_patient"] = {
                "name": f"{first_name} {last_name}",
                "age": age,
                "gender": gender,
                "id": f"ED-WALKIN-{age}{gender[0]}",
                "path": str(project_root / "data" / "samples" / "records100" / "00000" / "00008_lr"),
                "status": "Live Walk-in (Zero Past Records)",
            }
            st.session_state["view"] = "results"
            st.rerun()

    with c_scen2:
        st.markdown(
            """
            <div class='hub-card' style='border-color: #F59E0B;'>
                <div style='font-size: 1.8rem;'>🟠</div>
                <b>Nausea & Diaphoresis</b>
                <div style='font-size: 0.85rem; opacity: 0.8;'>Simulates acute inferior injury (RCA)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Acquire & Analyze (RCA Case)", use_container_width=True):
            inf_path = grouped_cohort["inferior"][0]["path"] if grouped_cohort["inferior"] else str(project_root / "data" / "samples" / "records100" / "00000" / "00001_lr")
            st.session_state["active_patient"] = {
                "name": f"{first_name} {last_name}",
                "age": age,
                "gender": gender,
                "id": f"ED-WALKIN-{age}{gender[0]}",
                "path": inf_path,
                "status": "Live Walk-in (Zero Past Records)",
            }
            st.session_state["view"] = "results"
            st.rerun()

    with c_scen3:
        st.markdown(
            """
            <div class='hub-card' style='border-color: #10B981;'>
                <div style='font-size: 1.8rem;'>🟢</div>
                <b>Routine / Normal Rhythm</b>
                <div style='font-size: 0.85rem; opacity: 0.8;'>Simulates stable normal sinus rhythm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Acquire & Analyze (Normal Case)", use_container_width=True):
            st.session_state["active_patient"] = {
                "name": f"{first_name} {last_name}",
                "age": age,
                "gender": gender,
                "id": f"ED-WALKIN-{age}{gender[0]}",
                "path": str(project_root / "data" / "samples" / "records100" / "00000" / "00001_lr"),
                "status": "Live Walk-in (Zero Past Records)",
            }
            st.session_state["view"] = "results"
            st.rerun()


# =========================================================
# VIEW 3: GROUPED CLINICAL CASE LIBRARY
# =========================================================
elif st.session_state["view"] == "library":
    st.markdown("## 📂 Verified Clinical Case Library")
    st.caption("Browse 297 clinical records grouped cleanly by verified pathology.")

    c_g1, c_g2, c_g3 = st.columns(3)
    with c_g1:
        st.metric("🔴 Anterior STEMI (LAD)", f"{len(grouped_cohort.get('anterior', []))} Cases")
    with c_g2:
        st.metric("🟠 Inferior STEMI (RCA)", f"{len(grouped_cohort.get('inferior', []))} Cases")
    with c_g3:
        st.metric("🟢 Normal Controls", f"{len(grouped_cohort.get('normal', []))} Cases")

    st.markdown("<hr style='margin:15px 0; opacity:0.2;'>", unsafe_allow_html=True)

    group_select = st.radio(
        "Select Category to Inspect:",
        ["🔴 Anterior STEMIs (LAD Artery Blocked)", "🟠 Inferior STEMIs (RCA Artery Blocked)", "🟢 Normal Controls (No Ischemia)"],
        horizontal=True,
    )

    if "Anterior" in group_select:
        active_list = grouped_cohort.get("anterior", [])
    elif "Inferior" in group_select:
        active_list = grouped_cohort.get("inferior", [])
    else:
        active_list = grouped_cohort.get("normal", [])

    st.markdown("#### Select Patient Record:")
    grid_cols = st.columns(3)
    for idx, case in enumerate(active_list[:12]):
        col_slot = grid_cols[idx % 3]
        with col_slot:
            st.markdown(
                f"""
                <div class='hub-card' style='padding: 14px; margin-bottom: 8px;'>
                    <b>Patient #{case['ecg_id']}</b><br>
                    <span style='font-size:0.9rem;'>Age: <b>{case['age']}</b> | Gender: <b>{case['gender']}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if col_slot.button(f"Analyze Patient #{case['ecg_id']}", key=f"btn_case_{case['ecg_id']}", use_container_width=True):
                st.session_state["active_patient"] = {
                    "name": f"Patient #{case['ecg_id']}",
                    "age": case["age"],
                    "gender": case["gender"],
                    "id": f"PTB-XL-{case['ecg_id']}",
                    "path": case["path"],
                    "status": f"Archived Case ({case['category']})",
                }
                st.session_state["view"] = "results"
                st.rerun()


# =========================================================
# VIEW 4: LIVE RESULTS WITH FULL-SCREEN COLOR ATMOSPHERE
# =========================================================
elif st.session_state["view"] == "results" and st.session_state["active_patient"]:
    pt = st.session_state["active_patient"]
    raw_signals, fields = wfdb.rdsamp(pt["path"])
    fs = fields["fs"]
    lead_names = fields["sig_name"]

    ens_res = ensemble.predict(raw_signals, fs=fs, lead_names=lead_names)

    if ens_res["status"] == "rejected":
        st.markdown("<div class='alarm-banner'>🚨 SIGNAL REJECTED: DISCONNECTED ELECTRODE</div>", unsafe_allow_html=True)
        st.error(f"Gatekeeper Rejection: {ens_res['reason']}. Check physical lead contacts.")
        if st.button("⬅ Return to Dashboard"):
            st.session_state["view"] = "welcome"
            st.rerun()
        st.stop()

    tri_res = triage_engine.evaluate(ens_res)
    tier = tri_res["final_tier"]
    st.session_state["triage_tier"] = tier

    # Top Hero Banner
    if tier == "TIER_3_ESCALATION":
        st.markdown("<div class='alarm-banner'>🚨 TIER 3 EMERGENCY: ACUTE INFARCTION DETECTED — CATH LAB ACTIVATION REQUIRED</div>", unsafe_allow_html=True)
    elif tier == "TIER_1_ROUTINE":
        st.markdown("<div class='safe-banner'>🟢 TIER 1 SAFE: NORMAL BASELINE CONDUCTION — ROUTINE FOLLOW-UP</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='neutral-banner'>⚪ TIER 2 REVIEW: INTERMEDIATE SUSPICION — REPEAT ECG IN 15 MIN</div>", unsafe_allow_html=True)

    # Patient Identity Bar
    st.markdown(
        f"""
        <div class='patient-header-card'>
            <div>👤 <b>Patient:</b> {pt['name']}</div>
            <div>🆔 <b>ID:</b> {pt['id']}</div>
            <div>🎂 <b>Age:</b> {pt['age']} yrs</div>
            <div>⚧ <b>Gender:</b> {pt['gender']}</div>
            <div>📋 <b>Intake:</b> {pt['status']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if tri_res["override_active"]:
        st.warning(
            f"⚠️ **DETERMINISTIC CLINICAL SAFETY OVERRIDE ACTIVE**  \n"
            + "\n".join([f"- {r}" for r in tri_res["override_reasons"]])
        )

    # 4 Key Metrics
    bm = ens_res["clinical_biomarkers"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Effective Risk", f"{tri_res['effective_risk_pct']:.1f}%")
    c2.metric("Suspected Culprit", ens_res["culprit_artery"])
    c3.metric("Heart Rate", f"{bm['heart_rate_bpm']:.1f} BPM")
    c4.metric("QRS Conduction", f"{bm['qrs_duration_ms']:.1f} ms")

    st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)

    # 2 Column Visuals
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🤖 Dual-Engine Risk Calibration")
        p_rf = ens_res["engine_probabilities"]["interpretable_rf"] * 100
        p_cnn = ens_res["engine_probabilities"]["deep_cnn"] * 100
        p_eff = tri_res["effective_risk_pct"]

        fig_ai, ax_ai = plt.subplots(figsize=(6, 3.5))
        ax_ai.bar(
            ["Interpretable RF", "Deep 1D-CNN", "Final Effective Risk"],
            [p_rf, p_cnn, p_eff],
            color=["#38BDF8", "#FB923C", "#EF4444" if p_eff >= 65 else "#10B981"],
        )
        ax_ai.set_ylim(0, 100)
        ax_ai.set_ylabel("Risk Score (%)")
        ax_ai.grid(True, linestyle="--", alpha=0.3)
        fig_ai.tight_layout()
        st.pyplot(fig_ai)
        plt.close()

    with col_r:
        st.markdown("### 🫀 Coronary Artery Ischemic Burden")
        tb = ens_res["territory_breakdown"]
        t_names = ["Anterior (LAD)", "Inferior (RCA)", "Lateral (LCx)"]
        t_vals = [
            tb["Anterior / Septal (LAD)"]["max_st_mv"],
            tb["Inferior (RCA)"]["max_st_mv"],
            tb["Lateral (LCx)"]["max_st_mv"],
        ]
        fig_cor, ax_cor = plt.subplots(figsize=(6, 3.5))
        bar_colors = ["#EF4444" if v >= 0.10 else "#64748B" for v in t_vals]
        ax_cor.bar(t_names, t_vals, color=bar_colors)
        ax_cor.axhline(0.10, color="crimson", linestyle="--", label="Limb Alarm (0.10 mV)")
        ax_cor.axhline(0.20, color="darkred", linestyle=":", label="Precordial Alarm (0.20 mV)")
        ax_cor.set_ylabel("Max ST Displacement (mV)")
        ax_cor.legend(loc="upper right", fontsize=8)
        ax_cor.grid(True, linestyle="--", alpha=0.3)
        fig_cor.tight_layout()
        st.pyplot(fig_cor)
        plt.close()

    st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)

    # Waveform Inspector
    st.markdown("### 📈 Interactive Electrophysiology Inspector")
    inspect_lead = st.selectbox("Inspect Waveform Lead:", lead_names, index=lead_names.index("II") if "II" in lead_names else 1)
    lead_idx = lead_names.index(inspect_lead)

    cleaned = filter_ecg(raw_signals, fs=fs)
    lead_sig = cleaned[:, lead_idx]
    time_ax = np.arange(len(lead_sig)) / fs
    r_peaks = detect_r_peaks(lead_sig, fs=fs)

    if len(r_peaks) >= 3:
        target_r = r_peaks[2]
        lm = delineate_beat_landmarks(lead_sig, target_r, fs=fs)
        start_s = target_r - int(0.25 * fs)
        end_s = target_r + int(0.40 * fs)

        # lm is None when this beat sits too close to the record edges to
        # window; the inspector then has nothing honest to draw.
        if lm is not None:
            # S trough = deepest sample inside the QRS. Plot-only landmark: the
            # measurement chain stops at the J point.
            s_trough = int(np.argmin(lead_sig[lm["q_onset_idx"]:lm["j_point_idx"] + 1])) + lm["q_onset_idx"]

            fig_wave, ax_wave = plt.subplots(figsize=(10, 3.8))
            ax_wave.plot(time_ax[start_s:end_s], lead_sig[start_s:end_s], color="white" if st.session_state["theme"] == "dark" else "black", linewidth=2.0)
            ax_wave.scatter(time_ax[lm["baseline_idx"]], lm["baseline_voltage"], color="#38BDF8", s=60, label="PR Baseline", zorder=5)
            ax_wave.scatter(time_ax[lm["q_onset_idx"]], lead_sig[lm["q_onset_idx"]], color="#A855F7", s=60, label="Q-Onset", zorder=5)
            ax_wave.scatter(time_ax[target_r], lead_sig[target_r], color="#EF4444", s=60, label="R-Peak", zorder=5)
            ax_wave.scatter(time_ax[s_trough], lead_sig[s_trough], color="#EC4899", s=60, label="S-Trough", zorder=5)
            ax_wave.scatter(time_ax[lm["j_point_idx"]], lead_sig[lm["j_point_idx"]], color="#F59E0B", s=60, label="ST @ J (measured)", zorder=5)
            if lm["st_at_j60_idx"] is not None:
                j60 = lm["st_at_j60_idx"]
                ax_wave.scatter(time_ax[j60], lead_sig[j60], color="#94A3B8", s=40, label="J+60 (legacy)", zorder=5)
            ax_wave.axhline(lm["baseline_voltage"], color="#38BDF8", linestyle=":", alpha=0.6)
            ax_wave.set_title(
                f"Lead {inspect_lead} Beat Landmark Delineation | "
                f"QRS {lm['qrs_duration_ms']:.0f} ms | ST at J: {lm['st_deviation_mv']:+.3f} mV",
                fontsize=10,
            )
            ax_wave.legend(loc="upper right", ncol=6, fontsize=8)
            ax_wave.grid(True, linestyle="--", alpha=0.3)
            fig_wave.tight_layout()
            st.pyplot(fig_wave)
            plt.close()
        else:
            st.info("Selected beat is too close to the edge of the recording to delineate.")

    # SBAR Report
    report_text = generate_clinical_report(pt["id"], ens_res, tri_res)
    with st.expander("📄 View Auditable Clinical SBAR Note", expanded=False):
        st.code(report_text, language="text")

    st.download_button("📥 Download SBAR Clinical Note (.txt)", data=report_text, file_name=f"{pt['id']}_report.txt")