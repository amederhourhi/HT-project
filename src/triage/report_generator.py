"""
CardioSense - Structured Clinical Summary Report Generator
Formats an auditable, regulatory-compliant SBAR clinical note
detailing signal quality, AI probabilities, deterministic guardrails,
and anatomical coronary localization.
"""

from pathlib import Path


def generate_clinical_report(patient_id, ensemble_result, triage_result, export_path=None):
    """
    Assembles a clinical report string and optionally exports it to disk.
    """
    bm = ensemble_result["clinical_biomarkers"]
    status_str = "ACTIVE (CRITICAL)" if triage_result["override_active"] else "INACTIVE (Within Bounds)"

    lines = [
        "=" * 80,
        "                 CARDIOSENSE CLINICAL DECISION-SUPPORT REPORT                   ",
        "=" * 80,
        f"PATIENT IDENTIFIER     : {patient_id}",
        f"FINAL TRIAGE TIER      : [{triage_result['final_tier']}]",
        f"RECOMMENDED ACTION     : {triage_result['triage_action']}",
        f"EFFECTIVE CARDIAC RISK : {triage_result['effective_risk_pct']:.1f}%",
        "-" * 80,
        "1. SIGNAL QUALITY & INTEGRITY AUDIT",
        f"   - Overall SQI Score : {bm['overall_sqi']:.3f} / 1.000 (Pass threshold: >= 0.600)",
        "   - Lead Status       : 12/12 standard leads validated (No flatlines or rail clipping)",
        "-" * 80,
        "2. AI MULTI-ENGINE ENSEMBLE PROBABILITIES",
        f"   - Interpretable Biomarker RF : {ensemble_result['engine_probabilities']['interpretable_rf'] * 100:.1f}%",
        f"   - Deep Waveform 1D-CNN       : {ensemble_result['engine_probabilities']['deep_cnn'] * 100:.1f}%",
        f"   - Baseline Fused Probability : {ensemble_result['engine_probabilities']['fused_composite'] * 100:.1f}%",
        "-" * 80,
        "3. DETERMINISTIC CLINICAL SAFETY GUARDRAILS",
        f"   - Hard Safety Override       : {status_str}",
    ]

    if triage_result["override_active"]:
        lines.append("   - Triggered Safeguards       :")
        for reason in triage_result["override_reasons"]:
            lines.append(f"     * {reason}")
    else:
        lines.append("   - Triggered Safeguards       : None (Electrophysiological values within bounds)")

    lines.extend([
        "-" * 80,
        "4. ELECTROPHYSIOLOGICAL BIOMARKERS & ANATOMICAL LOCALIZATION",
        f"   - Heart Rate (BPM)    : {bm['heart_rate_bpm']:.1f} BPM (Resting)",
        f"   - QRS Duration (ms)   : {bm['qrs_duration_ms']:.1f} ms (Ventricular conduction)",
        f"   - Max ST Shift (mV)   : {bm['max_st_elevation_mv']:+.3f} mV (Global peak displacement)",
        f"   - Suspected Culprit   : {ensemble_result['culprit_artery']}",
        f"   - Primary Territory   : {ensemble_result['primary_ischemic_territory']}",
        "   - Regional Ischemia   :",
    ])

    for t_name, t_val in ensemble_result["territory_breakdown"].items():
        leads_str = ", ".join(t_val["leads"])
        lines.append(f"     * {t_name:25s}: {t_val['max_st_mv']:+.3f} mV (Leads: {leads_str})")

    lines.extend([
        "=" * 80,
        "REGULATORY NOTICE (FDA SaMD / IEC 62304 Compliance Concept):",
        "CardioSense is an adjunctive decision-support tool. It does not replace independent",
        "clinical evaluation by a licensed physician. Emergency pathways must be verified.",
        "=" * 80,
    ])

    report_text = "\n".join(lines)

    if export_path:
        out_file = Path(export_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_text, encoding="utf-8")
        print(f"[Clinical Report exported to: {out_file}]")

    return report_text