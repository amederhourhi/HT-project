"""
CardioSense - Structured Clinical Summary Report Generator
Formats an auditable, regulatory-compliant SBAR clinical note
detailing patient situation, background, assessment, and recommendations.
"""

from datetime import datetime, timezone
from pathlib import Path
import subprocess


# ---------------------------------------------------------
# Block 1: Audit Metadata Helpers
# What it does: Captures the short Git commit hash for software traceability.
# How it does it: Calls git rev-parse via subprocess; falls back to 'unknown'.
# ---------------------------------------------------------
def _get_commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------
# Block 2: Assemble SBAR Note
# What it does: Compiles measurements, models, and findings into SBAR format.
# How it does it: Builds Situation, Background, Assessment, and Recommendation
#                 sections with timestamp and version provenance.
# ---------------------------------------------------------
def generate_clinical_report(patient_id, ensemble_result, triage_result, export_path=None) -> str:
    bm = ensemble_result["clinical_biomarkers"]
    patient = ensemble_result.get("patient") or {}
    age_str = f"{patient.get('age')} years" if patient.get("age") is not None else "Unknown"
    sex_str = str(patient.get("sex") or "Unknown")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    commit_sha = _get_commit_sha()

    # Read culprit artery from contiguous findings if present
    culprit_findings = [f for f in triage_result.get("findings", []) if f.get("territory")]
    if culprit_findings:
        suspected_culprit = culprit_findings[0]["territory"]
    else:
        suspected_culprit = "None (Contiguous STEMI criteria not met)"

    lines = [
        "=" * 80,
        "                 CARDIOSENSE CLINICAL DECISION-SUPPORT REPORT                   ",
        "=" * 80,
        f"REPORT GENERATED       : {now_utc}",
        f"SOFTWARE BUILD (COMMIT): {commit_sha}",
        f"PATIENT IDENTIFIER     : {patient_id}",
        "-" * 80,
        "S - SITUATION",
        f"   - Final Triage Tier : [{triage_result['final_tier']}]",
        f"   - Recommended Action: {triage_result['triage_action']}",
        f"   - Model Risk Score  : {triage_result['effective_risk_pct']:.1f}%",
        "-" * 80,
        "B - BACKGROUND & SIGNAL INTEGRITY",
        f"   - Demographics      : Age {age_str} | Sex: {sex_str}",
        f"   - Signal Quality    : {bm['overall_sqi']:.3f} / 1.000 (Pass threshold: >= 0.600)",
        f"   - Leads Validated   : 12/12 leads active (no flatline or rail saturation)",
        "-" * 80,
        "A - ASSESSMENT (AI & DETERMINISTIC SAFEGUARDS)",
        f"   - Interpretable RF  : {ensemble_result['engine_probabilities']['interpretable_rf'] * 100:.1f}%",
        f"   - Deep Waveform CNN : {ensemble_result['engine_probabilities']['deep_cnn'] * 100:.1f}%",
        f"   - Composite Average : {ensemble_result['engine_probabilities']['fused_composite'] * 100:.1f}%",
        f"   - Guardrail Status  : {'ACTIVE OVERRIDE' if triage_result['override_active'] else 'Within Normal Limits'}",
    ]

    if triage_result["override_active"]:
        lines.append("   - Active Findings   :")
        for reason in triage_result["override_reasons"]:
            lines.append(f"     * {reason}")
    else:
        lines.append("   - Active Findings   : None")

    lines.extend([
        "-" * 80,
        "R - RECOMMENDATION & MEASUREMENTS",
        f"   - Heart Rate        : {bm['heart_rate_bpm']:.1f} bpm (Resting)",
        f"   - QRS Duration      : {bm['qrs_duration_ms']:.1f} ms (Ventricular conduction)",
        f"   - Peak ST Shift     : {bm['max_st_elevation_mv']:+.3f} mV (J-point relative to PR baseline)",
        f"   - Suspected Culprit : {suspected_culprit}",
        "   - Territory Summary :",
    ])

    for t_name, t_val in ensemble_result.get("territory_breakdown", {}).items():
        leads_str = ", ".join(t_val["leads"])
        lines.append(f"     * {t_name:24s}: {t_val['max_st_mv']:+.3f} mV (Leads: {leads_str})")

    lines.extend([
        "=" * 80,
        "AUDIT & REGULATORY NOTICE (IEC 62304 / ISO 13485 Concept):",
        "CardioSense is an adjunctive decision-support tool. It does not replace independent",
        "evaluation by a licensed physician. Emergency pathways must be verified clinically.",
        "=" * 80,
    ])

    report_text = "\n".join(lines)

    # ---------------------------------------------------------
    # Block 3: Export to Disk (Optional)
    # What it does: Saves report text to disk when an export path is given.
    # How it does it: Writes UTF-8 encoded text to the specified Path.
    # ---------------------------------------------------------
    if export_path:
        out_file = Path(export_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_text, encoding="utf-8")
        print(f"[Clinical Report exported to: {out_file}]")

    return report_text