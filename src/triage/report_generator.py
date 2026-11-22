
"""
CardioSense - Research SBAR Report Generator.

Produces an auditable research report using:
    - Legacy V1 model outputs
    - Deterministic research triage findings
    - Shared ST measurement presentation contract
    - Feature-quality and provenance warnings
    - Software revision and UTC timestamp

IMPORTANT:
    All displayed ST measurements come from
    build_st_measurement_view().

    Historical classifier aggregate placeholders
    are NEVER displayed as measured ECG values.

    Measurement presentation does not modify:
        - RF features
        - CNN inputs
        - Model probabilities
        - Clinical guardrail decisions
        - Final triage routing

Research prototype only.
Not clinically validated or approved for clinical use.
"""

from datetime import datetime, timezone
from pathlib import Path
import subprocess

from src.triage.st_measurement_view import (
    build_st_measurement_view,
)


# =========================================================
# 1. Audit metadata
# =========================================================

def _get_commit_sha() -> str:
    """Return the current Git revision when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode != 0:
            return "unknown"

        return result.stdout.strip() or "unknown"

    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


# =========================================================
# 2. Compatibility helpers
# =========================================================

def _finite_st_readings(ensemble_result):
    """Return the same reportable readings as the API.

    The shared measurement view decides availability.
    This helper does not implement a second policy.
    """

    view = build_st_measurement_view(ensemble_result)

    return {
        lead: item["st_mv"]
        for lead, item in view["per_lead"].items()
        if item["available"]
    }


def _available_st_leads(ensemble_result):
    """Return standard lead names with reportable readings."""

    return list(_finite_st_readings(ensemble_result))


# =========================================================
# 3. Format quality warnings
# =========================================================

def _format_quality_warnings(quality):
    """Display existing advisory quality assessments.

    The quality handoff is metadata-only.
    It does not alter the triage decision.
    """

    if quality.get("status") != "assessed":
        return [
            (
                "   - Quality Review    : "
                "Unavailable (feature-quality metadata missing)"
            ),
            (
                "   - Quality Warning   : "
                "Measurement reliability has not been established"
            ),
        ]

    review_features = sorted(set(
        quality.get("review_required_features") or []
    ))

    unavailable_features = sorted(set(
        quality.get("unavailable_features") or []
    ))

    lines = [
        (
            "   - Quality Review    : "
            "Advisory assessment available"
        ),
        (
            "   - Quality Warning   : "
            "Independent clinical measurement validation "
            "not established"
        ),
        (
            "   - Review Required   : "
            + (
                ", ".join(review_features)
                if review_features
                else "None flagged by available heuristics"
            )
        ),
        (
            "   - Features Unavailable: "
            + (
                ", ".join(unavailable_features)
                if unavailable_features
                else "None reported by quality contract"
            )
        ),
    ]

    assessments = (
        quality.get("st_finding_assessments") or []
    )

    flagged_pairs = [
        pair
        for pair in assessments
        if (
            isinstance(pair, dict)
            and pair.get("known_quality_concern")
        )
    ]

    if flagged_pairs:
        lines.append(
            "   - ST Quality Concern: "
            "Known quality concern in triggered ST finding"
        )

        for pair in flagged_pairs:
            pair_name = "/".join(
                str(lead)
                for lead in pair.get("leads", [])
            ) or "Unknown pair"

            flagged_leads = (
                pair.get("flagged_leads") or []
            )

            lines.append(
                f"     * ST pair {pair_name}: "
                "flagged leads "
                + ", ".join(
                    str(lead) for lead in flagged_leads
                )
            )

            lead_quality = (
                pair.get("lead_quality") or {}
            )

            for lead in flagged_leads:
                item = lead_quality.get(lead) or {}

                reasons = item.get("reasons") or []

                reason_text = (
                    ", ".join(map(str, reasons))
                    if reasons
                    else "reason not recorded"
                )

                lines.append(
                    f"       - {lead}: "
                    f"{item.get('status', 'unknown')} "
                    f"({reason_text})"
                )

    elif quality.get("st_finding_quality_concern"):
        lines.append(
            "   - ST Quality Concern: "
            "Reported without detailed lead information"
        )

    else:
        lines.append(
            "   - ST Quality Concern: "
            "No triggered ST pair flagged by available "
            "heuristics; clinical validity not established"
        )

    return lines


# =========================================================
# 4. Format ST measurements using the shared contract
# =========================================================

def _format_st_measurements(view):
    """Format global and territory measurements.

    All values originate from the same presentation
    contract returned by the API.

    Never read territory_breakdown or historical
    classifier aggregates for displayed measurements.
    """

    lines = []

    # -----------------------------------------------------
    # A. Global maximum ST reading
    # -----------------------------------------------------

    peak = view["peak_st"]

    if peak["available"]:
        maximum = peak["max_st_mv"]
        sources = ", ".join(
            peak["source_leads"]
        )

        lines.append(
            f"   - Peak ST Shift     : {maximum:+.3f} mV "
            f"(maximum of available lead readings: {sources})"
        )
    else:
        lines.append(
            "   - Peak ST Shift     : "
            "Unavailable (no reportable lead measurements)"
        )

    # -----------------------------------------------------
    # B. Territory-specific measurements
    # -----------------------------------------------------

    lines.append("   - Territory Summary :")

    for territory_name, territory in (
        view["territories"].items()
    ):
        available_count = territory[
            "available_leads"
        ]

        total_count = territory[
            "total_leads"
        ]

        if not territory["available"]:
            lines.append(
                f"     * {territory_name}: Unavailable "
                f"(0/{total_count} territory leads measured)"
            )
            continue

        maximum = territory["max_st_mv"]

        source_leads = ", ".join(
            territory["source_leads"]
        )

        lines.append(
            f"     * {territory_name}: {maximum:+.3f} mV "
            f"(from {source_leads}; "
            f"{available_count}/{total_count} "
            "territory leads available)"
        )

        if territory["coverage"] == "partial":
            lines.append(
                "       - Coverage: Partial; territory "
                "maximum does not represent all leads"
            )

    # -----------------------------------------------------
    # C. Aggregate provenance inconsistencies
    # -----------------------------------------------------

    for concern in view[
        "aggregate_provenance_concerns"
    ]:
        feature = concern["feature"]
        reason = concern["reason"]

        lines.append(
            "   - Aggregate Provenance: "
            f"{feature} flagged ({reason}); "
            "legacy feature value not displayed "
            "as measurement"
        )

    # -----------------------------------------------------
    # D. Interpretation limitation
    # -----------------------------------------------------

    lines.append(
        "   - ST Interpretation : "
        "Available readings are algorithmic estimates, "
        "not independently validated"
    )

    return lines


# =========================================================
# 5. Generate complete SBAR research report
# =========================================================

def generate_clinical_report(
    patient_id,
    ensemble_result,
    triage_result,
    export_path=None,
) -> str:
    """Generate a research report without changing results."""

    biomarkers = ensemble_result["clinical_biomarkers"]

    patient = ensemble_result.get("patient") or {}

    age = patient.get("age")
    sex = patient.get("sex")

    age_str = (
        f"{age} years"
        if age is not None
        else "Unknown"
    )

    sex_str = (
        str(sex)
        if sex is not None
        else "Unknown"
    )

    # -----------------------------------------------------
    # A. Build the same measurement view as the API
    # -----------------------------------------------------

    view = build_st_measurement_view(
        ensemble_result
    )

    missing_leads = view["missing_leads"]

    quality = (
        triage_result.get("quality_assessment") or {}
    )

    quality_status = quality.get(
        "status",
        "not_available"
    )

    # -----------------------------------------------------
    # B. Interpret configured guardrail status
    # -----------------------------------------------------

    override_active = triage_result[
        "override_active"
    ]

    if override_active:
        guardrail_status = (
            "Configured guardrail findings detected"
        )
    else:
        guardrail_status = (
            "No configured guardrail triggered; "
            "ECG normality not established"
        )

    finding_territories = [
        finding.get("territory")
        for finding in triage_result.get(
            "findings", []
        )
        if finding.get("territory")
    ]

    finding_territory = (
        finding_territories[0]
        if finding_territories
        else (
            "No qualifying contiguous ST pair "
            "identified in available readings"
        )
    )

    probabilities = ensemble_result[
        "engine_probabilities"
    ]

    generated = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    # -----------------------------------------------------
    # C. Report header and Situation
    # -----------------------------------------------------

    lines = [
        "=" * 80,
        "                 CARDIOSENSE RESEARCH ECG REPORT",
        "=" * 80,

        f"REPORT GENERATED       : {generated}",
        f"SOFTWARE BUILD (COMMIT): {_get_commit_sha()}",
        f"PATIENT IDENTIFIER     : {patient_id}",

        "-" * 80,
        "S - SITUATION",

        (
            "   - Final Triage Tier : "
            f"[{triage_result['final_tier']}]"
        ),
        (
            "   - Recommended Action: "
            f"{triage_result['triage_action']}"
        ),
        (
            "   - Research Model Score: "
            f"{triage_result['effective_risk_pct']:.1f}%"
        ),

        # -------------------------------------------------
        # D. Background and signal quality
        # -------------------------------------------------

        "-" * 80,
        "B - BACKGROUND & SIGNAL INTEGRITY",

        (
            "   - Demographics      : "
            f"Age {age_str} | Sex: {sex_str}"
        ),
        (
            "   - Signal Quality    : "
            f"{biomarkers['overall_sqi']:.3f} / 1.000 "
            "(Algorithmic SQI; not clinical validation)"
        ),
        (
            "   - ST Readings       : "
            f"{view['available_lead_count']}/12 available "
            "for report display"
        ),
        (
            "   - ST Readings Missing: "
            + (
                ", ".join(missing_leads)
                if missing_leads
                else "None"
            )
        ),
        (
            "   - Measurement View  : "
            f"{view['version']}"
        ),
        (
            "   - Quality Metadata  : "
            f"{quality_status}"
        ),
        (
            "   - Measurement Validity: "
            "Not independently established"
        ),

        # -------------------------------------------------
        # E. Assessment
        # -------------------------------------------------

        "-" * 80,
        "A - ASSESSMENT (AI & DETERMINISTIC SAFEGUARDS)",

        (
            "   - Interpretable RF  : "
            f"{probabilities['interpretable_rf'] * 100:.1f}%"
        ),
        (
            "   - Deep Waveform CNN : "
            f"{probabilities['deep_cnn'] * 100:.1f}%"
        ),
        (
            "   - Composite Average : "
            f"{probabilities['fused_composite'] * 100:.1f}%"
        ),
        (
            "   - Guardrail Status  : "
            f"{guardrail_status}"
        ),
    ]

    # -----------------------------------------------------
    # F. Quality concerns
    # -----------------------------------------------------

    lines.extend(
        _format_quality_warnings(quality)
    )

    # -----------------------------------------------------
    # G. Active clinical guardrail findings
    # -----------------------------------------------------

    if override_active:
        lines.append(
            "   - Active Findings   :"
        )

        for reason in triage_result.get(
            "override_reasons", []
        ):
            lines.append(
                f"     * {reason}"
            )
    else:
        lines.append(
            "   - Active Findings   : "
            "None detected by configured rules"
        )

    # -----------------------------------------------------
    # H. Clinical measurements
    # -----------------------------------------------------

    lines.extend([
        "-" * 80,
        "R - RECOMMENDATION & MEASUREMENTS",

        (
            "   - Heart Rate        : "
            f"{biomarkers['heart_rate_bpm']:.1f} bpm"
        ),
        (
            "   - QRS Duration      : "
            f"{biomarkers['qrs_duration_ms']:.1f} ms"
        ),
        (
            "   - ST Finding Territory: "
            f"{finding_territory}"
        ),
    ])

    # Every displayed ST measurement uses the
    # shared API measurement-view contract.
    lines.extend(
        _format_st_measurements(view)
    )

    # -----------------------------------------------------
    # I. Distinguish legacy routing and presentation
    # -----------------------------------------------------

    lines.append(
        "   - Routing Basis     : "
        "Legacy V1 rule inputs; measurement display "
        "exclusions do not modify triage decisions"
    )

    # -----------------------------------------------------
    # J. Research limitations
    # -----------------------------------------------------

    lines.extend([
        "=" * 80,

        "RESEARCH LIMITATIONS & AUDIT NOTICE:",

        (
            "CardioSense is an experimental research "
            "decision-support prototype."
        ),
        (
            "Model scores and rule-based triage categories "
            "are not clinically validated."
        ),
        (
            "Available ECG measurements are algorithmic "
            "estimates, not independently validated."
        ),
        (
            "Missing or quality-excluded ST measurements "
            "are not represented as zero-valued readings."
        ),
        (
            "The absence of triggered guardrails does not "
            "establish a normal ECG or exclude disease."
        ),
        (
            "The legacy ensemble and clinical routing "
            "are unchanged by measurement presentation."
        ),
        (
            "This report must not independently determine "
            "diagnosis, disposition, or emergency treatment."
        ),

        "=" * 80,
    ])

    # -----------------------------------------------------
    # K. Assemble complete report
    # -----------------------------------------------------

    report_text = "\n".join(lines)

    # -----------------------------------------------------
    # L. Optional UTF-8 export
    # -----------------------------------------------------

    if export_path:
        output_file = Path(export_path)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file.write_text(
            report_text,
            encoding="utf-8"
        )

        print(
            f"[Clinical Report exported to: {output_file}]"
        )

    return report_text