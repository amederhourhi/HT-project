"""
CardioSense - Guardrail Layer Scorer
Runs the deterministic triage rules over a whole feature cohort and reports how
often each ground-truth class is escalated to emergency care.

This is the regression gate for all triage work. `src.triage.triage_rules` is
imported and executed directly rather than being reimplemented here, so the
figures below always describe the rules as shipped - the exact failure that the
cascade metrics were written to prevent.
"""

import sys
from pathlib import Path

# Add project root to the Python search path so `src` resolves when run as a script
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import argparse
import json

import pandas as pd

from src.evaluation.triage_metrics import (
    RecordOutcome,
    summary_to_jsonable,
    summarize_guardrails,
)
from src.triage.triage_rules import ClinicalTriageEngine


# ---------------------------------------------------------
# Block 1: Adapt a feature row into the engine's input shape
# ---------------------------------------------------------
def build_triage_input(row: pd.Series) -> dict:
    """
    Reconstruct the slice of an ensemble result that the guardrails read.

    `ClinicalTriageEngine.evaluate()` touches the trusted per-lead ST map, the
    patient demographics and the rhythm/conduction biomarkers. One row of
    `clinical_features.csv` carries all of them, which means the deterministic
    layer can be scored over a whole cohort with no model checkpoints loaded - the
    reason this audit is cheap enough to re-run on every change to the rules.

    Baseline risk is pinned to zero on purpose. With no statistical component the
    only escalations left standing are the ones the rules themselves cause.
    """
    # A lead missing from `leads_trusted` was never measured, and stays missing
    # here: handing the engine a 0.0 would let a delineation failure read as a
    # normal ST segment in that lead.
    trusted = {lead for lead in str(row["leads_trusted"]).split(",") if lead}

    return {
        "risk_score_pct": 0.0,
        "patient": {"age": row.get("age"), "sex": row.get("sex")},
        "st_by_lead": {lead: float(row[f"st_{lead.upper()}"]) for lead in trusted},
        "clinical_biomarkers": {
            "heart_rate_bpm": float(row["heart_rate_bpm"]),
            "qrs_duration_ms": float(row["qrs_duration_ms"]),
            # pandas reads the CSV's True/False column as bool. Comparing the text
            # guards against that column degrading to object dtype, where an empty
            # cell would make the string "False" - and every string is truthy.
            "qrs_plausible": str(row["qrs_plausible"]).lower() == "true",
        },
    }


def score_row(engine: ClinicalTriageEngine, row: pd.Series) -> RecordOutcome:
    """Run one cohort record through the shipped rules and keep the evidence."""
    triage_input = build_triage_input(row)
    verdict = engine.evaluate(triage_input)

    # Escalation is the tier, not the presence of a finding: a REVIEW-only record
    # (a blocked QRS, say) is handed to a clinician, not to the cath lab, and it
    # must not be charged against the false-alarm budget that a REVIEW finding is
    # specifically there to protect.
    escalated = verdict["final_tier"] == ClinicalTriageEngine.TIER_ESCALATION

    return RecordOutcome(
        ecg_id=int(row["ecg_id"]),
        label=int(row["target_label"]),
        strat_fold=int(row["strat_fold"]),
        escalated=escalated,
        rule_ids=tuple(finding["rule_id"] for finding in verdict["findings"]),
        reasons=tuple(verdict["override_reasons"]),
        measurements={
            "st_anterior_max": float(row["st_anterior_max"]),
            "st_inferior_max": float(row["st_inferior_max"]),
            "st_lateral_max": float(row["st_lateral_max"]),
            "heart_rate_bpm": float(row["heart_rate_bpm"]),
            "qrs_duration_ms": float(row["qrs_duration_ms"]),
            # How much of the tracing was readable at all. An escalation on two
            # trusted leads and one on ten are different amounts of evidence.
            "leads_trusted": float(len(triage_input["st_by_lead"])),
            "age": float(row["age"]) if pd.notnull(row.get("age")) else float("nan"),
            "sex": float(row["sex"]) if pd.notnull(row.get("sex")) else float("nan"),
        },
    )


# ---------------------------------------------------------
# Block 2: Reporting
# ---------------------------------------------------------
def print_summary(scope: str, outcomes: list[RecordOutcome]) -> dict:
    """Print the escalation table and the individual false alarms, return summary."""
    summary = summarize_guardrails(outcomes)

    # Records a guardrail pulled out of routine follow-up without escalating. They
    # are not in the escalation budget by design, but they are the reason the wide
    # QRS rule still exists, so the count belongs on the page.
    review_only = [
        outcome for outcome in outcomes if outcome.rule_ids and not outcome.escalated
    ]

    print("\n" + "=" * 74)
    print(f"   CARDIOSENSE DETERMINISTIC GUARDRAIL LAYER - ESCALATION AUDIT ({scope})")
    print("=" * 74)
    print(f"Records scored                     : {summary['n_records']}")
    print(f"Rules under test                   : src/triage/triage_rules.py (as shipped)")
    print(f"Cut-points source                  : src/triage/leads.py (Fourth UDMI 2018)")
    print("-" * 74)
    print(f"{'Ground-truth class':<22}{'n':>6}{'escalated':>12}{'rate':>9}   {'95% CI (Wilson)':<20}")
    print("-" * 74)

    for class_name, stats in summary["per_class"].items():
        interval = f"[{stats['ci95_low'] * 100:.1f}%, {stats['ci95_high'] * 100:.1f}%]"
        print(
            f"{class_name:<22}{stats['n']:>6}{stats['escalated']:>12}"
            f"{stats['rate'] * 100:>8.1f}%   {interval:<20}"
        )

    print("-" * 74)
    print("Fires per guardrail rule (records, by ground truth)")
    print(f"{'rule_id':<24}{'records':>9}{'normal':>9}{'infarction':>13}")
    for rule_id, counts in sorted(summary["per_rule"].items()):
        print(
            f"{rule_id:<24}{counts['records']:>9}{counts['normal']:>9}"
            f"{counts['infarction']:>13}"
        )

    false_alarms = summary["false_escalations"]
    print("-" * 74)
    print(
        f"Healthy records escalated to TIER_3 : {len(false_alarms)} / "
        f"{summary['per_class']['normal']['n']}"
    )
    print("Each of these is a cath-lab activation raised on a normal ECG.")
    print(
        f"Healthy records sent to review only : "
        f"{sum(1 for o in review_only if o.label == 0)} / "
        f"{summary['per_class']['normal']['n']}"
    )
    print("These are clinician reads, not activations, and are not charged above.")

    if false_alarms:
        print("-" * 74)
        print(
            f"{'ecg_id':>8} {'fold':>5} {'ant':>8} {'inf':>8} {'lat':>8} "
            f"{'HR':>7} {'QRS':>7} {'leads':>6}"
        )
        print("-" * 74)
        for outcome in false_alarms:
            values = outcome.measurements
            print(
                f"{outcome.ecg_id:>8} {outcome.strat_fold:>5}"
                f"{values['st_anterior_max']:>8.3f}{values['st_inferior_max']:>8.3f}"
                f"{values['st_lateral_max']:>8.3f}{values['heart_rate_bpm']:>7.1f}"
                f"{values['qrs_duration_ms']:>7.1f}{values['leads_trusted']:>6.0f}"
            )
            for reason in outcome.reasons:
                print(f"         -> {reason}")

    print("=" * 74)
    return summary


# ---------------------------------------------------------
# Block 3: Entry point
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the deterministic guardrail layer over a feature cohort."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root / "data" / "processed" / "clinical_features.csv",
        help="Feature CSV produced by src/features/batch_extract.py",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "train", "test"],
        default="all",
        help="Cohort slice to score, using the official PTB-XL stratified folds",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the machine-readable summary for CI use",
    )
    parser.add_argument(
        "--fail-above",
        type=float,
        default=None,
        help="Exit non-zero if the healthy-class escalation rate exceeds this fraction "
        "(e.g. 0.05 for 5%%). Turns the audit into a regression gate.",
    )
    return parser.parse_args()


def select_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Filter to the requested fold slice (folds 1-8 train, 9-10 test)."""
    if scope == "train":
        return df[df["strat_fold"] <= 8]
    if scope == "test":
        return df[df["strat_fold"] > 8]
    return df


def main() -> int:
    args = parse_args()

    if not args.features.exists():
        print(f"[ERROR] Feature CSV not found: {args.features}")
        print("        Run `python src/features/batch_extract.py` first.")
        return 2

    df = select_scope(pd.read_csv(args.features), args.scope)
    engine = ClinicalTriageEngine()
    outcomes = [score_row(engine, row) for _, row in df.iterrows()]

    summary = print_summary(args.scope, outcomes)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"scope": args.scope, **summary_to_jsonable(summary)}, indent=2),
            encoding="utf-8",
        )
        print(f"[Saved machine-readable summary to: {args.json_out}]")

    if args.fail_above is not None:
        normal_rate = summary["per_class"]["normal"]["rate"]
        if normal_rate > args.fail_above:
            print(
                f"\n[REGRESSION] Healthy escalation rate {normal_rate * 100:.1f}% "
                f"exceeds the {args.fail_above * 100:.1f}% budget."
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
