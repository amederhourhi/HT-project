"""
CardioSense - Cascade Metrics
Measures the triage decision the system actually emits, not the probability
that feeds it.

Why this module exists: the engineering log historically reported specificity
for Engine A, Engine B and the fused ensemble in isolation. Those numbers
describe the statistical model only. `ClinicalTriageEngine` may override the
statistical risk with a deterministic rule, and it is that override - not the
probability - that decides which tier a patient is routed to. Any claim about
false alarms therefore has to be made against the post-override decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Ground-truth labels used across the cohort (mirrors `target_label` in the CSV)
LABEL_NORMAL = 0
LABEL_INFARCTION = 1
CLASS_NAMES = {LABEL_NORMAL: "normal", LABEL_INFARCTION: "infarction"}


@dataclass(frozen=True)
class RecordOutcome:
    """One record's deterministic triage verdict, paired with its ground truth."""

    ecg_id: int
    label: int
    strat_fold: int
    # True when at least one deterministic guardrail fired, i.e. the record
    # would be routed to emergency escalation irrespective of model risk.
    escalated: bool
    reasons: tuple[str, ...]
    # The measurements the rules read, kept so a reviewer can see *why* a
    # record escalated without re-running the pipeline.
    measurements: dict[str, float]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    The cohort slices here are small - the official test fold holds only 20
    healthy records - so a point estimate like "25%" carries no information on
    its own. Wilson is preferred over the normal approximation because it stays
    inside [0, 1] and behaves sensibly when the count is zero or total.
    """
    if total == 0:
        return (0.0, 1.0)

    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total

    centre = (proportion + z_squared / (2.0 * total)) / denominator
    spread = (z / denominator) * (
        proportion * (1.0 - proportion) / total + z_squared / (4.0 * total * total)
    ) ** 0.5

    return (max(0.0, centre - spread), min(1.0, centre + spread))


def summarize_guardrails(outcomes: Sequence[RecordOutcome]) -> dict:
    """
    Roll per-record verdicts up into escalation rates per ground-truth class.

    A false escalation is the clinical failure mode this report is built to
    expose: a cath-lab activation raised on a healthy ECG. `unescalated_
    infarctions` is deliberately *not* called a miss - the deterministic layer
    only has to catch what the statistical layer cannot, and the model half of
    that judgement needs checkpoints.
    """
    per_class: dict[str, dict] = {}

    for label, class_name in CLASS_NAMES.items():
        group = [outcome for outcome in outcomes if outcome.label == label]
        hits = [outcome for outcome in group if outcome.escalated]
        low, high = wilson_interval(len(hits), len(group))

        per_class[class_name] = {
            "n": len(group),
            "escalated": len(hits),
            "rate": len(hits) / len(group) if group else 0.0,
            "ci95_low": low,
            "ci95_high": high,
        }

    return {
        "n_records": len(outcomes),
        "per_class": per_class,
        "false_escalations": [
            outcome for outcome in outcomes if outcome.escalated and outcome.label == LABEL_NORMAL
        ],
        "unescalated_infarctions": [
            outcome
            for outcome in outcomes
            if not outcome.escalated and outcome.label == LABEL_INFARCTION
        ],
    }


def summary_to_jsonable(summary: dict) -> dict:
    """Convert dataclass outcomes into plain dicts so the summary can be dumped."""
    jsonable = dict(summary)
    for key in ("false_escalations", "unescalated_infarctions"):
        jsonable[key] = [
            {
                "ecg_id": outcome.ecg_id,
                "strat_fold": outcome.strat_fold,
                "reasons": list(outcome.reasons),
                "measurements": outcome.measurements,
            }
            for outcome in summary[key]
        ]
    return jsonable
