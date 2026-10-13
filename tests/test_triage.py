"""
CardioSense - Deterministic Guardrail Verification Test
Validates the triage rules themselves: contiguity, demographic cut-points,
severity routing, and the treatment of leads that were never measured.

Why synthetic inputs instead of records: the guardrails are a pure function of
the ensemble payload, so their logic can be pinned exactly - including the
branches no record in this cohort exercises - with no model checkpoints and no
WFDB files. End-to-end behaviour on real records is measured over the whole
cohort by `src/evaluation/score_guardrails.py`, which is the better instrument
for that question anyway: one record proves an anecdote, a cohort proves a rate.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.triage.triage_rules import ESCALATE, REVIEW, ClinicalTriageEngine

engine = ClinicalTriageEngine()
checks_run = []


def payload(st_by_lead, age=None, sex=None, hr=72.0, qrs=95.0, risk=0.0):
    """
    Assemble the slice of an ensemble result the guardrails read.

    `st_by_lead` is passed through untouched: whether a lead is present at all is
    part of the contract under test, since an unmeasured lead must be absent from
    the map rather than present as zero.
    """
    return {
        "risk_score_pct": risk,
        "patient": {"age": age, "sex": sex},
        "st_by_lead": dict(st_by_lead),
        "clinical_biomarkers": {
            "heart_rate_bpm": hr,
            "qrs_duration_ms": qrs,
            "qrs_plausible": True,
        },
    }


def check(label, condition, detail=""):
    """Assert with the evidence printed, so a failure reads as a finding."""
    assert condition, f"FAILED: {label}{(' - ' + detail) if detail else ''}"
    checks_run.append(label)
    print(f"[PASS] {label}")


# ---------------------------------------------------------
# Block 1: Contiguity - the criterion needs two neighbouring leads
# ---------------------------------------------------------
# A lone markedly elevated V2 is common in healthy young men (high take-off /
# benign early repolarization) and is not a STEMI. This is the defect the old
# regional-maximum rule was built without.
verdict = engine.evaluate(payload({"V2": 0.30, "V3": 0.04, "V4": 0.02}, age=65, sex="MALE"))
check("lone V2 at +0.30 mV does not escalate",
      verdict["final_tier"] == engine.TIER_ROUTINE, verdict["final_tier"])
check("lone V2 raises no finding at all", verdict["findings"] == [])

# The same tracing, with its neighbour elevated, meets the definition.
verdict = engine.evaluate(payload({"V2": 0.30, "V3": 0.25}, age=65, sex="MALE"))
check("V2/V3 contiguous pair escalates", verdict["final_tier"] == engine.TIER_ESCALATION)
check("the pair reports the STEMI rule id",
      [f["rule_id"] for f in verdict["findings"]] == ["G1_CONTIGUOUS_STE"])
check("the pair names the LAD territory",
      verdict["findings"][0]["territory"] == "Anterior / Septal (LAD)")

# Outside V2-V3 the generic cut-point is 0.10 mV, which the old 0.20 mV
# precordial threshold missed entirely.
verdict = engine.evaluate(payload({"V4": 0.12, "V5": 0.11}, age=65, sex="MALE"))
check("V4/V5 pair fires on the generic 0.10 mV cut-point",
      verdict["final_tier"] == engine.TIER_ESCALATION)

# Two elevated leads that are not contiguous do not meet it.
verdict = engine.evaluate(payload({"V2": 0.40, "V3": 0.02, "V6": 0.40}, age=65, sex="MALE"))
check("non-contiguous V2 + V6 does not escalate", verdict["findings"] == [])

# ---------------------------------------------------------
# Block 2: Demographic cut-points
# ---------------------------------------------------------
# The three V2/V3 thresholds are 0.15 (women), 0.20 (men 40+) and 0.25 (men <40).
# Two ST levels separate all three: +0.18 mV clears only the women's criterion,
# and +0.22 mV clears the women's and the men-40+ criteria but not the young
# male's. One tracing, three patients, three answers.
below_all_male = {"V2": 0.18, "V3": 0.18}
check("women: +0.18 mV in V2/V3 meets the 0.15 mV criterion",
      engine.evaluate(payload(below_all_male, age=29, sex="FEMALE"))["final_tier"] == engine.TIER_ESCALATION)
check("men 40+: +0.18 mV stays under the 0.20 mV criterion",
      engine.evaluate(payload(below_all_male, age=65, sex="MALE"))["final_tier"] == engine.TIER_ROUTINE)
check("men <40: +0.18 mV stays under the 0.25 mV criterion",
      engine.evaluate(payload(below_all_male, age=34, sex="MALE"))["final_tier"] == engine.TIER_ROUTINE)

past_middle = {"V2": 0.22, "V3": 0.22}
check("men 40+: +0.22 mV meets the 0.20 mV criterion",
      engine.evaluate(payload(past_middle, age=65, sex="MALE"))["final_tier"] == engine.TIER_ESCALATION)
check("men <40: +0.22 mV still stays under the 0.25 mV criterion",
      engine.evaluate(payload(past_middle, age=34, sex="MALE"))["final_tier"] == engine.TIER_ROUTINE)

# Unknown demographics must not relax a threshold: the young-male exemption is the
# only relaxation on offer, and it requires knowing the patient is a young man.
verdict = engine.evaluate(payload(below_all_male))
check("unknown age/sex falls back to the most sensitive cut-point",
      verdict["final_tier"] == engine.TIER_ESCALATION)
check("the fallback is stated in the finding rather than hidden",
      "age/sex unknown" in verdict["override_reasons"][0], verdict["override_reasons"][0])

# PTB-XL encodes sex as 0.0/1.0 floats, which the engine normalises at its boundary.
check("cohort sex codes decode (0.0 = male, 1.0 = female)",
      engine.evaluate(payload(below_all_male, age=34, sex=0.0))["final_tier"] == engine.TIER_ROUTINE
      and engine.evaluate(payload(below_all_male, age=34, sex=1.0))["final_tier"] == engine.TIER_ESCALATION)

# ---------------------------------------------------------
# Block 3: An unmeasured lead is unknown, not normal
# ---------------------------------------------------------
# A lead whose delineation failed has to be absent from the map. Delivered as 0.0
# it would silently fail the criterion and the record would read as a normal ECG.
verdict = engine.evaluate(payload({"V2": 0.40}, age=65, sex="MALE"))
check("a missing partner lead does not escalate", verdict["findings"] == [])

# ---------------------------------------------------------
# Block 4: Severity routing - ESCALATE versus REVIEW
# ---------------------------------------------------------
# A blocked QRS makes the ST analysis unreliable. That is not an occlusion, so it
# must not activate a cath lab; and it is not a normal ECG, so it must not be
# discharged either.
verdict = engine.evaluate(payload({}, qrs=156.0, risk=12.0))
check("wide QRS alone does not escalate", verdict["final_tier"] != engine.TIER_ESCALATION)
check("wide QRS alone lifts a low-risk patient out of routine follow-up",
      verdict["final_tier"] == engine.TIER_REVIEW)
check("wide QRS reports REVIEW severity",
      verdict["highest_severity"] == REVIEW
      and verdict["findings"][0]["rule_id"] == "G3_WIDE_QRS")
check("a REVIEW finding still counts as an active override", verdict["override_active"] is True)

# REVIEW never argues a high statistical risk downward.
verdict = engine.evaluate(payload({}, qrs=156.0, risk=80.0))
check("wide QRS on a high risk score stays escalated",
      verdict["final_tier"] == engine.TIER_ESCALATION)
check("no ESCALATE rule means no upward risk floor",
      verdict["effective_risk_pct"] == 80.0)

# An ESCALATE rule outranks a clean conduction interval.
verdict = engine.evaluate(payload({"II": 0.20, "AVF": 0.18}, qrs=90.0, risk=5.0))
check("inferior contiguous pair escalates at the 0.10 mV limb cut-point",
      verdict["highest_severity"] == ESCALATE
      and verdict["final_tier"] == engine.TIER_ESCALATION)

# ---------------------------------------------------------
# Block 5: Rhythm guardrails
# ---------------------------------------------------------
check("severe tachycardia escalates",
      engine.evaluate(payload({}, hr=145.0))["final_tier"] == engine.TIER_ESCALATION)
verdict = engine.evaluate(payload({}, hr=35.0))
check("severe bradycardia escalates", verdict["final_tier"] == engine.TIER_ESCALATION)
check("bradycardia is identified by its own rule id",
      verdict["findings"][0]["rule_id"] == "G2_SEVERE_BRADYCARDIA")
check("a rate inside physiological bounds raises nothing",
      engine.evaluate(payload({}, hr=118.0))["findings"] == [])

# ---------------------------------------------------------
# Block 6: No finding, no override
# ---------------------------------------------------------
verdict = engine.evaluate(payload({"V2": 0.05, "V3": 0.04}, age=45, sex="FEMALE", risk=8.0))
check("a normal tracing at low risk is TIER_1_ROUTINE",
      verdict["final_tier"] == engine.TIER_ROUTINE and not verdict["override_active"])
check("with no finding the risk passes through unchanged",
      verdict["effective_risk_pct"] == 8.0)

print(f"\nAll {len(checks_run)} deterministic guardrail assertions passed.")
