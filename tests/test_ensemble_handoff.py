"""
CardioSense - Ensemble-to-Guardrail Handoff Verification Test

What this pins that nothing else does
-------------------------------------
`test_triage.py` feeds the guardrails a payload assembled by hand, and
`score_guardrails.py` assembles one from a row of `clinical_features.csv`. Both
therefore assume that `CardioSenseEnsemble.predict()` builds the same shape the
guardrails read. That assumption is exactly what broke during the ST measurement
fix, and no existing test would have caught it.

Here a real WFDB record goes through the real ensemble, and the dictionary that
comes out is handed straight to the real triage engine - no adapter in between.

Why the injected controls matter
--------------------------------
The pinned cohort contains no acute STEMI, so nothing in it can show that an
escalation is even reachable through the production payload. A green cohort audit
is consistent with the guardrails being silently disconnected. The controls below
prove the wiring carries signal in both directions: contiguity, the demographic
cut-points, and the unknown-demographic fallback all still bite once the numbers
travel through `predict()`'s output shape.
"""

import copy
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import wfdb

from src.triage.risk_fusion import CardioSenseEnsemble
from src.triage.triage_rules import ClinicalTriageEngine

engine = ClinicalTriageEngine()
checks_run = []

# A clean, twelve-lead, gate-passing record from the tracked samples.
RECORD = project_root / "data" / "samples" / "records100" / "00000" / "00001_lr"


def check(label, condition, detail=""):
    """Assert with the evidence printed, so a failure reads as a finding."""
    assert condition, f"FAILED: {label}{(' - ' + detail) if detail else ''}"
    checks_run.append(label)
    print(f"[PASS] {label}")


def with_st(base, overrides):
    """
    Return a copy of a real payload with selected per-lead ST readings replaced.

    Deep-copied on purpose: the controls below must not leak into each other
    through a shared `st_by_lead` dictionary.
    """
    tampered = copy.deepcopy(base)
    tampered["st_by_lead"].update(overrides)
    return tampered


def with_patient(base, age, sex):
    """Return a copy of a real payload carrying different demographics."""
    tampered = copy.deepcopy(base)
    tampered["patient"] = {"age": age, "sex": sex}
    return tampered


# ---------------------------------------------------------
# Block 1: Run a real record through the real ensemble
# ---------------------------------------------------------
print(f"Loading {RECORD.name} and running the full dual-engine path...")
raw_sig, fields = wfdb.rdsamp(str(RECORD))

ensemble = CardioSenseEnsemble()
payload = ensemble.predict(
    raw_sig,
    fs=fields["fs"],
    lead_names=fields["sig_name"],
    age=55,
    sex="MALE",
)

check("the ensemble returns a scored result, not a quality rejection",
      payload.get("status") == "success", payload.get("reason", ""))

print(f"\n  risk {payload['risk_score_pct']}%  |  ensemble tier {payload['risk_tier']}"
      f"  |  {len(payload['st_by_lead'])} trusted leads"
      f"  |  ST {min(payload['st_by_lead'].values()):+.3f}"
      f" to {max(payload['st_by_lead'].values()):+.3f} mV\n")

# ---------------------------------------------------------
# Block 2: The payload carries every key the guardrails read
# ---------------------------------------------------------
# These four keys are the whole contract. If predict() stops emitting any of
# them, the guardrails degrade quietly rather than failing loudly, which is the
# worst possible failure mode for a safety layer.
check("payload carries st_by_lead", isinstance(payload.get("st_by_lead"), dict))
check("payload carries patient demographics", "patient" in payload)
check("payload carries qrs_plausible", "qrs_plausible" in payload["clinical_biomarkers"])
check("a clean record yields all twelve leads as trusted",
      len(payload["st_by_lead"]) == 12, str(sorted(payload["st_by_lead"])))
check("trusted lead names are all recognised leads",
      set(payload["st_by_lead"]) <= set(
          {"I", "II", "III", "AVR", "AVL", "AVF",
           "V1", "V2", "V3", "V4", "V5", "V6"}))

# ---------------------------------------------------------
# Block 3: The untouched record does not activate
# ---------------------------------------------------------
verdict = engine.evaluate(payload)
check("an unmodified clean record raises no activation",
      verdict["final_tier"] != engine.TIER_ESCALATION, verdict["final_tier"])

# ---------------------------------------------------------
# Block 4: Positive control - contiguous elevation reaches TIER_3
# ---------------------------------------------------------
# +0.30/+0.32 mV in V2/V3 clears every demographic cut-point. If the guardrails
# are wired to the production payload at all, this must activate.
stemi = engine.evaluate(with_st(payload, {"V2": 0.30, "V3": 0.32}))
check("contiguous V2/V3 elevation activates through the production payload",
      stemi["final_tier"] == engine.TIER_ESCALATION, stemi["final_tier"])
check("the activation is attributed to the contiguous-lead rule",
      any(f["rule_id"] == "G1_CONTIGUOUS_STE" for f in stemi["findings"]),
      str([f["rule_id"] for f in stemi["findings"]]))

# ---------------------------------------------------------
# Block 5: Contiguity survives the handoff
# ---------------------------------------------------------
# The identical elevation in V2 alone has no partner lead, so it is a high
# take-off, not an occlusion.
lone = engine.evaluate(with_st(payload, {"V2": 0.30}))
check("the same elevation in V2 alone does not activate",
      lone["final_tier"] != engine.TIER_ESCALATION, lone["final_tier"])

# ---------------------------------------------------------
# Block 6: The demographic cut-points survive the handoff
# ---------------------------------------------------------
# +0.18 mV in V2/V3 sits between the female threshold (0.15) and the
# male >= 40 y threshold (0.20), so sex alone decides it.
borderline = with_st(payload, {"V2": 0.18, "V3": 0.18})
as_man = engine.evaluate(with_patient(borderline, 45, "MALE"))
as_woman = engine.evaluate(with_patient(borderline, 45, "FEMALE"))
check("0.18 mV in V2/V3 stays under the male >= 40 y cut-point",
      as_man["final_tier"] != engine.TIER_ESCALATION, as_man["final_tier"])
check("the identical reading crosses the female cut-point and activates",
      as_woman["final_tier"] == engine.TIER_ESCALATION, as_woman["final_tier"])

# ---------------------------------------------------------
# Block 7: Unknown demographics resolve conservatively
# ---------------------------------------------------------
# Missing demographics must never relax a criterion. With no age or sex the
# engine holds V2/V3 to the most sensitive threshold, so the record above must
# still activate.
as_unknown = engine.evaluate(with_patient(borderline, None, None))
check("unknown demographics fall back to the sensitive cut-point and activate",
      as_unknown["final_tier"] == engine.TIER_ESCALATION, as_unknown["final_tier"])

print(f"\nAll {len(checks_run)} ensemble-to-guardrail handoff assertions passed.")
