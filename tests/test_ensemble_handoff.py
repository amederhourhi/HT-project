
"""
CardioSense - Ensemble and Clinical Triage Integration Tests.

Verifies:
- Real ECG inference through the ensemble
- Clinical triage integration
- Prediction output compatibility
- Contiguous ST elevation detection
- Demographic thresholds
- Unknown-demographic fallback
"""

import copy
import sys
from pathlib import Path

import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.risk_fusion import CardioSenseEnsemble
from src.triage.triage_rules import ClinicalTriageEngine


RECORD = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "records100"
    / "00000"
    / "00001_lr"
)

engine = ClinicalTriageEngine()
checks_run = []


def check(label, condition, detail=""):
    assert condition, f"FAILED: {label} - {detail}"
    checks_run.append(label)
    print(f"[PASS] {label}")


def with_st(base, overrides):
    modified = copy.deepcopy(base)
    modified["st_by_lead"].update(overrides)
    return modified


def with_patient(base, age, sex):
    modified = copy.deepcopy(base)
    modified["patient"] = {
        "age": age,
        "sex": sex,
    }
    return modified


def run_tests():
    print("\n=== CARDIOSENSE INTEGRATION TEST ===\n")

    if not RECORD.with_suffix(".hea").exists():
        raise FileNotFoundError(f"Missing ECG record: {RECORD}")

    raw_sig, fields = wfdb.rdsamp(str(RECORD))

    ensemble = CardioSenseEnsemble()

    payload = ensemble.predict(
        raw_sig,
        fs=fields["fs"],
        lead_names=fields["sig_name"],
        age=55,
        sex="MALE",
    )

    # 1. Verify inference succeeded.
    check(
        "Ensemble returns successful prediction",
        payload.get("status") == "success",
        str(payload.get("reason", "")),
    )

    # 2. Verify clinical triage is integrated.
    check(
        "Statistical tier is available",
        "statistical_tier" in payload,
    )

    check(
        "Clinical triage output is available",
        isinstance(payload.get("clinical_triage"), dict),
    )

    clinical = payload["clinical_triage"]

    check(
        "Clinical triage contains final tier",
        "final_tier" in clinical,
    )

    check(
        "Clinical triage contains findings",
        isinstance(clinical.get("findings"), list),
    )

    check(
        "Final risk tier matches clinical decision",
        payload["risk_tier"] == clinical["final_tier"],
    )

    check(
        "Final action matches clinical decision",
        payload["triage_action"] == clinical["triage_action"],
    )

    check(
        "Clinical decision matches independent evaluation",
        clinical == engine.evaluate(payload),
    )

    print("\n=== CLINICAL TRIAGE OUTPUT ===")
    print("ML risk score:", payload["risk_score_pct"])
    print("Statistical tier:", payload["statistical_tier"])
    print("Final risk tier:", payload["risk_tier"])
    print("Triage action:", payload["triage_action"])
    print("Override active:", clinical["override_active"])
    print("Clinical findings:", clinical["findings"])

    # 3. Verify guardrail input contract.
    biomarkers = payload["clinical_biomarkers"]

    check(
        "Payload includes ST readings",
        isinstance(payload.get("st_by_lead"), dict),
    )

    check(
        "Payload includes patient demographics",
        isinstance(payload.get("patient"), dict),
    )

    check(
        "Payload includes QRS plausibility",
        "qrs_plausible" in biomarkers,
    )

    check(
        "All twelve ECG leads are trusted",
        len(payload["st_by_lead"]) == 12,
        str(sorted(payload["st_by_lead"])),
    )

    valid_leads = {
        "I", "II", "III", "AVR", "AVL", "AVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    }

    check(
        "Lead names are recognized",
        set(payload["st_by_lead"]) <= valid_leads,
    )

    # 4. Real-record baseline.
    verdict = engine.evaluate(payload)

    check(
        "Clean ECG does not activate emergency rules",
        verdict["final_tier"] != engine.TIER_ESCALATION,
        verdict["final_tier"],
    )

    # 5. Contiguous ST elevation positive control.
    stemi_payload = with_st(
        payload,
        {"V2": 0.30, "V3": 0.32},
    )

    stemi = engine.evaluate(stemi_payload)

    check(
        "Contiguous V2/V3 elevation triggers escalation",
        stemi["final_tier"] == engine.TIER_ESCALATION,
        stemi["final_tier"],
    )

    check(
        "ST elevation rule is identified",
        any(
            finding["rule_id"] == "G1_CONTIGUOUS_STE"
            for finding in stemi["findings"]
        ),
    )

    # 6. Isolated ST elevation negative control.
    lone_payload = with_st(
        payload,
        {"V2": 0.30},
    )

    lone = engine.evaluate(lone_payload)

    check(
        "Isolated V2 elevation does not trigger escalation",
        lone["final_tier"] != engine.TIER_ESCALATION,
        lone["final_tier"],
    )

    # 7. Demographic threshold checks.
    borderline = with_st(
        payload,
        {"V2": 0.18, "V3": 0.18},
    )

    male = engine.evaluate(
        with_patient(borderline, 45, "MALE")
    )

    female = engine.evaluate(
        with_patient(borderline, 45, "FEMALE")
    )

    check(
        "Male threshold does not activate at 0.18 mV",
        male["final_tier"] != engine.TIER_ESCALATION,
        male["final_tier"],
    )

    check(
        "Female threshold activates at 0.18 mV",
        female["final_tier"] == engine.TIER_ESCALATION,
        female["final_tier"],
    )

    # 8. Missing demographic fallback.
    unknown = engine.evaluate(
        with_patient(borderline, None, None)
    )

    check(
        "Unknown demographics use sensitive threshold",
        unknown["final_tier"] == engine.TIER_ESCALATION,
        unknown["final_tier"],
    )

    # 9. Guardrail escalation must not lower risk.
    low_risk_control = copy.deepcopy(stemi_payload)
    low_risk_control["risk_score_pct"] = 10.0

    escalation = engine.evaluate(low_risk_control)

    check(
        "Critical ECG finding overrides low ML score",
        escalation["final_tier"] == engine.TIER_ESCALATION,
        escalation["final_tier"],
    )

    check(
        "Override preserves original ML probability",
        escalation["baseline_ensemble_risk_pct"] == 10.0,
    )

    print("\n===================================")
    print(f"ALL {len(checks_run)} CHECKS PASSED")
    print("===================================")


if __name__ == "__main__":
    run_tests()
