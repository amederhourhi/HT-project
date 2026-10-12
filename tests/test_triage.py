"""
CardioSense - Clinical Safety Guardrail & Triage Verification Test
Validates:
1. Patient 1 (Normal Control): TIER_1_ROUTINE without safety overrides.
2. Patient 8 (Anterior ST Elevation): Automatic SAFETY OVERRIDE to TIER_3_ESCALATION,
   protecting against statistical model underestimation.
Exports the finalized clinical SBAR report to docs/reports/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import wfdb

from src.triage.risk_fusion import CardioSenseEnsemble
from src.triage.triage_rules import ClinicalTriageEngine
from src.triage.report_generator import generate_clinical_report

# ---------------------------------------------------------
# Block 1: Initialize Pipeline
# ---------------------------------------------------------
print("Initializing CardioSense Safety-Guardrailed Triage Engine...")
ensemble = CardioSenseEnsemble()
triage_engine = ClinicalTriageEngine()

samples_dir = project_root / "data" / "samples" / "records100" / "00000"
reports_dir = project_root / "docs" / "reports"

test_cases = [
    ("00001_lr", "Patient 1 (Normal Control)"),
    ("00008_lr", "Patient 8 (Acute Anterior Infarction)"),
]

# ---------------------------------------------------------
# Block 2: Run End-to-End Triage & Validation
# ---------------------------------------------------------
for pid, label in test_cases:
    rec_path = samples_dir / pid
    sig, fields = wfdb.rdsamp(str(rec_path))

    # 1. Ensemble Risk Fusion
    ens_res = ensemble.predict(sig, fs=fields["fs"], lead_names=fields["sig_name"])

    # 2. Deterministic Safety Guardrail Evaluation
    tri_res = triage_engine.evaluate(ens_res)

    # 3. Generate Clinical SBAR Report
    report_file = reports_dir / f"patient_{pid}_triage_report.txt"
    report = generate_clinical_report(pid, ens_res, tri_res, export_path=report_file)

    print("\n" + report + "\n")

    # 4. Programmatic Assertions
    if pid == "00001_lr":
        assert tri_res["final_tier"] == "TIER_1_ROUTINE", "Patient 1 should be TIER_1_ROUTINE"
        assert not tri_res["override_active"], "Patient 1 should NOT trigger a safety override"
        print(f"[ASSERTION PASSED: {label} correctly assigned TIER_1_ROUTINE with zero overrides]")

    elif pid == "00008_lr":
        assert tri_res["final_tier"] == "TIER_3_ESCALATION", "Patient 8 MUST be escalated to TIER_3_ESCALATION"
        assert tri_res["override_active"], "Patient 8 MUST trigger safety override due to anterior ST elevation"
        assert tri_res["effective_risk_pct"] >= 85.0, "Effective risk should be elevated to emergency tier (>= 85%)"
        print(f"[ASSERTION PASSED: {label} successfully caught by safety guardrail and escalated to TIER_3_ESCALATION!]")

print("\nAll clinical safety guardrail assertions passed successfully!")