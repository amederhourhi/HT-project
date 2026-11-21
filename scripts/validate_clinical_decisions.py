
"""CardioSense - Clinical decision evidence audit.

Replays actual feature extraction and deterministic triage over
the local PTB-XL cohort.

What this evaluates:
- extraction success and failure
- deterministic escalation and review frequencies
- quality warnings affecting triggered ST-elevation pairs
- results stratified by diagnostic label and fold
- cases requiring independent clinical adjudication

What it does NOT evaluate:
- confirmed STEMI sensitivity or specificity
- correctness of emergency intervention decisions
- model calibration
- clinical safety or medical-device certification

The statistical ensemble risk is fixed to zero to isolate
the deterministic rule layer.

No existing feature CSV, checkpoint, or source file is modified.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import wfdb

from src.features.extract_features import (
    extract_clinical_features,
)
from src.triage.triage_rules import ClinicalTriageEngine
from src.triage.leads import (
    find_stemi_pairs,
    normalise_age,
    normalise_sex,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/cohort/cohort_manifest.csv"
RECORDS = ROOT / "data/cohort"

OUTPUT = (
    ROOT
    / "docs/clinical_validation"
    / "rule_quality_audit_v1.csv"
)

REVIEW_OUTPUT = (
    ROOT
    / "docs/clinical_validation"
    / "adjudication_candidates_v1.csv"
)

SQI_VERSION = "v3"


def optional_value(row, name):
    value = row.get(name)
    return None if pd.isna(value) else value


def evaluate_record(row, engine):
    ecg_id = int(row["ecg_id"])

    result = {
        "ecg_id": ecg_id,
        "strat_fold": int(row["strat_fold"]),
        "target_label": int(row["target_label"]),
        "age": optional_value(row, "age"),
        "sex": optional_value(row, "sex"),
        "extraction_status": "error",
        "extraction_reason": "",
        "final_tier": "",
        "rule_ids": "",
        "st_pair_count": 0,
        "st_pairs": "",
        "quality_flagged_pair_count": 0,
        "quality_flagged_pairs": "",
        "quality_unavailable_pair_count": 0,
        "review_feature_count": 0,
        "unavailable_feature_count": 0,
        "baseline_review_leads": "",
        "qrs_duration_ms": None,
        "heart_rate_bpm": None,
        "qrs_plausible": None,
    }

    path = RECORDS / str(row["filename_lr"])

    if not (
        path.with_suffix(".hea").is_file()
        and path.with_suffix(".dat").is_file()
    ):
        result["extraction_status"] = "missing_waveform"
        result["extraction_reason"] = "Waveform unavailable"
        return result

    try:
        raw, fields = wfdb.rdsamp(str(path))

        extracted = extract_clinical_features(
            raw,
            fs=fields["fs"],
            lead_names=fields["sig_name"],
            sqi_version=SQI_VERSION,
        )

        result["extraction_status"] = extracted["status"]
        result["extraction_reason"] = extracted.get(
            "reason", ""
        )

        if extracted["status"] != "success":
            return result

        features = extracted["features"]
        quality = extracted.get("quality_contract") or {}
        feature_quality = quality.get("feature_quality") or {}
        sqi = extracted.get("sqi_details") or {}

        result["review_feature_count"] = len(
            quality.get("review_required_features", [])
        )
        result["unavailable_feature_count"] = len(
            quality.get("unavailable_features", [])
        )
        result["baseline_review_leads"] = ",".join(
            sqi.get("baseline_drift_review_leads", [])
        )

        result["qrs_duration_ms"] = features[
            "qrs_duration_ms"
        ]
        result["heart_rate_bpm"] = features[
            "heart_rate_bpm"
        ]
        result["qrs_plausible"] = features[
            "qrs_plausible"
        ]

        trusted = {
            name.strip().upper()
            for name in features["leads_trusted"].split(",")
            if name.strip()
        }

        st_map = {
            lead: float(features[f"st_{lead}"])
            for lead in trusted
        }

        patient = {
            "age": result["age"],
            "sex": result["sex"],
        }

        # Isolate deterministic rules from statistical predictions.
        # This is NOT the ensemble or complete clinical cascade.
        payload = {
            "risk_score_pct": 0.0,
            "patient": patient,
            "st_by_lead": st_map,
            "clinical_biomarkers": {
                "heart_rate_bpm": features["heart_rate_bpm"],
                "qrs_duration_ms": features["qrs_duration_ms"],
                "qrs_plausible": features["qrs_plausible"],
            },
            "quality_contract": quality,
            "st_provenance": extracted.get("st_provenance"),
        }

        verdict = engine.evaluate(payload)

        result["final_tier"] = verdict["final_tier"]

        findings = verdict.get("findings", [])
        result["rule_ids"] = ",".join(
            sorted({
                finding["rule_id"]
                for finding in findings
            })
        )

        # Analyze the same contiguous-lead criterion used by triage.
        pairs = find_stemi_pairs(
            st_map,
            normalise_age(patient["age"]),
            normalise_sex(patient["sex"]),
        )

        pair_names = []
        flagged_pairs = []
        unavailable_pairs = []

        for pair in pairs:
            lead_a = pair.lead_a
            lead_b = pair.lead_b
            name = f"{lead_a}/{lead_b}"
            pair_names.append(name)

            statuses = [
                feature_quality.get(
                    f"st_{lead.upper()}", {}
                ).get("status", "unknown")
                for lead in (lead_a, lead_b)
            ]

            if any(
                status in ("review_required", "unknown")
                for status in statuses
            ):
                flagged_pairs.append(name)

            if "unavailable" in statuses:
                unavailable_pairs.append(name)

        result["st_pair_count"] = len(pair_names)
        result["st_pairs"] = ";".join(pair_names)

        result["quality_flagged_pair_count"] = len(
            flagged_pairs
        )
        result["quality_flagged_pairs"] = ";".join(
            flagged_pairs
        )

        result["quality_unavailable_pair_count"] = len(
            unavailable_pairs
        )

    except Exception as error:
        result["extraction_status"] = "error"
        result["extraction_reason"] = (
            f"{type(error).__name__}: {error}"
        )

    return result


def main():
    manifest = pd.read_csv(MANIFEST)
    engine = ClinicalTriageEngine()

    records = []

    for index, (_, row) in enumerate(
        manifest.iterrows(), start=1
    ):
        records.append(evaluate_record(row, engine))

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(manifest)}"
            )

    table = pd.DataFrame(records)

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )
    table.to_csv(OUTPUT, index=False)

    successful = table[
        table["extraction_status"] == "success"
    ].copy()

    print("\n=== EXTRACTION COVERAGE ===")
    print(
        table["extraction_status"]
        .value_counts()
        .to_string()
    )

    print("\n=== DETERMINISTIC TRIAGE TIERS ===")
    print(
        successful["final_tier"]
        .value_counts()
        .to_string()
    )

    print("\n=== TRIAGE BY DIAGNOSTIC LABEL ===")
    print(
        pd.crosstab(
            successful["target_label"],
            successful["final_tier"],
        ).to_string()
    )

    print("\n=== TRIAGE BY FOLD ===")
    print(
        pd.crosstab(
            successful["strat_fold"],
            successful["final_tier"],
        ).to_string()
    )

    print("\n=== RULE FIRING COUNTS ===")
    counter = Counter()

    for value in successful["rule_ids"]:
        if value:
            counter.update(
                name for name in value.split(",")
                if name
            )

    for name, count in counter.most_common():
        print(f"{name}: {count}")

    st_cases = successful[
        successful["st_pair_count"] > 0
    ]

    flagged = st_cases[
        st_cases["quality_flagged_pair_count"] > 0
    ]

    unavailable = st_cases[
        st_cases["quality_unavailable_pair_count"] > 0
    ]

    print("\n=== ST FINDING QUALITY AUDIT ===")
    print("ECGs with qualifying ST pairs:", len(st_cases))
    print("ECGs with quality-flagged ST pairs:", len(flagged))
    print(
        "ECGs with unavailable measurements in ST pairs:",
        len(unavailable),
    )

    print("\n=== QUALITY-FLAGGED ST CASES ===")
    if flagged.empty:
        print("None found in this cohort")
    else:
        print(
            flagged[
                [
                    "ecg_id",
                    "strat_fold",
                    "target_label",
                    "st_pairs",
                    "quality_flagged_pairs",
                    "final_tier",
                ]
            ].to_string(index=False)
        )

    # This produces a candidate queue for expert review,
    # not a medically adjudicated ground-truth dataset.
    candidates = successful[
        (successful["st_pair_count"] > 0)
        | (successful["quality_flagged_pair_count"] > 0)
        | (successful["final_tier"] == "TIER_3_ESCALATION")
    ].copy()

    candidates.to_csv(
        REVIEW_OUTPUT, index=False
    )

    print("\n=== AUDIT FILES ===")
    print("Full audit:", OUTPUT)
    print("Review candidates:", REVIEW_OUTPUT)
    print("Candidates for expert review:", len(candidates))

    print(
        "\nIMPORTANT: PTB-XL MI labels do not provide "
        "an emergency-revascularization ground truth."
    )
    print(
        "This audit measures behavior and quality conflicts, "
        "not validated emergency-triage accuracy."
    )


if __name__ == "__main__":
    main()
