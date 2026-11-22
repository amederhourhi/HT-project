
"""Compare CardioSense measurements at original 100 Hz and 500 Hz.

Records: PTB-XL ECG 1918 and ECG 14639.

This experiment evaluates:
- ST deviation at the J point
- QRS duration
- trusted leads
- SQI V3 quality warnings
- qualifying contiguous ST-elevation pairs
- measurement differences across sampling rates

It does not execute the trained CNN/RF models or modify
existing model checkpoints.
"""

from pathlib import Path

import pandas as pd
import wfdb

from src.features.extract_features import extract_clinical_features
from src.triage.leads import (
    find_stemi_pairs,
    normalise_age,
    normalise_sex,
)


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "cohort"
MANIFEST = COHORT / "cohort_manifest.csv"

OUTPUT = (
    ROOT
    / "docs"
    / "clinical_validation"
    / "sampling_rate_comparison.csv"
)

ECG_IDS = (1918, 14639)
LEADS = ("V1", "V2", "V3", "V4")


def load_record(relative_path, expected_fs):
    path = COHORT / str(relative_path)

    for extension in (".hea", ".dat"):
        required = path.with_suffix(extension)

        if not required.is_file():
            raise FileNotFoundError(
                f"Missing original waveform file: {required}"
            )

    raw, fields = wfdb.rdsamp(str(path))

    if fields["fs"] != expected_fs:
        raise ValueError(
            f"Expected {expected_fs} Hz, "
            f"received {fields['fs']} Hz"
        )

    if raw.shape[1] != 12:
        raise ValueError(
            f"Expected 12 leads, received {raw.shape[1]}"
        )

    return raw, fields


def extract_record(relative_path, expected_fs):
    raw, fields = load_record(relative_path, expected_fs)

    result = extract_clinical_features(
        raw,
        fs=fields["fs"],
        lead_names=fields["sig_name"],
        sqi_version="v3",
    )

    return result, raw.shape


def summarize(result, age, sex):
    if result["status"] != "success":
        return {
            "status": result["status"],
            "reason": result.get("reason", ""),
        }

    features = result["features"]
    sqi = result.get("sqi_details") or {}

    trusted = {
        lead.strip().upper()
        for lead in features["leads_trusted"].split(",")
        if lead.strip()
    }

    st_by_lead = {
        lead: features[f"st_{lead}"]
        for lead in trusted
    }

    pairs = find_stemi_pairs(
        st_by_lead,
        normalise_age(age),
        normalise_sex(sex),
    )

    pair_labels = sorted({
        f"{pair.lead_a}/{pair.lead_b}"
        for pair in pairs
    })

    summary = {
        "status": "success",
        "reason": "OK",
        "qrs_duration_ms": features["qrs_duration_ms"],
        "qrs_plausible": features["qrs_plausible"],
        "heart_rate_bpm": features["heart_rate_bpm"],
        "overall_sqi": features["overall_sqi"],
        "trusted_leads": sorted(trusted),
        "baseline_drift_leads": sorted(
            sqi.get("baseline_drift_review_leads", [])
        ),
        "technical_failure_leads": sorted(
            sqi.get("technical_failure_leads", [])
        ),
        "st_pairs": pair_labels,
        "st_pair_count": len(pair_labels),
    }

    quality = result.get("quality_contract") or {}
    per_feature_quality = quality.get("feature_quality") or {}

    for lead in LEADS:
        feature_name = f"st_{lead}"

        summary[feature_name] = features[feature_name]
        summary[f"{feature_name}_quality"] = (
            per_feature_quality
            .get(feature_name, {})
            .get("status", "unknown")
        )

    return summary


def main():
    manifest = pd.read_csv(MANIFEST).set_index("ecg_id")

    rows = []

    for ecg_id in ECG_IDS:
        record = manifest.loc[ecg_id]

        age = None if pd.isna(record["age"]) else record["age"]
        sex = None if pd.isna(record["sex"]) else record["sex"]

        print(f"\n{'=' * 65}")
        print(f"ECG {ecg_id} | Age: {age} | Sex: {sex}")

        comparisons = {}

        for rate, path_column in (
            (100, "filename_lr"),
            (500, "filename_hr"),
        ):
            try:
                result, shape = extract_record(
                    record[path_column],
                    rate,
                )

                summary = summarize(result, age, sex)
                comparisons[rate] = summary

                print(f"\n--- {rate} Hz ---")
                print("Waveform shape:", shape)
                print("Extraction status:", summary["status"])

                if summary["status"] != "success":
                    print("Reason:", summary["reason"])
                    continue

                print(
                    "QRS duration:",
                    summary["qrs_duration_ms"], "ms"
                )
                print(
                    "QRS plausible:",
                    summary["qrs_plausible"]
                )
                print(
                    "ST pairs:",
                    summary["st_pairs"]
                )
                print(
                    "Baseline review:",
                    summary["baseline_drift_leads"]
                )
                print(
                    "Trusted leads:",
                    summary["trusted_leads"]
                )

                for lead in LEADS:
                    name = f"st_{lead}"
                    print(
                        f"{lead}: "
                        f"{summary[name]:+.3f} mV "
                        f"| {summary[name + '_quality']}"
                    )

            except Exception as error:
                comparisons[rate] = {
                    "status": "error",
                    "reason": (
                        f"{type(error).__name__}: {error}"
                    ),
                }

                print(f"\n--- {rate} Hz ERROR ---")
                print(comparisons[rate]["reason"])

        low = comparisons.get(100, {})
        high = comparisons.get(500, {})

        row = {
            "ecg_id": ecg_id,
            "status_100hz": low.get("status"),
            "status_500hz": high.get("status"),
            "reason_100hz": low.get("reason"),
            "reason_500hz": high.get("reason"),
        }

        if (
            low.get("status") == "success"
            and high.get("status") == "success"
        ):
            print("\n--- 500 Hz minus 100 Hz ---")

            for lead in LEADS:
                key = f"st_{lead}"
                delta = high[key] - low[key]

                row[f"{key}_100hz_mv"] = low[key]
                row[f"{key}_500hz_mv"] = high[key]
                row[f"{key}_delta_mv"] = round(delta, 6)

                print(
                    f"{lead}: {delta:+.3f} mV "
                    f"({delta * 1000:+.1f} µV)"
                )

            qrs_delta = (
                high["qrs_duration_ms"]
                - low["qrs_duration_ms"]
            )

            row["qrs_100hz_ms"] = low["qrs_duration_ms"]
            row["qrs_500hz_ms"] = high["qrs_duration_ms"]
            row["qrs_delta_ms"] = qrs_delta

            row["st_pairs_100hz"] = ";".join(
                low["st_pairs"]
            )
            row["st_pairs_500hz"] = ";".join(
                high["st_pairs"]
            )
            row["st_pair_decision_changed"] = (
                low["st_pairs"] != high["st_pairs"]
            )

            print(
                f"QRS difference: {qrs_delta:+.1f} ms"
            )
            print(
                "ST pair classification changed:",
                row["st_pair_decision_changed"]
            )

            if row["st_pair_decision_changed"]:
                print(
                    "REVIEW REQUIRED: Sampling-rate change "
                    "altered the contiguous ST findings."
                )

        rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)

    print(f"\nSaved comparison: {OUTPUT}")

    incomplete = [
        row["ecg_id"]
        for row in rows
        if (
            row["status_100hz"] != "success"
            or row["status_500hz"] != "success"
        )
    ]

    if incomplete:
        raise SystemExit(
            f"Comparison incomplete for ECGs: {incomplete}. "
            "Inspect the errors above."
        )

    print("SUCCESS: Both sampling-rate comparisons completed.")


if __name__ == "__main__":
    main()
