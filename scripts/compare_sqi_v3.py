
"""Compare complete CardioSense feature extraction using SQI V1 and V3."""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.extract_features import (
    FEATURE_COLUMNS,
    extract_clinical_features,
)


MANIFEST = ROOT / "data/cohort/cohort_manifest.csv"
RECORDS_ROOT = ROOT / "data/cohort"
OUTPUT = ROOT / "docs/sqi_audit/full_pipeline_v1_vs_v3.csv"


def process_record(raw, fs, leads, version):
    try:
        result = extract_clinical_features(
            raw,
            fs=fs,
            lead_names=leads,
            sqi_version=version,
        )

        sqi = result.get("sqi_details") or {}
        features = result.get("features")

        if result["status"] == "success":
            if not isinstance(features, dict):
                raise ValueError("Successful extraction has no features")

            missing = [
                name for name in FEATURE_COLUMNS
                if name not in features
            ]

            if missing:
                raise ValueError(
                    f"Missing classifier features: {missing}"
                )

        return {
            "status": result["status"],
            "reason": result.get("reason", ""),
            "overall_sqi": sqi.get("overall_sqi"),
            "technical_failures": ",".join(
                sqi.get("technical_failure_leads", [])
            ),
            "requires_review": sqi.get("requires_review", False),
            "feature_count": len(features) if features else 0,
        }

    except Exception as error:
        return {
            "status": "error",
            "reason": f"{type(error).__name__}: {error}",
            "overall_sqi": None,
            "technical_failures": "",
            "requires_review": False,
            "feature_count": 0,
        }


def main():
    manifest = pd.read_csv(MANIFEST)

    rows = []

    print(f"Manifest records: {len(manifest)}")

    for index, record in enumerate(
        manifest.itertuples(index=False), start=1
    ):
        record_id = int(record.ecg_id)
        record_path = RECORDS_ROOT / str(record.filename_lr)

        row = {
            "ecg_id": record_id,
            "strat_fold": int(record.strat_fold),
            "target_label": int(record.target_label),
        }

        if (
            not record_path.with_suffix(".hea").is_file()
            or not record_path.with_suffix(".dat").is_file()
        ):
            for version in ("v1", "v3"):
                row[f"{version}_status"] = "missing"
                row[f"{version}_reason"] = "Waveform files unavailable"

            rows.append(row)
            continue

        try:
            raw, fields = wfdb.rdsamp(str(record_path))

            for version in ("v1", "v3"):
                outcome = process_record(
                    raw,
                    fields["fs"],
                    fields["sig_name"],
                    version,
                )

                for key, value in outcome.items():
                    row[f"{version}_{key}"] = value

        except Exception as error:
            for version in ("v1", "v3"):
                row[f"{version}_status"] = "error"
                row[f"{version}_reason"] = (
                    f"Waveform loading: {type(error).__name__}: {error}"
                )

        rows.append(row)

        if index % 25 == 0:
            print(f"Processed {index}/{len(manifest)}")

    table = pd.DataFrame(rows)

    print("\n=== FULL PIPELINE RESULTS ===")

    for version in ("v1", "v3"):
        print(f"\nSQI {version.upper()}")
        print(
            table[f"{version}_status"]
            .value_counts(dropna=False)
            .to_string()
        )

    valid = table[
        (table["v1_status"] != "missing")
        & (table["v3_status"] != "missing")
    ].copy()

    print("\n=== STATUS TRANSITIONS ===")
    print(
        pd.crosstab(
            valid["v1_status"],
            valid["v3_status"],
            dropna=False,
        ).to_string()
    )

    recovered = valid[
        (valid["v1_status"] == "rejected")
        & (valid["v3_status"] == "success")
    ]

    lost = valid[
        (valid["v1_status"] == "success")
        & (valid["v3_status"] != "success")
    ]

    print("\nRecovered records:", len(recovered))
    print("Previously successful records lost:", len(lost))

    print("\n=== V3 SUCCESS BY FOLD ===")
    print(
        table.assign(
            v3_success=table["v3_status"].eq("success")
        )
        .groupby("strat_fold")["v3_success"]
        .agg(["sum", "count"])
        .to_string()
    )

    print("\n=== V3 SUCCESS BY LABEL ===")
    print(
        table.assign(
            v3_success=table["v3_status"].eq("success")
        )
        .groupby("target_label")["v3_success"]
        .agg(["sum", "count"])
        .to_string()
    )

    print("\n=== SELECTED ECGs ===")
    print(
        table[
            table["ecg_id"].isin([6795, 7706, 18646])
        ][
            [
                "ecg_id",
                "v1_status",
                "v3_status",
                "v3_reason",
                "v3_technical_failures",
            ]
        ].to_string(index=False)
    )

    failures = table[
        table["v3_status"].isin(["rejected", "error"])
    ]

    print("\n=== V3 FAILURE REASONS ===")
    for reason, count in Counter(
        failures["v3_reason"].fillna("")
    ).most_common(10):
        print(f"{count:3} | {reason}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print("Original feature CSV and checkpoints unchanged.")


if __name__ == "__main__":
    main()
