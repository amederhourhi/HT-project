"""Read-only comparison of SQI v1, v2, and v3 on the local cohort."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import wfdb

from src.dsp.filters import filter_ecg
from src.dsp.sqi import assess_ecg_quality
from src.dsp.sqi_v2 import assess_ecg_quality_v2
from src.dsp.sqi_v3 import assess_ecg_quality_v3


def main():
    manifest = pd.read_csv(ROOT / "data/cohort/cohort_manifest.csv")
    records_root = ROOT / "data/cohort"
    results = []
    for _, record in manifest.iterrows():
        record_id = int(record["ecg_id"])
        path = records_root / str(record["filename_lr"])
        if not path.with_suffix(".hea").exists() or not path.with_suffix(".dat").exists():
            results.append({"ecg_id": record_id, "status": "missing"})
            continue
        try:
            signal, fields = wfdb.rdsamp(str(path))
            cleaned = filter_ecg(signal, fs=fields["fs"])
            common = {"lead_names": fields["sig_name"], "fs": fields["fs"]}
            # V1's original signature may not accept fs: keep its native call.
            v1 = assess_ecg_quality(cleaned, lead_names=fields["sig_name"])
            v2 = assess_ecg_quality_v2(cleaned, **common)
            v3 = assess_ecg_quality_v3(cleaned, **common)
            results.append({
                "ecg_id": record_id, "status": "ok", "strat_fold": record["strat_fold"],
                "target_label": record["target_label"],
                "v1_acceptable": bool(v1["is_acceptable"]),
                "v2_acceptable": bool(v2["is_acceptable"]),
                "v3_acceptable": bool(v3["is_acceptable"]),
                "v3_requires_review": v3["requires_review"],
                "v3_ready": v3["ready_for_automated_inference"],
                "v3_review_leads": ",".join(v3["sustained_instability_leads"]),
            })
        except Exception as exc:
            results.append({"ecg_id": record_id, "status": "error", "error": repr(exc)})

    table = pd.DataFrame(results)
    ok = table[table["status"] == "ok"]
    print("\n=== COHORT COUNTS ===")
    print(table["status"].value_counts().to_string())
    if len(ok):
        for col in ("v1_acceptable", "v2_acceptable", "v3_acceptable", "v3_requires_review", "v3_ready"):
            print(f"{col}: {int(ok[col].sum())}/{len(ok)}")
        print("\n=== SELECTED RECORDS ===")
        print(ok[ok["ecg_id"].isin([6795, 7706, 18646])].to_string(index=False))
        print("\n=== REVIEW BY FOLD AND LABEL ===")
        print(ok.groupby(["strat_fold", "target_label"])["v3_requires_review"].agg(["sum", "count"]).to_string())
    folder = ROOT / "docs/sqi_audit"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "sqi_v3_comparison.csv"
    table.to_csv(target, index=False)
    print(f"\nSaved: {target}")
    print("No models or feature tables were modified.")


if __name__ == "__main__":
    main()
