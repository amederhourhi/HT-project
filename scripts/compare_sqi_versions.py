"""Read-only side-by-side SQI audit on local PTB-XL cohort.

Run from repository root: python scripts/compare_sqi_versions.py
Does not alter feature tables, model checkpoints or production SQI.
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dsp.filters import filter_ecg
from src.dsp.sqi import assess_ecg_quality
from src.dsp.sqi_v2 import assess_ecg_quality_v2

MANIFEST_CANDIDATES = [ROOT / "data/cohort/cohort_manifest.csv", ROOT / "data/cohort/manifest.csv"]
RECORD_ROOT = ROOT / "data/cohort"
OUTPUT = ROOT / "docs/sqi_audit/sqi_v1_vs_v2.csv"


def load_manifest():
    for path in MANIFEST_CANDIDATES:
        if path.is_file():
            frame = pd.read_csv(path)
            if "filename_lr" in frame.columns:
                return frame, path
    matches = [p for p in (ROOT / "data/cohort").rglob("*.csv") if p.is_file()]
    for path in matches:
        frame = pd.read_csv(path)
        if "filename_lr" in frame.columns:
            return frame, path
    raise FileNotFoundError("No cohort CSV with filename_lr found under data/cohort")


def main():
    manifest, path = load_manifest()
    print(f"Manifest: {path.relative_to(ROOT)} ({len(manifest)} rows)")
    rows = []
    for _, record in manifest.iterrows():
        rel = str(record["filename_lr"]).replace("\\", "/")
        name = RECORD_ROOT / rel
        if not name.with_suffix(".hea").is_file() or not name.with_suffix(".dat").is_file():
            rows.append({"ecg_id":record.get("ecg_id"), "status":"missing_waveform"})
            continue
        try:
            signals, info = wfdb.rdsamp(str(name))
            filtered = filter_ecg(signals, fs=info["fs"])
            old = assess_ecg_quality(filtered, lead_names=info["sig_name"])
            new = assess_ecg_quality_v2(filtered, lead_names=info["sig_name"], fs=info["fs"])
            review = [k for k, v in new["lead_scores"].items() if v["review_flags"]]
            rows.append({
                "ecg_id":record.get("ecg_id"), "strat_fold":record.get("strat_fold"),
                "target_label":record.get("target_label"), "status":"ok",
                "v1_accept":old["is_acceptable"], "v2_accept":new["is_acceptable"],
                "v1_sqi":old["overall_sqi"], "v2_sqi":new["overall_sqi"],
                "v1_passed":old["passed_leads"], "v2_passed":new["passed_leads"],
                "review_leads":",".join(review),
            })
        except Exception as exc:
            rows.append({"ecg_id":record.get("ecg_id"), "status":"error", "error":f"{type(exc).__name__}: {exc}"})
    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    good = result.loc[result["status"] == "ok"].copy()
    print("\n=== AUDIT ===")
    print("Processed:", len(good), "Missing:", int((result.status == "missing_waveform").sum()), "Errors:", int((result.status == "error").sum()))
    if not good.empty:
        print("v1 accepted:", int(good.v1_accept.sum()), "v2 accepted:", int(good.v2_accept.sum()))
        print("v1 reject -> v2 accept:", int(((~good.v1_accept) & good.v2_accept).sum()))
        print("v1 accept -> v2 reject:", int((good.v1_accept & (~good.v2_accept)).sum()))
        print("Review flags (record count):", int(good.review_leads.astype(bool).sum()))
        print("\n=== SELECTED RECORDS ===")
        print(good.loc[good.ecg_id.isin([6795, 18646, 7706])].to_string(index=False))
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
