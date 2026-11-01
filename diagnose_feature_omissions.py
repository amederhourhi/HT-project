"""Diagnose omitted CardioSense feature rows without changing datasets or models."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.extract_features import extract_clinical_features

# IDs identified as having usable waveforms but no row in clinical_features.csv.
MISSING_IDS = (
    2804, 12683, 15733, 17032, 4454, 6896, 8230, 1472, 6795,
    2536, 7735, 9015, 16202, 18991, 1259, 5175, 8366, 10285,
    10606, 4869, 6499, 7706, 13495, 18058, 18646, 773, 2542,
    3418, 3583, 13598, 19464, 19749, 4981, 10464, 12051,
    17553, 10804, 11880, 17523, 472, 2935, 7491, 8195,
    10272, 10806, 14585,
)


def locate_record(ecg_id):
    stem = f"{ecg_id:05d}_lr"
    # Standard PTB-XL layout: records100/01000/01234_lr.hea
    relative = Path("records100") / f"{(ecg_id // 1000) * 1000:05d}" / stem
    for base in (ROOT / "data" / "samples", ROOT / "data", ROOT):
        candidate = base / relative
        if candidate.with_suffix(".hea").is_file():
            return candidate
    matches = list((ROOT / "data").rglob(stem + ".hea"))
    if len(matches) == 1:
        return matches[0].with_suffix("")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous waveform paths for {ecg_id}: {matches}")
    raise FileNotFoundError(f"Cannot locate {stem}.hea below {ROOT / 'data'}")


def main():
    summary = Counter()
    reasons = defaultdict(list)
    for ecg_id in MISSING_IDS:
        try:
            path = locate_record(ecg_id)
            signal, metadata = wfdb.rdsamp(str(path))
            result = extract_clinical_features(
                signal, fs=metadata["fs"], lead_names=metadata["sig_name"]
            )
            status = result.get("status", "missing_status")
            if status == "success":
                features = result.get("features") or {}
                # A successful extraction may still contain nonfinite classifier inputs.
                from src.features.extract_features import FEATURE_COLUMNS
                invalid = [name for name in FEATURE_COLUMNS
                           if name not in features or
                           not isinstance(features[name], (int, float, np.number)) or
                           not np.isfinite(features[name])]
                category = "success_with_invalid_features" if invalid else "success_now"
                detail = ", ".join(invalid) if invalid else "No rejection; investigate batch exporter / historical versions"
            else:
                category = "rejected"
                detail = str(result.get("reason", "No rejection reason provided"))
        except Exception as exc:
            category = "exception"
            detail = f"{type(exc).__name__}: {exc}"
        summary[category] += 1
        reasons[(category, detail)].append(ecg_id)
        print(f"{ecg_id:5d} | {category:29s} | {detail}")

    print("\n=== SUMMARY ===")
    for category, n in summary.most_common():
        print(f"{category}: {n}")
    print("\n=== GROUPED REASONS ===")
    for (category, detail), ids in sorted(reasons.items(), key=lambda x: -len(x[1])):
        print(f"{len(ids):2d} | {category:29s} | {detail}\n     IDs: {ids}")
    print("\nRead-only diagnostic. No models, CSVs or checkpoints modified.")


if __name__ == "__main__":
    main()
