
"""Research-only R-peak benchmark on previously used LUDB development records 1-30.

Expert lead-II N-peak annotations are matched one-to-one to unchanged CardioSense
R-peak detections. No model changes, data download, or reserved-record access.
At 100 Hz, the 500 Hz signal is resampled for a controlled rate comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ludb_reference/data"

RECORD_IDS = tuple(range(1, 31))
RATES = (500, 100)
TOLERANCES_MS = (50.0, 100.0)


def _validated_samples(values, label, *, allow_empty=False):
    """Require unique, increasing, nonnegative integer sample positions."""

    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label}: invalid sample positions"
        ) from exc

    if arr.ndim != 1 or (arr.size == 0 and not allow_empty):
        raise ValueError(
            f"{label}: expected nonempty 1D array"
        )

    if (
        not np.isfinite(arr).all()
        or (arr < 0).any()
        or not np.equal(arr, np.floor(arr)).all()
        or (np.diff(arr) <= 0).any()
    ):
        raise ValueError(
            f"{label}: non-unique, unsorted or invalid positions"
        )

    return arr.astype(np.int64)


def match_peaks(reference, detected, *, fs, tolerance_ms):
    """Find optimal monotone one-to-one peak matches.

    Prioritize the maximum number of true matches.
    Among those solutions, minimize total absolute timing error.

    Unmatched expert beats count as FN.
    Unmatched detections count as FP.
    """

    if (
        not np.isfinite(fs)
        or fs <= 0
        or not np.isfinite(tolerance_ms)
        or tolerance_ms <= 0
    ):
        raise ValueError(
            "Sampling rate and tolerance must be positive and finite"
        )

    ref = _validated_samples(reference, "reference")

    det = _validated_samples(
        detected,
        "detected",
        allow_empty=True,
    )

    ref_ms = ref.astype(float) * 1000.0 / fs
    det_ms = det.astype(float) * 1000.0 / fs

    n = len(ref)
    m = len(det)

    dp = [
        [(0, 0.0) for _ in range(m + 1)]
        for _ in range(n + 1)
    ]

    action = [
        [None for _ in range(m + 1)]
        for _ in range(n + 1)
    ]

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):

            options = [
                (dp[i + 1][j], "skip_ref"),
                (dp[i][j + 1], "skip_det"),
            ]

            difference = abs(ref_ms[i] - det_ms[j])

            if difference <= tolerance_ms + 1e-9:

                count, total_error = dp[i + 1][j + 1]

                options.insert(
                    0,
                    (
                        (
                            count + 1,
                            total_error + difference,
                        ),
                        "match",
                    ),
                )

            best, choice = max(
                options,
                key=lambda item: (
                    item[0][0],
                    -item[0][1],
                ),
            )

            dp[i][j] = best
            action[i][j] = choice

    i = 0
    j = 0
    errors = []

    while i < n and j < m:

        step = action[i][j]

        if step == "match":
            errors.append(
                float(det_ms[j] - ref_ms[i])
            )
            i += 1
            j += 1

        elif step == "skip_ref":
            i += 1

        else:
            j += 1

    tp = len(errors)
    fp = m - tp
    fn = n - tp

    precision = tp / m if m else 0.0
    recall = tp / n

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if tp
        else 0.0
    )

    return {
        "reference_beats": n,
        "detected_peaks": m,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_timing_errors_ms": errors,
    }


def summarize(rows):
    """Compute micro-averaged metrics over all recordings."""

    if not rows:
        raise ValueError(
            "Cannot summarize empty evaluations"
        )

    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)

    n = tp + fn
    m = tp + fp

    if n <= 0:
        raise ValueError(
            "No expert reference events"
        )

    errors = [
        e
        for row in rows
        for e in row["matched_timing_errors_ms"]
    ]

    absolute = np.abs(
        np.asarray(errors, dtype=float)
    )

    return {
        "records": len(rows),
        "reference_beats": n,
        "detected_peaks": m,
        "tp": tp,
        "fp": fp,
        "fn": fn,

        "precision": (
            tp / m if m else 0.0
        ),

        "recall": tp / n,

        "f1": (
            2 * tp / (2 * tp + fp + fn)
            if tp
            else 0.0
        ),

        "median_absolute_timing_error_ms": (
            float(np.median(absolute))
            if len(errors)
            else None
        ),

        "p95_absolute_timing_error_ms": (
            float(np.percentile(absolute, 95))
            if len(errors)
            else None
        ),

        "records_with_missed_beats": sum(
            r["fn"] > 0 for r in rows
        ),

        "records_with_extra_detections": sum(
            r["fp"] > 0 for r in rows
        ),
    }


def evaluate_record(record_id):
    """Evaluate only a locally stored development recording."""

    if (
        type(record_id) is not int
        or record_id not in RECORD_IDS
    ):
        raise ValueError(
            "Only LUDB development records 1-30 are allowed"
        )

    base = DATA / str(record_id)

    # Never automatically download or access reserved data.
    for ext in ("hea", "dat", "ii"):

        if not (
            DATA / f"{record_id}.{ext}"
        ).is_file():

            raise FileNotFoundError(
                f"Missing local LUDB file: {record_id}.{ext}; "
                "download development data with: "
                "python -m scripts.benchmark_ludb_landmarks "
                "--count 30 --download-only"
            )

    import wfdb

    from scripts.benchmark_ludb_landmarks import (
        qrs_triplets,
    )

    from src.dsp.filters import filter_ecg
    from src.features.r_peaks import detect_r_peaks

    signal, fields = wfdb.rdsamp(str(base))

    if (
        float(fields["fs"]) != 500.0
        or signal.ndim != 2
        or signal.shape[1] != 12
    ):
        raise ValueError(
            f"LUDB {record_id}: expected "
            "12 leads sampled at 500 Hz"
        )

    names = [
        str(name).lower()
        for name in fields["sig_name"]
    ]

    if names.count("ii") != 1:
        raise ValueError(
            f"LUDB {record_id}: exactly one lead II required"
        )

    lead_ii = signal[
        :,
        names.index("ii"),
    ]

    annotations = wfdb.rdann(
        str(base),
        "ii",
    )

    references_500 = _validated_samples(
        [
            t[1]
            for t in qrs_triplets(annotations)
        ],
        "expert N peaks",
    )

    result = []

    for rate in RATES:

        if rate == 500:
            samples = lead_ii
            references = references_500

        else:
            samples = resample_poly(
                lead_ii,
                1,
                5,
            )

            references = np.rint(
                references_500 / 5
            ).astype(np.int64)

            references = _validated_samples(
                references,
                "downsampled reference",
            )

        # The production detector remains unchanged.
        filtered = filter_ecg(
            samples,
            fs=rate,
        )

        detected = detect_r_peaks(
            filtered,
            fs=rate,
        )

        for tolerance in TOLERANCES_MS:

            score = match_peaks(
                references,
                detected,
                fs=rate,
                tolerance_ms=tolerance,
            )

            result.append({
                "record_id": record_id,
                "fs_hz": rate,
                "tolerance_ms": tolerance,
                **score,
            })

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional new JSON path; "
            "never overwrites existing results"
        ),
    )

    args = parser.parse_args()

    results = []

    for rid in RECORD_IDS:

        results.extend(
            evaluate_record(rid)
        )

        print(
            f"LUDB {rid:03d}: "
            "evaluated at 500 and 100 Hz"
        )

    groups = []

    for rate in RATES:
        for tolerance in TOLERANCES_MS:

            selected = [
                r
                for r in results
                if (
                    r["fs_hz"] == rate
                    and r["tolerance_ms"] == tolerance
                )
            ]

            groups.append({
                "fs_hz": rate,
                "tolerance_ms": tolerance,
                **summarize(selected),
            })

    report = {
        "study": "cardiosense_rpeak_ludb_dev_v1",

        "data_role": (
            "previously_used_development_records_"
            "not_untouched_holdout"
        ),

        "record_ids": [1, 30],

        "expert_reference": (
            "lead_ii_N_QRS_triplet_peaks"
        ),

        "matching": (
            "one_to_one_maximum_tp_then_"
            "minimum_absolute_timing_error"
        ),

        "rate_100_hz_source": (
            "resample_poly_500_hz_by_factor_5_"
            "not_native_100_hz_device"
        ),

        "summary": groups,
        "per_record": results,

        "restrictions": [
            "No model training or modification",
            "No LUDB 31-200 evaluation",
            "No PTB-XL fold-10 evaluation",
            "N-only annotation parsing may omit other beat types",
            "Timing tolerance is predefined; metric is not clinical sensitivity",
        ],
    }

    print(
        "\n=== R-PEAK DETECTION: "
        "DEVELOPMENT RECORDS 1-30 ==="
    )

    for item in groups:

        print(
            f"{item['fs_hz']:3d} Hz | "
            f"+/-{item['tolerance_ms']:.0f} ms | "
            f"TP={item['tp']:3d} "
            f"FP={item['fp']:3d} "
            f"FN={item['fn']:3d} | "
            f"precision={item['precision']:.3f} "
            f"recall={item['recall']:.3f} "
            f"F1={item['f1']:.3f} | "
            "median abs error="
            f"{item['median_absolute_timing_error_ms']}"
        )

    if args.output is not None:

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "x",
            encoding="utf-8",
        ) as out:

            json.dump(
                report,
                out,
                indent=2,
                allow_nan=False,
            )

            out.write("\n")

        print(
            "Saved new research report:",
            args.output,
        )

    else:
        print(
            "READ-ONLY: no files were written"
        )


if __name__ == "__main__":
    main()