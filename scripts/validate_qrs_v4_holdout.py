
"""CardioSense — Frozen QRS V4 held-out evaluation.

LUDB 1–30: previously used for development.
LUDB 31–100: held-out evaluation (this script).
LUDB 101–200: reserved for final evaluation.

No retraining, parameter tuning or production modification.

Run:
    python -m scripts.validate_qrs_v4_holdout
"""

import csv
from pathlib import Path

import numpy as np

from scripts.benchmark_ludb_landmarks import download_record
from scripts.benchmark_ludb_qrs_v4 import evaluate


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "docs"
    / "clinical_validation"
    / "ludb_qrs_v4_holdout.csv"
)

START = 31
END = 100

METHODS = ("current", "qrs_v4")
METRICS = (
    "onset_error_ms",
    "offset_error_ms",
    "duration_error_ms",
)


def summarize(group, title):
    print(f"\n=== {title} ===")
    print(f"Successful recordings: {len(group)}")

    for metric in METRICS:
        values = np.asarray(
            [float(row[metric]) for row in group],
            dtype=float,
        )

        if len(values) == 0:
            print(f"{metric}: no valid results")
            continue

        abs_values = np.abs(values)

        print(
            f"{metric}: "
            f"bias={np.median(values):+.1f} ms | "
            f"median_abs={np.median(abs_values):.1f} ms | "
            f"p95_abs={np.percentile(abs_values, 95):.1f} ms | "
            f"max_abs={np.max(abs_values):.1f} ms"
        )


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for rid in range(START, END + 1):
        try:
            # Existing downloader skips files already present.
            download_record(rid)

            results = evaluate(rid)

            by_method = {
                row["method"]: row
                for row in results
            }

            # Fail closed: both algorithms must succeed
            # for a record to enter the paired comparison.
            for method in METHODS:
                result = by_method.get(method)

                if result is None:
                    raise RuntimeError(
                        f"Missing {method} result"
                    )

                if result.get("status") != "success":
                    raise RuntimeError(
                        f"{method}: "
                        f"{result.get('reason', 'unknown failure')}"
                    )

                for metric in METRICS:
                    value = float(result[metric])
                    if not np.isfinite(value):
                        raise ValueError(
                            f"{method}: nonfinite {metric}"
                        )

            for method in METHODS:
                result = by_method[method]

                rows.append({
                    "record_id": rid,
                    "method": method,
                    "status": "success",
                    "reason": "",
                    **{
                        metric: float(result[metric])
                        for metric in METRICS
                    },
                })

            print(f"LUDB {rid:03d}: evaluated")

        except Exception as exc:
            print(
                f"LUDB {rid:03d}: FAILED "
                f"{type(exc).__name__}: {exc}"
            )

            for method in METHODS:
                rows.append({
                    "record_id": rid,
                    "method": method,
                    "status": "failed",
                    "reason": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })

    columns = [
        "record_id",
        "method",
        "status",
        "reason",
        *METRICS,
    ]

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)

    successful = [
        row for row in rows
        if row["status"] == "success"
    ]

    records_ok = {
        row["record_id"]
        for row in successful
        if row["method"] == "current"
    }

    failures = [
        rid for rid in range(START, END + 1)
        if rid not in records_ok
    ]

    print("\n" + "=" * 65)
    print("FROZEN V4 — HELD-OUT LUDB RESULTS")
    print("=" * 65)

    print(
        f"Paired records: {len(records_ok)}/"
        f"{END - START + 1}"
    )
    print("Failed record IDs:", failures)

    for method in METHODS:
        group = [
            row for row in successful
            if row["method"] == method
        ]
        summarize(group, method.upper())

    current = {
        row["record_id"]: row
        for row in successful
        if row["method"] == "current"
    }

    candidate = {
        row["record_id"]: row
        for row in successful
        if row["method"] == "qrs_v4"
    }

    paired_ids = sorted(
        current.keys() & candidate.keys()
    )

    print("\n=== RECORD-LEVEL COMPARISON ===")

    for metric in METRICS:
        improved = []
        worsened = []
        unchanged = []

        for rid in paired_ids:
            old_error = abs(
                current[rid][metric]
            )
            new_error = abs(
                candidate[rid][metric]
            )

            if new_error < old_error:
                improved.append(rid)
            elif new_error > old_error:
                worsened.append((
                    rid,
                    old_error,
                    new_error,
                ))
            else:
                unchanged.append(rid)

        print(f"\n{metric}")
        print("Improved:", len(improved))
        print("Worsened:", len(worsened))
        print("Unchanged:", len(unchanged))

        print("Five largest regressions:")
        for rid, before, after in sorted(
            worsened,
            key=lambda item: item[2] - item[1],
            reverse=True,
        )[:5]:
            print(
                f"  LUDB {rid}: "
                f"{before:.1f} -> {after:.1f} ms"
            )

    print("\nSaved:", OUTPUT)

    if failures:
        print(
            "\nWARNING: Incomplete held-out evaluation. "
            "Do not ignore failed recordings."
        )

    print(
        "\nDo not change V4 based on these results "
        "and then reuse this set as untouched test data."
    )


if __name__ == "__main__":
    main()
