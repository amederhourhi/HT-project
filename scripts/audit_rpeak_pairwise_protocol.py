
"""CardioSense Phase II: fixed-protocol, paired R-peak evaluation audit.

Run:
    python -m scripts.audit_rpeak_pairwise_protocol --dev-only

The executable data runner is limited to LUDB 1-30.
Statistical functions accept in-memory rows and never open ECG files.

No downloads, parameter tuning, model changes, or reserved-set access.
"""

import argparse

import numpy as np

from scripts.compare_rpeak_refinement import eval_record


# ---------------------------------------------------------
# 1. Frozen development evaluation settings
# ---------------------------------------------------------

DEV_IDS = tuple(range(1, 31))

RATES = (500, 100)

TOLERANCES_MS = (50.0, 100.0)

METHODS = (
    "current",
    "asymmetric_50_20",
)

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261126


# ---------------------------------------------------------
# 2. Verify cohort completeness and scientific integrity
# ---------------------------------------------------------

def validate_cohort(rows, expected_ids):
    """Reject missing, duplicate, or inconsistent score rows."""

    expected_ids = tuple(expected_ids)

    if (
        not expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or any(
            type(rid) is not int or rid < 1
            for rid in expected_ids
        )
    ):
        raise ValueError(
            "Expected unique positive integer record IDs"
        )

    expected_keys = {
        (rid, fs, tolerance, method)
        for rid in expected_ids
        for fs in RATES
        for tolerance in TOLERANCES_MS
        for method in METHODS
    }

    indexed = {}

    for row in rows:

        key = (
            row["record_id"],
            row["fs_hz"],
            row["tolerance_ms"],
            row["method"],
        )

        if key in indexed:
            raise ValueError(
                f"Duplicate score row: {key}"
            )

        if key not in expected_keys:
            raise ValueError(
                f"Unexpected score row: {key}"
            )

        indexed[key] = row

        # All event counts must be nonnegative integers.
        for field in (
            "reference_beats",
            "detected_peaks",
            "tp",
            "fp",
            "fn",
            "unscored_outside_window",
        ):

            value = row[field]

            if type(value) is not int or value < 0:
                raise ValueError(
                    f"Invalid {field} in {key}"
                )

        if row["reference_beats"] == 0:
            raise ValueError(
                f"Empty reference in {key}"
            )

        # True positives and false negatives must
        # account for every expert reference beat.
        if (
            row["tp"] + row["fn"]
            != row["reference_beats"]
        ):
            raise ValueError(
                f"Inconsistent event counts in {key}"
            )

        # True positives and false positives must
        # account for every scored detection.
        if (
            row["tp"] + row["fp"]
            != row["detected_peaks"]
        ):
            raise ValueError(
                f"Inconsistent event counts in {key}"
            )

        errors = np.asarray(
            row["matched_timing_errors_ms"],
            dtype=float,
        )

        if (
            errors.ndim != 1
            or len(errors) != row["tp"]
            or not np.isfinite(errors).all()
        ):
            raise ValueError(
                f"Invalid timing errors in {key}"
            )

    # Never silently omit failed or missing records.
    missing = expected_keys.difference(indexed)

    if missing:
        raise ValueError(
            f"Missing score rows: {sorted(missing)[:4]}"
        )

    # Validate the same reference denominator is used
    # across algorithms, tolerances, and sampling rates.
    for rid in expected_ids:

        reference_counts = {
            indexed[
                (rid, fs, tolerance, method)
            ]["reference_beats"]
            for fs in RATES
            for tolerance in TOLERANCES_MS
            for method in METHODS
        }

        if len(reference_counts) != 1:
            raise ValueError(
                f"Reference denominator mismatch: LUDB {rid}"
            )

        for fs in RATES:
            for method in METHODS:

                strict = indexed[
                    (rid, fs, 50.0, method)
                ]

                lenient = indexed[
                    (rid, fs, 100.0, method)
                ]

                # Increasing the matching tolerance
                # cannot reduce the maximum match count.
                if (
                    strict["detected_peaks"]
                    != lenient["detected_peaks"]
                    or strict["unscored_outside_window"]
                    != lenient["unscored_outside_window"]
                    or strict["tp"] > lenient["tp"]
                ):
                    raise ValueError(
                        f"Tolerance inconsistency: "
                        f"LUDB {rid}, {fs} Hz, {method}"
                    )

    return indexed


# ---------------------------------------------------------
# 3. Calculate micro-averaged detection performance
# ---------------------------------------------------------

def _pooled_score(rows):
    """Pool event counts, rather than average record F1."""

    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        raise ValueError(
            "No scored events"
        )

    errors = np.asarray(
        [
            error
            for row in rows
            for error in row[
                "matched_timing_errors_ms"
            ]
        ],
        dtype=float,
    )

    absolute = np.abs(errors)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": 2 * tp / denominator,
        "unscored": sum(
            row["unscored_outside_window"]
            for row in rows
        ),
        "mean_abs_ms": (
            float(np.mean(absolute))
            if len(absolute)
            else None
        ),
        "p95_abs_ms": (
            float(np.percentile(absolute, 95))
            if len(absolute)
            else None
        ),
    }


# ---------------------------------------------------------
# 4. Paired bootstrap uncertainty estimation
# ---------------------------------------------------------

def paired_f1_bootstrap(
    indexed,
    record_ids,
    fs,
    tolerance_ms,
    *,
    replicates=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):
    """Resample complete records while preserving detector pairing.

    Both algorithms are evaluated on the same sampled records.
    Individual heartbeats are not treated as independent patients.
    """

    ids = tuple(record_ids)

    if (
        not ids
        or type(replicates) is not int
        or replicates < 1
    ):
        raise ValueError(
            "Require records and positive bootstrap repetitions"
        )

    baseline = [
        indexed[
            (rid, fs, tolerance_ms, METHODS[0])
        ]
        for rid in ids
    ]

    candidate = [
        indexed[
            (rid, fs, tolerance_ms, METHODS[1])
        ]
        for rid in ids
    ]

    observed_delta = (
        _pooled_score(candidate)["f1"]
        - _pooled_score(baseline)["f1"]
    )

    rng = np.random.default_rng(seed)

    draws = np.empty(
        replicates,
        dtype=float,
    )

    for i in range(replicates):

        # Resample record indices with replacement.
        selected = rng.integers(
            0,
            len(ids),
            size=len(ids),
        )

        old = _pooled_score([
            baseline[j]
            for j in selected
        ])

        new = _pooled_score([
            candidate[j]
            for j in selected
        ])

        draws[i] = (
            new["f1"] - old["f1"]
        )

    lower, upper = np.percentile(
        draws,
        [2.5, 97.5],
    )

    return (
        float(observed_delta),
        float(lower),
        float(upper),
    )


# ---------------------------------------------------------
# 5. Complete paired evaluation summary
# ---------------------------------------------------------

def summarize_cohort(
    rows,
    expected_ids,
    *,
    replicates=BOOTSTRAP_REPS,
):
    """Validate every recording before producing a comparison."""

    ids = tuple(expected_ids)

    indexed = validate_cohort(
        rows,
        ids,
    )

    output = []

    for fs in RATES:
        for tolerance in TOLERANCES_MS:

            old = _pooled_score([
                indexed[
                    (rid, fs, tolerance, METHODS[0])
                ]
                for rid in ids
            ])

            new = _pooled_score([
                indexed[
                    (rid, fs, tolerance, METHODS[1])
                ]
                for rid in ids
            ])

            delta, lower, upper = paired_f1_bootstrap(
                indexed,
                ids,
                fs,
                tolerance,
                replicates=replicates,
            )

            # Inspect every recording independently.
            # Aggregate improvement must not hide regressions.
            regressions = []

            for rid in ids:

                before = indexed[
                    (rid, fs, tolerance, METHODS[0])
                ]

                after = indexed[
                    (rid, fs, tolerance, METHODS[1])
                ]

                before_f1 = _pooled_score(
                    [before]
                )["f1"]

                after_f1 = _pooled_score(
                    [after]
                )["f1"]

                if (
                    after["fn"] > before["fn"]
                    or after_f1 < before_f1 - 1e-12
                ):
                    regressions.append(rid)

            output.append({
                "fs_hz": fs,
                "tolerance_ms": tolerance,
                "baseline": old,
                "candidate": new,
                "delta_f1": delta,
                "bootstrap_95_pct": (
                    lower,
                    upper,
                ),
                "regression_records": regressions,
            })

    return output


# ---------------------------------------------------------
# 6. Development-only execution
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    # Deliberately require the development-only flag.
    # No option permits running reserved LUDB records.
    parser.add_argument(
        "--dev-only",
        action="store_true",
        required=True,
        help=(
            "Evaluate ONLY existing local LUDB "
            "development records 1-30"
        ),
    )

    args = parser.parse_args()

    if not args.dev_only:
        parser.error(
            "The development-only flag is required"
        )

    rows = []

    for rid in DEV_IDS:

        # Existing evaluator rejects all non-development IDs.
        rows.extend(
            eval_record(rid)
        )

        print(
            f"LUDB {rid:03d}: evaluated"
        )

    report = summarize_cohort(
        rows,
        DEV_IDS,
    )

    print(
        "\n=== PAIRED R-PEAK PROTOCOL AUDIT: "
        "DEVELOPMENT ONLY ==="
    )

    for result in report:

        old = result["baseline"]
        new = result["candidate"]

        lower, upper = result[
            "bootstrap_95_pct"
        ]

        print(
            f"{result['fs_hz']} Hz | "
            f"+/-{int(result['tolerance_ms'])} ms | "
            f"F1 {old['f1']:.4f} -> {new['f1']:.4f} | "
            f"delta {result['delta_f1']:+.4f} | "
            f"paired record bootstrap 95% "
            f"[{lower:+.4f}, {upper:+.4f}]"
        )

        print(
            f"  baseline TP/FP/FN="
            f"{old['tp']}/{old['fp']}/{old['fn']} | "
            f"candidate="
            f"{new['tp']}/{new['fp']}/{new['fn']} | "
            f"unscored="
            f"{old['unscored']}/{new['unscored']} | "
            f"regression records="
            f"{result['regression_records']}"
        )

        print(
            "  matched-event absolute error (ms): "
            f"mean {old['mean_abs_ms']:.2f} -> "
            f"{new['mean_abs_ms']:.2f}, "
            f"p95 {old['p95_abs_ms']:.2f} -> "
            f"{new['p95_abs_ms']:.2f}"
        )

    print(
        "\nNo independent-data access or writes; "
        "candidate unchanged."
    )

    print(
        "Bootstrap interval is descriptive, "
        "not clinical validation."
    )


if __name__ == "__main__":
    main()