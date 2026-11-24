
"""Read-only audit of previously saved LUDB 31-100 expert QRS errors.

This audits record-level QRS delineation with expert R-peaks supplied.

It does not validate automated R-peak detection, ST amplitude, or diagnosis.
LUDB 101-200 and PTB-XL fold 10 remain untouched.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HOLDOUT = (
    ROOT
    / "docs/clinical_validation/ludb_qrs_v4_holdout.csv"
)

RECORDS = tuple(range(31, 101))

METHODS = (
    "current",
    "qrs_v4",
)

ERRORS = (
    "onset_error_ms",
    "offset_error_ms",
    "duration_error_ms",
)

REQUIRED = {
    "record_id",
    "method",
    "status",
    "reason",
    *ERRORS,
}


def verify_holdout(frame, *, expected_records=RECORDS):
    """Require a complete, uniquely paired saved experiment."""

    if (
        not isinstance(frame, pd.DataFrame)
        or not REQUIRED.issubset(frame.columns)
    ):
        raise ValueError(
            "Missing required LUDB holdout columns"
        )

    if frame.empty:
        raise ValueError("Empty LUDB holdout")

    expected = tuple(expected_records)

    if (
        not expected
        or len(expected) != len(set(expected))
    ):
        raise ValueError(
            "Invalid expected LUDB record IDs"
        )

    data = frame.copy()

    ids = pd.to_numeric(
        data["record_id"],
        errors="coerce",
    )

    if (
        ids.isna().any()
        or not np.isfinite(ids).all()
        or not (ids % 1 == 0).all()
    ):
        raise ValueError(
            "Invalid LUDB record IDs"
        )

    data["record_id"] = ids.astype(int)

    # Require the exact expected recording IDs.
    # This prevents silently including reserved records.
    if set(data["record_id"]) != set(expected):
        raise ValueError(
            "Unexpected or missing LUDB record IDs"
        )

    if data[
        ["record_id", "method"]
    ].isna().any().any():
        raise ValueError(
            "Missing record or method"
        )

    # A record cannot appear twice for one method.
    if data.duplicated(
        ["record_id", "method"]
    ).any():
        raise ValueError(
            "Duplicate record-method pair"
        )

    if set(data["method"]) != set(METHODS):
        raise ValueError(
            "Unexpected QRS method"
        )

    # Require both algorithms for every recording.
    for rid, group in data.groupby("record_id"):

        if (
            len(group) != len(METHODS)
            or set(group["method"]) != set(METHODS)
        ):
            raise ValueError(
                f"Incomplete method pairing for LUDB {rid}"
            )

    # Do not silently exclude failed recordings.
    if not data["status"].eq("success").all():

        failed = sorted(
            set(
                data.loc[
                    data["status"] != "success",
                    "record_id",
                ]
            )
        )

        raise ValueError(
            "Incomplete held-out evaluation; "
            f"failed records: {failed}"
        )

    # All signed landmark errors must be real numbers.
    for column in ERRORS:

        try:
            data[column] = pd.to_numeric(
                data[column],
                errors="raise",
            )

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {column}"
            ) from exc

        if not np.isfinite(
            data[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Nonfinite {column}"
            )

    # Mathematical consistency:
    #
    # duration = offset - onset
    #
    # Therefore:
    # duration_error = offset_error - onset_error

    if not np.allclose(
        data["duration_error_ms"].to_numpy(
            dtype=float
        ),
        (
            data["offset_error_ms"]
            - data["onset_error_ms"]
        ).to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError(
            "QRS duration/error identity mismatch"
        )

    return (
        data.sort_values(
            ["record_id", "method"]
        )
        .reset_index(drop=True)
    )


def _describe(errors):
    """Summarize signed and absolute errors in milliseconds."""

    values = np.asarray(
        errors,
        dtype=float,
    )

    magnitudes = np.abs(values)

    return {
        "median_signed_error_ms": float(
            np.median(values)
        ),

        "median_absolute_error_ms": float(
            np.median(magnitudes)
        ),

        "mean_absolute_error_ms": float(
            magnitudes.mean()
        ),

        "p95_absolute_error_ms": float(
            np.percentile(magnitudes, 95)
        ),

        "max_absolute_error_ms": float(
            magnitudes.max()
        ),
    }


def make_report(
    frame,
    *,
    expected_records=RECORDS,
    draws=3000,
    seed=42,
):
    """Compare QRS errors and flag record-level regressions.

    Bootstrap resampling uses recording IDs as units,
    with matched current/V4 values in every draw.
    """

    if (
        not isinstance(draws, int)
        or draws < 1
    ):
        raise ValueError(
            "Bootstrap draws must be positive"
        )

    data = verify_holdout(
        frame,
        expected_records=expected_records,
    )

    ids = sorted(
        set(data["record_id"])
    )

    # Align both algorithms to the same recordings.
    indexed = {
        method: (
            data.loc[
                data["method"].eq(method)
            ]
            .set_index("record_id")
            .loc[ids]
        )
        for method in METHODS
    }

    rng = np.random.default_rng(seed)

    # Paired resampling: both methods use the
    # same sampled recordings in every iteration.
    resamples = rng.integers(
        0,
        len(ids),
        size=(draws, len(ids)),
    )

    results = {}

    for column in ERRORS:

        old = indexed["current"][
            column
        ].to_numpy(dtype=float)

        new = indexed["qrs_v4"][
            column
        ].to_numpy(dtype=float)

        before = np.abs(old)
        after = np.abs(new)

        # Positive delta: V4 is worse.
        # Negative delta: V4 is better.
        delta = after - before

        boot = (
            delta[resamples]
            .mean(axis=1)
        )

        # Identify the largest accuracy regressions.
        worst = sorted(
            (
                {
                    "record_id": int(rid),

                    "current_absolute_error_ms": float(a),

                    "qrs_v4_absolute_error_ms": float(b),

                    "increase_ms": float(d),
                }

                for rid, a, b, d in zip(
                    ids,
                    before,
                    after,
                    delta,
                )

                if d > 1e-9
            ),

            key=lambda item: (
                -item["increase_ms"],
                item["record_id"],
            ),
        )[:5]

        results[column] = {

            "current": _describe(old),

            "qrs_v4": _describe(new),

            "paired": {
                "improved": int(
                    np.sum(delta < -1e-9)
                ),

                "worsened": int(
                    np.sum(delta > 1e-9)
                ),

                "unchanged": int(
                    np.sum(
                        np.abs(delta) <= 1e-9
                    )
                ),

                "mean_absolute_error_difference_v4_minus_current_ms": float(
                    delta.mean()
                ),

                "bootstrap_95pct_interval_difference_ms": [
                    float(x)
                    for x in np.percentile(
                        boot,
                        [2.5, 97.5],
                    )
                ],

                "largest_regressions": worst,
            },
        }

    return {
        "audit": (
            "saved_ludb_qrs_expert_reference_v1"
        ),

        "evaluation_records": len(ids),

        "record_id_min": int(ids[0]),
        "record_id_max": int(ids[-1]),

        "methods": list(METHODS),

        "successful_paired_records": len(ids),

        "bootstrap": {
            "resampling_unit": "LUDB_recording",
            "seed": seed,
            "draws": draws,
            "interval_kind": (
                "descriptive_record_bootstrap"
            ),
        },

        "metrics": results,

        "reserved_ludb_101_to_200_evaluated": False,
        "ptbxl_fold10_evaluated": False,

        "limitations": [
            (
                "Saved results use expert-supplied R peaks; "
                "R-peak detection is not evaluated"
            ),
            (
                "LUDB 31-100 was already evaluated; "
                "this is not a fresh untouched holdout"
            ),
            (
                "Errors are against an annotated global "
                "QRS reference, not ST amplitude ground truth"
            ),
            (
                "Per-record summaries do not establish "
                "beat-level or clinical reliability"
            ),
            (
                "No model promotion, tuning, new waveform "
                "processing or diagnosis"
            ),
        ],
    }


def main():
    """Audit committed data and print JSON without writing files."""

    result = make_report(
        pd.read_csv(HOLDOUT)
    )

    print(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        )
    )

    print(
        "READ-ONLY: no waveforms, model changes, "
        "or reserved-set evaluation"
    )


if __name__ == "__main__":
    main()