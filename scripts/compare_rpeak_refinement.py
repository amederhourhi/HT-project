
"""Compare an isolated asymmetric R-peak snap against deployed CardioSense.

Development-only LUDB 1-30. No reserved data, training, file writes or changes
in production implementation. Experiment results must not be treated as an
independent holdout or a deployment approval.
"""

import numpy as np
from scipy import signal
from scipy.signal import resample_poly

from src.features.r_peaks import (
    detect_r_peaks,
    qrs_energy,
)
from scripts.benchmark_ludb_rpeaks_phase2 import (
    DATA,
    match_peaks,
    summarize,
)


DEV_IDS = tuple(range(1, 31))
RATES = (500, 100)
TOLERANCES_MS = (50.0, 100.0)
WINDOW_MARGIN_MS = 100.0


def snap_candidate(
    samples,
    index,
    fs,
    *,
    back_ms=50.0,
    forward_ms=50.0,
):
    """Select strongest absolute voltage within a candidate window."""

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling frequency")

    if (
        not np.isfinite(back_ms)
        or not np.isfinite(forward_ms)
        or back_ms < 0
        or forward_ms < 0
    ):
        raise ValueError("Invalid refinement window")

    x = np.asarray(samples, dtype=float)

    if (
        x.ndim != 1
        or x.size == 0
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite 1D ECG signal")

    if (
        not isinstance(index, (int, np.integer))
        or index < 0
        or index >= len(x)
    ):
        raise ValueError("Invalid energy candidate index")

    before = int(back_ms * fs / 1000.0)
    after = int(forward_ms * fs / 1000.0)

    lo = max(0, index - before)
    hi = min(len(x), index + after + 1)

    return int(
        lo + np.argmax(np.abs(x[lo:hi]))
    )


def detect_snap_variant(
    samples,
    fs,
    *,
    back_ms=50.0,
    forward_ms=20.0,
):
    """Keep energy detection unchanged; vary only the refinement window."""

    x = np.asarray(samples, dtype=float)

    if (
        x.ndim != 1
        or len(x) < 3
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite 1D ECG signal")

    if not np.isfinite(fs) or fs < 4:
        raise ValueError("Invalid sampling rate")

    energy = qrs_energy(x)

    threshold = (
        0.3 * np.percentile(energy, 98)
    )

    candidates, _ = signal.find_peaks(
        energy,
        height=threshold,
        distance=int(0.25 * fs),
    )

    refined = [
        snap_candidate(
            x,
            int(p),
            fs,
            back_ms=back_ms,
            forward_ms=forward_ms,
        )
        for p in candidates
    ]

    return np.asarray(
        sorted(set(refined)),
        dtype=int,
    )


def score_annotated_window(
    reference,
    detected,
    fs,
    tolerance_ms,
):
    """Score only within a fixed expert-annotation interval.

    A 100 ms margin is used for both matching tolerances.

    Detections outside the interval are marked unscored,
    not automatically correct or incorrect.
    """

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling frequency")

    ref = np.asarray(reference, dtype=int)
    det = np.asarray(detected, dtype=int)

    if ref.ndim != 1 or ref.size == 0:
        raise ValueError("Missing expert reference")

    lower_ms = (
        ref[0] * 1000.0 / fs - WINDOW_MARGIN_MS
    )

    upper_ms = (
        ref[-1] * 1000.0 / fs + WINDOW_MARGIN_MS
    )

    times_ms = det * 1000.0 / fs

    keep = (
        (times_ms >= lower_ms)
        & (times_ms <= upper_ms)
    )

    result = match_peaks(
        ref,
        det[keep],
        fs=fs,
        tolerance_ms=tolerance_ms,
    )

    result["unscored_outside_window"] = int(
        (~keep).sum()
    )

    return result


def eval_record(record_id):
    """Evaluate one locally available LUDB development recording."""

    if (
        type(record_id) is not int
        or record_id not in DEV_IDS
    ):
        raise ValueError(
            "Only LUDB development records 1-30 permitted"
        )

    for ext in ("hea", "dat", "ii"):
        path = DATA / f"{record_id}.{ext}"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing LUDB {record_id}.{ext}"
            )

    import wfdb

    from scripts.benchmark_ludb_landmarks import (
        qrs_triplets,
    )
    from src.dsp.filters import filter_ecg

    raw, meta = wfdb.rdsamp(
        str(DATA / str(record_id))
    )

    names = [
        str(s).lower()
        for s in meta["sig_name"]
    ]

    if (
        float(meta["fs"]) != 500
        or raw.ndim != 2
        or raw.shape[1] != 12
        or names.count("ii") != 1
    ):
        raise ValueError(
            f"LUDB {record_id}: unexpected signal format"
        )

    lead_ii = raw[
        :, names.index("ii")
    ]

    annotation = wfdb.rdann(
        str(DATA / str(record_id)),
        "ii",
    )

    ref500 = np.asarray(
        [
            t[1]
            for t in qrs_triplets(annotation)
        ],
        dtype=int,
    )

    rows = []

    for fs in RATES:

        wave = (
            lead_ii
            if fs == 500
            else resample_poly(lead_ii, 1, 5)
        )

        refs = (
            ref500
            if fs == 500
            else np.rint(ref500 / 5).astype(int)
        )

        filtered = filter_ecg(
            wave,
            fs=fs,
        )

        # Actual unchanged production detector.
        deployed = detect_r_peaks(
            filtered,
            fs=fs,
        )

        # Reproduce the production detector with a
        # symmetric 50/50 ms window.
        parity = detect_snap_variant(
            filtered,
            fs,
            back_ms=50.0,
            forward_ms=50.0,
        )

        if not np.array_equal(
            deployed,
            parity,
        ):
            raise AssertionError(
                "Baseline reproduction differs "
                "from deployed detector"
            )

        # Experimental asymmetric 50/20 ms window.
        candidate = detect_snap_variant(
            filtered,
            fs,
            back_ms=50.0,
            forward_ms=20.0,
        )

        for tolerance in TOLERANCES_MS:

            for method, peaks in (
                ("current", deployed),
                ("asymmetric_50_20", candidate),
            ):

                result = score_annotated_window(
                    refs,
                    peaks,
                    fs,
                    tolerance,
                )

                rows.append({
                    "record_id": record_id,
                    "fs_hz": fs,
                    "tolerance_ms": tolerance,
                    "method": method,
                    **result,
                })

    return rows


def main():
    all_rows = []

    for record_id in DEV_IDS:

        all_rows.extend(
            eval_record(record_id)
        )

        print(
            f"LUDB {record_id:03d}: evaluated"
        )

    print(
        "\n=== ANNOTATION-WINDOW "
        "REFINEMENT COMPARISON ==="
    )

    for fs in RATES:

        for tolerance in TOLERANCES_MS:

            for method in (
                "current",
                "asymmetric_50_20",
            ):

                subset = [
                    row
                    for row in all_rows
                    if (
                        row["fs_hz"] == fs
                        and row["tolerance_ms"] == tolerance
                        and row["method"] == method
                    )
                ]

                summary = summarize(subset)

                unscored = sum(
                    row["unscored_outside_window"]
                    for row in subset
                )

                print(
                    f"{fs} Hz | "
                    f"+/-{int(tolerance)} ms | "
                    f"{method:16} | "
                    f"TP={summary['tp']:3d} "
                    f"FP={summary['fp']:3d} "
                    f"FN={summary['fn']:3d} "
                    f"F1={summary['f1']:.4f} | "
                    f"unscored={unscored}"
                )

    print(
        "\n=== LUDB 023 MATCHED TIMING ERRORS "
        "AT +/-100 MS ==="
    )

    for fs in RATES:

        for method in (
            "current",
            "asymmetric_50_20",
        ):

            score = next(
                row
                for row in all_rows
                if (
                    row["record_id"] == 23
                    and row["fs_hz"] == fs
                    and row["tolerance_ms"] == 100.0
                    and row["method"] == method
                )
            )

            print(
                f"{fs} Hz {method}: "
                f"{score['matched_timing_errors_ms']}"
            )

    print(
        "\n=== STRICT-TOLERANCE "
        "RECORD-LEVEL REGRESSIONS ==="
    )

    for fs in RATES:

        strict = [
            row
            for row in all_rows
            if (
                row["fs_hz"] == fs
                and row["tolerance_ms"] == 50.0
            )
        ]

        scores = {
            (row["record_id"], row["method"]): row
            for row in strict
        }

        regressions = []
        improvements = []

        for rid in DEV_IDS:

            old = scores[(rid, "current")]
            new = scores[(rid, "asymmetric_50_20")]

            # One FP and one FN may describe one
            # incorrectly localized R-peak event.
            delta = (
                new["fp"] + new["fn"]
                - old["fp"] - old["fn"]
            )

            if delta > 0:
                regressions.append((
                    rid,
                    delta,
                    old["fp"],
                    old["fn"],
                    new["fp"],
                    new["fn"],
                ))

            elif delta < 0:
                improvements.append((
                    rid,
                    delta,
                ))

        print(
            f"{fs} Hz | "
            f"worsened: {regressions} | "
            f"improved: {improvements}"
        )

    print(
        "\nNo production files changed; "
        "development-only evidence; "
        "unscored edges remain unverified."
    )


if __name__ == "__main__":
    main()