
"""Compare QRS detectors against LUDB expert annotations.

Evaluates:
- Original QRS detector
- Common-grid detector
- QRS V2
- QRS V3 hybrid

Uses LUDB development records 1-30 only.

Run:
    python -m scripts.benchmark_ludb_qrs_v3
"""

from pathlib import Path
import csv

import numpy as np
import wfdb

from src.dsp.filters import filter_ecg
from src.features import fiducials as old
from src.features import fiducials_common_grid as common
from src.features import fiducials_qrs_v2 as new
from src.features import fiducials_qrs_v3 as hybrid

from scripts.benchmark_ludb_landmarks import (
    qrs_triplets,
    expert_global_beat_references,
)


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/ludb_reference/data"

OUTPUT = (
    ROOT
    / "docs/clinical_validation/ludb_qrs_v3_development.csv"
)


def evaluate(record_id):
    path = DATA / str(record_id)

    signal, fields = wfdb.rdsamp(str(path))

    if fields["fs"] != 500 or signal.shape[1] != 12:
        raise ValueError(
            "Expected 12-lead 500 Hz LUDB waveform"
        )

    peaks = np.array(
        [
            x[1]
            for x in qrs_triplets(
                wfdb.rdann(str(path), "ii")
            )
        ],
        dtype=int,
    )

    refs = expert_global_beat_references(
        path,
        peaks,
    )

    lookup = {
        r["anchor"]: r
        for r in refs
    }

    pre = round(old.BEAT_PRE_MS * 0.5)
    post = round(old.BEAT_POST_MS * 0.5)

    rr = np.diff(peaks)
    median_rr = float(np.median(rr))

    selected = []

    for i, peak in enumerate(peaks):
        if (
            peak - pre < 0
            or peak + post >= len(signal)
            or peak not in lookup
        ):
            continue

        if (
            i > 0
            and abs(rr[i - 1] - median_rr) / median_rr
            > old.ECTOPIC_RR_TOLERANCE
        ):
            continue

        selected.append(lookup[peak])

    if len(selected) < 3:
        raise ValueError(
            "Insufficient matched annotated beats"
        )

    reference_on = float(
        np.median([
            (r["onset"] - r["anchor"]) * 2
            for r in selected
        ])
    )

    reference_off = float(
        np.median([
            (r["offset"] - r["anchor"]) * 2
            for r in selected
        ])
    )

    filtered = filter_ecg(signal, fs=500)

    rows = []

    methods = [
        ("current", old.analyze_12lead_features),
        ("common_grid", common.analyze_12lead_features),
        ("qrs_v2", new.analyze_12lead_features),
        ("qrs_v3", hybrid.analyze_12lead_features),
    ]

    for method, analyzer in methods:
        result = analyzer(
            filtered,
            peaks,
            lead_names=fields["sig_name"],
            fs=500,
        )

        if result["status"] != "success":
            rows.append({
                "record_id": record_id,
                "method": method,
                "status": result["status"],
                "reason": result.get("reason", ""),
            })
            continue

        onset = (
            result["qrs_onset_idx"] - pre
        ) * 2

        offset = (
            result["qrs_offset_idx"] - pre
        ) * 2

        rows.append({
            "record_id": record_id,
            "method": method,
            "status": "success",
            "onset_error_ms": onset - reference_on,
            "offset_error_ms": offset - reference_off,
            "duration_error_ms": (
                (offset - onset)
                - (reference_off - reference_on)
            ),
            "isolated_leads": len(
                result.get("isolated_boundaries", [])
            ),
        })

    return rows


def main():
    rows = []

    for rid in range(1, 31):
        try:
            rows.extend(evaluate(rid))
            print(f"LUDB {rid:03d}: evaluated")

        except Exception as exc:
            print(
                f"LUDB {rid:03d}: FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

            for method in (
                "current",
                "common_grid",
                "qrs_v2",
                "qrs_v3",
            ):
                rows.append({
                    "record_id": rid,
                    "method": method,
                    "status": "error",
                    "reason": str(exc),
                })

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "record_id",
        "method",
        "status",
        "reason",
        "onset_error_ms",
        "offset_error_ms",
        "duration_error_ms",
        "isolated_leads",
    ]

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        "\nEXPERT LANDMARK ERRORS "
        "(lower absolute error is better)"
    )

    for method in (
        "current",
        "common_grid",
        "qrs_v2",
        "qrs_v3",
    ):
        subset = [
            r for r in rows
            if (
                r["method"] == method
                and r["status"] == "success"
            )
        ]

        print(
            f"\n{method}: {len(subset)}/30 successful"
        )

        if not subset:
            continue

        for col in (
            "onset_error_ms",
            "offset_error_ms",
            "duration_error_ms",
        ):
            v = np.array(
                [r[col] for r in subset],
                float,
            )

            print(
                f"{col}: "
                f"signed_median={np.median(v):+.1f} ms, "
                f"median_abs={np.median(abs(v)):.1f} ms, "
                f"p95_abs={np.percentile(abs(v), 95):.1f} ms"
            )

    baseline = {
        r["record_id"]: r
        for r in rows
        if (
            r["method"] == "current"
            and r["status"] == "success"
        )
    }

    modified = {
        r["record_id"]: r
        for r in rows
        if (
            r["method"] == "qrs_v3"
            and r["status"] == "success"
        )
    }

    improved = 0
    worsened = 0
    tied = 0

    print(
        "\nRECORD-LEVEL QRS DURATION REGRESSIONS"
    )

    for rid in sorted(
        baseline.keys() & modified.keys()
    ):
        x = abs(
            baseline[rid]["duration_error_ms"]
        )
        y = abs(
            modified[rid]["duration_error_ms"]
        )

        if y < x:
            improved += 1

        elif y > x:
            worsened += 1
            print(
                f"LUDB {rid}: "
                f"{x:.1f} -> {y:.1f} ms"
            )

        else:
            tied += 1

    print(
        f"Improved {improved}, "
        f"worsened {worsened}, "
        f"unchanged {tied}"
    )

    print("Saved:", OUTPUT)
    print(
        "Experimental only: "
        "do not connect qrs_v3 to triage."
    )


if __name__ == "__main__":
    main()
