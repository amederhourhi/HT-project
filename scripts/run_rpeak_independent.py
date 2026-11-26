
"""CardioSense Phase II: frozen independent R-peak evaluation runner.

Development smoke test:
    python -m scripts.run_rpeak_independent --dev-smoke

Reserved evaluation:
    Disabled unless explicitly authorized on a clean,
    frozen main commit with the full expected commit SHA.

No automatic downloads, detector tuning, training, or production changes.
"""

import argparse
import hashlib
import json
import re
import subprocess

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


from scripts.benchmark_ludb_rpeaks_phase2 import (
    DATA,
    _validated_samples,
)

from scripts.benchmark_ludb_landmarks import qrs_triplets

from scripts.compare_rpeak_refinement import (
    detect_snap_variant,
    score_annotated_window,
)

from scripts.audit_rpeak_pairwise_protocol import (
    summarize_cohort,
    validate_cohort,
)

from src.dsp.filters import filter_ecg
from src.features.r_peaks import detect_r_peaks


# ---------------------------------------------------------
# 1. Fixed cohorts and execution configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DEV_IDS = tuple(range(1, 31))
HOLDOUT_IDS = tuple(range(101, 201))

RATES = (500, 100)
TOLERANCES_MS = (50.0, 100.0)

METHODS = (
    "current",
    "asymmetric_50_20",
)

CONFIRMATION = "AUTHORIZE_LUDB_101_200_ONE_TIME"

OUTPUT = (
    ROOT
    / "artifacts"
    / "rpeak_independent"
    / "ludb_101_200_result.json"
)

ATTEMPT_MARKER = (
    ROOT
    / "artifacts"
    / "rpeak_independent"
    / "ludb_101_200_attempt.json"
)


# ---------------------------------------------------------
# 2. Validate an exact, predetermined cohort
# ---------------------------------------------------------

def check_cohort(record_ids, allowed_ids):
    """Prevent accidental changes to the cohort."""

    ids = tuple(record_ids)

    if ids != tuple(allowed_ids):
        raise ValueError(
            "Recording cohort differs from the "
            "predefined evaluation cohort"
        )

    return ids


def preflight_files(record_ids, data_dir=DATA):
    """Check every required file before reading any waveform."""

    missing = []

    for rid in record_ids:
        for extension in ("hea", "dat", "ii"):

            path = data_dir / f"{rid}.{extension}"

            if not path.is_file():
                missing.append(str(path))

    if missing:
        preview = "\n".join(missing[:12])

        raise FileNotFoundError(
            f"{len(missing)} required files are missing.\n"
            f"First missing files:\n{preview}\n"
            "No evaluation has started."
        )


# ---------------------------------------------------------
# 3. Evaluate both frozen methods on one recording
# ---------------------------------------------------------

def evaluate_local_record(
    record_id,
    *,
    permitted_ids,
    data_dir=DATA,
):
    """Run both methods with identical preprocessing.

    The caller must explicitly provide an allowed cohort.
    """

    if (
        type(record_id) is not int
        or record_id not in permitted_ids
    ):
        raise ValueError(
            "Record ID outside permitted cohort"
        )

    import wfdb

    base = data_dir / str(record_id)

    ecg, fields = wfdb.rdsamp(str(base))

    names = [
        str(name).lower()
        for name in fields["sig_name"]
    ]

    if (
        float(fields["fs"]) != 500.0
        or ecg.ndim != 2
        or ecg.shape[1] != 12
        or names.count("ii") != 1
        or not np.isfinite(ecg).all()
    ):
        raise ValueError(
            f"LUDB {record_id}: invalid signal contract"
        )

    lead_ii = ecg[:, names.index("ii")]

    ann = wfdb.rdann(str(base), "ii")

    reference500 = _validated_samples(
        [
            triplet[1]
            for triplet in qrs_triplets(ann)
        ],
        "expert QRS references",
    )

    rows = []

    for fs in RATES:

        if fs == 500:
            samples = lead_ii
            reference = reference500

        else:
            # Resample waveform, never use expert peaks
            # to guide the detector.
            samples = resample_poly(
                lead_ii,
                1,
                5,
            )

            reference = _validated_samples(
                np.rint(
                    reference500 / 5
                ).astype(np.int64),
                "resampled expert references",
            )

        filtered = filter_ecg(
            samples,
            fs=fs,
        )

        # Unmodified production detector.
        baseline = detect_r_peaks(
            filtered,
            fs=fs,
        )

        # Verify our symmetric implementation still
        # reproduces the production baseline exactly.
        parity = detect_snap_variant(
            filtered,
            fs,
            back_ms=50.0,
            forward_ms=50.0,
        )

        if not np.array_equal(
            baseline,
            parity,
        ):
            raise AssertionError(
                f"Baseline parity failure: "
                f"LUDB {record_id}, {fs} Hz"
            )

        # Frozen research candidate.
        candidate = detect_snap_variant(
            filtered,
            fs,
            back_ms=50.0,
            forward_ms=20.0,
        )

        for tolerance in TOLERANCES_MS:

            for method, detections in (
                ("current", baseline),
                ("asymmetric_50_20", candidate),
            ):

                score = score_annotated_window(
                    reference,
                    detections,
                    fs,
                    tolerance,
                )

                rows.append({
                    "record_id": record_id,
                    "fs_hz": fs,
                    "tolerance_ms": tolerance,
                    "method": method,
                    **score,
                })

    return rows


# ---------------------------------------------------------
# 4. Verify frozen Git revision before reserved access
# ---------------------------------------------------------

def git_output(*arguments):
    """Run a read-only Git command in the repository."""

    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def authorize_holdout(
    confirmation,
    expected_sha,
):
    """Require an exact clean, frozen main commit."""

    if confirmation != CONFIRMATION:
        raise PermissionError(
            "Explicit independent-evaluation "
            "confirmation is required"
        )

    if not re.fullmatch(
        r"[0-9a-f]{40}",
        expected_sha or "",
    ):
        raise PermissionError(
            "A full 40-character commit SHA is required"
        )

    branch = git_output(
        "branch",
        "--show-current",
    )

    actual_sha = git_output(
        "rev-parse",
        "HEAD",
    )

    status = git_output(
        "status",
        "--porcelain",
        "--untracked-files=all",
    )

    if branch != "main":
        raise PermissionError(
            "Reserved evaluation requires main"
        )

    if actual_sha != expected_sha:
        raise PermissionError(
            "Frozen commit SHA does not match HEAD"
        )

    if status:
        raise PermissionError(
            "Working tree must be completely clean"
        )

    return actual_sha


# ---------------------------------------------------------
# 5. Dataset checksums for reproducibility
# ---------------------------------------------------------

def file_sha256(path):
    """Calculate SHA256 without loading the whole file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def dataset_manifest(record_ids, data_dir=DATA):
    """Record the exact local files used in evaluation."""

    manifest = []

    for rid in record_ids:

        for extension in ("hea", "dat", "ii"):

            path = data_dir / f"{rid}.{extension}"

            manifest.append({
                "record_id": rid,
                "extension": extension,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })

    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    )

    return {
        "files": manifest,
        "manifest_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


# ---------------------------------------------------------
# 6. Persistent marker prevents silent repeat execution
# ---------------------------------------------------------

def create_attempt_marker(
    path,
    *,
    commit_sha,
):
    """Create an exclusive marker before evaluation starts."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "study": "P2-RPEAK-03",
        "status": "started",
        "commit_sha": commit_sha,
        "record_ids": [101, 200],
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    # Mode "x" refuses to overwrite an existing file.
    with path.open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )
        handle.write("\n")


# ---------------------------------------------------------
# 7. Development smoke test: no reserved data
# ---------------------------------------------------------

def run_development_smoke():
    """Compare the new runner with previously frozen results."""

    from scripts.compare_rpeak_refinement import (
        eval_record,
    )

    preflight_files(DEV_IDS)

    rows = []

    for rid in DEV_IDS:

        current = evaluate_local_record(
            rid,
            permitted_ids=DEV_IDS,
        )

        previous = eval_record(rid)

        # Verify equality of the complete evaluation
        # outputs, not merely their summary scores.
        if current != previous:
            raise AssertionError(
                f"Development parity failed: LUDB {rid:03d}"
            )

        rows.extend(current)

        print(
            f"LUDB {rid:03d}: parity PASS"
        )

    validate_cohort(
        rows,
        DEV_IDS,
    )

    report = summarize_cohort(
        rows,
        DEV_IDS,
    )

    print(
        "\n=== DEVELOPMENT RUNNER PARITY ==="
    )

    print_summary(report)

    print(
        "\nPASS: runner reproduces frozen "
        "development evaluations exactly."
    )

    print(
        "No reserved data accessed. No files written."
    )


# ---------------------------------------------------------
# 8. Human-readable results
# ---------------------------------------------------------

def print_summary(report):
    """Print paired results and regressions."""

    for row in report:

        old = row["baseline"]
        new = row["candidate"]

        lower, upper = row[
            "bootstrap_95_pct"
        ]

        print(
            f"{row['fs_hz']} Hz | "
            f"+/-{int(row['tolerance_ms'])} ms | "
            f"F1 {old['f1']:.4f} -> "
            f"{new['f1']:.4f} | "
            f"delta {row['delta_f1']:+.4f} | "
            f"95% bootstrap "
            f"[{lower:+.4f}, {upper:+.4f}]"
        )

        print(
            "  TP/FP/FN: "
            f"{old['tp']}/{old['fp']}/{old['fn']} -> "
            f"{new['tp']}/{new['fp']}/{new['fn']}"
        )

        print(
            "  Unscored: "
            f"{old['unscored']} -> {new['unscored']} | "
            "Regressions: "
            f"{row['regression_records']}"
        )


# ---------------------------------------------------------
# 9. Protected, one-time independent evaluation
# ---------------------------------------------------------

def run_holdout(
    *,
    confirmation,
    expected_sha,
):
    """Evaluate the complete reserved cohort once authorized."""

    commit_sha = authorize_holdout(
        confirmation,
        expected_sha,
    )

    check_cohort(
        HOLDOUT_IDS,
        tuple(range(101, 201)),
    )

    # Fail before accessing any waveform if data is missing.
    preflight_files(HOLDOUT_IDS)

    if OUTPUT.exists():
        raise FileExistsError(
            "Independent result already exists"
        )

    # Claim the attempt before any ECG is evaluated.
    # An interrupted run requires explicit audit;
    # deleting the marker is not part of this workflow.
    create_attempt_marker(
        ATTEMPT_MARKER,
        commit_sha=commit_sha,
    )

    rows = []
    failures = []

    try:
        manifest = dataset_manifest(
            HOLDOUT_IDS,
        )

        for rid in HOLDOUT_IDS:

            try:
                rows.extend(
                    evaluate_local_record(
                        rid,
                        permitted_ids=HOLDOUT_IDS,
                    )
                )

                print(
                    f"LUDB {rid:03d}: evaluated"
                )

            except Exception as exc:

                failures.append({
                    "record_id": rid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

                print(
                    f"LUDB {rid:03d}: FAILED: {exc}"
                )

        # Never calculate a selected-subset comparison.
        if failures:
            report = None
            status = "incomplete"

        else:
            report = summarize_cohort(
                rows,
                HOLDOUT_IDS,
            )
            status = "complete"

        result = {
            "study": "P2-RPEAK-03",
            "status": status,
            "cohort_role": "reserved_one_time_evaluation",
            "commit_sha": commit_sha,
            "record_ids": list(HOLDOUT_IDS),
            "reference": "lead_ii_expert_N_peaks",
            "scoring": (
                "first_to_last_annotated_QRS_"
                "plus_100ms_margin"
            ),
            "rate_100_source": (
                "resampled_from_native_500Hz"
            ),
            "methods": list(METHODS),
            "rates_hz": list(RATES),
            "tolerances_ms": list(TOLERANCES_MS),
            "dataset": manifest,
            "failures": failures,
            "per_record": rows,
            "summary": report,
        }

        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Never overwrite a previous independent result.
        with OUTPUT.open(
            "x",
            encoding="utf-8",
        ) as handle:
            json.dump(
                result,
                handle,
                indent=2,
                allow_nan=False,
            )

            handle.write("\n")

        print(
            "\nIndependent result saved:",
            OUTPUT,
        )

        if status != "complete":
            raise RuntimeError(
                "Independent evaluation incomplete. "
                "Review the saved failure report. "
                "Do not rerun or select successful records."
            )

        print_summary(report)

        print(
            "\nIndependent evaluation completed. "
            "Candidate remains research-only."
        )

    except Exception:
        print(
            "\nEvaluation attempt recorded. "
            "Do not silently repeat or discard it."
        )
        raise


# ---------------------------------------------------------
# 10. Command-line entry point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(
        required=True,
    )

    mode.add_argument(
        "--dev-smoke",
        action="store_true",
        help="Evaluate only LUDB 1-30",
    )

    mode.add_argument(
        "--holdout",
        action="store_true",
        help="Protected independent evaluation",
    )

    parser.add_argument(
        "--confirm",
        default=None,
    )

    parser.add_argument(
        "--frozen-commit",
        default=None,
    )

    args = parser.parse_args()

    if args.dev_smoke:

        if (
            args.confirm is not None
            or args.frozen_commit is not None
        ):
            parser.error(
                "Development mode does not accept "
                "reserved-evaluation authorization"
            )

        run_development_smoke()

    else:
        run_holdout(
            confirmation=args.confirm,
            expected_sha=args.frozen_commit,
        )


if __name__ == "__main__":
    main()