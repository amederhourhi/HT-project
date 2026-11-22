
"""CardioSense external QRS-landmark evaluation on LUDB.

Compares CardioSense QRS onset and offset against expert
annotations. Uses annotated lead-II QRS peaks to isolate
fiducial delineation errors.

Does not validate ST voltage, STEMI decisions or live devices.

Run:
    python -m scripts.benchmark_ludb_landmarks --count 30
"""

import argparse
import csv
import statistics
import shutil

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import wfdb

from src.dsp.filters import filter_ecg
from src.features import fiducials as current
from src.features import fiducials_common_grid as candidate


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "ludb_reference"

REPORT = ROOT / "docs" / "clinical_validation"

BASE = "https://physionet.org/files/ludb/1.0.1/data/"

LEADS = (
    "i", "ii", "iii", "avr", "avl", "avf",
    "v1", "v2", "v3", "v4", "v5", "v6"
)

FS = 500


def download_record(record_id):
    """Download the waveform and all lead annotations."""

    folder = DATA / "data"
    folder.mkdir(parents=True, exist_ok=True)

    for extension in ("hea", "dat", *LEADS):

        target = folder / f"{record_id}.{extension}"

        if target.exists() and target.stat().st_size > 0:
            continue

        url = BASE + f"{record_id}.{extension}"

        temporary = target.with_suffix(
            target.suffix + ".part"
        )

        try:
            with urlopen(url, timeout=90) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)

            temporary.replace(target)

        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    return folder / str(record_id)


def qrs_triplets(ann):
    """Extract annotated QRS onset, peak and offset.

    Expected annotation sequence:
        ( N )

    P-wave and T-wave annotations are ignored.
    """

    symbols = list(ann.symbol)
    samples = list(map(int, ann.sample))

    output = []

    for i in range(1, len(symbols) - 1):

        if (
            symbols[i] == "N"
            and symbols[i - 1] == "("
            and symbols[i + 1] == ")"
        ):

            onset, peak, offset = samples[i - 1:i + 2]

            if onset < peak < offset:
                output.append((onset, peak, offset))

    return output


def expert_global_beat_references(
    record_base,
    lead_ii_peaks,
    tolerance_samples=40,
):
    """Calculate global QRS boundaries from expert annotations.

    Each lead's annotated QRS peak is matched to the closest
    lead-II reference peak within the tolerance.

    Global onset = earliest matching lead onset.
    Global offset = latest matching lead offset.

    At least eight annotated leads must match.
    """

    waves = {
        lead: qrs_triplets(
            wfdb.rdann(str(record_base), lead)
        )
        for lead in LEADS
    }

    references = []

    for anchor in lead_ii_peaks:

        matches = []

        for lead, triplets in waves.items():

            near = [
                triplet
                for triplet in triplets
                if abs(triplet[1] - anchor)
                <= tolerance_samples
            ]

            if near:
                matches.append(
                    min(
                        near,
                        key=lambda t: abs(t[1] - anchor)
                    )
                )

        if len(matches) >= 8:

            references.append({
                "anchor": int(anchor),
                "onset": min(t[0] for t in matches),
                "offset": max(t[2] for t in matches),
                "n_leads": len(matches),
            })

    return references


def audit_record(record_id, base):
    """Compare both algorithms against annotated QRS boundaries."""

    ecg, fields = wfdb.rdsamp(str(base))

    if fields["fs"] != FS or ecg.shape[1] != 12:
        raise ValueError(
            f"Unexpected waveform format: "
            f"{fields['fs']} Hz {ecg.shape}"
        )

    names = [
        str(name).lower()
        for name in fields["sig_name"]
    ]

    if set(names) != set(LEADS):
        raise ValueError(
            f"Unexpected leads: {names}"
        )

    annotations = wfdb.rdann(str(base), "ii")

    peaks = np.asarray(
        [
            t[1]
            for t in qrs_triplets(annotations)
        ],
        dtype=int,
    )

    if len(peaks) < 2:
        raise ValueError(
            "Fewer than two annotated lead-II QRS complexes"
        )

    references = expert_global_beat_references(
        base,
        peaks,
    )

    if len(references) < 3:
        raise ValueError(
            "Too few cross-lead annotated reference beats"
        )

    pre = int(0.250 * FS)
    post = int(0.450 * FS)

    median_rr = np.median(np.diff(peaks))

    valid_peaks = []

    for i, peak in enumerate(peaks):

        if peak - pre < 0:
            continue

        if peak + post >= len(ecg):
            continue

        if i > 0:

            rr = peaks[i] - peaks[i - 1]

            if (
                abs(rr - median_rr) / median_rr
                > current.ECTOPIC_RR_TOLERANCE
            ):
                continue

        valid_peaks.append(int(peak))

    # Match the expert events to the median-beat input.
    reference_by_peak = {
        item["anchor"]: item
        for item in references
    }

    selected = [
        reference_by_peak[p]
        for p in valid_peaks
        if p in reference_by_peak
    ]

    if len(selected) < 3:
        raise ValueError(
            "Too few usable annotated beats "
            "after edge and RR filtering"
        )

    # Estimate expert boundaries relative to the R peak.
    expert_onset = float(
        np.median([
            (item["onset"] - item["anchor"]) * 2
            for item in selected
        ])
    )

    expert_offset = float(
        np.median([
            (item["offset"] - item["anchor"]) * 2
            for item in selected
        ])
    )

    filtered = filter_ecg(ecg, fs=FS)

    output = []

    methods = [
        ("current", current.analyze_12lead_features),
        ("common_grid", candidate.analyze_12lead_features),
    ]

    for name, method in methods:

        result = method(
            filtered,
            peaks,
            lead_names=fields["sig_name"],
            fs=FS,
        )

        if result["status"] != "success":
            raise ValueError(
                f"{name} rejected record: "
                f"{result.get('reason')}"
            )

        # Both implementations use 500 Hz indices here.
        measured_onset = (
            result["qrs_onset_idx"] - pre
        ) * 2.0

        measured_offset = (
            result["qrs_offset_idx"] - pre
        ) * 2.0

        expert_duration = expert_offset - expert_onset

        output.append({
            "record_id": record_id,
            "method": name,

            "reference_beats": len(selected),
            "detected_beats_used": result["beats_used"],

            "expert_onset_ms": expert_onset,
            "expert_offset_ms": expert_offset,

            "measured_onset_ms": measured_onset,
            "measured_offset_ms": measured_offset,

            "onset_error_ms":
                measured_onset - expert_onset,

            "offset_error_ms":
                measured_offset - expert_offset,

            "abs_onset_error_ms":
                abs(measured_onset - expert_onset),

            "abs_offset_error_ms":
                abs(measured_offset - expert_offset),

            "expert_qrs_ms": expert_duration,
            "measured_qrs_ms": result["qrs_duration_ms"],

            "abs_qrs_error_ms":
                abs(
                    result["qrs_duration_ms"]
                    - expert_duration
                ),
        })

    return output


def p95(values):
    return float(np.percentile(values, 95))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of development records, maximum 30",
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not 1 <= args.count <= 30:
        parser.error(
            "Development limited to records 1-30. "
            "Reserve records 31-200."
        )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    failures = []

    for record_id in range(1, args.count + 1):

        try:

            base = download_record(record_id)

            if not args.download_only:
                rows.extend(
                    audit_record(record_id, base)
                )

            action = (
                "downloaded"
                if args.download_only
                else "evaluated"
            )

            print(f"{record_id:03d}: {action}")

        except Exception as exc:

            failures.append({
                "record_id": record_id,
                "error":
                    f"{type(exc).__name__}: {exc}",
            })

            print(
                f"{record_id:03d}: FAILED: {exc}"
            )

    if args.download_only:
        print(
            "Download finished. "
            "Benchmark was not executed."
        )
        return

    output_file = (
        REPORT / "ludb_expert_landmark_development.csv"
    )

    if rows:
        with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(rows)

    failure_file = (
        REPORT / "ludb_expert_landmark_failures.csv"
    )

    with failure_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "error"],
        )

        writer.writeheader()
        writer.writerows(failures)

    print(
        f"\nAnalyzed {len(rows)//2}/{args.count}; "
        f"failures: {len(failures)}"
    )

    print("Report:", output_file)

    if not rows:
        print("No valid benchmark results.")
        return

    for name in ("current", "common_grid"):

        subset = [
            item
            for item in rows
            if item["method"] == name
        ]

        print(f"\n{name}: {len(subset)} records")

        for field in (
            "abs_onset_error_ms",
            "abs_offset_error_ms",
            "abs_qrs_error_ms",
        ):

            values = [
                item[field]
                for item in subset
            ]

            print(
                f"  {field}: "
                f"median={statistics.median(values):.1f} ms, "
                f"p95={p95(values):.1f} ms"
            )

    print(
        "\nBoth algorithms use the same delineation "
        "at native 500 Hz."
    )

    print(
        "This benchmark evaluates landmark accuracy "
        "against external annotations."
    )

    print(
        "Keep LUDB records 31-200 untouched "
        "for later evaluation."
    )


if __name__ == "__main__":
    main()
