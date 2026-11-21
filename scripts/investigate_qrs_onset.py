
"""CardioSense: QRS-onset failure investigation.

Inspects LUDB development/previously evaluated records only.
Does not train, tune, or modify any detector.

Outputs:
  docs/clinical_validation/onset_investigation/
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wfdb

from src.dsp.filters import filter_ecg
from src.features import fiducials as original
from src.features import fiducials_qrs_v4 as v4
from scripts.benchmark_ludb_landmarks import qrs_triplets


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ludb_reference/data"
OUT = (
    ROOT / "docs/clinical_validation/"
    "onset_investigation"
)

# Regression cases + two comparison records.
RECORDS = [72, 85, 31, 45, 59, 20, 23]

FS = 500
PRE_MS = 150
POST_MS = 200


def get_annotations(path, extension):
    """Return expert-annotated onset, R peak, offset triplets."""
    ann = wfdb.rdann(str(path), extension)
    return qrs_triplets(ann)


def collect_reference(triplets, peaks):
    """Expert lead-II landmarks matched to supplied R peaks."""
    lookup = {int(r): (int(on), int(off))
              for on, r, off in triplets}

    matched = []
    for peak in peaks:
        peak = int(peak)
        if peak in lookup:
            onset, offset = lookup[peak]
            matched.append((onset, peak, offset))

    return matched


def plot_median_leads(
    record_id,
    filtered,
    lead_names,
    peaks,
    current_result,
    v4_result,
):
    """Plot all 12 median waveforms and algorithm boundaries."""
    pre = round(original.BEAT_PRE_MS * FS / 1000)

    fig, axes = plt.subplots(
        4, 3, figsize=(15, 12),
        sharex=True
    )

    for i, (ax, lead) in enumerate(
        zip(axes.flat, lead_names)
    ):
        beat, r_idx, count = original.build_median_beat(
            filtered[:, i],
            peaks,
            fs=FS,
        )

        if beat is None:
            ax.set_title(f"{lead}: no median beat")
            continue

        x = (
            np.arange(len(beat)) - r_idx
        ) * 1000 / FS

        ax.plot(
            x, beat,
            linewidth=1.3,
            color="black",
        )

        for result, color, label in [
            (current_result, "blue", "Original onset"),
            (v4_result, "red", "V4 onset"),
        ]:
            onset_ms = (
                result["qrs_onset_idx"] - pre
            ) * 1000 / FS

            ax.axvline(
                onset_ms,
                color=color,
                linestyle="--",
                linewidth=1.3,
                label=label,
            )

        offset_ms = (
            v4_result["qrs_offset_idx"] - pre
        ) * 1000 / FS

        ax.axvline(
            offset_ms,
            color="green",
            linestyle=":",
            linewidth=1.2,
            label="V4 J-point",
        )

        ax.axvline(
            0, color="gray",
            linewidth=0.8,
            alpha=0.6
        )

        ax.set_title(f"{lead} | {count} beats")
        ax.set_xlim(-110, 160)
        ax.grid(alpha=0.2)

    axes.flat[0].legend(fontsize=8)

    fig.suptitle(
        f"LUDB {record_id}: Median QRS waveforms\n"
        "Original vs V4 global onset",
        fontsize=14
    )

    fig.tight_layout()
    fig.savefig(
        OUT / f"ludb_{record_id}_12lead.png",
        dpi=160,
    )
    plt.close(fig)


def plot_expert_beats(
    record_id,
    signal,
    lead_names,
    matched,
    current_result,
    v4_result,
):
    """Show beat-level expert annotations and median-beat estimates.

    Algorithm lines are representative global offsets relative to R,
    NOT individual beat predictions.
    """
    index = [
        n.lower() for n in lead_names
    ].index("ii")

    waveform = signal[:, index]

    pre_idx = int(PRE_MS * FS / 1000)
    post_idx = int(POST_MS * FS / 1000)

    valid = [
        (on, r, off)
        for on, r, off in matched
        if r - pre_idx >= 0
        and r + post_idx < len(waveform)
    ]

    if not valid:
        return

    # Spread examples across recording, rather than
    # selecting only visually unusual beats.
    indices = np.unique(
        np.linspace(
            0, len(valid) - 1,
            min(12, len(valid)),
            dtype=int,
        )
    )

    fig, axes = plt.subplots(
        4, 3, figsize=(15, 12),
        sharex=True
    )

    old_on = (
        (current_result["qrs_onset_idx"] - round(original.BEAT_PRE_MS * FS / 1000)) * 1000 / FS
    )
    new_on = (
        v4_result["qrs_onset_ms_relative_to_r"]
    )

    for ax, j in zip(axes.flat, indices):
        expert_on, peak, expert_off = valid[j]

        segment = waveform[
            peak - pre_idx:peak + post_idx
        ]

        x = (
            np.arange(len(segment)) - pre_idx
        ) * 1000 / FS

        ax.plot(
            x, segment,
            color="black",
            linewidth=1.2,
        )

        ax.axvline(
            (expert_on - peak) * 1000 / FS,
            color="green",
            linewidth=2,
            label="Expert lead-II onset",
        )

        ax.axvline(
            (expert_off - peak) * 1000 / FS,
            color="green",
            linestyle=":",
            linewidth=1,
        )

        ax.axvline(
            old_on,
            color="blue",
            linestyle="--",
            label="Original global median onset",
        )

        ax.axvline(
            new_on,
            color="red",
            linestyle="--",
            label="V4 global median onset",
        )

        ax.set_title(f"Beat {j + 1}")
        ax.grid(alpha=0.2)

    for ax in list(axes.flat)[len(indices):]:
        ax.axis("off")

    axes.flat[0].legend(fontsize=7)

    fig.suptitle(
        f"LUDB {record_id}: Expert-annotated lead-II beats\n"
        "Global algorithm boundaries are shown only "
        "as representative positions",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(
        OUT / f"ludb_{record_id}_expert_beats.png",
        dpi=160,
    )
    plt.close(fig)


def evaluate_record(rid):
    path = DATA / str(rid)

    signal, fields = wfdb.rdsamp(str(path))

    if fields["fs"] != FS:
        raise ValueError("Expected 500 Hz")

    lead_names = fields["sig_name"]

    annotations = get_annotations(path, "ii")
    peaks = np.array(
        [int(r) for _, r, _ in annotations],
        dtype=int,
    )

    filtered = filter_ecg(signal, fs=FS)

    current_result = original.analyze_12lead_features(
        filtered, peaks,
        lead_names=lead_names, fs=FS
    )

    v4_result = v4.analyze_12lead_features(
        filtered, peaks,
        lead_names=lead_names, fs=FS
    )

    if (
        current_result["status"] != "success"
        or v4_result["status"] != "success"
    ):
        raise RuntimeError("Delineation unsuccessful")

    matched = collect_reference(
        annotations, peaks
    )

    expert_relative = np.array([
        (on - r) * 1000 / FS
        for on, r, _ in matched
    ])

    if len(expert_relative) == 0:
        raise RuntimeError("No expert annotations")

    plot_median_leads(
        rid, filtered, lead_names, peaks,
        current_result, v4_result
    )

    plot_expert_beats(
        rid, signal, lead_names, matched,
        current_result, v4_result
    )

    old_on = (current_result["qrs_onset_idx"] - round(original.BEAT_PRE_MS * FS / 1000)) * 1000 / FS
    new_on = v4_result[
        "qrs_onset_ms_relative_to_r"
    ]

    return {
        "record_id": rid,
        "expert_lead_ii_median_onset_ms": float(
            np.median(expert_relative)
        ),
        "expert_lead_ii_onset_iqr_ms": float(
            np.percentile(expert_relative, 75)
            - np.percentile(expert_relative, 25)
        ),
        "original_global_onset_ms": old_on,
        "v4_global_onset_ms": new_on,
        "v4_minus_original_ms": new_on - old_on,
        "expert_beats": len(matched),
        "isolated_offset_lead": v4_result.get(
            "isolated_energy_offset_lead"
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    for rid in RECORDS:
        try:
            row = evaluate_record(rid)
            rows.append(row)

            print(
                f"LUDB {rid}: completed | "
                f"onset shift "
                f"{row['v4_minus_original_ms']:+.1f} ms"
            )

        except Exception as exc:
            print(
                f"LUDB {rid}: FAILED "
                f"{type(exc).__name__}: {exc}"
            )

    if rows:
        output = OUT / "onset_investigation.csv"

        with output.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

        print("\nInvestigation saved to:", OUT)
        print("Records completed:", len(rows))


if __name__ == "__main__":
    main()
