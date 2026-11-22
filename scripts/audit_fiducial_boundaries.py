
"""Audit QRS boundary detection at 100 Hz and 500 Hz.

Read-only diagnostic for CardioSense.
Does not modify ECG files, models, or clinical rules.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from src.dsp.filters import filter_ecg
from src.features.r_peaks import detect_r_peaks
from src.features.fiducials import (
    build_median_beat,
    find_global_qrs_bounds,
    estimate_isoelectric_baseline,
    measure_lead_st,
    _lead_energy_envelope,
    _bounds_from_envelope,
    QRS_SEARCH_BACK_MS,
    QRS_SEARCH_FWD_MS,
    QRS_ENERGY_FRACTION,
    LOUDEST_LEAD_QUALITY_SHARE,
)

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "cohort"

manifest = pd.read_csv(
    COHORT / "cohort_manifest.csv"
).set_index("ecg_id")


def audit(ecg_id, signals, fs, lead_names, peaks):
    print(f"\n{'=' * 70}")
    print(f"ECG {ecg_id} | {fs} Hz")
    print(f"Supplied R peaks: {len(peaks)}")

    beats = []
    r_index = None
    counts = []

    for column in range(signals.shape[1]):
        beat, idx, count = build_median_beat(
            signals[:, column], peaks, fs=fs
        )

        if beat is None:
            print("Median beat construction failed.")
            return

        beats.append(beat)
        counts.append(count)
        if r_index is None:
            r_index = idx

    median_beats = np.stack(beats)

    onset, offset = find_global_qrs_bounds(
        median_beats, r_index, fs=fs
    )

    if onset is None:
        print("QRS delineation failed.")
        return

    print("Beats used:", counts[0])
    print("QRS duration:", round(
        (offset - onset) * 1000 / fs, 1
    ), "ms")

    print("QRS onset relative to R:", round(
        (onset - r_index) * 1000 / fs, 1
    ), "ms")

    print("QRS offset relative to R:", round(
        (offset - r_index) * 1000 / fs, 1
    ), "ms")

    start = max(
        0,
        r_index - int(QRS_SEARCH_BACK_MS * fs / 1000)
    )
    end = min(
        median_beats.shape[1] - 1,
        r_index + int(QRS_SEARCH_FWD_MS * fs / 1000)
    )

    envelopes = [
        _lead_energy_envelope(beat, start, end, fs)
        for beat in median_beats
    ]

    loudest = max(
        float(np.max(env))
        for env, _ in envelopes
    )

    hold = max(2, envelopes[0][1])

    candidates = []

    print("\nPer-lead QRS boundaries:")
    print(
        "Lead | Start ms | End ms | "
        "Width ms | Eligible"
    )

    for name, (env, width) in zip(lead_names, envelopes):
        peak = float(np.max(env))

        eligible = (
            loudest > 0
            and peak >= LOUDEST_LEAD_QUALITY_SHARE * loudest
        )

        if not eligible:
            print(f"{name:4} | excluded (low energy)")
            continue

        local_onset, local_offset = _bounds_from_envelope(
            env,
            int(np.argmax(env)),
            QRS_ENERGY_FRACTION * peak,
            hold,
        )

        if local_offset <= local_onset:
            print(f"{name:4} | invalid boundary")
            continue

        absolute_onset = start + local_onset
        absolute_offset = start + local_offset

        onset_ms = (
            absolute_onset - r_index
        ) * 1000 / fs

        offset_ms = (
            absolute_offset - r_index
        ) * 1000 / fs

        width_ms = (
            absolute_offset - absolute_onset
        ) * 1000 / fs

        print(
            f"{name:4} | "
            f"{onset_ms:+8.1f} | "
            f"{offset_ms:+6.1f} | "
            f"{width_ms:8.1f} | yes"
        )

        candidates.append(
            (name, absolute_onset, absolute_offset)
        )

    if candidates:
        earliest = min(c[1] for c in candidates)
        latest = max(c[2] for c in candidates)

        onset_leads = [
            c[0] for c in candidates if c[1] == earliest
        ]
        offset_leads = [
            c[0] for c in candidates if c[2] == latest
        ]

        print("\nEarliest-onset leads:", onset_leads)
        print("Latest-offset leads:", offset_leads)

    print("\nEnvelope smoothing samples:", envelopes[0][1])
    print(
        "Actual smoothing duration:",
        round(envelopes[0][1] * 1000 / fs, 1),
        "ms"
    )
    print("Boundary hold samples:", hold)
    print(
        "Boundary hold duration:",
        round(hold * 1000 / fs, 1),
        "ms"
    )

    print("\nST measurement details:")

    for lead in ("V1", "V2", "V3", "V4"):
        idx = [
            name.upper() for name in lead_names
        ].index(lead)

        beat = median_beats[idx]

        baseline, baseline_idx = (
            estimate_isoelectric_baseline(
                beat, onset, fs=fs
            )
        )

        reading = measure_lead_st(
            beat, baseline, offset, fs=fs
        )

        baseline_time = (
            baseline_idx - r_index
        ) * 1000 / fs

        print(
            f"{lead}: "
            f"ST={reading['st_at_j_mv']:+.3f} mV | "
            f"baseline={baseline:+.3f} mV | "
            f"baseline position={baseline_time:+.1f} ms"
        )


def main():
    for ecg_id in (1918, 14639):
        record = manifest.loc[ecg_id]

        raw, fields = wfdb.rdsamp(
            str(COHORT / record["filename_hr"])
        )

        lead_names = fields["sig_name"]
        filtered_500 = filter_ecg(raw, fs=500)
        filtered_100 = resample_poly(
            filtered_500, 1, 5, axis=0
        )

        lead_ii = [
            name.upper() for name in lead_names
        ].index("II")

        peaks_500 = np.asarray(
            detect_r_peaks(
                filtered_500[:, lead_ii],
                fs=500
            ),
            dtype=int
        )

        peaks_100 = np.unique(
            np.rint(peaks_500 / 5).astype(int)
        )

        audit(
            ecg_id, filtered_500,
            500, lead_names, peaks_500
        )

        audit(
            ecg_id, filtered_100,
            100, lead_names, peaks_100
        )


if __name__ == "__main__":
    main()
