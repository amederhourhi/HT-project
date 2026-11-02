
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import medfilt

from src.dsp.filters import filter_ecg

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data/cohort"
OUTPUT = ROOT / "docs/sqi_audit/baseline_regressions"

CASES = [
    (4072, "V5"),
    (5436, "V6"),
    (14316, "V4"),
]


def main():
    manifest = pd.read_csv(
        RECORDS / "cohort_manifest.csv"
    ).set_index("ecg_id")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for ecg_id, lead in CASES:
        filename = manifest.loc[ecg_id, "filename_lr"]
        raw, fields = wfdb.rdsamp(
            str(RECORDS / filename)
        )

        fs = fields["fs"]
        names = [
            name.upper() for name in fields["sig_name"]
        ]
        index = names.index(lead)

        filtered = filter_ecg(
            raw, fs=fs, lowcut=0.5, highcut=40.0
        )

        raw_lead = raw[:, index]
        filtered_lead = filtered[:, index]

        kernel = max(3, int(round(0.8 * fs)) | 1)
        baseline = medfilt(
            raw_lead, kernel_size=kernel
        )

        time = np.arange(len(raw_lead)) / fs

        print(f"\nECG {ecg_id} — {lead}")
        print(
            "Raw range (mV):",
            round(float(np.ptp(raw_lead)), 4)
        )
        print(
            "Baseline P95-P5 (mV):",
            round(float(
                np.percentile(baseline, 95)
                - np.percentile(baseline, 5)
            ), 4)
        )
        print(
            "Filtered range (mV):",
            round(float(np.ptp(filtered_lead)), 4)
        )

        fig, axes = plt.subplots(
            3, 1,
            figsize=(13, 8),
            sharex=True
        )

        axes[0].plot(
            time, raw_lead, linewidth=0.7
        )
        axes[0].plot(
            time, baseline,
            linewidth=2,
            label="Estimated baseline"
        )
        axes[0].set_title(
            f"ECG {ecg_id} — {lead}: Raw ECG"
        )
        axes[0].legend()

        axes[1].plot(
            time, baseline,
            linewidth=1.5
        )
        axes[1].set_title(
            "0.8-second median baseline trend"
        )

        axes[2].plot(
            time, filtered_lead,
            linewidth=0.7
        )
        axes[2].set_title(
            "Filtered ECG (0.5–40 Hz)"
        )
        axes[2].set_xlabel("Time (seconds)")

        for ax in axes:
            ax.grid(alpha=0.25)
            ax.set_ylabel("mV")

        plt.tight_layout()

        filename = OUTPUT / (
            f"ecg_{ecg_id}_{lead}_baseline.png"
        )
        plt.savefig(filename, dpi=160)
        plt.close(fig)

        print("Saved:", filename)

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
