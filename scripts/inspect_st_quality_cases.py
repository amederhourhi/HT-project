
"""Visual investigation of ST-elevation quality conflicts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from src.dsp.filters import filter_ecg
from src.features.extract_features import extract_clinical_features

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/cohort"
OUTPUT = ROOT / "docs/clinical_validation/st_quality_cases"

CASES = (1918, 14639)
LEADS_TO_INSPECT = ("V1", "V2", "V3", "V4")


def main():
    manifest = pd.read_csv(
        COHORT / "cohort_manifest.csv"
    ).set_index("ecg_id")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for ecg_id in CASES:
        record = manifest.loc[ecg_id]
        raw, fields = wfdb.rdsamp(
            str(COHORT / record["filename_lr"])
        )

        fs = fields["fs"]
        names = [str(n).upper() for n in fields["sig_name"]]

        filtered = filter_ecg(
            raw,
            fs=fs,
            lowcut=0.5,
            highcut=40.0,
        )

        extracted = extract_clinical_features(
            raw,
            fs=fs,
            lead_names=fields["sig_name"],
            sqi_version="v3",
        )

        print(f"\n{'=' * 65}")
        print(f"ECG {ecg_id}")
        print("Extraction:", extracted["status"])

        if extracted["status"] != "success":
            print("Reason:", extracted["reason"])
            continue

        features = extracted["features"]
        contract = extracted["quality_contract"]

        print(
            "Baseline drift leads:",
            extracted["sqi_details"].get(
                "baseline_drift_review_leads", []
            ),
        )

        fig, axes = plt.subplots(
            len(LEADS_TO_INSPECT),
            2,
            figsize=(15, 11),
            sharex=True,
        )

        time = np.arange(len(raw)) / fs

        for i, lead in enumerate(LEADS_TO_INSPECT):
            index = names.index(lead)

            quality = contract["feature_quality"][f"st_{lead}"]
            st_value = features[f"st_{lead}"]

            print(
                f"{lead}: ST={st_value:+.3f} mV "
                f"| quality={quality['status']} "
                f"| reasons={quality['reasons']}"
            )

            axes[i, 0].plot(
                time, raw[:, index], linewidth=0.8
            )
            axes[i, 0].set_title(f"{lead} — Raw")

            axes[i, 1].plot(
                time, filtered[:, index], linewidth=0.8
            )
            axes[i, 1].set_title(
                f"{lead} — Filtered | "
                f"ST={st_value:+.3f} mV | "
                f"{quality['status']}"
            )

            for ax in axes[i]:
                ax.set_ylabel("mV")
                ax.grid(alpha=0.25)

        for ax in axes[-1]:
            ax.set_xlabel("Time (seconds)")

        fig.suptitle(
            f"CardioSense ECG {ecg_id} — ST Quality Review",
            fontsize=14,
        )

        fig.tight_layout()

        path = OUTPUT / f"ecg_{ecg_id}_st_review.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)

        print("Saved:", path)

    print("\nST quality inspection completed.")


if __name__ == "__main__":
    main()
