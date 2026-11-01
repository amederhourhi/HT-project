"""Read-only investigation of CardioSense's 46 SQI-rejected ECG records."""
from collections import Counter, defaultdict
import numpy as np
import wfdb

from diagnose_feature_omissions import MISSING_IDS, locate_record
from src.dsp.filters import filter_ecg
from src.dsp.sqi import assess_ecg_quality, check_flatline, check_clipping, compute_lead_sqi


def main():
    causes = Counter()
    failed_leads = Counter()
    score_histogram = Counter()
    gaps = []
    print("ECG ID | avg SQI | passed | failed lead details")
    print("-" * 78)

    for ecg_id in MISSING_IDS:
        path = locate_record(ecg_id)
        raw, metadata = wfdb.rdsamp(str(path))
        cleaned = filter_ecg(raw, fs=metadata["fs"], lowcut=0.5, highcut=40.0)
        names = metadata["sig_name"]
        quality = assess_ecg_quality(cleaned, lead_names=names, accept_threshold=0.60)
        failures = []

        for index, lead in enumerate(names):
            filtered = cleaned[:, index]
            raw_lead = raw[:, index]
            score = compute_lead_sqi(filtered)
            if score >= 0.60:
                continue

            flat_filtered = check_flatline(filtered)
            flat_raw = check_flatline(raw_lead)
            clip_filtered = check_clipping(filtered)
            clip_raw = check_clipping(raw_lead)
            if flat_filtered:
                cause = "flatline_filter"
            elif clip_filtered:
                cause = "clipping_filter"
            else:
                cause = "morphology_or_amplitude_score"
            causes[cause] += 1
            failed_leads[lead] += 1
            score_histogram[f"{score:.1f}"] += 1
            if (flat_filtered != flat_raw) or (clip_filtered != clip_raw):
                gaps.append((ecg_id, lead, flat_raw, flat_filtered, clip_raw, clip_filtered))
            failures.append(f"{lead}:{score:.3f}({cause})")

        print(f"{ecg_id:5d} | {quality['overall_sqi']:.3f} | {quality['passed_leads']:>5} | {', '.join(failures)}")

    print("\n=== FAILURE COUNTS BY CATEGORY (lead instances) ===")
    for key, n in causes.most_common():
        print(f"{key}: {n}")
    print("\n=== MOST FREQUENTLY FAILED LEADS ===")
    for lead, n in failed_leads.most_common():
        print(f"{lead}: {n}/46")
    print("\n=== RAW VS FILTERED CHECK DISAGREEMENTS ===")
    print(f"Total lead instances: {len(gaps)}")
    for row in gaps[:15]:
        print("ecg_id, lead, raw_flat, filtered_flat, raw_clip, filtered_clip:", row)
    print("\nNo data or checkpoints were modified.")


if __name__ == "__main__":
    main()
