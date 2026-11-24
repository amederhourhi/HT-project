"""Read-only fold-9 V1/V3 feature availability and selection audit.

Reuses the frozen paired-prediction integrity checks from Milestone 6.

No model inference, retraining, test-fold evaluation, or artifact writes.
"""

import json

import pandas as pd

from scripts.audit_paired_fold9 import (
    PIN,
    V1_FEATURES,
    V3_FEATURES,
    load_verified_pairs,
)


FOLD = 9
CATEGORIES = (
    "both",
    "v1_only",
    "v3_only",
    "neither",
)


def _indexed(name, frame):
    """Validate cohort metadata and index records by ECG ID."""

    required = {
        "ecg_id",
        "target_label",
        "strat_fold",
    }

    if (
        not isinstance(frame, pd.DataFrame)
        or not required <= set(frame.columns)
    ):
        raise ValueError(
            f"{name}: required cohort metadata missing"
        )

    if (
        frame.empty
        or frame["ecg_id"].isna().any()
        or not frame["ecg_id"].is_unique
    ):
        raise ValueError(
            f"{name}: missing or duplicate ECG identifiers"
        )

    if not frame["target_label"].isin([0, 1]).all():
        raise ValueError(
            f"{name}: invalid diagnostic labels"
        )

    if not frame["strat_fold"].isin(range(1, 11)).all():
        raise ValueError(
            f"{name}: invalid fold assignments"
        )

    return frame.set_index(
        "ecg_id",
        
    ).copy()


def summarize_validation_coverage(
    pin,
    v1,
    v3,
    pairs,
    report,
):
    """Audit fold-9 selection without predicting excluded ECGs."""

    # -----------------------------------------------------
    # A. Validate the canonical cohort and feature tables
    # -----------------------------------------------------

    cohort = _indexed("pin", pin)

    datasets = {
        "v1": _indexed("v1", v1),
        "v3": _indexed("v3", v3),
    }

    for name, data in datasets.items():

        if not data.index.isin(cohort.index).all():
            raise ValueError(
                f"{name}: ECG outside pinned cohort"
            )

        reference = cohort.loc[data.index]

        for column in (
            "target_label",
            "strat_fold",
        ):
            if not data[column].equals(
                reference[column]
            ):
                raise ValueError(
                    f"{name}: {column} disagrees with pin"
                )

    # -----------------------------------------------------
    # B. Select fold 9 only
    # -----------------------------------------------------

    validation = cohort.loc[
        cohort["strat_fold"].eq(FOLD)
    ]

    if (
        validation.empty
        or set(validation["target_label"]) != {0, 1}
    ):
        raise ValueError(
            "Fold 9 must contain both diagnostic labels"
        )

    v1_ids = (
        set(datasets["v1"].index)
        & set(validation.index)
    )

    v3_ids = (
        set(datasets["v3"].index)
        & set(validation.index)
    )

    common_ids = v1_ids & v3_ids

    # -----------------------------------------------------
    # C. Verify the paired predictions cover exactly
    #    the intersection of the two feature tables
    # -----------------------------------------------------

    required_pairs = {
        "ecg_id",
        "target_label",
        "v1_prediction",
        "v3_prediction",
    }

    if (
        not isinstance(pairs, pd.DataFrame)
        or not required_pairs <= set(pairs.columns)
        or pairs["ecg_id"].isna().any()
        or not pairs["ecg_id"].is_unique
        or set(pairs["ecg_id"]) != common_ids
    ):
        raise ValueError(
            "Paired prediction IDs differ "
            "from fold-9 intersection"
        )

    aligned = pairs.set_index(
        "ecg_id",
        
    )

    if not aligned["target_label"].equals(
        validation.loc[
            aligned.index,
            "target_label",
        ]
    ):
        raise ValueError(
            "Paired labels disagree with the frozen cohort"
        )

    for column in (
        "v1_prediction",
        "v3_prediction",
    ):
        if not aligned[column].isin([0, 1]).all():
            raise ValueError(
                f"Invalid saved predicted class: {column}"
            )

    # -----------------------------------------------------
    # D. Preserve fold-10 reservation
    # -----------------------------------------------------

    if (
        report.get("evaluation_fold") != FOLD
        or report.get("test_fold_evaluated") is not False
    ):
        raise ValueError(
            "Fold-10 reservation is not established"
        )

    # -----------------------------------------------------
    # E. Verify historical coverage counts
    # -----------------------------------------------------

    frozen_rows = report.get(
        "coverage_by_label"
    )

    if (
        not isinstance(frozen_rows, list)
        or len(frozen_rows) != 2
    ):
        raise ValueError(
            "Frozen per-label coverage report missing"
        )

    if {
        row.get("target_label")
        for row in frozen_rows
    } != {0, 1}:
        raise ValueError(
            "Frozen coverage labels invalid"
        )

    frozen = {
        row["target_label"]: row
        for row in frozen_rows
    }

    by_label = []

    overall = {
        name: 0
        for name in CATEGORIES
    }

    # -----------------------------------------------------
    # F. Count selected and excluded ECGs by label
    # -----------------------------------------------------

    for label in (0, 1):

        group = set(
            validation.index[
                validation["target_label"].eq(label)
            ]
        )

        n = len(group)

        counts = {
            "both": len(
                group & common_ids
            ),
            "v1_only": len(
                (group & v1_ids) - v3_ids
            ),
            "v3_only": len(
                (group & v3_ids) - v1_ids
            ),
            "neither": len(
                group - (v1_ids | v3_ids)
            ),
        }

        if sum(counts.values()) != n:
            raise ValueError(
                "Coverage partition does not sum to cohort"
            )

        for name in CATEGORIES:
            overall[name] += counts[name]

        v1_count = (
            counts["both"] + counts["v1_only"]
        )

        v3_count = (
            counts["both"] + counts["v3_only"]
        )

        expectation = frozen[label]

        expected_counts = {
            "manifest_records": n,
            "v1_accepted": v1_count,
            "v3_eligible": v3_count,
            "paired_records": counts["both"],
        }

        for name, value in expected_counts.items():

            if expectation.get(name) != value:
                raise ValueError(
                    "Frozen coverage mismatch: "
                    f"label {label}, {name}"
                )

        expected_rates = (
            ("v1_coverage", v1_count / n),
            ("v3_coverage", v3_count / n),
        )

        for name, value in expected_rates:

            recorded = expectation.get(name)

            if (
                not isinstance(recorded, (int, float))
                or abs(recorded - value) > 1e-12
            ):
                raise ValueError(
                    "Frozen coverage mismatch: "
                    f"label {label}, {name}"
                )

        by_label.append({
            "target_label": label,
            "manifest_records": n,
            **counts,
            "v1_feature_rows": v1_count,
            "v3_eligible_rows": v3_count,
            "v1_coverage": v1_count / n,
            "v3_coverage": v3_count / n,
            "paired_coverage": counts["both"] / n,
        })

    # -----------------------------------------------------
    # G. Confirm total coverage
    # -----------------------------------------------------

    total = len(validation)

    if (
        sum(overall.values()) != total
        or len(common_ids) != overall["both"]
    ):
        raise ValueError(
            "Overall coverage accounting mismatch"
        )

    # -----------------------------------------------------
    # H. Count correct predictions on matched ECGs only
    # -----------------------------------------------------

    v1_correct = int(
        (
            aligned["v1_prediction"]
            == aligned["target_label"]
        ).sum()
    )

    v3_correct = int(
        (
            aligned["v3_prediction"]
            == aligned["target_label"]
        ).sum()
    )

    paired_count = len(aligned)

    # Do NOT assign any predicted class to an ECG
    # absent from the saved matched predictions.

    return {
        "audit": "fold9_feature_availability_v1",
        "validation_fold": FOLD,
        "manifest_records": total,
        "test_fold_evaluated": False,

        "feature_availability": {
            "v1_feature_rows": len(v1_ids),
            "v1_without_feature_row": (
                total - len(v1_ids)
            ),
            "v1_coverage": len(v1_ids) / total,

            "v3_eligible_rows": len(v3_ids),
            "v3_not_eligible": (
                total - len(v3_ids)
            ),
            "v3_coverage": len(v3_ids) / total,
        },

        "availability_overlap": {
            **overall,
            "paired_fraction_of_fold": (
                paired_count / total
            ),
        },

        "by_label": by_label,

        "matched_predictions_only": {
            "records": paired_count,
            "v1_correct": v1_correct,
            "v1_incorrect": (
                paired_count - v1_correct
            ),
            "v3_correct": v3_correct,
            "v3_incorrect": (
                paired_count - v3_correct
            ),
            "v1_accuracy": (
                v1_correct / paired_count
            ),
            "v3_accuracy": (
                v3_correct / paired_count
            ),
        },

        "limitations": [
            (
                "Feature-row absence is not equivalent "
                "to a negative clinical finding."
            ),
            (
                "V1 absence may reflect extraction failure; "
                "V3 absence reflects quality eligibility filtering."
            ),
            (
                "Predictions for unpaired records are not "
                "available in the saved matched CSV."
            ),
            (
                "Matched-subset accuracy is not full-fold "
                "accuracy or diagnostic sensitivity."
            ),
            (
                "Small, selected development fold; no "
                "external or clinical validation."
            ),
        ],
    }


def main():
    """Print the verified coverage report without writing files."""

    # Milestone 6 verifies the prediction artifact hashes,
    # labels, patients, fold assignments and frozen metrics.
    pairs, report = load_verified_pairs()

    result = summarize_validation_coverage(
        pd.read_csv(PIN),
        pd.read_csv(V1_FEATURES),
        pd.read_csv(V3_FEATURES),
        pairs,
        report,
    )

    print(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        )
    )

    print(
        "READ-ONLY: saved metadata and predictions only; "
        "no fold-10 evaluation"
    )


if __name__ == "__main__":
    main()