
"""Read-only audit of saved matched fold-9 V1/V3 Random Forest predictions.

Does not load models, retrain, score ECGs, inspect waveform files, or use
reserved fold 10 for evaluation. No file is written by this script.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sqi_ablation_uncertainty import (
    paired_decisions,
    patient_bootstrap,
)
from scripts.train_v3_rf_candidate import (
    calculate_metrics,
    sha256_file,
)
from src.features.extract_features import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = (
    ROOT / "docs/model_contracts/v1_v3_matched_fold9_predictions.csv"
)
REPORT = ROOT / "docs/model_contracts/v1_v3_matched_fold9.json"
PIN = ROOT / "data/processed/cohort_ecg_ids.csv"
V1_FEATURES = ROOT / "data/processed/clinical_features_validated.csv"
V3_FEATURES = ROOT / "data/processed/clinical_features_v3_candidate.csv"

METRICS = (
    "accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
)


def load_verified_pairs(
    predictions_path=PREDICTIONS,
    report_path=REPORT,
    pin_path=PIN,
    v1_path=V1_FEATURES,
    v3_path=V3_FEATURES,
):
    """Verify hashes, patient IDs, folds, labels, and frozen metrics."""

    paths = {
        "predictions": Path(predictions_path),
        "report": Path(report_path),
        "pin": Path(pin_path),
        "v1": Path(v1_path),
        "v3": Path(v3_path),
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    # -----------------------------------------------------
    # A. Verify frozen evaluation contract
    # -----------------------------------------------------

    report = json.loads(
        paths["report"].read_text(encoding="utf-8")
    )

    if (
        report.get("contract_version")
        != "v1_v3_matched_fold9_v1"
        or report.get("evaluation_fold") != 9
        or report.get("test_fold_evaluated") is not False
        or report.get("feature_order") != FEATURE_COLUMNS
    ):
        raise ValueError(
            "Unsupported or unsafe paired evaluation contract"
        )

    # -----------------------------------------------------
    # B. Verify stored artifact fingerprints
    # -----------------------------------------------------

    artifacts = report.get("artifacts") or {}

    for key, path in (
        ("paired_predictions_sha256", paths["predictions"]),
        ("v1_dataset_sha256", paths["v1"]),
        ("v3_dataset_sha256", paths["v3"]),
    ):
        if artifacts.get(key) != sha256_file(path):
            raise ValueError(
                f"Provenance hash mismatch: {key}"
            )

    # -----------------------------------------------------
    # C. Load existing files without modifying them
    # -----------------------------------------------------

    predictions = pd.read_csv(paths["predictions"])
    pin = pd.read_csv(paths["pin"])
    v1 = pd.read_csv(paths["v1"])
    v3 = pd.read_csv(paths["v3"])

    required_predictions = {
        "ecg_id",
        "patient_id",
        "target_label",
        "v1_probability",
        "v3_probability",
        "v1_prediction",
        "v3_prediction",
    }

    required_cohort = {
        "ecg_id",
        "strat_fold",
        "target_label",
    }

    if not required_predictions <= set(predictions.columns):
        raise ValueError("Missing paired prediction fields")

    for name, frame in (
        ("pin", pin),
        ("v1", v1),
        ("v3", v3),
    ):
        if not required_cohort <= set(frame.columns):
            raise ValueError(
                f"Missing cohort fields in {name}"
            )

    if "patient_id" not in v3.columns:
        raise ValueError(
            "Missing patient identities in V3 dataset"
        )

    if predictions.empty:
        raise ValueError("Empty prediction artifact")

    # -----------------------------------------------------
    # D. Reject missing or duplicate ECG identifiers
    # -----------------------------------------------------

    for name, frame in (
        ("predictions", predictions),
        ("pin", pin),
        ("v1", v1),
        ("v3", v3),
    ):
        if (
            frame["ecg_id"].isna().any()
            or not frame["ecg_id"].is_unique
        ):
            raise ValueError(
                f"Missing or duplicate ECG ID in {name}"
            )

    # -----------------------------------------------------
    # E. Verify that every prediction belongs to fold 9
    # -----------------------------------------------------

    ids = predictions["ecg_id"]
    aligned = predictions.copy()

    for name, frame in (
        ("pin", pin),
        ("v1", v1),
        ("v3", v3),
    ):
        selected = (
            frame.set_index("ecg_id")
            .reindex(ids)
        )

        if selected["target_label"].isna().any():
            raise ValueError(
                f"Prediction ECG absent from {name}"
            )

        if not selected["strat_fold"].eq(9).all():
            raise ValueError(
                f"Non-fold-9 ECG in {name}"
            )

        if not np.array_equal(
            aligned["target_label"].to_numpy(),
            selected["target_label"].to_numpy(),
        ):
            raise ValueError(
                f"Diagnostic labels differ in {name}"
            )

        # Check paired patient identities against V3 data.
        if name == "v3":
            recorded = pd.to_numeric(
                aligned["patient_id"],
                errors="coerce",
            )

            expected = pd.to_numeric(
                selected["patient_id"],
                errors="coerce",
            )

            if (
                recorded.isna().any()
                or expected.isna().any()
                or not np.array_equal(
                    recorded.to_numpy(),
                    expected.to_numpy(),
                )
            ):
                raise ValueError(
                    "Patient IDs differ from V3 dataset"
                )

    # -----------------------------------------------------
    # F. Check V3 patient separation
    # -----------------------------------------------------

    if (
        v3["patient_id"].isna().any()
        or not v3["strat_fold"].isin(
            range(1, 11)
        ).all()
    ):
        raise ValueError(
            "Invalid V3 patient or fold metadata"
        )

    splits = np.where(
        v3["strat_fold"].eq(10),
        "test",
        np.where(
            v3["strat_fold"].eq(9),
            "validation",
            "train",
        ),
    )

    patient_splits = pd.DataFrame({
        "patient": v3["patient_id"],
        "split": splits,
    })

    if (
        patient_splits
        .groupby("patient", dropna=False)["split"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "V3 patients overlap across partitions"
        )

    # -----------------------------------------------------
    # G. Validate the matched cohort size and classes
    # -----------------------------------------------------

    if set(aligned["target_label"]) != {0, 1}:
        raise ValueError(
            "Paired validation needs both diagnostic classes"
        )

    expected_records = (
        report.get("validation_matched") or {}
    )

    if any(
        (
            expected_records.get(name) or {}
        ).get("records") != len(aligned)
        for name in ("v1", "v3")
    ):
        raise ValueError(
            "Paired record count differs from frozen report"
        )

    # -----------------------------------------------------
    # H. Reproduce the saved evaluation metrics
    # -----------------------------------------------------

    for name in ("v1", "v3"):

        probabilities = pd.to_numeric(
            aligned[f"{name}_probability"],
            errors="coerce",
        ).to_numpy()

        saved_predictions = pd.to_numeric(
            aligned[f"{name}_prediction"],
            errors="coerce",
        ).to_numpy()

        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                f"Invalid probabilities: {name}"
            )

        if (
            not np.isfinite(saved_predictions).all()
            or not np.array_equal(
                saved_predictions,
                (probabilities >= 0.5).astype(int),
            )
        ):
            raise ValueError(
                "Predicted classes do not match "
                f"probabilities: {name}"
            )

        metrics = calculate_metrics(
            aligned["target_label"],
            probabilities,
        )

        frozen_metrics = expected_records[name]

        for metric in METRICS:
            if not np.isclose(
                metrics[metric],
                frozen_metrics.get(metric, np.nan),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Recorded {name} metric mismatch: {metric}"
                )

    return (
        aligned.sort_values("ecg_id").reset_index(drop=True),
        report,
    )


def paired_summary(pairs, *, draws=3000, seed=42):
    """Compute paired uncertainty using saved probabilities only."""

    if (
        pairs.empty
        or not pairs["patient_id"].notna().all()
    ):
        raise ValueError("Invalid paired data")

    y = pairs["target_label"].to_numpy(dtype=int)
    p1 = pairs["v1_probability"].to_numpy(dtype=float)
    p3 = pairs["v3_probability"].to_numpy(dtype=float)

    v1 = calculate_metrics(y, p1)
    v3 = calculate_metrics(y, p3)

    # Reuse the previously tested patient-bootstrap
    # implementation. Rename its input columns only;
    # these are NOT the SQI-ablation predictions.
    boot_data = pairs.rename(columns={
        "v1_probability": "v1_with_sqi",
        "v3_probability": "v3_with_sqi",
    })

    return {
        "scope": "matched_fold9_v1_vs_v3_rf_research",
        "matched_ecgs": len(pairs),
        "distinct_patients": int(
            pairs["patient_id"].nunique()
        ),
        "positives": int(y.sum()),
        "evaluation_fold": 9,
        "test_fold_evaluated": False,

        "metrics": {
            "v1": v1,
            "v3": v3,
        },

        "v3_minus_v1": {
            metric: float(
                v3[metric] - v1[metric]
            )
            for metric in METRICS
        },

        "paired_decisions": paired_decisions(
            y,
            p1,
            p3,
        ),

        "patient_bootstrap": patient_bootstrap(
            boot_data,
            draws=draws,
            seed=seed,
        ),

        "limitations": [
            "Fold 9 has been used during model development",
            "V1 and V3 training cohorts differ; "
            "differences are not causal SQI effects",
            "Small, quality-selected matched sample; "
            "intervals are descriptive",
            "No held-out fold-10 evaluation, "
            "external validation, or clinical claim",
        ],
    }


def main():
    """Print the read-only paired evaluation audit."""

    pairs, _ = load_verified_pairs()

    result = paired_summary(pairs)

    print(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        )
    )

    print(
        "READ-ONLY: no model training, "
        "inference, or file changes"
    )


if __name__ == "__main__":
    main()