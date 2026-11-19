
"""Audit patient-level uncertainty in the matched SQI RF ablation.

Reads saved fold-9 predictions. Does not train models or inspect fold 10.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "docs/model_contracts/sqi_rf_ablation_fold9_predictions.csv"
ABLATION = ROOT / "docs/model_contracts/sqi_rf_ablation_fold9.json"
MANIFEST = ROOT / "data/cohort/cohort_manifest.csv"
OUTPUT = ROOT / "docs/model_contracts/sqi_ablation_uncertainty_fold9.json"

VARIANTS = ("v1_with_sqi", "v3_with_sqi", "shared_without_sqi")
THRESHOLD = 0.5


def sha256(path):
    """Hash an existing file without modifying it."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_verified_inputs(predictions_path, report_path, manifest_path):
    """Validate saved predictions against the original ablation."""

    predictions_path = Path(predictions_path)
    report_path = Path(report_path)
    manifest_path = Path(manifest_path)

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    if report.get("experiment") != "sqi_rf_ablation_v1":
        raise ValueError("Unexpected ablation report")

    if report.get("test_fold_evaluated") is not False:
        raise ValueError("Test-fold reservation is not confirmed")

    if report.get("predictions_sha256") != sha256(predictions_path):
        raise ValueError("Prediction file hash mismatch")

    if report.get("input_sha256", {}).get("cohort_manifest") != sha256(
        manifest_path
    ):
        raise ValueError("Manifest hash mismatch")

    predictions = pd.read_csv(predictions_path)
    manifest = pd.read_csv(
        manifest_path,
        usecols=[
            "ecg_id", "patient_id",
            "target_label", "strat_fold",
        ],
    )

    expected = {"ecg_id", "target_label", *VARIANTS}

    if not expected.issubset(predictions.columns):
        raise ValueError("Missing prediction fields")

    if predictions.empty or predictions["ecg_id"].isna().any():
        raise ValueError("Empty predictions or missing ECG IDs")

    if (
        not predictions["ecg_id"].is_unique
        or not manifest["ecg_id"].is_unique
    ):
        raise ValueError("Duplicate ECG IDs")

    if set(predictions["ecg_id"]) != set(
        report.get("validation_ecg_ids", [])
    ):
        raise ValueError("Prediction ECG IDs differ from ablation report")

    data = predictions.merge(
        manifest,
        on="ecg_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_manifest"),
    )

    if data["patient_id"].isna().any():
        raise ValueError("Missing patient IDs")

    if not data["strat_fold"].eq(9).all():
        raise ValueError("Predictions contain non-validation ECGs")

    if not data["target_label"].eq(
        data["target_label_manifest"]
    ).all():
        raise ValueError("Prediction labels do not match the manifest")

    if set(data["target_label"]) != {0, 1}:
        raise ValueError("Validation set must contain both labels")

    for name in VARIANTS:
        probabilities = data[name].to_numpy(dtype=float)

        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite predictions: {name}")

        if (
            (probabilities < 0)
            | (probabilities > 1)
        ).any():
            raise ValueError(f"Invalid probabilities: {name}")

        # Check predictions against the original reported AUC.
        expected_auc = report["validation_metrics"][name]["roc_auc"]

        actual_auc = roc_auc_score(
            data["target_label"],
            probabilities,
        )

        if not np.isclose(
            actual_auc,
            expected_auc,
            atol=1e-12,
        ):
            raise ValueError(f"AUC does not match report: {name}")

    return (
        data.sort_values("ecg_id").reset_index(drop=True),
        report,
    )


def paired_decisions(labels, p_v1, p_v3):
    """Count changes in binary classification correctness."""

    y = np.asarray(labels, dtype=int)
    a = np.asarray(p_v1, dtype=float)
    b = np.asarray(p_v3, dtype=float)

    if (
        y.ndim != 1
        or y.size == 0
        or y.shape != a.shape
        or y.shape != b.shape
    ):
        raise ValueError("Invalid paired prediction arrays")

    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("Labels must be binary")

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Probabilities must be finite")

    if (
        ((a < 0) | (a > 1)).any()
        or ((b < 0) | (b > 1)).any()
    ):
        raise ValueError(
            "Probabilities must be between zero and one"
        )

    correct_v1 = (a >= THRESHOLD) == y
    correct_v3 = (b >= THRESHOLD) == y

    gains = int((~correct_v1 & correct_v3).sum())
    losses = int((correct_v1 & ~correct_v3).sum())
    discordant = gains + losses

    return {
        "v3_correct_v1_wrong": gains,
        "v1_correct_v3_wrong": losses,
        "both_correct": int(
            (correct_v1 & correct_v3).sum()
        ),
        "both_wrong": int(
            (~correct_v1 & ~correct_v3).sum()
        ),
        "changed_predicted_classes": int(
            (
                (a >= THRESHOLD)
                != (b >= THRESHOLD)
            ).sum()
        ),
        "mcnemar_exact_two_sided_p": (
            float(
                binomtest(
                    gains,
                    discordant,
                    0.5,
                ).pvalue
            )
            if discordant
            else 1.0
        ),
    }


def patient_bootstrap(data, *, draws=3000, seed=42):
    """Resample patients rather than treating ECGs as independent."""

    if (
        draws < 1
        or data.empty
        or data["patient_id"].isna().any()
    ):
        raise ValueError(
            "Missing patients or invalid bootstrap settings"
        )

    groups = [
        group.index.to_numpy(dtype=int)
        for _, group in (
            data.reset_index(drop=True)
            .groupby("patient_id", sort=True)
        )
    ]

    y = data["target_label"].to_numpy(dtype=int)
    p1 = data["v1_with_sqi"].to_numpy(dtype=float)
    p3 = data["v3_with_sqi"].to_numpy(dtype=float)

    generator = np.random.default_rng(seed)

    delta_accuracy = []
    delta_auc = []

    for _ in range(draws):
        choices = generator.integers(
            0,
            len(groups),
            size=len(groups),
        )

        indices = np.concatenate([
            groups[int(i)] for i in choices
        ])

        labels = y[indices]
        first = p1[indices]
        second = p3[indices]

        delta_accuracy.append(float(
            np.mean(
                (second >= THRESHOLD) == labels
            )
            - np.mean(
                (first >= THRESHOLD) == labels
            )
        ))

        # ROC-AUC requires both diagnostic classes.
        if len(np.unique(labels)) == 2:
            delta_auc.append(float(
                roc_auc_score(labels, second)
                - roc_auc_score(labels, first)
            ))

    def interval(values):
        """Calculate a percentile bootstrap interval."""
        return [
            float(value)
            for value in np.quantile(
                values,
                [0.025, 0.975],
            )
        ]

    return {
        "sampling_unit": "patient_id",
        "seed": seed,
        "draws": draws,
        "auc_draws_with_both_classes": len(delta_auc),
        "v3_minus_v1_accuracy_95pct_interval": interval(
            delta_accuracy
        ),
        "v3_minus_v1_roc_auc_95pct_interval": (
            interval(delta_auc)
            if delta_auc
            else None
        ),
    }


def main():
    """Write the uncertainty audit without retraining."""

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}"
        )

    data, _ = load_verified_inputs(
        PREDICTIONS,
        ABLATION,
        MANIFEST,
    )

    y = data["target_label"].to_numpy(dtype=int)
    first = data["v1_with_sqi"].to_numpy(dtype=float)
    second = data["v3_with_sqi"].to_numpy(dtype=float)

    first_ok = (first >= THRESHOLD) == y
    second_ok = (second >= THRESHOLD) == y

    result = {
        "audit_version": "sqi_ablation_uncertainty_v1",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "validation_fold": 9,
        "records": len(data),
        "patients": int(data["patient_id"].nunique()),
        "positive_records": int(y.sum()),
        "test_evaluated": False,
        "comparison": (
            "V3 + SQI versus V1 + SQI, same matched ECGs"
        ),
        "observed_accuracy_difference_v3_minus_v1": float(
            second_ok.mean() - first_ok.mean()
        ),
        "observed_roc_auc_difference_v3_minus_v1": float(
            roc_auc_score(y, second)
            - roc_auc_score(y, first)
        ),
        "paired_decisions": paired_decisions(
            y,
            first,
            second,
        ),
        "patient_bootstrap": patient_bootstrap(data),
        "provenance": {
            "predictions_sha256": sha256(PREDICTIONS),
            "ablation_report_sha256": sha256(ABLATION),
            "manifest_sha256": sha256(MANIFEST),
        },
        "limitations": [
            "Small fold-9 cohort examined repeatedly",
            "Bootstrap intervals are descriptive, not external validation",
            "Results do not justify model promotion or clinical use",
        ],
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = OUTPUT.with_name(
        OUTPUT.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    temp.replace(OUTPUT)

    counts = result["paired_decisions"]
    boot = result["patient_bootstrap"]

    print("\nCardioSense | SQI Ablation Uncertainty")
    print(f"Matched ECGs          : {len(data)}")
    print(f"Distinct patients     : {result['patients']}")
    print(
        "V3 corrected V1 errors: "
        f"{counts['v3_correct_v1_wrong']}"
    )
    print(
        "V3 introduced errors  : "
        f"{counts['v1_correct_v3_wrong']}"
    )
    print(
        "McNemar exact p       : "
        f"{counts['mcnemar_exact_two_sided_p']:.4f}"
    )
    print(
        "Accuracy difference   : "
        f"{result['observed_accuracy_difference_v3_minus_v1']:+.4f}"
    )
    print(
        "Accuracy 95% interval : "
        f"{boot['v3_minus_v1_accuracy_95pct_interval']}"
    )
    print(
        "ROC-AUC difference    : "
        f"{result['observed_roc_auc_difference_v3_minus_v1']:+.4f}"
    )
    print(
        "ROC-AUC 95% interval  : "
        f"{boot['v3_minus_v1_roc_auc_95pct_interval']}"
    )
    print(f"Report                : {OUTPUT}")
    print("Test fold and model checkpoints: unchanged")


if __name__ == "__main__":
    main()
