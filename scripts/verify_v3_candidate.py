"""Read-only integrity gate for the SQI V3 Random Forest research candidate.

A historical manifest SHA mismatch is NOT erased or rewritten. A narrowly
scoped research exception is possible when the current manifest is exactly
reproducible from the pinned cohort and local PTB-XL source metadata, and its
clinical/dataset metadata agrees with the frozen quality audit and V3 rows.

No checkpoint writes, model training, predictions, or test-fold evaluation.
An exception is NOT historical byte-for-byte provenance or clinical validation.
"""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

from src.data.download_ptbxl import build_cohort_manifest, load_pinned_cohort
from src.features.extract_features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]

MODEL = "models/checkpoints/interpretable_model_v3_research_candidate.joblib"
REPORT = "docs/model_contracts/interpretable_model_v3_research_candidate.json"
DATASET = "data/processed/clinical_features_v3_candidate.csv"
DATASET_CONTRACT = "docs/sqi_audit/v3_dataset_contract.json"
QUALITY_AUDIT = "docs/sqi_audit/v1_v3_feature_shift_records.csv"
TRAINER = "scripts/train_v3_rf_candidate.py"
MANIFEST = "data/cohort/cohort_manifest.csv"
PIN = "data/processed/cohort_ecg_ids.csv"
OFFICIAL_METADATA = "data/raw/ptbxl_database.csv"
BASELINE_AUDIT = "docs/model_contract_audit.json"
BASELINE_RF = "models/checkpoints/interpretable_model.joblib"
BASELINE_CNN = "models/checkpoints/ecg_cnn.pt"

META_COLUMNS = (
    "ecg_id", "patient_id", "target_label",
    "strat_fold", "age", "sex",
)


def sha256_file(path):
    """Calculate the SHA-256 fingerprint of a local file without editing it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_numeric(left, right, columns):
    """Check exact values, treating matching missing values as equal.

    Used only after both frames have been aligned by ECG ID. Parsing failures
    fail closed; NA is never silently converted into a numeric zero.
    """
    for column in columns:
        a = pd.to_numeric(left[column], errors="raise").reset_index(drop=True)
        b = pd.to_numeric(right[column], errors="raise").reset_index(drop=True)
        equal = a.eq(b) | (a.isna() & b.isna())
        if not bool(equal.fillna(False).all()):
            return False
    return True


def verify_reconstructed_manifest(root):
    """Check independent reproducibility and all frozen cohort relationships.

    Does NOT recover the historical manifest bytes. Only proves the presently
    materialized manifest is reproducible from the canonical pinned cohort and
    the current local official PTB-XL metadata, and matches frozen audit/data.

    Returns (valid, explanation). No persistent files are changed.
    """
    root = Path(root)
    try:
        with TemporaryDirectory(prefix="cardiosense_manifest_audit_") as tmp:
            build_cohort_manifest(
                pinned_ids_csv=root / PIN,
                database_csv=root / OFFICIAL_METADATA,
                cohort_dir=Path(tmp),
                download=False,
            )
            regenerated = Path(tmp) / "cohort_manifest.csv"
            if sha256_file(regenerated) != sha256_file(root / MANIFEST):
                return False, "Current manifest does not reproduce byte-for-byte from pinned IDs and local PTB-XL metadata"

        # The pin is the authoritative 300-record membership definition.
        pin = load_pinned_cohort(root / PIN)
        manifest = pd.read_csv(root / MANIFEST)
        audit = pd.read_csv(root / QUALITY_AUDIT)
        dataset = pd.read_csv(root / DATASET)

        required_manifest = set(META_COLUMNS) | {"filename_lr"}
        required_audit = {"ecg_id", "target_label", "strat_fold", "quality_decision"}
        required_dataset = set(META_COLUMNS) | set(FEATURE_COLUMNS)
        if not required_manifest.issubset(manifest.columns):
            return False, "Manifest lacks required metadata fields"
        if not required_audit.issubset(audit.columns):
            return False, "Frozen quality audit lacks expected fields"
        if not required_dataset.issubset(dataset.columns):
            return False, "Frozen V3 dataset lacks metadata or classifier features"

        if (
            len(manifest) != len(pin)
            or len(audit) != len(pin)
            or manifest.ecg_id.isna().any()
            or audit.ecg_id.isna().any()
            or dataset.ecg_id.isna().any()
            or not manifest.ecg_id.is_unique
            or not audit.ecg_id.is_unique
            or not dataset.ecg_id.is_unique
        ):
            return False, "Missing or duplicated ECG identifiers in cohort/audit/V3 dataset"

        if manifest.ecg_id.tolist() != pin.ecg_id.tolist():
            return False, "Manifest changed pinned ECG order or membership"

        if not _aligned_numeric(manifest, pin, ("ecg_id", "target_label", "strat_fold")):
            return False, "Manifest label or fold does not match canonical pin"

        # Every pinned record must retain its historical audit label/fold.
        aligned_audit = audit.set_index("ecg_id").reindex(pin.ecg_id).reset_index()
        if aligned_audit["quality_decision"].isna().any():
            return False, "Frozen quality audit is missing pinned ECGs"
        if not _aligned_numeric(
            manifest,
            aligned_audit,
            ("ecg_id", "target_label", "strat_fold"),
        ):
            return False, "Manifest labels or folds differ from frozen quality audit"

        if not audit.quality_decision.isin(("eligible", "abstain")).all():
            return False, "Unexpected quality selection decisions"
        eligible = set(audit.loc[audit.quality_decision.eq("eligible"), "ecg_id"])
        if set(dataset.ecg_id) != eligible:
            return False, "V3 feature IDs differ from frozen quality-eligible ECGs"

        # Align all SIX metadata fields for every V3 candidate row.
        aligned_manifest = manifest.set_index("ecg_id").reindex(dataset.ecg_id).reset_index()
        if not _aligned_numeric(dataset, aligned_manifest, META_COLUMNS):
            return False, "V3 feature dataset and manifest disagree on ECG/patient/label/fold/age/sex"

        # Patient-level separation is essential, including records excluded by SQI.
        if manifest.patient_id.isna().any():
            return False, "Manifest contains missing patient identifiers"
        if manifest.groupby("patient_id").strat_fold.nunique().gt(1).any():
            return False, "Patient appears in more than one stratified fold"

        return True, "Reconstructed manifest matches current bytes, pin, frozen quality audit, and all V3 metadata"

    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        return False, f"Cannot verify reconstructed manifest: {exc}"


def audit_candidate(root=ROOT):
    """Verify frozen candidate provenance with an explicit manifest exception."""
    root = Path(root)
    checks = {}
    issues = []
    warnings = []
    model_schema = None
    manifest_provenance = "unverified"
    expected_manifest_sha256 = None
    actual_manifest_sha256 = None

    def record(name, passed, failure):
        checks[name] = bool(passed)
        if not passed:
            issues.append(failure)

    def read_json(name, relative):
        try:
            value = json.loads((root / relative).read_text(encoding="utf-8"))
            record(name, isinstance(value, dict), f"Invalid JSON object: {relative}")
            return value if isinstance(value, dict) else None
        except (OSError, ValueError) as exc:
            record(name, False, f"Cannot read {relative}: {exc}")
            return None

    def hash_check(name, relative, expected):
        path = root / relative
        good = (
            isinstance(expected, str)
            and len(expected) == 64
            and path.is_file()
            and sha256_file(path) == expected
        )
        record(name, good, f"Missing or changed file: {relative}")
        return good

    report = read_json("training_report_readable", REPORT)
    contract = read_json("dataset_contract_readable", DATASET_CONTRACT)
    baseline = read_json("baseline_audit_readable", BASELINE_AUDIT)

    if report is None or contract is None or baseline is None:
        return {
            "status": "blocked", "checks": checks, "issues": issues,
            "warnings": warnings, "model_schema": None,
            "manifest_provenance": manifest_provenance,
        }

    record(
        "training_contract_identity",
        report.get("contract_version") == "rf_v3_research_candidate_v1"
        and report.get("status") == "research_candidate"
        and report.get("sqi_version") == "v3"
        and report.get("quality_policy") == "fail_closed_research_gate"
        and report.get("feature_order") == FEATURE_COLUMNS
        and report.get("class_order") == [0, 1],
        "Training contract identity or feature/class schema mismatch",
    )
    record(
        "frozen_dataset_identity",
        contract.get("contract_version") == "v3_research_dataset_v1"
        and contract.get("sqi_version") == "v3"
        and contract.get("feature_order") == FEATURE_COLUMNS
        and contract.get("quality_policy") == "fail_closed_research_gate",
        "Dataset contract identity or classifier feature order mismatch",
    )

    train = report.get("training") or {}
    validation = report.get("validation") or {}
    test = report.get("test") or {}
    counts = {key: 0 for key in ("train", "validation", "test")}
    for item in contract.get("split_label_counts", []):
        if item.get("split") in counts:
            counts[item["split"]] += int(item.get("count", 0))

    record(
        "locked_partitions",
        train.get("folds") == list(range(1, 9))
        and validation.get("fold") == 9
        and test.get("fold") == 10
        and test.get("evaluated") is False
        and train.get("records") == counts["train"]
        and (validation.get("metrics") or {}).get("records") == counts["validation"]
        and test.get("records") == counts["test"]
        and sum(counts.values()) == contract.get("record_count"),
        "Training/validation/test partition mismatch or fold 10 has been evaluated",
    )
    record(
        "no_promotion_claim",
        report.get("deployed_model_replaced") is False
        and report.get("clinical_validation") == "not_established",
        "Candidate report claims deployment or clinical validation",
    )

    artifacts = report.get("artifacts") or {}
    record(
        "candidate_path_identity",
        artifacts.get("candidate_model") == MODEL,
        "Research checkpoint path differs from frozen candidate path",
    )
    hash_check("candidate_model_hash", MODEL, artifacts.get("candidate_model_sha256"))
    hash_check("training_dataset_hash", DATASET, artifacts.get("training_dataset_sha256"))
    hash_check("dataset_contract_hash", DATASET_CONTRACT, artifacts.get("dataset_contract_sha256"))
    hash_check("quality_audit_hash", QUALITY_AUDIT, artifacts.get("quality_audit_sha256"))
    hash_check("training_script_hash", TRAINER, artifacts.get("training_script_sha256"))
    record(
        "dataset_hash_agreement",
        artifacts.get("training_dataset_sha256") == (contract.get("dataset") or {}).get("sha256"),
        "Candidate and dataset contracts disagree on dataset hash",
    )
    record(
        "quality_audit_hash_agreement",
        artifacts.get("quality_audit_sha256") == (contract.get("inputs") or {}).get("quality_audit_sha256"),
        "Candidate and dataset contracts disagree on quality audit hash",
    )

    for name, expected in (contract.get("source_sha256") or {}).items():
        hash_check(f"source:{name}", name, expected)

    deployed = baseline.get("deployed_models") or {}
    rf_hash = ((deployed.get("random_forest") or {}).get("artifact") or {}).get("sha256")
    cnn_hash = ((deployed.get("cnn") or {}).get("artifact") or {}).get("sha256")
    hash_check("deployed_rf_unchanged", BASELINE_RF, rf_hash)
    hash_check("deployed_cnn_unchanged", BASELINE_CNN, cnn_hash)

    # Only the historical manifest hash is eligible for the narrow exception.
    expected_manifest_sha256 = (contract.get("inputs") or {}).get("manifest_sha256")
    manifest_file = root / MANIFEST
    if manifest_file.is_file():
        actual_manifest_sha256 = sha256_file(manifest_file)
    same_manifest = (
        isinstance(expected_manifest_sha256, str)
        and len(expected_manifest_sha256) == 64
        and actual_manifest_sha256 == expected_manifest_sha256
    )
    checks["historical_manifest_hash_matches"] = same_manifest

    if same_manifest:
        manifest_provenance = "historical_hash_matched"
        checks["manifest_reconstruction_verified"] = None  # Not needed.
    else:
        verified, explanation = verify_reconstructed_manifest(root)
        checks["manifest_reconstruction_verified"] = verified
        if verified:
            manifest_provenance = "reconstructed_metadata_aligned"
            warnings.append(
                "Historical manifest SHA-256 mismatch is unresolved. "
                "Current manifest can be regenerated and matches frozen "
                "cohort/feature metadata; historical byte identity is NOT verified."
            )
        else:
            record(
                "manifest_reconstruction_accepted",
                False,
                f"Historical manifest hash mismatch; {explanation}",
            )

    # joblib uses pickle. Never deserialize if any mandatory check fails.
    if not issues:
        try:
            model = joblib.load(root / MODEL)
            names = [str(name) for name in model.feature_names_in_]
            classes = [int(label) for label in model.classes_]
            count = int(model.n_features_in_)
            model_schema = {
                "feature_count": count,
                "classes": classes,
                "feature_order_matches": names == FEATURE_COLUMNS,
            }
            record(
                "candidate_rf_structure",
                names == FEATURE_COLUMNS
                and count == len(FEATURE_COLUMNS)
                and classes == [0, 1],
                "V3 research checkpoint feature schema or class order mismatch",
            )
        except Exception as exc:
            record("candidate_rf_structure", False, f"Cannot inspect verified local checkpoint: {exc}")

    if issues:
        status = "blocked"
    elif manifest_provenance == "historical_hash_matched":
        status = "integrity_verified"
    else:
        status = "research_integrity_verified_with_manifest_exception"

    return {
        "status": status,
        "checks": checks,
        "issues": issues,
        "warnings": warnings,
        "model_schema": model_schema,
        "manifest_provenance": manifest_provenance,
        "expected_historical_manifest_sha256": expected_manifest_sha256,
        "actual_local_manifest_sha256": actual_manifest_sha256,
        "clinical_validation": "not_established",
        "test_fold_evaluated_by_this_audit": False,
    }


def main():
    result = audit_candidate()
    print("CardioSense V3 candidate | read-only integrity gate")
    for name, passed in result["checks"].items():
        if passed is None:
            label = "N/A"
        elif name == "historical_manifest_hash_matches" and not passed:
            label = "MISMATCH"
        else:
            label = "PASS" if passed else "FAIL"
        print(f"{label} | {name}")

    print(f"\nSTATUS: {result['status']}")
    print(f"MANIFEST PROVENANCE: {result['manifest_provenance']}")
    print(f"EXPECTED HISTORICAL MANIFEST SHA: {result['expected_historical_manifest_sha256']}")
    print(f"ACTUAL LOCAL MANIFEST SHA      : {result['actual_local_manifest_sha256']}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    for issue in result["issues"]:
        print(f"BLOCKER: {issue}")
    print("No training, prediction, or test-fold evaluation was performed.")
    return 0 if result["status"] in (
        "integrity_verified", "research_integrity_verified_with_manifest_exception"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())