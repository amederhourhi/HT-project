
"""Milestone 9: audit saved LUDB boundary regressions and research readiness.

Read-only: no waveform access, model inference, fitting, or reserved-set evaluation.
Existing LUDB 31-100 results are previously evaluated development evidence,
not a fresh independent test set.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_ludb_qrs_accuracy import RECORDS, make_report, verify_holdout


ROOT = Path(__file__).resolve().parents[1]

HOLDOUT = ROOT / "docs/clinical_validation/ludb_qrs_v4_holdout.csv"
OFFSET_ONLY = ROOT / "docs/clinical_validation/ludb_qrs_v4_holdout_offset_only.csv"
ONSET_NOTES = (
    ROOT
    / "docs/clinical_validation/onset_investigation/onset_investigation.csv"
)

FOCUS = (31, 72, 78, 85)

ERRORS = (
    "onset_error_ms",
    "offset_error_ms",
    "duration_error_ms",
)

TOLERANCE_MS = 1e-9


def fingerprint(path):
    """Identify input bytes; this is not a frozen-hash verification."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _numeric(frame, column, *, name):
    """Read finite numeric values or reject invalid evidence."""

    try:
        values = pd.to_numeric(
            frame[column],
            errors="raise",
        ).to_numpy(dtype=float)

    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{name}: invalid {column}"
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name}: nonfinite {column}"
        )

    return values


def _integer_ids(frame, *, name):
    """Require unique integer recording identifiers."""

    if "record_id" not in frame or frame.empty:
        raise ValueError(
            f"{name}: missing record IDs"
        )

    ids = _numeric(
        frame,
        "record_id",
        name=name,
    )

    if not np.equal(
        ids,
        np.floor(ids),
    ).all():
        raise ValueError(
            f"{name}: noninteger record IDs"
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"{name}: duplicate record IDs"
        )

    return ids.astype(int)


def verify_counterfactual(
    frame,
    verified,
    *,
    expected_records=RECORDS,
):
    """Validate saved current-onset + V4-offset counterfactual."""

    needed = {
        "record_id",
        "method",
        *ERRORS,
    }

    if (
        not isinstance(frame, pd.DataFrame)
        or not needed.issubset(frame)
    ):
        raise ValueError(
            "Counterfactual: missing columns"
        )

    data = frame.copy()

    data["record_id"] = _integer_ids(
        data,
        name="Counterfactual",
    )

    if set(data.record_id) != set(expected_records):
        raise ValueError(
            "Counterfactual: missing or reserved record IDs"
        )

    if not data.method.eq(
        "offset_only_counterfactual"
    ).all():
        raise ValueError(
            "Counterfactual: unexpected method"
        )

    for key in ERRORS:
        data[key] = _numeric(
            data,
            key,
            name="Counterfactual",
        )

    base = (
        verified.loc[
            verified.method.eq("current")
        ]
        .set_index("record_id")
    )

    v4 = (
        verified.loc[
            verified.method.eq("qrs_v4")
        ]
        .set_index("record_id")
    )

    cf = data.set_index("record_id")

    ids = sorted(expected_records)

    expected_onset = base.loc[
        ids,
        "onset_error_ms",
    ].to_numpy()

    expected_offset = v4.loc[
        ids,
        "offset_error_ms",
    ].to_numpy()

    checks = (
        (
            "onset",
            cf.loc[ids, "onset_error_ms"].to_numpy(),
            expected_onset,
        ),
        (
            "offset",
            cf.loc[ids, "offset_error_ms"].to_numpy(),
            expected_offset,
        ),
        (
            "duration",
            cf.loc[ids, "duration_error_ms"].to_numpy(),
            expected_offset - expected_onset,
        ),
    )

    for name, actual, expected in checks:
        if not np.allclose(
            actual,
            expected,
            rtol=0,
            atol=TOLERANCE_MS,
        ):
            raise ValueError(
                f"Counterfactual: inconsistent {name} error"
            )

    return cf


def verify_onset_notes(
    frame,
    verified,
    *,
    expected_records=RECORDS,
):
    """Validate recorded onset investigation and separate development IDs."""

    needed = {
        "record_id",
        "expert_lead_ii_median_onset_ms",
        "expert_lead_ii_onset_iqr_ms",
        "original_global_onset_ms",
        "v4_global_onset_ms",
        "v4_minus_original_ms",
        "expert_beats",
    }

    if (
        not isinstance(frame, pd.DataFrame)
        or not needed.issubset(frame)
    ):
        raise ValueError(
            "Onset notes: missing columns"
        )

    data = frame.copy()

    data["record_id"] = _integer_ids(
        data,
        name="Onset notes",
    )

    # Historical notes include LUDB 20 and 23.
    # They are development cases, not holdout outcomes.
    allowed = (
        set(expected_records)
        | set(range(1, 31))
    )

    if not set(data.record_id).issubset(allowed):
        raise ValueError(
            "Onset notes: unexpected or reserved record"
        )

    for name in needed - {"record_id"}:
        data[name] = _numeric(
            data,
            name,
            name="Onset notes",
        )

    if (
        (data["expert_beats"] <= 0).any()
        or not np.equal(
            data["expert_beats"],
            np.floor(data["expert_beats"]),
        ).all()
    ):
        raise ValueError(
            "Onset notes: invalid expert-beat count"
        )

    if (
        data["expert_lead_ii_onset_iqr_ms"] < 0
    ).any():
        raise ValueError(
            "Onset notes: negative onset IQR"
        )

    if not np.allclose(
        (
            data["v4_global_onset_ms"]
            - data["original_global_onset_ms"]
        ),
        data["v4_minus_original_ms"],
        rtol=0,
        atol=TOLERANCE_MS,
    ):
        raise ValueError(
            "Onset notes: inconsistent onset displacement"
        )

    paired = verified.pivot(
        index="record_id",
        columns="method",
        values=list(ERRORS),
    )

    for row in data.itertuples(index=False):
        rid = int(row.record_id)

        if rid not in expected_records:
            continue

        current_err = float(
            paired.loc[
                rid,
                ("onset_error_ms", "current"),
            ]
        )

        candidate_err = float(
            paired.loc[
                rid,
                ("onset_error_ms", "qrs_v4"),
            ]
        )

        # Both algorithm errors must imply the same
        # annotated global QRS-onset reference.
        reference_from_current = (
            row.original_global_onset_ms
            - current_err
        )

        reference_from_v4 = (
            row.v4_global_onset_ms
            - candidate_err
        )

        if not np.isclose(
            reference_from_current,
            reference_from_v4,
            rtol=0,
            atol=TOLERANCE_MS,
        ):
            raise ValueError(
                f"Onset notes: reference mismatch for {rid}"
            )

    return data.set_index("record_id")


def analyze_failure_cases(
    holdout,
    counterfactual,
    onset_notes,
    *,
    expected_records=RECORDS,
    focus=FOCUS,
    draws=3000,
    seed=42,
):
    """Identify observed boundary-error mechanisms, not unproven causes."""

    # Reuse Milestone 8's complete expert-reference gate.
    verified = verify_holdout(
        holdout,
        expected_records=expected_records,
    )

    cf = verify_counterfactual(
        counterfactual,
        verified,
        expected_records=expected_records,
    )

    notes = verify_onset_notes(
        onset_notes,
        verified,
        expected_records=expected_records,
    )

    if (
        not focus
        or len(set(focus)) != len(focus)
        or not set(focus).issubset(
            set(expected_records)
        )
    ):
        raise ValueError(
            "Invalid focused record IDs"
        )

    summary = make_report(
        verified,
        expected_records=expected_records,
        draws=draws,
        seed=seed,
    )

    indexed = {
        method: (
            verified.loc[
                verified.method.eq(method)
            ]
            .set_index("record_id")
        )
        for method in ("current", "qrs_v4")
    }

    cases = []

    for rid in focus:
        a = indexed["current"].loc[rid]
        b = indexed["qrs_v4"].loc[rid]

        onset_shift = float(
            b.onset_error_ms - a.onset_error_ms
        )

        offset_shift = float(
            b.offset_error_ms - a.offset_error_ms
        )

        duration_shift = float(
            b.duration_error_ms - a.duration_error_ms
        )

        # QRS duration = offset - onset.
        # This identity distinguishes which boundary
        # accounts for the observed duration change.
        if not np.isclose(
            duration_shift,
            offset_shift - onset_shift,
            rtol=0,
            atol=TOLERANCE_MS,
        ):
            raise ValueError(
                f"QRS boundary identity mismatch: {rid}"
            )

        moved_onset = (
            abs(onset_shift) > TOLERANCE_MS
        )

        moved_offset = (
            abs(offset_shift) > TOLERANCE_MS
        )

        mechanism = (
            "both_boundaries_shifted"
            if moved_onset and moved_offset
            else "onset_shift_only"
            if moved_onset
            else "offset_shift_only"
            if moved_offset
            else "no_boundary_shift"
        )

        item = {
            "record_id": int(rid),
            "observed_mechanism": mechanism,

            "onset_shift_v4_minus_current_ms":
                onset_shift,

            "offset_shift_v4_minus_current_ms":
                offset_shift,

            "duration_shift_v4_minus_current_ms":
                duration_shift,

            "current_absolute_duration_error_ms":
                float(abs(a.duration_error_ms)),

            "v4_absolute_duration_error_ms":
                float(abs(b.duration_error_ms)),

            "absolute_duration_error_increase_ms":
                float(
                    abs(b.duration_error_ms)
                    - abs(a.duration_error_ms)
                ),

            "offset_only_counterfactual_absolute_duration_error_ms":
                float(
                    abs(cf.loc[rid, "duration_error_ms"])
                ),

            "lead_ii_reference": None,
        }

        # Expert lead-II onset and global onset are
        # different reference definitions.
        # Never compare their errors as if they
        # represented the same ground truth.
        if rid in notes.index:
            n = notes.loc[rid]

            old_ii = abs(float(
                n.original_global_onset_ms
                - n.expert_lead_ii_median_onset_ms
            ))

            new_ii = abs(float(
                n.v4_global_onset_ms
                - n.expert_lead_ii_median_onset_ms
            ))

            old_global = abs(
                float(a.onset_error_ms)
            )

            new_global = abs(
                float(b.onset_error_ms)
            )

            item["lead_ii_reference"] = {
                "expert_beats": int(n.expert_beats),

                "expert_lead_ii_onset_iqr_ms":
                    float(n.expert_lead_ii_onset_iqr_ms),

                "current_distance_to_lead_ii_onset_ms":
                    float(old_ii),

                "v4_distance_to_lead_ii_onset_ms":
                    float(new_ii),

                "current_distance_to_global_onset_ms":
                    float(old_global),

                "v4_distance_to_global_onset_ms":
                    float(new_global),

                "reference_dependent_ranking": bool(
                    (new_ii < old_ii)
                    != (new_global < old_global)
                ),
            }

        cases.append(item)

    dev_only = sorted(
        int(x)
        for x in notes.index
        if x not in expected_records
    )

    return {
        "audit":
            "m09_recorded_failure_mechanism_and_readiness_v1",

        "evaluated_saved_ludb_records":
            len(expected_records),

        "focus_cases": cases,

        "development_only_onset_note_ids":
            dev_only,

        "duration_comparison": summary[
            "metrics"
        ]["duration_error_ms"]["paired"],

        "research_readiness": {
            "software_measurement_audit":
                "reproducible_from_saved_artifacts",

            "qrs_v4_promotion": "blocked",

            "clinical_readiness": "not_established",

            "unverified_causes": [
                (
                    "why specific leads and waveform "
                    "features trigger different onset boundaries"
                ),
                "full-pipeline automated R-peak performance",
                (
                    "independent ST amplitude accuracy "
                    "and diagnostic validity"
                ),
                "patient-level clinical generalization",
                (
                    "unresolved historical manifest byte identity "
                    "and V1 checkpoint training lineage"
                ),
            ],
        },

        "reserved_data_read_by_this_audit": {
            "ludb_101_200": False,
            "ptbxl_fold_10": False,
        },

        "limitations": [
            (
                "LUDB 31-100 had already been evaluated; "
                "not a fresh holdout"
            ),
            (
                "Lead-II and global QRS onset definitions "
                "are not interchangeable"
            ),
            (
                "Boundary displacement is observed; "
                "physical algorithmic root cause is unproven"
            ),
            (
                "Offset-only results are post hoc counterfactuals, "
                "not an independent method validation"
            ),
            (
                "No waveforms, models, new predictions, "
                "or reserved records used"
            ),
        ],
    }


def main():
    """Read archived measurements and print a research-readiness report."""

    source = {
        "holdout": HOLDOUT,
        "offset_only": OFFSET_ONLY,
        "onset_notes": ONSET_NOTES,
    }

    data = {
        key: pd.read_csv(path)
        for key, path in source.items()
    }

    result = analyze_failure_cases(
        data["holdout"],
        data["offset_only"],
        data["onset_notes"],
    )

    # Observed fingerprints identify the inputs used.
    # They are not proof of historical byte identity.
    result["input_sha256_observed_not_pinned"] = {
        key: fingerprint(path)
        for key, path in source.items()
    }

    print(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        )
    )

    print(
        "READ-ONLY: no waveform or checkpoint changes; "
        "no reserved-set evaluation"
    )


if __name__ == "__main__":
    main()