
"""
CardioSense — REST API.

FastAPI backend for the CardioSense research prototype.

Responsibilities:
    - Discover available ECG recordings.
    - Return standardized 12-lead ECG waveforms.
    - Run legacy V1 ensemble inference.
    - Evaluate deterministic research triage rules.
    - Present quality-aware ST measurements.
    - Generate research SBAR reports.

Input contract:
    All WFDB recordings pass through load_standardized_ecg().

Measurement contract:
    Raw legacy classifier outputs remain available for
    backward compatibility.

    Presentation measurements use the separate ST
    measurement view, which represents unavailable values
    as None rather than fabricated numerical measurements.

Safety:
    - Unknown record IDs return HTTP 404.
    - Never substitute another patient's recording.
    - Never interpret missing ST measurements as zero.
    - Measurement availability does not imply validation.

Research use only.
Not a clinically validated medical device.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    PlainTextResponse,
    RedirectResponse,
)
from pydantic import BaseModel

from src.data.ecg_loader import load_standardized_ecg
from src.dsp.filters import filter_ecg

from src.triage.leads import (
    cut_point_mv,
    find_stemi_pairs,
    normalise_age,
    normalise_sex,
)

from src.triage.report_generator import (
    generate_clinical_report,
)

from src.triage.risk_fusion import (
    CardioSenseEnsemble,
)

from src.triage.st_measurement_view import (
    build_st_measurement_view,
)

from src.triage.triage_rules import (
    ClinicalTriageEngine,
)


# =========================================================
# 1. Application configuration
# =========================================================

app = FastAPI(
    title="CardioSense API",
    description=(
        "Research API for standardized 12-lead ECG analysis, "
        "experimental model inference, quality-aware "
        "measurements, and auditable research findings."
    ),
    version="0.1.0",
)

# Preserve the existing local-development CORS policy.
# Production CORS requires a separate security review.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 2. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
COHORT_DIR = PROJECT_ROOT / "data" / "cohort"
MANIFEST_PATH = COHORT_DIR / "cohort_manifest.csv"

# Preserve names used by existing code and tests.
project_root = PROJECT_ROOT
samples_dir = SAMPLES_DIR
cohort_dir = COHORT_DIR
manifest_path = MANIFEST_PATH


# =========================================================
# 3. Research pipeline initialization
# =========================================================

# Existing startup policy:
# Load the V1-trained RF and CNN checkpoints once.
#
# Missing checkpoints prevent API initialization.
# V3 quality assessment remains a separate pathway.
ensemble_pipeline = CardioSenseEnsemble()

triage_engine = ClinicalTriageEngine()


# =========================================================
# 4. Request schema
# =========================================================

class EvaluationRequest(BaseModel):
    """Request for research ECG evaluation."""

    record_id: str
    age: Optional[float] = None
    sex: Optional[str] = None


# =========================================================
# 5. Root and health endpoints
# =========================================================

@app.get("/")
def root():
    """Redirect to the interactive API documentation."""

    return RedirectResponse(url="/docs")


@app.get("/api/health")
def health():
    """Return basic service information."""

    return {
        "status": "ok",
        "service": "CardioSense API",
        "version": "0.1.0",
    }


# =========================================================
# 6. Record discovery
# =========================================================

@app.get("/api/records")
def list_records():
    """List locally available sample and cohort ECGs."""

    records = []

    # -----------------------------------------------------
    # A. Discover sample recordings
    # -----------------------------------------------------

    for header in sorted(samples_dir.glob("**/*.hea")):

        record_path = header.with_suffix("")

        if not record_path.with_suffix(".dat").is_file():
            continue

        record_id = header.stem

        records.append({
            "id": record_id,
            "category": "sample",
            "path": str(record_path),
            "description": f"Sample record ({record_id})",
        })

    # -----------------------------------------------------
    # B. Discover pinned cohort recordings
    # -----------------------------------------------------

    if manifest_path.is_file():

        manifest = pd.read_csv(manifest_path)

        for _, row in manifest.iterrows():

            relative_path = row["filename_lr"]
            record_path = cohort_dir / relative_path

            if not (
                record_path.with_suffix(".hea").is_file()
                and record_path.with_suffix(".dat").is_file()
            ):
                continue

            ecg_id = int(row["ecg_id"])

            sex_value = row.get("sex")

            if pd.isna(sex_value):
                sex_label = None
            elif int(sex_value) == 0:
                sex_label = "Male"
            elif int(sex_value) == 1:
                sex_label = "Female"
            else:
                sex_label = None

            records.append({
                "id": str(ecg_id),
                "category": "cohort",
                "path": str(record_path),

                "age": (
                    None
                    if pd.isna(row.get("age"))
                    else float(row["age"])
                ),

                "sex": sex_label,

                "diagnosis": (
                    "infarction"
                    if int(row["target_label"]) == 1
                    else "normal"
                ),

                "description": f"Patient #{ecg_id:05d}",
            })

    return {
        "count": len(records),
        "records": records,
    }


# =========================================================
# 7. Exact ECG record lookup
# =========================================================

def _find_record_path(record_id: str) -> Path:
    """Resolve the exact requested WFDB record.

    Unknown recordings return HTTP 404.
    Never substitute another record.
    """

    # -----------------------------------------------------
    # A. Validate record identifier
    # -----------------------------------------------------

    if (
        not record_id
        or len(record_id) > 64
        or not all(
            character.isascii()
            and (
                character.isalnum()
                or character in "_-"
            )
            for character in record_id
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid ECG record ID.",
        )

    # -----------------------------------------------------
    # B. Search pinned cohort manifest
    # -----------------------------------------------------

    if (
        record_id.isdecimal()
        and manifest_path.is_file()
    ):

        manifest = pd.read_csv(
            manifest_path,
            dtype={"ecg_id": "string"},
        )

        normalized_id = str(int(record_id))

        matches = manifest[
            manifest["ecg_id"] == normalized_id
        ]

        if len(matches) > 1:

            raise HTTPException(
                status_code=500,
                detail="Duplicate ECG ID in cohort manifest.",
            )

        if not matches.empty:

            relative_path = matches.iloc[0][
                "filename_lr"
            ]

            if not isinstance(relative_path, str):

                raise HTTPException(
                    status_code=500,
                    detail="Invalid cohort waveform path.",
                )

            # Prevent path traversal through manifest data.
            base_dir = cohort_dir.resolve()

            record_path = (
                cohort_dir / relative_path
            ).resolve()

            if not record_path.is_relative_to(base_dir):

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Cohort manifest contains "
                        "an unsafe path."
                    ),
                )

            header = record_path.with_suffix(".hea")
            signal = record_path.with_suffix(".dat")

            if header.is_file() and signal.is_file():
                return record_path

            raise HTTPException(
                status_code=404,
                detail=(
                    f"ECG '{record_id}' exists in the cohort "
                    "but its waveform files are missing."
                ),
            )

    # -----------------------------------------------------
    # C. Search local sample recordings
    # -----------------------------------------------------

    sample_names = [record_id]

    if record_id.isdecimal():

        sample_names.append(
            f"{int(record_id):05d}_lr"
        )

    for sample_name in dict.fromkeys(sample_names):

        matches = sorted(
            samples_dir.glob(
                f"**/{sample_name}.hea"
            )
        )

        for header in matches:

            record_path = header.with_suffix("")

            if record_path.with_suffix(".dat").is_file():
                return record_path

    # -----------------------------------------------------
    # D. Return 404 for unknown records
    # -----------------------------------------------------

    raise HTTPException(
        status_code=404,
        detail=f"ECG record '{record_id}' not found.",
    )


# =========================================================
# 8. Standardized ECG loading
# =========================================================

def _load_record(record_id: str):
    """Load one recording through the shared ECG contract.

    Returns:
        signals_mv: ECG matrix, in millivolts.
        fs: Sampling frequency in Hz.
        lead_names: Canonical ECG lead ordering.
    """

    record_path = _find_record_path(
        record_id
    )

    try:

        signals_mv, fs, lead_names = (
            load_standardized_ecg(record_path)
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Invalid ECG input: {exc}",
        ) from exc

    except (FileNotFoundError, OSError) as exc:

        raise HTTPException(
            status_code=422,
            detail=(
                "Unable to read the requested "
                "ECG waveform."
            ),
        ) from exc

    return signals_mv, fs, lead_names


# =========================================================
# 9. Waveform endpoint
# =========================================================

@app.get("/api/records/{record_id}/signals")
def get_record_signals(record_id: str):
    """Return standardized ECG waveforms.

    Raw and filtered signals use:
        - Millivolts
        - Canonical 12-lead ordering
    """

    # -----------------------------------------------------
    # A. Load standardized ECG
    # -----------------------------------------------------

    signals_mv, fs, lead_names = _load_record(
        record_id
    )

    # -----------------------------------------------------
    # B. Build time axis
    # -----------------------------------------------------

    time_sec = [
        round(index / fs, 3)
        for index in range(len(signals_mv))
    ]

    # -----------------------------------------------------
    # C. Filter ECG
    # -----------------------------------------------------

    cleaned = filter_ecg(
        signals_mv,
        fs=fs,
    )

    # -----------------------------------------------------
    # D. Convert NumPy data for JSON
    # -----------------------------------------------------

    raw_by_lead = {
        lead: [
            round(float(value), 3)
            for value in signals_mv[:, index]
        ]
        for index, lead in enumerate(lead_names)
    }

    filtered_by_lead = {
        lead: [
            round(float(value), 3)
            for value in cleaned[:, index]
        ]
        for index, lead in enumerate(lead_names)
    }

    # Preserve the existing response structure.
    return {
        "record_id": record_id,
        "sampling_rate": fs,
        "time_seconds": time_sec,
        "lead_names": lead_names,
        "raw_signals": raw_by_lead,
        "filtered_signals": filtered_by_lead,
    }


# =========================================================
# 10. Quality-aware API ST table
# =========================================================

def _build_lead_st_table(
    lead_names,
    age,
    sex,
    measurement_view,
):
    """Build the API per-lead measurement table.

    CRITICAL:
    Read displayed ST values from measurement_view,
    not directly from legacy classifier features.

    Missing values remain None (JSON null).

    A review-required value remains visible with a
    warning; it is not considered clinically validated.

    Clinical thresholds are calculated using the
    same demographic normalization as the triage engine.
    """

    normalized_age = normalise_age(age)
    normalized_sex = normalise_sex(sex)

    per_lead = measurement_view["per_lead"]

    lead_table = []

    for lead in lead_names:

        lead_name = lead.upper()

        threshold, basis = cut_point_mv(
            lead_name,
            normalized_age,
            normalized_sex,
        )

        measurement = per_lead[lead_name]

        available = measurement["available"]

        reading = measurement["st_mv"]

        # An unavailable reading has no comparison result.
        # Do not convert None to 0.0.
        cleared = (
            reading >= threshold
            if available
            else None
        )

        lead_table.append({
            # Existing fields.
            "lead": lead,
            "st_mv": reading,
            "cut_point_mv": threshold,
            "basis": basis,
            "cleared": cleared,

            # Milestone 4 fields.
            "available": available,

            "quality_status": measurement[
                "quality_status"
            ],

            "quality_reasons": measurement[
                "reasons"
            ],

            # Availability != clinical validation.
            "clinical_validation": "not_established",
        })

    return lead_table


# =========================================================
# 11. Research evaluation endpoint
# =========================================================

@app.post("/api/triage/evaluate")
def evaluate_record(req: EvaluationRequest):
    """Run the legacy V1 research evaluation pipeline.

    Workflow:
        1. Load standardized ECG.
        2. Run legacy V1 ensemble.
        3. Evaluate deterministic triage rules.
        4. Build quality-aware measurement view.
        5. Generate research report.
        6. Return results with provenance.

    The measurement view does not modify the outputs
    produced by the original ensemble or triage engine.
    """

    # -----------------------------------------------------
    # A. Load the requested ECG
    # -----------------------------------------------------

    signals_mv, fs, lead_names = _load_record(
        req.record_id
    )

    # -----------------------------------------------------
    # B. Run existing V1 ensemble
    # -----------------------------------------------------

    try:

        ensemble_res = ensemble_pipeline.predict(
            signals_mv,
            fs=fs,
            lead_names=lead_names,
            age=req.age,
            sex=req.sex,
            units="mV",
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # -----------------------------------------------------
    # C. Handle rejected recordings
    # -----------------------------------------------------

    if ensemble_res.get("status") == "rejected":

        return {
            "status": "rejected",
            "reason": ensemble_res.get("reason"),
            "sqi_details": ensemble_res.get(
                "sqi_details"
            ),
        }

    # -----------------------------------------------------
    # D. Apply existing deterministic triage rules
    # -----------------------------------------------------

    triage_res = triage_engine.evaluate(
        ensemble_res
    )

    # -----------------------------------------------------
    # E. Build the measurement presentation contract
    # -----------------------------------------------------

    # This is the Milestone 4 API integration.
    #
    # The presentation view excludes unavailable ST
    # readings and ignores legacy classifier aggregates.
    #
    # The original ensemble result is not modified.

    measurement_view = build_st_measurement_view(
        ensemble_res
    )

    # -----------------------------------------------------
    # F. Generate research report
    # -----------------------------------------------------

    report_text = generate_clinical_report(
        req.record_id,
        ensemble_res,
        triage_res,
    )

    # -----------------------------------------------------
    # G. Preserve original research ST rule findings
    # -----------------------------------------------------

    # Existing triage findings are preserved.
    # Measurement quality warnings are advisory and
    # do not silently change the configured rule engine.
    #
    # The measurement view may flag a value as unavailable
    # for presentation even if the legacy rule engine
    # previously used it. Such concerns must be visible,
    # not silently interpreted as clinical validation.

    st_by_lead = ensemble_res["st_by_lead"]

    normalized_age = normalise_age(req.age)
    normalized_sex = normalise_sex(req.sex)

    stemi_pairs = [
        pair._asdict()
        for pair in find_stemi_pairs(
            st_by_lead,
            normalized_age,
            normalized_sex,
        )
    ]

    # -----------------------------------------------------
    # H. Build the quality-aware per-lead table
    # -----------------------------------------------------

    lead_table = _build_lead_st_table(
        lead_names,
        req.age,
        req.sex,
        measurement_view,
    )

    # -----------------------------------------------------
    # I. Return original and presentation results
    # -----------------------------------------------------

    return {
        "status": "success",
        "record_id": req.record_id,

        # Existing fields retained.
        "triage": triage_res,
        "ensemble": ensemble_res,
        "contiguous_stemi_pairs": stemi_pairs,
        "lead_st_table": lead_table,
        "sbar_report": report_text,

        # New quality-aware presentation.
        "measurement_view": measurement_view,

        # Explicitly describe the two interpretations.
        "measurement_presentation": {
            "version": "st_measurement_view_v1",
            "units": "mV",

            "clinical_validation": "not_established",

            "legacy_classifier_aggregates_used": False,

            "model_scoring_modified": False,
            "triage_routing_modified": False,

            "note": (
                "Use measurement_view and lead_st_table "
                "for displayed ST measurements. "
                "The ensemble object preserves legacy "
                "research classifier outputs, which may "
                "contain historical numeric placeholders."
            ),
        },
    }


# =========================================================
# 12. Plain-text research report endpoint
# =========================================================

@app.get(
    "/api/records/{record_id}/report",
    response_class=PlainTextResponse,
)
def get_record_report(
    record_id: str,
    age: Optional[float] = None,
    sex: Optional[str] = None,
):
    """Return the research SBAR report as plain text."""

    req = EvaluationRequest(
        record_id=record_id,
        age=age,
        sex=sex,
    )

    # Reuse the standardized evaluation pipeline.
    evaluation = evaluate_record(req)

    if evaluation.get("status") == "rejected":

        raise HTTPException(
            status_code=400,
            detail=evaluation.get("reason"),
        )

    return evaluation["sbar_report"]