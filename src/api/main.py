
"""
CardioSense — REST API

FastAPI backend for the CardioSense research prototype.

Responsibilities:
    - Discover available ECG recordings
    - Return standardized 12-lead waveforms
    - Run legacy V1 ensemble inference
    - Generate research triage reports

Input contract:
    Every WFDB recording passes through the centralized
    load_standardized_ecg() function before processing.

    The loader guarantees:
        - Valid 12-lead ECG structure
        - Standardized channel ordering
        - Voltage conversion to millivolts
        - Valid sampling frequency
        - Finite waveform samples

Safety:
    Unknown record IDs must return HTTP 404.
    Never substitute a different patient's recording.

Research use only. Not a clinically validated medical device.
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
)
from src.triage.report_generator import generate_clinical_report
from src.triage.risk_fusion import CardioSenseEnsemble
from src.triage.triage_rules import ClinicalTriageEngine


# =========================================================
# 1. Application configuration
# =========================================================

app = FastAPI(
    title="CardioSense API",
    description=(
        "Research API for 12-lead ECG analysis, "
        "experimental model inference, and auditable "
        "signal-derived findings."
    ),
    version="0.1.0",
)

# Preserve the existing local-development CORS policy.
# Production origins and credentials require separate review.
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


# Preserve the original module-level names for compatibility
# with existing code that may import or monkeypatch them.
project_root = PROJECT_ROOT
samples_dir = SAMPLES_DIR
cohort_dir = COHORT_DIR
manifest_path = MANIFEST_PATH


# =========================================================
# 3. Research pipeline initialization
# =========================================================

# Load existing V1 RF and CNN checkpoints once.
#
# This preserves the current application's startup policy.
# Missing checkpoints will prevent API initialization.
#
# V3 assessment remains separate from legacy V1 scoring.
ensemble_pipeline = CardioSenseEnsemble()

triage_engine = ClinicalTriageEngine()


# =========================================================
# 4. Request schemas
# =========================================================

class EvaluationRequest(BaseModel):
    """Input for the research evaluation endpoint."""

    record_id: str
    age: Optional[float] = None
    sex: Optional[str] = None


# =========================================================
# 5. API root and health check
# =========================================================

@app.get("/")
def root():
    """Redirect users to the interactive API documentation."""
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
    """List locally available sample and cohort recordings."""

    records = []

    # -----------------------------------------------------
    # A. Discover sample recordings.
    # -----------------------------------------------------
    for header in sorted(samples_dir.glob("**/*.hea")):

        record_path = header.with_suffix("")

        # A usable WFDB record needs both files.
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
    # B. Discover cohort recordings.
    # -----------------------------------------------------
    if manifest_path.is_file():

        manifest = pd.read_csv(manifest_path)

        for _, row in manifest.iterrows():

            relative_path = row["filename_lr"]
            record_path = cohort_dir / relative_path

            # Skip incomplete local files.
            # The canonical cohort definition is unchanged.
            if not (
                record_path.with_suffix(".hea").is_file()
                and record_path.with_suffix(".dat").is_file()
            ):
                continue

            ecg_id = int(row["ecg_id"])

            # Preserve existing response fields while avoiding
            # a false female assignment for missing sex data.
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
    """Return the exact requested WFDB record path.

    Unknown records return HTTP 404.

    No fallback to a different sample is permitted.
    """

    # -----------------------------------------------------
    # A. Validate the requested identifier.
    # -----------------------------------------------------
    # Prevent path separators, wildcard characters, and
    # unsupported filename expressions.
    if (
        not record_id
        or len(record_id) > 64
        or not all(
            char.isascii()
            and (char.isalnum() or char in "_-")
            for char in record_id
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid ECG record ID.",
        )

    # -----------------------------------------------------
    # B. Check the official cohort manifest.
    # -----------------------------------------------------
    if record_id.isdecimal() and manifest_path.is_file():

        manifest = pd.read_csv(
            manifest_path,
            dtype={"ecg_id": "string"},
        )

        normalized_id = str(int(record_id))

        matches = manifest[
            manifest["ecg_id"] == normalized_id
        ]

        # A duplicate is a manifest integrity error.
        if len(matches) > 1:
            raise HTTPException(
                status_code=500,
                detail="Duplicate ECG ID in cohort manifest.",
            )

        if not matches.empty:

            relative_path = matches.iloc[0]["filename_lr"]

            if not isinstance(relative_path, str):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid cohort waveform path.",
                )

            # Resolve and constrain the path to data/cohort.
            # Metadata must never point outside that directory.
            base_dir = cohort_dir.resolve()
            record_path = (
                cohort_dir / relative_path
            ).resolve()

            if not record_path.is_relative_to(base_dir):
                raise HTTPException(
                    status_code=500,
                    detail="Cohort manifest contains an unsafe path.",
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
    # C. Check local sample recordings.
    # -----------------------------------------------------
    # Examples:
    #   "00001_lr" -> "00001_lr"
    #   "1"        -> "00001_lr"
    sample_names = [record_id]

    if record_id.isdecimal():
        sample_names.append(
            f"{int(record_id):05d}_lr"
        )

    # Use exact filename matching rather than a prefix
    # such as 'record_id*', which may match other records.
    for sample_name in dict.fromkeys(sample_names):

        matches = sorted(
            samples_dir.glob(f"**/{sample_name}.hea")
        )

        for header in matches:

            record_path = header.with_suffix("")

            if record_path.with_suffix(".dat").is_file():
                return record_path

    # -----------------------------------------------------
    # D. Not found.
    # -----------------------------------------------------
    # Never return the first available sample.
    raise HTTPException(
        status_code=404,
        detail=f"ECG record '{record_id}' not found.",
    )


# =========================================================
# 8. Centralized API ECG loading
# =========================================================

def _load_record(record_id: str):
    """Load one exact recording through the shared ECG contract.

    Returns
    -------
    signals_mv:
        Unfiltered ECG with shape (samples, 12), in mV.

    fs:
        Sampling frequency in Hz.

    lead_names:
        Standardized ECG lead names.

    Raises
    ------
    HTTPException:
        400 for malformed record identifiers.
        404 for missing recordings.
        422 for invalid ECG waveform metadata or samples.
    """

    # Resolve the exact requested ECG.
    record_path = _find_record_path(record_id)

    try:
        # The shared loader handles:
        # - WFDB waveform reading
        # - Sampling frequency validation
        # - Channel and lead-name validation
        # - Physical voltage unit conversion
        # - Reordering into canonical 12-lead order
        signals_mv, fs, lead_names = load_standardized_ecg(
            record_path
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid ECG input: {exc}",
        ) from exc

    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Unable to read the requested ECG waveform.",
        ) from exc

    return signals_mv, fs, lead_names


# =========================================================
# 9. Waveform streaming endpoint
# =========================================================

@app.get("/api/records/{record_id}/signals")
def get_record_signals(record_id: str):
    """Return standardized raw and filtered ECG waveforms."""

    # -----------------------------------------------------
    # A. Load and standardize the requested ECG.
    # -----------------------------------------------------
    signals_mv, fs, lead_names = _load_record(
        record_id
    )

    # -----------------------------------------------------
    # B. Create the time axis.
    # -----------------------------------------------------
    time_sec = [
        round(i / fs, 3)
        for i in range(len(signals_mv))
    ]

    # -----------------------------------------------------
    # C. Filter standardized signals.
    # -----------------------------------------------------
    # signals_mv is already in millivolts and the
    # canonical lead order.
    #
    # Do not convert the units a second time.
    cleaned = filter_ecg(
        signals_mv,
        fs=fs,
    )

    # -----------------------------------------------------
    # D. Convert NumPy arrays into JSON-compatible data.
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

    # Preserve the existing API response structure.
    return {
        "record_id": record_id,
        "sampling_rate": fs,
        "time_seconds": time_sec,
        "lead_names": lead_names,
        "raw_signals": raw_by_lead,
        "filtered_signals": filtered_by_lead,
    }


# =========================================================
# 10. Research ensemble evaluation endpoint
# =========================================================

@app.post("/api/triage/evaluate")
def evaluate_record(req: EvaluationRequest):
    """Run the legacy V1 ensemble on a validated ECG."""

    # -----------------------------------------------------
    # A. Load the requested ECG.
    # -----------------------------------------------------
    signals_mv, fs, lead_names = _load_record(
        req.record_id
    )

    # -----------------------------------------------------
    # B. Run the existing legacy V1 ensemble.
    # -----------------------------------------------------
    # The loader already converted physical units to mV.
    #
    # Passing units="mV" is essential:
    # supplying the original WFDB units here would
    # incorrectly convert the waveform twice.
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
    # C. Handle rejected ECG recordings.
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
    # D. Run the existing research triage rules.
    # -----------------------------------------------------
    triage_res = triage_engine.evaluate(
        ensemble_res
    )

    # -----------------------------------------------------
    # E. Generate the existing report.
    # -----------------------------------------------------
    report_text = generate_clinical_report(
        req.record_id,
        ensemble_res,
        triage_res,
    )

    # -----------------------------------------------------
    # F. Compute contiguous ST lead pairs.
    # -----------------------------------------------------
    st_by_lead = ensemble_res["st_by_lead"]

    stemi_pairs = [
        pair._asdict()
        for pair in find_stemi_pairs(
            st_by_lead,
            req.age,
            req.sex,
        )
    ]

    # -----------------------------------------------------
    # G. Build per-lead research measurement table.
    # -----------------------------------------------------
    lead_table = []

    for lead in lead_names:

        threshold, basis = cut_point_mv(
            lead,
            req.age,
            req.sex,
        )

        # Missing values are not equivalent to a true
        # zero-millivolt measurement.
        reading = st_by_lead.get(lead.upper())

        lead_table.append({
            "lead": lead,
            "st_mv": reading,
            "cut_point_mv": threshold,
            "basis": basis,
            "cleared": (
                None
                if reading is None
                else reading >= threshold
            ),
        })

    # -----------------------------------------------------
    # H. Preserve the existing response structure.
    # -----------------------------------------------------
    return {
        "status": "success",
        "record_id": req.record_id,
        "triage": triage_res,
        "ensemble": ensemble_res,
        "contiguous_stemi_pairs": stemi_pairs,
        "lead_st_table": lead_table,
        "sbar_report": report_text,
    }


# =========================================================
# 11. Plain-text research report endpoint
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
    """Return the formatted research report."""

    req = EvaluationRequest(
        record_id=record_id,
        age=age,
        sex=sex,
    )

    # Reuse the same validated evaluation pipeline.
    evaluation = evaluate_record(req)

    if evaluation.get("status") == "rejected":
        raise HTTPException(
            status_code=400,
            detail=evaluation.get("reason"),
        )

    return evaluation["sbar_report"]
