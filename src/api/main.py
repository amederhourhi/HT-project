"""
CardioSense - Clinician Decision-Support REST API
FastAPI backend delivering record catalogues, 12-lead waveform time-series,
and evidence-grounded clinical triage evaluations.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel
import wfdb

from src.dsp.filters import filter_ecg
from src.triage.leads import cut_point_mv, find_stemi_pairs
from src.triage.report_generator import generate_clinical_report
from src.triage.risk_fusion import CardioSenseEnsemble
from src.triage.triage_rules import ClinicalTriageEngine

# ---------------------------------------------------------
# Block 1: App Setup & Cross-Origin (CORS) Configuration
# What it does: Initializes FastAPI and enables CORS for local frontend dev.
# How it does it: Instantiates FastAPI and applies CORSMiddleware.
# ---------------------------------------------------------
app = FastAPI(
    title="CardioSense API",
    description="Clinical Decision-Support System for Myocardial Infarction & Ischemia",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_root = Path(__file__).resolve().parents[2]
samples_dir = project_root / "data" / "samples"
cohort_dir = project_root / "data" / "cohort"
manifest_path = cohort_dir / "cohort_manifest.csv"

# Global pipeline singletons loaded once on server start
ensemble_pipeline = CardioSenseEnsemble()
triage_engine = ClinicalTriageEngine()


# ---------------------------------------------------------
# Block 2: Root Redirect & Health Check
# What it does: Automatically routes visitors to interactive API docs.
# How it does it: Returns an HTTP 307 redirect to /docs.
# ---------------------------------------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "CardioSense API", "version": "0.1.0"}


# ---------------------------------------------------------
# Block 3: Data Schemas
# What it does: Defines structured inputs for evaluation requests.
# How it does it: Uses Pydantic BaseModel for type validation.
# ---------------------------------------------------------
class EvaluationRequest(BaseModel):
    record_id: str
    age: Optional[float] = None
    sex: Optional[str] = None


# ---------------------------------------------------------
# Block 4: Record Catalogue Discovery
# What it does: Lists all available sample and cohort ECGs on disk.
# How it does it: Scans data/samples and data/cohort for .hea files.
# ---------------------------------------------------------
@app.get("/api/records")
def list_records():
    records = []

    # 1. Sample records
    for hea_path in sorted(samples_dir.glob("**/*.hea")):
        base_name = hea_path.stem
        records.append({
            "id": base_name,
            "category": "sample",
            "path": str(hea_path.with_suffix("")),
            "description": f"Sample record ({base_name})",
        })

    # 2. Cohort records (if manifest exists)
    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            wf_path = cohort_dir / f"{row['filename_lr']}.hea"
            if wf_path.exists():
                records.append({
                    "id": str(row["ecg_id"]),
                    "category": "cohort",
                    "path": str(cohort_dir / row["filename_lr"]),
                    "age": row.get("age"),
                    "sex": "Male" if row.get("sex") == 0.0 else "Female",
                    "diagnosis": "infarction" if row.get("target_label") == 1 else "normal",
                    "description": f"Patient #{int(row['ecg_id']):05d}",
                })

    return {"count": len(records), "records": records}


# ---------------------------------------------------------
# Block 5: Waveform Time-Series Stream
# What it does: Streams 12-lead raw and filtered voltage arrays.
# How it does it: Loads via wfdb.rdsamp(), applies zero-phase filter,
#                 and returns millivolt arrays indexed by lead name.
# ---------------------------------------------------------
@app.get("/api/records/{record_id}/signals")
def get_record_signals(record_id: str):
    record_path = _find_record_path(record_id)
    raw_signals, fields = wfdb.rdsamp(str(record_path))
    fs = float(fields["fs"])
    lead_names = fields["sig_name"]
    time_sec = [round(i / fs, 3) for i in range(len(raw_signals))]

    cleaned_signals = filter_ecg(raw_signals, fs=fs)

    leads_raw = {lead: [round(float(v), 3) for v in raw_signals[:, i]] for i, lead in enumerate(lead_names)}
    leads_cleaned = {lead: [round(float(v), 3) for v in cleaned_signals[:, i]] for i, lead in enumerate(lead_names)}

    return {
        "record_id": record_id,
        "sampling_rate": fs,
        "time_seconds": time_sec,
        "lead_names": lead_names,
        "raw_signals": leads_raw,
        "filtered_signals": leads_cleaned,
    }


# ---------------------------------------------------------
# Block 6: Full Clinical Triage Evaluation
# What it does: Executes dual-engine inference and Fourth UDMI guardrails.
# How it does it: Computes ML probabilities, evaluates contiguous lead pairs,
#                 and formats the auditable SBAR clinical note.
# ---------------------------------------------------------
@app.post("/api/triage/evaluate")
def evaluate_record(req: EvaluationRequest):
    record_path = _find_record_path(req.record_id)
    raw_signals, fields = wfdb.rdsamp(str(record_path))
    fs = float(fields["fs"])
    lead_names = fields["sig_name"]

    ensemble_res = ensemble_pipeline.predict(
        raw_signals,
        fs=fs,
        lead_names=lead_names,
        age=req.age,
        sex=req.sex,
    )

    if ensemble_res.get("status") == "rejected":
        return {
            "status": "rejected",
            "reason": ensemble_res.get("reason"),
            "sqi_details": ensemble_res.get("sqi_details"),
        }

    triage_res = triage_engine.evaluate(ensemble_res)
    report_text = generate_clinical_report(req.record_id, ensemble_res, triage_res)

    # Compute contiguous pairs grid for UI display
    st_by_lead = ensemble_res["st_by_lead"]
    stemi_pairs = [p._asdict() for p in find_stemi_pairs(st_by_lead, req.age, req.sex)]

    # Compute per-lead cut-point table
    lead_table = []
    for lead in lead_names:
        thresh, basis = cut_point_mv(lead, req.age, req.sex)
        val = st_by_lead.get(lead.upper(), 0.0)
        lead_table.append({
            "lead": lead,
            "st_mv": val,
            "cut_point_mv": thresh,
            "basis": basis,
            "cleared": val >= thresh,
        })

    return {
        "status": "success",
        "record_id": req.record_id,
        "triage": triage_res,
        "ensemble": ensemble_res,
        "contiguous_stemi_pairs": stemi_pairs,
        "lead_st_table": lead_table,
        "sbar_report": report_text,
    }


# ---------------------------------------------------------
# Block 7: Plain-Text SBAR Report Download
# What it does: Returns the clean formatted clinical text note.
# How it does it: Calls evaluate_record and outputs PlainTextResponse.
# ---------------------------------------------------------
@app.get("/api/records/{record_id}/report", response_class=PlainTextResponse)
def get_record_report(record_id: str, age: Optional[float] = None, sex: Optional[str] = None):
    req = EvaluationRequest(record_id=record_id, age=age, sex=sex)
    eval_res = evaluate_record(req)
    if eval_res.get("status") == "rejected":
        raise HTTPException(status_code=400, detail=eval_res.get("reason"))
    return eval_res["sbar_report"]


# ---------------------------------------------------------
# Block 8: Record Path Resolver
# What it does: Matches a requested record ID to its file path on disk.
# How it does it: Tries direct match, zero-padded integer, and cohort manifest.
# ---------------------------------------------------------
def _find_record_path(record_id: str) -> Path:
    # 1. Direct glob matching in data/samples
    for hea in samples_dir.glob(f"**/{record_id}*.hea"):
        return hea.with_suffix("")

    # 2. Numeric zero-padded matching (e.g., '1' -> '00001_lr')
    if record_id.isdigit():
        padded = f"{int(record_id):05d}"
        for hea in samples_dir.glob(f"**/{padded}*.hea"):
            return hea.with_suffix("")

    # 3. Check data/cohort if manifest exists
    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)
        match = df[df["ecg_id"].astype(str) == str(record_id)]
        if not match.empty:
            rel = match.iloc[0]["filename_lr"]
            return cohort_dir / rel

    # 4. Fallback to first available sample record
    samples = list(samples_dir.glob("**/*.hea"))
    if samples:
        return samples[0].with_suffix("")

    raise HTTPException(status_code=404, detail=f"ECG Record '{record_id}' not found on disk.")