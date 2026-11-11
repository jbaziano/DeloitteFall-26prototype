"""
Wire-fraud scoring API (FastAPI).

  GET  /health          service status and model version
  GET  /model           model version and test-set results (aggregates only)
  POST /score           score one wire -> risk, lane, top 3 reasons, next steps
  POST /feedback        save an analyst decision as a new label (feeds the next retrain)
  POST /retrain         retrain on simulated data + analyst labels, then hot-reload the model

There is deliberately no endpoint that returns wire data, customer data, or files:
the API scores and learns, it does not export.

Run:  uvicorn api.main:app --reload        (docs at http://localhost:8000/docs)

Environment:
  API_KEY           if set, /feedback and /retrain require header  X-API-Key: <value>
  ALLOWED_ORIGINS   comma-separated CORS origins for the front end (default "*")
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml import scoring
from ml.features import COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
AUDIT_LOG = ARTIFACTS / "audit_log.jsonl"
FEEDBACK_CSV = DATA_DIR / "feedback.csv"

app = FastAPI(title="Meridian wire fraud scoring", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])

_lock = Lock()
_state: dict = {}


def weights() -> dict:
    if "weights" not in _state:
        _state["weights"] = scoring.load_weights(ARTIFACTS / "model_weights.json")
    return _state["weights"]


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


class Thresholds(BaseModel):
    release: float = Field(0.05, ge=0, le=1)
    hold: float = Field(0.40, ge=0, le=1)


class ScoreRequest(BaseModel):
    wire: dict = Field(..., description="Signals for one wire; see ml/scoring.py for keys")
    thresholds: Optional[Thresholds] = None
    override: Optional[float] = Field(None, ge=0, le=1, description="Force a score (demos and testing only)")
    caseId: Optional[str] = None


class FeedbackRequest(BaseModel):
    caseId: str
    wire: dict
    label: Literal["fraud", "legitimate"]
    employeeId: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modelVersion": weights()["version"]}


@app.get("/model")
def model_info() -> dict:
    W = weights()
    return {"version": W["version"], "thresholds": W["thresholds"], "testMetrics": W["metrics"],
            "features": [f["label"] for f in W["features"]]}


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    th = req.thresholds.model_dump() if req.thresholds else None
    result = scoring.score(req.wire, thresholds=th, override=req.override, weights=weights())
    # Audit trail: decision metadata only (no names, accounts or amounts).
    entry = {"at": result["scoredAt"], "caseId": req.caseId, "modelVersion": result["modelVersion"],
             "score": result["score"], "lane": result["lane"], "scoreLane": result["scoreLane"],
             "policy": result["policy"], "overridden": result["overridden"],
             "reasons": [r["key"] for r in result["reasons"]]}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with _lock, AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return result


@app.post("/feedback", dependencies=[Depends(require_key)])
def feedback(req: FeedbackRequest) -> dict:
    raw = scoring.raw_features(req.wire)
    row = dict(zip(COLUMNS, raw)) | {"is_fraud": int(req.label == "fraud"), "case_id": req.caseId,
                                     "employee_id": req.employeeId,
                                     "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        pd.DataFrame([row]).to_csv(FEEDBACK_CSV, mode="a", header=not FEEDBACK_CSV.exists(), index=False)
        total = sum(1 for _ in FEEDBACK_CSV.open()) - 1
    return {"saved": True, "analystLabels": total}


@app.post("/retrain", dependencies=[Depends(require_key)])
def retrain() -> dict:
    from ml.train import train
    with _lock:
        metrics = train(data_dir=DATA_DIR, artifacts=ARTIFACTS, frontend=None, quiet=True)
        _state.pop("weights", None)
    t = metrics["test"]["semi_supervised"]
    return {"modelVersion": weights()["version"], "analystLabels": metrics["data"]["analyst_labels"],
            "auc": round(t["auc"], 4), "fraudCaught": round(t["fraud_caught"], 4),
            "legitChallenged": round(t["legit_challenged"], 4)}
