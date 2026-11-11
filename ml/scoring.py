"""
Score a wire with the trained weights: 0-1 risk, top 3 reasons, lane, and next steps.

Uses artifacts/model_weights.json (written by `python -m ml.train`), the same weights
the browser engine uses, so the API and the front end always give the same answer.

Input keys may be camelCase (browser/API) or snake_case (Python):
  amount + usualAmount  (or amountVsUsual / amount_vs_usual), outsideHours, debtToIncome,
  newDevice, typingMismatch, remoteAccess, simSwap7d, muleListHit, beneficiaryAgeDays,
  bankDetailsChanged30d, highRiskDestination, firstTimePayee, beneficiarySenders30d,
  channel ("online" | "mobile" | "branch" | "voice")

Policy: a wire requested by voice is ALWAYS routed to "confirm", whatever the score.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ml.features import COLUMNS, JS_KEYS, reason_text

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model_weights.json"
_SNAKE = dict(zip(JS_KEYS, COLUMNS))
_SNAKE.update({"amountVsUsual": "amount_vs_usual", "usualAmount": "usual_amount"})
LANE_NAMES = {"release": "Release", "confirm": "Confirm", "hold": "Hold"}


def load_weights(path: Path = WEIGHTS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m ml.train` first.")
    return json.loads(path.read_text())


def _norm(tx: dict) -> dict:
    return {_SNAKE.get(k, k): v for k, v in (tx or {}).items()}


def _num(v, default=0.0) -> float:
    try:
        return float(v) if v is not None and v != "" else default
    except (TypeError, ValueError):
        return default


def raw_features(tx: dict) -> list[float]:
    t = _norm(tx)
    if "amount_vs_usual" in t:
        ratio = _num(t["amount_vs_usual"], 1.0)
    else:
        ratio = max(_num(t.get("amount"), 1), 1) / max(_num(t.get("usual_amount"), 1), 1)
    vals = {
        "amount_vs_usual": max(ratio, 0.01),
        "debt_to_income": min(1.0, max(0.0, _num(t.get("debt_to_income")))),
        "typing_mismatch": min(1.0, max(0.0, _num(t.get("typing_mismatch")))),
        "beneficiary_age_days": max(1, round(_num(t.get("beneficiary_age_days"), 1))),
        "beneficiary_senders_30d": max(1, round(_num(t.get("beneficiary_senders_30d"), 1))),
    }
    return [vals[c] if c in vals else (1.0 if t.get(c) else 0.0) for c in COLUMNS]


def is_voice(tx: dict) -> bool:
    t = _norm(tx)
    return str(t.get("channel") or "").lower() == "voice" or t.get("voiceRequest") is True


def decide(score: float, thresholds: dict) -> str:
    return "release" if score < thresholds["release"] else "confirm" if score < thresholds["hold"] else "hold"


def next_steps(lane: str, tx: dict) -> list[dict]:
    t = _norm(tx)
    if lane == "confirm" and is_voice(tx):
        return [
            {"title": "Passkey on the customer\u2019s verified device",
             "detail": "Voice requests are always confirmed out of band; the voice is never proof of identity."},
            {"title": "Customer confirms the payee name in the app",
             "detail": "Typed on the verified device, not spoken on the call, so a cloned voice can\u2019t pass it."},
            {"title": "Release, or move to hold if either step fails", "detail": "Outcome is logged as a label."},
        ]
    if lane == "release":
        return [{"title": "Release automatically", "detail": "No extra step for the customer. Score and reasons are logged."}]
    if lane == "confirm":
        return [
            {"title": "Passkey on the customer\u2019s verified device" if t.get("new_device") else "Passkey on the current device",
             "detail": "Approval is pushed to the registered device when the session is on a new device or IP."},
            {"title": "Customer confirms the payee name", "detail": "Matched to the beneficiary account name before release."},
            {"title": "Release, or move to hold if either step fails", "detail": "Outcome is logged as a label."},
        ]
    return [
        {"title": "Hold the wire and step up identity (Okta)", "detail": "Phishing-resistant re-verification."},
        {"title": "Call the trusted contact on file", "detail": "Only a contact unchanged for at least ~2 weeks is used."},
        {"title": "Analyst decides; otherwise blocked", "detail": "If identity can\u2019t be confirmed, the customer sends in person."},
    ]


def score(tx: dict, thresholds: dict | None = None, override: float | None = None,
          weights: dict | None = None) -> dict:
    W = weights or load_weights()
    th = {**W["thresholds"], **(thresholds or {})}
    raw = raw_features(tx)
    contrib = []
    for j, (x, f) in enumerate(zip(raw, W["features"])):
        v = math.log(x) if f["transform"] == "log" else -math.log(x + 1) if f["transform"] == "neglog1p" else x
        contrib.append(W["w"][j] * (v - W["mu"][j]) / W["sd"][j])
    model_score = 1 / (1 + math.exp(-(W["b"] + W["shift"] + sum(contrib))))
    overridden = isinstance(override, (int, float)) and math.isfinite(override)
    final = min(1.0, max(0.0, float(override))) if overridden else model_score
    score_lane = decide(final, th)
    voice = is_voice(tx)
    lane = "confirm" if voice else score_lane
    reasons = sorted(({"key": W["features"][j]["jsKey"], "label": W["features"][j]["label"],
                       "text": reason_text(COLUMNS[j], raw[j]), "contribution": round(c, 3)}
                      for j, c in enumerate(contrib) if round(c, 3) > 0),
                     key=lambda r: -r["contribution"])[:3]
    return {
        "score": round(final, 4), "modelScore": round(model_score, 4), "overridden": overridden,
        "lane": lane, "scoreLane": score_lane, "policy": "voice_always_confirm" if voice else None,
        "channel": "voice" if voice else (_norm(tx).get("channel") or "online"),
        "reasons": reasons, "steps": next_steps(lane, tx),
        "modelVersion": W["version"], "scoredAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
