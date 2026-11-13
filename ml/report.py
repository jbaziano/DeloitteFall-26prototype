"""
Build the data behind the "Model & data" page (frontend/model.html).

Reads the simulated data and the trained model, then writes frontend/model_report.js
with summaries only: counts, rates, chart points, and six example wires.

Run after training:  python -m ml.train && python -m ml.report
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from ml import scoring
from ml.features import COLUMNS, FEATURES, LABELS, transform
from ml.train import THRESHOLDS, logit, rules_flag, split, DEPLOY_FRAUD_RATE

ROOT = Path(__file__).resolve().parents[1]
BINARY = {"outside_hours", "new_device", "remote_access", "sim_swap_7d", "mule_list_hit",
          "bank_details_changed_30d", "high_risk_destination", "first_time_payee"}
TYPE_NAMES = {"account_takeover": "Account takeover", "business_email_compromise": "Business email compromise",
              "impersonation_scam": "Impersonation scam (voice)", "synthetic_identity_mule": "Synthetic-identity mule"}


def _lane(p: float) -> str:
    return "release" if p < THRESHOLDS["release"] else "confirm" if p < THRESHOLDS["hold"] else "hold"


def _roc(y, p, n=60):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(n, len(fpr))).round().astype(int))
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]


def build(data_dir: Path = ROOT / "data", artifacts: Path = ROOT / "artifacts",
          frontend: Path | None = ROOT / "frontend") -> dict:
    wires = pd.read_csv(data_dir / "wires.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    metrics = json.loads((artifacts / "metrics.json").read_text())
    weights = json.loads((artifacts / "model_weights.json").read_text())
    bundle = joblib.load(artifacts / "model.joblib")
    scaler, model, shift = bundle["scaler"], bundle["model"], bundle["shift"]

    labeled, unlabeled, test = split(wires)
    y = test.is_fraud.to_numpy(int)
    p = 1 / (1 + np.exp(-(model.decision_function(scaler.transform(transform(test))) + shift)))
    sup = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(transform(labeled)), labeled.is_fraud)
    p_sup = 1 / (1 + np.exp(-(sup.decision_function(scaler.transform(transform(test)))
                               + logit(DEPLOY_FRAUD_RATE) - logit(labeled.is_fraud.mean()))))
    rules = rules_flag(test)

    # ---- data ----
    fraud = wires[wires.is_fraud == 1]
    data = {
        "customers": int(len(customers)),
        "segments": customers.segment.value_counts().to_dict(),
        "wires": int(len(wires)),
        "fraud": int(len(fraud)),
        "fraudRate": round(float(wires.is_fraud.mean()), 4),
        "firstDay": wires.timestamp.min()[:10], "lastDay": wires.timestamp.max()[:10],
        "fraudTypes": [{"name": TYPE_NAMES[k], "count": int(v)} for k, v in fraud.fraud_type.value_counts().items()],
        "medianAmount": round(float(wires.amount.median()), 2),
    }

    # ---- features: weight + how fraud and legit wires differ ----
    feats = []
    for j, (col, js, tf, group) in enumerate(FEATURES):
        legit_v, fraud_v = wires.loc[wires.is_fraud == 0, col], wires.loc[wires.is_fraud == 1, col]
        kind = "rate" if col in BINARY else "median"
        agg = (lambda s: float(s.mean())) if kind == "rate" else (lambda s: float(s.median()))
        feats.append({"label": LABELS[col], "group": group, "weight": round(float(weights["w"][j]), 3),
                      "kind": kind, "legit": round(agg(legit_v), 4), "fraud": round(agg(fraud_v), 4)})

    # ---- score distribution: share of each class per score bin ----
    edges = [0, .01, .02, .05, .1, .2, .4, .6, .8, .9, .95, 1.0001]
    hist = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        hist.append({"lo": lo, "hi": min(hi, 1), "legit": int((m & (y == 0)).sum()), "fraud": int((m & (y == 1)).sum())})

    # ---- catch rate by fraud type on the test set ----
    by_type = []
    for k, name in TYPE_NAMES.items():
        m = (test.fraud_type == k).to_numpy()
        if m.sum():
            by_type.append({"name": name, "n": int(m.sum()), "model": round(float((p[m] >= THRESHOLDS["release"]).mean()), 4),
                            "rules": round(float(rules[m].mean()), 4)})

    # ---- example wires: right and wrong ----
    t = test.reset_index(drop=True).assign(p=p)
    picks = []
    for k in ("account_takeover", "business_email_compromise", "synthetic_identity_mule"):
        sub = t[(t.fraud_type == k) & (t.p >= THRESHOLDS["hold"])]
        if len(sub):                                   # a typical catch, not the most extreme one
            picks.append(((sub.p - sub.p.median()).abs().idxmin(), "Caught"))
    legit = t[t.is_fraud == 0]
    released = legit[(legit.p < THRESHOLDS["release"]) & legit.beneficiary_age_days.between(90, 3650)]
    typical = (released.p - released.p.median()).abs().idxmin() if len(released) else legit.p.idxmin()
    picks += [(typical, "Correctly released"), (legit.p.idxmax(), "False alarm")]
    fr = t[t.is_fraud == 1]
    picks.append((fr.p.idxmin(), "Missed" if fr.p.min() < THRESHOLDS["release"] else "Lowest-scoring fraud"))
    examples = []
    for i, verdict in picks:
        r = t.loc[i]
        s = scoring.score(r[COLUMNS].to_dict(), weights=weights)
        examples.append({"verdict": verdict, "truth": TYPE_NAMES.get(r.fraud_type, "Legitimate") if r.is_fraud else "Legitimate",
                         "segment": "Business" if r.segment == "small_business" else "Retail",
                         "amount": round(float(r.amount), 2), "amountVsUsual": round(float(r.amount_vs_usual), 1),
                         "beneficiaryAgeDays": int(r.beneficiary_age_days), "firstTimePayee": bool(r.first_time_payee),
                         "newDevice": bool(r.new_device), "score": round(float(r.p), 4), "lane": _lane(float(r.p)),
                         "reasons": [x["text"] for x in s["reasons"]] if _lane(float(r.p)) != "release" else []})

    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": weights["version"], "thresholds": THRESHOLDS,
        "data": data, "split": metrics["data"], "selfTraining": metrics["self_training"],
        "test": {k: {kk: vv for kk, vv in v.items() if kk != "lanes"} for k, v in metrics["test"].items()},
        "lanes": metrics["test"]["semi_supervised"]["lanes"],
        "features": feats, "histogram": hist, "byType": by_type, "examples": examples,
        "roc": {"semi": _roc(y, p), "supervised": _roc(y, p_sup),
                "rules": [round(float(rules[y == 0].mean()), 4), round(float(rules[y == 1].mean()), 4)]},
    }
    if frontend is not None:
        (frontend / "model_report.js").write_text(
            "// Generated by `python -m ml.report`. Summaries of synthetic data only.\n"
            f"window.MODEL_REPORT = {json.dumps(report, indent=1)};\n")
    return report


if __name__ == "__main__":
    r = build()
    print(f"Wrote frontend/model_report.js for model {r['version']} "
          f"({r['data']['wires']:,} wires, {len(r['examples'])} examples)")
