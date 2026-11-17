"""
Build the data behind the "User Data and Patterns" page (frontend/customer.html).

Picks one simulated small-business customer, presents them as the demo customer
(Prairie Ridge Farms), names their payees, and scores every wire in their history
with the trained model. Writes frontend/customer_report.js.

Run after training:  python -m ml.train && python -m ml.customer_report
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml import scoring
from ml.features import COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DEMO_USUAL = 9000          # the demo's "usual wire" for Prairie Ridge Farms
TYPE_NAMES = {"account_takeover": "Account takeover", "business_email_compromise": "Business email compromise",
              "impersonation_scam": "Impersonation scam (voice)", "synthetic_identity_mule": "Synthetic-identity mule"}

# Known payees, most frequent first. The first two match the demo's payee list.
VENDORS = ["Midstate Steel Supply", "Robin Theodore Coleman", "Great Lakes Packaging", "Summit Payroll Services",
           "Atlas Freight Partners", "Bluestem Insurance", "Keystone Office Systems", "Corridor Building Supply",
           "Westfield Property Mgmt", "Hoosier Seed & Grain", "Wabash Equipment Rental", "Tri-County Fuel Co-op",
           "Lakeland Veterinary Group", "Crossroads Equipment Leasing", "Prairie State Agronomy", "Union Mills Feed",
           "Brookston Irrigation", "White River Electric", "Fairfield Accounting LLP", "Monroe County Treasurer"]
MULES = ["Orion Trade Holdings", "Apex Global Sourcing", "Silverline Ventures", "Northgate Imports Ltd",
         "Harborview Logistics LLC", "Pinnacle Asset Partners"]
BEC_NAME = "Granite Point Consulting"   # a vendor whose bank details were swapped (matches the demo scenario)


def pick_customer(wires: pd.DataFrame) -> str:
    sb = wires[wires.segment == "small_business"]
    g = sb.groupby("customer_id").agg(n=("wire_id", "size"), amt=("amount", "median"),
                                      types=("fraud_type", lambda s: s.dropna().nunique()))
    g = g[g.n.between(50, 90) & (g.types >= 3)]
    if g.empty:                                   # fall back to the busiest business customer
        g = sb.groupby("customer_id").agg(n=("wire_id", "size"), amt=("amount", "median"))
        return str(g.n.idxmax())
    return str((g.amt - DEMO_USUAL).abs().idxmin())


def _lane(p: float, th: dict) -> str:
    return "release" if p < th["release"] else "confirm" if p < th["hold"] else "hold"


def build(data_dir: Path = ROOT / "data", artifacts: Path = ROOT / "artifacts",
          frontend: Path | None = ROOT / "frontend") -> dict:
    wires = pd.read_csv(data_dir / "wires.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    weights = scoring.load_weights(artifacts / "model_weights.json")
    th = weights["thresholds"]

    cid = pick_customer(wires)
    cust = customers[customers.customer_id == cid].iloc[0]
    hist = wires[wires.customer_id == cid].sort_values("timestamp").reset_index(drop=True)
    scale = DEMO_USUAL / float(hist.loc[hist.is_fraud == 0, "amount"].median())

    # name payees: legit accounts by frequency, fraud accounts by type
    legit_freq = hist[hist.is_fraud == 0].beneficiary_account.value_counts().index.tolist()
    names = {acct: VENDORS[i % len(VENDORS)] for i, acct in enumerate(legit_freq)}
    mule_i = 0
    for acct, ftype in hist[hist.is_fraud == 1][["beneficiary_account", "fraud_type"]].itertuples(index=False):
        if acct in names:
            continue
        if ftype == "business_email_compromise":
            names[acct] = BEC_NAME
        else:
            names[acct] = MULES[mule_i % len(MULES)]
            mule_i += 1

    tx = []
    for r in hist.itertuples(index=False):
        s = scoring.score({c: getattr(r, c) for c in COLUMNS} | {"channel": r.channel}, weights=weights)
        p = s["modelScore"]
        lane = s["lane"]
        if lane == "release":
            outcome = "Sent"
        elif lane == "confirm":
            outcome = "Blocked: failed confirmation" if r.is_fraud else "Sent after customer confirmed"
        else:
            outcome = "Blocked as fraud" if r.is_fraud else "Released by analyst"
        if r.is_fraud and lane == "release":
            outcome = "Sent: fraud reported later"
        tx.append({
            "id": r.wire_id, "time": r.timestamp, "payee": names[r.beneficiary_account],
            "amount": round(float(r.amount) * scale, 2), "ratio": round(float(r.amount_vs_usual), 2),
            "channel": r.channel, "score": round(p, 4), "lane": lane, "outcome": outcome,
            "fraud": TYPE_NAMES.get(r.fraud_type, "") if r.is_fraud else "",
            "newDevice": bool(r.new_device), "offHours": bool(r.outside_hours), "firstPayee": bool(r.first_time_payee),
            "reasons": [x["text"] for x in s["reasons"]] if lane != "release" else [],
        })

    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelVersion": weights["version"], "thresholds": th,
        "customer": {"name": "Prairie Ridge Farms", "id": "CIF-" + cid[1:], "segment": "Small business",
                     "since": "2014", "accounts": ["Business Operating ••4821", "Payroll ••7730"],
                     "hours": [int(cust.hours_start), int(cust.hours_end)],
                     "debtToIncome": round(float(cust.debt_to_income), 3), "usualWire": DEMO_USUAL},
        "transactions": tx,
    }
    if frontend is not None:
        (frontend / "customer_report.js").write_text(
            "// Generated by `python -m ml.customer_report`. Synthetic customer history, scored by the trained model.\n"
            f"window.CUSTOMER_REPORT = {json.dumps(report, indent=1)};\n")
    return report


if __name__ == "__main__":
    r = build()
    t = r["transactions"]
    print(f"Wrote frontend/customer_report.js: {r['customer']['name']} ({len(t)} wires, "
          f"{sum(1 for x in t if x['fraud'])} fraud) scored by model {r['modelVersion']}")
