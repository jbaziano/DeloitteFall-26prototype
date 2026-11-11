"""
Feature definitions shared by training (Python), the API, and the browser engine (JavaScript).

Each feature has:
  column  - name in data/wires.csv and in Python inputs
  js_key  - name used by the browser and the API (camelCase)
  tf      - transform applied before scaling, chosen so a higher value always means riskier
            ("id" = as is, "log" = natural log, "neglog1p" = -log(x + 1))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    # column                       js_key                   tf          group
    ("amount_vs_usual",           "amountRatio",            "log",      "Customer history"),
    ("outside_hours",             "outsideHours",           "id",       "Customer history"),
    ("debt_to_income",            "debtToIncome",           "id",       "Customer history"),
    ("new_device",                "newDevice",              "id",       "Device and behavior"),
    ("typing_mismatch",           "typingMismatch",         "id",       "Device and behavior"),
    ("remote_access",             "remoteAccess",           "id",       "Device and behavior"),
    ("sim_swap_7d",               "simSwap7d",              "id",       "External signals"),
    ("mule_list_hit",             "muleListHit",            "id",       "External signals"),
    ("beneficiary_age_days",      "beneficiaryAgeDays",     "neglog1p", "Beneficiary"),
    ("bank_details_changed_30d",  "bankDetailsChanged30d",  "id",       "Beneficiary"),
    ("high_risk_destination",     "highRiskDestination",    "id",       "Beneficiary"),
    ("first_time_payee",          "firstTimePayee",         "id",       "Payment network"),
    ("beneficiary_senders_30d",   "beneficiarySenders30d",  "log",      "Payment network"),
]
COLUMNS = [f[0] for f in FEATURES]
JS_KEYS = [f[1] for f in FEATURES]
TRANSFORMS = [f[2] for f in FEATURES]

LABELS = {
    "amount_vs_usual": "Amount vs. usual",
    "outside_hours": "Outside usual hours",
    "debt_to_income": "Debt-to-income",
    "new_device": "New device or IP",
    "typing_mismatch": "Typing-pattern mismatch",
    "remote_access": "Remote-access tool",
    "sim_swap_7d": "Recent SIM swap",
    "mule_list_hit": "Consortium mule list",
    "beneficiary_age_days": "Beneficiary account age",
    "bank_details_changed_30d": "Payee bank details changed",
    "high_risk_destination": "High-risk destination",
    "first_time_payee": "First wire to payee",
    "beneficiary_senders_30d": "Senders into beneficiary",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clip raw feature values into valid ranges."""
    out = df[COLUMNS].astype(float).copy()
    out["amount_vs_usual"] = out["amount_vs_usual"].clip(lower=0.01)
    out["debt_to_income"] = out["debt_to_income"].clip(0, 1)
    out["typing_mismatch"] = out["typing_mismatch"].clip(0, 1)
    out["beneficiary_age_days"] = out["beneficiary_age_days"].clip(lower=1).round()
    out["beneficiary_senders_30d"] = out["beneficiary_senders_30d"].clip(lower=1).round()
    for c in ("outside_hours", "new_device", "remote_access", "sim_swap_7d", "mule_list_hit",
              "bank_details_changed_30d", "high_risk_destination", "first_time_payee"):
        out[c] = (out[c] > 0).astype(float)
    return out


def transform(df: pd.DataFrame) -> np.ndarray:
    """Raw features -> model inputs (before scaling)."""
    raw = clean(df)
    cols = []
    for col, tf in zip(COLUMNS, TRANSFORMS):
        x = raw[col].to_numpy(float)
        cols.append(np.log(x) if tf == "log" else -np.log(x + 1) if tf == "neglog1p" else x)
    return np.column_stack(cols)


def reason_text(column: str, value: float) -> str:
    """Plain-English reason shown to fraud analysts (never to customers)."""
    v = value
    return {
        "amount_vs_usual": f"Amount is {v:.1f}\u00d7 this customer\u2019s usual wire",
        "outside_hours": "Sent outside this customer\u2019s usual hours",
        "debt_to_income": f"Debt-to-income is {round(v * 100)}%",
        "new_device": "New device or IP address for this customer",
        "typing_mismatch": "Typing rhythm doesn\u2019t match this customer",
        "remote_access": "Remote-access software running in session",
        "sim_swap_7d": "SIM swap on the customer\u2019s phone in the last 7 days",
        "mule_list_hit": "Beneficiary is on the consortium mule list",
        "beneficiary_age_days": f"Beneficiary account opened {int(v)} day{'' if int(v) == 1 else 's'} ago",
        "bank_details_changed_30d": "Payee\u2019s bank details changed in the last 30 days",
        "high_risk_destination": "High-risk or new destination country",
        "first_time_payee": "First wire to this payee",
        "beneficiary_senders_30d": f"{int(v)} different senders paid this beneficiary in 30 days",
    }[column]
