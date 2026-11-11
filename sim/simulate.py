"""
Simulate Meridian customers, beneficiaries, and wire transfers.

Everything here is synthetic. Each wire's features are computed from the
customer's own simulated history (their usual amounts, devices, payees and
hours) and from the payment network (who pays whom), the same way they
would be computed from real bank data in production.

Run:  python -m sim.simulate            -> writes data/customers.csv and data/wires.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FRAUD_TYPES = (
    "account_takeover",            # attacker logs in as the customer (new device, SIM swap)
    "business_email_compromise",   # real customer pays a known vendor whose bank details were swapped
    "impersonation_scam",          # customer coached by a deepfake voice call, often with remote access
    "synthetic_identity_mule",     # money sent to a fake-identity mule account that receives from many victims
)
FRAUD_MIX = (0.30, 0.25, 0.25, 0.20)
START = datetime(2026, 5, 1)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _clip(x, lo, hi):
    return float(min(hi, max(lo, x)))


def simulate(n_customers: int = 1500, days: int = 180, fraud_rate: float = 0.03,
             seed: int = 20261103) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (customers, wires) DataFrames."""
    rng = np.random.default_rng(seed)

    # ---- customers --------------------------------------------------------
    business = rng.random(n_customers) < 0.6
    customers = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n_customers)],
        "segment": np.where(business, "small_business", "retail"),
        "profile_usual_amount": np.where(business, rng.lognormal(np.log(9000), 0.6, n_customers),
                                         rng.lognormal(np.log(2500), 0.6, n_customers)).round(2),
        "debt_to_income": np.clip(rng.normal(0.30, 0.14, n_customers), 0, 1).round(3),
        "hours_start": rng.integers(7, 10, n_customers),
        "hours_end": rng.integers(16, 20, n_customers),
        "wires_per_month": np.where(business, rng.gamma(3, 2.0, n_customers), rng.gamma(2, 0.5, n_customers)).round(2),
    })
    cust = customers.to_dict("records")

    # ---- legitimate beneficiaries -----------------------------------------
    n_ben = 5000
    ben_open = -rng.lognormal(6.8, 0.9, n_ben)          # days before START the account was opened
    young = rng.random(n_ben) < 0.10                     # new businesses and contractors: legit but young accounts
    ben_open[young] = -rng.lognormal(3.3, 1.0, young.sum())
    young_ids = np.flatnonzero(young)
    hub = rng.random(n_ben) < 0.03                      # title companies, payroll processors, big suppliers
    hub_ids = np.flatnonzero(hub)
    ben_weight = np.where(hub, 30.0, 1.0)
    ben_weight /= ben_weight.sum()
    ben_change_day = np.where(rng.random(n_ben) < 0.03, rng.uniform(0, days, n_ben), np.nan)
    ben_intl = rng.random(n_ben) < 0.05
    regular = [list(rng.choice(n_ben, size=int(rng.integers(3, 9)), replace=False, p=ben_weight))
               for _ in range(n_customers)]

    # ---- mule accounts (created as fraud happens) -------------------------
    mules: list[dict] = []

    def active_mule(t: float, busy: bool = False) -> int:
        live = [i for i, m in enumerate(mules) if m["first_use"] is None or t - m["first_use"] < 30]
        if live and rng.random() < (0.85 if busy else 0.6):
            if busy:  # synthetic-identity mules concentrate many victims on a few accounts
                uses = np.array([mules[i]["uses"] + 1 for i in live], float)
                return int(rng.choice(live, p=uses / uses.sum()))
            return int(rng.choice(live))
        seasoned = rng.random() < 0.50                   # bought or rented older accounts
        mules.append({"open": t - (rng.uniform(90, 900) if seasoned else rng.uniform(2, 60)),
                      "on_list": bool(rng.random() < 0.30),
                      "intl": bool(rng.random() < 0.25), "first_use": None, "uses": 0})
        return len(mules) - 1

    # ---- wire events -------------------------------------------------------
    events = []
    for c, cu in enumerate(cust):
        n = rng.poisson(cu["wires_per_month"] * days / 30)
        events += [(float(t), c) for t in rng.uniform(0, days, n)]
    events.sort()

    amounts = defaultdict(list)          # legit wire history per customer
    paid = {c: {f"B{b}" for b in regular[c]} for c in range(n_customers)}   # payees paid before the window
    devices = {c: {f"D{c}-{k}" for k in range(int(rng.integers(1, 3)))} for c in range(n_customers)}
    inflow = defaultdict(deque)          # beneficiary account -> deque[(day, customer)]
    off_hours = lambda h0, h1: int(rng.choice([*range(0, h0), *range(h1, 24)]))

    rows = []
    for wid, (t, c) in enumerate(events):
        cu = cust[c]
        h0, h1 = int(cu["hours_start"]), int(cu["hours_end"])
        usual = float(np.median(amounts[c][-20:])) if len(amounts[c]) >= 3 else float(cu["profile_usual_amount"])
        fraud = rng.random() < fraud_rate
        ftype = ""
        sim_swap = rng.random() < 0.005
        remote = rng.random() < 0.01
        hour = int(rng.integers(h0, h1)) if rng.random() < 0.92 else int(rng.integers(0, 24))
        typing = _clip(rng.normal(0.14, 0.08), 0, 1)
        device = str(rng.choice(sorted(devices[c]))) if rng.random() < 0.9 else f"D{c}-n{wid}"
        bank_changed = False

        if not fraud:
            r = rng.random()
            if r < 0.02:                                    # one-off large legit payment, e.g. a home closing
                ben = int(rng.choice(hub_ids))
                amount = usual * rng.uniform(5, 15)
            elif r < 0.08:                                  # first payment to a new vendor or contractor
                ben = int(rng.choice(young_ids))
                amount = usual * rng.lognormal(0.5, 0.6)
                if rng.random() < 0.3:
                    hour = int(rng.integers(0, 24))
            else:
                ben = int(rng.choice(regular[c])) if rng.random() < 0.85 else int(rng.choice(n_ben, p=ben_weight))
                amount = usual * rng.lognormal(0, 0.5)
            payee_key = account = f"B{ben}"
            age = t - ben_open[ben]
            bank_changed = bool(not np.isnan(ben_change_day[ben]) and 0 <= t - ben_change_day[ben] < 30)
            intl = bool(ben_intl[ben])
            mule_hit = False
            if device not in devices[c]:
                typing = _clip(typing + 0.08, 0, 1)
        else:
            ftype = FRAUD_TYPES[int(rng.choice(4, p=FRAUD_MIX))]
            m = active_mule(t, busy=(ftype == "synthetic_identity_mule"))
            mu = mules[m]
            payee_key = account = f"M{m}"
            age = t - mu["open"]
            mule_hit = mu["on_list"]
            intl = mu["intl"]
            if ftype == "account_takeover":
                device = f"ATK{wid}" if rng.random() < 0.7 else device
                sim_swap = rng.random() < 0.35
                remote = rng.random() < 0.15
                typing = _clip(rng.normal(0.40, 0.18), 0, 1)
                amount = usual * rng.lognormal(1.0, 0.7)
                if rng.random() < 0.45:
                    hour = off_hours(h0, h1)
            elif ftype == "business_email_compromise":
                payee_key = f"B{int(rng.choice(regular[c]))}"   # the vendor name the customer knows...
                bank_changed = rng.random() < 0.65               # ...but its bank details now point to a mule
                # (35% of swaps go unseen, e.g. changed on an invoice rather than in online banking)
                amount = usual * rng.lognormal(0.2, 0.5)
            elif ftype == "impersonation_scam":
                remote = rng.random() < 0.35
                typing = _clip(rng.normal(0.24, 0.12), 0, 1)
                amount = usual * rng.lognormal(1.2, 0.7)
                if rng.random() < 0.2:
                    hour = off_hours(h0, h1)
            else:  # synthetic_identity_mule
                amount = usual * rng.lognormal(0.5, 0.6)
            mu["uses"] += 1
            mu["first_use"] = t if mu["first_use"] is None else mu["first_use"]

        q = inflow[account]                    # network feature: distinct senders into this account, last 30 days
        while q and t - q[0][0] > 30:
            q.popleft()
        senders = len({s for _, s in q} | {c})

        rows.append({
            "wire_id": f"W{wid:06d}",
            "timestamp": (START + timedelta(days=int(t), hours=hour, minutes=int(rng.integers(0, 60)))).isoformat(),
            "customer_id": cu["customer_id"],
            "segment": cu["segment"],
            "beneficiary_account": account,
            "amount": round(float(amount), 2),
            "channel": str(rng.choice(["online", "voice", "branch"], p=[0.90, 0.06, 0.04])),
            # ---- model features (computed from customer history and the payment network) ----
            "amount_vs_usual": round(float(amount) / usual, 4),
            "outside_hours": int(not (h0 <= hour < h1)),
            "debt_to_income": float(cu["debt_to_income"]),
            "new_device": int(device not in devices[c]),
            "typing_mismatch": round(typing, 4),
            "remote_access": int(remote),
            "sim_swap_7d": int(sim_swap),
            "mule_list_hit": int(mule_hit),
            "beneficiary_age_days": max(1, int(round(age))),
            "bank_details_changed_30d": int(bank_changed),
            "high_risk_destination": int(intl),
            "first_time_payee": int(payee_key not in paid[c]),
            "beneficiary_senders_30d": senders,
            # ---- labels ----
            "is_fraud": int(fraud),
            "fraud_type": ftype,
        })

        q.append((t, c))
        if not fraud:                         # confirmed fraud doesn't become part of "normal" history
            amounts[c].append(float(amount))
            paid[c].add(payee_key)
            devices[c].add(device)

    return customers, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--customers", type=int, default=1500)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--fraud-rate", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    a = ap.parse_args()
    customers, wires = simulate(a.customers, a.days, a.fraud_rate, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    customers.to_csv(a.out / "customers.csv", index=False)
    wires.to_csv(a.out / "wires.csv", index=False)
    print(f"{len(customers):,} customers, {len(wires):,} wires, {int(wires.is_fraud.sum()):,} fraud "
          f"({wires.is_fraud.mean():.1%}) -> {a.out}")
    print(wires[wires.is_fraud == 1].fraud_type.value_counts().to_string())


if __name__ == "__main__":
    main()
