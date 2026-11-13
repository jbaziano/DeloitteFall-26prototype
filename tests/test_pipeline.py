"""End-to-end tests: simulation -> semi-supervised training -> scoring -> API (and browser parity)."""
import importlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml import scoring
from ml.features import COLUMNS, transform
from ml.train import split, train
from sim.simulate import simulate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    base = tmp_path_factory.mktemp("run")
    metrics = train(regenerate=True, data_dir=base / "data", artifacts=base / "artifacts", frontend=None, quiet=True)
    return base, metrics


def test_simulation_produces_features_and_labels():
    customers, wires = simulate(n_customers=200, days=60, seed=7)
    assert set(COLUMNS + ["is_fraud", "fraud_type", "customer_id"]).issubset(wires.columns)
    assert 0.01 < wires.is_fraud.mean() < 0.06
    assert set(wires.loc[wires.is_fraud == 1, "fraud_type"]) <= {
        "account_takeover", "business_email_compromise", "impersonation_scam", "synthetic_identity_mule"}
    assert wires.customer_id.isin(customers.customer_id).all()


def test_split_sizes_and_time_order(trained):
    base, _ = trained
    wires = pd.read_csv(base / "data" / "wires.csv")
    labeled, unlabeled, test = split(wires)
    assert (len(labeled), len(unlabeled), len(test)) == (1000, 9000, 3000)
    assert labeled.is_fraud.sum() == 300
    assert test.timestamp.min() >= max(labeled.timestamp.max(), unlabeled.timestamp.max())


def test_model_beats_static_rules(trained):
    _, m = trained
    semi, rules = m["test"]["semi_supervised"], m["test"]["static_rules"]
    assert semi["auc"] > 0.90
    assert semi["fraud_caught"] >= rules["fraud_caught"]
    assert semi["legit_challenged"] < rules["legit_challenged"]


def test_pseudo_labels_are_mostly_right(trained):
    _, m = trained
    st = m["self_training"]
    assert st["pseudo_labeled"] > 1000
    assert st["pseudo_label_errors"] / st["pseudo_labeled"] < 0.03


def test_scoring_matches_sklearn_model(trained):
    base, _ = trained
    bundle = joblib.load(base / "artifacts" / "model.joblib")
    W = scoring.load_weights(base / "artifacts" / "model_weights.json")
    test = split(pd.read_csv(base / "data" / "wires.csv"))[2].head(200)
    z = bundle["model"].decision_function(bundle["scaler"].transform(transform(test))) + bundle["shift"]
    expected = 1 / (1 + np.exp(-z))
    got = [scoring.score(r, weights=W)["modelScore"] for r in test[COLUMNS].to_dict("records")]
    assert np.allclose(got, expected, atol=1e-3)


def test_voice_requests_always_confirm(trained):
    base, _ = trained
    W = scoring.load_weights(base / "artifacts" / "model_weights.json")
    risky = {"amount": 85000, "usualAmount": 9000, "remoteAccess": True, "beneficiaryAgeDays": 12,
             "firstTimePayee": True, "muleListHit": True}
    safe = {"amount": 8000, "usualAmount": 9000, "beneficiaryAgeDays": 1500, "beneficiarySenders30d": 2}
    assert scoring.score(risky, weights=W)["lane"] == "hold"
    for wire in (risky, safe):
        r = scoring.score(wire | {"channel": "voice"}, weights=W)
        assert r["lane"] == "confirm" and r["policy"] == "voice_always_confirm"
        assert "in the app" in r["steps"][1]["title"]


def test_api_scores_learns_and_never_exports(trained, monkeypatch):
    from fastapi.testclient import TestClient
    base, _ = trained
    monkeypatch.setenv("DATA_DIR", str(base / "data"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(base / "artifacts"))
    monkeypatch.delenv("API_KEY", raising=False)
    import api.main as api_main
    api_main = importlib.reload(api_main)
    client = TestClient(api_main.app)

    assert client.get("/health").json()["status"] == "ok"
    wire = {"amount": 22000, "usualAmount": 7000, "newDevice": True, "simSwap7d": True, "typingMismatch": 0.6,
            "beneficiaryAgeDays": 45, "firstTimePayee": True, "highRiskDestination": True}
    r = client.post("/score", json={"wire": wire, "caseId": "W-1"}).json()
    assert r["lane"] == "hold" and len(r["reasons"]) == 3
    log = (base / "artifacts" / "audit_log.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["caseId"] == "W-1" and "amount" not in log[-1]

    for label in ("fraud", "legitimate"):
        resp = client.post("/feedback", json={"caseId": "W-1", "wire": wire, "label": label, "employeeId": "MCB-2041"})
        assert resp.status_code == 200
    out = client.post("/retrain").json()
    assert out["analystLabels"] == 2 and out["auc"] > 0.9

    for path in ("/data", "/wires", "/export", "/customers", "/feedback.csv", "/audit_log"):
        assert client.get(path).status_code in (404, 405)


def test_api_key_protects_learning_endpoints(trained, monkeypatch):
    from fastapi.testclient import TestClient
    base, _ = trained
    monkeypatch.setenv("DATA_DIR", str(base / "data"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(base / "artifacts"))
    monkeypatch.setenv("API_KEY", "secret")
    import api.main as api_main
    client = TestClient(importlib.reload(api_main).app)
    body = {"caseId": "W-2", "wire": {"amount": 1}, "label": "fraud", "employeeId": "MCB-2041"}
    assert client.post("/feedback", json=body).status_code == 401
    assert client.post("/feedback", json=body, headers={"X-API-Key": "secret"}).status_code == 200


@pytest.mark.skipif(shutil.which("node") is None or not (ROOT / "artifacts" / "model_weights.json").exists(),
                    reason="needs Node.js and a trained model in artifacts/")
def test_browser_engine_matches_python():
    W = scoring.load_weights(ROOT / "artifacts" / "model_weights.json")
    wires = [{"amount": 85000, "usualAmount": 9000, "remoteAccess": True, "beneficiaryAgeDays": 12, "firstTimePayee": True},
             {"amount": 8400, "usualAmount": 9000, "beneficiaryAgeDays": 1460, "beneficiarySenders30d": 2},
             {"amount": 96000, "usualAmount": 9000, "beneficiaryAgeDays": 2900, "firstTimePayee": True,
              "beneficiarySenders30d": 18, "channel": "voice"}]
    js = ("const W=require('./frontend/engine.js');"
          f"console.log(JSON.stringify({json.dumps(wires)}.map(t=>{{const r=W.score(t);return [r.score,r.lane]}})))")
    out = json.loads(subprocess.run(["node", "-e", js], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    for (s, lane), wire in zip(out, wires):
        py = scoring.score(wire, weights=W)
        assert math.isclose(s, py["score"], abs_tol=1e-4) and lane == py["lane"]


def test_model_report_has_summaries_only(trained, tmp_path):
    from ml.report import build
    base, _ = trained
    r = build(data_dir=base / "data", artifacts=base / "artifacts", frontend=tmp_path)
    assert (tmp_path / "model_report.js").exists()
    assert r["data"]["wires"] > 10000 and len(r["features"]) == 13 and len(r["examples"]) == 6
    assert sum(h["legit"] + h["fraud"] for h in r["histogram"]) == r["split"]["test"]
    text = (tmp_path / "model_report.js").read_text()
    assert "customer_id" not in text and "beneficiary_account" not in text   # no identifiers, summaries only
