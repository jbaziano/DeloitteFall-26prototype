"""
Train the wire-fraud model with semi-supervised self-training (scikit-learn).

Steps
  1. Simulate customers and wires (or reuse data/wires.csv).
  2. Split by time: the most recent 3,000 wires are the held-out test set.
     From earlier wires: 1,000 labeled (700 normal / 300 fraud) and 9,000 unlabeled.
     Analyst decisions saved by the API (data/feedback.csv) are added as extra labels.
  3. Fit StandardScaler + LogisticRegression inside SelfTrainingClassifier:
     the model labels unlabeled wires it is at least 95% sure about, then retrains.
  4. Correct for the training fraud mix so scores reflect a ~3% real-world fraud rate.
  5. Compare against supervised-only and today's static rules on the test set.
  6. Save artifacts/model.joblib, artifacts/metrics.json, artifacts/model_weights.json,
     and frontend/model_weights.js (so the browser demo scores with these exact weights).

Run:  python -m ml.train            (add --regenerate to re-simulate data)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier

from ml.features import COLUMNS, JS_KEYS, LABELS, TRANSFORMS, transform
from sim.simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {"release": 0.05, "hold": 0.40}
DEPLOY_FRAUD_RATE = 0.03
SEED = 20261103


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def rules_flag(df: pd.DataFrame) -> np.ndarray:
    """Static rules like Meridian's current system: big amount, new payee, or risky destination."""
    return ((df.amount_vs_usual > 2.5) | (df.first_time_payee == 1) | (df.high_risk_destination == 1)).to_numpy()


def lane_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    flagged = p >= THRESHOLDS["release"]
    hold = p >= THRESHOLDS["hold"]
    return _flag_metrics(flagged, y) | {
        "fraud_held": float((hold & (y == 1)).sum() / max(1, (y == 1).sum())),
        "lanes": {
            "release": {"legit": int(((~flagged) & (y == 0)).sum()), "fraud": int(((~flagged) & (y == 1)).sum())},
            "confirm": {"legit": int((flagged & ~hold & (y == 0)).sum()), "fraud": int((flagged & ~hold & (y == 1)).sum())},
            "hold": {"legit": int((hold & (y == 0)).sum()), "fraud": int((hold & (y == 1)).sum())},
        },
    }


def _flag_metrics(flagged: np.ndarray, y: np.ndarray) -> dict:
    fraud, legit = y == 1, y == 0
    return {
        "fraud_caught": float((flagged & fraud).sum() / max(1, fraud.sum())),
        "legit_challenged": float((flagged & legit).sum() / max(1, legit.sum())),
        "flags_that_were_legit": float((flagged & legit).sum() / max(1, flagged.sum())),
    }


def split(wires: pd.DataFrame, seed: int = SEED, n_test: int = 3000, n_lab_fraud: int = 300,
          n_lab_normal: int = 700, n_unlabeled: int = 9000):
    wires = wires.sort_values("timestamp").reset_index(drop=True)
    test = wires.tail(n_test)
    pool = wires.iloc[: len(wires) - n_test]
    rng = np.random.default_rng(seed)
    f_idx = rng.choice(pool.index[pool.is_fraud == 1], n_lab_fraud, replace=False)
    n_idx = rng.choice(pool.index[pool.is_fraud == 0], n_lab_normal, replace=False)
    labeled = pool.loc[np.concatenate([f_idx, n_idx])]
    rest = pool.drop(labeled.index)
    unlabeled = rest.loc[rng.choice(rest.index, min(n_unlabeled, len(rest)), replace=False)]
    return labeled, unlabeled, test


def train(regenerate: bool = False, data_dir: Path = ROOT / "data", artifacts: Path = ROOT / "artifacts",
          frontend: Path | None = ROOT / "frontend", quiet: bool = False) -> dict:
    say = (lambda *a: None) if quiet else print
    wires_csv = data_dir / "wires.csv"
    if regenerate or not wires_csv.exists():
        customers, wires = simulate(seed=SEED)
        data_dir.mkdir(parents=True, exist_ok=True)
        customers.to_csv(data_dir / "customers.csv", index=False)
        wires.to_csv(wires_csv, index=False)
    wires = pd.read_csv(wires_csv)

    labeled, unlabeled, test = split(wires)
    fb_csv = data_dir / "feedback.csv"
    feedback = pd.read_csv(fb_csv) if fb_csv.exists() else pd.DataFrame(columns=COLUMNS + ["is_fraud"])
    if len(feedback):
        labeled = pd.concat([labeled, feedback[COLUMNS + ["is_fraud"]]], ignore_index=True)
    say(f"Labeled {len(labeled):,} (incl. {len(feedback)} analyst labels), unlabeled {len(unlabeled):,}, test {len(test):,}")

    X_lab, X_unl, X_test = transform(labeled), transform(unlabeled), transform(test)
    y_lab = labeled.is_fraud.to_numpy(int)
    y_test = test.is_fraud.to_numpy(int)

    scaler = StandardScaler().fit(np.vstack([X_lab, X_unl]))
    Z_lab, Z_unl, Z_test = scaler.transform(X_lab), scaler.transform(X_unl), scaler.transform(X_test)

    # ---- semi-supervised self-training ----
    self_training = SelfTrainingClassifier(LogisticRegression(C=1.0, max_iter=2000),
                                           threshold=0.95, criterion="threshold", max_iter=10)
    y_semi = np.concatenate([y_lab, -np.ones(len(Z_unl), int)])     # -1 marks "unlabeled"
    self_training.fit(np.vstack([Z_lab, Z_unl]), y_semi)
    lr = self_training.estimator_

    used = self_training.labeled_iter_ >= 0
    train_prior = float(self_training.transduction_[used].mean())
    shift = logit(DEPLOY_FRAUD_RATE) - logit(train_prior)

    # pseudo-label audit (possible only because the data is synthetic)
    pl = self_training.labeled_iter_[len(Z_lab):]
    pseudo = pl > 0
    pseudo_y = self_training.transduction_[len(Z_lab):][pseudo]
    truth = unlabeled.is_fraud.to_numpy(int)[pseudo]
    rounds = [{"round": int(r), "added": int((pl == r).sum()),
               "fraud": int(((pl == r) & (self_training.transduction_[len(Z_lab):] == 1)).sum())}
              for r in range(1, int(pl.max()) + 1)] if pseudo.any() else []

    # ---- comparisons ----
    def deploy_proba(model, Z, prior):
        z = model.decision_function(Z) + (logit(DEPLOY_FRAUD_RATE) - logit(prior))
        return 1 / (1 + np.exp(-z))

    p_semi = deploy_proba(lr, Z_test, train_prior)
    supervised = LogisticRegression(C=1.0, max_iter=2000).fit(Z_lab, y_lab)
    p_sup = deploy_proba(supervised, Z_test, float(y_lab.mean()))

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"labeled": len(labeled), "analyst_labels": len(feedback), "unlabeled": len(unlabeled),
                 "test": len(test), "test_fraud": int(y_test.sum()), "test_split": "most recent wires (time-based)"},
        "self_training": {"pseudo_labeled": int(pseudo.sum()), "pseudo_labeled_fraud": int(pseudo_y.sum()),
                          "pseudo_label_errors": int((pseudo_y != truth).sum()),
                          "never_confident": int((pl == -1).sum()), "rounds": rounds,
                          "training_fraud_share": round(train_prior, 4)},
        "test": {
            "semi_supervised": {"auc": float(roc_auc_score(y_test, p_semi)),
                                "avg_precision": float(average_precision_score(y_test, p_semi))} | lane_metrics(p_semi, y_test),
            "supervised_only": {"auc": float(roc_auc_score(y_test, p_sup)),
                                "avg_precision": float(average_precision_score(y_test, p_sup))} | lane_metrics(p_sup, y_test),
            "static_rules": _flag_metrics(rules_flag(test), y_test),
        },
    }

    version = f"v{datetime.now(timezone.utc):%Y%m%d}-{len(feedback)}fb"
    weights = {
        "version": version,
        "features": [{"column": c, "jsKey": k, "transform": t, "label": LABELS[c]}
                     for c, k, t in zip(COLUMNS, JS_KEYS, TRANSFORMS)],
        "mu": [round(float(v), 6) for v in scaler.mean_],
        "sd": [round(float(v), 6) for v in scaler.scale_],
        "w": [round(float(v), 6) for v in lr.coef_[0]],
        "b": round(float(lr.intercept_[0]), 6),
        "shift": round(shift, 6),
        "thresholds": THRESHOLDS,
        "deployFraudRate": DEPLOY_FRAUD_RATE,
        "metrics": {"auc": round(metrics["test"]["semi_supervised"]["auc"], 4),
                    "fraudCaught": round(metrics["test"]["semi_supervised"]["fraud_caught"], 4),
                    "legitChallenged": round(metrics["test"]["semi_supervised"]["legit_challenged"], 4)},
    }

    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": lr, "shift": shift, "columns": COLUMNS, "version": version},
                artifacts / "model.joblib")
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifacts / "model_weights.json").write_text(json.dumps(weights, indent=2))
    if frontend is not None:
        frontend.mkdir(parents=True, exist_ok=True)
        (frontend / "model_weights.js").write_text(
            "// Generated by `python -m ml.train`. Do not edit by hand.\n"
            f"window.MODEL_WEIGHTS = {json.dumps(weights, indent=2)};\n")

    t = metrics["test"]
    say(f"Self-training pseudo-labeled {metrics['self_training']['pseudo_labeled']:,} wires "
        f"({metrics['self_training']['pseudo_label_errors']} wrong)")
    say(f"Test AUC  semi-supervised {t['semi_supervised']['auc']:.3f} | supervised only {t['supervised_only']['auc']:.3f}")
    for name in ("semi_supervised", "supervised_only", "static_rules"):
        m = t[name]
        say(f"  {name:<16} fraud caught {m['fraud_caught']:.0%}  legit challenged {m['legit_challenged']:.0%}  "
            f"flags that were legit {m['flags_that_were_legit']:.0%}")
    say(f"Saved model {version} -> {artifacts}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerate", action="store_true", help="re-simulate customers and wires first")
    train(regenerate=ap.parse_args().regenerate)


if __name__ == "__main__":
    main()
