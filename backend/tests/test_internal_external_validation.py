"""
Internal validation (refit bootstrap optimism + k-fold CV) and
external validation (logistic) — endpoint-level tests.

Covers:
  - /api/model_diagnostics/model_validation in predictors mode (logistic + Cox)
    with proper Harrell optimism correction and k-fold CV.
  - /api/model_diagnostics/external_validation_logistic discrimination +
    calibration on a fresh cohort, incl. miscalibration detection and dev->val drop.
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed(df: pd.DataFrame, session_id: str) -> str:
    store.save(session_id, df)
    return session_id


def _make_logistic(n=600, n_signal=3, n_noise=0, seed=0):
    """Binary outcome driven by n_signal informative predictors (+ optional noise)."""
    rng = np.random.default_rng(seed)
    cols = {}
    lp = np.full(n, -0.3)
    for i in range(n_signal):
        x = rng.normal(0, 1, n)
        cols[f"sig{i}"] = x
        lp = lp + 0.8 * x
    for j in range(n_noise):
        cols[f"noise{j}"] = rng.normal(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-lp))
    cols["event"] = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame(cols)


def _make_survival(n=400, seed=7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    lp = 0.7 * x1 + 0.5 * x2
    base = rng.exponential(scale=8.0, size=n)
    dur = base * np.exp(-lp)
    cens = rng.uniform(0, 12, n)
    duration = np.minimum(dur, cens)
    event = (dur <= cens).astype(int)
    return pd.DataFrame({"X1": x1, "X2": x2, "duration": duration, "event": event})


# ─────────────────────────────────────────────────────────────────────────────
# Internal validation — logistic predictors mode
# ─────────────────────────────────────────────────────────────────────────────

def test_internal_logistic_predictors_optimism_and_cv():
    df = _make_logistic(n=600, n_signal=3, seed=1)
    sid = _seed(df, "iv_logit_good")
    r = client.post("/api/model_diagnostics/model_validation", json={
        "session_id": sid, "model_type": "logistic", "outcome": "event",
        "predictors": ["sig0", "sig1", "sig2"], "n_boot": 80, "cv_folds": 5,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    apparent = d["apparent"]["auc"]
    corrected = d["corrected"]["auc"]
    assert 0.6 < apparent <= 1.0
    # Optimism is non-negative; corrected should not exceed apparent.
    assert corrected <= apparent + 1e-6
    assert d["optimism"]["auc"] >= -1e-6
    # CV block present and sane.
    assert d["cv"]["folds"] == 5
    assert 0.5 < d["cv"]["auc"] < 1.0


def test_internal_logistic_overfit_shows_large_gap():
    # Small n, many noise predictors -> apparent AUC inflated, big optimism.
    df = _make_logistic(n=120, n_signal=2, n_noise=15, seed=2)
    preds = ["sig0", "sig1"] + [f"noise{j}" for j in range(15)]
    sid = _seed(df, "iv_logit_overfit")
    r = client.post("/api/model_diagnostics/model_validation", json={
        "session_id": sid, "model_type": "logistic", "outcome": "event",
        "predictors": preds, "n_boot": 100, "cv_folds": 5,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    # Overfitting: apparent clearly above optimism-corrected.
    assert d["apparent"]["auc"] - d["corrected"]["auc"] > 0.03
    assert d["optimism"]["auc"] > 0.03
    assert "overfit_gap" in d


# ─────────────────────────────────────────────────────────────────────────────
# Internal validation — Cox predictors mode
# ─────────────────────────────────────────────────────────────────────────────

def test_internal_cox_predictors_optimism():
    df = _make_survival(n=400, seed=7)
    sid = _seed(df, "iv_cox")
    r = client.post("/api/model_diagnostics/model_validation", json={
        "session_id": sid, "model_type": "cox", "duration_col": "duration",
        "event_col": "event", "predictors": ["X1", "X2"],
        "n_boot": 60, "cv_folds": 5,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert 0.55 < d["apparent"]["c_index"] <= 1.0
    assert d["corrected"]["c_index"] <= d["apparent"]["c_index"] + 1e-6
    assert d["cv"]["folds"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# External validation — logistic
# ─────────────────────────────────────────────────────────────────────────────

def test_external_logistic_well_calibrated():
    # Generate a cohort and well-calibrated predicted probabilities for it.
    df = _make_logistic(n=800, n_signal=3, seed=3)
    from sklearn.linear_model import LogisticRegression
    X = df[["sig0", "sig1", "sig2"]].values
    y = df["event"].values
    probs = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
    df["pred"] = probs
    sid = _seed(df, "ev_good")
    r = client.post("/api/model_diagnostics/external_validation_logistic", json={
        "session_id": sid, "outcome": "event", "prob_column": "pred",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["discrimination"]["auc"] > 0.65
    assert len(d["discrimination"]["auc_ci"]) == 2
    # In-sample probs are by construction well calibrated.
    assert 0.7 <= d["calibration"]["slope"] <= 1.3
    assert abs(d["calibration"]["intercept"]) <= 0.3
    assert d["calibration"]["acceptable"] is True
    assert len(d["calibration_plot"]) >= 5


def test_external_logistic_miscalibrated_detected_and_drop():
    df = _make_logistic(n=800, n_signal=3, seed=4)
    from sklearn.linear_model import LogisticRegression
    X = df[["sig0", "sig1", "sig2"]].values
    y = df["event"].values
    probs = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
    # Deliberately miscalibrate: push probabilities toward 1 (systematic over-risk).
    bad = np.clip(probs ** 0.4, 1e-4, 1 - 1e-4)
    df["pred_bad"] = bad
    sid = _seed(df, "ev_bad")
    r = client.post("/api/model_diagnostics/external_validation_logistic", json={
        "session_id": sid, "outcome": "event", "prob_column": "pred_bad",
        "dev_auc": 0.85, "dev_calibration_slope": 1.0,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    # Discrimination roughly preserved (monotone transform), calibration broken.
    assert d["calibration"]["acceptable"] is False
    assert d["dev_vs_val"] is not None
    assert "auc_drop" in d["dev_vs_val"]
    assert "slope_shift" in d["dev_vs_val"]


def test_external_logistic_bad_inputs():
    df = _make_logistic(n=50, n_signal=2, seed=5)
    sid = _seed(df, "ev_badinput")
    # Missing column.
    r1 = client.post("/api/model_diagnostics/external_validation_logistic", json={
        "session_id": sid, "outcome": "event", "prob_column": "does_not_exist",
    })
    assert r1.status_code == 400
    # prob_column not in [0,1].
    df2 = df.copy()
    df2["raw"] = np.arange(len(df2)) * 1.0  # clearly out of [0,1]
    sid2 = _seed(df2, "ev_badrange")
    r2 = client.post("/api/model_diagnostics/external_validation_logistic", json={
        "session_id": sid2, "outcome": "event", "prob_column": "raw",
    })
    assert r2.status_code == 400
