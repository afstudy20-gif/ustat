"""Tests for diagnostics, model_diagnostics, decision_curve, model_compare."""
import numpy as np
import pandas as pd
from conftest import make_session


# ── Linear diagnostics ───────────────────────────────────────────────────────

def test_linear_full(client):
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = 2*x1 + 3*x2 + np.random.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    sid = make_session(df, "diag1")
    r = client.post("/api/diagnostics/linear_full", json={
        "session_id": sid, "outcome": "y", "predictors": ["x1", "x2"]
    })
    assert r.status_code == 200
    d = r.json()
    assert "vif" in d
    assert len(d["vif"]) == 2
    assert all(v["vif"] < 5 for v in d["vif"])  # no collinearity
    assert "heteroscedasticity" in d
    assert "breusch_pagan" in d["heteroscedasticity"]
    assert "autocorrelation" in d
    assert "plots" in d
    assert "result_text" in d
    assert "r_code" in d


def test_linear_full_collinearity(client):
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = x1 + np.random.normal(0, 0.01, n)  # nearly identical
    y = x1 + np.random.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    sid = make_session(df, "diag2")
    r = client.post("/api/diagnostics/linear_full", json={
        "session_id": sid, "outcome": "y", "predictors": ["x1", "x2"]
    })
    assert r.status_code == 200
    d = r.json()
    # At least one VIF should be very high
    max_vif = max(v["vif"] for v in d["vif"])
    assert max_vif > 10


# ── Logistic diagnostics ─────────────────────────────────────────────────────

def test_logistic_diagnostics(client):
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-(x1 + 0.5*x2)))
    y = np.random.binomial(1, prob)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    sid = make_session(df, "logdiag1")
    r = client.post("/api/model_diagnostics/logistic_diagnostics", json={
        "session_id": sid, "outcome": "y", "predictors": ["x1", "x2"]
    })
    assert r.status_code == 200
    d = r.json()
    assert "calibration" in d
    assert "brier_score" in d
    assert d["brier_score"] < 0.5  # better than random
    assert "hosmer_lemeshow" in d
    assert "c_statistic" in d
    assert "result_text" in d


# ── Cox diagnostics ──────────────────────────────────────────────────────────

def test_cox_diagnostics(client):
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    time = np.random.exponential(10, n) * np.exp(-0.5*x1)
    event = np.random.binomial(1, 0.7, n)
    df = pd.DataFrame({"time": time, "event": event, "x1": x1})
    sid = make_session(df, "coxdiag1")
    r = client.post("/api/model_diagnostics/cox_diagnostics", json={
        "session_id": sid, "duration_col": "time", "event_col": "event", "predictors": ["x1"]
    })
    assert r.status_code == 200
    d = r.json()
    assert "ph_test" in d
    assert "c_index" in d
    assert d["c_index"] > 0.5  # better than chance
    assert "result_text" in d


# ── Calibration ──────────────────────────────────────────────────────────────

def test_calibration(client):
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-x))
    y = np.random.binomial(1, prob)
    df = pd.DataFrame({"y": y, "x": x})
    sid = make_session(df, "cal1")
    r = client.post("/api/decision_curve/calibration", json={
        "session_id": sid, "outcome": "y", "predictors": ["x"]
    })
    assert r.status_code == 200
    d = r.json()
    assert "bins" in d
    assert "calibration_slope" in d
    assert "brier_score" in d
    assert "r_code" in d


# ── DCA ──────────────────────────────────────────────────────────────────────

def test_dca(client):
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-x))
    y = np.random.binomial(1, prob)
    df = pd.DataFrame({"y": y, "x": x})
    sid = make_session(df, "dca1")
    r = client.post("/api/decision_curve/dca", json={
        "session_id": sid, "outcome": "y", "predictors": ["x"]
    })
    assert r.status_code == 200
    d = r.json()
    assert "curves" in d
    assert "model" in d["curves"]
    assert "treat_all" in d["curves"]
    assert "treat_none" in d["curves"]
    assert "result_text" in d


# ── Nested LR test ───────────────────────────────────────────────────────────

def test_nested_lr(client):
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-(x1 + 0.5*x2)))
    y = np.random.binomial(1, prob)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    sid = make_session(df, "lr1")
    r = client.post("/api/model_compare/nested_lr_test", json={
        "session_id": sid, "outcome": "y",
        "predictors_reduced": ["x1"],
        "predictors_full": ["x1", "x2"],
        "model_type": "logistic"
    })
    assert r.status_code == 200
    d = r.json()
    assert "lr_stat" in d
    assert "p" in d
    assert "reduced" in d
    assert "full" in d
    assert "r_code" in d


# ── Model comparison ─────────────────────────────────────────────────────────

def test_compare_models(client):
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-(x1 + 0.5*x2)))
    y = np.random.binomial(1, prob)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    sid = make_session(df, "cmp1")
    r = client.post("/api/model_compare/compare_models", json={
        "session_id": sid, "outcome": "y",
        "model_specs": [
            {"name": "Model 1", "predictors": ["x1"]},
            {"name": "Model 2", "predictors": ["x1", "x2"]},
            {"name": "Model 3", "predictors": ["x1", "x2", "x3"]},
        ],
        "model_type": "logistic"
    })
    assert r.status_code == 200
    d = r.json()
    assert "models" in d
    assert len(d["models"]) == 3
    assert "best_model" in d
    assert "r_code" in d


# ── Bootstrap/permutation ────────────────────────────────────────────────────

def test_bootstrap_ci():
    from services.stat_utils import bootstrap_ci
    data = np.random.RandomState(42).normal(10, 2, 50)
    result = bootstrap_ci(data, np.mean, n_boot=500)
    assert 9 < result["ci_low"] < result["ci_high"] < 11
    assert result["method"] == "percentile bootstrap"


def test_permutation_test():
    from services.stat_utils import permutation_test
    x = np.random.RandomState(42).normal(10, 1, 30)
    y = np.random.RandomState(42).normal(12, 1, 30)
    result = permutation_test(x, y, n_perm=1000)
    assert result["p_permutation"] < 0.05  # significant difference
    assert result["significant"] == True
