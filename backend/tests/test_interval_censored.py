"""Interval-censored survival (/api/survival_advanced/interval_censored).

The event time is bracketed in [L, R] by periodic inspection visits. Plain KM
is biased here; we verify the Turnbull NPMLE + Weibull-AFT regression path.
"""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed(seed: int = 0) -> str:
    """True Weibull event times with a covariate + group effect, then
    interval-censor by inspecting on a fixed visit grid."""
    rng = np.random.default_rng(seed)
    n = 500
    x = rng.normal(0, 1, n)
    g = rng.integers(0, 2, n)
    # Larger x / group-1 → longer survival (protective; AFT time-ratio > 1).
    t_true = rng.weibull(1.5, n) * np.exp(0.5 * x + 0.4 * g) * 10.0
    visits = np.arange(0.0, 90.0, 5.0)
    lower = np.array([visits[visits <= t].max() if (visits <= t).any() else 0.0 for t in t_true])
    upper = np.array([visits[visits > t].min() if (visits > t).any() else np.inf for t in t_true])
    # Encode right-censored upper bound as blank (NaN) — the API maps it to +inf.
    upper_col = [u if np.isfinite(u) else np.nan for u in upper]
    df = pd.DataFrame({"L": lower, "R": upper_col, "x": x, "grp": g})
    sid = f"ic_{seed}"
    store.save(sid, df)
    return sid


def test_interval_censored_npmle_and_median():
    sid = _seed(1)
    r = client.post("/api/survival_advanced/interval_censored", json={
        "session_id": sid, "lower_col": "L", "upper_col": "R",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n"] == 500
    assert d["n_interval_censored"] > 0
    assert d["n_right_censored"] >= 0
    curve = d["npmle_curve"]
    assert len(curve) >= 2
    surv = [p["survival"] for p in curve]
    # Survival is monotone non-increasing (allow tiny numeric wobble).
    assert all(surv[i] + 1e-6 >= surv[i + 1] for i in range(len(surv) - 1))
    assert d["median_survival_time"] is not None and d["median_survival_time"] > 0


def test_interval_censored_regression_recovers_effect():
    sid = _seed(2)
    r = client.post("/api/survival_advanced/interval_censored", json={
        "session_id": sid, "lower_col": "L", "upper_col": "R",
        "covariates": ["x", "grp"],
    })
    assert r.status_code == 200, r.text
    reg = r.json()["regression"]
    assert reg is not None and "coefficients" in reg
    by = {c["variable"]: c for c in reg["coefficients"]}
    assert "x" in by
    # x lengthens survival → time ratio > 1 and a protective hazard ratio < 1.
    assert by["x"]["time_ratio"] > 1.0
    assert by["x"]["hazard_ratio"] < 1.0
    assert by["x"]["p"] < 0.05


def test_interval_censored_group_curves():
    sid = _seed(3)
    r = client.post("/api/survival_advanced/interval_censored", json={
        "session_id": sid, "lower_col": "L", "upper_col": "R", "group_col": "grp",
    })
    assert r.status_code == 200, r.text
    groups = r.json()["groups"]
    assert groups is not None and len(groups) == 2
    assert all(len(g["curve"]) >= 1 for g in groups)


def test_interval_censored_rejects_too_few():
    df = pd.DataFrame({"L": [1.0, 2.0, 3.0], "R": [2.0, 3.0, 4.0]})
    store.save("ic_small", df)
    r = client.post("/api/survival_advanced/interval_censored", json={
        "session_id": "ic_small", "lower_col": "L", "upper_col": "R",
    })
    assert r.status_code == 422


def test_interval_censored_rejects_missing_column():
    sid = _seed(4)
    r = client.post("/api/survival_advanced/interval_censored", json={
        "session_id": sid, "lower_col": "L", "upper_col": "NOPE",
    })
    assert r.status_code == 422
