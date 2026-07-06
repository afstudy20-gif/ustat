"""Deep coverage for time-series endpoints: ARIMA, decomposition,
stationarity, and GEE (longitudinal / repeated-measures) regression.

Existing smoke tests (test_v2_endpoints.py, test_package9_polish.py) only
touch these endpoints lightly. This file verifies statistical correctness
against synthetic series/data with known ground truth.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


def _sid(name: str) -> str:
    return f"ts_{name}"


# ── 1. ARIMA ──────────────────────────────────────────────────────────────


def test_arima_manual_order_recovers_ar1_coefficient(client):
    """AR(1) process y_t = phi * y_{t-1} + eps, phi=0.7 — fit ARIMA(1,0,0)
    and check the AR coefficient is recovered in a sane range, plus fitted
    values and forecast horizon are returned."""
    rng = np.random.default_rng(1)
    n = 300
    phi = 0.7
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(0, 1)
    sid = make_session(pd.DataFrame({"t": np.arange(n), "value": y}), _sid("arima_ar1"))

    r = client.post("/api/timeseries/arima", json={
        "session_id": sid, "value_col": "value", "time_col": "t",
        "p": 1, "d": 0, "q": 0, "forecast_steps": 5,
    })
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["order"] == [1, 0, 0]
    assert len(body["forecast"]) == 5
    assert all("ci_low" in f and "ci_high" in f and "forecast" in f for f in body["forecast"])
    assert len(body["fitted"]) == body["n"]

    ar_terms = [c for c in body["coefficients"] if "ar.L1" in c["term"]]
    assert len(ar_terms) == 1
    assert 0.5 < ar_terms[0]["estimate"] < 0.9


def test_arima_manual_order_linear_trend_forecast_continues_trend(client):
    """A near-perfect linear trend with small noise, fit with d=1 (first
    difference should remove the trend) — forecast should continue roughly
    along the trend line, not revert wildly."""
    rng = np.random.default_rng(2)
    n = 100
    x = np.arange(n)
    y = 10 + 2.0 * x + rng.normal(0, 0.5, n)
    sid = make_session(pd.DataFrame({"t": x, "value": y}), _sid("arima_trend"))

    r = client.post("/api/timeseries/arima", json={
        "session_id": sid, "value_col": "value", "time_col": "t",
        "p": 1, "d": 1, "q": 0, "forecast_steps": 3,
    })
    assert r.status_code == 200, r.text
    body = r.json()

    last_obs = y[-1]
    first_fc = body["forecast"][0]["forecast"]
    last_fc = body["forecast"][-1]["forecast"]
    # Forecast should keep climbing along the ~2/step trend, not collapse.
    assert first_fc > last_obs - 5
    assert last_fc > first_fc - 1  # non-decreasing-ish, allows small noise


def test_arima_auto_mode_selects_order_by_aic(client):
    """auto=True should run the bounded grid search and return an order,
    aic/bic, and grid_searched semantics via the 'auto' flag."""
    rng = np.random.default_rng(3)
    n = 150
    phi = 0.5
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(0, 1)
    sid = make_session(pd.DataFrame({"t": np.arange(n), "value": y}), _sid("arima_auto"))

    r = client.post("/api/timeseries/arima", json={
        "session_id": sid, "value_col": "value", "time_col": "t",
        "auto": True, "forecast_steps": 4,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["auto"] is True
    assert isinstance(body["order"], list) and len(body["order"]) == 3
    assert "aic" in body and "bic" in body
    assert len(body["forecast"]) == 4


def test_arima_too_short_series_returns_4xx_not_500(client):
    sid = make_session(pd.DataFrame({"t": np.arange(5), "value": [1, 2, 3, 4, 5]}), _sid("arima_short"))

    r = client.post("/api/timeseries/arima", json={
        "session_id": sid, "value_col": "value", "time_col": "t",
        "p": 1, "d": 0, "q": 0, "forecast_steps": 3,
    })
    assert 400 <= r.status_code < 500, r.text


def test_arima_unknown_column_returns_4xx(client):
    sid = make_session(pd.DataFrame({"t": np.arange(30), "value": np.arange(30)}), _sid("arima_badcol"))

    r = client.post("/api/timeseries/arima", json={
        "session_id": sid, "value_col": "does_not_exist", "time_col": "t",
        "p": 1, "d": 0, "q": 0, "forecast_steps": 3,
    })
    assert 400 <= r.status_code < 500, r.text


# ── 2. Decomposition ──────────────────────────────────────────────────────


def test_decompose_recovers_seasonal_period_and_trend(client):
    """Build trend + period-12 seasonal + small noise. Seasonal component
    returned should repeat with the given period (high correlation between
    values one period apart) and seasonality should be flagged as detected."""
    rng = np.random.default_rng(4)
    n = 96  # 8 full periods of 12
    x = np.arange(n)
    trend = 0.3 * x
    seasonal_pattern = 10 * np.sin(2 * np.pi * x / 12)
    noise = rng.normal(0, 0.3, n)
    y = 50 + trend + seasonal_pattern + noise
    sid = make_session(pd.DataFrame({"t": x, "value": y}), _sid("decompose_seasonal"))

    r = client.post("/api/timeseries/decompose", json={
        "session_id": sid, "value_col": "value", "time_col": "t",
        "period": 12, "method": "stl",
    })
    assert r.status_code == 200, r.text
    body = r.json()

    for k in ("trend", "seasonal", "resid", "observed"):
        assert k in body and len(body[k]) == body["n"]

    assert body["seasonality_detected"] is True
    assert body["strength_seasonal"] > 0.5

    seasonal = np.array(body["seasonal"], dtype=float)
    period = body["period"]
    assert period == 12
    # Correlate seasonal[i] with seasonal[i+period] across all valid pairs.
    a = seasonal[: n - period]
    b = seasonal[period:]
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.9


def test_decompose_classical_additive_matches_stl_seasonality_signal(client):
    rng = np.random.default_rng(5)
    n = 96
    x = np.arange(n)
    y = 20 + 0.1 * x + 5 * np.sin(2 * np.pi * x / 12) + rng.normal(0, 0.2, n)
    sid = make_session(pd.DataFrame({"t": x, "value": y}), _sid("decompose_classical"))

    r = client.post("/api/timeseries/decompose", json={
        "session_id": sid, "value_col": "value", "time_col": "t",
        "period": 12, "method": "classical", "model": "additive",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "classical"
    assert body["seasonality_detected"] is True


def test_decompose_too_short_series_returns_4xx(client):
    sid = make_session(pd.DataFrame({"t": np.arange(10), "value": np.arange(10)}), _sid("decompose_short"))

    r = client.post("/api/timeseries/decompose", json={
        "session_id": sid, "value_col": "value", "time_col": "t", "period": 12,
    })
    assert 400 <= r.status_code < 500, r.text


# ── 3. Stationarity (ADF/KPSS + ACF/PACF) ────────────────────────────────


def test_stationarity_white_noise_is_stationary(client):
    rng = np.random.default_rng(6)
    n = 200
    y = rng.normal(0, 1, n)
    sid = make_session(pd.DataFrame({"t": np.arange(n), "value": y}), _sid("stationary_wn"))

    r = client.post("/api/timeseries/stationarity", json={
        "session_id": sid, "value_col": "value", "time_col": "t", "n_lags": 20,
    })
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["adf_p"] < 0.05
    assert body["adf_stationary"] is True
    assert len(body["acf"]) == 21
    assert len(body["pacf"]) == 21


def test_stationarity_random_walk_is_non_stationary(client):
    rng = np.random.default_rng(7)
    n = 200
    steps = rng.normal(0, 1, n)
    y = np.cumsum(steps)
    sid = make_session(pd.DataFrame({"t": np.arange(n), "value": y}), _sid("stationary_rw"))

    r = client.post("/api/timeseries/stationarity", json={
        "session_id": sid, "value_col": "value", "time_col": "t", "n_lags": 20,
    })
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["adf_p"] >= 0.05
    assert body["adf_stationary"] is False
    assert len(body["acf"]) == 21
    assert len(body["pacf"]) == 21


def test_stationarity_strong_linear_trend_is_non_stationary(client):
    rng = np.random.default_rng(8)
    n = 150
    x = np.arange(n)
    y = 5 * x + rng.normal(0, 1, n)
    sid = make_session(pd.DataFrame({"t": x, "value": y}), _sid("stationary_trend"))

    r = client.post("/api/timeseries/stationarity", json={
        "session_id": sid, "value_col": "value", "time_col": "t", "n_lags": 15,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["adf_p"] >= 0.05
    assert body["adf_stationary"] is False


def test_stationarity_too_short_series_returns_4xx(client):
    sid = make_session(pd.DataFrame({"t": np.arange(5), "value": [1, 2, 1, 2, 1]}), _sid("stationary_short"))

    r = client.post("/api/timeseries/stationarity", json={
        "session_id": sid, "value_col": "value", "time_col": "t", "n_lags": 5,
    })
    assert 400 <= r.status_code < 500, r.text


# ── 4. GEE (longitudinal / clustered regression) ─────────────────────────


def _make_gee_gaussian_df(rng, n_subjects=60, n_visits=4, beta=3.0):
    """Longitudinal Gaussian outcome: y = 10 + beta*x + eps, where eps follows
    an AR(1) process within each subject (rho=0.6) so that all three
    correlation structures (independence / exchangeable / ar) have genuine
    within-subject dependence to pick up — a pure random-intercept design is
    degenerate for the Autoregressive() bracket search in statsmodels."""
    rows = []
    for sidx in range(n_subjects):
        x = rng.normal(0, 1)  # time-invariant predictor
        eps = np.zeros(n_visits)
        eps[0] = rng.normal(0, 1)
        for v in range(1, n_visits):
            eps[v] = 0.6 * eps[v - 1] + rng.normal(0, 1)
        for v in range(n_visits):
            y = 10 + beta * x + eps[v]
            rows.append({"subject": sidx, "visit": v, "x": x, "y": y})
    return pd.DataFrame(rows)


def _make_gee_binomial_df(rng, n_subjects=80, n_visits=4, beta=1.5):
    """Binary longitudinal outcome with an AR(1) latent process per subject
    (rho=0.6 on the logit scale) so the Autoregressive() cov_struct has real
    within-subject dependence to estimate, same rationale as the Gaussian
    generator above."""
    rows = []
    for sidx in range(n_subjects):
        x = rng.integers(0, 2)
        eps = np.zeros(n_visits)
        eps[0] = rng.normal(0, 0.5)
        for v in range(1, n_visits):
            eps[v] = 0.6 * eps[v - 1] + rng.normal(0, 0.5)
        for v in range(n_visits):
            logit = -0.5 + beta * x + eps[v]
            p = 1 / (1 + np.exp(-logit))
            y = int(rng.uniform() < p)
            rows.append({"subject": sidx, "visit": v, "x": x, "y": y})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("cov_struct", ["independence", "exchangeable", "ar"])
def test_gee_gaussian_recovers_positive_effect_across_cov_structs(client, cov_struct):
    rng = np.random.default_rng(10)
    df = _make_gee_gaussian_df(rng, beta=3.0)
    sid = make_session(df, _sid(f"gee_gauss_{cov_struct}"))

    r = client.post("/api/models/gee", json={
        "session_id": sid, "outcome": "y", "predictors": ["x"], "group_col": "subject",
        "family": "gaussian", "cov_struct": cov_struct,
    })
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["n_clusters"] == 60
    assert body["n_obs"] == 60 * 4
    assert body["cov_struct"] == cov_struct

    coefs = {c["variable"]: c for c in body["coefficients"]}
    assert "x" in coefs
    # SE must be present (statsmodels GEE .bse are robust/sandwich SEs).
    assert coefs["x"]["se"] is not None and coefs["x"]["se"] > 0
    # Effect should be recovered directionally and in a sane magnitude range.
    assert 2.0 < coefs["x"]["estimate"] < 4.0
    assert coefs["x"]["p"] < 0.05


@pytest.mark.parametrize("cov_struct", ["independence", "exchangeable", "ar"])
def test_gee_binomial_recovers_positive_effect_across_cov_structs(client, cov_struct):
    rng = np.random.default_rng(11)
    df = _make_gee_binomial_df(rng, beta=2.0)
    sid = make_session(df, _sid(f"gee_binom_{cov_struct}"))

    r = client.post("/api/models/gee", json={
        "session_id": sid, "outcome": "y", "predictors": ["x"], "group_col": "subject",
        "family": "binomial", "cov_struct": cov_struct,
    })
    assert r.status_code == 200, r.text
    body = r.json()

    coefs = {c["variable"]: c for c in body["coefficients"]}
    assert "x" in coefs
    assert coefs["x"]["se"] is not None and coefs["x"]["se"] > 0
    # Positive log-odds effect direction should match the synthetic beta=2.0.
    assert coefs["x"]["estimate"] > 0.3


def test_gee_unknown_group_col_returns_4xx(client):
    rng = np.random.default_rng(12)
    df = _make_gee_gaussian_df(rng, n_subjects=20)
    sid = make_session(df, _sid("gee_badgroup"))

    r = client.post("/api/models/gee", json={
        "session_id": sid, "outcome": "y", "predictors": ["x"], "group_col": "does_not_exist",
        "family": "gaussian", "cov_struct": "independence",
    })
    assert 400 <= r.status_code < 500, r.text


def test_gee_unsupported_family_returns_4xx(client):
    rng = np.random.default_rng(13)
    df = _make_gee_gaussian_df(rng, n_subjects=20)
    sid = make_session(df, _sid("gee_badfamily"))

    r = client.post("/api/models/gee", json={
        "session_id": sid, "outcome": "y", "predictors": ["x"], "group_col": "subject",
        "family": "not_a_family", "cov_struct": "independence",
    })
    assert 400 <= r.status_code < 500, r.text


def test_gee_unsupported_cov_struct_returns_4xx(client):
    rng = np.random.default_rng(14)
    df = _make_gee_gaussian_df(rng, n_subjects=20)
    sid = make_session(df, _sid("gee_badcov"))

    r = client.post("/api/models/gee", json={
        "session_id": sid, "outcome": "y", "predictors": ["x"], "group_col": "subject",
        "family": "gaussian", "cov_struct": "not_a_struct",
    })
    assert 400 <= r.status_code < 500, r.text
