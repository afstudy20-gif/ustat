"""Tests for /api/causal/* — instrumental variable (2SLS), mediation, target trial."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


# ── Instrumental Variable (2SLS) ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def iv_sid():
    rng = np.random.default_rng(5)
    n = 400
    Z = rng.normal(0, 1, n)                  # instrument
    U = rng.normal(0, 1, n)                  # unmeasured confounder
    X = 0.8 * Z + 0.7 * U + rng.normal(0, 0.5, n)   # endogenous
    Y = 1.5 * X + 1.0 * U + rng.normal(0, 1, n)     # true effect = 1.5; OLS biased upward
    Z2 = 0.6 * Z + rng.normal(0, 0.5, n)     # second instrument (for over-id)
    df = pd.DataFrame({"Z": Z, "Z2": Z2, "X": X, "Y": Y, "age": rng.normal(50, 8, n)})
    return make_session(df, "iv_main")


def test_iv_recovers_true_effect(client, iv_sid):
    r = client.post("/api/causal/iv_2sls", json={
        "session_id": iv_sid, "outcome": "Y", "endogenous": "X",
        "instruments": ["Z"], "covariates": ["age"],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    # IV near true 1.5, and clearly less biased than OLS (~2.x)
    assert abs(d["iv_estimate"]["estimate"] - 1.5) < 0.25
    assert d["ols_estimate"]["estimate"] > d["iv_estimate"]["estimate"] + 0.2
    assert d["first_stage"]["weak_instruments"] is False
    assert d["first_stage"]["f_stat"] > 10
    assert d["wu_hausman"]["endogenous"] is True       # endogeneity present
    assert d["sargan"] is None                          # just-identified


def test_iv_overidentified_sargan(client, iv_sid):
    r = client.post("/api/causal/iv_2sls", json={
        "session_id": iv_sid, "outcome": "Y", "endogenous": "X", "instruments": ["Z", "Z2"],
    })
    assert r.status_code == 200, r.text
    sg = r.json()["sargan"]
    assert sg is not None and sg["df"] == 1 and 0.0 <= sg["p"] <= 1.0


def test_iv_instrument_equals_endogenous_400(client, iv_sid):
    r = client.post("/api/causal/iv_2sls", json={
        "session_id": iv_sid, "outcome": "Y", "endogenous": "X", "instruments": ["X"],
    })
    assert r.status_code == 400, r.text


def test_iv_missing_column_400(client, iv_sid):
    r = client.post("/api/causal/iv_2sls", json={
        "session_id": iv_sid, "outcome": "Y", "endogenous": "X", "instruments": ["nope"],
    })
    assert r.status_code == 400, r.text


# ── Mediation ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def med_sid():
    rng = np.random.default_rng(7)
    n = 400
    X = rng.normal(0, 1, n)
    M = 0.6 * X + rng.normal(0, 0.8, n)          # a = 0.6
    Y = 0.5 * M + 0.2 * X + rng.normal(0, 1, n)  # b = 0.5, c' = 0.2 → ACME ≈ 0.30
    df = pd.DataFrame({"X": X, "M": M, "Y": Y, "age": rng.normal(50, 8, n)})
    return make_session(df, "med_main")


def test_mediation_detects_indirect_effect(client, med_sid):
    r = client.post("/api/causal/mediation", json={
        "session_id": med_sid, "outcome": "Y", "treatment": "X", "mediator": "M",
        "covariates": ["age"], "bootstrap": 400,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["acme_significant"] is True
    assert d["effects"]["acme"] > 0.1
    assert d["effects"]["acme_ci"][0] > 0          # bootstrap CI excludes 0
    assert 0.0 < d["effects"]["proportion_mediated"] <= 1.0
    assert d["sobel"]["p"] < 0.05


def test_mediation_requires_distinct_columns(client, med_sid):
    r = client.post("/api/causal/mediation", json={
        "session_id": med_sid, "outcome": "Y", "treatment": "X", "mediator": "X",
    })
    assert r.status_code == 400, r.text


# ── Target trial emulation ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tt_sid():
    rng = np.random.default_rng(9)
    n = 600
    age = rng.normal(60, 10, n)
    sev = rng.normal(0, 1, n)
    ps = 1 / (1 + np.exp(-(-0.5 + 0.03 * age + 0.8 * sev)))
    trt = (rng.uniform(0, 1, n) < ps).astype(int)          # confounded by age/severity
    py = 1 / (1 + np.exp(-(-1 + 0.02 * age + 0.6 * sev - 0.7 * trt)))
    died = (rng.uniform(0, 1, n) < py).astype(int)         # true protective effect
    df = pd.DataFrame({"age": age, "severity": sev, "treatment": trt, "died": died})
    return make_session(df, "tt_main")


def test_target_trial_recovers_protective_effect(client, tt_sid):
    r = client.post("/api/causal/target_trial", json={
        "session_id": tt_sid, "treatment": "treatment", "outcome": "died",
        "confounders": ["age", "severity"],
        "eligibility": [{"column": "age", "op": "gte", "value": 40}],
        "bootstrap": 300,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_eligible"] <= d["n_screened"]
    assert d["effect"]["risk_difference"] < 0          # protective
    assert d["effect"]["rd_ci"][1] < 0                  # CI excludes 0 (negative)
    assert len(d["protocol"]) == 7
    assert "caveats" in d and len(d["caveats"]) >= 1
    assert len(d["balance"]) == 2


def test_target_trial_needs_confounders(client, tt_sid):
    r = client.post("/api/causal/target_trial", json={
        "session_id": tt_sid, "treatment": "treatment", "outcome": "died", "confounders": [],
    })
    assert r.status_code == 400, r.text


# ── Difference-in-Differences / RDD / DAG ─────────────────────────────────────

def test_did_recovers_interaction(client):
    rng = np.random.default_rng(11)
    n = 400
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    Y = 10 + 2 * g + 1 * t + 3 * (g * t) + rng.normal(0, 2, n)   # true DiD = 3
    sid = make_session(pd.DataFrame({"Y": Y, "grp": g, "time": t, "age": rng.normal(50, 8, n)}), "did_main")
    r = client.post("/api/causal/did", json={
        "session_id": sid, "outcome": "Y", "group_col": "grp", "time_col": "time", "covariates": ["age"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert abs(d["did_estimate"] - 3.0) < 0.7
    assert d["significant"] is True
    assert set(d["cell_means"]) == {"control_pre", "control_post", "treated_pre", "treated_post"}


def test_rdd_detects_jump(client):
    rng = np.random.default_rng(12)
    n = 500
    x = rng.uniform(-5, 5, n)
    Y = 2 + 0.8 * x + 5 * (x >= 0) + rng.normal(0, 1.5, n)        # jump of 5 at 0
    sid = make_session(pd.DataFrame({"score": x, "out": Y}), "rdd_main")
    r = client.post("/api/causal/rdd", json={"session_id": sid, "outcome": "out", "running": "score", "cutoff": 0})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ci_low"] <= 5 <= d["ci_high"]      # CI covers the true jump
    assert d["significant"] is True
    assert len(d["binned"]) > 0


def test_dag_roles_and_adjustment_set(client):
    r = client.post("/api/causal/dag_adjustment", json={
        "edges": [["Z", "T"], ["Z", "Y"], ["T", "M"], ["M", "Y"], ["T", "C"], ["Y", "C"]],
        "treatment": "T", "outcome": "Y"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["roles"]["Z"] == "confounder"
    assert d["roles"]["M"] == "mediator"
    assert d["roles"]["C"] == "collider"
    assert d["adjustment_set"] == ["Z"]
    assert set(d["do_not_adjust"]) == {"C", "M"}


def test_dag_treatment_not_in_edges_400(client):
    r = client.post("/api/causal/dag_adjustment", json={
        "edges": [["A", "B"]], "treatment": "X", "outcome": "B"})
    assert r.status_code == 400, r.text
