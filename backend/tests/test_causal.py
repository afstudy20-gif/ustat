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
