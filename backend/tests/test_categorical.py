"""Tests for categorical endpoints."""
import numpy as np
import pandas as pd
from conftest import make_session


def test_binomial_known(client):
    df = pd.DataFrame({"outcome": [1]*60 + [0]*40})
    sid = make_session(df, "bin1")
    r = client.post("/api/categorical/binomial", json={"session_id": sid, "column": "outcome"})
    assert r.status_code == 200
    d = r.json()
    assert d["k"] == 60
    assert d["n"] == 100
    assert "r_code" in d
    assert "result_text" in d


def test_one_proportion(client):
    df = pd.DataFrame({"x": [1]*70 + [0]*30})
    sid = make_session(df, "op1")
    r = client.post("/api/categorical/one_proportion", json={"session_id": sid, "column": "x", "null_proportion": 0.5})
    assert r.status_code == 200
    d = r.json()
    assert d["significant"] is True
    assert "r_code" in d


def test_two_proportions(client):
    df = pd.DataFrame({
        "outcome": [1]*30 + [0]*20 + [1]*15 + [0]*35,
        "group": ["A"]*50 + ["B"]*50,
    })
    sid = make_session(df, "tp1")
    r = client.post("/api/categorical/two_proportions", json={"session_id": sid, "column": "outcome", "group_column": "group"})
    assert r.status_code == 200
    d = r.json()
    assert "effect_sizes" in d
    assert d["effect_sizes"][0]["name"] == "cohens_h"
    assert "r_code" in d


def test_mcnemar_known(client):
    # Classic McNemar: discordant pairs b=20, c=5
    df = pd.DataFrame({
        "before": [1]*30 + [0]*20 + [1]*5 + [0]*45,
        "after":  [1]*30 + [1]*20 + [0]*5 + [0]*45,
    })
    sid = make_session(df, "mc1")
    r = client.post("/api/categorical/mcnemar", json={"session_id": sid, "col1": "before", "col2": "after"})
    assert r.status_code == 200
    d = r.json()
    assert "r_code" in d
    assert "result_text" in d


def test_cochran_q(client):
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "t1": np.random.binomial(1, 0.3, n),
        "t2": np.random.binomial(1, 0.5, n),
        "t3": np.random.binomial(1, 0.7, n),
    })
    sid = make_session(df, "cq1")
    r = client.post("/api/categorical/cochran_q", json={"session_id": sid, "columns": ["t1", "t2", "t3"]})
    assert r.status_code == 200
    d = r.json()
    assert "Q" in d or "chi2" in d or "test" in d
    assert "r_code" in d


def test_mantel_haenszel(client):
    data = []
    for stratum in ["Hospital_A", "Hospital_B"]:
        base_or = 2.0 if stratum == "Hospital_A" else 1.5
        for _ in range(50):
            treat = np.random.binomial(1, 0.5)
            p_event = 0.3 * base_or if treat else 0.3
            event = np.random.binomial(1, min(p_event, 0.9))
            data.append({"treatment": treat, "event": event, "hospital": stratum})
    df = pd.DataFrame(data)
    sid = make_session(df, "mh1")
    r = client.post("/api/categorical/mantel_haenszel", json={
        "session_id": sid, "row_col": "treatment", "col_col": "event", "strata_col": "hospital"
    })
    assert r.status_code == 200
    d = r.json()
    assert "r_code" in d
    assert "result_text" in d
