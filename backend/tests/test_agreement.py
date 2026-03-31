"""Tests for agreement and reliability endpoints."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session


def test_bland_altman(client):
    np.random.seed(42)
    n = 50
    true_val = np.random.normal(100, 15, n)
    df = pd.DataFrame({"method1": true_val + np.random.normal(0, 5, n),
                       "method2": true_val + np.random.normal(2, 5, n)})
    sid = make_session(df, "ba1")
    r = client.post("/api/agreement/bland_altman", json={"session_id": sid, "method1": "method1", "method2": "method2"})
    assert r.status_code == 200
    d = r.json()
    assert "limits_of_agreement" in d
    assert d["limits_of_agreement"]["upper"] > d["limits_of_agreement"]["lower"]
    assert "plot_data" in d
    assert "summary" in d
    assert "r_code" in d


def test_deming(client):
    np.random.seed(42)
    n = 40
    true_val = np.random.normal(50, 10, n)
    df = pd.DataFrame({"x": true_val + np.random.normal(0, 3, n),
                       "y": 1.1 * true_val + 5 + np.random.normal(0, 3, n)})
    sid = make_session(df, "dem1")
    r = client.post("/api/agreement/deming", json={"session_id": sid, "method1": "x", "method2": "y"})
    assert r.status_code == 200
    d = r.json()
    assert "slope" in d
    assert "intercept" in d
    assert "r_code" in d


def test_passing_bablok(client):
    np.random.seed(42)
    n = 30
    x = np.random.normal(100, 20, n)
    y = 0.95 * x + 3 + np.random.normal(0, 5, n)
    df = pd.DataFrame({"ref": x, "new": y})
    sid = make_session(df, "pb1")
    r = client.post("/api/agreement/passing_bablok", json={"session_id": sid, "method1": "ref", "method2": "new"})
    assert r.status_code == 200
    d = r.json()
    assert "slope" in d
    assert "intercept" in d
    assert "r_code" in d


def test_concordance(client):
    np.random.seed(42)
    n = 50
    x = np.random.normal(100, 15, n)
    y = x + np.random.normal(0, 5, n)  # high agreement
    df = pd.DataFrame({"m1": x, "m2": y})
    sid = make_session(df, "ccc1")
    r = client.post("/api/agreement/concordance", json={"session_id": sid, "method1": "m1", "method2": "m2"})
    assert r.status_code == 200
    d = r.json()
    assert "ccc" in d
    assert d["ccc"] > 0.8  # should be high agreement
    assert "precision" in d
    assert "accuracy" in d
    assert "r_code" in d


def test_cronbach(client):
    np.random.seed(42)
    n = 50
    # Simulate a 5-item scale with good internal consistency
    latent = np.random.normal(0, 1, n)
    items = {f"item{i+1}": latent + np.random.normal(0, 0.5, n) for i in range(5)}
    df = pd.DataFrame(items)
    sid = make_session(df, "cron1")
    r = client.post("/api/reliability/cronbach", json={"session_id": sid, "items": list(items.keys())})
    assert r.status_code == 200
    d = r.json()
    assert "alpha" in d
    assert d["alpha"] > 0.7  # should be good with correlated items
    assert "item_stats" in d
    assert len(d["item_stats"]) == 5
    assert "alpha_if_deleted" in d["item_stats"][0]
    assert "r_code" in d


def test_missing_pattern(client):
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, 2, 3, None, 5],
        "c": [1, 2, 3, 4, 5],
    })
    sid = make_session(df, "miss1")
    r = client.post("/api/missing_data/pattern", json={"session_id": sid})
    assert r.status_code == 200
    d = r.json()
    assert "per_column" in d
    assert len(d["per_column"]) == 3
    assert "n_complete" in d


def test_mcar(client):
    np.random.seed(42)
    n = 100
    x = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    # Make some values missing completely at random
    x_miss = x.copy()
    x_miss[np.random.choice(n, 15, replace=False)] = np.nan
    y_miss = y.copy()
    y_miss[np.random.choice(n, 10, replace=False)] = np.nan
    df = pd.DataFrame({"x": x_miss, "y": y_miss})
    sid = make_session(df, "mcar1")
    r = client.post("/api/missing_data/mcar_test", json={"session_id": sid, "columns": ["x", "y"]})
    assert r.status_code == 200
    d = r.json()
    assert "chi2" in d or "test" in d
    assert "result_text" in d


def test_imputation_compare(client):
    np.random.seed(42)
    n = 50
    x = np.random.normal(10, 2, n)
    x[np.random.choice(n, 10, replace=False)] = np.nan
    df = pd.DataFrame({"x": x, "y": np.random.normal(5, 1, n)})
    sid = make_session(df, "imp1")
    r = client.post("/api/missing_data/imputation_compare", json={
        "session_id": sid, "columns": ["x"], "strategies": ["mean", "median"]
    })
    assert r.status_code == 200
    d = r.json()
    assert "comparisons" in d
    assert len(d["comparisons"]) >= 2
