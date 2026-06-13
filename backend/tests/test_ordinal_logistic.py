"""Proportional-odds ordinal logistic regression (/api/models/ordinal)."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed() -> str:
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0, 1, n)
    # Latent ordinal outcome driven by x → 3 ordered categories.
    lin = 1.2 * x + rng.logistic(0, 1, n)
    y = np.where(lin < -0.5, 1, np.where(lin < 1.0, 2, 3))  # codes 1/2/3
    df = pd.DataFrame({"stage": y, "x": x, "grp": rng.integers(0, 2, n)})
    sid = "ord_log"
    store.save(sid, df)
    return sid


def test_ordinal_returns_proportional_odds():
    sid = _seed()
    r = client.post("/api/models/ordinal", json={
        "session_id": sid, "outcome": "stage", "predictors": ["x", "grp"],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["categories_in_rank_order"] == ["1", "2", "3"]
    assert d["n"] == 300
    # One coefficient per predictor (proportional odds — NOT one per category).
    names = {c["variable"] for c in d["coefficients"]}
    assert names == {"x", "grp"}
    cx = next(c for c in d["coefficients"] if c["variable"] == "x")
    assert "odds_ratio" in cx and cx["odds_ratio"] > 1  # x drives higher stage
    assert cx["or_ci_low"] is not None and cx["or_ci_high"] is not None
    # Cumulative cut-points present (K-1 = 2 thresholds).
    assert len(d["thresholds"]) == 2
    assert d["pseudo_r2"] is not None


def test_brant_supports_proportional_odds_when_true():
    # Data simulated WITH the proportional-odds structure (one shared slope) —
    # the Brant test should NOT reject.
    rng = np.random.default_rng(11)
    n = 1200
    x = rng.normal(0, 1, n)
    g = rng.integers(0, 2, n).astype(float)
    lin = 1.0 * x + 0.8 * g + rng.logistic(0, 1, n)
    y = np.digitize(lin, [-0.6, 0.8]) + 1  # 3 ordered categories 1/2/3
    df = pd.DataFrame({"stage": y, "x": x, "grp": g})
    store.save("ord_brant_ok", df)
    r = client.post("/api/models/ordinal", json={
        "session_id": "ord_brant_ok", "outcome": "stage", "predictors": ["x", "grp"],
    })
    assert r.status_code == 200, r.text
    brant = r.json()["brant_proportional_odds"]
    assert brant["computed"] is True
    assert brant["omnibus"]["violation"] is False, brant["omnibus"]
    assert {b["variable"] for b in brant["by_predictor"]} == {"x", "grp"}


def test_brant_flags_nonproportional_predictor():
    # x's effect differs across the two cumulative thresholds → Brant must
    # reject overall AND isolate x (not grp) as the violator.
    import scipy.special as sp
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.normal(0, 1, n)
    g = rng.integers(0, 2, n).astype(float)
    p_gt0 = sp.expit(0.5 + 0.3 * x + 0.5 * g)
    p_gt1 = sp.expit(-1.0 + 2.0 * x + 0.5 * g)
    y = (rng.uniform(size=n) < p_gt0).astype(int) + (rng.uniform(size=n) < p_gt1).astype(int) + 1
    df = pd.DataFrame({"stage": y, "x": x, "grp": g})
    store.save("ord_brant_bad", df)
    r = client.post("/api/models/ordinal", json={
        "session_id": "ord_brant_bad", "outcome": "stage", "predictors": ["x", "grp"],
    })
    assert r.status_code == 200, r.text
    brant = r.json()["brant_proportional_odds"]
    assert brant["computed"] is True
    assert brant["omnibus"]["violation"] is True, brant["omnibus"]
    by = {b["variable"]: b for b in brant["by_predictor"]}
    assert by["x"]["violation"] is True
    assert by["grp"]["violation"] is False


def test_ordinal_requires_three_categories():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"bin": rng.integers(0, 2, 50), "x": rng.normal(0, 1, 50)})
    store.save("ord_bin", df)
    r = client.post("/api/models/ordinal", json={
        "session_id": "ord_bin", "outcome": "bin", "predictors": ["x"],
    })
    assert r.status_code == 422
