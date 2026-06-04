"""
Time-horizon sensitivity Cox endpoint — /api/models/survival/cox_horizons.

Runs the same Cox model at several administrative-censoring windows and
returns per-horizon HR + CI + forest-ready rows. Verifies:
  - multi-horizon fit returns one forest row per window (+ full follow-up)
  - administrative censoring shrinks event counts at shorter horizons
  - a strong binary predictor yields HR > 1 with CI excluding 1
  - bad inputs (missing column, no horizons) → 4xx
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed(df: pd.DataFrame, sid: str) -> str:
    store.save(sid, df)
    return sid


def _make_survival(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Binary risk factor `grp` with a genuine elevated hazard."""
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 2, n)                      # 0/1 exposure
    # Higher hazard for grp==1 → shorter event times.
    base = rng.exponential(scale=800.0, size=n)
    dur = base * np.exp(-1.0 * grp)                  # HR ~ e^1 ≈ 2.7
    cens = rng.uniform(200, 2000, n)
    time = np.minimum(dur, cens)
    event = (dur <= cens).astype(int)
    age = rng.normal(60, 10, n)
    return pd.DataFrame({"time": time, "event": event, "grp": grp, "age": age})


def test_cox_horizons_basic_forest_rows():
    df = _make_survival(n=600, seed=1)
    sid = _seed(df, "coxh_basic")
    r = client.post("/api/models/survival/cox_horizons", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "predictor": "grp", "horizons": [365, 730],
        "horizon_labels": ["1 year", "2 years"], "include_full": True,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    # 2 horizons + full follow-up = 3 forest rows.
    assert len(d["forest_rows"]) == 3
    labels = [row["label"] for row in d["forest_rows"]]
    assert labels == ["1 year", "2 years", "Full follow-up"]
    for row in d["forest_rows"]:
        assert row["est"] is not None
        assert row["ci_low"] is not None and row["ci_high"] is not None
        assert row["ci_low"] <= row["est"] <= row["ci_high"]


def test_cox_horizons_event_counts_increase_with_window():
    df = _make_survival(n=600, seed=2)
    sid = _seed(df, "coxh_events")
    r = client.post("/api/models/survival/cox_horizons", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "predictor": "grp", "horizons": [365, 730], "include_full": True,
    })
    assert r.status_code == 200, r.text
    h = r.json()["horizons"]
    ev = [x["n_events"] for x in h]
    # Monotone non-decreasing: shorter window ≤ longer window ≤ full.
    assert ev[0] <= ev[1] <= ev[2]
    assert ev[0] < ev[2]   # full follow-up strictly more events than 1-year


def test_cox_horizons_strong_predictor_hr_excludes_one():
    df = _make_survival(n=700, seed=3)
    sid = _seed(df, "coxh_strong")
    r = client.post("/api/models/survival/cox_horizons", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "predictor": "grp", "covariates": ["age"], "horizons": [730],
    })
    assert r.status_code == 200, r.text
    full = r.json()["forest_rows"][-1]   # full follow-up row
    assert full["est"] > 1.0
    assert full["ci_low"] > 1.0          # CI excludes 1 → significant


def test_cox_horizons_bad_inputs():
    df = _make_survival(n=100, seed=4)
    sid = _seed(df, "coxh_bad")
    # Missing predictor column.
    r1 = client.post("/api/models/survival/cox_horizons", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "predictor": "nope", "horizons": [365],
    })
    assert r1.status_code == 422
    # No horizons.
    r2 = client.post("/api/models/survival/cox_horizons", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "predictor": "grp", "horizons": [],
    })
    assert r2.status_code == 422
