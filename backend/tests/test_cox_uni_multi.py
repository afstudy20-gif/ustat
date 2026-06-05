"""
Paired unadjusted (univariable) vs adjusted (multivariable) Cox HR endpoint —
backs the publication 'Figure 4' forest plot.
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed() -> str:
    rng = np.random.default_rng(3)
    n = 300
    age = rng.normal(60, 10, n)
    grp = rng.integers(1, 4, n)  # 1,2,3
    base = 0.02 * (age - 60) + 0.3 * (grp == 3)
    time = rng.exponential(scale=np.exp(-base) * 500).clip(1, 1800)
    event = (time < 1200).astype(int)
    df = pd.DataFrame({"time": time, "event": event, "age": age, "ldl_grp": grp})
    sid = "cox_um"
    store.save(sid, df)
    # Mark ldl_grp categorical so it expands to contrast rows (vs reference).
    store.save_kind_overrides(sid, {"ldl_grp": "categorical"})
    return sid


def test_cox_uni_multi_shape():
    sid = _seed()
    r = client.post("/api/models/survival/cox_uni_multi", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "predictors": ["ldl_grp", "age"],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n"] > 0 and d["n_events"] > 0
    rows = d["rows"]
    # ldl_grp -> 2 contrast rows (vs ref 1); age -> 1 numeric row
    terms = [row["term"] for row in rows]
    assert "age" in terms
    assert "ldl_grp=2" in terms and "ldl_grp=3" in terms
    # category rows carry reference for label building
    cat_rows = [row for row in rows if row["kind"] == "category"]
    assert all(row["reference"] == "1" for row in cat_rows)
    # both passes populated
    for row in rows:
        assert row["unadjusted"] is not None
        assert "hr" in row["unadjusted"]
        assert row["adjusted"] is not None


def test_cox_uni_multi_rejects_non_binary_event():
    sid = _seed()
    r = client.post("/api/models/survival/cox_uni_multi", json={
        "session_id": sid, "duration_col": "time", "event_col": "age",
        "predictors": ["ldl_grp"],
    })
    assert r.status_code == 422
