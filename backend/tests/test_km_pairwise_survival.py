"""
KM extras: landmark survival-at-time + pairwise log-rank.

Reproduces the reporting pattern: overall log-rank across 3 groups,
N-year survival per group, and pairwise comparisons identifying which
group pair drives the difference.
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


def _make_three_groups(seed: int = 0) -> pd.DataFrame:
    """3 LDL groups; group 0 has worse survival than 1 and 2 (which match)."""
    rng = np.random.default_rng(seed)
    rows = []
    # group 0: high hazard (short times), groups 1 & 2: similar low hazard
    specs = {0: 280.0, 1: 900.0, 2: 850.0}
    for g, scale in specs.items():
        n = 160
        base = rng.exponential(scale=scale, size=n)
        cens = rng.uniform(400, 2200, n)
        time = np.minimum(base, cens)
        event = (base <= cens).astype(int)
        for t, e in zip(time, event):
            rows.append({"time": t, "event": e, "ldl_grp": g})
    return pd.DataFrame(rows)


def test_km_survival_at_time():
    df = _make_three_groups(seed=1)
    sid = _seed(df, "km_survat")
    r = client.post("/api/models/survival/km", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "group_col": "ldl_grp", "survival_times": [365, 1825],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["groups"]) == 3
    for g in d["groups"]:
        sa = g["survival_at"]
        assert len(sa) == 2
        for pt in sa:
            assert 0.0 <= pt["survival"] <= 1.0
            assert pt["ci_low"] is None or pt["ci_low"] <= pt["survival"] + 1e-6
    # Worst group (0) should have lower 5-year survival than the others.
    by_grp = {g["group"]: g["survival_at"][1]["survival"] for g in d["groups"]}
    assert by_grp["0"] < by_grp["1"]
    assert by_grp["0"] < by_grp["2"]


def test_km_pairwise_logrank_identifies_driver():
    df = _make_three_groups(seed=2)
    sid = _seed(df, "km_pairwise")
    r = client.post("/api/models/survival/km", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "group_col": "ldl_grp", "pairwise": True, "pairwise_correction": "holm",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    # Overall multivariate log-rank present + significant.
    assert d["logrank"]["p"] < 0.05
    pw = d["pairwise"]["comparisons"]
    assert len(pw) == 3  # 3 choose 2
    for c in pw:
        assert "p_adj" in c
    # 0-vs-1 and 0-vs-2 significant; 1-vs-2 not.
    def find(a, b):
        for c in pw:
            if {c["group_a"], c["group_b"]} == {a, b}:
                return c
        raise AssertionError("pair missing")
    assert find("0", "1")["p"] < 0.05
    assert find("0", "2")["p"] < 0.05
    assert find("1", "2")["p"] > 0.05


def test_km_risk_table():
    df = _make_three_groups(seed=5)
    sid = _seed(df, "km_risk")
    r = client.post("/api/models/survival/km", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "group_col": "ldl_grp", "risk_times": [0, 365, 730, 1095],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["risk_times"] == [0, 365, 730, 1095]
    for g in d["groups"]:
        ar = g["at_risk"]
        assert len(ar) == 4
        # at-risk at t=0 equals group n; monotone non-increasing over time.
        assert ar[0] == g["n"]
        assert ar[0] >= ar[1] >= ar[2] >= ar[3]


def test_km_pairwise_skipped_for_two_groups():
    df = _make_three_groups(seed=3)
    df = df[df["ldl_grp"] != 2]  # leave 2 groups
    sid = _seed(df, "km_twogrp")
    r = client.post("/api/models/survival/km", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "group_col": "ldl_grp", "pairwise": True,
    })
    assert r.status_code == 200, r.text
    # pairwise only meaningful with >=3 groups → None for 2.
    assert r.json()["pairwise"] is None
