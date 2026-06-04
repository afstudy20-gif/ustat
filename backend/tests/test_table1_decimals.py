"""
Per-column decimal display for /api/stats/table1 + /api/stats/descriptive.

Industry convention (AMA / ICMJE) is that Table 1 statistics inherit the
precision of the source variable. Follow-up days are reported as integers,
weights in kg as 1–2 decimals. This module verifies that:

* Integer-dtype columns render with 0 decimals.
* Float columns whose values are all whole numbers (e.g. SPSS-imported
  day-counts as float64) also render with 0 decimals.
* True float columns keep the legacy 2-decimal default.
* Explicit per-column overrides take precedence over auto-detection.
* `/api/stats/descriptive` surfaces a `display_decimals` hint per column
  for the frontend to read.
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed(df: pd.DataFrame, session_id: str) -> str:
    store.save(session_id, df)
    return session_id


# ── synthetic dataset ────────────────────────────────────────────────────

def _make_dataset(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 200
    return pd.DataFrame({
        "fu_days_int": rng.integers(low=6, high=1827, size=n),                # int dtype
        "fu_days_float": rng.integers(low=6, high=1827, size=n).astype(float),  # float but whole
        "weight_kg": rng.normal(loc=73.4, scale=12.0, size=n).round(2),       # true float
        "group": rng.choice(["A", "B"], size=n),
    })


# ─────────────────────────────────────────────────────────────────────────
# Table 1 — auto integer detection
# ─────────────────────────────────────────────────────────────────────────

def test_table1_integer_dtype_renders_without_decimals():
    df = _make_dataset(seed=1)
    sid = _seed(df, "dec_int")
    r = client.post("/api/stats/table1", json={
        "session_id": sid,
        "variables": ["fu_days_int"],
        "selected_stats": ["median_iqr"],
    })
    assert r.status_code == 200, r.text
    overall = r.json()["rows"][0]["overall"]
    # No decimal point should appear anywhere in the rendered value.
    assert "." not in overall, f"integer column rendered with decimals: {overall!r}"
    assert "[" in overall and "]" in overall  # still has IQR brackets


def test_table1_float_holding_integers_also_integer_displayed():
    df = _make_dataset(seed=2)
    sid = _seed(df, "dec_float_int")
    r = client.post("/api/stats/table1", json={
        "session_id": sid,
        "variables": ["fu_days_float"],
        "selected_stats": ["median_iqr"],
    })
    assert r.status_code == 200, r.text
    overall = r.json()["rows"][0]["overall"]
    assert "." not in overall, f"whole-valued float column should still be integer-displayed: {overall!r}"


def test_table1_true_float_keeps_two_decimals():
    df = _make_dataset(seed=3)
    sid = _seed(df, "dec_true_float")
    r = client.post("/api/stats/table1", json={
        "session_id": sid,
        "variables": ["weight_kg"],
        "selected_stats": ["mean_sd"],
    })
    assert r.status_code == 200, r.text
    overall = r.json()["rows"][0]["overall"]
    # Should have something like "73.40 ± 11.97" — every numeric component
    # should carry two decimals.
    parts = overall.replace("±", " ").split()
    decimals_carrying = [p for p in parts if "." in p]
    assert len(decimals_carrying) == 2
    for p in decimals_carrying:
        # Each numeric token should have exactly two digits after the dot.
        assert len(p.split(".")[1]) == 2, f"weight_kg lost precision: {overall!r}"


# ─────────────────────────────────────────────────────────────────────────
# Table 1 — explicit overrides
# ─────────────────────────────────────────────────────────────────────────

def test_table1_request_override_wins_over_autodetect():
    df = _make_dataset(seed=4)
    sid = _seed(df, "dec_override")
    r = client.post("/api/stats/table1", json={
        "session_id": sid,
        "variables": ["fu_days_int"],
        "selected_stats": ["median_iqr"],
        "column_decimals": {"fu_days_int": 1},
    })
    assert r.status_code == 200, r.text
    overall = r.json()["rows"][0]["overall"]
    # Each numeric token should now have exactly one digit after the dot.
    for p in overall.replace("[", " ").replace("]", " ").replace("–", " ").split():
        if any(ch.isdigit() for ch in p) and "." in p:
            assert len(p.split(".")[1]) == 1, f"override ignored: {overall!r}"


def test_table1_session_persisted_override_honoured():
    df = _make_dataset(seed=5)
    sid = _seed(df, "dec_session_override")
    # Persist a per-session override the same way the UI does it.
    r1 = client.post(f"/api/sessions/{sid}/decimals", json={
        "column": "fu_days_int", "decimals": 2,
    })
    assert r1.status_code == 200, r1.text
    r2 = client.post("/api/stats/table1", json={
        "session_id": sid,
        "variables": ["fu_days_int"],
        "selected_stats": ["median_iqr"],
    })
    assert r2.status_code == 200, r2.text
    overall = r2.json()["rows"][0]["overall"]
    for p in overall.replace("[", " ").replace("]", " ").replace("–", " ").split():
        if any(ch.isdigit() for ch in p) and "." in p:
            assert len(p.split(".")[1]) == 2, f"session override ignored: {overall!r}"


# ─────────────────────────────────────────────────────────────────────────
# /api/stats/descriptive surface
# ─────────────────────────────────────────────────────────────────────────

def test_descriptive_endpoint_exposes_display_decimals():
    df = _make_dataset(seed=6)
    sid = _seed(df, "dec_descriptive")
    r = client.get(f"/api/stats/{sid}/descriptive")
    assert r.status_code == 200, r.text
    data = r.json()
    # Integer + integer-valued-float should both auto-resolve to 0.
    assert data["fu_days_int"]["display_decimals"] == 0
    assert data["fu_days_float"]["display_decimals"] == 0
    # True float column keeps the default 2.
    assert data["weight_kg"]["display_decimals"] == 2
