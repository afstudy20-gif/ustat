"""Tests for the improved subgroup bar chart (/api/charts/subgroup_bar):
global percentage target, t-distribution CI for means, Wilson CI for proportions.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session

BASE = "/api/charts/subgroup_bar"


@pytest.fixture(scope="module")
def sid():
    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame({
        "sex": rng.choice(["M", "F"], n),
        "arm": rng.choice(["A", "B"], n),
        "ef": rng.normal(55, 8, n),
        "died": rng.integers(0, 2, n),
    })
    return make_session(df, "sgbar_main")


def test_mean_mode_t_ci_symmetric(client, sid):
    r = client.post(BASE, json={"session_id": sid, "y_col": "ef", "subgroup_col": "sex",
                                "xaxis_col": "arm", "y_mode": "mean", "error_type": "ci"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "method_note" in b and "t-distribution" in b["method_note"]
    t = b["traces"][0]
    assert {"error_low", "error_high", "error", "ns"} <= set(t)
    # means → symmetric error bars
    assert all(abs(lo - hi) < 1e-9 for lo, hi in zip(t["error_low"], t["error_high"]))


def test_percentage_global_target_and_wilson_bounds(client, sid):
    r = client.post(BASE, json={"session_id": sid, "y_col": "died", "subgroup_col": "sex",
                                "xaxis_col": "arm", "y_mode": "percentage", "error_type": "ci"})
    assert r.status_code == 200, r.text
    b = r.json()
    # target resolved once, globally (not per cell)
    assert b["target_value"] == "1"
    for t in b["traces"]:
        for v, lo, hi in zip(t["y"], t["error_low"], t["error_high"]):
            assert 0.0 <= v <= 100.0
            # Wilson keeps the whole bar+error inside [0, 100]
            assert v - lo >= -1e-6 and v + hi <= 100.0 + 1e-6


def test_explicit_target_value(client, sid):
    r = client.post(BASE, json={"session_id": sid, "y_col": "died", "subgroup_col": "sex",
                                "xaxis_col": "arm", "y_mode": "percentage",
                                "target_value": "0", "error_type": "se"})
    assert r.status_code == 200, r.text
    assert r.json()["target_value"] == "0"


def test_missing_column_400(client, sid):
    r = client.post(BASE, json={"session_id": sid, "y_col": "nope", "subgroup_col": "sex",
                                "xaxis_col": "arm"})
    assert r.status_code == 400, r.text
