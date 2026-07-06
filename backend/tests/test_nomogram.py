"""Tests for routers/nomogram.py.

Covers POST /api/nomogram/build for both logistic and Cox model types:
  * logistic nomogram happy path (predictors_table, probability_mapping, model_summary)
  * cox nomogram happy path (requires a duration column alongside the event column)
  * error paths: missing session, unknown columns, invalid model_type,
    non-binary outcome for logistic, missing duration column for cox,
    insufficient data after listwise deletion.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session

SEED = 20240707


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(SEED)
    n = 300
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    dm = rng.integers(0, 2, n)
    logit_p = -5 + 0.05 * age + 0.01 * ldl + 0.6 * dm
    p = 1 / (1 + np.exp(-logit_p))
    event = (rng.uniform(0, 1, n) < p).astype(int)

    # Cox: event indicator + companion "<outcome>_time" duration column.
    base_hazard = 0.01 * np.exp(0.03 * (age - 60) + 0.4 * dm)
    time = rng.exponential(1 / base_hazard).clip(0.1, 60)

    bad_outcome = rng.integers(2, 5, n)  # 3 unique values, non-binary, for the binary-check error path

    return pd.DataFrame({
        "AGE": age,
        "LDL": ldl,
        "DM": dm,
        "event": event,
        "event_time": time,
        "bad_outcome": bad_outcome,
    })


@pytest.fixture(scope="module")
def sid(synth):
    return make_session(synth, "nomogram_main")


# ── logistic nomogram ─────────────────────────────────────────────────────────

def test_logistic_nomogram_happy_path(client, sid):
    r = client.post("/api/nomogram/build", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE", "LDL", "DM"],
        "model_type": "logistic",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_type"] == "logistic"
    assert body["test"] == "Nomogram"

    # predictors_table: one entry per predictor, each with points/scale info
    table = body["predictors_table"]
    assert isinstance(table, list) and len(table) == 3
    names = {row["predictor"] for row in table}
    assert names == {"AGE", "LDL", "DM"}
    for row in table:
        assert isinstance(row["values"], list) and len(row["values"]) >= 5
        assert isinstance(row["points"], list) and len(row["points"]) == len(row["values"])
        assert row["max_points"] >= 0
        assert "coefficient" in row and "reference" in row

    # total-points-to-outcome mapping
    prob_map = body["probability_mapping"]
    assert isinstance(prob_map, list) and len(prob_map) == 20
    for entry in prob_map:
        assert "total_points" in entry
        assert 0.0 <= entry["probability"] <= 1.0
    # monotonically non-decreasing probability with increasing total points
    points_sorted = [e["total_points"] for e in prob_map]
    assert points_sorted == sorted(points_sorted)

    assert len(body["total_points_range"]) == 2
    assert body["total_points_range"][0] <= body["total_points_range"][1]

    summary = body["model_summary"]
    assert summary["n"] > 0
    assert 0.0 <= summary["c_statistic"] <= 1.0
    assert isinstance(summary["aic"], float)

    assert isinstance(body["result_text"], str) and "Logistic regression nomogram" in body["result_text"]
    assert "rms" in body["r_code"]


def test_logistic_nomogram_non_binary_outcome(client, sid):
    r = client.post("/api/nomogram/build", json={
        "session_id": sid, "outcome": "bad_outcome",
        "predictors": ["AGE", "LDL"],
        "model_type": "logistic",
    })
    assert r.status_code == 400
    assert "binary" in r.json()["detail"].lower()


# ── cox nomogram ──────────────────────────────────────────────────────────────

def test_cox_nomogram_happy_path(client, sid):
    r = client.post("/api/nomogram/build", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE", "DM"],
        "model_type": "cox",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_type"] == "cox"

    table = body["predictors_table"]
    assert isinstance(table, list) and len(table) == 2
    names = {row["predictor"] for row in table}
    assert names == {"AGE", "DM"}
    for row in table:
        assert isinstance(row["points"], list) and len(row["points"]) == len(row["values"])

    prob_map = body["probability_mapping"]
    assert isinstance(prob_map, list) and len(prob_map) == 20
    for entry in prob_map:
        assert "total_points" in entry
        assert 0.0 <= entry["probability"] <= 1.0

    summary = body["model_summary"]
    assert summary["n"] > 0
    assert 0.0 <= summary["c_statistic"] <= 1.0

    assert "Cox regression nomogram" in body["result_text"]
    assert "cph(Surv(" in body["r_code"]


def test_cox_nomogram_missing_duration_column(client):
    # No "<outcome>_time"/"time"/"duration" companion column present.
    df = pd.DataFrame({
        "AGE": np.linspace(20, 90, 50),
        "outcome_only": np.tile([0, 1], 25),
    })
    sid2 = make_session(df, "nomogram_no_duration")
    r = client.post("/api/nomogram/build", json={
        "session_id": sid2, "outcome": "outcome_only",
        "predictors": ["AGE"],
        "model_type": "cox",
    })
    assert r.status_code == 400
    assert "duration" in r.json()["detail"].lower()


# ── error paths shared by both model types ────────────────────────────────────

def test_nomogram_session_not_found(client):
    r = client.post("/api/nomogram/build", json={
        "session_id": "does_not_exist", "outcome": "event",
        "predictors": ["AGE"],
        "model_type": "logistic",
    })
    assert r.status_code == 404


def test_nomogram_unknown_columns(client, sid):
    r = client.post("/api/nomogram/build", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["NOT_A_COLUMN"],
        "model_type": "logistic",
    })
    assert r.status_code == 400
    assert "NOT_A_COLUMN" in r.json()["detail"]


def test_nomogram_invalid_model_type(client, sid):
    r = client.post("/api/nomogram/build", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE"],
        "model_type": "random_forest",
    })
    assert r.status_code == 400
    assert "model_type" in r.json()["detail"]


def test_nomogram_insufficient_data(client):
    # Only a handful of rows -> below the "predictors + 10" minimum threshold.
    df = pd.DataFrame({
        "AGE": [20, 30, 40, 50, 60],
        "event": [0, 1, 0, 1, 0],
    })
    sid3 = make_session(df, "nomogram_tiny")
    r = client.post("/api/nomogram/build", json={
        "session_id": sid3, "outcome": "event",
        "predictors": ["AGE"],
        "model_type": "logistic",
    })
    assert r.status_code == 400
    assert "insufficient" in r.json()["detail"].lower()
