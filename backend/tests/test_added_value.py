"""Tests for /api/model_compare/added_value — incremental predictive value
(ΔAUC via DeLong, NRI, IDI, nested LR, calibration)."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session

BASE = "/api/model_compare/added_value"


@pytest.fixture(scope="module")
def sid():
    rng = np.random.default_rng(3)
    n = 300
    age = rng.normal(60, 10, n)
    sex = rng.integers(0, 2, n)
    biomarker = rng.normal(0, 1, n)          # strongly predictive
    noise = rng.normal(0, 1, n)              # useless
    lp = -2 + 0.04 * age + 0.3 * sex + 0.9 * biomarker
    p = 1 / (1 + np.exp(-lp))
    event = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({"age": age, "sex": sex, "biomarker": biomarker, "noise": noise, "event": event})
    return make_session(df, "addedval")


def test_strong_predictor_adds_value(client, sid):
    r = client.post(BASE, json={
        "session_id": sid, "outcome": "event",
        "base_predictors": ["age", "sex"], "new_predictors": ["biomarker"], "bootstrap": 200,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["added_value"] is True
    assert d["discrimination"]["delta_auc"] > 0.02
    assert d["discrimination"]["significant"] is True
    assert d["reclassification"]["idi"] > 0
    assert d["reclassification"]["idi_ci"][0] > 0          # CI excludes 0
    assert d["fit"]["lr_p"] < 0.05 and d["fit"]["delta_aic"] < 0
    assert "calibration" in d and "preserved" in d["calibration"]
    assert "predictive value" in d["result_text"]


def test_useless_predictor_no_value(client, sid):
    r = client.post(BASE, json={
        "session_id": sid, "outcome": "event",
        "base_predictors": ["age", "sex", "biomarker"], "new_predictors": ["noise"], "bootstrap": 200,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["added_value"] is False
    assert abs(d["discrimination"]["delta_auc"]) < 0.02


def test_cv_mode(client, sid):
    r = client.post(BASE, json={
        "session_id": sid, "outcome": "event",
        "base_predictors": ["age", "sex"], "new_predictors": ["biomarker"],
        "cv_folds": 5, "bootstrap": 0,
    })
    assert r.status_code == 200, r.text
    assert "cross-validated" in r.json()["prediction_basis"]


def test_new_already_in_base_400(client, sid):
    r = client.post(BASE, json={
        "session_id": sid, "outcome": "event",
        "base_predictors": ["age", "biomarker"], "new_predictors": ["age"], "bootstrap": 0,
    })
    assert r.status_code == 400, r.text


def test_non_logistic_rejected(client, sid):
    r = client.post(BASE, json={
        "session_id": sid, "outcome": "event", "base_predictors": ["age"],
        "new_predictors": ["biomarker"], "model_type": "linear",
    })
    assert r.status_code == 400, r.text
