import numpy as np
import pandas as pd
import pytest

from conftest import make_session


def _row(body: dict, predictor: str) -> dict:
    for row in body["rows"]:
        if row["predictor"] == predictor:
            return row
    raise AssertionError(f"Missing predictor row: {predictor}")


@pytest.fixture
def multi_sid():
    rng = np.random.default_rng(123)
    n = 100
    age = rng.normal(45, 8, n)
    bmi = rng.normal(27, 4, n)
    pcos_num = rng.integers(0, 2, n)
    pcos = np.where(pcos_num == 1, "Yes", "No")
    y1 = 1.5 + 0.12 * age + 0.45 * bmi + 2.0 * pcos_num + rng.normal(0, 0.05, n)
    y2 = -4.0 + 1.8 * age + 1.2 * bmi - 3.0 * pcos_num + rng.normal(0, 0.05, n)
    df = pd.DataFrame({
        "EDE-Q": y1,
        "EES-C": y2,
        "Age": age,
        "BMI": bmi,
        "PCOS": pcos,
    })
    return make_session(df, "multi_outcome_main")


def test_two_outcomes_recovers_known_coefficients(client, multi_sid):
    r = client.post("/api/models/multi_outcome_regression", json={
        "session_id": multi_sid,
        "outcomes": ["EDE-Q", "EES-C"],
        "predictors": ["Age", "BMI", "PCOS"],
        "covariates": [],
        "standardize": True,
        "imputation": "listwise",
        "robust_se": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["test"] == "Multi-outcome linear regression"
    assert body["outcomes"] == ["EDE-Q", "EES-C"]
    assert body["n_by_outcome"] == {"EDE-Q": 100, "EES-C": 100}
    assert body["predictors_order"] == ["(Intercept)", "Age", "BMI", "PCOS_Yes"]

    age = _row(body, "Age")["by_outcome"]
    bmi = _row(body, "BMI")["by_outcome"]
    pcos = _row(body, "PCOS_Yes")["by_outcome"]

    assert age["EDE-Q"]["B"] == pytest.approx(0.12, abs=0.01)
    assert bmi["EDE-Q"]["B"] == pytest.approx(0.45, abs=0.01)
    assert pcos["EDE-Q"]["B"] == pytest.approx(2.0, abs=0.03)
    assert age["EES-C"]["B"] == pytest.approx(1.8, abs=0.01)
    assert bmi["EES-C"]["B"] == pytest.approx(1.2, abs=0.01)
    assert pcos["EES-C"]["B"] == pytest.approx(-3.0, abs=0.03)
    assert age["EDE-Q"]["beta"] is not None
    assert body["model_fit"]["EDE-Q"]["n"] == 100
    assert body["model_fit"]["EDE-Q"]["k"] == 3
    assert "significant predictors" in body["result_text"]


def test_single_outcome_works(client, multi_sid):
    r = client.post("/api/models/multi_outcome_regression", json={
        "session_id": multi_sid,
        "outcomes": ["EDE-Q"],
        "predictors": ["Age", "BMI"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcomes"] == ["EDE-Q"]
    assert body["n_by_outcome"] == {"EDE-Q": 100}
    assert set(body["model_fit"]) == {"EDE-Q"}
    assert _row(body, "Age")["by_outcome"]["EDE-Q"]["B"] is not None


def test_different_missingness_by_outcome_changes_n(client):
    rng = np.random.default_rng(321)
    n = 40
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y1 = 1 + 0.4 * x1 + 0.2 * x2 + rng.normal(0, 0.1, n)
    y2 = 2 - 0.6 * x1 + 0.3 * x2 + rng.normal(0, 0.1, n)
    y1[:3] = np.nan
    y2[:7] = np.nan
    sid = make_session(pd.DataFrame({"Y1": y1, "Y2": y2, "X1": x1, "X2": x2}), "multi_missing")

    r = client.post("/api/models/multi_outcome_regression", json={
        "session_id": sid,
        "outcomes": ["Y1", "Y2"],
        "predictors": ["X1", "X2"],
    })
    assert r.status_code == 200, r.text
    assert r.json()["n_by_outcome"] == {"Y1": 37, "Y2": 33}


def test_empty_outcomes_400(client, multi_sid):
    r = client.post("/api/models/multi_outcome_regression", json={
        "session_id": multi_sid,
        "outcomes": [],
        "predictors": ["Age"],
    })
    assert r.status_code == 400, r.text


def test_empty_predictors_400(client, multi_sid):
    r = client.post("/api/models/multi_outcome_regression", json={
        "session_id": multi_sid,
        "outcomes": ["EDE-Q"],
        "predictors": [],
    })
    assert r.status_code == 400, r.text


def test_binary_outcome_rejected_422(client):
    n = 30
    sid = make_session(pd.DataFrame({
        "Y": [0, 1] * 15,
        "X": np.linspace(0, 1, n),
    }), "multi_binary")
    r = client.post("/api/models/multi_outcome_regression", json={
        "session_id": sid,
        "outcomes": ["Y"],
        "predictors": ["X"],
    })
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "Continuous outcomes only"


def test_unknown_column_400(client, multi_sid):
    r = client.post("/api/models/multi_outcome_regression", json={
        "session_id": multi_sid,
        "outcomes": ["EDE-Q"],
        "predictors": ["Nope"],
    })
    assert r.status_code == 400, r.text
    assert "Unknown column" in r.json()["detail"]


def test_session_not_found_404(client):
    r = client.post("/api/models/multi_outcome_regression", json={
        "session_id": "does-not-exist",
        "outcomes": ["Y"],
        "predictors": ["X"],
    })
    assert r.status_code == 404, r.text
