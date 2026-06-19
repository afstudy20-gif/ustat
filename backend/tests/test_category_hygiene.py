import pandas as pd

from conftest import make_session


def _dirty_sex_session(name: str = "category_hygiene"):
    df = pd.DataFrame({
        "age": [61, 58, 63, 55, 70, 66, 52, 59, 64, 57, 62, 68],
        "bmi": [27.1, 25.4, 28.2, 24.9, 31.0, 29.4, 26.5, 23.8, 27.9, 25.1, 30.2, 28.7],
        "sbp": [130, 124, 138, 118, 145, 136, 122, 119, 132, 126, 140, 134],
        "event": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        "sex": ["M", "F", "Female", "x", None, "M", "F", "M", "F", "M", "F", "M"],
        "diabetes": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        "nyha": [1, 2, 3, 2, 4, 3, 1, 2, 3, 2, 4, 3],
        "patient_id": list(range(12)),
    })
    return make_session(df, name)


def test_two_group_stats_normalize_dirty_binary_group(client):
    sid = _dirty_sex_session("category_hygiene_stats")

    ttest = client.post("/api/stats/ttest", json={
        "session_id": sid, "column": "age", "group_column": "sex",
    })
    assert ttest.status_code == 200, ttest.text
    assert {ttest.json()["group1"], ttest.json()["group2"]} == {"Female", "Male"}
    assert ttest.json()["warnings"][0]["dropped_levels"][0]["level"] == "x"

    mw = client.post("/api/stats/mannwhitney", json={
        "session_id": sid, "column": "bmi", "group_column": "sex",
    })
    assert mw.status_code == 200, mw.text
    assert {mw.json()["group1"], mw.json()["group2"]} == {"Female", "Male"}


def test_chisquare_cleans_dirty_sex_levels_before_crosstab(client):
    sid = _dirty_sex_session("category_hygiene_chisq")
    r = client.post("/api/stats/chisquare", json={
        "session_id": sid, "row_column": "diabetes", "col_column": "sex",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["crosstab"].keys()) == {"Female", "Male"}
    assert body["warnings"][0]["dropped_levels"][0]["level"] == "x"


def test_equivalence_noninferiority_and_two_prop_accept_dirty_sex(client):
    sid = _dirty_sex_session("category_hygiene_equiv")
    tost = client.post("/api/stats/tost", json={
        "session_id": sid, "column": "sbp", "group_column": "sex",
        "low": -10, "high": 10, "test_type": "independent",
    })
    assert tost.status_code == 200, tost.text
    assert set(tost.json()["group_labels"]) == {"Female", "Male"}

    ni = client.post("/api/stats/noninferiority", json={
        "session_id": sid, "outcome_col": "event", "group_col": "sex",
        "outcome_type": "binary", "effect": "RR", "margin": 2.0,
    })
    assert ni.status_code == 200, ni.text
    assert {ni.json()["test_group"], ni.json()["ref_group"]} == {"Female", "Male"}

    prop = client.post("/api/categorical/two_proportions", json={
        "session_id": sid, "column": "event", "group_column": "sex",
    })
    assert prop.status_code == 200, prop.text
    assert set(prop.json()["summary"].keys()) >= {"Female", "Male"}


def test_bayesian_ttest_and_table1_use_same_clean_group_labels(client):
    sid = _dirty_sex_session("category_hygiene_bayes_table")
    bayes = client.post("/api/bayesian", json={
        "session_id": sid, "analysis_type": "ttest_ind",
        "outcome": "age", "predictor": "sex",
    })
    assert bayes.status_code == 200, bayes.text
    assert bayes.json()["n"] == 10

    table = client.post("/api/stats/table1", json={
        "session_id": sid, "group_column": "sex",
        "variables": ["age", "diabetes", "nyha"],
        "variable_kinds": {"age": "numeric", "diabetes": "categorical", "nyha": "categorical"},
    })
    assert table.status_code == 200, table.text
    assert table.json()["group_labels"] == ["Female", "Male"]
    assert "x" not in table.json()["group_ns"]


def test_glm_and_stepwise_handle_dirty_binary_predictor(client):
    sid = _dirty_sex_session("category_hygiene_models")

    poisson = client.post("/api/models/poisson", json={
        "session_id": sid, "outcome": "event", "predictors": ["sex", "age"],
    })
    assert poisson.status_code == 200, poisson.text
    assert poisson.json()["n_excluded"] == 2
    assert poisson.json()["warnings"][0]["dropped_levels"][0]["level"] == "x"

    ordinal = client.post("/api/models/ordinal", json={
        "session_id": sid, "outcome": "nyha", "predictors": ["sex", "age"],
    })
    assert ordinal.status_code == 200, ordinal.text
    assert ordinal.json()["brant_proportional_odds"]

    stepwise = client.post("/api/models/stepwise", json={
        "session_id": sid, "model_type": "logistic", "outcome": "event",
        "candidates": ["sex", "age", "bmi"], "direction": "forward",
    })
    assert stepwise.status_code == 200, stepwise.text
    assert stepwise.json()["n_excluded"] == 2
