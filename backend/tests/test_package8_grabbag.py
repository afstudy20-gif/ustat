import numpy as np
import pandas as pd

from conftest import make_session


def test_mediation_rejects_binary_outcome(client):
    sid = make_session(pd.DataFrame({
        "y": [0, 1] * 20,
        "x": np.linspace(0, 1, 40),
        "m": np.linspace(1, 2, 40),
    }), "pkg8_mediation")
    r = client.post("/api/causal/mediation", json={
        "session_id": sid, "outcome": "y", "treatment": "x", "mediator": "m",
        "bootstrap": 0,
    })
    assert r.status_code == 422
    assert "continuous outcome" in r.json()["detail"]


def test_power_logistic_and_cox_solve_effect_size(client):
    r1 = client.post("/api/stats/power", json={
        "test": "logistic", "solve_for": "effect_size",
        "n": 300, "power": 0.8, "p_event": 0.3,
    })
    assert r1.status_code == 200, r1.text
    assert r1.json()["result"] > 1

    r2 = client.post("/api/stats/power", json={
        "test": "survival_cox", "solve_for": "effect_size",
        "n": 300, "power": 0.8, "event_rate": 0.35, "p_exposed": 0.5,
    })
    assert r2.status_code == 200, r2.text
    assert r2.json()["result"] > 1


def test_forest_and_meta_reject_nonpositive_log_scale_effects(client):
    forest = client.post("/api/charts/forest", json={
        "x_axis": "log",
        "rows": [{"label": "bad", "est": -1, "ci_low": 0.5, "ci_high": 2.0}],
    })
    assert forest.status_code == 422

    meta = client.post("/api/meta/analyze", json={
        "measure": "OR",
        "studies": [
            {"label": "bad", "effect": -1, "ci_low": 0.5, "ci_high": 2.0},
            {"label": "ok", "effect": 1.2, "ci_low": 1.0, "ci_high": 1.5},
        ],
    })
    assert meta.status_code == 422


def test_evalue_legacy_endpoint_includes_consistent_aliases(client):
    r = client.post("/api/survival_advanced/evalue", json={
        "estimate": 2.0, "ci_low": 1.2, "ci_high": 3.0,
        "measure_type": "RR", "baseline_risk": 0.1,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["evalue_point"] == body["e_value_point_estimate"]
    assert body["evalue_ci"] == body["e_value_ci"]


def test_logistic_or_table_reports_row_total_not_predictor_count(client):
    n = 60
    sid = make_session(pd.DataFrame({
        "y": [0, 1] * (n // 2),
        "x1": np.linspace(0, 1, n),
        "x2": np.tile([0, 1, 1], n // 3),
    }), "pkg8_or_table")
    r = client.post("/api/models/logistic_table", json={
        "session_id": sid, "outcome": "y", "predictors": ["x1", "x2"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_total"] == n
    assert body["n_predictors"] == 2


def test_iptw_warns_on_extreme_weights(client):
    n = 200
    z = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    treat = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    treat[0] = 1
    df = pd.DataFrame({"treat": treat, "z": z, "y": np.tile([0, 1], n // 2)})
    sid = make_session(df, "pkg8_iptw")
    r = client.post("/api/models/iptw", json={
        "session_id": sid, "treatment_col": "treat",
        "covariates": ["z"], "outcome_col": "y",
        "stabilize": False,
    })
    assert r.status_code == 200, r.text
    assert any("Extreme IPTW weight" in w for w in r.json()["warnings"])
