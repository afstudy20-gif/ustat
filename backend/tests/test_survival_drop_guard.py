from pathlib import Path


COHORT = Path(__file__).resolve().parents[2] / "qa" / "cohort_test.csv"


def _upload_cohort(client):
    with COHORT.open("rb") as f:
        r = client.post("/api/upload/", files={"file": ("cohort_test.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _has_dropped_fu_warning(body: dict) -> bool:
    return any("row 100 dropped: non-positive fu_days" in w for w in body.get("warnings", []))


def test_km_drops_negative_duration_with_warning(client):
    sid = _upload_cohort(client)
    r = client.post("/api/models/survival/km", json={
        "session_id": sid,
        "duration_col": "fu_days",
        "event_col": "event",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["groups"][0]["n"] == 99
    assert d["n_invalid_survival"] == 1
    assert _has_dropped_fu_warning(d)


def test_cox_and_diagnostics_share_negative_duration_policy(client):
    sid = _upload_cohort(client)
    payload = {
        "session_id": sid,
        "duration_col": "fu_days",
        "event_col": "event",
        "predictors": ["age", "ldl", "nyha"],
    }
    cox = client.post("/api/models/survival/cox", json=payload)
    assert cox.status_code == 200, cox.text
    cox_d = cox.json()
    assert cox_d["n_invalid_survival"] == 1
    assert _has_dropped_fu_warning(cox_d)

    diag = client.post("/api/model_diagnostics/cox_diagnostics", json=payload)
    assert diag.status_code == 200, diag.text
    diag_d = diag.json()
    assert diag_d["n"] == cox_d["n_analyzed"]
    assert diag_d["n_invalid_survival"] == 1
    assert _has_dropped_fu_warning(diag_d)


def test_rmst_and_landmark_drop_negative_duration_with_warning(client):
    sid = _upload_cohort(client)
    rmst = client.post("/api/survival_advanced/rmst", json={
        "session_id": sid,
        "duration_col": "fu_days",
        "event_col": "event",
        "tau": 365,
    })
    assert rmst.status_code == 200, rmst.text
    assert rmst.json()["n_invalid_survival"] == 1
    assert _has_dropped_fu_warning(rmst.json())

    landmark = client.post("/api/survival_advanced/landmark", json={
        "session_id": sid,
        "duration_col": "fu_days",
        "event_col": "event",
        "landmark_time": 30,
        "group_col": "diabetes",
    })
    assert landmark.status_code == 200, landmark.text
    assert landmark.json()["n_invalid_survival"] == 1
    assert _has_dropped_fu_warning(landmark.json())


def test_cox_horizons_and_rcs_drop_negative_duration_with_warning(client):
    sid = _upload_cohort(client)
    horizons = client.post("/api/models/survival/cox_horizons", json={
        "session_id": sid,
        "duration_col": "fu_days",
        "event_col": "event",
        "predictor": "diabetes",
        "covariates": ["age", "ldl"],
        "horizons": [365],
    })
    assert horizons.status_code == 200, horizons.text
    assert horizons.json()["n_invalid_survival"] == 1
    assert _has_dropped_fu_warning(horizons.json())

    rcs = client.post("/api/models/survival/cox_rcs", json={
        "session_id": sid,
        "duration_col": "fu_days",
        "event_col": "event",
        "spline_terms": [{"column": "age", "n_knots": 3}],
        "covariates": ["ldl"],
    })
    assert rcs.status_code == 200, rcs.text
    assert rcs.json()["n_invalid_survival"] == 1
    assert _has_dropped_fu_warning(rcs.json())
