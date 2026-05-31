"""Coverage tests for routers/survival_advanced.py.

Covers the /api/survival_advanced/* endpoints: mice, fine_gray, evalue,
landmark, rmst, recurrent_lwyy, survival_validation, discrete_time.
Focuses on gaps + edge cases not covered by test_v2_endpoints.py.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

SEED = 20260531
PREFIX = "/api/survival_advanced"


@pytest.fixture(scope="module")
def surv_df():
    """Synthetic survival dataset with a competing-risks event column."""
    rng = np.random.default_rng(SEED)
    n = 240
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    sex = rng.integers(0, 2, n)
    dm = rng.integers(0, 2, n)
    duration = rng.exponential(500, n).clip(1, 1825)
    event = (rng.uniform(0, 1, n) < 0.45).astype(int)

    # 3-level competing-risks event: 0=censored, 1=event of interest, 2=competing
    u = rng.uniform(0, 1, n)
    comp_event = np.where(u < 0.35, 1, np.where(u < 0.6, 2, 0)).astype(int)

    # A column with missing values for MICE
    chol = rng.normal(200, 40, n)
    miss_mask = rng.uniform(0, 1, n) < 0.2
    chol[miss_mask] = np.nan

    df = pd.DataFrame({
        "AGE": age,
        "LDL": ldl,
        "SEX": sex,
        "DM": dm,
        "duration": duration,
        "event": event,
        "comp_event": comp_event,
        "CHOL": chol,
    })
    return df


@pytest.fixture(scope="module")
def sid(surv_df):
    return make_session(surv_df, "tsadv_main")


@pytest.fixture(scope="module")
def recur_df():
    """Counting-process recurrent-event dataset (multiple intervals/subject)."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for subj in range(60):
        trt = int(subj % 2)
        age = float(rng.normal(60, 10))
        n_events = rng.integers(1, 5)
        start = 0.0
        for _k in range(n_events):
            gap = float(rng.exponential(100) + 5)
            stop = start + gap
            ev = int(rng.uniform(0, 1) < 0.7)
            rows.append({
                "subj_id": subj,
                "start": start,
                "stop": stop,
                "rec_event": ev,
                "TRT": trt,
                "AGE": age,
            })
            start = stop
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sid_recur(recur_df):
    return make_session(recur_df, "tsadv_recur")


# ── /mice ────────────────────────────────────────────────────────────────────
# NOTE: MICE persists the imputed DataFrame back to the session (store.save),
# so each MICE test uses its own dedicated session to stay order-independent.

def test_mice_happy_path(client, surv_df):
    s = make_session(surv_df.copy(), "tsadv_mice_happy")
    r = client.post(f"{PREFIX}/mice", json={
        "session_id": s,
        "columns": ["CHOL"],
        "n_imputations": 2,
        "max_iter": 3,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "MICE Multiple Imputation"
    assert d["total_imputed"] > 0
    assert len(d["columns"]) == 1
    assert d["columns"][0]["column"] == "CHOL"
    assert d["columns"][0]["n_imputed"] > 0
    assert "assumptions" in d and "r_code" in d


def test_mice_no_missing_returns_422(client, sid):
    # AGE has no missing values
    r = client.post(f"{PREFIX}/mice", json={
        "session_id": sid,
        "columns": ["AGE"],
        "n_imputations": 2,
    })
    assert r.status_code == 422, r.text


def test_mice_missing_column_400(client, sid):
    r = client.post(f"{PREFIX}/mice", json={
        "session_id": sid,
        "columns": ["NOPE"],
    })
    assert r.status_code == 400, r.text


def test_mice_mnar_mechanism_flagged(client, surv_df):
    s = make_session(surv_df.copy(), "tsadv_mice_mnar")
    r = client.post(f"{PREFIX}/mice", json={
        "session_id": s,
        "columns": ["CHOL"],
        "n_imputations": 2,
        "max_iter": 3,
        "mechanism": "MNAR",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    mech_assumption = next(a for a in d["assumptions"] if a["name"] == "Missing mechanism")
    assert mech_assumption["met"] is False


# ── /fine_gray ───────────────────────────────────────────────────────────────

def test_fine_gray_cif_only(client, sid):
    """No predictors -> CIF curves only, no regression_result."""
    r = client.post(f"{PREFIX}/fine_gray", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "comp_event",
        "event_of_interest": 1,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "Fine-Gray Competing Risks"
    assert 1 in d["event_types"] and 2 in d["event_types"]
    assert d["regression_result"] is None
    assert "plot" in d and d["plot"]["data"]
    assert d["event_counts"]["All"]["competing_events"] > 0


def test_fine_gray_with_group_grays_test(client, sid):
    r = client.post(f"{PREFIX}/fine_gray", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "comp_event",
        "event_of_interest": 1,
        "group_col": "DM",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["gray_p"] is not None
    assert 0.0 <= d["gray_p"] <= 1.0
    assert len(d["event_counts"]) == 2


def test_fine_gray_bad_event_of_interest_422(client, sid):
    r = client.post(f"{PREFIX}/fine_gray", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "comp_event",
        "event_of_interest": 9,  # not present
    })
    assert r.status_code == 422, r.text


def test_fine_gray_missing_column_400(client, sid):
    r = client.post(f"{PREFIX}/fine_gray", json={
        "session_id": sid,
        "duration_col": "nope",
        "event_col": "comp_event",
    })
    assert r.status_code == 400, r.text


# ── /evalue ──────────────────────────────────────────────────────────────────

def test_evalue_hr(client):
    r = client.post(f"{PREFIX}/evalue", json={
        "estimate": 2.0,
        "ci_low": 1.5,
        "ci_high": 2.7,
        "measure_type": "HR",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "E-value (Unmeasured Confounding)"
    assert d["evalue_point"] > 1.0
    # HR 2.0 -> E = 2 + sqrt(2*1) ~= 3.414
    assert abs(d["evalue_point"] - 3.4142) < 0.01
    assert d["evalue_ci"] > 1.0


def test_evalue_or_with_baseline(client):
    r = client.post(f"{PREFIX}/evalue", json={
        "estimate": 1.8,
        "ci_low": 1.2,
        "ci_high": 2.5,
        "measure_type": "OR",
        "baseline_risk": 0.1,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["rr_converted"] is not None
    assert d["evalue_point"] > 1.0


def test_evalue_ci_crosses_null(client):
    # CI crosses 1.0 -> E-value for CI must be 1.0
    r = client.post(f"{PREFIX}/evalue", json={
        "estimate": 1.3,
        "ci_low": 0.8,
        "ci_high": 2.1,
        "measure_type": "RR",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["evalue_ci"] == 1.0


# ── /landmark ────────────────────────────────────────────────────────────────

def test_landmark_single_group(client, sid):
    r = client.post(f"{PREFIX}/landmark", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "landmark_time": 100,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "Landmark Survival Analysis"
    assert d["landmark_time"] == 100
    assert d["n_landmark"] <= d["n_total"]
    assert "All" in d["km_summaries"]


def test_landmark_two_group_with_predictors(client, sid):
    r = client.post(f"{PREFIX}/landmark", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "landmark_time": 100,
        "group_col": "DM",
        "predictors": ["AGE", "LDL"],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["logrank_p"] is not None
    assert d["cox_results"] is not None
    assert len(d["km_summaries"]) == 2


def test_landmark_too_high_422(client, sid):
    # Landmark beyond max time -> <10 remain
    r = client.post(f"{PREFIX}/landmark", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "landmark_time": 5000,
    })
    assert r.status_code == 422, r.text


# ── /rmst ────────────────────────────────────────────────────────────────────

def test_rmst_tau_exceeds_max_422(client, sid):
    r = client.post(f"{PREFIX}/rmst", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "tau": 100000,
    })
    assert r.status_code == 422, r.text


def test_rmst_nonpositive_tau_422(client, sid):
    r = client.post(f"{PREFIX}/rmst", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "tau": 0,
    })
    assert r.status_code == 422, r.text


def test_rmst_group_contrast_shape(client, sid):
    r = client.post(f"{PREFIX}/rmst", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "tau": 800,
        "group_col": "DM",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["rmst_by_group"]) == 2
    assert len(d["contrasts"]) == 1
    c = d["contrasts"][0]
    assert "delta_rmst" in c and "ci_low" in c and "ci_high" in c


# ── /recurrent_lwyy ──────────────────────────────────────────────────────────

def test_recurrent_lwyy_happy(client, sid_recur):
    r = client.post(f"{PREFIX}/recurrent_lwyy", json={
        "session_id": sid_recur,
        "id_col": "subj_id",
        "start_col": "start",
        "stop_col": "stop",
        "event_col": "rec_event",
        "predictors": ["TRT", "AGE"],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "Recurrent events — LWYY model"
    assert d["n_subjects"] > 0
    assert d["n_events"] > 0
    assert len(d["coefficients"]) >= 1
    for c in d["coefficients"]:
        assert "rate_ratio" in c and "robust_se" in c
    assert d["plot"]["data"]


def test_recurrent_lwyy_no_predictors_422(client, sid_recur):
    r = client.post(f"{PREFIX}/recurrent_lwyy", json={
        "session_id": sid_recur,
        "id_col": "subj_id",
        "start_col": "start",
        "stop_col": "stop",
        "event_col": "rec_event",
        "predictors": [],
    })
    assert r.status_code == 422, r.text


def test_recurrent_lwyy_missing_column_400(client, sid_recur):
    r = client.post(f"{PREFIX}/recurrent_lwyy", json={
        "session_id": sid_recur,
        "id_col": "subj_id",
        "start_col": "start",
        "stop_col": "nope",
        "event_col": "rec_event",
        "predictors": ["TRT"],
    })
    assert r.status_code == 400, r.text


def test_recurrent_lwyy_group_col_also_predictor(client, sid_recur):
    # Regression: group_col duplicating a predictor used to yield duplicate
    # DataFrame columns → pd.to_numeric TypeError → unhandled 500.
    r = client.post(f"{PREFIX}/recurrent_lwyy", json={
        "session_id": sid_recur,
        "id_col": "subj_id",
        "start_col": "start",
        "stop_col": "stop",
        "event_col": "rec_event",
        "predictors": ["TRT", "AGE"],
        "group_col": "TRT",
    })
    assert r.status_code == 200, r.text
    assert len(r.json()["coefficients"]) >= 1


# ── /survival_validation ─────────────────────────────────────────────────────

def test_survival_validation_happy(client, sid):
    r = client.post(f"{PREFIX}/survival_validation", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["AGE", "LDL", "DM"],
        "horizon": 500,
        "n_groups": 5,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "Survival model validation"
    assert d["horizon"] == 500
    assert 0.0 <= d["concordance"] <= 1.0
    assert len(d["calibration"]) >= 1
    assert len(d["coefficients"]) == 3
    if d["time_auc"] is not None:
        assert 0.0 <= d["time_auc"] <= 1.0


def test_survival_validation_bad_horizon_422(client, sid):
    r = client.post(f"{PREFIX}/survival_validation", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["AGE"],
        "horizon": 100000,
    })
    assert r.status_code == 422, r.text


def test_survival_validation_no_predictors_422(client, sid):
    r = client.post(f"{PREFIX}/survival_validation", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": [],
        "horizon": 500,
    })
    assert r.status_code == 422, r.text


# ── /discrete_time ───────────────────────────────────────────────────────────

def test_discrete_time_happy(client, sid):
    r = client.post(f"{PREFIX}/discrete_time", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["AGE", "DM"],
        "n_intervals": 4,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"].startswith("Discrete-time survival")
    assert d["n_intervals"] >= 2
    assert d["n_person_periods"] >= d["n_subjects"]
    kinds = {c["kind"] for c in d["coefficients"]}
    assert "covariate" in kinds
    for c in d["coefficients"]:
        assert "or" in c and "or_low" in c and "or_high" in c


def test_discrete_time_missing_column_400(client, sid):
    r = client.post(f"{PREFIX}/discrete_time", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["NOPE"],
        "n_intervals": 4,
    })
    assert r.status_code == 400, r.text


def test_discrete_time_session_not_found_404(client):
    r = client.post(f"{PREFIX}/discrete_time", json={
        "session_id": "tsadv_does_not_exist",
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["AGE"],
    })
    assert r.status_code == 404, r.text
