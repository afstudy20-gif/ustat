"""Tests for routers/models_causal.py — PSM and IPTW causal-inference endpoints.

Endpoints under test (mounted at /api/models):
  * POST /api/models/psm   — Propensity Score Matching
  * POST /api/models/iptw  — Inverse Probability of Treatment Weighting

Synthetic data: a binary treatment that depends on covariates (confounding),
a binary outcome, plus survival columns so the survival branches are exercised.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_session

SEED = 7
PREFIX = "tcaus"


def _make_df(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    sex = rng.integers(0, 2, n)
    dm = rng.integers(0, 2, n)
    # Confounded treatment: older / higher-LDL patients more likely treated.
    logit = -6.0 + 0.05 * age + 0.01 * ldl + 0.4 * dm
    p_treat = 1.0 / (1.0 + np.exp(-logit))
    treat = (rng.uniform(0, 1, n) < p_treat).astype(int)
    # Outcome depends on covariates + a real treatment effect.
    out_logit = -2.0 + 0.02 * age + 0.005 * ldl - 0.6 * treat
    p_out = 1.0 / (1.0 + np.exp(-out_logit))
    outcome = (rng.uniform(0, 1, n) < p_out).astype(int)
    event = (rng.uniform(0, 1, n) < 0.4).astype(int)
    duration = rng.exponential(500, n).clip(1, 1825)
    region = rng.integers(0, 3, n)  # categorical for exact-match
    return pd.DataFrame({
        "AGE": age, "LDL": ldl, "SEX": sex, "DM": dm,
        "treat": treat, "outcome": outcome,
        "event": event, "duration": duration, "region": region,
    })


@pytest.fixture(scope="module")
def sid() -> str:
    return make_session(_make_df(), f"{PREFIX}_main")


COVS = ["AGE", "LDL", "SEX", "DM"]


# ── PSM ──────────────────────────────────────────────────────────────────────

def test_psm_happy_path_greedy(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat",
        "covariates": COVS, "outcome_col": "outcome",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matching_method"] == "greedy"
    assert body["n_treated"] > 0 and body["n_control"] > 0
    assert body["n_matched_pairs"] >= 1
    assert body["n_matched_controls"] >= body["n_matched_pairs"]
    assert set(body["smd_before"].keys()) == set(COVS)
    assert set(body["smd_after"].keys()) == set(COVS)
    assert 0.0 <= body["avg_smd_after"] < 5.0
    assert isinstance(body["balance_achieved"], bool)
    assert body["matched_session_id"] == sid + "_psm"
    assert body["score_method"] == "logistic"
    assert "ps_distribution" in body


def test_psm_optimal_matching(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat",
        "covariates": COVS, "outcome_col": "outcome",
        "matching_method": "optimal",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matching_method"] == "optimal"
    assert body["n_matched_pairs"] >= 1
    # 1:1 optimal → equal treated and controls
    assert body["n_matched_controls"] == body["n_matched_pairs"]


def test_psm_optimal_ratio_falls_back_to_greedy(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat",
        "covariates": COVS, "matching_method": "optimal", "ratio": 2,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    # Optimal supports 1:1 only; ratio>1 falls back to greedy with a warning.
    assert body["matching_method"] == "greedy"
    assert body["matching_warning"]


def test_psm_outcome_conditional_logistic(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat",
        "covariates": COVS, "outcome_col": "outcome",
    })
    assert r.status_code == 200, r.text
    res = r.json()["outcome_result"]
    assert res is not None
    # Either a fitted conditional logistic / fallback, or a structured error.
    if "error" not in res:
        assert res["type"] in ("conditional_logistic", "logistic_robust")
        assert isinstance(res["coefficients"], list) and len(res["coefficients"]) >= 1
        c0 = res["coefficients"][0]
        assert c0["variable"] == "treat"
        assert "or" in c0 and c0["or"] > 0


def test_psm_survival_outcome(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat", "covariates": COVS,
        "outcome_type": "survival",
        "survival_duration_col": "duration", "survival_event_col": "event",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome_type"] == "survival"
    res = body["outcome_result"]
    assert res is not None
    if "error" not in res:
        assert res["type"] == "stratified_cox"
        assert res["coefficients"][0]["hr"] > 0


def test_psm_rosenbaum_bounds(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat",
        "covariates": COVS, "outcome_col": "outcome",
        "compute_rosenbaum": True, "rosenbaum_gamma_max": 2.5,
    })
    assert r.status_code == 200, r.text
    rb = r.json()["rosenbaum"]
    assert rb is not None
    assert "applicable" in rb
    if rb["applicable"]:
        assert "critical_gamma" in rb
        assert isinstance(rb["curve"], list) and len(rb["curve"]) >= 2


def test_psm_exact_match_and_raw_caliper(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat", "covariates": COVS,
        "exact_match": ["region"], "caliper_scale": "raw", "caliper": 0.5,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["caliper_scale"] == "raw"
    assert body["exact_match"] == ["region"]


def test_psm_trim_common_support(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat", "covariates": COVS,
        "trim_common_support": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_trimmed_common_support"] >= 0


def test_psm_non_binary_treatment_422(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "AGE", "covariates": ["LDL", "SEX"],
    })
    assert r.status_code == 422, r.text
    assert "binary" in r.text.lower()


def test_psm_missing_column_422(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat",
        "covariates": ["AGE", "NOPE"],
    })
    assert r.status_code == 422, r.text
    assert "not found" in r.text.lower()


def test_psm_bad_outcome_type_422(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat", "covariates": COVS,
        "outcome_type": "continuous",
    })
    assert r.status_code == 422, r.text


# ── IPTW ─────────────────────────────────────────────────────────────────────

def test_iptw_estimand_overlap(client, sid):
    r = client.post("/api/models/iptw", json={
        "session_id": sid, "treatment_col": "treat",
        "covariates": COVS, "outcome_col": "outcome", "estimand": "overlap",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["estimand"] == "overlap"
    # Overlap weights are bounded in (0,1).
    assert body["weight_summary"]["max"] <= 1.0 + 1e-6


def test_iptw_survival_outcome(client, sid):
    r = client.post("/api/models/iptw", json={
        "session_id": sid, "treatment_col": "treat", "covariates": COVS,
        "outcome_type": "survival",
        "survival_duration_col": "duration", "survival_event_col": "event",
    })
    assert r.status_code == 200, r.text
    res = r.json()["outcome_result"]
    assert res is not None
    if "error" not in res:
        assert res["type"].startswith("weighted_cox")
        assert res["coefficients"][0]["hr"] > 0


def test_iptw_bad_estimand_422(client, sid):
    r = client.post("/api/models/iptw", json={
        "session_id": sid, "treatment_col": "treat",
        "covariates": COVS, "estimand": "frobnicate",
    })
    assert r.status_code == 422, r.text


