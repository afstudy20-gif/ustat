"""Smoke tests for the 14 Tier-1 v2.0.0 endpoints.

These cover endpoint registration, response shape, and a sanity check on
the numbers. Heavier statistical correctness lives in the per-test files
(test_diagnostics.py, etc).
"""
import io
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(42)
    n = 200
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    sex = rng.integers(0, 2, n)
    dm = rng.integers(0, 2, n)
    ht = rng.integers(0, 2, n)
    logit_p = -4 + 0.04 * age + 0.01 * ldl + 0.5 * dm
    p = 1 / (1 + np.exp(-logit_p))
    event = (rng.uniform(0, 1, n) < p).astype(int)
    duration = rng.exponential(500, n).clip(1, 1825)
    severity = pd.qcut(age + rng.normal(0, 5, n), q=4, labels=False).astype(int) + 1
    sid = np.repeat(np.arange(n // 4), 4)[:n]
    base = rng.integers(1, 5, n)
    r1 = base.copy()
    r2 = np.where(rng.uniform(0, 1, n) < 0.85, base, rng.integers(1, 5, n))
    r3 = np.where(rng.uniform(0, 1, n) < 0.80, base, rng.integers(1, 5, n))
    return pd.DataFrame({
        "AGE": age, "LDL": ldl, "SEX": sex, "DM": dm, "HT": ht,
        "event": event, "duration": duration,
        "severity": severity, "sid": sid,
        "rater1": r1, "rater2": r2, "rater3": r3,
    })


@pytest.fixture(scope="module")
def sid(synth):
    return make_session(synth, "v2_session")


@pytest.fixture(scope="module")
def sid_competing(synth):
    """Augment `synth` with a 3-level competing-risks event column.

    0 = censored, 1 = cause of interest, 2 = competing event.
    Probabilities depend mildly on AGE / LDL / DM so the Fine-Gray
    sHR regression has signal to pick up.
    """
    rng = np.random.default_rng(7)
    n = len(synth)
    base = synth.copy().reset_index(drop=True)
    # Latent intensities for the two competing causes
    lin_int = -3 + 0.03 * base["AGE"].values + 0.02 * base["DM"].values - 0.01 * base["LDL"].values
    lin_comp = -3.5 + 0.02 * base["AGE"].values + 0.04 * base["HT"].values
    p_int = 1 / (1 + np.exp(-lin_int))
    p_comp = 1 / (1 + np.exp(-lin_comp))
    u = rng.uniform(0, 1, n)
    comp_event = np.zeros(n, dtype=int)
    comp_event[u < p_int] = 1
    comp_event[(u >= p_int) & (u < p_int + p_comp)] = 2
    base["comp_event"] = comp_event
    return make_session(base, "v2_competing_session")


@pytest.fixture(scope="module")
def sid_tv(synth):
    # Long-format 2 intervals per subject for Cox-TV
    rows = []
    for i, row in synth.head(50).reset_index(drop=True).iterrows():
        mid = float(row["duration"]) / 2
        rows.append({"sid": i, "start": 0.0, "stop": mid, "event": 0,
                     "AGE": row["AGE"], "LDL": row["LDL"]})
        rows.append({"sid": i, "start": mid, "stop": float(row["duration"]),
                     "event": int(row["event"]), "AGE": row["AGE"], "LDL": row["LDL"] * 1.05})
    return make_session(pd.DataFrame(rows), "v2_tv_session")


# 1. VIF in linear coef rows
def test_linear_has_vif(client, sid):
    r = client.post("/api/models/linear",
                    json={"session_id": sid, "outcome": "AGE", "predictors": ["LDL", "DM", "HT"]})
    assert r.status_code == 200, r.text
    coefs = r.json()["coefficients"]
    assert all("vif" in c for c in coefs)


# 2. Schoenfeld auto-attach + VIF on Cox
def test_cox_auto_schoenfeld_and_vif(client, sid):
    r = client.post("/api/models/survival/cox",
                    json={"session_id": sid, "duration_col": "duration", "event_col": "event",
                          "predictors": ["AGE", "LDL", "DM"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ph_test") is not None
    assert any(c.get("vif") is not None for c in body["coefficients"])


# 3. Hosmer-Lemeshow standalone
def test_hosmer_lemeshow(client, sid):
    r = client.post("/api/decision_curve/hosmer_lemeshow",
                    json={"session_id": sid, "outcome": "event", "predictors": ["AGE", "LDL"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "chi2" in d and "df" in d and "p" in d
    assert 0 <= d["p"] <= 1


# 4. ROC threshold table carries LR+/LR-/PPV/NPV
def test_roc_threshold_diagnostics(client, sid):
    r = client.post("/api/stats/roc",
                    json={"session_id": sid, "score_column": "LDL", "outcome_column": "event"})
    assert r.status_code == 200, r.text
    sample = r.json()["curve"][len(r.json()["curve"]) // 2]
    for k in ("sensitivity", "specificity", "ppv", "npv", "lr_pos", "lr_neg", "youden_j"):
        assert k in sample, f"missing {k} in ROC threshold curve point"


# 5. Fleiss kappa (>=3 raters)
def test_fleiss_kappa(client, sid):
    r = client.post("/api/stats/fleiss_kappa",
                    json={"session_id": sid, "rater_cols": ["rater1", "rater2", "rater3"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "kappa" in d and "interpretation" in d
    assert d["n_raters"] == 3


# 6. TOST equivalence
@pytest.mark.parametrize("test_type", ["independent", "paired", "one_sample"])
def test_tost(client, sid, test_type):
    body = {"session_id": sid, "column": "LDL", "low": -10, "high": 10, "test_type": test_type}
    if test_type == "independent":
        body["group_column"] = "DM"
    elif test_type == "paired":
        body["paired_column"] = "AGE"
    r = client.post("/api/stats/tost", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "p_overall" in d and "equivalent" in d


# 7. GEE
@pytest.mark.parametrize("fam,cov", [
    ("binomial", "exchangeable"),
    ("gaussian", "independence"),
    ("poisson", "ar"),
])
def test_gee(client, sid, fam, cov):
    out = {"binomial": "event", "gaussian": "AGE", "poisson": "DM"}[fam]
    preds = ["LDL", "HT"] if fam != "gaussian" else ["LDL", "DM"]
    r = client.post("/api/models/gee",
                    json={"session_id": sid, "outcome": out, "predictors": preds,
                          "group_col": "sid", "family": fam, "cov_struct": cov})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_clusters"] > 0 and d["n_obs"] > 0


# 8. Ordinal logistic
def test_ordinal(client, sid):
    r = client.post("/api/models/ordinal",
                    json={"session_id": sid, "outcome": "severity", "predictors": ["LDL", "SEX"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["categories_in_rank_order"]) >= 3
    assert len(d["coefficients"]) >= 1
    assert "brant_proportional_odds" in d


# 9. Power: logistic
def test_power_logistic(client):
    r = client.post("/api/stats/power",
                    json={"test": "logistic", "solve_for": "n", "alpha": 0.05, "power": 0.8,
                          "log_or": 1.5, "p_event": 0.2, "tails": 2})
    assert r.status_code == 200, r.text
    assert r.json()["result"] is not None and r.json()["result"] > 0


# 10. Power: survival_cox
def test_power_survival_cox(client):
    r = client.post("/api/stats/power",
                    json={"test": "survival_cox", "solve_for": "n", "alpha": 0.05, "power": 0.8,
                          "hr": 1.7, "event_rate": 0.3, "p_exposed": 0.5, "tails": 2})
    assert r.status_code == 200, r.text
    assert r.json()["result"] is not None and r.json()["result"] > 0


# 11. Forest plot + DL meta-analysis
def test_forest_meta(client):
    rows = [
        {"label": "S1", "est": 1.4, "ci_low": 1.0, "ci_high": 2.0},
        {"label": "S2", "est": 1.7, "ci_low": 1.2, "ci_high": 2.4},
        {"label": "S3", "est": 0.9, "ci_low": 0.7, "ci_high": 1.2},
    ]
    r = client.post("/api/charts/forest",
                    json={"rows": rows, "effect_label": "OR", "x_axis": "log", "do_meta": True})
    assert r.status_code == 200, r.text
    m = r.json()["meta"]
    assert m is not None
    for k in ("pooled_est", "pooled_ci_low", "pooled_ci_high", "I_squared_pct", "Q", "tau2"):
        assert k in m


# 12. Cox time-varying covariates
def test_cox_tv(client, sid_tv):
    r = client.post("/api/models/survival/cox_tv",
                    json={"session_id": sid_tv, "id_col": "sid", "start_col": "start",
                          "stop_col": "stop", "event_col": "event", "predictors": ["AGE", "LDL"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_subjects"] > 0 and d["n_events"] >= 0


# 13. Stepwise selection
def test_stepwise(client, sid):
    r = client.post("/api/models/stepwise",
                    json={"session_id": sid, "model_type": "logistic", "outcome": "event",
                          "candidates": ["AGE", "LDL", "DM", "HT", "SEX"],
                          "direction": "both", "criterion": "aic"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "selected" in d and "final_aic" in d and "trace" in d


# 15. IPTW — Inverse Probability of Treatment Weighting (v2.1.0)
@pytest.mark.parametrize("estimand,outcome_type,trunc", [
    ("ate",     "binary",   "percentile"),
    ("att",     "binary",   "hard"),
    ("overlap", "binary",   "none"),
    ("ate",     "survival", "percentile"),
])
def test_iptw_estimands_and_outcomes(client, sid, estimand, outcome_type, trunc):
    body = {
        "session_id": sid,
        "treatment_col": "DM",  # binary 0/1 treatment
        "covariates": ["AGE", "LDL", "SEX", "HT"],
        "estimand": estimand,
        "stabilize": True,
        "weight_truncation": trunc,
        "weight_truncation_max": 10,
        "outcome_type": outcome_type,
        "se_method": "robust",
    }
    if outcome_type == "binary":
        body["outcome_col"] = "event"
    else:
        body["survival_duration_col"] = "duration"
        body["survival_event_col"] = "event"
    r = client.post("/api/models/iptw", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["method"] == "iptw"
    assert d["estimand"] == estimand
    assert "weight_summary" in d
    assert "smd_after" in d and "smd_before" in d
    assert d["outcome_result"] is not None
    out = d["outcome_result"]
    assert "error" not in out, out
    expected_kind = "weighted_cox" if outcome_type == "survival" else "weighted_glm"
    assert out["type"].startswith(expected_kind)
    assert len(out["coefficients"]) >= 1


# 16. Fine-Gray subdistribution-hazard regression (v2.1.1)
def test_fine_gray_regression(client, sid_competing):
    r = client.post("/api/survival_advanced/fine_gray", json={
        "session_id": sid_competing,
        "duration_col": "duration",
        "event_col": "comp_event",
        "event_of_interest": 1,
        "predictors": ["AGE", "LDL", "DM"],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert "regression_result" in d and d["regression_result"]
    reg = d["regression_result"]
    assert reg["method"] == "fine_gray_regression"
    assert reg["n_events_of_interest"] > 0
    assert reg["n_competing"] > 0
    coefs = reg["coefficients"]
    names = {c["variable"] for c in coefs}
    # Numeric predictors stay under their original names; DM is binary 0/1 and
    # is treated as numeric by the encoder, so DM should be in the coef list.
    assert {"AGE", "LDL", "DM"} <= names
    for c in coefs:
        assert "shr" in c and "p" in c and "ci_low" in c and "ci_high" in c


# 17. RMST — Restricted Mean Survival Time (v2.1.2)
def test_rmst_single_group(client, sid):
    r = client.post("/api/survival_advanced/rmst", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "tau": 800,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "Restricted Mean Survival Time"
    assert d["rmst_by_group"]["All"]["rmst"] > 0
    assert d["rmst_by_group"]["All"]["rmst"] <= 800


def test_rmst_two_group_contrast(client, sid):
    r = client.post("/api/survival_advanced/rmst", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "tau": 800,
        "group_col": "DM",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    groups = d["rmst_by_group"]
    assert len(groups) >= 2
    contrasts = d["contrasts"]
    assert len(contrasts) == 1
    c0 = contrasts[0]
    for k in ("delta_rmst", "se", "z", "p", "ci_low", "ci_high"):
        assert k in c0


# 18. Method appendix DOCX
def test_method_appendix(client, sid):
    # First ensure SOME audit-loggable analysis has run
    client.post("/api/models/linear",
                json={"session_id": sid, "outcome": "AGE", "predictors": ["LDL"]})
    r = client.post("/api/pub_export/method_appendix",
                    json={"session_id": sid, "title": "Test Methods"})
    assert r.status_code == 200, r.text
    ctype = r.headers.get("content-type", "")
    assert "wordprocessingml" in ctype
    assert len(r.content) > 1000


# 19. Machine learning — random forest (classification)
def test_ml_random_forest_classification(client, sid):
    r = client.post("/api/ml/random_forest", json={
        "session_id": sid,
        "outcome": "event",
        "predictors": ["AGE", "LDL", "SEX", "DM", "HT"],
        "n_estimators": 120, "cv_folds": 4, "n_permutation_repeats": 4,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["task"] == "classification"
    assert 0.0 <= d["auc"] <= 1.0
    assert "confusion" in d and "roc_curve" in d and len(d["roc_curve"]) > 1
    assert len(d["importance"]) == 5
    assert all("permutation" in i for i in d["importance"])


# 20. Machine learning — gradient boosting (regression auto-detected)
def test_ml_gradient_boosting_regression(client, sid):
    r = client.post("/api/ml/gradient_boosting", json={
        "session_id": sid,
        "outcome": "LDL",
        "predictors": ["AGE", "SEX", "DM"],
        "task": "regression",
        "n_estimators": 120, "cv_folds": 4, "n_permutation_repeats": 4,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["task"] == "regression"
    for k in ("r2", "rmse", "mae", "scatter", "importance"):
        assert k in d


# 21. Time series — ARIMA fit + forecast
def test_ts_arima(client, sid):
    r = client.post("/api/timeseries/arima", json={
        "session_id": sid, "value_col": "AGE",
        "p": 1, "d": 1, "q": 1, "forecast_steps": 6,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["order"] == [1, 1, 1]
    assert "aic" in d and "bic" in d
    assert len(d["forecast"]) == 6
    assert all("ci_low" in f and "ci_high" in f for f in d["forecast"])
    assert len(d["coefficients"]) >= 1


# 22. Time series — stationarity ADF + KPSS + ACF/PACF
def test_ts_stationarity(client, sid):
    r = client.post("/api/timeseries/stationarity", json={
        "session_id": sid, "value_col": "AGE", "n_lags": 15,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert "adf_p" in d and "kpss_p" in d
    assert isinstance(d["adf_stationary"], bool)
    assert len(d["acf"]) == 16 and len(d["pacf"]) == 16


# 23. Time series — decomposition
def test_ts_decompose(client, sid):
    r = client.post("/api/timeseries/decompose", json={
        "session_id": sid, "value_col": "AGE", "period": 4, "method": "stl",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ("observed", "trend", "seasonal", "resid"):
        assert k in d and len(d[k]) == d["n"]
    assert "strength_trend" in d and "strength_seasonal" in d


# 24. Meta-analysis — random-effects pooling
_META_STUDIES = [
    {"label": "A", "effect": 0.75, "ci_low": 0.55, "ci_high": 1.02, "subgroup": "EU", "moderator": 2010},
    {"label": "B", "effect": 0.82, "ci_low": 0.60, "ci_high": 1.12, "subgroup": "EU", "moderator": 2013},
    {"label": "C", "effect": 1.10, "ci_low": 0.80, "ci_high": 1.51, "subgroup": "US", "moderator": 2016},
    {"label": "D", "effect": 0.68, "ci_low": 0.45, "ci_high": 1.03, "subgroup": "US", "moderator": 2019},
    {"label": "E", "e1": 12, "n1": 100, "e2": 20, "n2": 100, "subgroup": "US", "moderator": 2021},
]


def test_meta_analyze(client):
    r = client.post("/api/meta/analyze", json={"studies": _META_STUDIES, "measure": "OR", "tau2_method": "DL"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["k"] == 5
    assert "random" in d and "fixed" in d
    assert 0 <= d["I2_pct"] <= 100
    assert len(d["studies"]) == 5
    assert abs(sum(s["weight_pct"] for s in d["studies"]) - 100) < 1.0


def test_meta_subgroup_and_regression(client):
    rs = client.post("/api/meta/subgroup", json={"studies": _META_STUDIES, "measure": "OR"})
    assert rs.status_code == 200, rs.text
    assert len(rs.json()["subgroups"]) == 2
    rr = client.post("/api/meta/regression", json={"studies": _META_STUDIES, "measure": "OR"})
    assert rr.status_code == 200, rr.text
    assert "slope" in rr.json() and "slope_p" in rr.json()


def test_meta_bias(client):
    r = client.post("/api/meta/bias", json={"studies": _META_STUDIES, "measure": "OR"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "egger_p" in d and "trim_fill_missing" in d
    assert len(d["funnel"]) == 5


# 25. Weighted descriptives (survey weights)
def test_weighted_descriptive(client, synth):
    # add a weight column to the synth fixture via a fresh session
    df = synth.copy()
    rngw = np.random.default_rng(11)
    df["wt"] = rngw.uniform(0.5, 3.0, len(df))
    s = make_session(df, "weighted_session")
    r = client.post("/api/stats/weighted_descriptive", json={
        "session_id": s, "weight_col": "wt", "value_cols": ["AGE", "LDL", "DM"], "group_col": "SEX",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    rows = {x["column"]: x for x in d["results"]}
    assert "AGE" in rows and "w_mean" in rows["AGE"] and "ess_kish" in rows["AGE"]
    # DM is binary -> weighted proportion present
    assert "w_proportion" in rows["DM"]
    assert d["comparison"] is not None and "p" in d["comparison"]


# 26. Recurrent events — LWYY model
@pytest.fixture(scope="module")
def sid_recurrent():
    rng = np.random.default_rng(5)
    rows = []
    for pid in range(80):
        trt = pid % 2
        age = rng.normal(60, 8)
        rate = 0.4 * np.exp(-0.5 * trt + 0.02 * (age - 60))
        t, horizon = 0.0, 10.0
        while True:
            gap = rng.exponential(1 / rate)
            start, stop = t, t + gap
            if stop >= horizon:
                rows.append({"pid": pid, "t0": start, "t1": horizon, "ev": 0, "trt": trt, "age": age})
                break
            rows.append({"pid": pid, "t0": start, "t1": stop, "ev": 1, "trt": trt, "age": age})
            t = stop
    return make_session(pd.DataFrame(rows), "recurrent_session")


def test_recurrent_lwyy(client, sid_recurrent):
    r = client.post("/api/survival_advanced/recurrent_lwyy", json={
        "session_id": sid_recurrent, "id_col": "pid", "start_col": "t0",
        "stop_col": "t1", "event_col": "ev", "predictors": ["trt", "age"],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_subjects"] == 80 and d["n_events"] > 0
    coefs = {c["variable"]: c for c in d["coefficients"]}
    assert "trt" in coefs and "rate_ratio" in coefs["trt"]
    # treatment lowers the recurrence rate → RR < 1
    assert coefs["trt"]["rate_ratio"] < 1.0
    assert d["plot"]["data"] and len(d["plot"]["data"][0]["x"]) > 1


# 27. Multistage gatekeeping (truncated Hochberg)
def test_gatekeeping_serial(client):
    payload = {
        "method": "hochberg", "logic": "serial", "alpha": 0.05,
        "families": [
            {"name": "Primary", "hypotheses": [{"label": "Death", "p": 0.012}]},
            {"name": "Secondary", "hypotheses": [
                {"label": "MI", "p": 0.02}, {"label": "Stroke", "p": 0.04}, {"label": "HF", "p": 0.30},
            ]},
        ],
    }
    r = client.post("/api/multiplicity/gatekeeping", json=payload)
    assert r.status_code == 200, r.text
    d = r.json()
    fams = d["families"]
    assert len(fams) == 2
    # primary significant -> rejected
    assert fams[0]["hypotheses"][0]["reject"] is True
    # adjusted p present and monotone-ish
    assert all("p_adjusted" in h for f in fams for h in f["hypotheses"])


def test_gatekeeping_gate_closes(client):
    # primary NOT significant -> secondary must be blocked regardless of tiny p
    payload = {
        "method": "hochberg", "logic": "serial",
        "families": [
            {"name": "Primary", "hypotheses": [{"label": "Death", "p": 0.20}]},
            {"name": "Secondary", "hypotheses": [{"label": "MI", "p": 0.001}]},
        ],
    }
    r = client.post("/api/multiplicity/gatekeeping", json=payload)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["families"][1]["hypotheses"][0]["reject"] is False


# 28. Non-inferiority margin test
def test_noninferiority_binary_rr(client, synth):
    # event ~ DM; treat DM as the arm, event as outcome
    r = client.post("/api/stats/noninferiority", json={
        "session_id": make_session(synth, "ni_session"),
        "outcome_col": "event", "group_col": "DM",
        "outcome_type": "binary", "effect": "RR", "margin": 1.20, "bound": "upper", "alpha": 0.05,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ci_level"] == 90.0           # one-sided 0.05 -> 90% CI
    assert "estimate" in d and "ci_low" in d and "ci_high" in d
    assert isinstance(d["non_inferior"], bool)
    assert "p_noninferiority" in d


def test_noninferiority_continuous(client, synth):
    r = client.post("/api/stats/noninferiority", json={
        "session_id": make_session(synth, "ni_session2"),
        "outcome_col": "LDL", "group_col": "SEX",
        "outcome_type": "continuous", "margin": 20.0, "bound": "upper", "alpha": 0.05,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["effect"] == "Mean difference"
    assert d["ci_level"] == 90.0
