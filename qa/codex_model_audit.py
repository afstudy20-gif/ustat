from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_via_testclient import boot  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "qa" / "cohort_test.csv"


def clean_df() -> pd.DataFrame:
    df = pd.read_csv(CSV, na_values=["", "NA", "n/a"])
    for col in ["age", "bmi", "ldl", "sbp", "diabetes", "nyha", "fu_days", "event"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df


def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def call(client, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        if method == "GET":
            r = client.get(path)
        else:
            r = client.post(path, json=payload or {})
        out = {"path": path, "status": r.status_code}
        try:
            out["json"] = r.json()
        except Exception:
            out["text"] = r.text[:1000]
        return out
    except Exception as exc:
        return {"path": path, "status": "EXC", "exception": repr(exc)}


def summarize_response(resp: dict[str, Any]) -> dict[str, Any]:
    out = {"path": resp["path"], "status": resp["status"]}
    body = resp.get("json")
    if isinstance(body, dict):
        for key in [
            "model",
            "n",
            "n_obs",
            "n_total",
            "n_excluded",
            "r_squared",
            "concordance",
            "logrank",
            "e_value",
            "observed_expected_ratio",
            "oe_ratio",
            "random_effect_variance",
            "residual_variance",
            "icc",
        ]:
            if key in body:
                out[key] = body[key]
        if "coefficients" in body:
            out["coefficients"] = body["coefficients"][:8]
        if "groups" in body:
            out["groups"] = [
                {k: g.get(k) for k in ("group", "n", "events", "median_survival")}
                for g in body["groups"][:8]
            ]
        if "results" in body:
            out["results"] = body["results"][:5] if isinstance(body["results"], list) else body["results"]
        if "detail" in body:
            out["detail"] = body["detail"]
        for key in ["brant_proportional_odds", "horizons", "forest_rows", "turnbull", "weibull", "calibration", "metrics", "performance", "apparent"]:
            if key in body:
                out[key] = body[key]
    else:
        out["body"] = resp.get("text") or resp.get("exception")
    if "exception" in resp:
        out["exception"] = resp["exception"]
    return out


def endpoint_payloads(sid: str) -> list[tuple[str, str, dict[str, Any] | None]]:
    return [
        ("POST", "/api/models/linear", {"session_id": sid, "outcome": "bmi", "predictors": ["age", "sex", "ldl"]}),
        ("POST", "/api/models/logistic", {"session_id": sid, "outcome": "event", "predictors": ["age", "sex", "ldl", "nyha"]}),
        ("POST", "/api/models/firth_logistic", {"session_id": sid, "outcome": "event", "predictors": ["age", "sex", "ldl", "nyha"]}),
        ("POST", "/api/models/logistic_table", {"session_id": sid, "outcome": "event", "predictors": ["age", "sex", "ldl", "nyha"]}),
        ("POST", "/api/models/ordinal", {"session_id": sid, "outcome": "nyha", "predictors": ["age", "sex", "ldl"]}),
        ("POST", "/api/models/poisson", {"session_id": sid, "outcome": "event", "predictors": ["age", "sex", "ldl", "nyha"]}),
        ("POST", "/api/models/negbinom", {"session_id": sid, "outcome": "event", "predictors": ["age", "sex", "ldl", "nyha"]}),
        ("POST", "/api/models/gamma", {"session_id": sid, "outcome": "bmi", "predictors": ["age", "sex", "ldl"], "link": "log"}),
        ("POST", "/api/models/linear_diag", {"session_id": sid, "outcome": "bmi", "predictors": ["age", "sex", "ldl"]}),
        ("POST", "/api/model_diagnostics/logistic_diagnostics", {"session_id": sid, "outcome": "event", "predictors": ["age", "ldl", "nyha"]}),
        ("POST", "/api/model_diagnostics/cox_diagnostics", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "predictors": ["age", "ldl", "nyha"]}),
        ("POST", "/api/models/survival/km", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "group_col": "nyha", "risk_times": [0, 180, 365, 730], "survival_times": [180, 365, 730], "pairwise": True}),
        ("POST", "/api/models/survival/cox", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "predictors": ["age", "ldl", "nyha"]}),
        ("POST", "/api/models/survival/cox_horizons", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "predictor": "age", "covariates": ["ldl", "nyha"], "horizons": [180, 365, 730]}),
        ("POST", "/api/models/survival/cox_tv", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "predictors": ["age"], "time_varying_col": "ldl"}),
        ("POST", "/api/models/rcs", {"session_id": sid, "outcome": "bmi", "predictor": "age", "covariates": ["ldl"], "knots": 4}),
        ("POST", "/api/models/survival/cox_rcs", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "spline_var": "age", "covariates": ["ldl"], "knots": 4}),
        ("POST", "/api/models/survival/cox_uni_multi", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "predictors": ["age", "ldl", "nyha"], "multivariable_predictors": ["age", "ldl", "nyha"]}),
        ("POST", "/api/models/lmm", {"session_id": sid, "outcome": "bmi", "fixed_effects": ["age"], "group_col": "nyha"}),
        ("POST", "/api/models/gee", {"session_id": sid, "outcome": "bmi", "predictors": ["age", "ldl"], "group_col": "nyha", "family": "gaussian", "cov_struct": "exchangeable"}),
        ("POST", "/api/models/stepwise", {"session_id": sid, "model_type": "logistic", "outcome": "event", "candidates": ["age", "sex", "ldl", "nyha"], "direction": "both"}),
        ("POST", "/api/models/polynomial", {"session_id": sid, "outcome": "bmi", "predictor": "age", "degree": 2, "covariates": ["ldl"]}),
        ("POST", "/api/survival_advanced/rmst", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "group_col": "nyha", "tau": 365}),
        ("POST", "/api/survival_advanced/fine_gray", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "event_of_interest": 1, "group_col": "nyha", "predictors": ["age", "ldl"]}),
        ("POST", "/api/survival_advanced/landmark", {"session_id": sid, "duration_col": "fu_days", "event_col": "event", "landmark_time": 365, "group_col": "nyha", "predictors": ["age", "ldl"]}),
        ("POST", "/api/survival_advanced/evalue", {"estimate": 1.2, "ci_low": 0.8, "ci_high": 1.8, "measure_type": "HR"}),
        ("POST", "/api/survival_advanced/interval_censored", {"session_id": sid, "lower_col": "age", "upper_col": "fu_days", "covariates": ["ldl"], "group_col": "nyha"}),
    ]


def references() -> dict[str, Any]:
    df = clean_df()
    refs: dict[str, Any] = {}
    try:
        import statsmodels.api as sm
        lin = df[["bmi", "age", "sex", "ldl"]].dropna()
        lin = lin[lin["sex"].isin(["F", "M"])]
        X = pd.get_dummies(lin[["age", "sex", "ldl"]], drop_first=True).astype(float)
        X = sm.add_constant(X)
        m = sm.OLS(lin["bmi"], X).fit()
        refs["linear_clean"] = {
            "n": int(m.nobs),
            "r_squared": float(m.rsquared),
            "coef": {str(k): float(v) for k, v in m.params.items()},
        }
    except Exception as exc:
        refs["linear_clean_error"] = repr(exc)
    try:
        import statsmodels.api as sm
        log = df[["event", "age", "sex", "ldl", "nyha"]].dropna()
        log = log[log["sex"].isin(["F", "M"])]
        X = pd.get_dummies(log[["age", "sex", "ldl", "nyha"]], drop_first=True).astype(float)
        X = sm.add_constant(X)
        m = sm.Logit(log["event"].astype(int), X).fit(disp=False)
        refs["logistic_clean"] = {
            "n": int(m.nobs),
            "or": {str(k): float(math.exp(v)) for k, v in m.params.items()},
            "coef": {str(k): float(v) for k, v in m.params.items()},
        }
    except Exception as exc:
        refs["logistic_clean_error"] = repr(exc)
    try:
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import multivariate_logrank_test
        surv = df[["fu_days", "event", "age", "ldl", "nyha"]].dropna()
        surv_pos = surv[surv["fu_days"] >= 0].copy()
        lr = multivariate_logrank_test(surv_pos["fu_days"], surv_pos["nyha"], surv_pos["event"].astype(int))
        cox = surv_pos.copy()
        cox["age10"] = cox["age"] / 10.0
        cph = CoxPHFitter()
        cph.fit(cox[["fu_days", "event", "age10", "ldl", "nyha"]], duration_col="fu_days", event_col="event")
        refs["survival_clean"] = {
            "n": int(len(surv_pos)),
            "excluded_negative_or_missing": int(len(df) - len(surv_pos)),
            "logrank_p": float(lr.p_value),
            "cox_age10_hr": float(cph.hazard_ratios_["age10"]),
            "cox_age10_loghr": float(cph.params_["age10"]),
        }
        kmf = KaplanMeierFitter().fit(surv_pos["fu_days"], surv_pos["event"].astype(int))
        try:
            from lifelines.utils import restricted_mean_survival_time
            refs["rmst_clean_365"] = float(restricted_mean_survival_time(kmf, t=365))
        except Exception as exc:
            refs["rmst_clean_365_error"] = repr(exc)
    except Exception as exc:
        refs["survival_clean_error"] = repr(exc)
    return refs


def main() -> None:
    client, sid = boot()
    print("sid", sid)
    session_info = call(client, "GET", f"/api/sessions/{sid}")
    print("SESSION", jdump(summarize_response(session_info)))
    for method, path, payload in endpoint_payloads(sid):
        resp = call(client, method, path, payload)
        print("CALL", path)
        print(jdump(summarize_response(resp)))
    print("REFERENCES")
    print(jdump(references()))


if __name__ == "__main__":
    main()
