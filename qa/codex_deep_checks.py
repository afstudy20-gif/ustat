from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_via_testclient import boot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "qa" / "cohort_test.csv"


def print_json(label, obj):
    print(label)
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def safe_call(client, path, payload):
    try:
        r = client.post(path, json=payload)
        try:
            body = r.json()
        except Exception:
            body = r.text
        return {"status": r.status_code, "body": body}
    except Exception as exc:
        return {"status": "EXC", "body": repr(exc)}


def add_derived_columns(sid):
    sys.path.insert(0, str(ROOT / "backend"))
    from services import store
    from lifelines import CoxPHFitter
    import statsmodels.api as sm

    df = store.get(sid).copy()
    # Locale-aware numeric helper for derived audit-only columns.
    for c in ["age", "bmi", "ldl", "nyha", "fu_days", "event"]:
        df[f"{c}_num"] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["start0"] = 0.0
    df["stop_fu"] = df["fu_days_num"]

    surv = df[["fu_days_num", "event_num", "age_num", "ldl_num", "nyha_num"]].dropna()
    surv = surv[surv["fu_days_num"] > 0].copy()
    cph = CoxPHFitter()
    cph.fit(surv, duration_col="fu_days_num", event_col="event_num")
    lp = pd.Series(np.nan, index=df.index, dtype=float)
    lp.loc[surv.index] = cph.predict_log_partial_hazard(surv).astype(float).values
    df["cox_lp_same_cohort"] = lp

    log = df[["event_num", "age_num", "ldl_num", "nyha_num"]].dropna()
    X = sm.add_constant(log[["age_num", "ldl_num", "nyha_num"]].astype(float))
    lm = sm.Logit(log["event_num"].astype(int), X).fit(disp=False)
    prob = pd.Series(np.nan, index=df.index, dtype=float)
    prob.loc[log.index] = lm.predict(X).astype(float).values
    df["event_prob_same_cohort"] = prob

    store.save(sid, df, track_undo=False)
    return {
        "cox_c_index_dev": float(cph.concordance_index_),
        "cox_n": int(len(surv)),
        "cox_events": int(surv["event_num"].sum()),
        "logit_n": int(len(log)),
        "logit_mean_obs": float(log["event_num"].mean()),
        "logit_mean_pred": float(prob.loc[log.index].mean()),
        "logit_oe": float(log["event_num"].mean() / prob.loc[log.index].mean()),
    }


def reference_numbers():
    import statsmodels.api as sm
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test
    from lifelines.utils import concordance_index, restricted_mean_survival_time

    raw = pd.read_csv(CSV, na_values=["", "NA", "n/a"])
    for c in ["age", "bmi", "ldl", "sbp", "diabetes", "nyha", "fu_days", "event"]:
        raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    refs = {}
    lin_clean = raw[["bmi", "age", "sex", "ldl"]].dropna()
    lin_clean = lin_clean[lin_clean["sex"].isin(["F", "M"])]
    X = sm.add_constant(pd.get_dummies(lin_clean[["age", "sex", "ldl"]], drop_first=True).astype(float))
    m = sm.OLS(lin_clean["bmi"], X).fit()
    refs["linear_locale_clean"] = {"n": int(m.nobs), "r2": float(m.rsquared)}

    lin_strict = raw[["bmi", "age", "sex", "ldl"]].copy()
    # Emulate backend's listwise/dropna without locale conversion and then its astype(float).
    strict_complete = pd.read_csv(CSV, na_values=["", "NA", "n/a"])[["bmi", "age", "sex", "ldl"]].dropna()
    refs["linear_backend_complete_before_float"] = {"n": int(len(strict_complete)), "comma_cells_remaining": int(strict_complete["bmi"].astype(str).str.contains(",", regex=False).sum())}

    log = raw[["event", "age", "sex", "ldl", "nyha"]].dropna()
    X_dirty = sm.add_constant(pd.get_dummies(log[["age", "sex", "ldl", "nyha"]], drop_first=True).astype(float))
    md = sm.Logit(log["event"].astype(int), X_dirty).fit(disp=False)
    clean = log[log["sex"].isin(["F", "M"])]
    X_clean = sm.add_constant(pd.get_dummies(clean[["age", "sex", "ldl", "nyha"]], drop_first=True).astype(float))
    mc = sm.Logit(clean["event"].astype(int), X_clean).fit(disp=False)
    refs["logistic_dirty_levels"] = {"n": int(md.nobs), "or": {str(k): float(np.exp(v)) for k, v in md.params.items()}}
    refs["logistic_clean_sex"] = {"n": int(mc.nobs), "or": {str(k): float(np.exp(v)) for k, v in mc.params.items()}}

    surv = raw[["fu_days", "event", "age", "ldl", "nyha"]].dropna()
    surv_pos = surv[surv["fu_days"] >= 0].copy()
    lr = multivariate_logrank_test(surv_pos["fu_days"], surv_pos["nyha"], surv_pos["event"].astype(int))
    cox = surv_pos.copy()
    cox["age10"] = cox["age"] / 10.0
    cph = CoxPHFitter().fit(cox[["fu_days", "event", "age10", "ldl", "nyha"]], "fu_days", "event")
    kmf = KaplanMeierFitter().fit(surv_pos["fu_days"], surv_pos["event"].astype(int))
    refs["survival_positive_clean"] = {
        "n": int(len(surv_pos)),
        "excluded": int(len(raw) - len(surv_pos)),
        "logrank_p": float(lr.p_value),
        "cox_age10_hr": float(cph.hazard_ratios_["age10"]),
        "rmst_365": float(restricted_mean_survival_time(kmf, t=365)),
    }
    return refs


def main():
    client, sid = boot()
    derived = add_derived_columns(sid)
    print_json("DERIVED", derived)
    calls = [
        ("/api/models/survival/cox_tv", {"session_id": sid, "id_col": "patient_id", "start_col": "start0", "stop_col": "stop_fu", "event_col": "event_num", "predictors": ["age_num", "ldl_num"]}),
        ("/api/models/rcs", {"session_id": sid, "model_type": "linear", "outcome": "bmi_num", "predictor": "age_num", "covariates": ["ldl_num"], "n_knots": 4}),
        ("/api/models/rcs", {"session_id": sid, "model_type": "logistic", "outcome": "event_num", "predictor": "age_num", "covariates": ["ldl_num", "nyha_num"], "n_knots": 4}),
        ("/api/models/rcs", {"session_id": sid, "model_type": "cox", "duration_col": "fu_days_num", "event_col": "event_num", "predictor": "age_num", "covariates": ["ldl_num"], "n_knots": 4}),
        ("/api/models/survival/cox_rcs", {"session_id": sid, "duration_col": "fu_days_num", "event_col": "event_num", "spline_terms": [{"column": "age_num", "n_knots": 4}], "covariates": ["ldl_num"]}),
        ("/api/model_diagnostics/model_validation", {"session_id": sid, "model_type": "binary", "outcome": "event_num", "predictors": ["age_num", "ldl_num", "nyha_num"], "n_boot": 50, "cv_folds": 5}),
        ("/api/model_diagnostics/model_validation", {"session_id": sid, "model_type": "cox", "duration_col": "fu_days_num", "event_col": "event_num", "predictors": ["age_num", "ldl_num", "nyha_num"], "n_boot": 50, "cv_folds": 5}),
        ("/api/model_diagnostics/external_validation_logistic", {"session_id": sid, "outcome": "event_num", "prob_column": "event_prob_same_cohort", "dev_auc": 0.6, "dev_calibration_slope": 1.0}),
        ("/api/survival_advanced/external_validation", {"session_id": sid, "duration_col": "fu_days_num", "event_col": "event_num", "predicted_lp_col": "cox_lp_same_cohort", "dev_metrics": {"c_index": derived["cox_c_index_dev"], "calibration_slope": 1.0}}),
        ("/api/survival_advanced/evalue", {"estimate": 1.2, "ci_low": 0.8, "ci_high": 1.8, "measure_type": "HR"}),
        ("/api/survival_advanced/fine_gray", {"session_id": sid, "duration_col": "fu_days_num", "event_col": "event_num", "event_of_interest": 1, "group_col": "nyha_num", "predictors": ["age_num", "ldl_num"]}),
        ("/api/survival_advanced/landmark", {"session_id": sid, "duration_col": "fu_days_num", "event_col": "event_num", "landmark_time": 365, "group_col": "nyha_num", "predictors": ["age_num", "ldl_num"]}),
        ("/api/survival_advanced/rmst", {"session_id": sid, "duration_col": "fu_days_num", "event_col": "event_num", "group_col": "nyha_num", "tau": 365}),
    ]
    for path, payload in calls:
        print_json(path, safe_call(client, path, payload))
    print_json("REFERENCES", reference_numbers())


if __name__ == "__main__":
    main()
