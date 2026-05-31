"""
Survival Router

Contains:
- Kaplan-Meier (/survival/km)
- Cox Proportional Hazards (/survival/cox)
- Cox with time-varying covariates (/survival/cox_tv)

Extracted from the old monolithic models.py as part of the ongoing split.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from services import store
from services.impute import apply_imputation
from services.assumptions import check_cox_assumptions_from_ph_test, add_assumption_warnings_to_result
from services.missing_data import mice_multiple, pool_cox_results, missing_pattern_summary, add_missing_data_diagnostics

router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(v):
    """Return float or None for inf/nan values that aren't JSON-serializable."""
    try:
        f = float(v)
        if np.isfinite(f):
            return f
        return None
    except (TypeError, ValueError):
        return None


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _km_fit_groups(df: pd.DataFrame, duration_col: str, event_col: str, group_col: Optional[str]) -> list:
    groups = df[group_col].unique() if group_col else [None]
    results = []
    for grp in groups:
        subset = df[df[group_col] == grp] if group_col else df
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(
                subset[duration_col].astype(float),
                subset[event_col].astype(int),
                label=str(grp) if grp is not None else "All",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"KM fitting error: {exc}")
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["time", "survival"]
        curve = [
            {"time": _safe_float(row["time"]), "survival": _safe_float(row["survival"])}
            for _, row in sf.iterrows()
        ]
        results.append({
            "group": str(grp) if grp is not None else "All",
            "n": int(len(subset)),
            "events": int(subset[event_col].sum()),
            "median_survival": _safe_float(kmf.median_survival_time_),
            "curve": curve,
        })
    return results


def _km_logrank(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str) -> Optional[dict]:
    groups = df[group_col].unique()
    if len(groups) < 2:
        return None
    try:
        if len(groups) == 2:
            g0 = df[df[group_col] == groups[0]]
            g1 = df[df[group_col] == groups[1]]
            lr = logrank_test(
                g0[duration_col], g1[duration_col],
                event_observed_A=g0[event_col].astype(int),
                event_observed_B=g1[event_col].astype(int),
            )
            return {"test": "Log-rank", "p": _safe_float(lr.p_value)}
        else:
            lr = multivariate_logrank_test(df[duration_col], df[group_col], df[event_col].astype(int))
            return {"test": "Log-rank (multivariate)", "p": _safe_float(lr.p_value)}
    except Exception:
        return None


def _compute_vif(X: pd.DataFrame) -> dict:
    """Variance Inflation Factor per column."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xn = X.copy().astype(float)
    if "const" in Xn.columns:
        Xn = Xn.drop(columns=["const"])
    if Xn.shape[1] < 2:
        return {c: 1.0 for c in Xn.columns}
    arr = Xn.values
    out: dict = {}
    for i, col in enumerate(Xn.columns):
        try:
            v = float(variance_inflation_factor(arr, i))
            if not np.isfinite(v):
                v = None
        except Exception:
            v = None
        out[str(col)] = v
    return out


# ── Kaplan-Meier ───────────────────────────────────────────────────────────────

class KMRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    group_col: Optional[str] = None
    stratify_col: Optional[str] = None
    imputation: Optional[str] = "listwise"


@router.post("/survival/km")
def kaplan_meier(req: KMRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col] = pd.to_numeric(df_full[req.event_col], errors="coerce")

    km_cols = [req.duration_col, req.event_col]
    df = apply_imputation(df_full, km_cols, req.imputation)
    n_excluded = n_total - len(df)

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after coercing duration/event columns to numeric. Check that both columns contain numbers.")

    event_vals = sorted(df[req.event_col].dropna().unique())
    if set(event_vals) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1 (0=censored, 1=event). Found: {event_vals[:10]}")

    if (df[req.duration_col] < 0).any():
        raise HTTPException(status_code=422, detail="Duration column contains negative values. All durations must be ≥ 0.")

    if req.stratify_col:
        if req.stratify_col not in df_full.columns:
            raise HTTPException(status_code=422, detail=f"Stratify column '{req.stratify_col}' not found.")
        strata_vals = sorted(df[req.stratify_col].dropna().unique(), key=lambda x: (isinstance(x, str), x))
        strata_out = []
        for sv in strata_vals:
            sub = df[df[req.stratify_col] == sv].copy()
            if len(sub) == 0:
                continue
            grp_results = _km_fit_groups(sub, req.duration_col, req.event_col, req.group_col)
            lr = _km_logrank(sub, req.duration_col, req.event_col, req.group_col) if req.group_col else None
            strata_out.append({"label": str(sv), "n": int(len(sub)), "groups": grp_results, "logrank": lr})
        return {
            "model": "Kaplan-Meier",
            "strata": strata_out,
            "stratify_col": req.stratify_col,
            "n_total": n_total,
            "n_excluded": n_excluded,
            "imputation": req.imputation,
        }

    results = _km_fit_groups(df, req.duration_col, req.event_col, req.group_col)
    logrank = _km_logrank(df, req.duration_col, req.event_col, req.group_col) if req.group_col else None

    return {
        "model": "Kaplan-Meier",
        "groups": results,
        "logrank": logrank,
        "n_total": n_total,
        "n_excluded": n_excluded,
        "imputation": req.imputation,
    }


# ── Cox Proportional Hazards ───────────────────────────────────────────────────

class CoxRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    interactions: Optional[List[List[str]]] = None


@router.post("/survival/cox")
def cox_regression(req: CoxRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col] = pd.to_numeric(df_full[req.event_col], errors="coerce")

    cox_cols = [req.duration_col, req.event_col] + req.predictors
    imputation_method = req.imputation or "listwise"
    use_mice_pooled = False

    # === Phase 3: MICE + Pooling for Cox ===
    if imputation_method == "mice":
        imp_result = mice_multiple(df_full, cox_cols, n_imputations=5)
        imputed_dfs = imp_result.imputed_datasets

        individual_results = []
        for df_imp in imputed_dfs:
            try:
                cph_imp = CoxPHFitter()
                cph_imp.fit(df_imp[cox_cols], duration_col=req.duration_col, event_col=req.event_col)
                # Store log HRs
                loghrs = {var: float(np.log(cph_imp.hazard_ratios_.get(var, 1.0))) for var in req.predictors}
                individual_results.append({"coefficients": loghrs})
            except Exception:
                continue

        pooled = pool_cox_results(individual_results) if individual_results else {}

        # Use first imputation for main fit + diagnostics
        df = imputed_dfs[0]
        n_excluded = n_total - len(df)
        use_mice_pooled = True
    else:
        df = apply_imputation(df_full, cox_cols, imputation_method)
        n_excluded = n_total - len(df)
        use_mice_pooled = False
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after coercing duration/event columns to numeric.")

    event_vals = sorted(df[req.event_col].dropna().unique())
    if set(event_vals) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1. Found: {event_vals[:10]}")
    if (df[req.duration_col] < 0).any():
        raise HTTPException(status_code=422, detail="Duration column contains negative values.")

    pred_raw = df[req.predictors].copy()
    numeric_pred: list[str] = []
    cat_pred: list[str] = []
    for c in req.predictors:
        col = pred_raw[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric_pred.append(c)
        else:
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().mean() >= 0.8 and len(coerced.dropna().unique()) > 2:
                pred_raw[c] = coerced
                numeric_pred.append(c)
            else:
                cat_pred.append(c)

    num_part = pred_raw[numeric_pred].apply(pd.to_numeric, errors="coerce") if numeric_pred else pd.DataFrame(index=pred_raw.index)
    cat_part = pd.get_dummies(pred_raw[cat_pred], drop_first=True, dummy_na=False) if cat_pred else pd.DataFrame(index=pred_raw.index)
    enc = pd.concat([num_part, cat_part], axis=1).astype(float)

    interaction_cols: list[str] = []
    if req.interactions:
        def _members(name: str) -> list[str]:
            if name in enc.columns:
                return [name]
            prefix = f"{name}_"
            return [c for c in enc.columns if c.startswith(prefix)]

        for pair in req.interactions:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise HTTPException(status_code=422, detail=f"Each interaction must be a [colA, colB] pair. Got: {pair}")
            a_members = _members(pair[0])
            b_members = _members(pair[1])
            if not a_members or not b_members:
                raise HTTPException(status_code=422, detail=f"Interaction '{pair[0]} × {pair[1]}': one or both columns are not in the predictor list.")
            for a in a_members:
                for b in b_members:
                    new_col = f"{a}:{b}"
                    enc[new_col] = enc[a] * enc[b]
                    interaction_cols.append(new_col)

    fit_df = pd.concat([df[[req.duration_col, req.event_col]], enc], axis=1).dropna()
    if len(fit_df) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows after encoding (need ≥ 10, got {len(fit_df)}).")

    cph = CoxPHFitter()
    try:
        cph.fit(fit_df, duration_col=req.duration_col, event_col=req.event_col)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cox fitting error: {exc}")

    summary = cph.summary.reset_index()
    vifs = _compute_vif(enc) if 'enc' in locals() else {}
    coefs = []
    for _, row in summary.iterrows():
        name = str(row["covariate"])
        coefs.append({
            "variable": name,
            "log_hr": _safe_float(row["coef"]),
            "hr": _safe_float(row["exp(coef)"]),
            "se": _safe_float(row["se(coef)"]),
            "z": _safe_float(row["z"]),
            "p": _safe_float(row["p"]),
            "hr_ci_low": _safe_float(row["exp(coef) lower 95%"]),
            "hr_ci_high": _safe_float(row["exp(coef) upper 95%"]),
            "vif": vifs.get(name),
        })

    ph_test = None
    try:
        from lifelines.statistics import proportional_hazard_test
        ph_res = proportional_hazard_test(cph, fit_df, time_transform="rank")
        ph_summary = ph_res.summary.reset_index() if hasattr(ph_res.summary, "reset_index") else ph_res.summary
        per_term = []
        for _, row in ph_summary.iterrows():
            per_term.append({
                "variable": str(row.get("index", row.get("covariate", ""))),
                "test_stat": _safe_float(row.get("test_statistic")),
                "p": _safe_float(row.get("p")),
            })
        from scipy.stats import chi2 as _chi2
        chi_vals = [t["test_stat"] for t in per_term if t["test_stat"] is not None]
        if chi_vals:
            global_chi = float(sum(chi_vals))
            global_df = len(chi_vals)
            global_p = float(1 - _chi2.cdf(global_chi, global_df)) if global_df > 0 else None
        else:
            global_chi, global_df, global_p = None, None, None
        ph_test = {"global": {"chi2": global_chi, "df": global_df, "p": global_p}, "per_term": per_term}
    except Exception as exc:
        ph_test = {"error": str(exc)}

    result = {
        "model": "Cox Proportional Hazards",
        "n": int(cph.event_observed.sum()),
        "n_total": n_total,
        "n_excluded": n_excluded,
        "imputation": req.imputation,
        "log_likelihood": _safe_float(cph.log_likelihood_),
        "concordance": _safe_float(cph.concordance_index_),
        "coefficients": coefs,
        "interactions_used": interaction_cols,
        "ph_test": ph_test,
    }

    # === Phase 3: Apply MICE pooling if used ===
    if use_mice_pooled and 'pooled' in locals() and pooled:
        result["coefficients"] = pooled.get("coefficients", result.get("coefficients", []))
        result["pooled_from_imputations"] = True
        result["imputation"] = "mice (pooled)"

    # --- Assumption Checking (Phase 1) ---
    cox_assumption_report = check_cox_assumptions_from_ph_test(ph_test)
    result = add_assumption_warnings_to_result(result, cox_assumption_report)

    # Attach missing data diagnostics
    missing_info = missing_pattern_summary(df_full, cox_cols)
    result = add_missing_data_diagnostics(result, missing_info)

    return result


# ── Cox with time-varying covariates ───────────────────────────────────────────

class CoxTVRequest(BaseModel):
    session_id: str
    id_col: str
    start_col: str
    stop_col: str
    event_col: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"


@router.post("/survival/cox_tv")
def cox_time_varying(req: CoxTVRequest):
    from lifelines import CoxTimeVaryingFitter

    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    cols = [req.id_col, req.start_col, req.stop_col, req.event_col] + req.predictors
    missing = [c for c in cols if c not in df_full.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing}")
    df = apply_imputation(df_full, cols, req.imputation or "listwise")

    for c in [req.start_col, req.stop_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[req.event_col] = pd.to_numeric(df[req.event_col], errors="coerce")
    df = df.dropna(subset=[req.start_col, req.stop_col, req.event_col, req.id_col])
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Need ≥10 valid (subject, interval) rows.")
    if (df[req.stop_col] <= df[req.start_col]).any():
        raise HTTPException(status_code=422, detail="Found rows with stop ≤ start; each interval must have stop > start.")
    if set(df[req.event_col].unique()) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail="Event column must be 0/1.")
    n_excluded = n_total - len(df)

    enc = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    if enc.shape[1] == 0:
        raise HTTPException(status_code=422, detail="No usable predictor columns after encoding.")
    fit_df = pd.concat([df[[req.id_col, req.start_col, req.stop_col, req.event_col]], enc], axis=1)

    ctv = CoxTimeVaryingFitter()
    try:
        ctv.fit(fit_df, id_col=req.id_col, start_col=req.start_col, stop_col=req.stop_col, event_col=req.event_col)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cox-TV fitting failed: {exc}")

    summary = ctv.summary.reset_index()
    vifs = _compute_vif(enc)
    coefs = []
    for _, row in summary.iterrows():
        name = str(row["covariate"])
        coefs.append({
            "variable": name,
            "log_hr": _safe_float(row["coef"]),
            "hr": _safe_float(row["exp(coef)"]),
            "se": _safe_float(row["se(coef)"]),
            "z": _safe_float(row["z"]),
            "p": _safe_float(row["p"]),
            "hr_ci_low": _safe_float(row["exp(coef) lower 95%"]),
            "hr_ci_high": _safe_float(row["exp(coef) upper 95%"]),
            "vif": vifs.get(name),
        })

    n_subjects = int(df[req.id_col].nunique())
    n_events = int(df[req.event_col].sum())
    return {
        "model": "Cox Proportional Hazards (time-varying covariates)",
        "n_intervals": int(len(df)),
        "n_subjects": n_subjects,
        "n_events": n_events,
        "n_total": int(n_total),
        "n_excluded": int(n_excluded),
        "imputation": req.imputation or "listwise",
        "log_likelihood": _safe_float(ctv.log_likelihood_),
        "concordance": _safe_float(getattr(ctv, "concordance_index_", None)),
        "coefficients": coefs,
        "result_text": (
            f"Cox regression with time-varying covariates on {n_subjects} subjects "
            f"({len(df)} interval rows, {n_events} events; {n_excluded} excluded). "
            f"Predictors: {', '.join(req.predictors)}."
        ),
    }
