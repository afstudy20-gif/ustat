from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
from scipy.stats import chi2

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from services import store
from services.impute import apply_imputation
from services.assumptions import (
    check_cox_assumptions_from_ph_test,
    add_assumption_warnings_to_result,
)
from services.missing_data import (
    mice_multiple,
    pool_cox_results,
    missing_pattern_summary,
    add_missing_data_diagnostics,
)
from services.rcs_basis import (
    KNOT_PERCENTILES as _KNOT_PERCENTILES,
    rcs_basis as _rcs_basis,
    resolve_knots as _resolve_knots,
)

router = APIRouter()

# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    """Return float or None for inf/nan values that aren't JSON-serializable."""
    try:
        f = float(v)
        if np.isfinite(f):
            return f
        return None
    except (TypeError, ValueError):
        return None


def _clean(arr):
    return [_safe_float(v) for v in arr]


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
            logger.exception("KM fitting failed")
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
        logger.exception("KM logrank test failed")
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
            logger.exception("VIF calculation failed in Cox router")
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
async def cox_regression(req: CoxRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col] = pd.to_numeric(df_full[req.event_col], errors="coerce")

    cox_cols = [req.duration_col, req.event_col] + req.predictors
    imputation_method = req.imputation or "listwise"
    use_mice_pooled = False

    if imputation_method == "mice":
        imp_result = mice_multiple(df_full, cox_cols, n_imputations=5)
        imputed_dfs = imp_result.imputed_datasets

        individual_results = []
        for df_imp in imputed_dfs:
            try:
                cph_imp = CoxPHFitter()
                await asyncio.to_thread(cph_imp.fit, df_imp[cox_cols], duration_col=req.duration_col, event_col=req.event_col)
                loghrs = {var: float(np.log(cph_imp.hazard_ratios_.get(var, 1.0))) for var in req.predictors}
                individual_results.append({"coefficients": loghrs})
            except Exception:
                logger.exception("Cox fit failed for one imputation step")
                continue

        pooled = pool_cox_results(individual_results) if individual_results else {}

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
        await asyncio.to_thread(cph.fit, fit_df, duration_col=req.duration_col, event_col=req.event_col)
    except Exception as exc:
        logger.exception("Cox fitting failed")
        raise HTTPException(status_code=400, detail=f"Cox fitting error: {exc}")

    summary = cph.summary.reset_index()
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
        logger.exception("Proportional hazards test failed")
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

    if use_mice_pooled and 'pooled' in locals() and pooled:
        result["coefficients"] = pooled.get("coefficients", result.get("coefficients", []))
        result["pooled_from_imputations"] = True
        result["imputation"] = "mice (pooled)"

    cox_assumption_report = check_cox_assumptions_from_ph_test(ph_test)
    result = add_assumption_warnings_to_result(result, cox_assumption_report)

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
async def cox_time_varying(req: CoxTVRequest):
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
        await asyncio.to_thread(ctv.fit, fit_df, id_col=req.id_col, start_col=req.start_col, stop_col=req.stop_col, event_col=req.event_col)
    except Exception as exc:
        logger.exception("Cox-TV fitting failed")
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


# ── Restricted Cubic Splines (RCS) ───────────────────────────────────────────

class RCSRequest(BaseModel):
    session_id: str
    predictor: str
    outcome: Optional[str] = None
    covariates: List[str] = []
    n_knots: int = 4
    ref_value: Optional[float] = None
    model_type: str = "logistic"
    imputation: str = "listwise"
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    knot_positions: Optional[List[float]] = None
    interaction_covariates: Optional[List[str]] = None


@router.post("/rcs")
async def rcs_regression(req: RCSRequest):
    import statsmodels.api as sm_api
    if req.n_knots not in _KNOT_PERCENTILES:
        raise HTTPException(status_code=422, detail=f"n_knots must be 3, 4, or 5. Got: {req.n_knots}")

    model_type = (req.model_type or "logistic").lower()
    if model_type not in ("logistic", "linear", "cox"):
        raise HTTPException(status_code=422, detail=f"Unknown model_type: {req.model_type}")

    is_cox = model_type == "cox"

    if is_cox:
        if not req.duration_col or not req.event_col:
            raise HTTPException(status_code=422, detail="duration_col and event_col are required when model_type='cox'.")
        cols_needed = [req.predictor, req.duration_col, req.event_col] + req.covariates
    else:
        if not req.outcome:
            raise HTTPException(status_code=422, detail="outcome is required when model_type is 'logistic' or 'linear'.")
        cols_needed = [req.predictor, req.outcome] + req.covariates

    df_full = _get_df(req.session_id)
    missing_cols = [c for c in cols_needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found in session: {missing_cols}")

    df = df_full[cols_needed].copy()
    n_total = len(df)

    required_numeric = [req.predictor]
    if is_cox:
        required_numeric += [req.duration_col, req.event_col]
    else:
        required_numeric += [req.outcome]
    for c in required_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cov_df = None
    if req.covariates:
        cov_raw = df[req.covariates].copy()
        numeric_cov: list[str] = []
        cat_cov: list[str] = []
        for c in req.covariates:
            col = cov_raw[c]
            if pd.api.types.is_numeric_dtype(col):
                numeric_cov.append(c)
            else:
                coerced = pd.to_numeric(col, errors="coerce")
                if coerced.notna().mean() >= 0.8 and len(coerced.dropna().unique()) > 2:
                    cov_raw[c] = coerced
                    numeric_cov.append(c)
                else:
                    cat_cov.append(c)
        num_part = cov_raw[numeric_cov].apply(pd.to_numeric, errors="coerce") if numeric_cov else pd.DataFrame(index=cov_raw.index)
        cat_part = pd.get_dummies(cov_raw[cat_cov], drop_first=True, dummy_na=False) if cat_cov else pd.DataFrame(index=cov_raw.index)
        cov_df = pd.concat([num_part, cat_part], axis=1).astype(float)
        df = pd.concat([df.drop(columns=req.covariates), cov_df], axis=1)
    df = df.dropna()
    n = len(df)
    if n < 10:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows (need ≥ 10). Got {n} after dropping rows with missing predictor / outcome / covariates.")

    x_raw = df[req.predictor].values.astype(float)

    n_unique_x = len(np.unique(x_raw))
    if n_unique_x < req.n_knots + 2:
        raise HTTPException(status_code=422, detail=f"Predictor '{req.predictor}' has only {n_unique_x} unique values — need ≥ {req.n_knots + 2} for {req.n_knots}-knot spline.")

    if is_cox:
        duration = df[req.duration_col].values.astype(float)
        event = df[req.event_col].values.astype(float)
        if np.any(duration < 0):
            raise HTTPException(status_code=422, detail=f"duration_col '{req.duration_col}' must be ≥ 0.")
        unique_e = sorted(set(event.tolist()))
        if set(unique_e) - {0.0, 1.0}:
            raise HTTPException(status_code=422, detail=f"event_col '{req.event_col}' must be binary 0/1.")
    else:
        y = df[req.outcome].values.astype(float)
        if model_type == "logistic":
            unique_y = sorted(set(y.tolist()))
            if set(unique_y) - {0.0, 1.0}:
                raise HTTPException(status_code=422, detail=f"Logistic RCS requires binary 0/1 outcome.")

    try:
        knots = _resolve_knots(x_raw, req.n_knots, req.knot_positions, req.predictor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    spline_cols = _rcs_basis(x_raw, knots)

    if cov_df is not None and cov_df.shape[1] > 0:
        cov_names = list(cov_df.columns)
        cov_mat = df[cov_names].values.astype(float)
    else:
        cov_names = []
        cov_mat = None

    interaction_cov_names: list[str] = []
    interaction_extra_names: list[str] = []
    interaction_extra: list[np.ndarray] = []
    interaction_extra_meta: list[tuple[str, int]] = []
    if req.interaction_covariates:
        def _resolve_cov(name: str) -> list[str]:
            if name in cov_names:
                return [name]
            prefix = f"{name}_"
            return [c for c in cov_names if c.startswith(prefix)]

        spline_design = np.column_stack([x_raw, spline_cols])
        spline_part_names = ["lin"] + [f"sp{i+1}" for i in range(spline_cols.shape[1])]
        for cov in req.interaction_covariates:
            members = _resolve_cov(cov)
            if not members:
                raise HTTPException(status_code=422, detail=f"interaction_covariates entry '{cov}' is not in the selected covariates list.")
            interaction_cov_names.append(cov)
            for m in members:
                cov_vec = df[m].values.astype(float)
                for i in range(spline_design.shape[1]):
                    interaction_extra.append(spline_design[:, i] * cov_vec)
                    interaction_extra_names.append(f"{m}:{req.predictor}_{spline_part_names[i]}")
                    interaction_extra_meta.append((m, i))

    if interaction_extra:
        interaction_mat = np.column_stack(interaction_extra)
    else:
        interaction_mat = None

    try:
        if is_cox:
            feat_cols = [f"_x_lin"] + [f"_spl_{i}" for i in range(spline_cols.shape[1])]
            fit_df = pd.DataFrame(
                np.column_stack([x_raw, spline_cols]),
                columns=feat_cols,
                index=df.index,
            )
            for c in cov_names:
                fit_df[c] = df[c].values
            for i, ix_name in enumerate(interaction_extra_names):
                fit_df[ix_name] = interaction_extra[i]
            fit_df["_dur_"] = duration
            fit_df["_evt_"] = event
            cph = CoxPHFitter()
            await asyncio.to_thread(cph.fit, fit_df, duration_col="_dur_", event_col="_evt_")
            design_cols = feat_cols + cov_names + interaction_extra_names
            params = cph.params_.reindex(design_cols).values
            cov_params = cph.variance_matrix_.reindex(index=design_cols, columns=design_cols).values
            aic_val = None
            try:
                aic_val = float(getattr(cph, "AIC_partial_", np.nan))
                if np.isnan(aic_val):
                    aic_val = None
            except Exception:
                aic_val = None
            log_lik = float(cph.log_likelihood_)
            concordance = float(cph.concordance_index_)
            n_events = int(np.sum(event))
        else:
            X_parts = [np.ones(n), x_raw, spline_cols]
            if cov_mat is not None:
                X_parts.append(cov_mat)
            if interaction_mat is not None:
                X_parts.append(interaction_mat)
            X = np.column_stack(X_parts)
            if model_type == "logistic":
                result = await asyncio.to_thread(lambda: sm_api.Logit(y, X).fit(disp=0, maxiter=200))
            else:
                result = await asyncio.to_thread(lambda: sm_api.OLS(y, X).fit())
            params = result.params
            cov_params = result.cov_params()
            try:
                aic_val = float(result.aic)
                if np.isnan(aic_val) or np.isinf(aic_val):
                    aic_val = None
            except Exception:
                aic_val = None
            log_lik = float(getattr(result, "llf", np.nan))
            if np.isnan(log_lik) or np.isinf(log_lik):
                log_lik = None
            concordance = None
            n_events = int(y.sum()) if model_type == "logistic" else None
    except Exception as exc:
        logger.exception("RCS Model fitting failed")
        raise HTTPException(status_code=400, detail=f"Model fitting error: {exc}")

    interaction_result = None
    if interaction_extra_names:
        try:
            if is_cox:
                reduced_df = fit_df.drop(columns=interaction_extra_names)
                cph_red = CoxPHFitter()
                await asyncio.to_thread(cph_red.fit, reduced_df, duration_col="_dur_", event_col="_evt_")
                ll_red = float(cph_red.log_likelihood_)
                ll_full = float(log_lik)
            else:
                X_red_parts = [np.ones(n), x_raw, spline_cols]
                if cov_mat is not None:
                    X_red_parts.append(cov_mat)
                X_red = np.column_stack(X_red_parts)
                if model_type == "logistic":
                    res_red = await asyncio.to_thread(lambda: sm_api.Logit(y, X_red).fit(disp=0, maxiter=200))
                else:
                    res_red = await asyncio.to_thread(lambda: sm_api.OLS(y, X_red).fit())
                ll_red = float(getattr(res_red, "llf", np.nan))
                ll_full = float(log_lik) if log_lik is not None else float(getattr(result, "llf", np.nan))
            lr_stat = 2.0 * (ll_full - ll_red)
            df_lr = len(interaction_extra_names)
            p_lr = float(chi2.sf(lr_stat, df=df_lr))
            interaction_result = {
                "covariates": interaction_cov_names,
                "lr_stat": round(lr_stat, 4),
                "df": df_lr,
                "p": round(p_lr, 6),
                "log_lik_full": round(ll_full, 4),
                "log_lik_reduced": round(ll_red, 4),
            }
        except Exception as exc:
            logger.exception("RCS Interaction LR test failed")
            interaction_result = {"covariates": interaction_cov_names, "error": str(exc)}

    x_lo, x_hi = float(np.percentile(x_raw, 1)), float(np.percentile(x_raw, 99))
    x_syn = np.linspace(x_lo, x_hi, 200)
    sp_syn = _rcs_basis(x_syn, knots)

    cov_means_by_name: dict[str, float] = {}
    if cov_names:
        cov_means = cov_mat.mean(axis=0)
        for cn, mean_val in zip(cov_names, cov_means):
            cov_means_by_name[cn] = float(mean_val)

    spline_design_syn = np.column_stack([x_syn, sp_syn])

    if is_cox:
        if cov_mat is not None:
            X_syn = np.column_stack([x_syn, sp_syn, np.tile(cov_means, (200, 1))])
        else:
            X_syn = np.column_stack([x_syn, sp_syn])
    else:
        if cov_mat is not None:
            X_syn = np.column_stack([np.ones(200), x_syn, sp_syn, np.tile(cov_means, (200, 1))])
        else:
            X_syn = np.column_stack([np.ones(200), x_syn, sp_syn])

    if interaction_extra_meta:
        ix_syn = np.column_stack([
            spline_design_syn[:, spi] * cov_means_by_name.get(member, 0.0)
            for (member, spi) in interaction_extra_meta
        ])
        X_syn = np.column_stack([X_syn, ix_syn])

    lp_syn = X_syn @ params

    ref_val = req.ref_value if req.ref_value is not None else float(np.median(x_raw))
    ref_val = float(np.clip(ref_val, x_lo, x_hi))
    ref_idx = int(np.argmin(np.abs(x_syn - ref_val)))
    lp_ref = lp_syn[ref_idx]
    rel_lp = lp_syn - lp_ref

    diffs = X_syn - X_syn[ref_idx]
    var_lp = np.einsum("ij,jk,ik->i", diffs, cov_params, diffs)
    se_lp = np.sqrt(np.maximum(var_lp, 0))
    z95 = 1.96

    if is_cox or model_type == "logistic":
        or_vals = np.exp(rel_lp)
        ci_low = np.exp(rel_lp - z95 * se_lp)
        ci_high = np.exp(rel_lp + z95 * se_lp)
    else:
        or_vals = rel_lp
        ci_low = rel_lp - z95 * se_lp
        ci_high = rel_lp + z95 * se_lp

    effect_type = "HR" if is_cox else ("OR" if model_type == "logistic" else "mean_diff")

    cov_summary = []
    if cov_names:
        n_pre_cov = (0 if is_cox else 2) + (1 + spline_cols.shape[1])
        for offset, name in enumerate(cov_names):
            i = n_pre_cov + offset
            beta = float(params[i]) if i < len(params) else None
            se = float(np.sqrt(max(cov_params[i, i], 0.0))) if i < len(params) else None
            cov_summary.append({
                "name": name,
                "coef": round(beta, 6) if beta is not None else None,
                "effect": round(float(np.exp(beta)), 4) if (is_cox or model_type == "logistic") and beta is not None else (round(beta, 4) if beta is not None else None),
                "se": round(se, 6) if se is not None else None,
            })

    nonlin_p = None
    nonlin_wald = None
    nonlin_df = None
    try:
        lin_idx = 0 if is_cox else 1
        nl_start = lin_idx + 1
        nl_end = nl_start + spline_cols.shape[1]
        if nl_end > nl_start:
            idx = np.arange(nl_start, nl_end)
            beta_nl = np.asarray(params)[idx]
            cov_nl = np.asarray(cov_params)[np.ix_(idx, idx)]
            wald = float(beta_nl @ np.linalg.solve(cov_nl, beta_nl))
            df_nl = int(len(idx))
            nonlin_wald = round(wald, 4)
            nonlin_df = df_nl
            nonlin_p = round(float(chi2.sf(wald, df=df_nl)), 6)
    except Exception:
        logger.exception("Nonlinearity test failed")

    crude_block = None
    if cov_names or interaction_extra_names:
        try:
            if is_cox:
                feat_cols_c = [f"_x_lin"] + [f"_spl_{i}" for i in range(spline_cols.shape[1])]
                fit_df_c = pd.DataFrame(
                    np.column_stack([x_raw, spline_cols]),
                    columns=feat_cols_c,
                    index=df.index,
                )
                fit_df_c["_dur_"] = duration
                fit_df_c["_evt_"] = event
                cph_c = CoxPHFitter()
                await asyncio.to_thread(cph_c.fit, fit_df_c, duration_col="_dur_", event_col="_evt_")
                design_cols_c = feat_cols_c
                params_c = cph_c.params_.reindex(design_cols_c).values
                cov_params_c = cph_c.variance_matrix_.reindex(index=design_cols_c, columns=design_cols_c).values
                X_syn_c = np.column_stack([x_syn, sp_syn])
            else:
                X_parts_c = [np.ones(n), x_raw, spline_cols]
                X_c = np.column_stack(X_parts_c)
                if model_type == "logistic":
                    res_c = await asyncio.to_thread(lambda: sm_api.Logit(y, X_c).fit(disp=0, maxiter=200))
                else:
                    res_c = await asyncio.to_thread(lambda: sm_api.OLS(y, X_c).fit())
                params_c = res_c.params
                cov_params_c = res_c.cov_params()
                X_syn_c = np.column_stack([np.ones(200), x_syn, sp_syn])

            lp_syn_c = X_syn_c @ params_c
            ref_idx_c = int(np.argmin(np.abs(x_syn - ref_val)))
            rel_lp_c = lp_syn_c - lp_syn_c[ref_idx_c]
            diffs_c = X_syn_c - X_syn_c[ref_idx_c]
            var_lp_c = np.einsum("ij,jk,ik->i", diffs_c, cov_params_c, diffs_c)
            se_lp_c = np.sqrt(np.maximum(var_lp_c, 0))
            if is_cox or model_type == "logistic":
                or_vals_c = np.exp(rel_lp_c)
                ci_low_c = np.exp(rel_lp_c - z95 * se_lp_c)
                ci_high_c = np.exp(rel_lp_c + z95 * se_lp_c)
            else:
                or_vals_c = rel_lp_c
                ci_low_c = rel_lp_c - z95 * se_lp_c
                ci_high_c = rel_lp_c + z95 * se_lp_c

            crude_nl_p = None
            try:
                lin_idx_c = 0 if is_cox else 1
                nl_start_c = lin_idx_c + 1
                nl_end_c = nl_start_c + spline_cols.shape[1]
                idx_c = np.arange(nl_start_c, nl_end_c)
                beta_nl_c = np.asarray(params_c)[idx_c]
                cov_nl_c = np.asarray(cov_params_c)[np.ix_(idx_c, idx_c)]
                w_c = float(beta_nl_c @ np.linalg.solve(cov_nl_c, beta_nl_c))
                crude_nl_p = round(float(chi2.sf(w_c, df=int(len(idx_c)))), 6)
            except Exception:
                logger.exception("Crude RCS nonlinearity test failed")

            crude_block = {
                "x_values": _clean(x_syn),
                "or_values": _clean(or_vals_c),
                "ci_low": _clean(ci_low_c),
                "ci_high": _clean(ci_high_c),
                "nonlinearity_p": crude_nl_p,
            }
        except Exception:
            logger.exception("Crude RCS fit failed")
            crude_block = None

    return {
        "predictor": req.predictor,
        "outcome": req.outcome,
        "duration_col": req.duration_col,
        "event_col": req.event_col,
        "model_type": model_type,
        "effect_type": effect_type,
        "n": n,
        "n_total": n_total,
        "n_excluded": n_total - n,
        "n_events": n_events,
        "n_knots": req.n_knots,
        "knots": [round(float(kn), 2) for kn in knots],
        "knot_positions_custom": req.knot_positions is not None,
        "ref_value": round(ref_val, 4),
        "aic": _safe_float(aic_val),
        "log_likelihood": _safe_float(log_lik),
        "concordance": _safe_float(concordance),
        "covariates_requested": list(req.covariates or []),
        "covariates_used": cov_names,
        "covariates_summary": cov_summary,
        "interaction": interaction_result,
        "interaction_terms": interaction_extra_names,
        "nonlinearity_wald": nonlin_wald,
        "nonlinearity_df": nonlin_df,
        "nonlinearity_p": nonlin_p,
        "x_values": _clean(x_syn),
        "or_values": _clean(or_vals),
        "ci_low": _clean(ci_low),
        "ci_high": _clean(ci_high),
        "x_data": _clean(x_raw[:500]),
        "crude": crude_block,
    }


# ── Multivariable Cox-RCS Endpoint ─────────────────────────────────────────────

class SplineTerm(BaseModel):
    column: str
    n_knots: int = 4
    knot_positions: Optional[List[float]] = None
    ref_value: Optional[float] = None


class CoxRCSRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    spline_terms: List[SplineTerm]
    covariates: List[str] = []
    include_interaction: bool = False
    imputation: Optional[str] = "listwise"
    grid_size: int = 50


@router.post("/survival/cox_rcs")
async def cox_rcs(req: CoxRCSRequest):
    if not (1 <= len(req.spline_terms) <= 2):
        raise HTTPException(status_code=422, detail="spline_terms must contain 1 or 2 entries.")

    for term in req.spline_terms:
        if term.n_knots not in _KNOT_PERCENTILES:
            raise HTTPException(status_code=422, detail=f"n_knots for '{term.column}' must be 3, 4, or 5. Got: {term.n_knots}")

    if req.include_interaction and len(req.spline_terms) != 2:
        raise HTTPException(status_code=422, detail="include_interaction requires exactly 2 spline_terms.")

    df_full = _get_df(req.session_id)
    spline_cols = [t.column for t in req.spline_terms]
    cols_needed = list(dict.fromkeys(spline_cols + [req.duration_col, req.event_col] + req.covariates))
    missing_cols = [c for c in cols_needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found in session: {missing_cols}")

    df = df_full[cols_needed].copy()
    for c in cols_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if req.imputation and req.imputation != "listwise":
        df = apply_imputation(df, cols_needed, req.imputation)
    else:
        df = df.dropna()
    n = len(df)
    if n < 15:
        raise HTTPException(status_code=400, detail="Not enough complete rows (need ≥ 15).")

    duration = df[req.duration_col].values.astype(float)
    event = df[req.event_col].values.astype(float)
    if np.any(duration < 0):
        raise HTTPException(status_code=422, detail=f"duration_col '{req.duration_col}' must be ≥ 0.")
    if set(sorted(set(event.tolist()))) - {0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"event_col '{req.event_col}' must be binary 0/1.")
    if event.sum() < 5:
        raise HTTPException(status_code=400, detail="Need ≥ 5 events to fit a Cox model.")

    term_info = []
    for ti, term in enumerate(req.spline_terms):
        x_raw = df[term.column].values.astype(float)
        n_unique = len(np.unique(x_raw))
        if n_unique < term.n_knots + 2:
            raise HTTPException(status_code=422, detail=f"Spline term '{term.column}' has only {n_unique} unique values — need ≥ {term.n_knots + 2} for {term.n_knots}-knot spline.")
        try:
            knots = _resolve_knots(x_raw, term.n_knots, term.knot_positions, term.column)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        sp = _rcs_basis(x_raw, knots)
        lin_col = f"t{ti}_{term.column}_lin"
        sp_cols = [f"t{ti}_{term.column}_sp{i}" for i in range(sp.shape[1])]
        full_cols = [lin_col] + sp_cols
        term_info.append({
            "column": term.column,
            "knots": knots,
            "x_raw": x_raw,
            "design": np.column_stack([x_raw, sp]),
            "col_names": full_cols,
            "ref_value": term.ref_value if term.ref_value is not None else float(np.median(x_raw)),
        })

    feat_arrays = []
    feat_names: List[str] = []
    for ti in term_info:
        feat_arrays.append(ti["design"])
        feat_names.extend(ti["col_names"])

    cov_df = pd.DataFrame(index=df.index)
    if req.covariates:
        cov_raw = df_full.loc[df.index, req.covariates].copy()
        cov_df = pd.get_dummies(cov_raw, drop_first=True, dummy_na=False)
        for c in cov_df.columns:
            cov_df[c] = pd.to_numeric(cov_df[c], errors="coerce")
        cov_df = cov_df.dropna()
        df_aligned = df.loc[cov_df.index]
        duration = df_aligned[req.duration_col].values.astype(float)
        event = df_aligned[req.event_col].values.astype(float)
        for ti, term in enumerate(req.spline_terms):
            x_raw = df_aligned[term.column].values.astype(float)
            sp = _rcs_basis(x_raw, term_info[ti]["knots"])
            term_info[ti]["design"] = np.column_stack([x_raw, sp])
            term_info[ti]["x_raw"] = x_raw
        feat_arrays = [ti["design"] for ti in term_info]
        n = len(df_aligned)
        if n < 15:
            raise HTTPException(status_code=400, detail="Not enough complete rows after covariate handling (need ≥ 15).")

    main_design = np.column_stack(feat_arrays) if feat_arrays else np.empty((n, 0))

    interaction_design = None
    interaction_names: List[str] = []
    if req.include_interaction:
        a = term_info[0]["design"]
        b = term_info[1]["design"]
        a_names = term_info[0]["col_names"]
        b_names = term_info[1]["col_names"]
        ix_cols = []
        for i in range(a.shape[1]):
            for j in range(b.shape[1]):
                ix_cols.append(a[:, i] * b[:, j])
                interaction_names.append(f"ix_{a_names[i]}_x_{b_names[j]}")
        interaction_design = np.column_stack(ix_cols)

    full_design_arrays = [main_design]
    full_names = list(feat_names)
    if interaction_design is not None:
        full_design_arrays.append(interaction_design)
        full_names = full_names + interaction_names
    cov_names = list(cov_df.columns)
    if cov_names:
        full_design_arrays.append(cov_df.values.astype(float))
        full_names = full_names + cov_names

    full_design = np.column_stack(full_design_arrays) if full_design_arrays else np.empty((n, 0))

    full_df = pd.DataFrame(full_design, columns=full_names, index=range(n))
    full_df["_dur_"] = duration
    full_df["_evt_"] = event

    try:
        cph_full = CoxPHFitter()
        await asyncio.to_thread(cph_full.fit, full_df, duration_col="_dur_", event_col="_evt_")
    except Exception as exc:
        logger.exception("Cox-RCS fitting failed")
        raise HTTPException(status_code=400, detail=f"Cox-RCS fitting error: {exc}")

    params_full = cph_full.params_.reindex(full_names).values
    cov_full = cph_full.variance_matrix_.reindex(index=full_names, columns=full_names).values
    se_full = cph_full.standard_errors_.reindex(full_names).values
    p_full = None
    try:
        p_full = cph_full.summary["p"].reindex(full_names).values
    except Exception:
        logger.exception("Cox-RCS p-value extraction failed")
    ci_low_full = cph_full.confidence_intervals_.iloc[:, 0].reindex(full_names).values
    ci_high_full = cph_full.confidence_intervals_.iloc[:, 1].reindex(full_names).values
    log_lik_full = float(cph_full.log_likelihood_)

    coefs = []
    for i, name in enumerate(full_names):
        coef = float(params_full[i])
        se = float(se_full[i])
        z = coef / se if se > 0 else None
        p = float(p_full[i]) if (p_full is not None and not np.isnan(p_full[i])) else None
        coefs.append({
            "name": name,
            "coef": coef,
            "hr": float(np.exp(coef)),
            "se": se,
            "z": z,
            "p": p,
            "ci_low": float(np.exp(ci_low_full[i])),
            "ci_high": float(np.exp(ci_high_full[i])),
        })

    nonlinearity = {}
    for ti, term in enumerate(req.spline_terms):
        sp_names = term_info[ti]["col_names"][1:]
        idx = [full_names.index(n) for n in sp_names]
        if not idx:
            continue
        b = params_full[idx]
        cv = cov_full[np.ix_(idx, idx)]
        try:
            wald = float(b @ np.linalg.solve(cv, b))
            p_nl = float(chi2.sf(wald, df=len(idx)))
        except Exception:
            logger.exception("Cox-RCS nonlinearity Wald test failed")
            wald = None
            p_nl = None
        nonlinearity[term.column] = {
            "wald": wald,
            "df": len(idx),
            "p": p_nl,
        }

    interaction_result = None
    if req.include_interaction and interaction_design is not None:
        reduced_names = [n for n in full_names if n not in interaction_names]
        reduced_df = full_df[reduced_names + ["_dur_", "_evt_"]].copy()
        try:
            cph_red = CoxPHFitter()
            await asyncio.to_thread(cph_red.fit, reduced_df, duration_col="_dur_", event_col="_evt_")
            ll_red = float(cph_red.log_likelihood_)
            lr_stat = 2.0 * (log_lik_full - ll_red)
            df_lr = len(interaction_names)
            p_lr = float(chi2.sf(lr_stat, df=df_lr))
            interaction_result = {
                "lr_stat": lr_stat,
                "df": df_lr,
                "p": p_lr,
                "log_lik_full": log_lik_full,
                "log_lik_reduced": ll_red,
            }
        except Exception as exc:
            logger.exception("Cox-RCS interaction LR test failed")
            interaction_result = {"error": f"interaction LR fit failed: {exc}"}

    curves_1d = []
    cov_means = cov_df.values.astype(float).mean(axis=0) if cov_names else np.array([])

    for ti, term in enumerate(req.spline_terms):
        x_raw = term_info[ti]["x_raw"]
        x_lo, x_hi = float(np.percentile(x_raw, 1)), float(np.percentile(x_raw, 99))
        x_syn = np.linspace(x_lo, x_hi, 200)
        sp_syn = _rcs_basis(x_syn, term_info[ti]["knots"])
        this_design = np.column_stack([x_syn, sp_syn])
        other_idx = 1 - ti if len(term_info) == 2 else None
        other_design = None
        if other_idx is not None:
            other_term = term_info[other_idx]
            ref_x = other_term["ref_value"]
            ref_sp = _rcs_basis(np.array([ref_x]), other_term["knots"]).flatten()
            other_vec = np.concatenate([[ref_x], ref_sp])
            other_design = np.tile(other_vec, (200, 1))

        if ti == 0:
            main_syn = this_design if other_design is None else np.column_stack([this_design, other_design])
        else:
            main_syn = other_design if other_design is None else np.column_stack([other_design, this_design]) if other_idx == 0 else None
            if main_syn is None:
                main_syn = np.column_stack([other_design, this_design])

        if req.include_interaction and interaction_design is not None:
            a_syn = main_syn[:, :term_info[0]["design"].shape[1]]
            b_syn = main_syn[:, term_info[0]["design"].shape[1]:term_info[0]["design"].shape[1] + term_info[1]["design"].shape[1]]
            ix_syn = np.column_stack([a_syn[:, i] * b_syn[:, j]
                                       for i in range(a_syn.shape[1])
                                       for j in range(b_syn.shape[1])])
            main_syn = np.column_stack([main_syn, ix_syn])

        if cov_names:
            main_syn = np.column_stack([main_syn, np.tile(cov_means, (200, 1))])

        lp_syn = main_syn @ params_full

        own_ref = term_info[ti]["ref_value"]
        ref_idx_syn = int(np.argmin(np.abs(x_syn - own_ref)))
        ref_row = main_syn[ref_idx_syn].copy()

        diffs = main_syn - ref_row
        var_lp = np.einsum("ij,jk,ik->i", diffs, cov_full, diffs)
        se_lp = np.sqrt(np.maximum(var_lp, 0))
        rel_lp = lp_syn - lp_syn[ref_idx_syn]
        hr = np.exp(rel_lp)
        ci_low = np.exp(rel_lp - 1.96 * se_lp)
        ci_high = np.exp(rel_lp + 1.96 * se_lp)

        curves_1d.append({
            "column": term.column,
            "x": _clean(x_syn),
            "hr": _clean(hr),
            "lower": _clean(ci_low),
            "upper": _clean(ci_high),
            "knots": [round(float(k), 2) for k in term_info[ti]["knots"]],
            "ref": round(float(own_ref), 4),
        })

    surface_2d = None
    if req.include_interaction and interaction_design is not None and len(term_info) == 2:
        g = max(10, min(int(req.grid_size or 50), 100))
        xa = term_info[0]["x_raw"]
        xb = term_info[1]["x_raw"]
        a_lo, a_hi = float(np.percentile(xa, 1)), float(np.percentile(xa, 99))
        b_lo, b_hi = float(np.percentile(xb, 1)), float(np.percentile(xb, 99))
        a_grid = np.linspace(a_lo, a_hi, g)
        b_grid = np.linspace(b_lo, b_hi, g)
        A, B = np.meshgrid(a_grid, b_grid)
        a_flat = A.flatten()
        b_flat = B.flatten()
        a_basis = np.column_stack([a_flat, _rcs_basis(a_flat, term_info[0]["knots"])])
        b_basis = np.column_stack([b_flat, _rcs_basis(b_flat, term_info[1]["knots"])])
        ix_flat = np.column_stack([a_basis[:, i] * b_basis[:, j]
                                    for i in range(a_basis.shape[1])
                                    for j in range(b_basis.shape[1])])
        cov_block = np.tile(cov_means, (a_flat.size, 1)) if cov_names else np.empty((a_flat.size, 0))
        design = np.column_stack([a_basis, b_basis, ix_flat, cov_block])
        lp = design @ params_full
        ref_a = term_info[0]["ref_value"]
        ref_b = term_info[1]["ref_value"]
        ra_basis = np.column_stack([[ref_a], _rcs_basis(np.array([ref_a]), term_info[0]["knots"])])
        rb_basis = np.column_stack([[ref_b], _rcs_basis(np.array([ref_b]), term_info[1]["knots"])])
        rix = np.column_stack([ra_basis[:, i] * rb_basis[:, j]
                                for i in range(ra_basis.shape[1])
                                for j in range(rb_basis.shape[1])])
        rcov = np.tile(cov_means, (1, 1)) if cov_names else np.empty((1, 0))
        ref_design = np.column_stack([ra_basis, rb_basis, rix, rcov])
        lp_ref = float((ref_design @ params_full)[0])
        hr_flat = np.exp(lp - lp_ref)
        hr_grid = hr_flat.reshape(B.shape)

        def _gclean(mat):
            out = []
            for row in mat:
                rrow = []
                for v in row:
                    fv = float(v)
                    rrow.append(None if (np.isnan(fv) or np.isinf(fv)) else round(fv, 4))
                out.append(rrow)
            return out

        surface_2d = {
            "x_col": term_info[0]["column"],
            "y_col": term_info[1]["column"],
            "x": [round(float(v), 4) for v in a_grid],
            "y": [round(float(v), 4) for v in b_grid],
            "hr": _gclean(hr_grid),
            "ref": {term_info[0]["column"]: round(float(ref_a), 4),
                      term_info[1]["column"]: round(float(ref_b), 4)},
        }

    aic_partial = None
    try:
        aic_partial = float(getattr(cph_full, "AIC_partial_", np.nan))
        if np.isnan(aic_partial):
            aic_partial = None
    except Exception:
        logger.exception("Cox-RCS partial AIC extraction failed")

    return {
        "n": int(n),
        "n_events": int(event.sum()),
        "concordance": float(cph_full.concordance_index_),
        "log_likelihood": log_lik_full,
        "aic": aic_partial,
        "spline_terms": [
            {
                "column": t.column,
                "n_knots": t.n_knots,
                "knots": [round(float(k), 2) for k in term_info[i]["knots"]],
                "knot_positions_custom": t.knot_positions is not None,
                "ref": round(float(term_info[i]["ref_value"]), 4),
            }
            for i, t in enumerate(req.spline_terms)
        ],
        "covariates": req.covariates,
        "include_interaction": req.include_interaction,
        "coefficients": coefs,
        "nonlinearity": nonlinearity,
        "interaction": interaction_result,
        "curves_1d": curves_1d,
        "surface_2d": surface_2d,
    }
