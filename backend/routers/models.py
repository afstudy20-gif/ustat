import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services import store
from services.impute import apply_imputation

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


# ── Linear Regression ────────────────────────────────────────────────────────

class LinearRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"  # "listwise" | "median" | "mice"


@router.post("/linear")
def linear_regression(req: LinearRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = df[req.outcome].astype(float)
    model = sm.OLS(y, X).fit()

    coefs = []
    for var in model.params.index:
        coefs.append({
            "variable": str(var),
            "estimate": float(model.params[var]),
            "se": float(model.bse[var]),
            "t": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(model.conf_int().loc[var, 0]),
            "ci_high": float(model.conf_int().loc[var, 1]),
        })

    return {
        "model": "Linear Regression (OLS)",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_p": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "coefficients": coefs,
    }


# ── Logistic Regression ───────────────────────────────────────────────────────

class LogisticRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    scale_factors: Optional[dict] = None   # {column: divisor}  e.g. {"Platelet": 10000}
    selection: Optional[str] = "all"       # "all" | "p05" | "p10" | "forward" | "backward"
    imputation: Optional[str] = "listwise"  # "listwise" | "median" | "mice"


def _apply_scaling(df: pd.DataFrame, predictors: List[str], scale_factors: Optional[dict]):
    """Divide specified numeric columns by their scale factor and rename them.
    Returns (new_df, updated_predictor_list).
    """
    if not scale_factors:
        return df, predictors
    df = df.copy()
    new_predictors = []
    for pred in predictors:
        factor = scale_factors.get(pred)
        if factor is not None:
            try:
                factor_f = float(factor)
            except (TypeError, ValueError):
                factor_f = 1.0
        else:
            factor_f = 1.0
        if factor_f and factor_f != 1.0 and pred in df.columns:
            factor_label = int(factor_f) if factor_f == int(factor_f) else factor_f
            new_name = f"{pred} (per {factor_label} units)"
            df[new_name] = df[pred] / factor_f
            new_predictors.append(new_name)
        else:
            new_predictors.append(pred)
    return df, new_predictors


def _p_for_pred(pred: str, pvalues) -> float:
    """Return the representative p-value for a predictor from a fitted model.
    For numeric variables the name matches exactly; for categorical dummies it
    uses the most conservative (max) p-value across all dummy levels.
    """
    if pred in pvalues.index:
        return float(pvalues[pred])
    dummy_cols = [c for c in pvalues.index if c != "const" and c.startswith(pred + "_")]
    if dummy_cols:
        return float(pvalues[dummy_cols].max())
    return 1.0


def _uni_p_for_pred(pred: str, uni_results: dict) -> float:
    """Look up univariate p-value from uni_results dict (keyed by dummy names)."""
    if pred in uni_results:
        return uni_results[pred]["p"]
    matching = [v["p"] for k, v in uni_results.items() if k.startswith(pred + "_")]
    return min(matching) if matching else 1.0  # best (min) p across dummy levels


def _stepwise_forward(y, df: pd.DataFrame, pred_list: list, p_enter: float = 0.05) -> list:
    """Forward selection: greedily add the variable with lowest p < p_enter."""
    selected: list = []
    remaining = list(pred_list)
    while remaining:
        best_var, best_p = None, p_enter
        for var in remaining:
            candidate = selected + [var]
            X_enc = pd.get_dummies(df[candidate], drop_first=True).astype(float)
            X_const = sm.add_constant(X_enc, has_constant="add")
            try:
                m = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
                p = _p_for_pred(var, m.pvalues)
                if p < best_p:
                    best_p, best_var = p, var
            except Exception:
                pass
        if best_var is None:
            break
        selected.append(best_var)
        remaining.remove(best_var)
    return selected


def _stepwise_backward(y, df: pd.DataFrame, pred_list: list, p_remove: float = 0.10) -> list:
    """Backward elimination: iteratively remove the variable with highest p > p_remove."""
    selected = list(pred_list)
    while selected:
        X_enc = pd.get_dummies(df[selected], drop_first=True).astype(float)
        X_const = sm.add_constant(X_enc, has_constant="add")
        try:
            m = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
        except Exception:
            break
        worst_var, worst_p = None, p_remove
        for var in selected:
            p = _p_for_pred(var, m.pvalues)
            if p > worst_p:
                worst_p, worst_var = p, var
        if worst_var is None:
            break
        selected.remove(worst_var)
    return selected


@router.post("/logistic")
def logistic_regression(req: LogisticRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)
    X = pd.get_dummies(df[pred_list], drop_first=True).astype(float)
    X_const = sm.add_constant(X)
    y = df[req.outcome]  # outcome column not scaled
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.astype(int)

    model = sm.Logit(y, X_const).fit(disp=False)
    coefs = []
    for var in model.params.index:
        est = float(model.params[var])
        ci = model.conf_int()
        coefs.append({
            "variable": str(var),
            "log_odds": est,
            "odds_ratio": float(np.exp(est)),
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "or_ci_low": float(np.exp(ci.loc[var, 0])),
            "or_ci_high": float(np.exp(ci.loc[var, 1])),
        })

    return {
        "model": "Logistic Regression",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "pseudo_r2": float(model.prsquared),
        "log_likelihood": float(model.llf),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "coefficients": coefs,
    }


# ── Logistic OR Table (Univariate + Multivariate) ────────────────────────────

@router.post("/logistic_table")
def logistic_or_table(req: LogisticRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    # Apply unit scaling (renames columns & divides values)
    df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)

    # Encode outcome
    y_raw = df[req.outcome]
    if y_raw.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = y_raw.astype(int).values

    # Helper: fit logit and extract first non-const row OR all predictor rows
    def _fit_row(X_df, variable_names):
        X_enc = pd.get_dummies(X_df, drop_first=True).astype(float)
        X_const = sm.add_constant(X_enc, has_constant="add")
        try:
            m = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Model convergence error: {exc}")
        rows = {}
        ci = m.conf_int()
        for var in m.params.index:
            if var == "const":
                continue
            rows[var] = {
                "or": float(np.exp(m.params[var])),
                "ci_low": float(np.exp(ci.loc[var, 0])),
                "ci_high": float(np.exp(ci.loc[var, 1])),
                "p": float(m.pvalues[var]),
            }
        return rows

    # ── Univariate: one model per predictor (post-scaling) ───────────────────
    uni_results: dict = {}
    for pred in pred_list:
        try:
            rows = _fit_row(df[[pred]], [pred])
            for var, vals in rows.items():
                uni_results[var] = vals
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Univariate error for '{pred}': {exc}")

    # ── Variable selection for multivariate ──────────────────────────────────
    sel = (req.selection or "all").strip().lower()

    if sel == "p05":
        multi_pred_list = [p for p in pred_list if _uni_p_for_pred(p, uni_results) < 0.05]
        selection_label = "Univariate p < 0.05"
    elif sel == "p10":
        multi_pred_list = [p for p in pred_list if _uni_p_for_pred(p, uni_results) < 0.10]
        selection_label = "Univariate p < 0.10"
    elif sel == "forward":
        multi_pred_list = _stepwise_forward(y, df, pred_list, p_enter=0.05)
        selection_label = "Stepwise Forward (p_enter=0.05)"
    elif sel == "backward":
        multi_pred_list = _stepwise_backward(y, df, pred_list, p_remove=0.10)
        selection_label = "Stepwise Backward (p_remove=0.10)"
    else:  # "all"
        multi_pred_list = pred_list
        selection_label = "All variables (Enter)"

    # ── Multivariate: only selected predictors ────────────────────────────────
    multi_results: dict = {}
    if multi_pred_list:
        try:
            multi_results = _fit_row(df[multi_pred_list], multi_pred_list)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Multivariate error: {exc}")

    # ── Merge rows ────────────────────────────────────────────────────────────
    all_vars = list(dict.fromkeys(list(uni_results.keys()) + list(multi_results.keys())))
    table = []
    for var in all_vars:
        u = uni_results.get(var, {})
        m = multi_results.get(var, {})
        table.append({
            "variable": var,
            "uni_or": u.get("or"),
            "uni_ci_low": u.get("ci_low"),
            "uni_ci_high": u.get("ci_high"),
            "uni_p": u.get("p"),
            "multi_or": m.get("or"),
            "multi_ci_low": m.get("ci_low"),
            "multi_ci_high": m.get("ci_high"),
            "multi_p": m.get("p"),
        })

    return {
        "model": "Logistic OR Table",
        "outcome": req.outcome,
        "n": len(df),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "selection_method": selection_label,
        "n_multi": len(multi_pred_list),
        "n_total": len(pred_list),
        "table": table,
    }


# ── Kaplan-Meier Survival ─────────────────────────────────────────────────────

class KMRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    group_col: Optional[str] = None
    imputation: Optional[str] = "listwise"


def _safe_float(v):
    """Return float or None for inf/nan values that aren't JSON-serializable."""
    try:
        f = float(v)
        if np.isfinite(f):
            return f
        return None  # np.inf / -np.inf / nan → null in JSON
    except (TypeError, ValueError):
        return None


@router.post("/survival/km")
def kaplan_meier(req: KMRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    # Coerce to numeric first so imputation sees real NaN
    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col]    = pd.to_numeric(df_full[req.event_col],    errors="coerce")

    km_cols = [req.duration_col, req.event_col]
    df = apply_imputation(df_full, km_cols, req.imputation)
    n_excluded = n_total - len(df)

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after coercing duration/event columns to numeric. Check that both columns contain numbers.")

    results = []
    groups = df[req.group_col].unique() if req.group_col else [None]
    for grp in groups:
        subset = df[df[req.group_col] == grp] if req.group_col else df
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(
                subset[req.duration_col].astype(float),
                subset[req.event_col].astype(int),
                label=str(grp) if grp is not None else "All",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"KM fitting error: {exc}")

        sf = kmf.survival_function_.reset_index()
        sf.columns = ["time", "survival"]
        # Clean curve: replace inf/nan with None
        curve = [
            {"time": _safe_float(row["time"]), "survival": _safe_float(row["survival"])}
            for _, row in sf.iterrows()
        ]
        results.append({
            "group": str(grp) if grp is not None else "All",
            "n": int(len(subset)),
            "events": int(subset[req.event_col].sum()),
            "median_survival": _safe_float(kmf.median_survival_time_),
            "curve": curve,
        })

    # Log-rank test (only when group column supplied with ≥2 groups)
    logrank = None
    if req.group_col and len(groups) >= 2:
        try:
            if len(groups) == 2:
                g0 = df[df[req.group_col] == groups[0]]
                g1 = df[df[req.group_col] == groups[1]]
                lr = logrank_test(
                    g0[req.duration_col], g1[req.duration_col],
                    event_observed_A=g0[req.event_col].astype(int),
                    event_observed_B=g1[req.event_col].astype(int),
                )
                logrank = {"test": "Log-rank", "p": _safe_float(lr.p_value)}
            else:
                lr = multivariate_logrank_test(
                    df[req.duration_col], df[req.group_col], df[req.event_col].astype(int)
                )
                logrank = {"test": "Log-rank (multivariate)", "p": _safe_float(lr.p_value)}
        except Exception:
            pass

    return {
        "model": "Kaplan-Meier",
        "groups": results,
        "logrank": logrank,
        "n_total": n_total,
        "n_excluded": n_excluded,
        "imputation": req.imputation,
    }


# ── Cox Proportional Hazards ──────────────────────────────────────────────────

class CoxRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"


@router.post("/survival/cox")
def cox_regression(req: CoxRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col]    = pd.to_numeric(df_full[req.event_col],    errors="coerce")

    cox_cols = [req.duration_col, req.event_col] + req.predictors
    df = apply_imputation(df_full, cox_cols, req.imputation)
    n_excluded = n_total - len(df)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after coercing duration/event columns to numeric.")

    cph = CoxPHFitter()
    fit_df = df[[req.duration_col, req.event_col] + req.predictors]
    try:
        cph.fit(fit_df, duration_col=req.duration_col, event_col=req.event_col)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cox fitting error: {exc}")

    summary = cph.summary.reset_index()
    coefs = []
    for _, row in summary.iterrows():
        coefs.append({
            "variable": row["covariate"],
            "log_hr": _safe_float(row["coef"]),
            "hr": _safe_float(row["exp(coef)"]),
            "se": _safe_float(row["se(coef)"]),
            "z": _safe_float(row["z"]),
            "p": _safe_float(row["p"]),
            "hr_ci_low": _safe_float(row["exp(coef) lower 95%"]),
            "hr_ci_high": _safe_float(row["exp(coef) upper 95%"]),
        })

    return {
        "model": "Cox Proportional Hazards",
        "n": int(cph.event_observed.sum()),
        "n_total": n_total,
        "n_excluded": n_excluded,
        "imputation": req.imputation,
        "log_likelihood": _safe_float(cph.log_likelihood_),
        "concordance": _safe_float(cph.concordance_index_),
        "coefficients": coefs,
    }


# ── Restricted Cubic Splines ─────────────────────────────────────────────────

class RCSRequest(BaseModel):
    session_id: str
    predictor: str
    outcome: str
    covariates: List[str] = []
    n_knots: int = 4          # 3, 4, or 5
    ref_value: Optional[float] = None   # OR reference (median if None)
    model_type: str = "logistic"        # "logistic" | "linear"
    imputation: str = "listwise"

_KNOT_PERCENTILES = {
    3: [10, 50, 90],
    4: [5, 35, 65, 95],
    5: [5, 27.5, 50, 72.5, 95],
}

def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell restricted cubic spline basis (returns n_knots-2 spline columns)."""
    k = len(knots)
    cols = []
    kk = knots[-1]   # last knot
    k1 = knots[-2]   # second-to-last knot
    denom = (kk - knots[0]) ** 2
    for j in range(k - 2):
        t1 = np.maximum(x - knots[j], 0) ** 3
        t2 = np.maximum(x - k1, 0) ** 3
        t3 = np.maximum(x - kk, 0) ** 3
        col = t1 - ((kk - knots[j]) / (kk - k1)) * t2 + ((k1 - knots[j]) / (kk - k1)) * t3
        cols.append(col / denom)
    return np.column_stack(cols)

@router.post("/rcs")
def rcs_regression(req: RCSRequest):
    from services.impute import apply_imputation as _imp

    df_full = _get_df(req.session_id)
    cols_needed = [req.predictor, req.outcome] + req.covariates
    df = df_full[cols_needed].copy()
    for c in cols_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    n = len(df)
    if n < 10:
        raise HTTPException(status_code=400, detail="Not enough complete rows (need ≥ 10).")

    x_raw = df[req.predictor].values.astype(float)
    y     = df[req.outcome].values.astype(float)

    # Knot positions
    pcts = _KNOT_PERCENTILES.get(req.n_knots, _KNOT_PERCENTILES[4])
    knots = np.percentile(x_raw, pcts)

    # Build design matrix: intercept + x + spline cols + covariates
    spline_cols = _rcs_basis(x_raw, knots)
    X_parts = [np.ones(n), x_raw, spline_cols]
    if req.covariates:
        cov_mat = df[req.covariates].values.astype(float)
        X_parts.append(cov_mat)
    X = np.column_stack(X_parts)

    # Fit model
    try:
        if req.model_type == "logistic":
            result = sm.Logit(y, X).fit(disp=0, maxiter=200)
        else:
            result = sm.OLS(y, X).fit()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model fitting error: {exc}")

    # Synthetic X range for dose-response curve (1st–99th percentile of predictor)
    x_lo, x_hi = float(np.percentile(x_raw, 1)), float(np.percentile(x_raw, 99))
    x_syn = np.linspace(x_lo, x_hi, 200)
    sp_syn = _rcs_basis(x_syn, knots)

    # Build covariate means for synthetic rows
    n_cov = len(req.covariates)
    if n_cov:
        cov_means = df[req.covariates].mean().values.astype(float)
        X_syn = np.column_stack([np.ones(200), x_syn, sp_syn, np.tile(cov_means, (200, 1))])
    else:
        X_syn = np.column_stack([np.ones(200), x_syn, sp_syn])

    # Predict log-odds (or fitted values)
    lp_syn = X_syn @ result.params

    # Reference value
    ref_val = req.ref_value if req.ref_value is not None else float(np.median(x_raw))
    ref_val = float(np.clip(ref_val, x_lo, x_hi))
    # Find closest index in synthetic range
    ref_idx = int(np.argmin(np.abs(x_syn - ref_val)))
    lp_ref  = lp_syn[ref_idx]

    # Relative log-odds → OR
    rel_lp = lp_syn - lp_ref

    # 95% CI via delta method on the linear predictor difference
    # Var(lp(x) - lp(ref)) = (X_syn[i] - X_syn[ref]) @ cov @ (X_syn[i] - X_syn[ref])'
    cov_mat_param = result.cov_params()
    diffs = X_syn - X_syn[ref_idx]
    var_lp = np.einsum("ij,jk,ik->i", diffs, cov_mat_param, diffs)
    se_lp  = np.sqrt(np.maximum(var_lp, 0))
    z95    = 1.96

    if req.model_type == "logistic":
        or_vals  = np.exp(rel_lp)
        ci_low   = np.exp(rel_lp - z95 * se_lp)
        ci_high  = np.exp(rel_lp + z95 * se_lp)
    else:
        or_vals = rel_lp          # for linear: difference in means
        ci_low  = rel_lp - z95 * se_lp
        ci_high = rel_lp + z95 * se_lp

    # Summarise: n, events, AIC, knot positions
    events = int(y.sum()) if req.model_type == "logistic" else None

    def _ns(v):
        return None if (v is None or np.isnan(v) or np.isinf(v)) else round(float(v), 4)

    def _clean(arr):
        return [_ns(v) for v in arr]

    return {
        "predictor":   req.predictor,
        "outcome":     req.outcome,
        "model_type":  req.model_type,
        "n":           n,
        "n_events":    events,
        "n_knots":     req.n_knots,
        "knots":       [round(float(k), 2) for k in knots],
        "ref_value":   round(ref_val, 2),
        "aic":         _ns(result.aic),
        "x_values":    _clean(x_syn),
        "or_values":   _clean(or_vals),
        "ci_low":      _clean(ci_low),
        "ci_high":     _clean(ci_high),
        "x_data":      _clean(x_raw[:500]),  # raw data rug (first 500 points)
    }
