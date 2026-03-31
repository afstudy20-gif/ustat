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
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


# ── Linear Regression ────────────────────────────────────────────────────────

class LinearRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/linear")
def linear_regression(req: LinearRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = df[req.outcome].astype(float)
    base = sm.OLS(y, X)
    model = base.fit(cov_type="HC3", use_t=True) if req.robust_se else base.fit()

    coefs = []
    ci = model.conf_int()
    for var in model.params.index:
        coefs.append({
            "variable": str(var),
            "estimate": float(model.params[var]),
            "se": float(model.bse[var]),
            "t": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
        })

    # ── Predictor metadata for the interactive prediction panel ──────────────
    predictor_info: dict = {}
    for col in req.predictors:
        if col not in df_full.columns:
            continue
        s = df_full[col].dropna()
        if len(s) == 0:
            continue
        if pd.api.types.is_numeric_dtype(s):
            predictor_info[col] = {
                "type": "numeric",
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "median": float(s.median()),
            }
        else:
            vc = s.value_counts()
            predictor_info[col] = {
                "type": "categorical",
                "categories": vc.index.astype(str).tolist(),
                "counts": [int(v) for v in vc.values],
            }

    return {
        "model": f"Linear Regression (OLS){' [Robust SE]' if req.robust_se else ''}",
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
        "residual_se": float(np.sqrt(model.mse_resid)),
        "df_resid": int(model.df_resid),
        "predictors": req.predictors,
        "predictor_info": predictor_info,
    }


# ── Logistic Regression ───────────────────────────────────────────────────────

class LogisticRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    scale_factors: Optional[dict] = None
    selection: Optional[str] = "all"
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


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
        y = pd.to_numeric(y, errors="coerce")
        unique_vals = sorted(y.dropna().unique())
        if set(unique_vals) - {0, 1, 0.0, 1.0}:
            raise HTTPException(status_code=422, detail=f"Logistic regression requires a binary 0/1 outcome. Found values: {unique_vals[:10]}")
        y = y.astype(int)

    if len(set(y)) < 2:
        raise HTTPException(status_code=422, detail="Outcome column has only one unique value — logistic regression requires both 0 and 1.")

    cov_type = "HC3" if req.robust_se else "nonrobust"
    model = sm.Logit(y, X_const).fit(disp=False, cov_type=cov_type)
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
        "model": f"Logistic Regression{' [Robust SE]' if req.robust_se else ''}",
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


# ── Poisson Regression ───────────────────────────────────────────────────────

class PoissonRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/poisson")
def poisson_regression(req: PoissonRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if y.isna().all():
        raise HTTPException(status_code=422, detail="Outcome column has no numeric values.")
    if (y.dropna() < 0).any():
        raise HTTPException(status_code=422, detail="Poisson regression requires non-negative integer counts. Negative values found.")
    if (y.dropna() % 1 != 0).any():
        raise HTTPException(status_code=422, detail="Poisson regression requires integer counts. Fractional values found — consider Gamma regression instead.")
    cov_type = "HC3" if req.robust_se else "nonrobust"
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type=cov_type)
    ci = model.conf_int()
    coefs = []
    for var in model.params.index:
        est = float(model.params[var])
        coefs.append({
            "variable": str(var),
            "log_irr": est,
            "irr": float(np.exp(est)),
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
            "irr_ci_low":  float(np.exp(ci.loc[var, 0])),
            "irr_ci_high": float(np.exp(ci.loc[var, 1])),
        })
    return {
        "model": f"Poisson Regression{' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
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

    # Encode outcome — must be binary 0/1
    y_raw = df[req.outcome]
    if y_raw.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y_num = pd.to_numeric(y_raw, errors="coerce")
        unique_vals = sorted(y_num.dropna().unique())
        if set(unique_vals) - {0, 1, 0.0, 1.0}:
            raise HTTPException(status_code=422, detail=f"Logistic regression requires a binary 0/1 outcome. Found: {unique_vals[:10]}")
        y = y_num.astype(int).values
    if len(set(y)) < 2:
        raise HTTPException(status_code=422, detail="Outcome has only one unique value — needs both 0 and 1.")

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

    # Validate event indicator is binary 0/1
    event_vals = sorted(df[req.event_col].dropna().unique())
    if set(event_vals) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1 (0=censored, 1=event). Found: {event_vals[:10]}")

    # Validate duration is positive
    if (df[req.duration_col] < 0).any():
        raise HTTPException(status_code=422, detail="Duration column contains negative values. All durations must be ≥ 0.")

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

    # Validate event indicator is binary 0/1
    event_vals = sorted(df[req.event_col].dropna().unique())
    if set(event_vals) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1. Found: {event_vals[:10]}")
    if (df[req.duration_col] < 0).any():
        raise HTTPException(status_code=422, detail="Duration column contains negative values.")

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

    if req.n_knots not in _KNOT_PERCENTILES:
        raise HTTPException(status_code=422, detail=f"n_knots must be 3, 4, or 5. Got: {req.n_knots}")

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

    # Validate predictor has enough unique values for the spline
    n_unique_x = len(np.unique(x_raw))
    if n_unique_x < req.n_knots + 2:
        raise HTTPException(status_code=422, detail=f"Predictor '{req.predictor}' has only {n_unique_x} unique values — need ≥ {req.n_knots + 2} for {req.n_knots}-knot spline.")

    # Validate logistic outcome
    if req.model_type == "logistic":
        unique_y = sorted(set(y.tolist()))
        if set(unique_y) - {0.0, 1.0}:
            raise HTTPException(status_code=422, detail=f"Logistic RCS requires binary 0/1 outcome. Found: {unique_y[:10]}")

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


# ── Polynomial / Non-linear Regression ───────────────────────────────────────

class PolynomialRequest(BaseModel):
    session_id: str
    outcome: str
    predictor: str
    degree: int = 2          # 1–5
    covariates: List[str] = []
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/polynomial")
def polynomial_regression(req: PolynomialRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    cols = [req.outcome, req.predictor] + req.covariates
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    if req.degree < 1 or req.degree > 10:
        raise HTTPException(status_code=422, detail="Polynomial degree must be between 1 and 10.")

    x = df[req.predictor].astype(float)
    n_unique = x.nunique()
    if n_unique <= req.degree:
        raise HTTPException(status_code=422, detail=f"Predictor has only {n_unique} unique values — need more than degree ({req.degree}) for polynomial fit.")

    X_parts = {"const": np.ones(len(df))}
    for d in range(1, req.degree + 1):
        X_parts[f"{req.predictor}^{d}"] = x ** d
    for cov in req.covariates:
        X_parts[cov] = df[cov].astype(float)
    X = pd.DataFrame(X_parts)
    y = df[req.outcome].astype(float)

    base = sm.OLS(y, X)
    model = base.fit(cov_type="HC3", use_t=True) if req.robust_se else base.fit()
    ci = model.conf_int()

    coefs = []
    for var in model.params.index:
        coefs.append({
            "variable": str(var),
            "estimate": float(model.params[var]),
            "se": float(model.bse[var]),
            "t": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
        })

    # Curve for plotting (hold covariates at mean)
    x_lo, x_hi = float(x.min()), float(x.max())
    xs = np.linspace(x_lo, x_hi, 200)
    X_curve = np.column_stack([xs ** d for d in range(0, req.degree + 1)])
    cov_means = [float(df[c].mean()) for c in req.covariates]
    if cov_means:
        X_curve = np.hstack([X_curve, np.tile(cov_means, (len(xs), 1))])
    pred = model.get_prediction(X_curve)
    yhat = pred.predicted_mean
    ci_df = pred.conf_int()

    return {
        "model": f"Polynomial Regression (degree {req.degree}){' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "predictor": req.predictor,
        "degree": req.degree,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "residual_se": float(np.sqrt(model.mse_resid)),
        "coefficients": coefs,
        "curve": {
            "x": xs.tolist(),
            "y": yhat.tolist(),
            "ci_low":  ci_df[:, 0].tolist(),
            "ci_high": ci_df[:, 1].tolist(),
        },
        "scatter": {
            "x": x.tolist()[:2000],
            "y": y.tolist()[:2000],
        },
    }


# ── Linear Mixed Model (LMM) / GLMM auto-router ──────────────────────────────

class LMMRequest(BaseModel):
    session_id: str
    outcome: str
    fixed_effects: List[str]
    group_col: str
    imputation: Optional[str] = "listwise"


def _is_id_like(col: str, series: "pd.Series") -> bool:
    """Heuristic: column is likely a patient/subject identifier."""
    name_lower = col.lower()
    # Name-based check
    name_match = any(name_lower == tok or name_lower.endswith(tok) or name_lower.startswith(tok)
                     for tok in ("id", "no", "num", "number", "patient", "subject", "case", "record"))
    if name_match:
        return True
    # Value-based check: near-unique integers
    n = len(series.dropna())
    if n < 5:
        return False
    try:
        nunique = series.nunique()
        return (nunique / n) > 0.95 and pd.api.types.is_integer_dtype(series)
    except Exception:
        return False


@router.post("/lmm")
def linear_mixed_model(req: LMMRequest):
    import re
    import statsmodels.formula.api as smf

    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    # ── Guard: ID column as fixed effect ────────────────────────────────────
    id_in_fe = [c for c in req.fixed_effects if _is_id_like(c, df_full.get(c, pd.Series()))]
    if id_in_fe:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Column(s) {id_in_fe} look like patient/subject identifiers and cannot be fixed effects. "
                "Assign them as the Grouping variable (random intercept) instead."
            ),
        )

    cols = [req.outcome, req.group_col] + req.fixed_effects
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    # ── Detect binary outcome → route to GEE ────────────────────────────────
    outcome_vals = df[req.outcome].dropna().unique()
    is_binary = set(outcome_vals.tolist()) <= {0, 1, 0.0, 1.0}

    # Sanitize column names for formula
    def safe(c: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_]", "_", c)

    rename = {c: safe(c) for c in cols}
    df_r = df.rename(columns=rename)
    outcome_s = safe(req.outcome)
    group_s   = safe(req.group_col)
    fe_s      = [safe(f) for f in req.fixed_effects]
    formula   = f"{outcome_s} ~ " + (" + ".join(fe_s) if fe_s else "1")

    if is_binary:
        # ── GEE with Binomial/Logit — population-averaged GLMM alternative ──
        # statsmodels MixedLM does not support binomial; GEE is the standard
        # alternative for clustered binary outcomes (population-averaged effects).
        import statsmodels.api as sm_api
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Independence

        gee_model = GEE.from_formula(
            formula, group_s, data=df_r,
            family=Binomial(),
            cov_struct=Independence(),
        )
        result = gee_model.fit()
        ci = result.conf_int()
        coefs = []
        for var in result.params.index:
            p_val = float(result.pvalues[var])
            est   = float(result.params[var])
            coefs.append({
                "variable": str(var),
                "estimate": round(est, 6),
                "exp_estimate": round(float(np.exp(est)), 4),   # Odds Ratio
                "se": round(float(result.bse[var]), 6),
                "z": round(float(result.tvalues[var]), 4),
                "p": round(p_val, 6),
                "ci_low":  round(float(ci.loc[var, 0]), 4),
                "ci_high": round(float(ci.loc[var, 1]), 4),
                "or_low":  round(float(np.exp(ci.loc[var, 0])), 4),
                "or_high": round(float(np.exp(ci.loc[var, 1])), 4),
            })
        return {
            "model": "GEE — Binomial/Logit (Binary outcome)",
            "model_type": "gee_binomial",
            "note": (
                "Binary outcome detected. Fitted using Generalized Estimating Equations (GEE) "
                "with Binomial family and logit link — the population-averaged equivalent of a GLMM. "
                "Estimates are log-odds (logit scale); exp(β) = Odds Ratio."
            ),
            "outcome": req.outcome,
            "group": req.group_col,
            "n": int(result.nobs),
            "n_groups": int(df[req.group_col].nunique()),
            "n_excluded": n_excluded,
            "aic": float(result.aic) if hasattr(result, "aic") else None,
            "bic": float(result.bic) if hasattr(result, "bic") else None,
            "log_likelihood": float(result.llf) if hasattr(result, "llf") else None,
            "random_effect_variance": None,
            "residual_variance": None,
            "icc": None,
            "coefficients": coefs,
        }

    # ── Standard LMM (REML) for continuous outcomes ──────────────────────────
    model = smf.mixedlm(formula, df_r, groups=df_r[group_s]).fit(reml=True)

    fe_ci = model.conf_int()
    coefs = []
    for var in model.fe_params.index:
        coefs.append({
            "variable": str(var),
            "estimate": float(model.fe_params[var]),
            "se": float(model.bse_fe[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(fe_ci.loc[var, 0]),
            "ci_high": float(fe_ci.loc[var, 1]),
        })

    re_var = float(model.cov_re.iloc[0, 0]) if model.cov_re is not None and model.cov_re.size > 0 else None
    residual_var = float(model.scale)

    return {
        "model": "Linear Mixed Model (REML)",
        "model_type": "lmm",
        "outcome": req.outcome,
        "group": req.group_col,
        "n": int(model.nobs),
        "n_groups": int(df[req.group_col].nunique()),
        "n_excluded": n_excluded,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
        "random_effect_variance": re_var,
        "residual_variance": residual_var,
        "icc": (re_var / (re_var + residual_var)) if re_var is not None else None,
        "coefficients": coefs,
    }


# ── Wide → Long melt (repeated measures reshape) ──────────────────────────────

class MeltRequest(BaseModel):
    session_id: str
    id_col: str                  # e.g. "PatientID"
    value_cols: List[str]        # e.g. ["INHOSPITALEF", "EF", "CONTROLEF"]
    time_var_name: str = "TimePoint"
    value_var_name: str = "Value"
    time_labels: Optional[List[str]] = None  # custom labels; defaults to col names


@router.post("/melt")
def melt_wide_to_long(req: MeltRequest):
    """Reshape wide-format repeated measures into long format and save back to session."""
    df = _get_df(req.session_id)
    missing = [c for c in [req.id_col] + req.value_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing}")
    if len(req.value_cols) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 value columns to melt")

    labels = req.time_labels if req.time_labels and len(req.time_labels) == len(req.value_cols) \
             else req.value_cols

    # Keep other columns (non-melted) as covariates in the long frame
    other_cols = [c for c in df.columns if c not in req.value_cols and c != req.id_col]
    # Limit other cols to avoid explosion
    keep = [req.id_col] + req.value_cols + other_cols[:20]
    df_sub = df[[c for c in keep if c in df.columns]].copy()

    df_long = df_sub.melt(
        id_vars=[c for c in df_sub.columns if c not in req.value_cols],
        value_vars=req.value_cols,
        var_name=req.time_var_name,
        value_name=req.value_var_name,
    )
    # Replace column names with readable labels
    label_map = dict(zip(req.value_cols, labels))
    df_long[req.time_var_name] = df_long[req.time_var_name].map(label_map)

    # Persist the long-format DataFrame back to the session store
    from ..services.store import store
    store.save(req.session_id, df_long)

    return {
        "rows": len(df_long),
        "columns": list(df_long.columns),
        "time_var": req.time_var_name,
        "value_var": req.value_var_name,
        "time_points": labels,
        "preview": df_long.head(10).to_dict(orient="records"),
    }


# ── Gamma GLM ─────────────────────────────────────────────────────────────────

class GammaRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    link: str = "log"        # "log" | "identity" | "inverse"
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/gamma")
def gamma_regression(req: GammaRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if (y.dropna() <= 0).any():
        raise HTTPException(status_code=422, detail="Gamma regression requires strictly positive outcomes (> 0). Non-positive values found.")

    valid_links = {"log", "identity", "inverse"}
    if req.link and req.link not in valid_links:
        raise HTTPException(status_code=422, detail=f"Invalid link function '{req.link}'. Valid: {valid_links}")
    link_map = {"log": sm.families.links.Log(), "identity": sm.families.links.Identity(), "inverse": sm.families.links.InversePower()}
    family = sm.families.Gamma(link=link_map.get(req.link, sm.families.links.Log()))
    cov_type = "HC3" if req.robust_se else "nonrobust"
    model = sm.GLM(y, X, family=family).fit(cov_type=cov_type)
    ci = model.conf_int()

    coefs = []
    for var in model.params.index:
        b = float(model.params[var])
        coefs.append({
            "variable": str(var),
            "estimate": b,
            "exp_estimate": float(np.exp(b)) if req.link == "log" else None,
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
        })

    return {
        "model": f"Gamma GLM (link={req.link}){' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "link": req.link,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "deviance": float(model.deviance),
        "scale": float(model.scale),
        "coefficients": coefs,
    }


# ── Negative Binomial GLM ─────────────────────────────────────────────────────

class NegBinomRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/negbinom")
def negative_binomial_regression(req: NegBinomRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if (y.dropna() < 0).any():
        raise HTTPException(status_code=422, detail="Negative binomial requires non-negative integer counts.")
    if (y.dropna() % 1 != 0).any():
        raise HTTPException(status_code=422, detail="Negative binomial requires integer counts. Fractional values found.")
    cov_type = "HC3" if req.robust_se else "nonrobust"
    # Estimate alpha (dispersion) from Poisson residuals instead of fixed alpha=1
    try:
        poisson_fit = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        mu = poisson_fit.mu
        alpha_est = max(1e-6, float(((((y - mu) ** 2 - mu) / mu ** 2).mean())))
    except Exception:
        alpha_est = 1.0
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_est)).fit(cov_type=cov_type)
    ci = model.conf_int()

    coefs = []
    for var in model.params.index:
        b = float(model.params[var])
        coefs.append({
            "variable": str(var),
            "log_irr": b,
            "irr": float(np.exp(b)),
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
            "irr_ci_low":  float(np.exp(ci.loc[var, 0])),
            "irr_ci_high": float(np.exp(ci.loc[var, 1])),
        })

    return {
        "model": f"Negative Binomial Regression{' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "deviance": float(model.deviance),
        "coefficients": coefs,
    }


# ── Linear Regression Diagnostic Plots ────────────────────────────────────────

class DiagRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"


@router.post("/linear_diag")
def linear_diagnostics(req: DiagRequest):
    from scipy import stats as scipy_stats

    df_full = _get_df(req.session_id)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = df[req.outcome].astype(float)
    model = sm.OLS(y, X).fit()

    fitted   = model.fittedvalues.values
    resid    = model.resid.values
    std_res  = model.get_influence().resid_studentized_internal
    sqrt_abs = np.sqrt(np.abs(std_res))

    # QQ data
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(resid, dist="norm")
    qq_x_line = np.array([min(osm), max(osm)])
    qq_y_line  = slope * qq_x_line + intercept

    # Subsample for large datasets
    N = min(len(fitted), 2000)
    idx = np.random.choice(len(fitted), N, replace=False) if len(fitted) > N else np.arange(N)

    return {
        "residuals_fitted": {
            "x": fitted[idx].tolist(),
            "y": resid[idx].tolist(),
        },
        "qq": {
            "theoretical": osm[idx[:len(osm)]].tolist() if len(osm) > N else osm.tolist(),
            "sample":      osr[idx[:len(osr)]].tolist() if len(osr) > N else osr.tolist(),
            "line_x":      qq_x_line.tolist(),
            "line_y":      qq_y_line.tolist(),
        },
        "scale_location": {
            "x": fitted[idx].tolist(),
            "y": sqrt_abs[idx].tolist(),
        },
        "r_squared": float(model.rsquared),
        "residual_se": float(np.sqrt(model.mse_resid)),
        "n": int(model.nobs),
    }


# ── Propensity Score Matching (PSM) ───────────────────────────────────────────

class PSMRequest(BaseModel):
    session_id: str
    treatment_col: str
    covariates: List[str]
    outcome_col: Optional[str] = None
    caliper: Optional[float] = 0.2        # fraction of SD of PS
    ratio: Optional[int] = 1             # 1:ratio matching (1:1 default)
    imputation: Optional[str] = "listwise"


def _compute_smd(s_treated: pd.Series, s_control: pd.Series) -> float:
    """Standardized Mean Difference for one covariate."""
    # Convert categorical/object to numeric via label encoding
    if s_treated.dtype == object or str(s_treated.dtype).startswith("category"):
        combined = pd.concat([s_treated, s_control]).dropna()
        cats = sorted(combined.unique().tolist(), key=str)
        cat_map = {c: i for i, c in enumerate(cats)}
        s_treated = s_treated.map(cat_map)
        s_control = s_control.map(cat_map)

    s_treated = pd.to_numeric(s_treated, errors="coerce").dropna()
    s_control = pd.to_numeric(s_control, errors="coerce").dropna()

    if len(s_treated) == 0 or len(s_control) == 0:
        return 0.0

    n_uniq = pd.concat([s_treated, s_control]).nunique()
    if n_uniq <= 2:
        p1 = float(s_treated.mean())
        p0 = float(s_control.mean())
        denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
        return float(abs(p1 - p0) / denom) if denom > 1e-9 else 0.0
    # Continuous variable
    m1, m0 = float(s_treated.mean()), float(s_control.mean())
    sd1, sd0 = float(s_treated.std(ddof=1)), float(s_control.std(ddof=1))
    denom = np.sqrt((sd1 ** 2 + sd0 ** 2) / 2)
    return float(abs(m1 - m0) / denom) if denom > 1e-9 else 0.0


@router.post("/psm")
def propensity_score_matching(req: PSMRequest):
    import traceback
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    try:
        return _run_psm(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def _run_psm(req: PSMRequest):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    df_full = _get_df(req.session_id)
    needed = [req.treatment_col] + req.covariates + ([req.outcome_col] if req.outcome_col else [])
    missing_cols = [c for c in needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing_cols}")

    df = apply_imputation(df_full[needed], needed, req.imputation or "listwise").reset_index(drop=True)

    # Validate treatment is binary 0/1
    treat_vals = df[req.treatment_col].astype(float)
    if not set(treat_vals.unique().tolist()) <= {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422,
            detail=f"Treatment column '{req.treatment_col}' must be binary (0 = control, 1 = treated).")

    # ── Step 1: Propensity scores via Logistic Regression ────────────────────
    X = pd.get_dummies(df[req.covariates], drop_first=True).astype(float)
    y = treat_vals.astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    lr.fit(X_scaled, y)
    ps = lr.predict_proba(X_scaled)[:, 1]     # propensity score for each row

    df = df.copy()
    df["_ps_"] = ps
    df["_treat_"] = y

    # ── Step 2: Nearest-Neighbour matching with caliper (without replacement) ─
    caliper_dist = (req.caliper or 0.2) * ps.std()
    ratio = max(1, req.ratio or 1)

    treated_idx = np.where(y == 1)[0]
    control_idx = np.where(y == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise HTTPException(status_code=422, detail="Need both treated (1) and control (0) patients.")

    # KD-tree on control propensity scores
    ps_control = ps[control_idx].reshape(-1, 1)
    knn = NearestNeighbors(n_neighbors=min(ratio * 5, len(control_idx)), metric="euclidean")
    knn.fit(ps_control)

    matched_treated = []
    matched_controls = []
    used_controls = set()

    for ti in treated_idx:
        ps_t = np.array([[ps[ti]]])
        distances, neighbors = knn.kneighbors(ps_t)
        chosen = []
        for dist, nb in zip(distances[0], neighbors[0]):
            if dist <= caliper_dist and control_idx[nb] not in used_controls:
                chosen.append(control_idx[nb])
                used_controls.add(control_idx[nb])
                if len(chosen) == ratio:
                    break
        if len(chosen) == ratio:   # only keep fully matched treated units
            matched_treated.append(ti)
            matched_controls.extend(chosen)

    n_matched_treated = len(matched_treated)
    n_matched_controls = len(matched_controls)

    if n_matched_treated == 0:
        raise HTTPException(status_code=422,
            detail=f"No matches found within caliper {req.caliper}. "
                   "Try widening the caliper or check that treatment groups overlap in covariate space.")

    matched_all_idx = matched_treated + matched_controls
    df_matched = df.iloc[matched_all_idx].copy()

    # Assign match-set IDs for downstream paired/clustered analysis
    match_ids = []
    for i, ti in enumerate(matched_treated):
        match_ids.append(i)  # treated
    for i in range(len(matched_controls)):
        match_ids.append(i // ratio)  # controls get same match_id as their treated pair
    df_matched["_match_id_"] = match_ids

    # ── Step 3: SMD before and after matching ─────────────────────────────────
    smd_before, smd_after = {}, {}
    treat_mask = df["_treat_"].values   # numpy array, same length as df (reset index)
    for cov in req.covariates:
        col   = df[cov]
        col_m = df_matched[cov]
        smd_before[cov] = round(_compute_smd(
            col[treat_mask == 1], col[treat_mask == 0]), 4)
        smd_after[cov]  = round(_compute_smd(
            col_m[df_matched["_treat_"] == 1],
            col_m[df_matched["_treat_"] == 0]), 4)

    avg_smd_before = float(np.mean(list(smd_before.values())))
    avg_smd_after  = float(np.mean(list(smd_after.values())))
    reduction_pct  = float((avg_smd_before - avg_smd_after) / avg_smd_before * 100) if avg_smd_before > 0 else 0.0

    n_all_treated = int((y == 1).sum())
    n_all_control = int((y == 0).sum())
    n_unmatched   = n_all_treated - n_matched_treated

    # Balance flag: all SMDs < 0.10 after matching
    balance_achieved = all(v < 0.10 for v in smd_after.values())

    # ── Step 4: PS distribution for overlap plot ──────────────────────────────
    ps_dist = {
        "treated_unmatched": ps[treated_idx].tolist(),
        "control_unmatched": ps[control_idx].tolist(),
        "treated_matched":   ps[matched_treated].tolist(),
        "control_matched":   ps[matched_controls].tolist(),
    }

    # ── Outcome analysis on matched dataset ──────────────────────────────────
    outcome_result = None
    if req.outcome_col and req.outcome_col in df_matched.columns:
        try:
            y_out = pd.to_numeric(df_matched[req.outcome_col], errors="coerce")
            out_vals = set(y_out.dropna().unique().tolist())

            if not out_vals <= {0, 1, 0.0, 1.0}:
                outcome_result = {"error": f"Outcome must be binary 0/1 for matched analysis. Found: {sorted(out_vals)[:10]}"}
            else:
                # Use GEE with matched-pair clustering for valid SEs
                from statsmodels.genmod.generalized_estimating_equations import GEE
                from statsmodels.genmod.families import Binomial
                from statsmodels.genmod.cov_struct import Exchangeable

                df_out = df_matched[[req.treatment_col, req.outcome_col, "_match_id_"]].copy()
                df_out[req.outcome_col] = y_out.astype(int)
                df_out[req.treatment_col] = df_out[req.treatment_col].astype(float)

                formula = f"Q('{req.outcome_col}') ~ Q('{req.treatment_col}')"
                try:
                    gee_model = GEE.from_formula(
                        formula, groups="_match_id_", data=df_out,
                        family=Binomial(), cov_struct=Exchangeable()
                    )
                    m_out = gee_model.fit()
                    ci_out = m_out.conf_int()
                    coefs_out = []
                    for var in m_out.params.index:
                        est = float(m_out.params[var])
                        coefs_out.append({
                            "variable": str(var),
                            "estimate": round(est, 6),
                            "or": round(float(np.exp(est)), 4),
                            "se": round(float(m_out.bse[var]), 6),
                            "z": round(float(m_out.tvalues[var]), 4),
                            "p": round(float(m_out.pvalues[var]), 6),
                            "ci_low":  round(float(ci_out.loc[var, 0]), 4),
                            "ci_high": round(float(ci_out.loc[var, 1]), 4),
                            "or_low":  round(float(np.exp(ci_out.loc[var, 0])), 4),
                            "or_high": round(float(np.exp(ci_out.loc[var, 1])), 4),
                        })
                    outcome_result = {
                        "type": "gee_matched",
                        "model": "GEE Logistic (matched-pair clusters)",
                        "n": int(len(df_matched)),
                        "n_clusters": int(len(matched_treated)),
                        "coefficients": coefs_out,
                    }
                except Exception:
                    # Fallback to standard logistic with robust SE
                    X_out = sm.add_constant(df_out[[req.treatment_col]].astype(float))
                    m_out = sm.Logit(y_out.astype(int).values, X_out).fit(disp=False, cov_type="HC1")
                    ci_out = m_out.conf_int()
                    coefs_out = []
                    for var in m_out.params.index:
                        est = float(m_out.params[var])
                        coefs_out.append({
                            "variable": str(var),
                            "estimate": round(est, 6),
                            "or": round(float(np.exp(est)), 4),
                            "se": round(float(m_out.bse[var]), 6),
                            "z": round(float(m_out.tvalues[var]), 4),
                            "p": round(float(m_out.pvalues[var]), 6),
                            "ci_low":  round(float(ci_out.loc[var, 0]), 4),
                            "ci_high": round(float(ci_out.loc[var, 1]), 4),
                            "or_low":  round(float(np.exp(ci_out.loc[var, 0])), 4),
                            "or_high": round(float(np.exp(ci_out.loc[var, 1])), 4),
                        })
                    outcome_result = {
                        "type": "logistic_robust",
                        "model": "Logistic Regression [Robust SE] (matched cohort)",
                        "n": int(len(df_matched)),
                        "coefficients": coefs_out,
                        "aic": round(float(m_out.aic), 2),
                        "bic": round(float(m_out.bic), 2),
                    }
        except Exception as ex:
            outcome_result = {"error": str(ex)}

    # Persist matched dataset for downstream analysis (keep match_id for paired tests)
    df_export = df_matched.drop(columns=["_ps_", "_treat_"], errors="ignore")
    df_export = df_export.rename(columns={"_match_id_": "match_set_id"})
    store.save(req.session_id + "_psm", df_export)

    return {
        "n_total":            int(len(df)),
        "n_treated":          n_all_treated,
        "n_control":          n_all_control,
        "n_matched_pairs":    n_matched_treated,
        "n_matched_controls": n_matched_controls,
        "n_unmatched":        n_unmatched,
        "caliper_used":       round(float(caliper_dist), 6),
        "balance_achieved":   balance_achieved,
        "avg_smd_before":     round(avg_smd_before, 4),
        "avg_smd_after":      round(avg_smd_after, 4),
        "reduction_pct":      round(reduction_pct, 1),
        "smd_before":         smd_before,
        "smd_after":          smd_after,
        "ps_distribution":    ps_dist,
        "outcome_result":     outcome_result,
        "matched_session_id": req.session_id + "_psm",
    }
