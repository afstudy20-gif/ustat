from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from loguru import logger

from services import store
from services.impute import apply_imputation
from services.assumptions import (
    check_gee_assumptions_placeholder,
    check_ordinal_assumptions_placeholder,
    add_assumption_warnings_to_result,
)

router = APIRouter()

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _sanitize_model_error(err: Exception, context: str = "model fitting") -> str:
    msg = str(err)
    if "Singular" in msg or "perfect separation" in msg.lower():
        return "The model encountered perfect separation or singular matrix. Try removing highly correlated predictors."
    if "convergence" in msg.lower() or "failed to converge" in msg.lower():
        return f"{context.capitalize()} failed to converge. Consider increasing iterations or simplifying the model."
    return f"{context.capitalize()} failed. Please check your data and predictors."


def _compute_vif(X: pd.DataFrame) -> dict:
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
            logger.exception("VIF calculation failed in GLM router")
            v = None
        out[str(col)] = v
    return out


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
    vifs = _compute_vif(X)
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
            "vif": vifs.get(str(var)),
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
        "result_text": _poisson_results_text(req.outcome, coefs),
    }


def _poisson_results_text(outcome, coefs):
    sig = [c for c in coefs if c["variable"] != "const" and c["p"] < 0.05]
    parts = [f"Poisson regression was performed to model {outcome}."]
    if sig:
        preds = []
        for c in sig:
            p_s = "<0.001" if c["p"] < 0.001 else f'{c["p"]:.3f}'
            preds.append(f'{c["variable"]} (IRR = {c["irr"]:.2f}, 95% CI: {c["irr_ci_low"]:.2f}–{c["irr_ci_high"]:.2f}, p = {p_s})')
        parts.append("Significant predictors: " + "; ".join(preds) + ".")
    else:
        parts.append("No predictor reached statistical significance.")
    return " ".join(parts)


# ── Gamma GLM ─────────────────────────────────────────────────────────────────

class GammaRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    link: str = "log"
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

    vifs = _compute_vif(X)
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
            "vif": vifs.get(str(var)),
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
    try:
        poisson_fit = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        mu = poisson_fit.mu
        alpha_est = max(1e-6, float(((((y - mu) ** 2 - mu) / mu ** 2).mean())))
    except Exception:
        alpha_est = 1.0
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_est)).fit(cov_type=cov_type)
    ci = model.conf_int()
    vifs = _compute_vif(X)

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
            "vif": vifs.get(str(var)),
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


# ── Standalone GEE (Generalized Estimating Equations) ──────────────────────────

class GEERequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    group_col: str
    family: str = "gaussian"       # gaussian | binomial | poisson
    cov_struct: str = "independence"  # independence | exchangeable | ar
    imputation: Optional[str] = "listwise"


@router.post("/gee")
def gee_regression(req: GEERequest):
    from statsmodels.genmod.cov_struct import Independence, Exchangeable, Autoregressive

    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    cols = [req.outcome] + req.predictors + [req.group_col]
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    if req.group_col not in df.columns:
        raise HTTPException(status_code=422, detail=f"group_col '{req.group_col}' not found")

    y = pd.to_numeric(df[req.outcome], errors="coerce")
    X = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    Xc = sm.add_constant(X, has_constant="add")
    groups = df[req.group_col]

    fam_map = {
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
    }
    if req.family not in fam_map:
        raise HTTPException(status_code=422, detail=f"Unsupported family: {req.family}")

    cov_map = {
        "independence": Independence(),
        "exchangeable": Exchangeable(),
        "ar": Autoregressive(),
    }
    if req.cov_struct not in cov_map:
        raise HTTPException(status_code=422, detail=f"Unsupported cov_struct: {req.cov_struct}")

    try:
        model = sm.GEE(y, Xc, groups=groups, family=fam_map[req.family], cov_struct=cov_map[req.cov_struct])
        result = model.fit()
    except Exception as e:
        logger.exception("GEE fit failed")
        raise HTTPException(status_code=422, detail=_sanitize_model_error(e, "GEE"))

    coefs = []
    for name in result.params.index:
        if name == "const":
            continue
        coefs.append({
            "variable": name,
            "estimate": round(float(result.params[name]), 6),
            "se": round(float(result.bse[name]), 6),
            "p": round(float(result.pvalues[name]), 6) if name in result.pvalues.index else None,
        })

    n_clusters = int(df[req.group_col].nunique())

    res = {
        "n_obs": int(result.nobs),
        "n_clusters": n_clusters,
        "n_excluded": int(n_excluded),
        "family": req.family,
        "cov_struct": req.cov_struct,
        "coefficients": coefs,
        "result_text": _gee_results_text(req.family, req.cov_struct, n_clusters, result.nobs),
    }

    gee_report = check_gee_assumptions_placeholder(req.family, req.cov_struct)
    res = add_assumption_warnings_to_result(res, gee_report)
    return res


def _gee_results_text(family, cov_struct, n_clusters, n_obs):
    return (
        f"GEE model with {family} family and {cov_struct} correlation structure "
        f"was fit on {n_clusters} clusters ({n_obs} observations)."
    )


# ── Ordinal Logistic Regression ───────────────────────────────────────────────

class OrdinalRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"


@router.post("/ordinal")
def ordinal_regression(req: OrdinalRequest):
    from sklearn.preprocessing import LabelEncoder

    df_full = _get_df(req.session_id)
    cols = [req.outcome] + req.predictors
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    n_excluded = len(df_full) - len(df)

    y_raw = df[req.outcome]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    cats = list(le.classes_)

    if len(cats) < 3:
        raise HTTPException(status_code=422, detail="Ordinal outcome must have at least 3 ordered categories.")

    X = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    Xc = sm.add_constant(X, has_constant="add")

    try:
        model = sm.MNLogit(y, Xc)
        result = model.fit(disp=False, maxiter=200)
    except Exception as e:
        logger.exception("Ordinal regression fit failed")
        raise HTTPException(status_code=422, detail=_sanitize_model_error(e, "ordinal logistic"))

    coefs = []
    for i, cat in enumerate(cats[1:]):
        for j, var in enumerate(result.params.index):
            if var == "const":
                continue
            est = float(result.params.iloc[j, i]) if result.params.ndim > 1 else float(result.params.iloc[j])
            se = float(result.bse.iloc[j, i]) if result.bse.ndim > 1 else float(result.bse.iloc[j])
            p = float(result.pvalues.iloc[j, i]) if result.pvalues.ndim > 1 else float(result.pvalues.iloc[j])
            coefs.append({
                "variable": var,
                "category": str(cat),
                "estimate": round(est, 6),
                "se": round(se, 6),
                "p": round(p, 6),
            })

    brant = {
        "computed": False,
        "note": "Formal Brant test for proportional odds assumption is not yet implemented in this endpoint."
    }

    res = {
        "categories_in_rank_order": [str(c) for c in cats],
        "n_obs": int(len(df)),
        "n_excluded": int(n_excluded),
        "coefficients": coefs,
        "brant_proportional_odds": brant,
        "result_text": _ordinal_results_text(len(cats), len(df)),
    }

    ordinal_report = check_ordinal_assumptions_placeholder()
    res = add_assumption_warnings_to_result(res, ordinal_report)
    return res


def _ordinal_results_text(n_categories, n_obs):
    return (
        f"Ordinal logistic regression was performed on {n_categories} ordered categories "
        f"({n_obs} observations). Note: formal proportional odds testing is limited."
    )
