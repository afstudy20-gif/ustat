"""Linear / GLM model-fitting service.

Extracted from routers/models.py so the router is a thin HTTP adapter. Each
fit_* function takes an already-loaded DataFrame plus a duck-typed request and
returns a plain result dict. No session/store I/O lives here (the /melt
data-reshape endpoint, which mutates the session, stays in the router).

Public API:
    fit_linear, fit_delta_sensitivity, fit_polynomial, fit_lmm, fit_gamma,
    fit_negbinom, fit_gee, fit_ordinal, fit_stepwise, fit_linear_diag
plus helpers (design encoding, results text, effect refit, stepwise refit).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from fastapi import HTTPException

from services.impute import apply_imputation
from routers._models_shared import (
    compute_vif as _compute_vif,
    add_pairwise_interactions as _add_pairwise_interactions,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _encode_design(
    df: pd.DataFrame, predictors: List[str],
    reference_levels: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Dummy-code predictors, honouring a caller-chosen reference level per
    categorical column. Numeric columns pass through unchanged (matching
    pandas.get_dummies); categorical columns are reordered so the requested
    reference sorts first and is therefore dropped by drop_first=True.
    """
    refs = reference_levels or {}
    work = df[predictors].copy()
    for col in predictors:
        if col not in refs:
            continue
        if pd.api.types.is_numeric_dtype(work[col]):
            # Reference levels are meaningless for a continuous/numeric column.
            continue
        ref = str(refs[col])
        levels = [str(v) for v in pd.unique(work[col].dropna())]
        if ref not in levels:
            raise HTTPException(
                status_code=422,
                detail=f"reference level '{ref}' not found in column '{col}'. Levels: {sorted(levels)}",
            )
        others = sorted(l for l in levels if l != ref)
        work[col] = pd.Categorical(work[col].astype(str), categories=[ref] + others)
    return pd.get_dummies(work, drop_first=True).astype(float)




def _linear_results_text(outcome, coefs, model):
    sig = [c for c in coefs if c["variable"] != "const" and c["p"] < 0.05]
    f_p = "<0.001" if model.f_pvalue < 0.001 else f"{model.f_pvalue:.3f}"
    parts = [
        f"Multiple linear regression was performed to predict {outcome}. "
        f"The overall model was {'statistically significant' if model.f_pvalue < 0.05 else 'not significant'} "
        f"(F({int(model.df_model)},{int(model.df_resid)}) = {model.fvalue:.3f}, p = {f_p}), "
        f"explaining {model.rsquared*100:.1f}% of the variance (R² = {model.rsquared:.3f}, adjusted R² = {model.rsquared_adj:.3f})."
    ]
    if sig:
        preds = []
        for c in sig:
            p_s = "<0.001" if c["p"] < 0.001 else f'{c["p"]:.3f}'
            preds.append(f'{c["variable"]} (B = {c["estimate"]:.3f}, SE = {c["se"]:.3f}, p = {p_s})')
        parts.append("Significant predictors: " + "; ".join(preds) + ".")
    return " ".join(parts)


# ── Delta-scaling MNAR sensitivity ───────────────────────────────────────────



def _fit_effects(df: pd.DataFrame, model: str, outcome: str, predictors: List[str]):
    """Fit linear (OLS) or logistic (GLM-Binomial) and return per-predictor
    effect sizes — raw coefficient B for linear, odds ratio exp(β) for logistic.
    """
    X_enc = _encode_design(df, predictors, None)
    X = sm.add_constant(X_enc, has_constant="add")
    y = pd.to_numeric(df[outcome], errors="coerce").astype(float)
    if model == "logistic":
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        eff = {v: float(np.exp(res.params[v])) for v in res.params.index if v != "const"}
        return eff, "OR"
    res = sm.OLS(y, X).fit()
    eff = {v: float(res.params[v]) for v in res.params.index if v != "const"}
    return eff, "B"




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




def _fit_for_stepwise(model_type: str, df: pd.DataFrame, preds: list, outcome: str | None,
                      duration: str | None, event: str | None):
    """Fit a model for stepwise candidate-evaluation. Returns
    (aic, bic, p_per_predictor, llf). Raises on failure."""
    if model_type == "linear":
        X = pd.get_dummies(df[preds], drop_first=True).astype(float) if preds else pd.DataFrame(index=df.index)
        X = sm.add_constant(X, has_constant="add")
        y = pd.to_numeric(df[outcome], errors="coerce").astype(float)
        m = sm.OLS(y, X).fit()
        return float(m.aic), float(m.bic), {str(k): float(v) for k, v in m.pvalues.items() if k != "const"}, float(m.llf)
    if model_type == "logistic":
        X = pd.get_dummies(df[preds], drop_first=True).astype(float) if preds else pd.DataFrame(index=df.index)
        X = sm.add_constant(X, has_constant="add")
        y = pd.to_numeric(df[outcome], errors="coerce").astype(float)
        m = sm.Logit(y, X).fit(disp=False, maxiter=100)
        return float(m.aic), float(m.bic), {str(k): float(v) for k, v in m.pvalues.items() if k != "const"}, float(m.llf)
    if model_type == "cox":
        from lifelines import CoxPHFitter
        enc = pd.get_dummies(df[preds], drop_first=True).astype(float) if preds else pd.DataFrame(index=df.index)
        # Need at least one predictor for Cox to fit
        if enc.shape[1] == 0:
            raise ValueError("Cox requires ≥1 predictor.")
        fit_df = pd.concat([df[[duration, event]], enc], axis=1).dropna()
        cph = CoxPHFitter()
        cph.fit(fit_df, duration_col=duration, event_col=event)
        aic = float(cph.AIC_partial_) if hasattr(cph, "AIC_partial_") else float(-2 * cph.log_likelihood_ + 2 * len(cph.params_))
        # BIC = -2 log L + k log n
        n_events = int(cph.event_observed.sum())
        bic = float(-2 * cph.log_likelihood_ + len(cph.params_) * np.log(max(n_events, 1)))
        p_map = {str(k): float(v) for k, v in cph.summary["p"].items()}
        return aic, bic, p_map, float(cph.log_likelihood_)
    raise ValueError(f"Unknown model_type '{model_type}'")




def fit_linear(df_full, req):
    n_total = len(df_full)

    # ── Missing-indicator method ─────────────────────────────────────────────
    # Capture NA flags from the RAW data before any imputation, then pre-fill
    # the flagged columns so listwise deletion / imputation keeps their rows.
    mi_cols = [c for c in (req.missing_indicator or []) if c in req.predictors]
    for c in mi_cols:
        if c not in df_full.columns:
            raise HTTPException(status_code=422, detail=f"missing_indicator column '{c}' not found")
    df_src = df_full.copy()
    mi_flags: Dict[str, pd.Series] = {}
    for c in mi_cols:
        mi_flags[c] = df_src[c].isna().astype(int)
        if pd.api.types.is_numeric_dtype(df_src[c]):
            df_src[c] = df_src[c].fillna(df_src[c].median())
        else:
            mode = df_src[c].mode(dropna=True)
            df_src[c] = df_src[c].fillna(mode.iloc[0] if len(mode) else "")

    df = apply_imputation(df_src, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X_enc = _encode_design(df, req.predictors, req.reference_levels)
    # Append the missing-indicator dummies, aligned to the surviving rows.
    for c in mi_cols:
        X_enc[f"{c}__missing"] = mi_flags[c].reindex(df.index).fillna(0).astype(float).values
    X_enc, ix_added = _add_pairwise_interactions(X_enc, req.interactions, req.predictors)
    X = sm.add_constant(X_enc)
    y = df[req.outcome].astype(float)
    base = sm.OLS(y, X)
    model = base.fit(cov_type="HC3", use_t=True) if req.robust_se else base.fit()

    vifs = _compute_vif(X)
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
            "vif": vifs.get(str(var)),  # None for intercept
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
        "reference_levels": req.reference_levels or {},
        "missing_indicator_cols": mi_cols,
        "result_text": _linear_results_text(req.outcome, coefs, model),
    }




def fit_delta_sensitivity(df_full, req):
    if req.model not in ("linear", "logistic"):
        raise HTTPException(status_code=422, detail="model must be 'linear' or 'logistic'.")
    if req.imputation in ("listwise", "none", "", None):
        raise HTTPException(
            status_code=422,
            detail="Delta-scaling needs an imputing method (mean/median/mice); listwise leaves no imputed values to scale.",
        )
    if not req.deltas:
        raise HTTPException(status_code=422, detail="Provide at least one delta.")
    for c in [req.outcome] + req.predictors:
        if c not in df_full.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")

    if req.delta_cols:
        delta_cols = [c for c in req.delta_cols if c in req.predictors]
    else:
        delta_cols = [
            c for c in req.predictors
            if pd.api.types.is_numeric_dtype(df_full[c]) and bool(df_full[c].isna().any())
        ]
    if not delta_cols:
        raise HTTPException(
            status_code=422,
            detail="No numeric predictor with missing values to delta-scale. Pass delta_cols explicitly.",
        )

    if req.model == "logistic":
        yv = set(pd.to_numeric(df_full[req.outcome], errors="coerce").dropna().unique().tolist())
        if not yv.issubset({0, 1}) or len(yv) < 2:
            raise HTTPException(status_code=422, detail="Logistic outcome must be binary 0/1.")

    masks_full = {c: df_full[c].isna() for c in delta_cols}
    base_df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation)
    if len(base_df) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows after imputation (got {len(base_df)}).")
    masks = {c: masks_full[c].reindex(base_df.index).fillna(False).to_numpy() for c in delta_cols}
    n_scaled = {c: int(m.sum()) for c, m in masks.items()}

    base_eff, eff_label = _fit_effects(base_df, req.model, req.outcome, req.predictors)

    scenarios = []
    for d in req.deltas:
        dd = base_df.copy()
        for c in delta_cols:
            m = masks[c]
            if not m.any():
                continue
            col_vals = pd.to_numeric(dd[c], errors="coerce").to_numpy(dtype=float)
            col_vals[m] = col_vals[m] * float(d)
            dd[c] = col_vals
        try:
            eff, _ = _fit_effects(dd, req.model, req.outcome, req.predictors)
        except Exception:
            eff = {}
        scenarios.append({"delta": float(d), "effects": {k: round(v, 6) for k, v in eff.items()}})

    variables = list(base_eff.keys())
    table = []
    for v in variables:
        row: Dict[str, Any] = {"variable": v, "base": round(base_eff[v], 6)}
        for s in scenarios:
            ev = s["effects"].get(v)
            row[f"delta_{s['delta']}"] = ev if ev is not None else None
        table.append(row)

    return {
        "method": "delta_sensitivity",
        "model": req.model,
        "effect_label": eff_label,            # 'B' (linear) | 'OR' (logistic)
        "imputation": req.imputation,
        "delta_cols": delta_cols,
        "n_scaled_per_col": n_scaled,
        "deltas": [float(d) for d in req.deltas],
        "base": {v: round(base_eff[v], 6) for v in variables},
        "scenarios": scenarios,
        "table": table,
        "note": (
            "MNAR sensitivity by delta-scaling: values imputed for originally-missing "
            "cells in the listed columns are multiplied by Δ before refitting. Δ < 1 "
            "simulates true values below the imputed estimate, Δ > 1 above it. Effect "
            "estimates that stay stable across Δ support robustness to a "
            "missing-not-at-random mechanism."
        ),
    }


# ── Polynomial / Non-linear Regression ───────────────────────────────────────



def fit_polynomial(df_full, req):
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



def fit_lmm(df_full, req):
    import re
    import statsmodels.formula.api as smf

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
        # statsmodels MixedLM under REML leaves AIC/BIC undefined (NaN); coerce
        # any non-finite information criterion to None so the response is
        # JSON-serialisable instead of crashing FastAPI's encoder.
        "aic": (float(model.aic) if np.isfinite(model.aic) else None),
        "bic": (float(model.bic) if np.isfinite(model.bic) else None),
        "log_likelihood": (float(model.llf) if np.isfinite(model.llf) else None),
        "random_effect_variance": re_var,
        "residual_variance": residual_var,
        "icc": (re_var / (re_var + residual_var)) if re_var is not None else None,
        "coefficients": coefs,
    }


# ── Wide → Long melt (repeated measures reshape) ──────────────────────────────



def fit_gamma(df_full, req):
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



def fit_negbinom(df_full, req):
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


# ── Linear Regression Diagnostic Plots ────────────────────────────────────────

# ── GEE — Generalized Estimating Equations (standalone) ─────────────────────



def fit_gee(df_full, req):
    """Population-averaged regression for clustered / repeated-measures data.

    Uses statsmodels Generalized Estimating Equations (Liang & Zeger 1986).
    Family choices map outcome types:
      gaussian  → continuous outcome (default)
      binomial  → binary 0/1 outcome (logit link)
      poisson   → count outcome (log link)
    Working correlation structures:
      independence    → no within-cluster correlation (conservative)
      exchangeable    → equal correlation between any two observations in cluster
      ar / autoregressive → AR(1) — first-order autoregressive (time-ordered)
    """
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Gaussian, Binomial, Poisson
    from statsmodels.genmod.cov_struct import Independence, Exchangeable, Autoregressive

    fam_map = {"gaussian": Gaussian(), "binomial": Binomial(), "poisson": Poisson()}
    cov_map = {
        "independence": Independence(),
        "exchangeable": Exchangeable(),
        "ar": Autoregressive(),
        "autoregressive": Autoregressive(),
    }
    if req.family not in fam_map:
        raise HTTPException(status_code=422, detail=f"Unknown family '{req.family}'. Valid: gaussian/binomial/poisson.")
    if req.cov_struct not in cov_map:
        raise HTTPException(status_code=422, detail=f"Unknown cov_struct '{req.cov_struct}'.")

    n_total = len(df_full)
    cols = [req.outcome, req.group_col] + req.predictors
    if req.group_col not in df_full.columns:
        raise HTTPException(status_code=422, detail=f"Group column '{req.group_col}' not found.")
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Need ≥10 complete observations.")

    # Encode predictors (dummy-code categoricals, drop_first)
    X = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if y.isna().all():
        raise HTTPException(status_code=422, detail="Outcome has no numeric values.")

    # Binomial sanity
    if req.family == "binomial":
        uniq = set(y.dropna().unique())
        if not uniq <= {0, 1, 0.0, 1.0}:
            raise HTTPException(status_code=422, detail="Binomial GEE requires outcome coded as 0/1.")

    groups = df[req.group_col]
    fam = fam_map[req.family]
    cov = cov_map[req.cov_struct]

    try:
        model = GEE(y, X, groups=groups, family=fam, cov_struct=cov).fit()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"GEE fitting failed: {exc}")

    ci = model.conf_int()
    vifs = _compute_vif(X)
    coefs = []
    for var in model.params.index:
        b = float(model.params[var])
        row = {
            "variable": str(var),
            "estimate": b,
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
            "vif": vifs.get(str(var)),
        }
        # Add exp(beta) interpretation for binomial / poisson
        if req.family in ("binomial", "poisson"):
            row["exp_estimate"] = float(np.exp(b))
            row["exp_ci_low"] = float(np.exp(ci.loc[var, 0]))
            row["exp_ci_high"] = float(np.exp(ci.loc[var, 1]))
        coefs.append(row)

    effect_label = {"binomial": "Odds Ratio", "poisson": "Rate Ratio"}.get(req.family)

    return {
        "model": f"GEE ({req.family}, cov={req.cov_struct})",
        "outcome": req.outcome,
        "group_col": req.group_col,
        "family": req.family,
        "cov_struct": req.cov_struct,
        "n_obs": int(model.nobs),
        "n_clusters": int(groups.nunique()),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "coefficients": coefs,
        "effect_label": effect_label,
        "scale": float(model.scale) if hasattr(model, "scale") else None,
        "result_text": (
            f"GEE ({req.family} family, {req.cov_struct} working correlation) on {int(groups.nunique())} clusters "
            f"({int(model.nobs)} observations). Outcome: {req.outcome}; predictors: {', '.join(req.predictors)}."
        ),
    }


# ── Ordinal Logistic Regression (proportional odds) ─────────────────────────



def fit_ordinal(df_full, req):
    """Cumulative-link proportional-odds ordinal regression.

    Uses statsmodels.miscmodels.ordinal_model.OrderedModel. Returns one β
    per predictor (proportional-odds assumption) + cut-point thresholds
    α_k between successive categories. exp(β) is the cumulative odds ratio.

    Brant-style proportional-odds assumption check: fits a separate
    logistic model for each binary split y ≤ k vs y > k and compares
    slopes via a Wald χ² against the pooled estimate. p < 0.05 ⇒ PO
    assumption violated.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    if req.distr not in ("logit", "probit", "cloglog"):
        raise HTTPException(status_code=422, detail="distr must be logit | probit | cloglog")

    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    if len(df) < 20:
        raise HTTPException(status_code=400, detail="Need ≥20 complete observations for ordinal regression.")

    # Rank-code the outcome: lowest rank = 0, highest = K-1.
    y_raw = df[req.outcome]
    cats = sorted(y_raw.dropna().unique(), key=lambda v: (isinstance(v, str), v))
    if len(cats) < 3:
        raise HTTPException(status_code=422, detail=f"Ordinal regression requires ≥3 ordered categories; outcome has {len(cats)}.")
    rank_map = {c: i for i, c in enumerate(cats)}
    y = y_raw.map(rank_map).astype(int).values

    X = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    if X.shape[1] == 0:
        raise HTTPException(status_code=422, detail="No usable predictor columns after encoding.")

    try:
        # OrderedModel handles its own intercept-equivalents (cutpoints) — do
        # NOT add a column of ones, or the cutpoints become unidentified.
        model = OrderedModel(y, X, distr=req.distr).fit(method="bfgs", disp=False, maxiter=200)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ordinal regression fitting failed: {exc}")

    ci = model.conf_int()
    vifs = _compute_vif(X)
    coefs = []
    thresholds = []
    for name in model.params.index:
        b = float(model.params[name])
        se_val = float(model.bse[name])
        z_val = float(model.tvalues[name])
        p_val = float(model.pvalues[name])
        row = {
            "variable": str(name),
            "estimate": b,
            "se": se_val,
            "z": z_val,
            "p": p_val,
            "ci_low": float(ci.loc[name, 0]),
            "ci_high": float(ci.loc[name, 1]),
        }
        # OrderedModel parameter names: predictor names AND threshold names
        # ("0/1", "1/2", ...). Threshold params are reparameterised (log-diff
        # between cut-points except the first); skip them in the OR/CI block.
        is_threshold = "/" in str(name)
        if is_threshold:
            thresholds.append({
                "name": str(name),
                "estimate": b,
                "se": se_val,
                "p": p_val,
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
            })
        else:
            row["odds_ratio"] = float(np.exp(b))
            row["or_ci_low"] = float(np.exp(ci.loc[name, 0]))
            row["or_ci_high"] = float(np.exp(ci.loc[name, 1]))
            row["vif"] = vifs.get(str(name))
            coefs.append(row)

    # ── Brant-style proportional-odds assumption test ────────────────────────
    # Run K-1 separate binary logits (y > k vs y ≤ k) and compare each slope
    # to the pooled ordinal estimate via Wald χ² = (β_bin - β_ord)² / SE².
    # p < 0.05 ⇒ slopes differ across cut-points ⇒ PO assumption violated.
    brant = []
    K = len(cats)
    try:
        from scipy.stats import chi2 as _chi2
        for k_idx in range(K - 1):
            y_bin = (y > k_idx).astype(int)
            if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
                continue
            X_const = sm.add_constant(X, has_constant="add")
            try:
                bin_fit = sm.Logit(y_bin, X_const).fit(disp=False, maxiter=100)
            except Exception:
                continue
            row = {"cutpoint": f"{cats[k_idx]} | {cats[k_idx+1]}", "slopes": []}
            for var in X.columns:
                b_bin = float(bin_fit.params[var])
                se_bin = float(bin_fit.bse[var])
                b_ord = float(model.params[var])
                # Wald χ² of the difference: (β_bin - β_ord)² / SE_bin²
                if se_bin > 0:
                    chi2_val = (b_bin - b_ord) ** 2 / (se_bin ** 2)
                    p_val = float(1 - _chi2.cdf(chi2_val, 1))
                else:
                    chi2_val, p_val = None, None
                row["slopes"].append({
                    "variable": str(var),
                    "beta_binary": b_bin,
                    "beta_ordinal": b_ord,
                    "chi2": chi2_val,
                    "p": p_val,
                })
            brant.append(row)
    except Exception as exc:
        brant = [{"error": str(exc)}]

    return {
        "model": f"Ordinal Logistic Regression ({req.distr})",
        "outcome": req.outcome,
        "categories_in_rank_order": [str(c) for c in cats],
        "n": int(len(df)),
        "n_excluded": int(n_excluded),
        "imputation": req.imputation or "listwise",
        "coefficients": coefs,
        "thresholds": thresholds,
        "log_likelihood": float(model.llf),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "brant_proportional_odds": brant,
        "result_text": (
            f"Ordinal logistic regression ({req.distr} link) on {len(cats)} ordered categories "
            f"of {req.outcome} (n = {len(df)}, {n_excluded} excluded). Coefficients are interpreted as "
            f"log cumulative odds; exp(β) = cumulative OR for being in a higher category per unit of predictor."
        ),
    }


# ── Formal stepwise variable selection ──────────────────────────────────────



def fit_stepwise(df_full, req):
    """Formal forward / backward / both-direction variable selection.

    Criteria:
      - "aic": minimise AIC (default).
      - "bic": minimise BIC.
      - "p":   add when min p < p_enter, drop when max p > p_exit (Sasieni 1992).

    `forced_in` predictors are always retained regardless of criterion.
    Returns the trace (each step's chosen action + criterion value) plus
    the final selected predictor set.
    """
    if req.model_type not in ("linear", "logistic", "cox"):
        raise HTTPException(status_code=422, detail="model_type must be linear | logistic | cox.")
    if req.direction not in ("forward", "backward", "both"):
        raise HTTPException(status_code=422, detail="direction must be forward | backward | both.")
    if req.criterion not in ("aic", "bic", "p"):
        raise HTTPException(status_code=422, detail="criterion must be aic | bic | p.")
    if req.model_type in ("linear", "logistic") and not req.outcome:
        raise HTTPException(status_code=422, detail="outcome required for linear / logistic.")
    if req.model_type == "cox" and (not req.duration_col or not req.event_col):
        raise HTTPException(status_code=422, detail="duration_col + event_col required for cox.")

    needed = req.candidates + req.forced_in
    if req.model_type in ("linear", "logistic"):
        needed = [req.outcome] + needed
    else:
        needed = [req.duration_col, req.event_col] + needed
    df = apply_imputation(df_full, needed, req.imputation or "listwise")

    forced = list(req.forced_in)
    pool = [c for c in req.candidates if c not in forced]

    def _crit_val(aic_v, bic_v):
        return aic_v if req.criterion == "aic" else bic_v

    trace = []
    selected = list(forced) if req.direction != "backward" else list(forced) + list(pool)
    iterations = 0
    max_iter = max(20, 2 * len(req.candidates))

    while iterations < max_iter:
        iterations += 1
        # 1) Try adding (forward / both)
        best_add = None
        if req.direction in ("forward", "both"):
            remaining = [c for c in pool if c not in selected]
            for c in remaining:
                try_preds = selected + [c]
                try:
                    a, b, p_map, _ll = _fit_for_stepwise(req.model_type, df, try_preds, req.outcome,
                                                         req.duration_col, req.event_col)
                except Exception:
                    continue
                if req.criterion == "p":
                    # Pick the smallest p among new predictor(s) — handle dummy coding by checking c's matched cols
                    matched = [k for k in p_map if k.startswith(c)]
                    if not matched:
                        continue
                    p_new = min(p_map[k] for k in matched)
                    val = p_new
                    if best_add is None or val < best_add["value"]:
                        best_add = {"col": c, "value": val, "aic": a, "bic": b}
                else:
                    val = _crit_val(a, b)
                    if best_add is None or val < best_add["value"]:
                        best_add = {"col": c, "value": val, "aic": a, "bic": b}

        # 2) Try removing (backward / both)
        best_remove = None
        if req.direction in ("backward", "both"):
            removable = [c for c in selected if c not in forced]
            for c in removable:
                try_preds = [p for p in selected if p != c]
                if req.model_type == "cox" and not try_preds:
                    continue  # Cox needs ≥1 predictor
                try:
                    a, b, p_map, _ll = _fit_for_stepwise(req.model_type, df, try_preds, req.outcome,
                                                         req.duration_col, req.event_col)
                except Exception:
                    continue
                if req.criterion == "p":
                    # Drop if largest p among current model > p_exit
                    matched_c = [k for k in p_map if k.startswith(c)]
                    if matched_c:
                        continue  # column still present (shouldn't happen)
                    # value used = max p in remaining model (lower is better → invert)
                    max_p = max(p_map.values()) if p_map else 0.0
                    val = -max_p  # we want to drop largest p so lower(=negated) = better
                    if best_remove is None or val < best_remove["value"]:
                        best_remove = {"col": c, "value": val, "aic": a, "bic": b, "max_p": max_p}
                else:
                    val = _crit_val(a, b)
                    if best_remove is None or val < best_remove["value"]:
                        best_remove = {"col": c, "value": val, "aic": a, "bic": b}

        # Current model criterion (for AIC/BIC comparison)
        try:
            cur_a, cur_b, cur_p, _ll = _fit_for_stepwise(req.model_type, df, selected, req.outcome,
                                                         req.duration_col, req.event_col)
            cur_val = _crit_val(cur_a, cur_b)
        except Exception:
            cur_a, cur_b, cur_p, cur_val = float("inf"), float("inf"), {}, float("inf")

        action = None
        if req.criterion == "p":
            # Add if best_add p < p_enter; remove if best_remove max_p > p_exit
            if best_add and best_add["value"] < req.p_enter:
                selected.append(best_add["col"])
                action = {"step": iterations, "action": "add", "variable": best_add["col"],
                          "criterion_value": best_add["value"], "aic": best_add["aic"], "bic": best_add["bic"]}
            elif best_remove and best_remove.get("max_p", 0) > req.p_exit:
                selected.remove(best_remove["col"])
                action = {"step": iterations, "action": "remove", "variable": best_remove["col"],
                          "criterion_value": best_remove.get("max_p"), "aic": best_remove["aic"], "bic": best_remove["bic"]}
            else:
                break
        else:
            # AIC/BIC: pick the best improvement (lowest val) vs current
            candidates_step = []
            if best_add: candidates_step.append(("add", best_add))
            if best_remove: candidates_step.append(("remove", best_remove))
            if not candidates_step:
                break
            kind, best = min(candidates_step, key=lambda kv: kv[1]["value"])
            if best["value"] >= cur_val - 1e-9:
                break  # no improvement
            if kind == "add":
                selected.append(best["col"])
            else:
                selected.remove(best["col"])
            action = {"step": iterations, "action": kind, "variable": best["col"],
                      "criterion_value": best["value"], "aic": best["aic"], "bic": best["bic"]}
        trace.append(action)

    # Final fit
    final_aic, final_bic, final_p, final_llf = _fit_for_stepwise(
        req.model_type, df, selected, req.outcome, req.duration_col, req.event_col
    )
    return {
        "model_type": req.model_type,
        "direction": req.direction,
        "criterion": req.criterion,
        "forced_in": forced,
        "selected": selected,
        "trace": trace,
        "final_aic": final_aic,
        "final_bic": final_bic,
        "final_log_likelihood": final_llf,
        "n": int(len(df)),
        "result_text": (
            f"{req.direction.title()} stepwise ({req.criterion.upper()}) selected "
            f"{len(selected)} of {len(req.candidates)} candidates ({iterations} step(s)). "
            f"Final AIC = {final_aic:.2f}, BIC = {final_bic:.2f}."
        ),
    }




def fit_linear_diag(df_full, req):
    from scipy import stats as scipy_stats

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



