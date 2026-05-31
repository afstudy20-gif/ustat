"""
Joint Longitudinal-Survival Models (Phase 8)

Provides practical joint modeling of longitudinal biomarkers and time-to-event outcomes.

Current pragmatic implementation (no new heavy dependencies):
- Two-stage joint model (very common in practice):
  1. Fit Linear Mixed Model (LMM) on longitudinal data (random intercept + slope).
  2. Extract subject-specific random effects (empirical Bayes).
  3. Include them as covariates in a Cox PH model for survival.

This captures the association between biomarker trajectory and hazard while remaining
fast and implementable with statsmodels + lifelines.

All outputs are immutable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter


def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def fit_two_stage_joint_model(
    long_df: pd.DataFrame,
    surv_df: pd.DataFrame,
    id_col: str = "id",
    time_col: str = "time",
    y_col: str = "Y",
    long_predictors: Optional[List[str]] = None,
    surv_predictors: Optional[List[str]] = None,
    duration_col: str = "duration",
    event_col: str = "event",
) -> Dict[str, Any]:
    """
    Fit a two-stage joint longitudinal-survival model.

    Returns:
    - LMM summary
    - Extracted random effects per subject
    - Cox model that includes the random effects as predictors (association)
    - Basic diagnostics
    """
    if long_predictors is None:
        long_predictors = []
    if surv_predictors is None:
        surv_predictors = []

    # --- Stage 1: Linear Mixed Model ---
    # Clean potential duplicate columns and avoid 'id' label conflicts with statsmodels
    long_clean = long_df.loc[:, ~long_df.columns.duplicated()].copy()
    if id_col in long_clean.columns:
        long_clean = long_clean.rename(columns={id_col: "__subject_id__"})

    # Formula: Y ~ time + predictors + (1 + time | __subject_id__)
    formula = f"{y_col} ~ {time_col}"
    if long_predictors:
        formula += " + " + " + ".join(long_predictors)

    # Use random intercept + random slope for time
    md = smf.mixedlm(formula, long_clean, groups=long_clean["__subject_id__"], re_formula="~1 + " + time_col)
    mdf = md.fit(method=["lbfgs"], reml=True)

    # Extract random effects (empirical Bayes)
    re = mdf.random_effects
    re_df = pd.DataFrame.from_dict(re, orient="index")
    re_df = re_df.reset_index().rename(columns={"index": "__subject_id__"})
    re_df = re_df.rename(columns={"__subject_id__": id_col})
    # Typical columns after fit: Intercept, time (or whatever the slope var is)
    re_cols = [c for c in re_df.columns if c != id_col]

    # Merge random effects into survival data (avoid duplicate column issues)
    surv_work = surv_df.merge(re_df[[id_col] + re_cols], on=id_col, how="inner", suffixes=("", "_re"))

    # --- Stage 2: Cox model with random effects as covariates ---
    cox_covariates = surv_predictors + re_cols

    # Prepare design matrix
    X = surv_work[cox_covariates].copy()
    for c in cox_covariates:
        if X[c].dtype == object:
            X[c] = pd.Categorical(X[c]).codes

    cox_df = pd.concat([
        surv_work[[duration_col, event_col]].reset_index(drop=True),
        X.reset_index(drop=True)
    ], axis=1)

    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col, robust=True)

    # Format coefficients
    coefs = []
    for var in cph.params_.index:
        beta = float(cph.params_[var])
        coefs.append({
            "variable": str(var),
            "coef": round(beta, 5),
            "hr": round(np.exp(beta), 4),
            "se": round(float(cph.standard_errors_[var]), 5),
            "p": _safe(cph.summary.loc[var, "p"] if "p" in cph.summary.columns else None),
        })

    return {
        "model": "two_stage_joint_lmm_cox",
        "lmm_summary": {
            "params": mdf.params.to_dict(),
            "bse": mdf.bse.to_dict(),
            "pvalues": mdf.pvalues.to_dict(),
        },
        "random_effects": re_df.to_dict(orient="records"),
        "cox_coefficients": coefs,
        "cox_concordance": round(float(cph.concordance_index_), 4),
        "n_subjects": int(len(surv_work)),
        "note": "Two-stage joint model. Random effects from LMM used as covariates in Cox. Association parameters reflect link between biomarker trajectory and hazard.",
    }
