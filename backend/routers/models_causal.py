"""Causal-inference sub-router for the /api/models namespace.

Thin HTTP adapter: validates the request, loads the session DataFrame, delegates
all statistics to services.causal, and handles session/store persistence. The
actual PSM matching and IPTW weighting live in services/causal.py.

  * /psm   — Propensity Score Matching (greedy/optimal, caliper, exact match,
             SMD/variance ratio/KS balance, optional Crump trim, Rosenbaum
             bounds, conditional logistic / stratified Cox on the matched set).
  * /iptw  — Inverse Probability of Treatment Weighting (ATE/ATT/overlap,
             stabilised, percentile/hard truncation, weighted GLM/Cox with
             robust Lin-Wei sandwich SE or bootstrap percentile CI, Kish ESS).
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store, causal
from routers._models_shared import get_df as _get_df, cpu_bound

router = APIRouter()


# ── Propensity Score Matching ────────────────────────────────────────────────

class PSMRequest(BaseModel):
    session_id: str
    treatment_col: str
    covariates: List[str]
    outcome_col: Optional[str] = None
    caliper: Optional[float] = 0.2        # fraction of SD (of logit-PS if caliper_scale='logit', else PS)
    caliper_scale: Optional[str] = "logit"  # 'logit' (Austin 2011) or 'raw'
    ratio: Optional[int] = 1             # 1:ratio matching (1:1 default)
    imputation: Optional[str] = "listwise"
    trim_common_support: Optional[bool] = False  # Crump 2009 trimming
    random_state: Optional[int] = 42     # reproducibility for LR solver tie-breaking
    # Score-model alternatives
    score_method: Optional[str] = "logistic"   # 'logistic' | 'probit' | 'gbm'
    # Matching method
    matching_method: Optional[str] = "greedy"  # 'greedy' (NN+caliper) | 'optimal' (Hungarian, 1:1 only)
    # Exact-match strata (categorical columns that must agree before NN)
    exact_match: Optional[List[str]] = None
    # Outcome handling
    outcome_type: Optional[str] = "binary"     # 'binary' (default) | 'survival'
    survival_duration_col: Optional[str] = None
    survival_event_col:    Optional[str] = None
    # Sensitivity analysis
    compute_rosenbaum: Optional[bool] = False  # Rosenbaum bounds (1:1 binary only)
    rosenbaum_gamma_max: Optional[float] = 3.0


@router.post("/psm")
@cpu_bound
def propensity_score_matching(req: PSMRequest):
    import traceback
    try:
        df_full = _get_df(req.session_id)
        result, df_export = causal.run_psm(df_full, req)
        store.save(req.session_id + "_psm", df_export)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


# ── Inverse Probability of Treatment Weighting (IPTW) ────────────────────────
#
# Companion to PSM. PSM matches and discards; IPTW reweights every unit by the
# inverse propensity score, keeps the whole sample, and supports ATE / ATT /
# overlap estimands directly. Outcome models become weighted GLM (binary) or
# weighted Cox (survival), with robust (Lin & Wei sandwich) standard errors
# or, optionally, a bootstrap percentile CI.


class IPTWRequest(BaseModel):
    session_id: str
    treatment_col: str
    covariates: List[str]
    outcome_col: Optional[str] = None
    imputation: Optional[str] = "listwise"
    random_state: Optional[int] = 42
    score_method: Optional[str] = "logistic"        # 'logistic' | 'probit' | 'gbm'
    estimand: Optional[str] = "ate"                  # 'ate' | 'att' | 'overlap'
    stabilize: Optional[bool] = True
    trim_common_support: Optional[bool] = False
    weight_truncation: Optional[str] = "percentile"  # 'percentile' | 'hard' | 'none'
    weight_truncation_lo: Optional[float] = 0.01
    weight_truncation_hi: Optional[float] = 0.99
    weight_truncation_max: Optional[float] = 10.0
    outcome_type: Optional[str] = "binary"           # 'binary' | 'survival'
    survival_duration_col: Optional[str] = None
    survival_event_col: Optional[str] = None
    se_method: Optional[str] = "robust"              # 'robust' | 'bootstrap'
    bootstrap_reps: Optional[int] = 500


@router.post("/iptw")
@cpu_bound
def iptw(req: IPTWRequest):
    import traceback
    try:
        df_full = _get_df(req.session_id)
        result, audit = causal.run_iptw(df_full, req)
        try:
            store.log_action(req.session_id, "iptw", audit)
        except Exception:
            pass
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
