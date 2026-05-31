"""Linear / GLM sub-router for the /api/models namespace.

Thin HTTP adapter: validates the request, loads the session DataFrame, and
delegates model fitting to services.linear_models. The /melt data-reshape
endpoint, which mutates the session, keeps its logic here.

Endpoints: /linear, /delta_sensitivity, /polynomial, /lmm, /melt, /gamma,
/negbinom, /gee, /ordinal, /stepwise, /linear_diag.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store, linear_models
from routers._models_shared import get_df as _get_df, cpu_bound

router = APIRouter()


# ── Linear Regression ────────────────────────────────────────────────────────

class LinearRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False
    # Optional pairwise interactions — same semantics as the Cox endpoint:
    # each entry is [colA, colB]. Numeric × numeric is the element-wise
    # product; categorical columns expand into one interaction per surviving
    # dummy (e.g. SEX × AGE produces SEX_M:AGE).
    interactions: Optional[List[List[str]]] = None
    # Per-categorical reference level: {column: level}. The named level is
    # held out as the dummy-coding reference (e.g. {"coffee": "0 cups"} so all
    # other levels are contrasted against 0 cups/day). Columns not listed fall
    # back to the first sorted level (pandas drop_first default). Only applies
    # to columns pandas treats as categorical (object/category dtype).
    reference_levels: Optional[Dict[str, str]] = None
    # Missing-indicator method: for each listed predictor, add a `<col>__missing`
    # dummy (1 = value was missing in the raw data, 0 = observed) and impute the
    # column itself (median for numeric, mode for categorical) so incomplete rows
    # are retained rather than dropped. Lets a model absorb informative
    # missingness instead of losing the cases.
    missing_indicator: Optional[List[str]] = None



@router.post("/linear")
@cpu_bound
def linear_regression(req: LinearRequest):
    return linear_models.fit_linear(_get_df(req.session_id), req)


class DeltaSensitivityRequest(BaseModel):
    session_id: str
    model: str = "linear"               # 'linear' | 'logistic'
    outcome: str
    predictors: List[str]
    # Numeric predictors whose IMPUTED (originally-missing) cells are scaled by
    # each delta. Defaults to every numeric predictor that actually has missing
    # values.
    delta_cols: Optional[List[str]] = None
    deltas: List[float] = [0.9, 1.1]
    imputation: str = "mice"            # must impute (mean/median/mice), not listwise



@router.post("/delta_sensitivity")
@cpu_bound
def delta_sensitivity(req: DeltaSensitivityRequest):
    return linear_models.fit_delta_sensitivity(_get_df(req.session_id), req)


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
    return linear_models.fit_polynomial(_get_df(req.session_id), req)


# ── Linear Mixed Model (LMM) / GLMM auto-router ──────────────────────────────

class LMMRequest(BaseModel):
    session_id: str
    outcome: str
    fixed_effects: List[str]
    group_col: str
    imputation: Optional[str] = "listwise"



@router.post("/lmm")
@cpu_bound
def linear_mixed_model(req: LMMRequest):
    return linear_models.fit_lmm(_get_df(req.session_id), req)


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
    return linear_models.fit_gamma(_get_df(req.session_id), req)


# ── Negative Binomial GLM ─────────────────────────────────────────────────────

class NegBinomRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False




@router.post("/negbinom")
def negative_binomial_regression(req: NegBinomRequest):
    return linear_models.fit_negbinom(_get_df(req.session_id), req)


# ── GEE — Generalized Estimating Equations (standalone) ─────────────────────

class GEERequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    group_col: str                      # subject / cluster id
    family: str = "gaussian"            # gaussian | binomial | poisson
    cov_struct: str = "exchangeable"    # independence | exchangeable | ar | autoregressive
    imputation: Optional[str] = "listwise"




@router.post("/gee")
@cpu_bound
def gee_endpoint(req: GEERequest):
    return linear_models.fit_gee(_get_df(req.session_id), req)



# ── Ordinal Logistic Regression (proportional odds) ─────────────────────────

class OrdinalRequest(BaseModel):
    session_id: str
    outcome: str            # ordered categorical (will be integer-coded by rank)
    predictors: List[str]
    distr: str = "logit"    # logit | probit | cloglog (statsmodels OrderedModel)
    imputation: Optional[str] = "listwise"




@router.post("/ordinal")
@cpu_bound
def ordinal_regression(req: OrdinalRequest):
    return linear_models.fit_ordinal(_get_df(req.session_id), req)


# ── Formal stepwise variable selection ──────────────────────────────────────

class StepwiseRequest(BaseModel):
    session_id: str
    model_type: str               # "linear" | "logistic" | "cox"
    outcome: Optional[str] = None # required for linear / logistic
    duration_col: Optional[str] = None  # cox
    event_col: Optional[str] = None     # cox
    candidates: List[str]
    direction: str = "both"       # forward | backward | both
    criterion: str = "aic"        # aic | bic | p
    p_enter: float = 0.05         # criterion=p only
    p_exit: float = 0.10          # criterion=p only
    forced_in: List[str] = []     # always retained (background / known confounders)
    imputation: Optional[str] = "listwise"




@router.post("/stepwise")
@cpu_bound
def stepwise_selection(req: StepwiseRequest):
    return linear_models.fit_stepwise(_get_df(req.session_id), req)




class DiagRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"




@router.post("/linear_diag")
def linear_diagnostics(req: DiagRequest):
    return linear_models.fit_linear_diag(_get_df(req.session_id), req)


