"""Survival-analysis sub-router for the /api/models namespace.

Thin HTTP adapter: validates the request, loads the session DataFrame, and
delegates all model fitting to services.survival. main.py mounts this router
at the /api/models prefix so the public API is unchanged.

  * /survival/km      — Kaplan-Meier survival curves + log-rank
  * /survival/cox     — Cox proportional hazards
  * /survival/cox_tv  — Cox with time-varying covariates
  * /rcs              — Restricted cubic spline regression
  * /survival/cox_rcs — Cox PH with RCS terms + interaction surface
"""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from services import survival
from routers._models_shared import get_df as _get_df, cpu_bound

router = APIRouter()


# ── Kaplan-Meier Survival ─────────────────────────────────────────────────────

class KMRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    group_col: Optional[str] = None
    stratify_col: Optional[str] = None
    imputation: Optional[str] = "listwise"


@router.post("/survival/km")
def kaplan_meier(req: KMRequest):
    return survival.fit_kaplan_meier(_get_df(req.session_id), req)


# ── Cox Proportional Hazards ──────────────────────────────────────────────────

class CoxRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    # Optional pairwise interactions — each entry is [col_A, col_B] (or
    # written "A*B"/"A:B"). Both sides must already be in `predictors`
    # (or be encoded into dummies of a predictor). Numeric × numeric =
    # element-wise product; numeric × dummy and dummy × dummy work too.
    interactions: Optional[List[List[str]]] = None


@router.post("/survival/cox")
@cpu_bound
def cox_regression(req: CoxRequest):
    return survival.fit_cox(_get_df(req.session_id), req)


# ── Cox with time-varying covariates ────────────────────────────────────────

class CoxTVRequest(BaseModel):
    session_id: str
    id_col: str                         # subject id (long-format groups)
    start_col: str                      # start time of interval
    stop_col: str                       # stop time of interval
    event_col: str                      # 1 = event in this interval
    predictors: List[str]               # may include time-varying values
    imputation: Optional[str] = "listwise"


@router.post("/survival/cox_tv")
@cpu_bound
def cox_time_varying(req: CoxTVRequest):
    return survival.fit_cox_tv(_get_df(req.session_id), req)


# ── Restricted Cubic Splines ─────────────────────────────────────────────────

class RCSRequest(BaseModel):
    session_id: str
    predictor: str
    outcome: Optional[str] = None       # required for logistic/linear
    covariates: List[str] = []
    n_knots: int = 4                    # 3, 4, or 5
    ref_value: Optional[float] = None   # OR/HR reference (median if None)
    model_type: str = "logistic"        # "logistic" | "linear" | "cox"
    imputation: str = "listwise"
    # Cox-specific (required when model_type == "cox")
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    # Optional override for Harrell percentile knots
    knot_positions: Optional[List[float]] = None
    # Covariates to interact with the spline (must also be in `covariates`).
    # For each named covariate (dummy-encoded if categorical) we add
    # (n_knots − 1) × (#dummies) interaction columns multiplying the spline
    # basis. We then refit a reduced model without those columns and report
    # an LR test as the spline × covariate interaction p-value.
    interaction_covariates: Optional[List[str]] = None


@router.post("/rcs")
@cpu_bound
def rcs_regression(req: RCSRequest):
    return survival.fit_rcs(_get_df(req.session_id), req)


# ── Multivariable Cox-RCS (with optional RCS × RCS interaction) ──────────────

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
    grid_size: int = 50  # for prediction surface


@router.post("/survival/cox_rcs")
@cpu_bound
def cox_rcs(req: CoxRCSRequest):
    return survival.fit_cox_rcs(_get_df(req.session_id), req)
