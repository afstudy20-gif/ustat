"""Logistic-family sub-router for the /api/models namespace.

Thin HTTP adapter: validates the request, loads the session DataFrame, and
delegates all model fitting to services.regression. main.py mounts this router
at the /api/models prefix so the public API is unchanged.

  * /logistic       — Standard logistic regression (interactions, scaling,
                      stepwise, VIF, AUC, Hosmer-Lemeshow, classification table)
  * /firth_logistic — Firth's penalized likelihood logistic (for separation)
  * /poisson        — Poisson count regression
  * /logistic_table — Univariate + multivariate OR table
"""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from services import regression
from routers._models_shared import get_df as _get_df, cpu_bound

router = APIRouter()


# ── Logistic Regression ───────────────────────────────────────────────────────

class LogisticRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    scale_factors: Optional[dict] = None
    selection: Optional[str] = "all"
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False
    # Optional pairwise interactions — same semantics as the Cox endpoint.
    interactions: Optional[List[List[str]]] = None
    # When True the OR-table endpoint fits Firth's penalised logistic for
    # each row instead of standard MLE — required for rare events / (quasi-)
    # separated data where sm.Logit returns infinite ORs. Affects only the
    # /logistic_table endpoint; the standalone /logistic endpoint already
    # has its dedicated /firth_logistic peer.
    use_firth: Optional[bool] = False


@router.post("/logistic")
@cpu_bound
def logistic_regression(req: LogisticRequest):
    return regression.fit_logistic(_get_df(req.session_id), req)


# ── Firth Penalized Logistic Regression ──────────────────────────────────────

class FirthLogisticRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    scale_factors: Optional[dict] = None
    imputation: Optional[str] = "listwise"
    max_iter: int = 50
    tol: float = 1e-6
    interactions: Optional[List[List[str]]] = None


@router.post("/firth_logistic")
@cpu_bound
def firth_logistic_regression(req: FirthLogisticRequest):
    return regression.fit_firth(_get_df(req.session_id), req)


# ── Poisson Regression ───────────────────────────────────────────────────────

class PoissonRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/poisson")
def poisson_regression(req: PoissonRequest):
    return regression.fit_poisson(_get_df(req.session_id), req)


# ── Logistic OR Table (Univariate + Multivariate) ────────────────────────────

@router.post("/logistic_table")
def logistic_or_table(req: LogisticRequest):
    return regression.fit_or_table(_get_df(req.session_id), req)
