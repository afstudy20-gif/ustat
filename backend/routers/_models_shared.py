"""Shared helpers used by models.py, models_survival.py, models_causal.py, etc.

Extracted to avoid duplicating _get_df / _compute_vif across sub-routers.
"""

import asyncio
import functools
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException

from services import store


def cpu_bound(fn):
    """Decorator: run a sync endpoint handler in a thread via asyncio.to_thread.

    Frees the event loop while CPU-heavy statistical computations run.
    Preserves the original function's signature so FastAPI correctly resolves
    Pydantic body models (critical when `from __future__ import annotations`
    turns type hints into strings).
    """
    import inspect

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    # Explicitly copy the signature so FastAPI sees the original parameters
    # (req: SomeRequest) instead of (*args, **kwargs).
    wrapper.__signature__ = inspect.signature(fn)
    # Copy __globals__ so FastAPI can resolve string annotations (from
    # `from __future__ import annotations`) back to actual types.
    wrapper.__globals__.update(fn.__globals__)
    return wrapper


def get_df(session_id: str) -> pd.DataFrame:
    """Fetch the (optionally filtered) DataFrame for a session."""
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def compute_vif(X: pd.DataFrame) -> dict:
    """Variance Inflation Factor per column of the design matrix X.

    Excludes the intercept ('const' column) from the calculation. Returns
    {column_name: vif_float} so callers can splice into their coefficient
    rows. VIF > 5 ≈ moderate multicollinearity, > 10 ≈ severe.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xn = X.copy().astype(float)
    if "const" in Xn.columns:
        Xn = Xn.drop(columns=["const"])
    if Xn.shape[1] < 2:
        # VIF undefined for a single predictor — no other column to regress on.
        return {c: 1.0 for c in Xn.columns}
    arr = Xn.values
    out: dict = {}
    for i, col in enumerate(Xn.columns):
        try:
            v = float(variance_inflation_factor(arr, i))
            if not np.isfinite(v):
                v = None  # perfect collinearity → inf
        except Exception:
            v = None
        out[str(col)] = v
    return out


def add_pairwise_interactions(
    enc: pd.DataFrame,
    interactions: Optional[List[List[str]]],
    requested_predictors: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Append pairwise interaction columns (A × B = element-wise product on
    the dummy-coded design) to a design matrix. Numeric columns survive as
    themselves; categorical columns expand into one column per surviving
    dummy (so SEX × AGE on a 3-level SEX becomes SEX_M:AGE + SEX_O:AGE).

    Returns (new_enc, list_of_added_column_names). Raises HTTPException on
    invalid input.
    """
    if not interactions:
        return enc, []

    out = enc.copy()
    added: List[str] = []

    def _members(name: str) -> List[str]:
        if name in out.columns:
            return [name]
        prefix = f"{name}_"
        return [c for c in out.columns if c.startswith(prefix)]

    requested_set = set(requested_predictors)
    for pair in interactions:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise HTTPException(status_code=422, detail=f"Each interaction must be a [colA, colB] pair. Got: {pair}")
        a_name, b_name = pair
        for nm in (a_name, b_name):
            if nm not in requested_set:
                raise HTTPException(
                    status_code=422,
                    detail=f"Interaction '{a_name} × {b_name}': '{nm}' must already be in the predictors list."
                )
        a_members = _members(a_name)
        b_members = _members(b_name)
        if not a_members or not b_members:
            raise HTTPException(
                status_code=422,
                detail=f"Interaction '{a_name} × {b_name}': one or both columns did not survive dummy encoding."
            )
        for a in a_members:
            for b in b_members:
                col = f"{a}:{b}"
                if col in out.columns:
                    continue  # avoid duplicate when user lists same pair twice
                out[col] = out[a] * out[b]
                added.append(col)
    return out, added
