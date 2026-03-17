"""
Missing-data imputation helper shared across stats.py and models.py.

Strategies
----------
listwise  : drop any row missing ANY value in the selected columns (SPSS/R default)
median    : fill numeric columns with the column median, then drop remaining NaN
mean      : fill numeric columns with the column mean, then drop remaining NaN
mice      : Multiple Imputation by Chained Equations via sklearn IterativeImputer,
            falls back to median if sklearn is not available or fitting fails
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List


def apply_imputation(df: pd.DataFrame, cols: List[str], strategy: str = "listwise") -> pd.DataFrame:
    """Return a cleaned DataFrame after applying `strategy` to `cols`."""
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df

    # ------------------------------------------------------------------
    # Listwise (complete-case) — always the safe default
    # ------------------------------------------------------------------
    if strategy in ("listwise", "none", "", None):
        return df.dropna(subset=valid_cols)

    df = df.copy()
    num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(df[c])]

    # ------------------------------------------------------------------
    # Median imputation
    # ------------------------------------------------------------------
    if strategy == "median":
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

    # ------------------------------------------------------------------
    # Mean imputation (not recommended for skewed clinical data)
    # ------------------------------------------------------------------
    elif strategy == "mean":
        for col in num_cols:
            df[col] = df[col].fillna(df[col].mean())

    # ------------------------------------------------------------------
    # MICE (Multiple Imputation by Chained Equations)
    # ------------------------------------------------------------------
    elif strategy == "mice":
        if num_cols:
            try:
                # sklearn >= 0.21 requires the experimental flag in older builds
                try:
                    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                except ImportError:
                    pass
                from sklearn.impute import IterativeImputer

                imp = IterativeImputer(random_state=42, max_iter=10, verbose=0)
                df[num_cols] = imp.fit_transform(df[num_cols])
            except Exception:
                # Graceful fallback to median if MICE fails
                for col in num_cols:
                    df[col] = df[col].fillna(df[col].median())

    # Always drop rows still missing after imputation
    # (covers non-numeric columns and edge cases)
    return df.dropna(subset=valid_cols)


def missing_info(df: pd.DataFrame, cols: List[str]) -> dict:
    """Return a structured missing-value summary for the given columns."""
    valid_cols = [c for c in cols if c in df.columns]
    total = len(df)

    per_col: dict = {}
    for col in valid_cols:
        n = int(df[col].isna().sum())
        per_col[col] = {
            "count": n,
            "pct": round(n / total * 100, 1) if total > 0 else 0.0,
        }

    rows_affected = int(df[valid_cols].isna().any(axis=1).sum()) if valid_cols else 0
    return {
        "total_rows": total,
        "rows_affected": rows_affected,
        "pct_affected": round(rows_affected / total * 100, 1) if total > 0 else 0.0,
        "per_column": per_col,
    }
