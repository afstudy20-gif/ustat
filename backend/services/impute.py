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
from typing import Dict, List, Optional

from services.dirty_value_guard import flag_sentinels, plausibility_max_for_column, sentinel_values


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


def apply_passive_imputation(df: pd.DataFrame, formulas: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Recompute derived variables after imputation.

    Example: {"bmi": "weight / (height ** 2)"}. Expressions are evaluated with
    pandas.eval against dataframe columns only.
    """
    if not formulas:
        return df
    out = df.copy()
    for target, expr in formulas.items():
        try:
            out[target] = out.eval(expr)
        except Exception:
            continue
    return out


def add_survival_auxiliary_variables(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    *,
    prefix: str = "__surv_aux",
) -> pd.DataFrame:
    """
    Add survival-specific auxiliary variables for imputation models:
    log time and Nelson-Aalen cumulative hazard at each subject's follow-up.
    """
    out = df.copy()
    if duration_col not in out.columns or event_col not in out.columns:
        return out
    duration = pd.to_numeric(out[duration_col], errors="coerce")
    event = pd.to_numeric(out[event_col], errors="coerce").fillna(0).astype(int)
    out[f"{prefix}_log_time"] = np.log(np.clip(duration.astype(float), 1e-8, None))
    try:
        from lifelines import NelsonAalenFitter

        mask = duration.notna()
        naf = NelsonAalenFitter()
        naf.fit(duration[mask].astype(float), event_observed=event[mask].astype(int))
        cumulative = naf.cumulative_hazard_at_times(duration.fillna(duration.median()).astype(float)).to_numpy()
        out[f"{prefix}_nelson_aalen"] = np.asarray(cumulative, dtype=float)
    except Exception:
        order = duration.rank(method="average", pct=True)
        out[f"{prefix}_nelson_aalen"] = -np.log(np.clip(1.0 - order.fillna(order.median()), 1e-6, 1.0))
    return out


def missing_info(df: pd.DataFrame, cols: List[str]) -> dict:
    """Return a structured missing-value summary for the given columns."""
    valid_cols = [c for c in cols if c in df.columns]
    total = len(df)

    per_col: dict = {}
    for col in valid_cols:
        max_plausible = plausibility_max_for_column(col)
        raw_missing = df[col].isna()
        implausible = flag_sentinels(df[col], max_plausible)
        n = int((raw_missing | implausible).sum())
        per_col[col] = {
            "count": n,
            "raw_count": int(raw_missing.sum()),
            "n_implausible": int(implausible.sum()),
            "implausible_values": sorted(sentinel_values(df[col], max_plausible)),
            "pct": round(n / total * 100, 1) if total > 0 else 0.0,
        }

    if valid_cols:
        masks = []
        for col in valid_cols:
            max_plausible = plausibility_max_for_column(col)
            masks.append(df[col].isna() | flag_sentinels(df[col], max_plausible))
        rows_affected = int(pd.concat(masks, axis=1).any(axis=1).sum())
    else:
        rows_affected = 0
    return {
        "total_rows": total,
        "rows_affected": rows_affected,
        "pct_affected": round(rows_affected / total * 100, 1) if total > 0 else 0.0,
        "per_column": per_col,
    }
