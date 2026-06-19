"""Helpers for obvious numeric sentinels in clinical columns.

The upload pipeline normalises common locale decimals, but downstream code can
still see classic missing-code sentinels such as BMI=999.  These helpers keep
the policy central and conservative: values are flagged only when they are both
outside a caller-provided plausibility range and extreme relative to the body of
the column.
"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np
import pandas as pd


PLAUSIBLE_MAX_BY_NAME = {
    "age": 120.0,
    "bmi": 100.0,
    "body_mass_index": 100.0,
}


def plausibility_max_for_column(name: str | None) -> Optional[float]:
    if not name:
        return None
    key = str(name).strip().lower()
    if key in PLAUSIBLE_MAX_BY_NAME:
        return PLAUSIBLE_MAX_BY_NAME[key]
    if "bmi" in key:
        return PLAUSIBLE_MAX_BY_NAME["bmi"]
    if key in {"fu_days", "followup_days", "follow_up_days"}:
        return None
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Numeric coercion with support for simple comma decimals."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip()
    comma_decimal = text.str.match(r"^[+-]?\d+,\d+$", na=False)
    text = text.mask(comma_decimal, text.str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")


def flag_sentinels(series: pd.Series, max_plausible: Optional[float]) -> pd.Series:
    """Return a boolean mask for obvious high-side sentinel values."""
    mask = pd.Series(False, index=series.index)
    numeric = coerce_numeric(series)
    observed = numeric.dropna()
    if observed.empty:
        return mask

    body = observed
    if max_plausible is not None:
        body = observed[observed <= max_plausible]
    if body.empty:
        body = observed

    q1 = float(body.quantile(0.25))
    q3 = float(body.quantile(0.75))
    iqr = q3 - q1
    q99 = float(body.quantile(0.99))
    robust_high = q99 + 5.0 * iqr if iqr > 0 else q99 * 1.5

    if max_plausible is not None:
        threshold = max(max_plausible, robust_high)
    else:
        threshold = robust_high

    return (numeric > threshold).fillna(False)


def sentinel_values(series: pd.Series, max_plausible: Optional[float]) -> Set[float]:
    numeric = coerce_numeric(series)
    vals = numeric[flag_sentinels(series, max_plausible)].dropna().unique()
    return {float(v) for v in vals}


def mask_sentinels(series: pd.Series, max_plausible: Optional[float]) -> pd.Series:
    numeric = coerce_numeric(series)
    return numeric.mask(flag_sentinels(series, max_plausible), np.nan)
