"""Input-range checks shared by the survival endpoints.

The cohort dataset that survived ingest can still have impossible values —
a follow-up time of ``-10`` days, a duration column with zeros, an event
column with neither 0 nor 1. The fitters either crash on those, silently
drop the cohort, or (worst) produce a plot that looks fine but is wrong.

This module centralises the checks so KM, Cox, Fine-Gray, LWYY, landmark,
RMST etc. all behave the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException


def validate_survival_inputs(
    df: pd.DataFrame, duration_col: str, event_col: str,
) -> None:
    """Reject obvious dataset-level errors before any survival fit.

    Raises HTTPException 400 on:
    - missing columns
    - non-positive durations (``time ≤ 0``)
    - non-binary or non-numeric event flag

    Successful returns mean the caller can fit safely.
    """
    for c in (duration_col, event_col):
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found.")

    dur = pd.to_numeric(df[duration_col], errors="coerce")
    bad = int((dur <= 0).sum())
    if bad > 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{duration_col}' contains {bad} non-positive value(s) "
                f"(time ≤ 0). Follow-up time must be > 0; remove or recode "
                f"those rows before fitting a survival model."
            ),
        )

    ev = pd.to_numeric(df[event_col], errors="coerce").dropna()
    uniq = set(ev.unique())
    extras = uniq - {0, 1, 0.0, 1.0}
    if extras:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{event_col}' must be binary 0/1. Found values: "
                f"{sorted(str(v) for v in uniq)}."
            ),
        )


def warn_dev_eq_val(
    dev_session_id: str | None,
    val_session_id: str | None,
) -> str | None:
    """If the user passed the development cohort as the validation cohort,
    return a warning string the caller can attach to the response. (Not an
    error — a confused user wants to see this loudly, not be blocked.)"""
    if dev_session_id and val_session_id and dev_session_id == val_session_id:
        return (
            "Validation cohort is identical to the development cohort. "
            "Reported metrics describe in-sample fit, not external validation; "
            "load a separate held-out dataset for a meaningful result."
        )
    return None
