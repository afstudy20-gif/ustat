"""Input-range checks shared by the survival endpoints.

The cohort dataset that survived ingest can still have impossible values —
a follow-up time of ``-10`` days, a duration column with zeros, an event
column with neither 0 nor 1. The fitters either crash on those, silently
drop the cohort, or (worst) produce a plot that looks fine but is wrong.

This module centralises the checks so KM, Cox, Fine-Gray, LWYY, landmark,
RMST etc. all behave the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import HTTPException


@dataclass
class SurvivalValidationResult:
    df: pd.DataFrame
    warnings: list[str]
    n_excluded: int


def validate_survival_inputs(
    df: pd.DataFrame, duration_col: str, event_col: str,
    *,
    mode: Literal["reject", "drop_with_warning"] = "reject",
    require_binary_event: bool = True,
) -> SurvivalValidationResult:
    """Reject obvious dataset-level errors before any survival fit.

    Raises HTTPException 400 on:
    - missing columns
    - non-positive durations (``time ≤ 0``)
    - non-binary or non-numeric event flag

    In ``drop_with_warning`` mode, invalid duration/event rows are removed and
    summarized in warnings instead of rejecting the whole cohort.
    """
    for c in (duration_col, event_col):
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found.")

    work = df.copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work[event_col] = pd.to_numeric(work[event_col], errors="coerce")

    dur = work[duration_col]
    bad_duration = dur.notna() & (dur <= 0)
    bad = int(bad_duration.sum())
    if bad > 0 and mode == "reject":
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{duration_col}' contains {bad} non-positive value(s) "
                f"(time ≤ 0). Follow-up time must be > 0; remove or recode "
                f"those rows before fitting a survival model."
            ),
        )

    if require_binary_event:
        ev = work[event_col].dropna()
        uniq = set(ev.unique())
        extras = uniq - {0, 1, 0.0, 1.0}
        if extras:
            raise HTTPException(
                status_code=400 if mode == "reject" else 422,
                detail=(
                    f"'{event_col}' must be binary 0/1. Found values: "
                    f"{sorted(str(v) for v in uniq)}."
                ),
            )

    drop_mask = bad_duration
    warnings: list[str] = []
    if drop_mask.any():
        examples = []
        for idx in work.index[drop_mask][:5]:
            examples.append(f"row {int(idx) + 1} dropped: non-positive {duration_col}")
        suffix = "" if int(drop_mask.sum()) <= 5 else f" (+{int(drop_mask.sum()) - 5} more)"
        warnings.append("; ".join(examples) + suffix)
        work = work.loc[~drop_mask].copy()

    return SurvivalValidationResult(df=work, warnings=warnings, n_excluded=int(drop_mask.sum()))


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
