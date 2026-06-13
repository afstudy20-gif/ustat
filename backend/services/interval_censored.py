"""Interval-censored survival analysis.

The event time is known only to fall inside a bracket [L, R] — typical when
the outcome is detected at scheduled visits (imaging-detected recurrence,
seroconversion, radiographic progression) rather than observed exactly. Standard
Kaplan-Meier / Cox treat the event as happening at a single known time and are
biased here.

Provides:
  * Turnbull NPMLE survival curve (the interval-censored analogue of KM),
    overall and per group.
  * A parametric Weibull fit (interval-censored MLE) for a smooth curve and the
    median survival time.
  * Weibull accelerated-failure-time regression on covariates, reporting the
    time ratio (AFT-native) and the equivalent hazard ratio (Weibull PH
    duality, HR = exp(−β·shape)).

All fits use lifelines' interval-censoring estimators.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from services import store


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _to_upper(series: pd.Series) -> pd.Series:
    """Right bound: blanks / NaN / non-positive-infinity sentinels → +inf
    (right-censored — the event had not occurred by the last visit)."""
    s = pd.to_numeric(series, errors="coerce")
    return s.where(np.isfinite(s), np.inf)


def _curve_from_npmle(kmf) -> List[Dict[str, float]]:
    """Flatten a lifelines interval-censored KM fit to plottable points.

    The NPMLE is only identified up to the Turnbull intervals, so lifelines
    returns an upper and a lower step function; we report their midpoint as the
    estimate plus the band."""
    sf = kmf.survival_function_
    up_col = [c for c in sf.columns if "upper" in c.lower()]
    lo_col = [c for c in sf.columns if "lower" in c.lower()]
    if up_col and lo_col:
        up = sf[up_col[0]].to_numpy(dtype=float)
        lo = sf[lo_col[0]].to_numpy(dtype=float)
    else:
        up = lo = sf.iloc[:, 0].to_numpy(dtype=float)
    est = (up + lo) / 2.0
    times = sf.index.to_numpy(dtype=float)
    out: List[Dict[str, float]] = []
    for t, e, lo_i, up_i in zip(times, est, lo, up):
        if not math.isfinite(t):
            continue
        out.append({
            "time": round(float(t), 4),
            "survival": round(float(e), 4),
            "lower": round(float(min(lo_i, up_i)), 4),
            "upper": round(float(max(lo_i, up_i)), 4),
        })
    return out


def _weibull_regression(df: pd.DataFrame, lower: str, upper: str,
                        covariates: List[str]) -> Optional[Dict[str, Any]]:
    """Weibull AFT interval-censored regression → time ratios + derived HRs."""
    from lifelines import WeibullAFTFitter

    work = df[[lower, upper] + covariates].copy()
    work = pd.get_dummies(work, columns=[c for c in covariates
                                         if not pd.api.types.is_numeric_dtype(work[c])],
                          drop_first=True)
    design_cols = [c for c in work.columns if c not in (lower, upper)]
    if not design_cols:
        return None

    aft = WeibullAFTFitter()
    aft.fit_interval_censoring(work, lower_bound_col=lower, upper_bound_col=upper)

    # Weibull shape ρ (lifelines stores log-shape as the rho_ intercept).
    try:
        shape = float(np.exp(aft.params_.loc[("rho_", "Intercept")]))
    except Exception:
        shape = 1.0

    summ = aft.summary
    rows: List[Dict[str, Any]] = []
    for col in design_cols:
        try:
            row = summ.loc[("lambda_", col)]
        except KeyError:
            continue
        beta = float(row["coef"])
        lo = float(row["coef lower 95%"])
        hi = float(row["coef upper 95%"])
        p = float(row["p"])
        # AFT → PH duality for Weibull: HR = exp(−β·ρ). The transform is
        # monotone decreasing, so the CI bounds swap.
        rows.append({
            "variable": str(col),
            "time_ratio": round(math.exp(beta), 4),
            "tr_ci_low": round(math.exp(lo), 4),
            "tr_ci_high": round(math.exp(hi), 4),
            "hazard_ratio": round(math.exp(-beta * shape), 4),
            "hr_ci_low": round(math.exp(-hi * shape), 4),
            "hr_ci_high": round(math.exp(-lo * shape), 4),
            "p": round(p, 6),
        })
    return {
        "shape": round(shape, 4),
        "coefficients": rows,
        "log_likelihood": round(float(aft.log_likelihood_), 4),
        "aic": round(float(aft.AIC_), 4),
        "note": ("Weibull accelerated-failure-time model fitted to the interval-"
                 "censored data. Time ratio >1 = longer survival; the hazard ratio "
                 "is the Weibull PH-equivalent (HR = exp(−β·shape))."),
    }


def interval_censored_analysis(req) -> Dict[str, Any]:
    """Entry point used by the router. `req` carries session_id, lower_col,
    upper_col, optional covariates and group_col."""
    from lifelines import KaplanMeierFitter, WeibullFitter

    df = _get_df(req.session_id)
    lower, upper = req.lower_col, req.upper_col
    for c in (lower, upper):
        if c not in df.columns:
            raise HTTPException(status_code=422, detail=f"Column '{c}' not found.")

    work = df.copy()
    work[lower] = pd.to_numeric(work[lower], errors="coerce")
    work[upper] = _to_upper(work[upper])
    work = work[work[lower].notna()]
    # Keep only valid brackets: 0 ≤ L ≤ R (R may be +inf for right-censored).
    work = work[(work[lower] >= 0) & (work[upper] >= work[lower])]
    if len(work) < 10:
        raise HTTPException(status_code=422,
                            detail="Need ≥10 valid [lower, upper] intervals after cleaning.")

    n = int(len(work))
    n_exact = int((work[lower] == work[upper]).sum())
    n_right = int(np.isinf(work[upper]).sum())
    n_interval = n - n_exact - n_right

    # ── Turnbull NPMLE (overall) ──
    kmf = KaplanMeierFitter().fit_interval_censoring(work[lower], work[upper])
    overall_curve = _curve_from_npmle(kmf)

    # ── Parametric Weibull (median survival) ──
    median_survival = None
    try:
        wf = WeibullFitter().fit_interval_censoring(work[lower], work[upper])
        ms = float(wf.median_survival_time_)
        median_survival = round(ms, 4) if math.isfinite(ms) else None
    except Exception:
        median_survival = None

    result: Dict[str, Any] = {
        "model": "Interval-censored survival",
        "n": n,
        "n_exact": n_exact,
        "n_interval_censored": n_interval,
        "n_right_censored": n_right,
        "median_survival_time": median_survival,
        "npmle_curve": overall_curve,
        "groups": None,
        "regression": None,
    }

    # ── Per-group NPMLE curves ──
    group_col = getattr(req, "group_col", None)
    if group_col:
        if group_col not in work.columns:
            raise HTTPException(status_code=422, detail=f"Group column '{group_col}' not found.")
        groups = []
        for lvl, sub in work.groupby(group_col):
            if len(sub) < 5:
                continue
            try:
                g_kmf = KaplanMeierFitter().fit_interval_censoring(sub[lower], sub[upper])
                groups.append({
                    "level": str(lvl),
                    "n": int(len(sub)),
                    "curve": _curve_from_npmle(g_kmf),
                })
            except Exception:
                continue
        result["groups"] = groups or None

    # ── Weibull AFT regression on covariates (+ group) ──
    covariates = list(getattr(req, "covariates", None) or [])
    if group_col and group_col not in covariates:
        covariates = covariates + [group_col]
    if covariates:
        missing = [c for c in covariates if c not in work.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Covariate(s) not found: {missing}")
        try:
            reg = work[[lower, upper] + covariates].dropna()
            if len(reg) >= max(20, 5 * len(covariates)):
                result["regression"] = _weibull_regression(reg, lower, upper, covariates)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - numerical failure path
            result["regression"] = {"error": f"Regression failed: {exc}"}

    result["result_text"] = _result_text(result)
    return result


def _result_text(r: Dict[str, Any]) -> str:
    parts = [
        f"Interval-censored survival analysis on {r['n']} observations "
        f"({r['n_interval_censored']} interval-censored, {r['n_right_censored']} right-censored). "
        "The Turnbull NPMLE survival curve is reported (the interval-censored analogue of "
        "Kaplan-Meier, which would be biased here)."
    ]
    if r.get("median_survival_time") is not None:
        parts.append(f" Weibull-estimated median survival time = {r['median_survival_time']}.")
    reg = r.get("regression")
    if reg and reg.get("coefficients"):
        sig = [c["variable"] for c in reg["coefficients"] if c["p"] < 0.05]
        if sig:
            parts.append(f" Significant predictor(s) of the event time: {', '.join(sig)}.")
    return "".join(parts)
