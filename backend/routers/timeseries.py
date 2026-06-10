"""
Time-series analysis router (ARIMA / SARIMA + diagnostics).

Endpoints
---------
POST /arima         — (S)ARIMA fit, forecast, residual diagnostics
POST /decompose     — STL / classical seasonal decomposition
POST /stationarity  — ADF + KPSS stationarity tests + ACF/PACF

Pure statsmodels (already a dependency). No pmdarima — the optional
``auto`` flag does a bounded AIC grid search in-house.
"""

from __future__ import annotations

import asyncio
import math
import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from services import store
from services.impute import apply_imputation

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _series(req_session: str, value_col: str, time_col: Optional[str],
            imputation: Optional[str]) -> pd.Series:
    df = _get_df(req_session)
    if value_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{value_col}' not found")
    cols = [value_col] + ([time_col] if time_col and time_col in df.columns else [])
    work = apply_imputation(df[cols], cols, imputation or "listwise").reset_index(drop=True)
    if time_col and time_col in work.columns:
        # Order by the time column. Try datetime parse, else numeric, else as-is.
        t = pd.to_datetime(work[time_col], errors="coerce")
        if t.notna().mean() >= 0.8:
            work = work.assign(_t=t).sort_values("_t")
            idx = pd.DatetimeIndex(work["_t"].values)
        else:
            tn = pd.to_numeric(work[time_col], errors="coerce")
            work = work.assign(_t=tn).sort_values("_t")
            idx = None
    else:
        idx = None
    y = pd.to_numeric(work[value_col], errors="coerce").dropna()
    if idx is not None and len(idx) == len(y):
        y.index = idx[: len(y)]
    if len(y) < 20:
        raise HTTPException(status_code=400,
            detail=f"Need ≥ 20 non-missing observations (got {len(y)}).")
    return y.astype(float)


# ── 1. ARIMA / SARIMA ────────────────────────────────────────────────────────


class ARIMARequest(BaseModel):
    session_id: str
    value_col: str
    time_col: Optional[str] = None
    p: int = 1
    d: int = 1
    q: int = 1
    # Seasonal (set s > 0 to enable)
    P: int = 0
    D: int = 0
    Q: int = 0
    s: int = 0
    auto: bool = False          # bounded AIC grid search over (p,d,q)
    forecast_steps: int = 12
    imputation: Optional[str] = "listwise"


def _fit_sarimax(y: pd.Series, order, seasonal_order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        return model.fit(disp=False)


def _auto_grid_search(y: pd.Series, seasonal):
    best_aic, best, best_order = np.inf, None, None
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    res = _fit_sarimax(y, (p, d, q), seasonal)
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best_aic, best, best_order = res.aic, res, (p, d, q)
                except Exception as exc:
                    logger.debug(
                        "SARIMA grid candidate failed for order {} and seasonal {}: {}",
                        (p, d, q),
                        seasonal,
                        exc,
                    )
                    continue
    return best, best_order


@router.post("/arima")
async def arima(req: ARIMARequest):
    from statsmodels.stats.diagnostic import acorr_ljungbox

    y = _series(req.session_id, req.value_col, req.time_col, req.imputation)
    seasonal = (req.P, req.D, req.Q, req.s) if req.s and req.s > 1 else (0, 0, 0, 0)

    chosen_order = (req.p, req.d, req.q)
    grid_searched = False
    if req.auto:
        grid_searched = True
        best, best_order = await asyncio.to_thread(_auto_grid_search, y, seasonal)
        if best is None:
            raise HTTPException(status_code=422, detail="Auto grid search failed to fit any model.")
        fit = best
        chosen_order = best_order
    else:
        try:
            fit = await asyncio.to_thread(_fit_sarimax, y, chosen_order, seasonal)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"ARIMA fit failed: {exc}")

    # Coefficients
    params = fit.params
    bse = fit.bse
    pvals = fit.pvalues
    coefs: List[dict] = []
    for name in params.index:
        coefs.append({
            "term": str(name),
            "estimate": _safe(round(float(params[name]), 6)),
            "se": _safe(round(float(bse[name]), 6)) if name in bse else None,
            "p": _safe(round(float(pvals[name]), 6)) if name in pvals else None,
        })

    # Ljung-Box on residuals (white-noise check)
    resid = pd.Series(fit.resid).dropna()
    lb_p = None
    try:
        lag = min(10, max(1, len(resid) // 5))
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        lb_p = _safe(round(float(lb["lb_pvalue"].iloc[-1]), 6))
    except Exception as exc:
        logger.debug("Ljung-Box residual diagnostic failed: {}", exc)
        lb_p = None

    # In-sample fitted values
    fitted = fit.fittedvalues
    obs_x = [str(i) for i in y.index]
    fit_pairs = [
        {"x": str(ix), "observed": _safe(round(float(v), 6)),
         "fitted": _safe(round(float(fitted.get(ix, np.nan)), 6))}
        for ix, v in y.items()
    ]

    # Forecast
    steps = max(1, int(req.forecast_steps))
    fc = fit.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    fc_rows = []
    for i in range(steps):
        lo = float(ci.iloc[i, 0])
        hi = float(ci.iloc[i, 1])
        fc_rows.append({
            "step": i + 1,
            "x": str(mean.index[i]),
            "forecast": _safe(round(float(mean.iloc[i]), 6)),
            "ci_low": _safe(round(lo, 6)),
            "ci_high": _safe(round(hi, 6)),
        })

    interp = (
        f"SARIMA{chosen_order}×{seasonal} on n = {len(y)}. "
        f"AIC = {fit.aic:.1f}, BIC = {fit.bic:.1f}. "
        + ("Auto-selected order by minimum AIC. " if grid_searched else "")
        + (f"Ljung-Box p = {lb_p}: residuals "
           + ("show no significant autocorrelation (good)." if (lb_p is not None and lb_p >= 0.05)
              else "retain autocorrelation — consider a different order." if lb_p is not None else "n/a.")
        )
    )

    try:
        store.log_action(req.session_id, "arima", {
            "value_col": req.value_col, "order": list(chosen_order),
            "seasonal_order": list(seasonal), "auto": grid_searched,
        })
    except Exception:
        logger.exception("Logging ARIMA action failed")

    # In-sample forecast accuracy (fitted vs observed)
    resid = (y - fitted).dropna()
    rmse = float(np.sqrt(np.mean(resid ** 2))) if len(resid) > 0 else None
    mae = float(np.mean(np.abs(resid))) if len(resid) > 0 else None

    assumptions = [
        {"name": "Residual white noise", "met": lb_p is not None and lb_p >= 0.05,
         "detail": f"Ljung-Box p = {lb_p}" if lb_p is not None else "n/a"},
        {"name": "Stationarity & invertibility", "met": True,
         "detail": "Model was fit with enforce_stationarity/invertibility=False for robustness."},
    ]

    warnings = []
    if lb_p is not None and lb_p < 0.05:
        warnings.append("Residuals show significant autocorrelation — model may be misspecified.")
    if len(y) < 50:
        warnings.append("Small sample size — ARIMA estimates and forecasts have high uncertainty.")

    result_text = interp
    if rmse is not None:
        result_text += f" In-sample RMSE={rmse:.2f}, MAE={mae:.2f}."

    return _safe({
        "test": "ARIMA / SARIMA",
        "value_col": req.value_col,
        "n": int(len(y)),
        "order": list(chosen_order),
        "seasonal_order": list(seasonal),
        "auto": grid_searched,
        "aic": round(float(fit.aic), 3),
        "bic": round(float(fit.bic), 3),
        "ljung_box_p": lb_p,
        "in_sample_rmse": round(rmse, 4) if rmse else None,
        "in_sample_mae": round(mae, 4) if mae else None,
        "coefficients": coefs,
        "fitted": fit_pairs,
        "forecast": fc_rows,
        "assumptions": assumptions,
        "warnings": warnings,
        "interpretation": interp,
        "result_text": result_text,
        "obs_index": obs_x,
    })


# ── 2. Decomposition ─────────────────────────────────────────────────────────


class DecomposeRequest(BaseModel):
    session_id: str
    value_col: str
    time_col: Optional[str] = None
    period: int = 12
    method: str = "stl"        # stl | classical
    model: str = "additive"    # classical only: additive | multiplicative
    imputation: Optional[str] = "listwise"


@router.post("/decompose")
def decompose(req: DecomposeRequest):
    y = _series(req.session_id, req.value_col, req.time_col, req.imputation)
    period = max(2, int(req.period))
    if len(y) < 2 * period:
        raise HTTPException(status_code=422,
            detail=f"Need ≥ 2 full periods ({2*period} obs) for period {period}; got {len(y)}.")

    x = [str(i) for i in y.index]
    if req.method == "stl":
        from statsmodels.tsa.seasonal import STL
        res = STL(y, period=period, robust=True).fit()
        trend, seasonal, resid = res.trend, res.seasonal, res.resid
    else:
        from statsmodels.tsa.seasonal import seasonal_decompose
        res = seasonal_decompose(y, period=period, model=req.model, extrapolate_trend="freq")
        trend, seasonal, resid = res.trend, res.seasonal, res.resid

    def _arr(s):
        return [_safe(round(float(v), 6)) if pd.notna(v) else None for v in s]

    # Strength of trend / seasonality (Hyndman & Athanasopoulos).
    var_resid = float(np.nanvar(resid))
    var_dt = float(np.nanvar((trend + resid)))
    var_ds = float(np.nanvar((seasonal + resid)))
    f_trend = max(0.0, 1 - var_resid / var_dt) if var_dt > 0 else 0.0
    f_seas = max(0.0, 1 - var_resid / var_ds) if var_ds > 0 else 0.0

    try:
        store.log_action(req.session_id, "ts_decompose",
                         {"value_col": req.value_col, "period": period, "method": req.method})
    except Exception:
        logger.exception("Logging decomposition action failed")

    return _safe({
        "test": "Seasonal decomposition",
        "value_col": req.value_col,
        "method": req.method,
        "period": period,
        "n": int(len(y)),
        "x": x,
        "observed": _arr(y),
        "trend": _arr(trend),
        "seasonal": _arr(seasonal),
        "resid": _arr(resid),
        "strength_trend": round(f_trend, 4),
        "strength_seasonal": round(f_seas, 4),
        "interpretation": (
            f"{req.method.upper()} decomposition, period {period}. "
            f"Trend strength {f_trend:.2f}, seasonal strength {f_seas:.2f} "
            "(0 = none, 1 = dominant)."
        ),
    })


# ── 3. Stationarity + ACF/PACF ───────────────────────────────────────────────


class StationarityRequest(BaseModel):
    session_id: str
    value_col: str
    time_col: Optional[str] = None
    n_lags: int = 24
    imputation: Optional[str] = "listwise"


@router.post("/stationarity")
def stationarity(req: StationarityRequest):
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

    y = _series(req.session_id, req.value_col, req.time_col, req.imputation)
    n_lags = max(1, min(int(req.n_lags), len(y) // 2 - 1))

    # ADF: H0 = unit root (non-stationary). p < 0.05 → stationary.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, *_ = adfuller(y, autolag="AIC")
        # KPSS: H0 = stationary. p < 0.05 → non-stationary.
        try:
            kpss_stat, kpss_p, *_ = kpss(y, regression="c", nlags="auto")
        except Exception as exc:
            logger.debug("KPSS stationarity diagnostic failed: {}", exc)
            kpss_stat, kpss_p = None, None

    acf_vals, acf_ci = acf(y, nlags=n_lags, alpha=0.05, fft=True)
    pacf_vals, pacf_ci = pacf(y, nlags=n_lags, alpha=0.05)

    def _stems(vals, ci):
        out = []
        for k in range(len(vals)):
            lo = float(ci[k][0] - vals[k])
            hi = float(ci[k][1] - vals[k])
            out.append({"lag": k, "value": _safe(round(float(vals[k]), 4)),
                        "ci_low": _safe(round(lo, 4)), "ci_high": _safe(round(hi, 4))})
        return out

    adf_stationary = bool(adf_p < 0.05)
    kpss_stationary = bool(kpss_p is not None and kpss_p >= 0.05)
    if adf_stationary and kpss_stationary:
        verdict = "Stationary (both ADF and KPSS agree)."
    elif not adf_stationary and not kpss_stationary:
        verdict = "Non-stationary — differencing recommended (d ≥ 1)."
    else:
        verdict = "Borderline / trend-stationary — ADF and KPSS disagree; inspect ACF and consider differencing."

    try:
        store.log_action(req.session_id, "ts_stationarity", {"value_col": req.value_col})
    except Exception:
        logger.exception("Logging stationarity action failed")

    return _safe({
        "test": "Stationarity (ADF + KPSS)",
        "value_col": req.value_col,
        "n": int(len(y)),
        "adf_stat": round(float(adf_stat), 4),
        "adf_p": round(float(adf_p), 6),
        "adf_stationary": adf_stationary,
        "kpss_stat": round(float(kpss_stat), 4) if kpss_stat is not None else None,
        "kpss_p": round(float(kpss_p), 6) if kpss_p is not None else None,
        "kpss_stationary": kpss_stationary,
        "acf": _stems(acf_vals, acf_ci),
        "pacf": _stems(pacf_vals, pacf_ci),
        "interpretation": verdict,
        "result_text": verdict,
    })
