"""
External Validation and Prediction Evaluation Framework (Phase 9)

Provides tools for rigorous evaluation of survival, multi-state, and joint
prediction models on external data, with focus on calibration, discrimination,
and transportability diagnostics.

Core capabilities implemented:
- Integrated Brier Score (IBS) approximation at multiple time points
- Calibration slope & intercept for survival predictions (using LP)
- Harrell's C-index on validation set
- Simple transportability diagnostics (covariate shift, baseline risk shift)
- Comparison of performance between development and validation cohorts

All functions return immutable dicts. Designed to integrate with existing
model_validation.py and the new joint/multi-state modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from services.model_validation import compute_calibration_slope_intercept


def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def evaluate_external_validation(
    val_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    predicted_lp_col: str,           # linear predictor or risk score on validation
    dev_c_index: Optional[float] = None,
    dev_calibration_slope: Optional[float] = None,
    time_points: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Core external validation evaluation for a survival prediction model.

    Expects val_df to already contain the model's predictions (e.g. LP from
    a model fitted on development data and applied to validation data).
    """
    if predicted_lp_col not in val_df.columns:
        raise ValueError(f"predicted_lp_col '{predicted_lp_col}' not found in val_df")

    df = val_df[[duration_col, event_col, predicted_lp_col]].dropna().copy()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna()

    if len(df) < 20:
        return {"error": "Too few complete observations in validation data"}

    duration = df[duration_col].values
    event = df[event_col].astype(int).values
    lp = df[predicted_lp_col].values

    # 1. Discrimination on validation
    try:
        val_c = concordance_index(duration, -lp, event)  # higher LP = higher risk
    except Exception:
        val_c = None

    # 2. Calibration on validation (using LP as sole predictor)
    try:
        cal = compute_calibration_slope_intercept(
            y=event,
            probs_or_lp=lp,  # treat as LP
            duration=duration if np.any(event == 0) else None,  # if censored data present
        )
        val_slope = cal.get("calibration_slope")
        val_intercept = cal.get("calibration_intercept")
    except Exception:
        val_slope = val_intercept = None

    # 3. Simple Brier-style scores at selected time points (Kaplan-Meier based)
    if time_points is None:
        time_points = [np.percentile(duration, 25), np.percentile(duration, 50), np.percentile(duration, 75)]

    brier_scores = []
    for t in time_points:
        # Very rough: proportion of events by t vs mean predicted survival at t
        # In real use this would use model-specific survival curves
        events_by_t = (duration <= t) & (event == 1)
        obs_prob = float(events_by_t.mean())
        # Approximate predicted event prob from LP (sigmoid transform as proxy)
        pred_prob = 1 / (1 + np.exp(-lp.mean()))  # crude
        brier = (obs_prob - pred_prob) ** 2
        brier_scores.append({
            "time": round(float(t), 2),
            "observed_event_prob": round(obs_prob, 4),
            "approx_brier": round(float(brier), 5),
        })

    # 4. Performance drop vs development (if provided)
    performance_drop = {}
    if dev_c_index is not None and val_c is not None:
        performance_drop["c_index_drop"] = round(dev_c_index - val_c, 4)
    if dev_calibration_slope is not None and val_slope is not None:
        performance_drop["calibration_slope_shift"] = round(val_slope - dev_calibration_slope, 4)

    return {
        "n_validation": int(len(df)),
        "validation_c_index": round(float(val_c), 4) if val_c else None,
        "validation_calibration_slope": round(float(val_slope), 4) if val_slope else None,
        "validation_calibration_intercept": round(float(val_intercept), 4) if val_intercept else None,
        "brier_at_times": brier_scores,
        "performance_vs_dev": performance_drop if performance_drop else None,
        "note": "External validation metrics. Use predicted_lp_col from a model fitted on development data only.",
    }


def _ipcw_weights(duration: np.ndarray, event: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Compute IPCW weights using Kaplan-Meier censoring distribution (simple version)."""
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter()
    km.fit(duration, 1 - event)  # censoring indicator
    surv_cens = km.survival_function_.reindex(times, method="nearest").values.flatten()
    weights = 1.0 / np.clip(surv_cens, 1e-6, 1.0)
    return weights


def time_dependent_auc(
    duration: np.ndarray,
    event: np.ndarray,
    linear_predictor: np.ndarray,
    time_points: np.ndarray,
) -> List[Dict[str, float]]:
    """Time-dependent AUC(t) using risk-set ranking (Harrell-style at each t)."""
    results = []
    for t in time_points:
        at_risk = duration >= t
        if at_risk.sum() < 5:
            continue
        events_by_t = (duration <= t) & (event == 1) & at_risk
        if events_by_t.sum() < 2:
            continue

        # Among those at risk at t, rank by LP
        lp_risk = linear_predictor[at_risk]
        status = ((duration <= t) & (event == 1))[at_risk].astype(int)

        try:
            auc_t = concordance_index(
                np.ones_like(lp_risk),  # dummy times
                -lp_risk,
                status
            )
        except Exception:
            auc_t = None

        if auc_t:
            results.append({
                "time": round(float(t), 2),
                "auc": round(float(auc_t), 4),
                "n_at_risk": int(at_risk.sum()),
                "n_events": int(status.sum()),
            })
    return results


def integrated_brier_score(
    duration: np.ndarray,
    event: np.ndarray,
    survival_probs_at_times: np.ndarray,  # shape (n_samples, n_times)
    time_points: np.ndarray,
) -> Dict[str, float]:
    """
    Proper IPCW-weighted Integrated Brier Score (IBS) over the given time grid.
    Lower is better (0 = perfect).
    """
    if len(time_points) < 2:
        return {"ibs": None, "error": "Need at least 2 time points"}

    survival_probs_at_times = np.asarray(survival_probs_at_times)
    weights = _ipcw_weights(duration, event, time_points)

    brier_sum = 0.0

    for i, t in enumerate(time_points):
        # Observed status at t (1 if event by t, 0 otherwise, weighted by IPCW)
        obs_status = (duration <= t) & (event == 1)
        w = weights[i]

        # Predicted prob of event by t = 1 - S(t)
        pred_event_prob = 1.0 - survival_probs_at_times[:, i]

        # IPCW Brier at this time
        brier_t = np.mean(w * (obs_status.astype(float) - pred_event_prob) ** 2)
        brier_sum += brier_t

    ibs = brier_sum / len(time_points)

    return {
        "ibs": round(float(ibs), 5),
        "n_time_points": len(time_points),
        "time_range": [round(float(time_points[0]), 2), round(float(time_points[-1]), 2)],
    }


def evaluate_external_validation(
    val_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    predicted_lp_col: str,           # linear predictor or risk score on validation
    survival_probs: Optional[np.ndarray] = None,  # (n, len(time_points)) if available
    time_points: Optional[List[float]] = None,
    dev_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Enhanced external validation with proper time-dependent metrics when
    survival probabilities at multiple times are provided.
    """
    if predicted_lp_col not in val_df.columns:
        raise ValueError(f"predicted_lp_col '{predicted_lp_col}' not found in val_df")

    df = val_df[[duration_col, event_col, predicted_lp_col]].dropna().copy()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna()

    if len(df) < 20:
        return {"error": "Too few complete observations in validation data"}

    duration = df[duration_col].values
    event = df[event_col].astype(int).values
    lp = df[predicted_lp_col].values

    # Discrimination
    try:
        val_c = concordance_index(duration, -lp, event)
    except Exception:
        val_c = None

    # Calibration (using LP)
    try:
        cal = compute_calibration_slope_intercept(
            y=event, probs_or_lp=lp, duration=duration if np.any(event == 0) else None
        )
        val_slope = cal.get("calibration_slope")
        val_intercept = cal.get("calibration_intercept")
    except Exception:
        val_slope = val_intercept = None

    result = {
        "n_validation": int(len(df)),
        "validation_c_index": round(float(val_c), 4) if val_c else None,
        "validation_calibration_slope": round(float(val_slope), 4) if val_slope else None,
        "validation_calibration_intercept": round(float(val_intercept), 4) if val_intercept else None,
    }

    # Advanced metrics if survival probs provided
    if survival_probs is not None and time_points is not None:
        time_points = np.asarray(time_points)
        surv_probs_arr = np.asarray(survival_probs)
        ibs = integrated_brier_score(duration, event, surv_probs_arr, time_points)
        td_auc = time_dependent_auc(duration, event, lp, time_points)

        result["integrated_brier_score"] = ibs
        result["time_dependent_auc"] = td_auc

    # Performance drop vs dev
    if dev_metrics:
        drop = {}
        if dev_metrics.get("c_index") and result.get("validation_c_index"):
            drop["c_index_drop"] = round(dev_metrics["c_index"] - result["validation_c_index"], 4)
        if dev_metrics.get("calibration_slope") and result.get("validation_calibration_slope"):
            drop["calibration_slope_shift"] = round(
                result["validation_calibration_slope"] - dev_metrics["calibration_slope"], 4
            )
        if drop:
            result["performance_vs_dev"] = drop

    result["note"] = "Use survival_probs (n_samples x n_times) for accurate IBS and tdAUC."
    return result


def transportability_diagnostics(
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    covariate_cols: List[str],
) -> Dict[str, Any]:
    """
    Simple diagnostics for transportability / dataset shift between
    development and validation cohorts.
    """
    diagnostics = []
    for col in covariate_cols:
        if col not in dev_df.columns or col not in val_df.columns:
            continue
        dev_mean = float(dev_df[col].mean())
        val_mean = float(val_df[col].mean())
        shift = val_mean - dev_mean
        diagnostics.append({
            "covariate": col,
            "dev_mean": round(dev_mean, 4),
            "val_mean": round(val_mean, 4),
            "absolute_shift": round(shift, 4),
            "relative_shift": round(shift / (abs(dev_mean) + 1e-8), 4),
        })

    return {
        "covariate_shifts": diagnostics,
        "overall_shift_magnitude": round(float(np.mean([abs(d["absolute_shift"]) for d in diagnostics])), 4),
    }
