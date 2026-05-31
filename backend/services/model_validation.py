"""
Model Validation & Calibration Module (Phase 4)

Provides tools for proper internal validation of prediction models,
with focus on:
- Bootstrapped performance metrics (AUC, Brier, calibration slope/intercept)
- Optimism-corrected estimates
- Calibration assessment

This module is designed to support mid-to-advanced biostatistical practice.
"""

from __future__ import annotations

from typing import Dict, Any, List, Literal, Optional
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_calibration_slope_intercept(
    y: np.ndarray,
    probs: Optional[np.ndarray] = None,
    probs_or_lp: Optional[np.ndarray] = None,
    duration: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute calibration slope and intercept by regressing observed outcomes
    on predicted probabilities (logistic calibration).
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y).ravel()
    del duration  # accepted for survival-call compatibility; binary calibration uses y/probs.
    probs = probs if probs is not None else probs_or_lp
    if probs is None:
        raise ValueError("Provide probs or probs_or_lp")
    probs = np.asarray(probs).ravel()
    if np.nanmin(probs) < 0 or np.nanmax(probs) > 1:
        probs = 1.0 / (1.0 + np.exp(-probs))
    probs = np.clip(probs, 1e-6, 1 - 1e-6)

    # Use logit of probabilities as predictor
    logits = np.log(probs / (1 - probs + 1e-12))

    X = logits.reshape(-1, 1)
    cal_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    cal_model.fit(X, y)

    intercept = float(cal_model.intercept_[0])
    slope = float(cal_model.coef_[0][0])

    return {
        "calibration_intercept": round(intercept, 4),
        "calibration_slope": round(slope, 4),
    }


def flexible_calibration_curve(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    duration: Optional[np.ndarray] = None,
    time_horizon: Optional[float] = None,
    n_knots: int = 4,
    n_grid: int = 50,
) -> Dict[str, Any]:
    """
    Flexible calibration using restricted-cubic-spline-style basis over logit(p).

    For survival predictions, pass duration + time_horizon; the observed endpoint
    becomes event by horizon. This is a pragmatic calibration curve for API use.
    """
    import statsmodels.api as sm
    from patsy import dmatrix

    y = np.asarray(y).ravel().astype(float)
    probs = np.clip(np.asarray(probs).ravel().astype(float), 1e-6, 1 - 1e-6)
    if duration is not None and time_horizon is not None:
        duration = np.asarray(duration).ravel().astype(float)
        y = ((duration <= float(time_horizon)) & (y == 1)).astype(float)
    if len(y) != len(probs) or len(y) < 20 or len(np.unique(y)) < 2:
        return {"available": False, "reason": "Need at least 20 rows with both outcome classes."}

    lp = np.log(probs / (1 - probs))
    df = pd.DataFrame({"y": y, "lp": lp})
    spline_df = max(3, int(n_knots))
    try:
        X = dmatrix(f"cr(lp, df={spline_df})", df, return_type="dataframe")
        model = sm.GLM(df["y"], X, family=sm.families.Binomial()).fit()
        grid_p = np.linspace(max(0.001, float(np.percentile(probs, 1))), min(0.999, float(np.percentile(probs, 99))), n_grid)
        grid_lp = np.log(grid_p / (1 - grid_p))
        Xg = dmatrix(f"cr(lp, df={spline_df})", pd.DataFrame({"lp": grid_lp}), return_type="dataframe")
        observed = np.clip(model.predict(Xg), 0, 1)
        ici = float(np.mean(np.abs(model.predict(X) - probs)))
        return {
            "available": True,
            "method": "restricted_cubic_spline_logistic_calibration",
            "n": int(len(y)),
            "ici": round(ici, 5),
            "curve": [
                {"predicted": round(float(p), 5), "observed": round(float(o), 5)}
                for p, o in zip(grid_p, observed)
            ],
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def competing_risks_calibration(
    df: pd.DataFrame,
    duration_col: str,
    status_col: str,
    predicted_cif_col: str,
    *,
    event_code: int = 1,
    time_horizon: Optional[float] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calibration table for predicted CIF against observed cause-specific event by horizon."""
    if time_horizon is None:
        time_horizon = float(np.percentile(pd.to_numeric(df[duration_col], errors="coerce").dropna(), 75))
    work = df[[duration_col, status_col, predicted_cif_col]].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work[status_col] = pd.to_numeric(work[status_col], errors="coerce")
    work[predicted_cif_col] = pd.to_numeric(work[predicted_cif_col], errors="coerce")
    work = work.dropna()
    if len(work) < 20:
        return {"available": False, "reason": "Need at least 20 complete rows."}
    pred = np.clip(work[predicted_cif_col].to_numpy(dtype=float), 0, 1)
    obs = ((work[duration_col].to_numpy(dtype=float) <= time_horizon) & (work[status_col].to_numpy(dtype=int) == event_code)).astype(float)
    try:
        bins = pd.qcut(pred, q=min(n_bins, len(np.unique(pred))), duplicates="drop")
    except Exception:
        bins = pd.cut(pred, bins=min(n_bins, max(2, len(np.unique(pred)))), duplicates="drop")
    frame = pd.DataFrame({"pred": pred, "obs": obs, "bin": bins})
    rows = []
    for _, g in frame.groupby("bin", observed=False):
        rows.append({
            "n": int(len(g)),
            "mean_predicted_cif": round(float(g["pred"].mean()), 5),
            "observed_cumulative_incidence": round(float(g["obs"].mean()), 5),
            "absolute_error": round(float(abs(g["obs"].mean() - g["pred"].mean())), 5),
        })
    return {
        "available": True,
        "event_code": int(event_code),
        "time_horizon": round(float(time_horizon), 4),
        "bins": rows,
        "ici": round(float(np.mean([r["absolute_error"] for r in rows])), 5) if rows else None,
        "note": "Observed CIF is approximated as cause-specific event by horizon; use Aalen-Johansen when full competing-risk curves are available.",
    }


def compute_cox_calibration_slope(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    linear_predictor: np.ndarray
) -> Dict[str, float]:
    """
    For Cox models: regress the original data on the linear predictor
    from the original model. The coefficient on the LP is the calibration slope.
    """
    from lifelines import CoxPHFitter

    work = df[[duration_col, event_col]].copy()
    work["lp"] = linear_predictor

    try:
        cph = CoxPHFitter()
        cph.fit(work, duration_col=duration_col, event_col=event_col)
        slope = float(cph.params_["lp"])
        return {
            "calibration_slope": round(slope, 4),
        }
    except Exception:
        return {"calibration_slope": None}


def bootstrap_performance(
    y: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 500,
    metrics: List[str] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap performance metrics with 95% CI.

    Supported metrics: 'auc', 'brier', 'calibration_slope', 'calibration_intercept'
    """
    if metrics is None:
        metrics = ['auc', 'brier', 'calibration_slope', 'calibration_intercept']

    y = np.asarray(y).ravel()
    probs = np.asarray(probs).ravel()
    n = len(y)

    rng = np.random.default_rng(random_state)
    results = {m: [] for m in metrics}

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb = y[idx]
        pb = probs[idx]

        if 'auc' in metrics:
            try:
                from sklearn.metrics import roc_auc_score
                results['auc'].append(roc_auc_score(yb, pb))
            except:
                results['auc'].append(np.nan)

        if 'brier' in metrics:
            brier = np.mean((pb - yb) ** 2)
            results['brier'].append(brier)

        if 'calibration_slope' in metrics or 'calibration_intercept' in metrics:
            try:
                cal = compute_calibration_slope_intercept(yb, pb)
                if 'calibration_slope' in metrics:
                    results['calibration_slope'].append(cal['calibration_slope'])
                if 'calibration_intercept' in metrics:
                    results['calibration_intercept'].append(cal['calibration_intercept'])
            except:
                if 'calibration_slope' in metrics:
                    results['calibration_slope'].append(np.nan)
                if 'calibration_intercept' in metrics:
                    results['calibration_intercept'].append(np.nan)

    summary = {}
    for m in metrics:
        vals = np.array(results[m])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 10:
            summary[m] = {
                "mean": round(float(np.mean(vals)), 4),
                "ci_low": round(float(np.percentile(vals, 2.5)), 4),
                "ci_high": round(float(np.percentile(vals, 97.5)), 4),
            }
        else:
            summary[m] = {"mean": None, "ci_low": None, "ci_high": None}

    return summary


def optimism_corrected_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Very practical optimism correction using bootstrap (similar to Harrell's approach).

    Returns apparent performance, optimism, and optimism-corrected estimates
    for AUC and calibration slope.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).ravel()
    probs = np.asarray(probs).ravel()

    # Apparent performance
    apparent_auc = roc_auc_score(y, probs)
    cal = compute_calibration_slope_intercept(y, probs)
    apparent_slope = cal['calibration_slope']

    rng = np.random.default_rng(random_state)
    n = len(y)

    optimism_auc = []
    optimism_slope = []

    for _ in range(n_boot):
        # Bootstrap sample
        idx = rng.choice(n, n, replace=True)
        yb = y[idx]
        pb = probs[idx]

        # Refit a simple logistic on bootstrap sample for "model"
        # For simplicity we use the same probabilities (common practical approximation)
        # More rigorous would be to refit the original model on bootstrap data.

        # Performance on bootstrap sample (apparent in bootstrap)
        try:
            boot_auc = roc_auc_score(yb, pb)
        except:
            continue

        # Performance on original sample using bootstrap model predictions
        # Here we approximate by using the bootstrap probabilities on original data
        orig_auc = roc_auc_score(y, pb)

        optimism_auc.append(boot_auc - orig_auc)

        # Calibration slope optimism
        try:
            boot_cal = compute_calibration_slope_intercept(yb, pb)
            orig_cal = compute_calibration_slope_intercept(y, pb)
            optimism_slope.append(boot_cal['calibration_slope'] - orig_cal['calibration_slope'])
        except:
            continue

    opt_auc = float(np.mean(optimism_auc)) if optimism_auc else 0.0
    opt_slope = float(np.mean(optimism_slope)) if optimism_slope else 0.0

    return {
        "apparent_auc": round(apparent_auc, 4),
        "optimism_auc": round(opt_auc, 4),
        "optimism_corrected_auc": round(apparent_auc - opt_auc, 4),
        "apparent_calibration_slope": round(apparent_slope, 4),
        "optimism_calibration_slope": round(opt_slope, 4),
        "optimism_corrected_calibration_slope": round(apparent_slope - opt_slope, 4),
        "n_boot": n_boot,
    }


def add_validation_to_result(
    result: dict, 
    y: np.ndarray, 
    probs: np.ndarray,
    include_optimism: bool = True,
    n_boot: int = 300
) -> dict:
    """Convenience function to attach validation metrics to an existing result dict."""
    perf = bootstrap_performance(y, probs, n_boot=n_boot)
    result["bootstrap_performance"] = perf

    if include_optimism:
        opt = optimism_corrected_metrics(y, probs, n_boot=min(n_boot, 200))
        result["optimism_correction"] = opt

    return result
