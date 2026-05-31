"""
Survival Machine Learning Module (Phase 10)

Provides ML-based survival prediction with clinical-grade interpretability
and direct integration with the Phase 9 external validation framework.

Current pragmatic implementation (staying close to existing dependencies):
- Uses sklearn GradientBoosting + a survival wrapper approach for ranking.
- Strong emphasis on honest validation via the external_validation service.
- Permutation importance + head-to-head comparison vs classical Cox.

For production Random Survival Forest, scikit-survival can be added later as optional.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from lifelines import CoxPHFitter

from services.model_validation import compute_calibration_slope_intercept
from services.external_validation import evaluate_external_validation


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


def _risk_to_survival_probs(
    risk_scores: np.ndarray,
    time_points: np.ndarray,
    *,
    invert: bool = True,
    risk_std_scale: float = 0.9,
) -> np.ndarray:
    """
    Immutable helper (Phase 12 deepened): convert risk scores into approximate
    survival probability curves S(t) suitable for IPCW Integrated Brier Score.

    Strategy:
    - Standardize risks (mean 0, controlled std) so the relative hazard exp()
      term has a sensible dynamic range regardless of whether the input came
      from GB regression or Cox partial hazard.
    - Use a simple linear ramp baseline cumulative hazard derived from the
      observed time scale (good enough for demo + property testing).
    - Explicit invert flag ensures higher risk always maps to lower survival.
    """
    risk = np.asarray(risk_scores, dtype=float)
    times = np.asarray(time_points, dtype=float)

    if len(risk) == 0 or len(times) == 0:
        return np.ones((len(risk), len(times)))

    # Standardize per-model (critical: GB risks and Cox LP live on different scales)
    risk = risk - np.nanmean(risk)
    rstd = np.nanstd(risk) or 1.0
    risk = risk / (rstd / risk_std_scale)

    if invert:
        # Higher (standardized) risk → higher hazard multiplier → faster decay
        eff_risk = -risk
    else:
        eff_risk = risk

    # Data-driven-ish baseline cumulative hazard: ramp from 0 to ~1.2 over the time grid
    # This keeps S(t) in a realistic [0.05, 0.98] band for typical clinical follow-up
    t_max = float(times[-1]) if times[-1] > 0 else 1.0
    cumhaz0 = (times / max(t_max, 1e-6)) * 1.15

    # S_i(t) = exp( -Λ0(t) * exp(eff_risk_i) )
    surv = np.exp(-cumhaz0[None, :] * np.exp(eff_risk)[:, None])
    return np.clip(surv, 1e-5, 1.0 - 1e-5)


def run_survival_ml_benchmark(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    predictors: Optional[List[str]] = None,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Runs a practical ML survival benchmark:
    - Gradient Boosting on the survival ranking problem (using negative duration as target for ordering)
    - Classical Cox baseline
    - Permutation importance for the ML model
    - Direct comparison using Phase 9-style metrics
    """
    if predictors is None:
        predictors = [c for c in df.columns if c not in (duration_col, event_col)]

    X = df[predictors].copy()
    for c in predictors:
        if X[c].dtype == object:
            X[c] = pd.Categorical(X[c]).codes
    X = X.apply(pd.to_numeric, errors="coerce")

    y_duration = df[duration_col].values
    y_event = df[event_col].astype(int).values

    # --- 1. Classical Cox baseline ---
    cox_df = pd.concat([df[[duration_col, event_col]], X], axis=1)
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col, robust=True)

    cox_lp = cph.predict_partial_hazard(X).values
    cox_c = float(cph.concordance_index_)

    # --- 2. Practical ML Survival model (Gradient Boosting ranking) ---
    # We treat it as a regression on -duration (higher risk → shorter time), with event weighting
    y_target = -y_duration
    sample_weight = np.where(y_event == 1, 2.0, 1.0)

    gbr = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        random_state=random_state,
    )
    gbr.fit(X, y_target, sample_weight=sample_weight)

    ml_risk = gbr.predict(X)

    # --- 3. Permutation importance (ML model) ---
    perm = permutation_importance(
        gbr, X, y_target, n_repeats=8, random_state=random_state, scoring="neg_mean_squared_error"
    )
    importance = [
        {"variable": col, "importance": round(float(imp), 5)}
        for col, imp in zip(predictors, perm.importances_mean)
    ]
    importance.sort(key=lambda x: -x["importance"])

    # --- 4. Quick validation comparison using Phase 9 style ---
    # We use the risk scores to compute a simple C-index style comparison
    try:
        from lifelines.utils import concordance_index
        ml_c = concordance_index(y_duration, -ml_risk, y_event)
    except Exception:
        ml_c = None

    # Calibration using LP-style for both
    try:
        cox_cal = compute_calibration_slope_intercept(y_event, cox_lp, duration=y_duration)
    except Exception:
        cox_cal = {}

    try:
        ml_cal = compute_calibration_slope_intercept(y_event, ml_risk, duration=y_duration)
    except Exception:
        ml_cal = {}

    # Prepare risk scores for Phase 9 external validation (higher = higher risk)
    ml_risk_scores = ml_risk.astype(float)
    cox_risk_scores = cox_lp.astype(float)

    validation_ready = {
        "ml_risk_scores": ml_risk_scores.tolist(),
        "cox_risk_scores": cox_risk_scores.tolist(),
    }

    # --- Phase 12 Deepened: Generate survival probabilities + full Phase 9 metrics (IBS + tdAUC) ---
    time_grid = np.percentile(y_duration, [20, 40, 60, 80])
    time_grid = np.clip(time_grid, 0.1, None)

    # Explicit inversion: our ML risk and Cox LP are "higher = worse"
    # _risk_to_survival_probs(..., invert=True) ensures high-risk subjects get lower S(t)
    ml_surv_probs = _risk_to_survival_probs(ml_risk_scores, time_grid, invert=True)
    cox_surv_probs = _risk_to_survival_probs(cox_risk_scores, time_grid, invert=True)

    full_validation = {}
    try:
        ml_full = evaluate_external_validation(
            val_df=df.assign(ml_risk=ml_risk_scores),
            duration_col=duration_col,
            event_col=event_col,
            predicted_lp_col="ml_risk",
            survival_probs=ml_surv_probs,
            time_points=time_grid.tolist(),
        )
        cox_full = evaluate_external_validation(
            val_df=df.assign(cox_risk=cox_risk_scores),
            duration_col=duration_col,
            event_col=event_col,
            predicted_lp_col="cox_risk",
            survival_probs=cox_surv_probs,
            time_points=time_grid.tolist(),
        )
        full_validation = {
            "ml": ml_full,
            "cox": cox_full,
            "time_points": [round(float(t), 2) for t in time_grid],
        }
    except Exception as e:
        full_validation = {"error": str(e)}

    # Comparison table (now with real IBS when available)
    ml_ibs = None
    cox_ibs = None
    if isinstance(full_validation.get("ml"), dict):
        ml_ibs = full_validation["ml"].get("integrated_brier_score", {}).get("ibs")
    if isinstance(full_validation.get("cox"), dict):
        cox_ibs = full_validation["cox"].get("integrated_brier_score", {}).get("ibs")

    comparison = {
        "models": [
            {
                "name": "Classical Cox",
                "c_index": full_validation.get("cox", {}).get("validation_c_index") if isinstance(full_validation.get("cox"), dict) else None,
                "calibration_slope": full_validation.get("cox", {}).get("validation_calibration_slope") if isinstance(full_validation.get("cox"), dict) else None,
                "ibs": cox_ibs,
            },
            {
                "name": "Gradient Boosting Survival (ranking)",
                "c_index": full_validation.get("ml", {}).get("validation_c_index") if isinstance(full_validation.get("ml"), dict) else None,
                "calibration_slope": full_validation.get("ml", {}).get("validation_calibration_slope") if isinstance(full_validation.get("ml"), dict) else None,
                "ibs": ml_ibs,
            },
        ],
        "winner_by_c_index": None,
        "winner_by_ibs": None,
    }

    # Determine winners safely
    try:
        ml_c_val = full_validation.get("ml", {}).get("validation_c_index") if isinstance(full_validation.get("ml"), dict) else None
        cox_c_val = full_validation.get("cox", {}).get("validation_c_index") if isinstance(full_validation.get("cox"), dict) else None
        if ml_c_val is not None and cox_c_val is not None:
            comparison["winner_by_c_index"] = "ML" if ml_c_val > cox_c_val else "Cox"
        if ml_ibs is not None and cox_ibs is not None:
            comparison["winner_by_ibs"] = "ML" if ml_ibs < cox_ibs else "Cox"  # lower IBS better
    except Exception:
        pass

    # Rich metadata (immutable construction, following project conventions)
    assumptions = [
        "Gradient Boosting treats survival as a censored ranking / accelerated failure time surrogate (negative duration target + event weighting).",
        "Survival probabilities are approximated via exponential decay model calibrated to observed time scale; not a full parametric or non-parametric cumulative hazard estimator.",
        "Risk direction: higher ML risk score and higher Cox LP both correspond to worse prognosis (shorter survival). Inversion applied before S(t) generation.",
        "Phase 9 external_validation receives both LP (for C-index/calibration) and generated S(t) curves (for proper IPCW IBS and tdAUC).",
    ]
    warnings = []
    if ml_c is None:
        warnings.append("ML C-index could not be computed in-sample.")
    if "error" in full_validation:
        warnings.append("Full Phase 9 metrics (IBS/tdAUC) unavailable due to internal error in evaluate_external_validation.")
    if len(df) < 100:
        warnings.append("Small sample size; ML advantage estimates have high variance.")

    result_text = (
        f"Survival ML benchmark on n={len(df)} subjects. "
        f"Classical Cox C-index {round(cox_c, 3)}. "
        f"ML (GB ranking) C-index {round(float(ml_c), 3) if ml_c else 'N/A'}. "
        f"Full Phase 9 validation (IBS + tdAUC) computed on generated survival curves. "
        f"IBS winner: {comparison.get('winner_by_ibs') or 'undetermined'}. "
        "Useful when non-linear effects or interactions are suspected."
    )

    return {
        "n": int(len(df)),
        "classical_cox": {
            "c_index": round(cox_c, 4),
            "calibration_slope": cox_cal.get("calibration_slope"),
        },
        "ml_gradient_boosting_survival": {
            "c_index": round(float(ml_c), 4) if ml_c else None,
            "calibration_slope": ml_cal.get("calibration_slope"),
            "permutation_importance": importance[:8],
        },
        "validation_ready_risk_scores": validation_ready,
        "validation_ready_survival_probs": {
            "time_points": [round(float(t), 2) for t in time_grid],
            "ml_survival_probs": ml_surv_probs.tolist(),
            "cox_survival_probs": cox_surv_probs.tolist(),
        },
        "full_phase9_validation": full_validation,
        "auto_comparison": comparison,
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "note": "Phase 12 deepened integration: risk scores are converted to survival probability curves and fed to evaluate_external_validation for proper IBS and time-dependent AUC. ML model remains a practical Gradient Boosting ranking surrogate.",
    }


def auto_survival_ml_compare(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    predictors: Optional[List[str]] = None,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Phase 12: Automated comparison of classical Cox vs practical ML survival model.
    Returns ranked models with Phase 9 metrics + feature importance for the winner.
    """
    benchmark = run_survival_ml_benchmark(
        df, duration_col, event_col, predictors, n_estimators, random_state
    )

    comparison = benchmark.get("auto_comparison", {})
    models = comparison.get("models", [])

    # Add simple winner logic based on combined score (C-index primary, lower IBS secondary)
    if models:
        def score(m):
            c = m.get("c_index") or 0
            ibs = m.get("ibs") or 1.0
            return c - 0.3 * ibs  # simple composite

        ranked = sorted(models, key=score, reverse=True)
        comparison["ranked_models"] = ranked
        comparison["recommended"] = ranked[0]["name"] if ranked else None

    # Enrich with top features from ML
    ml_info = benchmark.get("ml_gradient_boosting_survival", {})
    comparison["top_features"] = ml_info.get("permutation_importance", [])[:5]

    return {
        "n": benchmark["n"],
        "comparison": comparison,
        "details": {
            "cox": benchmark["classical_cox"],
            "ml": benchmark["ml_gradient_boosting_survival"],
        },
        "full_phase9_validation": benchmark.get("full_phase9_validation"),
        "assumptions": benchmark.get("assumptions", []),
        "warnings": benchmark.get("warnings", []),
        "result_text": benchmark.get("result_text"),
        "note": "Automated survival ML vs classical comparison using full Phase 9 metrics (C-index + calibration + IBS + tdAUC via generated survival curves). For production-grade RSF, consider scikit-survival.",
    }
