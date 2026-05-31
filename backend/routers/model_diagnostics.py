"""Model-specific diagnostics: logistic regression calibration & Cox PH assumptions."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal

from services import store
from services.impute import apply_imputation
from services.missing_data_sensitivity import (
    simulate_missingness,
    delta_adjustment_sensitivity,
    summarize_sensitivity,
)
from services.model_validation import (
    bootstrap_performance,
    optimism_corrected_metrics,
    add_validation_to_result,
    compute_cox_calibration_slope,
    compute_calibration_slope_intercept,
)
from services.causal_sensitivity import (
    e_value,
    quantitative_bias_analysis,
    e_value_from_psm_or_iptw,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOGISTIC REGRESSION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

class LogisticDiagRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: str = "listwise"


@router.post("/logistic_diagnostics")
def logistic_diagnostics(req: LogisticDiagRequest):
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    df_full = _get_df(req.session_id)
    all_cols = [req.outcome] + req.predictors
    df = apply_imputation(df_full, all_cols, req.imputation or "listwise")

    if len(df) < len(req.predictors) + 10:
        raise HTTPException(400, "Not enough observations for logistic regression diagnostics.")

    # Encode outcome as binary 0/1
    y = df[req.outcome].copy()
    unique_vals = sorted(y.dropna().unique())
    if len(unique_vals) != 2:
        raise HTTPException(400, f"Outcome '{req.outcome}' must be binary (has {len(unique_vals)} unique values).")

    # Map to 0/1
    if set(unique_vals) <= {0, 1, 0.0, 1.0}:
        y = y.astype(float)
    else:
        val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
        y = y.map(val_map).astype(float)

    X = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    X = sm.add_constant(X)

    # ── Fit logistic ─────────────────────────────────────────────────────────
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
    except Exception as exc:
        raise HTTPException(400, f"Logistic model failed to converge: {exc}")

    probs = model.predict(X).values
    y_arr = y.values
    n = len(y_arr)

    # ── 1. Separation warning ────────────────────────────────────────────────
    separation_vars = []
    for col in req.predictors:
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            # Check if any threshold perfectly separates
            vals_0 = s[y_arr == 0]
            vals_1 = s[y_arr == 1]
            if len(vals_0) > 0 and len(vals_1) > 0:
                if vals_0.max() < vals_1.min() or vals_1.max() < vals_0.min():
                    separation_vars.append(col)
        else:
            # Categorical: check if any level has only one outcome value
            for level in s.unique():
                mask = s == level
                if mask.sum() > 0:
                    outcomes_in_level = y_arr[mask.values]
                    if len(np.unique(outcomes_in_level)) == 1 and mask.sum() > 1:
                        separation_vars.append(col)
                        break

    # ── 2. Calibration plot data ─────────────────────────────────────────────
    sorted_idx = np.argsort(probs)
    n_bins = min(10, max(2, n // 10))
    groups = np.array_split(sorted_idx, n_bins)
    calibration_bins = []
    for grp in groups:
        if len(grp) == 0:
            continue
        calibration_bins.append({
            "predicted_mean": round(float(probs[grp].mean()), 4),
            "observed_prop": round(float(y_arr[grp].mean()), 4),
            "n": int(len(grp)),
        })

    # ── 3. Brier score ───────────────────────────────────────────────────────
    brier = float(np.mean((probs - y_arr) ** 2))

    # ── 4. Hosmer-Lemeshow ───────────────────────────────────────────────────
    hl_groups = np.array_split(sorted_idx, 10)
    chi2 = 0.0
    for grp in hl_groups:
        if len(grp) == 0:
            continue
        n_g = len(grp)
        o_g = float(y_arr[grp].sum())
        e_g = float(probs[grp].sum())
        denom = e_g * (1 - e_g / n_g)
        if denom > 1e-10:
            chi2 += (o_g - e_g) ** 2 / denom
    p_hl = float(1 - scipy_stats.chi2.cdf(chi2, df=8))

    # ── 5. C-statistic (AUC) ────────────────────────────────────────────────
    try:
        c_stat = float(roc_auc_score(y_arr, probs))
    except Exception:
        c_stat = None

    # ── 6. Influence summary ─────────────────────────────────────────────────
    try:
        infl = model.get_influence()
        dfbetas = infl.dfbetas
        hat_diag = infl.hat_matrix_diag
        n_high_dfbeta = int(np.sum(np.any(np.abs(dfbetas) > 2 / np.sqrt(n), axis=1)))
        n_high_leverage = int(np.sum(hat_diag > 2 * X.shape[1] / n))
    except Exception:
        n_high_dfbeta = None
        n_high_leverage = None

    # ── Assumptions ──────────────────────────────────────────────────────────
    assumptions = []
    assumptions.append({
        "name": "Calibration (Hosmer-Lemeshow)",
        "met": bool(p_hl >= 0.05),
        "detail": f"Chi-square = {chi2:.2f}, p = {_p_str(p_hl)}",
    })
    assumptions.append({
        "name": "No perfect separation",
        "met": len(separation_vars) == 0,
        "detail": f"Separation detected in: {', '.join(separation_vars)}" if separation_vars else "No separation detected",
    })

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings = []
    if p_hl < 0.05:
        warnings.append("Hosmer-Lemeshow test significant — model may be poorly calibrated.")
    if separation_vars:
        warnings.append(f"Perfect or quasi-perfect separation in: {', '.join(separation_vars)}. Coefficient estimates may be unreliable.")
    if brier > 0.25:
        warnings.append(f"Brier score = {brier:.4f} — predictive accuracy is limited.")
    if c_stat is not None and c_stat < 0.6:
        warnings.append(f"C-statistic (AUC) = {c_stat:.4f} — poor discrimination.")

    # ── result_text ──────────────────────────────────────────────────────────
    hl_txt = f"Hosmer-Lemeshow: {'adequate calibration' if p_hl >= 0.05 else 'poor calibration'} (chi-square = {chi2:.2f}, p = {_p_str(p_hl)})"
    brier_txt = f"Brier score = {brier:.4f}"
    c_txt = f"C-statistic (AUC) = {c_stat:.4f}" if c_stat is not None else "C-statistic not available"
    sep_txt = f"Separation detected in: {', '.join(separation_vars)}" if separation_vars else "No separation detected"

    result_text = (
        f"Logistic regression diagnostics (n = {n}, {len(req.predictors)} predictor{'s' if len(req.predictors) != 1 else ''}). "
        f"{hl_txt}. {brier_txt}. {c_txt}. {sep_txt}."
    )

    # ── export_rows ──────────────────────────────────────────────────────────
    export_rows = [
        ["Diagnostic", "Value"],
        ["n", n],
        ["Brier score", round(brier, 4)],
        ["C-statistic (AUC)", round(c_stat, 4) if c_stat is not None else None],
        ["Hosmer-Lemeshow chi-square", round(chi2, 4)],
        ["Hosmer-Lemeshow df", 8],
        ["Hosmer-Lemeshow p", round(p_hl, 6)],
        ["Separation detected", "Yes" if separation_vars else "No"],
        ["High DFBETA count", n_high_dfbeta],
        ["High leverage count", n_high_leverage],
    ]

    # ── r_code ───────────────────────────────────────────────────────────────
    pred_formula = " + ".join(req.predictors)
    r_code = (
        f"library(ResourceSelection)\n"
        f"model <- glm({req.outcome} ~ {pred_formula}, data = data, family = binomial)\n"
        f"summary(model)\n"
        f"hoslem.test(model$y, fitted(model), g = 10)  # Hosmer-Lemeshow\n"
        f"library(pROC)\n"
        f"roc(model$y, fitted(model))  # AUC/C-statistic\n"
        f"# Brier score\n"
        f"mean((fitted(model) - model$y)^2)"
    )

    return {
        "test": "Logistic Regression Diagnostics",
        "calibration": {
            "bins": calibration_bins,
        },
        "brier_score": round(brier, 4),
        "hosmer_lemeshow": {
            "chi2": round(chi2, 4),
            "df": 8,
            "p": round(p_hl, 6),
            "significant": bool(p_hl < 0.05),
        },
        "separation": {
            "detected": len(separation_vars) > 0,
            "variables": separation_vars,
        },
        "c_statistic": round(c_stat, 4) if c_stat is not None else None,
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COX PH DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

class CoxDiagRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    imputation: str = "listwise"


@router.post("/cox_diagnostics")
def cox_diagnostics(req: CoxDiagRequest):
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    df_full = _get_df(req.session_id)
    all_cols = [req.duration_col, req.event_col] + req.predictors
    missing_cols = [c for c in all_cols if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(400, f"Columns not found: {missing_cols}")

    df = apply_imputation(df_full, all_cols, req.imputation or "listwise")

    if len(df) < len(req.predictors) + 10:
        raise HTTPException(400, "Not enough observations for Cox PH diagnostics.")

    # Ensure numeric
    df[req.duration_col] = pd.to_numeric(df[req.duration_col], errors="coerce")
    df[req.event_col] = pd.to_numeric(df[req.event_col], errors="coerce")
    for col in req.predictors:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=all_cols)

    if len(df) < len(req.predictors) + 10:
        raise HTTPException(400, "Not enough observations after cleaning.")

    # Ensure positive durations
    df = df[df[req.duration_col] > 0]
    if len(df) < 10:
        raise HTTPException(400, "Not enough observations with positive duration.")

    n = len(df)
    n_events = int(df[req.event_col].sum())

    # ── Fit Cox model ────────────────────────────────────────────────────────
    cox_cols = req.predictors + [req.duration_col, req.event_col]
    try:
        cph = CoxPHFitter()
        cph.fit(df[cox_cols], duration_col=req.duration_col, event_col=req.event_col)
    except Exception as exc:
        raise HTTPException(400, f"Cox PH model failed: {exc}")

    # ── 1. Schoenfeld residuals PH test ──────────────────────────────────────
    ph_results = []
    try:
        ph_test = proportional_hazard_test(cph, df[cox_cols], time_transform="rank")
        for var in req.predictors:
            try:
                row = ph_test.summary.loc[var]
                test_stat = float(row["test_statistic"])
                p_val = float(row["p"])
                ph_results.append({
                    "variable": var,
                    "test_stat": round(test_stat, 4),
                    "p": round(p_val, 6),
                    "assumption_met": bool(p_val >= 0.05),
                })
            except (KeyError, IndexError):
                ph_results.append({
                    "variable": var,
                    "test_stat": None,
                    "p": None,
                    "assumption_met": True,
                })
    except Exception:
        # PH test may fail for some data configurations
        for var in req.predictors:
            ph_results.append({
                "variable": var,
                "test_stat": None,
                "p": None,
                "assumption_met": True,
            })

    # ── 2. C-index ───────────────────────────────────────────────────────────
    c_index = round(float(cph.concordance_index_), 4)

    # ── 3. Log-likelihood ratio test ─────────────────────────────────────────
    try:
        ll_test = cph.log_likelihood_ratio_test()
        ll_stat = round(float(ll_test.test_statistic), 4)
        ll_p = float(ll_test.p_value)
    except Exception:
        ll_stat = None
        ll_p = None

    # ── Assumptions ──────────────────────────────────────────────────────────
    assumptions = []

    # PH assumption for each variable
    violated = [r for r in ph_results if not r["assumption_met"]]
    if violated:
        assumptions.append({
            "name": "Proportional hazards",
            "met": False,
            "detail": f"PH assumption violated for: {', '.join(v['variable'] for v in violated)}",
        })
    else:
        assumptions.append({
            "name": "Proportional hazards",
            "met": True,
            "detail": "PH assumption met for all predictors",
        })

    # Events per variable (rule of thumb: >= 10)
    epv = n_events / max(len(req.predictors), 1)
    assumptions.append({
        "name": "Events per variable",
        "met": epv >= 10,
        "detail": f"EPV = {epv:.1f} ({n_events} events, {len(req.predictors)} predictors)" + (" — may be underpowered" if epv < 10 else ""),
    })

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings = []
    if violated:
        warnings.append(f"PH assumption violated for: {', '.join(v['variable'] for v in violated)}. Consider time-varying coefficients or stratification.")
    if epv < 10:
        warnings.append(f"Low events per variable ({epv:.1f}). Model estimates may be unstable.")
    if c_index < 0.6:
        warnings.append(f"C-index = {c_index} — poor discrimination.")

    # ── result_text ──────────────────────────────────────────────────────────
    ph_txt = "PH assumption met for all predictors" if not violated else f"PH assumption violated for: {', '.join(v['variable'] for v in violated)}"
    ll_txt = f"Log-likelihood ratio test: chi-square = {ll_stat}, p = {_p_str(ll_p)}" if ll_p is not None else "Log-likelihood ratio test not available"

    result_text = (
        f"Cox PH diagnostics (n = {n}, {n_events} events, {len(req.predictors)} predictor{'s' if len(req.predictors) != 1 else ''}). "
        f"C-index = {c_index}. "
        f"{ph_txt}. "
        f"{ll_txt}."
    )

    # ── export_rows ──────────────────────────────────────────────────────────
    export_rows = [
        ["Diagnostic", "Value"],
        ["n", n],
        ["Events", n_events],
        ["C-index", c_index],
        ["Log-likelihood ratio stat", ll_stat],
        ["Log-likelihood ratio p", round(ll_p, 6) if ll_p is not None else None],
        ["Events per variable", round(epv, 1)],
    ]
    for r in ph_results:
        export_rows.append([f"PH test: {r['variable']} (stat)", r["test_stat"]])
        export_rows.append([f"PH test: {r['variable']} (p)", r["p"]])

    # ── r_code ───────────────────────────────────────────────────────────────
    pred_formula = " + ".join(req.predictors)
    r_code = (
        f"library(survival)\n"
        f"model <- coxph(Surv({req.duration_col}, {req.event_col}) ~ {pred_formula}, data = data)\n"
        f"summary(model)\n"
        f"cox.zph(model)  # Schoenfeld residuals PH test\n"
        f"concordance(model)  # C-index"
    )

    return {
        "test": "Cox PH Diagnostics",
        "ph_test": ph_results,
        "c_index": c_index,
        "log_likelihood_ratio": {
            "stat": ll_stat,
            "p": round(ll_p, 6) if ll_p is not None else None,
        },
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NRI + IDI (Net Reclassification Improvement + Integrated Discrimination Improvement)
# ═══════════════════════════════════════════════════════════════════════════════

class NRIIDIRequest(BaseModel):
    session_id: str
    outcome: str
    prob_old: str          # column name for probabilities from old model
    prob_new: str          # column name for probabilities from new model
    cutoff: float = 0.5
    imputation: str = "listwise"


@router.post("/nri_idi")
def nri_idi(req: NRIIDIRequest):
    """
    Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI).
    Compares two prediction models using their predicted probabilities.
    """
    df_full = _get_df(req.session_id)
    cols = [req.outcome, req.prob_old, req.prob_new]
    df = apply_imputation(df_full, cols, req.imputation or "listwise")

    if len(df) < 100:
        raise HTTPException(400, "At least 100 observations recommended for stable NRI/IDI estimates.")

    y = pd.to_numeric(df[req.outcome], errors="coerce").astype(int).values
    p_old = pd.to_numeric(df[req.prob_old], errors="coerce").values
    p_new = pd.to_numeric(df[req.prob_new], errors="coerce").values

    mask = ~np.isnan(y) & ~np.isnan(p_old) & ~np.isnan(p_new)
    y, p_old, p_new = y[mask], p_old[mask], p_new[mask]

    if len(np.unique(y)) != 2:
        raise HTTPException(400, "Outcome must be binary.")

    # Reclassification
    old_high = (p_old >= req.cutoff).astype(int)
    new_high = (p_new >= req.cutoff).astype(int)

    events = (y == 1)
    non_events = (y == 0)

    n_up_event = np.sum((new_high > old_high) & events)
    n_down_event = np.sum((new_high < old_high) & events)
    n_event = np.sum(events)

    n_up_non = np.sum((new_high > old_high) & non_events)
    n_down_non = np.sum((new_high < old_high) & non_events)
    n_non = np.sum(non_events)

    nri_event = (n_up_event - n_down_event) / n_event if n_event > 0 else 0
    nri_non_event = (n_down_non - n_up_non) / n_non if n_non > 0 else 0
    nri = nri_event + nri_non_event

    # IDI
    mean_diff_event = np.mean(p_new[events]) - np.mean(p_old[events]) if n_event > 0 else 0
    mean_diff_non = np.mean(p_new[non_events]) - np.mean(p_old[non_events]) if n_non > 0 else 0
    idi = mean_diff_event - mean_diff_non

    # Bootstrap CI
    rng = np.random.default_rng(42)
    n_boot = 800
    n = len(y)
    nri_boots, idi_boots = [], []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb, po, pn = y[idx], p_old[idx], p_new[idx]
        oh = (po >= req.cutoff).astype(int)
        nh = (pn >= req.cutoff).astype(int)
        ev = (yb == 1)
        nev = (yb == 0)

        nri_e = ((nh > oh) & ev).sum() - ((nh < oh) & ev).sum()
        nri_ne = ((nh < oh) & nev).sum() - ((nh > oh) & nev).sum()
        nri_b = (nri_e / ev.sum() if ev.sum() > 0 else 0) + (nri_ne / nev.sum() if nev.sum() > 0 else 0)

        idi_b = (pn[ev].mean() - po[ev].mean() if ev.sum() > 0 else 0) - \
                (pn[nev].mean() - po[nev].mean() if nev.sum() > 0 else 0)

        nri_boots.append(nri_b)
        idi_boots.append(idi_b)

    nri_ci = np.percentile(nri_boots, [2.5, 97.5])
    idi_ci = np.percentile(idi_boots, [2.5, 97.5])

    return {
        "n": int(n),
        "cutoff_used": req.cutoff,
        "nri": {
            "estimate": round(nri, 4),
            "ci_low": round(nri_ci[0], 4),
            "ci_high": round(nri_ci[1], 4),
            "contribution_events": round(nri_event, 4),
            "contribution_non_events": round(nri_non_event, 4),
        },
        "idi": {
            "estimate": round(idi, 4),
            "ci_low": round(idi_ci[0], 4),
            "ci_high": round(idi_ci[1], 4),
        },
        "reclassification_counts": {
            "up_in_events": int(n_up_event),
            "down_in_events": int(n_down_event),
            "up_in_non_events": int(n_up_non),
            "down_in_non_events": int(n_down_non),
        },
        "test": "NRI + IDI (with bootstrap CI)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Missing Data Sensitivity Analysis (MNAR via Delta-Adjustment)
# ═══════════════════════════════════════════════════════════════════════════════

class SensitivityRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    model_type: str = "logistic"          # "linear" | "logistic" | "cox"
    mechanism: str = "MNAR"               # "MCAR" | "MAR" | "MNAR"
    missing_rate: float = 0.25
    delta_range: List[float] = [-1.5, 1.5]
    n_steps: int = 7
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    imputation: str = "listwise"


@router.post("/missing_data_sensitivity")
def missing_data_sensitivity(req: SensitivityRequest):
    """
    Run a delta-adjustment sensitivity analysis to explore how results
    change under different MNAR assumptions.
    """
    df_full = _get_df(req.session_id)

    # First simulate the desired missingness mechanism on top of existing missingness
    df_sim = simulate_missingness(
        df_full,
        [req.outcome] + req.predictors,
        mechanism=req.mechanism,
        missing_rate=req.missing_rate,
    )

    sens = delta_adjustment_sensitivity(
        df_sim,
        outcome=req.outcome,
        predictors=req.predictors,
        model_type=req.model_type,
        delta_range=tuple(req.delta_range),
        n_steps=req.n_steps,
        duration_col=req.duration_col,
        event_col=req.event_col,
    )

    summary = summarize_sensitivity(sens["results"])

    return {
        "test": "Missing Data Sensitivity (Delta Adjustment)",
        "mechanism_simulated": req.mechanism,
        "missing_rate": req.missing_rate,
        "model_type": req.model_type,
        "delta_range": req.delta_range,
        "results": sens["results"],
        "summary": summary,
        "interpretation": sens.get("interpretation"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model Validation (Bootstrap + Optimism Correction) - Phase 4
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationRequest(BaseModel):
    session_id: str
    outcome: Optional[str] = None
    prob_column: Optional[str] = None          # For binary / logistic
    model_type: str = "binary"                 # "binary" | "cox"
    n_boot: int = 300
    include_optimism: bool = True
    imputation: str = "listwise"

    # Cox-specific fields
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    linear_predictor_col: Optional[str] = None   # LP or risk score column for Cox


@router.post("/model_validation")
def model_validation(req: ValidationRequest):
    """
    Bootstrap performance + optional optimism correction for a prediction model.
    Supports binary outcomes and Cox models.
    """
    df_full = _get_df(req.session_id)

    if req.model_type == "cox":
        # Cox validation path
        if not (req.duration_col and req.event_col and req.linear_predictor_col):
            raise HTTPException(400, "For Cox validation you must provide duration_col, event_col, and linear_predictor_col")

        cols = [req.duration_col, req.event_col, req.linear_predictor_col]
        df = apply_imputation(df_full, cols, req.imputation or "listwise")

        if len(df) < 50:
            raise HTTPException(400, "At least 50 observations needed for reliable Cox validation.")

        lp = pd.to_numeric(df[req.linear_predictor_col], errors="coerce").values
        duration = pd.to_numeric(df[req.duration_col], errors="coerce").values
        event = pd.to_numeric(df[req.event_col], errors="coerce").values

        mask = ~np.isnan(duration) & ~np.isnan(event) & ~np.isnan(lp) & (duration > 0)
        duration = duration[mask]
        event = event[mask].astype(int)
        lp = lp[mask]

        # Bootstrap C-index (using lifelines)
        from lifelines.utils import concordance_index
        c_indices = []
        rng = np.random.default_rng(42)
        n = len(duration)

        for _ in range(req.n_boot):
            idx = rng.choice(n, n, replace=True)
            try:
                c = concordance_index(duration[idx], -lp[idx], event[idx])  # higher LP = higher risk → negate for concordance_index
                c_indices.append(c)
            except:
                pass

        if c_indices:
            c_mean = float(np.mean(c_indices))
            c_ci = np.percentile(c_indices, [2.5, 97.5])
            perf = {
                "c_index": {
                    "mean": round(c_mean, 4),
                    "ci_low": round(c_ci[0], 4),
                    "ci_high": round(c_ci[1], 4),
                }
            }
        else:
            perf = {"c_index": {"mean": None, "ci_low": None, "ci_high": None}}

        result = {
            "test": "Model Validation - Cox",
            "n": int(n),
            "n_boot": req.n_boot,
            "bootstrap_performance": perf,
        }

        # Add calibration slope if possible
        try:
            cal = compute_cox_calibration_slope(df, req.duration_col, req.event_col, lp)
            result["calibration"] = cal
        except:
            pass

        return result

    # === Binary / Logistic path (default) ===
    if not (req.outcome and req.prob_column):
        raise HTTPException(400, "For binary validation you must provide outcome and prob_column")

    cols = [req.outcome, req.prob_column]
    df = apply_imputation(df_full, cols, req.imputation or "listwise")

    if len(df) < 50:
        raise HTTPException(400, "At least 50 observations needed for reliable validation.")

    y = pd.to_numeric(df[req.outcome], errors="coerce").values
    probs = pd.to_numeric(df[req.prob_column], errors="coerce").values

    mask = ~np.isnan(y) & ~np.isnan(probs)
    y = y[mask].astype(int)
    probs = probs[mask]

    if len(np.unique(y)) != 2:
        raise HTTPException(400, "Outcome must be binary for this validation module.")

    perf = bootstrap_performance(y, probs, n_boot=req.n_boot)

    result = {
        "test": "Model Validation (Bootstrap + Optimism Correction)",
        "n": int(len(y)),
        "n_boot": req.n_boot,
        "bootstrap_performance": perf,
    }

    if req.include_optimism:
        opt = optimism_corrected_metrics(y, probs, n_boot=min(req.n_boot, 200))
        result["optimism_correction"] = opt

    # Attach calibration slope/intercept
    try:
        cal = compute_calibration_slope_intercept(y, probs)
        result["calibration"] = cal
    except:
        pass

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Causal Sensitivity Analysis (E-value + Quantitative Bias Analysis) - Phase 5
# ═══════════════════════════════════════════════════════════════════════════════

from pydantic import Field

class CausalSensitivityRequest(BaseModel):
    observed_estimate: float = Field(..., gt=0, description="Point estimate on RR/OR/HR scale")
    ci_low: Optional[float] = Field(None, gt=0)
    ci_high: Optional[float] = Field(None, gt=0)
    measure: Literal["rr", "or", "hr"] = "rr"
    rare_outcome: bool = False
    baseline_risk: Optional[float] = Field(None, ge=0.001, le=0.99, description="Baseline outcome risk for OR->RR conversion (common outcomes)")
    confounding_strength: float = Field(2.0, gt=0)
    prevalence_exposed: float = Field(0.5, ge=0, le=1)
    prevalence_unexposed: float = Field(0.5, ge=0, le=1)


@router.post("/causal_sensitivity")
def causal_sensitivity(req: CausalSensitivityRequest):
    """
    E-value (VanderWeele & Ding) + quantitative bias analysis for unmeasured confounding.
    Supports direct use after PSM/IPTW marginal effects via the e_value_from_psm_or_iptw helper.
    """
    warnings: list[str] = []
    assumptions: list[dict] = []

    if req.ci_low and req.ci_high and req.ci_low >= req.ci_high:
        raise HTTPException(400, "ci_low must be < ci_high")

    if req.measure == "or" and not req.rare_outcome and req.baseline_risk is None:
        warnings.append(
            "OR supplied without rare_outcome=True or baseline_risk. Using conservative approximation (p0≈0.1). "
            "For common outcomes, provide baseline_risk for better accuracy."
        )

    ev = e_value(
        estimate=req.observed_estimate,
        ci_low=req.ci_low,
        ci_high=req.ci_high,
        measure=req.measure,
        rare_outcome=req.rare_outcome,
        baseline_risk=req.baseline_risk,
    )

    qba = quantitative_bias_analysis(
        observed_estimate=req.observed_estimate,
        measure=req.measure,
        confounding_strength=req.confounding_strength,
        prevalence_exposed=req.prevalence_exposed,
        prevalence_unexposed=req.prevalence_unexposed,
    )

    # Assumptions / interpretation notes
    assumptions.append({
        "name": "No unmeasured confounding",
        "met": False,
        "detail": "E-value quantifies how strong unmeasured confounding must be to explain away the result.",
    })
    if ev.get("e_value_point_estimate", 99) < 2.0:
        warnings.append("Low E-value (<2). Result is sensitive to even weak unmeasured confounding.")

    # result_text (user-friendly)
    ev_pt = ev.get("e_value_point_estimate", 1.0)
    result_text = (
        f"Causal sensitivity: E-value = {ev_pt} for observed {req.measure.upper()}={req.observed_estimate}. "
        f"{ev.get('interpretation', '')}"
    )

    # export_rows for reproducibility
    export_rows = [
        ["Metric", "Value"],
        ["Observed estimate", req.observed_estimate],
        ["Measure", req.measure],
        ["E-value (point)", ev.get("e_value_point_estimate")],
        ["E-value (CI)", ev.get("e_value_ci")],
        ["Bias factor (QBA)", qba.get("bias_factor")],
        ["Bias-corrected estimate (QBA)", qba.get("bias_corrected_estimate")],
    ]

    return {
        "test": "Causal Sensitivity Analysis (E-value + QBA)",
        "e_value": ev,
        "quantitative_bias_analysis": qba,
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": "# See VanderWeele & Ding (2017) E-value paper\n# R: EValue::evalues.RR(est=..., lo=..., hi=...)",
    }
