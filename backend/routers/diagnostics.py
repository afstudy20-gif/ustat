"""Full linear regression diagnostics: VIF, leverage, Cook's D, residual plots,
heteroscedasticity tests, autocorrelation — all in a single call."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from services import store
from services.impute import apply_imputation

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# POST /linear_full  — Full linear regression diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

class LinearDiagFullRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: str = "listwise"


@router.post("/linear_full")
def linear_full_diagnostics(req: LinearDiagFullRequest):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.stats.stattools import durbin_watson

    df_full = _get_df(req.session_id)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")

    if len(df) < len(req.predictors) + 2:
        raise HTTPException(400, "Not enough observations after listwise deletion.")

    # ── Fit OLS ──────────────────────────────────────────────────────────────
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = df[req.outcome].astype(float)
    model = sm.OLS(y, X).fit()

    n = int(model.nobs)
    p = X.shape[1] - 1  # exclude constant
    fitted = model.fittedvalues.values
    resid = model.resid.values

    # ── 1. VIF ───────────────────────────────────────────────────────────────
    vif_list = []
    for i in range(X.shape[1]):
        col_name = X.columns[i]
        if col_name == "const":
            continue
        try:
            vif_val = float(variance_inflation_factor(X.values, i))
        except Exception:
            vif_val = float("nan")
        flag = None
        if not np.isnan(vif_val):
            if vif_val > 10:
                flag = "severe"
            elif vif_val > 5:
                flag = "moderate"
        vif_list.append({
            "variable": str(col_name),
            "vif": round(vif_val, 4) if not np.isnan(vif_val) else None,
            "flag": flag,
        })

    # ── 2–4. Influence: leverage, Cook's D, studentized residuals ────────────
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]
    student_resid = influence.resid_studentized_external

    high_lev_thresh = 2.0 * (p + 1) / n
    cooks_thresh = 4.0 / n

    high_leverage_count = int(np.sum(leverage > high_lev_thresh))
    high_cooks_count = int(np.sum(cooks_d > cooks_thresh))
    large_resid_count = int(np.sum(np.abs(student_resid) > 2))

    # Subsample for plotting (max 2000 points)
    N_plot = min(n, 2000)
    if n > N_plot:
        idx = np.sort(np.random.choice(n, N_plot, replace=False))
    else:
        idx = np.arange(n)

    influence_section = {
        "leverage": leverage[idx].tolist(),
        "cooks_distance": cooks_d[idx].tolist(),
        "studentized_residuals": student_resid[idx].tolist(),
        "high_leverage_count": high_leverage_count,
        "high_cooks_count": high_cooks_count,
        "large_residual_count": large_resid_count,
        "high_leverage_threshold": round(high_lev_thresh, 6),
        "cooks_threshold": round(cooks_thresh, 6),
    }

    # ── 5. Breusch-Pagan ────────────────────────────────────────────────────
    try:
        bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(model.resid, model.model.exog)
        bp_result = {
            "stat": round(float(bp_stat), 4),
            "p": float(bp_p),
            "significant": bool(bp_p < 0.05),
        }
    except Exception:
        bp_result = {"stat": None, "p": None, "significant": False}

    # ── 6. White test ────────────────────────────────────────────────────────
    try:
        white_stat, white_p, white_f, white_fp = het_white(model.resid, model.model.exog)
        white_result = {
            "stat": round(float(white_stat), 4),
            "p": float(white_p),
            "significant": bool(white_p < 0.05),
        }
    except Exception:
        white_result = {"stat": None, "p": None, "significant": False}

    # ── 7. Residual plots data ───────────────────────────────────────────────
    std_res_internal = influence.resid_studentized_internal
    sqrt_abs = np.sqrt(np.abs(std_res_internal))

    # QQ
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(resid, dist="norm")
    qq_x_line = np.array([float(min(osm)), float(max(osm))])
    qq_y_line = slope * qq_x_line + intercept

    # Subsample QQ arrays too
    qq_n = min(len(osm), N_plot)
    qq_idx = idx[:qq_n] if len(osm) > N_plot else np.arange(len(osm))

    plots = {
        "residuals_fitted": {
            "x": fitted[idx].tolist(),
            "y": resid[idx].tolist(),
        },
        "qq": {
            "theoretical": osm[qq_idx].tolist(),
            "sample": osr[qq_idx].tolist(),
            "line_x": qq_x_line.tolist(),
            "line_y": qq_y_line.tolist(),
        },
        "scale_location": {
            "x": fitted[idx].tolist(),
            "y": sqrt_abs[idx].tolist(),
        },
        "residual_histogram": resid.tolist(),
    }

    # ── 8. Durbin-Watson ─────────────────────────────────────────────────────
    dw = float(durbin_watson(model.resid))
    if dw < 1.5:
        dw_interp = "Positive autocorrelation"
    elif dw > 2.5:
        dw_interp = "Negative autocorrelation"
    else:
        dw_interp = "No autocorrelation"

    # ── Assumptions summary ──────────────────────────────────────────────────
    assumptions = []

    # Multicollinearity
    severe_vif = [v for v in vif_list if v["flag"] == "severe"]
    moderate_vif = [v for v in vif_list if v["flag"] == "moderate"]
    if severe_vif:
        assumptions.append({
            "name": "Multicollinearity (VIF)",
            "met": False,
            "detail": f"Severe multicollinearity: {', '.join(v['variable'] for v in severe_vif)} have VIF > 10",
        })
    elif moderate_vif:
        assumptions.append({
            "name": "Multicollinearity (VIF)",
            "met": True,
            "detail": f"Moderate: {', '.join(v['variable'] for v in moderate_vif)} have VIF > 5 but <= 10",
        })
    else:
        assumptions.append({
            "name": "Multicollinearity (VIF)",
            "met": True,
            "detail": "All VIF values <= 5",
        })

    # Homoscedasticity
    het_met = not bp_result["significant"]
    assumptions.append({
        "name": "Homoscedasticity (Breusch-Pagan)",
        "met": het_met,
        "detail": f"Breusch-Pagan p = {_p_str(bp_result['p'])}" if bp_result["p"] is not None else "Could not compute",
    })

    # Independence
    auto_met = 1.5 <= dw <= 2.5
    assumptions.append({
        "name": "Independence of residuals (Durbin-Watson)",
        "met": auto_met,
        "detail": f"DW = {dw:.4f} — {dw_interp}",
    })

    # Normality of residuals (Shapiro-Wilk on subsample if n > 5000)
    resid_sample = resid[:5000] if n > 5000 else resid
    try:
        sw_stat, sw_p = scipy_stats.shapiro(resid_sample)
        norm_met = bool(sw_p >= 0.05)
        assumptions.append({
            "name": "Normality of residuals (Shapiro-Wilk)",
            "met": norm_met,
            "detail": f"W = {sw_stat:.4f}, p = {_p_str(sw_p)}",
        })
    except Exception:
        assumptions.append({
            "name": "Normality of residuals",
            "met": True,
            "detail": "Could not compute Shapiro-Wilk",
        })

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings = []
    if severe_vif:
        warnings.append(f"Severe multicollinearity detected for: {', '.join(v['variable'] for v in severe_vif)}. Consider removing or combining predictors.")
    if bp_result["significant"]:
        warnings.append("Heteroscedasticity detected (Breusch-Pagan). Consider robust standard errors (HC3) or a transformation.")
    if not auto_met:
        warnings.append(f"Residual autocorrelation ({dw_interp.lower()}, DW = {dw:.2f}). Consider time-series methods or adding lag terms.")
    if high_cooks_count > 0:
        warnings.append(f"{high_cooks_count} observation(s) exceed Cook's distance threshold ({cooks_thresh:.4f}). Inspect for influential outliers.")

    # ── result_text ──────────────────────────────────────────────────────────
    vif_summary = "no multicollinearity detected"
    if severe_vif:
        vif_summary = f"{', '.join(v['variable'] for v in severe_vif)} {'has' if len(severe_vif) == 1 else 'have'} VIF > 10"
    elif moderate_vif:
        vif_summary = f"{', '.join(v['variable'] for v in moderate_vif)} {'has' if len(moderate_vif) == 1 else 'have'} moderate VIF (> 5)"

    bp_summary = f"no heteroscedasticity (p = {_p_str(bp_result['p'])})" if bp_result["p"] is not None and not bp_result["significant"] else "heteroscedasticity detected"
    dw_summary = f"{dw_interp.lower()} (DW = {dw:.2f})"
    cook_summary = f"{high_cooks_count} influential observation(s) flagged by Cook's distance" if high_cooks_count > 0 else "no influential observations flagged by Cook's distance"

    result_text = (
        f"Linear regression diagnostics (n = {n}, {p} predictor{'s' if p != 1 else ''}). "
        f"R-squared = {model.rsquared:.4f}, Adj. R-squared = {model.rsquared_adj:.4f}. "
        f"VIF check: {vif_summary}. "
        f"Breusch-Pagan: {bp_summary}. "
        f"Durbin-Watson: {dw_summary}. "
        f"{cook_summary.capitalize()}."
    )

    # ── export_rows ──────────────────────────────────────────────────────────
    export_rows = [
        ["Diagnostic", "Value"],
        ["n", n],
        ["Predictors", p],
        ["R-squared", round(float(model.rsquared), 4)],
        ["Adj. R-squared", round(float(model.rsquared_adj), 4)],
        ["Residual SE", round(float(np.sqrt(model.mse_resid)), 4)],
        ["Durbin-Watson", round(dw, 4)],
        ["Breusch-Pagan stat", bp_result["stat"]],
        ["Breusch-Pagan p", round(float(bp_result["p"]), 6) if bp_result["p"] is not None else None],
        ["White stat", white_result["stat"]],
        ["White p", round(float(white_result["p"]), 6) if white_result["p"] is not None else None],
        ["High leverage count", high_leverage_count],
        ["High Cook's D count", high_cooks_count],
        ["Large residual count (|t| > 2)", large_resid_count],
    ]
    for v in vif_list:
        export_rows.append([f"VIF: {v['variable']}", v["vif"]])

    # ── r_code ───────────────────────────────────────────────────────────────
    pred_formula = " + ".join(req.predictors)
    r_code = (
        f"library(car)\n"
        f"model <- lm({req.outcome} ~ {pred_formula}, data = data)\n"
        f"vif(model)\n"
        f"plot(model)  # 4 diagnostic plots\n"
        f"bptest(model)  # Breusch-Pagan\n"
        f"dwtest(model)  # Durbin-Watson"
    )

    return {
        "test": "Linear Regression Diagnostics",
        "n": n,
        "p": p,
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "residual_se": round(float(np.sqrt(model.mse_resid)), 4),
        "vif": vif_list,
        "influence": influence_section,
        "heteroscedasticity": {
            "breusch_pagan": bp_result,
            "white": white_result,
        },
        "autocorrelation": {
            "durbin_watson": round(dw, 4),
            "interpretation": dw_interp,
        },
        "plots": plots,
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }
