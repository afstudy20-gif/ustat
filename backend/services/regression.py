"""Regression service: logistic-family model fitting.

Extracted from routers/models_logistic.py so the router is a thin HTTP adapter.
Each fit_* function takes an already-loaded DataFrame plus a duck-typed request
object and returns a plain result dict. No session/store I/O lives here.

Public API:
    fit_logistic(df_full, req)
    fit_firth(df_full, req)
    fit_poisson(df_full, req)
    fit_or_table(df_full, req)
plus the reusable helpers (scaling, stepwise selection, results text, Firth MLE).
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from fastapi import HTTPException

from services.impute import apply_imputation
from routers._models_shared import (
    compute_vif as _compute_vif,
    add_pairwise_interactions as _add_pairwise_interactions,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _apply_scaling(df: pd.DataFrame, predictors: List[str], scale_factors: Optional[dict]):
    """Divide specified numeric columns by their scale factor and rename them.
    Returns (new_df, updated_predictor_list).
    """
    if not scale_factors:
        return df, predictors
    df = df.copy()
    new_predictors = []
    for pred in predictors:
        factor = scale_factors.get(pred)
        if factor is not None:
            try:
                factor_f = float(factor)
            except (TypeError, ValueError):
                factor_f = 1.0
        else:
            factor_f = 1.0
        if factor_f and factor_f != 1.0 and pred in df.columns:
            factor_label = int(factor_f) if factor_f == int(factor_f) else factor_f
            new_name = f"{pred} (per {factor_label} units)"
            df[new_name] = df[pred] / factor_f
            new_predictors.append(new_name)
        else:
            new_predictors.append(pred)
    return df, new_predictors


def _p_for_pred(pred: str, pvalues) -> float:
    """Return the representative p-value for a predictor from a fitted model.
    For numeric variables the name matches exactly; for categorical dummies it
    uses the most conservative (max) p-value across all dummy levels.
    """
    if pred in pvalues.index:
        return float(pvalues[pred])
    dummy_cols = [c for c in pvalues.index if c != "const" and c.startswith(pred + "_")]
    if dummy_cols:
        return float(pvalues[dummy_cols].max())
    return 1.0


def _uni_p_for_pred(pred: str, uni_results: dict) -> float:
    """Look up univariate p-value from uni_results dict (keyed by dummy names)."""
    if pred in uni_results:
        return uni_results[pred]["p"]
    matching = [v["p"] for k, v in uni_results.items() if k.startswith(pred + "_")]
    return min(matching) if matching else 1.0  # best (min) p across dummy levels


def _stepwise_forward(y, df: pd.DataFrame, pred_list: list, p_enter: float = 0.05) -> list:
    """Forward selection: greedily add the variable with lowest p < p_enter."""
    selected: list = []
    remaining = list(pred_list)
    while remaining:
        best_var, best_p = None, p_enter
        for var in remaining:
            candidate = selected + [var]
            X_enc = pd.get_dummies(df[candidate], drop_first=True).astype(float)
            X_const = sm.add_constant(X_enc, has_constant="add")
            try:
                m = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
                p = _p_for_pred(var, m.pvalues)
                if p < best_p:
                    best_p, best_var = p, var
            except Exception:
                pass
        if best_var is None:
            break
        selected.append(best_var)
        remaining.remove(best_var)
    return selected


def _stepwise_backward(y, df: pd.DataFrame, pred_list: list, p_remove: float = 0.10) -> list:
    """Backward elimination: iteratively remove the variable with highest p > p_remove."""
    selected = list(pred_list)
    while selected:
        X_enc = pd.get_dummies(df[selected], drop_first=True).astype(float)
        X_const = sm.add_constant(X_enc, has_constant="add")
        try:
            m = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
        except Exception:
            break
        worst_var, worst_p = None, p_remove
        for var in selected:
            p = _p_for_pred(var, m.pvalues)
            if p > worst_p:
                worst_p, worst_var = p, var
        if worst_var is None:
            break
        selected.remove(worst_var)
    return selected



def _logistic_results_text(outcome, coefs, chi2_val, df, chi2_p, nagelkerke, hl, classification, auc):
    """Generate a publication-style results paragraph for logistic regression."""
    sig_coefs = [c for c in coefs if c["variable"] != "const" and c["p"] < 0.05]
    ns_coefs = [c for c in coefs if c["variable"] != "const" and c["p"] >= 0.05]
    n_pred = len([c for c in coefs if c["variable"] != "const"])

    parts = []
    # Omnibus
    p_str = "<0.001" if chi2_p < 0.001 else f"{chi2_p:.3f}"
    parts.append(f"A binary logistic regression was performed to predict {outcome}. "
                 f"The omnibus test indicated the model was {'statistically significant' if chi2_p < 0.05 else 'not statistically significant'} "
                 f"(χ²({df}) = {chi2_val:.3f}, p = {p_str}).")

    # Model fit
    parts.append(f"The model explained {nagelkerke*100:.1f}% of the variance (Nagelkerke R²) "
                 f"and correctly classified {classification['accuracy']*100:.1f}% of cases." if classification else "")

    # HL
    if hl:
        hl_p_str = "<0.001" if hl["p"] < 0.001 else f'{hl["p"]:.3f}'
        parts.append(f"Hosmer-Lemeshow test indicated {'adequate' if hl['p'] >= 0.05 else 'poor'} model fit (p = {hl_p_str}).")

    # AUC
    if auc:
        parts.append(f"The area under the ROC curve was {auc:.3f}.")

    # Significant predictors
    if sig_coefs:
        pred_strs = []
        for c in sig_coefs:
            p_s = "<0.001" if c["p"] < 0.001 else f'{c["p"]:.3f}'
            pred_strs.append(f'{c["variable"]} (OR = {c["odds_ratio"]:.2f}, 95% CI: {c["or_ci_low"]:.2f}–{c["or_ci_high"]:.2f}, p = {p_s})')
        parts.append("Significant predictors were: " + "; ".join(pred_strs) + ".")
    else:
        parts.append("No predictor reached statistical significance at the 0.05 level.")

    return " ".join(p for p in parts if p)


def _firth_fit(X: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-6):
    """Penalized MLE via Firth (Newton-Raphson). Returns (beta, vcov, llf, n_iter, converged).

    X must already include a constant column. y must be 0/1 integer. The
    penalized log-likelihood is l*(β) = l(β) + 0.5 log|I(β)|, whose score is
    s* = X' (y - π) + X' diag(h)(1/2 - π) and observed information is X' W X
    (the curvature of the penalty contributes O(1/n) and is dropped, following
    Heinze 2002 — Wald inference uses the unpenalized Fisher information).
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        eta = X @ beta
        # Clip eta for numerical stability with separated data.
        eta = np.clip(eta, -30, 30)
        pi = 1.0 / (1.0 + np.exp(-eta))
        W = pi * (1.0 - pi)
        # Information matrix I = X' W X.
        XtWX = (X.T * W) @ X
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            # Ridge a tiny diagonal if the information is singular (collinear).
            XtWX_inv = np.linalg.inv(XtWX + 1e-8 * np.eye(p))
        # Hat-matrix diagonal h_i = w_i x_i' (X'WX)^{-1} x_i.
        sqrtW = np.sqrt(W)
        XW = X * sqrtW[:, None]
        H = XW @ XtWX_inv @ XW.T
        h = np.diag(H)
        # Firth score: X' [(y - pi) + h (1/2 - pi)].
        score = X.T @ ((y - pi) + h * (0.5 - pi))
        step = XtWX_inv @ score
        beta_new = beta + step
        # Halving safeguards monotonic increase in penalized log-likelihood.
        for _ in range(10):
            eta_new = np.clip(X @ beta_new, -30, 30)
            pi_new = 1.0 / (1.0 + np.exp(-eta_new))
            ll_new = float(np.sum(y * eta_new - np.log1p(np.exp(eta_new))))
            try:
                sign, logdet = np.linalg.slogdet((X.T * (pi_new * (1.0 - pi_new))) @ X)
                pen_ll_new = ll_new + 0.5 * logdet
            except Exception:
                pen_ll_new = ll_new
            ll_old = float(np.sum(y * eta - np.log1p(np.exp(eta))))
            try:
                sign_o, logdet_o = np.linalg.slogdet(XtWX)
                pen_ll_old = ll_old + 0.5 * logdet_o
            except Exception:
                pen_ll_old = ll_old
            if pen_ll_new >= pen_ll_old - 1e-10 or np.linalg.norm(step) < tol:
                break
            step = step / 2.0
            beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    # Final fit-quality summaries.
    eta = np.clip(X @ beta, -30, 30)
    pi = 1.0 / (1.0 + np.exp(-eta))
    W = pi * (1.0 - pi)
    XtWX = (X.T * W) @ X
    try:
        vcov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        vcov = np.linalg.inv(XtWX + 1e-8 * np.eye(p))
    ll = float(np.sum(y * eta - np.log1p(np.exp(eta))))
    try:
        sign, logdet = np.linalg.slogdet(XtWX)
        penalized_ll = ll + 0.5 * logdet
    except Exception:
        penalized_ll = ll
    return beta, vcov, ll, penalized_ll, n_iter, converged, pi


def _poisson_results_text(outcome, coefs):
    sig = [c for c in coefs if c["variable"] != "const" and c["p"] < 0.05]
    parts = [f"Poisson regression was performed to model {outcome}."]
    if sig:
        preds = []
        for c in sig:
            p_s = "<0.001" if c["p"] < 0.001 else f'{c["p"]:.3f}'
            preds.append(f'{c["variable"]} (IRR = {c["irr"]:.2f}, 95% CI: {c["irr_ci_low"]:.2f}–{c["irr_ci_high"]:.2f}, p = {p_s})')
        parts.append("Significant predictors: " + "; ".join(preds) + ".")
    else:
        parts.append("No predictor reached statistical significance.")
    return " ".join(parts)


# ── Logistic OR Table (Univariate + Multivariate) ────────────────────────────


def _ortable_results_text(outcome, table, stats, selection):
    """Auto-generate results paragraph for OR Table."""
    parts = []
    uni_sig = [r for r in table if r.get("uni_p") is not None and r["uni_p"] < 0.05]
    multi_sig = [r for r in table if r.get("multi_p") is not None and r["multi_p"] < 0.05]

    parts.append(f"Univariate logistic regression identified {len(uni_sig)} of {len(table)} variables "
                 f"as significantly associated with {outcome} (p < 0.05).")

    if stats:
        om = stats.get("omnibus")
        if om:
            p_s = "<0.001" if om["p"] < 0.001 else f'{om["p"]:.3f}'
            parts.append(f"The multivariate model ({selection}) was {'significant' if om['p'] < 0.05 else 'not significant'} "
                         f"(χ²({om['df']}) = {om['chi2']:.3f}, p = {p_s}), "
                         f"Nagelkerke R² = {stats.get('nagelkerke_r2', 0):.3f}.")
        if stats.get("auc"):
            parts.append(f"AUC = {stats['auc']:.3f}.")
        cl = stats.get("classification")
        if cl:
            parts.append(f"Overall classification accuracy was {cl['accuracy']*100:.1f}%.")

    if multi_sig:
        pred_strs = []
        for r in multi_sig:
            p_s = "<0.001" if r["multi_p"] < 0.001 else f'{r["multi_p"]:.3f}'
            pred_strs.append(f'{r["variable"]} (OR = {r["multi_or"]:.2f}, 95% CI: {r["multi_ci_low"]:.2f}–{r["multi_ci_high"]:.2f}, p = {p_s})')
        parts.append("Independent predictors: " + "; ".join(pred_strs) + ".")

    return " ".join(parts)





def fit_logistic(df_full, req):
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)
    X = pd.get_dummies(df[pred_list], drop_first=True).astype(float)
    X, _ix_added = _add_pairwise_interactions(X, req.interactions, pred_list)
    X_const = sm.add_constant(X)
    y = df[req.outcome]  # outcome column not scaled
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = pd.to_numeric(y, errors="coerce")
        unique_vals = sorted(y.dropna().unique())
        if set(unique_vals) - {0, 1, 0.0, 1.0}:
            raise HTTPException(status_code=422, detail=f"Logistic regression requires a binary 0/1 outcome. Found values: {unique_vals[:10]}")
        y = y.astype(int)

    if len(set(y)) < 2:
        raise HTTPException(status_code=422, detail="Outcome column has only one unique value — logistic regression requires both 0 and 1.")

    cov_type = "HC3" if req.robust_se else "nonrobust"
    model = sm.Logit(y, X_const).fit(disp=False, cov_type=cov_type)

    # ── Variables in the Equation (B, SE, Wald, df, Sig, Exp(B), CI) ──
    vifs = _compute_vif(X_const)
    coefs = []
    ci = model.conf_int()
    for var in model.params.index:
        est = float(model.params[var])
        se_val = float(model.bse[var])
        z_val = float(model.tvalues[var])
        wald = z_val ** 2  # Wald = z²
        coefs.append({
            "variable": str(var),
            "B": est,
            "log_odds": est,
            "se": se_val,
            "wald": round(wald, 4),
            "df": 1,
            "p": float(model.pvalues[var]),
            "odds_ratio": float(np.exp(est)),
            "z": z_val,
            "or_ci_low": float(np.exp(ci.loc[var, 0])),
            "or_ci_high": float(np.exp(ci.loc[var, 1])),
            "vif": vifs.get(str(var)),
        })

    # ── SPSS-style Model-Level Statistics ──
    from scipy.stats import chi2 as chi2_dist
    from sklearn.metrics import roc_auc_score, confusion_matrix

    n = float(model.nobs)
    llf = float(model.llf)
    llnull = float(model.llnull)

    # Omnibus Test of Model Coefficients (LR test)
    omnibus_chi2 = -2 * (llnull - llf)
    omnibus_df = len(model.params) - 1  # exclude intercept
    omnibus_p = float(1 - chi2_dist.cdf(omnibus_chi2, omnibus_df)) if omnibus_df > 0 else 1.0

    # -2 Log Likelihood
    minus2ll = -2 * llf

    # Cox & Snell R²
    cox_snell_r2 = 1 - np.exp((2 / n) * (llnull - llf))

    # Nagelkerke R²
    max_r2 = 1 - np.exp((2 / n) * llnull)
    nagelkerke_r2 = float(cox_snell_r2 / max_r2) if max_r2 != 0 else 0.0

    # Predicted probabilities
    pred_probs = model.predict(X_const)
    y_arr = np.array(y)

    # Hosmer-Lemeshow test
    try:
        order = np.argsort(pred_probs)
        groups = np.array_split(order, 10)
        hl_chi2_val = 0.0
        for grp in groups:
            obs_1 = y_arr[grp].sum()
            obs_0 = len(grp) - obs_1
            exp_1 = pred_probs[grp].sum()
            exp_0 = len(grp) - exp_1
            if exp_1 > 0:
                hl_chi2_val += (obs_1 - exp_1) ** 2 / exp_1
            if exp_0 > 0:
                hl_chi2_val += (obs_0 - exp_0) ** 2 / exp_0
        hl_df = 8  # g - 2, where g = 10
        hl_p = float(1 - chi2_dist.cdf(hl_chi2_val, hl_df))
        hosmer_lemeshow = {"chi2": round(hl_chi2_val, 4), "df": hl_df, "p": round(hl_p, 6)}
    except Exception:
        hosmer_lemeshow = None

    # Classification table
    y_pred = (pred_probs >= 0.5).astype(int)
    try:
        cm = confusion_matrix(y_arr, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = float((tp + tn) / (tp + tn + fp + fn))
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        classification = {
            "accuracy": round(accuracy, 4), "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4), "ppv": round(ppv, 4), "npv": round(npv, 4),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    except Exception:
        classification = None

    # AUC
    try:
        auc = float(roc_auc_score(y_arr, pred_probs))
    except Exception:
        auc = None

    return {
        "model": f"Logistic Regression{' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        # Model Summary
        "minus2ll": round(minus2ll, 4),
        "cox_snell_r2": round(float(cox_snell_r2), 4),
        "nagelkerke_r2": round(float(nagelkerke_r2), 4),
        "pseudo_r2": float(model.prsquared),
        "log_likelihood": llf,
        "aic": float(model.aic),
        "bic": float(model.bic),
        # Omnibus Test
        "omnibus": {"chi2": round(omnibus_chi2, 4), "df": omnibus_df, "p": round(omnibus_p, 6)},
        # Hosmer-Lemeshow
        "hosmer_lemeshow": hosmer_lemeshow,
        # Classification
        "classification": classification,
        # AUC
        "auc": round(auc, 4) if auc is not None else None,
        # Coefficients
        "coefficients": coefs,
        # Auto-generated results text
        "result_text": _logistic_results_text(req.outcome, coefs, omnibus_chi2, omnibus_df, omnibus_p, nagelkerke_r2, hosmer_lemeshow, classification, auc),
    }


def fit_firth(df_full, req):
    """Penalized (Firth) logistic regression — bias-corrected MLE that handles
    complete or quasi-complete separation and rare-event outcomes that crash
    or produce infinite ORs in standard logistic regression.
    """
    from scipy.stats import chi2 as chi2_dist
    from scipy.stats import norm as sp_norm
    from sklearn.metrics import roc_auc_score, confusion_matrix

    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)
    X_df = pd.get_dummies(df[pred_list], drop_first=True).astype(float)
    X_df, _ix_added = _add_pairwise_interactions(X_df, req.interactions, pred_list)
    X_const_df = sm.add_constant(X_df, has_constant="add")
    y = df[req.outcome]
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = pd.to_numeric(y, errors="coerce")
        unique_vals = sorted(y.dropna().unique())
        if set(unique_vals) - {0, 1, 0.0, 1.0}:
            raise HTTPException(status_code=422,
                detail=f"Firth logistic requires a binary 0/1 outcome. Found values: {unique_vals[:10]}")
        y = y.astype(int)
    if len(set(y)) < 2:
        raise HTTPException(status_code=422,
            detail="Outcome column has only one unique value — logistic regression requires both 0 and 1.")

    X = X_const_df.values.astype(float)
    y_arr = np.asarray(y, dtype=int)

    beta, vcov, ll, penalized_ll, n_iter, converged, pi = _firth_fit(
        X, y_arr, max_iter=req.max_iter, tol=req.tol
    )
    se = np.sqrt(np.diag(vcov))

    # Null model (intercept-only) for LR test under the penalized likelihood.
    X_null = np.ones((len(y_arr), 1))
    _, _, ll0, pen_ll0, _, _, _ = _firth_fit(X_null, y_arr, max_iter=req.max_iter, tol=req.tol)

    # Wald CIs + p-values.
    z = beta / np.where(se > 0, se, np.nan)
    p_two = 2.0 * (1.0 - sp_norm.cdf(np.abs(z)))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    coefs = []
    for i, var in enumerate(X_const_df.columns):
        coefs.append({
            "variable": str(var),
            "B": float(beta[i]),
            "log_odds": float(beta[i]),
            "se": float(se[i]),
            "wald": round(float(z[i] ** 2), 4),
            "df": 1,
            "p": float(p_two[i]),
            "odds_ratio": float(np.exp(beta[i])),
            "z": float(z[i]),
            "or_ci_low": float(np.exp(ci_low[i])),
            "or_ci_high": float(np.exp(ci_high[i])),
            "vif": None,
        })

    n = len(y_arr)
    minus2ll = -2.0 * ll
    # Penalized LR omnibus test (preferred under Firth).
    omnibus_chi2 = float(2.0 * (penalized_ll - pen_ll0))
    omnibus_df = X.shape[1] - 1
    omnibus_p = float(1 - chi2_dist.cdf(omnibus_chi2, omnibus_df)) if omnibus_df > 0 else 1.0

    # Pseudo-R² (Cox & Snell and Nagelkerke from unpenalized likelihoods).
    cox_snell_r2 = float(1 - np.exp((2 / n) * (ll0 - ll))) if n > 0 else 0.0
    max_r2 = float(1 - np.exp((2 / n) * ll0)) if n > 0 else 1.0
    nagelkerke_r2 = float(cox_snell_r2 / max_r2) if max_r2 != 0 else 0.0

    # AIC / BIC use the penalized log-likelihood per Heinze (2002).
    k = X.shape[1]
    aic = float(-2.0 * penalized_ll + 2 * k)
    bic = float(-2.0 * penalized_ll + np.log(n) * k)

    # Classification at the 0.5 threshold.
    y_pred = (pi >= 0.5).astype(int)
    try:
        cm = confusion_matrix(y_arr, y_pred)
        tn, fp, fn, tp = cm.ravel()
        classification = {
            "accuracy": round(float((tp + tn) / (tp + tn + fp + fn)), 4),
            "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
            "specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
            "ppv": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4),
            "npv": round(float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0, 4),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    except Exception:
        classification = None
    try:
        auc = float(roc_auc_score(y_arr, pi))
    except Exception:
        auc = None

    return {
        "model": "Firth Penalized Logistic Regression",
        "outcome": req.outcome,
        "n": int(n),
        "n_events": int(y_arr.sum()),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "converged": bool(converged),
        "iterations": int(n_iter),
        "minus2ll": round(minus2ll, 4),
        "penalized_ll": round(float(penalized_ll), 4),
        "log_likelihood": float(ll),
        "cox_snell_r2": round(cox_snell_r2, 4),
        "nagelkerke_r2": round(nagelkerke_r2, 4),
        "pseudo_r2": round(cox_snell_r2, 4),
        "aic": round(aic, 4),
        "bic": round(bic, 4),
        "omnibus": {"chi2": round(omnibus_chi2, 4), "df": omnibus_df, "p": round(omnibus_p, 6)},
        "auc": round(auc, 4) if auc is not None else None,
        "classification": classification,
        "coefficients": coefs,
        "method_note": (
            "Firth (1993) bias-corrected logistic regression with Jeffreys-prior "
            "penalty; recommended for rare events or (quasi-)separated data. "
            "Wald CIs are reported; profile-penalized-likelihood CIs are not "
            "computed in this version. Reference: Heinze & Schemper, Stat Med 2002."
        ),
        "result_text": (
            f"Firth penalized logistic regression was used to model {req.outcome} "
            f"(n = {n}, events = {int(y_arr.sum())}). The model "
            f"{'converged' if converged else 'did not fully converge'} in {n_iter} "
            f"iterations. Omnibus penalized likelihood ratio χ² = {omnibus_chi2:.3f} "
            f"on {omnibus_df} df, p = "
            f"{'<0.001' if omnibus_p < 0.001 else f'{omnibus_p:.3f}'}. "
            f"Nagelkerke R² = {nagelkerke_r2:.3f}"
            + (f", AUC = {auc:.3f}." if auc is not None else ".")
        ),
    }


def fit_poisson(df_full, req):
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if y.isna().all():
        raise HTTPException(status_code=422, detail="Outcome column has no numeric values.")
    if (y.dropna() < 0).any():
        raise HTTPException(status_code=422, detail="Poisson regression requires non-negative integer counts. Negative values found.")
    if (y.dropna() % 1 != 0).any():
        raise HTTPException(status_code=422, detail="Poisson regression requires integer counts. Fractional values found — consider Gamma regression instead.")
    cov_type = "HC3" if req.robust_se else "nonrobust"
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type=cov_type)
    ci = model.conf_int()
    vifs = _compute_vif(X)
    coefs = []
    for var in model.params.index:
        est = float(model.params[var])
        coefs.append({
            "variable": str(var),
            "log_irr": est,
            "irr": float(np.exp(est)),
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
            "irr_ci_low":  float(np.exp(ci.loc[var, 0])),
            "irr_ci_high": float(np.exp(ci.loc[var, 1])),
            "vif": vifs.get(str(var)),
        })
    return {
        "model": f"Poisson Regression{' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "aic": float(model.aic),
        "bic": float(model.bic),
        "coefficients": coefs,
        "result_text": _poisson_results_text(req.outcome, coefs),
    }



def fit_or_table(df_full, req):
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    # Apply unit scaling (renames columns & divides values)
    df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)

    # Encode outcome — must be binary 0/1
    y_raw = df[req.outcome]
    if y_raw.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y_num = pd.to_numeric(y_raw, errors="coerce")
        unique_vals = sorted(y_num.dropna().unique())
        if set(unique_vals) - {0, 1, 0.0, 1.0}:
            raise HTTPException(status_code=422, detail=f"Logistic regression requires a binary 0/1 outcome. Found: {unique_vals[:10]}")
        y = y_num.values
    # Drop rows where outcome is NaN
    valid_mask = ~pd.isna(y)
    y = np.array(y[valid_mask], dtype=int)
    df = df.loc[valid_mask].reset_index(drop=True)
    if len(set(y)) < 2:
        raise HTTPException(status_code=422, detail="Outcome has only one unique value — needs both 0 and 1.")

    # Helper: fit logit (MLE or Firth) and extract OR rows for every predictor
    # (dummy-encoded categorical levels included). When use_firth=True the
    # _firth_fit Newton-Raphson is run instead — same row shape comes out so
    # both univariate and multivariate codepaths share their downstream
    # rendering.
    use_firth = bool(getattr(req, "use_firth", False))
    from scipy.stats import norm as _sp_norm

    def _fit_row(X_df, variable_names, return_model=False):
        X_enc = pd.get_dummies(X_df, drop_first=True).astype(float)
        # Build a combined frame to drop NaN from both predictors and outcome together
        combined = X_enc.copy()
        combined["__y__"] = y
        combined = combined.dropna()
        if len(combined) < 10:
            raise HTTPException(status_code=422, detail=f"Insufficient data after removing missing values ({len(combined)} rows)")
        y_clean = combined["__y__"].values.astype(int)
        X_clean = combined.drop(columns=["__y__"])
        if len(set(y_clean)) < 2:
            raise HTTPException(status_code=422, detail="After NaN removal, outcome has only one unique value.")
        X_const = sm.add_constant(X_clean, has_constant="add")

        if use_firth:
            # ── Firth's penalised MLE branch ──
            try:
                X_arr = X_const.values.astype(float)
                y_arr_in = y_clean.astype(int)
                beta, vcov, ll, penalized_ll, n_iter, converged, pi = _firth_fit(X_arr, y_arr_in)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Firth fit failed: {exc}")
            se = np.sqrt(np.diag(vcov))
            z = beta / np.where(se > 0, se, np.nan)
            p_two = 2.0 * (1.0 - _sp_norm.cdf(np.abs(z)))
            ci_lo_arr = beta - 1.95996 * se
            ci_hi_arr = beta + 1.95996 * se
            rows = {}
            for i, var in enumerate(X_const.columns):
                if var == "const":
                    continue
                or_val = float(np.exp(beta[i]))
                ci_lo = float(np.exp(ci_lo_arr[i]))
                ci_hi = float(np.exp(ci_hi_arr[i]))
                p_val = float(p_two[i])
                if not np.isfinite(or_val): or_val = 9999.0
                if not np.isfinite(ci_lo): ci_lo = 0.0
                if not np.isfinite(ci_hi): ci_hi = 9999.0
                rows[var] = {"or": or_val, "ci_low": ci_lo, "ci_high": ci_hi, "p": p_val}
            if return_model:
                # Build a minimal results bag the multivariable downstream
                # code can read — mirrors the statsmodels Results attributes
                # actually consulted below (nobs / llf / llnull / params /
                # predict). Null log-likelihood comes from an intercept-only
                # Firth fit so the omnibus LR test stays on the penalised
                # likelihood scale.
                X_null = np.ones((len(y_arr_in), 1))
                _, _, ll0, _, _, _, _ = _firth_fit(X_null, y_arr_in)
                class _FirthResults:
                    def __init__(self, beta, vcov, ll, ll0, pi, X_const, n_params):
                        self.params = pd.Series(beta, index=X_const.columns)
                        self.bse = pd.Series(np.sqrt(np.diag(vcov)), index=X_const.columns)
                        self.tvalues = self.params / self.bse.replace(0, np.nan)
                        self.pvalues = pd.Series(p_two, index=X_const.columns)
                        self.llf = float(ll)
                        self.llnull = float(ll0)
                        self.nobs = float(len(pi))
                        self.aic = float(-2.0 * ll + 2 * n_params)
                        self.bic = float(-2.0 * ll + np.log(self.nobs) * n_params)
                        # McFadden's pseudo-R² so downstream model-summary
                        # code can read `prsquared` the same way it does for
                        # a real statsmodels Results object.
                        self.prsquared = float(1.0 - ll / ll0) if ll0 != 0 else 0.0
                        self._pi = pi
                    def predict(self, _X):
                        return self._pi
                    def conf_int(self):
                        return pd.DataFrame({0: ci_lo_arr, 1: ci_hi_arr}, index=X_const.columns)
                m = _FirthResults(beta, vcov, ll, ll0, pi, X_const, X_arr.shape[1])
                return rows, m, X_const, y_clean
            return rows

        # ── Standard MLE branch (statsmodels Logit) ──
        try:
            m = sm.Logit(y_clean, X_const).fit(disp=False, maxiter=200)
        except np.linalg.LinAlgError:
            raise HTTPException(status_code=422, detail="Perfect separation detected — model cannot converge. Try removing collinear predictors or switch on Firth's penalised likelihood (use_firth=true).")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Model convergence error: {exc}")
        rows = {}
        ci = m.conf_int()
        for var in m.params.index:
            if var == "const":
                continue
            or_val = float(np.exp(m.params[var]))
            ci_lo = float(np.exp(ci.loc[var, 0]))
            ci_hi = float(np.exp(ci.loc[var, 1]))
            p_val = float(m.pvalues[var])
            # Cap extreme ORs for JSON safety
            if not np.isfinite(or_val): or_val = 9999.0
            if not np.isfinite(ci_lo): ci_lo = 0.0
            if not np.isfinite(ci_hi): ci_hi = 9999.0
            rows[var] = {"or": or_val, "ci_low": ci_lo, "ci_high": ci_hi, "p": p_val}
        if return_model:
            return rows, m, X_const, y_clean
        return rows

    # ── Univariate: one model per predictor (post-scaling) ───────────────────
    uni_results: dict = {}
    skipped: list = []
    for pred in pred_list:
        try:
            rows = _fit_row(df[[pred]], [pred])
            for var, vals in rows.items():
                uni_results[var] = vals
        except HTTPException as he:
            # Skip this predictor with a warning instead of crashing
            skipped.append(f"{pred}: {he.detail}")
        except Exception as exc:
            skipped.append(f"{pred}: {exc}")

    # ── Variable selection for multivariate ──────────────────────────────────
    sel = (req.selection or "all").strip().lower()

    if sel == "p05":
        multi_pred_list = [p for p in pred_list if _uni_p_for_pred(p, uni_results) < 0.05]
        selection_label = "Univariate p < 0.05"
    elif sel == "p10":
        multi_pred_list = [p for p in pred_list if _uni_p_for_pred(p, uni_results) < 0.10]
        selection_label = "Univariate p < 0.10"
    elif sel == "forward":
        multi_pred_list = _stepwise_forward(y, df, pred_list, p_enter=0.05)
        selection_label = "Stepwise Forward (p_enter=0.05)"
    elif sel == "backward":
        multi_pred_list = _stepwise_backward(y, df, pred_list, p_remove=0.10)
        selection_label = "Stepwise Backward (p_remove=0.10)"
    else:  # "all"
        multi_pred_list = pred_list
        selection_label = "All variables (Enter)"

    # ── Multivariate: only selected predictors ────────────────────────────────
    multi_results: dict = {}
    multi_error = None
    model_stats = None
    if multi_pred_list:
        try:
            multi_results, multi_model, multi_X, multi_y = _fit_row(df[multi_pred_list], multi_pred_list, return_model=True)
            # SPSS-style model-level stats from multivariate model
            from scipy.stats import chi2 as chi2_dist
            from sklearn.metrics import roc_auc_score, confusion_matrix as _cm

            n_m = float(multi_model.nobs)
            llf_m = float(multi_model.llf)
            llnull_m = float(multi_model.llnull)
            omnibus_chi2 = -2 * (llnull_m - llf_m)
            omnibus_df = len(multi_model.params) - 1
            omnibus_p = float(1 - chi2_dist.cdf(omnibus_chi2, omnibus_df)) if omnibus_df > 0 else 1.0
            minus2ll = -2 * llf_m
            cox_snell = 1 - np.exp((2 / n_m) * (llnull_m - llf_m))
            max_r2 = 1 - np.exp((2 / n_m) * llnull_m)
            nagelkerke = float(cox_snell / max_r2) if max_r2 != 0 else 0.0
            pred_probs = multi_model.predict(multi_X)
            # AUC
            try: auc_val = float(roc_auc_score(multi_y, pred_probs))
            except Exception: auc_val = None
            # Hosmer-Lemeshow
            try:
                order = np.argsort(pred_probs)
                groups = np.array_split(order, 10)
                hl_chi2_val = 0.0
                for grp in groups:
                    o1 = multi_y[grp].sum(); o0 = len(grp) - o1
                    e1 = pred_probs[grp].sum(); e0 = len(grp) - e1
                    if e1 > 0: hl_chi2_val += (o1 - e1)**2 / e1
                    if e0 > 0: hl_chi2_val += (o0 - e0)**2 / e0
                hl_p = float(1 - chi2_dist.cdf(hl_chi2_val, 8))
                hl = {"chi2": round(hl_chi2_val, 4), "df": 8, "p": round(hl_p, 6)}
            except Exception: hl = None
            # Classification
            try:
                y_pred = (pred_probs >= 0.5).astype(int)
                tn, fp, fn, tp = _cm(multi_y, y_pred).ravel()
                classification = {
                    "accuracy": round(float((tp+tn)/(tp+tn+fp+fn)), 4),
                    "sensitivity": round(float(tp/(tp+fn)), 4) if (tp+fn) > 0 else 0,
                    "specificity": round(float(tn/(tn+fp)), 4) if (tn+fp) > 0 else 0,
                    "ppv": round(float(tp/(tp+fp)), 4) if (tp+fp) > 0 else 0,
                    "npv": round(float(tn/(tn+fn)), 4) if (tn+fn) > 0 else 0,
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                }
            except Exception: classification = None

            model_stats = {
                "omnibus": {"chi2": round(omnibus_chi2, 4), "df": omnibus_df, "p": round(omnibus_p, 6)},
                "minus2ll": round(minus2ll, 4),
                "cox_snell_r2": round(float(cox_snell), 4),
                "nagelkerke_r2": round(nagelkerke, 4),
                "pseudo_r2": round(float(multi_model.prsquared), 4),
                "auc": round(auc_val, 4) if auc_val else None,
                "hosmer_lemeshow": hl,
                "classification": classification,
            }
        except HTTPException as he:
            multi_error = he.detail
        except Exception as exc:
            multi_error = str(exc)

    # ── Merge rows ────────────────────────────────────────────────────────────
    all_vars = list(dict.fromkeys(list(uni_results.keys()) + list(multi_results.keys())))
    table = []
    for var in all_vars:
        u = uni_results.get(var, {})
        m = multi_results.get(var, {})
        table.append({
            "variable": var,
            "uni_or": u.get("or"),
            "uni_ci_low": u.get("ci_low"),
            "uni_ci_high": u.get("ci_high"),
            "uni_p": u.get("p"),
            "multi_or": m.get("or"),
            "multi_ci_low": m.get("ci_low"),
            "multi_ci_high": m.get("ci_high"),
            "multi_p": m.get("p"),
        })

    return {
        "model": ("Firth Penalised Logistic OR Table" if use_firth else "Logistic OR Table"),
        "use_firth": bool(use_firth),
        "outcome": req.outcome,
        "n": len(df),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "selection_method": selection_label,
        "n_multi": len(multi_pred_list),
        "n_total": len(pred_list),
        "table": table,
        "model_stats": model_stats,
        "result_text": _ortable_results_text(req.outcome, table, model_stats, selection_label),
        "warnings": (skipped if skipped else []) + ([f"Multivariate: {multi_error}"] if multi_error else []),
    }


