import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services import store
from services.impute import apply_imputation
from services.rcs_basis import (
    KNOT_PERCENTILES as _KNOT_PERCENTILES,
    rcs_basis as _rcs_basis,
    resolve_knots as _resolve_knots,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


# ── Linear Regression ────────────────────────────────────────────────────────

class LinearRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/linear")
def linear_regression(req: LinearRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = df[req.outcome].astype(float)
    base = sm.OLS(y, X)
    model = base.fit(cov_type="HC3", use_t=True) if req.robust_se else base.fit()

    coefs = []
    ci = model.conf_int()
    for var in model.params.index:
        coefs.append({
            "variable": str(var),
            "estimate": float(model.params[var]),
            "se": float(model.bse[var]),
            "t": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
        })

    # ── Predictor metadata for the interactive prediction panel ──────────────
    predictor_info: dict = {}
    for col in req.predictors:
        if col not in df_full.columns:
            continue
        s = df_full[col].dropna()
        if len(s) == 0:
            continue
        if pd.api.types.is_numeric_dtype(s):
            predictor_info[col] = {
                "type": "numeric",
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "median": float(s.median()),
            }
        else:
            vc = s.value_counts()
            predictor_info[col] = {
                "type": "categorical",
                "categories": vc.index.astype(str).tolist(),
                "counts": [int(v) for v in vc.values],
            }

    return {
        "model": f"Linear Regression (OLS){' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_p": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "coefficients": coefs,
        "residual_se": float(np.sqrt(model.mse_resid)),
        "df_resid": int(model.df_resid),
        "predictors": req.predictors,
        "predictor_info": predictor_info,
        "result_text": _linear_results_text(req.outcome, coefs, model),
    }


def _linear_results_text(outcome, coefs, model):
    sig = [c for c in coefs if c["variable"] != "const" and c["p"] < 0.05]
    f_p = "<0.001" if model.f_pvalue < 0.001 else f"{model.f_pvalue:.3f}"
    parts = [
        f"Multiple linear regression was performed to predict {outcome}. "
        f"The overall model was {'statistically significant' if model.f_pvalue < 0.05 else 'not significant'} "
        f"(F({int(model.df_model)},{int(model.df_resid)}) = {model.fvalue:.3f}, p = {f_p}), "
        f"explaining {model.rsquared*100:.1f}% of the variance (R² = {model.rsquared:.3f}, adjusted R² = {model.rsquared_adj:.3f})."
    ]
    if sig:
        preds = []
        for c in sig:
            p_s = "<0.001" if c["p"] < 0.001 else f'{c["p"]:.3f}'
            preds.append(f'{c["variable"]} (B = {c["estimate"]:.3f}, SE = {c["se"]:.3f}, p = {p_s})')
        parts.append("Significant predictors: " + "; ".join(preds) + ".")
    return " ".join(parts)


# ── Logistic Regression ───────────────────────────────────────────────────────

class LogisticRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    scale_factors: Optional[dict] = None
    selection: Optional[str] = "all"
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


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


@router.post("/logistic")
def logistic_regression(req: LogisticRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)
    X = pd.get_dummies(df[pred_list], drop_first=True).astype(float)
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


# ── Poisson Regression ───────────────────────────────────────────────────────

class PoissonRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/poisson")
def poisson_regression(req: PoissonRequest):
    df_full = _get_df(req.session_id)
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

@router.post("/logistic_table")
def logistic_or_table(req: LogisticRequest):
    df_full = _get_df(req.session_id)
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

    # Helper: fit logit and extract first non-const row OR all predictor rows
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
        try:
            m = sm.Logit(y_clean, X_const).fit(disp=False, maxiter=200)
        except np.linalg.LinAlgError:
            raise HTTPException(status_code=422, detail="Perfect separation detected — model cannot converge. Try removing collinear predictors.")
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
        "model": "Logistic OR Table",
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


# ── Kaplan-Meier Survival ─────────────────────────────────────────────────────

class KMRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    group_col: Optional[str] = None
    stratify_col: Optional[str] = None
    imputation: Optional[str] = "listwise"


def _safe_float(v):
    """Return float or None for inf/nan values that aren't JSON-serializable."""
    try:
        f = float(v)
        if np.isfinite(f):
            return f
        return None  # np.inf / -np.inf / nan → null in JSON
    except (TypeError, ValueError):
        return None


def _km_fit_groups(df: pd.DataFrame, duration_col: str, event_col: str, group_col: Optional[str]) -> list:
    groups = df[group_col].unique() if group_col else [None]
    results = []
    for grp in groups:
        subset = df[df[group_col] == grp] if group_col else df
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(
                subset[duration_col].astype(float),
                subset[event_col].astype(int),
                label=str(grp) if grp is not None else "All",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"KM fitting error: {exc}")
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["time", "survival"]
        curve = [
            {"time": _safe_float(row["time"]), "survival": _safe_float(row["survival"])}
            for _, row in sf.iterrows()
        ]
        results.append({
            "group": str(grp) if grp is not None else "All",
            "n": int(len(subset)),
            "events": int(subset[event_col].sum()),
            "median_survival": _safe_float(kmf.median_survival_time_),
            "curve": curve,
        })
    return results


def _km_logrank(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str) -> Optional[dict]:
    groups = df[group_col].unique()
    if len(groups) < 2:
        return None
    try:
        if len(groups) == 2:
            g0 = df[df[group_col] == groups[0]]
            g1 = df[df[group_col] == groups[1]]
            lr = logrank_test(
                g0[duration_col], g1[duration_col],
                event_observed_A=g0[event_col].astype(int),
                event_observed_B=g1[event_col].astype(int),
            )
            return {"test": "Log-rank", "p": _safe_float(lr.p_value)}
        else:
            lr = multivariate_logrank_test(df[duration_col], df[group_col], df[event_col].astype(int))
            return {"test": "Log-rank (multivariate)", "p": _safe_float(lr.p_value)}
    except Exception:
        return None


@router.post("/survival/km")
def kaplan_meier(req: KMRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col]    = pd.to_numeric(df_full[req.event_col],    errors="coerce")

    km_cols = [req.duration_col, req.event_col]
    df = apply_imputation(df_full, km_cols, req.imputation)
    n_excluded = n_total - len(df)

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after coercing duration/event columns to numeric. Check that both columns contain numbers.")

    event_vals = sorted(df[req.event_col].dropna().unique())
    if set(event_vals) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1 (0=censored, 1=event). Found: {event_vals[:10]}")

    if (df[req.duration_col] < 0).any():
        raise HTTPException(status_code=422, detail="Duration column contains negative values. All durations must be ≥ 0.")

    # ── Stratified mode ──────────────────────────────────────────────────────────
    if req.stratify_col:
        if req.stratify_col not in df_full.columns:
            raise HTTPException(status_code=422, detail=f"Stratify column '{req.stratify_col}' not found.")
        strata_vals = sorted(df[req.stratify_col].dropna().unique(), key=lambda x: (isinstance(x, str), x))
        strata_out = []
        for sv in strata_vals:
            sub = df[df[req.stratify_col] == sv].copy()
            if len(sub) == 0:
                continue
            grp_results = _km_fit_groups(sub, req.duration_col, req.event_col, req.group_col)
            lr = _km_logrank(sub, req.duration_col, req.event_col, req.group_col) if req.group_col else None
            strata_out.append({"label": str(sv), "n": int(len(sub)), "groups": grp_results, "logrank": lr})
        return {
            "model": "Kaplan-Meier",
            "strata": strata_out,
            "stratify_col": req.stratify_col,
            "n_total": n_total,
            "n_excluded": n_excluded,
            "imputation": req.imputation,
        }

    # ── Standard (non-stratified) mode ──────────────────────────────────────────
    results = _km_fit_groups(df, req.duration_col, req.event_col, req.group_col)
    logrank = _km_logrank(df, req.duration_col, req.event_col, req.group_col) if req.group_col else None

    return {
        "model": "Kaplan-Meier",
        "groups": results,
        "logrank": logrank,
        "n_total": n_total,
        "n_excluded": n_excluded,
        "imputation": req.imputation,
    }


# ── Cox Proportional Hazards ──────────────────────────────────────────────────

class CoxRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    # Optional pairwise interactions — each entry is [col_A, col_B] (or
    # written "A*B"/"A:B"). Both sides must already be in `predictors`
    # (or be encoded into dummies of a predictor). Numeric × numeric =
    # element-wise product; numeric × dummy and dummy × dummy work too.
    interactions: Optional[List[List[str]]] = None


@router.post("/survival/cox")
def cox_regression(req: CoxRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col]    = pd.to_numeric(df_full[req.event_col],    errors="coerce")

    cox_cols = [req.duration_col, req.event_col] + req.predictors
    df = apply_imputation(df_full, cox_cols, req.imputation)
    n_excluded = n_total - len(df)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after coercing duration/event columns to numeric.")

    # Validate event indicator is binary 0/1
    event_vals = sorted(df[req.event_col].dropna().unique())
    if set(event_vals) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1. Found: {event_vals[:10]}")
    if (df[req.duration_col] < 0).any():
        raise HTTPException(status_code=422, detail="Duration column contains negative values.")

    # Encode predictors: numeric stays numeric, categorical / text columns
    # are dummy-coded with drop_first=True so lifelines receives only float
    # columns. Without this, a SEX column with values 'M' / 'F' would crash
    # CoxPHFitter with a 'could not convert string to float' error.
    pred_raw = df[req.predictors].copy()
    numeric_pred: list[str] = []
    cat_pred:     list[str] = []
    for c in req.predictors:
        col = pred_raw[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric_pred.append(c)
        else:
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().mean() >= 0.8 and len(coerced.dropna().unique()) > 2:
                pred_raw[c] = coerced
                numeric_pred.append(c)
            else:
                cat_pred.append(c)
    num_part = pred_raw[numeric_pred].apply(pd.to_numeric, errors="coerce") if numeric_pred else pd.DataFrame(index=pred_raw.index)
    cat_part = pd.get_dummies(pred_raw[cat_pred], drop_first=True, dummy_na=False) if cat_pred else pd.DataFrame(index=pred_raw.index)
    enc = pd.concat([num_part, cat_part], axis=1).astype(float)

    # ── Interaction columns ──────────────────────────────────────────────
    # Build A:B columns by multiplying encoded design slots. For numeric
    # columns we use the numeric column directly; for categoricals we use
    # every surviving dummy (the user's natural mental model "SEX × AGE"
    # then becomes SEX_M:AGE — one row in the output per dummy).
    interaction_cols: list[str] = []
    if req.interactions:
        def _members(name: str) -> list[str]:
            """Return the encoded design columns that represent `name`.
            Numeric → [name]; categorical → every dummy whose name starts
            with `name_`."""
            if name in enc.columns:
                return [name]
            prefix = f"{name}_"
            return [c for c in enc.columns if c.startswith(prefix)]
        for pair in req.interactions:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise HTTPException(status_code=422, detail=f"Each interaction must be a [colA, colB] pair. Got: {pair}")
            a_members = _members(pair[0])
            b_members = _members(pair[1])
            if not a_members or not b_members:
                raise HTTPException(status_code=422, detail=f"Interaction '{pair[0]} × {pair[1]}': one or both columns are not in the predictor list.")
            for a in a_members:
                for b in b_members:
                    new_col = f"{a}:{b}"
                    enc[new_col] = enc[a] * enc[b]
                    interaction_cols.append(new_col)

    fit_df = pd.concat([df[[req.duration_col, req.event_col]], enc], axis=1).dropna()
    if len(fit_df) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows after encoding (need ≥ 10, got {len(fit_df)}).")

    cph = CoxPHFitter()
    try:
        cph.fit(fit_df, duration_col=req.duration_col, event_col=req.event_col)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cox fitting error: {exc}")

    summary = cph.summary.reset_index()
    coefs = []
    for _, row in summary.iterrows():
        coefs.append({
            "variable": row["covariate"],
            "log_hr": _safe_float(row["coef"]),
            "hr": _safe_float(row["exp(coef)"]),
            "se": _safe_float(row["se(coef)"]),
            "z": _safe_float(row["z"]),
            "p": _safe_float(row["p"]),
            "hr_ci_low": _safe_float(row["exp(coef) lower 95%"]),
            "hr_ci_high": _safe_float(row["exp(coef) upper 95%"]),
        })

    return {
        "model": "Cox Proportional Hazards",
        "n": int(cph.event_observed.sum()),
        "n_total": n_total,
        "n_excluded": n_excluded,
        "imputation": req.imputation,
        "log_likelihood": _safe_float(cph.log_likelihood_),
        "concordance": _safe_float(cph.concordance_index_),
        "coefficients": coefs,
        "interactions_used": interaction_cols,
    }


# ── Restricted Cubic Splines ─────────────────────────────────────────────────

class RCSRequest(BaseModel):
    session_id: str
    predictor: str
    outcome: Optional[str] = None       # required for logistic/linear
    covariates: List[str] = []
    n_knots: int = 4                    # 3, 4, or 5
    ref_value: Optional[float] = None   # OR/HR reference (median if None)
    model_type: str = "logistic"        # "logistic" | "linear" | "cox"
    imputation: str = "listwise"
    # Cox-specific (required when model_type == "cox")
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    # Optional override for Harrell percentile knots
    knot_positions: Optional[List[float]] = None
    # Covariates to interact with the spline (must also be in `covariates`).
    # For each named covariate (dummy-encoded if categorical) we add
    # (n_knots − 1) × (#dummies) interaction columns multiplying the spline
    # basis. We then refit a reduced model without those columns and report
    # an LR test as the spline × covariate interaction p-value.
    interaction_covariates: Optional[List[str]] = None


@router.post("/rcs")
def rcs_regression(req: RCSRequest):
    if req.n_knots not in _KNOT_PERCENTILES:
        raise HTTPException(status_code=422, detail=f"n_knots must be 3, 4, or 5. Got: {req.n_knots}")

    model_type = (req.model_type or "logistic").lower()
    if model_type not in ("logistic", "linear", "cox"):
        raise HTTPException(status_code=422, detail=f"Unknown model_type: {req.model_type}")

    is_cox = model_type == "cox"

    if is_cox:
        if not req.duration_col or not req.event_col:
            raise HTTPException(status_code=422, detail="duration_col and event_col are required when model_type='cox'.")
        cols_needed = [req.predictor, req.duration_col, req.event_col] + req.covariates
    else:
        if not req.outcome:
            raise HTTPException(status_code=422, detail="outcome is required when model_type is 'logistic' or 'linear'.")
        cols_needed = [req.predictor, req.outcome] + req.covariates

    df_full = _get_df(req.session_id)
    missing_cols = [c for c in cols_needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found in session: {missing_cols}")

    df = df_full[cols_needed].copy()
    n_total = len(df)

    # Numeric coerce: only required columns (predictor, outcome / duration+event).
    # Covariates are coerced separately below so categorical columns survive.
    required_numeric = [req.predictor]
    if is_cox:
        required_numeric += [req.duration_col, req.event_col]
    else:
        required_numeric += [req.outcome]
    for c in required_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Encode covariates: numeric stays numeric (coerce), categorical → dummies
    # (drop_first=True). The encoded design slice lives in cov_df.
    cov_df = None
    if req.covariates:
        cov_raw = df[req.covariates].copy()
        numeric_cov: list[str] = []
        cat_cov: list[str] = []
        for c in req.covariates:
            col = cov_raw[c]
            # Treat as numeric if dtype is numeric AND coerces cleanly for ≥80%
            # of rows; otherwise treat as categorical.
            if pd.api.types.is_numeric_dtype(col):
                numeric_cov.append(c)
            else:
                coerced = pd.to_numeric(col, errors="coerce")
                if coerced.notna().mean() >= 0.8 and len(coerced.dropna().unique()) > 2:
                    cov_raw[c] = coerced
                    numeric_cov.append(c)
                else:
                    cat_cov.append(c)
        # Numeric covariate slice
        num_part = cov_raw[numeric_cov].apply(pd.to_numeric, errors="coerce") if numeric_cov else pd.DataFrame(index=cov_raw.index)
        # Categorical covariate slice → dummies (drop_first=True for identifiability)
        cat_part = pd.get_dummies(cov_raw[cat_cov], drop_first=True, dummy_na=False) if cat_cov else pd.DataFrame(index=cov_raw.index)
        cov_df = pd.concat([num_part, cat_part], axis=1).astype(float)
        # Re-assign cov_df back onto df.index then drop NAs across the joined frame
        df = pd.concat([df.drop(columns=req.covariates), cov_df], axis=1)
    df = df.dropna()
    n = len(df)
    if n < 10:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows (need ≥ 10). Got {n} after dropping rows with missing predictor / outcome / covariates (total available: {n_total}).")

    x_raw = df[req.predictor].values.astype(float)

    n_unique_x = len(np.unique(x_raw))
    if n_unique_x < req.n_knots + 2:
        raise HTTPException(status_code=422, detail=f"Predictor '{req.predictor}' has only {n_unique_x} unique values — need ≥ {req.n_knots + 2} for {req.n_knots}-knot spline.")

    if is_cox:
        duration = df[req.duration_col].values.astype(float)
        event    = df[req.event_col].values.astype(float)
        if np.any(duration < 0):
            raise HTTPException(status_code=422, detail=f"duration_col '{req.duration_col}' must be ≥ 0.")
        unique_e = sorted(set(event.tolist()))
        if set(unique_e) - {0.0, 1.0}:
            raise HTTPException(status_code=422, detail=f"event_col '{req.event_col}' must be binary 0/1. Found: {unique_e[:10]}")
    else:
        y = df[req.outcome].values.astype(float)
        if model_type == "logistic":
            unique_y = sorted(set(y.tolist()))
            if set(unique_y) - {0.0, 1.0}:
                raise HTTPException(status_code=422, detail=f"Logistic RCS requires binary 0/1 outcome. Found: {unique_y[:10]}")

    # Resolve knot positions (Harrell percentiles or user-supplied)
    try:
        knots = _resolve_knots(x_raw, req.n_knots, req.knot_positions, req.predictor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    spline_cols = _rcs_basis(x_raw, knots)
    # Final covariate design slice — column order matches cov_df.columns
    if cov_df is not None and cov_df.shape[1] > 0:
        cov_names = list(cov_df.columns)
        cov_mat = df[cov_names].values.astype(float)
    else:
        cov_names = []
        cov_mat = None

    # ── Spline × covariate interaction columns ────────────────────────────
    # Each interaction covariate multiplies BOTH the linear x term and every
    # spline basis column, so the spline shape is allowed to differ across
    # levels (numeric × spline = single set; categorical × spline = one set
    # per surviving dummy).
    interaction_cov_names: list[str] = []
    interaction_extra_names: list[str] = []  # names of the added columns
    interaction_extra: list[np.ndarray] = []  # the actual column arrays
    interaction_extra_meta: list[tuple[str, int]] = []  # (member_col, spline_part_idx) for X_syn rebuild
    if req.interaction_covariates:
        # Validate: every requested interaction covariate must appear in the
        # encoded covariate slice (either directly when numeric, or via a
        # dummy prefix when categorical).
        def _resolve_cov(name: str) -> list[str]:
            if name in cov_names:
                return [name]
            prefix = f"{name}_"
            return [c for c in cov_names if c.startswith(prefix)]

        spline_design = np.column_stack([x_raw, spline_cols])  # (n, k-1)
        spline_part_names = ["lin"] + [f"sp{i+1}" for i in range(spline_cols.shape[1])]
        for cov in req.interaction_covariates:
            members = _resolve_cov(cov)
            if not members:
                raise HTTPException(status_code=422, detail=f"interaction_covariates entry '{cov}' is not in the selected covariates list.")
            interaction_cov_names.append(cov)
            for m in members:
                cov_vec = df[m].values.astype(float)
                for i in range(spline_design.shape[1]):
                    interaction_extra.append(spline_design[:, i] * cov_vec)
                    interaction_extra_names.append(f"{m}:{req.predictor}_{spline_part_names[i]}")
                    interaction_extra_meta.append((m, i))

    if interaction_extra:
        interaction_mat = np.column_stack(interaction_extra)
    else:
        interaction_mat = None

    # ── Fit model ────────────────────────────────────────────────────────────
    try:
        if is_cox:
            # lifelines wants a single DataFrame with duration + event + features
            feat_cols = [f"_x_lin"] + [f"_spl_{i}" for i in range(spline_cols.shape[1])]
            fit_df = pd.DataFrame(
                np.column_stack([x_raw, spline_cols]),
                columns=feat_cols,
                index=df.index,
            )
            for c in cov_names:
                fit_df[c] = df[c].values
            for i, ix_name in enumerate(interaction_extra_names):
                fit_df[ix_name] = interaction_extra[i]
            fit_df["_dur_"] = duration
            fit_df["_evt_"] = event
            cph = CoxPHFitter()
            cph.fit(fit_df, duration_col="_dur_", event_col="_evt_")
            # Aligned column ordering for the design matrix
            design_cols = feat_cols + cov_names + interaction_extra_names
            params = cph.params_.reindex(design_cols).values
            cov_params = cph.variance_matrix_.reindex(index=design_cols, columns=design_cols).values
            aic_val = None  # lifelines exposes AIC_partial_ in newer versions
            try:
                aic_val = float(getattr(cph, "AIC_partial_", np.nan))
                if np.isnan(aic_val):
                    aic_val = None
            except Exception:
                aic_val = None
            log_lik = float(cph.log_likelihood_)
            concordance = float(cph.concordance_index_)
            n_events = int(np.sum(event))
        else:
            # Logistic / linear: intercept + x + spline + covariates + interactions
            X_parts = [np.ones(n), x_raw, spline_cols]
            if cov_mat is not None:
                X_parts.append(cov_mat)
            if interaction_mat is not None:
                X_parts.append(interaction_mat)
            X = np.column_stack(X_parts)
            if model_type == "logistic":
                result = sm.Logit(y, X).fit(disp=0, maxiter=200)
            else:
                result = sm.OLS(y, X).fit()
            params = result.params
            cov_params = result.cov_params()
            try:
                aic_val = float(result.aic)
                if np.isnan(aic_val) or np.isinf(aic_val):
                    aic_val = None
            except Exception:
                aic_val = None
            log_lik = float(getattr(result, "llf", np.nan))
            if np.isnan(log_lik) or np.isinf(log_lik):
                log_lik = None
            concordance = None
            n_events = int(y.sum()) if model_type == "logistic" else None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model fitting error: {exc}")

    # ── Interaction LR test ────────────────────────────────────────────────
    # Refit a reduced model that omits the spline × covariate interaction
    # columns and compute 2*(logL_full − logL_reduced) ~ χ²(#interaction_cols).
    interaction_result = None
    if interaction_extra_names:
        try:
            if is_cox:
                reduced_df = fit_df.drop(columns=interaction_extra_names)
                cph_red = CoxPHFitter()
                cph_red.fit(reduced_df, duration_col="_dur_", event_col="_evt_")
                ll_red = float(cph_red.log_likelihood_)
                ll_full = float(log_lik)
            else:
                X_red_parts = [np.ones(n), x_raw, spline_cols]
                if cov_mat is not None:
                    X_red_parts.append(cov_mat)
                X_red = np.column_stack(X_red_parts)
                if model_type == "logistic":
                    res_red = sm.Logit(y, X_red).fit(disp=0, maxiter=200)
                else:
                    res_red = sm.OLS(y, X_red).fit()
                ll_red = float(getattr(res_red, "llf", np.nan))
                ll_full = float(log_lik) if log_lik is not None else float(getattr(result, "llf", np.nan))
            lr_stat = 2.0 * (ll_full - ll_red)
            from scipy.stats import chi2 as _chi2
            df_lr = len(interaction_extra_names)
            p_lr = float(_chi2.sf(lr_stat, df=df_lr))
            interaction_result = {
                "covariates": interaction_cov_names,
                "lr_stat": round(lr_stat, 4),
                "df": df_lr,
                "p": round(p_lr, 6),
                "log_lik_full": round(ll_full, 4),
                "log_lik_reduced": round(ll_red, 4),
            }
        except Exception as exc:
            interaction_result = {"covariates": interaction_cov_names, "error": str(exc)}

    # ── Dose-response curve ─────────────────────────────────────────────────
    x_lo, x_hi = float(np.percentile(x_raw, 1)), float(np.percentile(x_raw, 99))
    x_syn = np.linspace(x_lo, x_hi, 200)
    sp_syn = _rcs_basis(x_syn, knots)

    # Lookup mean of each encoded covariate column so we can recreate the
    # interaction synthetic columns at the same anchor.
    cov_means_by_name: dict[str, float] = {}
    if cov_names:
        cov_means = cov_mat.mean(axis=0)
        for cn, mean_val in zip(cov_names, cov_means):
            cov_means_by_name[cn] = float(mean_val)

    spline_design_syn = np.column_stack([x_syn, sp_syn])  # (200, k-1)

    if is_cox:
        # Cox design has NO intercept (baseline hazard absorbs it)
        if cov_mat is not None:
            X_syn = np.column_stack([x_syn, sp_syn, np.tile(cov_means, (200, 1))])
        else:
            X_syn = np.column_stack([x_syn, sp_syn])
    else:
        if cov_mat is not None:
            X_syn = np.column_stack([np.ones(200), x_syn, sp_syn, np.tile(cov_means, (200, 1))])
        else:
            X_syn = np.column_stack([np.ones(200), x_syn, sp_syn])

    # Append synthetic interaction columns: each = spline_part(x_syn) × mean(covariate dummy)
    if interaction_extra_meta:
        ix_syn = np.column_stack([
            spline_design_syn[:, spi] * cov_means_by_name.get(member, 0.0)
            for (member, spi) in interaction_extra_meta
        ])
        X_syn = np.column_stack([X_syn, ix_syn])

    lp_syn = X_syn @ params

    ref_val = req.ref_value if req.ref_value is not None else float(np.median(x_raw))
    ref_val = float(np.clip(ref_val, x_lo, x_hi))
    ref_idx = int(np.argmin(np.abs(x_syn - ref_val)))
    lp_ref  = lp_syn[ref_idx]
    rel_lp  = lp_syn - lp_ref

    diffs   = X_syn - X_syn[ref_idx]
    var_lp  = np.einsum("ij,jk,ik->i", diffs, cov_params, diffs)
    se_lp   = np.sqrt(np.maximum(var_lp, 0))
    z95     = 1.96

    if is_cox or model_type == "logistic":
        or_vals  = np.exp(rel_lp)
        ci_low   = np.exp(rel_lp - z95 * se_lp)
        ci_high  = np.exp(rel_lp + z95 * se_lp)
    else:
        or_vals = rel_lp
        ci_low  = rel_lp - z95 * se_lp
        ci_high = rel_lp + z95 * se_lp

    def _ns(v):
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if np.isnan(fv) or np.isinf(fv):
            return None
        return round(fv, 4)

    def _clean(arr):
        return [_ns(v) for v in arr]

    # `effect_type` tells the frontend which interpretation to render:
    #   logistic → OR, cox → HR, linear → mean-difference.
    effect_type = "HR" if is_cox else ("OR" if model_type == "logistic" else "mean_diff")

    # Per-covariate coefficient summary so the frontend can show "adjusted for X, Y, Z"
    # AND surface the actual hazard / odds ratios for the adjustment terms.
    cov_summary = []
    if cov_names:
        n_pre_cov = (0 if is_cox else 2) + (1 + spline_cols.shape[1])  # intercept (if any) + linear + spline cols
        for offset, name in enumerate(cov_names):
            i = n_pre_cov + offset
            beta = float(params[i]) if i < len(params) else None
            se = float(np.sqrt(max(cov_params[i, i], 0.0))) if i < len(params) else None
            cov_summary.append({
                "name": name,
                "coef": round(beta, 6) if beta is not None else None,
                "effect": round(float(np.exp(beta)), 4) if (is_cox or model_type == "logistic") and beta is not None else (round(beta, 4) if beta is not None else None),
                "se":   round(se, 6) if se is not None else None,
            })

    return {
        "predictor":      req.predictor,
        "outcome":        req.outcome,
        "duration_col":   req.duration_col,
        "event_col":      req.event_col,
        "model_type":     model_type,
        "effect_type":    effect_type,
        "n":              n,
        "n_total":        n_total,
        "n_excluded":     n_total - n,
        "n_events":       n_events,
        "n_knots":        req.n_knots,
        "knots":          [round(float(kn), 2) for kn in knots],
        "knot_positions_custom": req.knot_positions is not None,
        "ref_value":      round(ref_val, 4),
        "aic":            _ns(aic_val),
        "log_likelihood": _ns(log_lik),
        "concordance":    _ns(concordance),
        "covariates_requested": list(req.covariates or []),
        "covariates_used":      cov_names,
        "covariates_summary":   cov_summary,
        "interaction":          interaction_result,
        "interaction_terms":    interaction_extra_names,
        "x_values":       _clean(x_syn),
        "or_values":      _clean(or_vals),   # kept for backward compat; really effect_type values
        "ci_low":         _clean(ci_low),
        "ci_high":        _clean(ci_high),
        "x_data":         _clean(x_raw[:500]),  # raw data rug (first 500 points)
    }


# ── Multivariable Cox-RCS (with optional RCS × RCS interaction) ──────────────

class SplineTerm(BaseModel):
    column: str
    n_knots: int = 4
    knot_positions: Optional[List[float]] = None
    ref_value: Optional[float] = None


class CoxRCSRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    spline_terms: List[SplineTerm]
    covariates: List[str] = []
    include_interaction: bool = False
    imputation: Optional[str] = "listwise"
    grid_size: int = 50  # for prediction surface


@router.post("/survival/cox_rcs")
def cox_rcs(req: CoxRCSRequest):
    """Multivariable Cox proportional hazards with restricted cubic splines.

    Supports 1 or 2 RCS terms plus additional linear covariates.
    When 2 spline terms are supplied and `include_interaction=True`, tensor-
    product columns are added and an LR test against the main-effects-only
    model is reported.
    """
    if not (1 <= len(req.spline_terms) <= 2):
        raise HTTPException(status_code=422, detail="spline_terms must contain 1 or 2 entries.")

    for term in req.spline_terms:
        if term.n_knots not in _KNOT_PERCENTILES:
            raise HTTPException(status_code=422, detail=f"n_knots for '{term.column}' must be 3, 4, or 5. Got: {term.n_knots}")

    if req.include_interaction and len(req.spline_terms) != 2:
        raise HTTPException(status_code=422, detail="include_interaction requires exactly 2 spline_terms.")

    df_full = _get_df(req.session_id)
    spline_cols = [t.column for t in req.spline_terms]
    cols_needed = list(dict.fromkeys(spline_cols + [req.duration_col, req.event_col] + req.covariates))
    missing_cols = [c for c in cols_needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found in session: {missing_cols}")

    # Coerce involved columns to numeric (matches existing /rcs and /cox behaviour)
    df = df_full[cols_needed].copy()
    for c in cols_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if req.imputation and req.imputation != "listwise":
        df = apply_imputation(df, cols_needed, req.imputation)
    else:
        df = df.dropna()
    n = len(df)
    if n < 15:
        raise HTTPException(status_code=400, detail="Not enough complete rows (need ≥ 15).")

    duration = df[req.duration_col].values.astype(float)
    event    = df[req.event_col].values.astype(float)
    if np.any(duration < 0):
        raise HTTPException(status_code=422, detail=f"duration_col '{req.duration_col}' must be ≥ 0.")
    if set(sorted(set(event.tolist()))) - {0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"event_col '{req.event_col}' must be binary 0/1.")
    if event.sum() < 5:
        raise HTTPException(status_code=400, detail="Need ≥ 5 events to fit a Cox model.")

    # ── Build spline basis for each term ────────────────────────────────────
    term_info = []  # list of dicts: {column, knots, x_raw, n_basis, ref_value, col_names}
    for ti, term in enumerate(req.spline_terms):
        x_raw = df[term.column].values.astype(float)
        n_unique = len(np.unique(x_raw))
        if n_unique < term.n_knots + 2:
            raise HTTPException(status_code=422, detail=f"Spline term '{term.column}' has only {n_unique} unique values — need ≥ {term.n_knots + 2} for {term.n_knots}-knot spline.")
        try:
            knots = _resolve_knots(x_raw, term.n_knots, term.knot_positions, term.column)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        sp = _rcs_basis(x_raw, knots)  # shape (n, n_knots-2)
        # Per-term design columns: linear + spline curves → n_knots-1 columns
        lin_col = f"t{ti}_{term.column}_lin"
        sp_cols = [f"t{ti}_{term.column}_sp{i}" for i in range(sp.shape[1])]
        full_cols = [lin_col] + sp_cols
        term_info.append({
            "column": term.column,
            "knots": knots,
            "x_raw": x_raw,
            "design": np.column_stack([x_raw, sp]),
            "col_names": full_cols,
            "ref_value": term.ref_value if term.ref_value is not None else float(np.median(x_raw)),
        })

    # ── Assemble main-effects design DataFrame ──────────────────────────────
    feat_arrays = []
    feat_names: List[str] = []
    for ti in term_info:
        feat_arrays.append(ti["design"])
        feat_names.extend(ti["col_names"])

    # Covariates: numeric raw, categorical dummies via get_dummies
    cov_df = pd.DataFrame(index=df.index)
    if req.covariates:
        cov_raw = df_full.loc[df.index, req.covariates].copy()
        cov_df = pd.get_dummies(cov_raw, drop_first=True, dummy_na=False)
        # Re-coerce remaining numeric columns
        for c in cov_df.columns:
            cov_df[c] = pd.to_numeric(cov_df[c], errors="coerce")
        cov_df = cov_df.dropna()
        # Align after potential row drops
        df_aligned = df.loc[cov_df.index]
        duration = df_aligned[req.duration_col].values.astype(float)
        event    = df_aligned[req.event_col].values.astype(float)
        # Rebuild spline basis on aligned rows
        for ti, term in enumerate(req.spline_terms):
            x_raw = df_aligned[term.column].values.astype(float)
            sp = _rcs_basis(x_raw, term_info[ti]["knots"])
            term_info[ti]["design"] = np.column_stack([x_raw, sp])
            term_info[ti]["x_raw"] = x_raw
        feat_arrays = [ti["design"] for ti in term_info]
        n = len(df_aligned)
        if n < 15:
            raise HTTPException(status_code=400, detail="Not enough complete rows after covariate handling (need ≥ 15).")

    main_design = np.column_stack(feat_arrays) if feat_arrays else np.empty((n, 0))

    # ── Interaction tensor-product columns ──────────────────────────────────
    interaction_design = None
    interaction_names: List[str] = []
    if req.include_interaction:
        a = term_info[0]["design"]
        b = term_info[1]["design"]
        # Skip the linear×linear term — it is already represented by the two
        # marginal linear columns multiplied; including would re-encode the
        # main linear effect interaction. Use the FULL tensor (Harrell standard).
        a_names = term_info[0]["col_names"]
        b_names = term_info[1]["col_names"]
        ix_cols = []
        for i in range(a.shape[1]):
            for j in range(b.shape[1]):
                ix_cols.append(a[:, i] * b[:, j])
                interaction_names.append(f"ix_{a_names[i]}_x_{b_names[j]}")
        interaction_design = np.column_stack(ix_cols)

    # ── Fit full model ──────────────────────────────────────────────────────
    full_design_arrays = [main_design]
    full_names = list(feat_names)
    if interaction_design is not None:
        full_design_arrays.append(interaction_design)
        full_names = full_names + interaction_names
    cov_names = list(cov_df.columns)
    if cov_names:
        full_design_arrays.append(cov_df.values.astype(float))
        full_names = full_names + cov_names

    full_design = np.column_stack(full_design_arrays) if full_design_arrays else np.empty((n, 0))

    full_df = pd.DataFrame(full_design, columns=full_names, index=range(n))
    full_df["_dur_"] = duration
    full_df["_evt_"] = event

    try:
        cph_full = CoxPHFitter()
        cph_full.fit(full_df, duration_col="_dur_", event_col="_evt_")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cox-RCS fitting error: {exc}")

    params_full = cph_full.params_.reindex(full_names).values
    cov_full    = cph_full.variance_matrix_.reindex(index=full_names, columns=full_names).values
    se_full     = cph_full.standard_errors_.reindex(full_names).values
    p_full      = None
    try:
        p_full  = cph_full.summary["p"].reindex(full_names).values
    except Exception:
        pass
    ci_low_full = cph_full.confidence_intervals_.iloc[:, 0].reindex(full_names).values
    ci_high_full = cph_full.confidence_intervals_.iloc[:, 1].reindex(full_names).values
    log_lik_full = float(cph_full.log_likelihood_)

    # Coefficients table
    coefs = []
    for i, name in enumerate(full_names):
        coef = float(params_full[i])
        se   = float(se_full[i])
        z    = coef / se if se > 0 else None
        p    = float(p_full[i]) if (p_full is not None and not np.isnan(p_full[i])) else None
        coefs.append({
            "name": name,
            "coef": coef,
            "hr":   float(np.exp(coef)),
            "se":   se,
            "z":    z,
            "p":    p,
            "ci_low":  float(np.exp(ci_low_full[i])),
            "ci_high": float(np.exp(ci_high_full[i])),
        })

    # ── Nonlinearity Wald test per spline term ──────────────────────────────
    nonlinearity = {}
    for ti, term in enumerate(req.spline_terms):
        # Spline columns (excluding the leading linear column) for this term
        sp_names = term_info[ti]["col_names"][1:]
        idx = [full_names.index(n) for n in sp_names]
        if not idx:
            continue
        b = params_full[idx]
        cv = cov_full[np.ix_(idx, idx)]
        try:
            wald = float(b @ np.linalg.solve(cv, b))
            from scipy.stats import chi2 as _chi2
            p_nl = float(_chi2.sf(wald, df=len(idx)))
        except Exception:
            wald = None
            p_nl = None
        nonlinearity[term.column] = {
            "wald": wald,
            "df":   len(idx),
            "p":    p_nl,
        }

    # ── Interaction LR test ─────────────────────────────────────────────────
    interaction_result = None
    if req.include_interaction and interaction_design is not None:
        reduced_names = [n for n in full_names if n not in interaction_names]
        reduced_df = full_df[reduced_names + ["_dur_", "_evt_"]].copy()
        try:
            cph_red = CoxPHFitter()
            cph_red.fit(reduced_df, duration_col="_dur_", event_col="_evt_")
            ll_red = float(cph_red.log_likelihood_)
            lr_stat = 2.0 * (log_lik_full - ll_red)
            df_lr   = len(interaction_names)
            from scipy.stats import chi2 as _chi2
            p_lr    = float(_chi2.sf(lr_stat, df=df_lr))
            interaction_result = {
                "lr_stat": lr_stat,
                "df":      df_lr,
                "p":       p_lr,
                "log_lik_full":    log_lik_full,
                "log_lik_reduced": ll_red,
            }
        except Exception as exc:
            interaction_result = {"error": f"interaction LR fit failed: {exc}"}

    # ── 1D dose-response curves per spline term (covariates at mean,
    # other spline term held at its ref value) ──────────────────────────────
    curves_1d = []
    cov_means = cov_df.values.astype(float).mean(axis=0) if cov_names else np.array([])

    for ti, term in enumerate(req.spline_terms):
        x_raw = term_info[ti]["x_raw"]
        x_lo, x_hi = float(np.percentile(x_raw, 1)), float(np.percentile(x_raw, 99))
        x_syn = np.linspace(x_lo, x_hi, 200)
        sp_syn = _rcs_basis(x_syn, term_info[ti]["knots"])
        this_design = np.column_stack([x_syn, sp_syn])
        # Other term: held at its ref value (linear) + spline basis at that value
        other_idx = 1 - ti if len(term_info) == 2 else None
        other_design = None
        if other_idx is not None:
            other_term = term_info[other_idx]
            ref_x = other_term["ref_value"]
            ref_sp = _rcs_basis(np.array([ref_x]), other_term["knots"]).flatten()
            other_vec = np.concatenate([[ref_x], ref_sp])
            other_design = np.tile(other_vec, (200, 1))

        # Build the full design row for synthetic predictions
        if ti == 0:
            main_syn = this_design if other_design is None else np.column_stack([this_design, other_design])
        else:
            main_syn = other_design if other_design is None else np.column_stack([other_design, this_design]) if other_idx == 0 else None
            if main_syn is None:
                main_syn = np.column_stack([other_design, this_design])

        if req.include_interaction and interaction_design is not None:
            # Recompute interaction columns from synthetic marginals
            a_syn = main_syn[:, :term_info[0]["design"].shape[1]]
            b_syn = main_syn[:, term_info[0]["design"].shape[1]:term_info[0]["design"].shape[1] + term_info[1]["design"].shape[1]]
            ix_syn = np.column_stack([a_syn[:, i] * b_syn[:, j]
                                       for i in range(a_syn.shape[1])
                                       for j in range(b_syn.shape[1])])
            main_syn = np.column_stack([main_syn, ix_syn])

        if cov_names:
            main_syn = np.column_stack([main_syn, np.tile(cov_means, (200, 1))])

        lp_syn = main_syn @ params_full

        # Reference: this term at its ref, other at its ref
        own_ref = term_info[ti]["ref_value"]
        ref_row = np.zeros_like(main_syn[0])
        # Build a reference row by replicating the synthetic-row pattern with
        # this term set to own_ref:
        ref_idx_syn = int(np.argmin(np.abs(x_syn - own_ref)))
        ref_row = main_syn[ref_idx_syn].copy()

        diffs   = main_syn - ref_row
        var_lp  = np.einsum("ij,jk,ik->i", diffs, cov_full, diffs)
        se_lp   = np.sqrt(np.maximum(var_lp, 0))
        rel_lp  = lp_syn - lp_syn[ref_idx_syn]
        hr      = np.exp(rel_lp)
        ci_low  = np.exp(rel_lp - 1.96 * se_lp)
        ci_high = np.exp(rel_lp + 1.96 * se_lp)

        def _cln(arr):
            out = []
            for v in arr:
                fv = float(v)
                out.append(None if (np.isnan(fv) or np.isinf(fv)) else round(fv, 4))
            return out

        curves_1d.append({
            "column":   term.column,
            "x":        _cln(x_syn),
            "hr":       _cln(hr),
            "lower":    _cln(ci_low),
            "upper":    _cln(ci_high),
            "knots":    [round(float(k), 2) for k in term_info[ti]["knots"]],
            "ref":      round(float(own_ref), 4),
        })

    # ── 2D HR surface for interaction ───────────────────────────────────────
    surface_2d = None
    if req.include_interaction and interaction_design is not None and len(term_info) == 2:
        g = max(10, min(int(req.grid_size or 50), 100))
        xa = term_info[0]["x_raw"]; xb = term_info[1]["x_raw"]
        a_lo, a_hi = float(np.percentile(xa, 1)), float(np.percentile(xa, 99))
        b_lo, b_hi = float(np.percentile(xb, 1)), float(np.percentile(xb, 99))
        a_grid = np.linspace(a_lo, a_hi, g)
        b_grid = np.linspace(b_lo, b_hi, g)
        A, B = np.meshgrid(a_grid, b_grid)  # B is rows, A is cols
        a_flat = A.flatten(); b_flat = B.flatten()
        a_basis = np.column_stack([a_flat, _rcs_basis(a_flat, term_info[0]["knots"])])
        b_basis = np.column_stack([b_flat, _rcs_basis(b_flat, term_info[1]["knots"])])
        ix_flat = np.column_stack([a_basis[:, i] * b_basis[:, j]
                                    for i in range(a_basis.shape[1])
                                    for j in range(b_basis.shape[1])])
        cov_block = np.tile(cov_means, (a_flat.size, 1)) if cov_names else np.empty((a_flat.size, 0))
        design = np.column_stack([a_basis, b_basis, ix_flat, cov_block])
        lp = design @ params_full
        # Reference row: both at their ref values
        ref_a = term_info[0]["ref_value"]
        ref_b = term_info[1]["ref_value"]
        ra_basis = np.column_stack([[ref_a], _rcs_basis(np.array([ref_a]), term_info[0]["knots"])])
        rb_basis = np.column_stack([[ref_b], _rcs_basis(np.array([ref_b]), term_info[1]["knots"])])
        rix = np.column_stack([ra_basis[:, i] * rb_basis[:, j]
                                for i in range(ra_basis.shape[1])
                                for j in range(rb_basis.shape[1])])
        rcov = np.tile(cov_means, (1, 1)) if cov_names else np.empty((1, 0))
        ref_design = np.column_stack([ra_basis, rb_basis, rix, rcov])
        lp_ref = float((ref_design @ params_full)[0])
        hr_flat = np.exp(lp - lp_ref)
        hr_grid = hr_flat.reshape(B.shape)

        def _gclean(mat):
            out = []
            for row in mat:
                rrow = []
                for v in row:
                    fv = float(v)
                    rrow.append(None if (np.isnan(fv) or np.isinf(fv)) else round(fv, 4))
                out.append(rrow)
            return out

        surface_2d = {
            "x_col": term_info[0]["column"],
            "y_col": term_info[1]["column"],
            "x":     [round(float(v), 4) for v in a_grid],
            "y":     [round(float(v), 4) for v in b_grid],
            "hr":    _gclean(hr_grid),
            "ref":   {term_info[0]["column"]: round(float(ref_a), 4),
                      term_info[1]["column"]: round(float(ref_b), 4)},
        }

    # AIC (partial likelihood)
    aic_partial = None
    try:
        aic_partial = float(getattr(cph_full, "AIC_partial_", np.nan))
        if np.isnan(aic_partial):
            aic_partial = None
    except Exception:
        aic_partial = None

    return {
        "n":              int(n),
        "n_events":       int(event.sum()),
        "concordance":    float(cph_full.concordance_index_),
        "log_likelihood": log_lik_full,
        "aic":            aic_partial,
        "spline_terms": [
            {
                "column":         t.column,
                "n_knots":        t.n_knots,
                "knots":          [round(float(k), 2) for k in term_info[i]["knots"]],
                "knot_positions_custom": t.knot_positions is not None,
                "ref":            round(float(term_info[i]["ref_value"]), 4),
            }
            for i, t in enumerate(req.spline_terms)
        ],
        "covariates":     req.covariates,
        "include_interaction": req.include_interaction,
        "coefficients":   coefs,
        "nonlinearity":   nonlinearity,
        "interaction":    interaction_result,
        "curves_1d":      curves_1d,
        "surface_2d":     surface_2d,
    }


# ── Polynomial / Non-linear Regression ───────────────────────────────────────

class PolynomialRequest(BaseModel):
    session_id: str
    outcome: str
    predictor: str
    degree: int = 2          # 1–5
    covariates: List[str] = []
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/polynomial")
def polynomial_regression(req: PolynomialRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    cols = [req.outcome, req.predictor] + req.covariates
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    if req.degree < 1 or req.degree > 10:
        raise HTTPException(status_code=422, detail="Polynomial degree must be between 1 and 10.")

    x = df[req.predictor].astype(float)
    n_unique = x.nunique()
    if n_unique <= req.degree:
        raise HTTPException(status_code=422, detail=f"Predictor has only {n_unique} unique values — need more than degree ({req.degree}) for polynomial fit.")

    X_parts = {"const": np.ones(len(df))}
    for d in range(1, req.degree + 1):
        X_parts[f"{req.predictor}^{d}"] = x ** d
    for cov in req.covariates:
        X_parts[cov] = df[cov].astype(float)
    X = pd.DataFrame(X_parts)
    y = df[req.outcome].astype(float)

    base = sm.OLS(y, X)
    model = base.fit(cov_type="HC3", use_t=True) if req.robust_se else base.fit()
    ci = model.conf_int()

    coefs = []
    for var in model.params.index:
        coefs.append({
            "variable": str(var),
            "estimate": float(model.params[var]),
            "se": float(model.bse[var]),
            "t": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
        })

    # Curve for plotting (hold covariates at mean)
    x_lo, x_hi = float(x.min()), float(x.max())
    xs = np.linspace(x_lo, x_hi, 200)
    X_curve = np.column_stack([xs ** d for d in range(0, req.degree + 1)])
    cov_means = [float(df[c].mean()) for c in req.covariates]
    if cov_means:
        X_curve = np.hstack([X_curve, np.tile(cov_means, (len(xs), 1))])
    pred = model.get_prediction(X_curve)
    yhat = pred.predicted_mean
    ci_df = pred.conf_int()

    return {
        "model": f"Polynomial Regression (degree {req.degree}){' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "predictor": req.predictor,
        "degree": req.degree,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "residual_se": float(np.sqrt(model.mse_resid)),
        "coefficients": coefs,
        "curve": {
            "x": xs.tolist(),
            "y": yhat.tolist(),
            "ci_low":  ci_df[:, 0].tolist(),
            "ci_high": ci_df[:, 1].tolist(),
        },
        "scatter": {
            "x": x.tolist()[:2000],
            "y": y.tolist()[:2000],
        },
    }


# ── Linear Mixed Model (LMM) / GLMM auto-router ──────────────────────────────

class LMMRequest(BaseModel):
    session_id: str
    outcome: str
    fixed_effects: List[str]
    group_col: str
    imputation: Optional[str] = "listwise"


def _is_id_like(col: str, series: "pd.Series") -> bool:
    """Heuristic: column is likely a patient/subject identifier."""
    name_lower = col.lower()
    # Name-based check
    name_match = any(name_lower == tok or name_lower.endswith(tok) or name_lower.startswith(tok)
                     for tok in ("id", "no", "num", "number", "patient", "subject", "case", "record"))
    if name_match:
        return True
    # Value-based check: near-unique integers
    n = len(series.dropna())
    if n < 5:
        return False
    try:
        nunique = series.nunique()
        return (nunique / n) > 0.95 and pd.api.types.is_integer_dtype(series)
    except Exception:
        return False


@router.post("/lmm")
def linear_mixed_model(req: LMMRequest):
    import re
    import statsmodels.formula.api as smf

    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    # ── Guard: ID column as fixed effect ────────────────────────────────────
    id_in_fe = [c for c in req.fixed_effects if _is_id_like(c, df_full.get(c, pd.Series()))]
    if id_in_fe:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Column(s) {id_in_fe} look like patient/subject identifiers and cannot be fixed effects. "
                "Assign them as the Grouping variable (random intercept) instead."
            ),
        )

    cols = [req.outcome, req.group_col] + req.fixed_effects
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    # ── Detect binary outcome → route to GEE ────────────────────────────────
    outcome_vals = df[req.outcome].dropna().unique()
    is_binary = set(outcome_vals.tolist()) <= {0, 1, 0.0, 1.0}

    # Sanitize column names for formula
    def safe(c: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_]", "_", c)

    rename = {c: safe(c) for c in cols}
    df_r = df.rename(columns=rename)
    outcome_s = safe(req.outcome)
    group_s   = safe(req.group_col)
    fe_s      = [safe(f) for f in req.fixed_effects]
    formula   = f"{outcome_s} ~ " + (" + ".join(fe_s) if fe_s else "1")

    if is_binary:
        # ── GEE with Binomial/Logit — population-averaged GLMM alternative ──
        # statsmodels MixedLM does not support binomial; GEE is the standard
        # alternative for clustered binary outcomes (population-averaged effects).
        import statsmodels.api as sm_api
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Independence

        gee_model = GEE.from_formula(
            formula, group_s, data=df_r,
            family=Binomial(),
            cov_struct=Independence(),
        )
        result = gee_model.fit()
        ci = result.conf_int()
        coefs = []
        for var in result.params.index:
            p_val = float(result.pvalues[var])
            est   = float(result.params[var])
            coefs.append({
                "variable": str(var),
                "estimate": round(est, 6),
                "exp_estimate": round(float(np.exp(est)), 4),   # Odds Ratio
                "se": round(float(result.bse[var]), 6),
                "z": round(float(result.tvalues[var]), 4),
                "p": round(p_val, 6),
                "ci_low":  round(float(ci.loc[var, 0]), 4),
                "ci_high": round(float(ci.loc[var, 1]), 4),
                "or_low":  round(float(np.exp(ci.loc[var, 0])), 4),
                "or_high": round(float(np.exp(ci.loc[var, 1])), 4),
            })
        return {
            "model": "GEE — Binomial/Logit (Binary outcome)",
            "model_type": "gee_binomial",
            "note": (
                "Binary outcome detected. Fitted using Generalized Estimating Equations (GEE) "
                "with Binomial family and logit link — the population-averaged equivalent of a GLMM. "
                "Estimates are log-odds (logit scale); exp(β) = Odds Ratio."
            ),
            "outcome": req.outcome,
            "group": req.group_col,
            "n": int(result.nobs),
            "n_groups": int(df[req.group_col].nunique()),
            "n_excluded": n_excluded,
            "aic": float(result.aic) if hasattr(result, "aic") else None,
            "bic": float(result.bic) if hasattr(result, "bic") else None,
            "log_likelihood": float(result.llf) if hasattr(result, "llf") else None,
            "random_effect_variance": None,
            "residual_variance": None,
            "icc": None,
            "coefficients": coefs,
        }

    # ── Standard LMM (REML) for continuous outcomes ──────────────────────────
    model = smf.mixedlm(formula, df_r, groups=df_r[group_s]).fit(reml=True)

    fe_ci = model.conf_int()
    coefs = []
    for var in model.fe_params.index:
        coefs.append({
            "variable": str(var),
            "estimate": float(model.fe_params[var]),
            "se": float(model.bse_fe[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(fe_ci.loc[var, 0]),
            "ci_high": float(fe_ci.loc[var, 1]),
        })

    re_var = float(model.cov_re.iloc[0, 0]) if model.cov_re is not None and model.cov_re.size > 0 else None
    residual_var = float(model.scale)

    return {
        "model": "Linear Mixed Model (REML)",
        "model_type": "lmm",
        "outcome": req.outcome,
        "group": req.group_col,
        "n": int(model.nobs),
        "n_groups": int(df[req.group_col].nunique()),
        "n_excluded": n_excluded,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
        "random_effect_variance": re_var,
        "residual_variance": residual_var,
        "icc": (re_var / (re_var + residual_var)) if re_var is not None else None,
        "coefficients": coefs,
    }


# ── Wide → Long melt (repeated measures reshape) ──────────────────────────────

class MeltRequest(BaseModel):
    session_id: str
    id_col: str                  # e.g. "PatientID"
    value_cols: List[str]        # e.g. ["INHOSPITALEF", "EF", "CONTROLEF"]
    time_var_name: str = "TimePoint"
    value_var_name: str = "Value"
    time_labels: Optional[List[str]] = None  # custom labels; defaults to col names


@router.post("/melt")
def melt_wide_to_long(req: MeltRequest):
    """Reshape wide-format repeated measures into long format and save back to session."""
    df = _get_df(req.session_id)
    missing = [c for c in [req.id_col] + req.value_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing}")
    if len(req.value_cols) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 value columns to melt")

    labels = req.time_labels if req.time_labels and len(req.time_labels) == len(req.value_cols) \
             else req.value_cols

    # Keep other columns (non-melted) as covariates in the long frame
    other_cols = [c for c in df.columns if c not in req.value_cols and c != req.id_col]
    # Limit other cols to avoid explosion
    keep = [req.id_col] + req.value_cols + other_cols[:20]
    df_sub = df[[c for c in keep if c in df.columns]].copy()

    df_long = df_sub.melt(
        id_vars=[c for c in df_sub.columns if c not in req.value_cols],
        value_vars=req.value_cols,
        var_name=req.time_var_name,
        value_name=req.value_var_name,
    )
    # Replace column names with readable labels
    label_map = dict(zip(req.value_cols, labels))
    df_long[req.time_var_name] = df_long[req.time_var_name].map(label_map)

    # Persist the long-format DataFrame back to the session store
    from ..services.store import store
    store.save(req.session_id, df_long)

    return {
        "rows": len(df_long),
        "columns": list(df_long.columns),
        "time_var": req.time_var_name,
        "value_var": req.value_var_name,
        "time_points": labels,
        "preview": df_long.head(10).to_dict(orient="records"),
    }


# ── Gamma GLM ─────────────────────────────────────────────────────────────────

class GammaRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    link: str = "log"        # "log" | "identity" | "inverse"
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/gamma")
def gamma_regression(req: GammaRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if (y.dropna() <= 0).any():
        raise HTTPException(status_code=422, detail="Gamma regression requires strictly positive outcomes (> 0). Non-positive values found.")

    valid_links = {"log", "identity", "inverse"}
    if req.link and req.link not in valid_links:
        raise HTTPException(status_code=422, detail=f"Invalid link function '{req.link}'. Valid: {valid_links}")
    link_map = {"log": sm.families.links.Log(), "identity": sm.families.links.Identity(), "inverse": sm.families.links.InversePower()}
    family = sm.families.Gamma(link=link_map.get(req.link, sm.families.links.Log()))
    cov_type = "HC3" if req.robust_se else "nonrobust"
    model = sm.GLM(y, X, family=family).fit(cov_type=cov_type)
    ci = model.conf_int()

    coefs = []
    for var in model.params.index:
        b = float(model.params[var])
        coefs.append({
            "variable": str(var),
            "estimate": b,
            "exp_estimate": float(np.exp(b)) if req.link == "log" else None,
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
        })

    return {
        "model": f"Gamma GLM (link={req.link}){' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "link": req.link,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "deviance": float(model.deviance),
        "scale": float(model.scale),
        "coefficients": coefs,
    }


# ── Negative Binomial GLM ─────────────────────────────────────────────────────

class NegBinomRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/negbinom")
def negative_binomial_regression(req: NegBinomRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if (y.dropna() < 0).any():
        raise HTTPException(status_code=422, detail="Negative binomial requires non-negative integer counts.")
    if (y.dropna() % 1 != 0).any():
        raise HTTPException(status_code=422, detail="Negative binomial requires integer counts. Fractional values found.")
    cov_type = "HC3" if req.robust_se else "nonrobust"
    # Estimate alpha (dispersion) from Poisson residuals instead of fixed alpha=1
    try:
        poisson_fit = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        mu = poisson_fit.mu
        alpha_est = max(1e-6, float(((((y - mu) ** 2 - mu) / mu ** 2).mean())))
    except Exception:
        alpha_est = 1.0
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_est)).fit(cov_type=cov_type)
    ci = model.conf_int()

    coefs = []
    for var in model.params.index:
        b = float(model.params[var])
        coefs.append({
            "variable": str(var),
            "log_irr": b,
            "irr": float(np.exp(b)),
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
            "irr_ci_low":  float(np.exp(ci.loc[var, 0])),
            "irr_ci_high": float(np.exp(ci.loc[var, 1])),
        })

    return {
        "model": f"Negative Binomial Regression{' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "deviance": float(model.deviance),
        "coefficients": coefs,
    }


# ── Linear Regression Diagnostic Plots ────────────────────────────────────────

class DiagRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"


@router.post("/linear_diag")
def linear_diagnostics(req: DiagRequest):
    from scipy import stats as scipy_stats

    df_full = _get_df(req.session_id)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = df[req.outcome].astype(float)
    model = sm.OLS(y, X).fit()

    fitted   = model.fittedvalues.values
    resid    = model.resid.values
    std_res  = model.get_influence().resid_studentized_internal
    sqrt_abs = np.sqrt(np.abs(std_res))

    # QQ data
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(resid, dist="norm")
    qq_x_line = np.array([min(osm), max(osm)])
    qq_y_line  = slope * qq_x_line + intercept

    # Subsample for large datasets
    N = min(len(fitted), 2000)
    idx = np.random.choice(len(fitted), N, replace=False) if len(fitted) > N else np.arange(N)

    return {
        "residuals_fitted": {
            "x": fitted[idx].tolist(),
            "y": resid[idx].tolist(),
        },
        "qq": {
            "theoretical": osm[idx[:len(osm)]].tolist() if len(osm) > N else osm.tolist(),
            "sample":      osr[idx[:len(osr)]].tolist() if len(osr) > N else osr.tolist(),
            "line_x":      qq_x_line.tolist(),
            "line_y":      qq_y_line.tolist(),
        },
        "scale_location": {
            "x": fitted[idx].tolist(),
            "y": sqrt_abs[idx].tolist(),
        },
        "r_squared": float(model.rsquared),
        "residual_se": float(np.sqrt(model.mse_resid)),
        "n": int(model.nobs),
    }


# ── Propensity Score Matching (PSM) ───────────────────────────────────────────

class PSMRequest(BaseModel):
    session_id: str
    treatment_col: str
    covariates: List[str]
    outcome_col: Optional[str] = None
    caliper: Optional[float] = 0.2        # fraction of SD (of logit-PS if caliper_scale='logit', else PS)
    caliper_scale: Optional[str] = "logit"  # 'logit' (Austin 2011) or 'raw'
    ratio: Optional[int] = 1             # 1:ratio matching (1:1 default)
    imputation: Optional[str] = "listwise"
    trim_common_support: Optional[bool] = False  # Crump 2009 trimming
    random_state: Optional[int] = 42     # reproducibility for LR solver tie-breaking
    # Score-model alternatives
    score_method: Optional[str] = "logistic"   # 'logistic' | 'probit' | 'gbm'
    # Matching method
    matching_method: Optional[str] = "greedy"  # 'greedy' (NN+caliper) | 'optimal' (Hungarian, 1:1 only)
    # Exact-match strata (categorical columns that must agree before NN)
    exact_match: Optional[List[str]] = None
    # Outcome handling
    outcome_type: Optional[str] = "binary"     # 'binary' (default) | 'survival'
    survival_duration_col: Optional[str] = None
    survival_event_col:    Optional[str] = None
    # Sensitivity analysis
    compute_rosenbaum: Optional[bool] = False  # Rosenbaum bounds (1:1 binary only)
    rosenbaum_gamma_max: Optional[float] = 3.0


def _smd_columns(s_treated: pd.Series, s_control: pd.Series) -> tuple[pd.Series, pd.Series, bool]:
    """Coerce two covariate series to numeric (label-encoded if categorical).

    Returns (treated_numeric, control_numeric, is_binary).
    """
    if s_treated.dtype == object or str(s_treated.dtype).startswith("category"):
        combined = pd.concat([s_treated, s_control]).dropna()
        cats = sorted(combined.unique().tolist(), key=str)
        cat_map = {c: i for i, c in enumerate(cats)}
        s_treated = s_treated.map(cat_map)
        s_control = s_control.map(cat_map)

    s_treated = pd.to_numeric(s_treated, errors="coerce").dropna()
    s_control = pd.to_numeric(s_control, errors="coerce").dropna()

    is_binary = pd.concat([s_treated, s_control]).nunique() <= 2
    return s_treated, s_control, is_binary


def _compute_smd(s_treated: pd.Series, s_control: pd.Series,
                 denom_sd: Optional[float] = None) -> float:
    """Standardized Mean Difference for one covariate.

    Args:
        denom_sd: If supplied, uses this pre-computed pooled SD (Austin 2011
            convention: pooled SD from the UNMATCHED sample is the denominator
            both before and after matching, so that the change reflects only
            the numerator shift). Defaults to in-sample pooled SD when None.
    """
    s_treated, s_control, is_binary = _smd_columns(s_treated, s_control)
    if len(s_treated) == 0 or len(s_control) == 0:
        return 0.0

    if is_binary:
        p1 = float(s_treated.mean())
        p0 = float(s_control.mean())
        denom = denom_sd if denom_sd is not None else np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
        return float(abs(p1 - p0) / denom) if denom > 1e-9 else 0.0

    m1, m0 = float(s_treated.mean()), float(s_control.mean())
    if denom_sd is None:
        sd1, sd0 = float(s_treated.std(ddof=1)), float(s_control.std(ddof=1))
        denom_sd = np.sqrt((sd1 ** 2 + sd0 ** 2) / 2)
    return float(abs(m1 - m0) / denom_sd) if denom_sd > 1e-9 else 0.0


def _pooled_sd(s_treated: pd.Series, s_control: pd.Series) -> float:
    """Pooled SD denominator (Austin 2011) from the unmatched sample."""
    s_treated, s_control, is_binary = _smd_columns(s_treated, s_control)
    if len(s_treated) == 0 or len(s_control) == 0:
        return 0.0
    if is_binary:
        p1 = float(s_treated.mean()); p0 = float(s_control.mean())
        return float(np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2))
    sd1 = float(s_treated.std(ddof=1)); sd0 = float(s_control.std(ddof=1))
    return float(np.sqrt((sd1 ** 2 + sd0 ** 2) / 2))


def _variance_ratio(s_treated: pd.Series, s_control: pd.Series) -> Optional[float]:
    """Rubin's variance ratio σ²_treated / σ²_control. Target range 0.5–2.0."""
    s_treated, s_control, is_binary = _smd_columns(s_treated, s_control)
    if is_binary or len(s_treated) < 2 or len(s_control) < 2:
        return None
    v1 = float(s_treated.var(ddof=1)); v0 = float(s_control.var(ddof=1))
    if v0 <= 1e-12:
        return None
    return round(v1 / v0, 4)


def _ks_p(s_treated: pd.Series, s_control: pd.Series) -> Optional[float]:
    """Two-sample KS test p-value for distributional balance."""
    s_treated, s_control, is_binary = _smd_columns(s_treated, s_control)
    if is_binary or len(s_treated) < 2 or len(s_control) < 2:
        return None
    from scipy.stats import ks_2samp
    try:
        _, p = ks_2samp(s_treated.values, s_control.values)
        return round(float(p), 6)
    except Exception:
        return None


def _fit_propensity_scores(X_scaled: np.ndarray, y: np.ndarray,
                            method: str, random_state: Optional[int]) -> np.ndarray:
    """Fit propensity scores using one of several models.

    Args:
        X_scaled: standardized design matrix.
        y: binary treatment vector.
        method: 'logistic' | 'probit' | 'gbm'.
        random_state: seed for reproducibility.

    Returns:
        1D array of PS = P(treatment = 1 | covariates) for each row.
    """
    method = (method or "logistic").lower()
    if method == "logistic":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0,
                               random_state=random_state)
        m.fit(X_scaled, y)
        return m.predict_proba(X_scaled)[:, 1]
    if method == "probit":
        import statsmodels.api as _sm
        X_const = _sm.add_constant(X_scaled, has_constant="add")
        m = _sm.Probit(y, X_const).fit(disp=False, maxiter=200)
        return np.asarray(m.predict(X_const))
    if method == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier
        m = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=random_state,
        )
        m.fit(X_scaled, y)
        return m.predict_proba(X_scaled)[:, 1]
    raise HTTPException(status_code=422, detail=f"Unknown score_method: {method}")


def _match_greedy(
    treated_idx: np.ndarray,
    control_idx: np.ndarray,
    distance_vec: np.ndarray,
    caliper_dist: float,
    ratio: int,
) -> tuple[list[int], list[int]]:
    """Greedy nearest-neighbour matching with caliper. Hardest-first order.

    Returns:
        (matched_treated, matched_controls) — flat lists of row indices.
        matched_controls has length ratio * len(matched_treated); each
        consecutive block of `ratio` controls belongs to one treated unit.
    """
    from sklearn.neighbors import NearestNeighbors

    matched_t: list[int] = []
    matched_c: list[int] = []
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return matched_t, matched_c

    ctrl_dist = distance_vec[control_idx].reshape(-1, 1)
    knn = NearestNeighbors(n_neighbors=min(ratio * 5, len(control_idx)), metric="euclidean")
    knn.fit(ctrl_dist)

    used: set[int] = set()
    ordered = treated_idx[np.argsort(-distance_vec[treated_idx])]
    for ti in ordered:
        q = np.array([[distance_vec[ti]]])
        distances, neighbours = knn.kneighbors(q)
        chosen: list[int] = []
        for dist, nb in zip(distances[0], neighbours[0]):
            c_real = int(control_idx[nb])
            if dist <= caliper_dist and c_real not in used:
                chosen.append(c_real)
                used.add(c_real)
                if len(chosen) == ratio:
                    break
        if len(chosen) == ratio:
            matched_t.append(int(ti))
            matched_c.extend(chosen)
    return matched_t, matched_c


def _match_optimal(
    treated_idx: np.ndarray,
    control_idx: np.ndarray,
    distance_vec: np.ndarray,
    caliper_dist: float,
) -> tuple[list[int], list[int]]:
    """Optimal 1:1 matching via Hungarian algorithm (minimises total distance).

    Distances above the caliper are set to +inf so they are never selected.
    If a treated unit has no feasible control under the caliper, it is dropped
    after the assignment (the assignment may have paired it nominally, but
    we check the cost and discard).
    """
    from scipy.optimize import linear_sum_assignment

    if len(treated_idx) == 0 or len(control_idx) == 0:
        return [], []

    dt = distance_vec[treated_idx]
    dc = distance_vec[control_idx]
    cost = np.abs(dt[:, None] - dc[None, :])
    # Caliper enforcement: anything beyond the caliper is unmatchable
    cost = np.where(cost <= caliper_dist, cost, np.inf)

    # Pad columns when controls < treated so the assignment is feasible.
    if cost.shape[1] < cost.shape[0]:
        pad = np.full((cost.shape[0], cost.shape[0] - cost.shape[1]), np.inf)
        cost_padded = np.hstack([cost, pad])
    else:
        cost_padded = cost

    # Replace remaining inf with a very large finite cost so the LSA solver
    # converges; we filter infeasible pairs after the fact.
    big = np.nanmax(cost_padded[np.isfinite(cost_padded)], initial=0.0) * 10.0 + 1.0
    work = np.where(np.isinf(cost_padded), big, cost_padded)
    row_ind, col_ind = linear_sum_assignment(work)

    matched_t: list[int] = []
    matched_c: list[int] = []
    for r, c in zip(row_ind, col_ind):
        if c >= cost.shape[1]:
            continue  # treated paired to a padding column = unmatched
        if not np.isfinite(cost[r, c]):
            continue
        matched_t.append(int(treated_idx[r]))
        matched_c.append(int(control_idx[c]))
    return matched_t, matched_c


def _run_match_strata(
    df: pd.DataFrame,
    treated_idx: np.ndarray,
    control_idx: np.ndarray,
    distance_vec: np.ndarray,
    caliper_dist: float,
    ratio: int,
    method: str,
    exact_match_cols: Optional[List[str]],
) -> tuple[list[int], list[int]]:
    """Run matching, partitioning by exact-match strata when requested."""
    match = _match_optimal if method == "optimal" else _match_greedy
    if not exact_match_cols:
        if method == "optimal" and ratio > 1:
            # Hungarian here is 1:1 only. Fall back to greedy for higher ratios.
            return _match_greedy(treated_idx, control_idx, distance_vec, caliper_dist, ratio)
        if method == "optimal":
            return match(treated_idx, control_idx, distance_vec, caliper_dist)
        return match(treated_idx, control_idx, distance_vec, caliper_dist, ratio)

    # Group by the exact-match key tuple
    keys = df[exact_match_cols].astype(str).agg("||".join, axis=1).values
    matched_t: list[int] = []
    matched_c: list[int] = []
    treated_keys = keys[treated_idx]
    control_keys = keys[control_idx]
    for key in pd.unique(treated_keys):
        t_sub = treated_idx[treated_keys == key]
        c_sub = control_idx[control_keys == key]
        if len(c_sub) == 0:
            continue
        if method == "optimal" and ratio == 1:
            mt, mc = match(t_sub, c_sub, distance_vec, caliper_dist)
        else:
            # Greedy (or optimal-with-ratio-fallback)
            mt, mc = _match_greedy(t_sub, c_sub, distance_vec, caliper_dist, ratio)
        matched_t.extend(mt)
        matched_c.extend(mc)
    return matched_t, matched_c


def _rosenbaum_bounds(
    pair_outcomes: list[tuple[int, int]],
    gamma_max: float = 3.0,
    n_gamma: int = 60,
    alpha: float = 0.05,
) -> dict:
    """Rosenbaum bounds for 1:1 matched pairs with binary outcomes.

    For each matched pair, take (treated_outcome, control_outcome) ∈ {0,1}².
    Discordant pairs are those where the two outcomes differ — call b the count
    of (treated=1, control=0) and c the count of (treated=0, control=1).

    Under H0 (no treatment effect) and no hidden bias, b ~ Binomial(b+c, 0.5).
    With hidden bias Γ ≥ 1, the worst-case upper bound on P(b ≥ b_obs) uses
    p+ = Γ / (1 + Γ) (and the lower bound uses p- = 1 / (1 + Γ)).

    Returns the largest Γ for which the upper-bound one-sided p-value remains
    ≤ alpha; this is the standard Rosenbaum critical Γ.
    """
    from scipy.stats import binom

    b = sum(1 for t, c in pair_outcomes if t == 1 and c == 0)
    c = sum(1 for t, c in pair_outcomes if t == 0 and c == 1)
    discordant = b + c
    if discordant == 0:
        return {"applicable": False, "reason": "No discordant pairs.", "b": b, "c": c}
    b_obs = max(b, c)

    p_at_no_bias = float(binom.sf(b_obs - 1, discordant, 0.5))
    gammas = np.linspace(1.0, max(1.0, gamma_max), max(2, int(n_gamma)))
    rows = []
    crit_gamma: Optional[float] = None
    for g in gammas:
        p_plus = g / (1.0 + g)
        p_upper = float(binom.sf(b_obs - 1, discordant, p_plus))
        rows.append({"gamma": round(float(g), 3), "p_upper": round(p_upper, 6)})
        if crit_gamma is None and p_upper > alpha:
            crit_gamma = round(float(g), 3)
    return {
        "applicable": True,
        "b": b,
        "c": c,
        "discordant_pairs": discordant,
        "p_unbiased": round(p_at_no_bias, 6),
        "critical_gamma": crit_gamma,
        "alpha": alpha,
        "gamma_max": gamma_max,
        "curve": rows,
    }


@router.post("/psm")
def propensity_score_matching(req: PSMRequest):
    import traceback
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    try:
        return _run_psm(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def _run_psm(req: PSMRequest):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    df_full = _get_df(req.session_id)
    outcome_type = (req.outcome_type or "binary").lower()
    if outcome_type not in ("binary", "survival"):
        raise HTTPException(status_code=422, detail="outcome_type must be 'binary' or 'survival'.")

    extra_outcome_cols: List[str] = []
    if outcome_type == "binary" and req.outcome_col:
        extra_outcome_cols.append(req.outcome_col)
    if outcome_type == "survival":
        if not req.survival_duration_col or not req.survival_event_col:
            raise HTTPException(status_code=422, detail="Survival outcome requires survival_duration_col and survival_event_col.")
        extra_outcome_cols.extend([req.survival_duration_col, req.survival_event_col])

    exact_match_cols = list(req.exact_match or [])
    needed = list(dict.fromkeys(
        [req.treatment_col] + req.covariates + extra_outcome_cols + exact_match_cols
    ))
    missing_cols = [c for c in needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing_cols}")

    df = apply_imputation(df_full[needed], needed, req.imputation or "listwise").reset_index(drop=True)

    # Validate treatment is binary 0/1
    treat_vals = df[req.treatment_col].astype(float)
    if not set(treat_vals.unique().tolist()) <= {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422,
            detail=f"Treatment column '{req.treatment_col}' must be binary (0 = control, 1 = treated).")

    # ── Step 1: Propensity scores (logistic / probit / GBM) ──────────────────
    X = pd.get_dummies(df[req.covariates], drop_first=True).astype(float)
    y = treat_vals.astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    score_method = (req.score_method or "logistic").lower()
    ps = _fit_propensity_scores(X_scaled, y, score_method, req.random_state)
    # Logit of PS — Austin 2011 recommends matching on this scale because the
    # raw PS is bounded [0,1] and gets compressed near the tails.
    ps_clip = np.clip(ps, 1e-6, 1 - 1e-6)
    logit_ps = np.log(ps_clip / (1.0 - ps_clip))

    df = df.copy()
    df["_ps_"] = ps
    df["_logit_ps_"] = logit_ps
    df["_treat_"] = y

    # ── Step 2: Optional Crump 2009 common-support trim ──────────────────────
    treated_idx_all = np.where(y == 1)[0]
    control_idx_all = np.where(y == 0)[0]
    if len(treated_idx_all) == 0 or len(control_idx_all) == 0:
        raise HTTPException(status_code=422, detail="Need both treated (1) and control (0) patients.")

    support_lo, support_hi = float(ps.min()), float(ps.max())
    n_trimmed = 0
    keep_mask = np.ones_like(y, dtype=bool)
    if req.trim_common_support:
        support_lo = max(ps[treated_idx_all].min(), ps[control_idx_all].min())
        support_hi = min(ps[treated_idx_all].max(), ps[control_idx_all].max())
        keep_mask = (ps >= support_lo) & (ps <= support_hi)
        n_trimmed = int((~keep_mask).sum())

    treated_idx = np.where((y == 1) & keep_mask)[0]
    control_idx = np.where((y == 0) & keep_mask)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise HTTPException(status_code=422, detail="No units remain after common-support trim. Disable trimming or widen the support.")

    # ── Step 3: Matching (greedy NN or optimal Hungarian) with caliper ──────
    scale = (req.caliper_scale or "logit").lower()
    if scale not in ("logit", "raw"):
        raise HTTPException(status_code=422, detail="caliper_scale must be 'logit' or 'raw'.")
    distance_vec = logit_ps if scale == "logit" else ps
    caliper_sd = float(distance_vec[keep_mask].std())
    caliper_dist = (req.caliper or 0.2) * caliper_sd
    ratio = max(1, req.ratio or 1)

    matching_method = (req.matching_method or "greedy").lower()
    if matching_method not in ("greedy", "optimal"):
        raise HTTPException(status_code=422, detail="matching_method must be 'greedy' or 'optimal'.")
    matching_warning: Optional[str] = None
    if matching_method == "optimal" and ratio != 1:
        matching_warning = "Optimal (Hungarian) matching supports 1:1 only. Falling back to greedy for ratio > 1."
        matching_method = "greedy"

    # Validate exact_match columns exist
    if exact_match_cols:
        bad = [c for c in exact_match_cols if c not in df.columns]
        if bad:
            raise HTTPException(status_code=422, detail=f"exact_match columns not found: {bad}")

    matched_treated, matched_controls = _run_match_strata(
        df=df,
        treated_idx=treated_idx,
        control_idx=control_idx,
        distance_vec=distance_vec,
        caliper_dist=caliper_dist,
        ratio=ratio,
        method=matching_method,
        exact_match_cols=exact_match_cols if exact_match_cols else None,
    )

    n_matched_treated = len(matched_treated)
    n_matched_controls = len(matched_controls)

    if n_matched_treated == 0:
        raise HTTPException(status_code=422,
            detail=f"No matches found within caliper {req.caliper}. "
                   "Try widening the caliper or check that treatment groups overlap in covariate space.")

    matched_all_idx = matched_treated + matched_controls
    df_matched = df.iloc[matched_all_idx].copy()

    # Assign match-set IDs for downstream paired/clustered analysis
    match_ids = []
    for i, ti in enumerate(matched_treated):
        match_ids.append(i)  # treated
    for i in range(len(matched_controls)):
        match_ids.append(i // ratio)  # controls get same match_id as their treated pair
    df_matched["_match_id_"] = match_ids

    # ── Step 4: SMD before and after matching ────────────────────────────────
    # Austin 2011: use the pooled SD from the UNMATCHED sample as the common
    # denominator for both before and after, so the change reflects only the
    # numerator shift (not a moving-target SD).
    smd_before, smd_after = {}, {}
    var_ratio_before, var_ratio_after = {}, {}
    ks_before, ks_after = {}, {}
    treat_mask = df["_treat_"].values
    for cov in req.covariates:
        col   = df[cov]
        col_m = df_matched[cov]
        denom = _pooled_sd(col[treat_mask == 1], col[treat_mask == 0])
        smd_before[cov] = round(_compute_smd(col[treat_mask == 1], col[treat_mask == 0],
                                              denom_sd=denom), 4)
        smd_after[cov]  = round(_compute_smd(col_m[df_matched["_treat_"] == 1],
                                              col_m[df_matched["_treat_"] == 0],
                                              denom_sd=denom), 4)
        var_ratio_before[cov] = _variance_ratio(col[treat_mask == 1], col[treat_mask == 0])
        var_ratio_after[cov]  = _variance_ratio(col_m[df_matched["_treat_"] == 1],
                                                 col_m[df_matched["_treat_"] == 0])
        ks_before[cov] = _ks_p(col[treat_mask == 1], col[treat_mask == 0])
        ks_after[cov]  = _ks_p(col_m[df_matched["_treat_"] == 1],
                                col_m[df_matched["_treat_"] == 0])

    avg_smd_before = float(np.mean(list(smd_before.values())))
    avg_smd_after  = float(np.mean(list(smd_after.values())))
    reduction_pct  = float((avg_smd_before - avg_smd_after) / avg_smd_before * 100) if avg_smd_before > 0 else 0.0

    n_all_treated = int((y == 1).sum())
    n_all_control = int((y == 0).sum())
    n_unmatched   = n_all_treated - n_matched_treated

    # Balance flag: all SMDs < 0.10 after matching AND every variance ratio in [0.5, 2.0]
    var_ratios_ok = all(
        (v is None) or (0.5 <= v <= 2.0)
        for v in var_ratio_after.values()
    )
    balance_achieved = bool(all(v < 0.10 for v in smd_after.values()) and var_ratios_ok)

    # ── Step 4: PS distribution for overlap plot ──────────────────────────────
    ps_dist = {
        "treated_unmatched": ps[treated_idx].tolist(),
        "control_unmatched": ps[control_idx].tolist(),
        "treated_matched":   ps[matched_treated].tolist(),
        "control_matched":   ps[matched_controls].tolist(),
    }

    # ── Outcome analysis on matched dataset ──────────────────────────────────
    outcome_result = None
    rosenbaum_result = None

    if outcome_type == "survival":
        try:
            from lifelines import CoxPHFitter
            dur = pd.to_numeric(df_matched[req.survival_duration_col], errors="coerce")
            evt = pd.to_numeric(df_matched[req.survival_event_col], errors="coerce")
            if np.any(dur.dropna() < 0):
                outcome_result = {"error": f"survival_duration_col '{req.survival_duration_col}' must be ≥ 0."}
            elif set(evt.dropna().unique().tolist()) - {0.0, 1.0}:
                outcome_result = {"error": f"survival_event_col '{req.survival_event_col}' must be binary 0/1."}
            else:
                cox_df = pd.DataFrame({
                    "_dur_":  dur.values.astype(float),
                    "_evt_":  evt.values.astype(int),
                    req.treatment_col: pd.to_numeric(df_matched[req.treatment_col], errors="coerce").astype(float).values,
                    "_match_id_": df_matched["_match_id_"].values.astype(int),
                }).dropna()
                cph = CoxPHFitter()
                # Stratify on match set so each matched pair has its own baseline hazard
                cph.fit(cox_df, duration_col="_dur_", event_col="_evt_", strata=["_match_id_"])
                coef = float(cph.params_.iloc[0])
                se   = float(cph.standard_errors_.iloc[0])
                ci   = cph.confidence_intervals_.iloc[0]
                try:
                    p_val = float(cph.summary["p"].iloc[0])
                except Exception:
                    p_val = None
                outcome_result = {
                    "type":   "stratified_cox",
                    "model":  "Cox PH stratified by matched set",
                    "n":      int(len(cox_df)),
                    "n_events": int(cox_df["_evt_"].sum()),
                    "concordance": round(float(cph.concordance_index_), 4),
                    "coefficients": [{
                        "variable": req.treatment_col,
                        "estimate": round(coef, 6),
                        "hr":       round(float(np.exp(coef)), 4),
                        "se":       round(se, 6),
                        "z":        round(coef / se, 4) if se > 0 else None,
                        "p":        round(p_val, 6) if p_val is not None else None,
                        "ci_low":   round(float(ci.iloc[0]), 4),
                        "ci_high":  round(float(ci.iloc[1]), 4),
                        "hr_low":   round(float(np.exp(ci.iloc[0])), 4),
                        "hr_high":  round(float(np.exp(ci.iloc[1])), 4),
                    }],
                }
        except Exception as ex:
            outcome_result = {"error": f"Stratified Cox failed: {ex}"}

    elif req.outcome_col and req.outcome_col in df_matched.columns:
        try:
            y_out = pd.to_numeric(df_matched[req.outcome_col], errors="coerce")
            out_vals = set(y_out.dropna().unique().tolist())

            if not out_vals <= {0, 1, 0.0, 1.0}:
                outcome_result = {"error": f"Outcome must be binary 0/1 for matched analysis. Found: {sorted(out_vals)[:10]}"}
            else:
                # Use GEE with matched-pair clustering for valid SEs
                from statsmodels.genmod.generalized_estimating_equations import GEE
                from statsmodels.genmod.families import Binomial
                from statsmodels.genmod.cov_struct import Exchangeable

                df_out = df_matched[[req.treatment_col, req.outcome_col, "_match_id_"]].copy()
                df_out[req.outcome_col] = y_out.astype(int)
                df_out[req.treatment_col] = df_out[req.treatment_col].astype(float)

                formula = f"Q('{req.outcome_col}') ~ Q('{req.treatment_col}')"
                try:
                    gee_model = GEE.from_formula(
                        formula, groups="_match_id_", data=df_out,
                        family=Binomial(), cov_struct=Exchangeable()
                    )
                    m_out = gee_model.fit()
                    ci_out = m_out.conf_int()
                    coefs_out = []
                    for var in m_out.params.index:
                        est = float(m_out.params[var])
                        coefs_out.append({
                            "variable": str(var),
                            "estimate": round(est, 6),
                            "or": round(float(np.exp(est)), 4),
                            "se": round(float(m_out.bse[var]), 6),
                            "z": round(float(m_out.tvalues[var]), 4),
                            "p": round(float(m_out.pvalues[var]), 6),
                            "ci_low":  round(float(ci_out.loc[var, 0]), 4),
                            "ci_high": round(float(ci_out.loc[var, 1]), 4),
                            "or_low":  round(float(np.exp(ci_out.loc[var, 0])), 4),
                            "or_high": round(float(np.exp(ci_out.loc[var, 1])), 4),
                        })
                    outcome_result = {
                        "type": "gee_matched",
                        "model": "GEE Logistic (matched-pair clusters)",
                        "n": int(len(df_matched)),
                        "n_clusters": int(len(matched_treated)),
                        "coefficients": coefs_out,
                    }
                except Exception:
                    # Fallback to standard logistic with robust SE
                    X_out = sm.add_constant(df_out[[req.treatment_col]].astype(float))
                    m_out = sm.Logit(y_out.astype(int).values, X_out).fit(disp=False, cov_type="HC1")
                    ci_out = m_out.conf_int()
                    coefs_out = []
                    for var in m_out.params.index:
                        est = float(m_out.params[var])
                        coefs_out.append({
                            "variable": str(var),
                            "estimate": round(est, 6),
                            "or": round(float(np.exp(est)), 4),
                            "se": round(float(m_out.bse[var]), 6),
                            "z": round(float(m_out.tvalues[var]), 4),
                            "p": round(float(m_out.pvalues[var]), 6),
                            "ci_low":  round(float(ci_out.loc[var, 0]), 4),
                            "ci_high": round(float(ci_out.loc[var, 1]), 4),
                            "or_low":  round(float(np.exp(ci_out.loc[var, 0])), 4),
                            "or_high": round(float(np.exp(ci_out.loc[var, 1])), 4),
                        })
                    outcome_result = {
                        "type": "logistic_robust",
                        "model": "Logistic Regression [Robust SE] (matched cohort)",
                        "n": int(len(df_matched)),
                        "coefficients": coefs_out,
                        "aic": round(float(m_out.aic), 2),
                        "bic": round(float(m_out.bic), 2),
                    }
        except Exception as ex:
            outcome_result = {"error": str(ex)}

    # ── Rosenbaum bounds (1:1 binary only) ───────────────────────────────────
    if (req.compute_rosenbaum and outcome_type == "binary" and ratio == 1
            and req.outcome_col and req.outcome_col in df_matched.columns):
        try:
            y_out = pd.to_numeric(df_matched[req.outcome_col], errors="coerce")
            out_vals = set(y_out.dropna().unique().tolist())
            if not out_vals <= {0, 1, 0.0, 1.0}:
                rosenbaum_result = {"applicable": False, "reason": "Rosenbaum bounds require binary 0/1 outcome."}
            else:
                pair_pairs: list[tuple[int, int]] = []
                df_rb = df_matched.copy()
                df_rb[req.outcome_col] = y_out.astype(int)
                df_rb[req.treatment_col] = pd.to_numeric(df_rb[req.treatment_col], errors="coerce").astype(int)
                for mid, grp in df_rb.groupby("_match_id_"):
                    t_rows = grp[grp[req.treatment_col] == 1]
                    c_rows = grp[grp[req.treatment_col] == 0]
                    if len(t_rows) == 1 and len(c_rows) == 1:
                        pair_pairs.append((int(t_rows[req.outcome_col].iloc[0]),
                                            int(c_rows[req.outcome_col].iloc[0])))
                if not pair_pairs:
                    rosenbaum_result = {"applicable": False, "reason": "No clean 1:1 matched pairs available."}
                else:
                    rosenbaum_result = _rosenbaum_bounds(
                        pair_pairs,
                        gamma_max=float(req.rosenbaum_gamma_max or 3.0),
                    )
        except Exception as ex:
            rosenbaum_result = {"applicable": False, "reason": f"Rosenbaum bounds failed: {ex}"}

    # Persist matched dataset for downstream analysis (keep match_id for paired tests)
    df_export = df_matched.drop(columns=["_ps_", "_logit_ps_", "_treat_"], errors="ignore")
    df_export = df_export.rename(columns={"_match_id_": "match_set_id"})
    store.save(req.session_id + "_psm", df_export)

    return {
        "n_total":            int(len(df)),
        "n_treated":          n_all_treated,
        "n_control":          n_all_control,
        "n_matched_pairs":    n_matched_treated,
        "n_matched_controls": n_matched_controls,
        "n_unmatched":        n_unmatched,
        "n_trimmed_common_support": n_trimmed,
        "score_method":       score_method,
        "matching_method":    matching_method,
        "matching_warning":   matching_warning,
        "exact_match":        exact_match_cols,
        "outcome_type":       outcome_type,
        "caliper_scale":      scale,
        "caliper_used":       round(float(caliper_dist), 6),
        "caliper_sd":         round(caliper_sd, 6),
        "common_support":     {"lo": round(support_lo, 6), "hi": round(support_hi, 6)},
        "balance_achieved":   balance_achieved,
        "avg_smd_before":     round(avg_smd_before, 4),
        "avg_smd_after":      round(avg_smd_after, 4),
        "reduction_pct":      round(reduction_pct, 1),
        "smd_before":         smd_before,
        "smd_after":          smd_after,
        "variance_ratio_before": var_ratio_before,
        "variance_ratio_after":  var_ratio_after,
        "ks_p_before":        ks_before,
        "ks_p_after":         ks_after,
        "ps_distribution":    ps_dist,
        "outcome_result":     outcome_result,
        "rosenbaum":          rosenbaum_result,
        "matched_session_id": req.session_id + "_psm",
    }
