"""
Missing Data Sensitivity Analysis Module (Phase 3 - C)

Provides tools for assessing how sensitive statistical results are
to different assumptions about the missing data mechanism,
with a focus on MNAR (Missing Not At Random) via delta-adjustment.

Core ideas implemented:
- Controlled simulation of MCAR / MAR / MNAR missingness
- Simple but useful delta-adjustment sensitivity analysis
- Support for linear, logistic, and Cox models
"""

from __future__ import annotations

from typing import Dict, Any, List, Literal, Optional
import numpy as np
import pandas as pd


def simulate_missingness(
    df: pd.DataFrame,
    cols: List[str],
    mechanism: Literal["MCAR", "MAR", "MNAR"] = "MAR",
    missing_rate: float = 0.2,
    seed: int = 42,
    **kwargs
) -> pd.DataFrame:
    """
    Introduce missing values into the DataFrame according to the specified mechanism.

    Parameters
    ----------
    mechanism : "MCAR" | "MAR" | "MNAR"
    missing_rate : target proportion of missing values in the selected columns
    """
    rng = np.random.default_rng(seed)
    df_miss = df.copy()

    for col in cols:
        if col not in df_miss.columns:
            continue

        n = len(df_miss)
        n_missing = int(n * missing_rate)

        if mechanism == "MCAR":
            miss_idx = rng.choice(n, size=n_missing, replace=False)
            df_miss.loc[miss_idx, col] = np.nan

        elif mechanism == "MAR":
            # Missingness depends on other observed variables (use first other numeric col as proxy)
            other_cols = [c for c in cols if c != col and pd.api.types.is_numeric_dtype(df_miss[c])]
            if not other_cols:
                # fallback to MCAR
                miss_idx = rng.choice(n, size=n_missing, replace=False)
            else:
                proxy = df_miss[other_cols[0]].fillna(df_miss[other_cols[0]].median())
                prob = 1 / (1 + np.exp(-0.8 * (proxy - proxy.mean()) / (proxy.std() + 1e-8)))
                prob = prob / prob.sum() * n_missing
                miss_idx = rng.choice(n, size=n_missing, replace=False, p=prob / prob.sum())
            df_miss.loc[miss_idx, col] = np.nan

        elif mechanism == "MNAR":
            # Missingness depends on the variable itself (or a latent version)
            vals = df_miss[col].fillna(df_miss[col].median())
            # Higher values more likely to be missing (common in clinical data, e.g. severe patients drop out)
            prob = 1 / (1 + np.exp(-1.2 * (vals - vals.mean()) / (vals.std() + 1e-8)))
            prob = prob / prob.sum() * n_missing
            miss_idx = rng.choice(n, size=n_missing, replace=False, p=prob / prob.sum())
            df_miss.loc[miss_idx, col] = np.nan

    return df_miss


def delta_adjustment_sensitivity(
    df: pd.DataFrame,
    outcome: str,
    predictors: List[str],
    model_type: Literal["linear", "logistic", "cox"] = "logistic",
    delta_range: tuple = (-2.0, 2.0),
    n_steps: int = 9,
    duration_col: Optional[str] = None,
    event_col: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform a simple delta-adjustment sensitivity analysis for MNAR.

    For each delta in the range, we add `delta` to the imputed values
    (or to the linear predictor in a pattern-mixture style) and refit the model.

    This gives an idea of how much the estimates change under different
    assumptions about the direction and strength of MNAR.

    Returns a list of results for each delta.
    """
    from services.missing_data import mice_multiple
    import statsmodels.api as sm
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(seed)
    deltas = np.linspace(delta_range[0], delta_range[1], n_steps)

    results = []
    base_cols = [outcome] + predictors

    # First do a standard MICE
    imp_result = mice_multiple(df, base_cols, n_imputations=3)
    base_pooled = None

    for delta in deltas:
        # Apply delta adjustment to the last imputed dataset (simple but illustrative)
        df_adj = imp_result.imputed_datasets[-1].copy()

        # Delta adjustment: shift the imputed values of the outcome (or a key predictor)
        # Here we shift the outcome for simplicity (common in pattern-mixture models)
        if model_type in ["linear", "logistic"]:
            # Only shift observed missing pattern in outcome
            miss_mask = df[outcome].isna()
            if miss_mask.any():
                df_adj.loc[miss_mask, outcome] = df_adj.loc[miss_mask, outcome] + delta

        # Refit model on the adjusted data
        try:
            if model_type == "linear":
                X = sm.add_constant(df_adj[predictors])
                y = df_adj[outcome]
                model = sm.OLS(y, X).fit()
                coef = model.params.iloc[1] if len(model.params) > 1 else model.params.iloc[0]
                se = model.bse.iloc[1] if len(model.bse) > 1 else model.bse.iloc[0]
                results.append({
                    "delta": round(float(delta), 3),
                    "estimate": round(float(coef), 4),
                    "se": round(float(se), 4),
                })

            elif model_type == "logistic":
                X = sm.add_constant(df_adj[predictors])
                y = df_adj[outcome].astype(int)
                model = sm.Logit(y, X).fit(disp=False, maxiter=100)
                coef = model.params.iloc[1] if len(model.params) > 1 else model.params.iloc[0]
                se = model.bse.iloc[1] if len(model.bse) > 1 else model.bse.iloc[0]
                results.append({
                    "delta": round(float(delta), 3),
                    "log_odds": round(float(coef), 4),
                    "odds_ratio": round(float(np.exp(coef)), 4),
                    "se": round(float(se), 4),
                })

            elif model_type == "cox":
                if not duration_col or not event_col:
                    raise ValueError("duration_col and event_col required for cox sensitivity")
                cph = CoxPHFitter()
                cph.fit(df_adj[[duration_col, event_col] + predictors],
                        duration_col=duration_col, event_col=event_col)
                hr = cph.hazard_ratios_.iloc[0] if len(cph.hazard_ratios_) > 0 else 1.0
                results.append({
                    "delta": round(float(delta), 3),
                    "hr": round(float(hr), 4),
                })

        except Exception as e:
            results.append({
                "delta": round(float(delta), 3),
                "error": str(e)[:80]
            })

    return {
        "model_type": model_type,
        "delta_range": delta_range,
        "n_steps": n_steps,
        "results": results,
        "interpretation": "How much the main effect estimate changes as we assume stronger MNAR (positive delta = worse outcomes among those with missing data)."
    }


def pattern_mixture_delta_model(
    df: pd.DataFrame,
    cols: List[str],
    *,
    delta_values: Optional[List[float]] = None,
    n_imputations: int = 5,
    passive_formulas: Optional[Dict[str, str]] = None,
    duration_col: Optional[str] = None,
    event_col: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Pattern-mixture MNAR sensitivity via delta adjustment.

    Missing cells are first imputed under MAR, then shifted by delta only for
    originally missing cells. This implements the common Little / Daniels-Hogan
    delta-adjustment PMM workflow.
    """
    from services.impute import add_survival_auxiliary_variables, apply_passive_imputation
    from services.missing_data import mice_multiple, mice_convergence_diagnostics, posterior_predictive_check

    delta_values = delta_values if delta_values is not None else [-2, -1, 0, 1, 2]
    work = df.copy()
    analysis_cols = [c for c in cols if c in work.columns]
    if duration_col and event_col and duration_col in work.columns and event_col in work.columns:
        work = add_survival_auxiliary_variables(work, duration_col, event_col)
        analysis_cols = list(dict.fromkeys(analysis_cols + [c for c in work.columns if c.startswith("__surv_aux")]))

    imp = mice_multiple(work, analysis_cols, n_imputations=n_imputations, random_state=random_state)
    missing_masks = {c: work[c].isna() for c in cols if c in work.columns}
    scenarios = []
    for delta in delta_values:
        summaries = []
        for m, imp_df in enumerate(imp.imputed_datasets, start=1):
            adj = imp_df.copy()
            for col, mask in missing_masks.items():
                if pd.api.types.is_numeric_dtype(adj[col]):
                    adj.loc[mask, col] = pd.to_numeric(adj.loc[mask, col], errors="coerce") + float(delta)
            adj = apply_passive_imputation(adj, passive_formulas)
            col_summary = {}
            for col in missing_masks:
                vals = pd.to_numeric(adj[col], errors="coerce")
                col_summary[col] = {
                    "mean": round(float(vals.mean()), 6) if vals.notna().any() else None,
                    "sd": round(float(vals.std(ddof=1)), 6) if vals.notna().sum() > 1 else None,
                }
            summaries.append({"imputation": m, "columns": col_summary})
        pooled = {}
        for col in missing_masks:
            means = [s["columns"][col]["mean"] for s in summaries if s["columns"][col]["mean"] is not None]
            pooled[col] = round(float(np.mean(means)), 6) if means else None
        scenarios.append({"delta": round(float(delta), 5), "pooled_means": pooled, "imputation_summaries": summaries})

    return {
        "method": "pattern_mixture_delta_adjustment",
        "n_imputations": int(n_imputations),
        "delta_values": [round(float(d), 5) for d in delta_values],
        "scenarios": scenarios,
        "mice_convergence": mice_convergence_diagnostics(imp, work, analysis_cols),
        "posterior_predictive_check": posterior_predictive_check(imp, work, analysis_cols),
        "interpretation": "Delta shifts apply only to originally missing cells; delta=0 is the MAR reference scenario.",
    }


def heckman_selection_model(
    df: pd.DataFrame,
    outcome_col: str,
    outcome_predictors: List[str],
    selection_predictors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Heckman-type two-stage selection model for continuous outcomes.

    Stage 1 models observation of outcome using probit. Stage 2 regresses the
    observed outcome on predictors plus inverse Mills ratio.
    """
    import statsmodels.api as sm
    from scipy.stats import norm

    selection_predictors = selection_predictors or outcome_predictors
    needed = list(dict.fromkeys([outcome_col] + outcome_predictors + selection_predictors))
    work = df[[c for c in needed if c in df.columns]].copy()
    observed = work[outcome_col].notna().astype(int)
    X_sel = pd.get_dummies(work[selection_predictors], drop_first=True).apply(pd.to_numeric, errors="coerce")
    X_sel = X_sel.fillna(X_sel.median(numeric_only=True))
    X_sel = sm.add_constant(X_sel, has_constant="add")
    try:
        probit = sm.Probit(observed, X_sel).fit(disp=False, maxiter=100)
        xb = np.asarray(probit.predict(X_sel, linear=True), dtype=float)
        mills = norm.pdf(xb) / np.clip(norm.cdf(xb), 1e-8, 1.0)
    except Exception as exc:
        return {"available": False, "reason": f"Selection equation failed: {exc}"}

    obs_mask = work[outcome_col].notna()
    X_out = pd.get_dummies(work.loc[obs_mask, outcome_predictors], drop_first=True).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(work.loc[obs_mask, outcome_col], errors="coerce")
    frame = pd.concat([y.rename("__y__"), X_out], axis=1).dropna()
    if len(frame) < 20:
        return {"available": False, "reason": "Need at least 20 observed outcomes for outcome equation."}
    X2 = sm.add_constant(frame.drop(columns=["__y__"]), has_constant="add")
    X2["inverse_mills_ratio"] = mills[frame.index]
    try:
        outcome_model = sm.OLS(frame["__y__"], X2).fit()
        coefs = [
            {
                "variable": str(k),
                "estimate": round(float(v), 6),
                "se": round(float(outcome_model.bse[k]), 6),
                "p": round(float(outcome_model.pvalues[k]), 6),
            }
            for k, v in outcome_model.params.items()
        ]
        imr_p = float(outcome_model.pvalues.get("inverse_mills_ratio", np.nan))
        return {
            "available": True,
            "method": "heckman_two_stage",
            "n_total": int(len(work)),
            "n_observed_outcome": int(obs_mask.sum()),
            "selection_rate": round(float(obs_mask.mean()), 5),
            "outcome_coefficients": coefs,
            "inverse_mills_ratio_p": round(imr_p, 6) if np.isfinite(imr_p) else None,
            "selection_bias_signal": bool(np.isfinite(imr_p) and imr_p < 0.05),
            "interpretation": "Significant inverse Mills ratio suggests outcome missingness is selection-related.",
        }
    except Exception as exc:
        return {"available": False, "reason": f"Outcome equation failed: {exc}"}


def isni_index(
    df: pd.DataFrame,
    outcome_col: str,
    predictors: List[str],
    missing_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Approximate ISNI (Index of Sensitivity to Non-Ignorability).

    For each variable with missingness, fit the analysis model with a missingness
    indicator and report indicator influence relative to the target coefficient.
    """
    import statsmodels.api as sm

    missing_cols = missing_cols or [c for c in [outcome_col] + predictors if c in df.columns and df[c].isna().any()]
    needed = list(dict.fromkeys([outcome_col] + predictors + missing_cols))
    work = df[[c for c in needed if c in df.columns]].copy()
    for c in work.columns:
        if pd.api.types.is_numeric_dtype(work[c]):
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(pd.to_numeric(work[c], errors="coerce").median())
    rows = []
    for miss_col in missing_cols:
        if miss_col not in df.columns:
            continue
        frame = work[[outcome_col] + predictors].copy()
        frame[f"__miss_{miss_col}"] = df[miss_col].isna().astype(int)
        frame = pd.get_dummies(frame, drop_first=True).apply(pd.to_numeric, errors="coerce").dropna()
        if len(frame) < 20:
            continue
        y = frame[outcome_col]
        X = sm.add_constant(frame.drop(columns=[outcome_col]), has_constant="add")
        try:
            if set(y.unique()).issubset({0, 1, 0.0, 1.0}):
                fit = sm.Logit(y, X).fit(disp=False, maxiter=100)
            else:
                fit = sm.OLS(y, X).fit()
            ind = f"__miss_{miss_col}"
            indicator_coef = float(fit.params.get(ind, 0.0))
            target = predictors[0] if predictors else ind
            target_coef = float(fit.params.get(target, np.nan))
            isni = abs(indicator_coef) / max(abs(target_coef), 1e-8) if np.isfinite(target_coef) else abs(indicator_coef)
            rows.append({
                "variable": miss_col,
                "missingness_indicator_coef": round(indicator_coef, 6),
                "target_coefficient": round(target_coef, 6) if np.isfinite(target_coef) else None,
                "isni": round(float(isni), 6),
                "high_sensitivity": bool(isni > 0.2),
            })
        except Exception as exc:
            rows.append({"variable": miss_col, "error": str(exc)})
    return {
        "method": "ISNI_local_sensitivity_proxy",
        "indices": rows,
        "interpretation": "Larger ISNI means the analysis coefficient is more locally sensitive to non-ignorability.",
    }


def survival_mnar_sensitivity(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    predictors: List[str],
    *,
    censoring_delta_values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Survival-specific MNAR sensitivity for informative censoring.

    Censored observations are up/down-weighted according to a delta sensitivity
    parameter and a Cox model is refit for each scenario.
    """
    from lifelines import CoxPHFitter

    censoring_delta_values = censoring_delta_values or [-1, -0.5, 0, 0.5, 1]
    needed = [duration_col, event_col] + predictors
    work = df[[c for c in needed if c in df.columns]].copy()
    for c in work.columns:
        if c in [duration_col, event_col] or pd.api.types.is_numeric_dtype(work[c]):
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    if len(work) < 20:
        return {"available": False, "reason": "Need at least 20 complete rows."}
    results = []
    for delta in censoring_delta_values:
        scenario = work.copy()
        censored = (scenario[event_col] == 0).astype(float)
        scenario["__mnar_weight__"] = np.exp(float(delta) * censored)
        try:
            cph = CoxPHFitter()
            cph.fit(
                scenario[[duration_col, event_col, "__mnar_weight__"] + predictors],
                duration_col=duration_col,
                event_col=event_col,
                weights_col="__mnar_weight__",
                robust=True,
            )
            coefs = [
                {"variable": str(k), "hr": round(float(v), 6)}
                for k, v in cph.hazard_ratios_.items()
            ]
            results.append({
                "delta": round(float(delta), 5),
                "censored_weight_multiplier": round(float(np.exp(delta)), 5),
                "coefficients": coefs,
                "concordance": round(float(cph.concordance_index_), 5),
            })
        except Exception as exc:
            results.append({"delta": round(float(delta), 5), "error": str(exc)})
    return {
        "available": True,
        "method": "informative_censoring_weight_shift",
        "results": results,
        "interpretation": "Positive delta gives censored observations more influence, probing informative censoring MNAR assumptions.",
    }


def summarize_sensitivity(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple helper to summarize how much estimates move across delta values."""
    estimates = [r.get("estimate") or r.get("log_odds") or r.get("hr") for r in results if "error" not in r]
    if not estimates:
        return {"range": None, "max_change": None}

    return {
        "min_estimate": round(float(min(estimates)), 4),
        "max_estimate": round(float(max(estimates)), 4),
        "range": round(float(max(estimates) - min(estimates)), 4),
        "most_extreme_delta": results[int(np.argmax(np.abs(np.array(estimates) - np.mean(estimates))))]["delta"]
    }
