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

from typing import List, Dict, Any, Literal, Optional
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
