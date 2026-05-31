"""
Advanced Missing Data Handling for uSTAT (Phase 3)

This module provides:
- Proper Multiple Imputation (MICE) with multiple datasets
- Rubin's Rules for pooling results across imputations
- Basic missing data diagnostics and pattern reporting
- Sensitivity analysis helpers (planned)

Goal: Move from "single imputation" to statistically proper multiple imputation
suitable for mid-to-advanced biostatistics work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class ImputationResult:
    """Container for multiple imputed datasets + metadata."""
    imputed_datasets: List[pd.DataFrame]
    original_missing_info: Dict[str, Any]
    n_imputations: int
    method: str = "mice"


def mice_multiple(
    df: pd.DataFrame,
    cols: List[str],
    n_imputations: int = 5,
    max_iter: int = 10,
    random_state: int = 42,
) -> ImputationResult:
    """
    Perform proper Multiple Imputation using IterativeImputer (MICE).

    Returns multiple completed datasets instead of a single one.
    This is the foundation for proper statistical inference with missing data.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return ImputationResult(
            imputed_datasets=[df.copy() for _ in range(max(1, n_imputations))],
            original_missing_info={},
            n_imputations=n_imputations,
            method="mice"
        )

    original_missing = missing_pattern_summary(df, valid_cols)

    num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        # Nothing to impute numerically → return copies
        return ImputationResult(
            imputed_datasets=[df.copy() for _ in range(n_imputations)],
            original_missing_info=original_missing,
            n_imputations=n_imputations,
            method="mice"
        )

    imputed_datasets = []
    base_imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
        skip_complete=True,
    )

    for i in range(n_imputations):
        imp = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state + i,  # different seed per imputation
            verbose=0,
            skip_complete=True,
        )
        df_imp = df.copy()
        df_imp[num_cols] = imp.fit_transform(df_imp[num_cols])
        imputed_datasets.append(df_imp)

    return ImputationResult(
        imputed_datasets=imputed_datasets,
        original_missing_info=original_missing,
        n_imputations=n_imputations,
        method="mice"
    )


def missing_pattern_summary(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    """Return richer missing data pattern information."""
    valid_cols = [c for c in cols if c in df.columns]
    total = len(df)

    per_col = {}
    for col in valid_cols:
        n = int(df[col].isna().sum())
        per_col[col] = {
            "count": n,
            "pct": round(n / total * 100, 1) if total > 0 else 0.0,
        }

    rows_affected = int(df[valid_cols].isna().any(axis=1).sum()) if valid_cols else 0

    # Simple pattern classification
    pattern = "unknown"
    if rows_affected == 0:
        pattern = "complete"
    elif rows_affected / total > 0.5:
        pattern = "heavy_missing"
    elif any(p["pct"] > 30 for p in per_col.values()):
        pattern = "high_in_some_variables"
    else:
        pattern = "moderate"

    return {
        "total_rows": total,
        "rows_affected": rows_affected,
        "pct_affected": round(rows_affected / total * 100, 1) if total > 0 else 0.0,
        "per_column": per_col,
        "pattern_severity": pattern,
    }


# =============================================================================
# Rubin's Rules (Pooling)
# =============================================================================

def pool_linear_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pool results from multiple imputed linear models using Rubin's Rules.

    Expects each result dict to have at least:
    - 'coefficients': list of {'variable': str, 'estimate': float, 'se': float, ...}
    - 'r_squared': float (optional)
    """
    if not results:
        return {}

    # Assume all imputations have the same variables
    variables = [c["variable"] for c in results[0]["coefficients"]]

    pooled_coefs = []
    for var in variables:
        estimates = []
        ses = []
        for res in results:
            for c in res["coefficients"]:
                if c["variable"] == var:
                    estimates.append(c["estimate"])
                    ses.append(c.get("se", np.nan))
                    break

        estimates = np.array(estimates)
        ses = np.array(ses)

        Q_bar = np.mean(estimates)
        U_bar = np.mean(ses ** 2)
        B = np.var(estimates, ddof=1) if len(estimates) > 1 else 0.0
        T = U_bar + (1 + 1 / len(results)) * B

        pooled_se = np.sqrt(max(T, 1e-12))
        denom = (1 + 1 / len(results)) * B if B > 0 else 1e-12
        df = max((len(results) - 1) * (1 + U_bar / denom) ** 2, 1.0)

        # Simple t approximation
        t_stat = Q_bar / pooled_se if pooled_se > 0 else 0.0

        pooled_coefs.append({
            "variable": var,
            "estimate": round(float(Q_bar), 6),
            "se": round(float(pooled_se), 6),
            "t": round(float(t_stat), 4),
            "df": round(float(df), 1),
        })

    # Simple pooled R² (average)
    r2s = [r.get("r_squared", np.nan) for r in results if "r_squared" in r]
    pooled_r2 = float(np.nanmean(r2s)) if r2s else None

    return {
        "method": "rubins_rules_pooled",
        "n_imputations": len(results),
        "coefficients": pooled_coefs,
        "r_squared": round(pooled_r2, 4) if pooled_r2 is not None else None,
    }


def pool_cox_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pool Cox PH results across multiple imputations using Rubin's Rules (on log HR scale).

    Expects each result to have:
    - 'coefficients': dict like {var: hr_value} or list of {'variable': , 'estimate': loghr or hr}
    """
    if not results:
        return {}

    # Normalize coefficients to log(HR) scale
    all_vars = set()
    for res in results:
        coefs = res.get("coefficients", {})
        if isinstance(coefs, dict):
            all_vars.update(coefs.keys())
        elif isinstance(coefs, list):
            for c in coefs:
                if "variable" in c:
                    all_vars.add(c["variable"])

    pooled = {}
    for var in sorted(all_vars):
        loghr_list = []
        for res in results:
            coefs = res.get("coefficients", {})
            val = None
            if isinstance(coefs, dict):
                val = coefs.get(var)
            elif isinstance(coefs, list):
                for c in coefs:
                    if c.get("variable") == var:
                        val = c.get("estimate") or c.get("log_hr") or c.get("hr")
                        break

            if val is not None and val > 0:
                loghr_list.append(np.log(val))

        if not loghr_list:
            continue

        loghr_arr = np.array(loghr_list)
        Q_bar = float(np.mean(loghr_arr))
        U_bar = float(np.var(loghr_arr, ddof=1)) / len(loghr_arr) if len(loghr_arr) > 1 else 0.0
        B = float(np.var(loghr_arr, ddof=1)) if len(loghr_arr) > 1 else 0.0
        T = U_bar + (1 + 1 / len(results)) * B

        pooled_se = float(np.sqrt(max(T, 1e-12)))
        hr = float(np.exp(Q_bar))

        pooled[var] = {
            "hr": round(hr, 4),
            "log_hr": round(Q_bar, 6),
            "se_log_hr": round(pooled_se, 6),
        }

    return {
        "method": "rubins_rules_pooled_cox",
        "n_imputations": len(results),
        "coefficients": pooled,
    }


def pool_logistic_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pool logistic regression results across multiple imputations using Rubin's Rules.

    Expects each result dict to contain:
    - 'coefficients': list of {'variable': str, 'log_odds': float, 'se': float, ...}
      (or 'estimate' as log-odds)

    Pooling is performed on the log-odds (coefficient) scale.
    Odds ratios are obtained by exponentiating the pooled log-odds.
    """
    if not results:
        return {}

    # Collect all unique variables
    first_coefs = results[0].get("coefficients", [])
    variables = [c.get("variable") or c.get("name") for c in first_coefs if c.get("variable") or c.get("name")]

    pooled_coefs = []
    for var in variables:
        log_odds_list = []
        se_list = []

        for res in results:
            coefs = res.get("coefficients", [])
            for c in coefs:
                if (c.get("variable") or c.get("name")) == var:
                    # Prefer 'log_odds', fall back to 'estimate' or 'B'
                    lod = c.get("log_odds") or c.get("estimate") or c.get("B")
                    se = c.get("se")
                    if lod is not None and se is not None:
                        log_odds_list.append(float(lod))
                        se_list.append(float(se))
                    break

        if not log_odds_list:
            continue

        log_odds_arr = np.array(log_odds_list)
        se_arr = np.array(se_list)

        Q_bar = np.mean(log_odds_arr)                    # Pooled log-odds
        U_bar = np.mean(se_arr ** 2)                     # Within-imputation variance
        B = np.var(log_odds_arr, ddof=1) if len(log_odds_arr) > 1 else 0.0  # Between-imputation variance
        T = U_bar + (1 + 1 / len(results)) * B           # Total variance

        pooled_se = np.sqrt(max(T, 1e-12))

        # Approximate degrees of freedom (Rubin's rules)
        if B > 0:
            df = (len(results) - 1) * (1 + U_bar / ((1 + 1 / len(results)) * B)) ** 2
        else:
            df = len(results) - 1

        # z-statistic and p-value (normal approximation, common in practice)
        z = Q_bar / pooled_se if pooled_se > 0 else 0.0
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z))) if pooled_se > 0 else None

        pooled_coefs.append({
            "variable": var,
            "log_odds": round(float(Q_bar), 6),
            "odds_ratio": round(float(np.exp(Q_bar)), 4),
            "se": round(float(pooled_se), 6),
            "z": round(float(z), 4),
            "p": round(float(p), 6) if p is not None else None,
            "df": round(float(df), 1) if df else None,
        })

    return {
        "method": "rubins_rules_pooled_logistic",
        "n_imputations": len(results),
        "coefficients": pooled_coefs,
    }


def add_missing_data_diagnostics(result: dict, missing_info: dict) -> dict:
    """Attach missing data diagnostics to an analysis result."""
    result["missing_data"] = missing_info
    if missing_info.get("pct_affected", 0) > 10:
        if "warnings" not in result:
            result["warnings"] = []
        result["warnings"].append(
            f"{missing_info['pct_affected']}% of rows had missing values in analysis variables. "
            "Consider using multiple imputation (mice)."
        )
    return result
