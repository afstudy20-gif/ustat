"""Missing data analysis: pattern detection, MCAR test, imputation comparison."""
import numpy as np
import pandas as pd
from scipy import stats as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

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
# 1. MISSING DATA PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class PatternRequest(BaseModel):
    session_id: str
    columns: Optional[List[str]] = None


@router.post("/pattern")
def pattern(req: PatternRequest):
    df = _get_df(req.session_id)
    cols = req.columns if req.columns else list(df.columns)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")

    sub = df[cols]
    n_rows = len(sub)

    # ── Per-column missing stats ─────────────────────────────────────────
    per_column = []
    for col in cols:
        n_missing = int(sub[col].isnull().sum())
        pct_missing = round(n_missing / n_rows * 100, 2) if n_rows > 0 else 0.0
        per_column.append({
            "col": col,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
        })

    # ── Missing pattern table ────────────────────────────────────────────
    is_null = sub.isnull()
    # Group rows by their missing pattern
    pattern_strs = is_null.apply(lambda row: tuple(row.values), axis=1)
    pattern_counts = pattern_strs.value_counts()

    patterns = []
    for pat_tuple, count in pattern_counts.items():
        pat_dict = {col: bool(val) for col, val in zip(cols, pat_tuple)}
        patterns.append({
            "pattern": pat_dict,
            "count": int(count),
            "pct": round(int(count) / n_rows * 100, 2) if n_rows > 0 else 0.0,
        })

    # ── Heatmap data (first 500 rows) ────────────────────────────────────
    heatmap_sub = is_null.head(500)
    heatmap = heatmap_sub.astype(int).values.tolist()

    # ── Complete-case count ───────────────────────────────────────────────
    n_complete = int(sub.dropna().shape[0])

    return {
        "test": "Missing Data Pattern Analysis",
        "per_column": per_column,
        "patterns": patterns,
        "heatmap": heatmap,
        "heatmap_columns": cols,
        "n_rows": n_rows,
        "n_complete": n_complete,
        "significant": False,
        "effect_sizes": [],
        "assumptions": [],
        "result_text": (
            f"Missing data pattern analysis was conducted on {len(cols)} variables (n = {n_rows} rows). "
            f"{n_complete} rows ({round(n_complete / n_rows * 100, 1) if n_rows > 0 else 0}%) had complete data across all variables. "
            f"{len(patterns)} unique missing data patterns were identified."
        ),
        "export_rows": [
            ["Column", "N Missing", "% Missing"],
            *[[p["col"], p["n_missing"], p["pct_missing"]] for p in per_column],
        ],
        "r_code": "library(mice)\nmd.pattern(data)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LITTLE'S MCAR TEST
# ═══════════════════════════════════════════════════════════════════════════════

class MCARRequest(BaseModel):
    session_id: str
    columns: Optional[List[str]] = None


@router.post("/mcar_test")
def mcar_test(req: MCARRequest):
    df = _get_df(req.session_id)
    cols = req.columns if req.columns else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(cols) < 2:
        raise HTTPException(400, "Need at least 2 numeric columns for MCAR test.")

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(400, f"Columns not found: {missing_cols}")

    df_num = df[cols].apply(pd.to_numeric, errors="coerce")

    # Check that there is actually missing data
    total_missing = df_num.isnull().sum().sum()
    if total_missing == 0:
        return {
            "test": "Little's MCAR Test",
            "statistic": 0.0,
            "chi2": 0.0,
            "df": 0,
            "p": 1.0,
            "significant": False,
            "effect_sizes": [],
            "assumptions": [],
            "interpretation": "No missing data detected — MCAR test not applicable.",
            "result_text": "No missing data were detected in the selected columns. Little's MCAR test is not applicable.",
            "export_rows": [["Statistic", "Value"], ["Note", "No missing data"]],
            "r_code": "library(naniar)\nmcar_test(data)",
        }

    # Drop rows that are entirely missing
    df_num = df_num.dropna(how="all")
    n = len(df_num)
    p = len(cols)

    if n < p + 1:
        raise HTTPException(400, "Not enough observations for MCAR test.")

    # ── Little's MCAR test implementation ────────────────────────────────
    # Grand means and covariance (pairwise complete)
    grand_mean = df_num.mean().values  # length p
    # Pairwise covariance
    grand_cov = df_num.cov().values  # p x p

    # Handle singular covariance by adding small ridge
    ridge = np.eye(p) * 1e-8
    grand_cov_reg = grand_cov + ridge

    # Get missing patterns
    is_missing = df_num.isnull()
    pattern_keys = is_missing.apply(lambda row: tuple(row.values), axis=1)
    unique_patterns = pattern_keys.unique()

    # Only keep patterns that have at least some observed values and are not complete
    chi2_val = 0.0
    df_val = 0

    for pat in unique_patterns:
        mask = pattern_keys == pat
        group = df_num[mask]
        n_j = len(group)

        # Which variables are observed (not missing) in this pattern
        observed = [i for i, v in enumerate(pat) if not v]

        if len(observed) == 0 or len(observed) == p:
            # Skip fully missing or fully observed patterns (fully observed contributes 0)
            if len(observed) == p:
                df_val += len(observed)
            continue

        # Observed means for this pattern
        obs_cols = [cols[i] for i in observed]
        group_means = group[obs_cols].mean().values

        # Grand means for observed variables
        gm_obs = grand_mean[observed]

        # Submatrix of covariance for observed variables
        cov_sub = grand_cov_reg[np.ix_(observed, observed)]

        try:
            cov_inv = np.linalg.inv(cov_sub)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_sub)

        diff = group_means - gm_obs
        chi2_val += float(n_j * diff @ cov_inv @ diff)
        df_val += len(observed)

    # Degrees of freedom = sum of observed vars across patterns - p
    df_val = df_val - p
    if df_val <= 0:
        raise HTTPException(400, "Not enough missing data patterns to compute MCAR test (df <= 0).")

    p_val = float(1 - sp.chi2.cdf(chi2_val, df_val))
    sig = bool(p_val < 0.05)

    if sig:
        interp = "Data are NOT MCAR — missingness may be systematic"
    else:
        interp = "Data are MCAR (missing completely at random)"

    ps = _p_str(p_val)

    return {
        "test": "Little's MCAR Test",
        "statistic": round(chi2_val, 4),
        "chi2": round(chi2_val, 4),
        "df": df_val,
        "p": float(p_val),
        "significant": sig,
        "effect_sizes": [],
        "assumptions": [
            {"name": "Multivariate normality", "met": True,
             "detail": "Little's MCAR test assumes multivariate normality of the data."},
        ],
        "interpretation": interp,
        "result_text": (
            f"Little's MCAR test was conducted on {p} variables (n = {n}). "
            f"The test was {'significant' if sig else 'not significant'} "
            f"(χ²({df_val}) = {chi2_val:.2f}, p = {ps}), suggesting that "
            f"{'the data are not missing completely at random and missingness may be systematic' if sig else 'the data are missing completely at random (MCAR)'}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Chi-square", round(chi2_val, 4)],
            ["df", df_val],
            ["p", round(float(p_val), 6)],
            ["MCAR", "No" if sig else "Yes"],
        ],
        "r_code": "library(naniar)\nmcar_test(data)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. IMPUTATION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

class ImputationCompareRequest(BaseModel):
    session_id: str
    columns: List[str]
    strategies: List[str]  # e.g. ["median", "mean", "mice"]


@router.post("/imputation_compare")
def imputation_compare(req: ImputationCompareRequest):
    df = _get_df(req.session_id)

    missing = [c for c in req.columns if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")

    valid_strategies = ["mean", "median", "mice", "listwise"]
    for s in req.strategies:
        if s not in valid_strategies:
            raise HTTPException(400, f"Unknown strategy '{s}'. Valid: {valid_strategies}")

    # ── Before stats (original data) ─────────────────────────────────────
    def _desc(series: pd.Series) -> dict:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        if len(valid) == 0:
            return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None}
        return {
            "n": int(len(valid)),
            "mean": round(float(valid.mean()), 4),
            "sd": round(float(valid.std(ddof=1)), 4),
            "median": round(float(valid.median()), 4),
            "min": round(float(valid.min()), 4),
            "max": round(float(valid.max()), 4),
        }

    comparisons = []
    for strategy in req.strategies:
        # Apply imputation
        df_imputed = apply_imputation(df.copy(), req.columns, strategy)

        col_results = []
        for col in req.columns:
            before = _desc(df[col])
            after = _desc(df_imputed[col])

            # KS test between original (non-missing) and imputed
            orig_valid = pd.to_numeric(df[col], errors="coerce").dropna().values
            imp_valid = pd.to_numeric(df_imputed[col], errors="coerce").dropna().values

            if len(orig_valid) >= 2 and len(imp_valid) >= 2:
                ks_stat, ks_p = sp.ks_2samp(orig_valid, imp_valid)
            else:
                ks_stat, ks_p = 0.0, 1.0

            col_results.append({
                "col": col,
                "before": before,
                "after": after,
                "ks_stat": round(float(ks_stat), 4),
                "ks_p": round(float(ks_p), 4),
            })

        comparisons.append({
            "strategy": strategy,
            "columns": col_results,
        })

    # ── Build result text ────────────────────────────────────────────────
    strat_names = ", ".join(req.strategies)
    result_text = (
        f"Imputation comparison was conducted on {len(req.columns)} variables using "
        f"{len(req.strategies)} strategies ({strat_names}). "
        f"Kolmogorov-Smirnov tests were used to assess distributional shift "
        f"between original and imputed values for each column and strategy."
    )

    # ── Export rows ──────────────────────────────────────────────────────
    export_rows = [
        ["Strategy", "Column", "Before N", "Before Mean", "Before SD",
         "After N", "After Mean", "After SD", "KS Stat", "KS p"],
    ]
    for comp in comparisons:
        for cr in comp["columns"]:
            export_rows.append([
                comp["strategy"], cr["col"],
                cr["before"]["n"], cr["before"]["mean"], cr["before"]["sd"],
                cr["after"]["n"], cr["after"]["mean"], cr["after"]["sd"],
                cr["ks_stat"], cr["ks_p"],
            ])

    return {
        "test": "Imputation Comparison",
        "comparisons": comparisons,
        "significant": False,
        "effect_sizes": [],
        "assumptions": [],
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": (
            "library(mice)\n"
            "# Compare imputation strategies\n"
            f"imp <- mice(data[, c({', '.join(repr(c) for c in req.columns)})], method = 'pmm', m = 5)\n"
            "complete(imp, 1)"
        ),
    }
