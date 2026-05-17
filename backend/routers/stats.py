import numpy as np
import pandas as pd
import json as _json
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException, Response, Query
from pydantic import BaseModel
from typing import Optional, List
from services import store
from services.impute import apply_imputation, missing_info
from services.text_generators import (
    methods_ttest_ind, methods_ttest_one, methods_chisquare, methods_mannwhitney,
    methods_fisher, methods_kruskal, methods_anova,
    results_ttest_ind, results_ttest_one, results_chisquare, results_mannwhitney,
    results_fisher, results_kruskal, results_anova,
    r_ttest_ind, r_ttest_one, r_chisquare, r_mannwhitney, r_fisher, r_kruskal, r_anova,
)
from services.stat_utils import (
    cohen_d, cohen_d_one_sample, eta_squared, partial_eta_squared, omega_squared,
    rank_biserial_r, cramers_v, odds_ratio_effect, epsilon_squared,
    check_normality, check_equal_variances, group_summary,
    adjust_pvalues, pairwise_t_tests, pairwise_wilcoxon, tukey_hsd, games_howell, dunn_test,
)


def _safe_json(obj) -> Response:
    """Serialize obj to JSON, replacing NaN/Inf with null."""
    text = _json.dumps(obj, allow_nan=False, default=lambda x: None
                       if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else str(x))
    return Response(content=text, media_type="application/json")


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None in dicts/lists."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


# ── Missing Data Summary ─────────────────────────────────────────────────────

@router.get("/{session_id}/missing")
def get_missing(session_id: str, columns: str = Query("")):
    """
    Return per-column missing counts and total rows affected for the given
    comma-separated list of column names.
    """
    df = _get_df(session_id)
    cols = [c.strip() for c in columns.split(",") if c.strip() and c.strip() in df.columns]
    if not cols:
        cols = df.columns.tolist()
    return missing_info(df, cols)


# ── Descriptive Statistics ──────────────────────────────────────────────────

@router.get("/{session_id}/descriptive")
def descriptive(session_id: str, column: Optional[str] = None):
    df = _get_df(session_id)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if column:
        if column not in num_cols:
            raise HTTPException(status_code=400, detail="Column not numeric")
        num_cols = [column]

    results = {}
    for col in num_cols:
        s = df[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 3:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        n = len(s)
        if n < 50:
            _, p_norm = scipy_stats.shapiro(s)
            norm_test = "Shapiro-Wilk"
        elif n <= 2000:
            from statsmodels.stats.diagnostic import lilliefors as _lilliefors
            _, p_norm = _lilliefors(s.values, dist="norm")
            norm_test = "Kolmogorov-Smirnov (Lilliefors)"
        elif abs(float(scipy_stats.skew(s))) <= 1.5:
            p_norm = 0.999  # CLT bypass — mild skewness at large n
            norm_test = "Skewness (CLT bypass)"
        else:
            from statsmodels.stats.diagnostic import lilliefors as _lilliefors
            _, p_norm = _lilliefors(s.values, dist="norm")
            norm_test = "Kolmogorov-Smirnov (Lilliefors)"
        results[col] = {
            "n": int(s.count()),
            "missing": int(df[col].isna().sum()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "se": float(s.sem()),
            "min": float(s.min()),
            "max": float(s.max()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
            "normality_p": float(p_norm),
            "normality_test": norm_test,
            "normality_label": "Normal" if p_norm > 0.05 else "Non-normal",
        }
    return results


# ── Frequency Table ─────────────────────────────────────────────────────────

@router.get("/{session_id}/frequency")
def frequency(session_id: str, column: str):
    df = _get_df(session_id)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
    counts = df[column].value_counts(dropna=False)
    total = len(df)
    return [
        {"value": str(k), "count": int(v), "pct": round(v / total * 100, 2)}
        for k, v in counts.items()
    ]


# ── T-Tests ─────────────────────────────────────────────────────────────────

class TTestRequest(BaseModel):
    session_id: str
    column: str
    group_column: Optional[str] = None
    mu: Optional[float] = 0.0
    equal_var: bool = True


@router.post("/ttest")
def ttest(req: TTestRequest):
    df = _get_df(req.session_id)
    col = df[req.column].dropna()

    if req.group_column:
        groups = df[req.group_column].dropna().unique()
        if len(groups) != 2:
            raise HTTPException(status_code=400, detail="Group column must have exactly 2 groups")
        g1 = df[df[req.group_column] == groups[0]][req.column].dropna().astype(float).values
        g2 = df[df[req.group_column] == groups[1]][req.column].dropna().astype(float).values

        # Assumption checks
        assumptions = [check_normality(g1, str(groups[0])), check_normality(g2, str(groups[1])),
                       check_equal_variances([g1, g2], [str(groups[0]), str(groups[1])])]
        use_welch = not assumptions[2]["met"]
        stat, p = scipy_stats.ttest_ind(g1, g2, equal_var=not use_welch)
        sig = bool(p < 0.05)
        es = cohen_d(g1, g2)
        p_str = '<0.001' if p < 0.001 else f'{p:.4f}'

        ret = {
            "test": f"Independent samples t-test{' (Welch)' if use_welch else ''}",
            "group1": str(groups[0]), "n1": len(g1), "mean1": float(g1.mean()),
            "group2": str(groups[1]), "n2": len(g2), "mean2": float(g2.mean()),
            "t": float(stat), "p": float(p), "df": int(len(g1) + len(g2) - 2),
            "significant": sig,
            "effect_sizes": [es],
            "assumptions": assumptions,
            "summary": {str(groups[0]): group_summary(g1, str(groups[0])),
                        str(groups[1]): group_summary(g2, str(groups[1]))},
            "interpretation": f"{'Significant' if sig else 'No significant'} difference between groups (t = {stat:.3f}, p = {p_str}, Hedges' g = {es['value']:.3f} [{es['magnitude']}])",
            "methods_text": methods_ttest_ind(req.column, req.group_column, use_welch),
            "r_code": r_ttest_ind(req.column, req.group_column),
        }
        ret["result_text"] = results_ttest_ind(ret)
        return ret
    else:
        x = col.astype(float).values
        stat, p = scipy_stats.ttest_1samp(x, req.mu)
        sig = bool(p < 0.05)
        es = cohen_d_one_sample(x, req.mu)
        p_str = '<0.001' if p < 0.001 else f'{p:.4f}'

        ret = {
            "test": "One-sample t-test",
            "mu": req.mu, "n": len(x),
            "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "t": float(stat), "p": float(p), "df": int(len(x) - 1),
            "significant": sig,
            "effect_sizes": [es],
            "assumptions": [check_normality(x, req.column)],
            "summary": {"sample": group_summary(x, "Sample")},
            "interpretation": f"Mean {'differs from' if sig else 'does not differ from'} {req.mu} (t = {stat:.3f}, p = {p_str}, Cohen's d = {es['value']:.3f} [{es['magnitude']}])",
            "methods_text": methods_ttest_one(req.column, req.mu),
            "r_code": r_ttest_one(req.column, req.mu),
        }
        ret["result_text"] = results_ttest_one(ret)
        return ret


# ── Chi-Square ───────────────────────────────────────────────────────────────

class ChiSqRequest(BaseModel):
    session_id: str
    row_column: str
    col_column: str


@router.post("/chisquare")
def chisquare(req: ChiSqRequest):
    df = _get_df(req.session_id)
    ct = pd.crosstab(df[req.row_column], df[req.col_column])
    chi2, p, dof, expected = scipy_stats.chi2_contingency(ct)
    sig = bool(p < 0.05)
    n = ct.values.sum()
    min_dim = min(ct.shape)
    es = cramers_v(chi2, n, min_dim)
    # Odds ratio for 2x2 tables
    effect_sizes = [es]
    if ct.shape == (2, 2):
        effect_sizes.append(odds_ratio_effect(ct.values))
    # Warning for small expected counts
    warnings = []
    if (expected < 5).any():
        warnings.append("Some expected cell counts < 5. Consider Fisher's exact test instead.")
    p_str = '<0.001' if p < 0.001 else f'{p:.4f}'
    ret = {
        "test": "Chi-square test of independence",
        "chi2": float(chi2), "p": float(p), "dof": int(dof), "n": int(n),
        "significant": sig,
        "effect_sizes": effect_sizes,
        "warnings": warnings,
        "crosstab": ct.to_dict(),
        "interpretation": f"{'Significant' if sig else 'No significant'} association (\u03C7\u00B2({dof}) = {chi2:.2f}, p = {p_str}, Cramer's V = {es['value']:.3f} [{es['magnitude']}])",
        "methods_text": methods_chisquare(req.row_column, req.col_column),
        "r_code": r_chisquare(req.row_column, req.col_column),
    }
    ret["result_text"] = results_chisquare(ret)
    return ret


# ── Correlation ──────────────────────────────────────────────────────────────

@router.get("/{session_id}/correlation")
def correlation(session_id: str, method: str = "pearson"):
    df = _get_df(session_id)
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method=method)
    p_values = {}
    for c1 in corr.columns:
        p_values[c1] = {}
        for c2 in corr.columns:
            if c1 == c2:
                p_values[c1][c2] = 0.0
            else:
                pair = num_df[[c1, c2]].dropna()
                if len(pair) < 3 or pair[c1].std() == 0 or pair[c2].std() == 0:
                    p_values[c1][c2] = None  # too few obs or constant
                    continue
                s1, s2 = pair.values.T
                try:
                    if method == "pearson":
                        _, p = scipy_stats.pearsonr(s1, s2)
                    elif method == "spearman":
                        _, p = scipy_stats.spearmanr(s1, s2)
                    else:
                        _, p = scipy_stats.kendalltau(s1, s2)
                    p_values[c1][c2] = float(p)
                except Exception:
                    p_values[c1][c2] = None
    return {
        "method": method,
        "columns": corr.columns.tolist(),
        "matrix": corr.round(4).where(pd.notnull(corr), None).to_dict(),
        "p_values": p_values,
    }


# ── Mann-Whitney U ────────────────────────────────────────────────────────────

class MannWhitneyRequest(BaseModel):
    session_id: str
    column: str
    group_column: str


@router.post("/mannwhitney")
def mannwhitney(req: MannWhitneyRequest):
    df = _get_df(req.session_id)
    groups = df[req.group_column].dropna().unique()
    if len(groups) != 2:
        raise HTTPException(status_code=400, detail="Group column must have exactly 2 groups")
    g1 = df[df[req.group_column] == groups[0]][req.column].dropna().astype(float).values
    g2 = df[df[req.group_column] == groups[1]][req.column].dropna().astype(float).values
    stat, p = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    sig = bool(p < 0.05)
    es = rank_biserial_r(float(stat), len(g1), len(g2))
    p_str = '<0.001' if p < 0.001 else f'{p:.4f}'
    ret = {
        "test": "Mann-Whitney U test",
        "group1": str(groups[0]), "n1": int(len(g1)),
        "median1": float(np.median(g1)), "iqr1": float(np.percentile(g1, 75) - np.percentile(g1, 25)),
        "group2": str(groups[1]), "n2": int(len(g2)),
        "median2": float(np.median(g2)), "iqr2": float(np.percentile(g2, 75) - np.percentile(g2, 25)),
        "U": float(stat), "p": float(p),
        "significant": sig,
        "effect_sizes": [es],
        "summary": {str(groups[0]): group_summary(g1, str(groups[0])),
                    str(groups[1]): group_summary(g2, str(groups[1]))},
        "interpretation": f"{'Significant' if sig else 'No significant'} difference (U = {stat:.1f}, p = {p_str}, r = {es['value']:.3f} [{es['magnitude']}])",
        "methods_text": methods_mannwhitney(req.column, req.group_column),
        "r_code": r_mannwhitney(req.column, req.group_column),
    }
    ret["result_text"] = results_mannwhitney(ret)
    return ret


# ── Fisher's Exact Test ───────────────────────────────────────────────────────

class FisherRequest(BaseModel):
    session_id: str
    row_column: str
    col_column: str


@router.post("/fisher")
def fisher_exact(req: FisherRequest):
    df = _get_df(req.session_id)
    ct = pd.crosstab(df[req.row_column], df[req.col_column])
    if ct.shape != (2, 2):
        raise HTTPException(status_code=400, detail="Fisher's exact test requires a 2\u00D72 table")
    table = ct.values.tolist()
    or_val, p = scipy_stats.fisher_exact(ct.values)
    sig = bool(p < 0.05)
    es = odds_ratio_effect(ct.values)
    p_str = '<0.001' if p < 0.001 else f'{p:.4f}'
    ret = {
        "test": "Fisher's exact test",
        "odds_ratio": float(or_val), "p": float(p),
        "significant": sig,
        "effect_sizes": [es],
        "table": table,
        "row_labels": ct.index.tolist(),
        "col_labels": ct.columns.tolist(),
        "interpretation": f"{'Significant' if sig else 'No significant'} association (p = {p_str}, OR = {es['value']:.2f}, 95% CI: {es['ci_low']:.2f}\u2013{es['ci_high']:.2f})",
        "methods_text": methods_fisher(req.row_column, req.col_column),
        "r_code": r_fisher(req.row_column, req.col_column),
    }
    ret["result_text"] = results_fisher(ret)
    return ret


# ── Kruskal-Wallis ────────────────────────────────────────────────────────────

class KruskalRequest(BaseModel):
    session_id: str
    column: str
    group_column: str


@router.post("/kruskal")
def kruskal(req: KruskalRequest):
    df = _get_df(req.session_id)
    grp_dict = {str(name): g[req.column].dropna().astype(float).values
                for name, g in df.groupby(req.group_column)}
    group_data = list(grp_dict.values())
    if len(group_data) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 groups")
    stat, p = scipy_stats.kruskal(*group_data)
    sig = bool(p < 0.05)
    n_total = sum(len(g) for g in group_data)
    es = epsilon_squared(float(stat), n_total)
    p_str = '<0.001' if p < 0.001 else f'{p:.4f}'

    # Post-hoc: Dunn's test (only if significant and > 2 groups)
    posthoc = dunn_test(grp_dict, correction="holm") if sig and len(grp_dict) > 2 else []

    group_stats = df.groupby(req.group_column)[req.column].agg(
        n="count", median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    ).reset_index()
    ret = {
        "test": "Kruskal-Wallis test",
        "H": float(stat), "p": float(p),
        "significant": sig,
        "effect_sizes": [es],
        "posthoc": posthoc,
        "posthoc_method": "Dunn's test (Holm correction)" if posthoc else None,
        "groups": [
            {k: (float(v) if hasattr(v, '__float__') else str(v)) for k, v in row.items()}
            for row in group_stats.to_dict(orient="records")
        ],
        "interpretation": f"{'Significant' if sig else 'No significant'} difference across groups (H = {stat:.2f}, p = {p_str}, \u03B5\u00B2 = {es['value']:.3f} [{es['magnitude']}])",
        "methods_text": methods_kruskal(req.column, req.group_column),
        "r_code": r_kruskal(req.column, req.group_column),
    }
    ret["result_text"] = results_kruskal(ret)
    return ret


# ── ROC Analysis ──────────────────────────────────────────────────────────────

class ROCRequest(BaseModel):
    session_id: str
    score_column: str
    outcome_column: str
    manual_cutoff: Optional[float] = None
    imputation: Optional[str] = "listwise"  # "listwise" | "median" | "mice"


def _roc_metrics_at_cutoff(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Compute full diagnostic metrics at a given threshold."""
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv   = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc   = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("inf")
    return {
        "cutoff": round(float(threshold), 6),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "accuracy": round(acc, 4),
        "lr_pos": round(lr_pos, 4) if not np.isinf(lr_pos) else None,
        "lr_neg": round(lr_neg, 4) if not np.isinf(lr_neg) else None,
        "youden_j": round(sens + spec - 1, 4),
    }


def _delong_placement_values(y: np.ndarray, scores: np.ndarray):
    """Return (V_pos, V_neg) placement value arrays for DeLong variance."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n1, n0 = len(pos_idx), len(neg_idx)
    s_pos = scores[pos_idx]
    s_neg = scores[neg_idx]
    # Placement value for each positive: Pr(neg < pos) + 0.5*Pr(neg == pos)
    V_pos = (
        np.sum(s_neg[:, None] < s_pos[None, :], axis=0).astype(float)
        + 0.5 * np.sum(s_neg[:, None] == s_pos[None, :], axis=0).astype(float)
    ) / n0
    # Placement value for each negative: Pr(neg > pos) + 0.5*Pr(neg == pos)
    V_neg = (
        np.sum(s_pos[:, None] > s_neg[None, :], axis=0).astype(float)
        + 0.5 * np.sum(s_pos[:, None] == s_neg[None, :], axis=0).astype(float)
    ) / n1
    return V_pos, V_neg


def _delong_compare(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """DeLong 1988 non-parametric AUC comparison.
    Returns AUCs, ΔAUC, 95% CI of ΔAUC, Z, p, and individual AUC 95% CIs."""
    V_pos1, V_neg1 = _delong_placement_values(y, s1)
    V_pos2, V_neg2 = _delong_placement_values(y, s2)
    n_pos, n_neg = len(V_pos1), len(V_neg1)
    auc1 = float(np.mean(V_pos1))
    auc2 = float(np.mean(V_pos2))

    # Variance-covariance matrix of [AUC1, AUC2] via empirical Mann-Whitney U
    s11 = np.var(V_pos1, ddof=1) / n_pos + np.var(V_neg1, ddof=1) / n_neg
    s22 = np.var(V_pos2, ddof=1) / n_pos + np.var(V_neg2, ddof=1) / n_neg
    s12 = (np.cov(V_pos1, V_pos2, ddof=1)[0, 1] / n_pos
           + np.cov(V_neg1, V_neg2, ddof=1)[0, 1] / n_neg)

    # 95% CI for ΔAUC = AUC1 − AUC2
    var_diff = max(s11 + s22 - 2 * s12, 1e-12)
    diff = auc1 - auc2
    se_diff = np.sqrt(var_diff)
    z = diff / se_diff
    p = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
    z95 = 1.95996   # scipy_stats.norm.ppf(0.975)
    ci_diff_low  = float(diff - z95 * se_diff)
    ci_diff_high = float(diff + z95 * se_diff)

    # 95% CI for each individual AUC (DeLong SE, no bootstrap needed)
    se1 = np.sqrt(max(s11, 1e-12))
    se2 = np.sqrt(max(s22, 1e-12))
    ci1_low  = max(0.0, float(auc1 - z95 * se1))
    ci1_high = min(1.0, float(auc1 + z95 * se1))
    ci2_low  = max(0.0, float(auc2 - z95 * se2))
    ci2_high = min(1.0, float(auc2 + z95 * se2))

    return {
        "auc_1": round(auc1, 4),
        "auc_2": round(auc2, 4),
        "ci_1_low":  round(ci1_low, 4),
        "ci_1_high": round(ci1_high, 4),
        "ci_2_low":  round(ci2_low, 4),
        "ci_2_high": round(ci2_high, 4),
        "difference":    round(diff, 4),
        "ci_diff_low":   round(ci_diff_low, 4),
        "ci_diff_high":  round(ci_diff_high, 4),
        "se_diff": round(float(se_diff), 6),
        "z": round(float(z), 4),
        "p": round(p, 6),
        "significant": bool(p < 0.05),
    }


def _validate_roc_inputs(df: pd.DataFrame, score_col: str, outcome_col: str,
                         imputation: str = "listwise"):
    """Validate + return (scores_arr, y_arr, clean_df). Raises HTTPException on error."""
    for col in [score_col, outcome_col]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    df = apply_imputation(df, [score_col, outcome_col], imputation)
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Not enough data (need ≥ 10 rows after removing missing)")
    try:
        y = df[outcome_col].astype(float).astype(int)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Outcome '{outcome_col}' could not be converted to 0/1")
    uniq = sorted(y.unique().tolist())
    if len(uniq) != 2:
        raise HTTPException(status_code=400, detail=f"Outcome must have exactly 2 unique values. Found: {uniq[:6]}")
    if set(uniq) != {0, 1}:
        raise HTTPException(status_code=400, detail=f"Outcome values must be 0 and 1. Found: {uniq}")
    try:
        scores = df[score_col].astype(float)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Score column '{score_col}' must be numeric")
    if scores.nunique() < 2:
        raise HTTPException(status_code=400, detail=f"Score column '{score_col}' has no variation (constant)")
    return scores.values, y.values, df


@router.post("/roc")
def roc_analysis(req: ROCRequest):
    from sklearn.metrics import roc_curve, roc_auc_score

    df_full = _get_df(req.session_id)
    scores_arr, y_arr, df = _validate_roc_inputs(
        df_full, req.score_column, req.outcome_column,
        imputation=req.imputation or "listwise"
    )

    try:
        fpr, tpr, thresholds = roc_curve(y_arr, scores_arr)
        auc = float(roc_auc_score(y_arr, scores_arr))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ROC computation failed: {exc}")

    # Youden's J optimal cutoff
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    optimal = _roc_metrics_at_cutoff(scores_arr, y_arr, best_thresh)

    # Manual cutoff (if provided)
    manual = None
    if req.manual_cutoff is not None:
        manual = _roc_metrics_at_cutoff(scores_arr, y_arr, req.manual_cutoff)

    # Downsample curve to 300 points for response size
    n_pts = len(fpr)
    step = max(1, n_pts // 300)
    # Always include first and last points
    indices = list(range(0, n_pts, step))
    if (n_pts - 1) not in indices:
        indices.append(n_pts - 1)
    # Each curve point now carries the full clinical diagnostic table
    # (sens / spec / PPV / NPV / LR+ / LR-) so the UI can render an interactive
    # threshold-table without a second round-trip.
    curve = []
    for i in indices:
        thr = float(thresholds[i])
        m = _roc_metrics_at_cutoff(scores_arr, y_arr, thr)
        curve.append({
            "fpr": round(float(fpr[i]), 6),
            "tpr": round(float(tpr[i]), 6),
            "threshold": round(thr, 6),
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "ppv": m["ppv"],
            "npv": m["npv"],
            "lr_pos": m["lr_pos"],
            "lr_neg": m["lr_neg"],
            "youden_j": m["youden_j"],
        })

    return _sanitize({
        "test": "ROC Analysis",
        "n": len(df),
        "n_positive": int(y_arr.sum()),
        "n_negative": int((y_arr == 0).sum()),
        "auc": round(auc, 4),
        # Optimal (Youden's J) — kept at top level for backward compat
        "optimal_cutoff": optimal["cutoff"],
        "sensitivity": optimal["sensitivity"],
        "specificity": optimal["specificity"],
        "tp": optimal["tp"], "tn": optimal["tn"],
        "fp": optimal["fp"], "fn": optimal["fn"],
        # Full metric objects
        "optimal": optimal,
        "manual": manual,
        "curve": curve,
        "interpretation": (
            f"AUC = {auc:.3f} — "
            f"{'Excellent' if auc >= 0.9 else 'Good' if auc >= 0.8 else 'Fair' if auc >= 0.7 else 'Poor'} "
            "discriminative ability"
        ),
        "result_text": (
            f"ROC analysis was performed for {req.score_column} predicting {req.outcome_column} (n = {len(df)}). "
            f"The area under the curve was {auc:.3f}, indicating "
            f"{'excellent' if auc >= 0.9 else 'good' if auc >= 0.8 else 'fair' if auc >= 0.7 else 'poor'} discrimination. "
            f"At the optimal cutoff ({optimal['cutoff']:.3f}, Youden's J), sensitivity was {optimal['sensitivity']*100:.1f}% "
            f"and specificity was {optimal['specificity']*100:.1f}%."
        ),
    })


# ── ROC Comparison (DeLong Test) ──────────────────────────────────────────────

class ROCCompareRequest(BaseModel):
    session_id: str
    score_column_1: str
    score_column_2: str
    outcome_column: str


@router.post("/roc_compare")
def roc_compare(req: ROCCompareRequest):
    from sklearn.metrics import roc_curve, roc_auc_score

    df_full = _get_df(req.session_id)
    s1_arr, y_arr, _  = _validate_roc_inputs(df_full, req.score_column_1, req.outcome_column)
    s2_arr, y_arr2, _ = _validate_roc_inputs(df_full, req.score_column_2, req.outcome_column)

    if not np.array_equal(y_arr, y_arr2):
        # Different NaN patterns — use common complete rows (DeLong requires paired data)
        df_clean = df_full.dropna(subset=[req.score_column_1, req.score_column_2, req.outcome_column])
        if len(df_clean) < 10:
            raise HTTPException(status_code=400, detail="Not enough complete rows for comparison (need ≥ 10)")
        y_arr  = df_clean[req.outcome_column].astype(float).astype(int).values
        s1_arr = df_clean[req.score_column_1].astype(float).values
        s2_arr = df_clean[req.score_column_2].astype(float).values

    result = _delong_compare(y_arr, s1_arr, s2_arr)
    result["score_1"] = req.score_column_1
    result["score_2"] = req.score_column_2
    result["n"] = int(len(y_arr))

    # ROC curves for both models (for the overlaid publication plot)
    def _roc_curve_pts(scores, y):
        fpr, tpr, _ = roc_curve(y, scores)
        n_pts = len(fpr)
        step = max(1, n_pts // 300)
        idx = list(range(0, n_pts, step))
        if (n_pts - 1) not in idx:
            idx.append(n_pts - 1)
        return [{"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6)} for i in idx]

    result["curve_1"] = _roc_curve_pts(s1_arr, y_arr)
    result["curve_2"] = _roc_curve_pts(s2_arr, y_arr)

    auc1, auc2 = result["auc_1"], result["auc_2"]
    diff = result["difference"]
    p = result["p"]
    p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
    ci_lo = result["ci_diff_low"]
    ci_hi = result["ci_diff_high"]
    winner = req.score_column_1 if diff > 0 else req.score_column_2
    loser  = req.score_column_2 if diff > 0 else req.score_column_1
    higher_auc = max(auc1, auc2)
    lower_auc  = min(auc1, auc2)

    # CI bounds should always be reported low→high
    ci_report_lo = min(ci_lo, ci_hi)
    ci_report_hi = max(ci_lo, ci_hi)

    if result["significant"]:
        result["interpretation"] = (
            f"{winner} significantly improved discrimination over {loser} "
            f"(AUC {higher_auc:.3f} vs. {lower_auc:.3f}; "
            f"ΔAUC = {abs(diff):.3f}, 95% CI: {ci_report_lo:.3f}–{ci_report_hi:.3f}, "
            f"DeLong p = {p_str})."
        )
    else:
        result["interpretation"] = (
            f"No significant difference between {req.score_column_1} and {req.score_column_2} "
            f"(AUC {auc1:.3f} vs. {auc2:.3f}; "
            f"ΔAUC = {abs(diff):.3f}, 95% CI: {ci_report_lo:.3f}–{ci_report_hi:.3f}, "
            f"DeLong p = {p_str})."
        )

    result["result_text"] = result["interpretation"]
    return _sanitize(result)


# ── ROC Combined Model ─────────────────────────────────────────────────────────

class ROCCombinedRequest(BaseModel):
    session_id: str
    predictor_columns: List[str]
    outcome_column: str
    model_name: Optional[str] = "Combined Model"


@router.post("/roc_combined")
def roc_combined(req: ROCCombinedRequest):
    """Fit a logistic regression on selected predictors, then run ROC on predicted probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    df_full = _get_df(req.session_id)

    if req.outcome_column not in df_full.columns:
        raise HTTPException(status_code=400, detail=f"Outcome column '{req.outcome_column}' not found")
    missing_cols = [c for c in req.predictor_columns if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Predictor column(s) not found: {missing_cols}")
    if len(req.predictor_columns) < 1:
        raise HTTPException(status_code=400, detail="At least one predictor column is required")

    cols = req.predictor_columns + [req.outcome_column]
    df = df_full.dropna(subset=cols)
    if len(df) < 20:
        raise HTTPException(status_code=400, detail="Not enough complete rows after removing missing (need ≥ 20)")

    # Encode predictors: numeric → use as-is, categorical → one-hot
    parts = []
    for col in req.predictor_columns:
        col_s = df[col]
        if pd.api.types.is_numeric_dtype(col_s):
            parts.append(col_s.rename(col).to_frame())
        else:
            parts.append(pd.get_dummies(col_s, prefix=col, drop_first=True))
    X = pd.concat(parts, axis=1).astype(float).values

    try:
        y = df[req.outcome_column].astype(float).astype(int).values
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Outcome could not be converted to 0/1 integers")
    uniq = sorted(set(y.tolist()))
    if set(uniq) != {0, 1}:
        raise HTTPException(status_code=400, detail=f"Outcome must be exactly 0 and 1. Found: {uniq}")

    # Fit logistic regression with cross-validated predictions to avoid overfitting bias
    try:
        from sklearn.model_selection import cross_val_predict
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        n_cv = min(10, max(3, len(y) // 10))  # adaptive CV folds
        prob = cross_val_predict(model, X_sc, y, cv=n_cv, method="predict_proba")[:, 1]
        # Also fit the full model for coefficients / reporting
        model.fit(X_sc, y)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model fitting failed: {exc}")

    # ROC on predicted probabilities
    try:
        fpr, tpr, thresholds = roc_curve(y, prob)
        auc = float(roc_auc_score(y, prob))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ROC computation failed: {exc}")

    # Youden's J optimal cutoff on probabilities
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    optimal = _roc_metrics_at_cutoff(prob, y, best_thresh)

    # Downsample curve to 300 points
    n_pts = len(fpr)
    step = max(1, n_pts // 300)
    indices = list(range(0, n_pts, step))
    if (n_pts - 1) not in indices:
        indices.append(n_pts - 1)
    curve = [
        {"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6)}
        for i in indices
    ]

    return _sanitize({
        "test": "ROC Analysis (Combined Model)",
        "model_name": req.model_name,
        "predictors": req.predictor_columns,
        "n": int(len(df)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "auc": round(auc, 4),
        "optimal": optimal,
        "curve": curve,
        "result_text": (
            f"A combined model ({req.model_name}) using {len(req.predictor_columns)} predictors "
            f"({', '.join(req.predictor_columns)}) was evaluated (n = {len(df)}). "
            f"The AUC was {auc:.3f}, indicating "
            f"{'excellent' if auc >= 0.9 else 'good' if auc >= 0.8 else 'fair' if auc >= 0.7 else 'poor'} discrimination. "
            f"At the optimal cutoff ({optimal['cutoff']:.3f}), sensitivity was {optimal['sensitivity']*100:.1f}% "
            f"and specificity was {optimal['specificity']*100:.1f}%."
        ),
    })


# ── Sparklines (mini per-column distribution data for variable lists) ─────────

@router.get("/{session_id}/sparklines")
def get_sparklines(session_id: str):
    df = _get_df(session_id)
    result = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            result[col] = {"type": "empty", "data": []}
            continue
        if pd.api.types.is_numeric_dtype(s):
            n_bins = min(14, max(4, int(len(s) ** 0.38)))
            counts, _ = np.histogram(s, bins=n_bins)
            result[col] = {"type": "numeric", "data": counts.tolist()}
        else:
            vc = s.value_counts(normalize=True)
            n_cats = min(6, len(vc))
            result[col] = {
                "type": "categorical",
                "data": [float(v) for v in vc.head(n_cats).values],
                "labels": vc.head(n_cats).index.astype(str).tolist(),
            }
    return result


# ── Raw column values (for SPLOM scatterplot matrix) ─────────────────────────

@router.get("/{session_id}/refresh")
def refresh_session(session_id: str):
    """Return updated session metadata after in-place operations (e.g. melt/compute)."""
    import json as _json
    df = _get_df(session_id)
    from routers.upload import _detect_kind
    columns = []
    for col in df.columns:
        kind = _detect_kind(df[col])
        columns.append({"name": col, "dtype": str(df[col].dtype), "kind": kind})
    preview_df = df.head(2000).replace([np.inf, -np.inf], np.nan)
    preview = _json.loads(preview_df.to_json(orient="records", default_handler=str, date_format="iso", date_unit="s"))
    return {"rows": len(df), "columns": columns, "preview": preview}


@router.get("/{session_id}/raw")
def get_raw_columns(session_id: str, columns: str = ""):
    df = _get_df(session_id)
    cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns] if columns else list(df.columns)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])][:12]  # limit to 12 numeric cols
    result = {}
    for col in cols:
        vals = df[col].where(df[col].notna(), other=None).tolist()[:3000]
        result[col] = vals
    return result


# ── Column Summary (Wizard-style: histogram+QQ or donut+bar) ─────────────────

@router.get("/{session_id}/column_summary")
def column_summary(session_id: str, column: str, kind: Optional[str] = None):
    df = _get_df(session_id)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
    s = df[column]

    # Use provided kind hint; fall back to dtype + nunique heuristic
    if kind == "numeric":
        is_num = True
    elif kind in ("categorical", "text", "boolean"):
        is_num = False
    else:
        is_num = pd.api.types.is_numeric_dtype(s) and s.nunique() > 10

    if is_num:
        s_clean = s.dropna().astype(float)
        n_clean = len(s_clean)
        # Histogram (auto bins, max 40)
        n_bins = min(40, max(10, int(np.sqrt(n_clean))))
        counts, edges = np.histogram(s_clean, bins=n_bins)
        histogram = [
            {"bin_start": float(edges[i]), "bin_end": float(edges[i+1]), "count": int(counts[i])}
            for i in range(len(counts))
        ]
        # QQ plot
        (theo, sample), _ = scipy_stats.probplot(s_clean)
        step = max(1, len(theo) // 300)
        qq = [{"x": float(theo[i]), "y": float(sample[i])} for i in range(0, len(theo), step)]
        # Normality: Shapiro-Wilk for n<50, Kolmogorov-Smirnov for n≥50
        p_norm, norm_test_name = _normality_test(s_clean)
        mean_val = float(s_clean.mean())
        std_val  = float(s_clean.std())
        q1, q3 = float(s_clean.quantile(0.25)), float(s_clean.quantile(0.75))
        iqr_val = q3 - q1
        # IQR-based Tukey fences
        fence_low  = q1 - 1.5 * iqr_val
        fence_high = q3 + 1.5 * iqr_val
        # Actual whisker ends = most-extreme non-outlier values
        non_out = s_clean[(s_clean >= fence_low) & (s_clean <= fence_high)]
        whisker_low  = float(non_out.min()) if len(non_out) else float(s_clean.min())
        whisker_high = float(non_out.max()) if len(non_out) else float(s_clean.max())
        # IQR outliers with 1-based row index
        out_mask = (s_clean < fence_low) | (s_clean > fence_high)
        outliers = [
            {"row": int(idx) + 1, "value": float(val)}
            for idx, val in zip(s_clean.index[out_mask], s_clean[out_mask])
        ]
        # Z-score extremes and Normality deviants
        z_extremes = []
        normality_deviants = []
        if std_val > 0 and n_clean >= 3:
            z_series = (s_clean - mean_val) / std_val
            s_sorted_idx = s_clean.sort_values().index
            s_sorted_vals = s_clean.loc[s_sorted_idx].values
            
            # Calculate theoretical positions and residuals for all points
            all_points_info = []
            for i, idx in enumerate(s_sorted_idx):
                val = float(s_sorted_vals[i])
                rank = i + 1
                theo_q = float(scipy_stats.norm.ppf((rank - 0.375) / (n_clean + 0.25)))
                expected_val = mean_val + std_val * theo_q
                residual = val - expected_val
                z = float(z_series[idx])
                
                info = {
                    "row": int(idx) + 1,
                    "value": round(val, 4),
                    "z": round(z, 3),
                    "residual": round(residual, 4),
                    "abs_residual": abs(residual),
                    "qq_x": round(theo_q, 4)
                }
                all_points_info.append(info)
                if abs(z) > 2.0:
                    z_extremes.append(info)
            
            # Sort by absolute residual to find points most responsible for non-normality
            all_points_info.sort(key=lambda d: d["abs_residual"], reverse=True)
            normality_deviants = all_points_info[:10]  # Top 10 worst offenders
            
            # Sort z_extremes by |z| desc
            z_extremes.sort(key=lambda d: abs(d["z"]), reverse=True)

        return {
            "type": "numeric",
            "n": int(s_clean.count()), "missing": int(s.isna().sum()),
            "mean": mean_val, "std": std_val,
            "median": float(s_clean.median()), "q1": q1, "q3": q3,
            "iqr": float(iqr_val), "min": float(s_clean.min()), "max": float(s_clean.max()),
            "skewness": float(s_clean.skew()), "kurtosis": float(s_clean.kurtosis()),
            "whisker_low": whisker_low, "whisker_high": whisker_high,
            "outliers": outliers,
            "z_extremes": z_extremes,
            "normality_deviants": normality_deviants,
            "histogram": histogram,
            "raw_values": s_clean.sample(min(2000, n_clean), random_state=42).tolist(),
            "qq": qq,
            "normality_p": float(p_norm),
            "normality_test": norm_test_name,
            "normal": bool(p_norm > 0.05),
            "normality_label": "Normally distributed" if p_norm > 0.05 else "Non-normal distribution",
        }

    else:
        total = len(s)
        vc = s.value_counts(dropna=False)
        categories = [
            {"value": str(k) if pd.notna(k) else "Missing",
             "count": int(v), "pct": round(v / total * 100, 1)}
            for k, v in vc.items()
        ]
        return {
            "type": "categorical",
            "n": int(s.count()), "missing": int(s.isna().sum()),
            "n_categories": int(s.nunique()),
            "categories": categories,
        }


# ── Table 1 (clinical baseline characteristics) ───────────────────────────────

class Table1Request(BaseModel):
    session_id: str
    group_column: Optional[str] = None
    variables: list[str]
    variable_kinds: Optional[dict] = None   # {col: "numeric"|"categorical"}
    selected_stats: Optional[list[str]] = None  # ["auto","mean_sd","median_iqr","se","ci95","variance","min_max","n","missing","p10","p25","p75","p90","p95"]
    normality_mode: Optional[str] = "overall"  # "overall" or "within_group"
    # within_group: run normality on each group separately; parametric path
    #   used only if EVERY group passes (p > 0.05). More conservative — matches
    #   the actual assumption of t-test/ANOVA. Falls back to overall when
    #   group_column is null or only one group has data.


def _fmt_p(p: float) -> str:
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"


# ── per-stat formatters ────────────────────────────────────────────────────────

_STAT_LABELS: dict[str, str] = {
    "mean_sd":    "Mean ± SD",
    "median_iqr": "Median [IQR]",
    "se":         "SE of Mean",
    "ci95":       "95% CI",
    "variance":   "Variance",
    "min_max":    "Min – Max",
    "n":          "N (non-missing)",
    "missing":    "Missing",
    "p10":        "10th Pctl",
    "p25":        "25th Pctl",
    "p75":        "75th Pctl",
    "p90":        "90th Pctl",
    "p95":        "95th Pctl",
}


def _f(v: float, d: int = 2) -> str:
    """Format a float safely; return '—' for NaN/Inf."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{d}f}"


def _fmt_one_stat(a: pd.Series, stat: str) -> str:
    """Format a single statistic for a series (already dropna'd & float)."""
    if len(a) == 0:
        return "—"
    if stat == "mean_sd":
        return f"{_f(a.mean())} ± {_f(a.std())}"
    if stat == "median_iqr":
        q1, q3 = a.quantile(0.25), a.quantile(0.75)
        return f"{_f(a.median())} [{_f(q1)}–{_f(q3)}]"
    if stat == "se":
        return _f(a.sem(), 3)
    if stat == "ci95":
        if len(a) < 2:
            return "—"
        se = a.sem()
        m = a.mean()
        t_crit = scipy_stats.t.ppf(0.975, df=len(a) - 1)
        ci = t_crit * se
        return f"{_f(m)} [{_f(m - ci)}–{_f(m + ci)}]"
    if stat == "variance":
        return _f(a.var(), 3)
    if stat == "min_max":
        return f"{_f(a.min())} – {_f(a.max())}"
    if stat == "n":
        return str(int(len(a)))
    if stat == "missing":
        return str(int(a.isna().sum()) if hasattr(a, 'isna') else 0)
    pct_map = {"p10": 0.10, "p25": 0.25, "p75": 0.75, "p90": 0.90, "p95": 0.95}
    if stat in pct_map:
        return _f(a.quantile(pct_map[stat]))
    return "—"


def _build_stat_rows(
    s_col: pd.Series,
    group_series: dict[str, pd.Series],  # gl → series (not yet dropna'd)
    stats: list[str],
    normal: bool,
) -> list[dict]:
    """Build a list of {label, overall, group_stats} for numeric variable."""
    rows_out = []
    s_all = s_col.dropna().astype(float)

    # Handle 'missing' stat specially (needs original series)
    for stat in stats:
        resolved = stat
        if stat == "auto":
            resolved = "mean_sd" if normal else "median_iqr"

        label = _STAT_LABELS.get(resolved, resolved)
        if resolved == "missing":
            overall_val = str(int(s_col.isna().sum()))
            grp_vals = {gl: str(int(gs.isna().sum())) for gl, gs in group_series.items()}
        else:
            overall_val = _fmt_one_stat(s_all, resolved)
            grp_vals = {
                gl: _fmt_one_stat(gs.dropna().astype(float), resolved)
                for gl, gs in group_series.items()
            }

        rows_out.append({"label": label, "overall": overall_val, "group_stats": grp_vals})
    return rows_out


def _normality_test(s_clean: pd.Series) -> tuple[float, str]:
    """Return (p_value, test_name).

    Tier 1: n < 50   → Shapiro-Wilk (most powerful for small samples)
    Tier 2: 50 ≤ n ≤ 2000 → Kolmogorov-Smirnov with Lilliefors correction
    Tier 3: n > 2000 → CLT skewness bypass → Lilliefors
    """
    n = len(s_clean)
    if n < 3:
        return 1.0, "—"
    if n < 50:
        _, p = scipy_stats.shapiro(s_clean)
        return float(p), "Shapiro-Wilk"
    if n <= 2000:
        from statsmodels.stats.diagnostic import lilliefors as _lilliefors
        _, p = _lilliefors(s_clean.values, dist="norm")
        return float(p), "Kolmogorov-Smirnov (Lilliefors)"
    # Large n — check skewness first (CLT bypass)
    skewness = float(scipy_stats.skew(s_clean))
    if abs(skewness) <= 1.5:
        return 0.999, "Skewness (CLT bypass)"
    from statsmodels.stats.diagnostic import lilliefors as _lilliefors
    _, p = _lilliefors(s_clean.values, dist="norm")
    return float(p), "Kolmogorov-Smirnov (Lilliefors)"


@router.post("/table1")
def table1(req: Table1Request):
    df = _get_df(req.session_id)
    rows = []

    # Default stats = ["auto"] (normality-based)
    sel_stats: list[str] = req.selected_stats if req.selected_stats else ["auto"]

    groups = None
    group_labels = []
    group_ns: dict = {}
    if req.group_column and req.group_column in df.columns:
        groups = sorted(df[req.group_column].dropna().unique().tolist(), key=str)
        group_labels = [str(g) for g in groups]
        group_ns = {str(g): int((df[req.group_column] == g).sum()) for g in groups}

    for var in req.variables:
        if var not in df.columns:
            continue
        s = df[var]

        provided_kind = (req.variable_kinds or {}).get(var)
        if provided_kind == "numeric":
            is_num = True
        elif provided_kind in ("categorical", "text", "boolean"):
            is_num = False
        else:
            is_num = pd.api.types.is_numeric_dtype(s) and s.nunique() > 10

        if is_num:
            s_all = s.dropna().astype(float)
            p_norm, norm_test_name = _normality_test(s_all)
            normal_overall = p_norm > 0.05

            # Build per-group series map  {label → raw series}
            group_series: dict[str, pd.Series] = {}
            group_arrs: list[pd.Series] = []
            if groups is not None:
                for g, gl in zip(groups, group_labels):
                    gs = df[df[req.group_column] == g][var]
                    group_series[gl] = gs
                    group_arrs.append(gs.dropna().astype(float))

            # Per-group normality (optional, opt-in via normality_mode).
            # Parametric assumption is "normal within each group" — stricter
            # than overall normality.
            per_group_norm: dict[str, dict] = {}
            if (req.normality_mode == "within_group" and groups is not None
                    and len(group_arrs) >= 2):
                for gl, arr in zip(group_labels, group_arrs):
                    if len(arr) >= 3:
                        pg, pg_name = _normality_test(arr)
                        per_group_norm[gl] = {
                            "p": round(float(pg), 4),
                            "test": pg_name,
                            "normal": bool(pg > 0.05),
                            "n": int(len(arr)),
                        }
                    else:
                        # Too few obs to test — treat as non-normal (forces
                        # non-parametric path, safer default).
                        per_group_norm[gl] = {
                            "p": None,
                            "test": "n<3",
                            "normal": False,
                            "n": int(len(arr)),
                        }
                # Parametric path only if EVERY group is normal
                normal = (len(per_group_norm) > 0
                          and all(v["normal"] for v in per_group_norm.values()))
            else:
                normal = normal_overall

            stat_rows = _build_stat_rows(s, group_series, sel_stats, normal)

            # Statistical test for group comparison
            p_value_str: Optional[str] = None
            test_name_str: Optional[str] = None
            significant = False
            if groups is not None and len(group_arrs) >= 2:
                try:
                    if len(groups) == 2:
                        if normal:
                            _, p_t = scipy_stats.ttest_ind(*group_arrs, equal_var=False)
                            test_name_str = "t-test"
                        else:
                            _, p_t = scipy_stats.mannwhitneyu(*group_arrs, alternative="two-sided")
                            test_name_str = "Mann-Whitney"
                    else:
                        if normal:
                            _, p_t = scipy_stats.f_oneway(*group_arrs)
                            test_name_str = "ANOVA"
                        else:
                            _, p_t = scipy_stats.kruskal(*group_arrs)
                            test_name_str = "Kruskal-Wallis"
                    p_value_str = _fmt_p(float(p_t))
                    significant = bool(float(p_t) < 0.05)
                except Exception:
                    p_value_str = "N/A"

            # SMD (Standardized Mean Difference) — only for 2-group comparisons
            smd_val: Optional[float] = None
            if groups is not None and len(group_arrs) == 2:
                try:
                    g1, g2 = group_arrs[0], group_arrs[1]
                    if len(g1) > 0 and len(g2) > 0:
                        pooled_std = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
                        if pooled_std > 0:
                            smd_val = round(float(abs(g1.mean() - g2.mean()) / pooled_std), 4)
                except Exception:
                    pass

            row: dict = {
                "variable": var,
                "type": "numeric",
                "overall_n": int(len(s_all)),
                "normal": normal,
                "normality_test": norm_test_name,
                "normality_p": round(p_norm, 4),
                "normality_mode": req.normality_mode or "overall",
                "per_group_normality": per_group_norm,  # {} when overall mode
                "stat_rows": stat_rows,
                "p_value": p_value_str,
                "test": test_name_str,
                "significant": significant,
                "smd": smd_val,
                # Legacy fields (for backward compat)
                "stat_label": stat_rows[0]["label"] if stat_rows else "",
                "overall": stat_rows[0]["overall"] if stat_rows else "",
                "group_stats": stat_rows[0]["group_stats"] if stat_rows else {},
            }

        else:
            # Categorical
            vc_all = s.value_counts(dropna=True)
            total_all = s.count()
            cats = [str(v) for v in vc_all.index.tolist()]
            sub_rows = []
            for cat in cats:
                n_all = int((s.astype(str) == cat).sum())
                pct_all = round(n_all / total_all * 100, 1) if total_all else 0
                sub: dict = {"category": cat, "overall": f"{n_all} ({pct_all}%)", "group_stats": {}}
                if groups is not None:
                    for g, gl in zip(groups, group_labels):
                        g_s = df[df[req.group_column] == g][var]
                        n_g = int((g_s.astype(str) == cat).sum())
                        t_g = g_s.count()
                        pct_g = round(n_g / t_g * 100, 1) if t_g else 0
                        sub["group_stats"][gl] = f"{n_g} ({pct_g}%)"
                sub_rows.append(sub)

            p_val: Optional[str] = None
            test_name: Optional[str] = None
            p_chi_raw: Optional[float] = None
            if groups is not None:
                try:
                    ct = pd.crosstab(df[var].astype(str), df[req.group_column])
                    chi2, p_chi_raw, dof, expected = scipy_stats.chi2_contingency(ct)
                    if ct.shape == (2, 2) and (expected < 5).any():
                        _, p_chi_raw = scipy_stats.fisher_exact(ct.values)
                        test_name = "Fisher"
                    else:
                        test_name = "Chi-square"
                    p_val = _fmt_p(float(p_chi_raw))
                except Exception:
                    p_val = "N/A"

            # SMD for categorical (proportion difference / pooled SE) — 2-group only
            cat_smd: Optional[float] = None
            if groups is not None and len(groups) == 2:
                try:
                    g1_s = df[df[req.group_column] == groups[0]][var].astype(str)
                    g2_s = df[df[req.group_column] == groups[1]][var].astype(str)
                    all_cats = sorted(set(g1_s.dropna()) | set(g2_s.dropna()))
                    if len(all_cats) == 2:
                        # Binary: use simple proportion SMD
                        target = all_cats[0]
                        p1 = (g1_s == target).mean()
                        p2 = (g2_s == target).mean()
                        pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
                        if pooled > 0:
                            cat_smd = round(float(abs(p1 - p2) / pooled), 4)
                    elif len(all_cats) > 2:
                        # Multi-category: Mahalanobis-like SMD (Yang & Dalton 2012)
                        p1_vec = np.array([(g1_s == c).mean() for c in all_cats[:-1]])
                        p2_vec = np.array([(g2_s == c).mean() for c in all_cats[:-1]])
                        s1 = np.diag(p1_vec * (1 - p1_vec))
                        s2 = np.diag(p2_vec * (1 - p2_vec))
                        s_pool = (s1 + s2) / 2
                        diff = p1_vec - p2_vec
                        det = np.linalg.det(s_pool)
                        if det > 1e-12:
                            cat_smd = round(float(np.sqrt(diff @ np.linalg.inv(s_pool) @ diff)), 4)
                except Exception:
                    pass

            row = {
                "variable": var,
                "type": "categorical",
                "stat_label": "n (%)",
                "overall": f"n={total_all}",
                "overall_n": int(total_all),
                "p_value": p_val,
                "test": test_name,
                "significant": bool(p_chi_raw is not None and p_chi_raw < 0.05),
                "sub_rows": sub_rows,
                "group_stats": {},
                "stat_rows": [],
                "smd": cat_smd,
            }
        rows.append(row)

    return _sanitize({
        "group_column": req.group_column,
        "group_labels": group_labels,
        "group_ns": group_ns,
        "total_n": len(df),
        "rows": rows,
    })


# ── ANOVA ─────────────────────────────────────────────────────────────────────

class AnovaRequest(BaseModel):
    session_id: str
    column: str
    group_column: str


@router.post("/anova")
def anova(req: AnovaRequest):
    df = _get_df(req.session_id)
    grp_dict = {str(name): g[req.column].dropna().astype(float).values
                for name, g in df.groupby(req.group_column)}
    group_arrays = list(grp_dict.values())
    group_names = list(grp_dict.keys())
    if len(group_arrays) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 groups")

    stat, p = scipy_stats.f_oneway(*group_arrays)
    sig = bool(p < 0.05)
    k = len(group_arrays)
    n_total = sum(len(g) for g in group_arrays)
    df_between = k - 1
    df_within = n_total - k
    # MS_within for omega-squared
    grand_mean = np.concatenate(group_arrays).mean()
    ss_within = sum(np.sum((g - g.mean())**2) for g in group_arrays)
    ms_within = ss_within / df_within if df_within > 0 else 1

    es_eta = eta_squared(float(stat), df_between, df_within)
    es_omega = omega_squared(float(stat), df_between, df_within, ms_within)

    # Assumption checks
    assumptions = [check_equal_variances(group_arrays, group_names)]
    for name, arr in grp_dict.items():
        assumptions.append(check_normality(arr, name))

    # Post-hoc tests (if significant and > 2 groups)
    posthoc = []
    posthoc_method = None
    if sig and k > 2:
        equal_var = assumptions[0]["met"]
        if equal_var:
            posthoc = tukey_hsd(grp_dict)
            posthoc_method = "Tukey HSD"
        else:
            posthoc = games_howell(grp_dict)
            posthoc_method = "Games-Howell (unequal variances)"

    p_str = '<0.001' if p < 0.001 else f'{p:.4f}'
    group_stats = df.groupby(req.group_column)[req.column].agg(["count", "mean", "std"]).reset_index()
    ret = {
        "test": "One-way ANOVA",
        "F": float(stat), "p": float(p),
        "df_between": df_between, "df_within": df_within,
        "significant": sig,
        "effect_sizes": [es_eta, es_omega],
        "assumptions": assumptions,
        "posthoc": posthoc,
        "posthoc_method": posthoc_method,
        "groups": [
            {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in row.items()}
            for row in group_stats.to_dict(orient="records")
        ],
        "interpretation": f"{'Significant' if sig else 'No significant'} difference across groups (F({df_between},{df_within}) = {stat:.2f}, p = {p_str}, \u03B7\u00B2 = {es_eta['value']:.3f} [{es_eta['magnitude']}])",
        "methods_text": methods_anova(req.column, req.group_column),
        "r_code": r_anova(req.column, req.group_column),
    }
    ret["result_text"] = results_anova(ret)
    return ret


# ── Pairwise Correlation ──────────────────────────────────────────────────────

class CorrelationPairRequest(BaseModel):
    session_id: str
    var1: str
    var2: str
    method: Optional[str] = "auto"   # "auto" | "pearson" | "spearman"
    imputation: Optional[str] = "listwise"


@router.post("/correlation_pair")
def correlation_pair(req: CorrelationPairRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.var1, req.var2], req.imputation or "listwise")
    x = df[req.var1].astype(float).values
    y = df[req.var2].astype(float).values
    n = len(x)
    n_excluded = n_total - n
    if n < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 observations")

    # ── Normality assessment ──────────────────────────────────────────────────
    # Three-tier strategy matching SPSS conventions:
    #
    # Tier 1 (n < 50): Shapiro-Wilk — most powerful for small samples.
    # Tier 2 (50 ≤ n ≤ 2000): Kolmogorov-Smirnov with Lilliefors correction.
    # Tier 3 (n > 2000): CLT bypass if |skewness| ≤ 1.5, else Lilliefors.

    def _assess_normality(arr: np.ndarray) -> dict:
        _n = len(arr)
        skewness = float(scipy_stats.skew(arr))

        if _n < 50:
            stat, p_val = scipy_stats.shapiro(arr)
            return {
                "statistic": float(stat),
                "p": float(p_val),
                "normal": bool(p_val >= 0.05),
                "skewness": skewness,
                "test": "Shapiro-Wilk",
                "bypass": None,
            }

        # Medium n (50–2000) — Kolmogorov-Smirnov with Lilliefors correction
        if _n <= 2000:
            from statsmodels.stats.diagnostic import lilliefors as _lilliefors
            stat, p_val = _lilliefors(arr, dist="norm")
            return {
                "statistic": float(stat),
                "p": float(p_val),
                "normal": bool(p_val >= 0.05),
                "skewness": skewness,
                "test": "Kolmogorov-Smirnov (Lilliefors)",
                "bypass": None,
            }

        # Large n (>2000) — CLT bypass if skewness is mild
        if abs(skewness) <= 1.5:
            return {
                "statistic": None,
                "p": None,
                "normal": True,
                "skewness": skewness,
                "test": "Skewness (CLT bypass)",
                "bypass": "clt_skew",
            }

        # Large n with marked skewness — Lilliefors
        from statsmodels.stats.diagnostic import lilliefors as _lilliefors
        stat, p_val = _lilliefors(arr, dist="norm")
        return {
            "statistic": float(stat),
            "p": float(p_val),
            "normal": bool(p_val >= 0.05),
            "skewness": skewness,
            "test": "Kolmogorov-Smirnov (Lilliefors)",
            "bypass": None,
        }

    norm1 = _assess_normality(x)
    norm2 = _assess_normality(y)
    normal1 = norm1["normal"]
    normal2 = norm2["normal"]

    # Top-level test label for display (most conservative test used)
    _tests_used = {norm1["test"], norm2["test"]}
    if any("Kolmogorov" in t or "Lilliefors" in t for t in _tests_used):
        norm_test_name = "Kolmogorov-Smirnov (Lilliefors)"
    elif "Shapiro-Wilk" in _tests_used:
        norm_test_name = "Shapiro-Wilk"
    else:
        norm_test_name = "Skewness (CLT bypass)"

    # Method selection
    method = req.method or "auto"
    if method == "auto":
        use_pearson = normal1 and normal2
    else:
        use_pearson = method == "pearson"

    if use_pearson:
        r, p = scipy_stats.pearsonr(x, y)
        method_used = "pearson"
        label = "r"
    else:
        r, p = scipy_stats.spearmanr(x, y)
        method_used = "spearman"
        label = "ρ"

    # 95% CI via Fisher z-transformation
    if abs(r) < 1.0:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        ci_low = float(np.tanh(z - 1.96 * se))
        ci_high = float(np.tanh(z + 1.96 * se))
    else:
        ci_low, ci_high = float(r), float(r)

    # Scatter data
    scatter_x = x.tolist()
    scatter_y = y.tolist()

    # Regression line (OLS) for plot
    slope, intercept, *_ = scipy_stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    # 95% CI band around regression line
    x_mean = x.mean()
    ss_x = np.sum((x - x_mean) ** 2)
    residuals = y - (slope * x + intercept)
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    t_crit = scipy_stats.t.ppf(0.975, df=n - 2)
    ci_band = t_crit * s_err * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_x)

    p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
    strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak" if abs(r) >= 0.2 else "negligible"
    direction = "positive" if r > 0 else "negative"

    return {
        "method": method_used,
        "label": label,
        "n": n,
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "r": float(r),
        "p": float(p),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "normality_test": norm_test_name,
        "normality": {
            req.var1: norm1,
            req.var2: norm2,
        },
        "scatter": {"x": scatter_x, "y": scatter_y},
        "regression_line": {
            "x": x_line.tolist(),
            "y": y_line.tolist(),
            "slope": float(slope),
            "intercept": float(intercept),
        },
        "ci_band": {
            "x": x_line.tolist(),
            "y_upper": (y_line + ci_band).tolist(),
            "y_lower": (y_line - ci_band).tolist(),
        },
        "result_text": (
            f"{'Pearson' if method_used == 'pearson' else 'Spearman'} correlation analysis revealed a "
            f"{strength} {direction} {'correlation' if p < 0.05 else 'but non-significant correlation'} "
            f"between {req.var1} and {req.var2} ({label} = {r:.3f}, 95% CI: {ci_low:.3f}–{ci_high:.3f}, "
            f"p = {p_str}, n = {n})."
        ),
    }


# ── Correlation Matrix ────────────────────────────────────────────────────────

class CorrelationMatrixRequest(BaseModel):
    session_id: str
    variables: List[str]
    method: Optional[str] = "pearson"
    imputation: Optional[str] = "listwise"


@router.post("/correlation_matrix")
def correlation_matrix_post(req: CorrelationMatrixRequest):
    raw = _get_df(req.session_id)[req.variables].apply(pd.to_numeric, errors="coerce")
    df = apply_imputation(raw, req.variables, req.imputation or "listwise")
    if len(req.variables) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 variables")

    method = req.method or "pearson"
    corr = df.corr(method=method)

    # p-value matrix (pairwise)
    p_matrix: dict = {}
    for c1 in req.variables:
        p_matrix[c1] = {}
        for c2 in req.variables:
            if c1 == c2:
                p_matrix[c1][c2] = None
            else:
                pair = df[[c1, c2]].dropna()
                if len(pair) < 3 or pair[c1].std() == 0 or pair[c2].std() == 0:
                    p_matrix[c1][c2] = None
                    continue
                try:
                    if method == "spearman":
                        _, pv = scipy_stats.spearmanr(pair[c1], pair[c2])
                    elif method == "kendall":
                        _, pv = scipy_stats.kendalltau(pair[c1], pair[c2])
                    else:
                        _, pv = scipy_stats.pearsonr(pair[c1], pair[c2])
                    p_matrix[c1][c2] = float(pv)
                except Exception:
                    p_matrix[c1][c2] = None

    # Multicollinearity warnings: |r| >= 0.70
    warnings = []
    vars_list = req.variables
    for i in range(len(vars_list)):
        for j in range(i + 1, len(vars_list)):
            r_val = corr.loc[vars_list[i], vars_list[j]]
            if abs(r_val) >= 0.70:
                warnings.append({
                    "var1": vars_list[i],
                    "var2": vars_list[j],
                    "r": float(r_val),
                    "severity": "high" if abs(r_val) >= 0.90 else "moderate",
                })

    matrix_dict = {c: {r: (float(corr.loc[r, c]) if not pd.isna(corr.loc[r, c]) else None)
                        for r in req.variables} for c in req.variables}

    return {
        "method": method,
        "variables": req.variables,
        "n": len(df),
        "matrix": matrix_dict,
        "p_matrix": p_matrix,
        "multicollinearity_warnings": warnings,
    }


# ── ICC(2,1) ──────────────────────────────────────────────────────────────────

class ICCRequest(BaseModel):
    session_id: str
    rater1_col: str
    rater2_col: str


@router.post("/icc")
def icc_endpoint(req: ICCRequest):
    df = _get_df(req.session_id).dropna(subset=[req.rater1_col, req.rater2_col])
    r1 = df[req.rater1_col].astype(float).values
    r2 = df[req.rater2_col].astype(float).values
    n = len(r1)
    k = 2  # raters
    if n < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 observations")

    # ANOVA decomposition for ICC(2,1) — Shrout & Fleiss 1979
    grand_mean = np.mean(np.stack([r1, r2]))
    subject_means = (r1 + r2) / 2.0
    rater_means = np.array([r1.mean(), r2.mean()])

    SS_b = k * np.sum((subject_means - grand_mean) ** 2)
    SS_r = n * np.sum((rater_means - grand_mean) ** 2)
    SS_total = np.sum((r1 - grand_mean) ** 2) + np.sum((r2 - grand_mean) ** 2)
    SS_e = SS_total - SS_b - SS_r

    df_b = n - 1
    df_r = k - 1
    df_e = (n - 1) * (k - 1)

    MS_b = SS_b / df_b
    MS_r = SS_r / df_r if df_r > 0 else 0.0
    MS_e = SS_e / df_e if df_e > 0 else 1e-9

    # ICC(2,1) absolute agreement
    icc_val = (MS_b - MS_e) / (MS_b + (k - 1) * MS_e + k * (MS_r - MS_e) / n)
    icc_val = float(np.clip(icc_val, -1.0, 1.0))

    # 95% CI (Shrout & Fleiss)
    F_lower = scipy_stats.f.ppf(0.975, df_b, df_e)
    F_upper = scipy_stats.f.ppf(0.025, df_b, df_e)
    F_obs = MS_b / MS_e if MS_e > 0 else 0.0
    ci_low = float((F_obs / F_lower - 1) / (F_obs / F_lower + k - 1)) if F_lower > 0 else 0.0
    ci_high = float((F_obs / F_upper - 1) / (F_obs / F_upper + k - 1)) if F_upper > 0 else 1.0
    ci_low = float(np.clip(ci_low, -1.0, 1.0))
    ci_high = float(np.clip(ci_high, -1.0, 1.0))

    # F-test p-value
    f_p = float(scipy_stats.f.sf(F_obs, df_b, df_e))

    # Interpretation
    if icc_val >= 0.90:
        interp = "Excellent"
    elif icc_val >= 0.75:
        interp = "Good"
    elif icc_val >= 0.50:
        interp = "Moderate"
    else:
        interp = "Poor"

    # Bland-Altman data
    means = ((r1 + r2) / 2).tolist()
    diffs = (r1 - r2).tolist()
    mean_diff = float(np.mean(r1 - r2))
    sd_diff = float(np.std(r1 - r2, ddof=1))
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    return {
        "icc": icc_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "f_stat": float(F_obs),
        "f_p": f_p,
        "n": n,
        "interpretation": interp,
        "bland_altman": {
            "means": means,
            "diffs": diffs,
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "loa_upper": float(loa_upper),
            "loa_lower": float(loa_lower),
        },
    }


# ── Cohen's Kappa ─────────────────────────────────────────────────────────────

class KappaRequest(BaseModel):
    session_id: str
    rater1_col: str
    rater2_col: str


@router.post("/cohens_kappa")
def cohens_kappa(req: KappaRequest):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix as sk_confusion

    df = _get_df(req.session_id).dropna(subset=[req.rater1_col, req.rater2_col])
    r1 = df[req.rater1_col].astype(str).values
    r2 = df[req.rater2_col].astype(str).values
    n = len(r1)
    if n < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 observations")

    kappa = float(cohen_kappa_score(r1, r2))

    # SE and 95% CI
    labels = sorted(set(r1) | set(r2))
    cm = sk_confusion(r1, r2, labels=labels)
    po = float(np.trace(cm) / n)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = float(np.sum(row_sums * col_sums) / (n ** 2))
    se = float(np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))) if (1 - pe) > 0 else 0.0
    ci_low = float(kappa - 1.96 * se)
    ci_high = float(kappa + 1.96 * se)

    # Landis & Koch interpretation
    if kappa >= 0.81:
        interp = "Almost Perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    elif kappa >= 0.0:
        interp = "Slight"
    else:
        interp = "Poor (< chance)"

    return {
        "kappa": kappa,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "n": n,
        "po": po,
        "pe": pe,
        "interpretation": interp,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


# ── Fleiss κ (≥3 raters) ───────────────────────────────────────────────────────

class FleissKappaRequest(BaseModel):
    session_id: str
    rater_cols: List[str]  # ≥3 raters


@router.post("/fleiss_kappa")
def fleiss_kappa_endpoint(req: FleissKappaRequest):
    """Fleiss κ for 3+ raters on a nominal/ordinal categorical outcome.

    Each rater column must contain the same set of categories. The aggregate
    table is N × k where N = subjects, k = categories. Each cell = number of
    raters assigning that category to that subject.

    Reports overall κ + per-category κ (a.k.a. PABAK / category-specific
    agreement) + Landis-Koch interpretation.
    """
    from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
    if len(req.rater_cols) < 3:
        raise HTTPException(status_code=422, detail="Fleiss κ requires ≥3 raters. Use Cohen's κ for 2 raters.")
    df = _get_df(req.session_id).dropna(subset=req.rater_cols)
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 subjects with complete ratings across all raters.")

    raters = df[req.rater_cols].astype(str).values  # shape (N, n_raters)
    table, categories = aggregate_raters(raters)  # table shape (N, k_categories)

    kappa = float(fleiss_kappa(table, method="fleiss"))
    # Standard error per Fleiss 1971: SE(κ) ≈ √(2/[Nn(n-1)]) under H₀=chance,
    # but Conger 1980 derived the proper SE. statsmodels has no SE; use the
    # asymptotic SE formula from Fleiss 1971 (chance-corrected, OK as 95% CI).
    n_subjects, k_cats = table.shape
    n_raters = int(table.sum(axis=1).mean())
    p_j = table.sum(axis=0) / (n_subjects * n_raters)
    p_e = float(np.sum(p_j ** 2))
    if (1 - p_e) > 0 and n_subjects > 0 and n_raters > 1:
        var_k = 2.0 / (n_subjects * n_raters * (n_raters - 1) * (1 - p_e) ** 2) * (
            p_e - (2 * n_raters - 3) * p_e ** 2 + 2 * (n_raters - 2) * float(np.sum(p_j ** 3))
        )
        se = float(np.sqrt(max(var_k, 0.0)))
    else:
        se = 0.0
    ci_low = float(kappa - 1.96 * se)
    ci_high = float(kappa + 1.96 * se)

    # Landis & Koch interpretation
    if kappa >= 0.81:
        interp = "Almost Perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    elif kappa >= 0.0:
        interp = "Slight"
    else:
        interp = "Poor (< chance)"

    # Per-category κ (Fleiss 1971, eq. 12 — proportion of agreement above chance for each category)
    per_category = []
    for j, cat in enumerate(categories):
        # κ_j = (p_jbar - p_j²) / (p_j (1 - p_j))
        p_j_val = float(p_j[j])
        # p_jbar = mean agreement on category j across subjects
        # using sum_i n_ij(n_ij - 1) / sum_i n_i(n_i - 1)
        num = float(np.sum(table[:, j] * (table[:, j] - 1)))
        den = float(np.sum(table.sum(axis=1) * (table.sum(axis=1) - 1)))
        p_jbar = num / den if den > 0 else 0.0
        if p_j_val > 0 and p_j_val < 1:
            kj = (p_jbar - p_j_val ** 2) / (p_j_val * (1 - p_j_val))
        else:
            kj = None
        per_category.append({
            "category": str(cat),
            "kappa": round(kj, 4) if kj is not None else None,
            "prevalence": round(p_j_val, 4),
        })

    return {
        "test": "Fleiss' κ",
        "kappa": round(kappa, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "se": round(se, 4),
        "n_subjects": int(n_subjects),
        "n_raters": int(n_raters),
        "n_categories": int(k_cats),
        "categories": [str(c) for c in categories],
        "per_category": per_category,
        "interpretation": interp,
        "result_text": (
            f"Fleiss' κ for {n_raters} raters on {n_subjects} subjects = {kappa:.3f} "
            f"(95% CI {ci_low:.3f} to {ci_high:.3f}) — {interp.lower()} agreement (Landis & Koch)."
        ),
    }


# ── Power Analysis ─────────────────────────────────────────────────────────────

class PowerRequest(BaseModel):
    test: str          # t_two | t_one | anova | correlation | proportion | chi2
    solve_for: str     # n | power | effect_size
    alpha: float = 0.05
    power: Optional[float] = None
    effect_size: Optional[float] = None
    n: Optional[int] = None
    tails: int = 2
    k_groups: int = 3   # ANOVA: number of groups; chi2: number of bins (df+1)
    ratio: float = 1.0  # n2/n1 for two-sample tests
    p1: Optional[float] = None
    p2: Optional[float] = None


@router.post("/power")
def run_power(req: PowerRequest):
    import numpy as np
    from scipy.stats import norm
    from statsmodels.stats.power import (
        TTestIndPower, TTestPower,
        FTestAnovaPower, NormalIndPower, GofChisquarePower,
    )

    alt = "two-sided" if req.tails == 2 else "larger"
    a   = req.alpha

    def _ceil(x): return int(np.ceil(float(x)))

    def _curve(pw_fn, n_end, n_start=4, steps=80):
        pts, step = [], max(1, (n_end - n_start) // steps)
        for n in range(n_start, n_end + 1, step):
            try:
                pwr = float(pw_fn(n))
                if 0 <= pwr <= 1:
                    pts.append({"n": n, "power": round(pwr, 4)})
            except Exception:
                pass
        return pts

    result, label, curve = None, "", []

    # ── Two-sample t-test ──────────────────────────────────────────────────────
    if req.test == "t_two":
        ana = TTestIndPower()
        ratio = req.ratio or 1.0
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt)

        if req.solve_for == "n":
            n1 = _ceil(ana.solve_power(effect_size=req.effect_size, nobs1=None, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            result = n1
            label  = f"n₁ = {n1},  n₂ = {_ceil(n1*ratio)},  total N = {n1 + _ceil(n1*ratio)}"
            curve  = _curve(pw, max(n1 * 4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs1=req.n, alpha=a, power=None, ratio=ratio, alternative=alt))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs1=req.n, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            label  = f"Minimum detectable Cohen's d = {result:.4f}"
            d = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=d, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt), max(int(req.n)*4, 100))

    # ── One-sample / paired t-test ─────────────────────────────────────────────
    elif req.test == "t_one":
        ana = TTestPower()
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, alternative=alt)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, alternative=alt))
            result, label, curve = n, f"n = {n}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, alternative=alt))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, alternative=alt))
            label  = f"Minimum detectable Cohen's d = {result:.4f}"
            d = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=d, nobs=n, alpha=a, power=None, alternative=alt), max(int(req.n)*4, 100))

    # ── One-way ANOVA ──────────────────────────────────────────────────────────
    elif req.test == "anova":
        ana, k = FTestAnovaPower(), req.k_groups
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, k_groups=k)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, k_groups=k))
            result, label, curve = n, f"n/group = {n},  total N = {n*k}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, k_groups=k))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, k_groups=k))
            label  = f"Minimum detectable Cohen's f = {result:.4f}"
            f_es = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=f_es, nobs=n, alpha=a, power=None, k_groups=k), max(int(req.n)*4, 100))

    # ── Pearson correlation (Fisher-z) ─────────────────────────────────────────
    elif req.test == "correlation":
        tails = req.tails

        def corr_power(r, n):
            if abs(r) >= 1 or n <= 3: return float("nan")
            ncp = np.arctanh(abs(r)) * np.sqrt(n - 3)
            z_c = norm.ppf(1 - a / (1 if tails == 1 else 2))
            return float(norm.sf(z_c - ncp) + (norm.cdf(-z_c - ncp) if tails == 2 else 0))

        def corr_solve_n(r, pwr):
            for n in range(4, 100001):
                if corr_power(r, n) >= pwr: return n
            return 100001

        def corr_solve_r(n, pwr):
            from scipy.optimize import brentq
            try:   return float(brentq(lambda r: corr_power(r, n) - pwr, 1e-6, 1 - 1e-6))
            except Exception: return None

        r_es = req.effect_size
        if req.solve_for == "n":
            n = corr_solve_n(r_es, req.power)
            result, label = n, f"n = {n}"
            curve = _curve(lambda n: corr_power(r_es, n), max(n*4, 100))
        elif req.solve_for == "power":
            result = corr_power(r_es, req.n)
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(lambda n: corr_power(r_es, n), max(int(req.n)*4, 100))
        else:
            r_sol = corr_solve_r(req.n, req.power)
            result = r_sol
            label  = f"Minimum detectable r = {r_sol:.4f}" if r_sol else "Could not converge"
            if r_sol:
                curve = _curve(lambda n: corr_power(r_sol, n), max(int(req.n)*4, 100))

    # ── Two proportions (Cohen's h) ────────────────────────────────────────────
    elif req.test == "proportion":
        ana   = NormalIndPower()
        ratio = req.ratio or 1.0
        p1    = req.p1 if req.p1 is not None else 0.5
        p2    = req.p2 if req.p2 is not None else 0.3
        h_from_p = abs(float(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))))

        if req.solve_for == "effect_size":
            eff = float(ana.solve_power(effect_size=None, nobs1=req.n, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            result = abs(eff)
            label  = f"Minimum detectable Cohen's h = {result:.4f}"
            h_sol = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=h_sol, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt), max(int(req.n)*4, 100))
        else:
            eff = req.effect_size if req.effect_size is not None else h_from_p
            def pw(n): return ana.solve_power(effect_size=eff, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt)
            if req.solve_for == "n":
                n1 = _ceil(ana.solve_power(effect_size=eff, nobs1=None, alpha=a, power=req.power, ratio=ratio, alternative=alt))
                result, label, curve = n1, f"n₁ = {n1},  n₂ = {_ceil(n1*ratio)},  total N = {n1+_ceil(n1*ratio)}", _curve(pw, max(n1*4, 100))
            else:
                result = float(ana.solve_power(effect_size=eff, nobs1=req.n, alpha=a, power=None, ratio=ratio, alternative=alt))
                label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
                curve  = _curve(pw, max(int(req.n)*4, 100))

    # ── Chi-square ─────────────────────────────────────────────────────────────
    elif req.test == "chi2":
        ana    = GofChisquarePower()
        n_bins = req.k_groups   # df = k_groups - 1
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, n_bins=n_bins)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, n_bins=n_bins))
            result, label, curve = n, f"n = {n}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, n_bins=n_bins))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, n_bins=n_bins))
            label  = f"Minimum detectable Cohen's w = {result:.4f}"
            w_es = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=w_es, nobs=n, alpha=a, power=None, n_bins=n_bins), max(int(req.n)*4, 100))
    else:
        raise HTTPException(400, f"Unknown test: {req.test}")

    result_text = _power_result_text(req, result)
    return {"result": float(result) if result is not None else None, "label": label, "curve": curve, "result_text": result_text}


def _power_result_text(req, result) -> str:
    """Generate a plain-English interpretation of the power analysis result."""
    if result is None:
        return ""

    test_names = {
        "t_two": "two-sample t-test", "t_one": "one-sample/paired t-test",
        "anova": "one-way ANOVA", "correlation": "correlation test",
        "proportion": "two-proportion z-test", "chi2": "chi-square test",
    }
    test_name = test_names.get(req.test, req.test)
    a_str = f"{req.alpha}" if req.alpha else "0.05"

    if req.solve_for == "n":
        n = int(np.ceil(result))
        total = n * 2 if req.test in ("t_two", "proportion") else n
        ratio_note = f" (ratio {req.ratio}:1)" if hasattr(req, "ratio") and req.ratio and req.ratio != 1 else ""
        return (
            f"You need {n} participants per group{ratio_note} (total N = {total}) "
            f"for a {test_name} to detect an effect size of {req.effect_size} "
            f"with {int((req.power or 0.8) * 100)}% power at alpha = {a_str}."
        )
    elif req.solve_for == "power":
        pwr = round(result * 100, 1)
        return (
            f"With n = {req.n} per group and effect size = {req.effect_size}, "
            f"your {test_name} has {pwr}% power to detect a real effect at alpha = {a_str}. "
            f"{'This exceeds the 80% minimum standard.' if result >= 0.8 else 'This is below the 80% minimum — consider increasing your sample size.'}"
        )
    elif req.solve_for == "effect_size":
        return (
            f"With n = {req.n} per group at {int((req.power or 0.8) * 100)}% power (alpha = {a_str}), "
            f"your {test_name} can detect a minimum effect size of {result:.3f}. "
            f"Effects smaller than this will likely be missed."
        )
    return ""
