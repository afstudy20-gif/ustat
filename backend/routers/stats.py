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


def _get_df(session_id: str, *, allow_missing: bool = False) -> pd.DataFrame | None:
    df = store.get_filtered(session_id)
    if df is None:
        if allow_missing:
            return None
        raise HTTPException(status_code=404, detail="Session not found")
    return df


# ── Missing Data Summary ─────────────────────────────────────────────────────

@router.get("/{session_id}/missing")
def get_missing(session_id: str, columns: str = Query("")):
    """
    Return per-column missing counts and total rows affected for the given
    comma-separated list of column names.
    """
    df = _get_df(session_id, allow_missing=True)
    if df is None:
        return {"columns": [], "total_rows": 0}
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
    # Multiple-comparison correction for Dunn's post-hoc test. Accepts
    # "holm" (default — uniformly more powerful than Bonferroni),
    # "bonferroni", "fdr" (Benjamini-Hochberg), or "none". The clinical
    # convention in most journals is Bonferroni; Holm matches its FWER
    # control while being strictly less conservative.
    posthoc_correction: Optional[str] = "holm"


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

    # Post-hoc: Dunn's test (only if significant and > 2 groups). Correction
    # method selectable — Bonferroni is the most conservative and most
    # widely-reported choice for tertile / quartile comparisons.
    pc = (req.posthoc_correction or "holm").lower()
    if pc not in {"holm", "bonferroni", "fdr", "none"}:
        raise HTTPException(status_code=422,
            detail=f"posthoc_correction must be holm | bonferroni | fdr | none, got '{req.posthoc_correction}'")
    posthoc = dunn_test(grp_dict, correction=pc) if sig and len(grp_dict) > 2 else []

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
        "posthoc_method": f"Dunn's test ({pc.title() if pc != 'fdr' else 'FDR'} correction)" if posthoc else None,
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


# ── Jonckheere-Terpstra trend test (ordered K-sample) ────────────────────────
# Non-parametric trend test for a continuous outcome across ≥3 ordered groups
# (e.g. tertiles or quartiles of a biomarker → continuous downstream measure).
# Statistic: J = Σ_{i<j} U(group_i, group_j) where U is the Mann-Whitney U
# (count of pairs (x_i, x_j) with x_j > x_i; ties count as 0.5). Under H₀ of
# no trend (groups are exchangeable):
#   E(J)   = (N² − Σ n_k²) / 4
#   Var(J) = (N²(2N + 3) − Σ n_k²(2n_k + 3)) / 72
# (Hollander & Wolfe 1973 §6.2). z = (J − E(J)) / √Var(J) is N(0,1) under H₀.
# Standard reference for the trend-in-medians clinical workflow that mirrors
# Cochran-Armitage on the categorical side.

class JonckheereRequest(BaseModel):
    session_id: str
    column: str           # continuous outcome
    group_column: str     # ordered (ordinal) exposure / group
    scores: Optional[List[float]] = None  # optional explicit group ordering — falls back to natural sort
    alpha: float = 0.05


@router.post("/jonckheere_terpstra")
def jonckheere_terpstra(req: JonckheereRequest):
    df = _get_df(req.session_id)
    for c in (req.column, req.group_column):
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")
    sub = df[[req.column, req.group_column]].dropna()
    if len(sub) < 5:
        raise HTTPException(422, "Need at least 5 non-null rows.")

    # Determine ordered group sequence. Custom scores → user controls order;
    # otherwise sort numerically when possible, else lexicographically.
    levels = sorted(sub[req.group_column].unique(), key=lambda x: (
        (0, float(x)) if isinstance(x, (int, float, np.integer, np.floating))
        or (isinstance(x, str) and x.replace(".", "", 1).replace("-", "", 1).isdigit())
        else (1, str(x))
    ))
    if req.scores is not None:
        if len(req.scores) != len(levels):
            raise HTTPException(422,
                f"Custom scores must match the number of levels ({len(levels)}); got {len(req.scores)}.")
        # Sort levels by user-supplied score so the resulting J is computed
        # in the user's intended order.
        levels = [lev for _, lev in sorted(zip(req.scores, levels), key=lambda t: t[0])]
    K = len(levels)
    if K < 3:
        raise HTTPException(422,
            f"Jonckheere-Terpstra requires ≥ 3 ordered groups; got {K}. "
            "For 2 groups use Mann-Whitney; for unordered groups use Kruskal-Wallis.")

    groups: list[np.ndarray] = []
    for lev in levels:
        vals = sub.loc[sub[req.group_column] == lev, req.column].astype(float).values
        if len(vals) == 0:
            raise HTTPException(422, f"Group '{lev}' has zero observations.")
        groups.append(vals)
    n_k = np.array([len(g) for g in groups], dtype=float)
    N = float(n_k.sum())

    # Compute J = Σ_{i<j} U(group_i, group_j). U counts pairs with x_j > x_i,
    # adding 0.5 per tie — the standard mid-rank convention.
    J = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            xi = groups[i][:, None]   # shape (n_i, 1)
            xj = groups[j][None, :]   # shape (1, n_j)
            J += float(np.sum(xj > xi) + 0.5 * np.sum(xj == xi))

    sum_n2 = float(np.sum(n_k ** 2))
    sum_n2_2n_p3 = float(np.sum(n_k ** 2 * (2 * n_k + 3)))
    E_J = (N ** 2 - sum_n2) / 4.0
    Var_J = (N ** 2 * (2 * N + 3) - sum_n2_2n_p3) / 72.0
    if Var_J <= 0:
        raise HTTPException(422, "Jonckheere-Terpstra variance is zero — group sizes degenerate.")
    z = (J - E_J) / np.sqrt(Var_J)
    p_two = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z)))
    sig = bool(p_two < req.alpha)
    p_str = "<0.001" if p_two < 0.001 else f"{p_two:.4f}"
    direction = "increasing" if z > 0 else "decreasing" if z < 0 else "flat"

    # Per-level medians for the UI table.
    level_rows = []
    for lev, g in zip(levels, groups):
        level_rows.append({
            "level": str(lev),
            "n": int(len(g)),
            "median": round(float(np.median(g)), 4),
            "q1": round(float(np.percentile(g, 25)), 4),
            "q3": round(float(np.percentile(g, 75)), 4),
            "mean": round(float(np.mean(g)), 4),
        })

    return _sanitize({
        "test": "Jonckheere-Terpstra trend test",
        "J": round(J, 4),
        "E_J": round(E_J, 4),
        "Var_J": round(Var_J, 6),
        "z": round(z, 4),
        "statistic": round(z, 4),
        "p": p_two,
        "significant": sig,
        "effect_sizes": [],
        "assumptions": [
            "Ordered (ordinal) exposure with ≥3 levels",
            "Continuous (or at least ordinal) outcome",
            "Independence between observations",
        ],
        "summary": {
            "n": int(N),
            "n_levels": K,
            "direction": direction,
            "levels": level_rows,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} monotone trend in "
            f"{req.column} across {K} ordered levels of {req.group_column} "
            f"(J = {J:.2f}, Z = {z:.3f}, p = {p_str}; direction: {direction})."
        ),
        "result_text": (
            f"The Jonckheere-Terpstra non-parametric trend test was used to "
            f"assess whether {req.column} changed monotonically across {K} "
            f"ordered levels of {req.group_column} (n = {int(N)}). The trend "
            f"was {'statistically significant' if sig else 'not statistically significant'} "
            f"(J = {J:.2f}, standardised Z = {z:.3f}, two-sided p = {p_str}), "
            f"with a {direction} trend in medians."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["J", round(J, 4)],
            ["E(J)", round(E_J, 4)],
            ["Var(J)", round(Var_J, 6)],
            ["Z", round(z, 4)],
            ["p", round(p_two, 6)],
            ["Levels", K],
            ["Total n", int(N)],
            ["Direction", direction],
        ],
        "r_code": (
            "# DescTools::JonckheereTerpstraTest(value ~ ordinal_group, "
            f"alternative='two.sided')   # column={req.column}, "
            f"ordinal={req.group_column}"
        ),
    })


# ── ROC Analysis ──────────────────────────────────────────────────────────────

class ROCRequest(BaseModel):
    session_id: str
    score_column: str
    outcome_column: str
    manual_cutoff: Optional[float] = None
    imputation: Optional[str] = "listwise"
    direction: Optional[str] = "auto"
    # NEW: Sex-specific / stratified ROC
    stratify_by: Optional[str] = None          # e.g. "SEX", "gender"
    stratify_values: Optional[List[str]] = None  # e.g. ["Male", "Female"] or [0, 1]


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

    # === Sex-specific / Stratified ROC (Phase 2 improvement) ===
    if req.stratify_by:
        if req.stratify_by not in df.columns:
            raise HTTPException(400, f"Stratification column '{req.stratify_by}' not found.")

        strata_results = {}
        strata_values = req.stratify_values or df[req.stratify_by].dropna().unique().tolist()

        for val in strata_values:
            mask = df[req.stratify_by] == val
            if mask.sum() < 20:
                continue

            s_scores = scores_arr[mask.values]
            s_y = y_arr[mask.values]

            try:
                fpr_s, tpr_s, th_s = roc_curve(s_y, s_scores)
                auc_s = float(roc_auc_score(s_y, s_scores))

                # Optimal cutoff (Youden)
                j_scores = tpr_s + (1 - fpr_s) - 1
                best_idx = int(np.argmax(j_scores))
                best_cut = float(th_s[best_idx])

                strata_results[str(val)] = {
                    "n": int(mask.sum()),
                    "auc": round(auc_s, 4),
                    "optimal_cutoff": round(best_cut, 4),
                    "sensitivity_at_opt": round(float(tpr_s[best_idx]), 4),
                    "specificity_at_opt": round(float(1 - fpr_s[best_idx]), 4),
                }
            except Exception:
                continue

        # If stratification requested, return early with strata results
        if strata_results:
            return {
                "test": "ROC Analysis (Stratified)",
                "stratified_by": req.stratify_by,
                "strata": strata_results,
                "note": "Separate ROC analysis performed within each stratum."
            }

    try:
        fpr, tpr, thresholds = roc_curve(y_arr, scores_arr)
        auc = float(roc_auc_score(y_arr, scores_arr))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ROC computation failed: {exc}")

    # Resolve score direction. "auto" flips the score sign when the naive AUC
    # is below 0.5 so the curve, AUC, optimal cutoff, and DeLong inference
    # all describe the biomarker in its true (protective) sense — i.e.
    # albumin, eGFR, haemoglobin etc. don't get pinned at AUC ≈ 0.31.
    direction_req = (req.direction or "auto").lower()
    if direction_req not in {"auto", "higher", "lower"}:
        raise HTTPException(status_code=422,
            detail=f"direction must be 'auto' | 'higher' | 'lower', got '{req.direction}'")
    flipped = False
    if direction_req == "lower" or (direction_req == "auto" and auc < 0.5):
        scores_arr = -scores_arr
        fpr, tpr, thresholds = roc_curve(y_arr, scores_arr)
        auc = float(roc_auc_score(y_arr, scores_arr))
        flipped = True
    # The reported direction (what the *returned* AUC describes).
    direction_used = "lower" if flipped else "higher"

    # DeLong non-parametric SE → 95% CI for AUC + z-test against H₀: AUC = 0.5
    # (i.e. the score has no discriminative ability). Reuses the same
    # placement-value machinery as /roc_compare so the single-curve and
    # paired-comparison endpoints stay on the same variance estimator.
    try:
        V_pos, V_neg = _delong_placement_values(y_arr.astype(int), scores_arr.astype(float))
        n_pos_d, n_neg_d = len(V_pos), len(V_neg)
        var_auc = float(
            np.var(V_pos, ddof=1) / max(n_pos_d, 1)
            + np.var(V_neg, ddof=1) / max(n_neg_d, 1)
        )
        var_auc = max(var_auc, 1e-12)
        se_auc = float(np.sqrt(var_auc))
        z95 = 1.95996  # scipy_stats.norm.ppf(0.975)
        ci_low = max(0.0, auc - z95 * se_auc)
        ci_high = min(1.0, auc + z95 * se_auc)
        # Two-sided z-test, H₀: AUC = 0.5.
        z_auc = (auc - 0.5) / se_auc
        p_auc = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(z_auc))))
    except Exception:
        # Defensive: fall back to None so the frontend renders "—" rather
        # than crashing if DeLong fails on a pathological input.
        se_auc = None
        ci_low = None
        ci_high = None
        z_auc = None
        p_auc = None

    # When flipped, the user supplies / sees thresholds in the original score
    # scale; internally we work on -score. Convert in and out of the user
    # scale so report values stay readable (e.g. "albumin ≤ 3.2 → death"
    # rather than "−albumin ≥ −3.2 → death").
    def _to_user(t: float) -> float:
        return -t if flipped else t

    def _from_user(t: float) -> float:
        return -t if flipped else t

    # Youden's J optimal cutoff
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    optimal = _roc_metrics_at_cutoff(scores_arr, y_arr, best_thresh)
    if flipped:
        optimal["cutoff"] = round(_to_user(best_thresh), 6)

    # Manual cutoff (if provided) — user supplies the value in original units
    manual = None
    if req.manual_cutoff is not None:
        thr_internal = _from_user(float(req.manual_cutoff))
        manual = _roc_metrics_at_cutoff(scores_arr, y_arr, thr_internal)
        if flipped:
            manual["cutoff"] = round(float(req.manual_cutoff), 6)

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
            # Always report thresholds in the *user* (original) score scale
            # so an interactive threshold table renders the values the user
            # recognises, regardless of internal sign flip.
            "threshold": round(_to_user(thr), 6),
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
        # DeLong inference for the AUC point estimate.
        "auc_se": round(se_auc, 6) if se_auc is not None else None,
        "ci_lower": round(ci_low, 4) if ci_low is not None else None,
        "ci_upper": round(ci_high, 4) if ci_high is not None else None,
        "auc_z": round(z_auc, 4) if z_auc is not None else None,
        "auc_p": round(p_auc, 6) if p_auc is not None else None,
        "auc_test": "H0: AUC = 0.5 (DeLong two-sided z-test)",
        # Score direction used for the reported AUC + curve. When the request
        # asked for "auto" and the naive AUC was < 0.5, the score sign was
        # flipped before computing everything below so the curve, optimal
        # cutoff, and DeLong inference all describe the biomarker in its
        # true (protective) direction — i.e. "low albumin → death".
        "direction_requested": direction_req,
        "direction_used": direction_used,
        "direction_flipped": flipped,
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
            f"The area under the curve was {auc:.2f}"
            + (f" (95% CI {ci_low:.2f}–{ci_high:.2f}, p = "
               f"{'<0.001' if (p_auc is not None and p_auc < 0.001) else f'{p_auc:.3f}' if p_auc is not None else 'n/a'})"
               if ci_low is not None and ci_high is not None else "")
            + ", indicating "
            f"{'excellent' if auc >= 0.9 else 'good' if auc >= 0.8 else 'fair' if auc >= 0.7 else 'poor'} discrimination "
            + ("(lower values predict the event — score sign auto-flipped from the request default). "
               if flipped else "(higher values predict the event). ")
            + f"At the optimal cutoff ({optimal['cutoff']:.2f}, Youden's J), sensitivity was {optimal['sensitivity']*100:.1f}% "
            f"and specificity was {optimal['specificity']*100:.1f}%."
        ),
    })


# ── ROC Comparison (DeLong Test) ──────────────────────────────────────────────

class ROCCompareRequest(BaseModel):
    session_id: str
    score_column_1: str
    score_column_2: str
    outcome_column: str
    # Per-score direction: same semantics as ROCRequest.direction. Default
    # "auto" flips a score's sign when its naive AUC < 0.5 so DeLong is
    # always run on protective-direction-corrected curves — otherwise the
    # ΔAUC vs an inverted biomarker (e.g. albumin) is meaningless.
    direction_1: Optional[str] = "auto"
    direction_2: Optional[str] = "auto"


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

    # Per-score direction: same auto-flip logic as the single-curve endpoint.
    # Critical for DeLong because comparing a protective biomarker (albumin,
    # AUC ≈ 0.31) against a risk biomarker (LAR, AUC ≈ 0.73) without flipping
    # the protective side reports a ΔAUC of 0.42 instead of the true ~0.04.
    def _resolve_direction(scores: np.ndarray, y: np.ndarray, req_dir: str):
        d = (req_dir or "auto").lower()
        if d not in {"auto", "higher", "lower"}:
            raise HTTPException(status_code=422,
                detail=f"direction must be 'auto' | 'higher' | 'lower', got '{req_dir}'")
        naive_auc = float(roc_auc_score(y, scores))
        flipped = (d == "lower") or (d == "auto" and naive_auc < 0.5)
        return (-scores if flipped else scores), flipped, d

    s1_arr, flipped_1, dir_req_1 = _resolve_direction(s1_arr, y_arr, req.direction_1 or "auto")
    s2_arr, flipped_2, dir_req_2 = _resolve_direction(s2_arr, y_arr, req.direction_2 or "auto")

    result = _delong_compare(y_arr, s1_arr, s2_arr)
    result["direction_1_requested"] = dir_req_1
    result["direction_2_requested"] = dir_req_2
    result["direction_1_used"] = "lower" if flipped_1 else "higher"
    result["direction_2_used"] = "lower" if flipped_2 else "higher"
    result["direction_1_flipped"] = bool(flipped_1)
    result["direction_2_flipped"] = bool(flipped_2)
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


# ── ROC Multi-Curve DeLong (K-way pairwise comparison) ───────────────────────


class ROCMultiCompareRequest(BaseModel):
    session_id: str
    score_columns: List[str]
    outcome_column: str
    # One direction per score column ('auto' | 'higher' | 'lower'). When the
    # list is shorter than score_columns, the remainder default to 'auto'.
    directions: Optional[List[str]] = None
    # Multiple-comparison adjustment over the K(K−1)/2 pairwise p-values.
    # Default Holm — strong-FWER, more powerful than Bonferroni.
    p_adjust: Optional[str] = "holm"


@router.post("/roc_multi_compare")
def roc_multi_compare(req: ROCMultiCompareRequest):
    """K-curve DeLong pairwise AUC comparison.

    Computes the per-column AUC with DeLong 95% CI plus every pairwise
    ΔAUC / z / p across the K = len(score_columns) curves on the SAME
    paired sample (rows with NaN in any score or the outcome are dropped
    once, before any computation, so every pair is tested on identical
    rows — required for the DeLong covariance to be valid).
    """
    from sklearn.metrics import roc_auc_score

    if len(req.score_columns) < 2:
        raise HTTPException(status_code=422,
            detail="Need at least 2 score columns to compare.")
    if len(req.score_columns) != len(set(req.score_columns)):
        raise HTTPException(status_code=422,
            detail="Duplicate entries in score_columns.")

    df_full = _get_df(req.session_id)
    for c in req.score_columns + [req.outcome_column]:
        if c not in df_full.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")

    # Single complete-case subset across every column — paired-sample
    # assumption for the DeLong covariance.
    df = df_full.dropna(subset=list(req.score_columns) + [req.outcome_column]).copy()
    if len(df) < 10:
        raise HTTPException(status_code=400,
            detail=f"Not enough complete rows after dropping NaN (need ≥ 10, got {len(df)}).")
    y_arr = df[req.outcome_column].astype(float).astype(int).values
    unique = set(np.unique(y_arr).tolist())
    if unique - {0, 1} or unique == {0} or unique == {1}:
        raise HTTPException(status_code=422,
            detail=f"Outcome must be binary 0/1 with both classes present (got {sorted(unique)}).")

    # Resolve per-score direction (auto-flip when naive AUC < 0.5) — same
    # convention as /roc and /roc_compare.
    dirs_in = list(req.directions or [])
    while len(dirs_in) < len(req.score_columns):
        dirs_in.append("auto")
    K = len(req.score_columns)
    scores: List[np.ndarray] = []
    scores_meta: List[dict] = []
    for col, d_in in zip(req.score_columns, dirs_in):
        d = (d_in or "auto").lower()
        if d not in {"auto", "higher", "lower"}:
            raise HTTPException(status_code=422,
                detail=f"direction for '{col}' must be 'auto'|'higher'|'lower', got '{d_in}'.")
        raw = df[col].astype(float).values
        naive_auc = float(roc_auc_score(y_arr, raw))
        flipped = (d == "lower") or (d == "auto" and naive_auc < 0.5)
        scores.append(-raw if flipped else raw)
        scores_meta.append({
            "name": col,
            "direction_requested": d,
            "direction_used": "lower" if flipped else "higher",
            "direction_flipped": bool(flipped),
        })

    # Pre-compute placement values per score (Hanley & McNeil / DeLong).
    place: List[tuple] = [_delong_placement_values(y_arr, s) for s in scores]
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())

    # Per-curve AUC + DeLong SE / 95% CI.
    z95 = 1.95996
    per_score: List[dict] = []
    aucs = np.zeros(K, dtype=float)
    ses  = np.zeros(K, dtype=float)
    for i, (V_pos, V_neg) in enumerate(place):
        auc = float(np.mean(V_pos))
        var = np.var(V_pos, ddof=1) / n_pos + np.var(V_neg, ddof=1) / n_neg
        se  = float(np.sqrt(max(var, 1e-12)))
        aucs[i] = auc
        ses[i]  = se
        ci_lo = max(0.0, auc - z95 * se)
        ci_hi = min(1.0, auc + z95 * se)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_arr, scores[i])
        curve_step = max(1, len(fpr) // 300)
        curve_idx = list(range(0, len(fpr), curve_step))
        if (len(fpr) - 1) not in curve_idx:
            curve_idx.append(len(fpr) - 1)
        per_score.append({
            **scores_meta[i],
            "auc": round(auc, 4),
            "se": round(se, 6),
            "ci_low":  round(ci_lo, 4),
            "ci_high": round(ci_hi, 4),
            "curve": [{"fpr": round(float(fpr[k]), 6), "tpr": round(float(tpr[k]), 6)} for k in curve_idx],
        })

    # Pairwise DeLong stats — share the precomputed placement values.
    pairs: List[dict] = []
    raw_ps: List[float] = []
    for i in range(K):
        Vpi, Vni = place[i]
        for j in range(i + 1, K):
            Vpj, Vnj = place[j]
            cov = (
                np.cov(Vpi, Vpj, ddof=1)[0, 1] / n_pos
                + np.cov(Vni, Vnj, ddof=1)[0, 1] / n_neg
            )
            var_diff = max(ses[i] ** 2 + ses[j] ** 2 - 2 * float(cov), 1e-12)
            se_diff  = float(np.sqrt(var_diff))
            diff = float(aucs[i] - aucs[j])
            z = diff / se_diff if se_diff > 0 else 0.0
            p = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
            ci_lo = float(diff - z95 * se_diff)
            ci_hi = float(diff + z95 * se_diff)
            pairs.append({
                "a": req.score_columns[i],
                "b": req.score_columns[j],
                "auc_a": round(float(aucs[i]), 4),
                "auc_b": round(float(aucs[j]), 4),
                "delta_auc": round(diff, 4),
                "se_diff":   round(se_diff, 6),
                "ci_low":    round(ci_lo, 4),
                "ci_high":   round(ci_hi, 4),
                "z": round(float(z), 4),
                "p_raw": round(p, 6),
            })
            raw_ps.append(p)

    # Multiple-comparison adjustment over the K(K-1)/2 raw p-values.
    method = (req.p_adjust or "holm").lower()
    m = len(raw_ps)
    if m == 0 or method == "none":
        p_adj_list = list(raw_ps)
    elif method == "bonferroni":
        p_adj_list = [min(1.0, p * m) for p in raw_ps]
    elif method == "holm":
        # Holm 1979 step-down.
        order = sorted(range(m), key=lambda k: raw_ps[k])
        p_adj_arr = [0.0] * m
        running = 0.0
        for rank, idx in enumerate(order):
            adj = (m - rank) * raw_ps[idx]
            running = max(running, adj)
            p_adj_arr[idx] = min(1.0, running)
        p_adj_list = p_adj_arr
    else:
        raise HTTPException(status_code=422,
            detail=f"p_adjust must be 'holm'|'bonferroni'|'none', got '{req.p_adjust}'.")

    for pair, p_adj in zip(pairs, p_adj_list):
        pair["p_adj"]      = round(float(p_adj), 6)
        pair["significant"] = bool(p_adj < 0.05)

    return _sanitize({
        "test": "ROC Multi-Curve DeLong",
        "n": int(len(y_arr)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "outcome": req.outcome_column,
        "scores": per_score,
        "pairs": pairs,
        "n_pairs": m,
        "p_adjust": method,
        "method_note": (
            "Per-column AUC reported with DeLong (1988) 95% confidence interval. "
            "Pairwise ΔAUC inference uses the same DeLong covariance machinery "
            "(placement values + Mann-Whitney U variance), so every pair is tested "
            "on the same paired sample (NaN-complete-case across every score and the "
            f"outcome). Multiple-comparison adjustment: {method}."
        ),
    })


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
    df = _get_df(session_id, allow_missing=True)
    if df is None:
        return {}
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


def _fisher_freeman_halton_mc(observed: np.ndarray, n_resamples: int = 5000, seed: int = 42) -> float:
    """Monte Carlo p-value for an r×c contingency table, conditional on
    the observed margins (Fisher-Freeman-Halton). Uses the chi-square
    statistic as the discrepancy measure; group labels are permuted
    `n_resamples` times. Returns the adjusted p-value
    (count_extreme + 1) / (n_resamples + 1) per Davison & Hinkley 1997.
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.sum() <= 0:
        return float("nan")
    n_rows, n_cols = obs.shape

    # Expand to long-form (item-level) arrays so we can permute group labels.
    cats_list: list[int] = []
    grps_list: list[int] = []
    for i in range(n_rows):
        for j in range(n_cols):
            n_ij = int(obs[i, j])
            if n_ij > 0:
                cats_list.extend([i] * n_ij)
                grps_list.extend([j] * n_ij)
    cats = np.asarray(cats_list, dtype=np.int64)
    grps = np.asarray(grps_list, dtype=np.int64)

    def _chi(ct: np.ndarray) -> float:
        rs = ct.sum(axis=1, keepdims=True)
        cs = ct.sum(axis=0, keepdims=True)
        total = ct.sum()
        if total <= 0:
            return 0.0
        e = rs * cs / total
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(((ct - e) ** 2 / np.where(e > 0, e, 1)).sum())

    obs_chi = _chi(obs)
    rng = np.random.default_rng(seed)
    minlength = n_rows * n_cols
    count = 0
    for _ in range(n_resamples):
        perm = rng.permutation(grps)
        enc = cats * n_cols + perm
        ct = np.bincount(enc, minlength=minlength).reshape(n_rows, n_cols).astype(float)
        if _chi(ct) >= obs_chi - 1e-9:
            count += 1
    return (count + 1) / (n_resamples + 1)


def _categorical_p_with_rule(ct: np.ndarray) -> tuple[float, str]:
    """Pick the right p-value for a contingency table.

    Rule (matches AMA/CONSORT convention):
      • If all expected cell counts ≥ 5 → Pearson chi-square.
      • Else, for a 2×2 table → Fisher's exact test.
      • Else, for an r×c table → Fisher-Freeman-Halton (Monte Carlo).
    """
    obs = np.asarray(ct, dtype=float)
    chi2, p_chi, dof, expected = scipy_stats.chi2_contingency(obs)
    if (expected < 5).any():
        if obs.shape == (2, 2):
            _, p_fisher = scipy_stats.fisher_exact(obs)
            return float(p_fisher), "Fisher"
        return _fisher_freeman_halton_mc(obs), "Fisher-Freeman-Halton (MC)"
    return float(p_chi), "Chi-square"


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

            # SMD (Standardized Mean Difference).
            # For 2 groups we report the standard Cohen's d (mean difference /
            # pooled SD). For k>2 groups we follow Austin 2011 / Yang-Dalton
            # 2012 and report the MAXIMUM pairwise SMD — the most conservative
            # measure of between-group imbalance for multi-arm Table 1.
            smd_val: Optional[float] = None
            if groups is not None and len(group_arrs) >= 2:
                try:
                    def _smd_num_pair(g1, g2) -> Optional[float]:
                        if len(g1) == 0 or len(g2) == 0:
                            return None
                        ps = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
                        if not np.isfinite(ps) or ps <= 0:
                            return None
                        return float(abs(g1.mean() - g2.mean()) / ps)
                    from itertools import combinations as _comb
                    pair_smds = []
                    for i, j in _comb(range(len(group_arrs)), 2):
                        s = _smd_num_pair(group_arrs[i], group_arrs[j])
                        if s is not None:
                            pair_smds.append(s)
                    if pair_smds:
                        smd_val = round(max(pair_smds), 4)
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
                    p_chi_raw, test_name = _categorical_p_with_rule(ct.values)
                    p_val = _fmt_p(float(p_chi_raw))
                except Exception:
                    p_val = "N/A"

            # SMD for categorical. 2-group uses Cohen's-style proportion SMD
            # (binary) or Yang-Dalton 2012 multinomial SMD; k>2 groups report
            # the MAXIMUM pairwise value (Austin 2011 convention).
            cat_smd: Optional[float] = None
            if groups is not None and len(groups) >= 2:
                try:
                    def _smd_cat_pair(g1_s: pd.Series, g2_s: pd.Series) -> Optional[float]:
                        all_cats = sorted(set(g1_s.dropna()) | set(g2_s.dropna()))
                        if len(all_cats) < 2:
                            return None
                        if len(all_cats) == 2:
                            target = all_cats[0]
                            p1 = (g1_s == target).mean()
                            p2 = (g2_s == target).mean()
                            pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
                            if pooled <= 0:
                                return None
                            return float(abs(p1 - p2) / pooled)
                        # Yang-Dalton 2012 multinomial SMD
                        p1_vec = np.array([(g1_s == c).mean() for c in all_cats[:-1]])
                        p2_vec = np.array([(g2_s == c).mean() for c in all_cats[:-1]])
                        s1 = np.diag(p1_vec * (1 - p1_vec))
                        s2 = np.diag(p2_vec * (1 - p2_vec))
                        s_pool = (s1 + s2) / 2
                        diff = p1_vec - p2_vec
                        det = np.linalg.det(s_pool)
                        if det <= 1e-12:
                            return None
                        return float(np.sqrt(diff @ np.linalg.inv(s_pool) @ diff))
                    from itertools import combinations as _comb
                    g_series = [df[df[req.group_column] == g][var].astype(str) for g in groups]
                    pair_smds = []
                    for i, j in _comb(range(len(g_series)), 2):
                        s = _smd_cat_pair(g_series[i], g_series[j])
                        if s is not None and np.isfinite(s):
                            pair_smds.append(s)
                    if pair_smds:
                        cat_smd = round(max(pair_smds), 4)
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


# ── TOST equivalence / non-inferiority tests ───────────────────────────────────

class TOSTRequest(BaseModel):
    session_id: str
    column: str               # continuous outcome
    group_column: Optional[str] = None   # for ind two-sample; None ⇒ one-sample vs mu
    paired_column: Optional[str] = None  # for paired version (col1, col2)
    low: float                # lower equivalence bound
    high: float               # upper equivalence bound
    mu: Optional[float] = 0.0  # reference for one-sample
    test_type: str = "independent"  # "independent" | "paired" | "one_sample"


@router.post("/tost")
def tost(req: TOSTRequest):
    """Two One-Sided Tests (TOST) for equivalence / non-inferiority.

    H0: difference is OUTSIDE the [low, high] equivalence margin.
    H1: difference lies WITHIN the equivalence margin.
    p < α ⇒ equivalence demonstrated.

    Three modes:
      - independent: ttost_ind on two groups defined by group_column.
      - paired: ttost_paired on two columns (column, paired_column).
      - one_sample: tests mean(column) - mu within [low, high].

    For non-inferiority pick a one-sided margin (e.g. low=-Inf, high=δ).
    """
    from statsmodels.stats.weightstats import ttost_ind, ttost_paired

    df = _get_df(req.session_id)
    if req.low >= req.high:
        raise HTTPException(status_code=422, detail="low must be < high")

    test_type = req.test_type
    n1 = n2 = 0
    mean1 = mean2 = std1 = std2 = None

    if test_type == "independent":
        if not req.group_column:
            raise HTTPException(status_code=422, detail="independent TOST requires group_column.")
        sub = df[[req.column, req.group_column]].dropna()
        groups = sub[req.group_column].unique()
        if len(groups) != 2:
            raise HTTPException(status_code=422, detail=f"group_column must have exactly 2 levels, found {len(groups)}.")
        a = sub.loc[sub[req.group_column] == groups[0], req.column].astype(float)
        b = sub.loc[sub[req.group_column] == groups[1], req.column].astype(float)
        n1, n2 = int(len(a)), int(len(b))
        if n1 < 2 or n2 < 2:
            raise HTTPException(status_code=400, detail="Each group needs ≥2 observations.")
        mean1, mean2 = float(a.mean()), float(b.mean())
        std1, std2 = float(a.std(ddof=1)), float(b.std(ddof=1))
        p_overall, (t_low, p_low, _df_low), (t_high, p_high, _df_high) = ttost_ind(a, b, low=req.low, upp=req.high, usevar="pooled")
        diff = mean1 - mean2
        group_labels = [str(groups[0]), str(groups[1])]
    elif test_type == "paired":
        if not req.paired_column:
            raise HTTPException(status_code=422, detail="paired TOST requires paired_column.")
        sub = df[[req.column, req.paired_column]].dropna()
        a = sub[req.column].astype(float)
        b = sub[req.paired_column].astype(float)
        n1 = n2 = int(len(a))
        if n1 < 2:
            raise HTTPException(status_code=400, detail="Need ≥2 paired observations.")
        mean1, mean2 = float(a.mean()), float(b.mean())
        std1, std2 = float(a.std(ddof=1)), float(b.std(ddof=1))
        p_overall, (t_low, p_low, _df_low), (t_high, p_high, _df_high) = ttost_paired(a, b, low=req.low, upp=req.high)
        diff = mean1 - mean2
        group_labels = [req.column, req.paired_column]
    elif test_type == "one_sample":
        from scipy.stats import t as _t
        col = df[req.column].dropna().astype(float)
        n1 = int(len(col))
        if n1 < 2:
            raise HTTPException(status_code=400, detail="Need ≥2 observations.")
        mean1 = float(col.mean())
        std1 = float(col.std(ddof=1))
        se = std1 / np.sqrt(n1)
        mu = float(req.mu or 0.0)
        # Lower one-sided: H0: mean - mu <= low ⇒ test (mean - mu - low) / SE > critical
        t_low = (mean1 - mu - req.low) / se if se > 0 else float("inf")
        p_low = float(_t.sf(t_low, df=n1 - 1))  # upper tail
        # Upper one-sided: H0: mean - mu >= high ⇒ test (mean - mu - high) / SE < -critical
        t_high = (mean1 - mu - req.high) / se if se > 0 else float("-inf")
        p_high = float(_t.cdf(t_high, df=n1 - 1))  # lower tail
        p_overall = max(p_low, p_high)
        diff = mean1 - mu
        group_labels = [req.column, f"μ₀ = {mu}"]
    else:
        raise HTTPException(status_code=422, detail=f"Unknown test_type '{test_type}'")

    equivalent = p_overall < 0.05
    interp = (
        f"Equivalence demonstrated (both one-sided p < 0.05) — observed difference is statistically "
        f"within the [{req.low}, {req.high}] margin."
        if equivalent else
        f"Equivalence NOT demonstrated (max of two one-sided p = {p_overall:.4f}) — cannot conclude "
        f"the difference lies within [{req.low}, {req.high}]."
    )
    return {
        "test": f"TOST ({test_type})",
        "test_type": test_type,
        "n1": n1, "n2": n2,
        "mean1": mean1, "mean2": mean2,
        "std1": std1, "std2": std2,
        "difference": float(diff),
        "low_bound": float(req.low),
        "high_bound": float(req.high),
        "t_low": float(t_low), "p_low": float(p_low),
        "t_high": float(t_high), "p_high": float(p_high),
        "p_overall": float(p_overall),
        "equivalent": bool(equivalent),
        "group_labels": group_labels,
        "interpretation": interp,
        "result_text": (
            f"Two One-Sided Tests for equivalence within [{req.low}, {req.high}]. "
            f"Lower bound test: t = {t_low:.3f}, p = {p_low:.4f}. "
            f"Upper bound test: t = {t_high:.3f}, p = {p_high:.4f}. "
            f"{interp}"
        ),
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
    # Logistic regression (req.test == "logistic")
    log_or: Optional[float] = None       # expected odds ratio
    p_event: Optional[float] = None      # baseline event probability
    r2_other: Optional[float] = 0.0      # R² of predictor against the rest (variance inflation)
    # Adjusted Cox / log-rank (req.test == "survival_cox")
    hr: Optional[float] = None           # expected hazard ratio
    event_rate: Optional[float] = None   # cumulative event probability
    p_exposed: Optional[float] = 0.5     # proportion exposed/treated


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

    # ── Logistic regression — Hsieh 1989 / 1998 formula ────────────────────────
    # n = (Z_{1-α/2} + Z_{1-β})² / (p (1-p) β² (1-R²))
    # where β = log(OR), p = baseline event probability, R² = predictor's
    # R² when regressed on the rest of the covariate matrix (variance
    # inflation due to adjustment; pass 0 for unadjusted).
    elif req.test == "logistic":
        from scipy.stats import norm as _norm

        def _required_n(log_or, p_event, power_target, alpha_target, r2_other, tails):
            z_a = _norm.ppf(1 - alpha_target / (2 if tails == 2 else 1))
            z_b = _norm.ppf(power_target)
            return float(((z_a + z_b) ** 2) / (p_event * (1 - p_event) * (log_or ** 2) * (1 - (r2_other or 0.0))))

        def _power_from_n(log_or, p_event, n_total, alpha_target, r2_other, tails):
            z_a = _norm.ppf(1 - alpha_target / (2 if tails == 2 else 1))
            se = float(np.sqrt(1.0 / (n_total * p_event * (1 - p_event) * (1 - (r2_other or 0.0)))))
            z = abs(log_or) / se if se > 0 else 0.0
            return float(_norm.cdf(z - z_a))

        if not req.log_or and req.effect_size is not None:
            # Convenience: accept effect_size as OR (front-end may pass it that way)
            log_or = float(np.log(req.effect_size))
        elif req.log_or is not None:
            # Accept either β (log OR) or OR > 0 in the same field — small ORs
            # under 0.05 are rare and confusable with logs, so treat positive
            # values > 0 as OR by convention and convert.
            log_or = float(req.log_or) if req.log_or <= 0 else float(np.log(req.log_or))
        else:
            raise HTTPException(400, "Logistic power needs 'log_or' (or 'effect_size' = OR).")
        if req.p_event is None or not (0 < req.p_event < 1):
            raise HTTPException(400, "Logistic power needs 'p_event' in (0, 1).")
        r2 = req.r2_other if req.r2_other is not None else 0.0

        def pw(n_): return _power_from_n(log_or, req.p_event, n_, a, r2, req.tails)
        if req.solve_for == "n":
            n_req = _ceil(_required_n(log_or, req.p_event, req.power or 0.8, a, r2, req.tails))
            result, label = n_req, f"n = {n_req}"
            curve = _curve(pw, max(n_req * 4, 200))
        elif req.solve_for == "power":
            result = float(pw(int(req.n)))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 200))
        else:
            # Solve for OR given n and power → invert numerically.
            from scipy.optimize import brentq
            try:
                f = lambda lo: pw(int(req.n)) - req.power if False else None
                or_solved = brentq(
                    lambda lo: _power_from_n(lo, req.p_event, int(req.n), a, r2, req.tails) - (req.power or 0.8),
                    1e-3, 5.0,
                )
                result = float(np.exp(or_solved))
                label  = f"Minimum detectable OR = {result:.3f}"
                # Curve: how power scales with n for this OR
                ll = float(or_solved)
                curve = _curve(lambda n_: _power_from_n(ll, req.p_event, n_, a, r2, req.tails), max(int(req.n)*4, 200))
            except Exception:
                result = None
                label = "Could not solve for OR — try different power / n combination."

    # ── Adjusted Cox / log-rank — Schoenfeld 1981 + Hsieh 1998 ────────────────
    # Required number of EVENTS d = (Z_{1-α/2} + Z_{1-β})² / (p_exp (1-p_exp) log(HR)²)
    # Then required N = d / event_rate. The (1 - R²) adjustment when present
    # inflates n for collinear covariate sets (Hsieh 1998).
    elif req.test == "survival_cox":
        from scipy.stats import norm as _norm

        if req.hr is None or req.hr <= 0:
            raise HTTPException(400, "Cox power needs 'hr' > 0.")
        if req.event_rate is None or not (0 < req.event_rate < 1):
            raise HTTPException(400, "Cox power needs 'event_rate' in (0, 1).")
        p_exp = req.p_exposed if req.p_exposed is not None else 0.5
        if not (0 < p_exp < 1):
            raise HTTPException(400, "'p_exposed' must be in (0, 1).")
        r2 = req.r2_other or 0.0
        log_hr = float(np.log(req.hr))

        def _events_required(power_target):
            z_a = _norm.ppf(1 - a / (2 if req.tails == 2 else 1))
            z_b = _norm.ppf(power_target)
            return ((z_a + z_b) ** 2) / (p_exp * (1 - p_exp) * (log_hr ** 2))

        def _n_required(power_target):
            d = _events_required(power_target)
            return d / (req.event_rate * (1 - r2))

        def _power_from_n(n_total):
            z_a = _norm.ppf(1 - a / (2 if req.tails == 2 else 1))
            d = n_total * req.event_rate * (1 - r2)
            if d <= 0:
                return 0.0
            se = float(np.sqrt(1.0 / (d * p_exp * (1 - p_exp))))
            z = abs(log_hr) / se if se > 0 else 0.0
            return float(_norm.cdf(z - z_a))

        def pw(n_): return _power_from_n(n_)
        if req.solve_for == "n":
            n_req = _ceil(_n_required(req.power or 0.8))
            d_req = _ceil(_events_required(req.power or 0.8))
            result, label = n_req, f"n = {n_req} (events = {d_req})"
            curve = _curve(pw, max(n_req * 4, 200))
        elif req.solve_for == "power":
            result = float(pw(int(req.n)))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 200))
        else:
            # Solve for HR
            from scipy.optimize import brentq
            try:
                hr_solved = brentq(
                    lambda lh: _power_from_n_with_hr(lh, int(req.n), p_exp, req.event_rate, r2, a, req.tails) - (req.power or 0.8) if False else 0,
                    0.01, 10.0,
                )
            except Exception:
                pass
            # Closed-form: events d = n × event_rate × (1 − R²);
            # log(HR) = (Z_α + Z_β) / √(d × p(1−p))
            d_total = int(req.n) * req.event_rate * (1 - r2)
            if d_total > 0:
                z_a = _norm.ppf(1 - a / (2 if req.tails == 2 else 1))
                z_b = _norm.ppf(req.power or 0.8)
                lh = (z_a + z_b) / np.sqrt(d_total * p_exp * (1 - p_exp))
                result = float(np.exp(lh))
                label  = f"Minimum detectable HR = {result:.3f}"
                lh_val = float(lh)
                curve = _curve(lambda n_: _power_from_n(n_), max(int(req.n) * 4, 200))
            else:
                result, label = None, "Insufficient events to solve for HR."

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


# ── Weighted descriptive statistics (survey / sampling weights) ───────────────
#
# Weights-only design-based estimation (no strata / cluster yet). Covers the
# common case where each row carries a sampling or post-stratification weight.
# Weighted mean / SD / SE / 95% CI and Kish's effective sample size come from
# statsmodels DescrStatsW; weighted quantiles are interpolated on the sorted
# cumulative-weight grid; a binary column additionally gets a Horvitz-Thompson
# weighted proportion. With a 2-level group column the endpoint also returns a
# weighted mean difference + DescrStatsW two-sample t-test.


class WeightedDescriptiveRequest(BaseModel):
    session_id: str
    value_cols: List[str]
    weight_col: str
    group_col: Optional[str] = None
    imputation: Optional[str] = "listwise"


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return float(np.interp(q, cw, v))


@router.post("/weighted_descriptive")
def weighted_descriptive(req: WeightedDescriptiveRequest):
    from statsmodels.stats.weightstats import DescrStatsW

    df_full = _get_df(req.session_id)
    for c in [req.weight_col, *req.value_cols] + ([req.group_col] if req.group_col else []):
        if c not in df_full.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.value_cols:
        raise HTTPException(status_code=422, detail="Select at least one value column.")

    cols = [req.weight_col, *req.value_cols] + ([req.group_col] if req.group_col else [])
    df = apply_imputation(df_full[cols], cols, req.imputation or "listwise").reset_index(drop=True)
    w_all = pd.to_numeric(df[req.weight_col], errors="coerce")
    if (w_all <= 0).any() or w_all.isna().all():
        # Drop non-positive / missing weights row-wise rather than failing hard.
        pass

    results: List[dict] = []
    for col in req.value_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & w_all.notna() & (w_all > 0)
        xv = x[mask].values.astype(float)
        wv = w_all[mask].values.astype(float)
        if len(xv) < 3:
            results.append({"column": col, "error": "fewer than 3 valid weighted observations"})
            continue
        d = DescrStatsW(xv, weights=wv, ddof=1)
        lo, hi = d.tconfint_mean(alpha=0.05)
        kish = float((wv.sum() ** 2) / np.sum(wv ** 2))   # effective sample size
        uniq = np.unique(xv)
        row = {
            "column": col,
            "n": int(len(xv)),
            "sum_weights": round(float(wv.sum()), 4),
            "ess_kish": round(kish, 2),
            "w_mean": round(float(d.mean), 6),
            "w_sd": round(float(d.std), 6),
            "w_se": round(float(d.std_mean), 6),
            "ci_low": round(float(lo), 6),
            "ci_high": round(float(hi), 6),
            "w_median": round(_weighted_quantile(xv, wv, 0.5), 6),
            "w_q1": round(_weighted_quantile(xv, wv, 0.25), 6),
            "w_q3": round(_weighted_quantile(xv, wv, 0.75), 6),
        }
        # Binary column → Horvitz-Thompson weighted proportion of the larger code.
        if set(uniq.tolist()) <= {0.0, 1.0} and len(uniq) == 2:
            p = float(np.sum(wv * xv) / np.sum(wv))
            se_p = float(np.sqrt(p * (1 - p) / kish))
            row["w_proportion"] = round(p, 6)
            row["w_proportion_ci_low"] = round(max(0.0, p - 1.959963984540054 * se_p), 6)
            row["w_proportion_ci_high"] = round(min(1.0, p + 1.959963984540054 * se_p), 6)
        results.append(row)

    # Optional weighted two-group comparison (first value column).
    comparison = None
    if req.group_col:
        groups = [g for g in df[req.group_col].dropna().unique()]
        if len(groups) == 2:
            col = req.value_cols[0]
            x = pd.to_numeric(df[col], errors="coerce")
            parts = []
            for g in groups:
                m = (df[req.group_col] == g) & x.notna() & w_all.notna() & (w_all > 0)
                parts.append((str(g), x[m].values.astype(float), w_all[m].values.astype(float)))
            if all(len(p[1]) >= 3 for p in parts):
                from statsmodels.stats.weightstats import CompareMeans, DescrStatsW as _D
                d1 = _D(parts[0][1], weights=parts[0][2], ddof=1)
                d2 = _D(parts[1][1], weights=parts[1][2], ddof=1)
                cm = CompareMeans(d1, d2)
                tstat, pval, dfree = cm.ttest_ind(usevar="unequal")
                diff = float(d1.mean - d2.mean)
                lo, hi = cm.tconfint_diff(alpha=0.05, usevar="unequal")
                comparison = {
                    "variable": col,
                    "group_a": parts[0][0], "group_b": parts[1][0],
                    "w_mean_a": round(float(d1.mean), 4), "w_mean_b": round(float(d2.mean), 4),
                    "diff": round(diff, 4),
                    "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                    "t": round(float(tstat), 4), "df": round(float(dfree), 2),
                    "p": round(float(pval), 6),
                }

    n_total = int((w_all.notna() & (w_all > 0)).sum())
    result_text = (
        f"Weighted descriptive statistics on n = {n_total} rows using '{req.weight_col}' as the "
        f"sampling weight (design-based, weights only). "
        + (f"Weighted {comparison['variable']}: {comparison['group_a']} = {comparison['w_mean_a']} vs "
           f"{comparison['group_b']} = {comparison['w_mean_b']}, Δ = {comparison['diff']} "
           f"(95% CI {comparison['ci_low']}–{comparison['ci_high']}), weighted t-test p = "
           f"{'<0.001' if comparison['p'] < 0.001 else round(comparison['p'], 3)}."
           if comparison else "")
    )

    export_rows = [["Variable", "n", "ESS", "Weighted mean", "Weighted SD", "95% CI low", "95% CI high", "Weighted median"]]
    for r in results:
        if "error" in r:
            continue
        export_rows.append([r["column"], r["n"], r["ess_kish"], r["w_mean"], r["w_sd"], r["ci_low"], r["ci_high"], r["w_median"]])

    try:
        store.log_action(req.session_id, "weighted_descriptive", {
            "weight_col": req.weight_col, "n_value_cols": len(req.value_cols),
            "group_col": req.group_col,
        })
    except Exception:
        pass

    return _sanitize({
        "test": "Weighted descriptive statistics",
        "weight_col": req.weight_col,
        "n": n_total,
        "results": results,
        "comparison": comparison,
        "assumptions": [
            {"name": "Weights-only design", "met": True,
             "detail": "Design-based estimation with sampling weights. Strata / cluster (full complex survey) not modelled — SEs assume independent weighted observations."},
            {"name": "Effective sample size", "met": True,
             "detail": "Kish's ESS = (Σw)² / Σw² reported per variable; large weight variation shrinks ESS and widens CIs."},
        ],
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": (
            "library(survey)\n"
            f"des <- svydesign(ids = ~1, weights = ~{req.weight_col}, data = data)\n"
            f"svymean(~{' + '.join(req.value_cols)}, des)\n"
            + (f"svyttest({req.value_cols[0]} ~ {req.group_col}, des)\n" if req.group_col else "")
        ),
    })


# ── Non-inferiority / superiority / equivalence (margin testing) ──────────────
#
# Regulatory-style margin testing for two-arm trials (ITT or per-protocol — the
# user supplies the relevant analysis dataset). A one-sided α corresponds to a
# two-sided (1 − 2α) confidence interval, the standard non-inferiority
# convention (α = 0.05 → 90% CI). Supports a binary outcome (risk ratio / risk
# difference / odds ratio) or a continuous outcome (mean difference).
#
# Non-inferiority is concluded from the appropriate CI bound vs the prespecified
# margin:
#   • bound = "upper": non-inferior if the upper CI bound < margin
#       (event is harmful, margin > 1 for RR/OR or > 0 for RD/mean-diff)
#   • bound = "lower": non-inferior if the lower CI bound > margin
#       (preserve benefit, margin < 1 or < 0)


class NonInferiorityRequest(BaseModel):
    session_id: str
    outcome_col: str
    group_col: str
    test_group: Optional[str] = None          # the new / experimental arm
    ref_group: Optional[str] = None           # the active control / reference
    outcome_type: str = "binary"              # binary | continuous
    effect: str = "RR"                        # binary: RR | RD | OR
    margin: float = 1.20
    bound: str = "upper"                      # "upper" | "lower"
    alpha: float = 0.05                       # ONE-SIDED alpha (→ (1−2α) CI)
    imputation: Optional[str] = "listwise"


@router.post("/noninferiority")
def noninferiority(req: NonInferiorityRequest):
    df = _get_df(req.session_id)
    for c in [req.outcome_col, req.group_col]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not (0.0 < req.alpha < 0.5):
        raise HTTPException(status_code=422, detail="One-sided alpha must be in (0, 0.5).")
    if req.bound not in ("upper", "lower"):
        raise HTTPException(status_code=422, detail="bound must be 'upper' or 'lower'.")

    cols = [req.outcome_col, req.group_col]
    work = apply_imputation(df[cols], cols, req.imputation or "listwise").dropna()
    groups = work[req.group_col].astype(str)
    levels = sorted(groups.unique().tolist())
    if len(levels) != 2:
        raise HTTPException(status_code=422,
            detail=f"Group column must have exactly 2 levels; found {len(levels)}: {levels}")
    test_g = str(req.test_group) if req.test_group is not None else levels[1]
    ref_g = str(req.ref_group) if req.ref_group is not None else levels[0]
    if test_g not in levels or ref_g not in levels or test_g == ref_g:
        raise HTTPException(status_code=422, detail=f"test_group / ref_group must be the 2 distinct levels {levels}.")

    z_one = float(scipy_stats.norm.ppf(1 - req.alpha))
    ci_level = round((1 - 2 * req.alpha) * 100, 1)
    log_margin = None
    is_log = False

    if req.outcome_type == "binary":
        y = pd.to_numeric(work[req.outcome_col], errors="coerce")
        if set(pd.unique(y.dropna())) - {0.0, 1.0}:
            raise HTTPException(status_code=422, detail="Binary outcome must be coded 0/1.")
        t = y[groups == test_g]; r = y[groups == ref_g]
        n1, n2 = int(t.notna().sum()), int(r.notna().sum())
        x1, x2 = int(t.sum()), int(r.sum())
        p1, p2 = x1 / n1, x2 / n2
        eff = req.effect.upper()
        if eff == "RD":
            est = p1 - p2
            se = float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
            lo, hi = est - z_one * se, est + z_one * se
            est_disp, lo_disp, hi_disp = est, lo, hi
        elif eff == "OR":
            is_log = True
            a, b, c, d = x1, n1 - x1, x2, n2 - x2
            if min(a, b, c, d) == 0:
                a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            le = np.log((a * d) / (b * c))
            se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
            lo, hi = le - z_one * se, le + z_one * se
            est_disp, lo_disp, hi_disp = float(np.exp(le)), float(np.exp(lo)), float(np.exp(hi))
            log_margin = float(np.log(req.margin))
        else:  # RR
            is_log = True
            if x1 == 0 or x2 == 0:
                x1a, x2a = x1 + 0.5, x2 + 0.5
                n1a, n2a = n1 + 0.5, n2 + 0.5
            else:
                x1a, x2a, n1a, n2a = x1, x2, n1, n2
            r1, r2 = x1a / n1a, x2a / n2a
            le = np.log(r1 / r2)
            se = float(np.sqrt((1 - r1) / x1a + (1 - r2) / x2a))
            lo, hi = le - z_one * se, le + z_one * se
            est_disp, lo_disp, hi_disp = float(np.exp(le)), float(np.exp(lo)), float(np.exp(hi))
            log_margin = float(np.log(req.margin))
        detail = {"n_test": n1, "n_ref": n2, "events_test": x1, "events_ref": x2,
                  "p_test": round(p1, 4), "p_ref": round(p2, 4)}
        scale_point = le if is_log else est
        scale_se = se
    elif req.outcome_type == "continuous":
        from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
        y = pd.to_numeric(work[req.outcome_col], errors="coerce")
        t = y[groups == test_g].dropna().values.astype(float)
        r = y[groups == ref_g].dropna().values.astype(float)
        if len(t) < 2 or len(r) < 2:
            raise HTTPException(status_code=422, detail="Each arm needs ≥ 2 observations.")
        cm = CompareMeans(DescrStatsW(t), DescrStatsW(r))
        est = float(t.mean() - r.mean())
        lo, hi = cm.tconfint_diff(alpha=2 * req.alpha, usevar="unequal")
        se = (hi - lo) / (2 * z_one)
        est_disp, lo_disp, hi_disp = est, float(lo), float(hi)
        detail = {"n_test": len(t), "n_ref": len(r),
                  "mean_test": round(float(t.mean()), 4), "mean_ref": round(float(r.mean()), 4)}
        scale_point, scale_se = est, se
        eff = "Mean difference"
    else:
        raise HTTPException(status_code=422, detail="outcome_type must be 'binary' or 'continuous'.")

    # NI verdict + one-sided p-value on the analysis scale (log for RR/OR).
    m_scale = (log_margin if is_log else req.margin)
    if req.bound == "upper":
        non_inferior = hi_disp < req.margin
        z = (m_scale - scale_point) / scale_se if scale_se > 0 else 0.0   # H0: effect ≥ margin
        p_ni = float(scipy_stats.norm.cdf(z))
        rule = f"upper {ci_level}% CI bound ({round(hi_disp, 4)}) < margin ({req.margin})"
    else:
        non_inferior = lo_disp > req.margin
        z = (scale_point - m_scale) / scale_se if scale_se > 0 else 0.0   # H0: effect ≤ margin
        p_ni = float(scipy_stats.norm.cdf(z))
        rule = f"lower {ci_level}% CI bound ({round(lo_disp, 4)}) > margin ({req.margin})"

    interp = (
        f"Non-inferiority test ({eff}, {test_g} vs {ref_g}). "
        f"{eff} = {round(est_disp, 4)} ({ci_level}% CI {round(lo_disp, 4)}–{round(hi_disp, 4)}); "
        f"prespecified margin = {req.margin}. One-sided α = {req.alpha} "
        f"(equivalently a two-sided {ci_level}% CI). "
        + (f"Non-inferiority DEMONSTRATED — {rule}, p = {'<0.001' if p_ni < 0.001 else round(p_ni, 4)}."
           if non_inferior else
           f"Non-inferiority NOT demonstrated — {rule} fails (p = {round(p_ni, 4)}).")
    )

    try:
        store.log_action(req.session_id, "noninferiority", {
            "outcome_col": req.outcome_col, "group_col": req.group_col,
            "effect": eff, "margin": req.margin, "bound": req.bound, "alpha": req.alpha,
        })
    except Exception:
        pass

    return _sanitize({
        "test": "Non-inferiority (margin) test",
        "outcome_type": req.outcome_type,
        "effect": eff,
        "test_group": test_g, "ref_group": ref_g,
        "estimate": round(est_disp, 5),
        "ci_level": ci_level,
        "ci_low": round(lo_disp, 5),
        "ci_high": round(hi_disp, 5),
        "margin": req.margin,
        "bound": req.bound,
        "alpha_one_sided": req.alpha,
        "non_inferior": bool(non_inferior),
        "p_noninferiority": round(p_ni, 6),
        **detail,
        "assumptions": [
            {"name": "Analysis population", "met": True,
             "detail": "Provide the ITT (or per-protocol) dataset — the test runs on the loaded rows as supplied."},
            {"name": "One-sided ↔ CI correspondence", "met": True,
             "detail": f"One-sided α = {req.alpha} corresponds to a two-sided {ci_level}% CI (regulatory convention)."},
            {"name": "Large-sample normal approx.", "met": (detail.get('n_test', 99) >= 10 and detail.get('n_ref', 99) >= 10),
             "detail": "Wald / log-Wald intervals assume adequate per-arm counts."},
        ],
        "result_text": interp,
        "interpretation": interp,
        "export_rows": [
            ["Metric", "Value"],
            [f"{eff} ({test_g} vs {ref_g})", round(est_disp, 5)],
            [f"{ci_level}% CI", f"{round(lo_disp, 4)} – {round(hi_disp, 4)}"],
            ["Margin", req.margin],
            ["Bound tested", req.bound],
            ["One-sided alpha", req.alpha],
            ["Non-inferior", "Yes" if non_inferior else "No"],
            ["p (non-inferiority)", round(p_ni, 6)],
        ],
        "r_code": (
            "# Non-inferiority: one-sided alpha = "
            f"{req.alpha} ↔ two-sided {ci_level}% CI\n"
            + ("library(epitools); riskratio(table)  # RR + CI\n" if req.effect.upper() == 'RR' and req.outcome_type == 'binary' else "")
            + f"# Non-inferior if {req.bound} {ci_level}% CI bound vs margin {req.margin}."
        ),
    })
