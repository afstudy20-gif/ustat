"""Categorical tests: binomial, proportion z-tests, McNemar, Cochran Q, Mantel-Haenszel."""
import numpy as np
import pandas as pd
from scipy import stats as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services import store
from services.stat_utils import cohens_h, adjust_pvalues, group_summary, kendalls_w

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BINOMIAL TEST
# ═══════════════════════════════════════════════════════════════════════════════

class BinomialRequest(BaseModel):
    session_id: str
    column: str
    expected_proportion: float = 0.5
    alpha: float = 0.05


@router.post("/binomial")
def binomial_test(req: BinomialRequest):
    df = _get_df(req.session_id)
    if req.column not in df.columns:
        raise HTTPException(400, f"Column '{req.column}' not found.")
    col = df[req.column].dropna()
    if len(col) < 1:
        raise HTTPException(400, "No non-null values in column.")

    n = len(col)
    # Count successes: if binary (0/1), count 1s; otherwise count most frequent value
    unique_vals = col.unique()
    if set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
        k = int((col.astype(float) == 1).sum())
        success_label = "1"
    else:
        most_frequent = col.value_counts().idxmax()
        k = int((col == most_frequent).sum())
        success_label = str(most_frequent)

    result = sp.binomtest(k, n, req.expected_proportion)
    p = float(result.pvalue)
    sig = bool(p < req.alpha)
    observed_prop = k / n
    ps = _p_str(p)

    ci = result.proportion_ci(confidence_level=0.95)
    ci_low = round(float(ci.low), 4)
    ci_high = round(float(ci.high), 4)

    es = cohens_h(observed_prop, req.expected_proportion)

    return {
        "test": "Binomial test",
        "k": k, "n": n, "observed_proportion": round(observed_prop, 4),
        "expected_proportion": req.expected_proportion,
        "p": p,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "ci_proportion": {"low": ci_low, "high": ci_high},
        "summary": {
            "success_value": success_label,
            "k": k, "n": n,
            "observed_proportion": round(observed_prop, 4),
            "expected_proportion": req.expected_proportion,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} difference from expected proportion "
            f"(observed = {observed_prop:.3f}, expected = {req.expected_proportion:.3f}, p = {ps})"
        ),
        "result_text": (
            f"A binomial test compared the observed proportion of '{success_label}' in {req.column} "
            f"({k}/{n} = {observed_prop:.3f}) against the expected proportion of {req.expected_proportion:.3f}. "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(p = {ps}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            f"Cohen's h = {es['value']:.3f} [{es['magnitude']}]."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["k (successes)", k],
            ["n (total)", n],
            ["Observed proportion", round(observed_prop, 4)],
            ["Expected proportion", req.expected_proportion],
            ["p", round(p, 6)],
            ["95% CI lower", ci_low],
            ["95% CI upper", ci_high],
            ["Cohen's h", es["value"]],
        ],
        "r_code": f"binom.test({k}, {n}, p = {req.expected_proportion})",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ONE-SAMPLE PROPORTION Z-TEST
# ═══════════════════════════════════════════════════════════════════════════════

class OneProportionRequest(BaseModel):
    session_id: str
    column: str
    null_proportion: float = 0.5
    alpha: float = 0.05


@router.post("/one_proportion")
def one_proportion_ztest(req: OneProportionRequest):
    from statsmodels.stats.proportion import proportions_ztest

    df = _get_df(req.session_id)
    if req.column not in df.columns:
        raise HTTPException(400, f"Column '{req.column}' not found.")
    col = df[req.column].dropna()
    if len(col) < 1:
        raise HTTPException(400, "No non-null values in column.")

    n = len(col)
    unique_vals = col.unique()
    if set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
        k = int((col.astype(float) == 1).sum())
        success_label = "1"
    else:
        most_frequent = col.value_counts().idxmax()
        k = int((col == most_frequent).sum())
        success_label = str(most_frequent)

    z_stat, p = proportions_ztest(k, n, value=req.null_proportion)
    z_stat = float(z_stat)
    p = float(p)
    sig = bool(p < req.alpha)
    observed_prop = k / n
    ps = _p_str(p)

    es = cohens_h(observed_prop, req.null_proportion)

    # Wald CI
    se = np.sqrt(observed_prop * (1 - observed_prop) / n) if n > 0 else 0
    ci_low = round(max(0, observed_prop - 1.96 * se), 4)
    ci_high = round(min(1, observed_prop + 1.96 * se), 4)

    return {
        "test": "One-sample proportion z-test",
        "z": round(z_stat, 4), "p": p,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "summary": {
            "success_value": success_label,
            "k": k, "n": n,
            "observed_proportion": round(observed_prop, 4),
            "null_proportion": req.null_proportion,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} difference from null proportion "
            f"(z = {z_stat:.3f}, p = {ps}, h = {es['value']:.3f} [{es['magnitude']}])"
        ),
        "result_text": (
            f"A one-sample proportion z-test compared the observed proportion of '{success_label}' in {req.column} "
            f"({k}/{n} = {observed_prop:.3f}) against the null proportion of {req.null_proportion:.3f}. "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(z = {z_stat:.3f}, p = {ps}). "
            f"95% CI for the proportion: [{ci_low:.3f}, {ci_high:.3f}]. "
            f"Cohen's h = {es['value']:.3f} [{es['magnitude']}]."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["z", round(z_stat, 4)],
            ["p", round(p, 6)],
            ["k (successes)", k],
            ["n (total)", n],
            ["Observed proportion", round(observed_prop, 4)],
            ["Null proportion", req.null_proportion],
            ["95% CI lower", ci_low],
            ["95% CI upper", ci_high],
            ["Cohen's h", es["value"]],
        ],
        "r_code": f"prop.test({k}, {n}, p = {req.null_proportion})",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TWO-SAMPLE PROPORTION Z-TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TwoProportionsRequest(BaseModel):
    session_id: str
    column: str
    group_column: str
    alpha: float = 0.05


@router.post("/two_proportions")
def two_proportions_ztest(req: TwoProportionsRequest):
    from statsmodels.stats.proportion import proportions_ztest

    df = _get_df(req.session_id)
    for c in [req.column, req.group_column]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.column, req.group_column]].dropna()
    groups = sub[req.group_column].unique()
    if len(groups) != 2:
        raise HTTPException(400, f"Group column must have exactly 2 groups, found {len(groups)}.")

    g1_data = sub[sub[req.group_column] == groups[0]][req.column]
    g2_data = sub[sub[req.group_column] == groups[1]][req.column]

    # Count successes (value == 1 or most frequent overall)
    all_vals = sub[req.column]
    unique_vals = all_vals.unique()
    if set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
        k1 = int((g1_data.astype(float) == 1).sum())
        k2 = int((g2_data.astype(float) == 1).sum())
        success_label = "1"
    else:
        most_frequent = all_vals.value_counts().idxmax()
        k1 = int((g1_data == most_frequent).sum())
        k2 = int((g2_data == most_frequent).sum())
        success_label = str(most_frequent)

    n1, n2 = len(g1_data), len(g2_data)
    p1, p2 = k1 / n1 if n1 > 0 else 0, k2 / n2 if n2 > 0 else 0

    z_stat, p = proportions_ztest([k1, k2], [n1, n2])
    z_stat = float(z_stat)
    p = float(p)
    sig = bool(p < req.alpha)
    ps = _p_str(p)

    es = cohens_h(p1, p2)

    return {
        "test": "Two-sample proportion z-test",
        "z": round(z_stat, 4), "p": p,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "summary": {
            str(groups[0]): {"n": n1, "k": k1, "proportion": round(p1, 4)},
            str(groups[1]): {"n": n2, "k": k2, "proportion": round(p2, 4)},
            "success_value": success_label,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} difference between proportions "
            f"({groups[0]}: {p1:.3f} vs {groups[1]}: {p2:.3f}, z = {z_stat:.3f}, p = {ps}, "
            f"h = {es['value']:.3f} [{es['magnitude']}])"
        ),
        "result_text": (
            f"A two-sample proportion z-test compared '{success_label}' rates between "
            f"{groups[0]} ({k1}/{n1} = {p1:.3f}) and {groups[1]} ({k2}/{n2} = {p2:.3f}). "
            f"The difference was {'statistically significant' if sig else 'not statistically significant'} "
            f"(z = {z_stat:.3f}, p = {ps}). "
            f"Cohen's h = {es['value']:.3f} [{es['magnitude']}]."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["z", round(z_stat, 4)],
            ["p", round(p, 6)],
            [f"{groups[0]}: k/n", f"{k1}/{n1}"],
            [f"{groups[0]}: proportion", round(p1, 4)],
            [f"{groups[1]}: k/n", f"{k2}/{n2}"],
            [f"{groups[1]}: proportion", round(p2, 4)],
            ["Cohen's h", es["value"]],
        ],
        "r_code": f"prop.test(c({k1}, {k2}), c({n1}, {n2}))",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. McNEMAR'S TEST
# ═══════════════════════════════════════════════════════════════════════════════

class McnemarRequest(BaseModel):
    session_id: str
    col1: str
    col2: str
    alpha: float = 0.05


@router.post("/mcnemar")
def mcnemar_test(req: McnemarRequest):
    from statsmodels.stats.contingency_tables import mcnemar

    df = _get_df(req.session_id)
    for c in [req.col1, req.col2]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.col1, req.col2]].dropna()
    if len(sub) < 5:
        raise HTTPException(400, "Need at least 5 paired observations.")

    # Build 2x2 contingency table from paired observations
    ct = pd.crosstab(sub[req.col1], sub[req.col2])
    if ct.shape != (2, 2):
        raise HTTPException(400, f"Expected 2x2 table from binary variables, got {ct.shape}.")

    table = ct.values
    a, b = table[0]
    c, d = table[1]

    # Use exact test when discordant pairs < 25
    exact = bool((b + c) < 25)
    result = mcnemar(table, exact=exact)
    stat = float(result.statistic)
    p = float(result.pvalue)
    sig = bool(p < req.alpha)
    ps = _p_str(p)

    # Effect size: odds ratio of discordant pairs
    or_val = float(b / c) if c > 0 else float('inf')
    or_str = f"{or_val:.3f}" if np.isfinite(or_val) else "Inf"
    es = {"name": "odds_ratio_discordant", "value": round(or_val, 4) if np.isfinite(or_val) else None,
          "ci_low": None, "ci_high": None, "magnitude": ""}
    if np.isfinite(or_val):
        from services.stat_utils import _es_magnitude
        es["magnitude"] = _es_magnitude("odds_ratio", or_val)

    return {
        "test": "McNemar's test",
        "statistic": round(stat, 4), "p": p,
        "exact": exact,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "contingency_table": {"a": int(a), "b": int(b), "c": int(c), "d": int(d)},
        "summary": {
            "discordant_b": int(b), "discordant_c": int(c),
            "concordant_a": int(a), "concordant_d": int(d),
            "n": int(len(sub)),
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} change between {req.col1} and {req.col2} "
            f"({'exact' if exact else 'chi-squared'} statistic = {stat:.3f}, p = {ps}, OR = {or_str})"
        ),
        "result_text": (
            f"McNemar's test ({'exact' if exact else 'asymptotic'}) assessed the change between "
            f"{req.col1} and {req.col2} (n = {len(sub)} pairs). "
            f"Discordant pairs: b = {b}, c = {c}. "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(statistic = {stat:.3f}, p = {ps}). "
            f"Odds ratio of discordant pairs = {or_str}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Test statistic", round(stat, 4)],
            ["p", round(p, 6)],
            ["Exact test", exact],
            ["a (both +)", int(a)],
            ["b (+ to -)", int(b)],
            ["c (- to +)", int(c)],
            ["d (both -)", int(d)],
            ["OR (b/c)", or_str],
        ],
        "r_code": f"mcnemar.test(table)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COCHRAN'S Q TEST
# ═══════════════════════════════════════════════════════════════════════════════

class CochranQRequest(BaseModel):
    session_id: str
    columns: List[str]
    alpha: float = 0.05


@router.post("/cochran_q")
def cochran_q_test(req: CochranQRequest):
    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_fn

    if len(req.columns) < 3:
        raise HTTPException(400, "Cochran's Q test requires at least 3 binary columns.")

    df = _get_df(req.session_id)
    for c in req.columns:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[req.columns].dropna()
    if len(sub) < 5:
        raise HTTPException(400, "Need at least 5 complete subjects.")

    # Convert to binary matrix
    mat = sub.values.astype(float)
    n, k = mat.shape

    # Manual Cochran's Q calculation
    # Gj = column sums, Li = row sums, T = grand total
    Gj = mat.sum(axis=0)  # column sums (k values)
    Li = mat.sum(axis=1)  # row sums (n values)
    T = mat.sum()

    numerator = (k - 1) * (k * np.sum(Gj ** 2) - T ** 2)
    denominator = k * T - np.sum(Li ** 2)

    if denominator == 0:
        raise HTTPException(400, "Cannot compute Q: all rows are identical.")

    Q = numerator / denominator
    df_q = k - 1
    p = float(1 - sp.chi2.cdf(Q, df_q))
    sig = bool(p < req.alpha)
    ps = _p_str(p)

    # Effect size: Kendall's W
    es = kendalls_w(float(Q), n, k)

    # Post-hoc: pairwise McNemar with Holm correction (if significant)
    posthoc = []
    if sig:
        raw_ps = []
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        for i, j in pairs:
            ct = pd.crosstab(sub[req.columns[i]], sub[req.columns[j]])
            # Ensure 2x2
            if ct.shape == (2, 2):
                table = ct.values
                exact = bool((table[0, 1] + table[1, 0]) < 25)
                try:
                    res = mcnemar_fn(table, exact=exact)
                    pv = float(res.pvalue)
                except Exception:
                    pv = 1.0
            else:
                pv = 1.0
            posthoc.append({
                "group1": req.columns[i], "group2": req.columns[j],
                "p": round(pv, 6),
            })
            raw_ps.append(pv)

        adj = adjust_pvalues(raw_ps, "holm")
        for idx, ph in enumerate(posthoc):
            ph["p_adj"] = round(adj[idx], 6)
            ph["significant"] = adj[idx] < req.alpha
            ph["correction"] = "holm"

    col_props = {c: round(float(Gj[i]) / n, 4) for i, c in enumerate(req.columns)}

    return {
        "test": "Cochran's Q test",
        "Q": round(float(Q), 4), "df": df_q, "p": p,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "posthoc": posthoc,
        "posthoc_method": "Pairwise McNemar (Holm correction)" if posthoc else None,
        "summary": {
            "n_subjects": n, "k_conditions": k,
            "proportions": col_props,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} difference across {k} conditions "
            f"(Q({df_q}) = {Q:.2f}, p = {ps}, Kendall's W = {es['value']:.3f} [{es['magnitude']}])"
        ),
        "result_text": (
            f"Cochran's Q test assessed differences across {k} related binary conditions "
            f"(n = {n} subjects). The test was {'statistically significant' if sig else 'not statistically significant'} "
            f"(Q({df_q}) = {Q:.2f}, p = {ps}). "
            f"Kendall's W = {es['value']:.3f} [{es['magnitude']}]."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Cochran's Q", round(float(Q), 4)],
            ["df", df_q],
            ["p", round(p, 6)],
            ["Kendall's W", es["value"]],
            ["n subjects", n],
            ["k conditions", k],
            *[[f"Proportion ({c})", col_props[c]] for c in req.columns],
        ],
        "r_code": (
            f"library(RVAideMemoire)\n"
            f"cochran.qtest(cbind({', '.join(req.columns)}) ~ 1, data = data)"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COCHRAN-MANTEL-HAENSZEL TEST
# ═══════════════════════════════════════════════════════════════════════════════

class MantelHaenszelRequest(BaseModel):
    session_id: str
    row_col: str
    col_col: str
    strata_col: str
    alpha: float = 0.05


@router.post("/mantel_haenszel")
def mantel_haenszel_test(req: MantelHaenszelRequest):
    from statsmodels.stats.contingency_tables import StratifiedTable

    df = _get_df(req.session_id)
    for c in [req.row_col, req.col_col, req.strata_col]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.row_col, req.col_col, req.strata_col]].dropna()
    if len(sub) < 10:
        raise HTTPException(400, "Need at least 10 observations.")

    row_levels = sorted(sub[req.row_col].unique())
    col_levels = sorted(sub[req.col_col].unique())
    if len(row_levels) != 2 or len(col_levels) != 2:
        raise HTTPException(400, "Row and column variables must each have exactly 2 levels for CMH test.")

    strata = sorted(sub[req.strata_col].unique())
    if len(strata) < 2:
        raise HTTPException(400, "Need at least 2 strata.")

    # Build list of 2x2 tables per stratum
    tables = []
    stratum_info = []
    for s in strata:
        s_data = sub[sub[req.strata_col] == s]
        ct = pd.crosstab(s_data[req.row_col], s_data[req.col_col])
        # Ensure 2x2 with correct ordering
        ct = ct.reindex(index=row_levels, columns=col_levels, fill_value=0)
        tables.append(ct.values.astype(float))
        stratum_info.append({"stratum": str(s), "n": int(len(s_data)),
                             "table": ct.values.tolist()})

    try:
        st = StratifiedTable(tables)
        result = st.test_null_odds()
        stat = float(result.statistic)
        p = float(result.pvalue)
    except Exception as exc:
        raise HTTPException(400, f"CMH test failed: {exc}")

    sig = bool(p < req.alpha)
    ps = _p_str(p)

    # Common odds ratio
    try:
        common_or = float(st.oddsratio_pooled)
        or_str = f"{common_or:.3f}"
    except Exception:
        common_or = None
        or_str = "N/A"

    return {
        "test": "Cochran-Mantel-Haenszel test",
        "statistic": round(stat, 4), "p": p,
        "significant": sig,
        "effect_sizes": [{"name": "common_odds_ratio", "value": round(common_or, 4) if common_or else None,
                          "ci_low": None, "ci_high": None, "magnitude": ""}],
        "assumptions": [],
        "summary": {
            "n_strata": len(strata),
            "n_total": int(len(sub)),
            "strata": stratum_info,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} association between {req.row_col} and {req.col_col} "
            f"after stratifying by {req.strata_col} "
            f"(CMH statistic = {stat:.3f}, p = {ps}, common OR = {or_str})"
        ),
        "result_text": (
            f"A Cochran-Mantel-Haenszel test examined the association between {req.row_col} and {req.col_col} "
            f"across {len(strata)} strata of {req.strata_col} (n = {len(sub)}). "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(CMH statistic = {stat:.3f}, p = {ps}). "
            f"Common odds ratio = {or_str}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["CMH statistic", round(stat, 4)],
            ["p", round(p, 6)],
            ["Common OR", or_str],
            ["Number of strata", len(strata)],
            ["Total n", int(len(sub))],
        ],
        "r_code": f"mantelhaen.test(table_array)",
    }
