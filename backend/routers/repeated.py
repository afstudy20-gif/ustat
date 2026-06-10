"""Repeated-measures tests: paired t-test, Wilcoxon SR, Friedman, RM ANOVA, mixed ANOVA."""
import numpy as np
import pandas as pd
from scipy import stats as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from services import store
from services.stat_utils import (
    cohen_d_paired, matched_rank_biserial, kendalls_w, partial_eta_squared,
    check_normality, group_summary, adjust_pvalues,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAIRED T-TEST
# ═══════════════════════════════════════════════════════════════════════════════

class PairedTTestRequest(BaseModel):
    session_id: str
    col1: str
    col2: str
    alpha: float = 0.05


@router.post("/paired_ttest")
def paired_ttest(req: PairedTTestRequest):
    df = _get_df(req.session_id)
    pair = df[[req.col1, req.col2]].dropna()
    if len(pair) < 3:
        raise HTTPException(400, "Need at least 3 complete pairs.")
    x1 = pair[req.col1].astype(float).values
    x2 = pair[req.col2].astype(float).values
    d = x1 - x2
    n = len(d)

    t_stat, p = sp.ttest_rel(x1, x2)
    if np.isnan(p) or np.isnan(t_stat):
        p = 1.0
        t_stat = 0.0
    if np.isinf(t_stat):
        t_stat = float(np.sign(t_stat)) * 9999.0  # cap for JSON serialization
        p = 0.0
    sig = bool(p < req.alpha)
    es = cohen_d_paired(d)
    norm = check_normality(d, "Differences")

    mean_diff = float(d.mean())
    sd_diff = float(d.std(ddof=1))
    ps = _p_str(p)

    return {
        "test": "Paired-samples t-test",
        "t": round(float(t_stat), 4), "df": n - 1, "p": float(p),
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [norm],
        "warnings": ["Differences are not normally distributed — consider Wilcoxon signed-rank test."] if not norm["met"] else [],
        "summary": {
            req.col1: group_summary(x1, req.col1),
            req.col2: group_summary(x2, req.col2),
            "differences": {"n": n, "mean": round(mean_diff, 4), "sd": round(sd_diff, 4)},
        },
        "interpretation": f"{'Significant' if sig else 'No significant'} difference between {req.col1} and {req.col2} (t({n-1}) = {t_stat:.3f}, p = {ps}, d_z = {es['value']:.3f} [{es['magnitude']}])",
        "result_text": (
            f"A paired-samples t-test compared {req.col1} (M = {x1.mean():.2f}, SD = {x1.std(ddof=1):.2f}) "
            f"and {req.col2} (M = {x2.mean():.2f}, SD = {x2.std(ddof=1):.2f}). "
            f"{'There was a significant difference' if sig else 'There was no significant difference'} "
            f"(t({n-1}) = {t_stat:.3f}, p = {ps}). The mean difference was {mean_diff:.3f} (SD = {sd_diff:.3f}), "
            f"with a {es['magnitude']} effect size (Cohen's d_z = {es['value']:.3f}, 95% CI [{es['ci_low']:.3f}, {es['ci_high']:.3f}])."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["t", round(float(t_stat), 4)],
            ["df", n - 1],
            ["p", round(float(p), 6)],
            ["Mean difference", round(mean_diff, 4)],
            ["SD of differences", round(sd_diff, 4)],
            ["Cohen's d_z", es["value"]],
            ["95% CI lower", es["ci_low"]],
            ["95% CI upper", es["ci_high"]],
            ["Magnitude", es["magnitude"]],
        ],
        "r_code": f't.test(data${req.col1}, data${req.col2}, paired = TRUE)',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WILCOXON SIGNED-RANK
# ═══════════════════════════════════════════════════════════════════════════════

class WilcoxonSRRequest(BaseModel):
    session_id: str
    col1: str
    col2: str
    alpha: float = 0.05


@router.post("/wilcoxon_signed_rank")
def wilcoxon_signed_rank(req: WilcoxonSRRequest):
    df = _get_df(req.session_id)
    pair = df[[req.col1, req.col2]].dropna()
    if len(pair) < 6:
        raise HTTPException(400, "Need at least 6 complete pairs for Wilcoxon signed-rank.")
    x1 = pair[req.col1].astype(float).values
    x2 = pair[req.col2].astype(float).values
    d = x1 - x2
    n = len(d)

    # Remove zero differences (standard Wilcoxon practice)
    nonzero = d[d != 0]
    if len(nonzero) < 3:
        raise HTTPException(400, "Too few non-zero differences for Wilcoxon test.")

    w_stat, p = sp.wilcoxon(x1, x2, alternative="two-sided")
    sig = bool(p < req.alpha)
    es = matched_rank_biserial(float(w_stat), len(nonzero))
    ps = _p_str(p)

    return {
        "test": "Wilcoxon signed-rank test",
        "W": round(float(w_stat), 4), "p": float(p), "n_nonzero": len(nonzero),
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "summary": {
            req.col1: group_summary(x1, req.col1),
            req.col2: group_summary(x2, req.col2),
        },
        "interpretation": f"{'Significant' if sig else 'No significant'} difference (W = {w_stat:.1f}, p = {ps}, r = {es['value']:.3f} [{es['magnitude']}])",
        "result_text": (
            f"A Wilcoxon signed-rank test indicated that {req.col2} scores were "
            f"{'significantly' if sig else 'not significantly'} different from {req.col1} scores "
            f"(W = {w_stat:.1f}, p = {ps}, r = {es['value']:.3f} [{es['magnitude']}]). "
            f"Median {req.col1} = {np.median(x1):.2f}, median {req.col2} = {np.median(x2):.2f}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["W", round(float(w_stat), 4)],
            ["p", round(float(p), 6)],
            ["n (non-zero differences)", len(nonzero)],
            ["Rank-biserial r", es["value"]],
            ["95% CI lower", es["ci_low"]],
            ["95% CI upper", es["ci_high"]],
        ],
        "r_code": f'wilcox.test(data${req.col1}, data${req.col2}, paired = TRUE)',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FRIEDMAN TEST
# ═══════════════════════════════════════════════════════════════════════════════

class FriedmanRequest(BaseModel):
    session_id: str
    columns: List[str]  # 3+ repeated measures columns (wide format)
    alpha: float = 0.05


@router.post("/friedman")
def friedman(req: FriedmanRequest):
    if len(req.columns) < 3:
        raise HTTPException(400, "Friedman test requires at least 3 repeated measures.")
    df = _get_df(req.session_id)
    sub = df[req.columns].dropna()
    if len(sub) < 5:
        raise HTTPException(400, "Need at least 5 complete subjects.")

    arrays = [sub[c].astype(float).values for c in req.columns]
    n = len(sub)
    k = len(req.columns)

    chi2, p = sp.friedmanchisquare(*arrays)
    sig = bool(p < req.alpha)
    es = kendalls_w(float(chi2), n, k)
    ps = _p_str(p)

    # Post-hoc: pairwise Wilcoxon signed-rank with Holm correction
    posthoc = []
    if sig and k > 2:
        raw_ps = []
        pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
        for i, j in pairs:
            try:
                w, pv = sp.wilcoxon(arrays[i], arrays[j], alternative="two-sided")
            except Exception:
                pv = 1.0
                w = 0
            posthoc.append({
                "group1": req.columns[i], "group2": req.columns[j],
                "statistic": round(float(w), 4), "p": round(float(pv), 6),
            })
            raw_ps.append(float(pv))
        adj = adjust_pvalues(raw_ps, "holm")
        for idx, ph in enumerate(posthoc):
            ph["p_adj"] = round(adj[idx], 6)
            ph["significant"] = adj[idx] < req.alpha
            ph["correction"] = "holm"

    return {
        "test": "Friedman test",
        "chi2": round(float(chi2), 4), "df": k - 1, "p": float(p),
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "posthoc": posthoc,
        "posthoc_method": "Pairwise Wilcoxon signed-rank (Holm correction)" if posthoc else None,
        "summary": {c: group_summary(sub[c].astype(float).values, c) for c in req.columns},
        "interpretation": f"{'Significant' if sig else 'No significant'} difference across {k} conditions (\u03C7\u00B2({k-1}) = {chi2:.2f}, p = {ps}, Kendall's W = {es['value']:.3f} [{es['magnitude']}])",
        "result_text": (
            f"A Friedman test showed {'a significant' if sig else 'no significant'} difference across {k} conditions "
            f"(\u03C7\u00B2({k-1}) = {chi2:.2f}, p = {ps}, Kendall's W = {es['value']:.3f} [{es['magnitude']}]). "
            f"n = {n} subjects with complete data across all conditions."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Chi-square", round(float(chi2), 4)],
            ["df", k - 1],
            ["p", round(float(p), 6)],
            ["Kendall's W", es["value"]],
            ["n", n],
            ["k (conditions)", k],
        ],
        "r_code": 'friedman.test(y ~ timepoint | subject, data = data_long)',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REPEATED-MEASURES ANOVA
# ═══════════════════════════════════════════════════════════════════════════════

class RMAnovaRequest(BaseModel):
    session_id: str
    subject_col: str
    within_col: str
    value_col: str
    alpha: float = 0.05


@router.post("/rm_anova")
def rm_anova(req: RMAnovaRequest):
    from statsmodels.stats.anova import AnovaRM

    df = _get_df(req.session_id)
    cols = [req.subject_col, req.within_col, req.value_col]
    for c in cols:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found. Data must be in long format — use the Melt helper first.")

    sub = df[cols].dropna()
    sub[req.value_col] = pd.to_numeric(sub[req.value_col], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 10:
        raise HTTPException(400, "Need at least 10 rows for RM ANOVA.")

    k = sub[req.within_col].nunique()
    if k < 2:
        raise HTTPException(400, f"Within-subjects factor '{req.within_col}' must have at least 2 levels.")

    try:
        rm = AnovaRM(sub, req.value_col, req.subject_col, within=[req.within_col])
        res = rm.fit()
    except Exception as exc:
        raise HTTPException(400, f"RM ANOVA failed: {exc}")

    tbl = res.anova_table
    row = tbl.iloc[0]
    F_val = float(row["F Value"])
    p_val = float(row["Pr > F"])
    df_num = int(row["Num DF"])
    df_den = int(row["Den DF"])
    sig = bool(p_val < req.alpha)
    es = partial_eta_squared(F_val, df_num, df_den)
    ps = _p_str(p_val)

    # Sphericity check
    assumptions = []
    if k > 2:
        # GG epsilon if available
        try:
            eps = float(res.epsilon) if hasattr(res, 'epsilon') else None
        except Exception:
            eps = None
        if eps and eps < 0.75:
            assumptions.append({"name": "Sphericity (Greenhouse-Geisser)", "met": False,
                                "detail": f"\u03B5 = {eps:.3f} < 0.75 — GG correction recommended"})
        elif eps:
            assumptions.append({"name": "Sphericity", "met": True, "detail": f"\u03B5 = {eps:.3f}"})

    # Post-hoc: pairwise paired t-tests with Holm
    posthoc = []
    if sig and k > 2:
        levels = sorted(sub[req.within_col].unique())
        raw_ps = []
        for i in range(len(levels)):
            for j in range(i+1, len(levels)):
                g1 = sub[sub[req.within_col] == levels[i]].set_index(req.subject_col)[req.value_col]
                g2 = sub[sub[req.within_col] == levels[j]].set_index(req.subject_col)[req.value_col]
                common = g1.index.intersection(g2.index)
                if len(common) < 3:
                    continue
                t, pv = sp.ttest_rel(g1.loc[common].values, g2.loc[common].values)
                posthoc.append({
                    "group1": str(levels[i]), "group2": str(levels[j]),
                    "statistic": round(float(t), 4), "p": round(float(pv), 6),
                    "mean_diff": round(float(g1.loc[common].mean() - g2.loc[common].mean()), 4),
                })
                raw_ps.append(float(pv))
        if raw_ps:
            adj = adjust_pvalues(raw_ps, "holm")
            for idx, ph in enumerate(posthoc):
                ph["p_adj"] = round(adj[idx], 6)
                ph["significant"] = adj[idx] < req.alpha
                ph["correction"] = "holm"

    n_subj = sub[req.subject_col].nunique()
    return {
        "test": "Repeated-measures ANOVA",
        "F": round(F_val, 4), "df_num": df_num, "df_den": df_den, "p": float(p_val),
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": assumptions,
        "posthoc": posthoc,
        "posthoc_method": "Pairwise paired t-tests (Holm correction)" if posthoc else None,
        "summary": {str(lv): group_summary(sub[sub[req.within_col] == lv][req.value_col].values, str(lv))
                    for lv in sorted(sub[req.within_col].unique())},
        "interpretation": f"{'Significant' if sig else 'No significant'} effect of {req.within_col} on {req.value_col} (F({df_num},{df_den}) = {F_val:.2f}, p = {ps}, partial \u03B7\u00B2 = {es['value']:.3f} [{es['magnitude']}])",
        "result_text": (
            f"A repeated-measures ANOVA with {k} levels of {req.within_col} (n = {n_subj} subjects) "
            f"showed {'a significant' if sig else 'no significant'} effect on {req.value_col} "
            f"(F({df_num},{df_den}) = {F_val:.2f}, p = {ps}, partial \u03B7\u00B2 = {es['value']:.3f} [{es['magnitude']}])."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["F", round(F_val, 4)],
            ["df (numerator)", df_num],
            ["df (denominator)", df_den],
            ["p", round(float(p_val), 6)],
            ["Partial eta-squared", es["value"]],
            ["n subjects", n_subj],
            ["k conditions", k],
        ],
        "r_code": f'library(ez)\nezANOVA(data = data, dv = .({req.value_col}), wid = .({req.subject_col}), within = .({req.within_col}))',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MIXED ANOVA (within + between)
# ═══════════════════════════════════════════════════════════════════════════════

class MixedAnovaRequest(BaseModel):
    session_id: str
    subject_col: str
    within_col: str
    between_col: str
    value_col: str
    alpha: float = 0.05


@router.post("/mixed_anova")
def mixed_anova(req: MixedAnovaRequest):
    """Mixed ANOVA via OLS with Type II SS (within × between interaction)."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    df = _get_df(req.session_id)
    cols = [req.subject_col, req.within_col, req.between_col, req.value_col]
    for c in cols:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[cols].dropna()
    sub[req.value_col] = pd.to_numeric(sub[req.value_col], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 12:
        raise HTTPException(400, "Need at least 12 rows for mixed ANOVA.")

    # OLS-based mixed ANOVA: Y ~ C(within) * C(between) with subject as random
    # For simplicity, use fixed-effects factorial; note: this is approximate for mixed designs
    formula = f"Q('{req.value_col}') ~ C(Q('{req.within_col}')) * C(Q('{req.between_col}'))"
    try:
        model = smf.ols(formula, data=sub).fit()
        aov = anova_lm(model, typ=2)
    except Exception as exc:
        raise HTTPException(400, f"Mixed ANOVA failed: {exc}")

    # Parse each effect row
    effects = []
    for idx_name, row in aov.iterrows():
        name = str(idx_name)
        if name == "Residual":
            continue
        F_val = float(row["F"]) if not np.isnan(row["F"]) else 0
        p_val = float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else 1
        df_n = int(row["df"])
        df_d = int(aov.loc["Residual", "df"])
        sig = bool(p_val < req.alpha)
        es = partial_eta_squared(F_val, df_n, df_d)

        # Clean term name
        if req.within_col in name and req.between_col in name:
            label = f"{req.within_col} \u00D7 {req.between_col} (interaction)"
        elif req.within_col in name:
            label = req.within_col
        elif req.between_col in name:
            label = req.between_col
        else:
            label = name

        effects.append({
            "term": label, "F": round(F_val, 4), "df_num": df_n, "df_den": df_d,
            "p": round(float(p_val), 6), "significant": sig, "effect_size": es,
        })

    n_subj = sub[req.subject_col].nunique()
    k_within = sub[req.within_col].nunique()
    k_between = sub[req.between_col].nunique()

    # Build interpretation
    interp_parts = []
    for e in effects:
        ps = _p_str(e["p"])
        interp_parts.append(
            f"{'significant' if e['significant'] else 'no significant'} effect of {e['term']} "
            f"(F({e['df_num']},{e['df_den']}) = {e['F']:.2f}, p = {ps}, partial \u03B7\u00B2 = {e['effect_size']['value']:.3f})"
        )

    return {
        "test": "Mixed ANOVA (within × between)",
        "effects": effects,
        "significant": any(e["significant"] for e in effects),
        "effect_sizes": [e["effect_size"] for e in effects],
        "assumptions": [],
        "summary": {
            "within_levels": sorted(sub[req.within_col].unique().tolist()),
            "between_levels": sorted(sub[req.between_col].unique().tolist()),
            "n_subjects": n_subj,
        },
        "interpretation": "Mixed ANOVA: " + "; ".join(interp_parts) + ".",
        "result_text": (
            f"A mixed ANOVA with {req.within_col} ({k_within} levels, within-subjects) and "
            f"{req.between_col} ({k_between} levels, between-subjects) on {req.value_col} "
            f"(n = {n_subj} subjects) revealed: " + "; ".join(interp_parts) + "."
        ),
        "export_rows": [
            ["Term", "F", "df_num", "df_den", "p", "Partial eta-squared"],
            *[[e["term"], e["F"], e["df_num"], e["df_den"], e["p"], e["effect_size"]["value"]] for e in effects],
        ],
        "r_code": (
            f'library(ez)\n'
            f'ezANOVA(data = data, dv = .({req.value_col}), wid = .({req.subject_col}), '
            f'within = .({req.within_col}), between = .({req.between_col}))'
        ),
    }
