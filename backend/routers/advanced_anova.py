"""Advanced ANOVA: ANCOVA, two-way ANOVA with interaction + estimated marginal means."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from services import store
from services.impute import apply_imputation
from services.stat_utils import (
    partial_eta_squared, check_normality, group_summary, tukey_hsd, sorted_groups,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _safe_col(name: str) -> str:
    """Make column name safe for statsmodels formula (wrap in Q())."""
    return f"Q('{name}')"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ANCOVA
# ═══════════════════════════════════════════════════════════════════════════════

class AncovaRequest(BaseModel):
    session_id: str
    outcome: str
    group_col: str
    covariates: List[str]
    alpha: float = 0.05
    imputation: str = "listwise"


@router.post("/ancova")
def ancova(req: AncovaRequest):
    df_full = _get_df(req.session_id)
    cols = [req.outcome, req.group_col] + req.covariates
    for c in cols:
        if c not in df_full.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    df = apply_imputation(df_full, cols, req.imputation)
    df[req.outcome] = pd.to_numeric(df[req.outcome], errors="coerce")
    for cov in req.covariates:
        df[cov] = pd.to_numeric(df[cov], errors="coerce")
    df = df.dropna(subset=cols)

    if len(df) < 10:
        raise HTTPException(400, "Need at least 10 complete rows.")

    groups = sorted_groups(df[req.group_col])
    if len(groups) < 2:
        raise HTTPException(400, "Group column must have at least 2 levels.")

    # Build formula: outcome ~ C(group) + cov1 + cov2 + ...
    cov_terms = " + ".join([_safe_col(c) for c in req.covariates])
    formula = f"{_safe_col(req.outcome)} ~ C({_safe_col(req.group_col)}) + {cov_terms}"

    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as exc:
        raise HTTPException(400, f"Model fitting error: {exc}")

    # Type II ANOVA table
    try:
        aov = anova_lm(model, typ=2)
    except Exception:
        aov = anova_lm(model, typ=1)

    # Extract group effect
    group_key = [k for k in aov.index if req.group_col in str(k)]
    if not group_key:
        raise HTTPException(500, "Could not find group effect in ANOVA table.")

    gk = group_key[0]
    F_val = float(aov.loc[gk, "F"])
    p_val = float(aov.loc[gk, "PR(>F)"])
    df_num = int(aov.loc[gk, "df"])
    df_den = int(aov.loc["Residual", "df"])
    sig = bool(p_val < req.alpha)
    es = partial_eta_squared(F_val, df_num, df_den)
    ps = _p_str(p_val)

    # Assumption checks
    assumptions = [check_normality(model.resid.values, "Residuals")]
    # Homogeneity of regression slopes: test group x covariate interaction
    for cov in req.covariates:
        try:
            int_formula = f"{_safe_col(req.outcome)} ~ C({_safe_col(req.group_col)}) * {_safe_col(cov)}"
            int_model = smf.ols(int_formula, data=df).fit()
            int_aov = anova_lm(int_model, typ=2)
            int_key = [k for k in int_aov.index if req.group_col in str(k) and cov in str(k)]
            if int_key:
                int_p = float(int_aov.loc[int_key[0], "PR(>F)"])
                met = bool(int_p >= 0.05)
                assumptions.append({
                    "name": f"Homogeneity of slopes ({cov})",
                    "met": met,
                    "detail": f"Interaction p = {int_p:.4f}" + (" — ANCOVA assumption violated" if not met else ""),
                })
        except Exception:
            pass

    # Estimated marginal means (EMMs)
    cov_means = {c: float(df[c].mean()) for c in req.covariates}
    emms = []
    for g in sorted(groups):
        row = {req.group_col: g, **cov_means}
        pred_df = pd.DataFrame([row])
        try:
            emm = float(model.predict(pred_df).iloc[0])
        except Exception:
            emm = None
        emms.append({"group": str(g), "emm": round(emm, 4) if emm else None})

    # Covariate effects
    cov_effects = []
    for cov in req.covariates:
        cov_key = [k for k in aov.index if cov in str(k) and req.group_col not in str(k)]
        if cov_key:
            ck = cov_key[0]
            cov_effects.append({
                "covariate": cov,
                "F": round(float(aov.loc[ck, "F"]), 4),
                "p": round(float(aov.loc[ck, "PR(>F)"]), 6),
                "significant": bool(aov.loc[ck, "PR(>F)"] < req.alpha),
            })

    # ANOVA table as export rows
    export_rows = [["Source", "SS", "df", "F", "p"]]
    for idx_name, row in aov.iterrows():
        if str(idx_name) != "Residual":
            export_rows.append([str(idx_name), round(float(row["sum_sq"]), 4),
                                int(row["df"]), round(float(row["F"]), 4), round(float(row["PR(>F)"]), 6)])
    export_rows.append(["Residual", round(float(aov.loc["Residual", "sum_sq"]), 4),
                         int(aov.loc["Residual", "df"]), "", ""])

    cov_list = ", ".join(req.covariates)
    return {
        "test": "ANCOVA",
        "F": round(F_val, 4), "df_num": df_num, "df_den": df_den, "p": float(p_val),
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": assumptions,
        "emms": emms,
        "covariate_effects": cov_effects,
        "summary": {str(g): group_summary(df[df[req.group_col] == g][req.outcome].values, str(g))
                    for g in sorted(groups)},
        "interpretation": (
            f"After controlling for {cov_list}, there was {'a significant' if sig else 'no significant'} "
            f"effect of {req.group_col} on {req.outcome} "
            f"(F({df_num},{df_den}) = {F_val:.2f}, p = {ps}, partial \u03B7\u00B2 = {es['value']:.3f} [{es['magnitude']}])"
        ),
        "result_text": (
            f"An ANCOVA was conducted with {req.outcome} as the dependent variable, {req.group_col} as the factor, "
            f"and {cov_list} as covariate(s). After controlling for the covariate(s), there was "
            f"{'a significant' if sig else 'no significant'} effect of {req.group_col} "
            f"(F({df_num},{df_den}) = {F_val:.2f}, p = {ps}, partial \u03B7\u00B2 = {es['value']:.3f} [{es['magnitude']}]). "
            f"Estimated marginal means: " + ", ".join([f"{e['group']} = {e['emm']}" for e in emms if e['emm']]) + "."
        ),
        "export_rows": export_rows,
        "r_code": (
            f'library(emmeans)\n'
            f'model <- lm({req.outcome} ~ {req.group_col} + {" + ".join(req.covariates)}, data = data)\n'
            f'anova(model)\n'
            f'emmeans(model, ~ {req.group_col})'
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TWO-WAY ANOVA
# ═══════════════════════════════════════════════════════════════════════════════

class TwoWayAnovaRequest(BaseModel):
    session_id: str
    outcome: str
    factor1: str
    factor2: str
    alpha: float = 0.05
    imputation: str = "listwise"


@router.post("/two_way_anova")
def two_way_anova(req: TwoWayAnovaRequest):
    df_full = _get_df(req.session_id)
    cols = [req.outcome, req.factor1, req.factor2]
    for c in cols:
        if c not in df_full.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    df = apply_imputation(df_full, cols, req.imputation)
    df[req.outcome] = pd.to_numeric(df[req.outcome], errors="coerce")
    df = df.dropna(subset=cols)

    if len(df) < 12:
        raise HTTPException(400, "Need at least 12 complete rows.")

    k1 = df[req.factor1].nunique()
    k2 = df[req.factor2].nunique()
    if k1 < 2 or k2 < 2:
        raise HTTPException(400, "Both factors must have at least 2 levels.")

    # Full factorial model: outcome ~ C(f1) * C(f2)
    formula = f"{_safe_col(req.outcome)} ~ C({_safe_col(req.factor1)}) * C({_safe_col(req.factor2)})"
    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as exc:
        raise HTTPException(400, f"Model fitting error: {exc}")

    try:
        aov = anova_lm(model, typ=2)
    except Exception:
        aov = anova_lm(model, typ=1)

    # Parse effects
    effects = []
    for idx_name, row in aov.iterrows():
        name = str(idx_name)
        if name == "Residual":
            continue
        F_val = float(row["F"]) if not np.isnan(row["F"]) else 0
        p_val = float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else 1
        df_n = int(row["df"])
        df_d = int(aov.loc["Residual", "df"])
        sig_e = bool(p_val < req.alpha)
        es_e = partial_eta_squared(F_val, df_n, df_d)

        # Determine readable term name
        if req.factor1 in name and req.factor2 in name:
            term_label = f"{req.factor1} \u00D7 {req.factor2} (interaction)"
        elif req.factor1 in name:
            term_label = req.factor1
        elif req.factor2 in name:
            term_label = req.factor2
        else:
            term_label = name

        effects.append({
            "term": term_label, "raw_term": name,
            "F": round(F_val, 4), "df_num": df_n, "df_den": df_d,
            "p": round(p_val, 6), "significant": sig_e, "effect_size": es_e,
        })

    # Assumptions
    assumptions = [check_normality(model.resid.values, "Residuals")]

    # EMMs per cell
    emms = []
    for g1 in sorted(df[req.factor1].unique()):
        for g2 in sorted(df[req.factor2].unique()):
            cell = df[(df[req.factor1] == g1) & (df[req.factor2] == g2)][req.outcome]
            emms.append({
                "factor1": str(g1), "factor2": str(g2),
                "n": int(len(cell)),
                "mean": round(float(cell.mean()), 4) if len(cell) > 0 else None,
                "sd": round(float(cell.std(ddof=1)), 4) if len(cell) > 1 else None,
            })

    # Post-hoc for significant main effects
    posthoc = []
    posthoc_method = None
    for eff in effects:
        if not eff["significant"] or "interaction" in eff["term"]:
            continue
        # Determine which factor
        if req.factor1 in eff["raw_term"] and req.factor2 not in eff["raw_term"]:
            factor = req.factor1
        elif req.factor2 in eff["raw_term"] and req.factor1 not in eff["raw_term"]:
            factor = req.factor2
        else:
            continue
        grp_dict = {str(name): g[req.outcome].astype(float).values
                    for name, g in df.groupby(factor)}
        ph = tukey_hsd(grp_dict)
        for p in ph:
            p["factor"] = factor
        posthoc.extend(ph)
        posthoc_method = "Tukey HSD"

    # Build interpretation
    interp_parts = []
    for e in effects:
        ps = _p_str(e["p"])
        interp_parts.append(
            f"{'significant' if e['significant'] else 'no significant'} effect of {e['term']} "
            f"(F({e['df_num']},{e['df_den']}) = {e['F']:.2f}, p = {ps}, partial \u03B7\u00B2 = {e['effect_size']['value']:.3f})"
        )

    # Export rows
    export_rows = [["Source", "SS", "df", "F", "p", "Partial eta-sq"]]
    for idx_name, row in aov.iterrows():
        name = str(idx_name)
        export_rows.append([
            name,
            round(float(row["sum_sq"]), 4),
            int(row["df"]),
            round(float(row["F"]), 4) if not np.isnan(row["F"]) else "",
            round(float(row["PR(>F)"]), 6) if not np.isnan(row["PR(>F)"]) else "",
            "",
        ])

    return {
        "test": f"Two-way ANOVA ({req.factor1} \u00D7 {req.factor2})",
        "effects": effects,
        "significant": any(e["significant"] for e in effects),
        "effect_sizes": [e["effect_size"] for e in effects],
        "assumptions": assumptions,
        "emms": emms,
        "posthoc": posthoc,
        "posthoc_method": posthoc_method,
        "summary": {
            f"{req.factor1}_levels": sorted(df[req.factor1].unique().tolist()),
            f"{req.factor2}_levels": sorted(df[req.factor2].unique().tolist()),
            "n": len(df),
        },
        "interpretation": "Two-way ANOVA: " + "; ".join(interp_parts) + ".",
        "result_text": (
            f"A two-way ANOVA examined the effects of {req.factor1} ({k1} levels) and {req.factor2} ({k2} levels) "
            f"on {req.outcome} (N = {len(df)}). " + "; ".join(interp_parts) + "."
        ),
        "export_rows": export_rows,
        "r_code": (
            f'model <- aov({req.outcome} ~ {req.factor1} * {req.factor2}, data = data)\n'
            f'summary(model)\n'
            f'TukeyHSD(model)\n'
            f'library(emmeans)\n'
            f'emmeans(model, ~ {req.factor1} | {req.factor2})'
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MANCOVA  (multivariate analysis of covariance)
# ═══════════════════════════════════════════════════════════════════════════════

class MancovaRequest(BaseModel):
    session_id: str
    outcomes: List[str]            # ≥2 dependent variables (e.g. BDNF, GDNF, NTF3, NGF)
    group_col: str
    covariates: List[str] = []
    alpha: float = 0.05
    imputation: str = "listwise"


def _eta_magnitude(v: float) -> str:
    if v >= 0.14:
        return "large"
    if v >= 0.06:
        return "medium"
    if v >= 0.01:
        return "small"
    return "negligible"


@router.post("/mancova")
def mancova(req: MancovaRequest):
    """One-way MANCOVA: several continuous outcomes vs a grouping factor while
    controlling for covariates. Returns the four multivariate test statistics
    (Pillai's trace, Wilks' lambda, Hotelling-Lawley trace, Roy's greatest root)
    for the group effect, with an F-approximation, p-value, and a multivariate
    partial η². Intended as the omnibus step before per-outcome ANCOVAs.
    """
    from statsmodels.multivariate.manova import MANOVA

    df_full = _get_df(req.session_id)
    if len(req.outcomes) < 2:
        raise HTTPException(400, "MANCOVA needs at least 2 outcome variables.")
    cols = list(dict.fromkeys(req.outcomes + [req.group_col] + req.covariates))
    for c in cols:
        if c not in df_full.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    df = apply_imputation(df_full, cols, req.imputation)
    for c in req.outcomes + req.covariates:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)

    min_n = max(10, len(req.outcomes) + len(req.covariates) + 3)
    if len(df) < min_n:
        raise HTTPException(400, f"Need at least {min_n} complete rows for this MANCOVA.")

    groups = sorted_groups(df[req.group_col])
    if len(groups) < 2:
        raise HTTPException(400, "Group column must have at least 2 levels.")

    lhs = " + ".join(_safe_col(o) for o in req.outcomes)
    rhs = " + ".join([f"C({_safe_col(req.group_col)})"] + [_safe_col(c) for c in req.covariates])
    formula = f"{lhs} ~ {rhs}"

    try:
        mv = MANOVA.from_formula(formula, data=df)
        mvtest = mv.mv_test()
    except Exception as exc:
        raise HTTPException(400, f"MANCOVA fitting error: {exc}")

    # Locate the grouping term in the multivariate test output.
    group_term = next((k for k in mvtest.results.keys() if req.group_col in str(k)), None)
    if group_term is None:
        raise HTTPException(500, "Could not find the group effect in the MANCOVA output.")

    stat_df = mvtest.results[group_term]["stat"]
    tests_out = []
    pillai = None
    for name in stat_df.index:
        row = stat_df.loc[name]
        entry = {
            "test": str(name),
            "value": round(float(row["Value"]), 5),
            "F": round(float(row["F Value"]), 4),
            "num_df": round(float(row["Num DF"]), 1),
            "den_df": round(float(row["Den DF"]), 1),
            "p": float(row["Pr > F"]),
            "significant": bool(float(row["Pr > F"]) < req.alpha),
        }
        tests_out.append(entry)
        if "Pillai" in str(name):
            pillai = entry

    if pillai is None:
        pillai = tests_out[0]

    # Multivariate partial η² from Pillai's trace: V / s, s = min(#outcomes, df_hypothesis)
    s = max(1, min(len(req.outcomes), len(groups) - 1))
    eta2 = float(pillai["value"]) / s
    eta2 = max(0.0, min(1.0, eta2))
    magnitude = _eta_magnitude(eta2)
    sig = bool(pillai["p"] < req.alpha)
    ps = _p_str(pillai["p"])

    cov_list = ", ".join(req.covariates) if req.covariates else "none"
    out_list = ", ".join(req.outcomes)

    export_rows = [["Multivariate test", "Value", "F", "Num df", "Den df", "p"]]
    for t in tests_out:
        export_rows.append([t["test"], t["value"], t["F"], t["num_df"], t["den_df"], round(t["p"], 6)])

    return {
        "test": "MANCOVA",
        "outcomes": req.outcomes,
        "group_col": req.group_col,
        "covariates": req.covariates,
        "n": int(len(df)),
        "n_groups": len(groups),
        "groups": [str(g) for g in groups],
        "multivariate_tests": tests_out,
        "pillai": pillai,
        "significant": sig,
        "effect_size": {"name": "partial η² (multivariate, from Pillai)",
                        "value": round(eta2, 4), "magnitude": magnitude},
        "interpretation": (
            f"After controlling for {cov_list}, there was {'a significant' if sig else 'no significant'} "
            f"multivariate effect of {req.group_col} on the combined outcomes "
            f"(Pillai's Trace = {pillai['value']:.3f}, F({pillai['num_df']:.0f}, {pillai['den_df']:.0f}) "
            f"= {pillai['F']:.2f}, p = {ps}, partial η² = {eta2:.3f} [{magnitude}])."
            + (" Follow up with per-outcome ANCOVAs." if sig else "")
        ),
        "result_text": (
            f"A one-way MANCOVA was conducted with {out_list} as dependent variables and {req.group_col} as the "
            f"between-subjects factor, controlling for {cov_list}. "
            f"There was {'a statistically significant' if sig else 'no statistically significant'} "
            f"multivariate effect of {req.group_col} (Pillai's Trace = {pillai['value']:.3f}, "
            f"F({pillai['num_df']:.0f}, {pillai['den_df']:.0f}) = {pillai['F']:.2f}, p = {ps}, "
            f"partial η² = {eta2:.3f}). "
            + ("Given the significant omnibus result, separate ANCOVAs were examined for each outcome."
               if sig else "As the omnibus test was non-significant, follow-up ANCOVAs are not warranted.")
        ),
        "export_rows": export_rows,
        "r_code": (
            f'library(car)\n'
            f'model <- lm(cbind({", ".join(req.outcomes)}) ~ {req.group_col}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "")
            + ', data = data)\n'
            'Manova(model, type = "II")  # Pillai by default'
        ),
    }
