"""Causal inference sub-router for the /api/models namespace.

Hosts:
  * /psm   — Propensity Score Matching (greedy/optimal, caliper, exact match,
             SMD/variance ratio/KS balance, optional Crump trim, Rosenbaum
             bounds, conditional logistic / stratified Cox on the matched set).
  * /iptw  — Inverse Probability of Treatment Weighting (ATE/ATT/overlap,
             stabilised, percentile/hard truncation, weighted GLM/Cox with
             robust Lin-Wei sandwich SE or bootstrap percentile CI, Kish ESS).

Split out of routers/models.py (which had grown to 4846 LOC). main.py mounts
this router at the same /api/models prefix so the public API is unchanged.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from lifelines import CoxPHFitter
from pydantic import BaseModel

from services import store
from services.impute import apply_imputation
from routers._models_shared import get_df as _get_df

router = APIRouter()


# ── Propensity Score Matching ────────────────────────────────────────────────

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
async def propensity_score_matching(req: PSMRequest):
    import asyncio
    import traceback

    try:
        return await asyncio.to_thread(_run_psm, req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def _run_psm(req: PSMRequest):
    from sklearn.preprocessing import StandardScaler

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
                # Matched binary outcome → CONDITIONAL LOGISTIC REGRESSION
                # (also known as clogit / Cox partial likelihood with
                # stratified pairs). Each matched set is its own stratum so
                # the intercept and any time-invariant pair-level effects
                # are absorbed into the conditioning, leaving only the
                # treatment effect and any within-pair covariates. For 1:1
                # matching with no other covariates this reduces to
                # McNemar's test on the discordant pairs.
                from statsmodels.discrete.conditional_models import ConditionalLogit

                df_out = df_matched[[req.treatment_col, req.outcome_col, "_match_id_"]].copy()
                df_out[req.outcome_col] = y_out.astype(int)
                df_out[req.treatment_col] = pd.to_numeric(df_out[req.treatment_col], errors="coerce").astype(float)
                df_out = df_out.dropna()

                # Drop matched sets that are uninformative for clogit:
                # a stratum where every outcome is 0 or every outcome is 1
                # contributes 0 to the conditional likelihood, so we count
                # them but exclude from the fit.
                grp_y = df_out.groupby("_match_id_")[req.outcome_col]
                informative_ids = grp_y.nunique().loc[lambda s: s > 1].index
                n_informative_pairs = int(len(informative_ids))
                df_clogit = df_out[df_out["_match_id_"].isin(informative_ids)].copy()

                if n_informative_pairs == 0:
                    outcome_result = {
                        "error": "No informative (discordant) matched sets — every pair has identical outcomes. Conditional logistic cannot fit.",
                    }
                else:
                    try:
                        X_out = df_clogit[[req.treatment_col]].astype(float).values
                        y_arr_out = df_clogit[req.outcome_col].astype(int).values
                        grp_arr = df_clogit["_match_id_"].values
                        mod_cl = ConditionalLogit(
                            y_arr_out, X_out, groups=grp_arr,
                        )
                        res_cl = mod_cl.fit(disp=False)
                        # ConditionalLogit exposes params in the same order
                        # as the columns of X — single coefficient for the
                        # treatment indicator here.
                        coef = float(res_cl.params[0])
                        se = float(res_cl.bse[0])
                        p_val = float(res_cl.pvalues[0])
                        ci_lo = coef - 1.959963984540054 * se
                        ci_hi = coef + 1.959963984540054 * se
                        coefs_out = [{
                            "variable": req.treatment_col,
                            "estimate": round(coef, 6),
                            "or": round(float(np.exp(coef)), 4),
                            "se": round(se, 6),
                            "z": round(coef / se, 4) if se > 0 else None,
                            "p": round(p_val, 6),
                            "ci_low":  round(ci_lo, 4),
                            "ci_high": round(ci_hi, 4),
                            "or_low":  round(float(np.exp(ci_lo)), 4),
                            "or_high": round(float(np.exp(ci_hi)), 4),
                        }]
                        outcome_result = {
                            "type": "conditional_logistic",
                            "model": "Conditional logistic regression (matched-set stratification)",
                            "n": int(len(df_clogit)),
                            "n_matched_sets": int(df_out["_match_id_"].nunique()),
                            "n_informative_sets": n_informative_pairs,
                            "n_uninformative_sets": int(df_out["_match_id_"].nunique()) - n_informative_pairs,
                            "coefficients": coefs_out,
                            "log_likelihood": round(float(res_cl.llf), 4),
                            "method_note": (
                                "Conditional likelihood treats each matched set as a stratum. "
                                "Uninformative (concordant) sets contribute 0 to the likelihood and are dropped. "
                                "For 1:1 matching with treatment as the only covariate this is equivalent to McNemar's test."
                            ),
                        }
                    except Exception as cl_exc:
                        # Fallback to standard logistic with HC1 — should be rare.
                        X_out = sm.add_constant(df_out[[req.treatment_col]].astype(float))
                        m_out = sm.Logit(df_out[req.outcome_col].astype(int).values, X_out).fit(disp=False, cov_type="HC1")
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
                            "model": "Logistic Regression [Robust SE] (matched cohort, clogit fallback)",
                            "n": int(len(df_matched)),
                            "coefficients": coefs_out,
                            "aic": round(float(m_out.aic), 2),
                            "bic": round(float(m_out.bic), 2),
                            "method_note": f"Conditional logistic fit failed ({cl_exc}); fell back to logistic with robust SE.",
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


# ── Inverse Probability of Treatment Weighting (IPTW) ────────────────────────
#
# Companion to PSM. PSM matches and discards; IPTW reweights every unit by the
# inverse propensity score, keeps the whole sample, and supports ATE / ATT /
# overlap estimands directly. Outcome models become weighted GLM (binary) or
# weighted Cox (survival), with robust (Lin & Wei sandwich) standard errors
# or, optionally, a bootstrap percentile CI.


class IPTWRequest(BaseModel):
    session_id: str
    treatment_col: str
    covariates: List[str]
    outcome_col: Optional[str] = None
    imputation: Optional[str] = "listwise"
    random_state: Optional[int] = 42
    score_method: Optional[str] = "logistic"        # 'logistic' | 'probit' | 'gbm'
    estimand: Optional[str] = "ate"                  # 'ate' | 'att' | 'overlap'
    stabilize: Optional[bool] = True
    trim_common_support: Optional[bool] = False
    weight_truncation: Optional[str] = "percentile"  # 'percentile' | 'hard' | 'none'
    weight_truncation_lo: Optional[float] = 0.01
    weight_truncation_hi: Optional[float] = 0.99
    weight_truncation_max: Optional[float] = 10.0
    outcome_type: Optional[str] = "binary"           # 'binary' | 'survival'
    survival_duration_col: Optional[str] = None
    survival_event_col: Optional[str] = None
    se_method: Optional[str] = "robust"              # 'robust' | 'bootstrap'
    bootstrap_reps: Optional[int] = 500


def _iptw_weights(ps: np.ndarray, t: np.ndarray, estimand: str,
                  stabilize: bool) -> np.ndarray:
    """Compute IPTW weights for one of three estimands.

    - ATE: w = T/ps + (1-T)/(1-ps)
    - ATT: w = T + (1-T) * ps / (1-ps)
    - Overlap (Crump 2009): w = (1-ps) for treated, ps for control.

    `stabilize` multiplies by the marginal P(T=1) (and 1-P(T=1) for controls)
    to bring weights closer to 1 and reduce standard-error inflation; the
    point estimate is unchanged.
    """
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    t = t.astype(int)
    p_t = float(t.mean())
    if estimand == "ate":
        w = np.where(t == 1, 1.0 / ps, 1.0 / (1.0 - ps))
        if stabilize:
            w = np.where(t == 1, w * p_t, w * (1 - p_t))
    elif estimand == "att":
        w = np.where(t == 1, 1.0, ps / (1.0 - ps))
        if stabilize:
            # Stabilised ATT divides by P(T=1) on controls (Hernán/Robins)
            w = np.where(t == 1, w, w * (1 - p_t))
    elif estimand == "overlap":
        w = np.where(t == 1, 1.0 - ps, ps)
        # Overlap weights are naturally bounded so stabilisation is a no-op
    else:
        raise HTTPException(status_code=422, detail=f"Unknown estimand: {estimand}")
    return w.astype(float)


def _truncate_weights(w: np.ndarray, mode: str, lo: float, hi: float,
                      max_w: float) -> Tuple[np.ndarray, dict]:
    """Truncate IPTW weights to control influence of extremes.

    'percentile' clips at the lo / hi sample percentiles (Cole & Hernán).
    'hard' clips at an absolute maximum.
    'none' returns the weights unchanged.

    Returns (truncated weights, diagnostic dict).
    """
    info: dict = {"mode": mode, "n_trimmed": 0,
                  "cap_lo": None, "cap_hi": None, "cap_max": None}
    if mode == "percentile":
        cap_lo = float(np.quantile(w, max(0.0, min(1.0, lo))))
        cap_hi = float(np.quantile(w, max(0.0, min(1.0, hi))))
        before = w.copy()
        w = np.clip(w, cap_lo, cap_hi)
        info.update({"cap_lo": round(cap_lo, 4), "cap_hi": round(cap_hi, 4),
                     "n_trimmed": int(((before < cap_lo) | (before > cap_hi)).sum())})
    elif mode == "hard":
        cap = float(max_w)
        before = w.copy()
        w = np.clip(w, 0.0, cap)
        info.update({"cap_max": round(cap, 4), "n_trimmed": int((before > cap).sum())})
    return w, info


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    ws = float(w.sum())
    if ws <= 0:
        return float("nan")
    return float(np.sum(x * w) / ws)


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    ws = float(w.sum())
    if ws <= 0:
        return float("nan")
    m = _weighted_mean(x, w)
    return float(np.sum(w * (x - m) ** 2) / ws)


def _weighted_smd(x_t: np.ndarray, w_t: np.ndarray,
                  x_c: np.ndarray, w_c: np.ndarray,
                  denom_sd: float, is_binary: bool) -> float:
    """Weighted standardized mean difference for one covariate."""
    if len(x_t) == 0 or len(x_c) == 0:
        return 0.0
    m_t = _weighted_mean(x_t, w_t)
    m_c = _weighted_mean(x_c, w_c)
    if is_binary:
        denom = denom_sd if denom_sd > 1e-9 else np.sqrt(
            (m_t * (1 - m_t) + m_c * (1 - m_c)) / 2
        )
    else:
        denom = denom_sd
    return float(abs(m_t - m_c) / denom) if denom > 1e-9 else 0.0


def _kish_ess(w: np.ndarray) -> float:
    """Kish effective sample size: (Σw)² / Σ(w²)."""
    if len(w) == 0:
        return 0.0
    num = float(w.sum()) ** 2
    den = float((w ** 2).sum())
    return float(num / den) if den > 0 else 0.0


@router.post("/iptw")
async def iptw(req: IPTWRequest):
    import asyncio
    import traceback
    try:
        return await asyncio.to_thread(_run_iptw, req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def _run_iptw(req: IPTWRequest):
    from sklearn.preprocessing import StandardScaler

    df_full = _get_df(req.session_id)
    outcome_type = (req.outcome_type or "binary").lower()
    if outcome_type not in ("binary", "survival"):
        raise HTTPException(status_code=422, detail="outcome_type must be 'binary' or 'survival'.")
    estimand = (req.estimand or "ate").lower()
    if estimand not in ("ate", "att", "overlap"):
        raise HTTPException(status_code=422, detail="estimand must be 'ate', 'att', or 'overlap'.")
    trunc_mode = (req.weight_truncation or "percentile").lower()
    if trunc_mode not in ("percentile", "hard", "none"):
        raise HTTPException(status_code=422, detail="weight_truncation must be 'percentile', 'hard', or 'none'.")
    se_method = (req.se_method or "robust").lower()
    if se_method not in ("robust", "bootstrap"):
        raise HTTPException(status_code=422, detail="se_method must be 'robust' or 'bootstrap'.")

    extra_outcome_cols: List[str] = []
    if outcome_type == "binary" and req.outcome_col:
        extra_outcome_cols.append(req.outcome_col)
    if outcome_type == "survival":
        if not req.survival_duration_col or not req.survival_event_col:
            raise HTTPException(status_code=422, detail="Survival outcome requires survival_duration_col and survival_event_col.")
        extra_outcome_cols.extend([req.survival_duration_col, req.survival_event_col])

    needed = list(dict.fromkeys([req.treatment_col] + req.covariates + extra_outcome_cols))
    missing_cols = [c for c in needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing_cols}")

    df = apply_imputation(df_full[needed], needed, req.imputation or "listwise").reset_index(drop=True)

    treat_vals = df[req.treatment_col].astype(float)
    if not set(treat_vals.unique().tolist()) <= {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422,
            detail=f"Treatment column '{req.treatment_col}' must be binary (0 = control, 1 = treated).")

    # Step 1: Propensity scores
    X = pd.get_dummies(df[req.covariates], drop_first=True).astype(float)
    y = treat_vals.astype(int).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    score_method = (req.score_method or "logistic").lower()
    ps = _fit_propensity_scores(X_scaled, y, score_method, req.random_state)
    df = df.copy()
    df["_ps_"] = ps
    df["_treat_"] = y

    # Step 2: Optional Crump 2009 common-support trim (pre-weighting)
    n_trimmed_common_support = 0
    keep_mask = np.ones_like(y, dtype=bool)
    if req.trim_common_support:
        treated_ps = ps[y == 1]
        control_ps = ps[y == 0]
        if len(treated_ps) > 0 and len(control_ps) > 0:
            lo = max(treated_ps.min(), control_ps.min())
            hi = min(treated_ps.max(), control_ps.max())
            keep_mask = (ps >= lo) & (ps <= hi)
            n_trimmed_common_support = int((~keep_mask).sum())

    df_keep = df[keep_mask].reset_index(drop=True).copy()
    ps_keep = df_keep["_ps_"].values
    t_keep = df_keep["_treat_"].values.astype(int)
    if (t_keep.sum() == 0) or ((1 - t_keep).sum() == 0):
        raise HTTPException(status_code=422, detail="No treated or no control units remain after common-support trim.")

    # Step 3: IPTW weights
    w_raw = _iptw_weights(ps_keep, t_keep, estimand, bool(req.stabilize))

    # Step 4: Truncate weights
    w, trunc_info = _truncate_weights(
        w_raw, trunc_mode,
        float(req.weight_truncation_lo or 0.01),
        float(req.weight_truncation_hi or 0.99),
        float(req.weight_truncation_max or 10.0),
    )
    df_keep["_w_"] = w

    # Weight summary diagnostics
    w_t = w[t_keep == 1]
    w_c = w[t_keep == 0]
    weight_summary = {
        "min": round(float(w.min()), 4),
        "max": round(float(w.max()), 4),
        "mean": round(float(w.mean()), 4),
        "median": round(float(np.median(w)), 4),
        "max_treated": round(float(w_t.max()) if len(w_t) > 0 else 0.0, 4),
        "max_control": round(float(w_c.max()) if len(w_c) > 0 else 0.0, 4),
        "ess_treated": round(_kish_ess(w_t), 2),
        "ess_control": round(_kish_ess(w_c), 2),
    }

    # Step 5: Weighted balance (SMD + variance ratio + KS) before vs after
    smd_before: dict = {}
    smd_after: dict = {}
    var_before: dict = {}
    var_after: dict = {}
    ks_before: dict = {}
    ks_after: dict = {}
    treat_mask_full = df["_treat_"].values
    for cov in req.covariates:
        col_full = df[cov]
        col_keep = df_keep[cov]

        col_t_un = col_full[treat_mask_full == 1]
        col_c_un = col_full[treat_mask_full == 0]
        denom = _pooled_sd(col_t_un, col_c_un)

        smd_before[cov] = round(_compute_smd(col_t_un, col_c_un, denom_sd=denom), 4)
        var_before[cov] = _variance_ratio(col_t_un, col_c_un)
        ks_before[cov] = _ks_p(col_t_un, col_c_un)

        # Weighted "after"
        try:
            col_arr = pd.to_numeric(col_keep, errors="coerce").astype(float).values
        except Exception:
            col_arr = col_keep.astype("category").cat.codes.values.astype(float)
        is_binary = pd.concat([col_t_un, col_c_un]).nunique() <= 2
        mask_t = t_keep == 1
        mask_c = t_keep == 0
        smd_w = _weighted_smd(
            col_arr[mask_t], w[mask_t],
            col_arr[mask_c], w[mask_c],
            denom_sd=denom, is_binary=bool(is_binary),
        )
        smd_after[cov] = round(smd_w, 4)
        if is_binary:
            var_after[cov] = None
            ks_after[cov] = None
        else:
            v_t = _weighted_var(col_arr[mask_t], w[mask_t])
            v_c = _weighted_var(col_arr[mask_c], w[mask_c])
            var_after[cov] = round(v_t / v_c, 4) if v_c > 1e-12 else None
            # For the KS p-value we approximate the weighted distribution by
            # resampling proportionally to the weights so we can re-use the
            # plain two-sample KS test. Cheap and matches user expectations.
            try:
                from scipy.stats import ks_2samp
                rs = np.random.default_rng(req.random_state if req.random_state is not None else 42)
                # Resample to the original group size, normalised weights.
                def _sample(arr: np.ndarray, ww: np.ndarray, n: int) -> np.ndarray:
                    p = ww / ww.sum()
                    return arr[rs.choice(len(arr), size=n, replace=True, p=p)]
                arr_t = _sample(col_arr[mask_t], w[mask_t], max(50, int(mask_t.sum())))
                arr_c = _sample(col_arr[mask_c], w[mask_c], max(50, int(mask_c.sum())))
                _, p_ks = ks_2samp(arr_t, arr_c)
                ks_after[cov] = round(float(p_ks), 6)
            except Exception:
                ks_after[cov] = None

    # Balance flag
    smd_after_vals = [v for v in smd_after.values() if v is not None]
    avg_before = float(np.mean(list(smd_before.values()))) if smd_before else 0.0
    avg_after = float(np.mean(smd_after_vals)) if smd_after_vals else 0.0
    balance_achieved = bool(all(v < 0.10 for v in smd_after_vals))

    # PS distribution (unweighted + weighted overlay)
    ps_dist = {
        "treated_unmatched": ps[y == 1].tolist(),
        "control_unmatched": ps[y == 0].tolist(),
        "treated_weighted": ps_keep[t_keep == 1].tolist(),
        "control_weighted": ps_keep[t_keep == 0].tolist(),
    }
    weight_distribution = {
        "treated": w_t.tolist(),
        "control": w_c.tolist(),
    }

    # ── Step 6: Weighted outcome model ───────────────────────────────────────
    outcome_result: Optional[dict] = None
    if outcome_type == "survival":
        try:
            dur = pd.to_numeric(df_keep[req.survival_duration_col], errors="coerce")
            evt = pd.to_numeric(df_keep[req.survival_event_col], errors="coerce")
            if np.any(dur.dropna() < 0):
                outcome_result = {"error": f"survival_duration_col '{req.survival_duration_col}' must be ≥ 0."}
            elif set(evt.dropna().unique().tolist()) - {0.0, 1.0}:
                outcome_result = {"error": f"survival_event_col '{req.survival_event_col}' must be binary 0/1."}
            else:
                cox_df = pd.DataFrame({
                    "_dur_": dur.values.astype(float),
                    "_evt_": evt.values.astype(int),
                    req.treatment_col: t_keep.astype(float),
                    "_w_": w,
                }).dropna()
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col="_dur_", event_col="_evt_",
                        weights_col="_w_", robust=True)
                coef = float(cph.params_.iloc[0])
                se   = float(cph.standard_errors_.iloc[0])
                ci   = cph.confidence_intervals_.iloc[0]
                try:
                    p_val = float(cph.summary["p"].iloc[0])
                except Exception:
                    p_val = None
                outcome_result = {
                    "type": "weighted_cox",
                    "model": f"IPTW-weighted Cox PH ({estimand.upper()} weights, robust Lin-Wei SE)",
                    "n": int(len(cox_df)),
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
                    "method_note": ("Cox proportional hazards weighted by the IPTW. "
                                    "Lin & Wei (1989) sandwich estimator handles the weighted score."),
                }
        except Exception as ex:
            outcome_result = {"error": f"Weighted Cox failed: {ex}"}

    elif req.outcome_col and req.outcome_col in df_keep.columns:
        try:
            y_out = pd.to_numeric(df_keep[req.outcome_col], errors="coerce")
            out_vals = set(y_out.dropna().unique().tolist())
            if not out_vals <= {0, 1, 0.0, 1.0}:
                outcome_result = {"error": f"Outcome must be binary 0/1 for weighted analysis. Found: {sorted(out_vals)[:10]}"}
            else:
                df_out = pd.DataFrame({
                    "_y_": y_out.astype(int).values,
                    req.treatment_col: t_keep.astype(float),
                    "_w_": w,
                }).dropna()
                # Weighted logistic via GLM with HC1 sandwich SE.
                import statsmodels.api as _sm_
                X_out = _sm_.add_constant(df_out[[req.treatment_col]].astype(float))
                glm = _sm_.GLM(df_out["_y_"].values, X_out,
                               family=_sm_.families.Binomial(),
                               freq_weights=df_out["_w_"].values)
                m_out = glm.fit(cov_type="HC1")
                ci_out = m_out.conf_int()
                coefs_out = []
                for var in m_out.params.index:
                    est = float(m_out.params[var])
                    se_v = float(m_out.bse[var])
                    coefs_out.append({
                        "variable": str(var),
                        "estimate": round(est, 6),
                        "or": round(float(np.exp(est)), 4),
                        "se": round(se_v, 6),
                        "z": round(float(m_out.tvalues[var]), 4) if se_v > 0 else None,
                        "p": round(float(m_out.pvalues[var]), 6),
                        "ci_low":  round(float(ci_out.loc[var, 0]), 4),
                        "ci_high": round(float(ci_out.loc[var, 1]), 4),
                        "or_low":  round(float(np.exp(ci_out.loc[var, 0])), 4),
                        "or_high": round(float(np.exp(ci_out.loc[var, 1])), 4),
                    })
                outcome_result = {
                    "type": "weighted_glm",
                    "model": f"IPTW-weighted logistic ({estimand.upper()} weights, robust HC1 SE)",
                    "n": int(len(df_out)),
                    "coefficients": coefs_out,
                    "method_note": ("Binomial GLM with frequency weights = IPTW. "
                                    "HC1 sandwich estimator handles the weighted score."),
                }
        except Exception as ex:
            outcome_result = {"error": f"Weighted GLM failed: {ex}"}

    # ── Step 7: Optional bootstrap CI (replaces robust CI when requested) ────
    if se_method == "bootstrap" and outcome_result and not outcome_result.get("error"):
        try:
            reps = max(50, int(req.bootstrap_reps or 500))
            rs = np.random.default_rng(req.random_state if req.random_state is not None else 42)
            df_full_keep = df_keep.copy()
            X_full = X_scaled  # NOTE: X_scaled is on the unfiltered df
            X_filt = X_full[keep_mask]
            n_keep = len(df_full_keep)
            est_idx = 0  # treatment-column index in coefficients list (after intercept for GLM)
            ests = []
            for _b in range(reps):
                idx_b = rs.integers(0, n_keep, size=n_keep)
                df_b = df_full_keep.iloc[idx_b].reset_index(drop=True)
                X_b = X_filt[idx_b]
                y_b = df_b["_treat_"].values.astype(int)
                if y_b.sum() == 0 or (1 - y_b).sum() == 0:
                    continue
                ps_b = _fit_propensity_scores(X_b, y_b, score_method, req.random_state)
                w_b = _iptw_weights(ps_b, y_b, estimand, bool(req.stabilize))
                w_b, _ = _truncate_weights(
                    w_b, trunc_mode,
                    float(req.weight_truncation_lo or 0.01),
                    float(req.weight_truncation_hi or 0.99),
                    float(req.weight_truncation_max or 10.0),
                )
                try:
                    if outcome_type == "survival":
                        cox_df_b = pd.DataFrame({
                            "_dur_": pd.to_numeric(df_b[req.survival_duration_col], errors="coerce").astype(float).values,
                            "_evt_": pd.to_numeric(df_b[req.survival_event_col], errors="coerce").astype(int).values,
                            req.treatment_col: y_b.astype(float),
                            "_w_": w_b,
                        }).dropna()
                        cph_b = CoxPHFitter()
                        cph_b.fit(cox_df_b, duration_col="_dur_", event_col="_evt_",
                                  weights_col="_w_", robust=False)
                        ests.append(float(cph_b.params_.iloc[0]))
                    else:
                        import statsmodels.api as _sm__
                        y_b_out = pd.to_numeric(df_b[req.outcome_col], errors="coerce").astype(int).values
                        X_b_out = _sm__.add_constant(pd.DataFrame({req.treatment_col: y_b.astype(float)}))
                        glm_b = _sm__.GLM(y_b_out, X_b_out,
                                          family=_sm__.families.Binomial(),
                                          freq_weights=w_b)
                        m_b = glm_b.fit()
                        ests.append(float(m_b.params.iloc[1] if hasattr(m_b.params, "iloc") else m_b.params[1]))
                except Exception:
                    continue
            if ests:
                lo = float(np.quantile(ests, 0.025))
                hi = float(np.quantile(ests, 0.975))
                # Update first coefficient (treatment) with bootstrap CI
                coef0 = outcome_result["coefficients"][est_idx]
                coef0["ci_low"] = round(lo, 4)
                coef0["ci_high"] = round(hi, 4)
                if outcome_type == "survival":
                    coef0["hr_low"] = round(float(np.exp(lo)), 4)
                    coef0["hr_high"] = round(float(np.exp(hi)), 4)
                else:
                    coef0["or_low"] = round(float(np.exp(lo)), 4)
                    coef0["or_high"] = round(float(np.exp(hi)), 4)
                outcome_result["type"] = outcome_result["type"] + "_bootstrap"
                outcome_result["method_note"] = (
                    outcome_result.get("method_note", "")
                    + f"\nBootstrap percentile CI over {len(ests)} resamples (PS refit each draw)."
                )
                outcome_result["bootstrap_reps"] = len(ests)
        except Exception as ex:
            outcome_result["bootstrap_error"] = str(ex)

    # Audit
    try:
        store.log_action(req.session_id, "iptw", {
            "score_method": score_method,
            "estimand": estimand,
            "stabilize": bool(req.stabilize),
            "weight_truncation": trunc_mode,
            "outcome_type": outcome_type,
            "se_method": se_method,
            "n_trimmed_common_support": int(n_trimmed_common_support),
            "n_weight_truncated": int(trunc_info.get("n_trimmed") or 0),
            "stabilize_used": bool(req.stabilize),
        })
    except Exception:
        pass

    return {
        "method": "iptw",
        "n_total": int(len(df)),
        "n_treated": int((y == 1).sum()),
        "n_control": int((y == 0).sum()),
        "n_trimmed_common_support": n_trimmed_common_support,
        "score_method": score_method,
        "estimand": estimand,
        "stabilize": bool(req.stabilize),
        "se_method": se_method,
        "weight_summary": weight_summary,
        "weight_truncation": trunc_info,
        "weight_distribution": weight_distribution,
        "balance_achieved": balance_achieved,
        "avg_smd_before": round(avg_before, 4),
        "avg_smd_after": round(avg_after, 4),
        "reduction_pct": round(((avg_before - avg_after) / avg_before) * 100, 2) if avg_before > 0 else 0.0,
        "smd_before": smd_before,
        "smd_after": smd_after,
        "variance_ratio_before": var_before,
        "variance_ratio_after": var_after,
        "ks_p_before": ks_before,
        "ks_p_after": ks_after,
        "ps_distribution": ps_dist,
        "outcome_result": outcome_result,
    }
