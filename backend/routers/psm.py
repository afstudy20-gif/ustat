"""
Dedicated router for Propensity Score Matching (PSM).

This module was extracted from the monolithic models.py to improve
maintainability and testability. All heavy statistical logic for PSM lives here
or in services/psm.py (pure functions).

Endpoint remains mounted at /api/models/psm for backward compatibility with
the frontend.
"""

from __future__ import annotations

import traceback
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from services import store
from services.impute import apply_imputation
from services.psm import (
    _compute_smd,
    _fit_propensity_scores,
    _ks_p,
    _match_greedy,
    _match_optimal,
    _pooled_sd,
    _rosenbaum_bounds,
    _variance_ratio,
)

router = APIRouter()


# ── Request Model ──────────────────────────────────────────────────────────────

class PSMRequest(BaseModel):
    session_id: str
    treatment_col: str
    covariates: List[str]
    outcome_col: Optional[str] = None
    caliper: Optional[float] = 0.2
    caliper_scale: Optional[str] = "logit"
    ratio: Optional[int] = 1
    imputation: Optional[str] = "listwise"
    trim_common_support: Optional[bool] = False
    random_state: Optional[int] = 42
    score_method: Optional[str] = "logistic"
    matching_method: Optional[str] = "greedy"
    exact_match: Optional[List[str]] = None
    outcome_type: Optional[str] = "binary"
    survival_duration_col: Optional[str] = None
    survival_event_col: Optional[str] = None
    compute_rosenbaum: Optional[bool] = False
    rosenbaum_gamma_max: Optional[float] = 3.0


# ── Internal helpers (kept here because they need _get_df + store access) ─────

def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _run_match_strata(
    df: pd.DataFrame,
    treated_idx: np.ndarray,
    control_idx: np.ndarray,
    distance_vec: np.ndarray,
    caliper_dist: float,
    ratio: int,
    method: str,
    exact_match_cols: Optional[List[str]],
) -> Tuple[list[int], list[int]]:
    """Run matching, partitioning by exact-match strata when requested."""
    match = _match_optimal if method == "optimal" else _match_greedy
    if not exact_match_cols:
        if method == "optimal" and ratio > 1:
            return _match_greedy(treated_idx, control_idx, distance_vec, caliper_dist, ratio)
        if method == "optimal":
            return match(treated_idx, control_idx, distance_vec, caliper_dist)
        return match(treated_idx, control_idx, distance_vec, caliper_dist, ratio)

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
            mt, mc = _match_greedy(t_sub, c_sub, distance_vec, caliper_dist, ratio)
        matched_t.extend(mt)
        matched_c.extend(mc)
    return matched_t, matched_c


# ── Public endpoint ────────────────────────────────────────────────────────────

@router.post("")
def propensity_score_matching(req: PSMRequest):
    """Main PSM endpoint (mounted at /api/models/psm)."""
    try:
        return _run_psm(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _run_psm(req: PSMRequest):
    """Core PSM implementation (moved from models.py)."""
    df_full = _get_df(req.session_id)
    outcome_type = (req.outcome_type or "binary").lower()
    if outcome_type not in ("binary", "survival"):
        raise HTTPException(status_code=422, detail="outcome_type must be 'binary' or 'survival'.")

    extra_outcome_cols: List[str] = []
    if outcome_type == "binary" and req.outcome_col:
        extra_outcome_cols.append(req.outcome_col)
    if outcome_type == "survival":
        if not req.survival_duration_col or not req.survival_event_col:
            raise HTTPException(
                status_code=422,
                detail="Survival outcome requires survival_duration_col and survival_event_col.",
            )
        extra_outcome_cols.extend([req.survival_duration_col, req.survival_event_col])

    exact_match_cols = list(req.exact_match or [])
    needed = list(
        dict.fromkeys(
            [req.treatment_col] + req.covariates + extra_outcome_cols + exact_match_cols
        )
    )
    missing_cols = [c for c in needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing_cols}")

    df_imputed_temp = apply_imputation(df_full[needed], needed, req.imputation or "listwise")
    df_full_imputed = df_full.loc[df_imputed_temp.index].copy().reset_index(drop=True)
    df = df_imputed_temp.reset_index(drop=True)

    treat_vals = df[req.treatment_col].astype(float)
    if not set(treat_vals.unique().tolist()) <= {0, 1, 0.0, 1.0}:
        raise HTTPException(
            status_code=422,
            detail=f"Treatment column '{req.treatment_col}' must be binary (0 = control, 1 = treated).",
        )

    X = pd.get_dummies(df[req.covariates], drop_first=True).astype(float)
    y = treat_vals.astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    score_method = (req.score_method or "logistic").lower()
    ps = _fit_propensity_scores(X_scaled, y, score_method, req.random_state)
    ps_clip = np.clip(ps, 1e-6, 1 - 1e-6)
    logit_ps = np.log(ps_clip / (1.0 - ps_clip))

    df = df.copy()
    df["_ps_"] = ps
    df["_logit_ps_"] = logit_ps
    df["_treat_"] = y

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
        raise HTTPException(
            status_code=422,
            detail="No units remain after common-support trim. Disable trimming or widen the support.",
        )

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
        raise HTTPException(
            status_code=422,
            detail=f"No matches found within caliper {req.caliper}. "
            "Try widening the caliper or check that treatment groups overlap in covariate space.",
        )

    matched_all_idx = matched_treated + matched_controls
    df_matched = df_full_imputed.iloc[matched_all_idx].copy()
    df_matched["_treat_"] = df["_treat_"].iloc[matched_all_idx].values
    df_matched["_ps_"] = df["_ps_"].iloc[matched_all_idx].values
    df_matched["_logit_ps_"] = df["_logit_ps_"].iloc[matched_all_idx].values

    match_ids = []
    for i, ti in enumerate(matched_treated):
        match_ids.append(i)
    for i in range(len(matched_controls)):
        match_ids.append(i // ratio)
    df_matched["_match_id_"] = match_ids

    smd_before, smd_after = {}, {}
    var_ratio_before, var_ratio_after = {}, {}
    ks_before, ks_after = {}, {}
    treat_mask = df["_treat_"].values
    for cov in req.covariates:
        col = df[cov]
        col_m = df_matched[cov]
        denom = _pooled_sd(col[treat_mask == 1], col[treat_mask == 0])
        smd_before[cov] = round(
            _compute_smd(col[treat_mask == 1], col[treat_mask == 0], denom_sd=denom), 4
        )
        smd_after[cov] = round(
            _compute_smd(
                col_m[df_matched["_treat_"] == 1],
                col_m[df_matched["_treat_"] == 0],
                denom_sd=denom,
            ),
            4,
        )
        var_ratio_before[cov] = _variance_ratio(col[treat_mask == 1], col[treat_mask == 0])
        var_ratio_after[cov] = _variance_ratio(
            col_m[df_matched["_treat_"] == 1], col_m[df_matched["_treat_"] == 0]
        )
        ks_before[cov] = _ks_p(col[treat_mask == 1], col[treat_mask == 0])
        ks_after[cov] = _ks_p(
            col_m[df_matched["_treat_"] == 1], col_m[df_matched["_treat_"] == 0]
        )

    avg_smd_before = float(np.mean(list(smd_before.values())))
    avg_smd_after = float(np.mean(list(smd_after.values())))
    reduction_pct = (
        float((avg_smd_before - avg_smd_after) / avg_smd_before * 100)
        if avg_smd_before > 0
        else 0.0
    )

    n_all_treated = int((y == 1).sum())
    n_all_control = int((y == 0).sum())
    n_unmatched = n_all_treated - n_matched_treated

    var_ratios_ok = all(
        (v is None) or (0.5 <= v <= 2.0) for v in var_ratio_after.values()
    )
    balance_achieved = bool(all(v < 0.10 for v in smd_after.values()) and var_ratios_ok)

    ps_dist = {
        "treated_unmatched": ps[treated_idx].tolist(),
        "control_unmatched": ps[control_idx].tolist(),
        "treated_matched": ps[matched_treated].tolist(),
        "control_matched": ps[matched_controls].tolist(),
    }

    outcome_result = None
    rosenbaum_result = None

    if outcome_type == "survival":
        try:
            from lifelines import CoxPHFitter

            dur = pd.to_numeric(df_matched[req.survival_duration_col], errors="coerce")
            evt = pd.to_numeric(df_matched[req.survival_event_col], errors="coerce")
            if np.any(dur.dropna() < 0):
                outcome_result = {
                    "error": f"survival_duration_col '{req.survival_duration_col}' must be ≥ 0."
                }
            elif set(evt.dropna().unique().tolist()) - {0.0, 1.0}:
                outcome_result = {
                    "error": f"survival_event_col '{req.survival_event_col}' must be binary 0/1."
                }
            else:
                cox_df = pd.DataFrame(
                    {
                        "_dur_": dur.values.astype(float),
                        "_evt_": evt.values.astype(int),
                        req.treatment_col: pd.to_numeric(
                            df_matched[req.treatment_col], errors="coerce"
                        )
                        .astype(float)
                        .values,
                        "_match_id_": df_matched["_match_id_"].values.astype(int),
                    }
                ).dropna()
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col="_dur_", event_col="_evt_", strata=["_match_id_"])
                coef = float(cph.params_.iloc[0])
                se = float(cph.standard_errors_.iloc[0])
                ci = cph.confidence_intervals_.iloc[0]
                try:
                    p_val = float(cph.summary["p"].iloc[0])
                except Exception:
                    p_val = None
                outcome_result = {
                    "type": "stratified_cox",
                    "model": "Cox PH stratified by matched set",
                    "n": int(len(cox_df)),
                    "n_events": int(cox_df["_evt_"].sum()),
                    "concordance": round(float(cph.concordance_index_), 4),
                    "coefficients": [
                        {
                            "variable": req.treatment_col,
                            "estimate": round(coef, 6),
                            "hr": round(float(np.exp(coef)), 4),
                            "se": round(se, 6),
                            "z": round(coef / se, 4) if se > 0 else None,
                            "p": round(p_val, 6) if p_val is not None else None,
                            "ci_low": round(float(ci.iloc[0]), 4),
                            "ci_high": round(float(ci.iloc[1]), 4),
                            "hr_low": round(float(np.exp(ci.iloc[0])), 4),
                            "hr_high": round(float(np.exp(ci.iloc[1])), 4),
                        }
                    ],
                }
        except Exception as ex:
            outcome_result = {"error": f"Stratified Cox failed: {ex}"}

    elif req.outcome_col and req.outcome_col in df_matched.columns:
        try:
            y_out = pd.to_numeric(df_matched[req.outcome_col], errors="coerce")
            out_vals = set(y_out.dropna().unique().tolist())

            if not out_vals <= {0, 1, 0.0, 1.0}:
                outcome_result = {
                    "error": f"Outcome must be binary 0/1 for matched analysis. Found: {sorted(out_vals)[:10]}"
                }
            else:
                from statsmodels.discrete.conditional_models import ConditionalLogit

                df_out = df_matched[[req.treatment_col, req.outcome_col, "_match_id_"]].copy()
                df_out[req.outcome_col] = y_out.astype(int)
                df_out[req.treatment_col] = pd.to_numeric(
                    df_out[req.treatment_col], errors="coerce"
                ).astype(float)
                df_out = df_out.dropna()

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
                        mod_cl = ConditionalLogit(y_arr_out, X_out, groups=grp_arr)
                        res_cl = mod_cl.fit(disp=False)
                        coef = float(res_cl.params[0])
                        se = float(res_cl.bse[0])
                        p_val = float(res_cl.pvalues[0])
                        ci_lo = coef - 1.959963984540054 * se
                        ci_hi = coef + 1.959963984540054 * se
                        coefs_out = [
                            {
                                "variable": req.treatment_col,
                                "estimate": round(coef, 6),
                                "or": round(float(np.exp(coef)), 4),
                                "se": round(se, 6),
                                "z": round(coef / se, 4) if se > 0 else None,
                                "p": round(p_val, 6),
                                "ci_low": round(ci_lo, 4),
                                "ci_high": round(ci_hi, 4),
                                "or_low": round(float(np.exp(ci_lo)), 4),
                                "or_high": round(float(np.exp(ci_hi)), 4),
                            }
                        ]
                        outcome_result = {
                            "type": "conditional_logistic",
                            "model": "Conditional logistic regression (matched-set stratification)",
                            "n": int(len(df_clogit)),
                            "n_matched_sets": int(df_out["_match_id_"].nunique()),
                            "n_informative_sets": n_informative_pairs,
                            "n_uninformative_sets": int(df_out["_match_id_"].nunique())
                            - n_informative_pairs,
                            "coefficients": coefs_out,
                            "log_likelihood": round(float(res_cl.llf), 4),
                            "method_note": (
                                "Conditional likelihood treats each matched set as a stratum. "
                                "Uninformative (concordant) sets contribute 0 to the likelihood and are dropped. "
                                "For 1:1 matching with treatment as the only covariate this is equivalent to McNemar's test."
                            ),
                        }
                    except Exception as cl_exc:
                        X_out = sm.add_constant(df_out[[req.treatment_col]].astype(float))
                        m_out = sm.Logit(
                            df_out[req.outcome_col].astype(int).values, X_out
                        ).fit(disp=False, cov_type="HC1")
                        ci_out = m_out.conf_int()
                        coefs_out = []
                        for var in m_out.params.index:
                            est = float(m_out.params[var])
                            coefs_out.append(
                                {
                                    "variable": str(var),
                                    "estimate": round(est, 6),
                                    "or": round(float(np.exp(est)), 4),
                                    "se": round(float(m_out.bse[var]), 6),
                                    "z": round(float(m_out.tvalues[var]), 4),
                                    "p": round(float(m_out.pvalues[var]), 6),
                                    "ci_low": round(float(ci_out.loc[var, 0]), 4),
                                    "ci_high": round(float(ci_out.loc[var, 1]), 4),
                                    "or_low": round(float(np.exp(ci_out.loc[var, 0])), 4),
                                    "or_high": round(float(np.exp(ci_out.loc[var, 1])), 4),
                                }
                            )
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

    if (
        req.compute_rosenbaum
        and outcome_type == "binary"
        and ratio == 1
        and req.outcome_col
        and req.outcome_col in df_matched.columns
    ):
        try:
            y_out = pd.to_numeric(df_matched[req.outcome_col], errors="coerce")
            out_vals = set(y_out.dropna().unique().tolist())
            if not out_vals <= {0, 1, 0.0, 1.0}:
                rosenbaum_result = {
                    "applicable": False,
                    "reason": "Rosenbaum bounds require binary 0/1 outcome.",
                }
            else:
                pair_pairs: list[tuple[int, int]] = []
                df_rb = df_matched.copy()
                df_rb[req.outcome_col] = y_out.astype(int)
                df_rb[req.treatment_col] = pd.to_numeric(
                    df_rb[req.treatment_col], errors="coerce"
                ).astype(int)
                for mid, grp in df_rb.groupby("_match_id_"):
                    t_rows = grp[grp[req.treatment_col] == 1]
                    c_rows = grp[grp[req.treatment_col] == 0]
                    if len(t_rows) == 1 and len(c_rows) == 1:
                        pair_pairs.append(
                            (
                                int(t_rows[req.outcome_col].iloc[0]),
                                int(c_rows[req.outcome_col].iloc[0]),
                            )
                        )
                if not pair_pairs:
                    rosenbaum_result = {
                        "applicable": False,
                        "reason": "No clean 1:1 matched pairs available.",
                    }
                else:
                    rosenbaum_result = _rosenbaum_bounds(
                        pair_pairs,
                        gamma_max=float(req.rosenbaum_gamma_max or 3.0),
                    )
        except Exception as ex:
            rosenbaum_result = {"applicable": False, "reason": f"Rosenbaum bounds failed: {ex}"}

    df_export = df_matched.drop(columns=["_ps_", "_logit_ps_", "_treat_"], errors="ignore")
    df_export = df_export.rename(columns={"_match_id_": "match_set_id"})
    store.save(req.session_id + "_psm", df_export)

    try:
        parent_metadata = store.get_metadata(req.session_id)
        if parent_metadata:
            store.save_metadata(req.session_id + "_psm", parent_metadata)

        parent_kinds = store.get_kind_overrides(req.session_id)
        if parent_kinds:
            kinds = {**parent_kinds, "match_set_id": "categorical"}
            store.set_kind_overrides(req.session_id + "_psm", kinds)

        parent_decimals = store.get_decimals(req.session_id)
        if parent_decimals:
            store.save_decimals(req.session_id + "_psm", parent_decimals)
    except Exception:
        pass

    return {
        "n_total": int(len(df)),
        "n_treated": n_all_treated,
        "n_control": n_all_control,
        "n_matched_pairs": n_matched_treated,
        "n_matched_controls": n_matched_controls,
        "n_unmatched": n_unmatched,
        "n_trimmed_common_support": n_trimmed,
        "score_method": score_method,
        "matching_method": matching_method,
        "matching_warning": matching_warning,
        "exact_match": exact_match_cols,
        "outcome_type": outcome_type,
        "caliper_scale": scale,
        "caliper_used": round(float(caliper_dist), 6),
        "caliper_sd": round(caliper_sd, 6),
        "common_support": {"lo": round(support_lo, 6), "hi": round(support_hi, 6)},
        "balance_achieved": balance_achieved,
        "avg_smd_before": round(avg_smd_before, 4),
        "avg_smd_after": round(avg_smd_after, 4),
        "reduction_pct": round(reduction_pct, 1),
        "smd_before": smd_before,
        "smd_after": smd_after,
        "variance_ratio_before": var_ratio_before,
        "variance_ratio_after": var_ratio_after,
        "ks_p_before": ks_before,
        "ks_p_after": ks_after,
        "ps_distribution": ps_dist,
        "outcome_result": outcome_result,
        "rosenbaum": rosenbaum_result,
        "matched_session_id": req.session_id + "_psm",
    }
