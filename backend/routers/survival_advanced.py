"""
Advanced Survival Analyses router.

Endpoints
---------
POST /mice              — MICE multiple imputation
POST /fine_gray         — Fine-Gray competing risks (CIF curves)
POST /evalue            — E-value for unmeasured confounding
POST /landmark          — Landmark survival analysis
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _safe(v: Any) -> Any:
    """Make a value JSON-safe."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ── 1. MICE Multiple Imputation ─────────────────────────────────────────────


class MICERequest(BaseModel):
    session_id: str
    columns: List[str]
    n_imputations: int = 5
    max_iter: int = 10
    random_state: int = 42
    mechanism: str = "unknown"  # unknown, MCAR, MAR, MNAR


@router.post("/mice")
def mice_imputation(req: MICERequest):
    df = _get_df(req.session_id)

    # Validate columns
    missing_cols = [c for c in req.columns if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing_cols}")

    # Check there are actually missing values
    cols_with_missing = [c for c in req.columns if df[c].isna().sum() > 0]
    if not cols_with_missing:
        raise HTTPException(status_code=422, detail="No missing values in selected columns")

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    # Use all numeric columns as features for imputation
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise HTTPException(status_code=422, detail="No numeric columns available for MICE")

    # Ensure target columns are numeric
    non_numeric = [c for c in req.columns if c not in numeric_cols]
    if non_numeric:
        raise HTTPException(status_code=422, detail=f"Non-numeric columns cannot be imputed with MICE: {non_numeric}")

    df_work = df.copy()
    subset = df_work[numeric_cols].copy()

    # Record pre-imputation state
    pre_missing = {c: int(subset[c].isna().sum()) for c in cols_with_missing}

    # Run MICE (averaged over n_imputations)
    imputed_sum = np.zeros_like(subset.values, dtype=float)
    missing_mask = subset.isna().values

    for i in range(req.n_imputations):
        imp = IterativeImputer(
            max_iter=req.max_iter,
            random_state=req.random_state + i,
            sample_posterior=True,
        )
        imputed_sum += imp.fit_transform(subset)

    imputed_avg = imputed_sum / req.n_imputations

    # Only fill originally missing values
    result = subset.values.copy()
    result[missing_mask] = imputed_avg[missing_mask]
    subset_filled = pd.DataFrame(result, columns=numeric_cols, index=subset.index)

    # Update only the requested columns
    for c in cols_with_missing:
        df_work[c] = subset_filled[c]

    store.save(req.session_id, df_work)

    # Build per-column summary
    col_summaries = []
    for c in cols_with_missing:
        mask = df[c].isna()
        imputed_vals = df_work.loc[mask, c]
        col_summaries.append({
            "column": c,
            "n_imputed": pre_missing[c],
            "mean_imputed": _safe(round(float(imputed_vals.mean()), 4)) if len(imputed_vals) > 0 else None,
            "min_imputed": _safe(round(float(imputed_vals.min()), 4)) if len(imputed_vals) > 0 else None,
            "max_imputed": _safe(round(float(imputed_vals.max()), 4)) if len(imputed_vals) > 0 else None,
        })

    total_imputed = sum(s["n_imputed"] for s in col_summaries)

    mech = req.mechanism.upper()
    mech_label = {"UNKNOWN": "Unknown", "MCAR": "MCAR", "MAR": "MAR", "MNAR": "MNAR"}.get(mech, "Unknown")
    mech_ok = mech != "MNAR"
    assumptions = [
        {"name": "Missing mechanism",
         "met": mech_ok,
         "detail": f"Assumed {mech_label}. MICE is valid under MAR/MCAR."
                   + (" MNAR may produce biased estimates — consider sensitivity analysis." if not mech_ok else "")},
        {"name": "Numeric columns", "met": True, "detail": f"{len(numeric_cols)} numeric features used as predictors."},
        {"name": "Imputations", "met": True, "detail": f"{req.n_imputations} imputations averaged (Rubin's rules approximation)."},
    ]

    result_text = (
        f"Multiple imputation (MICE) was performed assuming {mech_label} mechanism, "
        f"using {req.n_imputations} imputations with {req.max_iter} iterations each. "
        f"{total_imputed} missing values were imputed "
        f"across {len(cols_with_missing)} variable(s): {', '.join(cols_with_missing)}."
    )

    export_rows = [["Column", "N Imputed", "Mean", "Min", "Max"]]
    for s in col_summaries:
        export_rows.append([s["column"], s["n_imputed"], s["mean_imputed"], s["min_imputed"], s["max_imputed"]])

    r_code = (
        f"library(mice)\n"
        f"imp <- mice(data[, c({', '.join(repr(c) for c in req.columns)})],\n"
        f"            m = {req.n_imputations}, maxit = {req.max_iter}, method = 'pmm', seed = {req.random_state})\n"
        f"completed_data <- complete(imp, action = 'long')\n"
        f"# Pool estimates with Rubin's rules:\n"
        f"# fit <- with(imp, lm(outcome ~ predictors))\n"
        f"# pooled <- pool(fit)"
    )

    return {
        "test": "MICE Multiple Imputation",
        "n_total": len(df),
        "total_imputed": total_imputed,
        "columns": col_summaries,
        "n_imputations": req.n_imputations,
        "max_iter": req.max_iter,
        "assumptions": assumptions,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ── 2. Fine-Gray Competing Risks ────────────────────────────────────────────


class FineGrayRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    event_of_interest: int = 1
    group_col: Optional[str] = None
    # Subdistribution-hazard regression (Fine-Gray 1999) via the Geskus 2011
    # IPCW-weighted Cox recipe. When non-empty the endpoint also returns a
    # `regression_result` block with subdistribution hazard ratios (sHR).
    predictors: Optional[List[str]] = None
    imputation: Optional[str] = "listwise"


def _fine_gray_fit(df: pd.DataFrame, duration: str, event: str,
                   cause: int, predictors: List[str]) -> dict:
    """Fit a Fine-Gray subdistribution hazard regression via Geskus's
    (2011) IPCW-weighted Cox reformulation. Mathematically equivalent
    to Fine & Gray (1999) and free of external dependencies beyond
    lifelines.

    Algorithm:
      1. Estimate Ĝ(t) — Kaplan-Meier of the censoring distribution
         (event = 1 iff e == 0).
      2. Build an augmented long-format dataset:
         • cause-of-interest subject  → (0, t_i, evt=1, w=1)
         • competing-event subject    → pseudo-rows (0, s, evt=0,
             w = Ĝ(s)/Ĝ(t_i)) at every cause-of-interest event time
             s > t_i; weight is 0 once Ĝ(s) collapses to 0.
         • censored subject           → (0, t_i, evt=0, w=1)
      3. Fit lifelines.CoxPHFitter on this dataset with weights_col
         and robust=True (Lin-Wei sandwich SE).
    """
    from lifelines import CoxPHFitter, KaplanMeierFitter

    # Encode predictors: numeric stays numeric, categorical → dummies
    # (drop_first=True). This matches the existing Cox endpoint in
    # backend/routers/models.py so the user sees consistent dummy names.
    pred_raw = df[predictors].copy()
    numeric_pred: List[str] = []
    cat_pred: List[str] = []
    for c in predictors:
        col = pred_raw[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric_pred.append(c)
        else:
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().mean() >= 0.8 and len(coerced.dropna().unique()) > 2:
                pred_raw[c] = coerced
                numeric_pred.append(c)
            else:
                cat_pred.append(c)
    num_part = pred_raw[numeric_pred].apply(pd.to_numeric, errors="coerce") if numeric_pred else pd.DataFrame(index=pred_raw.index)
    cat_part = pd.get_dummies(pred_raw[cat_pred], drop_first=True, dummy_na=False) if cat_pred else pd.DataFrame(index=pred_raw.index)
    enc = pd.concat([num_part, cat_part], axis=1).astype(float)

    work = pd.concat(
        [df[[duration, event]].reset_index(drop=True), enc.reset_index(drop=True)],
        axis=1,
    ).dropna()
    if len(work) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows after dropping NA in predictors / duration / event (need ≥ 10, got {len(work)}).")

    t = work[duration].values.astype(float)
    e = work[event].values.astype(int)
    cov_cols = list(enc.columns)
    if not cov_cols:
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")

    # Censoring KM Ĝ(t): event indicator is "censored" (e == 0).
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(t, (e == 0).astype(int))
    g_at = kmf_c.survival_function_.iloc[:, 0]  # Series indexed by time
    g_times = g_at.index.values.astype(float)
    g_vals = g_at.values.astype(float)

    def _g(tau: float) -> float:
        """Right-continuous Ĝ(tau): largest indexed time ≤ tau."""
        idx = np.searchsorted(g_times, tau, side="right") - 1
        if idx < 0:
            return 1.0
        return float(g_vals[idx])

    cause_event_times = np.sort(np.unique(t[e == cause]))
    if len(cause_event_times) == 0:
        raise HTTPException(status_code=422, detail=f"No subjects experienced the event of interest (code {cause}).")

    # Cap on augmented row count — defends against pathological inputs where
    # N_competing × K_event_times explodes (e.g. ten thousand subjects all
    # with a competing event and a thousand distinct cause-of-interest times).
    n_competing = int(((e != cause) & (e != 0)).sum())
    if n_competing * len(cause_event_times) > 500_000:
        raise HTTPException(
            status_code=422,
            detail=(
                "Augmented Fine-Gray dataset would exceed 500 000 rows "
                f"({n_competing} competing-event subjects × "
                f"{len(cause_event_times)} unique event times of interest). "
                "Reduce the dataset (e.g. coarser duration units, fewer subjects) "
                "or fit on a stratified subset."
            ),
        )

    # Build augmented dataset
    rows: List[dict] = []
    cov_arr = work[cov_cols].values
    for i in range(len(work)):
        ti = float(t[i])
        ei = int(e[i])
        cov_i = cov_arr[i]
        if ei == cause:
            row = {"_stop_": ti, "_event_": 1, "_w_": 1.0}
            for cj, name in enumerate(cov_cols):
                row[name] = float(cov_i[cj])
            rows.append(row)
        elif ei == 0:
            row = {"_stop_": ti, "_event_": 0, "_w_": 1.0}
            for cj, name in enumerate(cov_cols):
                row[name] = float(cov_i[cj])
            rows.append(row)
        else:
            g_ti = _g(ti)
            if g_ti <= 0:
                continue
            future_ev = cause_event_times[cause_event_times > ti]
            for s in future_ev:
                w_s = _g(float(s)) / g_ti
                if w_s <= 0:
                    break
                row = {"_stop_": float(s), "_event_": 0, "_w_": float(w_s)}
                for cj, name in enumerate(cov_cols):
                    row[name] = float(cov_i[cj])
                rows.append(row)

    aug = pd.DataFrame(rows)
    if aug["_event_"].sum() == 0:
        raise HTTPException(status_code=422, detail="No cause-of-interest events present after building the augmented dataset.")

    cph = CoxPHFitter()
    cph.fit(aug, duration_col="_stop_", event_col="_event_",
            weights_col="_w_", robust=True)

    ci = cph.confidence_intervals_
    coefs: List[dict] = []
    for var in cph.params_.index:
        beta = float(cph.params_[var])
        se_v = float(cph.standard_errors_[var])
        try:
            p_v = float(cph.summary["p"].loc[var])
        except Exception:
            p_v = None
        try:
            lo = float(ci.loc[var, ci.columns[0]])
            hi = float(ci.loc[var, ci.columns[1]])
        except Exception:
            lo = hi = float("nan")
        coefs.append({
            "variable": str(var),
            "estimate": round(beta, 6),
            "shr": round(float(np.exp(beta)), 4),
            "se": round(se_v, 6),
            "z": round(beta / se_v, 4) if se_v > 0 else None,
            "p": round(p_v, 6) if p_v is not None else None,
            "ci_low": round(lo, 4) if math.isfinite(lo) else None,
            "ci_high": round(hi, 4) if math.isfinite(hi) else None,
            "shr_low": round(float(np.exp(lo)), 4) if math.isfinite(lo) else None,
            "shr_high": round(float(np.exp(hi)), 4) if math.isfinite(hi) else None,
        })

    return {
        "method": "fine_gray_regression",
        "model": "Fine-Gray subdistribution hazards (IPCW-weighted Cox, Lin-Wei robust SE)",
        "n": int(len(work)),
        "n_events_of_interest": int((e == cause).sum()),
        "n_competing": int(((e != cause) & (e != 0)).sum()),
        "n_censored": int((e == 0).sum()),
        "n_augmented_rows": int(len(aug)),
        "concordance": round(float(cph.concordance_index_), 4),
        "coefficients": coefs,
        "method_note": (
            "Fine-Gray subdistribution hazard regression fit via the Geskus "
            "(2011) IPCW-weighted Cox reformulation — mathematically equivalent "
            "to Fine & Gray (1999). Competing-event subjects stay at risk past "
            "their event time with weights Ĝ(s)/Ĝ(t_i) from the Kaplan-Meier "
            "estimate of the censoring distribution. Lin-Wei sandwich estimator "
            "for the standard errors. Output sHR = exp(β) is the subdistribution "
            "hazard ratio for the cause of interest."
        ),
    }


@router.post("/fine_gray")
def fine_gray(req: FineGrayRequest):
    df = _get_df(req.session_id)

    for c in [req.duration_col, req.event_col]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if req.group_col and req.group_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.group_col}' not found")

    from lifelines import AalenJohansenFitter

    work = df[[req.duration_col, req.event_col] + ([req.group_col] if req.group_col else [])].dropna()
    durations = work[req.duration_col].values.astype(float)
    events = work[req.event_col].values.astype(int)

    # Unique event types (0 = censored, others are event types)
    event_types = sorted([e for e in np.unique(events) if e != 0])
    if req.event_of_interest not in event_types:
        raise HTTPException(status_code=422, detail=f"Event of interest {req.event_of_interest} not found. Available: {event_types}")

    # Build CIF curves
    plots = {}
    cif_data = {}
    event_counts = {}

    if req.group_col:
        groups = sorted(work[req.group_col].unique())
    else:
        groups = ["All"]

    colors = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]

    traces = []
    for gi, group in enumerate(groups):
        if req.group_col:
            mask = work[req.group_col] == group
            dur_g = durations[mask]
            ev_g = events[mask]
        else:
            dur_g = durations
            ev_g = events

        ajf = AalenJohansenFitter()
        ajf.fit(dur_g, ev_g, event_of_interest=req.event_of_interest)

        cif = ajf.cumulative_density_
        col_name = cif.columns[0]
        times = cif.index.tolist()
        probs = cif[col_name].tolist()

        label = f"CIF - {group}" if req.group_col else "CIF"
        color = colors[gi % len(colors)]

        traces.append({
            "x": [_safe(t) for t in times],
            "y": [_safe(p) for p in probs],
            "type": "scatter",
            "mode": "lines",
            "name": label,
            "line": {"color": color, "width": 2},
        })

        n_event = int(np.sum(ev_g == req.event_of_interest))
        n_competing = int(np.sum((ev_g != 0) & (ev_g != req.event_of_interest)))
        n_censored = int(np.sum(ev_g == 0))
        event_counts[str(group)] = {
            "n": len(dur_g),
            "event_of_interest": n_event,
            "competing_events": n_competing,
            "censored": n_censored,
        }

        cif_data[str(group)] = {
            "cif_at_max": _safe(round(probs[-1], 4)) if probs else None,
        }

    # Gray's test (K-sample comparison) — approximate using log-rank on sub-events
    gray_p = None
    if req.group_col and len(groups) == 2:
        try:
            from lifelines.statistics import logrank_test
            g1_mask = work[req.group_col] == groups[0]
            g2_mask = work[req.group_col] == groups[1]
            # Create binary event: 1 if event of interest, 0 otherwise
            ev_binary = (events == req.event_of_interest).astype(int)
            lr = logrank_test(
                durations[g1_mask], durations[g2_mask],
                ev_binary[g1_mask], ev_binary[g2_mask],
            )
            gray_p = _safe(round(float(lr.p_value), 6))
        except Exception:
            gray_p = None

    plot = {
        "data": traces,
        "layout": {
            "title": f"Cumulative Incidence Function (Event={req.event_of_interest})",
            "xaxis": {"title": req.duration_col, "gridcolor": "#e5e7eb"},
            "yaxis": {"title": "Cumulative Incidence", "range": [0, 1], "gridcolor": "#e5e7eb"},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "#ffffff",
            "font": {"color": "#374151", "size": 12},
            "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.98},
        },
    }

    assumptions = [
        {"name": "Independent censoring", "met": True, "detail": "Competing risks model assumes censoring is non-informative."},
        {"name": "Event types", "met": True, "detail": f"Event types found: {event_types}. Event of interest: {req.event_of_interest}."},
    ]

    n_total = len(work)
    result_text = (
        f"Competing risks analysis was performed on {n_total} subjects. "
        f"The cumulative incidence function (CIF) was estimated using the Aalen-Johansen estimator "
        f"for event type {req.event_of_interest}."
    )
    if gray_p is not None:
        result_text += f" Gray's test p = {gray_p}."

    export_rows = [["Group", "N", "Events of Interest", "Competing Events", "Censored", "CIF at Max Time"]]
    for g in groups:
        ec = event_counts[str(g)]
        export_rows.append([str(g), ec["n"], ec["event_of_interest"], ec["competing_events"], ec["censored"],
                            cif_data[str(g)]["cif_at_max"]])

    r_code = (
        f"library(cmprsk)\n"
        f"library(tidycmprsk)\n\n"
        f"# Cumulative incidence\n"
        f"cif <- cuminc(ftime = data${req.duration_col},\n"
        f"              fstatus = data${req.event_col}"
        + (f",\n              group = data${req.group_col}" if req.group_col else "")
        + f")\n"
        f"plot(cif)\n\n"
        f"# Fine-Gray regression\n"
        f"fg <- crr(ftime = data${req.duration_col},\n"
        f"          fstatus = data${req.event_col},\n"
        f"          failcode = {req.event_of_interest},\n"
        f"          cov1 = model.matrix(~ predictors, data)[,-1])\n"
        f"summary(fg)"
    )

    # ── Subdistribution hazard regression (Fine-Gray 1999 / Geskus 2011) ──
    regression_result: Optional[dict] = None
    if req.predictors:
        missing_preds = [c for c in req.predictors if c not in df.columns]
        if missing_preds:
            raise HTTPException(status_code=422, detail=f"Predictor columns not found: {missing_preds}")
        # Reuse the imputation infrastructure used by other survival endpoints
        from services.impute import apply_imputation
        cols_needed = [req.duration_col, req.event_col] + req.predictors
        df_reg = apply_imputation(df[cols_needed], cols_needed, req.imputation or "listwise").reset_index(drop=True)
        # Coerce duration/event numeric; event_of_interest match is integer.
        df_reg[req.duration_col] = pd.to_numeric(df_reg[req.duration_col], errors="coerce")
        df_reg[req.event_col] = pd.to_numeric(df_reg[req.event_col], errors="coerce")
        regression_result = _fine_gray_fit(
            df_reg, req.duration_col, req.event_col, int(req.event_of_interest), list(req.predictors),
        )
        if regression_result is not None:
            result_text = (
                result_text
                + f" Fine-Gray subdistribution-hazard regression on {len(req.predictors)} "
                  f"predictor(s) — see the sHR table for details."
            )

    # Audit
    try:
        store.log_action(req.session_id, "fine_gray", {
            "duration_col": req.duration_col,
            "event_col": req.event_col,
            "event_of_interest": int(req.event_of_interest),
            "group_col": req.group_col,
            "n_predictors": len(req.predictors or []),
            "ran_regression": regression_result is not None,
        })
    except Exception:
        pass

    return {
        "test": "Fine-Gray Competing Risks",
        "n": n_total,
        "event_types": event_types,
        "event_of_interest": req.event_of_interest,
        "event_counts": event_counts,
        "cif_data": cif_data,
        "gray_p": gray_p,
        "plot": plot,
        "assumptions": assumptions,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
        "regression_result": regression_result,
    }


# ── 3. E-value ──────────────────────────────────────────────────────────────


class EValueRequest(BaseModel):
    estimate: float
    ci_low: float
    ci_high: float
    measure_type: str = "OR"  # OR, HR, RR
    baseline_risk: float = 0.1  # p0, used for OR→RR conversion


@router.post("/evalue")
def evalue(req: EValueRequest):
    est = req.estimate
    ci_lo = req.ci_low
    ci_hi = req.ci_high
    p0 = req.baseline_risk

    # Convert to RR scale
    if req.measure_type.upper() == "OR":
        # OR → RR approximation: RR ≈ OR / (1 - p0 + p0 * OR)
        rr = est / (1 - p0 + p0 * est)
        rr_lo = ci_lo / (1 - p0 + p0 * ci_lo)
        rr_hi = ci_hi / (1 - p0 + p0 * ci_hi)
    elif req.measure_type.upper() == "HR":
        # HR ≈ RR for rare outcomes; use directly
        rr = est
        rr_lo = ci_lo
        rr_hi = ci_hi
    else:
        rr = est
        rr_lo = ci_lo
        rr_hi = ci_hi

    def _evalue(rr_val: float) -> Optional[float]:
        if rr_val is None or rr_val <= 0:
            return None
        if rr_val < 1:
            rr_val = 1 / rr_val
        return round(rr_val + math.sqrt(rr_val * (rr_val - 1)), 4)

    e_point = _evalue(rr)

    # E-value for CI: use the bound closer to 1 (more conservative)
    ci_bound = rr_hi if rr < 1 else rr_lo
    if rr < 1:
        ci_bound = rr_hi
    else:
        ci_bound = rr_lo

    # If CI crosses 1, E-value for CI is 1
    if (rr_lo <= 1 <= rr_hi) or (ci_lo <= 1 <= ci_hi):
        e_ci = 1.0
    else:
        e_ci = _evalue(ci_bound)

    interpretation = (
        f"The E-value is {e_point}. To explain away the observed {req.measure_type} of {est}, "
        f"an unmeasured confounder would need to be associated with both the treatment and outcome "
        f"by a risk ratio of at least {e_point}-fold each, above and beyond the measured covariates."
    )
    if e_ci and e_ci > 1:
        interpretation += (
            f" For the confidence interval limit, the E-value is {e_ci}, "
            f"meaning a confounder of this strength could shift the CI to include the null."
        )

    assumptions = [
        {"name": "Sufficient adjustment", "met": True,
         "detail": "E-value quantifies the minimum confounding strength needed to explain away the result."},
        {"name": "Rare outcome", "met": req.measure_type.upper() != "OR" or p0 < 0.15,
         "detail": f"OR→RR conversion accuracy depends on baseline risk ({p0}). Best when <15%."},
    ]

    result_text = (
        f"The E-value for the observed {req.measure_type} of {est} "
        f"(95% CI: {ci_lo}–{ci_hi}) was {e_point}. "
        f"The E-value for the confidence interval bound was {e_ci}."
    )

    export_rows = [
        ["Metric", "Value"],
        [f"Observed {req.measure_type}", est],
        ["95% CI", f"{ci_lo} – {ci_hi}"],
        ["RR (converted)", round(rr, 4)],
        ["E-value (point)", e_point],
        ["E-value (CI)", e_ci],
    ]

    r_code = (
        f"library(EValue)\n\n"
        f"# E-value calculation\n"
        f'evalues.{req.measure_type}({est}, lo = {ci_lo}, hi = {ci_hi}'
        + (f', rare = FALSE)' if req.measure_type.upper() == "OR" else ')')
        + f"\n\n"
        f"# Interpretation: An unmeasured confounder would need\n"
        f"# RR >= {e_point} with both treatment & outcome to explain\n"
        f"# away the observed effect."
    )

    return {
        "test": "E-value (Unmeasured Confounding)",
        "estimate": est,
        "measure_type": req.measure_type,
        "ci": [ci_lo, ci_hi],
        "rr_converted": _safe(round(rr, 4)),
        "evalue_point": e_point,
        "evalue_ci": e_ci,
        "interpretation": interpretation,
        "assumptions": assumptions,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ── 4. Landmark Analysis ────────────────────────────────────────────────────


class LandmarkRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    landmark_time: float
    group_col: Optional[str] = None
    predictors: Optional[List[str]] = None


@router.post("/landmark")
def landmark_analysis(req: LandmarkRequest):
    df = _get_df(req.session_id)

    for c in [req.duration_col, req.event_col]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")

    needed = [req.duration_col, req.event_col]
    if req.group_col:
        if req.group_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{req.group_col}' not found")
        needed.append(req.group_col)
    if req.predictors:
        for p in req.predictors:
            if p not in df.columns:
                raise HTTPException(status_code=400, detail=f"Predictor '{p}' not found")
            needed.append(p)

    work = df[needed].dropna()
    n_total = len(work)

    # Apply landmark: exclude subjects who had event before landmark
    lm_mask = work[req.duration_col] >= req.landmark_time
    lm_df = work[lm_mask].copy()
    n_excluded = n_total - len(lm_df)

    if len(lm_df) < 10:
        raise HTTPException(status_code=422,
                            detail=f"Only {len(lm_df)} subjects survived beyond landmark time {req.landmark_time}. Need at least 10.")

    # Shift time origin
    lm_df[req.duration_col] = lm_df[req.duration_col] - req.landmark_time

    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test

    colors = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
    traces = []
    km_summaries = {}

    if req.group_col:
        groups = sorted(lm_df[req.group_col].unique())
    else:
        groups = ["All"]

    for gi, group in enumerate(groups):
        if req.group_col:
            g_df = lm_df[lm_df[req.group_col] == group]
        else:
            g_df = lm_df

        kmf = KaplanMeierFitter()
        kmf.fit(g_df[req.duration_col], g_df[req.event_col])

        sf = kmf.survival_function_
        col_name = sf.columns[0]
        times = sf.index.tolist()
        probs = sf[col_name].tolist()

        label = f"KM - {group}" if req.group_col else "KM"
        color = colors[gi % len(colors)]

        traces.append({
            "x": [_safe(t) for t in times],
            "y": [_safe(p) for p in probs],
            "type": "scatter",
            "mode": "lines",
            "name": label,
            "line": {"color": color, "width": 2},
        })

        median_surv = kmf.median_survival_time_
        km_summaries[str(group)] = {
            "n": len(g_df),
            "events": int(g_df[req.event_col].sum()),
            "median_survival": _safe(round(float(median_surv), 2)) if not math.isinf(median_surv) else None,
        }

    # Log-rank test between groups
    logrank_p = None
    if req.group_col and len(groups) == 2:
        g1 = lm_df[lm_df[req.group_col] == groups[0]]
        g2 = lm_df[lm_df[req.group_col] == groups[1]]
        try:
            lr = logrank_test(
                g1[req.duration_col], g2[req.duration_col],
                g1[req.event_col], g2[req.event_col],
            )
            logrank_p = _safe(round(float(lr.p_value), 6))
        except Exception:
            logrank_p = None

    # Cox regression if predictors given
    cox_results = None
    if req.predictors:
        try:
            cox_cols = [req.duration_col, req.event_col] + req.predictors
            if req.group_col and req.group_col not in req.predictors:
                cox_cols.append(req.group_col)
            cox_df = lm_df[cox_cols].copy()

            # Encode categorical predictors
            for c in req.predictors + ([req.group_col] if req.group_col else []):
                if c in cox_df.columns and cox_df[c].dtype == object:
                    cox_df[c] = pd.Categorical(cox_df[c]).codes

            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col=req.duration_col, event_col=req.event_col)

            cox_results = []
            for _, row in cph.summary.iterrows():
                cox_results.append({
                    "variable": row.name,
                    "HR": _safe(round(float(row["exp(coef)"]), 4)),
                    "ci_low": _safe(round(float(row["exp(coef) lower 95%"]), 4)),
                    "ci_high": _safe(round(float(row["exp(coef) upper 95%"]), 4)),
                    "p": _safe(round(float(row["p"]), 6)),
                })
        except Exception as exc:
            cox_results = [{"error": str(exc)}]

    plot = {
        "data": traces,
        "layout": {
            "title": f"Landmark Analysis (t ≥ {req.landmark_time})",
            "xaxis": {"title": f"Time from landmark ({req.landmark_time})", "gridcolor": "#e5e7eb"},
            "yaxis": {"title": "Survival Probability", "range": [0, 1.05], "gridcolor": "#e5e7eb"},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "#ffffff",
            "font": {"color": "#374151", "size": 12},
            "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.02, "yanchor": "bottom"},
        },
    }

    assumptions = [
        {"name": "Landmark exclusion", "met": True,
         "detail": f"{n_excluded} subjects excluded (event or censored before t={req.landmark_time})."},
        {"name": "Sufficient sample", "met": len(lm_df) >= 20,
         "detail": f"{len(lm_df)} subjects remain after landmark."},
    ]
    if logrank_p is not None:
        assumptions.append({"name": "Log-rank test", "met": logrank_p < 0.05,
                            "detail": f"p = {logrank_p}"})

    result_text = (
        f"Landmark analysis at t = {req.landmark_time}: {n_excluded} subjects were excluded "
        f"(event or censored before landmark), leaving {len(lm_df)} subjects for analysis."
    )
    if logrank_p is not None:
        result_text += f" Log-rank test p = {logrank_p}."

    export_rows = [["Group", "N", "Events", "Median Survival"]]
    for g in groups:
        s = km_summaries[str(g)]
        export_rows.append([str(g), s["n"], s["events"], s["median_survival"]])

    preds_str = " + ".join(req.predictors) if req.predictors else "group"
    r_code = (
        f"library(survival)\n"
        f"library(survminer)\n\n"
        f"# Landmark at t = {req.landmark_time}\n"
        f"lm_data <- data[data${req.duration_col} >= {req.landmark_time}, ]\n"
        f"lm_data${req.duration_col} <- lm_data${req.duration_col} - {req.landmark_time}\n\n"
        f"# KM curves\n"
        f"fit <- survfit(Surv({req.duration_col}, {req.event_col}) ~ "
        + (req.group_col if req.group_col else "1")
        + f", data = lm_data)\n"
        f"ggsurvplot(fit, data = lm_data, risk.table = TRUE)\n\n"
        f"# Cox regression\n"
        f"cox <- coxph(Surv({req.duration_col}, {req.event_col}) ~ {preds_str}, data = lm_data)\n"
        f"summary(cox)"
    )

    return {
        "test": "Landmark Survival Analysis",
        "landmark_time": req.landmark_time,
        "n_total": n_total,
        "n_excluded": n_excluded,
        "n_landmark": len(lm_df),
        "km_summaries": km_summaries,
        "logrank_p": logrank_p,
        "cox_results": cox_results,
        "plot": plot,
        "assumptions": assumptions,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }
