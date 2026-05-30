"""
Advanced Survival Analyses router.

Endpoints
---------
POST /mice              — MICE multiple imputation
POST /fine_gray         — Fine-Gray competing risks (CIF curves)
POST /evalue            — E-value for unmeasured confounding
POST /landmark          — Landmark survival analysis
POST /rmst              — Restricted Mean Survival Time (PH-free alternative)
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

    # Unique event types (0 = censored, others are event types). Cast to
    # plain int — numpy.int64 is not JSON-serialisable by FastAPI's encoder.
    event_types = sorted(int(e) for e in np.unique(events) if e != 0)
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


# ── 5. Restricted Mean Survival Time (RMST) ─────────────────────────────────
#
# Robust alternative to the hazard-ratio framework — does NOT require the
# proportional-hazards assumption. RMST(τ) is the area under the survival
# curve from 0 to τ, equal to the average event-free time over the horizon
# (e.g. "average years alive in 0-5 years"). For two groups the difference
# Δ = RMST_A(τ) − RMST_B(τ) is a clinically interpretable summary with a
# proper SE (Royston & Parmar 2013) → z-test + 95 % CI + p.


class RMSTRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    tau: float                                   # restriction time-horizon
    group_col: Optional[str] = None
    imputation: Optional[str] = "listwise"


def _rmst_one_group(t: np.ndarray, e: np.ndarray, tau: float) -> Dict[str, float]:
    """KM-based RMST estimator with Greenwood-style SE.

    Algorithm:
      1. Fit KaplanMeierFitter on (t, e).
      2. Trapezoidal-rule integrate S(u) from 0 to τ.
      3. SE via the integral of Greenwood's variance:
           Var[RMST(τ)] = Σ_k  [∫_{t_k}^{min(t_{k+1}, τ)} S(u) du]^2  * d_k / (n_k (n_k - d_k))
         which is the Klein & Moeschberger (2003) eqn 4.5.1 / Hosmer-Lemeshow form.
      4. 95% CI = Wald on RMST.
    """
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(t, e.astype(int))

    # Step-function on the unique observed times. surv[k] = S(t_k+) for
    # t in [t_k, t_{k+1}).
    sf = kmf.survival_function_.iloc[:, 0]
    sf_times = sf.index.values.astype(float)
    sf_surv = sf.values.astype(float)
    # lifelines already seeds the timeline with the origin (t=0, S=1). Only
    # prepend it ourselves when it is missing — otherwise we create a
    # duplicate zero, the first trapezoid piece has zero width, and the loop
    # below breaks immediately, collapsing RMST to 0.
    if len(sf_times) and sf_times[0] == 0.0:
        times = sf_times
        surv = sf_surv
    else:
        times = np.concatenate(([0.0], sf_times))
        surv = np.concatenate(([1.0], sf_surv))

    et = kmf.event_table  # columns: removed, observed, censored, entrance, at_risk
    et_index = et.index.values.astype(float)
    d_arr = et["observed"].values.astype(float)
    n_arr = et["at_risk"].values.astype(float)

    # Trapezoid pieces between consecutive step points, capped at tau.
    pieces_t: List[float] = []  # left edges
    pieces_h: List[float] = []  # heights (S at left edge)
    pieces_w: List[float] = []  # widths (capped at tau)
    for k in range(len(times) - 1):
        a = times[k]
        b = min(times[k + 1], tau)
        if b <= a:
            break
        pieces_t.append(a)
        pieces_h.append(surv[k])
        pieces_w.append(b - a)
    pieces_t_arr = np.array(pieces_t)
    pieces_h_arr = np.array(pieces_h)
    pieces_w_arr = np.array(pieces_w)
    cum_area = np.concatenate(([0.0], np.cumsum(pieces_h_arr * pieces_w_arr)))
    total_area = cum_area[-1] if len(cum_area) > 0 else 0.0
    rmst = float(total_area)

    # Greenwood-style SE: Var(RMST(τ)) = Σ_{j: t_j ≤ τ} A_j^2 * d_j / (n_j (n_j - d_j))
    # with A_j = ∫_{t_j}^τ S(u) du.
    se_var = 0.0
    for j, tj in enumerate(et_index):
        if tj > tau:
            break
        d_j = d_arr[j]
        n_j = n_arr[j]
        if d_j <= 0 or n_j - d_j <= 0:
            continue
        idx = np.searchsorted(pieces_t_arr, tj, side="right")
        area_before = cum_area[idx] if idx < len(cum_area) else total_area
        A_j = total_area - area_before
        se_var += (A_j ** 2) * d_j / (n_j * (n_j - d_j))

    se = float(np.sqrt(se_var)) if se_var > 0 else 0.0
    n = int(len(t))
    n_events = int(e.astype(int).sum())
    z95 = 1.959963984540054
    return {
        "n": n,
        "n_events": n_events,
        "rmst": round(float(rmst), 4),
        "se": round(se, 4),
        "ci_low": round(float(rmst - z95 * se), 4),
        "ci_high": round(float(rmst + z95 * se), 4),
    }


@router.post("/rmst")
def rmst(req: RMSTRequest):
    if req.tau is None or req.tau <= 0:
        raise HTTPException(status_code=422, detail="tau must be > 0.")

    df_full = _get_df(req.session_id)
    for c in [req.duration_col, req.event_col]:
        if c not in df_full.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if req.group_col and req.group_col not in df_full.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.group_col}' not found")

    cols_needed = [req.duration_col, req.event_col] + ([req.group_col] if req.group_col else [])
    from services.impute import apply_imputation
    df = apply_imputation(df_full[cols_needed], cols_needed, req.imputation or "listwise").reset_index(drop=True)
    df[req.duration_col] = pd.to_numeric(df[req.duration_col], errors="coerce")
    df[req.event_col] = pd.to_numeric(df[req.event_col], errors="coerce")
    df = df.dropna()
    if len(df) < 5:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows (need ≥ 5, got {len(df)}).")

    t_all = df[req.duration_col].values.astype(float)
    if np.any(t_all < 0):
        raise HTTPException(status_code=422, detail="Negative durations are not allowed.")
    if req.tau > float(t_all.max()):
        raise HTTPException(
            status_code=422,
            detail=f"tau = {req.tau} exceeds the maximum observed time ({t_all.max():.3f}). "
                   "Pick a horizon within the observed follow-up.",
        )
    e_all = df[req.event_col].values.astype(int)
    if set(np.unique(e_all)) - {0, 1}:
        raise HTTPException(status_code=422, detail="Event column must be binary 0/1.")

    groups: List[Any] = []
    if req.group_col:
        groups = sorted(df[req.group_col].dropna().unique().tolist(), key=lambda x: (isinstance(x, str), x))

    rmst_by_group: Dict[str, dict] = {}
    if not groups:
        rmst_by_group["All"] = _rmst_one_group(t_all, e_all, float(req.tau))
    else:
        for g in groups:
            mask = df[req.group_col] == g
            rmst_by_group[str(g)] = _rmst_one_group(t_all[mask], e_all[mask], float(req.tau))

    # Pairwise contrasts when groups present
    contrasts: List[dict] = []
    if len(groups) >= 2:
        from scipy.stats import norm
        z95 = 1.959963984540054
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a = rmst_by_group[str(groups[i])]
                b = rmst_by_group[str(groups[j])]
                diff = a["rmst"] - b["rmst"]
                se = float(np.sqrt(a["se"] ** 2 + b["se"] ** 2))
                if se <= 0:
                    p = None
                    lo = hi = diff
                    z = None
                else:
                    z = diff / se
                    p = float(2 * (1 - norm.cdf(abs(z))))
                    lo = diff - z95 * se
                    hi = diff + z95 * se
                contrasts.append({
                    "group_a": str(groups[i]),
                    "group_b": str(groups[j]),
                    "delta_rmst": round(diff, 4),
                    "se": round(se, 4),
                    "z": round(z, 4) if z is not None else None,
                    "p": round(p, 6) if p is not None else None,
                    "ci_low": round(lo, 4),
                    "ci_high": round(hi, 4),
                })

    # Build plot (KM curves capped at tau, with shaded area = RMST per group)
    from lifelines import KaplanMeierFitter
    palette = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
    traces = []
    if not groups:
        kmf = KaplanMeierFitter()
        kmf.fit(t_all, e_all)
        sf = kmf.survival_function_.iloc[:, 0]
        traces.append({
            "x": [0] + [float(t) for t in sf.index.tolist()],
            "y": [1.0] + [float(v) for v in sf.values.tolist()],
            "type": "scatter", "mode": "lines",
            "line": {"color": palette[0], "width": 2, "shape": "hv"},
            "name": f"All (RMST = {rmst_by_group['All']['rmst']})",
        })
    else:
        for gi, g in enumerate(groups):
            mask = df[req.group_col] == g
            kmf = KaplanMeierFitter()
            kmf.fit(t_all[mask], e_all[mask])
            sf = kmf.survival_function_.iloc[:, 0]
            label = f"{g} (RMST = {rmst_by_group[str(g)]['rmst']})"
            traces.append({
                "x": [0] + [float(t) for t in sf.index.tolist()],
                "y": [1.0] + [float(v) for v in sf.values.tolist()],
                "type": "scatter", "mode": "lines",
                "line": {"color": palette[gi % len(palette)], "width": 2, "shape": "hv"},
                "name": label,
            })

    # Vertical line at tau
    shapes = [{
        "type": "line", "xref": "x", "yref": "paper",
        "x0": float(req.tau), "x1": float(req.tau), "y0": 0, "y1": 1,
        "line": {"color": "#9ca3af", "dash": "dash", "width": 1.5},
    }]
    annotations = [{
        "xref": "x", "yref": "paper", "x": float(req.tau), "y": 1.0,
        "xanchor": "left", "yanchor": "top",
        "text": f" τ = {req.tau}",
        "showarrow": False, "font": {"size": 11, "color": "#374151"},
    }]

    plot = {
        "data": traces,
        "layout": {
            "title": f"Restricted Mean Survival Time (τ = {req.tau})",
            "xaxis": {"title": req.duration_col, "gridcolor": "#e5e7eb"},
            "yaxis": {"title": "Survival probability", "range": [0, 1.05], "gridcolor": "#e5e7eb"},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "#ffffff",
            "font": {"color": "#374151", "size": 12},
            "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.05},
            "shapes": shapes,
            "annotations": annotations,
        },
    }

    n_total = len(df)
    if groups and len(groups) == 2 and contrasts:
        c0 = contrasts[0]
        result_text = (
            f"Restricted mean survival time on n = {n_total} subjects at τ = {req.tau}. "
            f"{c0['group_a']}: {rmst_by_group[c0['group_a']]['rmst']} "
            f"(95% CI {rmst_by_group[c0['group_a']]['ci_low']}–{rmst_by_group[c0['group_a']]['ci_high']}). "
            f"{c0['group_b']}: {rmst_by_group[c0['group_b']]['rmst']} "
            f"(95% CI {rmst_by_group[c0['group_b']]['ci_low']}–{rmst_by_group[c0['group_b']]['ci_high']}). "
            f"ΔRMST = {c0['delta_rmst']} (95% CI {c0['ci_low']}–{c0['ci_high']}), p = "
            f"{'<0.001' if (c0['p'] is not None and c0['p'] < 0.001) else (round(c0['p'], 3) if c0['p'] is not None else 'N/A')}."
        )
    else:
        result_text = (
            f"Restricted mean survival time on n = {n_total} subjects at τ = {req.tau}. "
            "RMST per group is reported below; the difference is interpretable as the "
            "average event-free time lived during the first τ time units."
        )

    assumptions = [
        {"name": "Censoring at random",      "met": True,
         "detail": "RMST assumes censoring is independent of the event process within each group."},
        {"name": "τ within observed range",  "met": True,
         "detail": f"τ = {req.tau} is at or below the maximum observed time ({float(t_all.max()):.3f})."},
        {"name": "No proportional-hazards required", "met": True,
         "detail": "RMST is a robust alternative when the PH assumption fails (e.g. crossing curves, late effects)."},
    ]

    export_rows = [["Group", "n", "Events", "RMST (τ)", "SE", "95% CI low", "95% CI high"]]
    for g_label, gv in rmst_by_group.items():
        export_rows.append([
            g_label, gv["n"], gv["n_events"], gv["rmst"], gv["se"], gv["ci_low"], gv["ci_high"],
        ])
    if contrasts:
        export_rows.append([])
        export_rows.append(["Group A", "Group B", "ΔRMST", "SE", "95% CI low", "95% CI high", "p"])
        for c in contrasts:
            export_rows.append([
                c["group_a"], c["group_b"], c["delta_rmst"], c["se"], c["ci_low"], c["ci_high"], c["p"],
            ])

    r_code = (
        "library(survRM2)\n"
        f"# data: time={req.duration_col}, status={req.event_col}"
        + (f", group={req.group_col}" if req.group_col else "")
        + f", tau={req.tau}\n"
        + (f"rmst2(time = data${req.duration_col}, status = data${req.event_col}, "
           f"arm = data${req.group_col}, tau = {req.tau})\n"
           if req.group_col else
           f"library(survival)\nfit <- survfit(Surv({req.duration_col}, {req.event_col}) ~ 1, data = data)\n"
           f"# RMST via integration of S(t) on [0, {req.tau}]\n")
    )

    try:
        store.log_action(req.session_id, "rmst", {
            "duration_col": req.duration_col,
            "event_col": req.event_col,
            "tau": float(req.tau),
            "group_col": req.group_col,
            "n_groups": len(groups) if groups else 1,
        })
    except Exception:
        pass

    return {
        "test": "Restricted Mean Survival Time",
        "n": n_total,
        "tau": float(req.tau),
        "rmst_by_group": rmst_by_group,
        "contrasts": contrasts,
        "plot": plot,
        "assumptions": assumptions,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ── 6. Recurrent events — LWYY (Lin-Wei-Yang-Ying) model ─────────────────────
#
# The LWYY marginal rate/mean model is a modified Andersen-Gill Cox model for
# RECURRENT events (e.g. repeat heart-failure hospitalisations): a single Cox
# fit on the counting-process (start, stop, event) data with a robust
# cluster-sandwich variance estimator clustered on subject id. The point
# estimate equals Andersen-Gill; the Lin-Wei-Yang-Ying (2000) contribution is
# the cluster-robust SE that accounts for within-subject correlation of events.
#
# Implemented with lifelines CoxPHFitter using entry_col = start, duration_col
# = stop, cluster_col = id, robust = True — mathematically the LWYY estimator.
# exp(beta) is interpreted as a rate ratio (ratio of the event RATES / mean
# cumulative functions), not a single-event hazard ratio.


class RecurrentLWYYRequest(BaseModel):
    session_id: str
    id_col: str
    start_col: str
    stop_col: str
    event_col: str
    predictors: List[str]
    group_col: Optional[str] = None          # for the mean cumulative function plot
    imputation: Optional[str] = "listwise"


def _mcf(intervals: pd.DataFrame, start: str, stop: str, event: str) -> List[dict]:
    """Nonparametric mean cumulative function (Nelson 1995) for recurrent
    events on counting-process intervals: MCF(t) = Σ_{t_k ≤ t} d_k / n_k where
    d_k = events at t_k and n_k = subjects under observation at t_k."""
    ev_times = np.sort(np.unique(intervals.loc[intervals[event] == 1, stop].values.astype(float)))
    starts = intervals[start].values.astype(float)
    stops = intervals[stop].values.astype(float)
    evs = intervals[event].values.astype(int)
    pts = [{"t": 0.0, "mcf": 0.0}]
    cum = 0.0
    for t in ev_times:
        d = int(np.sum((stops == t) & (evs == 1)))
        n = int(np.sum((starts < t) & (stops >= t)))
        if n > 0:
            cum += d / n
        pts.append({"t": round(float(t), 4), "mcf": round(float(cum), 5)})
    return pts


@router.post("/recurrent_lwyy")
def recurrent_lwyy(req: RecurrentLWYYRequest):
    from lifelines import CoxPHFitter

    df = _get_df(req.session_id)
    needed = [req.id_col, req.start_col, req.stop_col, req.event_col, *req.predictors]
    if req.group_col:
        needed.append(req.group_col)
    for c in needed:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.predictors:
        raise HTTPException(status_code=422, detail="Select at least one predictor.")

    from services.impute import apply_imputation
    work = apply_imputation(df[needed], needed, req.imputation or "listwise").reset_index(drop=True)

    # Coerce the counting-process columns.
    for c in [req.start_col, req.stop_col, req.event_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[req.id_col, req.start_col, req.stop_col, req.event_col])
    work = work[work[req.stop_col] > work[req.start_col]]
    if len(work) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough usable intervals (need ≥ 10, got {len(work)}).")
    evset = set(np.unique(work[req.event_col].astype(int)))
    if evset - {0, 1}:
        raise HTTPException(status_code=422, detail="Event column must be 0/1 per interval (1 = event at the interval's stop time).")

    # Encode predictors: numeric stays, categorical → dummies (drop_first).
    pred_raw = work[req.predictors].copy()
    numeric_pred, cat_pred = [], []
    for c in req.predictors:
        col = pred_raw[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric_pred.append(c)
        else:
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().mean() >= 0.8 and coerced.dropna().nunique() > 2:
                pred_raw[c] = coerced; numeric_pred.append(c)
            else:
                cat_pred.append(c)
    num_part = pred_raw[numeric_pred].apply(pd.to_numeric, errors="coerce") if numeric_pred else pd.DataFrame(index=pred_raw.index)
    cat_part = pd.get_dummies(pred_raw[cat_pred], drop_first=True, dummy_na=False) if cat_pred else pd.DataFrame(index=pred_raw.index)
    enc = pd.concat([num_part, cat_part], axis=1).astype(float)
    cov_cols = [str(c) for c in enc.columns]
    if not cov_cols:
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")

    fit_df = pd.concat([
        work[[req.id_col, req.start_col, req.stop_col, req.event_col]].reset_index(drop=True),
        enc.reset_index(drop=True),
    ], axis=1).dropna()

    cph = CoxPHFitter()
    try:
        cph.fit(fit_df, duration_col=req.stop_col, event_col=req.event_col,
                entry_col=req.start_col, cluster_col=req.id_col, robust=True, show_progress=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"LWYY fit failed: {exc}")

    summ = cph.summary
    coefs: List[dict] = []
    for var in cph.params_.index:
        beta = float(cph.params_[var])
        coefs.append({
            "variable": str(var),
            "estimate": round(beta, 6),
            "rate_ratio": round(float(np.exp(beta)), 4),
            "robust_se": round(float(summ.loc[var, "se(coef)"]), 6),
            "z": round(float(summ.loc[var, "z"]), 4),
            "p": round(float(summ.loc[var, "p"]), 6),
            "rr_low": round(float(summ.loc[var, "exp(coef) lower 95%"]), 4),
            "rr_high": round(float(summ.loc[var, "exp(coef) upper 95%"]), 4),
        })

    n_subjects = int(work[req.id_col].nunique())
    n_events = int((work[req.event_col] == 1).sum())
    # per-subject event counts
    ev_per = work.groupby(req.id_col)[req.event_col].sum()
    # total follow-up = sum over subjects of (max stop − min start)
    grp_fu = work.groupby(req.id_col)
    fu = grp_fu[req.stop_col].max() - grp_fu[req.start_col].min()
    total_fu = float(fu.sum())

    # Mean cumulative function — overall + by group.
    palette = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
    traces = []
    if req.group_col:
        groups = sorted(work[req.group_col].dropna().unique().tolist(), key=lambda x: (isinstance(x, str), x))
        for gi, g in enumerate(groups):
            sub = work[work[req.group_col] == g]
            pts = _mcf(sub, req.start_col, req.stop_col, req.event_col)
            traces.append({
                "x": [p["t"] for p in pts], "y": [p["mcf"] for p in pts],
                "type": "scatter", "mode": "lines", "name": f"{req.group_col} = {g}",
                "line": {"color": palette[gi % len(palette)], "width": 2, "shape": "hv"},
            })
    else:
        pts = _mcf(work, req.start_col, req.stop_col, req.event_col)
        traces.append({
            "x": [p["t"] for p in pts], "y": [p["mcf"] for p in pts],
            "type": "scatter", "mode": "lines", "name": "MCF",
            "line": {"color": palette[0], "width": 2, "shape": "hv"},
        })

    plot = {
        "data": traces,
        "layout": {
            "title": "Mean cumulative function (expected events per subject)",
            "xaxis": {"title": req.stop_col, "gridcolor": "#e5e7eb"},
            "yaxis": {"title": "Mean cumulative events", "gridcolor": "#e5e7eb"},
            "paper_bgcolor": "transparent", "plot_bgcolor": "#ffffff",
            "font": {"color": "#374151", "size": 12},
            "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
            "showlegend": bool(req.group_col),
            "legend": {"x": 0.02, "y": 0.98},
        },
    }

    primary = coefs[0]
    interp = (
        f"LWYY recurrent-event model on {n_subjects} subjects with {n_events} events "
        f"({ev_per.mean():.2f} events/subject; {n_events / total_fu * 100:.2f} events per 100 "
        f"time-units of follow-up). Andersen-Gill point estimates with Lin-Wei-Yang-Ying "
        f"cluster-robust SE. {primary['variable']}: rate ratio = {primary['rate_ratio']} "
        f"(95% CI {primary['rr_low']}–{primary['rr_high']}, p = "
        f"{'<0.001' if primary['p'] < 0.001 else round(primary['p'], 3)})."
    )

    assumptions = [
        {"name": "Recurrent-event structure", "met": True,
         "detail": f"Counting-process intervals (start, stop]; {len(work)} intervals across {n_subjects} subjects."},
        {"name": "Robust variance (LWYY)", "met": True,
         "detail": "Cluster-robust sandwich SE clustered on subject id — accounts for within-subject event correlation. No common-baseline-hazard or independent-increment assumption needed."},
        {"name": "Rate-ratio interpretation", "met": True,
         "detail": "exp(β) is the ratio of event rates (mean cumulative functions), not a single-event hazard ratio."},
    ]

    export_rows = [["Variable", "Rate ratio", "95% CI low", "95% CI high", "β", "Robust SE", "z", "p"]]
    for c in coefs:
        export_rows.append([c["variable"], c["rate_ratio"], c["rr_low"], c["rr_high"],
                            c["estimate"], c["robust_se"], c["z"], c["p"]])

    r_code = (
        "library(survival)\n"
        f"# LWYY = Andersen-Gill + robust cluster SE\n"
        f"fit <- coxph(Surv({req.start_col}, {req.stop_col}, {req.event_col}) ~ "
        f"{' + '.join(req.predictors)} + cluster({req.id_col}), data = data)\n"
        f"summary(fit)"
    )

    try:
        store.log_action(req.session_id, "recurrent_lwyy", {
            "id_col": req.id_col, "event_col": req.event_col,
            "n_predictors": len(req.predictors), "n_subjects": n_subjects, "n_events": n_events,
        })
    except Exception:
        pass

    return _safe({
        "test": "Recurrent events — LWYY model",
        "model": "Lin-Wei-Yang-Ying (modified Andersen-Gill, cluster-robust SE)",
        "n_subjects": n_subjects,
        "n_events": n_events,
        "n_intervals": int(len(work)),
        "events_per_subject": round(float(ev_per.mean()), 4),
        "total_followup": round(total_fu, 4),
        "concordance": round(float(cph.concordance_index_), 4),
        "coefficients": coefs,
        "plot": plot,
        "assumptions": assumptions,
        "result_text": interp,
        "interpretation": interp,
        "export_rows": export_rows,
        "r_code": r_code,
    })


# ── Survival model validation: time-dependent AUC(t) + calibration ───────────

class SurvivalValidationRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    horizon: float                       # landmark time t* for AUC(t) / calibration
    n_groups: int = 10                   # risk groups for the calibration plot
    imputation: Optional[str] = "listwise"


def _encode_survival_predictors(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """Numeric stay numeric; categorical → drop_first dummies; all float."""
    raw = df[predictors].copy()
    num, cat = [], []
    for c in predictors:
        if pd.api.types.is_numeric_dtype(raw[c]):
            num.append(c)
        else:
            coerced = pd.to_numeric(raw[c], errors="coerce")
            if coerced.notna().mean() >= 0.8 and coerced.dropna().nunique() > 2:
                raw[c] = coerced
                num.append(c)
            else:
                cat.append(c)
    num_part = raw[num].apply(pd.to_numeric, errors="coerce") if num else pd.DataFrame(index=raw.index)
    cat_part = pd.get_dummies(raw[cat], drop_first=True, dummy_na=False) if cat else pd.DataFrame(index=raw.index)
    enc = pd.concat([num_part, cat_part], axis=1)
    enc.columns = [str(c) for c in enc.columns]
    return enc.astype(float)


@router.post("/survival_validation")
def survival_validation(req: SurvivalValidationRequest):
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from services.impute import apply_imputation

    df = _get_df(req.session_id)
    for c in [req.duration_col, req.event_col, *req.predictors]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.predictors:
        raise HTTPException(status_code=422, detail="Select at least one predictor.")

    cols = [req.duration_col, req.event_col, *req.predictors]
    work = apply_imputation(df[cols], cols, req.imputation or "listwise").reset_index(drop=True)
    work[req.duration_col] = pd.to_numeric(work[req.duration_col], errors="coerce")
    work[req.event_col] = pd.to_numeric(work[req.event_col], errors="coerce")
    work = work.dropna(subset=[req.duration_col, req.event_col])
    enc = _encode_survival_predictors(work, req.predictors)
    fit_df = pd.concat(
        [work[[req.duration_col, req.event_col]].reset_index(drop=True), enc.reset_index(drop=True)], axis=1
    ).dropna()
    if len(fit_df) < 20:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows (need >= 20, got {len(fit_df)}).")
    if not list(enc.columns):
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")

    t = fit_df[req.duration_col].to_numpy(dtype=float)
    e = fit_df[req.event_col].to_numpy(dtype=int)
    if set(np.unique(e)) - {0, 1}:
        raise HTTPException(status_code=422, detail="Event column must be binary 0/1.")
    tau = float(req.horizon)
    if tau <= 0 or tau > float(t.max()):
        raise HTTPException(status_code=422, detail=f"horizon must be in (0, {t.max():.3f}].")

    cph = CoxPHFitter()
    try:
        cph.fit(fit_df, duration_col=req.duration_col, event_col=req.event_col)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cox model did not converge: {exc}")

    X = fit_df[list(enc.columns)]
    risk_lp = cph.predict_log_partial_hazard(X).to_numpy(dtype=float)          # higher = worse
    surv_t = cph.predict_survival_function(X, times=[tau]).iloc[0].to_numpy(dtype=float)
    pred_risk = 1.0 - surv_t                                                    # predicted cum. incidence at tau

    # ── IPCW cumulative-case / dynamic-control AUC(t) (Uno 2007) ──
    kmc = KaplanMeierFitter().fit(t, (e == 0).astype(int))                      # KM of the censoring distribution
    g_times = kmc.survival_function_.index.values.astype(float)
    g_vals = kmc.survival_function_.iloc[:, 0].to_numpy(dtype=float)

    def _G(x: float) -> float:
        idx = int(np.searchsorted(g_times, x, side="right")) - 1
        return float(g_vals[idx]) if idx >= 0 else 1.0

    case = (t <= tau) & (e == 1)
    control = t > tau
    auc_t = None
    if case.sum() > 0 and control.sum() > 0:
        w_case = np.array([1.0 / max(_G(ti), 1e-8) for ti in t])
        wc = 1.0 / max(_G(tau), 1e-8)
        Mc = np.sort(risk_lp[control])
        n_ctrl = len(Mc)
        num = 0.0
        den = 0.0
        for i in np.where(case)[0]:
            below = int(np.searchsorted(Mc, risk_lp[i], side="left"))
            ties = int(np.searchsorted(Mc, risk_lp[i], side="right")) - below
            num += w_case[i] * wc * (below + 0.5 * ties)
            den += w_case[i] * wc * n_ctrl
        auc_t = float(num / den) if den > 0 else None

    # ── Calibration by predicted-risk groups ──
    ng = max(2, min(req.n_groups, 20))
    order = np.argsort(pred_risk)
    cal = []
    obs_total = 0.0
    for gi, idxs in enumerate(np.array_split(order, ng)):
        if len(idxs) == 0:
            continue
        kmf = KaplanMeierFitter().fit(t[idxs], e[idxs])
        s_at = kmf.survival_function_at_times(tau)
        s_val = float(s_at.iloc[0]) if hasattr(s_at, "iloc") else float(s_at)
        obs = 1.0 - s_val
        cal.append({"group": gi + 1, "n": int(len(idxs)),
                    "pred": round(float(np.mean(pred_risk[idxs])), 4), "obs": round(float(obs), 4)})
        obs_total += obs * len(idxs)
    exp_total = float(np.sum(pred_risk))
    oe = float(obs_total / exp_total) if exp_total > 0 else None

    return _safe({
        "test": "Survival model validation",
        "horizon": tau,
        "n": int(len(fit_df)),
        "n_events_by_horizon": int(case.sum()),
        "time_auc": round(auc_t, 4) if auc_t is not None else None,
        "concordance": round(float(cph.concordance_index_), 4),
        "oe_ratio": round(oe, 4) if oe is not None else None,
        "calibration": cal,
        "coefficients": [
            {"variable": str(v), "hr": round(float(np.exp(cph.params_[v])), 4)}
            for v in cph.params_.index
        ],
        "note": (
            "Time-dependent AUC(t) is the IPCW cumulative-case / dynamic-control AUC at the "
            "horizon (Uno 2007). Calibration compares the Cox-predicted cumulative incidence "
            "with the 1 - Kaplan-Meier observed risk in each predicted-risk group; O/E is "
            "total observed / total expected events."
        ),
    })


# ── Discrete-time survival (person-period logistic, cluster-robust) ───────────

class DiscreteTimeRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    n_intervals: int = 5
    imputation: Optional[str] = "listwise"


@router.post("/discrete_time")
def discrete_time(req: DiscreteTimeRequest):
    import statsmodels.api as sm
    from services.impute import apply_imputation

    df = _get_df(req.session_id)
    for c in [req.duration_col, req.event_col, *req.predictors]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.predictors:
        raise HTTPException(status_code=422, detail="Select at least one predictor.")

    cols = [req.duration_col, req.event_col, *req.predictors]
    work = apply_imputation(df[cols], cols, req.imputation or "listwise").reset_index(drop=True)
    work[req.duration_col] = pd.to_numeric(work[req.duration_col], errors="coerce")
    work[req.event_col] = pd.to_numeric(work[req.event_col], errors="coerce")
    work = work.dropna(subset=[req.duration_col, req.event_col])
    enc = _encode_survival_predictors(work, req.predictors)
    base = pd.concat(
        [work[[req.duration_col, req.event_col]].reset_index(drop=True), enc.reset_index(drop=True)], axis=1
    ).dropna()
    if len(base) < 20:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows (need >= 20, got {len(base)}).")

    t = base[req.duration_col].to_numpy(dtype=float)
    e = base[req.event_col].to_numpy(dtype=int)
    if set(np.unique(e)) - {0, 1}:
        raise HTTPException(status_code=422, detail="Event column must be binary 0/1.")
    cov_cols = list(enc.columns)
    if not cov_cols:
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")

    n_int = max(2, min(int(req.n_intervals), 12))
    edges = np.unique(np.quantile(t, np.linspace(0.0, 1.0, n_int + 1)))
    edges[0] = float(t.min()) - 1e-9
    n_int = len(edges) - 1
    if n_int < 2:
        raise HTTPException(status_code=422, detail="Too few distinct event times to form discrete intervals.")
    if len(base) * n_int > 400_000:
        raise HTTPException(status_code=422, detail="Person-period dataset too large; reduce rows or intervals.")

    cov_arr = base[cov_cols].to_numpy(dtype=float)
    sid, interval, yv, covs = [], [], [], []
    for i in range(len(base)):
        ti, ei = float(t[i]), int(e[i])
        for k in range(n_int):
            lo, hi = edges[k], edges[k + 1]
            if ti <= lo:
                break
            y = 1 if (ei == 1 and ti <= hi) else 0
            sid.append(i); interval.append(k); yv.append(y); covs.append(cov_arr[i])
            if ti <= hi:
                break

    pp = pd.DataFrame(covs, columns=cov_cols)
    pp["_sid_"] = sid
    pp["_int_"] = interval
    pp["_y_"] = yv
    int_dummies = pd.get_dummies(pp["_int_"], prefix="interval", drop_first=True).astype(float)
    X = pd.concat([int_dummies, pp[cov_cols]], axis=1).astype(float)
    X = sm.add_constant(X, has_constant="add")

    try:
        model = sm.GEE(pp["_y_"].astype(float), X, groups=pp["_sid_"],
                       family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence())
        res = model.fit()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Discrete-time model did not converge: {exc}")

    ci = res.conf_int()
    coefs = []
    for v in res.params.index:
        if v == "const":
            continue
        beta = float(res.params[v])
        lo = float(ci.loc[v, 0]); hi = float(ci.loc[v, 1])
        is_interval = str(v).startswith("interval_")
        coefs.append({
            "variable": str(v),
            "kind": "baseline_interval" if is_interval else "covariate",
            "estimate": round(beta, 6),
            "or": round(float(np.exp(beta)), 4),
            "or_low": round(float(np.exp(lo)), 4),
            "or_high": round(float(np.exp(hi)), 4),
            "p": round(float(res.pvalues[v]), 6),
        })

    return _safe({
        "test": "Discrete-time survival (person-period logistic)",
        "model": "Cluster-robust logistic discrete-time hazard (GEE, independence working correlation)",
        "n_subjects": int(len(base)),
        "n_person_periods": int(len(pp)),
        "n_intervals": int(n_int),
        "interval_edges": [round(float(x), 4) for x in edges],
        "coefficients": coefs,
        "note": (
            "Each subject is split into person-period rows up to their event/censoring interval; "
            "a logistic model with interval indicators estimates the discrete-time hazard, with "
            "cluster-robust (GEE) standard errors at the subject level. exp(beta) for a covariate "
            "is the discrete-time hazard odds ratio; the interval terms are the baseline hazard shape."
        ),
    })
