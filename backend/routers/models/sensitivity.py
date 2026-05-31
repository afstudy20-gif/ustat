from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Literal, Optional
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from services import store
from services.impute import (
    apply_imputation,
    add_survival_auxiliary_variables,
    apply_passive_imputation,
)
from services.causal_sensitivity import (
    e_value,
    e_value_for_smd,
    manski_bounds_binary,
    manski_bounds_from_data,
    multi_confounder_sensitivity,
    negative_control_analysis,
    quantitative_bias_analysis,
    rosenbaum_bounds_from_matched_data,
)
from services.missing_data import (
    auxiliary_variable_guidance,
    congeniality_assessment,
    mice_convergence_diagnostics,
    mice_multiple,
    posterior_predictive_check,
)
from services.missing_data_sensitivity import (
    delta_adjustment_sensitivity,
    heckman_selection_model,
    isni_index,
    pattern_mixture_delta_model,
    survival_mnar_sensitivity,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


# ── 1. Causal Sensitivity Analysis ──────────────────────────────────────────

class CausalSensitivityRequest(BaseModel):
    observed_estimate: float = Field(..., gt=0, description="Point estimate on RR/OR/HR scale")
    ci_low: Optional[float] = Field(None, gt=0)
    ci_high: Optional[float] = Field(None, gt=0)
    measure: Literal["rr", "or", "hr"] = "rr"
    rare_outcome: bool = False
    baseline_risk: Optional[float] = Field(None, ge=0.001, le=0.99, description="Baseline risk for OR->RR conversion")
    smd: Optional[float] = None

    confounding_strength: float = Field(2.0, gt=0)
    prevalence_exposed: float = Field(0.5, ge=0, le=1)
    prevalence_unexposed: float = Field(0.5, ge=0, le=1)
    unmeasured_confounders: List[Dict[str, Any]] = Field(default_factory=list)

    session_id: Optional[str] = None
    treatment_col: Optional[str] = None
    outcome_col: Optional[str] = None
    monotone_treatment_response: bool = False
    p_y1_treated: Optional[float] = Field(None, ge=0, le=1)
    p_y1_control: Optional[float] = Field(None, ge=0, le=1)
    p_treated: Optional[float] = Field(None, ge=0, le=1)

    match_id_col: Optional[str] = None
    rosenbaum_gamma_max: float = Field(3.0, gt=1)
    rosenbaum_n_gamma: int = Field(60, ge=2, le=500)

    negative_control_outcome_col: Optional[str] = None
    negative_control_covariates: List[str] = Field(default_factory=list)
    imputation: Optional[str] = "listwise"


@router.post("/causal_sensitivity")
async def causal_sensitivity(req: CausalSensitivityRequest):
    """
    Causal sensitivity suite: E-value, QBA, Manski bounds, Rosenbaum bounds,
    multi-confounder scenarios, SMD E-value, and negative-control screening.
    """
    measure = (req.measure or "rr").lower()
    if measure not in {"rr", "or", "hr"}:
        raise HTTPException(status_code=422, detail="measure must be rr, or, or hr")
    if req.ci_low is not None and req.ci_high is not None and req.ci_low >= req.ci_high:
        raise HTTPException(status_code=400, detail="ci_low must be < ci_high")

    # Run E-value and QBA (very fast, but thread pool is safe)
    ev = await asyncio.to_thread(
        e_value,
        estimate=req.observed_estimate,
        ci_low=req.ci_low,
        ci_high=req.ci_high,
        measure=measure,
        rare_outcome=req.rare_outcome,
        baseline_risk=req.baseline_risk,
    )

    qba = await asyncio.to_thread(
        quantitative_bias_analysis,
        observed_estimate=req.observed_estimate,
        measure=measure,
        confounding_strength=req.confounding_strength,
        prevalence_exposed=req.prevalence_exposed,
        prevalence_unexposed=req.prevalence_unexposed,
    )

    multi = await asyncio.to_thread(
        multi_confounder_sensitivity,
        req.observed_estimate,
        req.unmeasured_confounders,
        measure=measure,
    ) if req.unmeasured_confounders else {"available": False, "reason": "No unmeasured_confounders array supplied."}

    smd_ev = await asyncio.to_thread(
        e_value_for_smd,
        req.smd,
        baseline_risk=req.baseline_risk or 0.1
    ) if req.smd is not None else {"available": False, "reason": "No SMD supplied."}

    manski: Dict[str, Any]
    rosenbaum: Dict[str, Any] = {"applicable": False, "reason": "No matched data columns supplied."}
    negative_control: Dict[str, Any] = {"available": False, "reason": "No negative control outcome supplied."}
    df = None
    if req.session_id:
        df = _get_df(req.session_id)

    if df is not None and req.treatment_col and req.outcome_col:
        needed = [req.treatment_col, req.outcome_col]
        if req.match_id_col:
            needed.append(req.match_id_col)
        if req.negative_control_outcome_col:
            needed.append(req.negative_control_outcome_col)
        needed.extend(req.negative_control_covariates or [])
        missing = [c for c in set(needed) if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {sorted(missing)}")

        work = await asyncio.to_thread(
            apply_imputation,
            df[list(dict.fromkeys(needed))],
            list(dict.fromkeys(needed)),
            req.imputation or "listwise"
        )

        manski = await asyncio.to_thread(
            manski_bounds_from_data,
            work,
            req.treatment_col,
            req.outcome_col,
            monotone_treatment_response=req.monotone_treatment_response,
        )

        if req.match_id_col:
            rosenbaum = await asyncio.to_thread(
                rosenbaum_bounds_from_matched_data,
                work,
                req.match_id_col,
                req.treatment_col,
                req.outcome_col,
                gamma_max=req.rosenbaum_gamma_max,
                n_gamma=req.rosenbaum_n_gamma,
            )
        if req.negative_control_outcome_col:
            negative_control = await asyncio.to_thread(
                negative_control_analysis,
                work,
                req.treatment_col,
                req.negative_control_outcome_col,
                covariates=req.negative_control_covariates,
            )
    elif req.p_y1_treated is not None and req.p_y1_control is not None and req.p_treated is not None:
        manski = await asyncio.to_thread(
            manski_bounds_binary,
            p_y1_treated=req.p_y1_treated,
            p_y1_control=req.p_y1_control,
            p_treated=req.p_treated,
            monotone_treatment_response=req.monotone_treatment_response,
        )
        manski["available"] = True
    else:
        manski = {"available": False, "reason": "Supply session_id+treatment_col+outcome_col or Manski probabilities."}

    warnings = []
    if ev.get("e_value_point_estimate", 99) < 2:
        warnings.append("Low E-value (<2); result is sensitive to weak unmeasured confounding.")
    if rosenbaum.get("critical_gamma") is not None and rosenbaum.get("critical_gamma", 99) < 1.5:
        warnings.append("Rosenbaum critical gamma is low; matched result may be sensitive to small hidden bias.")
    if negative_control.get("flag_residual_bias"):
        warnings.append("Negative control outcome is associated with treatment; residual bias signal detected.")

    result_text = (
        f"Causal sensitivity suite: E-value {ev.get('e_value_point_estimate')} for observed "
        f"{measure.upper()}={req.observed_estimate}. QBA corrected estimate {qba.get('bias_corrected_estimate')}."
    )
    if manski.get("available"):
        result_text += f" Manski ATE bounds: {manski.get('ate_bounds')}."

    return {
        "test": "Causal Sensitivity Analysis (E-value + QBA + Partial Identification)",
        "e_value": ev,
        "e_value_smd": smd_ev,
        "quantitative_bias_analysis": qba,
        "multi_confounder_sensitivity": multi,
        "manski_bounds": manski,
        "rosenbaum_bounds": rosenbaum,
        "negative_control_analysis": negative_control,
        "warnings": warnings,
        "assumptions": [
            {"name": "No unmeasured confounding", "met": False,
             "detail": "Sensitivity methods quantify how violations could change inference."},
            {"name": "Manski partial identification", "met": manski.get("available", False),
             "detail": "Bounds avoid exchangeability assumptions but can be wide."},
            {"name": "Rosenbaum matched-pair sensitivity", "met": rosenbaum.get("applicable", False),
             "detail": "Requires clean 1:1 matched binary-outcome pairs."},
        ],
        "result_text": result_text,
        "export_rows": [
            ["Metric", "Value"],
            ["Observed estimate", req.observed_estimate],
            ["Measure", measure],
            ["E-value point", ev.get("e_value_point_estimate")],
            ["E-value CI", ev.get("e_value_ci")],
            ["QBA bias factor", qba.get("bias_factor")],
            ["QBA corrected estimate", qba.get("bias_corrected_estimate")],
            ["Manski ATE bounds", manski.get("ate_bounds")],
            ["Rosenbaum critical gamma", rosenbaum.get("critical_gamma")],
            ["Negative control p", negative_control.get("p")],
        ],
        "r_code": (
            "library(EValue)\n"
            "# E-values: EValue::evalues.RR(est, lo, hi)\n"
            "# Rosenbaum bounds: rbounds::psens(...)\n"
            "# Partial identification: report Manski lower/upper bounds."
        ),
    }


# ── 2. MNAR Sensitivity Analysis ─────────────────────────────────────────────

class MNARSensitivityRequest(BaseModel):
    session_id: str
    columns: List[str]
    outcome_col: Optional[str] = None
    predictors: List[str] = Field(default_factory=list)
    selection_predictors: List[str] = Field(default_factory=list)
    auxiliary_candidates: List[str] = Field(default_factory=list)
    delta_values: List[float] = Field(default_factory=lambda: [-1.0, 0.0, 1.0])
    n_imputations: int = 5
    max_iter: int = 10
    passive_formulas: Dict[str, str] = Field(default_factory=dict)
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    model_type: str = "logistic"
    run_heckman: bool = True
    run_isni: bool = True
    run_survival_mnar: bool = True
    imputation: Optional[str] = "listwise"


@router.post("/mnar_sensitivity")
async def mnar_sensitivity(req: MNARSensitivityRequest):
    """
    Run MNAR missing data sensitivity analysis including pattern mixture models,
    selection models, local sensitivity index (ISNI), and survival specific models.
    """
    df_full = _get_df(req.session_id)
    needed = list(dict.fromkeys(
        req.columns
        + ([req.outcome_col] if req.outcome_col else [])
        + req.predictors
        + req.selection_predictors
        + ([req.duration_col, req.event_col] if req.duration_col and req.event_col else [])
        + (req.auxiliary_candidates or [])
    ))
    missing = [c for c in needed if c and c not in df_full.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    if not req.columns:
        raise HTTPException(400, "Select at least one variable with missing data.")

    df = df_full.copy()
    if req.duration_col and req.event_col:
        df = await asyncio.to_thread(add_survival_auxiliary_variables, df, req.duration_col, req.event_col)

    imputation_cols = list(dict.fromkeys(req.columns + req.predictors + ([req.outcome_col] if req.outcome_col else [])))
    if req.duration_col and req.event_col:
        imputation_cols.extend([req.duration_col, req.event_col, "__surv_aux_log_time", "__surv_aux_nelson_aalen"])
    imputation_cols = [c for c in dict.fromkeys(imputation_cols) if c in df.columns]

    # Run heavy operations in background threads using asyncio.to_thread
    mice_result = await asyncio.to_thread(
        mice_multiple,
        df,
        imputation_cols,
        n_imputations=max(2, req.n_imputations),
        max_iter=req.max_iter,
    )

    passive_preview = await asyncio.to_thread(
        apply_passive_imputation,
        mice_result.imputed_datasets[0],
        req.passive_formulas
    )

    pmm = await asyncio.to_thread(
        pattern_mixture_delta_model,
        df,
        imputation_cols,
        delta_values=req.delta_values,
        n_imputations=max(2, req.n_imputations),
        passive_formulas=req.passive_formulas,
        duration_col=req.duration_col,
        event_col=req.event_col,
    )

    model_delta = None
    if req.outcome_col and req.predictors:
        try:
            model_delta = await asyncio.to_thread(
                delta_adjustment_sensitivity,
                df,
                outcome=req.outcome_col,
                predictors=req.predictors,
                model_type=req.model_type if req.model_type in {"linear", "logistic", "cox"} else "logistic",
                delta_range=(min(req.delta_values), max(req.delta_values)),
                n_steps=len(req.delta_values),
                duration_col=req.duration_col,
                event_col=req.event_col,
            )
        except Exception as exc:
            logger.exception("Delta adjustment sensitivity analysis failed")
            model_delta = {"available": False, "reason": str(exc)}

    heckman = {"available": False, "reason": "Heckman not requested or outcome/predictors missing."}
    if req.run_heckman and req.outcome_col and req.predictors:
        try:
            heckman = await asyncio.to_thread(
                heckman_selection_model,
                df,
                outcome_col=req.outcome_col,
                outcome_predictors=req.predictors,
                selection_predictors=req.selection_predictors or req.predictors,
            )
        except Exception as exc:
            logger.exception("Heckman selection model failed")
            heckman = {"available": False, "reason": str(exc)}

    isni = {"available": False, "reason": "ISNI not requested or outcome/predictors missing."}
    if req.run_isni and req.outcome_col and req.predictors:
        try:
            isni = await asyncio.to_thread(
                isni_index,
                df,
                req.outcome_col,
                req.predictors,
                missing_cols=req.columns
            )
            isni["available"] = True
        except Exception as exc:
            logger.exception("ISNI computation failed")
            isni = {"available": False, "reason": str(exc)}

    survival_mnar = {"available": False, "reason": "Survival MNAR not requested or duration/event/predictors missing."}
    if req.run_survival_mnar and req.duration_col and req.event_col and req.predictors:
        try:
            survival_mnar = await asyncio.to_thread(
                survival_mnar_sensitivity,
                df,
                req.duration_col,
                req.event_col,
                req.predictors,
                censoring_delta_values=req.delta_values,
            )
        except Exception as exc:
            logger.exception("Survival MNAR sensitivity failed")
            survival_mnar = {"available": False, "reason": str(exc)}

    convergence = await asyncio.to_thread(mice_convergence_diagnostics, mice_result, df, imputation_cols)
    ppc = await asyncio.to_thread(posterior_predictive_check, mice_result, df, imputation_cols)
    aux = await asyncio.to_thread(
        auxiliary_variable_guidance,
        df,
        req.columns,
        candidate_cols=req.auxiliary_candidates,
    )
    congeniality = await asyncio.to_thread(
        congeniality_assessment,
        imputation_cols,
        [c for c in [req.outcome_col, req.duration_col, req.event_col] + req.predictors if c],
        passive_formulas=req.passive_formulas,
    )

    passive_cols = {}
    for target in req.passive_formulas:
        if target in passive_preview.columns:
            vals = pd.to_numeric(passive_preview[target], errors="coerce")
            passive_cols[target] = {
                "n_nonmissing": int(vals.notna().sum()),
                "mean": round(float(vals.mean()), 6) if vals.notna().any() else None,
            }

    warnings = []
    high_rhat = [
        c for c, v in convergence.get("variables", {}).items()
        if v.get("r_hat_proxy") is not None and v.get("r_hat_proxy") > 1.1
    ]
    if high_rhat:
        warnings.append(f"Potential MICE convergence concern for: {', '.join(high_rhat)}.")
    if survival_mnar.get("available"):
        warnings.append("Survival MNAR sensitivity uses informative-censoring weight shifts; interpret as scenario analysis.")
    if heckman.get("selection_bias_signal"):
        warnings.append("Heckman inverse Mills ratio suggests possible selection bias.")

    return {
        "test": "MNAR Missing Data Sensitivity Analysis",
        "n": int(len(df_full)),
        "columns": req.columns,
        "pattern_mixture_model": pmm,
        "model_delta_sensitivity": model_delta,
        "heckman_selection_model": heckman,
        "isni": isni,
        "mice_convergence_diagnostics": convergence,
        "imputation_model_diagnostics": ppc,
        "congeniality_assessment": congeniality,
        "passive_imputation": {
            "formulas": req.passive_formulas,
            "preview": passive_cols,
        },
        "survival_specific_imputation": {
            "enabled": bool(req.duration_col and req.event_col),
            "auxiliary_variables": [c for c in ["__surv_aux_log_time", "__surv_aux_nelson_aalen"] if c in df.columns],
        },
        "auxiliary_variable_guidance": aux,
        "survival_mnar_sensitivity": survival_mnar,
        "warnings": warnings,
        "assumptions": [
            {"name": "MAR reference imputation", "met": True,
             "detail": "Delta-adjusted pattern-mixture scenarios start from MAR MICE imputations."},
            {"name": "MNAR scenario analysis", "met": True,
             "detail": "Delta values encode unverifiable assumptions about missing outcomes/covariates."},
            {"name": "Heckman exclusion restriction", "met": bool(req.selection_predictors),
             "detail": "Selection models are stronger with predictors of missingness not already in the outcome equation."},
        ],
        "result_text": (
            f"MNAR sensitivity analysis ran for {len(req.columns)} variable(s) across "
            f"{len(req.delta_values)} delta scenario(s), with {req.n_imputations} imputation chains."
        ),
        "r_code": (
            "library(mice)\n"
            "# Pattern-mixture delta adjustment: mice(...); complete(); shift missing cells by delta\n"
            "# Heckman: sampleSelection::selection(...)\n"
            "# Survival auxiliaries: include Nelson-Aalen cumulative hazard and log time in imputation model"
        ),
    }
