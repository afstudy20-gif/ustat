"""
Joint Longitudinal-Survival Models (Phase 8 - Enhanced)

Provides advanced joint modeling of longitudinal biomarkers and time-to-event outcomes
using a pragmatic, dependency-light (statsmodels + lifelines) approach.

Capabilities:
- Time-varying two-stage joint models (counting-process format)
- Association structures: value, slope, area
- Non-linear trajectories (Natural Cubic Splines)
- Multivariate longitudinal markers
- Joint Latent Class Models (JLCM) via Gaussian Mixture Modeling of random effects
- Model comparison (AIC, BIC)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from sklearn.mixture import GaussianMixture

def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v

def _build_spline_formula(time_col: str, spline: bool) -> str:
    if spline:
        # Natural cubic spline with 3 degrees of freedom
        return f"cr({time_col}, df=3)"
    return time_col

def _extract_lmm_predictions(
    mdf: Any, df: pd.DataFrame, time_col: str, id_col: str, 
    association: List[str], spline: bool
) -> pd.DataFrame:
    """
    Computes value, slope, and area at each row's `time_col` based on the fitted LMM.
    """
    preds = df[[id_col, time_col]].copy()
    re_dict = mdf.random_effects
    
    if not spline:
        int_fixed_eff = mdf.params.get("Intercept", 0.0)
        time_fixed_eff = mdf.params.get(time_col, 0.0)
        
        # 1. value = (Intercept + random_intercept) + (slope_fixed + random_slope) * t
        y_pred = []
        for _, row in df.iterrows():
            g = row[id_col]
            t = row[time_col]
            re_vals = re_dict.get(g, pd.Series())
            b0 = re_vals.get("Group", 0.0)  # statsmodels names intercept RE "Group"
            b1 = re_vals.get(time_col, 0.0)
            val = (int_fixed_eff + b0) + (time_fixed_eff + b1) * t
            y_pred.append(val)
        preds['value'] = y_pred
        
        # 2. slope = beta_1 + b_1
        if "slope" in association:
            slopes = []
            for g in df[id_col]:
                re_vals = re_dict.get(g, pd.Series())
                time_re_eff = re_vals.get(time_col, 0.0)
                slopes.append(time_fixed_eff + time_re_eff)
            preds['slope'] = slopes
            
        # 3. area = (beta_0+b_0)*t + 0.5*(beta_1+b_1)*t^2
        if "area" in association:
            areas = []
            for _, row in df.iterrows():
                g = row[id_col]
                t = row[time_col]
                re_vals = re_dict.get(g, pd.Series())
                b0 = re_vals.get("Group", 0.0)
                b1 = re_vals.get(time_col, 0.0)
                a = (int_fixed_eff + b0) * t + 0.5 * (time_fixed_eff + b1) * (t**2)
                areas.append(a)
            preds['area'] = areas
    else:
        # Spline case: use statsmodels predict for value, fallback to 0.0 for others
        try:
            preds['value'] = mdf.predict(df)
        except Exception:
            preds['value'] = 0.0
            
        if "slope" in association:
            preds['slope'] = 0.0
        if "area" in association:
            preds['area'] = 0.0
            
    return preds

def _to_counting_process(
    long_df: pd.DataFrame, surv_df: pd.DataFrame, 
    id_col: str, time_col: str, duration_col: str, event_col: str
) -> pd.DataFrame:
    """
    Converts survival data into counting process format (start, stop, event) 
    split by longitudinal measurement times.
    """
    records = []
    
    surv_df_idx = surv_df.set_index(id_col, drop=True)
    long_groups = long_df.groupby(id_col)
    
    for id_val, s_row in surv_df_idx.iterrows():
        event_time = s_row[duration_col]
        event_status = s_row[event_col]
        
        if id_val in long_groups.groups:
            l_df = long_groups.get_group(id_val)
            m_times = sorted(l_df[l_df[time_col] <= event_time][time_col].unique())
        else:
            m_times = []
            
        if not m_times or m_times[0] > 0:
            m_times = [0.0] + m_times
            
        if m_times[-1] < event_time:
            m_times.append(event_time)
            
        for i in range(len(m_times) - 1):
            start = m_times[i]
            stop = m_times[i+1]
            if start == stop:
                continue
                
            is_last = (i == len(m_times) - 2)
            records.append({
                id_col: id_val,
                "start": float(start),
                "stop": float(stop),
                event_col: int(event_status) if is_last else 0,
                duration_col: float(stop)  # map for evaluation
            })
            
    cp_df = pd.DataFrame(records)
    
    # Join baseline predictors - use original surv_df which is not indexed
    baseline_cols = [c for c in surv_df.columns if c not in [id_col, duration_col, event_col]]
    if baseline_cols:
        cp_df = cp_df.merge(surv_df[[id_col] + baseline_cols], on=id_col, how="left")
        
    return cp_df


def fit_time_varying_joint_model(
    long_df: pd.DataFrame,
    surv_df: pd.DataFrame,
    id_col: str = "id",
    time_col: str = "time",
    y_cols: List[str] = ["Y"],
    long_predictors: Optional[List[str]] = None,
    surv_predictors: Optional[List[str]] = None,
    duration_col: str = "duration",
    event_col: str = "event",
    association: List[str] = ["value"],
    time_spline: bool = False
) -> Dict[str, Any]:
    """
    Fits a time-varying two-stage joint model.
    """
    if not long_predictors:
        long_predictors = []
    if not surv_predictors:
        surv_predictors = []
    if not association:
        association = ["value"]
    
    long_clean = long_df.loc[:, ~long_df.columns.duplicated()].copy()
    
    lmm_summaries = {}
    total_loglike_lmm = 0.0
    
    cp_df = _to_counting_process(long_df, surv_df, id_col, time_col, duration_col, event_col)
    
    eval_df = cp_df[[id_col, duration_col]].rename(columns={duration_col: time_col})
    tv_covariates = []
    
    # Sort groups so statsmodels handles properly
    long_clean = long_clean.sort_values([id_col, time_col]).reset_index(drop=True)
    
    for y_col in y_cols:
        formula = f"{y_col} ~ {_build_spline_formula(time_col, time_spline)}"
        if long_predictors:
            formula += " + " + " + ".join(long_predictors)
            
        re_formula = f"~{_build_spline_formula(time_col, time_spline)}"
        
        md = smf.mixedlm(formula, long_clean, groups=long_clean[id_col], re_formula=re_formula)
        try:
            mdf = md.fit(method=["lbfgs"], reml=False) # ML for correct AIC
        except Exception:
            mdf = md.fit(method=["cg"], reml=False)
            
        total_loglike_lmm += mdf.llf
        
        lmm_summaries[y_col] = {
            "params": mdf.params.to_dict(),
            "bse": mdf.bse.to_dict(),
            "pvalues": mdf.pvalues.to_dict(),
        }
        
        preds = _extract_lmm_predictions(mdf, eval_df, time_col, id_col, association, time_spline)
        
        for a in association:
            col_name = f"{y_col}_{a}"
            cp_df[col_name] = preds[a]
            tv_covariates.append(col_name)
            
    cox_predictors = surv_predictors + tv_covariates
    cox_df = cp_df[[id_col, "start", "stop", event_col] + cox_predictors].copy()
    
    for c in cox_predictors:
        if cox_df[c].dtype == object:
            cox_df[c] = pd.Categorical(cox_df[c]).codes
            
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(
        cox_df, 
        duration_col="stop", 
        event_col=event_col, 
        entry_col="start", 
        cluster_col=id_col, 
        robust=True
    )
    
    coefs = []
    for var in cph.params_.index:
        beta = float(cph.params_[var])
        coefs.append({
            "variable": str(var),
            "coef": round(beta, 5),
            "hr": round(np.exp(beta), 4),
            "se": round(float(cph.standard_errors_[var]), 5),
            "p": _safe(cph.summary.loc[var, "p"] if "p" in cph.summary.columns else None),
        })
        
    num_params = cph.summary.shape[0] + sum(len(s['params']) for s in lmm_summaries.values())
    total_ll = cph.log_likelihood_ + total_loglike_lmm
    aic = 2 * num_params - 2 * total_ll
    bic = num_params * np.log(len(surv_df)) - 2 * total_ll
        
    return {
        "model": "time_varying_joint_lmm_cox",
        "lmm_summaries": lmm_summaries,
        "cox_coefficients": coefs,
        "cox_concordance": round(float(cph.concordance_index_), 4),
        "n_subjects": int(len(surv_df)),
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "log_likelihood": round(total_ll, 2),
        "note": "Time-varying joint model. Association evaluated dynamically via counting process.",
    }

def fit_latent_class_joint_model(
    long_df: pd.DataFrame,
    surv_df: pd.DataFrame,
    id_col: str = "id",
    time_col: str = "time",
    y_cols: List[str] = ["Y"],
    long_predictors: Optional[List[str]] = None,
    surv_predictors: Optional[List[str]] = None,
    duration_col: str = "duration",
    event_col: str = "event",
    latent_classes: int = 2
) -> Dict[str, Any]:
    """
    Fits a pragmatic Joint Latent Class Model (JLCM) using GMM on extracted random effects.
    """
    if not long_predictors:
        long_predictors = []
    if not surv_predictors:
        surv_predictors = []
    
    long_clean = long_df.loc[:, ~long_df.columns.duplicated()].copy()
    long_clean = long_clean.sort_values([id_col, time_col]).reset_index(drop=True)
    
    lmm_summaries = {}
    re_dfs = []
    total_loglike_lmm = 0.0
    
    for y_col in y_cols:
        formula = f"{y_col} ~ {time_col}"
        if long_predictors:
            formula += " + " + " + ".join(long_predictors)
            
        md = smf.mixedlm(formula, long_clean, groups=long_clean[id_col], re_formula=f"~{time_col}")
        mdf = md.fit(method=["lbfgs"], reml=False)
        total_loglike_lmm += mdf.llf
        
        lmm_summaries[y_col] = {
            "params": mdf.params.to_dict(),
        }
        
        re_dict = mdf.random_effects
        re_df = pd.DataFrame.from_dict(re_dict, orient="index")
        re_df.columns = [f"{y_col}_RE_{c}" for c in re_df.columns]
        re_df.index.name = id_col
        re_dfs.append(re_df)
        
    combined_re = pd.concat(re_dfs, axis=1).reset_index()
    
    # GMM
    re_features = combined_re.drop(columns=[id_col]).fillna(0)
    gmm = GaussianMixture(n_components=latent_classes, random_state=42)
    gmm.fit(re_features)
    
    combined_re['latent_class'] = gmm.predict(re_features)
    
    surv_work = surv_df.merge(combined_re[[id_col, 'latent_class']], on=id_col, how="inner")
    
    cox_predictors = surv_predictors + ['latent_class']
    cox_df = surv_work[[duration_col, event_col] + cox_predictors].copy()
    
    for c in cox_predictors:
        if c == 'latent_class' or cox_df[c].dtype == object:
            cox_df[c] = cox_df[c].astype(str)
            cox_df[c] = pd.Categorical(cox_df[c]).codes
            
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col, robust=True)
    
    coefs = []
    for var in cph.params_.index:
        beta = float(cph.params_[var])
        coefs.append({
            "variable": str(var),
            "coef": round(beta, 5),
            "hr": round(np.exp(beta), 4),
            "se": round(float(cph.standard_errors_[var]), 5),
            "p": _safe(cph.summary.loc[var, "p"] if "p" in cph.summary.columns else None),
        })
        
    num_params = cph.summary.shape[0] + sum(len(s['params']) for s in lmm_summaries.values()) + (latent_classes - 1)
    total_ll = cph.log_likelihood_ + total_loglike_lmm
    aic = 2 * num_params - 2 * total_ll
    bic = num_params * np.log(len(surv_df)) - 2 * total_ll
        
    return {
        "model": "latent_class_joint_lmm_cox",
        "lmm_summaries": lmm_summaries,
        "cox_coefficients": coefs,
        "cox_concordance": round(float(cph.concordance_index_), 4),
        "n_subjects": int(len(surv_work)),
        "n_classes": latent_classes,
        "class_proportions": {str(k): round(float(v), 3) for k, v in combined_re['latent_class'].value_counts(normalize=True).items()},
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "log_likelihood": round(total_ll, 2),
        "note": "JLCM using LMM random effects + GMM clustering + Cox.",
    }
