"""Nomogram builder: logistic regression and Cox regression nomograms."""
import numpy as np
import pandas as pd
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import store

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


class NomogramRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    model_type: str = "logistic"  # "logistic" or "cox"
    imputation: str = "listwise"


def _handle_missing(df: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame:
    """Handle missing data for model fitting."""
    if method == "listwise":
        return df[columns].dropna()
    elif method == "mean":
        result = df[columns].copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].mean())
            else:
                result[col] = result[col].fillna(result[col].mode().iloc[0] if len(result[col].mode()) > 0 else 0)
        return result.dropna()
    elif method == "median":
        result = df[columns].copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
            else:
                result[col] = result[col].fillna(result[col].mode().iloc[0] if len(result[col].mode()) > 0 else 0)
        return result.dropna()
    else:
        return df[columns].dropna()


@router.post("/build")
def build_nomogram(req: NomogramRequest):
    """Build a nomogram from logistic or Cox regression."""
    df = _get_df(req.session_id)

    # Validate columns
    missing_cols = [c for c in [req.outcome] + req.predictors if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing_cols}")

    if req.model_type not in ("logistic", "cox"):
        raise HTTPException(status_code=400, detail="model_type must be 'logistic' or 'cox'")

    if req.model_type == "logistic":
        return _build_logistic_nomogram(df, req)
    else:
        return _build_cox_nomogram(df, req)


def _build_logistic_nomogram(df: pd.DataFrame, req: NomogramRequest) -> dict:
    """Build nomogram from logistic regression."""
    import statsmodels.api as sm

    all_cols = [req.outcome] + req.predictors
    df_clean = _handle_missing(df, all_cols, req.imputation)

    if len(df_clean) < len(req.predictors) + 10:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data after handling missing values: {len(df_clean)} rows",
        )

    y = pd.to_numeric(df_clean[req.outcome], errors="coerce")
    X = df_clean[req.predictors].apply(pd.to_numeric, errors="coerce")

    # Drop any rows that couldn't be converted
    valid = y.notna() & X.notna().all(axis=1)
    y = y[valid].values
    X = X[valid].values
    predictor_names = req.predictors
    n = len(y)

    if n < len(predictor_names) + 10:
        raise HTTPException(status_code=400, detail=f"Insufficient valid data: {n} rows")

    # Unique outcome values check
    unique_y = np.unique(y)
    if len(unique_y) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Outcome must be binary for logistic regression. Found {len(unique_y)} unique values.",
        )

    X_with_const = sm.add_constant(X)
    try:
        model = sm.Logit(y, X_with_const).fit(disp=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model fitting failed: {str(e)}")

    coefficients = model.params[1:]  # Exclude intercept
    intercept = model.params[0]

    # Compute reference values (mean of each predictor in the data)
    X_df = pd.DataFrame(X, columns=predictor_names)
    references = X_df.mean().values

    # Scale: strongest predictor gets 0-100 points
    abs_coefs = np.abs(coefficients)
    ranges = X_df.max().values - X_df.min().values
    max_contributions = abs_coefs * ranges
    max_contribution = max_contributions.max()
    scale = 100.0 / max_contribution if max_contribution > 0 else 1.0

    # Build predictors table
    predictors_table = []
    for i, pred in enumerate(predictor_names):
        pred_min = float(X_df[pred].min())
        pred_max = float(X_df[pred].max())
        ref = float(references[i])
        coef = float(coefficients[i])

        # Generate evenly spaced values across the predictor range
        n_values = min(11, max(5, int(X_df[pred].nunique())))
        values = np.linspace(pred_min, pred_max, n_values)
        points = (coef * (values - ref)) * scale

        predictors_table.append({
            "predictor": pred,
            "coefficient": round(coef, 4),
            "reference": round(ref, 4),
            "values": [round(float(v), 4) for v in values],
            "points": [round(float(p), 2) for p in points],
            "max_points": round(float(np.max(np.abs(points))), 2),
        })

    # Total points to probability mapping
    min_total = sum(min(p["points"]) for p in predictors_table)
    max_total = sum(max(p["points"]) for p in predictors_table)
    total_points_range = np.linspace(min_total, max_total, 20)

    # Convert total points back to probability via inverse logit
    # Total points = scale * (sum of beta_i * (x_i - ref_i))
    # log-odds = intercept + total_points / scale
    probability_mapping = []
    for tp in total_points_range:
        log_odds = intercept + tp / scale
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        probability_mapping.append({
            "total_points": round(float(tp), 2),
            "probability": round(float(prob), 4),
        })

    # C-statistic (AUC)
    from sklearn.metrics import roc_auc_score
    try:
        y_pred_prob = model.predict(X_with_const)
        c_stat = float(roc_auc_score(y, y_pred_prob))
    except Exception:
        c_stat = float("nan")

    # R code
    pred_formula = " + ".join(predictor_names)
    r_code = f"""library(rms)
dd <- datadist(data)
options(datadist = "dd")
model <- lrm({req.outcome} ~ {pred_formula}, data = data)
nom <- nomogram(model)
plot(nom)"""

    result_text = (
        f"Logistic regression nomogram with {len(predictor_names)} predictors "
        f"(N = {n}). C-statistic = {c_stat:.3f}. "
        f"AIC = {model.aic:.1f}. "
        f"Total points range from {min_total:.1f} to {max_total:.1f}, "
        f"corresponding to predicted probabilities of "
        f"{probability_mapping[0]['probability']:.3f} to {probability_mapping[-1]['probability']:.3f}."
    )

    return {
        "test": "Nomogram",
        "model_type": "logistic",
        "predictors_table": predictors_table,
        "probability_mapping": probability_mapping,
        "total_points_range": [round(float(min_total), 2), round(float(max_total), 2)],
        "model_summary": {
            "aic": round(float(model.aic), 2),
            "c_statistic": round(c_stat, 4),
            "n": n,
        },
        "result_text": result_text,
        "r_code": r_code,
    }


def _build_cox_nomogram(df: pd.DataFrame, req: NomogramRequest) -> dict:
    """Build nomogram from Cox proportional hazards regression."""
    from lifelines import CoxPHFitter

    # For Cox, outcome should be time-to-event; we need a duration and event column.
    # Convention: outcome column is the event indicator, and we look for a
    # companion duration column (e.g., outcome + "_time" or "time" or "duration").
    event_col = req.outcome
    duration_candidates = [
        f"{event_col}_time", "time", "duration", "survival_time",
        "follow_up", "followup", "TIME", "Time",
    ]
    duration_col = None
    for cand in duration_candidates:
        if cand in df.columns:
            duration_col = cand
            break

    if duration_col is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cox model requires a duration/time column. "
                f"Looked for: {duration_candidates}. "
                f"Please rename your time variable to '{event_col}_time' or 'time'."
            ),
        )

    all_cols = [event_col, duration_col] + req.predictors
    df_clean = _handle_missing(df, all_cols, req.imputation)

    for col in all_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean = df_clean.dropna()
    n = len(df_clean)

    if n < len(req.predictors) + 10:
        raise HTTPException(status_code=400, detail=f"Insufficient valid data: {n} rows")

    cph = CoxPHFitter()
    try:
        cph.fit(
            df_clean[req.predictors + [duration_col, event_col]],
            duration_col=duration_col,
            event_col=event_col,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cox model fitting failed: {str(e)}")

    coefficients = cph.params_.values
    predictor_names = req.predictors

    X_df = df_clean[predictor_names]
    references = X_df.mean().values

    abs_coefs = np.abs(coefficients)
    ranges = X_df.max().values - X_df.min().values
    max_contributions = abs_coefs * ranges
    max_contribution = max_contributions.max()
    scale = 100.0 / max_contribution if max_contribution > 0 else 1.0

    predictors_table = []
    for i, pred in enumerate(predictor_names):
        pred_min = float(X_df[pred].min())
        pred_max = float(X_df[pred].max())
        ref = float(references[i])
        coef = float(coefficients[i])

        n_values = min(11, max(5, int(X_df[pred].nunique())))
        values = np.linspace(pred_min, pred_max, n_values)
        points = (coef * (values - ref)) * scale

        predictors_table.append({
            "predictor": pred,
            "coefficient": round(coef, 4),
            "reference": round(ref, 4),
            "values": [round(float(v), 4) for v in values],
            "points": [round(float(p), 2) for p in points],
            "max_points": round(float(np.max(np.abs(points))), 2),
        })

    min_total = sum(min(p["points"]) for p in predictors_table)
    max_total = sum(max(p["points"]) for p in predictors_table)
    total_points_range = np.linspace(min_total, max_total, 20)

    # For Cox, convert total points to survival probability at median time
    baseline_survival = cph.baseline_survival_
    median_time_idx = len(baseline_survival) // 2
    S0 = float(baseline_survival.iloc[median_time_idx].values[0])

    probability_mapping = []
    for tp in total_points_range:
        lp = tp / scale  # linear predictor
        surv_prob = S0 ** np.exp(lp)
        probability_mapping.append({
            "total_points": round(float(tp), 2),
            "probability": round(float(surv_prob), 4),
        })

    c_stat = float(cph.concordance_index_)

    # AIC approximation: -2 * log-likelihood + 2k
    try:
        aic = float(-2 * cph.log_likelihood_ + 2 * len(predictor_names))
    except Exception:
        aic = float("nan")

    pred_formula = " + ".join(predictor_names)
    r_code = f"""library(rms)
dd <- datadist(data)
options(datadist = "dd")
model <- cph(Surv({duration_col}, {event_col}) ~ {pred_formula}, data = data)
nom <- nomogram(model)
plot(nom)"""

    result_text = (
        f"Cox regression nomogram with {len(predictor_names)} predictors "
        f"(N = {n}). C-statistic = {c_stat:.3f}. "
        f"AIC = {aic:.1f}. "
        f"Total points range from {min_total:.1f} to {max_total:.1f}, "
        f"corresponding to predicted survival probabilities of "
        f"{probability_mapping[0]['probability']:.3f} to {probability_mapping[-1]['probability']:.3f} "
        f"at the median follow-up time."
    )

    return {
        "test": "Nomogram",
        "model_type": "cox",
        "predictors_table": predictors_table,
        "probability_mapping": probability_mapping,
        "total_points_range": [round(float(min_total), 2), round(float(max_total), 2)],
        "model_summary": {
            "aic": round(aic, 2),
            "c_statistic": round(c_stat, 4),
            "n": n,
        },
        "result_text": result_text,
        "r_code": r_code,
    }
