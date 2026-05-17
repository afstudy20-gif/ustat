"""Decision curve analysis and calibration for prediction models."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp
from sklearn.metrics import brier_score_loss, roc_auc_score
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services import store
from services.impute import apply_imputation

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _hosmer_lemeshow(y: np.ndarray, probs: np.ndarray, n_groups: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test.

    Bins observations by deciles of predicted probability (default g=10).
    χ² = Σ (O_g − E_g)² / [E_g (1 − E_g/n_g)]
    df = g − 2
    p > 0.05 ⇒ model fits the data (failure to reject good-fit null).

    Returns dict with chi2, df, p, n_groups_used, and the per-group table
    (predicted mean, observed events, expected events, n).
    """
    from scipy.stats import chi2 as _chi2

    y = np.asarray(y, dtype=float)
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    if n < n_groups * 2:
        # Not enough data; cap groups so each bin has ≥ 2 observations.
        n_groups = max(2, min(n_groups, n // 2))

    # Sort by predicted probability, then bin into equal-sized groups.
    order = np.argsort(probs)
    y_s = y[order]
    p_s = probs[order]
    # Use np.array_split for near-equal bin sizes even when n is not divisible.
    idx_split = np.array_split(np.arange(n), n_groups)

    chi2_val = 0.0
    groups = []
    used_groups = 0
    for idx_arr in idx_split:
        if len(idx_arr) == 0:
            continue
        used_groups += 1
        obs = float(y_s[idx_arr].sum())
        exp = float(p_s[idx_arr].sum())
        ng = int(len(idx_arr))
        # Denominator zero-guard: if expected events are 0 OR equal to n_g
        # (everyone perfectly predicted), the bin's contribution to χ² is 0.
        denom = exp * (1 - exp / ng) if ng > 0 else 0.0
        if denom > 0:
            chi2_val += (obs - exp) ** 2 / denom
        groups.append({
            "predicted_mean": round(float(p_s[idx_arr].mean()), 4),
            "observed_events": int(obs),
            "expected_events": round(exp, 4),
            "n": ng,
        })

    df_val = max(used_groups - 2, 1)
    p_val = float(1 - _chi2.cdf(chi2_val, df_val))
    return {
        "chi2": round(chi2_val, 4),
        "df": df_val,
        "p": p_val,
        "n_groups": used_groups,
        "interpretation": "Good fit (p > 0.05)" if p_val > 0.05 else "Poor fit (p ≤ 0.05) — model misspecified",
        "groups": groups,
    }


def _fit_logistic(df: pd.DataFrame, outcome: str, predictors: List[str]):
    """Fit logistic regression and return model + predicted probabilities."""
    X = pd.get_dummies(df[predictors], drop_first=True).astype(float)
    X = sm.add_constant(X)
    y = df[outcome].astype(float)

    if y.nunique() < 2:
        raise HTTPException(400, "Outcome must be binary (0/1) with at least one event and one non-event.")
    if not set(y.unique()).issubset({0, 1, 0.0, 1.0}):
        raise HTTPException(400, "Outcome must be coded as 0/1 for calibration analysis.")

    try:
        model = sm.Logit(y, X).fit(disp=False, maxiter=100)
    except Exception as exc:
        raise HTTPException(400, f"Logistic regression failed: {exc}")

    probs = model.predict(X)
    return model, X, y.values, probs.values


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CALIBRATION PLOT
# ═══════════════════════════════════════════════════════════════════════════════

class CalibrationRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    n_bins: int = 10
    imputation: str = "listwise"


@router.post("/calibration")
def calibration(req: CalibrationRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation)
    n_excluded = n_total - len(df)

    if len(df) < 20:
        raise HTTPException(400, "Need at least 20 complete observations for calibration analysis.")

    model, X, y, probs = _fit_logistic(df, req.outcome, req.predictors)

    # ── Bin predictions ────────────────────────────────────────────────────
    n_bins = max(2, min(req.n_bins, len(df) // 5))  # at least 5 per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    predicted_means = []
    observed_props = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)

        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue

        pred_mean = float(probs[mask].mean())
        obs_events = int(y[mask].sum())
        obs_prop = obs_events / n_in_bin

        # Wilson CI for observed proportion
        if n_in_bin > 0:
            ci_low, ci_high = _wilson_ci(obs_events, n_in_bin)
        else:
            ci_low, ci_high = 0.0, 1.0

        bins.append({
            "predicted_mean": round(pred_mean, 4),
            "observed_prop": round(obs_prop, 4),
            "n": n_in_bin,
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
        })
        predicted_means.append(pred_mean)
        observed_props.append(obs_prop)

    # ── Calibration slope & intercept ──────────────────────────────────────
    if len(predicted_means) >= 2:
        cal_X = sm.add_constant(np.array(predicted_means))
        cal_model = sm.OLS(np.array(observed_props), cal_X).fit()
        cal_intercept = round(float(cal_model.params[0]), 4)
        cal_slope = round(float(cal_model.params[1]), 4)
    else:
        cal_intercept = 0.0
        cal_slope = 1.0

    # ── E/O ratio ──────────────────────────────────────────────────────────
    expected_events = float(probs.sum())
    observed_events = float(y.sum())
    eo_ratio = round(expected_events / observed_events, 4) if observed_events > 0 else float("inf")

    # ── Discrimination metrics ─────────────────────────────────────────────
    brier = round(float(brier_score_loss(y, probs)), 4)
    try:
        c_stat = round(float(roc_auc_score(y, probs)), 4)
    except ValueError:
        c_stat = None

    # ── Result text ────────────────────────────────────────────────────────
    slope_interp = "well-calibrated" if 0.8 <= cal_slope <= 1.2 else ("overfitting" if cal_slope < 0.8 else "underfitting")
    # Hosmer-Lemeshow goodness-of-fit test (deciles).
    hl = _hosmer_lemeshow(y, probs, n_groups=10)

    result_text = (
        f"Calibration analysis of {req.outcome} predicted by {', '.join(req.predictors)} "
        f"(n = {len(df)}, {n_excluded} excluded). "
        f"Calibration slope = {cal_slope}, intercept = {cal_intercept} ({slope_interp}). "
        f"E/O ratio = {eo_ratio}. Brier score = {brier}. "
        f"Hosmer-Lemeshow χ²({hl['df']}) = {hl['chi2']}, p = {_p_str(hl['p'])}."
    )
    if c_stat is not None:
        result_text += f" C-statistic (AUC) = {c_stat}."

    preds_col = ", ".join(req.predictors)

    return {
        "test": "Calibration Analysis",
        "bins": bins,
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
        "eo_ratio": eo_ratio,
        "brier_score": brier,
        "hosmer_lemeshow": hl,
        "c_statistic": c_stat,
        "n": len(df),
        "n_excluded": n_excluded,
        "plot_data": {
            "predicted": predicted_means,
            "observed": observed_props,
            "identity_line": [0, 1],
        },
        "result_text": result_text,
        "export_rows": [
            ["Statistic", "Value"],
            ["Calibration slope", cal_slope],
            ["Calibration intercept", cal_intercept],
            ["E/O ratio", eo_ratio],
            ["Brier score", brier],
            ["C-statistic (AUC)", c_stat],
            ["n", len(df)],
            ["n excluded", n_excluded],
            *[
                [f"Bin {i+1}: predicted={b['predicted_mean']}, observed={b['observed_prop']}", f"n={b['n']}"]
                for i, b in enumerate(bins)
            ],
        ],
        "r_code": f"library(rms)\nval.prob(predicted, observed)",
    }


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = sp.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DECISION CURVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 1b. HOSMER-LEMESHOW (standalone)
# ═══════════════════════════════════════════════════════════════════════════════

class HLRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    n_groups: int = 10
    imputation: str = "listwise"


@router.post("/hosmer_lemeshow")
def hosmer_lemeshow_endpoint(req: HLRequest):
    """Standalone Hosmer-Lemeshow goodness-of-fit for a logistic model.

    Fits the logistic model from the predictors, bins predicted probabilities
    into n_groups deciles, returns χ²/df/p plus the per-group observed vs
    expected table. p > 0.05 ⇒ model fits.
    """
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation)
    n_excluded = n_total - len(df)

    if len(df) < req.n_groups * 2:
        raise HTTPException(400, f"Need at least {req.n_groups * 2} complete observations for an H-L test with {req.n_groups} groups.")

    _, _, y, probs = _fit_logistic(df, req.outcome, req.predictors)
    hl = _hosmer_lemeshow(y, probs, n_groups=req.n_groups)
    return {
        "test": "Hosmer-Lemeshow Goodness-of-Fit",
        **hl,
        "n": int(len(df)),
        "n_excluded": int(n_excluded),
        "outcome": req.outcome,
        "predictors": req.predictors,
        "result_text": (
            f"Hosmer-Lemeshow χ²({hl['df']}) = {hl['chi2']}, p = {_p_str(hl['p'])}. "
            f"{hl['interpretation']}."
        ),
    }


class DCARequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    threshold_range: List[float] = [0.01, 0.99]
    n_thresholds: int = 100
    imputation: str = "listwise"


@router.post("/dca")
def dca(req: DCARequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation)
    n_excluded = n_total - len(df)

    if len(df) < 20:
        raise HTTPException(400, "Need at least 20 complete observations for DCA.")

    model, X, y, probs = _fit_logistic(df, req.outcome, req.predictors)

    prevalence = float(y.mean())
    n = len(y)

    thresholds = np.linspace(
        max(0.001, req.threshold_range[0]),
        min(0.999, req.threshold_range[1]),
        req.n_thresholds,
    )

    model_nb = []
    all_nb = []
    none_nb = []

    for pt in thresholds:
        # Model net benefit
        pred_pos = probs >= pt
        tp = float((pred_pos & (y == 1)).sum()) / n
        fp = float((pred_pos & (y == 0)).sum()) / n
        nb = tp - fp * pt / (1 - pt) if pt < 1 else 0.0
        model_nb.append(round(float(nb), 6))

        # Treat-all net benefit
        nb_all = prevalence - (1 - prevalence) * pt / (1 - pt) if pt < 1 else 0.0
        all_nb.append(round(float(nb_all), 6))

        # Treat-none net benefit
        none_nb.append(0.0)

    thresholds_list = [round(float(t), 4) for t in thresholds]

    # ── Find range where model has positive net benefit ────────────────────
    useful_range = [t for t, nb in zip(thresholds_list, model_nb) if nb > 0]
    if useful_range:
        useful_min = min(useful_range)
        useful_max = max(useful_range)
        range_text = f"The model has positive net benefit across threshold probabilities {useful_min:.2f} to {useful_max:.2f}."
    else:
        range_text = "The model does not show positive net benefit at any threshold."

    preds_col = ", ".join(req.predictors)
    result_text = (
        f"Decision curve analysis for {req.outcome} predicted by {preds_col} "
        f"(n = {n}, prevalence = {prevalence:.3f}, {n_excluded} excluded). "
        f"{range_text}"
    )

    return {
        "test": "Decision Curve Analysis",
        "curves": {
            "model": {"thresholds": thresholds_list, "net_benefit": model_nb},
            "treat_all": {"thresholds": thresholds_list, "net_benefit": all_nb},
            "treat_none": {"thresholds": thresholds_list, "net_benefit": none_nb},
        },
        "prevalence": round(prevalence, 4),
        "n": n,
        "n_excluded": n_excluded,
        "result_text": result_text,
        "export_rows": [
            ["Threshold", "Model NB", "Treat All NB", "Treat None NB"],
            *[
                [thresholds_list[i], model_nb[i], all_nb[i], none_nb[i]]
                for i in range(len(thresholds_list))
            ],
        ],
        "r_code": f"library(dcurves)\ndca(outcome ~ {' + '.join(req.predictors)}, data = data)",
    }
