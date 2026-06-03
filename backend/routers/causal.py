"""Causal-inference methods that complement PSM/IPTW/E-value:

  * /iv_2sls          — instrumental-variable estimation (two-stage least
                        squares) with weak-instrument F, Wu-Hausman endogeneity
                        test, and Sargan over-identification test.

(Additive router — does not touch the existing PSM/IPTW/sensitivity panels.)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


def _design(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Numeric design block (categoricals → dummies, drop_first)."""
    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[cols], drop_first=True).astype(float)


class IV2SLSRequest(BaseModel):
    session_id: str
    outcome: str                       # continuous
    endogenous: str                    # the treatment/exposure suspected endogenous
    instruments: List[str]             # ≥1 instrument(s)
    covariates: List[str] = []         # exogenous controls
    imputation: str = "listwise"


@router.post("/iv_2sls")
def iv_2sls(req: IV2SLSRequest):
    """Two-stage least squares IV estimator for a continuous outcome with one
    endogenous regressor. Reports the IV (2SLS) effect with correct SE, the
    first-stage weak-instrument F, the Wu-Hausman endogeneity test, the Sargan
    over-identification test (when over-identified), and the naive OLS estimate
    for contrast.
    """
    if not req.instruments:
        raise HTTPException(400, "Provide at least one instrument.")
    if req.endogenous in req.instruments:
        raise HTTPException(400, "The endogenous variable cannot also be an instrument.")
    if req.endogenous in req.covariates:
        raise HTTPException(400, "The endogenous variable cannot also be a covariate.")

    df_full = _get_df(req.session_id)
    cols = list(dict.fromkeys([req.outcome, req.endogenous] + req.instruments + req.covariates))
    missing = [c for c in cols if c not in df_full.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")

    df = apply_imputation(df_full, cols, req.imputation)
    df[req.outcome] = pd.to_numeric(df[req.outcome], errors="coerce")
    df[req.endogenous] = pd.to_numeric(df[req.endogenous], errors="coerce")
    df = df.dropna(subset=cols)
    n = len(df)
    if n < len(cols) + 10:
        raise HTTPException(400, "Not enough complete observations for IV estimation.")

    y = df[req.outcome].astype(float).values
    X = df[[req.endogenous]].astype(float).values            # n×1 endogenous
    W = _design(df, req.covariates)                           # exogenous controls
    Z = _design(df, req.instruments)                          # instruments
    n_instr = Z.shape[1]
    if n_instr < 1:
        raise HTTPException(400, "Instruments produced no usable columns.")

    Wc = sm.add_constant(W, has_constant="add")               # add intercept
    exog_2nd_names = list(Wc.columns) + [req.endogenous]

    # ── First stage: endogenous ~ controls + instruments ─────────────────────
    fs_exog = pd.concat([Wc, Z], axis=1).astype(float)
    fs = sm.OLS(X.ravel(), fs_exog).fit()
    Xhat = fs.fittedvalues.values
    v = fs.resid.values                                       # first-stage residuals

    # Weak-instrument test: joint F for the excluded instruments.
    try:
        f_res = fs.f_test([f"{c} = 0" for c in Z.columns])
        first_stage_F = float(np.ravel(f_res.fvalue)[0])
        first_stage_F_p = float(f_res.pvalue)
    except Exception:
        first_stage_F, first_stage_F_p = float("nan"), float("nan")
    weak = bool(np.isfinite(first_stage_F) and first_stage_F < 10.0)

    # ── Second stage: outcome ~ controls + fitted endogenous ─────────────────
    ss_exog = pd.concat([Wc, pd.Series(Xhat, index=df.index, name=req.endogenous)], axis=1).astype(float)
    ss = sm.OLS(y, ss_exog).fit()
    beta = ss.params.values

    # Correct 2SLS covariance: residuals use the ACTUAL endogenous, not fitted.
    ss_actual = pd.concat([Wc, df[[req.endogenous]].astype(float)], axis=1).astype(float)
    resid = y - ss_actual.values @ beta
    k = ss_exog.shape[1]
    sigma2 = float(resid @ resid) / (n - k)
    XtX = ss_exog.values.T @ ss_exog.values
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        raise HTTPException(400, "Design matrix is singular — check for collinear instruments/covariates.")
    Vbeta = sigma2 * XtX_inv
    se = np.sqrt(np.clip(np.diag(Vbeta), 0, None))
    tvals = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = 2 * sp.t.sf(np.abs(tvals), df=n - k)
    tcrit = float(sp.t.ppf(0.975, n - k))

    coefs = []
    endo_row = None
    for i, name in enumerate(exog_2nd_names):
        row = {
            "variable": name,
            "estimate": round(float(beta[i]), 6),
            "se": round(float(se[i]), 6),
            "t": round(float(tvals[i]), 4),
            "p": float(pvals[i]),
            "ci_low": round(float(beta[i] - tcrit * se[i]), 6),
            "ci_high": round(float(beta[i] + tcrit * se[i]), 6),
        }
        coefs.append(row)
        if name == req.endogenous:
            endo_row = row

    # ── Naive OLS (for contrast — biased if the regressor is endogenous) ─────
    ols = sm.OLS(y, ss_actual).fit()
    ols_b = float(ols.params.get(req.endogenous, np.nan))
    ols_se = float(ols.bse.get(req.endogenous, np.nan))
    ols_p = float(ols.pvalues.get(req.endogenous, np.nan))

    # ── Wu-Hausman endogeneity test (control-function form) ──────────────────
    aug = pd.concat([ss_actual, pd.Series(v, index=df.index, name="_resid_v_")], axis=1).astype(float)
    haus = sm.OLS(y, aug).fit()
    wu_p = float(haus.pvalues.get("_resid_v_", np.nan))
    wu_t = float(haus.tvalues.get("_resid_v_", np.nan))
    endogenous_flag = bool(np.isfinite(wu_p) and wu_p < 0.05)

    # ── Sargan over-identification test (only if over-identified) ────────────
    sargan = None
    if n_instr > 1:
        sg = sm.OLS(resid, pd.concat([Wc, Z], axis=1).astype(float)).fit()
        sargan_stat = float(n * sg.rsquared)
        sargan_df = int(n_instr - 1)
        sargan_p = float(sp.chi2.sf(sargan_stat, sargan_df))
        sargan = {"stat": round(sargan_stat, 4), "df": sargan_df, "p": sargan_p,
                  "valid": bool(sargan_p >= 0.05)}

    iv_b = endo_row["estimate"]
    iv_p = endo_row["p"]
    result_text = (
        f"Instrumental-variable (2SLS) estimate of the effect of {req.endogenous} on {req.outcome} "
        f"using {', '.join(req.instruments)} as instrument(s) (n = {n}). "
        f"IV effect = {iv_b:.4f} (95% CI {endo_row['ci_low']:.4f} to {endo_row['ci_high']:.4f}, p = {_p_str(iv_p)}); "
        f"naive OLS = {ols_b:.4f} (p = {_p_str(ols_p)}). "
        f"First-stage F = {first_stage_F:.1f} "
        + ("(WEAK instruments, F < 10 — IV estimate unreliable). " if weak else "(instruments adequate, F ≥ 10). ")
        + f"Wu-Hausman endogeneity p = {_p_str(wu_p)} "
        + ("→ endogeneity present, IV preferred over OLS. " if endogenous_flag
           else "→ no strong evidence of endogeneity; OLS may suffice. ")
        + (f"Sargan over-identification p = {_p_str(sargan['p'])} "
           + ("(instruments jointly valid)." if sargan["valid"] else "(instrument validity in doubt).")
           if sargan else "Just-identified — over-identification not testable.")
    )

    return {
        "test": "Instrumental Variable (2SLS)",
        "outcome": req.outcome,
        "endogenous": req.endogenous,
        "instruments": req.instruments,
        "covariates": req.covariates,
        "n": int(n),
        "iv_estimate": endo_row,
        "coefficients": coefs,
        "ols_estimate": {"estimate": round(ols_b, 6), "se": round(ols_se, 6), "p": ols_p},
        "first_stage": {"f_stat": round(first_stage_F, 4), "f_p": first_stage_F_p,
                        "weak_instruments": weak, "n_instruments": int(n_instr)},
        "wu_hausman": {"t": round(wu_t, 4), "p": wu_p, "endogenous": endogenous_flag},
        "sargan": sargan,
        "result_text": result_text,
        "interpretation": result_text,
        "r_code": (
            f'library(AER)\n'
            f'iv <- ivreg({req.outcome} ~ {req.endogenous}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "")
            + f' | {" + ".join(req.instruments)}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "")
            + f', data = data)\n'
            f'summary(iv, diagnostics = TRUE)  # weak instruments, Wu-Hausman, Sargan'
        ),
    }
