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


# ── Causal mediation (X → M → Y) ──────────────────────────────────────────────

class MediationRequest(BaseModel):
    session_id: str
    outcome: str                       # continuous Y
    treatment: str                     # exposure X
    mediator: str                      # continuous M
    covariates: List[str] = []
    bootstrap: int = 1000              # nonparametric bootstrap reps for the ACME CI
    imputation: str = "listwise"


@router.post("/mediation")
def mediation(req: MediationRequest):
    """Linear causal mediation (Baron-Kenny / Preacher-Hayes) for a continuous
    mediator and outcome. Decomposes the total effect into the indirect effect
    through the mediator (ACME = a·b) and the direct effect (ADE = c'), with a
    percentile bootstrap CI for the indirect effect, a Sobel test, and the
    proportion mediated.
    """
    for nm, c in [("outcome", req.outcome), ("treatment", req.treatment), ("mediator", req.mediator)]:
        if not c:
            raise HTTPException(400, f"{nm} is required.")
    if len({req.outcome, req.treatment, req.mediator}) < 3:
        raise HTTPException(400, "Outcome, treatment, and mediator must be three distinct columns.")

    df_full = _get_df(req.session_id)
    cols = list(dict.fromkeys([req.outcome, req.treatment, req.mediator] + req.covariates))
    miss = [c for c in cols if c not in df_full.columns]
    if miss:
        raise HTTPException(400, f"Columns not found: {miss}")

    df = apply_imputation(df_full, cols, req.imputation)
    for c in [req.outcome, req.treatment, req.mediator]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)
    n = len(df)
    if n < len(cols) + 10:
        raise HTTPException(400, "Not enough complete observations for mediation analysis.")

    def _fit(d: pd.DataFrame):
        Wd = _design(d, req.covariates)
        Xm = sm.add_constant(pd.concat([d[[req.treatment]].astype(float), Wd], axis=1), has_constant="add")
        m_model = sm.OLS(d[req.mediator].astype(float).values, Xm).fit()
        a, a_se = float(m_model.params[req.treatment]), float(m_model.bse[req.treatment])
        Xy = sm.add_constant(pd.concat([d[[req.treatment, req.mediator]].astype(float), Wd], axis=1), has_constant="add")
        y_model = sm.OLS(d[req.outcome].astype(float).values, Xy).fit()
        b, b_se = float(y_model.params[req.mediator]), float(y_model.bse[req.mediator])
        cprime = float(y_model.params[req.treatment])
        return a, a_se, b, b_se, cprime

    a, a_se, b, b_se, cprime = _fit(df)
    acme = a * b
    ade = cprime
    total = acme + ade
    prop_med = (acme / total) if abs(total) > 1e-9 else float("nan")

    sobel_se = float(np.sqrt(b * b * a_se * a_se + a * a * b_se * b_se))
    sobel_z = float(acme / sobel_se) if sobel_se > 0 else float("nan")
    sobel_p = float(2 * sp.norm.sf(abs(sobel_z))) if np.isfinite(sobel_z) else float("nan")

    reps = int(req.bootstrap)
    acme_ci = ade_ci = total_ci = prop_ci = None
    if reps and reps >= 100:
        rng = np.random.default_rng(42)
        idx = np.arange(n)
        a_bs, ad_bs, t_bs, pm_bs = [], [], [], []
        dfr = df.reset_index(drop=True)
        for _ in range(reps):
            bi = rng.choice(idx, size=n, replace=True)
            try:
                ab, _, bb, _, cb = _fit(dfr.iloc[bi])
            except Exception:
                continue
            ind = ab * bb
            tot = ind + cb
            a_bs.append(ind); ad_bs.append(cb); t_bs.append(tot)
            if abs(tot) > 1e-9:
                pm_bs.append(ind / tot)
        if a_bs:
            def q(arr):
                return [round(float(np.quantile(arr, 0.025)), 5), round(float(np.quantile(arr, 0.975)), 5)]
            acme_ci, ade_ci, total_ci = q(a_bs), q(ad_bs), q(t_bs)
            prop_ci = q(pm_bs) if pm_bs else None

    acme_sig = bool(acme_ci is not None and (acme_ci[0] > 0 or acme_ci[1] < 0))
    result_text = (
        f"Causal mediation of {req.treatment} → {req.mediator} → {req.outcome} (n = {n}"
        + (f", adjusted for {', '.join(req.covariates)}" if req.covariates else "") + "). "
        f"Indirect effect (ACME = a·b) = {acme:.4f}"
        + (f" (95% bootstrap CI {acme_ci[0]} to {acme_ci[1]})" if acme_ci else "")
        + f"; direct effect (ADE) = {ade:.4f}; total effect = {total:.4f}. "
        f"Proportion mediated = {prop_med*100:.1f}%"
        + (f" (95% CI {prop_ci[0]*100:.1f}% to {prop_ci[1]*100:.1f}%)" if prop_ci else "") + ". "
        f"Sobel z = {sobel_z:.2f}, p = {_p_str(sobel_p)}. "
        + ("The indirect (mediated) effect is statistically significant (bootstrap CI excludes 0)."
           if acme_sig else "The indirect (mediated) effect is not statistically significant.")
    )

    return {
        "test": "Causal Mediation (linear)",
        "outcome": req.outcome, "treatment": req.treatment, "mediator": req.mediator,
        "covariates": req.covariates, "n": int(n),
        "paths": {"a": round(a, 6), "a_se": round(a_se, 6), "b": round(b, 6),
                  "b_se": round(b_se, 6), "c_prime": round(cprime, 6)},
        "effects": {
            "acme": round(acme, 6), "acme_ci": acme_ci,
            "ade": round(ade, 6), "ade_ci": ade_ci,
            "total": round(total, 6), "total_ci": total_ci,
            "proportion_mediated": round(prop_med, 4) if np.isfinite(prop_med) else None,
            "proportion_mediated_ci": prop_ci,
        },
        "sobel": {"z": round(sobel_z, 4), "p": sobel_p, "se": round(sobel_se, 6)},
        "acme_significant": acme_sig,
        "result_text": result_text,
        "interpretation": result_text,
        "r_code": (
            f'library(mediation)\n'
            f'm.med <- lm({req.mediator} ~ {req.treatment}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "") + ', data = data)\n'
            f'm.out <- lm({req.outcome} ~ {req.treatment} + {req.mediator}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "") + ', data = data)\n'
            f'med <- mediate(m.med, m.out, treat = "{req.treatment}", mediator = "{req.mediator}", boot = TRUE)\n'
            f'summary(med)  # ACME, ADE, total effect, proportion mediated'
        ),
    }
