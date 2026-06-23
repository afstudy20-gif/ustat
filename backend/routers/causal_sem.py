"""SEM / Path analysis endpoint (PROCESS Models 4/5/6/80/81 equivalent).

Supports multi-treatment / multi-mediator (parallel or serial) / multi-outcome
mediation designs via ``semopy`` (lavaan-style syntax). Reports labeled path
coefficients, indirect/direct/total effects with percentile bootstrap CIs, and
global fit (chi2, CFI, TLI, RMSEA, SRMR). Power user override: free-text
``lavaan_spec`` bypasses the structure builder entirely.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import semopy
import semopy.stats as smstats

from services import store
from services.impute import apply_imputation


router = APIRouter()


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _safe(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _round(x, nd: int = 4) -> Optional[float]:
    v = _safe(x)
    return None if v is None else round(v, nd)


# ── request schema ───────────────────────────────────────────────────────────

class SEMRequest(BaseModel):
    session_id: str
    treatments: List[str] = []
    mediators: List[str] = []
    outcomes: List[str] = []
    covariates: List[str] = []
    serial: bool = False
    bootstrap: int = 5000
    imputation: str = "listwise"
    lavaan_spec: Optional[str] = None


# ── model builder ────────────────────────────────────────────────────────────

def _safe_label(name: str) -> str:
    """semopy parameter labels must be plain identifiers."""
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "v"


def _build_spec(
    treatments: List[str],
    mediators: List[str],
    outcomes: List[str],
    covariates: List[str],
    serial: bool,
) -> Tuple[str, List[Dict], List[Dict]]:
    """Return (lavaan_spec, indirect_chains, direct_pairs).

    Each indirect_chain dict: {label, treatment, chain[List[str]], outcome,
                                path_labels [List[str]]}
    Each direct_pair dict:    {treatment, outcome, label}
    """
    lines: List[str] = []
    indirect_chains: List[Dict] = []
    direct_pairs: List[Dict] = []
    cov_part = (" + " + " + ".join(covariates)) if covariates else ""

    if serial and len(mediators) >= 2:
        # Serial chain: X -> M1 -> M2 -> ... -> Mn -> Y, with direct X -> Y.
        # M1 regressed on all X (+ cov). Mk (k>=2) regressed on M_{k-1} only (+ cov).
        # Each Y regressed on the LAST mediator + every X (direct paths).
        for ti, X in enumerate(treatments):
            lines.append(f"{mediators[0]} ~ a_{ti}_0*{X}{cov_part}")
        for j in range(1, len(mediators)):
            lines.append(f"{mediators[j]} ~ s_{j-1}_{j}*{mediators[j-1]}{cov_part}")
        for yi, Y in enumerate(outcomes):
            rhs_terms = [f"cp_{ti}_{yi}*{X}" for ti, X in enumerate(treatments)]
            rhs_terms.append(f"b_{yi}_last*{mediators[-1]}")
            lines.append(f"{Y} ~ " + " + ".join(rhs_terms) + cov_part)
        chain_path = mediators[:]
        for ti, X in enumerate(treatments):
            for yi, Y in enumerate(outcomes):
                labs = ([f"a_{ti}_0"]
                        + [f"s_{j-1}_{j}" for j in range(1, len(mediators))]
                        + [f"b_{yi}_last"])
                indirect_chains.append({
                    "label": f"{X} -> " + " -> ".join(mediators) + f" -> {Y}",
                    "treatment": X, "chain": chain_path, "outcome": Y, "path_labels": labs,
                })
                direct_pairs.append({"treatment": X, "outcome": Y, "label": f"cp_{ti}_{yi}"})
    else:
        # Parallel mediators (or single mediator)
        # Mk ~ a_{ti,k}*X (+ cov)  — for each (treatment, mediator)
        for ti, X in enumerate(treatments):
            for mi, M in enumerate(mediators):
                a_lab = f"a_{ti}_{mi}"
                lines.append(f"{M} ~ {a_lab}*{X}{cov_part}")
        # Y_yi ~ cp_{ti,yi}*X + b_{mi,yi}*M  (one regression per outcome with all X and all M)
        for yi, Y in enumerate(outcomes):
            rhs = []
            for ti, X in enumerate(treatments):
                rhs.append(f"cp_{ti}_{yi}*{X}")
            for mi, M in enumerate(mediators):
                rhs.append(f"b_{mi}_{yi}*{M}")
            lines.append(f"{Y} ~ " + " + ".join(rhs) + cov_part)
        # Indirect chains
        for ti, X in enumerate(treatments):
            for mi, M in enumerate(mediators):
                for yi, Y in enumerate(outcomes):
                    indirect_chains.append({
                        "label": f"{X} -> {M} -> {Y}",
                        "treatment": X, "chain": [M], "outcome": Y,
                        "path_labels": [f"a_{ti}_{mi}", f"b_{mi}_{yi}"],
                    })
                    # direct_pair tracked per (X,Y); dedupe later
        seen = set()
        for ti, X in enumerate(treatments):
            for yi, Y in enumerate(outcomes):
                key = (X, Y)
                if key in seen:
                    continue
                seen.add(key)
                direct_pairs.append({"treatment": X, "outcome": Y, "label": f"cp_{ti}_{yi}"})

    spec = "\n".join(lines)
    return spec, indirect_chains, direct_pairs


def _fit_and_extract(spec: str, df: pd.DataFrame) -> Dict[str, float]:
    """Fit a SEM and return a {label: estimate} dict for ~labels and a parsed
    inspect frame stash (for path table)."""
    model = semopy.Model(spec)
    model.fit(df)
    insp = model.inspect()
    # semopy doesn't expose the parameter LABELS through inspect — only (lval, op, rval).
    # We must map labels → (lval, rval) by re-parsing the spec.
    label_map = _extract_labels_from_spec(spec)  # {label: (lval, rval)}
    estimates: Dict[str, float] = {}
    for lab, (lval, rval) in label_map.items():
        row = insp[(insp["lval"] == lval) & (insp["op"] == "~") & (insp["rval"] == rval)]
        if len(row) == 1:
            estimates[lab] = float(row.iloc[0]["Estimate"])
        else:
            estimates[lab] = float("nan")
    return estimates, insp, model


_LABEL_RE = None
def _extract_labels_from_spec(spec: str) -> Dict[str, Tuple[str, str]]:
    """Walk lines like 'Y ~ a*X + b*M + c*Z' and return {a:(Y,X), b:(Y,M), c:(Y,Z)}.
    Unlabeled rhs terms are ignored (no covariate labeling).
    """
    out: Dict[str, Tuple[str, str]] = {}
    for raw in spec.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "~" not in line:
            continue
        lhs, rhs = line.split("~", 1)
        lval = lhs.strip()
        for term in rhs.split("+"):
            t = term.strip()
            if "*" in t:
                lab, rval = [s.strip() for s in t.split("*", 1)]
                if lab and rval:
                    out[lab] = (lval, rval)
    return out


def _srmr(model) -> Optional[float]:
    """Standardised Root Mean square Residual. semopy doesn't ship one; we
    compute from the observed vs. implied covariance matrices."""
    try:
        S = model.mx_cov            # observed cov
        Sigma = model.calc_sigma()[0]  # implied cov
        d = S.shape[0]
        std = np.sqrt(np.diag(S))
        # standardised residuals (S - Sigma) / (std_i * std_j), elementwise
        denom = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = (S - Sigma) / denom
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        # Take lower-triangle + diagonal
        idx = np.tril_indices(d)
        vals = R[idx]
        return float(np.sqrt(np.mean(vals ** 2)))
    except Exception:
        return None


# ── endpoint ─────────────────────────────────────────────────────────────────

@router.post("/sem")
def sem_path(req: SEMRequest):
    """Fit a structural equation / path model. Equivalent to PROCESS Models
    4 (parallel mediation), 6 (serial mediation), and beyond; also supports
    multi-outcome and multi-treatment designs that PROCESS cannot fit in a
    single model.
    """
    # ── validation ──────────────────────────────────────────────────────────
    if not req.lavaan_spec:
        if not req.treatments:
            raise HTTPException(400, "At least one treatment is required.")
        if not req.mediators:
            raise HTTPException(400, "At least one mediator is required.")
        if not req.outcomes:
            raise HTTPException(400, "At least one outcome is required.")
        all_vars = list(dict.fromkeys(req.treatments + req.mediators + req.outcomes + req.covariates))
        # No variable may play two roles
        roles = {"treatment": set(req.treatments), "mediator": set(req.mediators),
                 "outcome": set(req.outcomes), "covariate": set(req.covariates)}
        seen_role = {}
        for r, names in roles.items():
            for n in names:
                if n in seen_role:
                    raise HTTPException(400, f"Variable {n!r} appears as both {seen_role[n]} and {r}.")
                seen_role[n] = r
    else:
        all_vars = []

    df_full = _get_df(req.session_id)

    if req.lavaan_spec:
        # Discover model variables from the spec (lval / rval tokens). Crude
        # but works: anything appearing as a side of '~' or '~~' is a measured
        # variable in this path-only application.
        toks = set()
        for line in req.lavaan_spec.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for chunk in s.replace("~~", "~").split("~"):
                for term in chunk.split("+"):
                    t = term.strip()
                    if "*" in t:
                        t = t.split("*", 1)[1].strip()
                    if t and not t.replace(".", "").replace("-", "").isdigit():
                        toks.add(t)
        all_vars = [t for t in toks if t in df_full.columns]
        if not all_vars:
            raise HTTPException(400, "lavaan spec references no columns present in this dataset.")

    miss = [c for c in all_vars if c not in df_full.columns]
    if miss:
        raise HTTPException(400, f"Columns not found: {miss}")

    df = apply_imputation(df_full, all_vars, req.imputation)
    for c in all_vars:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=all_vars)
    n = len(df)
    if n < len(all_vars) + 10:
        raise HTTPException(400, "Not enough complete observations for the SEM model.")

    # Continuous outcome check (mirrors mediation endpoint)
    if not req.lavaan_spec:
        for Y in req.outcomes:
            y_levels = sorted(df[Y].dropna().unique().tolist())
            if len(y_levels) == 2 and set(y_levels).issubset({0, 1, 0.0, 1.0}):
                raise HTTPException(
                    status_code=422,
                    detail=(f"Outcome {Y!r} is binary. Linear SEM requires continuous outcomes; "
                            "binary outcomes need a probit/logit SEM, which is not implemented."),
                )

    # ── build spec ──────────────────────────────────────────────────────────
    if req.lavaan_spec:
        spec = req.lavaan_spec.strip()
        # In free-form mode we cannot infer indirect chains automatically
        indirect_chains: List[Dict] = []
        direct_pairs: List[Dict] = []
    else:
        spec, indirect_chains, direct_pairs = _build_spec(
            req.treatments, req.mediators, req.outcomes, req.covariates, req.serial,
        )

    # ── fit ─────────────────────────────────────────────────────────────────
    try:
        estimates, insp, model = _fit_and_extract(spec, df)
    except Exception as exc:
        raise HTTPException(400, f"SEM fit failed: {exc}")

    # ── path table ──────────────────────────────────────────────────────────
    paths = []
    # Reverse-map (lval, rval) → label for prettier rows when available
    label_map = _extract_labels_from_spec(spec)
    inv = {v: k for k, v in label_map.items()}
    for _, row in insp.iterrows():
        if row["op"] != "~":
            continue
        lab = inv.get((row["lval"], row["rval"]))
        paths.append({
            "label": lab,
            "from": row["rval"],
            "to": row["lval"],
            "est": _round(row["Estimate"], 6),
            "se": _round(row["Std. Err"], 6),
            "z": _round(row["z-value"], 4),
            "p": _safe(row["p-value"]),
        })

    # ── direct effects (point estimates) ────────────────────────────────────
    def _row(lval: str, rval: str):
        r = insp[(insp["lval"] == lval) & (insp["op"] == "~") & (insp["rval"] == rval)]
        return None if r.empty else r.iloc[0]

    direct_effects = []
    for d in direct_pairs:
        r = _row(d["outcome"], d["treatment"])
        if r is None:
            continue
        est = _safe(r["Estimate"]); se = _safe(r["Std. Err"]); p = _safe(r["p-value"])
        ci = None
        if est is not None and se is not None and np.isfinite(se):
            ci = [round(est - 1.96 * se, 6), round(est + 1.96 * se, 6)]
        direct_effects.append({
            "treatment": d["treatment"], "outcome": d["outcome"],
            "est": _round(est, 6), "se": _round(se, 6), "p": p, "ci": ci,
        })

    # ── point estimates of indirect effects + total ─────────────────────────
    def _indirect_pt(labels: List[str], ests: Dict[str, float]) -> float:
        v = 1.0
        for lab in labels:
            v *= ests.get(lab, float("nan"))
        return v

    indirect_pt = {}
    for ch in indirect_chains:
        indirect_pt[ch["label"]] = _indirect_pt(ch["path_labels"], estimates)

    total_pt: Dict[Tuple[str, str], float] = {}
    if not req.lavaan_spec:
        for X in req.treatments:
            for Y in req.outcomes:
                # sum of all indirect chains (X,Y) + direct (X,Y)
                ind_sum = 0.0
                for ch in indirect_chains:
                    if ch["treatment"] == X and ch["outcome"] == Y:
                        ind_sum += indirect_pt[ch["label"]]
                # direct
                d = next((dd for dd in direct_pairs if dd["treatment"] == X and dd["outcome"] == Y), None)
                dir_est = estimates.get(d["label"], 0.0) if d else 0.0
                total_pt[(X, Y)] = ind_sum + dir_est

    # ── bootstrap ───────────────────────────────────────────────────────────
    reps = max(0, min(20000, int(req.bootstrap or 0)))
    indirect_boot: Dict[str, List[float]] = {ch["label"]: [] for ch in indirect_chains}
    total_boot: Dict[Tuple[str, str], List[float]] = {k: [] for k in total_pt}
    boot_used = 0
    if reps >= 100 and indirect_chains:
        rng = np.random.default_rng(42)
        idx = np.arange(n)
        dfr = df.reset_index(drop=True)
        for _ in range(reps):
            bi = rng.choice(idx, size=n, replace=True)
            try:
                ests_b, _, _ = _fit_and_extract(spec, dfr.iloc[bi])
            except Exception:
                continue
            boot_used += 1
            for ch in indirect_chains:
                indirect_boot[ch["label"]].append(_indirect_pt(ch["path_labels"], ests_b))
            if not req.lavaan_spec:
                for X in req.treatments:
                    for Y in req.outcomes:
                        ind_sum = 0.0
                        for ch in indirect_chains:
                            if ch["treatment"] == X and ch["outcome"] == Y:
                                ind_sum += _indirect_pt(ch["path_labels"], ests_b)
                        d = next((dd for dd in direct_pairs
                                  if dd["treatment"] == X and dd["outcome"] == Y), None)
                        dir_est = ests_b.get(d["label"], 0.0) if d else 0.0
                        total_boot[(X, Y)].append(ind_sum + dir_est)

    def _q(arr: List[float]) -> Optional[List[float]]:
        arr = [a for a in arr if np.isfinite(a)]
        if not arr:
            return None
        return [round(float(np.quantile(arr, 0.025)), 6),
                round(float(np.quantile(arr, 0.975)), 6)]

    # ── indirect / total tables ─────────────────────────────────────────────
    indirect_effects = []
    for ch in indirect_chains:
        ci = _q(indirect_boot[ch["label"]]) if indirect_boot[ch["label"]] else None
        sig = bool(ci is not None and (ci[0] > 0 or ci[1] < 0))
        indirect_effects.append({
            "label": ch["label"], "treatment": ch["treatment"],
            "chain": ch["chain"], "outcome": ch["outcome"],
            "est": _round(indirect_pt[ch["label"]], 6),
            "boot_ci": ci, "significant": sig,
        })

    total_effects = []
    for (X, Y), v in total_pt.items():
        total_effects.append({
            "treatment": X, "outcome": Y,
            "est": _round(v, 6),
            "boot_ci": _q(total_boot[(X, Y)]) if total_boot.get((X, Y)) else None,
        })

    # ── fit indices ─────────────────────────────────────────────────────────
    fit: Dict[str, Optional[float]] = {}
    try:
        stats = semopy.calc_stats(model).T  # column "Value"
        col = stats.columns[0]
        s = stats[col]
        chi2 = _safe(s.get("chi2"))
        df_chi = _safe(s.get("DoF"))
        fit = {
            "chi2": _round(chi2, 4),
            "df": int(df_chi) if df_chi is not None else None,
            "p": _safe(s.get("chi2 p-value")),
            "cfi": _round(s.get("CFI"), 4),
            "tli": _round(s.get("TLI"), 4),
            "rmsea": _round(s.get("RMSEA"), 4),
            "srmr": _round(_srmr(model), 4),
            "aic": _round(s.get("AIC"), 2),
            "bic": _round(s.get("BIC"), 2),
            "n": int(n),
        }
    except Exception:
        fit = {"n": int(n)}

    # ── plain-English summary ───────────────────────────────────────────────
    sig_chains = [ie for ie in indirect_effects if ie["significant"]]
    parts = [
        f"SEM / path analysis with {len(req.treatments) if not req.lavaan_spec else '?'} treatment(s), "
        f"{len(req.mediators) if not req.lavaan_spec else '?'} mediator(s), "
        f"{len(req.outcomes) if not req.lavaan_spec else '?'} outcome(s) "
        f"(n = {n}{', serial chain' if req.serial else ''}"
        + (f", adjusted for {', '.join(req.covariates)}" if req.covariates else "") + ").",
    ]
    if fit.get("chi2") is not None:
        parts.append(
            f"Fit: chi2({fit.get('df')}) = {fit.get('chi2')}, p = {_p_str(fit.get('p') or float('nan'))}; "
            f"CFI = {fit.get('cfi')}, TLI = {fit.get('tli')}, RMSEA = {fit.get('rmsea')}, "
            f"SRMR = {fit.get('srmr')}."
        )
    if sig_chains:
        labs = "; ".join(c["label"] for c in sig_chains)
        parts.append(f"Significant indirect path(s) (95% bootstrap CI excludes 0): {labs}.")
    elif indirect_effects:
        parts.append("No indirect path reached significance at the 95% bootstrap CI.")
    if boot_used and reps:
        parts.append(f"Bootstrap: {boot_used}/{reps} resamples converged.")
    result_text = " ".join(parts)

    return {
        "test": "SEM / Path analysis",
        "n": int(n),
        "lavaan_spec": spec,
        "paths": paths,
        "indirect_effects": indirect_effects,
        "direct_effects": direct_effects,
        "total_effects": total_effects,
        "fit": fit,
        "covariates": req.covariates,
        "serial": bool(req.serial),
        "bootstrap_used": int(boot_used),
        "result_text": result_text,
        "interpretation": result_text,
    }
