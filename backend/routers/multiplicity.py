"""
Multiple-testing / multiplicity router.

Endpoints
---------
POST /gatekeeping  — multistage gatekeeping across ordered families of
                     hypotheses using a truncated Holm or truncated Hochberg
                     procedure within each family (Dmitrienko, Tamhane & Wiens
                     2008). Serial or parallel logic between families.

Pure python — no new deps.

Method
------
Families F1, …, Fm are tested in order. Within a family the per-rank critical
fraction for a truncation γ ∈ [0, 1] is

    c_i = γ / (m − i + 1) + (1 − γ) / m        (i = 1 … m, rank ascending)

γ = 1 reproduces the ordinary Holm (step-down) or Hochberg (step-up)
procedure; γ = 0 reduces to Bonferroni. A truncation < 1 in a non-terminal
family reserves Bonferroni mass so the "gate" to the next family can open
while still allowing within-family testing — this is the truncated
gatekeeping device.

Between families:
  • serial   — family k+1 is tested only if EVERY hypothesis in families
               1…k was rejected.
  • parallel — family k+1 is tested only if AT LEAST ONE hypothesis in the
               immediately preceding family was rejected.

Adjusted p-values are obtained as the infimum α at which the full procedure
rejects each hypothesis (the operational definition of a multiplicity-adjusted
p-value). Because the rejection region is monotone in α this is well defined;
it is computed by a fine grid scan, so the same engine serves both the Holm
and Hochberg variants exactly.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store

router = APIRouter()


class GKHypothesis(BaseModel):
    label: str
    p: float


class GKFamily(BaseModel):
    name: str
    hypotheses: List[GKHypothesis]
    gamma: Optional[float] = None      # truncation fraction; default 0.5 (non-terminal), 1.0 (terminal)


class GatekeepingRequest(BaseModel):
    session_id: Optional[str] = None
    families: List[GKFamily]
    method: str = "hochberg"           # hochberg | holm
    logic: str = "serial"              # serial | parallel
    alpha: float = 0.05


def _within_family_reject(pvals: List[float], gamma: float, alpha: float, method: str) -> List[bool]:
    """Return per-hypothesis rejection (original order) for a truncated
    Holm (step-down) or Hochberg (step-up) test at level `alpha`."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])      # ascending p
    crit = [gamma / (m - i) + (1.0 - gamma) / m for i in range(m)]  # i=0..m-1 → rank i+1
    rej_sorted = [False] * m

    if method == "holm":
        # Step-down: reject ranks 1.. until the first that fails.
        for r in range(m):
            if pvals[order[r]] <= alpha * crit[r]:
                rej_sorted[r] = True
            else:
                break
    else:  # hochberg step-up
        # From the largest p downward; first rank that passes → reject it and all smaller.
        cutoff = -1
        for r in range(m - 1, -1, -1):
            if pvals[order[r]] <= alpha * crit[r]:
                cutoff = r
                break
        for r in range(cutoff + 1):
            rej_sorted[r] = True

    out = [False] * m
    for r, idx in enumerate(order):
        out[idx] = rej_sorted[r]
    return out


def _gatekeep_at(families: List[GKFamily], gammas: List[float], alpha: float,
                 method: str, logic: str) -> List[List[bool]]:
    """Run the full multistage procedure at level alpha. Returns rejection
    booleans per family (original order)."""
    result: List[List[bool]] = []
    gate_open = True
    for k, fam in enumerate(families):
        ps = [h.p for h in fam.hypotheses]
        if not gate_open or len(ps) == 0:
            result.append([False] * len(ps))
            # once a serial gate closes it stays closed
            gate_open = False
            continue
        rej = _within_family_reject(ps, gammas[k], alpha, method)
        result.append(rej)
        # decide whether the next family's gate is open
        if logic == "serial":
            gate_open = all(rej)
        else:  # parallel — at least one rejected in this (preceding) family
            gate_open = any(rej)
    return result


@router.post("/gatekeeping")
def gatekeeping(req: GatekeepingRequest):
    if not req.families or len(req.families) < 1:
        raise HTTPException(status_code=422, detail="Provide at least one family of hypotheses.")
    if req.method not in ("hochberg", "holm"):
        raise HTTPException(status_code=422, detail="method must be 'hochberg' or 'holm'.")
    if req.logic not in ("serial", "parallel"):
        raise HTTPException(status_code=422, detail="logic must be 'serial' or 'parallel'.")
    for f in req.families:
        if not f.hypotheses:
            raise HTTPException(status_code=422, detail=f"Family '{f.name}' has no hypotheses.")
        for h in f.hypotheses:
            if not (0.0 <= h.p <= 1.0):
                raise HTTPException(status_code=422, detail=f"p-value out of [0,1] for '{h.label}'.")
    if not (0.0 < req.alpha < 1.0):
        raise HTTPException(status_code=422, detail="alpha must be in (0, 1).")

    n_fam = len(req.families)
    gammas: List[float] = []
    for k, f in enumerate(req.families):
        if f.gamma is not None:
            g = float(f.gamma)
        else:
            g = 1.0 if k == n_fam - 1 else 0.5     # terminal: full; non-terminal: half-truncated
        gammas.append(min(1.0, max(0.0, g)))

    # Adjusted p-value = inf{ alpha : hypothesis rejected }, via a fine grid.
    GRID = 2000
    grid = [i / GRID for i in range(1, GRID + 1)]    # 0.0005 … 1.0
    adj = [[1.0] * len(f.hypotheses) for f in req.families]
    found = [[False] * len(f.hypotheses) for f in req.families]
    for a in grid:
        rej = _gatekeep_at(req.families, gammas, a, req.method, req.logic)
        all_done = True
        for k in range(n_fam):
            for j in range(len(req.families[k].hypotheses)):
                if not found[k][j]:
                    if rej[k][j]:
                        adj[k][j] = round(a, 5)
                        found[k][j] = True
                    else:
                        all_done = False
        if all_done:
            break

    # Final decisions at the requested alpha.
    final = _gatekeep_at(req.families, gammas, req.alpha, req.method, req.logic)

    families_out = []
    rows_export = [["Family", "Hypothesis", "Raw p", "Adjusted p", "Reject"]]
    n_rejected = 0
    for k, f in enumerate(req.families):
        hyps = []
        for j, h in enumerate(f.hypotheses):
            rj = bool(final[k][j])
            n_rejected += int(rj)
            hyps.append({
                "label": h.label,
                "p_raw": round(float(h.p), 6),
                "p_adjusted": adj[k][j],
                "reject": rj,
            })
            rows_export.append([f.name, h.label, round(float(h.p), 6), adj[k][j], "Yes" if rj else "No"])
        families_out.append({
            "name": f.name,
            "gamma": gammas[k],
            "n": len(f.hypotheses),
            "n_rejected": sum(1 for x in hyps if x["reject"]),
            "hypotheses": hyps,
        })

    proc = f"truncated {'Hochberg' if req.method == 'hochberg' else 'Holm'}"
    interp = (
        f"Multistage {req.logic} gatekeeping across {n_fam} ordered "
        f"{'family' if n_fam == 1 else 'families'} using a {proc} procedure within each "
        f"family (FWER controlled at α = {req.alpha}). {n_rejected} hypothesis(es) rejected. "
        "Adjusted p-values are the smallest α at which each hypothesis is rejected by the full "
        "procedure; reject when the adjusted p ≤ α."
        + (" Hochberg step-up assumes independent or positively-dependent p-values within a family."
           if req.method == "hochberg" else "")
    )

    if req.session_id:
        try:
            store.log_action(req.session_id, "gatekeeping", {
                "method": req.method, "logic": req.logic, "alpha": req.alpha,
                "n_families": n_fam, "n_rejected": n_rejected,
            })
        except Exception:
            pass

    return {
        "test": "Multistage gatekeeping",
        "method": req.method,
        "logic": req.logic,
        "alpha": req.alpha,
        "n_families": n_fam,
        "n_rejected": n_rejected,
        "families": families_out,
        "result_text": interp,
        "interpretation": interp,
        "export_rows": rows_export,
        "r_code": (
            "library(gMCP)   # or lrstat::truncated procedures\n"
            "# Multistage gatekeeping with a truncated "
            f"{'Hochberg' if req.method == 'hochberg' else 'Holm'} test per family,\n"
            f"# {req.logic} logic, FWER = {req.alpha}.\n"
            "# Enter raw p-values and truncation gammas per family."
        ),
    }
