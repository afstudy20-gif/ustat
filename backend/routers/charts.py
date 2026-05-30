import json as _json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services import store

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


class ChartRequest(BaseModel):
    session_id: str
    x: str
    y: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None
    bins: int = 20


@router.post("/histogram")
def histogram(req: ChartRequest):
    df = _get_df(req.session_id)
    s = df[req.x].dropna()
    counts, edges = np.histogram(s, bins=req.bins)
    kde_x = np.linspace(s.min(), s.max(), 200)
    kde = scipy_stats.gaussian_kde(s)
    return {
        "type": "histogram",
        "x": req.x,
        "bins": [{"x0": float(edges[i]), "x1": float(edges[i+1]), "count": int(counts[i])} for i in range(len(counts))],
        "kde": [{"x": float(kx), "y": float(ky)} for kx, ky in zip(kde_x, kde(kde_x))],
        "stats": {"mean": float(s.mean()), "median": float(s.median()), "std": float(s.std())},
    }


@router.post("/scatter")
def scatter(req: ChartRequest):
    df = _get_df(req.session_id)

    # Build deduplicated column list
    needed = [req.x, req.y]
    if req.color and req.color not in needed:
        needed.append(req.color)
    if req.shape and req.shape not in needed:
        needed.append(req.shape)

    for col in needed:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    # Clean: replace inf→nan on numeric cols only, then drop missing
    sub = df[needed].copy()
    for col in needed:
        if sub[col].dtype.kind in ("f", "i", "u"):
            sub[col] = sub[col].replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna()

    if len(sub) < 2:
        raise HTTPException(status_code=400, detail="Not enough non-missing data points to draw scatter (need ≥ 2)")

    # Regression only when both axes are numeric
    x_numeric = df[req.x].dtype.kind in ("f", "i", "u")
    y_numeric = df[req.y].dtype.kind in ("f", "i", "u")

    reg: dict = {}
    if x_numeric and y_numeric:
        x_arr = sub[req.x].astype(float).tolist()
        y_arr = sub[req.y].astype(float).tolist()
        try:
            slope, intercept, r, p, se = scipy_stats.linregress(x_arr, y_arr)
            if np.isnan(r) or np.isinf(r):
                raise ValueError("degenerate")
            line_x = [float(sub[req.x].min()), float(sub[req.x].max())]
            line_y = [float(slope * lx + intercept) for lx in line_x]
            reg = {
                "slope": float(slope), "intercept": float(intercept),
                "r": float(r), "r2": float(r ** 2),
                "p": float(p), "se": float(se),
                "line_x": line_x, "line_y": line_y,
            }
        except Exception:
            reg = {
                "slope": None, "intercept": None,
                "r": None, "r2": None, "p": None, "se": None,
                "line_x": [], "line_y": [],
                "note": "Regression unavailable (constant or degenerate data)",
            }
    else:
        reg = {
            "slope": None, "intercept": None,
            "r": None, "r2": None, "p": None, "se": None,
            "line_x": [], "line_y": [],
            "note": "Regression requires two numeric axes",
        }

    # Serialize points safely (NaN → null via json round-trip)
    points = _json.loads(sub.to_json(orient="records", default_handler=str, date_format="iso", date_unit="s"))

    return {
        "type": "scatter",
        "x": req.x, "y": req.y,
        "points": points,
        "regression": reg,
        "color": req.color,
    }


@router.post("/boxplot")
def boxplot(req: ChartRequest):
    df = _get_df(req.session_id)
    if req.color:
        result = []
        for grp, sub in df.groupby(req.color):
            mask = sub[req.x].notna()
            vals = sub.loc[mask, req.x].tolist()
            indices = sub.loc[mask].index.tolist()
            result.append({"group": str(grp), "values": vals, "row_indices": indices})
    else:
        mask = df[req.x].notna()
        vals = df.loc[mask, req.x].tolist()
        indices = df.loc[mask].index.tolist()
        result = [{"group": "All", "values": vals, "row_indices": indices}]
    return {"type": "boxplot", "x": req.x, "groups": result}


class SplomRequest(BaseModel):
    session_id: str
    variables: List[str]
    color: Optional[str] = None


@router.post("/splom")
def splom(req: SplomRequest):
    df = _get_df(req.session_id)

    if len(req.variables) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 variables")

    needed = list(req.variables)
    if req.color and req.color not in needed:
        needed.append(req.color)

    for col in needed:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    sub = df[needed].replace([np.inf, -np.inf], np.nan).dropna()

    if len(sub) < 3:
        raise HTTPException(status_code=400, detail="Not enough data after removing missing values (need ≥ 3 rows)")

    # Build column arrays
    data_cols = {col: sub[col].tolist() for col in req.variables}
    color_values = sub[req.color].tolist() if req.color else None

    # Pairwise Pearson r matrix
    corr: dict = {}
    for a in req.variables:
        for b in req.variables:
            if a == b:
                corr[f"{a}||{b}"] = 1.0
            else:
                key = f"{a}||{b}"
                try:
                    r, _ = scipy_stats.pearsonr(sub[a].astype(float), sub[b].astype(float))
                    corr[key] = round(float(r), 4) if not (np.isnan(r) or np.isinf(r)) else None
                except Exception:
                    corr[key] = None

    return {
        "variables": req.variables,
        "n": len(sub),
        "data": data_cols,
        "color": req.color,
        "color_values": color_values,
        "corr": corr,
    }


@router.post("/bar")
def bar(req: ChartRequest):
    df = _get_df(req.session_id)
    if req.y:
        grp = df.groupby(req.x)[req.y].mean().reset_index()
        return {
            "type": "bar",
            "x": req.x, "y": req.y,
            "data": [{"label": str(row[req.x]), "value": float(row[req.y])} for _, row in grp.iterrows()],
        }
    else:
        counts = df[req.x].value_counts()
        return {
            "type": "bar",
            "x": req.x, "y": "count",
            "data": [{"label": str(k), "value": int(v)} for k, v in counts.items()],
        }


# ── Grouped bar (clustered bars with error bars) ─────────────────────────────

class GroupedBarRequest(BaseModel):
    session_id: str
    x: str                          # categorical subgroup → clusters on the x-axis
    series: Optional[str] = None    # categorical → paired/coloured bars within each cluster
    y: Optional[str] = None         # numeric value column; binary 0/1 → proportion
    error: str = "ci95"             # 'ci95' | 'se' (continuous only; proportions use Wilson 95%)


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


@router.post("/grouped_bar")
def grouped_bar(req: GroupedBarRequest):
    """Clustered bar chart with error bars (e.g. the WHI 'statin use by subgroup'
    figure): subgroups on the x-axis, one coloured bar per `series` level within
    each subgroup, error bars from the 95% CI / SE.

    - Continuous `y` → bar = group mean, error = t-based 95% CI (or SE).
    - Binary 0/1 `y` → bar = proportion, error = Wilson score 95% CI.
    - No `y` → bar = row count (no error bars).
    """
    df = _get_df(req.session_id)
    for c in [req.x] + ([req.series] if req.series else []) + ([req.y] if req.y else []):
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")

    proportion = False
    if req.y:
        yv_all = pd.to_numeric(df[req.y], errors="coerce")
        uniq = set(pd.unique(yv_all.dropna()))
        proportion = len(uniq) > 0 and uniq.issubset({0, 1})

    def _levels(col: str) -> list:
        vals = df[col].dropna().unique().tolist()
        return sorted(vals, key=lambda v: (isinstance(v, str), v))

    x_levels = [str(v) for v in _levels(req.x)]
    series_levels = [str(v) for v in _levels(req.series)] if req.series else ["All"]
    if not x_levels:
        raise HTTPException(status_code=400, detail=f"No non-missing values in '{req.x}'.")

    z = 1.959963984540054
    value_kind = "count"
    groups = []
    for s in series_levels:
        vals, lo, hi, ns = [], [], [], []
        for xl in x_levels:
            mask = df[req.x].astype(str) == xl
            if req.series:
                mask = mask & (df[req.series].astype(str) == s)
            sub = df[mask]
            if not req.y:
                value_kind = "count"
                n = int(len(sub))
                ns.append(n); vals.append(float(n)); lo.append(float(n)); hi.append(float(n))
                continue
            yv = pd.to_numeric(sub[req.y], errors="coerce").dropna()
            n = int(len(yv))
            ns.append(n)
            if n == 0:
                vals.append(0.0); lo.append(0.0); hi.append(0.0); continue
            if proportion:
                value_kind = "proportion"
                k = int(yv.sum())
                p = k / n
                l, h = _wilson_ci(k, n, z)
                vals.append(p); lo.append(l); hi.append(h)
            else:
                value_kind = "mean"
                m = float(yv.mean())
                vals.append(m)
                if n >= 2:
                    se = float(yv.std(ddof=1) / np.sqrt(n))
                    if req.error == "se":
                        l, h = m - se, m + se
                    else:
                        tcrit = float(scipy_stats.t.ppf(0.975, n - 1))
                        l, h = m - tcrit * se, m + tcrit * se
                    lo.append(l); hi.append(h)
                else:
                    lo.append(m); hi.append(m)
        groups.append({
            "series": s, "x": x_levels,
            "value": vals, "err_low": lo, "err_high": hi, "n": ns,
        })

    return {
        "type": "grouped_bar",
        "x": req.x, "series": req.series, "y": req.y,
        "value_kind": value_kind,      # 'mean' | 'proportion' | 'count'
        "error": req.error,
        "x_levels": x_levels,
        "series_levels": series_levels,
        "groups": groups,
    }


# ── Forest plot ─────────────────────────────────────────────────────────────────

class ForestRow(BaseModel):
    label: str
    est: float
    ci_low: float
    ci_high: float
    weight: Optional[float] = None  # for meta-analysis weighting
    group: Optional[str] = None     # optional group label (sub-heading)
    n: Optional[int] = None         # optional sample-size annotation


class ForestRequest(BaseModel):
    rows: List[ForestRow]
    effect_label: str = "OR"        # OR / HR / RR / β / Mean difference
    x_axis: str = "log"             # "log" for OR/HR/RR, "linear" for β/diff
    null_line: float = 1.0          # reference value (1.0 for log-scale, 0 for linear)
    title: Optional[str] = None
    sort_by: Optional[str] = None   # "effect" | "p" | None (preserve order)
    # Meta-analysis (optional):
    do_meta: bool = False
    meta_method: str = "DL"         # DerSimonian-Laird random-effects


@router.post("/forest")
def forest_plot(req: ForestRequest):
    """Forest plot data + optional DerSimonian-Laird meta-analysis pool.

    Accepts a flat row array of {label, est, ci_low, ci_high, weight?, group?}
    and returns Plotly-ready traces + (when do_meta=True) a pooled diamond
    with I² heterogeneity and τ². Same backend serves two UI hooks:
    univariate-OR screening (from logistic_table) and study-level
    meta-analysis (free-form upload).
    """
    rows = [r.dict() for r in req.rows]
    if not rows:
        raise HTTPException(status_code=422, detail="rows array is empty.")
    if req.sort_by == "effect":
        rows.sort(key=lambda r: r["est"])
    # SE inferred from CI assuming symmetric on the log/linear scale.
    log_scale = req.x_axis == "log"
    for r in rows:
        if log_scale:
            r["log_est"] = float(np.log(max(r["est"], 1e-12)))
            r["log_low"] = float(np.log(max(r["ci_low"], 1e-12)))
            r["log_high"] = float(np.log(max(r["ci_high"], 1e-12)))
            r["se"] = (r["log_high"] - r["log_low"]) / (2 * 1.96)
        else:
            r["se"] = (r["ci_high"] - r["ci_low"]) / (2 * 1.96)

    meta = None
    if req.do_meta and len(rows) >= 2:
        ests = np.array([r["log_est"] if log_scale else r["est"] for r in rows], dtype=float)
        ses  = np.array([r["se"] for r in rows], dtype=float)
        wts_fe = 1.0 / (ses ** 2)
        wts_fe = wts_fe / wts_fe.sum()  # normalise
        # Fixed-effect mean
        mu_fe = float(np.sum(wts_fe * ests))
        var_fe = float(1.0 / np.sum(1.0 / (ses ** 2)))
        # Cochran Q and τ² (DerSimonian-Laird)
        q = float(np.sum((1.0 / (ses ** 2)) * (ests - mu_fe) ** 2))
        dfree = len(rows) - 1
        c = float(np.sum(1.0 / (ses ** 2)) - np.sum((1.0 / (ses ** 2)) ** 2) / np.sum(1.0 / (ses ** 2)))
        tau2 = max(0.0, (q - dfree) / c if c > 0 else 0.0)
        # Random-effects re-weighting
        wts_re = 1.0 / (ses ** 2 + tau2)
        mu_re = float(np.sum(wts_re * ests) / np.sum(wts_re))
        var_re = float(1.0 / np.sum(wts_re))
        se_re = float(np.sqrt(var_re))
        ci_low_re = float(mu_re - 1.96 * se_re)
        ci_high_re = float(mu_re + 1.96 * se_re)
        i2 = max(0.0, (q - dfree) / q * 100.0) if q > 0 else 0.0
        from scipy.stats import chi2 as _chi2
        q_p = float(1 - _chi2.cdf(q, dfree)) if dfree > 0 else 1.0
        if log_scale:
            pooled_est = float(np.exp(mu_re))
            pooled_low = float(np.exp(ci_low_re))
            pooled_high = float(np.exp(ci_high_re))
        else:
            pooled_est = float(mu_re)
            pooled_low = float(ci_low_re)
            pooled_high = float(ci_high_re)
        meta = {
            "method": req.meta_method,
            "pooled_est": round(pooled_est, 4),
            "pooled_ci_low": round(pooled_low, 4),
            "pooled_ci_high": round(pooled_high, 4),
            "tau2": round(tau2, 6),
            "Q": round(q, 4),
            "Q_df": dfree,
            "Q_p": round(q_p, 4),
            "I_squared_pct": round(i2, 2),
            "k_studies": len(rows),
            "result_text": (
                f"DerSimonian-Laird random-effects meta-analysis (k = {len(rows)} studies). "
                f"Pooled {req.effect_label} = {pooled_est:.3f} (95% CI {pooled_low:.3f}–{pooled_high:.3f}). "
                f"Heterogeneity Q({dfree}) = {q:.2f}, p = {q_p:.4f}, I² = {i2:.1f}%, τ² = {tau2:.4f}."
            ),
        }

    return {
        "type": "forest",
        "effect_label": req.effect_label,
        "x_axis": req.x_axis,
        "null_line": req.null_line,
        "title": req.title,
        "rows": rows,
        "meta": meta,
    }
