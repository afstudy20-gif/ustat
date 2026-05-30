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


class SubgroupBarRequest(BaseModel):
    session_id: str
    y_col: str
    subgroup_col: str
    xaxis_col: str
    color_col: Optional[str] = None
    y_mode: str = "mean"  # "mean" or "percentage"
    target_value: Optional[str] = None
    error_type: str = "ci"  # "ci", "se", "sd", "none"


@router.post("/subgroup_bar")
def subgroup_bar(req: SubgroupBarRequest):
    df = _get_df(req.session_id)
    
    # Check if selected columns exist
    for col in [req.y_col, req.subgroup_col, req.xaxis_col]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
            
    if req.color_col and req.color_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.color_col}' not found")
        
    # Get subset of columns
    cols_to_use = [req.y_col, req.subgroup_col, req.xaxis_col]
    if req.color_col:
        cols_to_use.append(req.color_col)
        
    sub = df[cols_to_use].copy()
    
    # Drop missing in grouping columns, but handle Y missing per cell safely
    sub = sub.dropna(subset=[req.subgroup_col, req.xaxis_col] + ([req.color_col] if req.color_col else []))
    
    if len(sub) == 0:
        raise HTTPException(status_code=400, detail="No valid data points found after dropping missing values in grouping variables.")

    # Get unique groups, handling categories sorting nicely
    subgroups = sorted(sub[req.subgroup_col].dropna().unique())
    x_vals = sorted(sub[req.xaxis_col].dropna().unique())
    color_groups = sorted(sub[req.color_col].dropna().unique()) if req.color_col else ["All"]

    traces = []
    
    for cg in color_groups:
        trace_x_subgroup = []
        trace_x_xaxis = []
        trace_y = []
        trace_error = []
        trace_ns = []
        
        for sg in subgroups:
            for xv in x_vals:
                # Filter for this cell
                mask = (sub[req.subgroup_col] == sg) & (sub[req.xaxis_col] == xv)
                if req.color_col:
                    mask = mask & (sub[req.color_col] == cg)
                
                cell_data = sub.loc[mask, req.y_col].dropna()
                n = len(cell_data)
                
                val = 0.0
                err = 0.0
                
                if n > 0:
                    if req.y_mode == "percentage":
                        target = req.target_value
                        if target is None:
                            unique_vals = cell_data.unique()
                            if len(unique_vals) > 0:
                                target = str(unique_vals[0])
                            else:
                                target = "1"
                        
                        successes = sum(cell_data.astype(str) == str(target))
                        p = successes / n
                        val = p * 100.0
                        
                        sd = np.sqrt(p * (1 - p)) * 100.0
                        se = (np.sqrt(p * (1 - p) / n)) * 100.0
                        ci = 1.96 * se
                    else:
                        numeric_data = pd.to_numeric(cell_data, errors='coerce').dropna()
                        n_num = len(numeric_data)
                        if n_num > 0:
                            val = float(numeric_data.mean())
                            sd = float(numeric_data.std()) if n_num > 1 else 0.0
                            se = sd / np.sqrt(n_num)
                            ci = 1.96 * se
                        else:
                            val = 0.0
                            sd = 0.0
                            se = 0.0
                            ci = 0.0
                    
                    if req.error_type == "ci":
                        err = ci
                    elif req.error_type == "se":
                        err = se
                    elif req.error_type == "sd":
                        err = sd
                    else:
                        err = 0.0
                else:
                    val = 0.0
                    err = 0.0
                
                trace_x_subgroup.append(str(sg))
                trace_x_xaxis.append(str(xv))
                trace_y.append(val)
                trace_error.append(err)
                trace_ns.append(n)
                
        traces.append({
            "name": str(cg),
            "x_subgroup": trace_x_subgroup,
            "x_xaxis": trace_x_xaxis,
            "y": trace_y,
            "error": trace_error,
            "ns": trace_ns
        })
        
    return {
        "type": "subgroup_bar",
        "y_col": req.y_col,
        "subgroup_col": req.subgroup_col,
        "xaxis_col": req.xaxis_col,
        "color_col": req.color_col,
        "y_mode": req.y_mode,
        "target_value": req.target_value,
        "error_type": req.error_type,
        "traces": traces
    }

