from __future__ import annotations

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import json as _json
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException, Response, Query
from pydantic import BaseModel, Field
from loguru import logger

from services import store
from services.impute import apply_imputation, missing_info

router = APIRouter()


def _get_df(session_id: str, *, allow_missing: bool = False) -> pd.DataFrame | None:
    df = store.get_filtered(session_id)
    if df is None:
        if allow_missing:
            return None
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None in dicts/lists."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ── 1. Missing Data Summary ─────────────────────────────────────────────────────

@router.get("/{session_id}/missing")
def get_missing(session_id: str, columns: str = Query("")):
    """
    Return per-column missing counts and total rows affected for the given
    comma-separated list of column names.
    """
    df = _get_df(session_id, allow_missing=True)
    if df is None:
        return {"columns": [], "total_rows": 0}
    cols = [c.strip() for c in columns.split(",") if c.strip() and c.strip() in df.columns]
    if not cols:
        cols = df.columns.tolist()
    return missing_info(df, cols)


# ── 2. Descriptive Statistics ──────────────────────────────────────────────────

def _normality_test(s_clean: pd.Series) -> tuple[float, str]:
    """Return (p_value, test_name)."""
    n = len(s_clean)
    if n < 3:
        return 1.0, "—"
    if n < 50:
        _, p = scipy_stats.shapiro(s_clean)
        return float(p), "Shapiro-Wilk"
    if n <= 2000:
        from statsmodels.stats.diagnostic import lilliefors as _lilliefors
        _, p = _lilliefors(s_clean.values, dist="norm")
        return float(p), "Kolmogorov-Smirnov (Lilliefors)"

    skewness = float(scipy_stats.skew(s_clean))
    if abs(skewness) <= 1.5:
        return 0.999, "Skewness (CLT bypass)"
    from statsmodels.stats.diagnostic import lilliefors as _lilliefors
    _, p = _lilliefors(s_clean.values, dist="norm")
    return float(p), "Kolmogorov-Smirnov (Lilliefors)"


@router.get("/{session_id}/descriptive")
def descriptive(session_id: str, column: Optional[str] = None):
    df = _get_df(session_id)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if column:
        if column not in num_cols:
            raise HTTPException(status_code=400, detail="Column not numeric")
        num_cols = [column]

    # Resolve session-persisted decimal overrides once so each column
    # carries its display hint to the frontend (Summary tile, exports).
    decimals_override = _resolve_decimals_override(session_id, None)

    results = {}
    for col in num_cols:
        s = df[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 3:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        n = len(s)
        p_norm, norm_test = _normality_test(s)

        results[col] = {
            "n": int(n),
            "missing": int(df[col].isna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "se": float(s.sem()),
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "skewness": float(scipy_stats.skew(s)),
            "kurtosis": float(scipy_stats.kurtosis(s)),
            "normality_p": float(p_norm),
            "normality_test": norm_test,
            "normal": bool(p_norm > 0.05),
            # Suggested decimal places for displaying sample-valued stats
            # (mean, median, quartiles, min/max). Honours user overrides
            # and auto-detects integer-valued columns.
            "display_decimals": _col_decimals(df, col, decimals_override, fallback=2),
        }
    return _sanitize(results)


# ── 3. Frequency Table ─────────────────────────────────────────────────────────

@router.get("/{session_id}/frequency")
def frequency(session_id: str, column: Optional[str] = None):
    df = _get_df(session_id)
    cols = df.columns.tolist()
    if column:
        if column not in df.columns:
            raise HTTPException(status_code=400, detail="Column not found")
        cols = [column]

    results = {}
    for col in cols:
        s = df[col]
        total = len(s)
        vc = s.value_counts(dropna=False)
        categories = []
        for k, v in vc.items():
            categories.append({
                "value": str(k) if pd.notna(k) else "Missing",
                "count": int(v),
                "pct": round(v / total * 100, 2),
            })
        results[col] = {
            "n": int(s.count()),
            "missing": int(s.isna().sum()),
            "categories": categories,
        }
    return results


# ── 4. Sparklines ──────────────────────────────────────────────────────────────

@router.get("/{session_id}/sparklines")
def get_sparklines(session_id: str):
    df = _get_df(session_id, allow_missing=True)
    if df is None:
        return {}
    result = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            result[col] = {"type": "empty", "data": []}
            continue
        if pd.api.types.is_numeric_dtype(s):
            n_bins = min(14, max(4, int(len(s) ** 0.38)))
            counts, _ = np.histogram(s, bins=n_bins)
            result[col] = {"type": "numeric", "data": counts.tolist()}
        else:
            vc = s.value_counts(normalize=True)
            n_cats = min(6, len(vc))
            result[col] = {
                "type": "categorical",
                "data": [float(v) for v in vc.head(n_cats).values],
                "labels": vc.head(n_cats).index.astype(str).tolist(),
            }
    return result


# ── 5. Refresh ─────────────────────────────────────────────────────────────────

@router.get("/{session_id}/refresh")
def refresh_session(session_id: str):
    """Return updated session metadata after in-place operations."""
    df = _get_df(session_id)
    from routers.upload import _detect_kind
    columns = []
    for col in df.columns:
        kind = _detect_kind(df[col])
        columns.append({"name": col, "dtype": str(df[col].dtype), "kind": kind})
    preview_df = df.head(2000).replace([np.inf, -np.inf], np.nan)
    preview = _json.loads(preview_df.to_json(orient="records", default_handler=str, date_format="iso", date_unit="s"))
    return {"rows": len(df), "columns": columns, "preview": preview}


# ── 6. Raw Data Columns ────────────────────────────────────────────────────────

@router.get("/{session_id}/raw")
def get_raw_columns(session_id: str, columns: str = ""):
    df = _get_df(session_id)
    cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns] if columns else list(df.columns)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])][:12]
    result = {}
    for col in cols:
        vals = df[col].where(df[col].notna(), other=None).tolist()[:3000]
        result[col] = vals
    return result


# ── 7. Column Summary (QQ + Outliers) ──────────────────────────────────────────

@router.get("/{session_id}/column_summary")
def column_summary(session_id: str, column: str, kind: Optional[str] = None):
    df = _get_df(session_id)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
    s = df[column]

    if kind == "numeric":
        is_num = True
    elif kind in ("categorical", "text", "boolean"):
        is_num = False
    else:
        is_num = pd.api.types.is_numeric_dtype(s) and s.nunique() > 10

    if is_num:
        s_clean = s.dropna().astype(float)
        n_clean = len(s_clean)
        n_bins = min(40, max(10, int(np.sqrt(n_clean))))
        counts, edges = np.histogram(s_clean, bins=n_bins)
        histogram = [
            {"bin_start": float(edges[i]), "bin_end": float(edges[i+1]), "count": int(counts[i])}
            for i in range(len(counts))
        ]

        (theo, sample), _ = scipy_stats.probplot(s_clean)
        step = max(1, len(theo) // 300)
        qq = [{"x": float(theo[i]), "y": float(sample[i])} for i in range(0, len(theo), step)]

        p_norm, norm_test_name = _normality_test(s_clean)
        mean_val = float(s_clean.mean())
        std_val  = float(s_clean.std())
        q1, q3 = float(s_clean.quantile(0.25)), float(s_clean.quantile(0.75))
        iqr_val = q3 - q1

        fence_low  = q1 - 1.5 * iqr_val
        fence_high = q3 + 1.5 * iqr_val

        non_out = s_clean[(s_clean >= fence_low) & (s_clean <= fence_high)]
        whisker_low  = float(non_out.min()) if len(non_out) else float(s_clean.min())
        whisker_high = float(non_out.max()) if len(non_out) else float(s_clean.max())

        out_mask = (s_clean < fence_low) | (s_clean > fence_high)
        outliers = [
            {"row": int(idx) + 1, "value": float(val)}
            for idx, val in zip(s_clean.index[out_mask], s_clean[out_mask])
        ]

        z_extremes = []
        normality_deviants = []
        if std_val > 0 and n_clean >= 3:
            z_series = (s_clean - mean_val) / std_val
            s_sorted_idx = s_clean.sort_values().index
            s_sorted_vals = s_clean.loc[s_sorted_idx].values

            all_points_info = []
            for i, idx in enumerate(s_sorted_idx):
                val = float(s_sorted_vals[i])
                rank = i + 1
                theo_q = float(scipy_stats.norm.ppf((rank - 0.375) / (n_clean + 0.25)))
                expected_val = mean_val + std_val * theo_q
                residual = val - expected_val
                z = float(z_series[idx])

                info = {
                    "row": int(idx) + 1,
                    "value": round(val, 4),
                    "z": round(z, 3),
                    "residual": round(residual, 4),
                    "abs_residual": abs(residual),
                    "qq_x": round(theo_q, 4)
                }
                all_points_info.append(info)
                if abs(z) > 2.0:
                    z_extremes.append(info)

            all_points_info.sort(key=lambda d: d["abs_residual"], reverse=True)
            normality_deviants = all_points_info[:10]
            z_extremes.sort(key=lambda d: abs(d["z"]), reverse=True)

        return {
            "type": "numeric",
            "n": int(s_clean.count()), "missing": int(s.isna().sum()),
            "mean": mean_val, "std": std_val,
            "median": float(s_clean.median()), "q1": q1, "q3": q3,
            "iqr": float(iqr_val), "min": float(s_clean.min()), "max": float(s_clean.max()),
            "skewness": float(s_clean.skew()), "kurtosis": float(s_clean.kurtosis()),
            "whisker_low": whisker_low, "whisker_high": whisker_high,
            "outliers": outliers,
            "z_extremes": z_extremes,
            "normality_deviants": normality_deviants,
            "histogram": histogram,
            "raw_values": s_clean.sample(min(2000, n_clean), random_state=42).tolist(),
            "qq": qq,
            "normality_p": float(p_norm),
            "normality_test": norm_test_name,
            "normal": bool(p_norm > 0.05),
            "normality_label": "Normally distributed" if p_norm > 0.05 else "Non-normal distribution",
        }

    else:
        total = len(s)
        vc = s.value_counts(dropna=False)
        categories = [
            {"value": str(k) if pd.notna(k) else "Missing",
             "count": int(v), "pct": round(v / total * 100, 1)}
            for k, v in vc.items()
        ]
        return {
            "type": "categorical",
            "n": int(s.count()), "missing": int(s.isna().sum()),
            "n_categories": int(s.nunique()),
            "categories": categories,
        }


# ── 8. Table 1 (clinical characteristics) ──────────────────────────────────────

class Table1Request(BaseModel):
    session_id: str
    group_column: Optional[str] = None
    variables: list[str]
    variable_kinds: Optional[dict] = None
    selected_stats: Optional[list[str]] = None
    normality_mode: Optional[str] = "overall"
    # Optional per-column decimal overrides keyed by column name. Values
    # supplied here win over (a) the session-persisted decimals map and
    # (b) the auto integer-detection logic in _col_decimals().
    column_decimals: Optional[Dict[str, int]] = None


def _fmt_p(p: float) -> str:
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"


_STAT_LABELS: dict[str, str] = {
    "mean_sd":    "Mean ± SD",
    "median_iqr": "Median [IQR]",
    "se":         "SE of Mean",
    "ci95":       "95% CI",
    "variance":   "Variance",
    "min_max":    "Min – Max",
    "n":          "N (non-missing)",
    "missing":    "Missing",
    "p10":        "10th Pctl",
    "p25":        "25th Pctl",
    "p75":        "75th Pctl",
    "p90":        "90th Pctl",
    "p95":        "95th Pctl",
}


def _f(v: float, d: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{d}f}"


def _col_decimals(
    df: pd.DataFrame,
    col: str,
    override: Optional[Dict[str, int]] = None,
    fallback: int = 2,
) -> int:
    """Resolve per-column decimal places for descriptive display.

    Resolution order:
      1. Explicit ``override`` mapping (request body / session store).
      2. Integer-dtype column → 0 (e.g. counts, days, ages).
      3. Float column whose values are all whole numbers → 0 (e.g. days
         column re-read from SPSS as float64 but holding integers only).
      4. Otherwise the supplied fallback (default 2).

    Industry convention (AMA, ICMJE) is that Table 1 statistics inherit the
    precision of the source variable; a follow-up-days column should report
    integer medians, not 1697.50.
    """
    if override and col in override:
        try:
            return max(0, int(override[col]))
        except (TypeError, ValueError):
            pass
    if col not in df.columns:
        return fallback
    s = df[col]
    if pd.api.types.is_integer_dtype(s):
        return 0
    if pd.api.types.is_float_dtype(s):
        clean = s.dropna()
        if len(clean) > 0:
            try:
                if (clean.mod(1) == 0).all():
                    return 0
            except (TypeError, ValueError):
                pass
    return fallback


def _resolve_decimals_override(
    session_id: Optional[str],
    request_override: Optional[Dict[str, int]],
) -> Dict[str, int]:
    """Merge the persisted per-session decimals map with a request-supplied
    one. Request values take precedence so callers can preview overrides
    without committing them to the session."""
    base: Dict[str, int] = {}
    if session_id:
        try:
            base = dict(store.get_decimals(session_id) or {})
        except Exception:  # pragma: no cover — defensive
            base = {}
    if request_override:
        for k, v in request_override.items():
            try:
                base[k] = max(0, int(v))
            except (TypeError, ValueError):
                continue
    return base


def _fmt_one_stat(
    a: pd.Series,
    stat: str,
    *,
    df: Optional[pd.DataFrame] = None,
    col: Optional[str] = None,
    override: Optional[Dict[str, int]] = None,
) -> str:
    """Format a single Table 1 statistic.

    When ``df`` + ``col`` are supplied, the column's natural decimal places
    are honoured (integer columns render as integers, float overrides win).
    SE / variance keep an extra digit of precision per AMA convention.
    """
    if len(a) == 0:
        return "—"
    # Column-aware decimal places for sample-valued statistics (mean,
    # median, quartiles, min/max). Falls back to the legacy 2-decimal
    # default when no column context is supplied.
    if df is not None and col is not None:
        d = _col_decimals(df, col, override, fallback=2)
    else:
        d = 2

    def fc(v: float, dd: Optional[int] = None) -> str:
        return _f(v, d if dd is None else dd)

    if stat == "mean_sd":
        return f"{fc(a.mean())} ± {fc(a.std())}"
    if stat == "median_iqr":
        q1, q3 = a.quantile(0.25), a.quantile(0.75)
        return f"{fc(a.median())} [{fc(q1)}–{fc(q3)}]"
    if stat == "se":
        # SE keeps an extra digit of precision since it shrinks with √n.
        return fc(a.sem(), max(d, 3))
    if stat == "ci95":
        if len(a) < 2:
            return "—"
        se = a.sem()
        m = a.mean()
        t_crit = scipy_stats.t.ppf(0.975, df=len(a) - 1)
        ci = t_crit * se
        return f"{fc(m)} [{fc(m - ci)}–{fc(m + ci)}]"
    if stat == "variance":
        return fc(a.var(), max(d, 3))
    if stat == "min_max":
        return f"{fc(a.min())} – {fc(a.max())}"
    if stat == "n":
        return str(int(len(a)))
    if stat == "missing":
        return str(int(a.isna().sum()) if hasattr(a, 'isna') else 0)
    pct_map = {"p10": 0.10, "p25": 0.25, "p75": 0.75, "p90": 0.90, "p95": 0.95}
    if stat in pct_map:
        return fc(a.quantile(pct_map[stat]))
    return "—"


def _build_stat_rows(
    s_col: pd.Series,
    group_series: dict[str, pd.Series],
    stats: list[str],
    normal: bool,
    *,
    df: Optional[pd.DataFrame] = None,
    col: Optional[str] = None,
    override: Optional[Dict[str, int]] = None,
) -> list[dict]:
    rows_out = []
    s_all = s_col.dropna().astype(float)

    for stat in stats:
        resolved = stat
        if stat == "auto":
            resolved = "mean_sd" if normal else "median_iqr"

        label = _STAT_LABELS.get(resolved, resolved)
        if resolved == "missing":
            overall_val = str(int(s_col.isna().sum()))
            grp_vals = {gl: str(int(gs.isna().sum())) for gl, gs in group_series.items()}
        else:
            overall_val = _fmt_one_stat(
                s_all, resolved, df=df, col=col, override=override,
            )
            grp_vals = {
                gl: _fmt_one_stat(
                    gs.dropna().astype(float), resolved,
                    df=df, col=col, override=override,
                )
                for gl, gs in group_series.items()
            }

        rows_out.append({"label": label, "overall": overall_val, "group_stats": grp_vals})
    return rows_out


def _fisher_freeman_halton_mc(observed: np.ndarray, n_resamples: int = 5000, seed: int = 42) -> float:
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.sum() <= 0:
        return float("nan")
    n_rows, n_cols = obs.shape

    cats_list: list[int] = []
    grps_list: list[int] = []
    for i in range(n_rows):
        for j in range(n_cols):
            n_ij = int(obs[i, j])
            if n_ij > 0:
                cats_list.extend([i] * n_ij)
                grps_list.extend([j] * n_ij)
    cats = np.asarray(cats_list, dtype=np.int64)
    grps = np.asarray(grps_list, dtype=np.int64)

    def _chi(ct: np.ndarray) -> float:
        rs = ct.sum(axis=1, keepdims=True)
        cs = ct.sum(axis=0, keepdims=True)
        total = ct.sum()
        if total <= 0:
            return 0.0
        e = rs * cs / total
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(((ct - e) ** 2 / np.where(e > 0, e, 1)).sum())

    obs_chi = _chi(obs)
    rng = np.random.default_rng(seed)
    minlength = n_rows * n_cols
    count = 0
    for _ in range(n_resamples):
        perm = rng.permutation(grps)
        enc = cats * n_cols + perm
        ct = np.bincount(enc, minlength=minlength).reshape(n_rows, n_cols).astype(float)
        if _chi(ct) >= obs_chi - 1e-9:
            count += 1
    return (count + 1) / (n_resamples + 1)


def _categorical_p_with_rule(ct: np.ndarray) -> tuple[float, str]:
    obs = np.asarray(ct, dtype=float)
    chi2, p_chi, dof, expected = scipy_stats.chi2_contingency(obs)
    if (expected < 5).any():
        if obs.shape == (2, 2):
            _, p_fisher = scipy_stats.fisher_exact(obs)
            return float(p_fisher), "Fisher"
        return _fisher_freeman_halton_mc(obs), "Fisher-Freeman-Halton (MC)"
    return float(p_chi), "Chi-square"


@router.post("/table1")
def table1(req: Table1Request):
    df = _get_df(req.session_id)
    rows = []
    sel_stats: list[str] = req.selected_stats if req.selected_stats else ["auto"]
    # Per-column decimal overrides: merge the session-persisted map with
    # any request-supplied overrides (request wins). Auto-detection still
    # applies for columns absent from both.
    decimals_override = _resolve_decimals_override(req.session_id, req.column_decimals)

    groups = None
    group_labels = []
    group_ns: dict = {}
    if req.group_column and req.group_column in df.columns:
        groups = sorted(df[req.group_column].dropna().unique().tolist(), key=str)
        group_labels = [str(g) for g in groups]
        group_ns = {str(g): int((df[req.group_column] == g).sum()) for g in groups}

    for var in req.variables:
        if var not in df.columns:
            continue
        s = df[var]

        provided_kind = (req.variable_kinds or {}).get(var)
        if provided_kind == "numeric":
            is_num = True
        elif provided_kind in ("categorical", "text", "boolean"):
            is_num = False
        else:
            is_num = pd.api.types.is_numeric_dtype(s) and s.nunique() > 10

        if is_num:
            s_all = s.dropna().astype(float)
            p_norm, norm_test_name = _normality_test(s_all)
            normal_overall = p_norm > 0.05

            group_series: dict[str, pd.Series] = {}
            group_arrs: list[pd.Series] = []
            if groups is not None:
                for g, gl in zip(groups, group_labels):
                    gs = df[df[req.group_column] == g][var]
                    group_series[gl] = gs
                    group_arrs.append(gs.dropna().astype(float))

            per_group_norm: dict[str, dict] = {}
            if (req.normality_mode == "within_group" and groups is not None
                    and len(group_arrs) >= 2):
                for gl, arr in zip(group_labels, group_arrs):
                    if len(arr) >= 3:
                        pg, pg_name = _normality_test(arr)
                        per_group_norm[gl] = {
                            "p": round(float(pg), 4),
                            "test": pg_name,
                            "normal": bool(pg > 0.05),
                            "n": int(len(arr)),
                        }
                    else:
                        per_group_norm[gl] = {
                            "p": None,
                            "test": "n<3",
                            "normal": False,
                            "n": int(len(arr)),
                        }
                normal = (len(per_group_norm) > 0
                          and all(v["normal"] for v in per_group_norm.values()))
            else:
                normal = normal_overall

            stat_rows = _build_stat_rows(
                s, group_series, sel_stats, normal,
                df=df, col=var, override=decimals_override,
            )

            p_value_str: Optional[str] = None
            test_name_str: Optional[str] = None
            significant = False
            if groups is not None and len(group_arrs) >= 2:
                try:
                    if len(groups) == 2:
                        if normal:
                            _, p_t = scipy_stats.ttest_ind(*group_arrs, equal_var=False)
                            test_name_str = "t-test"
                        else:
                            _, p_t = scipy_stats.mannwhitneyu(*group_arrs, alternative="two-sided")
                            test_name_str = "Mann-Whitney"
                    else:
                        if normal:
                            _, p_t = scipy_stats.f_oneway(*group_arrs)
                            test_name_str = "ANOVA"
                        else:
                            _, p_t = scipy_stats.kruskal(*group_arrs)
                            test_name_str = "Kruskal-Wallis"
                    p_value_str = _fmt_p(float(p_t))
                    significant = bool(float(p_t) < 0.05)
                except Exception as exc:
                    logger.exception("Table 1 statistical test failed")
                    p_value_str = "N/A"

            smd_val: Optional[float] = None
            if groups is not None and len(group_arrs) >= 2:
                try:
                    def _smd_num_pair(g1, g2) -> Optional[float]:
                        if len(g1) == 0 or len(g2) == 0:
                            return None
                        ps = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
                        if not np.isfinite(ps) or ps <= 0:
                            return None
                        return float(abs(g1.mean() - g2.mean()) / ps)
                    from itertools import combinations as _comb
                    pair_smds = []
                    for i, j in _comb(range(len(group_arrs)), 2):
                        s_smd = _smd_num_pair(group_arrs[i], group_arrs[j])
                        if s_smd is not None:
                            pair_smds.append(s_smd)
                    if pair_smds:
                        smd_val = round(max(pair_smds), 4)
                except Exception as exc:
                    logger.exception("SMD numerical calculation failed")

            row: dict = {
                "variable": var,
                "type": "numeric",
                "overall_n": int(len(s_all)),
                "normal": normal,
                "normality_test": norm_test_name,
                "normality_p": round(p_norm, 4),
                "normality_mode": req.normality_mode or "overall",
                "per_group_normality": per_group_norm,
                "stat_rows": stat_rows,
                "p_value": p_value_str,
                "test": test_name_str,
                "significant": significant,
                "smd": smd_val,
                "stat_label": stat_rows[0]["label"] if stat_rows else "",
                "overall": stat_rows[0]["overall"] if stat_rows else "",
                "group_stats": stat_rows[0]["group_stats"] if stat_rows else {},
            }

        else:
            vc_all = s.value_counts(dropna=True)
            total_all = s.count()
            cats = [str(v) for v in vc_all.index.tolist()]
            sub_rows = []
            for cat in cats:
                n_all = int((s.astype(str) == cat).sum())
                pct_all = round(n_all / total_all * 100, 1) if total_all else 0
                sub: dict = {"category": cat, "overall": f"{n_all} ({pct_all}%)", "group_stats": {}}
                if groups is not None:
                    for g, gl in zip(groups, group_labels):
                        g_s = df[df[req.group_column] == g][var]
                        n_g = int((g_s.astype(str) == cat).sum())
                        t_g = g_s.count()
                        pct_g = round(n_g / t_g * 100, 1) if t_g else 0
                        sub["group_stats"][gl] = f"{n_g} ({pct_g}%)"
                sub_rows.append(sub)

            p_val: Optional[str] = None
            test_name: Optional[str] = None
            p_chi_raw: Optional[float] = None
            if groups is not None:
                try:
                    ct = pd.crosstab(df[var].astype(str), df[req.group_column])
                    p_chi_raw, test_name = _categorical_p_with_rule(ct.values)
                    p_val = _fmt_p(float(p_chi_raw))
                except Exception as exc:
                    logger.exception("Categorical test failed in Table 1")
                    p_val = "N/A"

            cat_smd: Optional[float] = None
            if groups is not None and len(groups) >= 2:
                try:
                    def _smd_cat_pair(g1_s: pd.Series, g2_s: pd.Series) -> Optional[float]:
                        all_cats = sorted(set(g1_s.dropna()) | set(g2_s.dropna()))
                        if len(all_cats) < 2:
                            return None
                        if len(all_cats) == 2:
                            target = all_cats[0]
                            p1 = (g1_s == target).mean()
                            p2 = (g2_s == target).mean()
                            pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
                            if pooled <= 0:
                                  return None
                            return float(abs(p1 - p2) / pooled)

                        p1_vec = np.array([(g1_s == c).mean() for c in all_cats[:-1]])
                        p2_vec = np.array([(g2_s == c).mean() for c in all_cats[:-1]])
                        s1 = np.diag(p1_vec * (1 - p1_vec))
                        s2 = np.diag(p2_vec * (1 - p2_vec))
                        s_pool = (s1 + s2) / 2
                        diff = p1_vec - p2_vec
                        det = np.linalg.det(s_pool)
                        if det <= 1e-12:
                            return None
                        return float(np.sqrt(diff @ np.linalg.inv(s_pool) @ diff))

                    from itertools import combinations as _comb
                    g_series = [df[df[req.group_column] == g][var].astype(str) for g in groups]
                    pair_smds = []
                    for i, j in _comb(range(len(g_series)), 2):
                        s_smd = _smd_cat_pair(g_series[i], g_series[j])
                        if s_smd is not None and np.isfinite(s_smd):
                            pair_smds.append(s_smd)
                    if pair_smds:
                        cat_smd = round(max(pair_smds), 4)
                except Exception as exc:
                    logger.exception("SMD categorical calculation failed")

            row = {
                "variable": var,
                "type": "categorical",
                "stat_label": "n (%)",
                "overall": f"n={total_all}",
                "overall_n": int(total_all),
                "p_value": p_val,
                "test": test_name,
                "significant": bool(p_chi_raw is not None and p_chi_raw < 0.05),
                "sub_rows": sub_rows,
                "group_stats": {},
                "stat_rows": [],
                "smd": cat_smd,
            }
        rows.append(row)

    return _sanitize({
        "group_column": req.group_column,
        "group_labels": group_labels,
        "group_ns": group_ns,
        "total_n": len(df),
        "rows": rows,
    })


# ── 9. Weighted Descriptive Statistics ────────────────────────────────────────

class WeightedDescriptiveRequest(BaseModel):
    session_id: str
    value_cols: List[str]
    weight_col: str
    group_col: Optional[str] = None
    imputation: Optional[str] = "listwise"


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return float(np.interp(q, cw, v))


@router.post("/weighted_descriptive")
def weighted_descriptive(req: WeightedDescriptiveRequest):
    from statsmodels.stats.weightstats import DescrStatsW

    df_full = _get_df(req.session_id)
    for c in [req.weight_col, *req.value_cols] + ([req.group_col] if req.group_col else []):
        if c not in df_full.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.value_cols:
        raise HTTPException(status_code=422, detail="Select at least one value column.")

    cols = [req.weight_col, *req.value_cols] + ([req.group_col] if req.group_col else [])
    df = apply_imputation(df_full[cols], cols, req.imputation or "listwise").reset_index(drop=True)
    w_all = pd.to_numeric(df[req.weight_col], errors="coerce")

    results: List[dict] = []
    for col in req.value_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & w_all.notna() & (w_all > 0)
        xv = x[mask].values.astype(float)
        wv = w_all[mask].values.astype(float)
        if len(xv) < 3:
            results.append({"column": col, "error": "fewer than 3 valid weighted observations"})
            continue
        d = DescrStatsW(xv, weights=wv, ddof=1)
        lo, hi = d.tconfint_mean(alpha=0.05)
        kish = float((wv.sum() ** 2) / np.sum(wv ** 2))
        uniq = np.unique(xv)
        row = {
            "column": col,
            "n": int(len(xv)),
            "sum_weights": round(float(wv.sum()), 4),
            "ess_kish": round(kish, 2),
            "w_mean": round(float(d.mean), 6),
            "w_sd": round(float(d.std), 6),
            "w_se": round(float(d.std_mean), 6),
            "ci_low": round(float(lo), 6),
            "ci_high": round(float(hi), 6),
            "w_median": round(_weighted_quantile(xv, wv, 0.5), 6),
            "w_q1": round(_weighted_quantile(xv, wv, 0.25), 6),
            "w_q3": round(_weighted_quantile(xv, wv, 0.75), 6),
        }
        if set(uniq.tolist()) <= {0.0, 1.0} and len(uniq) == 2:
            p = float(np.sum(wv * xv) / np.sum(wv))
            se_p = float(np.sqrt(p * (1 - p) / kish))
            row["w_proportion"] = round(p, 6)
            row["w_proportion_ci_low"] = round(max(0.0, p - 1.959963984540054 * se_p), 6)
            row["w_proportion_ci_high"] = round(min(1.0, p + 1.959963984540054 * se_p), 6)
        results.append(row)

    comparison = None
    if req.group_col:
        groups = [g for g in df[req.group_col].dropna().unique()]
        if len(groups) == 2:
            col = req.value_cols[0]
            x = pd.to_numeric(df[col], errors="coerce")
            parts = []
            for g in groups:
                m = (df[req.group_col] == g) & x.notna() & w_all.notna() & (w_all > 0)
                parts.append((str(g), x[m].values.astype(float), w_all[m].values.astype(float)))
            if all(len(p[1]) >= 3 for p in parts):
                from statsmodels.stats.weightstats import CompareMeans, DescrStatsW as _D
                d1 = _D(parts[0][1], weights=parts[0][2], ddof=1)
                d2 = _D(parts[1][1], weights=parts[1][2], ddof=1)
                cm = CompareMeans(d1, d2)
                tstat, pval, dfree = cm.ttest_ind(usevar="unequal")
                diff = float(d1.mean - d2.mean)
                lo, hi = cm.tconfint_diff(alpha=0.05, usevar="unequal")
                comparison = {
                    "variable": col,
                    "group_a": parts[0][0], "group_b": parts[1][0],
                    "w_mean_a": round(float(d1.mean), 4), "w_mean_b": round(float(d2.mean), 4),
                    "diff": round(diff, 4),
                    "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                    "t": round(float(tstat), 4), "df": round(float(dfree), 2),
                    "p": round(float(pval), 6),
                }

    n_total = int((w_all.notna() & (w_all > 0)).sum())
    result_text = (
        f"Weighted descriptive statistics on n = {n_total} rows using '{req.weight_col}' as the "
        f"sampling weight (design-based, weights only). "
        + (f"Weighted {comparison['variable']}: {comparison['group_a']} = {comparison['w_mean_a']} vs "
           f"{comparison['group_b']} = {comparison['w_mean_b']}, Δ = {comparison['diff']} "
           f"(95% CI {comparison['ci_low']}–{comparison['ci_high']}), weighted t-test p = "
           f"{'<0.001' if comparison['p'] < 0.001 else round(comparison['p'], 3)}."
           if comparison else "")
    )

    export_rows = [["Variable", "n", "ESS", "Weighted mean", "Weighted SD", "95% CI low", "95% CI high", "Weighted median"]]
    for r in results:
        if "error" in r:
            continue
        export_rows.append([r["column"], r["n"], r["ess_kish"], r["w_mean"], r["w_sd"], r["ci_low"], r["ci_high"], r["w_median"]])

    try:
        store.log_action(req.session_id, "weighted_descriptive", {
            "weight_col": req.weight_col, "n_value_cols": len(req.value_cols),
            "group_col": req.group_col,
        })
    except Exception as exc:
        logger.exception("Logging weighted descriptive action failed")

    return _sanitize({
        "test": "Weighted descriptive statistics",
        "weight_col": req.weight_col,
        "n": n_total,
        "results": results,
        "comparison": comparison,
        "assumptions": [
            {"name": "Weights-only design", "met": True,
             "detail": "Design-based estimation with sampling weights. Strata / cluster (full complex survey) not modelled — SEs assume independent weighted observations."},
            {"name": "Effective sample size", "met": True,
             "detail": "Kish's ESS = (Σw)² / Σw² reported per variable; large weight variation shrinks ESS and widens CIs."},
        ],
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": (
            "library(survey)\n"
            f"des <- svydesign(ids = ~1, weights = ~{req.weight_col}, data = data)\n"
            f"svymean(~{' + '.join(req.value_cols)}, des)\n"
            + (f"svyttest({req.value_cols[0]} ~ {req.group_col}, des)\n" if req.group_col else "")
        ),
    })
