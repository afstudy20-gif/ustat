import numpy as np
import pandas as pd
import json as _json
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException, Response, Query
from pydantic import BaseModel
from typing import Optional, List
from services import store
from services.impute import apply_imputation, missing_info


def _safe_json(obj) -> Response:
    """Serialize obj to JSON, replacing NaN/Inf with null."""
    text = _json.dumps(obj, allow_nan=False, default=lambda x: None
                       if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else str(x))
    return Response(content=text, media_type="application/json")


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None in dicts/lists."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


# ── Missing Data Summary ─────────────────────────────────────────────────────

@router.get("/{session_id}/missing")
def get_missing(session_id: str, columns: str = Query("")):
    """
    Return per-column missing counts and total rows affected for the given
    comma-separated list of column names.
    """
    df = _get_df(session_id)
    cols = [c.strip() for c in columns.split(",") if c.strip() and c.strip() in df.columns]
    if not cols:
        cols = df.columns.tolist()
    return missing_info(df, cols)


# ── Descriptive Statistics ──────────────────────────────────────────────────

@router.get("/{session_id}/descriptive")
def descriptive(session_id: str, column: Optional[str] = None):
    df = _get_df(session_id)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if column:
        if column not in num_cols:
            raise HTTPException(status_code=400, detail="Column not numeric")
        num_cols = [column]

    results = {}
    for col in num_cols:
        s = df[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 3:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        n = len(s)
        if n < 50:
            _, p_norm = scipy_stats.shapiro(s[:5000])
            norm_test = "Shapiro-Wilk"
        else:
            p_norm = scipy_stats.kstest(s, "norm", args=(float(s.mean()), float(s.std()))).pvalue
            norm_test = "Kolmogorov-Smirnov"
        results[col] = {
            "n": int(s.count()),
            "missing": int(df[col].isna().sum()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "se": float(s.sem()),
            "min": float(s.min()),
            "max": float(s.max()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
            "normality_p": float(p_norm),
            "normality_test": norm_test,
            "normality_label": "Normal" if p_norm > 0.05 else "Non-normal",
        }
    return results


# ── Frequency Table ─────────────────────────────────────────────────────────

@router.get("/{session_id}/frequency")
def frequency(session_id: str, column: str):
    df = _get_df(session_id)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
    counts = df[column].value_counts(dropna=False)
    total = len(df)
    return [
        {"value": str(k), "count": int(v), "pct": round(v / total * 100, 2)}
        for k, v in counts.items()
    ]


# ── T-Tests ─────────────────────────────────────────────────────────────────

class TTestRequest(BaseModel):
    session_id: str
    column: str
    group_column: Optional[str] = None
    mu: Optional[float] = 0.0
    equal_var: bool = True


@router.post("/ttest")
def ttest(req: TTestRequest):
    df = _get_df(req.session_id)
    col = df[req.column].dropna()

    if req.group_column:
        groups = df[req.group_column].dropna().unique()
        if len(groups) != 2:
            raise HTTPException(status_code=400, detail="Group column must have exactly 2 groups")
        g1 = df[df[req.group_column] == groups[0]][req.column].dropna()
        g2 = df[df[req.group_column] == groups[1]][req.column].dropna()
        stat, p = scipy_stats.ttest_ind(g1, g2, equal_var=req.equal_var)
        sig = bool(p < 0.05)
        return {
            "test": "Independent samples t-test",
            "group1": str(groups[0]), "n1": len(g1), "mean1": float(g1.mean()),
            "group2": str(groups[1]), "n2": len(g2), "mean2": float(g2.mean()),
            "t": float(stat), "p": float(p),
            "significant": sig,
            "interpretation": f"{'Significant' if sig else 'No significant'} difference between groups (p={'<0.001' if p < 0.001 else f'{p:.4f}'})",
        }
    else:
        stat, p = scipy_stats.ttest_1samp(col, req.mu)
        sig = bool(p < 0.05)
        return {
            "test": "One-sample t-test",
            "mu": req.mu, "n": len(col),
            "mean": float(col.mean()), "std": float(col.std()),
            "t": float(stat), "p": float(p),
            "significant": sig,
            "interpretation": f"Mean {'is' if not sig else 'is not'} equal to {req.mu} (p={'<0.001' if p < 0.001 else f'{p:.4f}'})",
        }


# ── Chi-Square ───────────────────────────────────────────────────────────────

class ChiSqRequest(BaseModel):
    session_id: str
    row_column: str
    col_column: str


@router.post("/chisquare")
def chisquare(req: ChiSqRequest):
    df = _get_df(req.session_id)
    ct = pd.crosstab(df[req.row_column], df[req.col_column])
    chi2, p, dof, expected = scipy_stats.chi2_contingency(ct)
    sig = bool(p < 0.05)
    return {
        "test": "Chi-square test of independence",
        "chi2": float(chi2), "p": float(p), "dof": int(dof),
        "significant": sig,
        "crosstab": ct.to_dict(),
        "interpretation": f"{'Significant' if sig else 'No significant'} association (p={'<0.001' if p < 0.001 else f'{p:.4f}'})",
    }


# ── Correlation ──────────────────────────────────────────────────────────────

@router.get("/{session_id}/correlation")
def correlation(session_id: str, method: str = "pearson"):
    df = _get_df(session_id)
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method=method)
    p_values = {}
    for c1 in corr.columns:
        p_values[c1] = {}
        for c2 in corr.columns:
            if c1 == c2:
                p_values[c1][c2] = 0.0
            else:
                s1, s2 = num_df[[c1, c2]].dropna().values.T
                if method == "pearson":
                    _, p = scipy_stats.pearsonr(s1, s2)
                elif method == "spearman":
                    _, p = scipy_stats.spearmanr(s1, s2)
                else:
                    _, p = scipy_stats.kendalltau(s1, s2)
                p_values[c1][c2] = float(p)
    return {
        "method": method,
        "columns": corr.columns.tolist(),
        "matrix": corr.round(4).where(pd.notnull(corr), None).to_dict(),
        "p_values": p_values,
    }


# ── Mann-Whitney U ────────────────────────────────────────────────────────────

class MannWhitneyRequest(BaseModel):
    session_id: str
    column: str
    group_column: str


@router.post("/mannwhitney")
def mannwhitney(req: MannWhitneyRequest):
    df = _get_df(req.session_id)
    groups = df[req.group_column].dropna().unique()
    if len(groups) != 2:
        raise HTTPException(status_code=400, detail="Group column must have exactly 2 groups")
    g1 = df[df[req.group_column] == groups[0]][req.column].dropna()
    g2 = df[df[req.group_column] == groups[1]][req.column].dropna()
    stat, p = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    sig = bool(p < 0.05)
    return {
        "test": "Mann-Whitney U test",
        "group1": str(groups[0]), "n1": int(len(g1)),
        "median1": float(g1.median()), "iqr1": float(g1.quantile(0.75) - g1.quantile(0.25)),
        "group2": str(groups[1]), "n2": int(len(g2)),
        "median2": float(g2.median()), "iqr2": float(g2.quantile(0.75) - g2.quantile(0.25)),
        "U": float(stat), "p": float(p),
        "significant": sig,
        "interpretation": f"{'Significant' if sig else 'No significant'} difference between groups (p={'<0.001' if p < 0.001 else f'{p:.4f}'})",
    }


# ── Fisher's Exact Test ───────────────────────────────────────────────────────

class FisherRequest(BaseModel):
    session_id: str
    row_column: str
    col_column: str


@router.post("/fisher")
def fisher_exact(req: FisherRequest):
    df = _get_df(req.session_id)
    ct = pd.crosstab(df[req.row_column], df[req.col_column])
    if ct.shape != (2, 2):
        raise HTTPException(status_code=400, detail="Fisher's exact test requires a 2×2 table")
    table = ct.values.tolist()
    odds_ratio, p = scipy_stats.fisher_exact(ct.values)
    sig = bool(p < 0.05)
    return {
        "test": "Fisher's exact test",
        "odds_ratio": float(odds_ratio), "p": float(p),
        "significant": sig,
        "table": table,
        "row_labels": ct.index.tolist(),
        "col_labels": ct.columns.tolist(),
        "interpretation": f"{'Significant' if sig else 'No significant'} association (p={'<0.001' if p < 0.001 else f'{p:.4f}'})",
    }


# ── Kruskal-Wallis ────────────────────────────────────────────────────────────

class KruskalRequest(BaseModel):
    session_id: str
    column: str
    group_column: str


@router.post("/kruskal")
def kruskal(req: KruskalRequest):
    df = _get_df(req.session_id)
    group_data = [g[req.column].dropna().values for _, g in df.groupby(req.group_column)]
    if len(group_data) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 groups")
    stat, p = scipy_stats.kruskal(*group_data)
    sig = bool(p < 0.05)
    group_stats = df.groupby(req.group_column)[req.column].agg(
        n="count", median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    ).reset_index()
    return {
        "test": "Kruskal-Wallis test",
        "H": float(stat), "p": float(p),
        "significant": sig,
        "groups": [
            {k: (float(v) if hasattr(v, '__float__') else str(v)) for k, v in row.items()}
            for row in group_stats.to_dict(orient="records")
        ],
        "interpretation": f"{'Significant' if sig else 'No significant'} difference across groups (p={'<0.001' if p < 0.001 else f'{p:.4f}'})",
    }


# ── ROC Analysis ──────────────────────────────────────────────────────────────

class ROCRequest(BaseModel):
    session_id: str
    score_column: str
    outcome_column: str
    manual_cutoff: Optional[float] = None
    imputation: Optional[str] = "listwise"  # "listwise" | "median" | "mice"


def _roc_metrics_at_cutoff(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Compute full diagnostic metrics at a given threshold."""
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv   = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc   = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("inf")
    return {
        "cutoff": round(float(threshold), 6),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "accuracy": round(acc, 4),
        "lr_pos": round(lr_pos, 4) if not np.isinf(lr_pos) else None,
        "lr_neg": round(lr_neg, 4) if not np.isinf(lr_neg) else None,
        "youden_j": round(sens + spec - 1, 4),
    }


def _delong_placement_values(y: np.ndarray, scores: np.ndarray):
    """Return (V_pos, V_neg) placement value arrays for DeLong variance."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n1, n0 = len(pos_idx), len(neg_idx)
    s_pos = scores[pos_idx]
    s_neg = scores[neg_idx]
    # Placement value for each positive: Pr(neg < pos) + 0.5*Pr(neg == pos)
    V_pos = (
        np.sum(s_neg[:, None] < s_pos[None, :], axis=0).astype(float)
        + 0.5 * np.sum(s_neg[:, None] == s_pos[None, :], axis=0).astype(float)
    ) / n0
    # Placement value for each negative: Pr(neg > pos) + 0.5*Pr(neg == pos)
    V_neg = (
        np.sum(s_pos[:, None] > s_neg[None, :], axis=0).astype(float)
        + 0.5 * np.sum(s_pos[:, None] == s_neg[None, :], axis=0).astype(float)
    ) / n1
    return V_pos, V_neg


def _delong_compare(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """DeLong 1988 non-parametric AUC comparison. Returns diff, Z, p."""
    V_pos1, V_neg1 = _delong_placement_values(y, s1)
    V_pos2, V_neg2 = _delong_placement_values(y, s2)
    n1, n0 = len(V_pos1), len(V_neg1)
    auc1 = float(np.mean(V_pos1))
    auc2 = float(np.mean(V_pos2))
    # Covariance matrix of [AUC1, AUC2]
    s11 = np.var(V_pos1, ddof=1) / n1 + np.var(V_neg1, ddof=1) / n0
    s22 = np.var(V_pos2, ddof=1) / n1 + np.var(V_neg2, ddof=1) / n0
    s12 = (np.cov(V_pos1, V_pos2, ddof=1)[0, 1] / n1
           + np.cov(V_neg1, V_neg2, ddof=1)[0, 1] / n0)
    var_diff = s11 + s22 - 2 * s12
    diff = auc1 - auc2
    if var_diff <= 0:
        return {"auc_1": round(auc1, 4), "auc_2": round(auc2, 4),
                "difference": round(diff, 4), "z": 0.0, "p": 1.0}
    z = diff / np.sqrt(var_diff)
    p = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
    return {
        "auc_1": round(auc1, 4),
        "auc_2": round(auc2, 4),
        "difference": round(diff, 4),
        "z": round(float(z), 4),
        "p": round(p, 6),
        "significant": bool(p < 0.05),
    }


def _validate_roc_inputs(df: pd.DataFrame, score_col: str, outcome_col: str,
                         imputation: str = "listwise"):
    """Validate + return (scores_arr, y_arr, clean_df). Raises HTTPException on error."""
    for col in [score_col, outcome_col]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    df = apply_imputation(df, [score_col, outcome_col], imputation)
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Not enough data (need ≥ 10 rows after removing missing)")
    try:
        y = df[outcome_col].astype(float).astype(int)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Outcome '{outcome_col}' could not be converted to 0/1")
    uniq = sorted(y.unique().tolist())
    if len(uniq) != 2:
        raise HTTPException(status_code=400, detail=f"Outcome must have exactly 2 unique values. Found: {uniq[:6]}")
    if set(uniq) != {0, 1}:
        raise HTTPException(status_code=400, detail=f"Outcome values must be 0 and 1. Found: {uniq}")
    try:
        scores = df[score_col].astype(float)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Score column '{score_col}' must be numeric")
    if scores.nunique() < 2:
        raise HTTPException(status_code=400, detail=f"Score column '{score_col}' has no variation (constant)")
    return scores.values, y.values, df


@router.post("/roc")
def roc_analysis(req: ROCRequest):
    from sklearn.metrics import roc_curve, roc_auc_score

    df_full = _get_df(req.session_id)
    scores_arr, y_arr, df = _validate_roc_inputs(
        df_full, req.score_column, req.outcome_column,
        imputation=req.imputation or "listwise"
    )

    try:
        fpr, tpr, thresholds = roc_curve(y_arr, scores_arr)
        auc = float(roc_auc_score(y_arr, scores_arr))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ROC computation failed: {exc}")

    # Youden's J optimal cutoff
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    optimal = _roc_metrics_at_cutoff(scores_arr, y_arr, best_thresh)

    # Manual cutoff (if provided)
    manual = None
    if req.manual_cutoff is not None:
        manual = _roc_metrics_at_cutoff(scores_arr, y_arr, req.manual_cutoff)

    # Downsample curve to 300 points for response size
    n_pts = len(fpr)
    step = max(1, n_pts // 300)
    # Always include first and last points
    indices = list(range(0, n_pts, step))
    if (n_pts - 1) not in indices:
        indices.append(n_pts - 1)
    curve = [{"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6),
               "threshold": round(float(thresholds[i]), 6)} for i in indices]

    return _sanitize({
        "test": "ROC Analysis",
        "n": len(df),
        "n_positive": int(y_arr.sum()),
        "n_negative": int((y_arr == 0).sum()),
        "auc": round(auc, 4),
        # Optimal (Youden's J) — kept at top level for backward compat
        "optimal_cutoff": optimal["cutoff"],
        "sensitivity": optimal["sensitivity"],
        "specificity": optimal["specificity"],
        "tp": optimal["tp"], "tn": optimal["tn"],
        "fp": optimal["fp"], "fn": optimal["fn"],
        # Full metric objects
        "optimal": optimal,
        "manual": manual,
        "curve": curve,
        "interpretation": (
            f"AUC = {auc:.3f} — "
            f"{'Excellent' if auc >= 0.9 else 'Good' if auc >= 0.8 else 'Fair' if auc >= 0.7 else 'Poor'} "
            "discriminative ability"
        ),
    })


# ── ROC Comparison (DeLong Test) ──────────────────────────────────────────────

class ROCCompareRequest(BaseModel):
    session_id: str
    score_column_1: str
    score_column_2: str
    outcome_column: str


@router.post("/roc_compare")
def roc_compare(req: ROCCompareRequest):
    from sklearn.metrics import roc_curve, roc_auc_score

    df_full = _get_df(req.session_id)
    s1_arr, y_arr, _  = _validate_roc_inputs(df_full, req.score_column_1, req.outcome_column)
    s2_arr, y_arr2, _ = _validate_roc_inputs(df_full, req.score_column_2, req.outcome_column)

    if not np.array_equal(y_arr, y_arr2):
        # Different NaN patterns — use common complete rows
        df_clean = df_full.dropna(subset=[req.score_column_1, req.score_column_2, req.outcome_column])
        if len(df_clean) < 10:
            raise HTTPException(status_code=400, detail="Not enough complete rows for comparison (need ≥ 10)")
        y_arr  = df_clean[req.outcome_column].astype(float).astype(int).values
        s1_arr = df_clean[req.score_column_1].astype(float).values
        s2_arr = df_clean[req.score_column_2].astype(float).values

    result = _delong_compare(y_arr, s1_arr, s2_arr)
    result["score_1"] = req.score_column_1
    result["score_2"] = req.score_column_2
    result["n"] = int(len(y_arr))

    auc1, auc2 = result["auc_1"], result["auc_2"]
    diff = result["difference"]
    p = result["p"]
    p_str = "<0.001" if p < 0.001 else f"{p:.4f}"
    winner = req.score_column_1 if diff > 0 else req.score_column_2
    loser  = req.score_column_2 if diff > 0 else req.score_column_1
    if result["significant"]:
        result["interpretation"] = (
            f"{winner} has significantly better AUC than {loser} "
            f"(ΔAUC = {abs(diff):.4f}, p = {p_str})"
        )
    else:
        result["interpretation"] = (
            f"No significant difference between AUCs "
            f"(ΔAUC = {abs(diff):.4f}, p = {p_str})"
        )

    return _sanitize(result)


# ── ROC Combined Model ─────────────────────────────────────────────────────────

class ROCCombinedRequest(BaseModel):
    session_id: str
    predictor_columns: List[str]
    outcome_column: str
    model_name: Optional[str] = "Combined Model"


@router.post("/roc_combined")
def roc_combined(req: ROCCombinedRequest):
    """Fit a logistic regression on selected predictors, then run ROC on predicted probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    df_full = _get_df(req.session_id)

    if req.outcome_column not in df_full.columns:
        raise HTTPException(status_code=400, detail=f"Outcome column '{req.outcome_column}' not found")
    missing_cols = [c for c in req.predictor_columns if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Predictor column(s) not found: {missing_cols}")
    if len(req.predictor_columns) < 1:
        raise HTTPException(status_code=400, detail="At least one predictor column is required")

    cols = req.predictor_columns + [req.outcome_column]
    df = df_full.dropna(subset=cols)
    if len(df) < 20:
        raise HTTPException(status_code=400, detail="Not enough complete rows after removing missing (need ≥ 20)")

    # Encode predictors: numeric → use as-is, categorical → one-hot
    parts = []
    for col in req.predictor_columns:
        col_s = df[col]
        if pd.api.types.is_numeric_dtype(col_s):
            parts.append(col_s.rename(col).to_frame())
        else:
            parts.append(pd.get_dummies(col_s, prefix=col, drop_first=True))
    X = pd.concat(parts, axis=1).astype(float).values

    try:
        y = df[req.outcome_column].astype(float).astype(int).values
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Outcome could not be converted to 0/1 integers")
    uniq = sorted(set(y.tolist()))
    if set(uniq) != {0, 1}:
        raise HTTPException(status_code=400, detail=f"Outcome must be exactly 0 and 1. Found: {uniq}")

    # Fit logistic regression
    try:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        model.fit(X_sc, y)
        prob = model.predict_proba(X_sc)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model fitting failed: {exc}")

    # ROC on predicted probabilities
    try:
        fpr, tpr, thresholds = roc_curve(y, prob)
        auc = float(roc_auc_score(y, prob))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ROC computation failed: {exc}")

    # Youden's J optimal cutoff on probabilities
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    optimal = _roc_metrics_at_cutoff(prob, y, best_thresh)

    # Downsample curve to 300 points
    n_pts = len(fpr)
    step = max(1, n_pts // 300)
    indices = list(range(0, n_pts, step))
    if (n_pts - 1) not in indices:
        indices.append(n_pts - 1)
    curve = [
        {"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6)}
        for i in indices
    ]

    return _sanitize({
        "test": "ROC Analysis (Combined Model)",
        "model_name": req.model_name,
        "predictors": req.predictor_columns,
        "n": int(len(df)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "auc": round(auc, 4),
        "optimal": optimal,
        "curve": curve,
    })


# ── Sparklines (mini per-column distribution data for variable lists) ─────────

@router.get("/{session_id}/sparklines")
def get_sparklines(session_id: str):
    df = _get_df(session_id)
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


# ── Raw column values (for SPLOM scatterplot matrix) ─────────────────────────

@router.get("/{session_id}/raw")
def get_raw_columns(session_id: str, columns: str = ""):
    df = _get_df(session_id)
    cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns] if columns else list(df.columns)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])][:12]  # limit to 12 numeric cols
    result = {}
    for col in cols:
        vals = df[col].where(df[col].notna(), other=None).tolist()[:3000]
        result[col] = vals
    return result


# ── Column Summary (Wizard-style: histogram+QQ or donut+bar) ─────────────────

@router.get("/{session_id}/column_summary")
def column_summary(session_id: str, column: str, kind: Optional[str] = None):
    df = _get_df(session_id)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
    s = df[column]

    # Use provided kind hint; fall back to dtype + nunique heuristic
    if kind == "numeric":
        is_num = True
    elif kind in ("categorical", "text", "boolean"):
        is_num = False
    else:
        is_num = pd.api.types.is_numeric_dtype(s) and s.nunique() > 10

    if is_num:
        s_clean = s.dropna().astype(float)
        # Histogram (auto bins, max 40)
        n_bins = min(40, max(10, int(np.sqrt(len(s_clean)))))
        counts, edges = np.histogram(s_clean, bins=n_bins)
        histogram = [
            {"bin_start": float(edges[i]), "bin_end": float(edges[i+1]), "count": int(counts[i])}
            for i in range(len(counts))
        ]
        # QQ plot
        (theo, sample), _ = scipy_stats.probplot(s_clean)
        step = max(1, len(theo) // 300)
        qq = [{"x": float(theo[i]), "y": float(sample[i])} for i in range(0, len(theo), step)]
        # Normality: Shapiro-Wilk for n<50, Kolmogorov-Smirnov for n≥50
        p_norm, norm_test_name = _normality_test(s_clean)
        q1, q3 = float(s_clean.quantile(0.25)), float(s_clean.quantile(0.75))
        return {
            "type": "numeric",
            "n": int(s_clean.count()), "missing": int(s.isna().sum()),
            "mean": float(s_clean.mean()), "std": float(s_clean.std()),
            "median": float(s_clean.median()), "q1": q1, "q3": q3,
            "iqr": float(q3 - q1), "min": float(s_clean.min()), "max": float(s_clean.max()),
            "skewness": float(s_clean.skew()), "kurtosis": float(s_clean.kurtosis()),
            "histogram": histogram,
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


# ── Table 1 (clinical baseline characteristics) ───────────────────────────────

class Table1Request(BaseModel):
    session_id: str
    group_column: Optional[str] = None
    variables: list[str]
    variable_kinds: Optional[dict] = None   # {col: "numeric"|"categorical"}
    selected_stats: Optional[list[str]] = None  # ["auto","mean_sd","median_iqr","se","ci95","variance","min_max","n","missing","p10","p25","p75","p90","p95"]


def _fmt_p(p: float) -> str:
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"


# ── per-stat formatters ────────────────────────────────────────────────────────

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
    """Format a float safely; return '—' for NaN/Inf."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{d}f}"


def _fmt_one_stat(a: pd.Series, stat: str) -> str:
    """Format a single statistic for a series (already dropna'd & float)."""
    if len(a) == 0:
        return "—"
    if stat == "mean_sd":
        return f"{_f(a.mean())} ± {_f(a.std())}"
    if stat == "median_iqr":
        q1, q3 = a.quantile(0.25), a.quantile(0.75)
        return f"{_f(a.median())} [{_f(q1)}–{_f(q3)}]"
    if stat == "se":
        return _f(a.sem(), 3)
    if stat == "ci95":
        if len(a) < 2:
            return "—"
        se = a.sem()
        m = a.mean()
        t_crit = scipy_stats.t.ppf(0.975, df=len(a) - 1)
        ci = t_crit * se
        return f"{_f(m)} [{_f(m - ci)}–{_f(m + ci)}]"
    if stat == "variance":
        return _f(a.var(), 3)
    if stat == "min_max":
        return f"{_f(a.min())} – {_f(a.max())}"
    if stat == "n":
        return str(int(len(a)))
    if stat == "missing":
        return str(int(a.isna().sum()) if hasattr(a, 'isna') else 0)
    pct_map = {"p10": 0.10, "p25": 0.25, "p75": 0.75, "p90": 0.90, "p95": 0.95}
    if stat in pct_map:
        return _f(a.quantile(pct_map[stat]))
    return "—"


def _build_stat_rows(
    s_col: pd.Series,
    group_series: dict[str, pd.Series],  # gl → series (not yet dropna'd)
    stats: list[str],
    normal: bool,
) -> list[dict]:
    """Build a list of {label, overall, group_stats} for numeric variable."""
    rows_out = []
    s_all = s_col.dropna().astype(float)

    # Handle 'missing' stat specially (needs original series)
    for stat in stats:
        resolved = stat
        if stat == "auto":
            resolved = "mean_sd" if normal else "median_iqr"

        label = _STAT_LABELS.get(resolved, resolved)
        if resolved == "missing":
            overall_val = str(int(s_col.isna().sum()))
            grp_vals = {gl: str(int(gs.isna().sum())) for gl, gs in group_series.items()}
        else:
            overall_val = _fmt_one_stat(s_all, resolved)
            grp_vals = {
                gl: _fmt_one_stat(gs.dropna().astype(float), resolved)
                for gl, gs in group_series.items()
            }

        rows_out.append({"label": label, "overall": overall_val, "group_stats": grp_vals})
    return rows_out


def _normality_test(s_clean: pd.Series) -> tuple[float, str]:
    """Return (p_value, test_name). Uses Shapiro-Wilk for n<50, K-S for n≥50."""
    n = len(s_clean)
    if n < 3:
        return 1.0, "—"
    if n < 50:
        _, p = scipy_stats.shapiro(s_clean[:5000])
        return float(p), "Shapiro-Wilk"
    else:
        # Kolmogorov-Smirnov against N(μ,σ) with estimated parameters
        _, p = scipy_stats.kstest(
            s_clean, "norm", args=(float(s_clean.mean()), float(s_clean.std()))
        )
        return float(p), "Kolmogorov-Smirnov"


@router.post("/table1")
def table1(req: Table1Request):
    df = _get_df(req.session_id)
    rows = []

    # Default stats = ["auto"] (normality-based)
    sel_stats: list[str] = req.selected_stats if req.selected_stats else ["auto"]

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
            normal = p_norm > 0.05

            # Build per-group series map  {label → raw series}
            group_series: dict[str, pd.Series] = {}
            group_arrs: list[pd.Series] = []
            if groups is not None:
                for g, gl in zip(groups, group_labels):
                    gs = df[df[req.group_column] == g][var]
                    group_series[gl] = gs
                    group_arrs.append(gs.dropna().astype(float))

            stat_rows = _build_stat_rows(s, group_series, sel_stats, normal)

            # Statistical test for group comparison
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
                except Exception:
                    p_value_str = "N/A"

            row: dict = {
                "variable": var,
                "type": "numeric",
                "overall_n": int(len(s_all)),
                "normal": normal,
                "normality_test": norm_test_name,
                "normality_p": round(p_norm, 4),
                "stat_rows": stat_rows,
                "p_value": p_value_str,
                "test": test_name_str,
                "significant": significant,
                # Legacy fields (for backward compat)
                "stat_label": stat_rows[0]["label"] if stat_rows else "",
                "overall": stat_rows[0]["overall"] if stat_rows else "",
                "group_stats": stat_rows[0]["group_stats"] if stat_rows else {},
            }

        else:
            # Categorical
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
                    chi2, p_chi_raw, dof, expected = scipy_stats.chi2_contingency(ct)
                    if ct.shape == (2, 2) and (expected < 5).any():
                        _, p_chi_raw = scipy_stats.fisher_exact(ct.values)
                        test_name = "Fisher"
                    else:
                        test_name = "Chi-square"
                    p_val = _fmt_p(float(p_chi_raw))
                except Exception:
                    p_val = "N/A"

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
            }
        rows.append(row)

    return _sanitize({
        "group_column": req.group_column,
        "group_labels": group_labels,
        "group_ns": group_ns,
        "total_n": len(df),
        "rows": rows,
    })


# ── ANOVA ─────────────────────────────────────────────────────────────────────

class AnovaRequest(BaseModel):
    session_id: str
    column: str
    group_column: str


@router.post("/anova")
def anova(req: AnovaRequest):
    df = _get_df(req.session_id)
    groups = [g[req.column].dropna().values for _, g in df.groupby(req.group_column)]
    if len(groups) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 groups")
    stat, p = scipy_stats.f_oneway(*groups)
    sig = bool(p < 0.05)
    group_stats = df.groupby(req.group_column)[req.column].agg(["count", "mean", "std"]).reset_index()
    return {
        "test": "One-way ANOVA",
        "F": float(stat), "p": float(p),
        "significant": sig,
        "groups": [
            {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in row.items()}
            for row in group_stats.to_dict(orient="records")
        ],
        "interpretation": f"{'Significant' if sig else 'No significant'} difference across groups (p={'<0.001' if p < 0.001 else f'{p:.4f}'})",
    }


# ── Pairwise Correlation ──────────────────────────────────────────────────────

class CorrelationPairRequest(BaseModel):
    session_id: str
    var1: str
    var2: str
    method: Optional[str] = "auto"   # "auto" | "pearson" | "spearman"
    imputation: Optional[str] = "listwise"


@router.post("/correlation_pair")
def correlation_pair(req: CorrelationPairRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.var1, req.var2], req.imputation or "listwise")
    x = df[req.var1].astype(float).values
    y = df[req.var2].astype(float).values
    n = len(x)
    n_excluded = n_total - n
    if n < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 observations")

    # Normality check (Shapiro-Wilk, only reliable for n≤5000)
    sw1_stat, sw1_p = scipy_stats.shapiro(x[:5000])
    sw2_stat, sw2_p = scipy_stats.shapiro(y[:5000])
    normal1 = bool(sw1_p >= 0.05)
    normal2 = bool(sw2_p >= 0.05)

    # Method selection
    method = req.method or "auto"
    if method == "auto":
        use_pearson = normal1 and normal2
    else:
        use_pearson = method == "pearson"

    if use_pearson:
        r, p = scipy_stats.pearsonr(x, y)
        method_used = "pearson"
        label = "r"
    else:
        r, p = scipy_stats.spearmanr(x, y)
        method_used = "spearman"
        label = "ρ"

    # 95% CI via Fisher z-transformation
    if abs(r) < 1.0:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        ci_low = float(np.tanh(z - 1.96 * se))
        ci_high = float(np.tanh(z + 1.96 * se))
    else:
        ci_low, ci_high = float(r), float(r)

    # Scatter data
    scatter_x = x.tolist()
    scatter_y = y.tolist()

    # Regression line (OLS) for plot
    slope, intercept, *_ = scipy_stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    # 95% CI band around regression line
    x_mean = x.mean()
    ss_x = np.sum((x - x_mean) ** 2)
    residuals = y - (slope * x + intercept)
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    t_crit = scipy_stats.t.ppf(0.975, df=n - 2)
    ci_band = t_crit * s_err * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_x)

    return {
        "method": method_used,
        "label": label,
        "n": n,
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "r": float(r),
        "p": float(p),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "normality": {
            req.var1: {"statistic": float(sw1_stat), "p": float(sw1_p), "normal": normal1},
            req.var2: {"statistic": float(sw2_stat), "p": float(sw2_p), "normal": normal2},
        },
        "scatter": {"x": scatter_x, "y": scatter_y},
        "regression_line": {
            "x": x_line.tolist(),
            "y": y_line.tolist(),
            "slope": float(slope),
            "intercept": float(intercept),
        },
        "ci_band": {
            "x": x_line.tolist(),
            "y_upper": (y_line + ci_band).tolist(),
            "y_lower": (y_line - ci_band).tolist(),
        },
    }


# ── Correlation Matrix ────────────────────────────────────────────────────────

class CorrelationMatrixRequest(BaseModel):
    session_id: str
    variables: List[str]
    method: Optional[str] = "pearson"
    imputation: Optional[str] = "listwise"


@router.post("/correlation_matrix")
def correlation_matrix_post(req: CorrelationMatrixRequest):
    raw = _get_df(req.session_id)[req.variables].apply(pd.to_numeric, errors="coerce")
    df = apply_imputation(raw, req.variables, req.imputation or "listwise")
    if len(req.variables) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 variables")

    method = req.method or "pearson"
    corr = df.corr(method=method)

    # p-value matrix (pairwise)
    p_matrix: dict = {}
    for c1 in req.variables:
        p_matrix[c1] = {}
        for c2 in req.variables:
            if c1 == c2:
                p_matrix[c1][c2] = None
            else:
                pair = df[[c1, c2]].dropna()
                if method == "spearman":
                    _, pv = scipy_stats.spearmanr(pair[c1], pair[c2])
                elif method == "kendall":
                    _, pv = scipy_stats.kendalltau(pair[c1], pair[c2])
                else:
                    _, pv = scipy_stats.pearsonr(pair[c1], pair[c2])
                p_matrix[c1][c2] = float(pv)

    # Multicollinearity warnings: |r| >= 0.70
    warnings = []
    vars_list = req.variables
    for i in range(len(vars_list)):
        for j in range(i + 1, len(vars_list)):
            r_val = corr.loc[vars_list[i], vars_list[j]]
            if abs(r_val) >= 0.70:
                warnings.append({
                    "var1": vars_list[i],
                    "var2": vars_list[j],
                    "r": float(r_val),
                    "severity": "high" if abs(r_val) >= 0.90 else "moderate",
                })

    matrix_dict = {c: {r: (float(corr.loc[r, c]) if not pd.isna(corr.loc[r, c]) else None)
                        for r in req.variables} for c in req.variables}

    return {
        "method": method,
        "variables": req.variables,
        "n": len(df),
        "matrix": matrix_dict,
        "p_matrix": p_matrix,
        "multicollinearity_warnings": warnings,
    }


# ── ICC(2,1) ──────────────────────────────────────────────────────────────────

class ICCRequest(BaseModel):
    session_id: str
    rater1_col: str
    rater2_col: str


@router.post("/icc")
def icc_endpoint(req: ICCRequest):
    df = _get_df(req.session_id).dropna(subset=[req.rater1_col, req.rater2_col])
    r1 = df[req.rater1_col].astype(float).values
    r2 = df[req.rater2_col].astype(float).values
    n = len(r1)
    k = 2  # raters
    if n < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 observations")

    # ANOVA decomposition for ICC(2,1) — Shrout & Fleiss 1979
    grand_mean = np.mean(np.stack([r1, r2]))
    subject_means = (r1 + r2) / 2.0
    rater_means = np.array([r1.mean(), r2.mean()])

    SS_b = k * np.sum((subject_means - grand_mean) ** 2)
    SS_r = n * np.sum((rater_means - grand_mean) ** 2)
    SS_total = np.sum((r1 - grand_mean) ** 2) + np.sum((r2 - grand_mean) ** 2)
    SS_e = SS_total - SS_b - SS_r

    df_b = n - 1
    df_r = k - 1
    df_e = (n - 1) * (k - 1)

    MS_b = SS_b / df_b
    MS_r = SS_r / df_r if df_r > 0 else 0.0
    MS_e = SS_e / df_e if df_e > 0 else 1e-9

    # ICC(2,1) absolute agreement
    icc_val = (MS_b - MS_e) / (MS_b + (k - 1) * MS_e + k * (MS_r - MS_e) / n)
    icc_val = float(np.clip(icc_val, -1.0, 1.0))

    # 95% CI (Shrout & Fleiss)
    F_lower = scipy_stats.f.ppf(0.975, df_b, df_e)
    F_upper = scipy_stats.f.ppf(0.025, df_b, df_e)
    F_obs = MS_b / MS_e if MS_e > 0 else 0.0
    ci_low = float((F_obs / F_lower - 1) / (F_obs / F_lower + k - 1)) if F_lower > 0 else 0.0
    ci_high = float((F_obs / F_upper - 1) / (F_obs / F_upper + k - 1)) if F_upper > 0 else 1.0
    ci_low = float(np.clip(ci_low, -1.0, 1.0))
    ci_high = float(np.clip(ci_high, -1.0, 1.0))

    # F-test p-value
    f_p = float(scipy_stats.f.sf(F_obs, df_b, df_e))

    # Interpretation
    if icc_val >= 0.90:
        interp = "Excellent"
    elif icc_val >= 0.75:
        interp = "Good"
    elif icc_val >= 0.50:
        interp = "Moderate"
    else:
        interp = "Poor"

    # Bland-Altman data
    means = ((r1 + r2) / 2).tolist()
    diffs = (r1 - r2).tolist()
    mean_diff = float(np.mean(r1 - r2))
    sd_diff = float(np.std(r1 - r2, ddof=1))
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    return {
        "icc": icc_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "f_stat": float(F_obs),
        "f_p": f_p,
        "n": n,
        "interpretation": interp,
        "bland_altman": {
            "means": means,
            "diffs": diffs,
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "loa_upper": float(loa_upper),
            "loa_lower": float(loa_lower),
        },
    }


# ── Cohen's Kappa ─────────────────────────────────────────────────────────────

class KappaRequest(BaseModel):
    session_id: str
    rater1_col: str
    rater2_col: str


@router.post("/cohens_kappa")
def cohens_kappa(req: KappaRequest):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix as sk_confusion

    df = _get_df(req.session_id).dropna(subset=[req.rater1_col, req.rater2_col])
    r1 = df[req.rater1_col].astype(str).values
    r2 = df[req.rater2_col].astype(str).values
    n = len(r1)
    if n < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 observations")

    kappa = float(cohen_kappa_score(r1, r2))

    # SE and 95% CI
    labels = sorted(set(r1) | set(r2))
    cm = sk_confusion(r1, r2, labels=labels)
    po = float(np.trace(cm) / n)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = float(np.sum(row_sums * col_sums) / (n ** 2))
    se = float(np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))) if (1 - pe) > 0 else 0.0
    ci_low = float(kappa - 1.96 * se)
    ci_high = float(kappa + 1.96 * se)

    # Landis & Koch interpretation
    if kappa >= 0.81:
        interp = "Almost Perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    elif kappa >= 0.0:
        interp = "Slight"
    else:
        interp = "Poor (< chance)"

    return {
        "kappa": kappa,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "n": n,
        "po": po,
        "pe": pe,
        "interpretation": interp,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


# ── Power Analysis ─────────────────────────────────────────────────────────────

class PowerRequest(BaseModel):
    test: str          # t_two | t_one | anova | correlation | proportion | chi2
    solve_for: str     # n | power | effect_size
    alpha: float = 0.05
    power: Optional[float] = None
    effect_size: Optional[float] = None
    n: Optional[int] = None
    tails: int = 2
    k_groups: int = 3   # ANOVA: number of groups; chi2: number of bins (df+1)
    ratio: float = 1.0  # n2/n1 for two-sample tests
    p1: Optional[float] = None
    p2: Optional[float] = None


@router.post("/power")
def run_power(req: PowerRequest):
    import numpy as np
    from scipy.stats import norm
    from statsmodels.stats.power import (
        TTestIndPower, TTestOneSamplePower,
        FTestAnovaPower, NormalIndPower, GofChisquarePower,
    )

    alt = "two-sided" if req.tails == 2 else "larger"
    a   = req.alpha

    def _ceil(x): return int(np.ceil(float(x)))

    def _curve(pw_fn, n_end, n_start=4, steps=80):
        pts, step = [], max(1, (n_end - n_start) // steps)
        for n in range(n_start, n_end + 1, step):
            try:
                pwr = float(pw_fn(n))
                if 0 <= pwr <= 1:
                    pts.append({"n": n, "power": round(pwr, 4)})
            except Exception:
                pass
        return pts

    result, label, curve = None, "", []

    # ── Two-sample t-test ──────────────────────────────────────────────────────
    if req.test == "t_two":
        ana = TTestIndPower()
        ratio = req.ratio or 1.0
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt)

        if req.solve_for == "n":
            n1 = _ceil(ana.solve_power(effect_size=req.effect_size, nobs1=None, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            result = n1
            label  = f"n₁ = {n1},  n₂ = {_ceil(n1*ratio)},  total N = {n1 + _ceil(n1*ratio)}"
            curve  = _curve(pw, max(n1 * 4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs1=req.n, alpha=a, power=None, ratio=ratio, alternative=alt))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs1=req.n, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            label  = f"Minimum detectable Cohen's d = {result:.4f}"
            d = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=d, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt), max(int(req.n)*4, 100))

    # ── One-sample / paired t-test ─────────────────────────────────────────────
    elif req.test == "t_one":
        ana = TTestOneSamplePower()
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, alternative=alt)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, alternative=alt))
            result, label, curve = n, f"n = {n}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, alternative=alt))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, alternative=alt))
            label  = f"Minimum detectable Cohen's d = {result:.4f}"
            d = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=d, nobs=n, alpha=a, power=None, alternative=alt), max(int(req.n)*4, 100))

    # ── One-way ANOVA ──────────────────────────────────────────────────────────
    elif req.test == "anova":
        ana, k = FTestAnovaPower(), req.k_groups
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, k_groups=k)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, k_groups=k))
            result, label, curve = n, f"n/group = {n},  total N = {n*k}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, k_groups=k))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, k_groups=k))
            label  = f"Minimum detectable Cohen's f = {result:.4f}"
            f_es = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=f_es, nobs=n, alpha=a, power=None, k_groups=k), max(int(req.n)*4, 100))

    # ── Pearson correlation (Fisher-z) ─────────────────────────────────────────
    elif req.test == "correlation":
        tails = req.tails

        def corr_power(r, n):
            if abs(r) >= 1 or n <= 3: return float("nan")
            ncp = np.arctanh(abs(r)) * np.sqrt(n - 3)
            z_c = norm.ppf(1 - a / (1 if tails == 1 else 2))
            return float(norm.sf(z_c - ncp) + (norm.cdf(-z_c - ncp) if tails == 2 else 0))

        def corr_solve_n(r, pwr):
            for n in range(4, 100001):
                if corr_power(r, n) >= pwr: return n
            return 100001

        def corr_solve_r(n, pwr):
            from scipy.optimize import brentq
            try:   return float(brentq(lambda r: corr_power(r, n) - pwr, 1e-6, 1 - 1e-6))
            except Exception: return None

        r_es = req.effect_size
        if req.solve_for == "n":
            n = corr_solve_n(r_es, req.power)
            result, label = n, f"n = {n}"
            curve = _curve(lambda n: corr_power(r_es, n), max(n*4, 100))
        elif req.solve_for == "power":
            result = corr_power(r_es, req.n)
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(lambda n: corr_power(r_es, n), max(int(req.n)*4, 100))
        else:
            r_sol = corr_solve_r(req.n, req.power)
            result = r_sol
            label  = f"Minimum detectable r = {r_sol:.4f}" if r_sol else "Could not converge"
            if r_sol:
                curve = _curve(lambda n: corr_power(r_sol, n), max(int(req.n)*4, 100))

    # ── Two proportions (Cohen's h) ────────────────────────────────────────────
    elif req.test == "proportion":
        ana   = NormalIndPower()
        ratio = req.ratio or 1.0
        p1    = req.p1 if req.p1 is not None else 0.5
        p2    = req.p2 if req.p2 is not None else 0.3
        h_from_p = abs(float(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))))

        if req.solve_for == "effect_size":
            eff = float(ana.solve_power(effect_size=None, nobs1=req.n, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            result = abs(eff)
            label  = f"Minimum detectable Cohen's h = {result:.4f}"
            h_sol = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=h_sol, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt), max(int(req.n)*4, 100))
        else:
            eff = req.effect_size if req.effect_size is not None else h_from_p
            def pw(n): return ana.solve_power(effect_size=eff, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt)
            if req.solve_for == "n":
                n1 = _ceil(ana.solve_power(effect_size=eff, nobs1=None, alpha=a, power=req.power, ratio=ratio, alternative=alt))
                result, label, curve = n1, f"n₁ = {n1},  n₂ = {_ceil(n1*ratio)},  total N = {n1+_ceil(n1*ratio)}", _curve(pw, max(n1*4, 100))
            else:
                result = float(ana.solve_power(effect_size=eff, nobs1=req.n, alpha=a, power=None, ratio=ratio, alternative=alt))
                label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
                curve  = _curve(pw, max(int(req.n)*4, 100))

    # ── Chi-square ─────────────────────────────────────────────────────────────
    elif req.test == "chi2":
        ana    = GofChisquarePower()
        n_bins = req.k_groups   # df = k_groups - 1
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, n_bins=n_bins)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, n_bins=n_bins))
            result, label, curve = n, f"n = {n}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, n_bins=n_bins))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, n_bins=n_bins))
            label  = f"Minimum detectable Cohen's w = {result:.4f}"
            w_es = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=w_es, nobs=n, alpha=a, power=None, n_bins=n_bins), max(int(req.n)*4, 100))
    else:
        raise HTTPException(400, f"Unknown test: {req.test}")

    return {"result": float(result) if result is not None else None, "label": label, "curve": curve}
