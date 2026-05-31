"""
Machine-learning predictive-modeling router.

Endpoints
---------
POST /random_forest      — Random Forest classifier / regressor
POST /gradient_boosting  — Gradient Boosting classifier / regressor
POST /feature_importance — permutation importance for a chosen model

Design notes
------------
- Pure scikit-learn (already a dependency). No xgboost / shap required for v1;
  feature importance uses sklearn's impurity importance plus model-agnostic
  permutation importance.
- Honest performance: classification metrics (AUC, calibration, confusion)
  come from out-of-fold predictions via ``cross_val_predict`` so they are not
  optimistic in-sample numbers. The final model is then refit on the full
  data for the importance ranking.
- Categorical predictors are one-hot encoded (drop_first=True) the same way
  the logistic / Cox endpoints in models.py do, so dummy names line up.
- Missing values handled through the shared ``apply_imputation`` service.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store
from services.impute import apply_imputation

router = APIRouter()

_Z95 = 1.959963984540054


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _safe(v: Any) -> Any:
    """JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _downsample_curve(fpr: np.ndarray, tpr: np.ndarray, max_pts: int = 300) -> List[dict]:
    n = len(fpr)
    step = max(1, n // max_pts)
    idx = list(range(0, n, step))
    if (n - 1) not in idx:
        idx.append(n - 1)
    return [{"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6)} for i in idx]


def _encode(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """One-hot encode categorical predictors, keep numerics, all float."""
    raw = df[predictors].copy()
    numeric, categorical = [], []
    for c in predictors:
        col = raw[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric.append(c)
        else:
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().mean() >= 0.8 and coerced.dropna().nunique() > 2:
                raw[c] = coerced
                numeric.append(c)
            else:
                categorical.append(c)
    num_part = raw[numeric].apply(pd.to_numeric, errors="coerce") if numeric else pd.DataFrame(index=raw.index)
    cat_part = pd.get_dummies(raw[categorical], drop_first=True, dummy_na=False) if categorical else pd.DataFrame(index=raw.index)
    enc = pd.concat([num_part, cat_part], axis=1)
    enc.columns = [str(c) for c in enc.columns]
    return enc.astype(float)


# ── Request model ───────────────────────────────────────────────────────────


class MLRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    task: Optional[str] = "auto"            # auto | classification | regression
    # Hyper-parameters (sensible clinical defaults)
    n_estimators: int = 300
    max_depth: Optional[int] = None          # None = grow until pure
    min_samples_leaf: int = 1
    learning_rate: float = 0.1               # gradient boosting only
    # Evaluation
    cv_folds: int = 5
    class_weight_balanced: bool = True       # classification imbalance
    n_permutation_repeats: int = 10
    random_state: int = 42
    imputation: Optional[str] = "listwise"


# ── Core evaluators ─────────────────────────────────────────────────────────


def _resolve_task(req: MLRequest, y: pd.Series) -> str:
    if req.task in ("classification", "regression"):
        return req.task
    # auto: binary / few-level integer → classification, else regression
    nun = y.nunique(dropna=True)
    if nun <= 2:
        return "classification"
    if pd.api.types.is_numeric_dtype(y) and nun > 10:
        return "regression"
    return "classification"


def _eval_classifier(estimator, X: pd.DataFrame, y: np.ndarray, req: MLRequest) -> dict:
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
    from sklearn.inspection import permutation_importance

    classes = np.unique(y)
    if len(classes) != 2:
        raise HTTPException(status_code=422,
            detail=f"Classification v1 supports a binary outcome (0/1). Found {len(classes)} classes: {classes.tolist()}.")
    # Map to 0/1 preserving the larger code as the positive event if it's {0,1}
    y01 = (y == classes.max()).astype(int)

    n = len(y01)
    n_pos = int(y01.sum())
    n_neg = n - n_pos
    folds = max(2, min(req.cv_folds, n_pos, n_neg))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=req.random_state)

    # Out-of-fold probabilities → honest AUC / calibration / confusion.
    proba = cross_val_predict(estimator, X.values, y01, cv=skf, method="predict_proba")[:, 1]

    auc = float(roc_auc_score(y01, proba))
    # Bootstrap percentile CI for AUC over the OOF probabilities.
    rng = np.random.default_rng(req.random_state)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, n, n)
        yb = y01[idx]
        if yb.min() == yb.max():
            continue
        boot.append(roc_auc_score(yb, proba[idx]))
    ci_low = float(np.percentile(boot, 2.5)) if boot else None
    ci_high = float(np.percentile(boot, 97.5)) if boot else None

    fpr, tpr, _ = roc_curve(y01, proba)
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y01 == 1)).sum())
    tn = int(((pred == 0) & (y01 == 0)).sum())
    fp = int(((pred == 1) & (y01 == 0)).sum())
    fn = int(((pred == 0) & (y01 == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    ppv = tp / (tp + fp) if (tp + fp) else None
    npv = tn / (tn + fn) if (tn + fn) else None
    acc = (tp + tn) / n
    brier = float(brier_score_loss(y01, proba))

    # Calibration bins (10 quantile bins of predicted probability).
    cal = []
    try:
        bins = pd.qcut(proba, q=min(10, len(np.unique(proba))), duplicates="drop")
        cal_df = pd.DataFrame({"p": proba, "y": y01, "bin": bins})
        for _, g in cal_df.groupby("bin", observed=True):
            cal.append({"pred": round(float(g["p"].mean()), 4),
                        "obs": round(float(g["y"].mean()), 4),
                        "n": int(len(g))})
    except Exception:
        cal = []

    # Refit on full data → impurity + permutation importance.
    estimator.fit(X.values, y01)
    imp_impurity = getattr(estimator, "feature_importances_", None)
    perm = permutation_importance(estimator, X.values, y01,
                                  n_repeats=req.n_permutation_repeats,
                                  random_state=req.random_state, scoring="roc_auc")
    importance = []
    for i, name in enumerate(X.columns):
        importance.append({
            "feature": str(name),
            "impurity": round(float(imp_impurity[i]), 6) if imp_impurity is not None else None,
            "permutation": round(float(perm.importances_mean[i]), 6),
            "permutation_sd": round(float(perm.importances_std[i]), 6),
        })
    importance.sort(key=lambda d: (d["permutation"] if d["permutation"] is not None else -1), reverse=True)

    interp = (
        f"{_estimator_label(estimator)} classifier on n = {n} "
        f"({n_pos} events, {n_neg} non-events), {folds}-fold cross-validated. "
        f"AUC = {auc:.3f}"
        + (f" (95% CI {ci_low:.3f}–{ci_high:.3f})" if ci_low is not None else "")
        + f". Accuracy {acc*100:.1f}%, sensitivity {sens*100:.1f}%, specificity {spec*100:.1f}% "
          f"at the 0.5 cutoff. Brier score {brier:.3f}. "
          f"Top predictor by permutation importance: {importance[0]['feature']}."
    )

    return {
        "task": "classification",
        "n": n, "n_events": n_pos, "n_non_events": n_neg, "cv_folds": folds,
        "auc": round(auc, 4), "auc_ci_low": round(ci_low, 4) if ci_low is not None else None,
        "auc_ci_high": round(ci_high, 4) if ci_high is not None else None,
        "accuracy": round(acc, 4), "sensitivity": _safe(round(sens, 4) if sens is not None else None),
        "specificity": _safe(round(spec, 4) if spec is not None else None),
        "ppv": _safe(round(ppv, 4) if ppv is not None else None),
        "npv": _safe(round(npv, 4) if npv is not None else None),
        "brier": round(brier, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "roc_curve": _downsample_curve(fpr, tpr),
        "calibration": cal,
        "importance": importance,
        "interpretation": interp,
    }


def _eval_regressor(estimator, X: pd.DataFrame, y: np.ndarray, req: MLRequest) -> dict:
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.inspection import permutation_importance

    n = len(y)
    folds = max(2, min(req.cv_folds, n))
    kf = KFold(n_splits=folds, shuffle=True, random_state=req.random_state)
    pred = cross_val_predict(estimator, X.values, y, cv=kf)

    r2 = float(r2_score(y, pred))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    resid = (y - pred)

    estimator.fit(X.values, y)
    imp_impurity = getattr(estimator, "feature_importances_", None)
    perm = permutation_importance(estimator, X.values, y,
                                  n_repeats=req.n_permutation_repeats,
                                  random_state=req.random_state, scoring="r2")
    importance = []
    for i, name in enumerate(X.columns):
        importance.append({
            "feature": str(name),
            "impurity": round(float(imp_impurity[i]), 6) if imp_impurity is not None else None,
            "permutation": round(float(perm.importances_mean[i]), 6),
            "permutation_sd": round(float(perm.importances_std[i]), 6),
        })
    importance.sort(key=lambda d: (d["permutation"] if d["permutation"] is not None else -1), reverse=True)

    # Predicted-vs-actual scatter (downsample to 500 points for payload).
    if n > 500:
        rng = np.random.default_rng(req.random_state)
        sel = rng.choice(n, 500, replace=False)
    else:
        sel = np.arange(n)
    scatter = [{"actual": round(float(y[i]), 4), "predicted": round(float(pred[i]), 4)} for i in sel]

    interp = (
        f"{_estimator_label(estimator)} regressor on n = {n}, {folds}-fold "
        f"cross-validated. R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}. "
        f"Top predictor by permutation importance: {importance[0]['feature']}."
    )

    return {
        "task": "regression",
        "n": n, "cv_folds": folds,
        "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
        "resid_mean": round(float(resid.mean()), 4), "resid_sd": round(float(resid.std(ddof=1)), 4),
        "scatter": scatter,
        "importance": importance,
        "interpretation": interp,
    }


def _estimator_label(est) -> str:
    name = type(est).__name__
    return {
        "RandomForestClassifier": "Random forest",
        "RandomForestRegressor": "Random forest",
        "GradientBoostingClassifier": "Gradient boosting",
        "GradientBoostingRegressor": "Gradient boosting",
    }.get(name, name)


def _prepare(req: MLRequest):
    df = _get_df(req.session_id)
    for c in [req.outcome, *req.predictors]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.predictors:
        raise HTTPException(status_code=422, detail="Select at least one predictor.")
    if req.outcome in req.predictors:
        raise HTTPException(status_code=422, detail="Outcome cannot also be a predictor.")

    cols = [req.outcome, *req.predictors]
    work = apply_imputation(df[cols], cols, req.imputation or "listwise").reset_index(drop=True)
    X = _encode(work, req.predictors)
    y_raw = work[req.outcome]
    keep = X.notna().all(axis=1) & y_raw.notna()
    X, y_raw = X[keep], y_raw[keep]
    if len(X) < 20:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows (need ≥ 20, got {len(X)}).")
    if X.shape[1] == 0:
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")
    task = _resolve_task(req, y_raw)
    y = pd.to_numeric(y_raw, errors="coerce").values if task == "regression" else y_raw.values
    if task == "regression" and np.isnan(y).any():
        raise HTTPException(status_code=422, detail="Regression outcome must be numeric.")
    return X, y, task


def _run(req: MLRequest, kind: str) -> dict:
    X, y, task = _prepare(req)
    cw = "balanced" if req.class_weight_balanced else None
    md = req.max_depth if (req.max_depth and req.max_depth > 0) else None

    if kind == "random_forest":
        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier
            est = RandomForestClassifier(
                n_estimators=req.n_estimators, max_depth=md,
                min_samples_leaf=req.min_samples_leaf, class_weight=cw,
                random_state=req.random_state, n_jobs=-1, oob_score=False)
        else:
            from sklearn.ensemble import RandomForestRegressor
            est = RandomForestRegressor(
                n_estimators=req.n_estimators, max_depth=md,
                min_samples_leaf=req.min_samples_leaf,
                random_state=req.random_state, n_jobs=-1)
        model_name = "Random Forest"
    else:  # gradient_boosting
        if task == "classification":
            from sklearn.ensemble import GradientBoostingClassifier
            est = GradientBoostingClassifier(
                n_estimators=req.n_estimators, max_depth=md or 3,
                min_samples_leaf=req.min_samples_leaf,
                learning_rate=req.learning_rate, random_state=req.random_state)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            est = GradientBoostingRegressor(
                n_estimators=req.n_estimators, max_depth=md or 3,
                min_samples_leaf=req.min_samples_leaf,
                learning_rate=req.learning_rate, random_state=req.random_state)
        model_name = "Gradient Boosting"

    result = _eval_classifier(est, X, y, req) if task == "classification" else _eval_regressor(est, X, y, req)
    result["model"] = model_name
    result["outcome"] = req.outcome
    result["predictors"] = req.predictors
    result["n_features"] = int(X.shape[1])

    try:
        store.log_action(req.session_id, kind, {
            "outcome": req.outcome, "n_predictors": len(req.predictors),
            "task": task, "n_estimators": req.n_estimators,
        })
    except Exception:
        pass
    return result


@router.post("/random_forest")
def random_forest(req: MLRequest):
    return _run(req, "random_forest")


@router.post("/gradient_boosting")
def gradient_boosting(req: MLRequest):
    return _run(req, "gradient_boosting")


@router.post("/feature_importance")
def feature_importance(req: MLRequest):
    """Permutation importance only (no curves) for a quick screen."""
    res = _run(req, "random_forest")
    return {
        "model": res["model"], "task": res["task"], "n": res["n"],
        "outcome": res["outcome"], "importance": res["importance"],
        "interpretation": res["interpretation"],
    }


# ── Penalised / kernel predictive pipeline ───────────────────────────────────
# Lasso (L1 logistic) or RBF-kernel SVM, with a held-out test set, stratified
# k-fold GridSearchCV tuning, calibration + Brier + AUC on the holdout, optional
# cubic-spline feature expansion, and partial-dependence plots.


class PredictiveRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    model: str = "lasso"                 # 'lasso' | 'svm_rbf'
    holdout_frac: float = 0.3            # stratified test fraction (default 70/30)
    cv_folds: int = 5
    spline: bool = False                 # expand numeric predictors via cubic splines
    spline_knots: int = 4
    pdp_features: Optional[List[str]] = None
    max_pdp: int = 4
    random_state: int = 42
    imputation: Optional[str] = "listwise"


def _build_design(df: pd.DataFrame, predictors: List[str], spline: bool, knots: int):
    """Encode predictors; when spline=True, expand numeric predictors into a
    natural cubic-spline (B-spline) basis. Returns a float design DataFrame."""
    raw = df[predictors].copy()
    numeric, categorical = [], []
    for c in predictors:
        if pd.api.types.is_numeric_dtype(raw[c]):
            numeric.append(c)
        else:
            coerced = pd.to_numeric(raw[c], errors="coerce")
            if coerced.notna().mean() >= 0.8 and coerced.dropna().nunique() > 2:
                raw[c] = coerced
                numeric.append(c)
            else:
                categorical.append(c)
    parts = []
    if numeric:
        num_df = raw[numeric].apply(pd.to_numeric, errors="coerce")
        if spline:
            from sklearn.preprocessing import SplineTransformer
            for c in numeric:
                col = num_df[[c]].astype(float)
                nun = int(col[c].nunique())
                if nun < 4:
                    parts.append(col.rename(columns={c: c}))
                    continue
                st = SplineTransformer(n_knots=min(max(3, knots), nun), degree=3,
                                       include_bias=False, extrapolation="constant")
                basis = st.fit_transform(col.values)
                cols = [f"{c}_sp{i+1}" for i in range(basis.shape[1])]
                parts.append(pd.DataFrame(basis, columns=cols, index=raw.index))
        else:
            parts.append(num_df)
    if categorical:
        parts.append(pd.get_dummies(raw[categorical], drop_first=True, dummy_na=False))
    if not parts:
        return pd.DataFrame(index=raw.index)
    enc = pd.concat(parts, axis=1)
    enc.columns = [str(c) for c in enc.columns]
    return enc.astype(float)


def _calibration_deciles(proba: np.ndarray, y01: np.ndarray) -> list:
    cal = []
    try:
        bins = pd.qcut(proba, q=min(10, len(np.unique(proba))), duplicates="drop")
        cdf = pd.DataFrame({"p": proba, "y": y01, "bin": bins})
        for _, g in cdf.groupby("bin", observed=True):
            cal.append({"pred": round(float(g["p"].mean()), 4),
                        "obs": round(float(g["y"].mean()), 4), "n": int(len(g))})
    except Exception:
        cal = []
    return cal


@router.post("/predictive")
async def predictive(req: PredictiveRequest):
    import asyncio
    return await asyncio.to_thread(_run_predictive, req)


def _run_predictive(req: PredictiveRequest):
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix

    if req.model not in ("lasso", "svm_rbf"):
        raise HTTPException(status_code=422, detail="model must be 'lasso' or 'svm_rbf'.")
    df = _get_df(req.session_id)
    for c in [req.outcome, *req.predictors]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.predictors:
        raise HTTPException(status_code=422, detail="Select at least one predictor.")

    cols = [req.outcome, *req.predictors]
    work = apply_imputation(df[cols], cols, req.imputation or "listwise").reset_index(drop=True)
    X = _build_design(work, req.predictors, req.spline, req.spline_knots)
    y_raw = work[req.outcome]
    keep = X.notna().all(axis=1) & y_raw.notna()
    X, y_raw = X[keep].reset_index(drop=True), y_raw[keep].reset_index(drop=True)
    if X.shape[1] == 0:
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")

    uniq = sorted(pd.unique(pd.to_numeric(y_raw, errors="coerce").dropna()).tolist()) \
        if pd.api.types.is_numeric_dtype(y_raw) else sorted(map(str, pd.unique(y_raw.dropna())))
    classes = sorted(map(str, pd.unique(y_raw.dropna())))
    if len(classes) != 2:
        raise HTTPException(status_code=422, detail=f"Predictive pipeline needs a binary outcome (got {len(classes)} levels).")
    pos = classes[-1]
    y = (y_raw.astype(str) == pos).astype(int).values
    if len(X) < 40:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows (need >= 40, got {len(X)}).")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=req.holdout_frac, stratify=y, random_state=req.random_state)

    if req.model == "lasso":
        from sklearn.linear_model import LogisticRegression
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                                    class_weight="balanced", max_iter=2000,
                                                    random_state=req.random_state))])
        grid = {"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]}
        model_label = "Lasso (L1 logistic)"
    else:
        from sklearn.svm import SVC
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", SVC(kernel="rbf", probability=True,
                                     class_weight="balanced", random_state=req.random_state))])
        grid = {"clf__C": [0.5, 1.0, 5.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]}
        model_label = ("Spline-SVM (RBF kernel)" if req.spline else "SVM (RBF kernel)")

    folds = max(2, min(req.cv_folds, int(np.bincount(y_tr).min())))
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(folds, shuffle=True, random_state=req.random_state),
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_

    proba = best.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, proba))
    brier = float(brier_score_loss(y_te, proba))
    fpr, tpr, _ = roc_curve(y_te, proba)
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    cal = _calibration_deciles(proba, y_te)
    # O/E ratio on the holdout
    oe = float(y_te.mean() / proba.mean()) if proba.mean() > 0 else None

    # Selected (non-zero) Lasso coefficients
    selected = None
    if req.model == "lasso":
        coef = best.named_steps["clf"].coef_.ravel()
        selected = [{"feature": str(c), "coef": round(float(b), 6),
                     "or": round(float(np.exp(b)), 4)}
                    for c, b in zip(X.columns, coef) if abs(b) > 1e-8]
        selected.sort(key=lambda d: abs(d["coef"]), reverse=True)

    # Partial-dependence plots
    pdp = []
    try:
        from sklearn.inspection import partial_dependence
        if req.pdp_features:
            want = [c for c in X.columns
                    if c in req.pdp_features or any(c == p or c.startswith(p + "_") or c.startswith(p + "_sp") for p in req.pdp_features)]
        else:
            # rank by training-set variance as a cheap proxy for "interesting"
            want = list(X.var().sort_values(ascending=False).index)
        want = want[:max(1, req.max_pdp)]
        for col in want:
            idx = X.columns.get_loc(col)
            pd_res = partial_dependence(best, X_tr, [idx], kind="average")
            grid_vals = pd_res["grid_values"][0] if "grid_values" in pd_res else pd_res["values"][0]
            avg = np.asarray(pd_res["average"]).ravel()
            pdp.append({
                "feature": str(col),
                "x": [round(float(v), 4) for v in grid_vals],
                "y": [round(float(v), 4) for v in avg],
            })
    except Exception:
        pdp = []

    _ = uniq  # (kept for potential numeric-class labelling)
    interp = (
        f"{model_label} trained on {len(X_tr)} cases ({int(np.sum(y_tr))} events) with "
        f"{folds}-fold GridSearchCV, evaluated on a held-out {len(X_te)} cases "
        f"({int(np.sum(y_te))} events). Holdout AUC = {auc:.3f}, Brier = {brier:.3f}, "
        f"O/E = {oe:.2f}." if oe is not None else
        f"{model_label}: holdout AUC = {auc:.3f}, Brier = {brier:.3f}."
    )

    try:
        store.log_action(req.session_id, f"ml_{req.model}", {
            "outcome": req.outcome, "n_predictors": len(req.predictors),
            "spline": req.spline, "holdout_frac": req.holdout_frac,
        })
    except Exception:
        pass

    return {
        "model": model_label,
        "task": "classification",
        "outcome": req.outcome,
        "predictors": req.predictors,
        "positive_class": pos,
        "n_total": int(len(X)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(X.shape[1]),
        "best_params": {k: _safe(v) for k, v in gs.best_params_.items()},
        "cv_best_auc": round(float(gs.best_score_), 4),
        "holdout": {
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "oe_ratio": round(oe, 4) if oe is not None else None,
            "confusion": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
            "roc_curve": _downsample_curve(fpr, tpr),
            "calibration": cal,
        },
        "selected_coefficients": selected,
        "pdp": pdp,
        "benchmark_note": "External benchmark models (e.g. EuroSCORE II) are not built in — supply their predicted risks as a column to compare.",
        "interpretation": interp,
    }
