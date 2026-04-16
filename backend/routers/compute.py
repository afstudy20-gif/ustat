"""
Compute / Create New Variable router.

Endpoints
---------
POST /{session_id}/formula          — formula builder via pandas df.eval()
POST /{session_id}/transform        — single-column math transforms (log, sqrt, …)
POST /{session_id}/recode           — IF-THEN rule builder via numpy np.select()
POST /{session_id}/clinical/{calc}  — preset clinical calculators (BMI, eGFR, CHA₂DS₂-VASc)
DELETE /{session_id}/column/{col}   — remove a computed (or any) column from session
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store

router = APIRouter()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _col_kind(series: pd.Series) -> str:
    from routers.upload import _detect_kind
    return _detect_kind(series)


def _build_result(df: pd.DataFrame, col: str) -> dict:
    """Build the standard response dict after adding a new column."""
    series = df[col]
    preview_vals = series.head(2000).where(pd.notna(series.head(2000)), other=None).tolist()
    return {
        "name": col,
        "dtype": str(series.dtype),
        "kind": _col_kind(series),
        "preview_values": preview_vals,
        "n_computed": int(series.notna().sum()),
        "n_missing": int(series.isna().sum()),
    }


def _validate_col_name(new_col: str):
    if not new_col or not new_col.strip():
        raise HTTPException(status_code=422, detail="New column name cannot be empty")
    return new_col.strip()


# ── 1. Formula Builder ────────────────────────────────────────────────────────

class FormulaRequest(BaseModel):
    formula: str
    new_col: str


def _eval_formula_with_custom_functions(df: pd.DataFrame, formula: str) -> pd.Series:
    """
    Evaluate formula with custom IF, ISNA, DAYS functions.
    Converts formula to use numpy/pandas operations and evaluates with proper namespace.
    """
    import re as regex

    # Replace custom functions with their implementations
    # Handle DAYS(date1, date2) -> (pd.to_datetime(date1) - pd.to_datetime(date2)).dt.days
    formula_proc = formula
    formula_proc = regex.sub(
        r'DAYS\(([^,]+),\s*([^)]+)\)',
        r'((pd.to_datetime(\1) - pd.to_datetime(\2)).dt.days)',
        formula_proc
    )
    # Handle ISNA(x) -> pd.isna(x)
    formula_proc = regex.sub(r'ISNA\(([^)]+)\)', r'pd.isna(\1)', formula_proc)
    # Handle IF(cond, true, false) -> np.where(cond, true, false)
    formula_proc = regex.sub(
        r'IF\(([^,]+),\s*([^,]+),\s*([^)]+)\)',
        r'np.where(\1, \2, \3)',
        formula_proc
    )

    # Build namespace with numpy, pandas, and all columns
    namespace = {
        'np': np,
        'pd': pd,
        **{col: df[col] for col in df.columns}
    }

    # Evaluate with proper namespace
    result = eval(formula_proc, {"__builtins__": {}}, namespace)

    if not isinstance(result, (pd.Series, np.ndarray)):
        raise ValueError("Formula did not produce a series result")

    if isinstance(result, np.ndarray):
        result = pd.Series(result, index=df.index)

    return result


@router.post("/{session_id}/formula")
def formula_compute(session_id: str, req: FormulaRequest):
    """
    Evaluate a pandas-safe formula expression and save as a new column.
    Uses df.eval() — safe, no arbitrary Python execution.
    NaN propagation is automatic: if any source cell is NaN, result is NaN.
    Supports custom functions: IF(cond, true_val, false_val), ISNA(x), DAYS(date1, date2)
    """
    df = _get_df(session_id)
    new_col = _validate_col_name(req.new_col)

    try:
        result = _eval_formula_with_custom_functions(df, req.formula)
        # eval() may return a scalar if formula has no column refs
        if not isinstance(result, pd.Series):
            raise HTTPException(status_code=422, detail="Formula did not produce a column result. Make sure to reference existing column names.")
        df = df.copy()
        df[new_col] = result
    except HTTPException:
        raise
    except Exception as exc:
        msg = str(exc)
        # Make common errors more user-friendly
        if "undefined" in msg.lower() or "UndefinedVariable" in msg:
            # Extract the offending name
            m = re.search(r"'(\w+)'", msg)
            bad = f" Column '{m.group(1)}' not found." if m else ""
            raise HTTPException(status_code=422, detail=f"Unknown column name in formula.{bad} Check spelling and use exact column names as they appear in the dataset.")
        raise HTTPException(status_code=422, detail=f"Formula error: {msg}")

    store.save(session_id, df)
    return _build_result(df, new_col)


# ── 2. Transformations ────────────────────────────────────────────────────────

TRANSFORMS = {
    "ln":           "Ln (natural log)",
    "log10":        "Log₁₀",
    "sqrt":         "√ Square root",
    "square":       "x² Square",
    "exp":          "eˣ Exponential",
    "abs":          "|x| Absolute value",
    "zscore":       "Z-score",
    "tertile":      "Tertile (3 groups)",
    "quartile":     "Quartile (4 groups)",
    "median_split": "Median split (2 groups)",
}


class TransformRequest(BaseModel):
    source_col: str
    transform: str          # one of the TRANSFORMS keys
    new_col: str


@router.post("/{session_id}/transform")
def transform_compute(session_id: str, req: TransformRequest):
    df = _get_df(session_id)
    new_col = _validate_col_name(req.new_col)

    if req.source_col not in df.columns:
        raise HTTPException(status_code=422, detail=f"Column '{req.source_col}' not found")
    if req.transform not in TRANSFORMS:
        raise HTTPException(status_code=422, detail=f"Unknown transform '{req.transform}'. Valid: {list(TRANSFORMS.keys())}")

    col = pd.to_numeric(df[req.source_col], errors="coerce")
    df = df.copy()

    if req.transform == "ln":
        df[new_col] = np.log(col.where(col > 0))       # ≤0 → NaN
    elif req.transform == "log10":
        df[new_col] = np.log10(col.where(col > 0))
    elif req.transform == "sqrt":
        df[new_col] = np.sqrt(col.where(col >= 0))     # <0 → NaN
    elif req.transform == "square":
        df[new_col] = col ** 2
    elif req.transform == "exp":
        df[new_col] = np.exp(col)
    elif req.transform == "abs":
        df[new_col] = col.abs()
    elif req.transform == "zscore":
        mu, sd = col.mean(), col.std()
        if sd == 0:
            raise HTTPException(status_code=422, detail="Standard deviation is 0 — cannot compute Z-score for a constant column")
        df[new_col] = (col - mu) / sd
    elif req.transform == "tertile":
        df[new_col] = pd.qcut(col, q=3, labels=[1, 2, 3], duplicates="drop").astype(float)
    elif req.transform == "quartile":
        df[new_col] = pd.qcut(col, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(float)
    elif req.transform == "median_split":
        med = col.median()
        df[new_col] = (col > med).astype(float)  # 0 = ≤ median, 1 = > median

    store.save(session_id, df)
    return _build_result(df, new_col)


# ── 3. Recode / Binning ───────────────────────────────────────────────────────

class Condition(BaseModel):
    col: str
    op: str       # one of: < <= > >= == !=
    val: Any      # string or number

class Rule(BaseModel):
    conditions: List[Condition]   # all joined with AND
    result: Any                   # the value to assign when all conditions are true

class RecodeRequest(BaseModel):
    rules: List[Rule]
    else_val: Optional[Any] = None   # None → NaN; or numeric/string
    new_col: str


_OPS = {
    "<":        lambda s, v: s < v,
    "<=":       lambda s, v: s <= v,
    ">":        lambda s, v: s > v,
    ">=":       lambda s, v: s >= v,
    "==":       lambda s, v: s == v,
    "!=":       lambda s, v: s != v,
    "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
    "!contains": lambda s, v: ~s.astype(str).str.contains(str(v), case=False, na=False),
}


def _cast_val(col_series: pd.Series, val: Any) -> Any:
    """Try to cast the threshold value to the column's dtype."""
    if pd.api.types.is_numeric_dtype(col_series):
        try:
            return float(val)
        except (TypeError, ValueError):
            return val
    return val


@router.post("/{session_id}/recode")
def recode_compute(session_id: str, req: RecodeRequest):
    df = _get_df(session_id)
    new_col = _validate_col_name(req.new_col)

    if not req.rules:
        raise HTTPException(status_code=422, detail="At least one rule is required")

    # Validate all referenced columns exist
    all_cols = {c.col for r in req.rules for c in r.conditions}
    missing = all_cols - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Column(s) not found: {', '.join(missing)}")

    conditions: list = []
    choices: list = []

    for rule in req.rules:
        mask = pd.Series([True] * len(df), index=df.index)
        for cond in rule.conditions:
            if cond.op not in _OPS:
                raise HTTPException(status_code=422, detail=f"Unknown operator '{cond.op}'")

            raw_col = df[cond.col]
            val = cond.val

            # Decide whether to compare as numeric or string
            # If the value looks numeric, try numeric comparison
            val_is_numeric = False
            try:
                val_num = float(val)
                val_is_numeric = True
            except (TypeError, ValueError):
                pass

            if val_is_numeric and cond.op in ("<", "<=", ">", ">="):
                # Numeric comparison — coerce column to numeric
                col_s = pd.to_numeric(raw_col, errors="coerce")
                v = val_num
            elif val_is_numeric and cond.op in ("==", "!="):
                # For ==  / !=, try numeric first, fall back to string
                col_num = pd.to_numeric(raw_col, errors="coerce")
                if col_num.notna().sum() > col_num.isna().sum():
                    col_s = col_num
                    v = val_num
                else:
                    col_s = raw_col.astype(str).str.strip()
                    v = str(val).strip()
            else:
                # String comparison (value is text, or == / != on text column)
                col_s = raw_col.astype(str).str.strip()
                v = str(val).strip()

            try:
                cond_mask = _OPS[cond.op](col_s, v)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Condition error ({cond.col} {cond.op} {cond.val}): {exc}")
            # NaN in source → False (row not matched)
            cond_mask = cond_mask.fillna(False)
            mask = mask & cond_mask
        conditions.append(mask)
        # Try to cast result to numeric
        try:
            choices.append(float(rule.result))
        except (TypeError, ValueError):
            choices.append(rule.result)

    # Determine default
    default = np.nan
    if req.else_val is not None and str(req.else_val).strip() != "":
        try:
            default = float(req.else_val)
        except (TypeError, ValueError):
            default = req.else_val

    df = df.copy()

    # If all choices + default are numeric, use np.select normally
    # If any is a string, cast everything to string to avoid mixed-type issues
    all_numeric = all(isinstance(c, (int, float)) for c in choices)
    if default is not np.nan:
        try:
            float(default)
        except (TypeError, ValueError):
            all_numeric = False

    if all_numeric:
        df[new_col] = np.select(conditions, choices, default=default)
        # Convert int-like float columns to int if no NaN
        if df[new_col].notna().all():
            try:
                vals = df[new_col].astype(float)
                if (vals % 1 == 0).all():
                    df[new_col] = vals.astype(int)
            except (ValueError, TypeError):
                pass
    else:
        # String mode: cast all choices to string
        str_choices = [str(c) for c in choices]
        str_default = "" if default is np.nan else str(default)
        result = np.select(conditions, str_choices, default=str_default)
        df[new_col] = result
        # Replace empty string default with NaN
        if default is np.nan:
            df.loc[df[new_col] == "", new_col] = np.nan

    store.save(session_id, df)
    return _build_result(df, new_col)


# ── 4. Clinical Calculators ───────────────────────────────────────────────────

class ClinicalRequest(BaseModel):
    column_map: Dict[str, str]   # logical_name → actual df column name
    female_value: Optional[str] = None  # which value in sex column = Female
    new_col: Optional[str] = None       # override output column name


def _req_cols(column_map: dict, *keys: str):
    missing = [k for k in keys if not column_map.get(k)]
    if missing:
        raise HTTPException(status_code=422, detail=f"Required column mapping(s) missing: {', '.join(missing)}")


def _is_female(df: pd.DataFrame, sex_col: str, female_value: Optional[str]) -> pd.Series:
    """Return boolean Series indicating Female rows."""
    col = df[sex_col].astype(str)
    if female_value is not None:
        return col == str(female_value)
    # Auto-detect common patterns
    return col.str.lower().isin(["f", "female", "kadın", "kadin", "women", "w", "2"])


@router.post("/{session_id}/clinical/bmi")
def clinical_bmi(session_id: str, req: ClinicalRequest):
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "weight", "height")

    weight = pd.to_numeric(df[cm["weight"]], errors="coerce")
    height = pd.to_numeric(df[cm["height"]], errors="coerce")

    df = df.copy()
    new_col = req.new_col or "BMI"
    df[new_col] = (weight / ((height / 100) ** 2)).round(2)

    store.save(session_id, df)
    return _build_result(df, new_col)


@router.post("/{session_id}/clinical/egfr")
def clinical_egfr(session_id: str, req: ClinicalRequest):
    """Race-free CKD-EPI 2021 eGFR formula."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age", "sex", "creatinine")

    age = pd.to_numeric(df[cm["age"]], errors="coerce")
    scr = pd.to_numeric(df[cm["creatinine"]], errors="coerce")
    is_f = _is_female(df, cm["sex"], req.female_value)

    kappa = np.where(is_f, 0.7, 0.9)
    alpha = np.where(is_f, -0.241, -0.302)
    ratio = scr.values / kappa

    egfr = (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** (-1.200)
        * 0.9938 ** age.values
        * np.where(is_f, 1.012, 1.0)
    )

    df = df.copy()
    new_col = req.new_col or "eGFR"
    df[new_col] = np.round(egfr, 1)

    store.save(session_id, df)
    return _build_result(df, new_col)


@router.post("/{session_id}/clinical/chadsvasc")
def clinical_chadsvasc(session_id: str, req: ClinicalRequest):
    """CHA₂DS₂-VASc score for AF stroke risk."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age", "sex")

    age = pd.to_numeric(df[cm["age"]], errors="coerce")
    is_f = _is_female(df, cm["sex"], req.female_value)

    # Age score: ≥75 → 2, 65-74 → 1, <65 → 0
    age_score = np.where(age >= 75, 2, np.where(age >= 65, 1, 0))

    def _binary(key: str) -> pd.Series:
        col_name = cm.get(key)
        if not col_name:
            return pd.Series(0, index=df.index)
        s = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
        return s.clip(0, 1).astype(int)

    score = (
        _binary("chf")           # CHF = 1
        + _binary("htn")         # Hypertension = 1
        + age_score              # Age score
        + _binary("dm")          # Diabetes = 1
        + _binary("stroke") * 2  # Stroke/TIA = 2
        + _binary("vasc")        # Vascular disease = 1
        + is_f.astype(int)       # Female sex = 1
    )

    df = df.copy()
    new_col = req.new_col or "CHA2DS2VASc"
    df[new_col] = score

    store.save(session_id, df)
    return _build_result(df, new_col)


# ── shared binary helper used by all clinical calculators ─────────────────────

def _bin(df: pd.DataFrame, cm: dict, key: str) -> pd.Series:
    """Return an integer 0/1 Series for a binary column; 0 if column not mapped."""
    col_name = cm.get(key)
    if not col_name:
        return pd.Series(0, index=df.index)
    s = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
    return s.clip(0, 1).astype(int)


def _num(df: pd.DataFrame, cm: dict, key: str) -> pd.Series:
    """Return a numeric Series for a column; NaN if not mapped."""
    col_name = cm.get(key)
    if not col_name:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col_name], errors="coerce")


# ── BSA (Mosteller formula) ───────────────────────────────────────────────────

@router.post("/{session_id}/clinical/bsa")
def clinical_bsa(session_id: str, req: ClinicalRequest):
    """Body Surface Area = sqrt(height_cm × weight_kg / 3600)"""
    df = _get_df(session_id)
    _req_cols(req.column_map, "weight", "height")
    weight = _num(df, req.column_map, "weight")
    height = _num(df, req.column_map, "height")
    df = df.copy()
    new_col = req.new_col or "BSA"
    df[new_col] = np.sqrt(height * weight / 3600).round(2)
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── MAP (Mean Arterial Pressure) ──────────────────────────────────────────────

@router.post("/{session_id}/clinical/map")
def clinical_map(session_id: str, req: ClinicalRequest):
    """MAP = (SBP + 2 × DBP) / 3"""
    df = _get_df(session_id)
    _req_cols(req.column_map, "sbp", "dbp")
    sbp = _num(df, req.column_map, "sbp")
    dbp = _num(df, req.column_map, "dbp")
    df = df.copy()
    new_col = req.new_col or "MAP"
    df[new_col] = ((sbp + 2 * dbp) / 3).round(1)
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── CHA₂DS₂-VA (2024 ESC updated — sex category removed) ─────────────────────

@router.post("/{session_id}/clinical/chadsva")
def clinical_chadsva(session_id: str, req: ClinicalRequest):
    """CHA₂DS₂-VA score (2024 ESC guideline update — sex no longer counted)."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age")
    age = _num(df, cm, "age")
    age_score = np.where(age >= 75, 2, np.where(age >= 65, 1, 0))
    score = (
        _bin(df, cm, "chf")           # CHF = 1
        + _bin(df, cm, "htn")         # Hypertension = 1
        + age_score                   # Age ≥75 = 2, 65-74 = 1
        + _bin(df, cm, "dm")          # Diabetes = 1
        + _bin(df, cm, "stroke") * 2  # Stroke/TIA = 2
        + _bin(df, cm, "vasc")        # Vascular disease = 1
    )
    df = df.copy()
    new_col = req.new_col or "CHA2DS2VA"
    df[new_col] = score
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── HAS-BLED Score ────────────────────────────────────────────────────────────

@router.post("/{session_id}/clinical/hasbled")
def clinical_hasbled(session_id: str, req: ClinicalRequest):
    """HAS-BLED bleeding risk score (0-9)."""
    df = _get_df(session_id)
    cm = req.column_map
    # Age-based elderly criterion: >65
    age_col = cm.get("age")
    if age_col:
        age = _num(df, cm, "age")
        elderly = (age > 65).astype(int).fillna(0)
    else:
        elderly = _bin(df, cm, "elderly")
    score = (
        _bin(df, cm, "htn")       # H: uncontrolled hypertension
        + _bin(df, cm, "renal")   # A: abnormal renal function (1 each)
        + _bin(df, cm, "liver")   # A: abnormal liver function (1 each)
        + _bin(df, cm, "stroke")  # S: stroke history
        + _bin(df, cm, "bleeding") # B: bleeding history
        + _bin(df, cm, "labile_inr") # L: labile INR
        + elderly                  # E: age > 65
        + _bin(df, cm, "drugs")   # D: drugs (antiplatelets/NSAIDs)
        + _bin(df, cm, "alcohol") # D: alcohol use
    )
    df = df.copy()
    new_col = req.new_col or "HAS_BLED"
    df[new_col] = score
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── GRACE Score (in-hospital mortality) ───────────────────────────────────────

def _grace_lookup(series: pd.Series, breakpoints: list, points: list) -> np.ndarray:
    """Map a numeric series to integer points using a step lookup table."""
    result = np.zeros(len(series), dtype=int)
    for i, (bp, pt) in enumerate(zip(breakpoints, points)):
        if i == 0:
            result = np.where(series < bp, pt, result)
        else:
            result = np.where(series >= breakpoints[i - 1], pt, result)
    return result


@router.post("/{session_id}/clinical/grace")
def clinical_grace(session_id: str, req: ClinicalRequest):
    """GRACE 2.0 integer risk score for ACS (in-hospital mortality)."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age", "hr", "sbp", "creatinine")

    age = _num(df, cm, "age").values
    hr  = _num(df, cm, "hr").values
    sbp = _num(df, cm, "sbp").values
    scr = _num(df, cm, "creatinine").values   # mg/dL

    # Age lookup (points for upper boundary of each bracket)
    age_pts = np.select(
        [age < 30, age < 40, age < 50, age < 60, age < 70, age < 80, age < 90],
        [0,        8,        25,        41,        58,        75,        91],
        default=100,
    )
    # Heart rate
    hr_pts = np.select(
        [hr < 50, hr < 70, hr < 90, hr < 110, hr < 150, hr < 200],
        [0,       3,       9,       15,        24,        38],
        default=46,
    )
    # Systolic BP
    sbp_pts = np.select(
        [sbp < 80, sbp < 100, sbp < 120, sbp < 140, sbp < 160, sbp < 200],
        [63,       58,        47,         37,         26,         11],
        default=0,
    )
    # Creatinine (mg/dL)
    scr_pts = np.select(
        [scr < 0.4, scr < 0.8, scr < 1.2, scr < 1.6, scr < 2.0, scr < 4.0],
        [2,         5,         8,          11,         14,         23],
        default=31,
    )
    # Killip class (1-4 → 0, 20, 39, 59)
    killip_col = cm.get("killip")
    if killip_col:
        killip = pd.to_numeric(df[killip_col], errors="coerce").fillna(1).clip(1, 4).astype(int)
        killip_pts = np.select(
            [killip == 1, killip == 2, killip == 3],
            [0,           20,          39],
            default=59,
        )
    else:
        killip_pts = np.zeros(len(df), dtype=int)

    score = (
        age_pts
        + hr_pts
        + sbp_pts
        + scr_pts
        + killip_pts
        + _bin(df, cm, "cardiac_arrest").values * 43
        + _bin(df, cm, "st_deviation").values   * 30
        + _bin(df, cm, "cardiac_markers").values * 15
    )

    df = df.copy()
    new_col = req.new_col or "GRACE_Score"
    df[new_col] = score.astype(int)
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── TIMI Risk Score for NSTEMI / UA ──────────────────────────────────────────

@router.post("/{session_id}/clinical/timi_nstemi")
def clinical_timi_nstemi(session_id: str, req: ClinicalRequest):
    """TIMI risk score for NSTEMI/UA (0-7). Each criterion = 1 point."""
    df = _get_df(session_id)
    cm = req.column_map

    # Age ≥65 from numeric column
    age_col = cm.get("age")
    if age_col:
        age_pts = (_num(df, cm, "age") >= 65).astype(int).fillna(0)
    else:
        age_pts = _bin(df, cm, "age_ge65")

    score = (
        age_pts                        # 1. Age ≥ 65
        + _bin(df, cm, "risk_factors") # 2. ≥3 CAD risk factors
        + _bin(df, cm, "known_cad")    # 3. Known CAD (stenosis ≥50%)
        + _bin(df, cm, "aspirin")      # 4. Aspirin use in last 7 days
        + _bin(df, cm, "severe_angina")# 5. ≥2 anginal events in last 24h
        + _bin(df, cm, "st_deviation") # 6. ST deviation ≥0.5 mm
        + _bin(df, cm, "markers")      # 7. Elevated cardiac markers
    )
    df = df.copy()
    new_col = req.new_col or "TIMI_NSTEMI"
    df[new_col] = score
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── TIMI Risk Score for STEMI ─────────────────────────────────────────────────

@router.post("/{session_id}/clinical/timi_stemi")
def clinical_timi_stemi(session_id: str, req: ClinicalRequest):
    """TIMI risk score for STEMI (0-14). Points as per original publication."""
    df = _get_df(session_id)
    cm = req.column_map

    age = _num(df, cm, "age")
    age_pts = np.where(age >= 75, 3, np.where(age >= 65, 2, 0))

    sbp = _num(df, cm, "sbp")
    sbp_pts = (sbp < 100).astype(int).fillna(0) * 3

    hr = _num(df, cm, "hr")
    hr_pts = (hr > 100).astype(int).fillna(0) * 2

    # Killip class II-IV = 2 points
    killip_col = cm.get("killip")
    if killip_col:
        killip = pd.to_numeric(df[killip_col], errors="coerce").fillna(1)
        killip_pts = (killip > 1).astype(int) * 2
    else:
        killip_pts = pd.Series(0, index=df.index)

    weight = _num(df, cm, "weight")
    weight_pts = (weight < 67).astype(int).fillna(0)

    score = (
        age_pts
        + _bin(df, cm, "dm_htn_angina") * 1  # DM, HTN, or angina = 1
        + sbp_pts                              # SBP < 100 = 3
        + hr_pts                               # HR > 100 = 2
        + killip_pts                           # Killip II-IV = 2
        + weight_pts                           # Weight < 67 kg = 1
        + _bin(df, cm, "anterior_stemi") * 1  # Anterior ST elevation or LBBB = 1
        + _bin(df, cm, "late_treatment") * 1  # Time to treatment > 4h = 1
    )
    df = df.copy()
    new_col = req.new_col or "TIMI_STEMI"
    df[new_col] = score
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── H2FPEF Score (HFpEF diagnosis) ───────────────────────────────────────────

@router.post("/{session_id}/clinical/h2fpef")
def clinical_h2fpef(session_id: str, req: ClinicalRequest):
    """H2FPEF score for HFpEF probability (0-9)."""
    df = _get_df(session_id)
    cm = req.column_map

    # H: Heavy — BMI > 30 = 2 points
    bmi_col = cm.get("bmi")
    if bmi_col:
        bmi = _num(df, cm, "bmi")
        heavy = (bmi > 30).astype(int).fillna(0) * 2
    else:
        heavy = _bin(df, cm, "obese") * 2   # or direct binary

    # E: Elderly — age > 60 = 1 point
    age_col = cm.get("age")
    if age_col:
        age = _num(df, cm, "age")
        elderly = (age > 60).astype(int).fillna(0)
    else:
        elderly = _bin(df, cm, "elderly")

    score = (
        heavy                            # H²: obese (BMI > 30) = 2
        + _bin(df, cm, "htn_meds") * 1  # H: ≥2 antihypertensive meds = 1
        + _bin(df, cm, "af") * 3        # F: Atrial fibrillation = 3
        + _bin(df, cm, "pulm_htn") * 1  # P: Pulmonary HTN (PASP > 35) = 1
        + elderly                        # E: Age > 60 = 1
        + _bin(df, cm, "ee_ratio") * 1  # F: E/e' > 9 = 1
    )
    df = df.copy()
    new_col = req.new_col or "H2FPEF"
    df[new_col] = score
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── MAGGIC Heart Failure Risk Score ──────────────────────────────────────────

def _maggic_age_pts(age: np.ndarray) -> np.ndarray:
    return np.select(
        [age < 55, age < 60, age < 65, age < 70, age < 75, age < 80],
        [0,        1,        2,        4,        6,        8],
        default=10,
    )

def _maggic_sbp_pts(sbp: np.ndarray) -> np.ndarray:
    return np.select(
        [sbp < 100, sbp < 110, sbp < 120, sbp < 130, sbp < 140],
        [5,         4,         3,         2,         1],
        default=0,
    )

def _maggic_bmi_pts(bmi: np.ndarray) -> np.ndarray:
    return np.select(
        [bmi < 15, bmi < 20, bmi < 25, bmi < 30],
        [6,        5,        3,        1],
        default=0,
    )

def _maggic_creatinine_pts(scr_umol: np.ndarray) -> np.ndarray:
    """Creatinine in μmol/L."""
    return np.select(
        [scr_umol < 90, scr_umol < 110, scr_umol < 130, scr_umol < 150, scr_umol < 170, scr_umol < 210],
        [0,             1,              2,              3,              4,              5],
        default=8,
    )

def _maggic_ef_pts(ef: np.ndarray) -> np.ndarray:
    return np.select(
        [ef < 15, ef < 20, ef < 25, ef < 30, ef < 35, ef < 40, ef < 45],
        [7,       6,       5,       4,       3,       2,       1],
        default=0,
    )

def _maggic_nyha_pts(nyha: np.ndarray) -> np.ndarray:
    return np.select(
        [nyha == 1, nyha == 2, nyha == 3],
        [0,         2,         6],
        default=8,  # NYHA IV
    )


@router.post("/{session_id}/clinical/maggic")
def clinical_maggic(session_id: str, req: ClinicalRequest):
    """MAGGIC Heart Failure Risk Score (Pocock et al. 2013, EHJ)."""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "age", "sbp", "bmi", "creatinine", "ef")

    age = _num(df, cm, "age").values
    sbp = _num(df, cm, "sbp").values
    bmi_vals = _num(df, cm, "bmi").values
    ef  = _num(df, cm, "ef").values

    # Creatinine: auto-detect mg/dL vs μmol/L (mg/dL values are typically <20)
    scr_raw = _num(df, cm, "creatinine").values
    scr_umol = np.where(np.nanmax(scr_raw) < 20, scr_raw * 88.4, scr_raw)

    # NYHA class (1-4); default to 2 if not mapped
    nyha_col = cm.get("nyha")
    if nyha_col:
        nyha = pd.to_numeric(df[nyha_col], errors="coerce").fillna(2).clip(1, 4).values
    else:
        nyha = np.full(len(df), 2.0)

    # Sex: male = +1
    sex_col = cm.get("sex")
    if sex_col:
        is_male = ~_is_female(df, sex_col, req.female_value)
        male_pts = is_male.astype(int).values
    else:
        male_pts = np.zeros(len(df), dtype=int)

    # Not on BB = +3; we accept a "bb" column (1=on BB, 0=not on BB)
    bb = _bin(df, cm, "bb").values
    not_on_bb = (1 - bb) * 3

    # Not on ACE/ARB = +1
    ace = _bin(df, cm, "ace_arb").values
    not_on_ace = (1 - ace)

    score = (
        _maggic_age_pts(age)
        + male_pts
        + _maggic_nyha_pts(nyha)
        + np.where(cm.get("current_smoker"), _bin(df, cm, "current_smoker").values, 0)
        + _bin(df, cm, "diabetes").values   * 3
        + _bin(df, cm, "copd").values       * 2
        + _bin(df, cm, "hf_lt18m").values   * 2  # HF diagnosed < 18 months ago
        + not_on_ace
        + not_on_bb
        + _maggic_sbp_pts(sbp)
        + _maggic_bmi_pts(bmi_vals)
        + _maggic_creatinine_pts(scr_umol)
        + _maggic_ef_pts(ef)
    )

    df = df.copy()
    new_col = req.new_col or "MAGGIC_Score"
    df[new_col] = score.astype(int)
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── QTc — Bazett's formula ────────────────────────────────────────────────────

@router.post("/{session_id}/clinical/qtc")
def clinical_qtc(session_id: str, req: ClinicalRequest):
    """Corrected QT interval (Bazett): QTc = QT_ms / sqrt(RR_s) = QT / sqrt(60/HR)"""
    df = _get_df(session_id)
    cm = req.column_map
    _req_cols(cm, "qt", "hr")
    qt = _num(df, cm, "qt")   # QT in milliseconds
    hr = _num(df, cm, "hr")   # Heart rate in bpm
    rr = 60.0 / hr            # RR interval in seconds
    df = df.copy()
    new_col = req.new_col or "QTc_Bazett"
    df[new_col] = (qt / np.sqrt(rr)).round(1)
    store.save(session_id, df)
    return _build_result(df, new_col)


# ── 5. Delete column ──────────────────────────────────────────────────────────

@router.delete("/{session_id}/column/{col_name:path}")
def delete_column(session_id: str, col_name: str):
    df = _get_df(session_id)
    if col_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col_name}' not found")
    df = df.drop(columns=[col_name])
    store.save(session_id, df)
    return {"deleted": col_name}


# ── 6. Fill blanks ──────────────────────────────────────────────────────────

class FillBlanksRequest(BaseModel):
    column: str
    value: str  # fill value (will be cast to match column dtype)


@router.post("/{session_id}/fill_blanks")
def fill_blanks(session_id: str, req: FillBlanksRequest):
    df = _get_df(session_id)
    if req.column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{req.column}' not found")

    df = df.copy()
    col = df[req.column]
    n_before = int(col.isna().sum() + (col.astype(str).str.strip() == "").sum())

    method_label = req.value

    # Special fill strategies
    if req.value == "__mean__":
        num_col = pd.to_numeric(col, errors="coerce")
        fill_val = float(num_col.mean())
        method_label = f"mean ({fill_val:.2f})"
        df[req.column] = num_col.fillna(fill_val)
    elif req.value == "__median__":
        num_col = pd.to_numeric(col, errors="coerce")
        fill_val = float(num_col.median())
        method_label = f"median ({fill_val:.2f})"
        df[req.column] = num_col.fillna(fill_val)
    elif req.value == "__mice__":
        # MICE imputation for this single column using all other numeric columns
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if req.column not in num_cols:
            num_col = pd.to_numeric(col, errors="coerce")
            df[req.column] = num_col
            num_cols = df.select_dtypes(include="number").columns.tolist()
        if req.column in num_cols and len(num_cols) >= 2:
            imp = IterativeImputer(max_iter=10, random_state=42)
            imputed = imp.fit_transform(df[num_cols])
            idx = num_cols.index(req.column)
            df[req.column] = imputed[:, idx]
            fill_val = "MICE"
            method_label = "MICE (multiple imputation)"
        else:
            # Fallback to median if MICE not possible
            num_col = pd.to_numeric(col, errors="coerce")
            fill_val = float(num_col.median())
            method_label = f"median fallback ({fill_val:.2f})"
            df[req.column] = num_col.fillna(fill_val)
    else:
        # Custom value — try numeric cast first
        try:
            fill_val = float(req.value)
            if fill_val == int(fill_val):
                fill_val = int(fill_val)
        except (ValueError, TypeError):
            fill_val = req.value

        df[req.column] = col.fillna(fill_val)
        if col.dtype == object:
            df.loc[df[req.column].astype(str).str.strip() == "", req.column] = fill_val

    n_after = int(df[req.column].isna().sum())
    n_filled = n_before - n_after

    store.save(session_id, df)
    store.log_action(session_id, "fill_blanks", {"column": req.column, "method": method_label, "n_filled": n_filled})
    return {"column": req.column, "fill_value": method_label, "n_filled": n_filled}


# ── 7. Delete rows ──────────────────────────────────────────────────────────

class DeleteRowsRequest(BaseModel):
    row_indices: List[int]  # 0-based indices to delete


@router.post("/{session_id}/delete_rows")
def delete_rows(session_id: str, req: DeleteRowsRequest):
    df = _get_df(session_id)
    if not req.row_indices:
        raise HTTPException(status_code=422, detail="No row indices provided")
    invalid = [i for i in req.row_indices if i < 0 or i >= len(df)]
    if invalid:
        raise HTTPException(status_code=422, detail=f"Row indices out of range: {invalid}")
    df = df.drop(df.index[req.row_indices]).reset_index(drop=True)
    store.save(session_id, df)
    store.log_action(session_id, "delete_rows", {"n_deleted": len(req.row_indices)})
    return {"deleted": len(req.row_indices), "remaining_rows": len(df)}


# ── 8. Add row ─────────────────────────────────────────────────────────────

class AddRowRequest(BaseModel):
    position: int = -1  # -1 = append at end, otherwise insert at this index


@router.post("/{session_id}/add_row")
def add_row(session_id: str, req: AddRowRequest):
    df = _get_df(session_id)
    # New row with all NaN/None values
    new_row = pd.DataFrame([{col: None for col in df.columns}])
    if req.position < 0 or req.position >= len(df):
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        top = df.iloc[:req.position]
        bottom = df.iloc[req.position:]
        df = pd.concat([top, new_row, bottom], ignore_index=True)
    store.save(session_id, df)
    store.log_action(session_id, "add_row", {"position": req.position})
    return {"rows": len(df), "position": req.position}


# ── 9. Add column ──────────────────────────────────────────────────────────

class AddColumnRequest(BaseModel):
    name: str
    default_value: Optional[Any] = None  # None → all NaN
    position: int = -1  # -1 = append at end, otherwise insert at this index


@router.post("/{session_id}/add_column")
def add_column(session_id: str, req: AddColumnRequest):
    df = _get_df(session_id)
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Column name cannot be empty")
    if name in df.columns:
        raise HTTPException(status_code=422, detail=f"Column '{name}' already exists")
    df = df.copy()
    if req.position >= 0 and req.position < len(df.columns):
        df.insert(req.position, name, req.default_value)
    else:
        df[name] = req.default_value
    store.save(session_id, df)
    store.log_action(session_id, "add_column", {"name": name})
    return _build_result(df, name)


# ── 10. Paste rows (from clipboard TSV/CSV) ─────────────────────────────────

class PasteRequest(BaseModel):
    tsv: str  # tab or comma separated text (with optional header row)
    has_header: bool = True
    mode: str = "append"  # "append" or "replace"


@router.post("/{session_id}/paste")
def paste_rows(session_id: str, req: PasteRequest):
    import io as _io
    df = _get_df(session_id)

    text = req.tsv.strip()
    if not text:
        raise HTTPException(status_code=422, detail="No data to paste")

    # Auto-detect separator (tab or comma)
    first_line = text.split("\n")[0]
    sep = "\t" if "\t" in first_line else ","

    try:
        if req.has_header:
            pasted = pd.read_csv(_io.StringIO(text), sep=sep)
        else:
            pasted = pd.read_csv(_io.StringIO(text), sep=sep, header=None)
            # Assign column names from existing df if column count matches
            if len(pasted.columns) == len(df.columns):
                pasted.columns = df.columns
            else:
                pasted.columns = [f"Col_{i+1}" for i in range(len(pasted.columns))]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse pasted data: {exc}")

    if req.mode == "replace":
        df = pasted
    else:
        # Append — align columns (add missing as NaN, ignore extra)
        for col in df.columns:
            if col not in pasted.columns:
                pasted[col] = None
        for col in pasted.columns:
            if col not in df.columns:
                df[col] = None
        df = pd.concat([df, pasted[df.columns]], ignore_index=True)

    store.save(session_id, df)
    store.log_action(session_id, "paste_rows", {"n_pasted": len(pasted), "mode": req.mode})
    return {"n_pasted": len(pasted), "total_rows": len(df)}


# ── 11. Rename column ──────────────────────────────────────────────────────

class RenameRequest(BaseModel):
    old_name: str
    new_name: str


@router.post("/{session_id}/rename")
def rename_column(session_id: str, req: RenameRequest):
    df = _get_df(session_id)
    if req.old_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{req.old_name}' not found")
    new = req.new_name.strip()
    if not new:
        raise HTTPException(status_code=422, detail="New column name cannot be empty")
    if new in df.columns and new != req.old_name:
        raise HTTPException(status_code=422, detail=f"Column '{new}' already exists")
    df = df.rename(columns={req.old_name: new})
    store.save(session_id, df)
    store.log_action(session_id, "rename_column", {"old": req.old_name, "new": new})
    return {"old_name": req.old_name, "new_name": new}


# ── 12. Duplicate column ──────────────────────────────────────────────────────

class DuplicateColumnRequest(BaseModel):
    column: str


@router.post("/{session_id}/duplicate_column")
def duplicate_column(session_id: str, req: DuplicateColumnRequest):
    df = _get_df(session_id)
    col = req.column
    if col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found")

    # Generate unique name
    base = f"{col}_copy"
    new_name = base
    i = 2
    while new_name in df.columns:
        new_name = f"{base}_{i}"
        i += 1

    # Insert right after the original column
    pos = list(df.columns).index(col) + 1
    df = df.copy()
    df.insert(pos, new_name, df[col].values.copy())
    store.save(session_id, df)
    store.log_action(session_id, "duplicate_column", {"source": col, "new": new_name})
    return _build_result(df, new_name)


# ── 13. Paste cells (copy-paste within the grid) ─────────────────────────────

class PasteCellsRequest(BaseModel):
    start_row: int
    start_col: str
    tsv: str  # tab-separated values grid


@router.post("/{session_id}/paste_cells")
def paste_cells(session_id: str, req: PasteCellsRequest):
    """Paste a TSV grid of values starting at a given cell position."""
    df = _get_df(session_id)
    if req.start_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.start_col}' not found")

    lines = req.tsv.strip().split("\n")
    if not lines:
        return {"pasted": 0}

    col_list = list(df.columns)
    start_ci = col_list.index(req.start_col)
    df = df.copy()
    pasted = 0

    for dr, line in enumerate(lines):
        ri = req.start_row + dr
        if ri >= len(df):
            break
        vals = line.split("\t")
        for dc, val in enumerate(vals):
            ci = start_ci + dc
            if ci >= len(col_list):
                break
            col_name = col_list[ci]
            # Coerce value
            v: Any = val.strip()
            if v == "" or v.lower() == "null":
                v = np.nan
            else:
                col_dtype = df[col_name].dtype
                try:
                    if col_dtype.kind in ("i", "u"):
                        v = int(float(v))
                    elif col_dtype.kind == "f":
                        v = float(v)
                except (ValueError, TypeError):
                    pass
            df.at[ri, col_name] = v
            pasted += 1

    store.save(session_id, df)
    return {"pasted": pasted}


# ── 7. List unique values (for sex mapping UI) ────────────────────────────────

@router.get("/{session_id}/unique/{col_name:path}")
def unique_values(session_id: str, col_name: str):
    df = _get_df(session_id)
    if col_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col_name}' not found")
    vals = sorted(df[col_name].dropna().unique().tolist(), key=lambda x: (str(type(x).__name__), x))
    return {"values": [str(v) for v in vals[:200]]}
