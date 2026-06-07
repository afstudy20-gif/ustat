import io
import uuid
import tempfile
import os
import pandas as pd
import pyreadstat
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from services import store

router = APIRouter()

# Hard cap on a single uploaded dataset. Protects the in-memory store from
# being exhausted by an oversized (or hostile) file. Override via env.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))  # 100 MB

import re

# Date/time patterns for auto-detection
_DATE_PATTERNS = [
    re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$"),        # 01/02/2024, 1-2-24
    re.compile(r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}$"),           # 2024-01-02
    re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$"),                     # 01:29:00, 1:29
    re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\s+\d{1,2}:\d{2}"),  # 01/02/2024 13:45
    re.compile(r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}[T ]\d{1,2}:\d{2}"),   # 2024-01-02T13:45
]

_LEADING_ZERO_RE = re.compile(r"^0\d")  # 0123 — keep as text (likely an ID code)


def coerce_numeric_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Restore numeric dtype for object columns whose every non-empty value is
    numeric-coercible.

    JSON session round-trips serialise with ``default_handler=str`` and some
    imports (Excel/SPSS with stray cells) leave genuinely-numeric columns as
    strings. Those then misclassify as 'text' in the Data Dictionary even
    though the data is numeric. We only convert when it is lossless — every
    non-empty value parses as a number — and we skip values with a leading
    zero (e.g. '0123') that are almost certainly identifier codes, not numbers.

    Mutates a copy and returns it; the input is left untouched.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if s.dtype != object:
            continue
        as_str = s.astype(str).str.strip()
        meaningful = s.notna() & (as_str != "") & (as_str.str.lower() != "nan")
        n = int(meaningful.sum())
        if n == 0:
            continue
        # Preserve identifier-like codes with leading zeros.
        if as_str[meaningful].str.match(_LEADING_ZERO_RE).any():
            continue
        coerced = pd.to_numeric(s, errors="coerce")
        if int(coerced[meaningful].notna().sum()) == n:
            out[col] = coerced
    return out


def _detect_kind(series: pd.Series) -> str:
    """Detect column kind with date/time and binary auto-detection."""
    import datetime as _dt
    dtype = str(series.dtype)

    # Already a datetime dtype (pandas parsed it)
    if "datetime" in dtype or "timedelta" in dtype:
        return "date"

    if dtype == "bool":
        return "categorical"  # treat bool as categorical

    if dtype.startswith("int") or dtype.startswith("float"):
        # Binary detection: if only 2 unique non-null values (typically 0/1)
        # → treat as categorical (e.g. SEX, DM, EXITUS)
        unique_vals = set(series.dropna().unique())
        if len(unique_vals) <= 2:
            return "categorical"
        return "numeric"

    # Object column: check for datetime.time / datetime.date / datetime.datetime objects
    # (SPSS/SAS often store these as Python objects, not pandas datetime)
    sample_vals = series.dropna().head(20)
    if len(sample_vals) > 0:
        first_nonnull = sample_vals.iloc[0]
        if isinstance(first_nonnull, (_dt.time, _dt.date, _dt.datetime)):
            return "date"

    # For object/string columns: check if values look like dates/times.
    # Numeric-separator/ISO/time forms via regex, plus TR/EN month-name dates
    # via the date parser. Pure numbers are NOT treated as dates here so Excel
    # serial numbers / integer IDs are never mislabelled (serial parsing stays
    # opt-in through the 'Parse as date' tool).
    from services.date_parser import parse_one
    _pure_num = re.compile(r"^-?\d+(\.\d+)?$")
    sample = series.dropna().head(50).astype(str)
    if len(sample) > 0:
        def _looks_date(v: str) -> bool:
            v = v.strip()
            if _pure_num.match(v):
                return False
            return any(p.match(v) for p in _DATE_PATTERNS) or parse_one(v) is not None
        matches = sum(1 for v in sample if _looks_date(v))
        if matches / len(sample) >= 0.7:  # ≥70% match → date
            return "date"

    # String binary detection: Yes/No, True/False, M/F, etc.
    unique_str = set(series.dropna().astype(str).str.strip().str.lower().unique())
    if len(unique_str) <= 2:
        return "categorical"

    n_unique = series.nunique()
    return "categorical" if n_unique <= 50 else "text"

SUPPORTED = {
    "csv": "text/csv",
    "xlsx": "excel",
    "xls": "excel",
    "sas7bdat": "sas",
    "sav": "spss",
    "dta": "stata",
}


def _read(filename: str, content: bytes) -> pd.DataFrame:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(io.BytesIO(content))
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(content))
    elif ext in ("sas7bdat", "sav", "dta"):
        # pyreadstat requires a real file path, not BytesIO
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            if ext == "sas7bdat":
                df, _ = pyreadstat.read_sas7bdat(tmp_path)
            elif ext == "sav":
                df, _ = pyreadstat.read_sav(tmp_path)
            elif ext == "dta":
                df, _ = pyreadstat.read_dta(tmp_path)
        finally:
            os.unlink(tmp_path)
        return df
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


@router.post("/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    _max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    # Cheap pre-check on the declared size (rejects before reading the body).
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {_max_mb} MB.")
    # Hard cap on the bytes actually read — defends against a missing or spoofed
    # Content-Length. Read one byte past the limit; if we got it, it's too big.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {_max_mb} MB.")
    try:
        df = _read(file.filename, content)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")

    session_id = str(uuid.uuid4())
    store.save(session_id, df)

    columns = []
    for col in df.columns:
        kind = _detect_kind(df[col])
        columns.append({"name": col, "dtype": str(df[col].dtype), "kind": kind})

    # Use pandas to_json → loads to guarantee NaN/Inf become null
    import numpy as np, json as _json
    preview_df = df.head(2000).replace([np.inf, -np.inf], np.nan)
    preview = _json.loads(preview_df.to_json(orient="records", default_handler=str, date_format="iso", date_unit="s"))

    return {
        "session_id": session_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": columns,
        "preview": preview,
    }
