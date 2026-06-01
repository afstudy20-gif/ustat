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

    # For object/string columns: check if values look like dates/times
    sample = series.dropna().head(50).astype(str)
    if len(sample) > 0:
        matches = sum(1 for v in sample if any(p.match(v.strip()) for p in _DATE_PATTERNS))
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
