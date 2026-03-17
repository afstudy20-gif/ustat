import io
import uuid
import tempfile
import os
import pandas as pd
import pyreadstat
from fastapi import APIRouter, UploadFile, File, HTTPException
from services import store

router = APIRouter()

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
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
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
        dtype = str(df[col].dtype)
        if dtype.startswith("int") or dtype.startswith("float"):
            kind = "numeric"
        elif dtype == "bool":
            kind = "boolean"
        else:
            n_unique = df[col].nunique()
            kind = "categorical" if n_unique <= 50 else "text"
        columns.append({"name": col, "dtype": dtype, "kind": kind})

    # Use pandas to_json → loads to guarantee NaN/Inf become null
    import numpy as np, json as _json
    preview_df = df.head(2000).replace([np.inf, -np.inf], np.nan)
    preview = _json.loads(preview_df.to_json(orient="records", default_handler=str))

    return {
        "session_id": session_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": columns,
        "preview": preview,
    }
