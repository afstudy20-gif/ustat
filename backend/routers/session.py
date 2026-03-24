"""Session management: cell editing and dataset export."""
import io
import json
import os
import tempfile
import numpy as np
import pandas as pd
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from services import store

router = APIRouter()


# ── Cell editing ───────────────────────────────────────────────────────────────

class CellUpdate(BaseModel):
    row_index: int
    column: str
    value: Optional[Any] = None  # string, number, or null from frontend


@router.patch("/{session_id}/cell")
async def update_cell(session_id: str, body: CellUpdate):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if body.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{body.column}' not found")
    if body.row_index < 0 or body.row_index >= len(df):
        raise HTTPException(status_code=400, detail=f"Row index {body.row_index} out of range")

    col_dtype = df[body.column].dtype
    val = body.value

    # Coerce to column dtype
    if val is not None and val != "":
        try:
            if col_dtype.kind in ("i", "u"):
                val = int(float(str(val)))
            elif col_dtype.kind == "f":
                val = float(str(val))
        except (ValueError, TypeError):
            pass  # keep as string
    else:
        val = np.nan  # blank → missing

    df.at[body.row_index, body.column] = val
    store.save(session_id, df)

    stored = df.at[body.row_index, body.column]
    try:
        if isinstance(stored, float) and (np.isnan(stored) or np.isinf(stored)):
            stored = None
    except (TypeError, ValueError):
        pass

    return {"row_index": body.row_index, "column": body.column, "value": stored}


# ── Export ─────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/export")
async def export_dataset(
    session_id: str,
    fmt: str = Query("csv", regex="^(csv|tsv|xlsx|sav)$"),
    filename: str = Query("data"),
    col_kinds: str = Query("{}"),   # JSON: {"colName": "numeric"|"categorical"|"boolean"|"text"}
):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Strip any extension the user might have included
    base = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Build Content-Disposition header safely for non-ASCII filenames (Turkish, etc.)
    from urllib.parse import quote
    ascii_base = base.encode("ascii", errors="replace").decode("ascii")  # fallback for latin-1
    utf8_base = quote(base, safe="")  # RFC 5987 percent-encoded
    def _cd(ext: str) -> dict:
        return {"Content-Disposition": f"attachment; filename=\"{ascii_base}.{ext}\"; filename*=UTF-8''{utf8_base}.{ext}"}

    if fmt == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        content = buf.getvalue().encode("utf-8-sig")  # BOM for Excel compat
        return Response(content=content, media_type="text/csv", headers=_cd("csv"))

    if fmt == "tsv":
        buf = io.StringIO()
        df.to_csv(buf, index=False, sep="\t")
        content = buf.getvalue().encode("utf-8-sig")
        return Response(content=content, media_type="text/tab-separated-values", headers=_cd("tsv"))

    if fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=_cd("xlsx"),
        )

    if fmt == "sav":
        import pyreadstat

        try:
            kinds: dict = json.loads(col_kinds)
        except Exception:
            kinds = {}

        # Build a clean copy of the dataframe suitable for pyreadstat
        df_sav = df.copy()

        variable_measure: dict = {}
        variable_value_labels: dict = {}

        for col in df_sav.columns:
            kind = kinds.get(col, "numeric")

            if kind == "date":
                variable_measure[col] = "scale"
                # Keep as string for SPSS — dates stored as text labels
            elif kind in ("categorical", "text"):
                variable_measure[col] = "nominal"
                # Numeric columns marked as categorical → add value labels so
                # SPSS knows the numbers map to categories
                if pd.api.types.is_numeric_dtype(df_sav[col]):
                    unique_vals = sorted(df_sav[col].dropna().unique())
                    variable_value_labels[col] = {float(v): str(v) for v in unique_vals}
            else:
                variable_measure[col] = "scale"
                # Ensure object columns declared numeric are cast to float
                if df_sav[col].dtype == object:
                    df_sav[col] = pd.to_numeric(df_sav[col], errors="coerce")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sav")
        os.close(tmp_fd)
        try:
            pyreadstat.write_sav(
                df_sav,
                tmp_path,
                variable_measure=variable_measure,
                variable_value_labels=variable_value_labels if variable_value_labels else None,
            )
            with open(tmp_path, "rb") as f:
                content = f.read()
        finally:
            os.unlink(tmp_path)

        return Response(content=content, media_type="application/octet-stream", headers=_cd("sav"))


# ── Select Cases ────────────────────────────────────────────────────────────────

class SelectCasesRequest(BaseModel):
    conditions: list  # [{column, operator, value, join}]


@router.post("/{session_id}/select_cases")
def select_cases(session_id: str, body: SelectCasesRequest):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    store.save_filter(session_id, body.conditions)
    from services.store import _apply_conditions
    df_filtered = _apply_conditions(df, body.conditions)
    return {"selected": len(df_filtered), "total": len(df)}


@router.delete("/{session_id}/select_cases")
def clear_cases(session_id: str):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    store.clear_filter(session_id)
    return {"selected": len(df), "total": len(df)}


# ── File Export ─────────────────────────────────────────────────────────────

@router.get("/{session_id}/export/csv")
def export_csv(session_id: str, filename: str = Query("export.csv")):
    """Export session data as CSV file."""
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Stream CSV directly instead of loading into memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/{session_id}/export/xlsx")
def export_xlsx(session_id: str, filename: str = Query("export.xlsx")):
    """Export session data as XLSX file."""
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        import openpyxl
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError:
        raise HTTPException(status_code=400, detail="XLSX export requires openpyxl")

    # Write to bytes buffer
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
    excel_buffer.seek(0)

    return StreamingResponse(
        iter([excel_buffer.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
