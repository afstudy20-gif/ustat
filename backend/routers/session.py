"""Session management: cell editing, dataset export, session save/load, audit."""
import io
import json
import os
import tempfile
import time
import uuid
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from services import store

router = APIRouter()


# ── Cell editing ───────────────────────────────────────────────────────────────

class CellUpdate(BaseModel):
    row_index: int
    column: str
    value: Optional[Any] = None  # string, number, or null from frontend


class ClearCellsRequest(BaseModel):
    cells: list  # [{row_index: int, column: str}, ...]


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

    df = df.copy()
    df.at[body.row_index, body.column] = val
    store.save(session_id, df)

    stored = df.at[body.row_index, body.column]
    try:
        if isinstance(stored, float) and (np.isnan(stored) or np.isinf(stored)):
            stored = None
    except (TypeError, ValueError):
        pass

    return {"row_index": body.row_index, "column": body.column, "value": stored}


@router.post("/{session_id}/clear_cells")
async def clear_cells(session_id: str, body: ClearCellsRequest):
    """Clear (set to NaN) multiple cells at once."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    df = df.copy()
    cleared = 0
    for cell in body.cells:
        r = cell.get("row_index") if isinstance(cell, dict) else None
        c = cell.get("column") if isinstance(cell, dict) else None
        if r is None or c is None:
            continue
        if c not in df.columns or r < 0 or r >= len(df):
            continue
        df.at[r, c] = np.nan
        cleared += 1

    store.save(session_id, df)
    return {"cleared": cleared}


@router.delete("/{session_id}/row/{row_index}")
async def delete_row(session_id: str, row_index: int):
    """Delete a specific row containing an outlier."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # row_index from frontend is 1-based, we need 0-based for pandas index
    # (or simply exactly matching the internal pd.Index we logged)
    target_idx = row_index - 1
    
    if target_idx < 0:
        raise HTTPException(status_code=400, detail="Invalid row index")

    success = store.delete_row(session_id, target_idx)
    if not success:
        raise HTTPException(status_code=400, detail="Row could not be deleted")
        
    return _session_preview(store.get(session_id), session_id)


class ReorderColumnsRequest(BaseModel):
    columns: list  # ordered list of column names


@router.post("/{session_id}/reorder_columns")
async def reorder_columns(session_id: str, body: ReorderColumnsRequest):
    """Reorder DataFrame columns to match frontend drag-and-drop order."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    new_order = [c for c in body.columns if c in df.columns]
    # Append any columns that weren't in the request (safety)
    for c in df.columns:
        if c not in new_order:
            new_order.append(c)

    df = df[new_order]
    store.save(session_id, df)
    return {"columns": list(df.columns)}


# ── Export ─────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/export")
async def export_dataset(
    session_id: str,
    fmt: str = Query("csv", pattern="^(csv|tsv|xlsx|sav)$"),
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
        col_metadata = store.get_metadata(session_id)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")

            # Build Value Labels sheet if any column has user-defined labels
            vl_rows = []
            for col in df.columns:
                user_labels = (col_metadata.get(col, {}) or {}).get("value_labels", {})
                if user_labels:
                    for val, label in sorted(user_labels.items(), key=lambda x: str(x[0])):
                        if label:  # skip empty labels
                            vl_rows.append({"Column": col, "Value": val, "Label": label})
            if vl_rows:
                vl_df = pd.DataFrame(vl_rows)
                vl_df.to_excel(writer, index=False, sheet_name="Value Labels")

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

        # Load user-defined value labels from session metadata
        col_metadata = store.get_metadata(session_id)

        # Build a clean copy of the dataframe suitable for pyreadstat
        df_sav = df.copy()

        variable_measure: dict = {}
        variable_value_labels: dict = {}

        for col in df_sav.columns:
            kind = kinds.get(col, "numeric")

            # Check for user-defined value labels (from Data Dictionary / Value Labels modal)
            user_labels = (col_metadata.get(col, {}) or {}).get("value_labels", {})

            if kind == "date":
                variable_measure[col] = "scale"
            elif kind in ("categorical", "text"):
                variable_measure[col] = "nominal"
                if pd.api.types.is_numeric_dtype(df_sav[col]):
                    if user_labels:
                        # Use custom value labels: convert keys to float for SPSS
                        variable_value_labels[col] = {float(k): str(v) for k, v in user_labels.items() if v}
                    else:
                        unique_vals = sorted(df_sav[col].dropna().unique())
                        variable_value_labels[col] = {float(v): str(v) for v in unique_vals}
            else:
                variable_measure[col] = "scale"
                # Even numeric/scale columns can have value labels
                if user_labels and pd.api.types.is_numeric_dtype(df_sav[col]):
                    variable_value_labels[col] = {float(k): str(v) for k, v in user_labels.items() if v}
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


# ── Session Save/Load ─────────────────────────────────────────────────────────

@router.get("/{session_id}/save_session")
async def save_session(session_id: str):
    """Export the full session as a downloadable JSON file."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build columns metadata (same shape as upload response). User-driven kind
    # overrides win over auto-detection so the dictionary classification
    # survives the save/load round-trip.
    from routers.upload import _detect_kind
    kind_overrides = store.get_kind_overrides(session_id)
    columns = []
    for col in df.columns:
        kind = kind_overrides.get(col) or _detect_kind(df[col])
        columns.append({"name": col, "dtype": str(df[col].dtype), "kind": kind})

    # User-chosen display name (set via /rename) wins over the hardcoded
    # fallback so the saved JSON round-trips the rename.
    user_filename = store.get_filename(session_id) or f"session_{session_id[:8]}.json"
    payload = {
        "version": "1.2",
        "filename": user_filename,
        "created": time.time(),
        "columns": columns,
        "col_metadata": store.get_metadata(session_id),
        "kind_overrides": kind_overrides,
        "decimals_overrides": store.get_decimals(session_id),
        "case_filter": store.get_filter(session_id),
        "audit": store.get_audit(session_id),
        "data": json.loads(
            df.replace([np.inf, -np.inf], np.nan).to_json(
                orient="records", date_format="iso", default_handler=str
            )
        ),
    }

    content = json.dumps(payload, allow_nan=False, default=str).encode("utf-8")
    safe_name = f"session_{session_id[:8]}.json"

    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@router.post("/load_session")
async def load_session(file: UploadFile = File(...)):
    """Restore a session from a previously saved JSON file."""
    raw = await file.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    if "data" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'data' key in session file")

    df = pd.DataFrame(payload["data"])
    new_session_id = str(uuid.uuid4())
    store.save(new_session_id, df)

    # Restore filters if present
    case_filter = payload.get("case_filter", [])
    if case_filter:
        store.save_filter(new_session_id, case_filter)

    # Restore column metadata if present
    col_metadata = payload.get("col_metadata", {})
    if col_metadata:
        store.save_metadata(new_session_id, col_metadata)

    # Restore user-driven kind overrides (v1.1+ session files). For older v1.0
    # files we fall back to the "columns" array on the payload, which already
    # carries the kind the user saw at save time.
    kind_overrides = payload.get("kind_overrides")
    if not kind_overrides and isinstance(payload.get("columns"), list):
        kind_overrides = {c["name"]: c["kind"] for c in payload["columns"] if c.get("name") and c.get("kind")}
    if kind_overrides:
        store.set_kind_overrides(new_session_id, kind_overrides)

    # Restore per-column decimal-places overrides (v1.2+ session files).
    decimals_overrides = payload.get("decimals_overrides") or {}
    if decimals_overrides:
        store.save_decimals(new_session_id, decimals_overrides)

    # Restore column metadata (labels, units, value_labels set at recode
    # time) so the Data Dictionary + legends repopulate after reload.
    col_metadata = payload.get("col_metadata") or {}
    if col_metadata:
        store.save_metadata(new_session_id, col_metadata)

    # Restore user-chosen display name so subsequent save_session calls
    # keep round-tripping the rename.
    restored_filename = payload.get("filename")
    if restored_filename:
        store.set_filename(new_session_id, restored_filename)

    # Build columns info — overrides win over auto-detection.
    from routers.upload import _detect_kind
    overrides = store.get_kind_overrides(new_session_id)
    columns = []
    for col in df.columns:
        kind = overrides.get(col) or _detect_kind(df[col])
        columns.append({"name": col, "dtype": str(df[col].dtype), "kind": kind})
    _attach_value_labels(columns, new_session_id)

    preview = json.loads(
        df.head(2000).replace([np.inf, -np.inf], np.nan).to_json(
            orient="records", default_handler=str, date_format="iso", date_unit="s"
        )
    )

    return {
        "session_id": new_session_id,
        "filename": payload.get("filename", file.filename),
        "rows": len(df),
        "columns": columns,
        "preview": preview,
    }


# ── Audit ─────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/audit")
async def get_audit(session_id: str):
    """Return the audit trail for a session."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return store.get_audit(session_id)


# ── Undo / Redo ──────────────────────────────────────────────────────────────

def _attach_value_labels(columns: list, session_id: str) -> list:
    """Merge persisted per-column value labels (set at recode time via the
    metadata endpoint) into the column objects, so the frontend Data
    Dictionary and legends see them after a refresh / reload. Without this
    the labels live only in the separate _metadata map and column.value_labels
    arrives empty."""
    meta = store.get_metadata(session_id) or {}
    for c in columns:
        vl = (meta.get(c.get("name"), {}) or {}).get("value_labels")
        if vl:
            c["value_labels"] = vl
    return columns


def _session_preview(df: pd.DataFrame, session_id: str | None = None) -> dict:
    """Build a session-like response from a DataFrame for frontend state update."""
    import json as _json
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "datetime" in dtype or "timedelta" in dtype:
            kind = "date"
        elif dtype.startswith("int") or dtype.startswith("float"):
            unique_vals = set(df[col].dropna().unique())
            kind = "categorical" if len(unique_vals) <= 2 else "numeric"
        elif dtype == "bool":
            kind = "categorical"
        else:
            kind = "categorical" if df[col].nunique() <= 50 else "text"
        columns.append({"name": col, "dtype": dtype, "kind": kind})
    if session_id:
        _attach_value_labels(columns, session_id)
    preview_df = df.head(2000).replace([np.inf, -np.inf], np.nan)
    preview = _json.loads(preview_df.to_json(orient="records", default_handler=str, date_format="iso", date_unit="s"))
    return {"rows": len(df), "columns": columns, "preview": preview}


@router.post("/{session_id}/undo")
async def undo_action(session_id: str):
    """Undo the last data mutation (backend DataFrame + return refreshed preview)."""
    restored = store.undo(session_id)
    if restored is None:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    store.log_action(session_id, "undo")
    result = _session_preview(restored, session_id)
    result["undo_depth"] = store.undo_depth(session_id)
    result["redo_depth"] = store.redo_depth(session_id)
    return result


@router.post("/{session_id}/redo")
async def redo_action(session_id: str):
    """Redo the last undone mutation."""
    restored = store.redo(session_id)
    if restored is None:
        raise HTTPException(status_code=400, detail="Nothing to redo")
    store.log_action(session_id, "redo")
    result = _session_preview(restored, session_id)
    result["undo_depth"] = store.undo_depth(session_id)
    result["redo_depth"] = store.redo_depth(session_id)
    return result


# ── Column Metadata ──────────────────────────────────────────────────────────

class ColumnMetadataRequest(BaseModel):
    columns: Dict[str, dict]  # {COL_NAME: {label, units, role, value_labels, description}}


@router.post("/{session_id}/metadata")
async def save_metadata(session_id: str, body: ColumnMetadataRequest):
    """Store column-level metadata for the session."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    store.save_metadata(session_id, body.columns)
    store.log_action(session_id, "metadata_updated", {"columns": list(body.columns.keys())})

    return {"status": "ok", "columns_updated": list(body.columns.keys())}


# ── Column kind override ─────────────────────────────────────────────────────

class KindOverrideRequest(BaseModel):
    column: str
    kind: str  # "numeric" | "categorical" | "text" | "date" | "boolean"


@router.post("/{session_id}/kind")
async def set_column_kind(session_id: str, body: KindOverrideRequest):
    """Persist a user-driven kind change (data-tab badge / dictionary)."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if body.column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{body.column}' not in session")
    if body.kind not in ("numeric", "categorical", "text", "date", "boolean"):
        raise HTTPException(status_code=422, detail=f"Invalid kind '{body.kind}'")

    store.save_kind_overrides(session_id, {body.column: body.kind})
    store.log_action(session_id, "kind_override", {"column": body.column, "kind": body.kind})
    return {"status": "ok", "column": body.column, "kind": body.kind}


# ── Decimal-places override ──────────────────────────────────────────────────
# Per-column display precision the user picks via the data-tab context menu.
# Persisted so save_session round-trips the formatting choice.

class DecimalRequest(BaseModel):
    column: str
    decimals: Optional[int] = None  # None ⇒ clear the override


@router.post("/{session_id}/decimals")
async def set_column_decimals(session_id: str, body: DecimalRequest):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if body.column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{body.column}' not in session")
    if body.decimals is None:
        store.clear_decimal(session_id, body.column)
        return {"status": "ok", "column": body.column, "decimals": None}
    if not (0 <= int(body.decimals) <= 10):
        raise HTTPException(status_code=422, detail="decimals must be between 0 and 10")
    store.set_decimal(session_id, body.column, int(body.decimals))
    return {"status": "ok", "column": body.column, "decimals": int(body.decimals)}


@router.get("/{session_id}/decimals")
async def get_column_decimals(session_id: str):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return store.get_decimals(session_id)


# ── Session rename ────────────────────────────────────────────────────────────
# Lets the user choose a display name for the session — surfaced in the
# header pill, the auto-save IndexedDB record, and the round-tripped JSON.

class RenameRequest(BaseModel):
    filename: str


@router.post("/{session_id}/rename")
async def rename_session(session_id: str, body: RenameRequest):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    name = (body.filename or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="filename cannot be empty")
    if len(name) > 200:
        raise HTTPException(status_code=422, detail="filename too long (>200 chars)")
    store.set_filename(session_id, name)
    return {"status": "ok", "filename": name}


@router.get("/{session_id}")
async def get_session_info(session_id: str):
    """Retrieve session details (filename, rows, columns, preview) for a saved session ID."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from routers.upload import _detect_kind
    kind_overrides = store.get_kind_overrides(session_id)
    columns = []
    for col in df.columns:
        kind = kind_overrides.get(col) or _detect_kind(df[col])
        columns.append({"name": col, "dtype": str(df[col].dtype), "kind": kind})
    _attach_value_labels(columns, session_id)

    import numpy as np, json as _json
    preview = _json.loads(
        df.head(2000).replace([np.inf, -np.inf], np.nan).to_json(
            orient="records", default_handler=str, date_format="iso", date_unit="s"
        )
    )

    return {
        "session_id": session_id,
        "filename": f"iptw_weighted_cohort.csv" if session_id.endswith("_iptw") else f"psm_matched_cohort.csv" if session_id.endswith("_psm") else f"session_{session_id[:8]}.csv",
        "rows": len(df),
        "columns": columns,
        "preview": preview,
    }
