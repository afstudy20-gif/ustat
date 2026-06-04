"""In-memory dataframe store keyed by session id with automatic cleanup."""
import os
import pandas as pd
from typing import Dict, List, Optional
import time
from threading import Lock
from fastapi import HTTPException

# Per-dataset size ceiling (rows × columns). Guards the in-memory store against
# a single oversized frame — from upload or from a runaway compute/merge.
# ~20M cells ≈ 200k rows × 100 cols. Override via env.
MAX_SESSION_CELLS = int(os.environ.get("MAX_SESSION_CELLS", str(20_000_000)))

_store: Dict[str, dict] = {}  # {session_id: {"df": DataFrame, "timestamp": float}}
_filters: Dict[str, List[dict]] = {}
_audit: Dict[str, list] = {}
_metadata: Dict[str, dict] = {}
_kinds: Dict[str, Dict[str, str]] = {}  # {session_id: {col: "numeric"|"categorical"|...}}
_decimals: Dict[str, Dict[str, int]] = {}  # {session_id: {col: decimal_places}} for cell-format overrides
_filenames: Dict[str, str] = {}  # {session_id: user-chosen display name}
_undo: Dict[str, list] = {}   # {session_id: [DataFrame snapshots]}
_redo: Dict[str, list] = {}
_lock = Lock()
MAX_UNDO = 30

# Session configuration
SESSION_TTL_SECONDS = 1800  # 30 minutes
MAX_SESSIONS = 20  # Limit concurrent sessions
_last_cleanup = time.time()


def _cleanup_old_sessions() -> None:
    """Remove sessions older than TTL, keeping only the most recent MAX_SESSIONS."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < 60:  # Cleanup every 60 seconds max
        return

    _last_cleanup = now
    with _lock:
        # Remove expired sessions
        expired = [sid for sid, entry in _store.items() if now - entry["timestamp"] > SESSION_TTL_SECONDS]
        for sid in expired:
            _store.pop(sid, None)
            _filters.pop(sid, None)
            _audit.pop(sid, None)
            _metadata.pop(sid, None)
            _kinds.pop(sid, None)
            _undo.pop(sid, None)
            _redo.pop(sid, None)

        # If still over limit, remove oldest sessions
        if len(_store) > MAX_SESSIONS:
            sorted_sids = sorted(_store.items(), key=lambda x: x[1]["timestamp"])
            to_remove = len(_store) - MAX_SESSIONS
            for sid, _ in sorted_sids[:to_remove]:
                _store.pop(sid, None)
                _filters.pop(sid, None)
                _audit.pop(sid, None)
                _kinds.pop(sid, None)
                _undo.pop(sid, None)
                _redo.pop(sid, None)
                _metadata.pop(sid, None)


def save(session_id: str, df: pd.DataFrame, track_undo: bool = True) -> None:
    """Save dataframe with timestamp for TTL tracking.
    If track_undo=True, pushes the previous state onto the undo stack.
    """
    _cleanup_old_sessions()
    n_cells = int(df.shape[0]) * int(df.shape[1])
    if n_cells > MAX_SESSION_CELLS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Dataset too large: {df.shape[0]:,} rows × {df.shape[1]:,} cols "
                f"= {n_cells:,} cells (limit {MAX_SESSION_CELLS:,})."
            ),
        )
    with _lock:
        # Push current state to undo stack before overwriting
        if track_undo and session_id in _store:
            old_df = _store[session_id]["df"]
            _undo.setdefault(session_id, []).append(old_df.copy())
            if len(_undo[session_id]) > MAX_UNDO:
                _undo[session_id] = _undo[session_id][-MAX_UNDO:]
            # Clear redo stack on new action
            _redo.pop(session_id, None)
        _store[session_id] = {"df": df, "timestamp": time.time()}
    log_action(session_id, "data_updated")


def delete_row(session_id: str, row_index: int) -> bool:
    """Delete a specific row by its pandas index and save to trigger undo tracking."""
    with _lock:
        entry = _store.get(session_id)
        if entry is None:
            return False
        df = entry["df"]
        
        # Verify the index exists to avoid KeyError
        if row_index not in df.index:
            return False
            
    # Drop outside the lock just in case then save through the standard pipeline
    # to handle undo tracking
    new_df = df.drop(index=row_index)
    save(session_id, new_df)
    log_action(session_id, "row_deleted", {"row_index": row_index})
    return True



def get(session_id: str) -> Optional[pd.DataFrame]:
    """Get dataframe and update access timestamp."""
    with _lock:
        entry = _store.get(session_id)
        if entry is None:
            return None
        # Update timestamp on access to keep active sessions alive
        entry["timestamp"] = time.time()
        return entry["df"]


def save_filter(session_id: str, conditions: List[dict]) -> None:
    _filters[session_id] = conditions


def get_filter(session_id: str) -> List[dict]:
    return _filters.get(session_id, [])


def clear_filter(session_id: str) -> None:
    _filters.pop(session_id, None)


def _apply_conditions(df: pd.DataFrame, conditions: List[dict]) -> pd.DataFrame:
    if not conditions:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    for i, cond in enumerate(conditions):
        col = cond.get("column", "")
        if col not in df.columns:
            continue
        op = cond.get("operator", "eq")
        val = cond.get("value", "")
        join = cond.get("join", "AND")

        if op == "missing":
            cond_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        elif op == "not_missing":
            cond_mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
        elif op == "contains":
            cond_mask = df[col].astype(str).str.contains(str(val), case=False, na=False)
        else:
            # Try numeric comparison first, fall back to string
            try:
                num_val = float(val)
                s = pd.to_numeric(df[col], errors="coerce")
                if op == "eq":  cond_mask = s == num_val
                elif op == "ne":  cond_mask = s != num_val
                elif op == "gt":  cond_mask = s > num_val
                elif op == "lt":  cond_mask = s < num_val
                elif op == "gte": cond_mask = s >= num_val
                elif op == "lte": cond_mask = s <= num_val
                else:             cond_mask = pd.Series([True] * len(df), index=df.index)
            except (ValueError, TypeError):
                s = df[col].astype(str)
                if op == "eq":  cond_mask = s == str(val)
                elif op == "ne":  cond_mask = s != str(val)
                else:             cond_mask = pd.Series([True] * len(df), index=df.index)

        if i == 0 or join == "AND":
            mask = mask & cond_mask
        else:
            mask = mask | cond_mask

    return df[mask]


def get_filtered(session_id: str) -> Optional[pd.DataFrame]:
    """Return the session dataframe with any active case filter applied."""
    with _lock:
        entry = _store.get(session_id)
        if entry is None:
            return None
        df = entry["df"]
        # Update access timestamp
        entry["timestamp"] = time.time()
        conditions = _filters.get(session_id, [])
    return _apply_conditions(df, conditions)


def delete(session_id: str) -> None:
    _store.pop(session_id, None)
    _filters.pop(session_id, None)
    _audit.pop(session_id, None)
    _metadata.pop(session_id, None)
    _undo.pop(session_id, None)
    _redo.pop(session_id, None)


def list_sessions() -> list[str]:
    return list(_store.keys())


def undo(session_id: str) -> Optional[pd.DataFrame]:
    """Pop the last undo snapshot and restore it. Returns the restored DataFrame or None."""
    with _lock:
        stack = _undo.get(session_id, [])
        if not stack:
            return None
        prev_df = stack.pop()
        # Push current state to redo
        if session_id in _store:
            _redo.setdefault(session_id, []).append(_store[session_id]["df"].copy())
            if len(_redo[session_id]) > MAX_UNDO:
                _redo[session_id] = _redo[session_id][-MAX_UNDO:]
        _store[session_id] = {"df": prev_df, "timestamp": time.time()}
    return prev_df


def redo(session_id: str) -> Optional[pd.DataFrame]:
    """Pop the last redo snapshot and restore it. Returns the restored DataFrame or None."""
    with _lock:
        stack = _redo.get(session_id, [])
        if not stack:
            return None
        next_df = stack.pop()
        # Push current state to undo
        if session_id in _store:
            _undo.setdefault(session_id, []).append(_store[session_id]["df"].copy())
            if len(_undo[session_id]) > MAX_UNDO:
                _undo[session_id] = _undo[session_id][-MAX_UNDO:]
        _store[session_id] = {"df": next_df, "timestamp": time.time()}
    return next_df


def undo_depth(session_id: str) -> int:
    return len(_undo.get(session_id, []))


def redo_depth(session_id: str) -> int:
    return len(_redo.get(session_id, []))


# ── Audit trail ───────────────────────────────────────────────────────────────

def log_action(session_id: str, action: str, params: Optional[dict] = None) -> None:
    """Append an audit entry for the given session."""
    entry = {"action": action, "params": params, "timestamp": time.time()}
    _audit.setdefault(session_id, []).append(entry)


def get_audit(session_id: str) -> list:
    """Return the audit trail for a session."""
    return _audit.get(session_id, [])


# ── Column metadata ──────────────────────────────────────────────────────────

def save_metadata(session_id: str, meta: dict) -> None:
    """Store column-level metadata for a session."""
    _metadata[session_id] = meta


def get_metadata(session_id: str) -> dict:
    """Return column-level metadata for a session."""
    return _metadata.get(session_id, {})


# ── Column kind overrides ────────────────────────────────────────────────────
# User-driven `numeric` ↔ `categorical` (etc.) flips made through the data-tab
# badge / dictionary. Persists alongside the dataframe so save_session can
# round-trip them — otherwise the next load re-runs auto-detection and the
# user's classification choices are silently discarded.

def save_kind_overrides(session_id: str, overrides: Dict[str, str]) -> None:
    """Merge a dict of {column: kind} into the per-session override map."""
    current = _kinds.get(session_id, {})
    current.update({k: v for k, v in overrides.items() if v})
    _kinds[session_id] = current


def set_kind_overrides(session_id: str, overrides: Dict[str, str]) -> None:
    """Replace the override map wholesale (used on load_session restore)."""
    _kinds[session_id] = dict(overrides or {})


def get_kind_overrides(session_id: str) -> Dict[str, str]:
    return _kinds.get(session_id, {})


def clear_kind_override(session_id: str, column: str) -> None:
    if session_id in _kinds:
        _kinds[session_id].pop(column, None)


# ── Column decimal-places overrides ──────────────────────────────────────────
# Per-cell display precision the user picks via the right-click context menu
# in the data table. Persisted server-side so save_session round-trips the
# choice — the previous frontend-only implementation forgot decimals on every
# JSON export.

def save_decimals(session_id: str, decimals: Dict[str, int]) -> None:
    """Replace the decimal-places map wholesale."""
    _decimals[session_id] = {k: int(v) for k, v in (decimals or {}).items()}


def get_decimals(session_id: str) -> Dict[str, int]:
    return _decimals.get(session_id, {})


def set_decimal(session_id: str, column: str, decimals: int) -> None:
    """Set the decimal-places override for a single column."""
    cur = _decimals.get(session_id, {})
    cur[column] = int(decimals)
    _decimals[session_id] = cur


def clear_decimal(session_id: str, column: str) -> None:
    if session_id in _decimals:
        _decimals[session_id].pop(column, None)


def rename_decimal_key(session_id: str, old: str, new: str) -> None:
    """Move the decimal entry to a new column name (called on rename)."""
    if session_id in _decimals and old in _decimals[session_id]:
        _decimals[session_id][new] = _decimals[session_id].pop(old)


# ── Session display name (user-facing rename) ────────────────────────────────

def set_filename(session_id: str, name: str) -> None:
    """Store a user-chosen display name for the session.

    Independent of the original upload filename. Used by save_session so
    the round-tripped JSON carries the renamed value, and by the
    /sessions/{sid}/filename endpoint so the React store can sync on
    rehydrate.
    """
    if name:
        _filenames[session_id] = str(name)


def get_filename(session_id: str) -> Optional[str]:
    return _filenames.get(session_id)


def clear_filename(session_id: str) -> None:
    _filenames.pop(session_id, None)


def get_df(session_id: str) -> "pd.DataFrame":
    """
    Convenience helper: return the (filtered) dataframe for a session.
    Raises 404 if the session does not exist.
    This used to live in routers/models.py as _get_df.
    """
    df = get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df
