"""In-memory dataframe store keyed by session id with automatic cleanup."""
import pandas as pd
from typing import Dict, List, Optional
import time
from threading import Lock

_store: Dict[str, dict] = {}  # {session_id: {"df": DataFrame, "timestamp": float}}
_filters: Dict[str, List[dict]] = {}
_audit: Dict[str, list] = {}
_metadata: Dict[str, dict] = {}
_lock = Lock()

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

        # If still over limit, remove oldest sessions
        if len(_store) > MAX_SESSIONS:
            sorted_sids = sorted(_store.items(), key=lambda x: x[1]["timestamp"])
            to_remove = len(_store) - MAX_SESSIONS
            for sid, _ in sorted_sids[:to_remove]:
                _store.pop(sid, None)
                _filters.pop(sid, None)
                _audit.pop(sid, None)
                _metadata.pop(sid, None)


def save(session_id: str, df: pd.DataFrame) -> None:
    """Save dataframe with timestamp for TTL tracking."""
    _cleanup_old_sessions()
    with _lock:
        _store[session_id] = {"df": df, "timestamp": time.time()}
    log_action(session_id, "data_updated")


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


def list_sessions() -> list[str]:
    return list(_store.keys())


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
