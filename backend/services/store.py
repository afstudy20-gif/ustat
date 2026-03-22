"""In-memory dataframe store keyed by session id."""
import pandas as pd
from typing import Dict, List, Optional

_store: Dict[str, pd.DataFrame] = {}

# Case filter conditions: [{column, operator, value, join}]
_filters: Dict[str, List[dict]] = {}


def save(session_id: str, df: pd.DataFrame) -> None:
    _store[session_id] = df


def get(session_id: str) -> Optional[pd.DataFrame]:
    return _store.get(session_id)


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
    df = _store.get(session_id)
    if df is None:
        return None
    conditions = _filters.get(session_id, [])
    return _apply_conditions(df, conditions)


def delete(session_id: str) -> None:
    _store.pop(session_id, None)
    _filters.pop(session_id, None)


def list_sessions() -> list[str]:
    return list(_store.keys())
