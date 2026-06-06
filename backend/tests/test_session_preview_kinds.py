"""
GET /sessions/{id} (and undo/redo, which share _session_preview) must honour
user kind overrides + value labels. Otherwise a multi-level numeric-coded
categorical (e.g. LDL groups 1/2/3) silently reverts to 'numeric' on any
refresh — e.g. right after a recode adds a column.
"""

import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed(sid: str) -> str:
    df = pd.DataFrame({"ldl_grp": [1, 2, 3, 1, 2, 3, 1, 2], "age": [50, 60, 70, 55, 65, 45, 58, 62]})
    store.save(sid, df)
    store.save_kind_overrides(sid, {"ldl_grp": "categorical"})
    client.post(f"/api/sessions/{sid}/metadata", json={
        "columns": {"ldl_grp": {"value_labels": {"1": "<100", "2": "100-130", "3": ">130"}}},
    })
    return sid


def test_get_session_honours_kind_override_and_labels():
    sid = _seed("sp_kinds")
    g = client.get(f"/api/sessions/{sid}").json()
    col = next(c for c in g["columns"] if c["name"] == "ldl_grp")
    assert col["kind"] == "categorical"           # not reverted to numeric
    assert col.get("value_labels") == {"1": "<100", "2": "100-130", "3": ">130"}


def test_kind_override_survives_recode_then_get():
    sid = _seed("sp_kinds_recode")
    # Recode adds a new column (does not touch ldl_grp).
    r = client.post(f"/api/compute/{sid}/recode", json={
        "rules": [{"conditions": [{"col": "age", "op": ">", "val": "60"}], "result": "1"}],
        "else_val": "0", "new_col": "old",
    })
    assert r.status_code == 200, r.text
    g = client.get(f"/api/sessions/{sid}").json()
    col = next(c for c in g["columns"] if c["name"] == "ldl_grp")
    assert col["kind"] == "categorical"
    assert col.get("value_labels") == {"1": "<100", "2": "100-130", "3": ">130"}
