"""
Value labels set at recode time must survive a session reload.

Bug: recode saved value labels to the metadata store, but the column
objects returned by GET /sessions/{id} (and load_session) were built
without merging them back, so the Data Dictionary showed empty inputs
after a refresh.
"""

import io
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed(df: pd.DataFrame, sid: str) -> str:
    store.save(sid, df)
    return sid


def test_value_labels_merged_into_get_session():
    df = pd.DataFrame({"LDL100": [0, 1, 0, 1, 1]})
    sid = _seed(df, "vl_get")
    # Save value labels the way the recode flow does.
    r = client.post(f"/api/sessions/{sid}/metadata", json={
        "columns": {"LDL100": {"value_labels": {"0": "<100 mg/dL", "1": "≥100 mg/dL"}}},
    })
    assert r.status_code == 200, r.text
    # GET session must echo them on the column object.
    g = client.get(f"/api/sessions/{sid}")
    assert g.status_code == 200, g.text
    col = next(c for c in g.json()["columns"] if c["name"] == "LDL100")
    assert col.get("value_labels") == {"0": "<100 mg/dL", "1": "≥100 mg/dL"}


def test_value_labels_survive_save_load_session():
    df = pd.DataFrame({"LDL100": [0, 1, 0, 1]})
    sid = _seed(df, "vl_roundtrip")
    client.post(f"/api/sessions/{sid}/metadata", json={
        "columns": {"LDL100": {"value_labels": {"0": "low", "1": "high"}}},
    })
    blob = client.get(f"/api/sessions/{sid}/save_session").content
    files = {"file": ("session.json", io.BytesIO(blob), "application/json")}
    r = client.post("/api/sessions/load_session", files=files)
    assert r.status_code == 200, r.text
    col = next(c for c in r.json()["columns"] if c["name"] == "LDL100")
    assert col.get("value_labels") == {"0": "low", "1": "high"}
