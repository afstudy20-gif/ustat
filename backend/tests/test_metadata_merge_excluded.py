"""
Per-column metadata merges (no wipe) and analysis_excluded round-trips.

Bug guarded: save_metadata used to REPLACE the whole map, so setting a flag on
one column wiped value_labels on others. It now merges per column.
"""

import io
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed(sid: str) -> str:
    df = pd.DataFrame({"NAME": ["a", "b"], "LDL": [1, 0], "AGE": [50, 60]})
    store.save(sid, df)
    return sid


def test_partial_update_does_not_wipe_other_columns():
    sid = _seed("meta_merge")
    # Set value labels on LDL.
    client.post(f"/api/sessions/{sid}/metadata", json={
        "columns": {"LDL": {"value_labels": {"0": "low", "1": "high"}}},
    })
    # Now exclude NAME from analysis — must NOT wipe LDL's value labels.
    client.post(f"/api/sessions/{sid}/metadata", json={
        "columns": {"NAME": {"analysis_excluded": True}},
    })
    g = client.get(f"/api/sessions/{sid}").json()
    ldl = next(c for c in g["columns"] if c["name"] == "LDL")
    name = next(c for c in g["columns"] if c["name"] == "NAME")
    assert ldl.get("value_labels") == {"0": "low", "1": "high"}
    assert name.get("analysis_excluded") is True


def test_analysis_excluded_roundtrips_save_load():
    sid = _seed("meta_excl_rt")
    client.post(f"/api/sessions/{sid}/metadata", json={
        "columns": {"NAME": {"analysis_excluded": True}},
    })
    blob = client.get(f"/api/sessions/{sid}/save_session").content
    files = {"file": ("s.json", io.BytesIO(blob), "application/json")}
    r = client.post("/api/sessions/load_session", files=files)
    assert r.status_code == 200, r.text
    name = next(c for c in r.json()["columns"] if c["name"] == "NAME")
    assert name.get("analysis_excluded") is True


def test_name_suggestions_endpoint():
    sid = _seed("meta_suggest")
    r = client.get(f"/api/sessions/{sid}/name_suggestions")
    assert r.status_code == 200, r.text
    sug = r.json()["suggestions"]
    # NAME -> "Name" (all-caps plain word), AGE -> "Age".
    assert sug.get("NAME") == "Name"
    assert sug.get("AGE") == "Age"
    # LDL is a known acronym -> unchanged -> omitted.
    assert "LDL" not in sug
