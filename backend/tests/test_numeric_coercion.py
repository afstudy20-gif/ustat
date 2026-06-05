"""
Numeric columns must not silently become 'text' across a save/load round-trip.

Bug: save_session serialises with default_handler=str, so a numeric column that
was object-typed (string numbers from import or a prior op) round-trips as
strings and then misclassifies as 'text' in the Data Dictionary. The columns
array also pinned that stale 'text' kind as an override on reload.
"""

import io
import json
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store
from routers.upload import coerce_numeric_objects

client = TestClient(app)


def test_coerce_clean_numeric_strings():
    df = pd.DataFrame({"AST": ["12", "34", "56.5", None, ""]})
    out = coerce_numeric_objects(df)
    assert pd.api.types.is_numeric_dtype(out["AST"])
    assert out["AST"].tolist()[:3] == [12.0, 34.0, 56.5]


def test_preserve_leading_zero_ids():
    df = pd.DataFrame({"ID": ["0123", "0456", "0789"]})
    out = coerce_numeric_objects(df)
    assert out["ID"].dtype == object  # untouched — identifier codes


def test_preserve_real_text():
    df = pd.DataFrame({"NOTE": ["high", "low", "12"]})
    out = coerce_numeric_objects(df)
    assert out["NOTE"].dtype == object  # 'high'/'low' aren't numbers


def test_numeric_kind_survives_roundtrip():
    # Seed a session whose AST column is string-typed numbers (object dtype).
    df = pd.DataFrame({"AST": ["12", "34", "56", "78", "90", "21", "43", "65"]})
    sid = "coerce_rt"
    store.save(sid, df)

    blob = client.get(f"/api/sessions/{sid}/save_session").content
    files = {"file": ("session.json", io.BytesIO(blob), "application/json")}
    r = client.post("/api/sessions/load_session", files=files)
    assert r.status_code == 200, r.text
    col = next(c for c in r.json()["columns"] if c["name"] == "AST")
    assert col["kind"] == "numeric", col
