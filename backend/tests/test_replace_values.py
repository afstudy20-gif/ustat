"""
Per-column value-map find & replace (in place) — backs the data-grid
'Find & Replace' modal. Distinct from the multi-column cleaning find_replace:
this one applies a value→value map to ONE column, auto-casts the result to
numeric when every value is a number, and remaps existing value labels.
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed(df: pd.DataFrame, sid: str) -> str:
    store.save(sid, df)
    return sid


def test_text_to_numeric_autocast():
    sid = _seed(pd.DataFrame({"sex": ["kadın", "erkek", "kadın", "erkek"]}), "rv_num")
    r = client.post(f"/api/compute/{sid}/replace_values", json={
        "column": "sex", "mapping": {"kadın": "0", "erkek": "1"},
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_replaced"] == 4
    # Stored as a real numeric dtype (not object strings). A 0/1 column is
    # auto-classified categorical (binary), which is correct — the point is the
    # data is numeric so models fit a single 0/1 term, not a text dummy.
    assert d["kind"] in ("numeric", "categorical")
    df = store.get(sid)
    assert pd.api.types.is_numeric_dtype(df["sex"])
    assert df["sex"].tolist() == [0, 1, 0, 1]


def test_value_labels_remap_follows_mapping():
    df = pd.DataFrame({"sex": ["kadın", "erkek"]})
    sid = _seed(df, "rv_vl")
    store.save_metadata(sid, {"sex": {"value_labels": {"kadın": "Female", "erkek": "Male"}}})
    r = client.post(f"/api/compute/{sid}/replace_values", json={
        "column": "sex", "mapping": {"kadın": "0", "erkek": "1"},
    })
    assert r.status_code == 200, r.text
    vl = store.get_metadata(sid)["sex"]["value_labels"]
    # Keys followed the replacement; labels preserved.
    assert vl == {"0": "Female", "1": "Male"}


def test_partial_mapping_keeps_unmapped_and_stays_text():
    df = pd.DataFrame({"grp": ["a", "b", "c"]})
    sid = _seed(df, "rv_partial")
    r = client.post(f"/api/compute/{sid}/replace_values", json={
        "column": "grp", "mapping": {"a": "x"},
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_replaced"] == 1
    assert store.get(sid)["grp"].tolist() == ["x", "b", "c"]
    # Mixed text → not coerced to numeric.
    assert d["kind"] != "numeric"


def test_numeric_coded_column_matches_string_keys():
    # Column stored as ints 1/2/3 — mapping keys come from the UI as strings.
    df = pd.DataFrame({"ldl": [1, 2, 3, 1]})
    sid = _seed(df, "rv_int")
    r = client.post(f"/api/compute/{sid}/replace_values", json={
        "column": "ldl", "mapping": {"1": "10", "3": "30"},
    })
    assert r.status_code == 200, r.text
    assert r.json()["n_replaced"] == 3
    assert store.get(sid)["ldl"].tolist() == [10, 2, 30, 10]


def test_errors():
    sid = _seed(pd.DataFrame({"x": [1, 2]}), "rv_err")
    assert client.post(f"/api/compute/{sid}/replace_values", json={
        "column": "missing", "mapping": {"1": "2"}}).status_code == 404
    assert client.post(f"/api/compute/{sid}/replace_values", json={
        "column": "x", "mapping": {}}).status_code == 422
