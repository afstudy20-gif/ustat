"""Tests for routers/session.py (mounted at /api/sessions).

Covers cell editing, column ops, filtering (select_cases), export, save/load,
audit, undo/redo, metadata, kind & decimals overrides.

Session id prefix for this module: "tsess".
"""
import io
import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_session

SEED = 7


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(SEED)
    n = 120
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    sex = rng.integers(0, 2, n)
    grp = rng.integers(0, 3, n)
    name = np.array([f"subj_{i}" for i in range(n)])
    return pd.DataFrame(
        {
            "AGE": age,
            "LDL": ldl,
            "SEX": sex,
            "GRP": grp,
            "NAME": name,
        }
    )


def _new_session(df, suffix):
    """Persist a fresh independent copy under a unique tsess id."""
    return make_session(df.copy(), f"tsess_{suffix}")


# ── Cell editing ───────────────────────────────────────────────────────────────

def test_update_cell_numeric(client, df):
    sid = _new_session(df, "cell")
    r = client.patch(f"/api/sessions/{sid}/cell", json={"row_index": 0, "column": "AGE", "value": 42})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["row_index"] == 0
    assert body["column"] == "AGE"
    assert abs(float(body["value"]) - 42.0) < 1e-9


def test_update_cell_blank_becomes_missing(client, df):
    sid = _new_session(df, "cell_blank")
    r = client.patch(f"/api/sessions/{sid}/cell", json={"row_index": 1, "column": "LDL", "value": ""})
    assert r.status_code == 200, r.text
    assert r.json()["value"] is None


def test_update_cell_bad_column(client, df):
    sid = _new_session(df, "cell_badcol")
    r = client.patch(f"/api/sessions/{sid}/cell", json={"row_index": 0, "column": "NOPE", "value": 1})
    assert r.status_code == 400, r.text


def test_update_cell_row_out_of_range(client, df):
    sid = _new_session(df, "cell_oob")
    r = client.patch(f"/api/sessions/{sid}/cell", json={"row_index": 99999, "column": "AGE", "value": 1})
    assert r.status_code == 400, r.text


def test_update_cell_missing_session(client):
    r = client.patch("/api/sessions/tsess_nope/cell", json={"row_index": 0, "column": "AGE", "value": 1})
    assert r.status_code == 404, r.text


def test_clear_cells(client, df):
    sid = _new_session(df, "clear")
    r = client.post(
        f"/api/sessions/{sid}/clear_cells",
        json={"cells": [{"row_index": 0, "column": "AGE"}, {"row_index": 2, "column": "LDL"}, {"row_index": -1, "column": "AGE"}]},
    )
    assert r.status_code == 200, r.text
    # two valid cells cleared, one invalid skipped
    assert r.json()["cleared"] == 2


# ── Column ops ─────────────────────────────────────────────────────────────────

def test_reorder_columns(client, df):
    sid = _new_session(df, "reorder")
    new_order = ["SEX", "AGE", "LDL"]  # subset; rest appended
    r = client.post(f"/api/sessions/{sid}/reorder_columns", json={"columns": new_order})
    assert r.status_code == 200, r.text
    cols = r.json()["columns"]
    assert cols[:3] == new_order
    assert set(cols) == {"AGE", "LDL", "SEX", "GRP", "NAME"}


def test_delete_row(client, df):
    sid = _new_session(df, "delrow")
    r = client.delete(f"/api/sessions/{sid}/row/2")  # 1-based -> index 1
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == len(df) - 1
    assert "columns" in body and "preview" in body


def test_delete_row_invalid_index(client, df):
    sid = _new_session(df, "delrow_bad")
    r = client.delete(f"/api/sessions/{sid}/row/0")  # 1-based 0 -> target -1
    assert r.status_code == 400, r.text


# ── Select cases (filtering) ───────────────────────────────────────────────────

def test_select_cases_numeric(client, df):
    sid = _new_session(df, "select")
    r = client.post(
        f"/api/sessions/{sid}/select_cases",
        json={"conditions": [{"column": "AGE", "operator": "gt", "value": 60, "join": "AND"}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == len(df)
    assert body["applied"] is True
    assert 0 <= body["selected"] <= len(df)
    # cross-check against the data
    expected = int((df["AGE"] > 60).sum())
    assert body["selected"] == expected


def test_clear_cases(client, df):
    sid = _new_session(df, "select_clear")
    client.post(
        f"/api/sessions/{sid}/select_cases",
        json={"conditions": [{"column": "AGE", "operator": "lt", "value": 50}]},
    )
    r = client.delete(f"/api/sessions/{sid}/select_cases")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["selected"] == len(df) == body["total"]


# ── Export ─────────────────────────────────────────────────────────────────────

def test_export_csv_generic(client, df):
    sid = _new_session(df, "exp_csv")
    r = client.get(f"/api/sessions/{sid}/export", params={"fmt": "csv", "filename": "out"})
    assert r.status_code == 200, r.text
    assert "attachment" in r.headers.get("content-disposition", "")
    text = r.content.decode("utf-8-sig")
    assert "AGE" in text.splitlines()[0]


def test_export_xlsx_generic(client, df):
    sid = _new_session(df, "exp_xlsx")
    r = client.get(f"/api/sessions/{sid}/export", params={"fmt": "xlsx", "filename": "out"})
    assert r.status_code == 200, r.text
    assert r.content[:2] == b"PK"  # xlsx is a zip


def test_export_csv_endpoint(client, df):
    sid = _new_session(df, "exp_csv2")
    r = client.get(f"/api/sessions/{sid}/export/csv", params={"filename": "data.csv"})
    assert r.status_code == 200, r.text
    text = r.content.decode("utf-8")
    assert "LDL" in text.splitlines()[0]


def test_export_missing_session(client):
    r = client.get("/api/sessions/tsess_missing/export", params={"fmt": "csv"})
    assert r.status_code == 404, r.text


# ── Save / Load session round-trip ─────────────────────────────────────────────

def test_save_and_load_session(client, df):
    sid = _new_session(df, "saveload")
    r = client.get(f"/api/sessions/{sid}/save_session")
    assert r.status_code == 200, r.text
    payload = json.loads(r.content.decode("utf-8"))
    assert "data" in payload and "columns" in payload
    assert len(payload["data"]) == len(df)

    # Round-trip via load_session (multipart file upload)
    files = {"file": ("session.json", io.BytesIO(r.content), "application/json")}
    r2 = client.post("/api/sessions/load_session", files=files)
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["rows"] == len(df)
    assert {c["name"] for c in body["columns"]} == set(df.columns)
    assert "session_id" in body


def test_save_and_load_session_preserves_case_filter(client, df):
    sid = _new_session(df, "saveload_filter")
    conditions = [{"column": "AGE", "operator": "gt", "value": 50, "join": "AND"}]
    selected = client.post(
        f"/api/sessions/{sid}/select_cases",
        json={"conditions": conditions},
    )
    assert selected.status_code == 200, selected.text

    saved = client.get(f"/api/sessions/{sid}/save_session")
    assert saved.status_code == 200, saved.text

    files = {"file": ("session.json", io.BytesIO(saved.content), "application/json")}
    loaded = client.post("/api/sessions/load_session", files=files)
    assert loaded.status_code == 200, loaded.text
    case_filter = loaded.json()["case_filter"]
    assert case_filter["conditions"] == conditions
    assert case_filter["selected"] == int((df["AGE"] > 50).sum())
    assert case_filter["total"] == len(df)


def test_load_session_invalid_json(client):
    files = {"file": ("bad.json", io.BytesIO(b"not json"), "application/json")}
    r = client.post("/api/sessions/load_session", files=files)
    assert r.status_code == 400, r.text


def test_load_session_missing_data_key(client):
    files = {"file": ("nodata.json", io.BytesIO(b'{"foo": 1}'), "application/json")}
    r = client.post("/api/sessions/load_session", files=files)
    assert r.status_code == 400, r.text


# ── Audit ──────────────────────────────────────────────────────────────────────

def test_audit_trail(client, df):
    sid = _new_session(df, "audit")
    # generate an audit entry via metadata save
    client.post(f"/api/sessions/{sid}/metadata", json={"columns": {"AGE": {"label": "Age"}}})
    r = client.get(f"/api/sessions/{sid}/audit")
    assert r.status_code == 200, r.text
    audit = r.json()
    assert isinstance(audit, list)
    assert any(e["action"] == "metadata_updated" for e in audit)


# ── Undo / Redo ────────────────────────────────────────────────────────────────

def test_undo_redo_cycle(client, df):
    sid = _new_session(df, "undo")
    # mutate a cell so there is something to undo
    client.patch(f"/api/sessions/{sid}/cell", json={"row_index": 0, "column": "AGE", "value": 1})
    r = client.post(f"/api/sessions/{sid}/undo")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "undo_depth" in body and "redo_depth" in body
    assert body["redo_depth"] >= 1
    # now redo
    r2 = client.post(f"/api/sessions/{sid}/redo")
    assert r2.status_code == 200, r2.text
    assert r2.json()["undo_depth"] >= 1


def test_undo_nothing(client, df):
    sid = _new_session(df, "undo_empty")
    r = client.post(f"/api/sessions/{sid}/undo")
    assert r.status_code == 400, r.text


# ── Column metadata ────────────────────────────────────────────────────────────

def test_save_metadata(client, df):
    sid = _new_session(df, "meta")
    r = client.post(
        f"/api/sessions/{sid}/metadata",
        json={"columns": {"LDL": {"label": "LDL cholesterol", "units": "mg/dL"}}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert "LDL" in body["columns_updated"]


# ── Kind override ──────────────────────────────────────────────────────────────

def test_set_kind(client, df):
    sid = _new_session(df, "kind")
    r = client.post(f"/api/sessions/{sid}/kind", json={"column": "SEX", "kind": "categorical"})
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "categorical"


def test_set_kind_invalid_kind(client, df):
    sid = _new_session(df, "kind_bad")
    r = client.post(f"/api/sessions/{sid}/kind", json={"column": "SEX", "kind": "bogus"})
    assert r.status_code == 422, r.text


def test_set_kind_unknown_column(client, df):
    sid = _new_session(df, "kind_nocol")
    r = client.post(f"/api/sessions/{sid}/kind", json={"column": "ZZZ", "kind": "numeric"})
    assert r.status_code == 404, r.text


# ── Decimals override ──────────────────────────────────────────────────────────

def test_set_and_get_decimals(client, df):
    sid = _new_session(df, "dec")
    r = client.post(f"/api/sessions/{sid}/decimals", json={"column": "AGE", "decimals": 3})
    assert r.status_code == 200, r.text
    assert r.json()["decimals"] == 3

    r2 = client.get(f"/api/sessions/{sid}/decimals")
    assert r2.status_code == 200, r2.text
    assert r2.json().get("AGE") == 3


def test_set_decimals_out_of_range(client, df):
    sid = _new_session(df, "dec_bad")
    r = client.post(f"/api/sessions/{sid}/decimals", json={"column": "AGE", "decimals": 99})
    assert r.status_code == 422, r.text


def test_clear_decimals(client, df):
    sid = _new_session(df, "dec_clear")
    client.post(f"/api/sessions/{sid}/decimals", json={"column": "AGE", "decimals": 2})
    r = client.post(f"/api/sessions/{sid}/decimals", json={"column": "AGE", "decimals": None})
    assert r.status_code == 200, r.text
    assert r.json()["decimals"] is None
