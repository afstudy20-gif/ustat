"""Missing-value imputation robustness — MICE panel + column fill_blanks.

Regression guards: categorical columns must not crash or hard-fail; they are
imputed with the most-frequent value. Numeric columns use MICE.
"""
import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from docx import Document

from main import app
from services import store

client = TestClient(app)


def _seed(sid: str) -> str:
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bp": rng.normal(120, 15, n),
        "sex": rng.choice(["kadın", "erkek"], n),
    })
    df.loc[rng.choice(n, 8, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "sex"] = np.nan
    store.save(sid, df)
    return sid


def test_panel_mice_mixed_numeric_and_categorical():
    sid = _seed("mi_panel")
    r = client.post("/api/survival_advanced/mice", json={
        "session_id": sid, "columns": ["age", "sex"], "n_imputations": 3, "max_iter": 5, "mechanism": "unknown",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    methods = {c["column"]: c["method"] for c in d["columns"]}
    assert methods["age"] == "PMM"
    assert methods["sex"] == "mode"
    df = store.get(sid)
    assert df["age"].isna().sum() == 0
    assert df["sex"].isna().sum() == 0


def test_column_fill_mice_on_categorical_uses_mode_not_crash():
    sid = _seed("mi_col_cat")
    r = client.post(f"/api/compute/{sid}/fill_blanks", json={"column": "sex", "value": "__mice__"})
    assert r.status_code == 200, r.text
    assert r.json()["n_filled"] == 5
    assert "most frequent" in r.json()["fill_value"]
    assert store.get(sid)["sex"].isna().sum() == 0


def test_column_fill_mice_on_numeric():
    sid = _seed("mi_col_num")
    r = client.post(f"/api/compute/{sid}/fill_blanks", json={"column": "age", "value": "__mice__"})
    assert r.status_code == 200, r.text
    assert r.json()["n_filled"] == 8
    assert store.get(sid)["age"].isna().sum() == 0


def test_column_fill_to_new_column_preserves_original():
    sid = _seed("mi_col_copy")
    before = store.get(sid)["age"].copy()
    r = client.post(f"/api/compute/{sid}/fill_blanks", json={
        "column": "age", "value": "__mice__", "new_column": "age_imp",
    })
    assert r.status_code == 200, r.text
    assert r.json()["column"] == "age_imp"
    assert r.json()["source_column"] == "age"
    assert r.json()["new_column"] is True
    df = store.get(sid)
    assert df["age"].equals(before)
    assert df["age_imp"].isna().sum() == 0
    assert list(df.columns).index("age_imp") == list(df.columns).index("age") + 1


def test_panel_mice_new_columns_preserves_originals():
    sid = _seed("mi_panel_copy")
    before_age = store.get(sid)["age"].copy()
    before_sex = store.get(sid)["sex"].copy()
    r = client.post("/api/survival_advanced/mice", json={
        "session_id": sid,
        "columns": ["age", "sex"],
        "n_imputations": 1,
        "max_iter": 5,
        "new_columns": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["preserved_originals"] is True
    assert body["new_column_map"] == {"age": "age_imp", "sex": "sex_imp"}
    df = store.get(sid)
    assert df["age"].equals(before_age)
    assert df["sex"].equals(before_sex)
    assert df["age_imp"].isna().sum() == 0
    assert df["sex_imp"].isna().sum() == 0


def test_missing_diagnostics_only_reports_selected_columns():
    sid = _seed("mi_diag_selected")
    r = client.post(f"/api/compute/{sid}/missing_diagnostics", json={
        "columns": ["age"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analyzed_columns"] == ["age"]
    assert [item["name"] for item in body["columns"]] == ["age"]


def test_panel_mice_returns_and_audits_methods_text():
    sid = _seed("mi_methods")
    r = client.post("/api/survival_advanced/mice", json={
        "session_id": sid,
        "columns": ["age", "sex"],
        "n_imputations": 1,
        "max_iter": 7,
        "random_state": 123,
        "mechanism": "MAR",
        "new_columns": True,
    })
    assert r.status_code == 200, r.text
    methods = r.json()["methods_text"]
    assert "predictive mean matching" in methods.lower()
    assert "7 iterations" in methods
    assert "random seed 123" in methods
    assert "age as age_imp" in methods

    audit = store.get_audit(sid)
    mice_entries = [entry for entry in audit if entry["action"] == "mice"]
    assert mice_entries
    assert mice_entries[-1]["params"]["methods_text"] == methods

    docx = client.post("/api/pub_export/method_appendix", json={
        "session_id": sid,
        "title": "Test Methods",
    })
    assert docx.status_code == 200, docx.text
    document = Document(io.BytesIO(docx.content))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert methods in text
