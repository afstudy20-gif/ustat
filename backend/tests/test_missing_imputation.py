"""Missing-value imputation robustness — MICE panel + column fill_blanks.

Regression guards: categorical columns must not crash or hard-fail; they are
imputed with the most-frequent value. Numeric columns use MICE.
"""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

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
    assert methods["age"] == "MICE"
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
