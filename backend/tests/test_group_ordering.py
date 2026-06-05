"""
Grouped output must follow value-code order, not data order or naive string
sort. A string sort breaks on multi-digit codes (1, 2, 10 -> 1, 10, 2).
"""

import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store
from services.stat_utils import sorted_groups

client = TestClient(app)


def test_sorted_groups_numeric_multidigit():
    s = pd.Series([10, 2, 1, 2, 10, 1])
    assert sorted_groups(s) == [1.0, 2.0, 10.0]  # not 1, 10, 2


def test_sorted_groups_strings():
    s = pd.Series(["beta", "alpha", "beta"])
    assert sorted_groups(s) == ["alpha", "beta"]


def test_sorted_groups_drops_na():
    s = pd.Series([3, None, 1, 2])
    assert sorted_groups(s) == [1.0, 2.0, 3.0]


def test_table1_group_columns_value_code_order():
    # Group codes seeded scrambled and including a multi-digit code.
    df = pd.DataFrame({
        "grp": [10, 1, 2, 1, 2, 10, 2, 1, 10, 2],
        "age": [50, 61, 55, 62, 58, 49, 57, 63, 48, 56],
    })
    sid = "t1_order"
    store.save(sid, df)
    r = client.post("/api/stats/table1", json={
        "session_id": sid, "variables": ["age"], "group_column": "grp",
    })
    assert r.status_code == 200, r.text
    labels = r.json().get("group_labels")
    assert labels == ["1", "2", "10"], labels
