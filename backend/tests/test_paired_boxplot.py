import numpy as np
import pandas as pd

from conftest import make_session


def _paired_df():
    # 5 matched pairs; pair_id links each control row to its diabetic partner.
    return pd.DataFrame({
        "PWD": [100, 105, 110, 98, 120, 108, 130, 95, 115, 125],
        "DM": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "match_set_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    })


def test_paired_box_basic(client):
    sid = make_session(_paired_df(), "paired_basic")
    r = client.post("/api/charts/paired_box", json={
        "session_id": sid, "y": "PWD", "group": "DM", "pair_id": "match_set_id",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "paired_box"
    assert len(body["groups"]) == 2
    assert body["n_pairs"] == 5
    assert body["n_unpaired"] == 0
    # Each pair's two values should trace back to the matching match_set_id rows
    for pair in body["pairs"]:
        assert pair["pair_id"] in {"1", "2", "3", "4", "5"}


def test_paired_box_requires_two_levels(client):
    df = _paired_df()
    df["DM"] = 0  # collapse to a single level
    sid = make_session(df, "paired_one_level")
    r = client.post("/api/charts/paired_box", json={
        "session_id": sid, "y": "PWD", "group": "DM", "pair_id": "match_set_id",
    })
    assert r.status_code == 400
    assert "2 levels" in r.json()["detail"]


def test_paired_box_unmatched_rows_excluded_from_pairs(client):
    df = _paired_df()
    # Break one pair's link — only its DM=1 side gets a match_set_id no
    # other row shares, so it can no longer be paired with DM=0's "5".
    df["match_set_id"] = df["match_set_id"].astype(str)
    df.loc[(df["match_set_id"] == "5") & (df["DM"] == 1), "match_set_id"] = "orphan"
    sid = make_session(df, "paired_orphan")
    r = client.post("/api/charts/paired_box", json={
        "session_id": sid, "y": "PWD", "group": "DM", "pair_id": "match_set_id",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_pairs"] == 4
    assert body["n_unpaired"] == 2
    # Both groups still report all their values in the box (unpaired included)
    assert sum(len(g["values"]) for g in body["groups"]) == 10


def test_paired_box_column_not_found(client):
    sid = make_session(_paired_df(), "paired_missing_col")
    r = client.post("/api/charts/paired_box", json={
        "session_id": sid, "y": "NOPE", "group": "DM", "pair_id": "match_set_id",
    })
    assert r.status_code == 400
