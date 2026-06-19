import pandas as pd

from conftest import make_session
from services import store


def _session(name: str = "summary_visual_numeric_hygiene"):
    df = pd.DataFrame({
        "age": ["40", "50", "199", "60", "55", "45"],
        "bmi": ["25,5", "999", "27.0", "26.2", None, "29.1"],
        "fu_days": ["10", "-5", "20", "30", "40", "50"],
        "weight": [1, 2, 1, 1, 1, 1],
        "xgrp": ["A", "A", "B", "B", "A", "B"],
        "subgrp": ["S1", "S2", "S1", "S2", "S1", "S2"],
        "other": [1, None, 3, 4, 5, 6],
    })
    sid = make_session(df, name)
    store.save_kind_overrides(sid, {"age": "numeric", "bmi": "numeric", "fu_days": "numeric"})
    return sid


def test_column_summary_and_histogram_coerce_text_numeric_with_warnings(client):
    sid = _session("summary_visual_column_hist")

    summary = client.get(f"/api/stats/{sid}/column_summary", params={
        "column": "age", "kind": "numeric",
    })
    assert summary.status_code == 200, summary.text
    body = summary.json()
    assert body["type"] == "numeric"
    assert body["warnings"][0]["implausible_values"] == [199.0]

    hist = client.post("/api/charts/histogram", json={
        "session_id": sid, "x": "bmi",
    })
    assert hist.status_code == 200, hist.text
    h = hist.json()
    assert h["stats"]["mean"] > 200
    assert h["warnings"][0]["implausible_values"] == [999.0]


def test_descriptive_and_table1_honor_stored_numeric_kind_for_text_columns(client):
    sid = _session("summary_visual_table1")

    desc = client.get(f"/api/stats/{sid}/descriptive")
    assert desc.status_code == 200, desc.text
    assert "bmi" in desc.json()
    assert desc.json()["age"]["warnings"][0]["implausible_values"] == [199.0]

    table = client.post("/api/stats/table1", json={
        "session_id": sid,
        "group_column": "xgrp",
        "variables": ["bmi"],
    })
    assert table.status_code == 200, table.text
    row = table.json()["rows"][0]
    assert row["type"] == "numeric"
    assert table.json()["warnings"][0]["implausible_values"] == [999.0]


def test_subgroup_bar_without_color_splits_traces_and_masks_bmi_sentinel(client):
    sid = _session("summary_visual_subgroup")

    r = client.post("/api/charts/subgroup_bar", json={
        "session_id": sid,
        "y_col": "bmi",
        "subgroup_col": "subgrp",
        "xaxis_col": "xgrp",
        "y_mode": "mean",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert [tr["name"] for tr in body["traces"]] == ["A", "B"]
    assert body["warnings"][0]["variable"] == "bmi"
    all_means = [y for tr in body["traces"] for y in tr["y"]]
    assert max(all_means) < 100


def test_weighted_descriptive_uses_per_column_complete_cases_by_default(client):
    sid = _session("summary_visual_weighted")

    r = client.post("/api/stats/weighted_descriptive", json={
        "session_id": sid,
        "weight_col": "weight",
        "value_cols": ["bmi", "other"],
    })
    assert r.status_code == 200, r.text
    rows = {row["column"]: row for row in r.json()["results"]}
    assert rows["bmi"]["n"] == 5
    assert rows["other"]["n"] == 5
