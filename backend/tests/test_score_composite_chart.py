import pandas as pd

from conftest import make_session


BASE = "/api/charts/score_composite"


def test_score_composite_builds_five_panel_plotly_figure(client):
    df = pd.DataFrame({
        "burden": ["LTB"] * 8 + ["HTB"] * 8,
        "cha2ds2_va": [1, 2, 2, 3, 3, 4, 4, 5, 2, 3, 3, 4, 4, 5, 6, 7],
        "atria": [0, 1, 2, 2, 4, 5, 6, 8, 1, 2, 3, 5, 6, 7, 9, 10],
        "htn": [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        "dm": [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        "age_65_74": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        "ckd": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "stroke": [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
    })
    sid = make_session(df, "score_composite_main")

    r = client.post(BASE, json={
        "session_id": sid,
        "group_col": "burden",
        "scores": [
            {
                "score_col": "cha2ds2_va",
                "label": "CHA2DS2-VA",
                "components": ["htn", "dm", "age_65_74"],
                "component_labels": {"htn": "HTN (H)", "dm": "Diabetes (D)", "age_65_74": "Age 65-74 (A)"},
            },
            {
                "score_col": "atria",
                "label": "ATRIA",
                "components": ["htn", "dm", "ckd", "stroke"],
                "component_labels": {"htn": "HTN", "dm": "DM", "ckd": "CKD", "stroke": "Stroke"},
            },
        ],
    })

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "score_composite"
    assert body["groups"] == ["LTB", "HTB"]
    assert "Mann-Whitney" in body["method_note"]
    assert len(body["scores"]) == 2
    assert body["scores"][0]["p_text"].startswith("p ")
    assert body["scores"][0]["components"][0]["label"] == "HTN (H)"

    figure = body["figure"]
    trace_types = {trace["type"] for trace in figure["data"]}
    assert {"histogram", "box", "bar"} <= trace_types
    assert figure["layout"]["xaxis4"]["ticktext"] == ["HTN (H)", "Diabetes (D)", "Age 65-74 (A)"]
    assert figure["layout"]["xaxis5"]["ticktext"] == ["HTN", "DM", "CKD", "Stroke"]
    assert figure["layout"]["yaxis4"]["range"] == [0, 108]


def test_score_composite_rejects_missing_component(client):
    sid = make_session(pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "score1": [1, 2, 3, 4],
        "score2": [2, 3, 4, 5],
        "component": [1, 0, 1, 1],
    }), "score_composite_missing")

    r = client.post(BASE, json={
        "session_id": sid,
        "group_col": "group",
        "scores": [
            {"score_col": "score1", "components": ["component"]},
            {"score_col": "score2", "components": ["missing"]},
        ],
    })

    assert r.status_code == 400
    assert "Column 'missing' not found" in r.text
