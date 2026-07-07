import numpy as np
import pandas as pd

from conftest import make_session


BASE = "/api/charts/km_composite"


def _trial_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    arm = rng.choice(["Pressure wire", "FFRangio"], n)

    def endpoint(rate: float):
        dur = rng.exponential(20, n).clip(0, 12)
        ev = (rng.random(n) < rate).astype(int)
        return dur, ev

    d0, e0 = endpoint(0.07)
    d1, e1 = endpoint(0.02)
    d2, e2 = endpoint(0.025)
    return pd.DataFrame({
        "arm": arm,
        "dur_primary": d0, "ev_primary": e0,
        "dur_death": d1, "ev_death": e1,
        "dur_mi": d2, "ev_mi": e2,
    })


def test_km_composite_builds_multi_panel_cuminc_figure(client):
    sid = make_session(_trial_df(), "km_composite_main")

    r = client.post(BASE, json={
        "session_id": sid,
        "group_col": "arm",
        "risk_times": [0, 3, 6, 9, 12],
        "title": "Composite Primary End Point and Individual Components",
        "endpoints": [
            {"duration_col": "dur_primary", "event_col": "ev_primary", "label": "Primary End Point"},
            {"duration_col": "dur_death", "event_col": "ev_death", "label": "Death from Any Cause"},
            {"duration_col": "dur_mi", "event_col": "ev_mi", "label": "Myocardial Infarction"},
        ],
    })

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "km_composite"
    assert set(body["groups"]) == {"Pressure wire", "FFRangio"}
    assert len(body["endpoints"]) == 3
    assert body["endpoints"][0]["p_text"].startswith("p ")
    # Cumulative incidence: each arm has a final accrued percentage per endpoint.
    for ep in body["endpoints"]:
        assert set(ep["final_by_group"]) == {"Pressure wire", "FFRangio"}
        assert set(ep["n_by_group"]) == {"Pressure wire", "FFRangio"}

    figure = body["figure"]
    # 3 endpoints x 2 arms x 2 (main + inset) = 12 traces.
    assert len(figure["data"]) == 12
    assert all(t["type"] == "scatter" for t in figure["data"])
    # Main axes present for all three panels.
    assert "xaxis" in figure["layout"] and "xaxis2" in figure["layout"] and "xaxis3" in figure["layout"]
    # Inset axes for each panel (offset by 4).
    assert "xaxis5" in figure["layout"] and "yaxis5" in figure["layout"]
    # No.-at-risk table + panel letters rendered as annotations.
    texts = [a.get("text", "") for a in figure["layout"]["annotations"]]
    assert any("No. at Risk" in t for t in texts)
    assert any(t.startswith("<b>A</b>") for t in texts)


def test_km_composite_survival_orientation_toggle(client):
    sid = make_session(_trial_df(120), "km_composite_surv")
    r = client.post(BASE, json={
        "session_id": sid,
        "group_col": "arm",
        "as_cumulative_incidence": False,
        "inset": False,
        "endpoints": [
            {"duration_col": "dur_primary", "event_col": "ev_primary", "label": "Primary"},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["as_cumulative_incidence"] is False
    # inset off -> one panel x two arms = 2 traces, no inset axes.
    assert len(body["figure"]["data"]) == 2
    assert "xaxis5" not in body["figure"]["layout"]


def test_km_composite_rejects_non_binary_event(client):
    df = pd.DataFrame({
        "arm": ["A", "A", "B", "B"],
        "dur": [1.0, 2.0, 3.0, 4.0],
        "ev": [0, 1, 2, 1],  # 2 is invalid
    })
    sid = make_session(df, "km_composite_badev")
    r = client.post(BASE, json={
        "session_id": sid,
        "group_col": "arm",
        "endpoints": [{"duration_col": "dur", "event_col": "ev"}],
    })
    assert r.status_code == 422
    assert "binary" in r.text.lower()


def test_km_composite_rejects_missing_column(client):
    sid = make_session(_trial_df(60), "km_composite_missing")
    r = client.post(BASE, json={
        "session_id": sid,
        "group_col": "arm",
        "endpoints": [{"duration_col": "dur_primary", "event_col": "nope"}],
    })
    assert r.status_code == 400
    assert "Column 'nope' not found" in r.text


def test_km_composite_rejects_too_many_endpoints(client):
    sid = make_session(_trial_df(60), "km_composite_over")
    r = client.post(BASE, json={
        "session_id": sid,
        "group_col": "arm",
        "endpoints": [
            {"duration_col": "dur_primary", "event_col": "ev_primary"},
            {"duration_col": "dur_death", "event_col": "ev_death"},
            {"duration_col": "dur_mi", "event_col": "ev_mi"},
            {"duration_col": "dur_primary", "event_col": "ev_primary"},
            {"duration_col": "dur_death", "event_col": "ev_death"},
        ],
    })
    assert r.status_code == 400
    assert "between 1 and 4" in r.text
