import numpy as np
import pandas as pd

from services import store
from services.journal_formatter import format_table1_for_journal


def _sid(name: str) -> str:
    return f"pkg9_{name}"


def test_select_cases_rejects_unknown_operator(client):
    sid = _sid("bad_operator")
    store.save(sid, pd.DataFrame({"age": [45, 70, 82]}))

    r = client.post(
        f"/api/sessions/{sid}/select_cases",
        json={"conditions": [{"column": "age", "operator": "==", "value": 70}]},
    )

    assert r.status_code == 422
    assert "unsupported operator" in r.json()["detail"]


def test_find_replace_reports_absent_value_without_claiming_change(client):
    sid = _sid("find_absent")
    store.save(sid, pd.DataFrame({"group": ["A", "B", "A"]}))

    r = client.post(
        f"/api/compute/{sid}/find_replace",
        json={"columns": ["group"], "find_value": "C", "replace_value": "D"},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["replaced_count"] == 0
    assert body["found"] is False
    assert body["changed"] is False
    assert any("No matching values" in w for w in body["warnings"])


def test_clean_outliers_counts_plausibility_removals_for_age(client):
    sid = _sid("age_outliers")
    store.save(sid, pd.DataFrame({"age": [52, 63, 71, 999, 130, None], "x": range(6)}))

    r = client.post(
        f"/api/compute/{sid}/clean_outliers",
        json={"columns": ["age"], "method": "iqr", "threshold": 1.5},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] == 2
    assert body["per_column_deleted"]["age"] == 2
    assert body["remaining_rows"] == 4
    assert any("plausible maximum" in w for w in body["warnings"])


def test_decompose_marks_weak_seasonality(client):
    sid = _sid("weak_seasonality")
    x = np.arange(72)
    y = 100 + 0.2 * x
    store.save(sid, pd.DataFrame({"t": x, "value": y}))

    r = client.post(
        "/api/timeseries/decompose",
        json={"session_id": sid, "value_col": "value", "time_col": "t", "period": 12, "method": "stl"},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seasonality_detected"] is False
    assert body["warnings"]


def test_auto_arima_warns_when_selected_model_has_no_ar_ma_terms(client):
    sid = _sid("auto_arima_baseline")
    rng = np.random.default_rng(42)
    store.save(sid, pd.DataFrame({"t": np.arange(80), "value": rng.normal(0, 1, 80)}))

    r = client.post(
        "/api/timeseries/arima",
        json={"session_id": sid, "value_col": "value", "time_col": "t", "auto": True, "forecast_steps": 3},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    if body["order"][0] == 0 and body["order"][2] == 0:
        assert any("p=0 and q=0" in w for w in body["warnings"])
    else:
        assert any("AIC overfit" in w for w in body["warnings"])


def test_journal_formatter_accepts_own_three_decimal_p_values():
    table1_result = {
        "group_column": "group",
        "group_labels": ["0", "1"],
        "group_ns": {"0": 10, "1": 12},
        "total_n": 22,
        "rows": [
            {
                "type": "numeric",
                "variable": "Age",
                "test": "Student t-test",
                "p_value": 0.055,
                "stat_rows": [
                    {"label": "mean \u00b1 SD", "group_stats": {"0": "60.1 \u00b1 8.0", "1": "65.4 \u00b1 7.5"}},
                    {"label": "median [IQR]", "group_stats": {"0": "60 [55-66]", "1": "66 [61-70]"}},
                ],
            }
        ],
    }

    formatted = format_table1_for_journal(table1_result)

    assert formatted["rows"][0]["p_value"] == "0.055"
    assert formatted["validation"]["p_value_formatting"] == "PASS"
    assert formatted["validation"]["status"] == "READY FOR SUBMISSION"


def test_meta_regression_labels_wls_r2(client):
    studies = [
        {"label": "A", "effect": 0.75, "ci_low": 0.55, "ci_high": 1.02, "moderator": 2010},
        {"label": "B", "effect": 0.90, "ci_low": 0.70, "ci_high": 1.16, "moderator": 2012},
        {"label": "C", "effect": 1.05, "ci_low": 0.82, "ci_high": 1.35, "moderator": 2014},
        {"label": "D", "effect": 1.20, "ci_low": 0.96, "ci_high": 1.50, "moderator": 2016},
    ]

    r = client.post("/api/meta/regression", json={"studies": studies, "measure": "OR"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["r2_label"] == "Weighted least-squares R\u00b2 (%)"
    assert "not a direct proportion of tau" in body["r2_note"]
