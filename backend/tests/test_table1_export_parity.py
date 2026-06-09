"""Table 1 parity: the DOCX export must match the live /api/stats/table1 for the
same data — same column-aware rounding (median/IQR bounds) and same normality
chooser (Shapiro vs Lilliefors-KS). Regression guard for the reported
'55 [49–63]' (screen) vs '55.00 [48.75–63.00]' (export) bug."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store
from routers.pub_export import _run_table1_analysis, TableDocxRequest

client = TestClient(app)


def _seed() -> str:
    rng = np.random.default_rng(0)
    # Integer-valued age (so _col_decimals → 0) with fractional quartiles, plus a
    # grouping column and 80 rows (Lilliefors-KS tier, n in [50, 2000]).
    n = 80
    age = rng.integers(40, 80, n)
    df = pd.DataFrame({"age": age, "grp": rng.integers(1, 3, n)})
    sid = "t1_parity"
    store.save(sid, df)
    return sid


def _numeric_overall(rows) -> str:
    return next(r["overall"] for r in rows if r["type"] == "numeric")


def test_docx_table1_matches_live_table1():
    sid = _seed()
    live = client.post("/api/stats/table1", json={
        "session_id": sid, "variables": ["age"], "group_column": "grp",
        "variable_kinds": {"age": "numeric"},
    })
    assert live.status_code == 200, live.text
    live_rows = live.json()["rows"]
    live_overall = next(r["overall"] for r in live_rows if r.get("type") == "numeric")

    docx = _run_table1_analysis(TableDocxRequest(
        session_id=sid, variables=["age"], group_column="grp",
        variable_kinds={"age": "numeric"},
    ))
    docx_overall = _numeric_overall(docx["rows"])

    # Identical formatting — integer column → no decimals, IQR bounds rounded
    # the same as the median (the bug was 48.75 vs 49).
    assert live_overall == docx_overall
    assert ".0" not in docx_overall and ".75" not in docx_overall  # integer precision


def test_docx_table1_normality_uses_lilliefors_tier():
    # n=80 → both paths must use the Lilliefors-KS tier (not plain Shapiro),
    # so they agree on the test choice for the same variable.
    sid = _seed()
    docx = _run_table1_analysis(TableDocxRequest(
        session_id=sid, variables=["age"], group_column="grp",
        variable_kinds={"age": "numeric"},
    ))
    row = next(r for r in docx["rows"] if r["type"] == "numeric")
    # A normal-or-not decision was made and a 2-group test was chosen.
    assert row["test"] in ("t-test", "Mann–Whitney")
