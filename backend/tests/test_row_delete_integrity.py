"""Row-delete data integrity: positional delete, no off-by-one, contiguous index
so a subsequent cell edit can't append a phantom row."""
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed(sid: str) -> str:
    store.save(sid, pd.DataFrame({"x": [10, 20, 30], "y": ["a", "b", "c"]}))
    return sid


def test_delete_first_row_works_and_is_positional():
    sid = _seed("del_first")
    # 0-based position 0 = the first row (x=10). Previously '-1' made it undeletable.
    r = client.delete(f"/api/sessions/{sid}/row/0")
    assert r.status_code == 200, r.text
    df = store.get(sid)
    assert df["x"].tolist() == [20, 30]
    assert list(df.index) == [0, 1]  # reset to a contiguous RangeIndex


def test_delete_then_edit_does_not_create_phantom_row():
    sid = _seed("del_edit")
    assert client.delete(f"/api/sessions/{sid}/row/1").status_code == 200  # drop x=20
    assert store.get(sid)["x"].tolist() == [10, 30]
    # Edit the now-2-row frame at position 1 — must update, not append a 3rd row.
    r = client.patch(f"/api/sessions/{sid}/cell", json={"row_index": 1, "column": "x", "value": 99})
    assert r.status_code == 200, r.text
    df = store.get(sid)
    assert len(df) == 2
    assert df["x"].tolist() == [10, 99]


def test_purge_clears_all_session_maps():
    sid = "purge_me"
    store.save(sid, pd.DataFrame({"a": [1, 2]}))
    store.save_kind_overrides(sid, {"a": "ordinal"})
    store.set_filename(sid, "mydata.csv")
    store.purge_session(sid)
    assert store.get(sid) is None
    assert store.get_kind_overrides(sid) == {}
    assert store.get_filename(sid) is None
