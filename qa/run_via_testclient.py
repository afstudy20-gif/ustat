"""Helper: load qa/cohort_test.csv into the backend via FastAPI TestClient.

Lets agents exercise every router without spinning up uvicorn / a frontend.
Returns a TestClient + the session_id of the loaded dataset.

Usage:
    from qa.run_via_testclient import boot
    client, sid = boot()
    r = client.post("/api/stats/ttest", json={...})
"""
from __future__ import annotations
import io, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # type: ignore
from main import app  # type: ignore


def boot(csv_path: str | None = None):
    csv_path = csv_path or os.path.join(os.path.dirname(__file__), "cohort_test.csv")
    client = TestClient(app)
    with open(csv_path, "rb") as f:
        files = {"file": (os.path.basename(csv_path), f, "text/csv")}
        r = client.post("/api/upload/", files=files)
    r.raise_for_status()
    sid = r.json()["session_id"]
    return client, sid


if __name__ == "__main__":
    client, sid = boot()
    print("session_id =", sid)
    info = client.get(f"/api/sessions/{sid}").json()
    print("columns =", [c["name"] for c in info.get("columns", [])])
    print("rows =", info.get("rows"))
