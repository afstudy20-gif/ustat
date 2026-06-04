import pandas as pd
from fastapi.testclient import TestClient
from main import app
from services import store

client = TestClient(app)

def test_rename_session_persists():
    df = pd.DataFrame({"a": [1, 2, 3]})
    store.save("rename_sid", df)
    r = client.post("/api/sessions/rename_sid/rename", json={"filename": "my_study.json"})
    assert r.status_code == 200, r.text
    assert r.json()["filename"] == "my_study.json"
    # save_session should now emit the renamed filename
    r2 = client.get("/api/sessions/rename_sid/save_session")
    assert r2.status_code == 200
    import json
    payload = json.loads(r2.content)
    assert payload["filename"] == "my_study.json"

def test_rename_empty_rejected():
    df = pd.DataFrame({"a": [1, 2, 3]})
    store.save("rename_sid2", df)
    r = client.post("/api/sessions/rename_sid2/rename", json={"filename": "   "})
    assert r.status_code == 422

if __name__ == "__main__":
    test_rename_session_persists()
    test_rename_empty_rejected()
    print("OK")
