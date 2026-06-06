"""
One exposure's adjusted HR across multiple model specifications — the
'sensitivity analyses' forest (Figure 6).
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _seed() -> str:
    rng = np.random.default_rng(5)
    n = 320
    age = rng.normal(60, 10, n)
    sex = rng.integers(0, 2, n)
    ef = rng.normal(40, 10, n)
    ldl_low = rng.integers(0, 2, n)  # binary exposure
    base = 0.03 * (age - 60) + 0.2 * ldl_low
    time = rng.exponential(scale=np.exp(-base) * 500).clip(1, 1800)
    event = (time < 1200).astype(int)
    df = pd.DataFrame({"time": time, "event": event, "age": age, "sex": sex,
                       "ef": ef, "ldl_low": ldl_low})
    sid = "cox_specs"
    store.save(sid, df)
    return sid


def test_model_specs_shape():
    sid = _seed()
    r = client.post("/api/models/survival/cox_model_specs", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "exposure": "ldl_low",
        "specs": [
            {"label": "Parsimonious", "covariates": ["age", "sex"]},
            {"label": "Alternative", "covariates": ["age", "ef"]},
            {"label": "Full", "covariates": ["age", "sex", "ef"]},
        ],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    labels = [s["label"] for s in d["specs"]]
    assert labels[0] == "Unadjusted"  # prepended
    assert "Parsimonious" in labels and "Full" in labels
    for s in d["specs"]:
        assert s["n"] > 0 and s["n_events"] > 0
        # binary exposure -> one term per spec
        assert len(s["terms"]) == 1
        t = s["terms"][0]
        assert t["term"] == "ldl_low"
        assert "hr" in t


def test_model_specs_rejects_bad_exposure():
    sid = _seed()
    r = client.post("/api/models/survival/cox_model_specs", json={
        "session_id": sid, "duration_col": "time", "event_col": "event",
        "exposure": "missing_col", "specs": [{"label": "m", "covariates": []}],
    })
    assert r.status_code == 422
