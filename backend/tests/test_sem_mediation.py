"""Tests for /api/causal/sem — SEM / path analysis endpoint."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture(scope="module")
def parallel_sid():
    """Two parallel mediators between X and two outcomes."""
    rng = np.random.default_rng(7)
    n = 400
    X = rng.normal(0, 1, n)
    C = rng.normal(0, 1, n)
    M1 = 0.6 * X + 0.2 * C + rng.normal(0, 1, n)
    M2 = 0.4 * X + 0.1 * C + rng.normal(0, 1, n)
    Y1 = 0.5 * M1 + 0.3 * M2 + 0.1 * X + 0.1 * C + rng.normal(0, 1, n)
    Y2 = 0.2 * M1 + 0.4 * M2 + 0.2 * C + rng.normal(0, 1, n)
    df = pd.DataFrame({"X": X, "M1": M1, "M2": M2, "Y1": Y1, "Y2": Y2, "C": C})
    return make_session(df, "sem_parallel")


@pytest.fixture(scope="module")
def serial_sid():
    """Serial chain: X -> M1 -> M2 -> Y."""
    rng = np.random.default_rng(11)
    n = 400
    X = rng.normal(0, 1, n)
    M1 = 0.7 * X + rng.normal(0, 1, n)
    M2 = 0.6 * M1 + rng.normal(0, 1, n)
    Y = 0.5 * M2 + 0.1 * X + rng.normal(0, 1, n)
    df = pd.DataFrame({"X": X, "M1": M1, "M2": M2, "Y": Y})
    return make_session(df, "sem_serial")


def test_parallel_mediation_multi_outcome(client, parallel_sid):
    r = client.post("/api/causal/sem", json={
        "session_id": parallel_sid,
        "treatments": ["X"], "mediators": ["M1", "M2"], "outcomes": ["Y1", "Y2"],
        "covariates": ["C"], "serial": False, "bootstrap": 200,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    # 1 treatment × 2 mediators × 2 outcomes = 4 indirect paths
    assert len(d["indirect_effects"]) == 4
    # 1 × 2 direct (X->Y1, X->Y2)
    assert len(d["direct_effects"]) == 2
    assert len(d["total_effects"]) == 2
    # X -> M1 -> Y1 should be largest and significant (true product = 0.6*0.5 = 0.30)
    by_label = {ie["label"]: ie for ie in d["indirect_effects"]}
    assert "X -> M1 -> Y1" in by_label
    ie = by_label["X -> M1 -> Y1"]
    assert ie["est"] is not None and 0.18 < ie["est"] < 0.45
    assert ie["significant"] is True
    # Fit indices present
    assert "fit" in d and "n" in d["fit"]
    assert d["fit"]["n"] == 400


def test_serial_chain(client, serial_sid):
    r = client.post("/api/causal/sem", json={
        "session_id": serial_sid,
        "treatments": ["X"], "mediators": ["M1", "M2"], "outcomes": ["Y"],
        "serial": True, "bootstrap": 200,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    # Serial = single chain X -> M1 -> M2 -> Y
    assert len(d["indirect_effects"]) == 1
    ie = d["indirect_effects"][0]
    assert ie["label"] == "X -> M1 -> M2 -> Y"
    assert ie["chain"] == ["M1", "M2"]
    # True product ~ 0.7*0.6*0.5 = 0.21
    assert ie["est"] is not None and 0.10 < ie["est"] < 0.35
    assert ie["significant"] is True
    assert d["serial"] is True


def test_bootstrap_reproducible(client, parallel_sid):
    payload = {
        "session_id": parallel_sid,
        "treatments": ["X"], "mediators": ["M1"], "outcomes": ["Y1"],
        "bootstrap": 200,
    }
    r1 = client.post("/api/causal/sem", json=payload)
    r2 = client.post("/api/causal/sem", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    ie1 = r1.json()["indirect_effects"][0]
    ie2 = r2.json()["indirect_effects"][0]
    # Same seed → identical bootstrap CIs
    assert ie1["boot_ci"] == ie2["boot_ci"]


def test_missing_treatment_400(client, parallel_sid):
    r = client.post("/api/causal/sem", json={
        "session_id": parallel_sid,
        "treatments": [], "mediators": ["M1"], "outcomes": ["Y1"],
        "bootstrap": 0,
    })
    assert r.status_code == 400, r.text


def test_role_overlap_400(client, parallel_sid):
    r = client.post("/api/causal/sem", json={
        "session_id": parallel_sid,
        "treatments": ["X"], "mediators": ["X"], "outcomes": ["Y1"],
        "bootstrap": 0,
    })
    assert r.status_code == 400, r.text


def test_unknown_column_400(client, parallel_sid):
    r = client.post("/api/causal/sem", json={
        "session_id": parallel_sid,
        "treatments": ["X"], "mediators": ["nope"], "outcomes": ["Y1"],
        "bootstrap": 0,
    })
    assert r.status_code == 400, r.text


def test_binary_outcome_422(client):
    rng = np.random.default_rng(3)
    n = 300
    X = rng.normal(0, 1, n)
    M = 0.5 * X + rng.normal(0, 1, n)
    Y = (rng.normal(0, 1, n) > 0).astype(int)
    sid = make_session(pd.DataFrame({"X": X, "M": M, "Yb": Y}), "sem_binary")
    r = client.post("/api/causal/sem", json={
        "session_id": sid,
        "treatments": ["X"], "mediators": ["M"], "outcomes": ["Yb"],
        "bootstrap": 0,
    })
    assert r.status_code == 422, r.text


def test_lavaan_override(client, parallel_sid):
    spec = "M1 ~ a*X\nY1 ~ b*M1 + cp*X"
    r = client.post("/api/causal/sem", json={
        "session_id": parallel_sid,
        "treatments": [], "mediators": [], "outcomes": [],
        "lavaan_spec": spec, "bootstrap": 0,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["lavaan_spec"] == spec
    # Indirect chains are NOT auto-derived in lavaan-spec mode
    assert d["indirect_effects"] == []
    # But paths table should include a and b
    labels = {p["label"] for p in d["paths"] if p["label"]}
    assert {"a", "b", "cp"}.issubset(labels)


def test_session_not_found(client):
    r = client.post("/api/causal/sem", json={
        "session_id": "does_not_exist",
        "treatments": ["X"], "mediators": ["M"], "outcomes": ["Y"],
        "bootstrap": 0,
    })
    assert r.status_code == 404
