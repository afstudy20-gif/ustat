"""Tests for the TOST equivalence endpoint (POST /api/stats/tost)."""
import numpy as np
import pandas as pd
from conftest import make_session


def _tost(client, sid, **kwargs):
    body = {"session_id": sid}
    body.update(kwargs)
    return client.post("/api/stats/tost", json=body)


def _assert_common_shape(d):
    assert "p_overall" in d and "equivalent" in d
    assert "t_low" in d and "p_low" in d
    assert "t_high" in d and "p_high" in d
    assert "difference" in d
    assert "low_bound" in d and "high_bound" in d
    assert d["low_bound"] < d["high_bound"]
    assert "group_labels" in d


# ── Independent samples ───────────────────────────────────────────────────────

def test_tost_independent_equivalent(client):
    np.random.seed(10)
    n = 100
    # Both groups drawn from same distribution -> clearly equivalent within wide margin
    g1 = np.random.normal(loc=50.0, scale=2.0, size=n)
    g2 = np.random.normal(loc=50.2, scale=2.0, size=n)
    df = pd.DataFrame({
        "value": np.concatenate([g1, g2]),
        "group": ["A"] * n + ["B"] * n,
    })
    sid = make_session(df, "tost_ind_equiv")
    r = _tost(client, sid, column="value", group_column="group", low=-5.0, high=5.0, test_type="independent")
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["p_overall"] < 0.05
    assert d["equivalent"] is True
    assert set(d["group_labels"]) == {"A", "B"}


def test_tost_independent_not_equivalent(client):
    np.random.seed(11)
    n = 30
    # Groups clearly far apart relative to a tight margin -> not equivalent
    g1 = np.random.normal(loc=0.0, scale=1.0, size=n)
    g2 = np.random.normal(loc=20.0, scale=1.0, size=n)
    df = pd.DataFrame({
        "value": np.concatenate([g1, g2]),
        "group": ["A"] * n + ["B"] * n,
    })
    sid = make_session(df, "tost_ind_not_equiv")
    r = _tost(client, sid, column="value", group_column="group", low=-1.0, high=1.0, test_type="independent")
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["p_overall"] >= 0.05
    assert d["equivalent"] is False


def test_tost_independent_missing_group_col(client):
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    sid = make_session(df, "tost_ind_missing_group")
    r = _tost(client, sid, column="value", low=-5.0, high=5.0, test_type="independent")
    assert r.status_code == 422, r.text


def test_tost_low_gte_high_rejected(client):
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "group": ["A", "A", "B", "B"]})
    sid = make_session(df, "tost_low_gte_high")
    r = _tost(client, sid, column="value", group_column="group", low=5.0, high=-5.0, test_type="independent")
    assert r.status_code == 422, r.text


def test_tost_independent_wrong_group_count(client):
    df = pd.DataFrame({
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "group": ["A", "B", "C", "A", "B", "C"],
    })
    sid = make_session(df, "tost_ind_3groups")
    r = _tost(client, sid, column="value", group_column="group", low=-5.0, high=5.0, test_type="independent")
    assert r.status_code == 422, r.text


# ── Paired ────────────────────────────────────────────────────────────────────

def test_tost_paired_equivalent(client):
    np.random.seed(12)
    n = 100
    before = np.random.normal(loc=50.0, scale=2.0, size=n)
    after = before + np.random.normal(loc=0.0, scale=0.2, size=n)  # tiny paired difference
    df = pd.DataFrame({"before": before, "after": after})
    sid = make_session(df, "tost_paired_equiv")
    r = _tost(client, sid, column="before", paired_column="after", low=-2.0, high=2.0, test_type="paired")
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["p_overall"] < 0.05
    assert d["equivalent"] is True


def test_tost_paired_not_equivalent(client):
    np.random.seed(13)
    n = 30
    before = np.random.normal(loc=0.0, scale=1.0, size=n)
    after = before + 10.0 + np.random.normal(loc=0.0, scale=0.5, size=n)
    df = pd.DataFrame({"before": before, "after": after})
    sid = make_session(df, "tost_paired_not_equiv")
    r = _tost(client, sid, column="before", paired_column="after", low=-1.0, high=1.0, test_type="paired")
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["p_overall"] >= 0.05
    assert d["equivalent"] is False


def test_tost_paired_missing_paired_column(client):
    df = pd.DataFrame({"before": [1.0, 2.0, 3.0, 4.0], "after": [1.1, 2.1, 3.1, 4.1]})
    sid = make_session(df, "tost_paired_missing")
    r = _tost(client, sid, column="before", low=-5.0, high=5.0, test_type="paired")
    assert r.status_code == 422, r.text


# ── One-sample ────────────────────────────────────────────────────────────────

def test_tost_one_sample_equivalent(client):
    np.random.seed(14)
    n = 100
    x = np.random.normal(loc=0.05, scale=1.0, size=n)  # mean very close to mu=0
    df = pd.DataFrame({"value": x})
    sid = make_session(df, "tost_one_sample_equiv")
    r = _tost(client, sid, column="value", mu=0.0, low=-2.0, high=2.0, test_type="one_sample")
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["p_overall"] < 0.05
    assert d["equivalent"] is True


def test_tost_one_sample_not_equivalent(client):
    np.random.seed(15)
    n = 30
    x = np.random.normal(loc=10.0, scale=1.0, size=n)  # mean far from mu=0
    df = pd.DataFrame({"value": x})
    sid = make_session(df, "tost_one_sample_not_equiv")
    r = _tost(client, sid, column="value", mu=0.0, low=-1.0, high=1.0, test_type="one_sample")
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["p_overall"] >= 0.05
    assert d["equivalent"] is False


def test_tost_unknown_test_type(client):
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    sid = make_session(df, "tost_unknown_type")
    r = _tost(client, sid, column="value", low=-5.0, high=5.0, test_type="bogus")
    assert r.status_code == 422, r.text
