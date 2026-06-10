"""Tests for repeated-measures endpoints: paired t, Wilcoxon SR, Friedman, RM ANOVA, mixed ANOVA."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp
from conftest import make_session


# ── Paired t-test ────────────────────────────────────────────────────────────

def test_paired_ttest_known_answer(client):
    before = [85, 90, 78, 92, 88, 76, 95, 82, 89, 91]
    after  = [90, 95, 80, 96, 92, 80, 99, 86, 93, 95]
    df = pd.DataFrame({"before": before, "after": after})
    sid = make_session(df, "pt1")
    r = client.post("/api/repeated/paired_ttest", json={"session_id": sid, "col1": "before", "col2": "after"})
    assert r.status_code == 200
    d = r.json()
    # Verify against scipy
    t_expected, p_expected = sp.ttest_rel(before, after)
    assert abs(d["t"] - t_expected) < 0.01
    assert abs(d["p"] - p_expected) < 0.001
    assert d["significant"] is True
    assert len(d["effect_sizes"]) == 1
    assert d["effect_sizes"][0]["name"] == "cohen_d_z"
    assert "paired" in d["result_text"].lower()
    assert "r_code" in d
    assert "paired = TRUE" in d["r_code"]


def test_paired_ttest_no_difference(client):
    vals = [10, 20, 30, 40, 50]
    df = pd.DataFrame({"a": vals, "b": vals})
    sid = make_session(df, "pt2")
    r = client.post("/api/repeated/paired_ttest", json={"session_id": sid, "col1": "a", "col2": "b"})
    assert r.status_code == 200
    d = r.json()
    assert d["significant"] is False
    assert d["p"] >= 0.99


# ── Wilcoxon signed-rank ─────────────────────────────────────────────────────

def test_wilcoxon_sr_known(client):
    before = [8, 7, 6, 9, 5, 7, 8, 6, 9, 7]
    after  = [9, 8, 7, 10, 6, 8, 9, 7, 10, 8]
    df = pd.DataFrame({"before": before, "after": after})
    sid = make_session(df, "wsr1")
    r = client.post("/api/repeated/wilcoxon_signed_rank", json={"session_id": sid, "col1": "before", "col2": "after"})
    assert r.status_code == 200
    d = r.json()
    w_exp, p_exp = sp.wilcoxon(before, after)
    assert abs(d["W"] - w_exp) < 0.5
    assert d["effect_sizes"][0]["name"] == "rank_biserial_r"
    assert "r_code" in d


def test_wilcoxon_sr_ties(client):
    df = pd.DataFrame({"a": [5, 5, 6, 6, 7, 7, 8, 8, 9, 9], "b": [6, 5, 7, 6, 8, 7, 9, 8, 10, 9]})
    sid = make_session(df, "wsr2")
    r = client.post("/api/repeated/wilcoxon_signed_rank", json={"session_id": sid, "col1": "a", "col2": "b"})
    assert r.status_code == 200


# ── Friedman ─────────────────────────────────────────────────────────────────

def test_friedman_known(client):
    np.random.seed(42)
    n = 15
    df = pd.DataFrame({
        "time1": np.random.normal(10, 2, n),
        "time2": np.random.normal(12, 2, n),
        "time3": np.random.normal(14, 2, n),
    })
    sid = make_session(df, "fr1")
    r = client.post("/api/repeated/friedman", json={"session_id": sid, "columns": ["time1", "time2", "time3"]})
    assert r.status_code == 200
    d = r.json()
    chi_exp, p_exp = sp.friedmanchisquare(df["time1"], df["time2"], df["time3"])
    assert abs(d["chi2"] - chi_exp) < 0.1
    assert d["effect_sizes"][0]["name"] == "kendalls_w"
    if d["significant"]:
        assert len(d["posthoc"]) > 0
    assert "r_code" in d


# ── RM ANOVA ─────────────────────────────────────────────────────────────────

def test_rm_anova_known(client):
    np.random.seed(42)
    subjects = list(range(10)) * 3
    timepoints = ["T1"] * 10 + ["T2"] * 10 + ["T3"] * 10
    values = list(np.random.normal(10, 2, 10)) + list(np.random.normal(12, 2, 10)) + list(np.random.normal(15, 2, 10))
    df = pd.DataFrame({"subject": subjects, "time": timepoints, "score": values})
    sid = make_session(df, "rma1")
    r = client.post("/api/repeated/rm_anova", json={
        "session_id": sid, "subject_col": "subject", "within_col": "time", "value_col": "score"
    })
    assert r.status_code == 200
    d = r.json()
    assert "F" in d
    assert d["effect_sizes"][0]["name"] == "partial_eta_squared"
    assert "r_code" in d
    assert "ezANOVA" in d["r_code"]


# ── Mixed ANOVA ──────────────────────────────────────────────────────────────

def test_mixed_anova_known(client):
    np.random.seed(42)
    data = []
    for subj in range(12):
        group = "A" if subj < 6 else "B"
        for t in ["pre", "post"]:
            val = np.random.normal(10 if t == "pre" else (14 if group == "A" else 11), 2)
            data.append({"subject": subj, "group": group, "time": t, "score": val})
    df = pd.DataFrame(data)
    sid = make_session(df, "ma1")
    r = client.post("/api/repeated/mixed_anova", json={
        "session_id": sid, "subject_col": "subject", "within_col": "time",
        "between_col": "group", "value_col": "score"
    })
    assert r.status_code == 200
    d = r.json()
    assert "effects" in d
    assert len(d["effects"]) >= 2
    assert "r_code" in d


# ── Contract checks ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("endpoint,payload", [
    ("/api/repeated/paired_ttest", {"col1": "a", "col2": "b"}),
    ("/api/repeated/wilcoxon_signed_rank", {"col1": "a", "col2": "b"}),
])
def test_result_contract_paired(client, endpoint, payload):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
    sid = make_session(df, f"contract_{endpoint.split('/')[-1]}")
    r = client.post(endpoint, json={"session_id": sid, **payload})
    assert r.status_code == 200
    d = r.json()
    assert isinstance(d.get("result_text", ""), str) and len(d["result_text"]) > 10
    assert isinstance(d.get("effect_sizes"), list) and len(d["effect_sizes"]) > 0
    assert isinstance(d.get("export_rows"), list) and len(d["export_rows"]) > 1
    assert "r_code" in d
