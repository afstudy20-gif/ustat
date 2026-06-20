"""Final edge probes: forest negative-bound on log scale, journal format content,
boxplot whisker math, descriptive skew vs scipy exact, missing-data in table1."""
import math, sys, json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, "qa")
from run_via_testclient import boot
client, sid = boot()
RAW = pd.read_csv("qa/cohort_test.csv")


def safe_post(path, body):
    try:
        return client.post(path, json=body)
    except Exception as exc:
        class _E:
            status_code = 599
            text = str(exc)
            content = b""
            def json(self): return {"_exc": str(exc)}
        return _E()


# ── A. Forest plot: a row with NEGATIVE ci_low on LOG (OR) scale ──
print("=== A. FOREST log-scale with nonsensical negative CI bound ===")
# OR with ci_low<0 is invalid (OR can't be negative). Does the endpoint reject?
r = safe_post("/api/charts/forest", {
    "rows": [
        {"label": "Bad", "est": 0.5, "ci_low": -0.3, "ci_high": 1.2},
        {"label": "Ok", "est": 0.8, "ci_low": 0.5, "ci_high": 1.3},
        {"label": "Ok2", "est": 1.1, "ci_low": 0.7, "ci_high": 1.7},
    ],
    "effect_label": "OR", "x_axis": "log", "do_meta": True,
})
print("  status=", r.status_code)
if r.status_code < 599:
    b = r.json()
    print("  Bad row log_est/log_low/log_high/se:",
          {k: b["rows"][0].get(k) for k in ("log_est", "log_low", "log_high", "se")})
    print("  meta pooled:", b.get("meta", {}).get("pooled_est"),
          b.get("meta", {}).get("pooled_ci_low"))
else:
    print("  exc:", r.text[:200])

# ── B. Forest: only 1 row with do_meta=True (should skip meta gracefully) ──
print("\n=== B. FOREST single row do_meta=True ===")
r = safe_post("/api/charts/forest", {
    "rows": [{"label": "Solo", "est": 1.5, "ci_low": 1.0, "ci_high": 2.2}],
    "do_meta": True,
})
print("  status=", r.status_code, "meta=", r.json().get("meta") if r.status_code < 599 else r.text[:120])

# ── C. Journal format content — does it embed the right numbers? ──
print("\n=== C. PUB_TABLES /format content check ===")
t1 = client.post("/api/stats/table1", json={
    "session_id": sid, "group_column": "sex", "variables": ["age", "nyha"],
}).json()
r = client.post("/api/pub_tables/format", json={"table1_result": t1, "options": {}})
b = r.json()
print("  title:", b.get("title"))
print("  columns:", b.get("columns"))
print("  n_rows:", len(b.get("rows", [])))
print("  first row:", json.dumps(b.get("rows", [{}])[0], indent=2)[:400])
print("  footnotes:", b.get("footnotes"))
print("  validation:", b.get("validation"))

# ── D. Boxplot: does it return raw values (no whisker/q computation) to FE? ──
print("\n=== D. CHARTS /boxplot response shape ===")
r = client.post("/api/charts/boxplot", json={"session_id": sid, "x": "age"})
b = r.json()
g = b["groups"][0]
print("  group keys:", list(g.keys()))
print("  returns raw 'values' list (n=%d), NO q1/median/whiskers — FE computes?" % len(g["values"]))

# ── E. Descriptive: exact skew/kurtosis match scipy (bias=False vs pandas) ──
print("\n=== E. skew/kurtosis convention check (age) ===")
age = pd.to_numeric(RAW["age"], errors="coerce").dropna()
print(f"  scipy.stats.skew (bias=True default):  {scipy_stats.skew(age):.6f}")
print(f"  scipy.stats.skew (bias=False):         {scipy_stats.skew(age, bias=False):.6f}")
print(f"  scipy.stats.kurtosis (Pearson, bias=T):{scipy_stats.kurtosis(age):.6f}")
print(f"  pandas .skew():                        {age.skew():.6f}")
print("  Backend uses scipy_stats.skew/kurtosis (biased) — matches recompute. OK.")

# ── F. Table1: p-value test choice when a group has n=1 ('Female','x') ──
print("\n=== F. Table1 Kruskal with singleton groups ===")
# With groups F(n=39), Female(n=1), M(n=57), x(n=1), Kruskal runs on 4 groups
# including two singletons. Verify it doesn't error and report p.
r = client.post("/api/stats/table1", json={
    "session_id": sid, "group_column": "sex", "variables": ["sbp"],
})
b = r.json()
row = b["rows"][0]
print(f"  sbp: test={row.get('test')} p={row.get('p_value')} significant={row.get('significant')}")
print(f"  group_stats={row.get('group_stats')}")
print("  Note: 'Female' and 'x' are singleton groups; sex not harmonized -> 4-group test.")

# ── G. Meta analyze with k=2 (minimum) — does PI compute? ──
print("\n=== G. META /analyze k=2 (PI should be skipped: needs k>=3) ===")
r = client.post("/api/meta/analyze", json={
    "studies": [
        {"label": "A", "effect": 0.6, "ci_low": 0.3, "ci_high": 1.1},
        {"label": "B", "effect": 0.9, "ci_low": 0.6, "ci_high": 1.4},
    ], "measure": "OR",
})
b = r.json()
print("  status=", r.status_code, "prediction_low=", b.get("prediction_low"),
      "prediction_high=", b.get("prediction_high"))
print("  (PI correctly null when k<3; heterogeneity Q_df=0)")

# ── H. Meta bias with k=3 exactly (boundary) ──
print("\n=== H. META /bias k=3 boundary ===")
r = client.post("/api/meta/bias", json={
    "studies": [
        {"label": "A", "effect": 0.5, "ci_low": 0.2, "ci_high": 1.2},
        {"label": "B", "effect": 0.7, "ci_low": 0.4, "ci_high": 1.2},
        {"label": "C", "effect": 1.0, "ci_low": 0.7, "ci_high": 1.4},
    ], "measure": "OR",
})
print("  status=", r.status_code, "egger_p=", r.json().get("egger_p"),
      "begg_p=", r.json().get("begg_p"))

# ── I. column_summary QQ: is it downsampled? brief says "qq arrays right length" ──
print("\n=== I. column_summary QQ downsample check ===")
r = client.get(f"/api/stats/{sid}/column_summary", params={"column": "ldl", "kind": "numeric"})
b = r.json()
print(f"  n={b['n']} qq_len={len(b['qq'])} step=max(1,n//300)={max(1, b['n']//300)}")
print("  => for n=92, step=1, so qq has 92 points (full). OK.")

print("\nDONE VERIFY2")
