"""Probe script for the Grok-composer slice (Visual/Reporting/Table1/Summary/Meta/TimeSeries).

Exercises every endpoint in scope against qa/cohort_test.csv and prints findings.
Does NOT touch production code.
"""
from __future__ import annotations
import json
import sys
import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, "qa")
from run_via_testclient import boot  # noqa: E402

client, sid = boot()

print("=== SESSION ===")
print("session_id =", sid)


def post(path, body):
    return client.post(path, json=body)


def get(path, **params):
    try:
        return client.get(path, params=params)
    except Exception as exc:
        class _Err:
            status_code = 599
            text = str(exc)
            content = b""

            def json(self):
                return {"_client_exception": str(exc)}
        return _Err()


def safe_post(path, body):
    """Wrap a POST so a 500/exception in the app is captured, not fatal."""
    try:
        return client.post(path, json=body)
    except Exception as exc:  # TestClient re-raises app exceptions
        class _Err:
            status_code = 599
            text = str(exc)
            content = b""

            def json(self):
                return {"_client_exception": str(exc)}
        return _Err()


# ════════════════════════════════════════════════════════════════════════════
# 1. SUMMARY / DESCRIPTIVE
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 1. DESCRIPTIVE (per column) ===")

# Load the raw CSV independently for recomputation
RAW = pd.read_csv("qa/cohort_test.csv")

for col in ["age", "bmi", "ldl", "sbp", "fu_days"]:
    r = get(f"/api/stats/{sid}/descriptive", column=col)
    print(f"\n--- {col} --- status={r.status_code}")
    body = r.json()
    if r.status_code != 200:
        print("  ERROR BODY:", body)
        continue
    # Body is {col: {stats}} when column param is set
    stats = body.get(col, body)
    print("  n=", stats.get("n"), "mean=", stats.get("mean"), "min=", stats.get("min"),
          "max=", stats.get("max"), "skew=", stats.get("skewness"), "kurt=",
          stats.get("kurtosis"), "test=", stats.get("normality_test"),
          "p=", stats.get("normality_p"))
    # Recompute against what the backend saw (numeric coercion)
    s = pd.to_numeric(RAW[col], errors="coerce").dropna()
    print(f"  [recompute as-coerced] n={len(s)} mean={s.mean():.4f} "
          f"min={s.min()} max={s.max()} skew={scipy_stats.skew(s):.4f} "
          f"kurt={scipy_stats.kurtosis(s):.4f}")


# ════════════════════════════════════════════════════════════════════════════
# 2. COLUMN SUMMARY (QQ + outliers + histogram)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. COLUMN_SUMMARY (QQ + outliers) ===")
for col in ["age", "fu_days", "ldl"]:
    r = get(f"/api/stats/{sid}/column_summary", column=col, kind="numeric")
    b = r.json()
    print(f"\n--- {col} --- status={r.status_code}")
    if r.status_code != 200:
        print("  ERR", b)
        continue
    print("  n=", b.get("n"), "missing=", b.get("missing"),
          "whisker_low=", b.get("whisker_low"), "whisker_high=", b.get("whisker_high"))
    outs = b.get("outliers", [])
    print("  outliers count=", len(outs), "sample=", outs[:5])
    print("  qq length=", len(b.get("qq", [])), "expected ~", min(300, b.get("n", 0)))
    hist = b.get("histogram", [])
    print("  histogram bins=", len(hist))
    if hist:
        print("    first bin=", hist[0], "last bin=", hist[-1])


# ════════════════════════════════════════════════════════════════════════════
# 3. CHARTS — histogram, boxplot, subgroup bar
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 3a. CHARTS /histogram ===")
for col in ["age", "fu_days"]:
    r = post("/api/charts/histogram", {"session_id": sid, "x": col, "bins": 10})
    b = r.json()
    print(f"  {col}: status={r.status_code} bins={len(b.get('bins', []))} "
          f"kde_pts={len(b.get('kde', []))}")
    if b.get("bins"):
        edges = [(bin_["x0"], bin_["x1"]) for bin_ in b["bins"]]
        print(f"    bin edges: x0={edges[0][0]} ... x1_last={edges[-1][1]}")
        print(f"    stats min/max in kde domain:", b.get("stats"))

print("\n=== 3b. CHARTS /boxplot (by sex) ===")
r = post("/api/charts/boxplot", {"session_id": sid, "x": "age", "color": "sex"})
b = r.json()
print("  status=", r.status_code, "groups=", [g["group"] for g in b.get("groups", [])])
for g in b.get("groups", []):
    print(f"    {g['group']}: n_values={len(g['values'])} min={min(g['values'])} max={max(g['values'])}")

print("\n=== 3c. CHARTS /subgroup_bar (event prop by nyha, subgroup==xaxis) ===")
r = safe_post("/api/charts/subgroup_bar", {
    "session_id": sid, "y_col": "event", "subgroup_col": "nyha",
    "xaxis_col": "nyha", "y_mode": "percentage", "error_type": "ci",
})
b = r.json() if r.status_code != 599 else {}
print("  status=", r.status_code)
if r.status_code >= 500:
    print("  EXCEPTION:", r.text[:300])
for tr in b.get("traces", []):
    print(f"    trace={tr['name']} xaxis={tr['x_xaxis']} y={tr['y']} ns={tr['ns']}")
print("  target_value=", b.get("target_value"))

# subgroup_bar mean mode (mean sbp by nyha)
print("\n=== 3d. CHARTS /subgroup_bar (mean sbp by nyha, subgroup==xaxis) ===")
r = safe_post("/api/charts/subgroup_bar", {
    "session_id": sid, "y_col": "sbp", "subgroup_col": "nyha",
    "xaxis_col": "nyha", "y_mode": "mean", "error_type": "ci",
})
b = r.json() if r.status_code != 599 else {}
print("  status=", r.status_code)
if r.status_code >= 500:
    print("  EXCEPTION:", r.text[:300])
for tr in b.get("traces", []):
    print(f"    trace={tr['name']} xaxis={tr['x_xaxis']} y={tr['y']} ns={tr['ns']}")

# Now the realistic case: event prop by nyha with subgroup != xaxis
print("\n=== 3e. CHARTS /subgroup_bar (event prop, subgroup=diabetes, xaxis=nyha) ===")
r = safe_post("/api/charts/subgroup_bar", {
    "session_id": sid, "y_col": "event", "subgroup_col": "diabetes",
    "xaxis_col": "nyha", "y_mode": "percentage", "error_type": "ci",
})
b = r.json() if r.status_code != 599 else {}
print("  status=", r.status_code)
if r.status_code >= 500:
    print("  EXCEPTION:", r.text[:300])
for tr in b.get("traces", []):
    print(f"    trace={tr['name']} xaxis={tr['x_xaxis']} y={[round(v,1) for v in tr['y']]} ns={tr['ns']}")
print("  method_note=", (b.get("method_note") or "")[:120])


# ════════════════════════════════════════════════════════════════════════════
# 4. TABLE 1 by sex (mixed coding)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 4. TABLE 1 by sex ===")
r = post("/api/stats/table1", {
    "session_id": sid, "group_column": "sex",
    "variables": ["age", "bmi", "sbp", "diabetes", "nyha"],
})
b = r.json()
print("  status=", r.status_code)
print("  group_labels=", b.get("group_labels"))
print("  group_ns=", b.get("group_ns"))
print("  total_n=", b.get("total_n"))
for row in b.get("rows", []):
    print(f"  var={row['variable']} type={row['type']} "
          f"stat_label={row.get('stat_label')} overall={row.get('overall')}")
    if row["type"] == "categorical":
        for sr in row.get("sub_rows", []):
            print(f"      cat={sr['category']} overall={sr['overall']}")
    if row.get("group_stats"):
        print(f"      group_stats={row['group_stats']}")
    print(f"      test={row.get('test')} p={row.get('p_value')} significant={row.get('significant')}")


# ════════════════════════════════════════════════════════════════════════════
# 5. WEIGHTED DESCRIPTIVE
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 5. WEIGHTED DESCRIPTIVE ===")
# Use sbp as a fake weight to exercise the path
r = post("/api/stats/weighted_descriptive", {
    "session_id": sid, "value_cols": ["age", "ldl"],
    "weight_col": "sbp", "group_col": "sex",
})
b = r.json()
print("  status=", r.status_code)
print("  n=", b.get("n"))
for res in b.get("results", []):
    print(f"    col={res.get('column')} w_mean={res.get('w_mean')} "
          f"w_sd={res.get('w_sd')} ess={res.get('ess_kish')}")
print("  comparison=", b.get("comparison"))


# ════════════════════════════════════════════════════════════════════════════
# 6. FOREST PLOT (with meta pool)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 6a. FOREST plot (OR, log scale, one negative-ish bound) ===")
r = post("/api/charts/forest", {
    "rows": [
        {"label": "Study A", "est": 0.45, "ci_low": 0.20, "ci_high": 1.00},
        {"label": "Study B", "est": 0.80, "ci_low": 0.50, "ci_high": 1.30},
        {"label": "Study C", "est": 1.20, "ci_low": 0.90, "ci_high": 1.70},
    ],
    "effect_label": "OR", "x_axis": "log", "null_line": 1.0,
    "do_meta": True, "meta_method": "DL",
})
b = r.json()
print("  status=", r.status_code)
meta = b.get("meta")
print("  meta=", json.dumps(meta, indent=2) if meta else None)

# Forest with a NEGATIVE lower bound on linear scale (mean diff) — does it complain?
print("\n=== 6b. FOREST plot (linear, negative bound) ===")
r = post("/api/charts/forest", {
    "rows": [
        {"label": "A", "est": -2.5, "ci_low": -5.0, "ci_high": 0.0},
        {"label": "B", "est": 1.0, "ci_low": -1.0, "ci_high": 3.0},
        {"label": "C", "est": 2.0, "ci_low": 0.5, "ci_high": 3.5},
    ],
    "effect_label": "Mean diff", "x_axis": "linear", "null_line": 0.0,
    "do_meta": True,
})
b = r.json()
print("  status=", r.status_code)
print("  meta=", json.dumps(b.get("meta"), indent=2))
print("  row SEs:", [(row["label"], row.get("se")) for row in b.get("rows", [])])


# ════════════════════════════════════════════════════════════════════════════
# 7. META — analyze / subgroup / regression / bias
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 7a. META /analyze (5 synthetic studies, OR) ===")
studies = [
    {"label": "S1", "effect": 0.50, "ci_low": 0.30, "ci_high": 0.85},
    {"label": "S2", "effect": 0.65, "ci_low": 0.40, "ci_high": 1.05},
    {"label": "S3", "effect": 0.80, "ci_low": 0.55, "ci_high": 1.15},
    {"label": "S4", "effect": 0.45, "ci_low": 0.25, "ci_high": 0.80},
    {"label": "S5", "effect": 1.10, "ci_low": 0.70, "ci_high": 1.70},
]
r = post("/api/meta/analyze", {"studies": studies, "measure": "OR", "tau2_method": "DL"})
b = r.json()
print("  status=", r.status_code)
print("  fixed=", b.get("fixed"))
print("  random=", b.get("random"))
print("  Q=", b.get("Q"), "Q_p=", b.get("Q_p"), "I2=", b.get("I2_pct"), "H2=", b.get("H2"))
print("  tau2=", b.get("tau2"))
print("  pred=", b.get("prediction_low"), b.get("prediction_high"))

# Recompute with statsmodels-style manual DL to verify
print("\n  [recompute DL by hand]")
y = np.array([math.log(s["effect"]) for s in studies])
selo = np.array([(math.log(s["ci_high"]) - math.log(s["ci_low"])) / (2 * 1.959963984540054) for s in studies])
v = selo ** 2
w = 1 / v
mu_fe = np.sum(w * y) / np.sum(w)
q = np.sum(w * (y - mu_fe) ** 2)
df = len(y) - 1
c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
tau2 = max(0.0, (q - df) / c)
wr = 1 / (v + tau2)
mu_re = np.sum(wr * y) / np.sum(wr)
se_re = math.sqrt(1 / np.sum(wr))
i2 = max(0.0, (q - df) / q * 100.0)
print(f"    mu_fe(log)={mu_fe:.4f} -> OR={math.exp(mu_fe):.4f}")
print(f"    mu_re(log)={mu_re:.4f} -> OR={math.exp(mu_re):.4f} CI "
      f"[{math.exp(mu_re-1.96*se_re):.4f}, {math.exp(mu_re+1.96*se_re):.4f}]")
print(f"    Q={q:.4f} I2={i2:.2f} tau2={tau2:.6f}")

print("\n=== 7b. META /subgroup ===")
r = post("/api/meta/subgroup", {
    "studies": [
        {"label": "S1", "effect": 0.50, "ci_low": 0.30, "ci_high": 0.85, "subgroup": "Adult"},
        {"label": "S2", "effect": 0.65, "ci_low": 0.40, "ci_high": 1.05, "subgroup": "Adult"},
        {"label": "S3", "effect": 1.20, "ci_low": 0.80, "ci_high": 1.80, "subgroup": "Pediatric"},
        {"label": "S4", "effect": 1.40, "ci_low": 0.90, "ci_high": 2.20, "subgroup": "Pediatric"},
        {"label": "S5", "effect": 1.10, "ci_low": 0.70, "ci_high": 1.70, "subgroup": "Pediatric"},
    ],
    "measure": "OR",
})
b = r.json()
print("  status=", r.status_code)
print("  subgroups=", json.dumps(b.get("subgroups"), indent=2))
print("  q_between=", b.get("q_between"), "p=", b.get("q_between_p"))

print("\n=== 7c. META /regression (with moderator) ===")
r = post("/api/meta/regression", {
    "studies": [
        {"label": "S1", "effect": 0.50, "ci_low": 0.30, "ci_high": 0.85, "moderator": 45},
        {"label": "S2", "effect": 0.65, "ci_low": 0.40, "ci_high": 1.05, "moderator": 50},
        {"label": "S3", "effect": 0.80, "ci_low": 0.55, "ci_high": 1.15, "moderator": 55},
        {"label": "S4", "effect": 0.95, "ci_low": 0.65, "ci_high": 1.40, "moderator": 60},
        {"label": "S5", "effect": 1.10, "ci_low": 0.70, "ci_high": 1.70, "moderator": 65},
    ],
    "measure": "OR",
})
b = r.json()
print("  status=", r.status_code)
print("  slope=", b.get("slope"), "slope_p=", b.get("slope_p"),
      "ci=", b.get("slope_ci_low"), b.get("slope_ci_high"))
print("  r2_pct=", b.get("r2_pct"), "tau2=", b.get("tau2"), "tau2_resid=", b.get("tau2_resid"))
print("  n_points=", len(b.get("points", [])), "line_y=", b.get("line_y"))

print("\n=== 7d. META /bias (funnel-asymmetric) ===")
# Deliberately asymmetric: small studies all show big effects
r = post("/api/meta/bias", {
    "studies": [
        {"label": "S1", "effect": 0.30, "ci_low": 0.10, "ci_high": 0.90},   # tiny, very imprecise, big effect
        {"label": "S2", "effect": 0.35, "ci_low": 0.15, "ci_high": 0.85},
        {"label": "S3", "effect": 0.80, "ci_low": 0.60, "ci_high": 1.05},
        {"label": "S4", "effect": 0.85, "ci_low": 0.70, "ci_high": 1.05},
        {"label": "S5", "effect": 0.90, "ci_low": 0.78, "ci_high": 1.05},
        {"label": "S6", "effect": 0.95, "ci_low": 0.85, "ci_high": 1.07},
        {"label": "S7", "effect": 1.00, "ci_low": 0.92, "ci_high": 1.08},
    ],
    "measure": "OR",
})
b = r.json()
print("  status=", r.status_code)
print("  egger_intercept=", b.get("egger_intercept"), "egger_p=", b.get("egger_p"))
print("  begg_tau=", b.get("begg_tau"), "begg_p=", b.get("begg_p"))
print("  trim_fill_missing=", b.get("trim_fill_missing"))
print("  funnel pts=", len(b.get("funnel", [])))


# ════════════════════════════════════════════════════════════════════════════
# 8. TIME SERIES
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 8a. TIMESERIES /stationarity on age (ordered by patient_id) ===")
r = post("/api/timeseries/stationarity", {
    "session_id": sid, "value_col": "age", "time_col": "patient_id", "n_lags": 20,
})
b = r.json()
print("  status=", r.status_code)
print("  n=", b.get("n"), "adf_p=", b.get("adf_p"), "kpss_p=", b.get("kpss_p"))
print("  adf_stationary=", b.get("adf_stationary"), "kpss_stationary=", b.get("kpss_stationary"))
print("  acf length=", len(b.get("acf", [])), "pacf length=", len(b.get("pacf", [])))
print("  interpretation=", b.get("interpretation"))

print("\n=== 8b. TIMESERIES /decompose (STL) on age ===")
r = post("/api/timeseries/decompose", {
    "session_id": sid, "value_col": "age", "time_col": "patient_id",
    "period": 7, "method": "stl",
})
b = r.json()
print("  status=", r.status_code)
if r.status_code == 200:
    print("  n=", b.get("n"), "strength_trend=", b.get("strength_trend"),
          "strength_seasonal=", b.get("strength_seasonal"))
    seas = b.get("seasonal", [])
    obs = b.get("observed", [])
    print("  seasonal len=", len(seas), "first 3=", seas[:3])
    print("  seasonal range=", min(seas), max(seas), "observed range=", min(obs), max(obs))
else:
    print("  ERR", b)

print("\n=== 8c. TIMESERIES /decompose (classical, no seasonality) ===")
r = post("/api/timeseries/decompose", {
    "session_id": sid, "value_col": "age", "time_col": "patient_id",
    "period": 2, "method": "classical", "model": "additive",
})
b = r.json()
print("  status=", r.status_code)
if r.status_code == 200:
    print("  strength_seasonal=", b.get("strength_seasonal"))
else:
    print("  ERR", b)

print("\n=== 8d. TIMESERIES /arima on age (auto) ===")
r = post("/api/timeseries/arima", {
    "session_id": sid, "value_col": "age", "time_col": "patient_id",
    "auto": True, "forecast_steps": 5,
})
b = r.json()
print("  status=", r.status_code)
if r.status_code == 200:
    print("  order=", b.get("order"), "aic=", b.get("aic"), "bic=", b.get("bic"))
    print("  ljung_box_p=", b.get("ljung_box_p"))
    print("  n_coefs=", len(b.get("coefficients", [])))
    for c in b.get("coefficients", []):
        print("    ", c)
    print("  warnings=", b.get("warnings"))
else:
    print("  ERR", b)

print("\n=== 8e. TIMESERIES /arima on fu_days (has -10 outlier) non-auto (0,0,0) ===")
r = post("/api/timeseries/arima", {
    "session_id": sid, "value_col": "fu_days", "time_col": "patient_id",
    "p": 0, "d": 0, "q": 0, "forecast_steps": 3,
})
b = r.json()
print("  status=", r.status_code)
if r.status_code == 200:
    print("  order=", b.get("order"), "aic=", b.get("aic"), "rmse=", b.get("in_sample_rmse"))
else:
    print("  ERR", b)


# ════════════════════════════════════════════════════════════════════════════
# 9. REPORTING — pub_export
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 9a. PUB_EXPORT /table_docx ===")
r = post("/api/pub_export/table_docx", {
    "session_id": sid, "group_column": "sex",
    "variables": ["age", "sbp", "nyha"],
})
print("  status=", r.status_code, "content-type=", r.headers.get("content-type"),
      "bytes=", len(r.content))

print("\n=== 9b. PUB_EXPORT /figure_caption ===")
for ftype in ["roc", "forest", "histogram", "scatter", "bar", "km"]:
    r = post("/api/pub_export/figure_caption", {
        "figure_type": ftype,
        "params": {"outcome": "Mortality", "auc": 0.82, "ci_lower": 0.75,
                   "ci_upper": 0.88, "cutoff": 120, "sensitivity": 0.80,
                   "specificity": 0.75, "group": "Treatment", "p_value": 0.03,
                   "median_survival": "14 mo", "analysis_type": "OR",
                   "i_squared": 35, "variable": "Age", "n_obs": 100,
                   "normality_p": 0.12, "x_var": "Age", "y_var": "SBP",
                   "r_value": 0.30},
    })
    b = r.json()
    print(f"  {ftype}: caption='{b.get('caption')}' footnote='{b.get('footnote')}'")

print("\n=== 9c. PUB_EXPORT /method_appendix (no prior audit) ===")
r = post("/api/pub_export/method_appendix", {"session_id": sid})
print("  status=", r.status_code, "bytes=", len(r.content))

print("\n=== 9d. PUB_TABLES /format (journal) ===")
# Build a table1 result and pipe through journal formatter
t1 = post("/api/stats/table1", {
    "session_id": sid, "group_column": "sex", "variables": ["age", "sbp"],
}).json()
r = post("/api/pub_tables/format", {"table1_result": t1, "options": {}})
print("  status=", r.status_code)
b = r.json()
print("  keys=", list(b.keys()) if isinstance(b, dict) else type(b))


# ════════════════════════════════════════════════════════════════════════════
# 10. EXTRA EDGE PROBES
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 10a. DESCRIPTIVE on bmi (text-typed) — should it error or coerce? ===")
r = get(f"/api/stats/{sid}/descriptive", column="bmi")
print("  status=", r.status_code, "body=", r.json())

print("\n=== 10b. column_summary on bmi text-typed with kind=numeric ===")
r = get(f"/api/stats/{sid}/column_summary", column="bmi", kind="numeric")
b = r.json()
print("  status=", r.status_code)
if r.status_code == 200:
    print("  n=", b.get("n"), "mean=", b.get("mean"), "min=", b.get("min"), "max=", b.get("max"))
    print("  [note: 999 / 'n/a' / comma-decimals present in raw bmi column]")

print("\n=== 10c. weighted_descriptive where weight col has missing ===")
r = post("/api/stats/weighted_descriptive", {
    "session_id": sid, "value_cols": ["sbp"], "weight_col": "ldl",
})
b = r.json()
print("  status=", r.status_code, "n=", b.get("n"))
for res in b.get("results", []):
    print("   ", res)

print("\nDONE PROBING")
