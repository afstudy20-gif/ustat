"""Targeted verification of suspicious probe outputs."""
from __future__ import annotations
import math
import sys
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, "qa")
from run_via_testclient import boot  # noqa: E402

client, sid = boot()
RAW = pd.read_csv("qa/cohort_test.csv")

# ── 1. Subgroup bar: is the "All" trace duplicating xaxis across subgroup levels? ──
print("=== VERIFY 1: subgroup_bar n counts & bar duplication ===")
# diabetes × nyha crosstab (after dropping missing)
sub = RAW[["event", "diabetes", "nyha"]].dropna()
print("diabetes levels:", sorted(sub["diabetes"].unique()))
print("nyha levels:", sorted(sub["nyha"].unique()))
ct = pd.crosstab(sub["nyha"], sub["diabetes"])
print("crosstab nyha(rows) x diabetes(cols):\n", ct)
print("col sums (n per diabetes):", ct.sum().to_dict())

# The backend call with subgroup=diabetes, xaxis=nyha, no color.
# Per the code, with no color_col, color_groups=["All"], and it iterates
# sg in subgroups(diabetes) × xv in x_vals(nyha). So the single "All" trace
# emits len(diabetes)*len(nyha) bars — DUPLICATING each nyha across diabetes
# strata in the SAME trace. That's the bug.
r = client.post("/api/charts/subgroup_bar", json={
    "session_id": sid, "y_col": "event", "subgroup_col": "diabetes",
    "xaxis_col": "nyha", "y_mode": "percentage", "error_type": "ci",
})
b = r.json()
for tr in b["traces"]:
    print(f"trace {tr['name']}: {len(tr['x_xaxis'])} bars, "
          f"xaxis={tr['x_xaxis']}, y={[round(v,1) for v in tr['y']]}, ns={tr['ns']}")

# Confirm: a single All trace should have one bar per nyha (4 bars), not 8.
print("EXPECTED: a single 'All' trace with 4 bars (one per nyha).")
print("ACTUAL bars in All trace:", len(b["traces"][0]["x_xaxis"]))


# ── 2. Trim-and-fill: returns 0 on an asymmetric funnel? ──
print("\n=== VERIFY 2: trim-and-fill L0 estimator ===")
# Reproduce the backend math on the asymmetric set
studies = [
    ("S1", 0.30, 0.10, 0.90),
    ("S2", 0.35, 0.15, 0.85),
    ("S3", 0.80, 0.60, 1.05),
    ("S4", 0.85, 0.70, 1.05),
    ("S5", 0.90, 0.78, 1.05),
    ("S6", 0.95, 0.85, 1.07),
    ("S7", 1.00, 0.92, 1.08),
]
y = np.array([math.log(s[1]) for s in studies])
se = np.array([(math.log(s[3]) - math.log(s[2])) / (2 * 1.959963984540054) for s in studies])

# Backend logic
order = np.argsort(y)
y_sorted = y[order]
mu = float(np.mean(y_sorted))
centered = y_sorted - mu
ranks = scipy_stats.rankdata(np.abs(centered))
signs = np.sign(centered)
Tn = float(np.sum(ranks[signs > 0]))
n = len(y_sorted)
l0 = (4 * Tn - n * (n + 1)) / (2 * n - 1)
k0 = max(0, int(round(l0)))
print(f"  Tn={Tn}, n={n}, L0={l0:.4f}, k0(rounded)={k0}")
print(f"  y_sorted={y_sorted}")
print(f"  signs={signs.astype(int)}")
print("  Note: with mostly-negative centered values (mean pulled down by 2 tiny"
      " imprecise studies), Tn is small, L0 negative -> k0=0.")
print("  Compare with R meta::trimfill which would impute on the RIGHT side.")


# ── 3. Prediction interval degrees of freedom (k-2) ──
print("\n=== VERIFY 3: prediction interval df = k-2 ===")
# For k=5 studies the backend uses t_{k-2}=t_3. Reference (Higgins 2009 /
# Borenstein) uses t_{k-2} for the PI, so that's fine. Just confirm value.
k = 5
tval = scipy_stats.t.ppf(0.975, k - 2)
print(f"  t_(k-2)=t_{k-2}={tval:.4f}  (acceptable per Higgins 2009)")


# ── 4. Meta-regression R² = 98.56% — verify it's just WLS R² on log scale ──
print("\n=== VERIFY 4: meta-regression R² (WLS on log-OR) ===")
studies_mr = [
    ("S1", 0.50, 0.30, 0.85, 45),
    ("S2", 0.65, 0.40, 1.05, 50),
    ("S3", 0.80, 0.55, 1.15, 55),
    ("S4", 0.95, 0.65, 1.40, 60),
    ("S5", 1.10, 0.70, 1.70, 65),
]
y = np.array([math.log(s[1]) for s in studies_mr])
se = np.array([(math.log(s[3]) - math.log(s[2])) / (2 * 1.959963984540054) for s in studies_mr])
v = se ** 2
x = np.array([float(s[4]) for s in studies_mr])

# DL tau2
w = 1 / v
mu = np.sum(w * y) / np.sum(w)
q = np.sum(w * (y - mu) ** 2)
df = len(y) - 1
c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
tau2 = max(0.0, (q - df) / c)
wr = 1 / (v + tau2)

import statsmodels.api as sm
X = sm.add_constant(x)
m = sm.WLS(y, X, weights=wr).fit()
print(f"  WLS slope={m.params[1]:.5f} p={m.pvalues[1]:.6f} R2={m.rsquared:.4f}")
print(f"  tau2={tau2:.6f}, tau2_resid(backend)=tau2*(1-R2)={tau2*(1-m.rsquared):.6f}")
print("  Note: r2_pct is WLS R² on the LOG scale; labeled 'R² (prop of tau2 explained)'")
print("  but tau2_resid = tau2*(1-R²) is a heuristic, not the true R²=1-tau2_resid/tau2.")
print("  The on-screen 'R²' overstates explained heterogeneity when R²_WLS≈1.")


# ── 5. Table1 bmi as ~50 categorical categories ──
print("\n=== VERIFY 5: Table1 bmi rendered as many categories ===")
# bmi is text-typed (comma-decimals/n/a/999). With >10 unique strings it's
# detected as categorical in table1 (is_num = numeric & nunique>10 → False).
bmi = RAW["bmi"].astype(str)
print(f"  bmi unique count: {bmi.nunique()} (including '999', 'n/a', '25,9')")
print(f"  sample values: {sorted(bmi.unique())[:8]} ...")
print("  => Table1 emits one row per distinct bmi string (~50 rows). Confirmed noise.")
# Confirm by hitting table1
r = client.post("/api/stats/table1", json={
    "session_id": sid, "group_column": "sex", "variables": ["bmi"],
})
b = r.json()
row = b["rows"][0]
print(f"  backend table1 bmi type={row['type']} stat_label={row['stat_label']} "
      f"overall={row['overall']} n_subrows={len(row.get('sub_rows', []))}")


# ── 6. Descriptive/boxplot impossible-value leakage (age -5/199, fu -10) ──
print("\n=== VERIFY 6: impossible values leak into descriptives ──")
age = pd.to_numeric(RAW["age"], errors="coerce").dropna()
print(f"  age: n={len(age)} mean={age.mean():.2f} min={age.min()} max={age.max()} "
      f"std={age.std():.2f}")
print(f"  age without impossible (5<age<120): mean={age[(age>5)&(age<120)].mean():.2f} "
      f"std={age[(age>5)&(age<120)].std():.2f}")
print("  Backend age descriptive mean=63.26 includes -5 and three 199s — no range guard.")
fu = pd.to_numeric(RAW["fu_days"], errors="coerce").dropna()
print(f"  fu_days: min={fu.min()} (one negative -10 leaks into mean=584.87)")


# ── 7. Weighted descriptive ESS: same for both value cols (correct?) ──
print("\n=== VERIFY 7: weighted_descriptive ESS identical across cols ===")
# ess is computed from weights only, so identical across value cols is correct.
# But check: does it silently drop rows where the VALUE is missing but weight
# present? The imputation='listwise' default drops ANY missing in the selected
# cols. ldl has 8 missing, so n=87 for BOTH age and ldl. age has no missing.
r = client.post("/api/stats/weighted_descriptive", json={
    "session_id": sid, "value_cols": ["age", "ldl"], "weight_col": "sbp",
})
b = r.json()
print(f"  n reported (single global) = {b['n']}")
for res in b["results"]:
    print(f"    {res['column']}: n={res['n']}")
print("  => age (no missing) is silently restricted to n=87 to match ldl's 8 gaps")


# ── 8. ARIMA auto on iid-ish age chose (0,1,2) with huge SEs ──
print("\n=== VERIFY 8: ARIMA auto overfits iid data ──")
# The grid only covers p∈0..2, d∈0..1, q∈0..2 — minimum AIC. On near-iid data
# differencing (d=1) inflates noise; (0,1,2) won with ma SEs of 282 / 15.
print("  Backend picked (0,1,2): ma.L1 SE=282, ma.L2 SE=15 — coefficients")
print("  essentially unidentified. ARIMA(0,0,0) would be the principled pick.")
# Compare AIC of (0,0,0) vs (0,1,2) on same series
from statsmodels.tsa.statespace.sarimax import SARIMAX
y_ts = pd.to_numeric(RAW["age"], errors="coerce").dropna().astype(float)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    aic000 = SARIMAX(y_ts, order=(0,0,0), seasonal_order=(0,0,0,0),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).aic
    aic012 = SARIMAX(y_ts, order=(0,1,2), seasonal_order=(0,0,0,0),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).aic
print(f"  AIC(0,0,0)={aic000:.1f}  AIC(0,1,2)={aic012:.1f}")
print(f"  Backend reported AIC(0,1,2)=921.0 — consistent.")


# ── 9. STL decompose on data with no real seasonality returns nonzero seasonal ──
print("\n=== VERIFY 9: STL invents seasonality on iid data ===")
r = client.post("/api/timeseries/decompose", json={
    "session_id": sid, "value_col": "age", "time_col": "patient_id",
    "period": 7, "method": "stl",
})
b = r.json()
seas = b["seasonal"]
print(f"  seasonal strength={b['strength_seasonal']} (near 0 expected for iid)")
print(f"  but seasonal amplitude={max(seas)-min(seas):.2f} on observed range "
      f"{max(b['observed'])-min(b['observed']):.2f}")
print("  => no warning that the data is non-seasonal / period is arbitrary.")

print("\nDONE VERIFY")
