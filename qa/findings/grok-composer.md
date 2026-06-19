# Grok-composer QA findings — Visual / Reporting / Table 1 / Summary / Meta / Time Series

Agent: **grok-composer** (driven via zcode)
Slice: Table 1 · Summary · Visual · Meta · Time Series · Reporting
Dataset: `qa/cohort_test.csv` (100×11), loaded via `qa/run_via_testclient.boot()`.
Reproducible helper scripts: `qa/probe_grok_composer.py`, `qa/verify_grok_composer.py`,
`qa/verify2_grok_composer.py`, `qa/verify3_grok_composer.py`. No production code was edited.

Key data-load facts that drive several findings below: on ingest `age` (int64),
`ldl`/`sbp`/`nyha`/`diabetes`/`event` (float64), `fu_days` (int64) are numeric, so the
impossible values `age=-5`, three `age=199`, and `fu_days=-10` enter the numeric frame
untouched. `bmi` is typed **text** (because of `"25,9"`, `"n/a"`, `"999"`); `sex` is
categorical with four distinct non-missing strings `M / F / x / Female` plus 2 blanks.

---

## [CRITICAL] Subgroup Bar chart 500-crashes when subgroup column == x-axis column
**Where:** Visual → Subgroup Bar
**Steps:** 1) Load cohort. 2) POST `/api/charts/subgroup_bar` with
`subgroup_col` and `xaxis_col` set to the same column (e.g. both `nyha`), `y_col=event`,
`y_mode=percentage`.
**Expected:** Either a single bar per `nyha` level, or a 400 telling the user the two
axes must differ.
**Actual:** Uncaught 500. `sorted_groups(sub[req.subgroup_col])` is called twice on the
same 2-column DataFrame slice, so `pd.Series(series)` receives an ndarray of shape
`(98, 2)` → `ValueError: Data must be 1-dimensional, got ndarray of shape (98, 2)`.
**Evidence:** Trace from `backend/routers/charts.py:357` →
`backend/services/stat_utils.py:27` (`sorted_groups`). Reproduced in
`qa/probe_grok_composer.py` §3c/3d (status 599). The TestClient re-raises the raw
exception; in the live app this is a 500 to the browser.
**Hypothesis:** `subgroup_col` and `xaxis_col` are independent selectors with no guard;
when equal, `sub[req.subgroup_col]` returns a 2-column DataFrame and every downstream
`sorted_groups`/mask explodes.

---

## [HIGH] Subgroup Bar with no color column emits one bar per (subgroup × xaxis) cell in a single "All" trace
**Where:** Visual → Subgroup Bar
**Steps:** 1) POST `/api/charts/subgroup_bar` with `y_col=event`, `subgroup_col=diabetes`,
`xaxis_col=nyha`, no `color_col`, `y_mode=percentage`.
**Expected:** With no color variable, one trace ("All") with **4 bars** — one per `nyha`
level — each pooling across diabetes.
**Actual:** The single "All" trace contains **8 bars**: `xaxis=['1.0','2.0','3.0','4.0',
'1.0','2.0','3.0','4.0']` with `y=[62.5, 31.4, 40.9, 57.1, 12.5, 16.7, 55.6, 100.0]` and
`ns=[8, 35, 22, 7, 8, 6, 9, 1]`. The first four bars are diabetes=0 cells, the next four
are diabetes=1 cells — i.e. the chart silently stratifies by `subgroup_col` even though
no color is shown, so two differently-derived bars land on the same x label with
different n's.
**Evidence:** `qa/verify_grok_composer.py` §1 — crosstab `nyha × diabetes` matches the
eight `ns` exactly (col sums 72/24); backend response in `qa/probe_grok_composer.py` §3e.
Loop is `for sg in subgroups: for xv in x_vals:` inside one trace regardless of
`color_col`.
**Hypothesis:** When `color_col` is None the code should collapse over `subgroup_col`
(or require a color), not iterate it.

---

## [HIGH] Subgroup Bar (mean mode) silently lets `"999"` dominate the BMI mean and drops comma-decimals
**Where:** Visual → Subgroup Bar
**Steps:** POST `/api/charts/subgroup_bar` `y_col=bmi` (text), `subgroup_col=diabetes`,
`xaxis_col=nyha`, `y_mode=mean`.
**Expected:** Either refuse (bmi is text), or coerce consistently — and never let a
sentinel `"999"` masquerade as a real measurement.
**Actual:** 200 OK. The cell diabetes=1 × nyha=3 reports mean **74.07** (vs ~27 in
other cells) because `pd.to_numeric(cell, errors="coerce")` happily parses `"999"` as
999 while simultaneously dropping the comma-decimal rows (`"25,9"` → NaN). So the
documented `bmi=999` sentinel leaks straight into a published-looking mean and skews the
bar chart, with no warning.
**Evidence:** `qa/verify3_grok_composer.py` — `All` trace
`y=[29.4, 28.1, 74.07, 26.1, 26.8, 27.4, 26.9, 28.6]`; the 74.07 spike corresponds to
the single `bmi="999"` row (P023, nyha=3).
**Hypothesis:** `pd.to_numeric(errors="coerce")` treats `"999"` as valid; there is no
sentinel/outlier guard before averaging.

---

## [HIGH] `column_summary` (Q-Q / outliers panel) 500-crashes on a text column when `kind=numeric`
**Where:** Summary → Descriptive → Q-Q / outlier detail
**Steps:** 1) Open the Summary panel for `bmi` (text-typed). 2)
GET `/api/stats/{sid}/column_summary?column=bmi&kind=numeric`.
**Expected:** A guarded 400 ("column is not numeric; coerce first") or a coerced result.
**Actual:** Uncaught 500: `s.dropna().astype(float)` raises
`ValueError: could not convert string to float: '25,9'` at
`backend/routers/stats/descriptive.py:247`.
**Evidence:** `qa/probe_grok_composer.py` §10b (full traceback). The comma-decimal
locale leakage in `bmi` is one of the documented dirty bits, so this path is reachable
from normal use.

---

## [HIGH] Histogram endpoint 500-crashes on a text column
**Where:** Visual → Charts (custom histogram) / Summary histogram picker
**Steps:** POST `/api/charts/histogram` `{"session_id": sid, "x": "bmi", "bins": 10}`.
**Expected:** 400 "column not numeric" (consistent with `/descriptive`, which does
return 400 for `bmi`), or coercion with a warning.
**Actual:** Uncaught 500 — `np.histogram(s, …)` on the object Series raises
`ufunc 'isfinite' not supported for the input types`.
**Evidence:** `qa/verify3_grok_composer.py` (status 599). Inconsistent with the
`/descriptive` endpoint, which guards `bmi` with a clean 400.

---

## [HIGH] Impossible values silently corrupt Summary descriptives (age, fu_days) with no range guard or warning
**Where:** Summary → Descriptive (mean / SD / skew / kurtosis / min / max)
**Steps:** GET `/api/stats/{sid}/descriptive?column=age` and `…?column=fu_days`.
**Expected:** Either a guard/flag on biologically impossible values (`age=-5`,
`age=199`×3, `fu_days=-10`), or at minimum a prominent warning that outliers are
included.
**Actual:** `age`: mean **63.26**, SD 27.78, min **−5**, max **199**, skew 3.32,
kurtosis 14.98 — all computed on the dirty series as-is. Recomputed independently with
scipy on the same coerced series: identical (skew 3.3168, kurtosis 14.9789), confirming
the *math* is right but the *data* is wrong. Cleaning to plausible ages (5<age<120)
gives mean **59.73**, SD 12.67 — i.e. the reported mean is inflated by ~3.5 units and
the SD more than doubled. `fu_days` reports min **−10** baked into mean 584.87.
**Evidence:** `qa/probe_grok_composer.py` §1, `qa/verify_grok_composer.py` §6. The
boxplot/Q-Q panel *does* flag these as outliers (`column_summary` outliers for `age`
include row 4 = −5.0 and rows 28/62/85 = 199.0), so the app detects them in one place
but still feeds them into the headline descriptives in another.
**Hypothesis:** `_normality_test`/`descriptive` operate on the raw numeric Series with
no plausibility filter; the outlier list shown elsewhere is never fed back as exclusions.

---

## [HIGH] Table 1 renders `bmi` as 75 categorical "n (%)" rows (text-typed leakage)
**Where:** Table (Table 1)
**Steps:** POST `/api/stats/table1` with `group_column=sex`, `variables=["bmi", …]`.
**Expected:** `bmi` treated as numeric (mean±SD / median[IQR]), or a warning that the
column is text and cannot be summarised continuously.
**Actual:** `bmi` comes back `type=categorical`, `stat_label="n (%)"`, `overall="n=97"`,
with **75 sub_rows** — one per distinct string value (`"26.0" 3 (3.1%)`, `"999" …`,
`"n/a" …`, `"25,9" …`). The table is unreadable noise.
**Evidence:** `qa/verify_grok_composer.py` §5 — `bmi.nunique()=76`; backend emits 75
sub-rows. The auto-typing rule `is_num = numeric & nunique>10` is False for an
object-dtype column, so it falls through to categorical. The `999` and comma-decimal
dirty bits are the direct cause.
**Hypothesis:** Table 1 has no "looks numeric but stored as text" recovery; it should
attempt `pd.to_numeric(errors="coerce")` and warn about the unparseable cells.

---

## [HIGH] Publication-bias trim-and-fill returns 0 missing studies on a deliberately asymmetric funnel (Egger p<0.001)
**Where:** Meta → Publication bias
**Steps:** POST `/api/meta/bias` with 7 OR studies whose two smallest, most-imprecise
studies both have large effects (S1 OR 0.30, S2 OR 0.35) and the rest cluster near 1.0.
**Expected:** Trim-and-fill should estimate missing studies on the right side of the
funnel (the asymmetric side), consistent with Egger/Begg both flagging asymmetry.
**Actual:** Egger intercept −2.45, **p<0.001**; Begg τ=−0.90, **p=0.003** — both scream
asymmetry — yet `trim_fill_missing = 0`. Recomputing the backend's own L0 estimator
(Tweedie formula in `meta.py`): `Tn=15, n=7, L0=(4·15−7·8)/(2·7−1)=0.31`,
`k0=round(0.31)=0`. The estimator is **one-sided** (it only counts positive-centred
ranks `Tn=Σ ranks[signs>0]`), so when the pooled mean is dragged below the bulk of
studies by the two tiny ones, almost all residuals go positive and L0 collapses to ~0.
**Evidence:** `qa/verify_grok_composer.py` §2 — full reproduction; R's
`meta::trimfill` would impute on the opposite side. The interpretation string still
claims "Trim-and-fill estimates 0 potentially missing study(ies)." next to the
significant Egger/Begg results, which is internally contradictory.
**Hypothesis:** The side of trimming is hard-coded from the sign of `centered`; a
correct L0 implementation chooses the side opposite the pooled mean and imputes there.

---

## [HIGH] Forest plot silently accepts a negative OR (ci_low<0) and corrupts the pooled meta
**Where:** Visual → Forest plot (and the univariate-OR screening hook that reuses
`/api/charts/forest`)
**Steps:** POST `/api/charts/forest` with `x_axis="log"`, `do_meta=true`, rows
`[{Bad: est .5, ci_low −.3, ci_high 1.2}, {Ok …}, {Ok2 …}]`.
**Expected:** 422 — an OR / CI bound must be positive on the log scale.
**Actual:** 200 OK. The bad row becomes `log_low = log(max(−0.3,1e−12)) = −27.63`,
`se = (0.18 − (−27.63))/(2·1.96) = 7.10`. That single garbage SE then perturbs the
inverse-variance meta pool (τ²/Q re-weighting) and the result text prints a pooled OR
with no complaint.
**Evidence:** `qa/verify2_grok_composer.py` §A — `Bad` row `se=7.095`, pooled
`0.9489 [0.6856, …]`. No validation on `ci_low`/`ci_high` sign for log-scale effects.
**Hypothesis:** Missing guard: on `x_axis=="log"`, reject any `est`/`ci_low`/`ci_high`
≤ 0.

---

## [MEDIUM] Weighted descriptive (listwise default) silently shrinks every column to the union of missingness
**Where:** Summary → Weighted
**Steps:** POST `/api/stats/weighted_descriptive` with `value_cols=["age","ldl"]`,
`weight_col="sbp"` (default `imputation="listwise"`).
**Expected:** Per-column `n` reflecting that column's own valid count (age=100, ldl=92),
or a clear note that listwise deletion aligns all columns to the smallest.
**Actual:** A single global `n=89` is reported, and **both** `age` (no missing) and
`ldl` (8 missing) come back with `n=89`. The user sees age's weighted mean computed on
89 rows with no indication that 11 age values were dropped because of *other* columns.
**Evidence:** `qa/verify_grok_composer.py` §7 — `n=89` for both; `age` has 0 missing in
the raw data. (89 vs the expected 92 is itself surprising — it means sbp's 3 blanks
+ ldl's 8 blanks overlap to remove 11 rows total.)
**Hypothesis:** `apply_imputation(…, "listwise")` drops any row missing in *any* of the
selected cols, then reports one n for all. Per-column n should be shown, or the
imbalance flagged.

---

## [MEDIUM] ARIMA `auto` overfits near-iid data to (0,1,2) with unidentified coefficients and no warning
**Where:** Models → Time Series → ARIMA (Auto)
**Steps:** POST `/api/timeseries/arima` `value_col=age`, `time_col=patient_id`,
`auto=true`. (Age ordered by patient_id is essentially iid.)
**Expected:** A parsimonious ARIMA(0,0,0) / (0,1,0), or at least a warning that the
selected order is overfit and coefficients are unidentifiable.
**Actual:** Picks `(0,1,2)` with `ma.L1 = −0.95 (SE 282.1, p 0.997)`, `ma.L2 = −0.05
(SE 15.0, p 0.997)` — SEs 2–3 orders of magnitude larger than the estimates. AIC 921.0
beats (0,0,0)'s 1121.7 purely because differencing absorbs the mean, but the MA terms
are noise. The response `warnings` array is **empty** despite the Ljung-Box being
trivially non-significant and coefficients being unidentified.
**Evidence:** `qa/probe_grok_composer.py` §8d; `qa/verify_grok_composer.py` §8 confirms
AIC consistency. Coefficient SEs of 282/15 vs estimates ≈1 are the red flag.
**Hypothesis:** The grid (`p∈0..2, d∈0..1, q∈0..2`) minimises AIC blindly; no check on
coefficient identifiability (SE/estimate ratio) or on differencing a stationary series.

---

## [MEDIUM] STL decomposition invents a seasonal component on non-seasonal data with no warning
**Where:** Models → Time Series → Decompose (STL)
**Steps:** POST `/api/timeseries/decompose` `value_col=age`, `period=7`, `method=stl`.
**Expected:** Either flat/zero seasonal component, or a warning that the period is
arbitrary and the series shows no seasonality.
**Actual:** Returns `strength_seasonal=0.0228` (correctly near 0) **but** a seasonal
series with amplitude **50.96** on an observed range of 204 — i.e. a non-trivial
wiggly seasonal curve plotted for the user even though the strength metric says it's
negligible. No `warnings` field at all on this endpoint.
**Evidence:** `qa/verify_grok_composer.py` §9 — seasonal min/max −36.98/13.98. The
strength metric and the plotted component contradict each other in magnitude.
**Hypothesis:** STL always returns a seasonal series of period `p` regardless of whether
seasonality exists; the endpoint should warn when `strength_seasonal` ≈ 0 and/or when
no real time ordering exists (patient_id is just a row label).

---

## [MEDIUM] Journal formatter's own validator flags its Table 1 output as "REVISION REQUIRED" (p-value formatting FAIL)
**Where:** Reporting → pub_tables journal format (Table 1 → AMA)
**Steps:** Build a Table 1 (`/api/stats/table1` by `sex`), then POST it to
`/api/pub_tables/format`.
**Expected:** The formatted AMA table passes the formatter's built-in validation.
**Actual:** Response includes `"validation": {"p_value_formatting": "FAIL",
"status": "REVISION REQUIRED"}` — the app's own self-check fails its own output.
The table otherwise renders (title, columns, footnotes all PASS), but the p-value
column (e.g. `"0.308"`, `"0.215"`) is rejected by the validator's p-format rule.
**Evidence:** `qa/verify2_grok_composer.py` §C — full validation dict. Also note the
columns inherit the un-harmonised sex groups `F / Female / M / x` (see next finding).
**Hypothesis:** The formatter emits raw `"0.308"` strings where the validator expects
APA-style (`p = .308`, `p < .001`); the two code paths disagree on p formatting.

---

## [MEDIUM] Table 1 / boxplot / journal export all propagate un-harmonised `sex` levels (F, Female, M, x) as 4 separate groups
**Where:** Table 1; Visual → boxplot by `sex`; Reporting journal export
**Steps:** Run any grouped-by-`sex` analysis (Table 1, boxplot, journal format).
**Expected:** `Female`→`F` harmonisation, `x` flagged as unknown, blanks excluded —
yielding two clean groups (M/F) plus a data-quality note.
**Actual:** Four groups are emitted everywhere: `group_labels=['F','Female','M','x']`,
`group_ns={'F':39,'Female':1,'M':57,'x':1}`. Consequences cascade:
- Table 1 runs a **4-group Kruskal-Wallis/ANOVA including two singleton groups** and
  prints `"124 ± —"` for the SD of `Female` (n=1).
- Boxplot draws 4 boxes, two of which (`Female`, `x`) are single points.
- The journal-format AMA table gets columns `F (n=39) | Female (n=1) | M (n=57) | x (n=1)`.
**Evidence:** `qa/probe_grok_composer.py` §3b/§4; `qa/verify2_grok_composer.py` §C/§F.
`frequency(sex)` confirms the raw levels (`M 57, F 39, Missing 2, x 1, Female 1`).
**Hypothesis:** No categorical-value harmonisation layer between ingest and grouped
analyses; the documented "x" / "Female" dirty bits flow straight through.

---

## [LOW] Meta-regression reports WLS R² (98.56%) labelled as "proportion of τ² explained", overstating explanatory power
**Where:** Meta → Meta-regression
**Steps:** POST `/api/meta/regression` with 5 OR studies and a moderator perfectly
correlated with log-OR.
**Expected:** R² analog close to but not exactly the WLS fit R², with wording that
distinguishes "regression R²" from "heterogeneity explained".
**Actual:** `r2_pct = 98.56` (= WLS `model.rsquared·100` on the log scale), reported
alongside `tau2=0.0315` and `tau2_resid = tau2·(1−R²) = 0.000454`. The implication
(0.000454/0.0315 ≈ 98.6% of τ² explained) is an artefact of multiplying τ² by (1−R²)
rather than re-estimating τ² on the residuals.
**Evidence:** `qa/verify_grok_composer.py` §4 — independent WLS reproduction gives
slope 0.03885, p 0.000738, R² 0.9856, matching backend exactly. The math is consistent;
the concern is the label/interpretation, not the arithmetic.
**Hypothesis:** `tau2_resid` should be refit (DL on the WLS residuals), not derived as
`tau2·(1−R²)`; the latter can overstate explained heterogeneity.

---

### Summary
15 findings: 1 CRITICAL, 8 HIGH, 5 MEDIUM, 1 LOW.

Recurring themes:
1. **Crash-on-dirty-data** (5 endpoints 500 on text bmi / equal axes / negative OR): the
   documented dirty bits reliably break Summary, Visual, Table 1, and Forest paths.
2. **Silent corruption** (impossible ages/fu_days; `999` in subgroup means; negative OR
   in forest): endpoints return 200 with wrong numbers instead of guarding.
3. **No categorical harmonisation**: `sex` (F/Female/M/x) cascades into every grouped
   output including the journal export.
4. **Self-contradiction**: journal formatter fails its own validator; trim-and-fill
   contradicts Egger/Begg; STL strength≈0 but amplitude≈51.

Note: a parallel grok-composer run previously logged overlapping findings for the
Subgroup-bar-crash, Forest negative-bound, sex harmonisation, and impossible-age topics;
those are re-confirmed here with independent scipy/statsmodels recomputation.
