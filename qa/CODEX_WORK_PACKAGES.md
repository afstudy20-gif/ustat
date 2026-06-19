# Codex work packages — Phase 4-9

Six independent packages. Each one is a self-contained prompt Codex can act
on alone. Phases 1-3 (the 12 CRITICAL bugs) are already committed —
Codex starts from `main` at commit `3048c38` or later.

For every package:

- **Read first:** `qa/findings/*.md` for the original repro, `qa/TEST_PLAN.md`
  for the finding schema, and `qa/run_via_testclient.py` for the
  TestClient-based driver against `qa/cohort_test.csv`.
- **Do not edit production code beyond what the package asks for.**
- After each fix, verify against `qa/cohort_test.csv` via the same TestClient
  pattern shown in Phases 1-3. Write a one-line summary of the verification
  to the commit message.
- One commit per package. Branch is `main`, commits go straight there.

---

## Package 4 — Numeric coercion: cover the rest of the leak surface (HIGH)

**Scope:** Phase 1 fixed comma-decimals + text sentinels at upload. Several
downstream paths still re-coerce or compute on raw strings and either drop
information or admit a sentinel like `"999"` as a real number.

**Findings to fix:**

- [zcode HIGH] `fill_blanks __mean__` on bmi returns a mean polluted by `999`
- [zcode HIGH] MICE mis-reports `n_imputed`; `999` poisons PMM neighbours
- [zcode HIGH] H2FPEF / clinical scores using bmi inherit the same `999` /
  comma-decimal bug
- [zcode HIGH] Missing-data audit undercounts bmi missingness (ignores `999`)
- [codex HIGH] Linear diagnostics, polynomial regression, LMM share the BMI
  coercion crash (verify they're fixed by Phase 1 — if not, add the missing
  guard)
- [codex HIGH] Gamma GLM crashes instead of fitting a positive continuous
  outcome
- [codex HIGH] GEE fails on locale-decimal BMI although the cleaned model
  fits — see if Phase 1 already covered it; otherwise patch the GEE call site

**Approach:**

1. Add a single `services/dirty_value_guard.py` helper that:
   - Returns the set of "obviously sentinel" values for a column (e.g. anything
     ≥ 99th percentile + 5×IQR over the body of the distribution AND outside a
     plausibility range you let the caller pass).
   - Provides `flag_sentinels(series, max_plausible)` → boolean mask.
2. Wire that helper into:
   - `services/missing_data.py` — count sentinels alongside NaNs in the
     missing-audit; flag them as "implausible (review)" in the response.
   - `services/impute.py` `fill_blanks __mean__` — exclude sentinels before
     computing the mean.
   - `services/missing_data.mice_multiple` — mask sentinels to NaN before
     PMM; bump the imputed count.
   - `routers/compute.py` clinical calculators (H2FPEF etc.) — same masking.
3. Add a test file `tests/test_dirty_value_guard.py` (4-6 tests).

**Verify:**

- Upload `qa/cohort_test.csv`. Missing audit on `bmi` should report
  `n_implausible >= 1` (the `999` row).
- `fill_blanks __mean__` on `bmi` should drop `999` → mean ≈ 27.8, not 38.1.
- Clinical-score endpoint that uses `bmi` should warn for the sentinel row.

---

## Package 5 — Categorical hygiene + JSON-NaN coverage (HIGH + MED)

**Scope:** Phase 2 added a global JSON-NaN handler and rare-level warnings to
logistic. The same pattern bites every other test that runs `pd.get_dummies`
or assumes 2-level categoricals.

**Findings to fix:**

- [kimi HIGH] Chi-square treats invalid sex codes as separate columns
- [kimi MED] Independent t-test fails because dirty sex codes inflate group
  count (return a clean 400, like Mann-Whitney does)
- [kimi MED] Mann-Whitney U fails because sex has >2 levels (already 400;
  improve the message to name the offending levels)
- [kimi MED] TOST fails on sbp ~ sex because dirty codes
- [kimi MED] Non-inferiority fails on event ~ sex because dirty codes
- [kimi MED] Bayesian t-test fails on age ~ sex after dropping all sex groups
- [kimi MED] Cochran's Q endpoint fails on cohort_test.csv
- [kimi MED] Two-proportion z endpoint fails on cohort_test.csv
- [codex HIGH] Stepwise logistic fails instead of dropping unstable dirty
  categorical levels
- [codex HIGH] Poisson + negative-binomial crash during JSON encoding on
  sparse categorical levels
- [codex HIGH] Ordinal logistic 500s on NYHA instead of returning a Brant test
- [grok-composer HIGH] Table 1 propagates un-harmonised sex levels as 4
  groups

**Approach:**

1. Extend `services/category_health.py` with `clean_two_level(series,
   keep="auto")` → keeps the two most frequent levels, returns a mask of rows
   to drop and the dropped count.
2. Add to every 2-group endpoint (t-test, Mann-Whitney, TOST, non-inferiority,
   2-prop, McNemar, Bayesian t-test): when the grouping column has >2 levels
   call `clean_two_level`; attach a warning naming the dropped levels.
3. For ≥3-group endpoints (Kruskal, chi², ANOVA, ordinal logistic, etc.):
   call the existing `rare_level_warnings` and attach to the response;
   continue the fit.
4. Apply the same wrapper to Poisson / NegBinom / Gamma GLM / GEE / stepwise.
5. Table 1: add a "label_map" parameter so the caller can collapse
   `"F"/"Female"` → `"F"`; or auto-normalise via simple Levenshtein + freq.

**Verify:** All listed endpoints return 200 (with warnings) instead of 422 /
500 / silent dummies on `qa/cohort_test.csv`.

---

## Package 6 — Apply Phase 3 survival guard to every survival endpoint (HIGH)

**Scope:** Phase 3 wired `validate_survival_inputs` into Fine-Gray and
external_validation only. KM, Cox, Cox-RCS, time-horizon HR, RMST, landmark,
LWYY, joint-model, interval-censored — none of them have the guard yet, and
the QA audit confirmed they each behave badly on the same negative-fu_days
row.

**Findings to fix:**

- [codex HIGH] Main survival endpoints reject the entire cohort for one
  negative follow-up value (KM / Cox / time-horizon / RMST / landmark)
- [codex HIGH] Cox diagnostics can pass on a model the main Cox endpoint
  refuses to fit (consistency)

**Approach:**

1. Add `validate_survival_inputs` to every survival fitter under
   `services/survival_advanced_service.py` and `routers/models/cox.py`.
2. The right behaviour is to **drop** the bad rows and warn, not reject the
   whole cohort. Update the helper with a `mode="reject"|"drop_with_warning"`
   parameter. Default `reject` for Fine-Gray / extval (already shipped),
   `drop_with_warning` for the routine fits (KM, Cox, etc.).
3. Make Cox diagnostics use the SAME row-set as the main Cox fit (read the
   `n_excluded` from the fit's response and propagate, or share the same
   prep helper).

**Verify:** With `fu_days = -10` on row 99, KM / Cox return 200 with `n=99`
and a `warnings: ["row 99 dropped: non-positive fu_days"]`. Cox diagnostics
report the same `n`.

---

## Package 7 — Summary / Visual / Table 1 numeric leakage (HIGH + MED)

**Scope:** When a column arrives as text-typed numeric, several display
endpoints either crash or render absurd results. Phase 1 fixed ingest, but
some endpoints still re-coerce or assume a column's `kind` matches its
dtype.

**Findings to fix:**

- [grok-composer HIGH] `column_summary` (Q-Q / outliers panel) 500-crashes
  on a text column when `kind=numeric`
- [grok-composer HIGH] Histogram endpoint 500-crashes on a text column
- [grok-composer HIGH] Impossible values silently corrupt Summary
  descriptives (age, fu_days) with no range guard or warning
- [grok-composer HIGH] Table 1 renders `bmi` as 75 categorical "n (%)" rows
- [grok-composer HIGH] Subgroup Bar with no color column emits one bar per
  (subgroup × xaxis) cell in a single "All" trace
- [grok-composer HIGH] Subgroup Bar (mean mode) silently lets `"999"`
  dominate the BMI mean and drops comma-decimals
- [grok-composer MED] Weighted descriptive (listwise default) silently
  shrinks every column to the union of missingness

**Approach:**

1. Each endpoint: read the column's stored `kind` from session metadata, and
   if `kind=numeric`, call `pd.to_numeric(errors="coerce")` once at the top
   so the rest of the function is dtype-stable.
2. Add a plausibility-range guard for `age` (≥0 ≤ 120), `fu_days` (>0),
   `bmi` (>10 <100) — but **don't drop**, just warn. Numbers like `999` light
   the warning.
3. Subgroup Bar: when `color_col` is empty, render one trace per `xaxis`
   level, not one fat "All" trace.
4. Weighted descriptive: switch default to per-column complete-case (not
   listwise across all columns).

---

## Package 8 — Causal / Mediation / Power / Meta + ROC fixes (HIGH + MED)

**Scope:** A grab-bag of single-endpoint bugs the QA audit caught.

**Findings to fix:**

- [grok-build HIGH] Mediation runs linear models on binary outcome without
  rejection (must detect and route to logistic-mediator-logistic, or 422)
- [grok-build HIGH] Power logistic + Cox lack `solve_for=effect_size`
  (minimum detectable OR/HR — invert Hsieh's and Schoenfeld's formulas)
- [grok-build HIGH] DCA on multi-predictor model with dirty cohort returns
  singular-matrix error (wrap the logistic with the rare-category prep from
  Package 5)
- [grok-build MED] DiD accepts any binary column as time_col without
  temporal validation
- [grok-build MED] PSM common-support trim on dirty data yields n=2 matches
- [grok-build MED] IPTW warnings empty when weights are concerning (add
  weight-extreme detection like `max_weight > 10` → warn)
- [grok-build MED] Causal routers trigger pandas SettingWithCopyWarning
  (clean copies before mutating)
- [grok-build MED] E-value endpoint returns different key names than its
  helper (response shape mismatch)
- [grok-composer HIGH] Publication-bias trim-and-fill returns 0 missing
  studies on an asymmetric funnel (Egger p<0.001) → the estimator's
  iteration loop has a bug; cross-check against `dmetar::tf` reference
- [grok-composer HIGH] Forest plot silently accepts a negative OR /
  ci_low<0 and corrupts the pooled meta → guard at the meta layer
- [kimi MED] Combined ROC model reports AUC < 0.5 without flipping direction
  (auto-flip when score is anti-predictive, like single-ROC already does)
- [codex MED] OR table reports `n_total=4` for a 100-row dataset (typo
  somewhere — read the path carefully)

**Approach:** Each finding is small (10-30 LOC). One commit can hold all of
them; group by file to minimise re-running tests.

---

## Package 9 — Data tools polish (MED + LOW)

**Scope:** Compute / Data tab UX papercuts.

**Findings to fix:**

- [zcode MED] `clean_outliers` on `age` deletes 5 rows but count doesn't
  match obvious impossibles (off-by-some)
- [zcode MED] `select_cases` silently selects ALL rows when given an
  operator it doesn't recognise (`==` instead of `eq`)
- [zcode MED] `find_replace` returns `replaced_count=0` with misleading
  "nothing happened" signal even when find value isn't present
- [grok-composer MED] ARIMA `auto` overfits near-iid data with no warning
  (heuristic: if AR/MA both 0 in differenced series → warn)
- [grok-composer MED] STL invents seasonal component on non-seasonal data
- [grok-composer MED] Journal formatter's own validator flags its Table 1
  output as "REVISION REQUIRED" (p-value formatting fail — fix the writer)
- [grok-composer LOW] Meta-regression labels WLS R² as "proportion of τ²
  explained" (cosmetic but wrong)
- [grok-composer LOW]  + [zcode LOW × 2] miscellaneous copy fixes

---

## Suggested order

1. **Package 4** (numeric leak) — leverages Phase 1's groundwork.
2. **Package 6** (survival guard) — small, mechanical, biggest win on
   reliability.
3. **Package 5** (categorical hygiene) — pairs with Package 4 thematically.
4. **Package 7** (summary/visual leakage) — visible to users.
5. **Package 8** (grab-bag) — once the data-quality dust settles.
6. **Package 9** (polish) — last.

Each package can be opened as its own Codex session. Keep `qa/` open and
verify with the TestClient after each one.
</content>
