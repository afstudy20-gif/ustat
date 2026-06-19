# uSTAT — Fix Plan (post-QA wave 1+2)

Three root causes explain ~80% of the 12 confirmed CRITICAL findings. Fix
them in this order. Each phase = one commit + targeted regression test.

## Phase 1 — Numeric coercion at ingest (highest leverage, biggest blast)

**Symptom:** `bmi` column with comma-decimals arrives as `dtype:object`.
Every downstream that expects numeric either crashes (linear, kruskal),
silently drops rows (transform), or returns text concat (`bmi*2` → `"30.630.6"`).

**Affected findings:** Z1, Z2, Z3, K4, C1 (5 of 12 CRITICAL).

**Root cause:** `services/upload.py` (or wherever pandas reads CSV) uses
default parser. Locale-mixed numerics aren't sniffed.

**Fix:**
1. On upload, after pandas reads, sweep each `object` column:
   - try `pd.to_numeric(s.str.replace(",", "."), errors="raise")` →
     if it parses, replace + flag as numeric.
   - guard with a sentinel pattern that recognizes the column as "mostly
     numeric with comma decimals" (≥80% parseable).
2. Also normalize text-as-missing sentinels in the same sweep:
   `["", "NA", "na", "n/a", "?", "-", ".", "NULL", "missing"]` → `NaN`.
3. Update `routers/sessions.py` `kind=numeric` override to **actually coerce**
   the column (`pd.to_numeric(errors="coerce")`), not just flip the metadata.

**Test:** Re-upload `qa/cohort_test.csv`. Expect:
- `bmi` dtype → `float64`
- Z1 transform → `n_computed=97` (not 94)
- Z2 formula `bmi*2` → `dtype:float64`, values are 2× bmi
- C1 linear with bmi as outcome → 200 OK
- K4 kruskal_bmi_nyha → 200 OK
- Missing badge count for `bmi` → 4 (3 truly empty + 1 "n/a")

## Phase 2 — Dirty categorical-coded values

**Symptom:** `sex` column has "M", "F", "x", "Female", "". Stats endpoints
either silently create extra dummies (logistic), or fail JSON serialization
because the test code can't handle the extra categories.

**Affected findings:** K1, K2, K5 (JSON-NaN crashes), C2 (silent extra dummy).
Also drives K3 to a clean 400 instead of useful behavior.

**Fix:**
1. At ingest, log a per-column "uncommon levels" list when a categorical has
   ≥3 levels but one of them is `<5%` of rows AND visually looks like a
   typo of another level (e.g. "x" vs "M"/"F"; "Female" vs "F").
2. In `routers/advanced_anova.py` and `routers/categorical.py`: catch
   `ValueError("Out of range float values are not JSON compliant")` →
   return 400 with `detail: "Statistic is NaN/inf, often caused by very
   small subgroups. Check for outliers / dirty category codes."`.
3. In `routers/models/logistic.py`: detect when a dummy column has `<5` rows
   and warn (don't crash, but flag in response under `warnings`).

**Test:**
- K1 ancova age~sex+ldl → 200 OK with `warnings: ["sex has 3+ rare categories"]`
- C2 logistic → response shows the rare-dummy warning
- K3 mw_bmi_sex → still 400 but with the right diagnostic message
- K5 mantel_haenszel → 400 with the new message, no crash

## Phase 3 — Survival input-range validation

**Symptom:** Survival endpoints accept impossible time values; external
validation accepts development cohort as validation cohort; IPTW reports
no rebalancing.

**Affected findings:** C3 (Fine-Gray neg fu_days), C4 (external_val nonsense
on same-cohort), G1 (IPTW smd_before==smd_after).

**Fix:**
1. `routers/survival_advanced.py` Fine-Gray + main Cox / KM: pre-flight
   check `(duration > 0).all()`. If not: 400 with `detail: "<col> contains
   <n> non-positive values; fu_days must be > 0."`.
2. `routers/survival_advanced.py` external_validation: detect when the
   validation cohort is exactly the development cohort (same `session_id` or
   same rows) → warning in response.
3. `routers/models/psm_iptw.py` IPTW: SMD should be recomputed on the
   weighted sample, not the unweighted. Check whether the bug is "same array"
   or "computed on wrong frame". Fix the calculation; expected `smd_after <
   smd_before` for a sensible weighting.

**Test:**
- C3: Fine-Gray with negative `fu_days` → 400 with the message
- C4: external_val with `validation_id == dev_id` → 200 + warning
- G1: IPTW SMD_before > SMD_after (the test data does improve)

## Out of scope for this fix round

- Visual + Meta + Time Series slices (never tested in wave-1)
- 21 HIGH findings: triaged separately, after Phases 1-3 stabilize
- the 3 LOW (cosmetic / wording)

## Verification dataset

`qa/cohort_test.csv` stays as the regression dataset. After each phase, the
cross-check script under `qa/` is re-run, expected status is documented in
`qa/crosscheck/SUMMARY.md`.
</content>
