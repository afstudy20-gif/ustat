# Cross-check brief — ZCode verifies Kimi + Codex CRITICAL findings

You are doing a wave-2 cross-check. Wave-1 produced findings in qa/findings/{zcode,kimi,codex,grok-build}.md.

Your job: independently reproduce the CRITICAL findings from **kimi.md** and **codex.md** on the same dataset (qa/cohort_test.csv via qa/run_via_testclient.py boot()). For each:
1. Re-run the exact endpoint.
2. Confirm: bug real? Quote response.
3. Or refute: explain.

Output to qa/crosscheck/zcode_xcheck.md using schema:

```
## [CONFIRM|REFUTE|PARTIAL] <agent>:<finding title>
**Verified by:** <command / endpoint>
**Actual response:** <key fields>
**Verdict reason:** <one sentence>
```

Kimi CRITICAL findings to verify (5):
- ANCOVA endpoint fails on age ~ sex + ldl
- Two-way ANOVA fails on age by sex and diabetes
- Mann-Whitney U crashes on comma-decimal BMI values
- Kruskal-Wallis (bmi ~ nyha) fails
- Mantel-Haenszel endpoint fails on cohort_test.csv

Codex CRITICAL findings to verify (4):
- Linear regression 500s on comma-decimal BMI values
- Logistic models silently create bogus predictors for dirty sex levels
- Fine-Gray accepts and plots negative follow-up time
- Survival external validation uses binary calibration and omits O/E

Print 'done zcode-xcheck verified=<C count>/<R count>/<P count>' at end. No code edits.

---

# Wave-2 cross-check results (ZCode verifying Kimi + Codex)

Re-ran via `qa/run_via_testclient.py boot()` then `qa/crosscheck/zcode_xcheck_verify.py`
(FastAPI TestClient, no uvicorn). sid for the run: `19cadf61-80aa-45dc-a03a-07e7d252fe75`.
Full log: `qa/logs/zcode_xcheck_verify.log`; stderr: `qa/logs/zcode_xcheck_verify.err`.

## [CONFIRM] kimi:ANCOVA endpoint fails on age ~ sex + ldl
**Verified by:** `POST /api/advanced_anova/ancova` `{outcome=age, group_col=sex, covariates=[ldl]}`
**Actual response:** `status=EXC` `ValueError('Out of range float values are not JSON compliant')` — TestClient could not return JSON because statsmodels produced a non-finite float (overflowing F from rank-deficient constraint covariance).
**Verdict reason:** Endpoint crashes identically; ANCOVA returns no usable result on this cohort.

## [CONFIRM] kimi:Two-way ANOVA fails on age by sex and diabetes
**Verified by:** `POST /api/advanced_anova/two_way_anova` `{outcome=age, factor1=sex, factor2=diabetes}`
**Actual response:** `status=EXC` `ValueError('Out of range float values are not JSON compliant')` — same JSON-non-compliant overflow path as ANCOVA (constraint covariance not full rank → division by zero → inf/NaN).
**Verdict reason:** Two-way ANOVA fails with the exact same non-JSON crash reported by Kimi.

## [CONFIRM] kimi:Mann-Whitney U crashes on comma-decimal BMI values
**Verified by:** `POST /api/stats/mannwhitney` `{column=bmi, group_column=diabetes}`
**Actual response:** `status=EXC` `ValueError("could not convert string to float: '34,3'")` — reproduces verbatim; locale-decimal BMI cell `'34,3'` is passed to `astype(float)` unconverted.
**Verdict reason:** Endpoint has no locale-aware coercion and raises on the first comma-decimal BMI cell.

## [CONFIRM] kimi:Kruskal-Wallis (bmi ~ nyha) fails
**Verified by:** `POST /api/stats/kruskal` `{column=bmi, group_column=nyha}`
**Actual response:** `status=EXC` `ValueError("could not convert string to float: '34,3'")` — same comma-decimal crash as Mann-Whitney; nonparametric helpers share the unguarded `astype(float)` path.
**Verdict reason:** Kruskal-Wallis fails for the identical reason (no BMI locale coercion).

## [CONFIRM] kimi:Mantel-Haenszel endpoint fails on cohort_test.csv
**Verified by:** `POST /api/categorical/mantel_haenszel` `{row_col=event, col_col=diabetes, strata_col=sex}`
**Actual response:** `status=EXC` `ValueError('Out of range float values are not JSON compliant')` — Mantel-Haenszel common-OR estimate / variance produces a non-finite float during JSON serialization (sparse strata × dirty sex levels create zero cells → `denom /= n²·(n-1)` divide warning → inf).
**Verdict reason:** Endpoint fails identically; no MH result is returned.

## [CONFIRM] codex:Linear regression 500s on comma-decimal BMI values
**Verified by:** `POST /api/models/linear` `{outcome=bmi, predictors=[age, sex, ldl]}`
**Actual response:** `status=EXC` `ValueError("could not convert string to float: '25,9'")` — TestClient raises before any JSON is produced; the first comma-decimal BMI cell crashes `astype(float)`.
**Verdict reason:** Linear regression cannot fit bmi as outcome on this cohort; same unguarded locale-decimal path as the nonparametric endpoints.

## [CONFIRM] codex:Logistic models silently create bogus predictors for dirty sex levels
**Verified by:** `POST /api/models/logistic` and `POST /api/models/logistic_table` with `event ~ age + sex + ldl + nyha`
**Actual response:** both `status=200`. Coefficient terms = `['const','age','ldl','nyha','sex_Female','sex_M','sex_x']`. `sex_Female` (n=1 row): `B=-76.78, OR=4.5258e-34, OR_CI=[0.0, null], p≈1.0`. `sex_x` (n=1 row): `B=-77.99, OR=1.3526e-34, OR_CI=[0.0, null], p≈1.0`. Logistic OR table reports the same `sex_Female`/`sex_x` rows with `multi_or≈3.5e-34` and capped `CI=[0.0, 9999.0]`. Backend also flags `separation_risk: critical, max|coef|≈78.0`. Independent statsmodels on valid `sex∈{M,F}` rows fits `n=86` with only `sex_M OR=1.291`.
**Verdict reason:** Dirty one-row sex levels are dummies-encoded as real model terms with bogus near-zero ORs and overflowed CIs instead of being normalized/excluded.

## [CONFIRM] codex:Fine-Gray accepts and plots the negative follow-up time
**Verified by:** `POST /api/survival_advanced/fine_gray` `{duration_col=fu_days, event_col=event, event_of_interest=1, group_col=nyha, predictors=[age,ldl]}`
**Actual response:** `status=200`, `n=98`, `n_excluded=null`. `plot.data[].x` min across traces = **`-10.0`** (the impossible `fu_days=-10` row for P100 is included); `regression_result.n=92`. The string `"-10"` appears in the serialized body. A duration-guarded fit would drop that row (CIF n would be 97, regression n 91).
**Verdict reason:** Fine-Gray accepts and plots the negative duration, unlike KM/Cox/RMST which reject the whole cohort for the same value.

## [CONFIRM] codex:Survival external validation uses binary calibration and omits O/E
**Verified by:** Derived `cox_lp_same_cohort` from a lifelines Cox fit on the cleaned cohort (dev C-index `0.5346`, n=89), then `POST /api/survival_advanced/external_validation` `{duration_col=fu_days_num, event_col=event_num, predicted_lp_col=cox_lp_same_cohort, dev_metrics={c_index=0.5346, calibration_slope=1.0}}`
**Actual response:** `status=200`. `validation_c_index=0.5346` (matches dev discrimination), but `validation_calibration_slope=1.4539`, `validation_calibration_intercept=-0.3437`. `performance_vs_dev` keys = `['c_index_drop','calibration_slope_shift']` only. No O/E-related key anywhere (`oe_ratio_present=false`, `oe_in_nested=false`). Same-cohort validation of a model's own LP should give slope ≈ 1.0; the reported slope ~1.45 and the absence of O/E are consistent with the binary-logistic calibration hypothesis.
**Verdict reason:** External validation reports matching discrimination but a wrong calibration slope (~1.45 instead of ~1.0) and provides no observed/expected ratio, confirming binary-calibration misuse and missing O/E.

done zcode-xcheck verified=9/0/0

