## [CRITICAL] Linear regression 500s on comma-decimal BMI values
**Where:** Models > Regression > Linear
**Steps:** 1) Load `qa/cohort_test.csv` with `boot()`. 2) POST `/api/models/linear` with `outcome="bmi"` and `predictors=["age","sex","ldl"]`.
**Expected:** The endpoint should either coerce locale comma decimals (`25,9`, `34,3`, `20,9`) or exclude/report those rows in `n_excluded`, then return an OLS result.
**Actual:** TestClient raises `ValueError("could not convert string to float: '25,9'")`; no JSON response is returned.
**Evidence:** Backend had 87 complete rows after listwise missing handling but still retained 3 comma-decimal BMI cells that then crashed `astype(float)`. Independent statsmodels OLS on a locale-clean copy with valid sex levels fits `n=85`, `R2=0.015674`; uSTAT returns no R2.
**Hypothesis (optional):** `apply_imputation()` runs before numeric locale normalization, so non-missing comma-decimal strings survive into `.astype(float)`.

## [HIGH] Linear diagnostics, polynomial regression, and LMM share the same BMI coercion crash
**Where:** Models > Regression > Linear diagnostics / Polynomial / LMM
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/models/linear_diag` with `bmi ~ age + sex + ldl`. 3) POST `/api/models/polynomial` with `outcome="bmi"`, `predictor="age"`, `degree=2`, `covariates=["ldl"]`. 4) POST `/api/models/lmm` with `bmi ~ age + (1|nyha)`.
**Expected:** These model helpers should use the same numeric coercion/exclusion rules as the main model and return diagnostics/model output with clear `n_excluded`.
**Actual:** `linear_diag` and `polynomial` raise `ValueError("could not convert string to float: '25,9'")`; LMM raises `ValueError("endog has evaluated to an array with multiple columns that has shape (95, 73)...")`.
**Evidence:** Independent clean fits are available: polynomial/OLS basis fits after comma conversion; MixedLM on locale-clean `bmi ~ age + (1|nyha)` fits `n=95`, `n_groups=4`, age coefficient `-0.330544`, random-effect variance `7.337567`, residual variance `9966.916139`.
**Hypothesis (optional):** Each endpoint has local `.astype(float)` or formula handling and bypasses shared robust numeric parsing.

## [HIGH] Gamma GLM crashes instead of fitting a positive continuous outcome
**Where:** Models > Regression > Gamma GLM
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/models/gamma` with `outcome="bmi"`, `predictors=["age","sex","ldl"]`, `link="log"`.
**Expected:** BMI is positive after parsing; bad/missing BMI cells should be excluded or coerced with `n_excluded`.
**Actual:** TestClient raises `ValueError("The first guess on the deviance function returned a nan...")`.
**Evidence:** Independent statsmodels Gamma(log) on the locale-clean complete cases fits `n=85`, `AIC=919.085051`; uSTAT returns a server exception. The same three comma-decimal BMI cells remain non-null before the endpoint converts `y` to numeric, producing NaNs inside GLM.
**Hypothesis (optional):** The endpoint validates positivity on `y.dropna()` but does not drop the NaNs introduced by `pd.to_numeric(..., errors="coerce")` before fitting.

## [HIGH] Ordinal logistic 500s on NYHA instead of returning a Brant test
**Where:** Models > Regression > Ordinal logistic
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/models/ordinal` with `outcome="nyha"` and `predictors=["age","sex","ldl"]`.
**Expected:** The endpoint should fit the proportional-odds model and return coefficients plus `brant_proportional_odds`, or reject dirty categorical values with a 4xx.
**Actual:** TestClient raises `OverflowError("math range error")`; no Brant result is returned.
**Evidence:** Independent statsmodels `OrderedModel` after dropping invalid sex levels fits `n=86`, `AIC=218.898531`, with parameters `age=0.003564`, `ldl=0.008801`, `sex_M=-0.116975`; uSTAT returns no model output.
**Hypothesis (optional):** Dirty category levels and/or unstable coefficients are exponentiated without overflow-safe handling while building OR CIs.

## [HIGH] Poisson and negative-binomial models crash during JSON encoding on sparse categorical levels
**Where:** Models > Regression > Poisson / Negative binomial
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/models/poisson` with `event ~ age + sex + ldl + nyha`. 3) POST `/api/models/negbinom` with the same payload.
**Expected:** Count models should return finite JSON-safe estimates or sanitize infinite CIs to `null`/warnings.
**Actual:** Both endpoints raise `ValueError("Out of range float values are not JSON compliant")`.
**Evidence:** Independent clean fits after dropping invalid sex levels complete: Poisson `n=86`, `AIC=142.847942`, `IRR_nyha=1.303711`; NegBinom `n=86`, `AIC=156.775828`, `IRR_nyha=1.289220`. uSTAT overflows CIs for the one-row dirty `sex_Female`/`sex_x` levels and then fails JSON serialization.
**Hypothesis (optional):** Unlike Cox helpers, GLM coefficient serialization does not pass estimates/CIs through a finite-value sanitizer.

## [CRITICAL] Logistic models silently create bogus predictors for dirty sex levels
**Where:** Models > Regression > Logistic and OR table
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/models/logistic` or `/api/models/logistic_table` with `event ~ age + sex + ldl + nyha`.
**Expected:** The app should normalize `Female` to `F`, reject/flag `x`, or count those rows as excluded rather than treating them as real model levels.
**Actual:** The model returns 200 with dummy variables `sex_Female` and `sex_x`, each based on a single dirty row, and publishes near-zero ORs.
**Evidence:** uSTAT logistic response: `n=88`, `sex_Female OR=4.5258469635e-34`, `sex_x OR=1.3525691587e-34`, with huge/overflowed CIs. Independent statsmodels reproduces those bogus rows only when dirty levels are kept (`n=88`); after valid-level filtering the clean model is `n=86` and contains only `sex_M OR=1.2905692062`.
**Hypothesis (optional):** Categorical model encoding has no data-dictionary/level validation before `pd.get_dummies()`.

## [MEDIUM] OR table reports `n_total=4` for a 100-row dataset
**Where:** Models > Regression > OR/HR table > Logistic OR table
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/models/logistic_table` with `event ~ age + sex + ldl + nyha`.
**Expected:** Count fields should distinguish dataset rows, analyzed rows, excluded rows, and number of model terms without overloading names.
**Actual:** Response reports `n=88`, `n_excluded=12`, `n_multi=4`, and `n_total=4`.
**Evidence:** The uploaded CSV has 100 data rows; the model analyzed 88 rows. `n_total=4` appears to be the number of requested predictors, not total rows. The same response table has 6 coefficient rows after dummy expansion.
**Hypothesis (optional):** The endpoint reuses `n_total` for predictor count or selected-variable count.

## [HIGH] Main survival endpoints reject the entire cohort for one negative follow-up value
**Where:** Models > Survival Advanced > KM / Cox PH / Time-horizon HR / Cox-RCS / Cox uni+multi / RMST
**Steps:** 1) Load `qa/cohort_test.csv`. 2) Run `/api/models/survival/km`, `/api/models/survival/cox`, `/api/models/survival/cox_horizons`, `/api/models/survival/cox_uni_multi`, `/api/models/survival/cox_rcs`, and `/api/survival_advanced/rmst` with `fu_days` and `event`.
**Expected:** One impossible duration should be excluded with an explicit `n_excluded` warning, allowing the remaining survival analysis to run.
**Actual:** The endpoints return 422 such as `"Duration column contains negative values"` or `"Negative durations are not allowed"` and produce no KM/Cox/RMST numbers.
**Evidence:** Independent lifelines on the same cohort after excluding missing/negative survival rows gives `n=89`, multigroup NYHA log-rank `p=0.543962`, Cox age per 10-year HR `0.986660`, and RMST at 365 days `322.917215`. uSTAT returns no values for these analyses.
**Hypothesis (optional):** Duration validation is implemented as all-or-nothing rather than row-level invalid-data exclusion.

## [CRITICAL] Fine-Gray accepts and plots the negative follow-up time
**Where:** Models > Survival Advanced > Fine-Gray
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/survival_advanced/fine_gray` with `duration_col="fu_days"`, `event_col="event"`, `event_of_interest=1`, `group_col="nyha"`, and predictors `["age","ldl"]`.
**Expected:** Fine-Gray should enforce the same nonnegative duration rule as KM/Cox/RMST or exclude the bad row with a warning.
**Actual:** Response is 200 and includes the negative time in the CIF curve.
**Evidence:** uSTAT reports `n=98` for grouped CIF and the group 1 curve starts with `x=-10.0`. Its regression block reports `n=92`. A clean duration rule would drop that impossible row; with group-only complete cases the count is 97 instead of 98, and with predictor complete cases it is 91 instead of 92.
**Hypothesis (optional):** `fit_fine_gray()` coerces/drops missing values but never checks `duration >= 0`.

## [HIGH] Cox diagnostics can pass on a model the main Cox endpoint refuses to fit
**Where:** Models > Diagnostics > Cox diagnostics
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/model_diagnostics/cox_diagnostics` with `fu_days`, `event`, and predictors `["age","ldl","nyha"]`. 3) POST the same model to `/api/models/survival/cox`.
**Expected:** Diagnostics and the fitted model should use the same analysis sample and invalid-duration policy.
**Actual:** Cox diagnostics returns 200 after silently filtering to positive durations, while the main Cox endpoint returns 422 for the negative duration.
**Evidence:** Independent clean survival sample is `n=89`, events `37`; main Cox returns `"Duration column contains negative values."` Diagnostics returns a successful assumptions payload without surfacing that it filtered the row differently from the model endpoint.
**Hypothesis (optional):** Diagnostics uses `df = df[df[duration_col] > 0]`, while the model endpoint treats any negative value as fatal.

## [CRITICAL] Survival external validation uses binary calibration and omits O/E
**Where:** Models > Validation > External validation (survival)
**Steps:** 1) Load `qa/cohort_test.csv`. 2) Derive `cox_lp_same_cohort` from a Cox model fit on the same cleaned cohort (`fu_days`, `event`, `age`, `ldl`, `nyha`). 3) POST `/api/survival_advanced/external_validation` with that LP as `predicted_lp_col`.
**Expected:** Same-cohort validation of the development Cox LP should give calibration slope approximately 1 and should report survival O/E at a stated time horizon.
**Actual:** Response reports matching discrimination but calibration slope `1.4539`, intercept `-0.3437`, and no O/E ratio.
**Evidence:** Independent lifelines Cox model on the same cleaned cohort gives development C-index `0.534562`; refitting Cox on its own LP gives calibration slope `1.000000048`. uSTAT external validation reports `validation_c_index=0.5346` but `validation_calibration_slope=1.4539` and `performance_vs_dev.calibration_slope_shift=0.4539`.
**Hypothesis (optional):** `evaluate_external_validation()` calls logistic `compute_calibration_slope_intercept()` on the event indicator, ignoring time and censoring; O/E is not implemented for survival validation.

## [HIGH] GEE fails on locale-decimal BMI although the cleaned model fits
**Where:** Models > Regression > GEE
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/models/gee` with `outcome="bmi"`, `predictors=["age","ldl"]`, `group_col="nyha"`, `family="gaussian"`, and `cov_struct="exchangeable"`.
**Expected:** GEE should apply the same numeric parsing/exclusion policy as other model endpoints and return the analyzed row count, clusters, and coefficients.
**Actual:** Response is 422 with only `"Gee failed. Please check your data and predictors."`
**Evidence:** Rerun log `qa/logs/codex_model_audit_rerun.log` shows the endpoint 422. Independent statsmodels GEE after comma-decimal conversion and missing-row exclusion fits `n=87`, `clusters=4`, `age=-0.297781`, `ldl=-0.010003`.
**Hypothesis (optional):** `pd.to_numeric(..., errors="coerce")` is applied to the outcome after listwise imputation, but the rows newly converted to NaN are not dropped before `sm.GEE()`.

## [HIGH] Stepwise logistic fails instead of dropping unstable dirty categorical levels
**Where:** Models > Regression > Stepwise selection
**Steps:** 1) Load `qa/cohort_test.csv`. 2) POST `/api/models/stepwise` with `model_type="logistic"`, `outcome="event"`, `candidates=["age","sex","ldl","nyha"]`, and `direction="both"`.
**Expected:** Stepwise selection should either normalize/reject dirty `sex` levels before fitting or return a selected model with a clear analyzed sample.
**Actual:** Response is 422: `"Stepwise selection failed to converge. Consider increasing iterations or simplifying the model."`
**Evidence:** The same dataset has one-row `sex="Female"` and `sex="x"` levels. Independent statsmodels on valid sex levels fits the full model at `n=86`, `AIC=123.712856`; backward p-removal selects `nyha` alone with `AIC=118.034714`. The endpoint returns no trace or selected predictors.
**Hypothesis (optional):** Stepwise encodes categorical predictors as integer category codes, then helper fits encounter separation/instability from dirty levels and collapse to a generic convergence error.
