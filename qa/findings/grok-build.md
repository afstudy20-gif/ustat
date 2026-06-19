# grok-build audit findings

## [CRITICAL] IV/2SLS with non-numeric instrument (admission_date) causes response serialization crash
**Where:** Causal+ > IV/2SLS
**Steps:** 1) boot() qa/cohort_test.csv 2) POST /api/causal/iv_2sls with outcome=ldl, endogenous=diabetes, instruments=["admission_date"]
**Expected:** 4xx validation (instrument must be numeric) or 200 with first-stage diagnostics and a note on coercion failure.
**Actual:** Unhandled exception during response rendering: ValueError("Out of range float values are not JSON compliant").
**Evidence:** Trace shows crash inside Starlette/FastAPI JSON encoder after second-stage 2SLS; earlier steps include OLS on string-coerced design matrix producing non-finite values.
**Hypothesis (optional):** _design(pd.get_dummies on date strings) + downstream matrix ops yield inf/NaN that are not sanitized before return.

## [CRITICAL] causal_sensitivity with rosenbaum + session fields returns non-finite values leading to JSON crash
**Where:** Causal sensitivity > E-value / Q-bias / Rosenbaum
**Steps:** 1) Load cohort 2) POST /api/survival_advanced/causal_sensitivity with observed_estimate, session_id, treatment_col=diabetes, outcome_col=event, match_id_col=patient_id, rosenbaum_* params.
**Expected:** Finite numbers or clear "applicable":false with reason; always JSON-serializable response.
**Actual:** Endpoint returns 200 payload containing inf/NaN; FastAPI json response fails with "Out of range float values are not JSON compliant".
**Evidence:** Reproduced multiple times; stack points at response render after _rosenbaum_bounds or negative-control paths.
**Hypothesis (optional):** Rosenbaum bounds or partial-id components emit +/-inf when match_id_col is not a real matched-set id or data produces zero/undefined pairs.

## [HIGH] Mediation runs linear models on binary outcome without rejection
**Where:** Causal+ > Mediation
**Steps:** 1) POST /api/causal/mediation outcome="event" (0/1), treatment="diabetes", mediator="ldl" (or bmi).
**Expected:** 422 or explicit warning that outcome must be continuous; or at minimum a note that linear decomposition on binary Y is not valid.
**Actual:** 200 with acme/ade/total/proportion_mediated computed via OLS on 0/1 outcome.
**Evidence:** Response contains effects with small ACME on binary scale and bootstrap CIs; no validation error.
**Hypothesis (optional):** Mediation path only checks distinctness and row count; never validates outcome dtype/unique values.

## [HIGH] Power logistic and Cox lack support for solve_for=effect_size (min detectable OR/HR)
**Where:** Power > Logistic / Cox
**Steps:** 1) POST /api/stats/power test=logistic (or survival_cox) solve_for=effect_size power=0.4 n=50 (or 200) plus required params (p_event or event_rate).
**Expected:** Returns a finite minimum detectable OR/HR (or NaN with explanation) per brief.
**Actual:** 400 "Logistic power needs 'log_or'..." or "Cox power needs 'hr' > 0." — the effect-size solve path is absent or requires the value it is trying to solve.
**Evidence:** Direct calls with effect_size or without pre-supplying log_or/hr fail; "power" and "n" solve_for work.
**Hypothesis (optional):** Logistic/Cox branches only implement "n" and "power" arms; missing the solve_for effect_size arm present for t/ANOVA/proportion.

## [HIGH] DCA on multi-predictor model with dirty cohort returns singular-matrix error
**Where:** DCA
**Steps:** 1) POST /api/decision_curve/dca outcome=event predictors=["age","diabetes","bmi","ldl","sbp"] on cohort_test.csv.
**Expected:** Either fits after excluding bad rows and reports n_excluded, or returns informative error + partial curve.
**Actual:** 400 {"detail":"Logistic regression failed: Singular matrix"}.
**Evidence:** Single-predictor ("age") succeeds and returns curves; adding the full dirty covariate set triggers perfect separation / collinearity inside _fit_logistic.
**Hypothesis (optional):** No pre-check for separation or rank before fitting; imputation leaves enough dirty contrast to make X singular.

## [MEDIUM] DiD accepts any binary column as time_col with no temporal validation or warning
**Where:** Causal+ > DiD
**Steps:** 1) POST /api/causal/did outcome=bmi, group_col=diabetes, time_col=event (or any 0/1).
**Expected:** 422 or warning that time_col should encode a genuine pre/post temporal indicator.
**Actual:** 200 with cell_means and DiD estimate; no warning.
**Evidence:** Using "event" as time_col produced a result; later using a created fu_days-based time also succeeded.
**Hypothesis (optional):** Endpoint only validates that group/time are {0,1}; semantic correctness of "time" is not enforced.

## [MEDIUM] PSM common-support trim on dirty data yields extremely few matches (n=2)
**Where:** PSM/IPTW > Propensity Score Matching
**Steps:** 1) POST /api/models/psm treatment=diabetes covariates=[age,bmi,ldl,sbp] trim_common_support=true outcome=event.
**Expected:** Reasonable matched n after trim, or clear guidance that overlap is poor.
**Actual:** n_matched_pairs=2, n_trimmed_common_support=75, n_unmatched=21.
**Evidence:** Response includes n_matched_pairs, n_unmatched, n_trimmed; post-trim balance may be based on tiny sample.
**Hypothesis (optional):** Propensity overlap is narrow due to impossible ages, extreme bmi, missingness; trim + caliper interaction leaves almost nothing.

## [MEDIUM] IPTW warnings list often empty even when weights or effective_n are concerning
**Where:** PSM/IPTW > IPTW
**Steps:** 1) POST /api/models/iptw with various estimands/stabilize settings.
**Expected:** Warnings populated for low effective_n, extreme max weight, or truncation.
**Actual:** In several runs (unstabilized and stabilized) warnings: [] even when max_w>3 or effective_n lower.
**Evidence:** Response weight_summary shows numbers but warnings remains [].
**Hypothesis (optional):** Warning emission only triggers on non-finite or hard truncation paths; low-eff-n or moderate max not flagged.

## [MEDIUM] Causal router paths trigger pandas SettingWithCopyWarning during imputation/numeric coercion
**Where:** Causal+ (IV, mediation, DiD)
**Steps:** Exercise iv_2sls, mediation, did on the loaded session.
**Expected:** No SettingWithCopy during request handling; defensive .copy() or .loc used.
**Actual:** Multiple warnings logged: "A value is trying to be set on a copy of a slice..."
**Evidence:** Trace lines point to causal.py:82 (outcome), :83 (endogenous), :254, :589.
**Hypothesis (optional):** apply_imputation returns a view in some paths; subsequent df[col] = ... mutates parent frame.

## [LOW] Power solve-for-power with low target (0.4) and small n returns a plausible but low number without special handling note
**Where:** Power > Logistic
**Steps:** POST power solve_for=power n=30 power=0.4 effect_size=2 p_event=0.3.
**Expected:** A number <=0.5 or explicit note that the design is under-powered.
**Actual:** ~0.413 returned; result_text still uses the generic template.
**Evidence:** No crash, but no extra guidance for sub-80% scenarios as mentioned in brief.
**Hypothesis (optional):** Implementation is correct per formula; only UX copy is generic.

## [MEDIUM] E-value endpoint returns different key names than internal e_value() helper
**Where:** Causal sensitivity > E-value
**Steps:** POST /api/survival_advanced/evalue vs inspect services.causal_sensitivity.e_value.
**Expected:** Consistent public contract (e_value_point_estimate / e_value_ci).
**Actual:** Router returns "evalue_point", "evalue_ci".
**Evidence:** Successful numeric results but shape mismatch vs service docstring and brief expectations.
**Hypothesis (optional):** Router was written against a different return shape.

