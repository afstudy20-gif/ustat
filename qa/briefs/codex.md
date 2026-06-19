# Codex brief — Regression + Survival + RCS + Validation

Read `/Users/yh/Documents/projects/wiz3/qa/briefs/common.md` first.

Your slice: all model-fitting endpoints. You are the math-heavy agent — verify
numbers, not just status codes.

## Scope

| Group | Endpoints |
|-------|-----------|
| Linear / Logistic / Firth / Ordinal / Poisson / NegBinom / Gamma GLM | `POST /api/models/{linear,logistic,firth_logistic,poisson,gamma,negbinom,ordinal}` |
| OR / Firth-OR / HR uni+multi tables | `POST /api/models/{logistic_table,survival/cox_uni_multi}` |
| Linear / Cox diagnostics + assumption checks | `POST /api/models/{linear_diag}`, `POST /api/model_diagnostics/*` |
| RCS dose-response + Cox-RCS | `POST /api/models/{rcs,survival/cox_rcs}` |
| Mixed-effects (LMM) / GEE | `POST /api/models/{lmm,gee}` |
| Stepwise selection / polynomial | `POST /api/models/{stepwise,polynomial}` |
| Survival: KM / Cox / time-varying / time-horizon / RMST / Fine-Gray / LWYY / **interval-censored** / Landmark / E-value | `POST /api/models/survival/*` and `/api/survival_advanced/*` |
| Validation (internal: bootstrap optimism, k-fold) | endpoints under model validation |
| External validation (calibration, transportability) | `POST /api/survival_advanced/external_validation` |

## What to probe specifically

- Linear regression `bmi ~ age + sex + ldl`: with comma-decimals on `bmi` does
  `n_excluded` make sense and does R² agree with statsmodels OLS on a clean
  copy?
- Logistic `event ~ age + sex + ldl + nyha`: does it handle `sex="Female"` /
  `sex="x"` (extra dummies?) and rare-event tilt? Compare OR to a manual fit.
- **Brant test** on ordinal logistic with `nyha` as outcome — does the new
  test return a sensible χ² and does it correctly flag a non-PO case if you
  simulate one?
- Firth logistic: rare-event subset (e.g. only `event=1` ≤ 5) — does it still
  converge?
- KM `fu_days, event` with `nyha` as group: log-rank p — does it match
  lifelines? And does the row with `fu_days=-10` survive into the fit?
- Cox PH on the same — HR for `age` per 10-year increment; verify against
  lifelines manually.
- Cox time-horizon at [180, 365, 730]: forest output shape, HR direction,
  CIs not crossing zero in log space.
- Interval-censored: pick two numeric columns and call them L/R — does the
  Turnbull curve land monotone non-increasing? Does the Weibull regression
  recover a covariate effect on the test data?
- RMST at τ = median follow-up — agrees with lifelines `restricted_mean_survival_time`?
- E-value: pass an HR with CI crossing 1 and confirm `e_value` ≥ 1 (else flag).
- LMM `bmi ~ age + (1|nyha)` — converge? sane variance components?
- External validation with a "validation cohort" that's actually the same as
  development: O/E ratio should be ≈ 1.

## Output

`/Users/yh/Documents/projects/wiz3/qa/findings/codex.md`
</content>
