# Grok-composer brief — Data ingest + Tests + Correlation + ROC + Visual + Meta + TS

Read `/Users/yh/Documents/projects/wiz3/qa/briefs/common.md` first.

You now cover **two merged slices** (originally Kimi + Grok-composer); ZCode +
Kimi are sitting this round out. Take it as one large pass.

## Scope

| Group | Endpoints (consult MANUAL.md + backend/routers/) |
|-------|-------------------------------------------------|
| Upload + type detection | `POST /api/upload/`, `GET /api/sessions/{sid}` |
| Dictionary / type override / decimals | `routers/session.py` |
| Compute (formula, transform, recode, clinical) | `routers/compute.py` |
| Missing audit + MICE | `routers/missing_data.py`, `routers/survival_advanced.py#mice` |
| Hypothesis tests + repeated + categorical + Bayesian + reliability + factor + non-inferiority + gatekeeping | `routers/stats/{inferential,nonparametric,correlation}.py`, `routers/{repeated,advanced_anova,reliability,agreement,bayesian,multiplicity,factor}.py` |
| Correlation + ROC + DeLong + multi + combined | `routers/stats/{correlation,nonparametric}.py` (ROC lives in nonparametric) |
| Summary / descriptive / histogram / boxplot / Q-Q | `routers/stats/descriptive.py` |
| Table 1 | `routers/pub_tables.py`, `services/journal_formatter.py` |
| Visual: charts, subgroup bar, forest, added value | `routers/charts.py`, plus the forest builder |
| Meta-analysis: pool, subgroup, regression, bias | `routers/meta.py` |
| Time series ARIMA / STL / stationarity | `routers/timeseries.py` |
| Reporting export | `routers/pub_export.py` |

## What to probe

### Data ingest + cleaning

- Comma-decimal `bmi`: kept numeric? Goes to mean/SD?
- `bmi="999"`, `bmi="n/a"`, `ldl="NA"` — how are they typed?
- `age=-5`, `age=250`, `fu_days=-10` — any guard?
- `sex` mixed coding ("M", "F", "x", "Female", blank) — how many cats?
- `admission_date` mixed formats — Parse-as-date end-to-end.
- Recode rule that matches no rows — warned?
- MICE imputation on `ldl` — converges with this many gaps?

### Tests + correlation + ROC

- Independent t-test `age ~ sex` with bad sex codes — n_per_group?
- ANOVA `bmi ~ nyha` — same.
- Mann-Whitney `bmi ~ sex` — comma-decimals reach the test?
- Kruskal-Wallis `bmi ~ nyha`.
- Chi² `diabetes × sex` — extra cells from "x" / "Female"?
- Fisher exact on a designed 2×2 with a zero cell.
- Cronbach α on `bmi, ldl, sbp` — NaN-resistant?
- ICC on the same.
- Pearson / Spearman `age × bmi` — agree to 4 dp with scipy.
- ROC `ldl` vs `event` — AUC matches sklearn `roc_auc_score`.
- DeLong compare AUC of `ldl` vs `sbp`.

### Summary + Table 1 + Visual + Meta + TS

- Descriptive on `age, bmi, ldl, sbp, fu_days` — bad cells leaking into mean/SD?
- Skewness/kurtosis match scipy.
- Histogram bins sensible despite outliers.
- Table 1 by `sex` — three columns or one + warning?
- Subgroup bar of `event` rate by `nyha` — ordinal order preserved?
- Forest builder smoke: 4 rows with HR + CI, one negative bound → log-axis handling.
- Meta-analysis: write 5 synthetic rows (effect+var) → pooled (random vs fixed), I².
- Meta-regression with a moderator, bubble shape correct.
- Publication-bias trim-and-fill on funnel-asymmetric set.
- ARIMA on `age` ordered by `patient_id`.
- STL on a flat series — crash or sane.
- Stationarity ADF/KPSS p-values + ACF/PACF lengths.

## Recompute key numbers independently

Where a result looks off, compute it with scipy/statsmodels/sklearn on the same
data and quote both numbers in the finding.

## Output

`/Users/yh/Documents/projects/wiz3/qa/findings/grok-composer.md`
</content>

---

## Plus the Causal + Power slice (Grok-build sat out)

Also exercise the causal stack and power endpoints since the dedicated agent could not finish:

| Group | Endpoints |
|-------|-----------|
| PSM | `POST /api/psm/*` |
| IPTW | `POST /api/iptw/*` |
| Causal+: IV/2SLS, mediation, target-trial, DiD, RDD, DAG | `POST /api/causal/{iv_2sls,mediation,target_trial,did,rdd,dag_adjustment}` |
| Decision-curve analysis | `POST /api/decision_curve/*` |
| Power / sample size (logistic, Cox especially) | `POST /api/stats/power` |
| Causal sensitivity (E-value / Q-bias) | `POST /api/survival_advanced/{evalue,causal_sensitivity}` |

Verify two power numbers by hand: Hsieh logistic n for OR=2, p_event=0.3, power=0.8; Schoenfeld Cox events for HR=1.5, event_rate=0.35.
