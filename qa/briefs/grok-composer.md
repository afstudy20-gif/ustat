# Grok-composer brief — Table 1 + Summary + Visual + Meta + Time Series

Read `/Users/yh/Documents/projects/wiz3/qa/briefs/common.md` first.

## Scope

| Group | Endpoints |
|-------|-----------|
| Summary — descriptive, histogram, boxplot, violin, Q-Q | `POST /api/stats/descriptive/*` |
| Weighted descriptives (survey weights) | weighted endpoint |
| Table 1 (clinical baseline) | `POST /api/pub_tables/table1` or `routers/pub_tables.py` |
| Visual: Models & Diagnostics, Charts (custom plots), Subgroup Bar, Forest plot, Added Predictive Value | `POST /api/charts/*`, forest endpoints |
| Meta-analysis: pooled, subgroup, meta-regression, publication bias | `POST /api/meta/{analyze,subgroup,regression,bias}` |
| Time series: ARIMA / STL / stationarity (ADF, KPSS, ACF, PACF) | `POST /api/timeseries/{arima,decompose,stationarity}` |
| Reporting: pub_export, journal_formatter | `routers/pub_export.py` |

## What to probe specifically

- Descriptive on `age, bmi, ldl, sbp, fu_days`: do the negative `age=-5`,
  `bmi=999`, `fu_days=-10` values leak into mean/SD? Skewness/kurtosis numbers
  match scipy?
- Histogram/Boxplot data shape: do bins look right with the impossible values?
- Table 1 by `sex` with mixed coding ("x", "Female", blank): how does the
  table present them — one column, three columns, or a warning?
- Subgroup bar of `event` proportion by `nyha`: does the bar chart x-axis
  preserve ordinal order or sort alphabetically?
- Forest plot builder: feed it a few HR rows incl. one with negative bound —
  does the log-scale handling complain or render off-page?
- Meta-analysis: build a tiny synthetic forest input (5 studies with
  effect+variance) — random vs fixed pool, I² in [0,1]?
- Meta-regression: add a moderator — bubble plot data shape correct?
- Publication-bias: trim-and-fill on a deliberately funnel-asymmetric set.
- Time series ARIMA on `age` ordered by `patient_id` (synthetic time order) —
  does ARIMA on essentially-iid data return AR(0) MA(0) sensibly?
- STL decomposition with no seasonality — does it crash or return flat
  seasonal?
- Stationarity ADF/KPSS — p-values within expected ranges; ACF/PACF arrays
  the right length?
- pub_export: produce a Word/HTML export of one regression — does it embed
  the right numbers? Are decimals respected?

## Output

`/Users/yh/Documents/projects/wiz3/qa/findings/grok-composer.md`
</content>
