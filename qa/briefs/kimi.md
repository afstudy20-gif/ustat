# Kimi brief — Hypothesis tests + Categorical + Correlation + ROC

Read `/Users/yh/Documents/projects/wiz3/qa/briefs/common.md` first.

Your slice: every classical hypothesis test + correlation + ROC.

## Scope (see MANUAL.md for full list)

| Group | Endpoints |
|-------|-----------|
| t-tests / ANOVA / ANCOVA / MANCOVA / two-way ANOVA | `POST /api/stats/{ttest,anova}`, `POST /api/advanced_anova/{ancova,two_way_anova,mancova}` |
| Mann-Whitney / Kruskal-Wallis / Jonckheere-Terpstra | `POST /api/stats/{mannwhitney,kruskal,jonckheere_terpstra}` |
| Repeated measures: paired t / Wilcoxon signed-rank / RM-ANOVA / Friedman / mixed ANOVA | `POST /api/repeated/{paired_ttest,wilcoxon_signed_rank,rm_anova,friedman,mixed_anova}` |
| Categorical: chi² / Fisher / binomial / proportions / McNemar / Cochran Q / Mantel-Haenszel / CA trend | `POST /api/stats/{chisquare,fisher}` + `routers/categorical.py` |
| Reliability: Cronbach α / ICC / Cohen's κ / Fleiss κ | `POST /api/reliability/cronbach`, `routers/agreement.py` |
| Non-inferiority + TOST | `POST /api/stats/{tost,noninferiority}` |
| Gatekeeping / multiplicity | `POST /api/multiplicity/gatekeeping` |
| Factor analysis / PCA | `POST /api/factor/factor_pca` |
| Bayesian (BF t-test, correlation, regression) | `POST /api/bayesian` |
| Correlation: Pearson / Spearman / Kendall | `POST /api/stats/correlation/*` (see `backend/routers/stats/correlation.py`) |
| ROC: single, DeLong compare, multi-curve, combined model | `POST /api/stats/{roc,roc_compare,roc_multi_compare,roc_combined}` |

## What to probe specifically

- Two-sample t-test on `age ~ sex`: do the rows with `sex=""`, `"x"`, `"Female"`
  get dropped, miscategorised, or counted? Does `n` make sense?
- Chi² on `diabetes × sex`: does the bad-coded `"x"`, `"Female"` create extra
  cells or break the table?
- Mann-Whitney on `bmi ~ sex`: comma-decimals — does the test see "30,6" or
  drop it?
- Kruskal-Wallis on `bmi ~ nyha`: 4-level ordinal with 2 missings — is `n_per_group`
  reported correctly?
- ROC of `ldl` vs `event`: when `ldl` is "NA" / blank, are those rows dropped
  before the AUC, or do they break the curve?
- DeLong: compare `ldl` AUC vs `sbp` AUC — does the p-value match scipy's
  expectation?
- Cronbach α on `bmi, ldl, sbp`: bad data — does it return NaN or a wrong
  positive number?
- Bayesian t-test: same `age ~ sex` — does the BF have the right sign?
- Fisher exact small-cell: design a 2×2 with a zero cell and run it.

## Output

`/Users/yh/Documents/projects/wiz3/qa/findings/kimi.md`
</content>
