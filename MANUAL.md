# uSTAT — LLM Reference Manual

**Purpose.** This file is written for a chat assistant (LLM). A user will say
something like *"I uploaded my data to uSTAT — which tests should I run and how
do I run them in uSTAT?"* Use this document to (1) recommend the right analysis
for their data and question, and (2) give exact, click-by-click steps for the
uSTAT web app at <https://ustat.drtr.uk>.

uSTAT is a browser-based clinical/biostatistics app. Data lives in server RAM
only, is never written to disk, and is auto-discarded ~30 min after the last
action. It is **not** a validated medical device — always tell the user to
confirm important results against SPSS/R/Stata/SAS.

> How to use this manual as an LLM:
> 1. Map the user's question + variable types to a row in the **Master Test
>    Index** below.
> 2. Open the matching **per-tab section** for the exact navigation path,
>    required variables, and steps.
> 3. Every panel in uSTAT also shows its own *When to use / Assumptions /
>    How to read* card — reassure the user those notes are on-screen.

---

## 1. How uSTAT is organized

After the user uploads a file, the top navigation bar shows these **tabs**
(left → right). Some tabs contain a second row of **sub-tabs**.

| Tab | Sub-tabs | What lives here |
|-----|----------|-----------------|
| **Data** | — | Spreadsheet view, variable typing, cleaning, Select Cases, Filter, missing badges |
| **Summary** | Descriptive · Weighted | Descriptive stats, histogram, boxplot, violin, Q-Q |
| **Table** (Table 1) | — | Clinical baseline "Table 1" by group |
| **Tests** | Hypothesis · Repeated Measures · Categorical · Reliability · Non-Inferiority · Gatekeeping · Factor Analysis · Bayesian Statistics | Classic hypothesis tests |
| **Correlation** | — | Pearson / Spearman / Kendall, correlation matrix |
| **ROC** | — | ROC curve, AUC, DeLong comparison, combined model |
| **Models** | Regression · Survival Advanced · Restricted Cubic Spline · Machine Learning · Time Series · Validation | Regression, survival, ML, forecasting, validation |
| **PSM** | — | Propensity-score matching |
| **IPTW** | — | Inverse-probability-of-treatment weighting |
| **Causal+** | — | IV/2SLS, mediation, target trial, DiD, RDD, DAG |
| **DCA** | — | Decision-curve analysis (net benefit) |
| **Meta** | — | Meta-analysis, subgroup, meta-regression, publication bias |
| **Missing** | — | Missing-data audit + multiple imputation (MICE) |
| **Visual** | Models & Diagnostics · Charts · Subgroup Bar · Forest plot · Added Predictive Value | Visual model combos, custom charts, forest builder |
| **Compute** | — | Recode, formula columns, clinical calculators (BMI, eGFR, CHA₂DS₂-VASc) |
| **Power** | — | Power / sample-size (no data needed; also on the splash screen) |

**Global search:** the search box at the top ("Search tests / models…") jumps
straight to any analysis by name or alias — tell users they can type e.g.
"Cox", "kappa", "mediation", or Turkish aliases ("lojistik", "eksik veri").

---

## 2. Master Test Index

Find the analysis, go to the tab/sub-tab, prepare the listed variable types.
*(num = numeric, cat = categorical, ord = ordered categorical; mark types in
the Data tab by clicking a column's type badge or via the Dictionary modal.)*

### Descriptive & baseline
| Analysis | Tab → sub-tab | Needs |
|----------|---------------|-------|
| Descriptive statistics | Summary → Descriptive | any |
| Histogram / Boxplot / Violin / Q-Q | Summary → Descriptive | 1 num (+ optional cat group) |
| Weighted descriptives (survey weights) | Summary → Weighted | num + weight col |
| Table 1 (clinical baseline) | Table | a grouping cat + any vars |

### Group comparisons (hypothesis tests)
| Analysis | Tab → sub-tab | Needs |
|----------|---------------|-------|
| One-sample t-test | Tests → Hypothesis | 1 num |
| Independent (two-sample) t-test | Tests → Hypothesis | 1 num + 1 cat (2 levels) |
| One-way ANOVA | Tests → Hypothesis | 1 num + 1 cat (≥3 levels) |
| Two-way ANOVA / ANCOVA / MANCOVA | Tests → Hypothesis | num outcome(s) + cat factors (+ num covariate) |
| Mann-Whitney U | Tests → Hypothesis | 1 num/ord + 1 cat (2 levels) |
| Kruskal-Wallis | Tests → Hypothesis | 1 num/ord + 1 cat (≥3 levels) |
| Jonckheere-Terpstra (ordered trend) | Tests → Hypothesis | 1 num + 1 ord group |
| Paired t-test / Wilcoxon signed-rank | Tests → Repeated Measures | 2 paired num cols |
| Repeated-measures / Mixed ANOVA / Friedman | Tests → Repeated Measures | ≥2 paired num cols (+ between factor) |

### Categorical associations
| Analysis | Tab → sub-tab | Needs |
|----------|---------------|-------|
| Chi-square / Fisher's exact | Tests → Categorical (or → Hypothesis) | 2 cat |
| Binomial / one- & two-proportion z | Tests → Categorical | 1–2 cat |
| McNemar / Cochran's Q | Tests → Categorical | paired cat |
| Mantel-Haenszel (stratified OR) | Tests → Categorical | 2 cat + 1 stratum cat |
| Cochran-Armitage trend | Tests → Categorical | ordered cat + binary |

### Agreement / reliability / structure
| Analysis | Tab → sub-tab | Needs |
|----------|---------------|-------|
| Cronbach's α | Tests → Reliability | ≥2 num items |
| ICC (intraclass correlation) | Tests → Reliability | ≥2 num rater cols |
| Fleiss' / Cohen's κ | Tests → Reliability | rater cat cols |
| Bland-Altman / Deming / Passing-Bablok | (Agreement — via search) | 2 num methods |
| PCA / Factor analysis | Tests → Factor Analysis | ≥3 num |
| Bayesian t-test / correlation / regression | Tests → Bayesian Statistics | per test |
| Gatekeeping (multiplicity, Hochberg/Holm) | Tests → Gatekeeping | a set of p-values |
| Non-inferiority (RR/RD/OR/mean + margin) | Tests → Non-Inferiority | per design |

### Correlation
| Analysis | Tab | Needs |
|----------|-----|-------|
| Pearson / Spearman / Kendall + matrix | Correlation | ≥2 num/ord |

### Diagnostic accuracy
| Analysis | Tab | Needs |
|----------|-----|-------|
| ROC curve + AUC | ROC | 1 num score + 1 binary outcome |
| DeLong AUC comparison / Multi-curve / Combined model | ROC | ≥2 num scores + 1 binary outcome |

### Regression
| Analysis | Tab → sub-tab | Needs |
|----------|---------------|-------|
| Linear regression | Models → Regression | num outcome + predictors |
| Logistic / Firth logistic | Models → Regression | binary outcome + predictors |
| OR table (uni + multivariable) | Models → Regression | binary outcome + predictors |
| Ordinal logistic (+ **Brant** prop-odds test) | Models → Regression | ordered-cat outcome + predictors |
| Poisson / Negative binomial / Gamma GLM | Models → Regression | count/positive outcome + predictors |
| Polynomial / Stepwise selection | Models → Regression | num outcome + predictors |
| Mixed-effects (LMM) / GEE | Models → Regression | outcome + predictors + cluster id |
| Restricted cubic spline (dose-response) | Models → Restricted Cubic Spline | num predictor + outcome |

### Survival (time-to-event)
| Analysis | Tab → sub-tab | Needs |
|----------|---------------|-------|
| Kaplan-Meier + log-rank | Models → Survival Advanced (Kaplan-Meier) | time num + event 0/1 (+ group) |
| Cox proportional hazards (+ Schoenfeld) | Models → Survival Advanced (Cox PH) | time + event + predictors |
| Time-horizon HR forest | Models → Survival Advanced (Time-horizon HR) | time + event + predictor |
| Landmark analysis | Models → Survival Advanced (Landmark) | time + event + predictors |
| RMST (restricted mean survival) | Models → Survival Advanced (RMST) | time + event + group |
| Fine-Gray competing risks (sHR/CIF) | Models → Survival Advanced (Fine-Gray) | time + event (≥2 causes) |
| Recurrent events (LWYY) | Models → Survival Advanced (Recurrent) | id + start/stop + event |
| **Interval-censored** (Turnbull NPMLE + Weibull) | Models → Survival Advanced (Interval-censored) | lower-bound + upper-bound num (+ covariates) |
| E-value (unmeasured confounding) | Models → Survival Advanced (E-value) | an estimate + CI |
| RCS-Cox (multivariable spline survival) | Models → Restricted Cubic Spline | time + event + spline var |

### Causal inference
| Analysis | Tab | Needs |
|----------|-----|-------|
| Propensity-score matching | PSM | treatment cat + covariates + outcome |
| IPTW (inverse-probability weighting) | IPTW | treatment cat + covariates + outcome |
| Instrumental variable (2SLS) | Causal+ | outcome + endogenous + instrument(s) + covariates |
| Causal mediation (ACME/ADE) | Causal+ | treatment + mediator + outcome |
| Target-trial emulation | Causal+ | eligibility + treatment + outcome |
| Difference-in-Differences | Causal+ | outcome + time + group |
| Regression discontinuity (RDD) | Causal+ | outcome + running var + cutoff |
| DAG backdoor analysis | Causal+ | a user-drawn DAG |

### Prediction model evaluation
| Analysis | Tab | Needs |
|----------|-----|-------|
| Decision-curve analysis (net benefit) | DCA | predicted risks/predictors + binary outcome |
| Internal validation (bootstrap optimism) | Models → Validation | model spec |
| External validation (calibration, O/E, DeLong) | Models → Validation | model + validation cols |
| Added predictive value (ΔAUC, NRI, IDI) | Visual → Added Predictive Value | base vs extended predictors + outcome |
| Random Forest / Gradient Boosting / feature importance | Models → Machine Learning | outcome + predictors |

### Meta-analysis & forecasting
| Analysis | Tab | Needs |
|----------|-----|-------|
| Meta-analysis (random/fixed) + subgroup + meta-regression | Meta | per-study effect + variance (or 2×2) |
| Publication bias (Egger/Begg/funnel/trim-fill) | Meta | study effects |
| ARIMA/SARIMA, STL decomposition, stationarity (ADF/KPSS) | Models → Time Series | a time-ordered num series |

### Design & data tools
| Analysis | Tab | Needs |
|----------|-----|-------|
| Power / sample size (t, ANOVA, χ², proportions, **logistic, Cox**) | Power | effect size + design params (no data) |
| Missing-data audit + MICE imputation | Missing | any (works on the whole dataset) |
| Recode / formula columns / clinical calculators | Compute | source columns |

---

## 3. Tab-by-tab how-to

### 3.1 Data — load, type, clean
![Data tab](docs/manual/img/01-data.png)

- **Get here:** the **Data** tab (default after upload).
- **Set variable types:** click a column's small badge (`num` / `cat` / `txt` /
  `date`) to cycle types, or open **Dictionary** for labels + value labels.
  Correct typing matters — many pickers only list numeric or only categorical
  columns. `Ordered Categorical` (ordinal) shows up in both.
- **Missing values:** each header shows a badge like `103✕ · 13%` (count + % of
  rows). Click **⚠ Missing** to view only rows with gaps.
- **Subset:** **Select Cases** (rule-based row filter) and **Filter**
  (per-column). **+ Row / + Column**, inline edit (double-click a cell),
  Undo/Redo, Freeze columns.
- **Tell the user:** clean and type the data here *first* — every other tab
  reads these definitions.

### 3.2 Summary — descriptives & distribution plots
![Summary tab](docs/manual/img/02-summary.png)

- **Get here:** **Summary → Descriptive**. (**Weighted** sub-tab for survey
  weights.)
- **Steps:** pick a numeric variable → see mean/SD/median/IQR/range/skew, plus
  **Histogram, Boxplot, Violin, Q-Q**. Add a categorical **group** to compare
  distributions side by side.
- **Use Q-Q** to judge normality before choosing a t-test vs Mann-Whitney.

### 3.3 Table — clinical "Table 1"
![Table 1](docs/manual/img/03-table1.png)

- **Get here:** **Table** tab.
- **Steps:** choose a **grouping variable** (e.g. treatment arm) → tick the
  baseline variables → uSTAT auto-picks mean±SD vs median[IQR] vs n(%) and the
  right test per row, with a p-value column. Export to publication format.

### 3.4 Tests — hypothesis tests
![Tests tab](docs/manual/img/04-tests.png)

- **Get here:** **Tests** tab. Sub-tabs: **Hypothesis** (t/ANOVA/Mann-Whitney/
  Kruskal/Jonckheere/χ²/Fisher), **Repeated Measures** (paired t, Wilcoxon,
  RM-ANOVA, Friedman, mixed ANOVA), **Categorical** (proportions, McNemar,
  Cochran Q, Mantel-Haenszel, trend), **Reliability** (Cronbach, ICC, κ),
  **Non-Inferiority**, **Gatekeeping**, **Factor Analysis** (PCA), **Bayesian
  Statistics**.
- **Steps:** pick the test in the left list → assign the outcome + group
  variables in the form → **Run**. Each test shows a *When to use / Assumptions
  / How to read* card (visible on the right) — point the user to it.
- **Picking parametric vs non-parametric:** continuous + roughly normal (check
  Summary Q-Q) → t-test/ANOVA; otherwise → Mann-Whitney/Kruskal.

### 3.5 Correlation
![Correlation tab](docs/manual/img/05-correlation.png)

- **Get here:** **Correlation** tab.
- **Steps:** select ≥2 numeric/ordinal variables → choose **Pearson** (linear,
  normal), **Spearman** (monotonic/ranked), or **Kendall** → get the coefficient
  matrix, p-values, and a heatmap/scatter.

### 3.6 ROC — diagnostic accuracy
![ROC tab](docs/manual/img/06-roc.png)

- **Get here:** **ROC** tab.
- **Steps:** pick a numeric **score/marker** + a **binary outcome (0/1)** →
  AUC with 95% CI, optimal cutoff (Youden), sensitivity/specificity. Switch to
  **multi-curve** to compare several markers; **DeLong** tests whether two AUCs
  differ; **Combined model** fits a logistic combination of markers.

### 3.7 Models — Regression
![Models · Regression](docs/manual/img/07-models-regression.png)

- **Get here:** **Models → Regression**.
- **Steps:** pick the model in the left list (Linear, Logistic, Firth, OR/HR
  table, Ordinal, Poisson, …) → choose the **Outcome** → tick **Predictors** →
  options (Robust SE, imputation, interactions) → **Fit**. Output: coefficient
  table (β / OR / IRR) with 95% CI + p, model fit, and a plain-English summary.
- **Ordinal logistic** also reports the **Brant test** of the proportional-odds
  assumption (green = assumption holds, amber = violated, with the offending
  predictors named).

### 3.8 Models — Survival Advanced
![Models · Survival Advanced](docs/manual/img/08-models-survival.png)

- **Get here:** **Models → Survival Advanced**. Left list = method
  (Kaplan-Meier, Cox PH, Time-horizon HR, Landmark, RMST, Fine-Gray, Recurrent
  LWYY, **Interval-censored**, E-value).
- **Common inputs:** **Duration (time)** numeric + **Event (0/1)** + optional
  **Group**/**Stratify**. For Cox, also tick **predictors**.
- **Interval-censored** (event known only within a bracket — e.g. recurrence
  found at a scheduled scan): pick the **lower** and **upper** bound columns
  (leave the upper blank for still-event-free), optional covariates → Turnbull
  NPMLE curve + Weibull time-ratio/HR table.

### 3.9 PSM — propensity-score matching
![PSM tab](docs/manual/img/09-psm.png)

- **Get here:** **PSM** tab.
- **Steps:** choose the **treatment** (binary), the **covariates** to balance
  on, and the **outcome** → uSTAT estimates the propensity score, matches
  treated/control, and reports balance (standardized mean differences, Love
  plot) plus the matched treatment effect.

### 3.10 IPTW — inverse-probability weighting
![IPTW tab](docs/manual/img/10-iptw.png)

- **Get here:** **IPTW** tab.
- **Steps:** same inputs as PSM (treatment + covariates + outcome) but instead
  of matching it weights every subject by 1/propensity → reports balance after
  weighting and the weighted (ATE) effect. Use when you don't want to discard
  unmatched subjects.

### 3.11 Causal+ — modern causal methods
![Causal+ tab](docs/manual/img/11-causal.png)

- **Get here:** **Causal+** tab. Methods: **IV/2SLS** (instrument), **Mediation**
  (treatment→mediator→outcome, ACME/ADE), **Target-trial emulation**,
  **Difference-in-Differences**, **Regression discontinuity**, **DAG backdoor
  analysis**.
- **Steps:** pick the method → assign its specific roles (e.g. IV needs outcome
  + endogenous exposure + instrument(s) + covariates) → **Run**.

### 3.12 DCA — decision-curve analysis
![DCA tab](docs/manual/img/12-dca.png)

- **Get here:** **DCA** tab.
- **Steps:** provide predicted risks (or predictors uSTAT will model) + the
  **binary outcome** → net-benefit curve across threshold probabilities vs
  *treat-all* / *treat-none*. Shows where a model is clinically useful.

### 3.13 Meta — meta-analysis
![Meta tab](docs/manual/img/13-meta.png)

- **Get here:** **Meta** tab.
- **Steps:** supply per-study effect sizes + variances (or 2×2 counts) →
  random/fixed-effects pooled estimate, **forest plot**, I²/τ² heterogeneity,
  **subgroup** analysis, **meta-regression**, and **publication-bias** checks
  (Egger, Begg, funnel, trim-and-fill).

### 3.14 Missing — audit & imputation
![Missing tab](docs/manual/img/14-missing.png)

- **Get here:** **Missing** tab.
- **Steps:** see the per-variable missingness pattern + mechanism hints, then
  run **MICE** multiple imputation to produce completed datasets that downstream
  models can pool (Rubin's rules). Recommend this before regression when data
  are not missing-completely-at-random.

### 3.15 Visual — model visuals, charts, forest builder
![Visual tab](docs/manual/img/15-visual.png)

- **Get here:** **Visual** tab. Sub-tabs: **Models & Diagnostics**, **Charts**
  (custom plots), **Subgroup Bar**, **Forest plot** (publication-ready forest
  builder — paste rows or load from a model), **Added Predictive Value**
  (ΔAUC / NRI / IDI for whether a new marker improves a model).

### 3.16 Power — power & sample size
![Power tab](docs/manual/img/16-power.png)

- **Get here:** **Power** tab (also reachable from the splash screen — *no data
  needed*).
- **Steps:** pick the test (**t-test, ANOVA, correlation, proportions,
  χ², logistic regression, Cox**) → choose **solve for** (sample size / power /
  minimum detectable effect) → enter the effect size + α + power and the
  test-specific fields (e.g. logistic needs the odds ratio + event prevalence;
  Cox needs the hazard ratio + event rate + exposed fraction) → read the result
  + power curve. Great for protocol/grant sample-size justification.

---

## 4. Suggested workflow (what to tell a new user)

1. **Data tab** — fix variable types, check the missing badges, subset if needed.
2. **Summary / Table 1** — describe the cohort, check distributions (Q-Q).
3. **Missing tab** — if gaps are substantial, run MICE before modelling.
4. **Pick the analysis** from the Master Test Index (Section 2) by question +
   variable types.
5. **Run** it in the matching tab; read the on-panel *When to use / Assumptions
   / How to read* card.
6. **Validate & report** — for prediction models use Validation/DCA/Added Value;
   export tables/plots; double-check key numbers in SPSS/R/Stata before
   publishing.

> Reminder for the user: uSTAT is not a validated medical device. Anonymise data
> before upload (no names, MRNs, DOBs).
