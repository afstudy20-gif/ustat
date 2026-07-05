# wiz3 Functional Spec

Medical/clinical statistics workbench. FastAPI backend (Python, stats/ML libs:
statsmodels, lifelines, scikit-learn, scipy) + React/TS SPA frontend (Vite,
Zustand, Plotly). Turkish-language UI. Ships as a web app and (via Tauri) a
desktop app.

## Architecture

```
frontend (React/Vite, Zustand store, ~36 Panel components, one per tab/sub-tab)
   |  axios -> /api/*
backend (FastAPI, 33 routers, 34 service modules, in-memory + disk-persisted
         session store keyed by session_id)
```

- **Session model**: upload a CSV/XLSX/STATA file -> backend parses into a
  DataFrame, infers column kinds (numeric/categorical/datetime), stores it
  in `services/store.py` keyed by `session_id`. All later panel calls pass
  that `session_id`. Sessions autosave to disk (`{sid}.pkl` + `.meta.json`)
  every 20s if dirty, and rehydrate on backend restart.
- **No auth** — single-user local/desktop tool.
- **No database** — sessions are the only persisted state, file-backed.
- **Blank workspace**: "New file" seeds a placeholder session (empty rows/
  columns) without requiring an upload; columns activate on rename, cells
  populate on edit.

## Frontend tabs -> panels (user-facing flow)

Each tab wraps 1+ Panel components in an ErrorBoundary (a panel crashing
doesn't take down the app). Typical flow per panel: pick variable(s) from
the session's columns -> click a "Run/Analyze" button -> panel calls one or
more `/api/...` endpoints -> renders a results table and/or a Plotly chart.

| Tab | Panel(s) | Core flow |
|---|---|---|
| Data | DataTable, DataDictionaryPanel | edit cells, rename/retype columns, value labels |
| Summary | DescriptivePanel, ChartsPanel | per-column stats, normality, freeform charts |
| Table | Table1Panel | grouped clinical characteristics table, p/SMD |
| Tests | HypothesisPanel, CategoricalTestsPanel, RepeatedMeasuresPanel, GatekeepingPanel, NonInferiorityPanel | parametric/non-parametric/categorical hypothesis tests |
| Correlation | CorrelationPanel | pairwise/matrix correlation, ICC, kappa |
| ROC | ROCPanel | ROC/AUC, DeLong compare, combined model |
| Models | ModelsPanel, VisualModelPanel, RCSPanel, SurvivalAdvancedPanel, MLPanel, InternalValidationPanel, AddedValuePanel, PowerPanel, TimeSeriesPanel | regression (linear/logistic/Cox/Poisson/ordinal/GEE/LMM), splines, survival, ML, validation, power, time series |
| PSM | PSMPanel | propensity matching, balance diagnostics |
| IPTW | IPTWPanel | inverse-probability weighting, balance diagnostics |
| Causal+ | CausalPanel | IV/2SLS, mediation, target trial, DiD, RDD, DAG, SEM |
| DCA | DecisionCurvePanel | net-benefit decision curve |
| Meta | MetaPanel | meta-analysis: pooled ES, subgroup, meta-regression, bias |
| Missing | MissingDataPanel | MICE, MCAR test, imputation comparison |
| Visual | ReliabilityPanel, FactorPCAPanel, WeightedStatsPanel, SubgroupBarPanel, ForestBuilderPanel, BayesianPanel, IntervalCensoredPanel | reliability, PCA/FA, survey-weighted stats, forest builder, Bayesian, interval-censored survival |
| Compute | ComputePanel, CodePanel | derived columns, formulas, clinical scores, sandboxed code execution |
| — | RecentSessionsPanel | session list/restore/delete (IndexedDB + disk) |

## Backend routers -> endpoints (by feature area)

33 routers, ~146 endpoints, 34 service modules. Full endpoint-by-endpoint
map with per-endpoint test-coverage status lives in the PR/assessment notes;
summary by area:

- **Descriptive/Table1**: column_summary, frequency, sparklines, table1, weighted_descriptive — fully covered.
- **Correlation/agreement**: correlation matrix/pair, ICC, Cohen/Fleiss kappa, Bland-Altman, Deming, Passing-Bablok, concordance — fully covered.
- **Hypothesis tests**: t-test, ANOVA/ANCOVA/two-way/MANCOVA, Mann-Whitney, Kruskal-Wallis, Jonckheere-Terpstra — fully covered.
- **Categorical**: chi-square, Fisher, binomial/proportions, McNemar, Cochran Q, Mantel-Haenszel, Cochran-Armitage — fully covered.
- **Equivalence**: TOST, non-inferiority — covered, TOST light.
- **Regression**: linear/logistic/ordinal/Cox/Poisson/GLM/multi-outcome (SUR), with stepwise, VIF, RCS, MICE pooling — fully covered (largest test surface, 40+ files).
- **ROC**: single/compare/multi-compare (DeLong)/combined — fully covered.
- **Survival**: KM, Cox uni/multivariate, cumulative hazard, competing risks, interval-censored, frailty, recurrent events, multistate, joint models — fully covered.
- **Causal inference**: PSM, IPTW, IV/2SLS, mediation, SEM, target trial, DiD, RDD, DAG adjustment, sensitivity (E-value) — fully covered.
- **Missing data**: pattern summary, MCAR test, imputation compare, MNAR sensitivity — fully covered.
- **Meta-analysis**: standardize ES, forest (fixed/random, I²), bias — **partial** (light test depth).
- **Bayesian**: conjugate posterior comparison — **partial** (1 minimal test file).
- **Power/sample size**: t/ANOVA/correlation/proportion/logistic/Cox/chi-square power — **partial** (light depth).
- **Reliability**: Cronbach alpha, test-retest, inter-rater ICC — fully covered.
- **Factor/dimensionality**: FA, PCA, clustering — fully covered.
- **ML**: classification/regression/ensemble, risk prediction — fully covered.
- **Time series**: ARIMA, decompose, stationarity, GEE — **partial**.
- **Decision/prediction**: decision curve, nomogram, risk prediction — covered, nomogram thin.
- **Data management**: formula/recode/transform/clinical-scores/row-col ops — fully covered.
- **Session/upload**: upload, list/get, save/restore, refresh — fully covered.
- **Publishing/export**: table/model export (DOCX/XLSX), citations — mostly covered, citations thin.
- **Diagnostics/charts**: residual/VIF diagnostics, general chart rendering — fully covered.
- **Article parser**: PDF/image table extraction — **no tests**.
- **Code runner**: sandboxed Python/R execution — covered.

## Statistical display conventions (must hold everywhere a result is shown)

- p-values and n (sample size) are italicized as statistical variables
  (`<i>p</i>`, `<i>n</i>`) in on-screen tables and Plotly chart text —
  journal/APA convention. Applies to live result displays only; excluded:
  CSV/XLSX/DOCX exports, native `title=` tooltips, help/prose text, ARIMA
  `(p,d,q)` labels, and props/state literally named `p`/`n`.
- p-value formatting is centralized in `frontend/src/lib/format.ts`
  (`fmtP`, `fmtPubP`, `fmtPFull`, `pCellTitle`, `fmtPubPHtml`) — new code
  should use these rather than ad hoc formatting.

## Known coverage gaps (input to the test-writing plan)

1. Frontend now has Vitest + React Testing Library + MSW wiring and focused
   formatter tests for the shared p-value display contract. Panel-level tests
   are still mostly absent; all 36 panels need smoke tests for render, request,
   loading/error, and result-table paths.
2. Backend: Bayesian, time series (ARIMA/GEE), meta-analysis, TOST, power
   analysis have shallow test depth. Article parser has none. Nomogram is
   thin.
