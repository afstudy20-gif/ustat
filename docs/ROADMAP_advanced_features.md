# uSTAT — Advanced Features Roadmap

Response to the feature-gap critique (offline, UI polish, advanced stats,
viz, reporting, data management). Every item below is graded against the
**current** codebase, not a greenfield assumption.

## Current stack (ground truth)

- **Backend**: FastAPI, 23 routers under `backend/routers/`. Stats engine =
  `scipy 1.15`, `statsmodels 0.14.4`, `scikit-learn 1.6.1`, `lifelines 0.30`,
  `pandas 2.2`. Audit trail already recorded per session via
  `store.log_action()` / `store.get_audit()`.
- **Frontend**: React + Plotly, flat tab array in `App.tsx` (Data, Summary,
  Table, Tests, Correlation, ROC, Models, Visual, Compute, PSM, Missing,
  Code). `ModelsPanel` already uses sub-tabs (Regression / Survival Advanced /
  RCS) — the pattern for nesting new analyses.
- **Reporting**: `pub_export.py` already emits a Methods-appendix DOCX from the
  audit log (`_ACTION_HUMAN` map) + Table 1 DOCX. python-docx present.
- **Already shipped** that the critique misses: IPTW weighted GLM + weighted
  Cox (PSM panel), DerSimonian-Laird random-effects forest (`charts/forest`),
  K-way DeLong, Firth penalised logistic, RCS, competing-risks Fine-Gray,
  RMST, MICE, calibration / DCA / Hosmer-Lemeshow.

**Headline**: the entire P1 tier below needs **zero new dependencies** —
sklearn + statsmodels already cover random forest, gradient boosting, ARIMA,
meta-regression, and weighted descriptives.

---

## Verdict on each critique point

| Critique | Verdict | Tier |
|----------|---------|------|
| Offline doesn't work | **True but expensive** — compute is server-side Python | P3 / defer |
| UI/UX not as polished as JASP/Jamovi | **Partly true** — cross-cutting track | P1-cross |
| Bayesian stats | **True** — heavy dep + needs job queue | P3 (light alt in P2) |
| Time series (ARIMA) | **True — cheap** (statsmodels SARIMAX) | **P1** |
| SEM | **True** — niche + complex UI | P3 / defer |
| ML / predictive (RF, etc.) | **True — cheap** (sklearn present) | **P1** |
| Advanced meta-analysis | **Partly** — DL forest exists, expand it | **P1** |
| ggplot-level viz | **True** — Plotly ≠ GoG, big rework | P3 (incremental in P2) |
| Auto report (RMarkdown/Quarto) | **Partly** — appendix exists, build assembler | **P2** |
| Large-data perf | **Overstated** — fine to ~100k rows | P2 (pagination) |
| dplyr/pandas manipulation limited | **Partly** — have formula/recode, lack joins/reshape | P2 |
| Complex survey / weighted | **Partly** — IPTW weighting exists | **P1** (weights-only) |

---

## P1 — Quick wins (no new deps, days each)

### 1. ML predictive modeling

**New router** `backend/routers/ml.py`, prefix `/api/ml`.

Endpoints:
- `POST /ml/random_forest` — `RandomForestClassifier` (binary/multiclass) or
  `RandomForestRegressor`. Honest performance via `cross_val_predict`
  (default 5-fold). Returns: CV AUC + DeLong-free bootstrap CI, accuracy,
  confusion matrix, sensitivity/specificity/PPV/NPV, ROC curve points,
  calibration bins, Gini + **permutation** importance (no new dep), OOB score.
- `POST /ml/gradient_boosting` — `GradientBoostingClassifier/Regressor`
  (sklearn-native; xgboost is an optional later upgrade, not required).
- `POST /ml/feature_importance` — standalone permutation importance on any
  fitted model spec.

Shared concerns:
- `class_weight="balanced"` toggle for imbalance.
- Categorical predictors → one-hot (reuse the dummy-encoding pattern from
  `models.py` Cox/logistic).
- Train/test split OR k-fold CV selectable; report which.
- Reuse `apply_imputation` from `services/impute.py`.

**Frontend**: `MLPanel.tsx` + new `ml` tab (or sub-tab under Models).
Controls: outcome, predictor multiselect, model (RF / GBM), n_estimators,
max_depth, CV folds vs test split, class-weight toggle. Results column:
metric tiles, ROC (reuse ROC trace style), feature-importance bar (reuse
ForestPlot bar), confusion matrix, calibration curve.

Files: `routers/ml.py` (new), `main.py` (+1 include), `api.ts` (+3),
`MLPanel.tsx` (new), `App.tsx` (tab + TEST_CATALOG entries +
`AboutModal` row). Effort **M (2-3 d)**.

### 2. Time series — ARIMA / SARIMA

**New router** `backend/routers/timeseries.py`, prefix `/api/timeseries`.

- `POST /timeseries/arima` — `statsmodels SARIMAX`. Inputs: value col,
  optional time/index col, `(p,d,q)`, seasonal `(P,D,Q,s)`, `auto` flag
  (small AIC grid search; pmdarima optional later for full auto). Outputs:
  coef table, AIC/BIC, Ljung-Box (white-noise residuals), n-step forecast +
  CI, ACF/PACF arrays, residual series.
- `POST /timeseries/decompose` — STL/seasonal_decompose (trend/seasonal/resid).
- `POST /timeseries/stationarity` — ADF + KPSS tests.

**Frontend**: `TimeSeriesPanel.tsx`, new `timeseries` tab. Plots:
observed + in-sample fit + forecast band; ACF/PACF stem plots; residual
diagnostics. Effort **M**.

### 3. Meta-analysis expansion

DL random-effects forest already exists in `charts/forest`. **New router**
`backend/routers/meta.py`, prefix `/api/meta`, to make it a first-class tool:

- `POST /meta/analyze` — fixed (inverse-variance) + random (DL, optional
  REML/PM τ²) pooling. Accepts pre-computed effect+SE rows OR raw 2×2 tables
  → OR/RR/RD with continuity correction. Returns pooled estimate + 95% CI +
  **95% prediction interval**, Q, I², τ², H².
- `POST /meta/subgroup` — stratified pooling + between-group Q test.
- `POST /meta/regression` — meta-regression via `statsmodels WLS`
  (effect ~ moderator, weights = 1/(SE²+τ²)).
- `POST /meta/bias` — Egger's regression test, Begg rank test, funnel-plot
  points, trim-and-fill (L0 estimator, pure numpy).

**Frontend**: `MetaPanel.tsx` (or extend `ForestBuilderPanel`) — paste/upload
study table, forest + funnel plots, bias panel. Effort **M**.

### 4. Weighted / survey descriptives

Weights-only (no strata/cluster) covers the common case; full complex-survey
(`samplics`) deferred.

- Extend `stats.py`: `POST /stats/weighted_descriptive` —
  `statsmodels DescrStatsW` (weighted mean, SD, quantiles, CI), weighted
  proportion (Horvitz-Thompson), weighted t-test, weighted chi-square.
- Add optional `weights_col` to existing GLM endpoints (logistic/linear/
  poisson) — statsmodels `freq_weights`/`var_weights`.

**Frontend**: weight-column dropdown on Descriptive + Models panels. Effort
**S-M**.

---

## P2 — Medium (infra mostly exists)

### 5. Auto-report assembler ("Report builder")

The audit log + `_ACTION_HUMAN` already drive the Methods appendix. Extend to
a full Results document.

- **Persist results**: add `store.pin_result(session_id, item)` storing the
  result payload (text, export_rows, figure PNG) when the user clicks a new
  **"Pin to report"** button (extend `ResultExporter`).
- `POST /pub_export/full_report` — assemble pinned items into a structured
  DOCX **and** HTML: Methods section (from audit) + Results section (pinned
  tables with auto-numbered captions + embedded figures) + References stub.
  Quarto-style sectioning, journal-ready.
- **Frontend**: a `report` tab — list pinned items, drag-reorder, edit
  captions, export DOCX/HTML. Effort **M-L**.

### 6. Data manipulation (joins / reshape / aggregate)

Extend `compute.py` (already has formula/recode/transform; melt in models):
- Long↔wide reshape (pivot/melt UI).
- Group-by aggregate (mean/median/n/sum per group → new columns or new table).
- Merge/join two sessions on a key (inner/left/outer).
- Visual filter/sort builder (compose row filters).

**Frontend**: extend `ComputePanel.tsx`. Effort **M**.

### 7. Large-data performance

- Audit `DataTable.tsx` (74 KB) for row virtualization; add if missing.
- Server-side preview sampling + payload caps; stream CSV export.
- Paginated `raw` endpoint already partial — formalize.
  Effort **M**, mostly frontend.

### 8. Light Bayesian (no heavy dep)

Conjugate / closed-form cases in pure numpy/scipy — no PyMC:
- Beta-binomial proportion (credible interval, posterior plot).
- Bayesian two-group comparison (BEST-style via t, or Bayes factor for t-test
  through `scipy`/closed form).
- Bayes factor for correlation / contingency (JZS where closed-form).

Covers the 80% clinical use without MCMC infra. Full PyMC stays P3. Effort
**M**.

---

## P3 — Defer (weeks+, heavy deps, low ROI for cardiology audience)

| Feature | Blocker | Note |
|---------|---------|------|
| **Offline compute** | Python compute is server-side | Pyodide/WASM port of scipy+statsmodels+lifelines is huge + partial. PWA already caches the shell; document "online compute required". Revisit only if a true offline build is a hard requirement. |
| **Full Bayesian (PyMC)** | Heavy dep + MCMC too slow for a request cycle | Needs async job queue (RQ/Celery + Redis) — an infra change. Light conjugate Bayesian (P2.8) covers most clinical needs first. |
| **SEM** | `semopy` dep + path-diagram UI | Niche for the cardiology audience; build only on direct demand. |
| **ggplot-level viz rework** | Plotly is not grammar-of-graphics | Incremental wins instead: raincloud, paired-slope, violin+box combos, more themes (cheap, P2-adjacent). Full GoG engine swap not justified. |
| **Complex survey (strata+cluster)** | `samplics` dep | Weights-only (P1.4) ships first; add design-based SE only if requested. |

---

## UI/UX polish (cross-cutting track)

Already in flight: 3-column controls/chart/results layout + drag-resizable
columns + publication-style ROC/RCS plots. Continue incrementally:
- Consistent ThreeCol rollout across remaining panels.
- Empty-state illustrations, loading skeletons, keyboard nav.
- Onboarding tour for first-time users (the JASP/Jamovi "feels guided" gap).
- Dark mode pass.
No single big rewrite — steady polish per panel.

---

## Dependency plan

- **P1**: none new (sklearn + statsmodels suffice).
- **P2**: none new for report/manipulation/light-Bayes; perf is frontend.
- **Optional upgrades** (add only when the feature lands): `xgboost`
  (GBM upgrade), `pmdarima` (true auto-ARIMA), `shap` (explainability),
  `samplics` (complex survey).
- **Deferred heavy**: `pymc`, `semopy`, Pyodide toolchain.

## Suggested sprint order

1. **Sprint 1 (P1, no deps)**: ML panel → ARIMA → meta expansion → weighted
   descriptives. Highest clinical value, zero dependency risk.
2. **Sprint 2 (P2)**: Report builder → data manipulation → light Bayesian.
3. **Sprint 3**: perf + optional deps (xgboost/shap/pmdarima) + viz extras.
4. **Defer**: Bayesian-PyMC / SEM / offline-compute unless explicitly demanded.

## Per-feature verification

Each new endpoint ships with: a `backend/tests/test_*.py` smoke test on the
synthetic fixture; a cross-check of one result against R/Python reference
(documented in the commit, not committed as a test); `tsc --noEmit` + `vite
build` green; an `AboutModal` changelog entry + Tests-&-Methods row; and a
live check on `ustat.drtr.uk`.
