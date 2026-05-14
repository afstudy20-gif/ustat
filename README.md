<div align="center">

<img src="frontend/public/logo.png" alt="uSTAT logo" width="120" />

# uSTAT — Statistical Analysis Platform

**Free, browser-based SPSS / R / Stata alternative for clinicians and researchers.**

[![Live](https://img.shields.io/badge/live-ustat.drtr.uk-4f46e5?style=flat-square)](https://ustat.drtr.uk)
[![Security scan](https://img.shields.io/badge/CI-bandit%20%2B%20pip--audit%20%2B%20semgrep-22c55e?style=flat-square)](.github/workflows/security-scan.yml)
[![No medical device](https://img.shields.io/badge/not-a%20medical%20device-dc2626?style=flat-square)](frontend/public/terms.html)

</div>

uSTAT is a free, server-hosted statistical analysis platform aimed at clinicians,
biostatisticians, and medical researchers. Upload a `CSV`, `Excel`, `SPSS`,
`SAS`, or `Stata` file and run 40+ analyses — t-tests, ANOVA, Cox PH, ROC, PSM,
RCS dose-response, Table 1, power analysis — without installing anything.

> ⚠️ **Not a medical device.** uSTAT has **not yet been validated through
> peer-reviewed publications**. Independent validation against SPSS, R, and
> Stata is ongoing. Verify any clinically or scientifically important result
> against an established statistics package before reporting.

---

## Table of contents

- [Live instance](#live-instance)
- [Features](#features)
- [Quick start](#quick-start-end-user)
- [Statistical pipeline (worked example)](#statistical-pipeline-worked-example)
- [Local development](#local-development)
- [Self-hosting](#self-hosting)
- [Architecture](#architecture)
- [Statistical methods & references](#statistical-methods--references)
- [Security & privacy](#security--privacy)
- [Code-execution sandbox](#code-execution-sandbox)
- [Contributing](#contributing)
- [Citation](#citation)
- [License & attribution](#license--attribution)

---

## Live instance

Production deployment: **<https://ustat.drtr.uk>**

Operated by **Dr. Yusuf Hoşoğlu**.
Contact: [adycovs@gmail.com](mailto:adycovs@gmail.com)
Vulnerability disclosure: [adycovs@gmail.com](mailto:adycovs@gmail.com?subject=%5BuSTAT-security%5D) (prefix subject with `[uSTAT-security]`) · RFC 9116 contact at `/.well-known/security.txt`.

---

## Features

| Area                  | What you can do |
|-----------------------|-----------------|
| **Data I/O**          | CSV, Excel (.xlsx / .xls), SPSS (.sav), SAS (.sas7bdat), Stata (.dta), JSON sessions |
| **Cleaning**          | Rename, recode, fill blanks (mean / median / MICE), filter cases, compute new variables, Ctrl+V paste from Excel/CSV |
| **Descriptive**       | Mean / SD / median / IQR, Shapiro-Wilk, Kolmogorov-Smirnov (Lilliefors), Q-Q plots |
| **Hypothesis tests**  | t-test (independent / paired), one-way ANOVA, ANCOVA, two-way ANOVA, repeated-measures ANOVA, Tukey HSD, Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman |
| **Categorical**       | χ², Fisher's exact, McNemar, Cochran's Q, Mantel-Haenszel, binomial, one- & two-proportion |
| **Correlation**       | Pearson, Spearman, Kendall, ICC, Cohen's κ |
| **Regression**        | Linear (+ HC3 robust SE), logistic (with OR + 95% CI), Poisson / negative binomial, polynomial, **restricted cubic splines (RCS)**, mixed-effects (LMM) |
| **Survival**          | Kaplan-Meier + log-rank, **Cox PH**, **Cox-RCS** with 1 or 2 spline terms and optional `rcs(X) × rcs(Y)` interaction (LR test + 2D HR contour), Fine-Gray competing risks, landmark, stratified Cox |
| **ROC / prediction**  | ROC curves, AUC, Youden index, paired ROC comparison (DeLong), calibration plot, Hosmer-Lemeshow, Brier score, decision curve analysis, nomogram |
| **Causal inference**  | **Propensity Score Matching (PSM)** — logistic / probit / GBM score models, greedy or optimal (Hungarian) matching, exact-match strata, common-support trim (Crump 2009), Austin 2011 logit-PS caliper, SMD + Rubin variance ratio + KS test, Love plot, GEE binary outcome **or** stratified Cox survival outcome, **Rosenbaum bounds** sensitivity analysis |
| **Missing data**      | MICE multiple imputation, Little's MCAR test, imputation comparison |
| **Tables & export**   | Publication-ready Table 1 with SMD, AMA-formatted journal Excel/Word export, charts at up to 600 DPI |
| **Power analysis**    | t-test / ANOVA / proportions / correlation power, Cox / log-rank sample size (Schoenfeld 1981, Freedman 1982) |
| **Code sandbox**      | Optional server-side Python runner with `df` pre-injected. Off by default — see [Code-execution sandbox](#code-execution-sandbox) |

---

## Quick start (end user)

1. Open <https://ustat.drtr.uk>.
2. Drop a file on the **Statistical Analysis** tile (or click to browse). uSTAT auto-detects variable types (numeric / categorical / date).
3. Use the left sidebar to pick an analysis (**Models**, **Tests**, **Survival**, **PSM**, **Power**, **Table**, …). Each panel lets you pick the variables and runs the appropriate test automatically — including normality-based selection between parametric and non-parametric paths.
4. Read the result. Every output includes effect sizes, 95% CIs, assumption diagnostics, and a plain-English summary.
5. Export charts at 600 DPI, copy tables to Word / Excel, or save the whole session as a JSON file to resume later.

Your data stays in server RAM only, is never written to disk, and is automatically discarded 30 minutes after your last activity (see [SECURITY.md](backend/SECURITY.md)).

---

## Statistical pipeline (worked example)

**Question:** how does serum LDL relate to all-cause mortality, after adjusting for age and other covariates?

| Step | UI path | Backend route | What it does |
|------|---------|----------------|--------------|
| 1a. Univariate Cox-RCS, Harrell knots | Models → RCS Dose-Response → Outcome **Cox** | `POST /api/models/rcs` | Cox PH on `rcs(LDL, 4)` only. Knots at the 5/35/65/95 percentile (Harrell standard). Returns HR curve + 95% CI + nonlinearity p. |
| 1b. Univariate Cox-RCS, clinical knots | Same panel, **Knot positions = Custom (70, 100, 130, 160)** | `POST /api/models/rcs` | Sensitivity analysis with clinically meaningful cut-points. |
| 2. Multivariable Cox-RCS | Models → **Cox-RCS (multivariable)** → LDL + AGE + covariates | `POST /api/models/survival/cox_rcs` | `Surv(time, event) ~ rcs(LDL,4) + rcs(AGE,4) + SEX + DM + HT + SMOKER`. Per-term Wald nonlinearity test. |
| 3. Interaction (primary hypothesis) | Same panel, toggle **Include RCS × RCS interaction** | `POST /api/models/survival/cox_rcs` | Tensor-product interaction columns + LR test of full vs main-effects model + 2D HR contour plot of `LDL × AGE`. |

Each step also exists as a code-runner template (Step 1 / Step 1b / Step 2 / Step 3) if you prefer to drive the analysis with lifelines directly — see [Code-execution sandbox](#code-execution-sandbox).

---

## Local development

### Prerequisites

- Python 3.11+
- Node 20+ (or compatible)
- ~3 GB free disk for the Python scientific stack

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is now on `http://localhost:8000`. Sanity check:

```bash
curl http://localhost:8000/api/health
# {"status":"ok","active_sessions":0,"memory":{...}}
```

### Frontend

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

Vite serves the UI on `http://localhost:5173` and proxies `/api/*` to the
backend automatically (`vite.config.ts`).

### Run both with one command

The repo ships a `.claude/launch.json` consumed by the Claude Preview MCP, but
in plain shell:

```bash
# terminal 1
cd backend && source .venv/bin/activate && uvicorn main:app --reload --port 8000
# terminal 2
cd frontend && npm run dev
```

### Enable the optional code sandbox (off by default)

```bash
ENABLE_CODE_RUNNER=1 uvicorn main:app --reload --port 8000
```

The frontend `Code` tab will then appear; before that, it is hidden by the
`/api/code/status` probe.

---

## Self-hosting

A `Dockerfile` and `render.yaml` ship in the repo root. The production
instance runs on Render's free plan.

```bash
docker build -t ustat .
docker run -p 8000:8000 ustat
```

Optional production env vars (see [SECURITY.md](backend/SECURITY.md) for the
full table):

```bash
ENABLE_CODE_RUNNER=0           # default — keep the runner off in production
CSP_ENFORCE=0                  # flip to 1 after a week of report-only telemetry
CODE_RUNNER_PER_MIN=2          # if you ever flip the runner on, tighten limits
CODE_RUNNER_MAX_CONCURRENT=1
SECURITY_CONTACT_EMAIL=...     # overrides the default in /.well-known/security.txt
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Browser (React 19 + Plotly.js + zustand)                           │
│   – no statistics here, just UI state + chart rendering            │
└──────────────────────────────────────────────┬─────────────────────┘
                                               │ HTTPS (HSTS, CSP, …)
┌──────────────────────────────────────────────▼─────────────────────┐
│ FastAPI 0.115 + Uvicorn 0.34                                       │
│   – /api/upload, /api/stats/*, /api/models/*, /api/sessions/*, …   │
│   – middleware/security_headers.py                                 │
│   – routers/code_runner.py (gated)                                 │
├────────────────────────────────────────────────────────────────────┤
│ services/store.py — in-RAM dataframe store (30 min TTL)            │
│ services/rcs_basis.py — Harrell RCS basis                          │
│ services/impute.py — listwise / mean / median / MICE               │
│ services/sandbox.py + code_runner_child.py — Python subprocess     │
│                       with rlimit + import allowlist + audit       │
├────────────────────────────────────────────────────────────────────┤
│ scipy 1.15 · statsmodels 0.14 · lifelines 0.30 · scikit-learn 1.6  │
│ pandas 2.2 · numpy 2.2 · patsy 0.5 · pyreadstat 1.2                │
└────────────────────────────────────────────────────────────────────┘
```

Key invariants:

- **Server-side computation.** The frontend renders results; it does not run
  statistical algorithms.
- **No persistence.** Uploaded data lives only in process memory and is
  discarded after `SESSION_TTL_SECONDS` (default 1800).
- **No accounts.** Sessions are an opaque UUID handed back to the browser; the
  backend never knows who you are.
- **Pure-Python deps.** All statistical computation runs on peer-reviewed
  open-source libraries — no custom black-box estimators.

---

## Statistical methods & references

Every test maps to a specific upstream function. The full table lives in the
in-app **About → Tests & Methods** modal. Highlights:

| Method                                           | Implementation                                  |
|--------------------------------------------------|-------------------------------------------------|
| Independent / paired t-test                      | `scipy.stats.ttest_ind` / `ttest_rel`           |
| ANOVA / ANCOVA / mixed                           | `statsmodels.formula.api.ols` + `anova_lm`, `mixedlm` |
| Mann-Whitney / Kruskal-Wallis                    | `scipy.stats.mannwhitneyu` / `kruskal`          |
| Shapiro-Wilk / Lilliefors                        | `scipy.stats.shapiro` / `statsmodels.stats.diagnostic.lilliefors` |
| χ² / Fisher's exact                              | `scipy.stats.chi2_contingency` / `fisher_exact` |
| Pearson / Spearman / Kendall                     | `scipy.stats.pearsonr` / `spearmanr` / `kendalltau` |
| Linear regression (HC3)                          | `statsmodels.OLS` with `cov_type="HC3"`         |
| Logistic regression (with OR + CI)               | `statsmodels.formula.api.logit`                 |
| Cox proportional hazards                         | `lifelines.CoxPHFitter`                         |
| Kaplan-Meier + log-rank                          | `lifelines.KaplanMeierFitter`, `logrank_test`   |
| Fine-Gray competing risks                        | custom on top of `lifelines`                    |
| Restricted cubic splines (Harrell)               | `services/rcs_basis.py` (Harrell 2015 §2.4.4)   |
| Propensity-score matching                        | sklearn `LogisticRegression` / `GradientBoostingClassifier` / statsmodels `Probit` + custom greedy and `scipy.optimize.linear_sum_assignment` |
| Rosenbaum bounds                                 | custom on `scipy.stats.binom`                   |
| MICE multiple imputation                         | `sklearn.impute.IterativeImputer`               |

Methodological references applied:

- Austin PC (2011). *Multivariate Behavioral Research* — PSM balance, logit-PS caliper, pooled-SD denominator.
- Crump RK et al. (2009). *Biometrika* — common-support trimming.
- Rosenbaum PR (2002). *Observational Studies* — bounds on hidden bias.
- Harrell FE (2015). *Regression Modeling Strategies* — RCS knot placement, Cox-RCS, nonlinearity tests.
- Vickers AJ, Elkin EB (2006). *Med Decis Making* — decision curve analysis.
- DeLong ER et al. (1988). *Biometrics* — paired ROC comparison.
- Schoenfeld DA (1981) — Cox model sample size.
- Hosmer DW, Lemeshow S — logistic calibration test.
- Benjamini Y, Hochberg Y (1995) — FDR control.

---

## Security & privacy

A summary; the canonical documents are:

- [`/privacy.html`](frontend/public/privacy.html) — Privacy Policy (data flow, retention, what we do NOT do)
- [`/terms.html`](frontend/public/terms.html) — Terms of Use (no medical device, no warranty, citation)
- [`/security.html`](frontend/public/security.html) — Security Overview (architecture, hardening, automated scans)
- [`backend/SECURITY.md`](backend/SECURITY.md) — sandbox threat model + env knobs
- [`/.well-known/security.txt`](backend/main.py) — RFC 9116 disclosure contact

Highlights:

- HSTS 1-year preload, CSP (report-only until tuned), X-Frame-Options DENY,
  X-Content-Type-Options nosniff, Permissions-Policy denying camera / mic /
  geolocation / payment / USB / motion sensors, COOP same-origin.
- Continuous scans on every push & weekly:
  [`.github/workflows/security-scan.yml`](.github/workflows/security-scan.yml)
  runs `bandit`, `pip-audit`, `npm audit`, `semgrep` (OWASP Top 10), and
  `gitleaks`.
- No persistent storage, no accounts, no analytics beyond an aggregate
  visitor map.

If you discover a vulnerability, email
[adycovs@gmail.com](mailto:adycovs@gmail.com?subject=%5BuSTAT-security%5D)
with the `[uSTAT-security]` subject prefix. We acknowledge within 5 business
days.

---

## Code-execution sandbox

uSTAT optionally exposes a sandboxed Python runner — a "Code" tab that
accepts arbitrary Python and runs it against your session DataFrame
(injected as `df`). Off by default in production.

| Knob                            | Default | Description                              |
|---------------------------------|---------|------------------------------------------|
| `ENABLE_CODE_RUNNER`            | 0       | Set to 1 to expose the endpoint          |
| `SANDBOX_MEM_BYTES`             | 512 MB  | Address-space rlimit per run             |
| `SANDBOX_CPU_SEC`               | 30 s    | CPU rlimit (max 60 s)                    |
| `CODE_RUNNER_PER_MIN`           | 6       | Runs / minute / session                  |
| `CODE_RUNNER_PER_HOUR`          | 30      | Runs / hour / session                    |
| `CODE_RUNNER_IP_PER_MIN`        | 10      | Runs / minute / IP                       |
| `CODE_RUNNER_IP_PER_HOUR`       | 60      | Runs / hour / IP                         |
| `CODE_RUNNER_GLOBAL_PER_MIN`    | 30      | Runs / minute server-wide                |
| `CODE_RUNNER_MAX_CONCURRENT`    | 2       | Concurrent in-flight runs                |

Defense-in-depth: subprocess with `python -I -u`, `resource.setrlimit` for
CPU / memory / fds / nproc / fsize, `sys.meta_path` import allowlist
(numpy / pandas / scipy / statsmodels / lifelines / sklearn / matplotlib /
seaborn + safe stdlib) with an explicit deny-list for `socket`, `urllib*`,
`http*`, `subprocess`, `ctypes`, `os`, `shutil`, `pickle`, `importlib`, …
On Linux the wrapper additionally runs the child under
`unshare --user --net` when available, stripping the network namespace.

Read the full threat model in [`backend/SECURITY.md`](backend/SECURITY.md)
before enabling in production.

---

## Contributing

Pull requests welcome. Before opening one:

1. Run the linters and the local checks the CI workflow runs.
2. Update [`backend/SECURITY.md`](backend/SECURITY.md) if you touch the
   sandbox, auth, or rate-limit layers.
3. Add or update a test if you change a statistical method.
4. If you add a new statistical method, also add a row to the Tests & Methods
   table in [`AboutModal.tsx`](frontend/src/components/AboutModal.tsx) and a
   line to the methods table in this README.

---

## Citation

If you publish results obtained with uSTAT, please cite the tool:

> Hoşoğlu Y. *uSTAT — browser-based statistical analysis platform.*
> https://ustat.drtr.uk (accessed YYYY-MM-DD).

BibTeX:

```bibtex
@software{ustat,
  author  = {Hoşoğlu, Yusuf},
  title   = {uSTAT — browser-based statistical analysis platform},
  url     = {https://ustat.drtr.uk},
  year    = {2026}
}
```

---

## License & attribution

© 2026 Dr. Yusuf Hoşoğlu. All rights reserved.

The source is published for transparency and community review. A formal
open-source license is on the roadmap — until one is added, no
redistribution rights are granted by default.

Statistical computation builds on `scipy`, `statsmodels`, `lifelines`,
`scikit-learn`, `pandas`, `numpy`, `patsy`, `pyreadstat` (each under its own
license — see the respective project pages).
