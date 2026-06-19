# uSTAT — Multi-Agent QA Test Plan

**Target:** Find every broken / buggy / misleading behaviour in the live uSTAT
web app (`https://ustat.drtr.uk` or local `frontend/` + `backend/`).
**Dataset:** `qa/cohort_test.csv` — 100 patients × 10 columns, deliberately
imperfect (typed missings, text-as-missing, comma-decimal locale leakage,
impossible values, mixed date formats, mild class imbalance).

The repo's `MANUAL.md` is the authoritative map of every feature; every
analysis listed there is in scope.

## Method

Three independent passes:

1. **Detection pass** (this stage). Five backend agents each cover a
   non-overlapping slice of the app, run the analyses end-to-end on the test
   dataset, and write findings to `qa/findings/<agent>.md`.
2. **Cross-check pass**. A second wave re-runs the *other* agent's top-priority
   findings on the *same* dataset (or a clean variant) to confirm or refute.
3. **Triage** (human + Claude). Collect, dedupe, prioritize, plan fixes.

## Finding schema (every entry must use this)

```
## [SEV] <one-line title>
**Where:** <tab → sub-tab → control>
**Steps:** 1) … 2) … 3) …
**Expected:** <what should happen>
**Actual:** <what happens>
**Evidence:** <network response shape, screenshot ref, console log excerpt>
**Hypothesis (optional):** <likely cause>
```

`SEV` ∈ `CRITICAL` (data corruption / wrong stat result / crash) ·
`HIGH` (analysis fails or returns wrong numbers) ·
`MEDIUM` (usability, missing affordance) · `LOW` (cosmetic / copy).

## Coverage assignment

| Agent | Slice | Tabs to exercise |
|-------|-------|------------------|
| ZCode | Data ingest + cleaning + Compute + Missing | Data, Compute, Missing, Dictionary |
| Kimi | Hypothesis tests + Categorical + Correlation + ROC | Tests (all sub-tabs), Correlation, ROC |
| Codex | Regression + Survival + RCS + Validation | Models (all sub-tabs incl. Survival Advanced) |
| Grok-build | Causal stack + Prediction utilities | PSM, IPTW, Causal+, DCA, Power |
| Grok-composer | Visual / reporting / Table 1 / Meta / Time Series | Table, Summary, Visual, Meta, Time Series |

Each agent runs the analyses listed for its slice in `MANUAL.md` against
`qa/cohort_test.csv`. The dataset is intentionally broken in places — flag
both "uSTAT crashed" *and* "uSTAT produced output that's silently wrong on
this data".

## Backend (compute) coverage

If the live app is not reachable, the same dataset can be loaded against the
local backend (`backend/`) via the API (see endpoint list in
`MANUAL.md` or under `backend/routers/`).

## Output

Each agent appends to `qa/findings/<agent>.md`. Claude collates after wave 1
and dispatches the cross-check wave.
</content>
