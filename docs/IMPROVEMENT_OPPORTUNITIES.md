# wiz3 Assessment: Bugs Found & Improvement Opportunities

Companion to [SPEC.md](SPEC.md). Produced from a full-repo assessment: spec
write-up, backend baseline fix, frontend test infra stand-up, and test
coverage for all 33 backend routers and all 36 frontend panels.

## Real bugs found and fixed during this pass

1. **Entire backend test suite was not running.** `hypothesis` (declared in
   `requirements-dev.txt`) wasn't installed in the venv — pytest collection
   died on `tests/simulation/test_property_based.py` before running anything,
   silently. All 601 pre-existing tests were not executing. Fixed by
   installing the dep; the suite runs and is green.
2. **Test bug, not app bug**: `test_logistic_table_all` asserted
   `n_total == 3` against a 300-row synthetic dataset (copy-paste typo from
   an adjacent `len(table) >= 3` assertion). Fixed to `300`.
3. **`POST /api/stats/power` returned a raw 500** (unhandled statsmodels
   `ValueError`) instead of a 400 whenever a required field was missing for
   `t_two`/`t_one`/`anova`/`chi2`/`proportion` test types. Fixed with upfront
   field validation in `routers/stats/inferential.py`.
4. **`POST /api/bayesian` returned a raw 500** (`KeyError`) when
   `analysis_type="ttest_ind"` was sent without `predictor` — `df[None]`
   indexing crashed before validation. Fixed with an explicit guard.
5. **Frontend `commandParser.ts` silently mis-parsed quoted column names
   containing connector words.** `"time vs event"` was split into
   `["time", "event"]` because the connector-splitting regex matched inside
   the quoted span. Fixed by protecting internal whitespace in quoted spans
   before splitting.

None of these were cosmetic — 1 and 2 meant the backend test suite gave a
false sense of safety (it wasn't running / had a broken assertion), and 3–5
are real user-facing correctness bugs (crash instead of clean error;
misparsed command-palette input).

## Dead code / design smells found (not fixed — flagging for a decision)

- **`ModelsPanel.tsx:262`**: `const isSurvival = false;  // KM/Cox moved to
  Survival Advanced tab`. The entire `isSurvival` branch (KM/Cox UI, ~line
  528 onward) is unreachable — dead code left behind after KM/Cox moved to
  `SurvivalAdvancedPanel`. Worth a cleanup pass to delete the dead branch
  rather than carry it forward.
- **`MissingGuard` async remount pattern**: several panels (`ROCPanel`,
  `VisualModelPanel`, `RCSPanel`, others) wrap their run button in a
  `MissingGuard` that fires an async check on mount and re-renders a *new*
  button element once it resolves. Multiple independent test-writing passes
  hit the same failure mode — a button reference captured before the async
  check resolves is already detached, so a click on it silently no-ops. This
  is not just a test-authoring footgun: a real user who double-clicks fast,
  or whose click lands right as the guard resolves, could plausibly hit a
  dead button in production. Worth a UX check (does the button render
  disabled/loading until the guard resolves, or is there a click-race
  window?).

## Test coverage delivered this pass

- **Backend**: 730 passed, 0 failed, 2 skipped (`kaleido`, optional). New:
  `test_bayesian.py`, `test_tost.py`, `test_power_analysis.py`,
  `test_meta_analysis.py`, `test_timeseries.py`, `test_article_parser.py`,
  `test_nomogram.py` (129 new tests filling the Bayesian/TOST/power/
  meta-analysis/time-series/article-parser/nomogram gaps flagged in SPEC.md).
- **Frontend**: 294 tests, 0 failed. Test infra stood up from scratch
  (vitest + RTL + MSW + jsdom). All 36 panels have a render/request/
  loading/error smoke test; `lib/` pure-logic modules have full unit tests.
- `tsc --noEmit` and `eslint . --quiet` both clean.

## Known residual gaps (honestly reported, not silently dropped)

- **`SurvivalAdvancedPanel`** (largest component, 2943 lines, ~11 modes):
  6 of ~11 modes covered (KM, Cox, Fine-Gray, RMST, E-value, no-session
  guard). Not covered: Cox Horizons, Landmark, Recurrent (LWYY),
  Interval-censored, Uni/Multi-variable Cox, Model Specs. Same test harness
  and pattern applies — straightforward to extend.
- **`article_parser`**: PDF-branch coverage is partial — no `reportlab`/
  `PyPDF2` fixture available in the venv to generate a real test PDF, so
  the PDF-extraction path is covered via `pdfplumber.open` monkeypatching
  rather than an end-to-end real-file test. `.txt`/`.docx` paths are fully
  end-to-end.
- **`DataTable.tsx`** (1877 lines, the core spreadsheet grid) and
  `CommandPalette.tsx`/`UploadZone.tsx` were out of scope for this pass
  (not "analysis flow" panels) — no tests written.
- No E2E/Playwright layer. Per this repo's typescript testing convention,
  Playwright is the intended E2E tool for critical flows (upload → run
  analysis → export) — current coverage is component-level (RTL+MSW) and
  backend API-level (pytest), not full-stack browser E2E.

## Suggestions for follow-up (not done, for you to prioritize)

1. **Wire CI**: none of this (backend pytest, frontend vitest/tsc/eslint)
   appears to run in `.github/workflows/` today — worth checking and adding
   a gate so these 1000+ tests actually block merges, not just exist locally.
2. **Delete the dead `isSurvival` branch** in `ModelsPanel.tsx` (Step-0 style
   cleanup — it's inert code that will confuse future readers).
3. **Extend `SurvivalAdvancedPanel.test.tsx`** to the remaining 5 modes
   using the same harness.
4. **A real PDF fixture for `article_parser`** — either commit a tiny sample
   PDF or generate one at test time with a lightweight PDF-writing lib, to
   get true end-to-end coverage of the PDF branch.
5. **Playwright E2E for the golden path** (upload → pick a panel → run →
   see result → export) would catch integration issues no unit/component
   test can (real Vite dev server + real backend + real Plotly rendering).
6. **Investigate the `MissingGuard` click-race** noted above — decide
   whether the run button should be disabled during the async check.
