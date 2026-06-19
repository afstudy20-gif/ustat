# Shared brief — read before starting

You are one of five agents auditing the uSTAT web app for bugs. The other
agents cover non-overlapping slices in parallel; you cover the slice named in
your own brief.

## Repo

`/Users/yh/Documents/projects/wiz3`  (the working directory you are launched in).

- `backend/` — FastAPI app (every analysis endpoint lives under `backend/routers/`).
- `frontend/` — React/Vite UI.
- `MANUAL.md` — authoritative map of every feature; consult it for which test
  lives where.

## Test dataset

`qa/cohort_test.csv` — 100 rows × 11 columns:

| col | type | dirty bits |
|-----|------|-----------|
| `patient_id` | id | clean |
| `age` | num | a few impossible (-5, 199, 0, 250) |
| `sex` | cat 'M'/'F' | blanks; one "x"; one "Female" inconsistency |
| `bmi` | num | comma-decimals on 3 rows ("30,6"); blanks; one "999"; one "n/a" |
| `ldl` | num | blanks + "NA" |
| `sbp` | num | a few blanks |
| `diabetes` | binary 0/1 | a couple blanks |
| `nyha` | ordinal 1-4 | a couple blanks |
| `fu_days` | num (time to event) | one negative |
| `event` | binary 0/1 | clean |
| `admission_date` | date | mixed ISO / dd/mm/yyyy / dd.mm.yyyy; one impossible "13/13/2024"; blanks |

Some breakage is intentional so the QA finds *how the app handles bad data*,
not just the happy path.

## How to drive the backend without uvicorn / frontend

```python
import sys; sys.path.insert(0, "qa")
from run_via_testclient import boot
client, sid = boot()  # loads qa/cohort_test.csv
# now: client.post("/api/stats/ttest", json={...}), etc.
```

Endpoints are all under `/api/...`. Inspect `backend/routers/*.py` to see
exact request shapes. `MANUAL.md` lists every analysis with its tab.

## What counts as a finding

A finding is anything that's wrong, misleading, broken, or surprising:

- 5xx crash on a sensible request
- a 200 OK that returns nonsense (e.g. a t-test with NaN, an OR of inf,
  a regression that ignores n_excluded)
- silent ingest of a clearly bad cell (e.g. "n/a" treated as a number)
- a panel that 200's but the response shape doesn't match what the frontend
  expects (read the component to verify)
- a tooltip / label that says the wrong thing
- a feature listed in MANUAL.md that returns 404 or is wired to a wrong tab
- inconsistent results between two ways of doing the same thing
  (e.g. linear regression of x on y vs Pearson r²)

## What to do

1. For your slice, run every analysis listed in MANUAL.md against
   `qa/cohort_test.csv` (via the `boot()` helper, or by reading code paths
   directly if a path is suspicious).
2. For each issue write an entry to your findings file using the schema in
   `qa/TEST_PLAN.md`.
3. Do **not** edit the production code — this is a detection pass only.
   You may create helper files inside `qa/`.

## Output file

Append all your findings to **the single file** named in your brief, e.g.
`qa/findings/zcode.md`. Use the SEV labels from TEST_PLAN.md.

## Verifying expected numeric output (optional but valuable)

Where you suspect a result is wrong, recompute it independently with scipy /
statsmodels / numpy on the same data and quote both numbers. That makes the
CRITICAL/HIGH calls defensible.

## When you are done

Print a one-line summary to stdout: `done <agent-name> findings=<count>` and
exit.
</content>
