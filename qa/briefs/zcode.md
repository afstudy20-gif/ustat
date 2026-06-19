# ZCode brief — Data ingest + cleaning + Compute + Missing

Read `/Users/yh/Documents/projects/wiz3/qa/briefs/common.md` first.

Your slice: everything the user touches **before** running a statistical test.

## Scope

| Area | Endpoints / panels to exercise |
|------|-------------------------------|
| Upload + type detection | `POST /api/upload/`, `GET /api/sessions/{sid}` — does it sniff numeric vs categorical vs date correctly on the dirty cohort? |
| Dictionary / type override | `POST /api/sessions/{sid}/kind`, `POST /api/sessions/{sid}/metadata` |
| Cell edits / delete row / undo | `POST /api/sessions/{sid}/undo`, `redo`, `/row/{i}` |
| Select cases + Filter | `POST /api/sessions/{sid}/select_cases` |
| Compute — formula / transform / recode / clinical | `POST /api/compute/{formula,transform,recode,clinical}` (see `backend/routers/compute.py`) |
| Missing audit + MICE | `POST /api/survival_advanced/mice`, `/api/missing/*` |
| Convert value modal (find & replace) | `POST /api/compute/...` for value swap |

## Deliberate breakage to probe

- `bmi` has comma-decimals: do they become numeric `30.6` or stay as text?
- `bmi="999"` and `bmi="n/a"` rows: does the type-sniffer keep the column numeric?
- `age` negatives / extreme values: any validation? Or do they flow into a
  later mean / regression silently?
- `admission_date` mixed formats + one impossible: does Parse-as-date handle?
- Recode rule that produces an all-empty new column: is there a warning?
- Missing values shown vs counted vs imputed — three numbers; are they consistent?
- Undo after deleting a row: does the new column added by Compute survive?

## Output

`/Users/yh/Documents/projects/wiz3/qa/findings/zcode.md`

Use the finding schema from `qa/TEST_PLAN.md`. Number-of-findings target: open-ended, but expect 8-15 on this slice. Finish with a one-line stdout summary.
</content>
