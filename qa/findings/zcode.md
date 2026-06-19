# ZCode agent — findings (Data ingest + cleaning + Compute + Missing)

Detector: ZCode (`zai/glm-5.2`)
Dataset: `qa/cohort_test.csv` (100 rows × 11 cols), loaded via the
`boot()` TestClient helper from `qa/run_via_testclient.py`. All endpoints hit
through FastAPI's `TestClient`; numbers re-derived independently with
pandas/numpy where a wrong result was suspected.

Scope covered (per `qa/briefs/zcode.md`): upload + type detection, Dictionary
kind/metadata, cell edits / delete-row / undo, Select Cases, Compute
(formula / transform / recode / clinical), Missing audit + MICE, Convert-value
(find & replace), parse-as-date.

Cross-cutting root cause. The cohort's `bmi` column is genuinely numeric but
arrives as `object`/`text` because three cells break `pd.read_csv` numeric
inference — `"25,9"` / `"34,3"` / `"20,9"` (comma-decimals), `"999"` (sentinel),
and `"n/a"` (read_csv *does* fold `"n/a"` to NaN, so it is fine). Because the
column stays `object`, every downstream that wants a number re-coerces with
`pd.to_numeric(errors="coerce")` — and that coercion is **lossy and inconsistent**:
the comma-decimals silently become NaN (data loss) and the `999` sentinel
silently becomes a real 999 (poisons means / models). Several findings below are
manifestations of this one root cause; they are listed separately because they
surface in different features with different user-visible symptoms.

---

## [CRITICAL] `bmi` with comma-decimals is silently dropped by Transform/clinical/mean — `25,9` → NaN

**Where:** Data → bmi column (object/text) → Compute → Transform (ln/sqrt/zscore…); also Compute → Clinical (H2FPEF, MAGGIC) and Fill-blanks `__mean__`.
**Steps:**
1. Upload `qa/cohort_test.csv`. `bmi` is detected as `text` (dtype `object`) because of `"25,9"`, `"999"`, `"n/a"`.
2. `POST /api/compute/{sid}/transform` `{"source_col":"bmi","transform":"ln","new_col":"ln_bmi"}`.
**Expected:** Either reject with a "please coerce bmi to numeric first" error, or coerce `"25,9"` → `25.9` and proceed. A user-visible warning that comma-decimals were reinterpreted.
**Actual:** `200 OK`, `n_computed=94, n_missing=6`. The three comma-decimal rows (`P006 25,9`, `P020 34,3`, `P042 20,9`) are silently coerced to NaN by `pd.to_numeric(errors="coerce")` (transform_compute, compute.py:229) and dropped from the result with no warning. `bmi="999"` is kept as `ln(999)=6.9`.
**Evidence:** Independent recompute — `pd.to_numeric("25,9")=NaN`, `pd.to_numeric("999")=999.0`. Transform response `n_computed=94` (100 − 3 comma − 3 blanks + 0 = 94), `n_missing=6`. No "warning"/"coerced" field in the response.
**Hypothesis:** `transform_compute` unconditionally `pd.to_numeric(..., errors="coerce")` and never reports how many cells were lost to coercion. Comma-decimal locale leakage is a documented dirty-bit in the dataset; it should be detected (e.g. "value matches `^\d+,\d+$`") and either auto-converted or surfaced.

---

## [CRITICAL] Formula builder concatenates instead of computing on a text-typed numeric column (`bmi*2` → `"30.630.6"`)

**Where:** Compute → Formula.
**Steps:**
1. Upload cohort. `bmi` is `object`/`text`.
2. `POST /api/compute/{sid}/formula` `{"formula":"bmi*2","new_col":"bmi2"}`.
**Expected:** Either an error ("bmi is text; coerce to numeric first") or numeric multiplication.
**Actual:** `200 OK`, returns a column of **strings**: preview head = `["30.630.6","34.334.3","27.127.1","27.527.5","26.026.0","25,925,9","30.630.6","32.932.9",...]`. `dtype="object"`, `kind="text"`, `n_computed=97`. Python `*` on object-Series repeats the string.
**Evidence:** Response preview above; the result column is unusable as a numeric predictor and will silently propagate as `text` into every analysis picker.
**Hypothesis:** `_eval_formula_with_custom_functions` (compute.py:102) binds `names={col: df[col]}` and lets Python operators apply element-wise; for `object` dtype, `*` is string repetition. There is no pre-check that arithmetic operands are numeric.

---

## [CRITICAL] Dictionary "kind → numeric" override does not coerce the column; `bmi*2` still concatenates after the user marks bmi numeric

**Where:** Data → Dictionary (or column badge) → set `bmi` kind to `numeric` → Compute → Formula.
**Steps:**
1. Upload cohort (bmi = object/text).
2. `POST /api/sessions/{sid}/kind` `{"column":"bmi","kind":"numeric"}` → `200 OK`.
3. `GET /api/sessions/{sid}` → bmi shows `{"dtype":"object","kind":"numeric"}`.
4. `POST /api/compute/{sid}/formula` `{"formula":"bmi*2","new_col":"bmi_dbl"}`.
**Expected:** After the user explicitly marks bmi numeric, the column should be coerced (or, at minimum, arithmetic on it should treat values as numbers). The string `"25,9"` should become `25.9`.
**Actual:** `200 OK` but `bmi*2` still returns string concatenation: `P006 (was "25,9") → "25,925,9"`, `P020 (was "34,3") → "34,334,3"`, `P023 (was "999") → "999999"`. The override is a display-only label; `df["bmi"]` is still `object` with the original strings.
**Evidence:** `set_column_kind` (session.py:615) only calls `store.save_kind_overrides` — no dtype coercion. Subsequent formula preview shows concatenation. `dtype="object"` reported alongside `kind="numeric"`.
**Hypothesis:** The kind override should attempt a best-effort numeric coercion (handling comma-decimals) when the user sets kind=numeric, or at least the formula/transform paths should honour the override by coercing before arithmetic.

---

## [HIGH] `fill_blanks __mean__` on bmi returns a mean polluted by the `999` sentinel (38.12 vs true 27.79)

**Where:** Data → context menu → Fill blanks → Mean.
**Steps:**
1. Upload cohort.
2. `POST /api/compute/{sid}/fill_blanks` `{"column":"bmi","value":"__mean__","new_column":"bmi_m"}`.
**Expected:** Mean computed over genuine BMI values, ignoring obvious sentinels (or at least flagging `999` as suspicious). True mean (excluding 999) ≈ 27.79.
**Actual:** `200 OK`, `fill_value="mean (38.12)"`. The reported mean is `38.12` because `pd.to_numeric("999")=999.0` is included. The 3 blank cells are filled with `38.12`; the `999` row is left untouched (not treated as missing), so a sentinel both inflates the imputation and remains in the column.
**Evidence:** Independent: `pd.to_numeric(bmi).mean()=38.12` (with 999) vs `bmi[bmi<100].mean()=27.79` (without). `fill_blanks` response `n_filled=3` (only the true NaNs, not the 999).
**Hypothesis:** `fill_blanks` `__mean__` branch (compute.py:999) computes the mean over the coerced column without any outlier/sentinel guard.

---

## [HIGH] MICE mis-reports `n_imputed` and silently imputes a text-missing cell; the `999` sentinel is kept as a real value and poisons PMM

**Where:** Missing → MICE (also reachable via Models → Survival Advanced → MICE).
**Steps:**
1. Upload cohort.
2. `POST /api/survival_advanced/mice` `{"session_id":sid,"columns":["bmi"],"n_imputations":1,"max_iter":5,"new_columns":false}`.
3. Inspect stored `bmi` for `P009` (blank), `P023` (`"999"`), `P048` (blank), `P055` (was `"n/a"` → already NaN), `P089` (blank).
**Expected:** `n_imputed` to equal the number of cells actually changed; sentinels like `999` to be flagged or excluded from the imputation model.
**Actual:** Response says `n_imputed=3`, `total_imputed=3`, `result_text="3 missing values were imputed"`. But the stored column shows **P055 was also changed** (`"n/a"`→NaN at read_csv→imputed to 24.8) without being counted, and **P023 stays `999.0`** — a value of 999 is fed into the PMM model as a real observation, biasing every imputed draw upward. So: undercount (P055 changed but not reported) + sentinel poisoning (999 in the model) in a single run.
**Evidence:** Stored values after MICE-in-place: `P009=21.4, P023=999.0, P048=25.1, P055=24.8, P089=21.6`. Response `summary=[{column:bmi, method:PMM, n_imputed:3, mean_imputed:23.77, min:21.4, max:25.1}]`. P055 (24.8) is outside [21.4, 25.1]? Actually 24.8 is inside — but it was changed silently and not counted; P023 (999) is kept and not in the imputed range, yet it participated in PMM fitting.
**Hypothesis:** `_missing_mask` (survival_advanced_service.py:520) flags only `isna()` + blank-string; it doesn't recognise sentinels. Then `_is_numeric` (line 536) says bmi is numeric because `pd.to_numeric` parses ≥80% of cells (the `999` parses, the comma-decimals and `n/a`→NaN don't, but 80% threshold is met), so PMM runs on a column where `999` is a real data point and `"n/a"` becomes an unreported NaN inside the imputer.

---

## [HIGH] H2FPEF (and any clinical score using bmi) silently mishandles comma-decimals and the `999` sentinel

**Where:** Compute → Clinical → H2FPEF (also MAGGIC, eGFR-when-bmi-mapped, etc. — any calculator that reads bmi).
**Steps:**
1. Upload cohort.
2. `POST /api/compute/{sid}/clinical/h2fpef` `{"column_map":{"bmi":"bmi","age":"age"},"new_col":"h2"}`.
3. Compare `P020` (bmi `"34,3"`, truly obese, age 85) and `P023` (bmi `"999"`, sentinel).
**Expected:** P020 → Heavy (BMI>30)=2 + Elderly(age>60)=1 = at least 3. P023 → sentinel recognised, score should not count "obese".
**Actual:** P020 → `h2=1` (only the elderly point; the comma-decimal `"34,3"` is `pd.to_numeric`→NaN→`NaN>30=False`, so the obesity point is missed). P023 → `h2=2` (`999>30=True`, so a sentinel is counted as obesity). P006 (`"25,9"`, correctly non-obese) → `h2=1` by accident.
**Evidence:** Independent: `pd.to_numeric("34,3")=NaN`, `pd.to_numeric("999")=999.0`, `999>30=True`. uSTAT H2FPEF outputs P020=1, P023=2.
**Hypothesis:** `_num` (compute.py:539) does `pd.to_numeric(..., errors="coerce")` with no comma-decimal repair and no sentinel guard; the same defect affects every clinical calculator that reads a comma-decimal/sentinel-laden column.

---

## [HIGH] Every missing-data audit undercounts bmi missingness by ignoring the `999` sentinel (reported 3; real-world "needs review" is 4+)

**Where:** Missing tab; also Data-tab header badge; also `GET /api/stats/{sid}/missing`; also `POST /api/missing_data/pattern`.
**Steps:**
1. Upload cohort.
2. Query all three missingness sources: `POST /api/compute/{sid}/missing_diagnostics`, `POST /api/missing_data/pattern`, `GET /api/stats/{sid}/missing`.
**Expected:** The Missing audit to flag `bmi` as having 3 NaN cells **plus** at least a warning that one cell is a likely sentinel (`999`), since `999` is a classic missing-code for BMI.
**Actual:** All three sources agree on `bmi: n_missing=3` and say nothing about the `999`. The audits are mutually consistent (good) but all three miss the sentinel. A user relying on the audit believes bmi is 97% complete when one of those 97 is a `999` that will distort any mean/regression.
**Evidence:** `missing_diagnostics` → `bmi: n_missing=3, pct=3.0`; `missing_data/pattern` → `bmi: n_missing=3, pct_missing=3.0`; `stats/{sid}/missing` → `bmi: count=3, pct=3.0`. Independent: bmi has 3 NaN + 1 sentinel (`999`) + 3 comma-decimals (valid but unread).
**Hypothesis:** All three paths use `isna()` (plus blank-string for object), with no outlier/sentinel heuristic. The Data-tab badge inherits the same number.

---

## [MEDIUM] `clean_outliers` on `age` deletes 5 rows but the count doesn't match the obvious impossible values (−5, three 199s)

**Where:** Data → Clean outliers (IQR).
**Steps:**
1. Upload cohort.
2. `POST /api/compute/{sid}/clean_outliers` `{"columns":["age"],"method":"iqr","threshold":1.5}`.
**Expected:** Drop the clearly impossible ages (−5, 199, 199, 199) — 4 rows — and report which fence was used.
**Actual:** `deleted=5, remaining_rows=95`. IQR fence is `[Q1−1.5·IQR, Q3+1.5·IQR]`; with this cohort that fence also catches one additional borderline age (the function is mathematically right), but the response gives no per-row detail, no fence values, and no breakdown of "impossible vs statistical" — so a user can't tell whether a genuine outlier or a data-entry error was removed.
**Evidence:** Response `{"deleted":5,"remaining_rows":95}` only. Independent IQR on age gives fence ≈ [21,101], which excludes −5, the three 199s, and one age ≤ 31.
**Hypothesis:** `clean_outliers` (compute.py:1462) returns only counts; consider returning the fence bounds and the deleted row IDs, and separately flagging physiologically impossible values (age<0, age>120).

---

## [MEDIUM] `select_cases` silently selects ALL rows when given an operator it doesn't recognise (`==` instead of `eq`)

**Where:** Data → Select Cases (defence-in-depth; the shipped frontend always sends `eq/ne/gt/...`, but any third-party/edited client, or a typo in a future operator, triggers this).
**Steps:**
1. Upload cohort.
2. `POST /api/sessions/{sid}/select_cases` `{"conditions":[{"column":"sex","operator":"==","value":"F","join":"AND"}],"apply":true}`.
**Expected:** A 422 error listing the valid operators (`eq,ne,gt,lt,gte,lte,contains,missing,not_missing`), since `==` is not in the accepted set.
**Actual:** `200 OK`, `selected=100, total=100`. The unknown operator falls through `_apply_conditions`'s `else` branch (store.py) which defaults `cond_mask = all True`, so the filter matches every row — and if `apply=true`, the session filter is silently set to "everything".
**Evidence:** With `operator:"eq"` → `selected=39` (correct). With `operator:"=="` → `selected=100`. The frontend's `OPERATORS` list (`SelectCasesModal.tsx:6`) only emits the lowercase names, so this is latent rather than user-triggered today.
**Hypothesis:** `_apply_conditions` treats an unknown operator as "match all" instead of raising.

---

## [MEDIUM] `find_replace` returns `replaced_count=0` and a misleading "nothing happened" signal when the find value isn't present, even though the operation is a no-op success

**Where:** Data → Find & Replace.
**Steps:**
1. Upload cohort. Note `"n/a"` was already folded to NaN by `pd.read_csv`, so `bmi` has no literal `"n/a"` cell.
2. `POST /api/compute/{sid}/find_replace` `{"columns":["bmi"],"find_value":"n/a","replace_value":"25"}`.
**Expected:** Either a clear "0 matches — nothing replaced" status, or guidance that `"n/a"` was auto-converted to missing at upload time.
**Actual:** `200 OK, replaced_count=0` — indistinguishable from a successful run that genuinely matched nothing. A user who knows their CSV had `"n/a"` and tries to recode it will believe the tool ignored them.
**Evidence:** Response `{"replaced_count":0}`. Independent: `bmi` has no `"n/a"` string (read_csv NA inference). Contrast: `find_value:"999"` → `replaced_count=1` (correct).
**Hypothesis:** `find_replace` (compute.py:1500) returns only a count; no per-column "matched / not-found" distinction, and no surfacing of read_csv's silent NA folding.

---

## [LOW] `_is_female` with an explicit `female_value="F"` silently treats `sex="Female"` as male in CHA₂DS₂-VASc (and every sex-aware clinical calc)

**Where:** Compute → Clinical → CHA₂DS₂-VASc / eGFR / MAGGIC (any calculator using sex).
**Steps:**
1. Upload cohort. `sex` has `F`, `M`, `Female` (P071), `x` (P034), and 2 blanks.
2. `POST /api/compute/{sid}/clinical/chadsvasc` `{"column_map":{"age":"age","sex":"sex","dm":"diabetes"},"female_value":"F","new_col":"c2v"}`.
**Expected:** Either accept `Female` as female, or warn that the supplied `female_value` doesn't cover all variants in the column.
**Actual:** P071 (`sex="Female"`, age 51) → `c2v=0` (no female point). With `female_value=null` (auto-detect) → `c2v=1` (correct). The explicit value short-circuits the auto-detector that would otherwise catch `female`/`women`/etc.
**Evidence:** `_is_female` (compute.py:433): when `female_value is not None`, it does `col == str(female_value)` and skips the `isin(["f","female",...])` branch. P071 score flips 0→1 between the two calls.
**Hypothesis:** Intentional (explicit override wins), but the UI lets the user pick a single value while the column has several synonyms; either union-match the synonyms or warn on unmapped values.

---

## [LOW] `recode` produces an all-NaN column with no warning when no rule matches and `else_val` is empty

**Where:** Compute → Recode.
**Steps:**
1. Upload cohort.
2. `POST /api/compute/{sid}/recode` `{"rules":[{"conditions":[{"col":"age","op":">","val":1000}],"result":99}],"else_val":null,"new_col":"never_match"}`.
**Expected:** A warning that 0/100 rows matched any rule (the recode effectively created an empty column), or a prompt to provide an `else` value.
**Actual:** `200 OK`, `n_computed=0, n_missing=100`, preview all `null`. A fully-empty column is silently added to the dataset and will show up in every downstream picker as a valid (but useless) variable.
**Evidence:** Response `{"name":"never_match","dtype":"float64","kind":"categorical","n_computed":0,"n_missing":100,"preview_values":[null,null,...]}`. No warning field.
**Hypothesis:** `recode_compute` (compute.py:302) has no "zero matches" guard analogous to the existing "need ≥2 distinct values" guard in `_quantile_groups`.

---

## Summary

12 findings: 3 CRITICAL, 4 HIGH, 3 MEDIUM, 2 LOW.

The single largest leverage point is the **comma-decimal + sentinel handling on
`bmi`**: fixing the upload-time coercion (detect `^\d+,\d+$`, rewrite to `.`,
and flag/strip obvious sentinels like `999`) would resolve findings #1, #3, #5,
#6, #7 and large parts of #2/#4 in one pass. The Dictionary "kind → numeric"
override (#3) should additionally perform that same coercion at override time so
the user's explicit type choice actually takes effect on the stored data.
