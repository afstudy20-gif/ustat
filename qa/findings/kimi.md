## [MEDIUM] Independent t-test (age ~ sex) fails because dirty sex codes inflate group count
**Where:** Tests → Hypothesis → Independent t-test
**Steps:** 1) POST /api/stats/ttest with column=age, group_column=sex on cohort_test.csv
**Expected:** 200 with statistics using only valid M/F rows (or clear recoding of invalid sex entries)
**Actual:** status=400, body={"detail":"Group column must have exactly 2 groups"}
**Evidence:** HTTP 400 response
**Hypothesis:** The test rejects the column because it sees 4 unique sex values instead of dropping/recoding '', 'x', 'Female'.

## [CRITICAL] ANCOVA endpoint fails on age ~ sex + ldl
**Where:** Tests → Hypothesis → ANCOVA
**Steps:** 1) POST /api/advanced_anova/ancova with outcome=age, group_col=sex, covariates=[ldl]
**Expected:** 200 with adjusted means / F / p
**Actual:** status=-1, body=exception: Out of range float values are not JSON compliant
Traceback (most recent call last):
  File "/Users/yh/Documents/projects/wiz3/qa/kimi_audit.py", line 84, in call
    r = self.client.post(path, json=payload)
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/starlette/testclient.py", line 555, in post
    return super().post(
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/httpx/_client.py", line 1144, in post
    r
**Evidence:** HTTP -1

## [CRITICAL] Two-way ANOVA fails on age by sex and diabetes
**Where:** Tests → Hypothesis → Two-way ANOVA
**Steps:** 1) POST /api/advanced_anova/two_way_anova with outcome=age, factor1=sex, factor2=diabetes
**Expected:** 200 with effects table
**Actual:** status=-1, body=exception: Out of range float values are not JSON compliant
Traceback (most recent call last):
  File "/Users/yh/Documents/projects/wiz3/qa/kimi_audit.py", line 84, in call
    r = self.client.post(path, json=payload)
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/starlette/testclient.py", line 555, in post
    return super().post(
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/httpx/_client.py", line 1144, in post
    r
**Evidence:** HTTP -1

## [MEDIUM] Mann-Whitney U (bmi ~ sex) fails because sex has more than 2 levels
**Where:** Tests → Hypothesis → Mann-Whitney U
**Steps:** 1) POST /api/stats/mannwhitney with column=bmi, group_column=sex
**Expected:** 200 with U statistic; invalid sex codes should be recoded/dropped
**Actual:** status=400, body={"detail":"Group column must have exactly 2 groups"}
**Evidence:** HTTP 400

## [CRITICAL] Mann-Whitney U crashes on comma-decimal BMI values
**Where:** Tests → Hypothesis → Mann-Whitney U
**Steps:** 1) POST /api/stats/mannwhitney with column=bmi, group_column=diabetes
**Expected:** 200 with U statistic after coercing comma decimals or excluding non-numeric BMI values
**Actual:** status=-1, body=exception: could not convert string to float: '34,3'
Traceback (most recent call last):
  File "/Users/yh/Documents/projects/wiz3/qa/kimi_audit.py", line 84, in call
    r = self.client.post(path, json=payload)
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/starlette/testclient.py", line 555, in post
    return super().post(
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/httpx/_client.py", line 1144, in post
    return s
**Evidence:** HTTP -1
**Hypothesis:** The endpoint calls astype(float) on the BMI column without locale-aware parsing, so '30,6' raises a Python exception.

## [CRITICAL] Kruskal-Wallis (bmi ~ nyha) fails
**Where:** Tests → Hypothesis → Kruskal-Wallis
**Steps:** 1) POST /api/stats/kruskal with column=bmi, group_column=nyha
**Expected:** 200 with H statistic and per-group n
**Actual:** status=-1, body=exception: could not convert string to float: '34,3'
Traceback (most recent call last):
  File "/Users/yh/Documents/projects/wiz3/qa/kimi_audit.py", line 84, in call
    r = self.client.post(path, json=payload)
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/starlette/testclient.py", line 555, in post
    return super().post(
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/httpx/_client.py", line 1144, in post
    return s
**Evidence:** HTTP -1

## [HIGH] Chi-square treats invalid sex codes as separate columns
**Where:** Tests → Categorical → Chi-square
**Steps:** 1) Run diabetes × sex chi-square
**Expected:** Invalid sex entries ('', 'x', 'Female') should be dropped or aggregated into a single 'other/missing' category
**Actual:** Returned columns include non-M/F keys: {'Female', 'x', 'F', 'M'}
**Evidence:** table={'F': {'0.0': 26, '1.0': 12}, 'Female': {'0.0': 1, '1.0': 0}, 'M': {'0.0': 42, '1.0': 14}, 'x': {'0.0': 1, '1.0': 0}}
**Hypothesis:** The cross-tab does not clean/recode the grouping variable before building the table.

## [MEDIUM] Two-proportion z endpoint fails on cohort_test.csv
**Where:** Tests → Categorical → Two-proportion z
**Steps:** 1) POST /api/categorical/two_proportions with {'session_id': 'a6ae1446-d25d-4bf4-a6b4-c6a21da685c1', 'column': 'event', 'group_column': 'sex'}
**Expected:** 200 with valid result
**Actual:** status=400, body={"detail":"Group column must have exactly 2 groups, found 4."}
**Evidence:** HTTP 400

## [MEDIUM] Cochran's Q endpoint fails on cohort_test.csv
**Where:** Tests → Categorical → Cochran's Q
**Steps:** 1) POST /api/categorical/cochran_q with {'session_id': 'a6ae1446-d25d-4bf4-a6b4-c6a21da685c1', 'columns': ['event', 'diabetes']}
**Expected:** 200 with valid result
**Actual:** status=400, body={"detail":"Cochran's Q test requires at least 3 binary columns."}
**Evidence:** HTTP 400

## [CRITICAL] Mantel-Haenszel endpoint fails on cohort_test.csv
**Where:** Tests → Categorical → Mantel-Haenszel
**Steps:** 1) POST /api/categorical/mantel_haenszel with {'session_id': 'a6ae1446-d25d-4bf4-a6b4-c6a21da685c1', 'row_col': 'event', 'col_col': 'diabetes', 'strata_col': 'sex'}
**Expected:** 200 with valid result
**Actual:** status=-1, body=exception: Out of range float values are not JSON compliant
Traceback (most recent call last):
  File "/Users/yh/Documents/projects/wiz3/qa/kimi_audit.py", line 84, in call
    r = self.client.post(path, json=payload)
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/starlette/testclient.py", line 555, in post
    return super().post(
  File "/Users/yh/Documents/projects/wiz3/backend/.venv/lib/python3.10/site-packages/httpx/_client.py", line 1144, in post
    r
**Evidence:** HTTP -1

## [HIGH] Cronbach's α is outside [0,1]
**Where:** Tests → Reliability → Cronbach's α
**Steps:** 1) Run Cronbach on bmi,ldl,sbp
**Expected:** Alpha in [0,1]
**Actual:** alpha=-0.1672
**Evidence:** response={'test': "Cronbach's Alpha Reliability Analysis", 'alpha': -0.1672, 'omega': 0.0002, 'n': 83, 'k': 3, 'significant': False, 'effect_sizes': [{'name': "Cronbach's alpha", 'value': -0.1672, 'magnitude': 'Unacceptable'}, {'name': "McDonald's omega", 'value': 0.0002, 'magnitude': 'Unacceptable'}], 'assumptions': [], 'item_stats': [{'item': 'bmi', 'mean': 39.7313, 'sd': 106.6804, 'item_total_r': -0.1598, 'alpha_if_deleted': 0.0125}, {'item': 'ldl', 'mean': 121.8096, 'sd': 31.743, 'item_total_r': -0.0203, 'alpha_if_deleted': -0.2163}, {'item': 'sbp', 'mean': 140.1084, 'sd': 20.5747, 'item_total_r': -0.251, 'alpha_if_deleted': -0.0232}], 'scale_summary': {'mean': 301.6494, 'sd': 107.3622, 'min': 199.4, 'max': 1206.9, 'skewness': 7.4438}, 'interpretation': 'Unacceptable', 'result_text': "A reliability analysis was conducted on a 3-item scale (n = 83). Cronbach's alpha was -0.167, indicating unacceptable internal consistency. The scale mean was 301.65 (SD = 107.36). Item-total correlations ranged from -0.251 to -0.020. McDonald's omega was 0.000.", 'export_rows': [['Item', 'Mean', 'SD', 'Item-Total r', 'Alpha if Deleted'], ['bmi', 39.7313, 106.6804, -0.1598, 0.0125], ['ldl', 121.8096, 31.743, -0.0203, -0.2163], ['sbp', 140.1084, 20.5747, -0.251, -0.0232], ['Scale Total', 301.6494, 107.3622, '', -0.1672]], 'r_code': 'library(psych)\nalpha(data[, c("bmi", "ldl", "sbp")])\nomega(data[, c("bmi", "ldl", "sbp")])'}

## [HIGH] Cohen's κ treats integer and float category labels as distinct categories
**Where:** Tests → Reliability → Cohen's κ
**Steps:** 1) Run Cohen's κ on event vs diabetes
**Expected:** A single 2×2 agreement table with categories 0 and 1
**Actual:** labels=['0', '0.0', '1', '1.0'], kappa=0.0, se=0.0, n=98
**Evidence:** confusion_matrix=[[0, 43, 0, 18], [0, 0, 0, 0], [0, 29, 0, 8], [0, 0, 0, 0]]
**Hypothesis:** event is stored as int 0/1 and diabetes as float 0.0/1.0; κ should unify the category encoding before building the table.

## [MEDIUM] Combined ROC model reports AUC < 0.5 without flipping direction
**Where:** ROC → Combined model
**Steps:** 1) Run combined ROC with ldl+sbp+age vs event
**Expected:** AUC should be ≥ 0.5 (invert predicted probability if the model direction is negative)
**Actual:** auc=0.4277
**Evidence:** response_keys=['test', 'model_name', 'predictors', 'n', 'n_positive', 'n_negative', 'auc', 'optimal', 'curve', 'result_text']
**Hypothesis:** Logistic model coefficients produce probabilities that are inversely related to the outcome; the endpoint does not auto-flip the classifier.

## [MEDIUM] TOST fails on sbp ~ sex because dirty sex codes create >2 groups
**Where:** Tests → Non-Inferiority → TOST
**Steps:** 1) POST /api/stats/tost with column=sbp, group_column=sex, low=-5, high=5
**Expected:** 200 with TOST result using only M/F or recoding invalid sex codes
**Actual:** status=422, body={"detail":"group_column must have exactly 2 levels, found 4."}
**Evidence:** HTTP 422

## [MEDIUM] Non-inferiority fails on event ~ sex because dirty sex codes create >2 groups
**Where:** Tests → Non-Inferiority
**Steps:** 1) POST /api/stats/noninferiority with outcome=event, group=sex
**Expected:** 200 with non-inferiority conclusion using only M/F
**Actual:** status=422, body={"detail":"Group column must have exactly 2 levels; found 4: ['F', 'Female', 'M', 'x']"}
**Evidence:** HTTP 422

## [MEDIUM] Bayesian t-test fails on age ~ sex after dropping all sex groups
**Where:** Tests → Bayesian Statistics → Bayesian t-test
**Steps:** 1) POST /api/bayesian analysis_type=ttest_ind, outcome=age, predictor=sex
**Expected:** 200 with BF10 (positive) using valid M/F rows
**Actual:** status=400, body={"detail":"Grouping variable must have exactly 2 groups. Found: []"}
**Evidence:** HTTP 400
**Hypothesis:** The cleaning/filtering step removes every row because it cannot reconcile the dirty sex labels, leaving zero groups.

