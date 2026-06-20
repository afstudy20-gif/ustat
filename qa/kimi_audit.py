"""Detection audit for the Kimi slice: hypothesis tests, categorical,
correlation, ROC, reliability, repeated measures, advanced ANOVA,
Bayesian, gatekeeping, factor/PCA.

Run from repo root with the backend venv active:
    cd /Users/yh/Documents/projects/wiz3
    source backend/.venv/bin/activate
    python qa/kimi_audit.py
"""
from __future__ import annotations
import sys, os, json, math, textwrap, itertools, warnings, traceback, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, "qa")
from run_via_testclient import boot

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "cohort_test.csv")
OUT = os.path.join(ROOT, "findings", "kimi.md")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_csv():
    return pd.read_csv(CSV, dtype=str, keep_default_na=False)


def has_bad_float(x):
    if x is None:
        return True
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return True
    return False


findings: list[dict] = []


def add_finding(sev, title, where, steps, expected, actual, evidence, hypothesis=None):
    findings.append(
        {
            "sev": sev,
            "title": title,
            "where": where,
            "steps": steps,
            "expected": expected,
            "actual": actual,
            "evidence": evidence,
            "hypothesis": hypothesis,
        }
    )


def write_findings():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for fd in findings:
            f.write(f"## [{fd['sev']}] {fd['title']}\n")
            f.write(f"**Where:** {fd['where']}\n")
            steps = fd["steps"]
            if isinstance(steps, (list, tuple)):
                steps = "\n".join(f"{i+1}) {s}" for i, s in enumerate(steps))
            f.write(f"**Steps:** {steps}\n")
            f.write(f"**Expected:** {fd['expected']}\n")
            f.write(f"**Actual:** {fd['actual']}\n")
            f.write(f"**Evidence:** {fd['evidence']}\n")
            if fd.get("hypothesis"):
                f.write(f"**Hypothesis:** {fd['hypothesis']}\n")
            f.write("\n")
    print(f"done kimi findings={len(findings)}")


class API:
    def __init__(self, client, sid):
        self.client = client
        self.sid = sid

    def call(self, method, path, payload=None):
        try:
            if method == "post":
                r = self.client.post(path, json=payload)
            else:
                r = self.client.get(path)
            return {"status": r.status_code, "json": r.json() if r.content else None, "text": r.text}
        except Exception as e:
            return {"status": -1, "json": None, "text": f"exception: {e}\n{traceback.format_exc()}"}


def post(api, path, payload):
    return api.call("post", path, payload)


# ---------------------------------------------------------------------------
# reference computations
# ---------------------------------------------------------------------------

def ref_ttest_age_sex():
    df = load_csv()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    g = df.groupby("sex")["age"]
    counts = g.count().to_dict()
    valid = df["sex"].isin(["M", "F"])
    a = df.loc[valid & (df["sex"] == "M"), "age"].dropna()
    b = df.loc[valid & (df["sex"] == "F"), "age"].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=True, nan_policy="omit")
    return {
        "all_counts": counts,
        "valid_M_n": len(a),
        "valid_F_n": len(b),
        "t": float(t),
        "p": float(p),
    }


def ref_chisq_diabetes_sex():
    df = load_csv()
    tab = pd.crosstab(df["diabetes"], df["sex"])
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    return {"table": tab.to_dict(), "chi2": chi2, "p": p, "dof": dof}


def ref_mannwhitney_bmi_sex():
    df = load_csv()
    df["bmi_num"] = pd.to_numeric(df["bmi"].str.replace(",", ".", regex=False), errors="coerce")
    valid = df["sex"].isin(["M", "F"])
    a = df.loc[valid & (df["sex"] == "M"), "bmi_num"].dropna()
    b = df.loc[valid & (df["sex"] == "F"), "bmi_num"].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"M_n": len(a), "F_n": len(b), "U": u, "p": p}


def ref_kruskal_bmi_nyha():
    df = load_csv()
    df["bmi_num"] = pd.to_numeric(df["bmi"].str.replace(",", ".", regex=False), errors="coerce")
    groups = []
    n_per = {}
    for lev in sorted(df["nyha"].dropna().unique()):
        subset = df.loc[df["nyha"] == lev, "bmi_num"].dropna()
        n_per[str(lev)] = len(subset)
        groups.append(subset.values)
    h, p = stats.kruskal(*groups)
    return {"n_per_group": n_per, "H": h, "p": p}


def ref_roc_ldl_event():
    df = load_csv()
    df["ldl"] = pd.to_numeric(df["ldl"], errors="coerce")
    valid = df.dropna(subset=["ldl", "event"])
    y = valid["event"].astype(int)
    score = valid["ldl"].astype(float)
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    return {"n": len(valid), "auc": auc}


# ---------------------------------------------------------------------------
# audit logic
# ---------------------------------------------------------------------------

def severity_for_status(status, is_validation=False):
    if status == -1:
        return "CRITICAL"
    if status >= 500:
        return "CRITICAL"
    if status == 400 or status == 422:
        return "MEDIUM" if is_validation else "HIGH"
    return "HIGH"


def audit_ttest(api):
    r = post(api, "/api/stats/ttest", {"session_id": api.sid, "column": "age", "group_column": "sex", "mu": 0, "equal_var": True})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"], is_validation=True),
            "Independent t-test (age ~ sex) fails because dirty sex codes inflate group count",
            "Tests → Hypothesis → Independent t-test",
            ["POST /api/stats/ttest with column=age, group_column=sex on cohort_test.csv"],
            "200 with statistics using only valid M/F rows (or clear recoding of invalid sex entries)",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']} response",
            "The test rejects the column because it sees 4 unique sex values instead of dropping/recoding '', 'x', 'Female'.",
        )
        return
    data = r["json"]
    ref = ref_ttest_age_sex()
    group_stats = data.get("group_stats", {})
    reported_ns = {k: v.get("n") for k, v in group_stats.items()}
    extra = set(reported_ns.keys()) - {"M", "F"}
    if extra:
        add_finding(
            "HIGH",
            "t-test miscategorises invalid sex codes as groups",
            "Tests → Hypothesis → Independent t-test",
            ["Run age ~ sex t-test on cohort_test.csv"],
            "Only M and F groups are used; empty/'x'/'Female' are treated as missing or cause a validation error",
            f"Backend returned groups {list(reported_ns.keys())} with n={reported_ns}",
            f"groups_json={reported_ns}",
            "The test likely uses the categorical column as-is without filtering to the two expected levels.",
        )
    valid_n = ref["valid_M_n"] + ref["valid_F_n"]
    total_reported = sum(v for v in reported_ns.values() if v is not None)
    if total_reported != valid_n:
        add_finding(
            "MEDIUM",
            "t-test total n does not match valid M+F rows",
            "Tests → Hypothesis → Independent t-test",
            ["Run age ~ sex t-test"],
            f"Total n = {valid_n} (M={ref['valid_M_n']}, F={ref['valid_F_n']})",
            f"Reported total n = {total_reported}, groups={reported_ns}",
            f"expected={ref['valid_M_n']+ref['valid_F_n']}, got={total_reported}",
        )
    for key in ["t_statistic", "p_value"]:
        val = data.get(key)
        if has_bad_float(val):
            add_finding(
                "HIGH",
                f"t-test returns invalid {key}",
                "Tests → Hypothesis → Independent t-test",
                ["Run age ~ sex t-test"],
                f"Finite numeric {key}",
                f"{key}={val}",
                f"response={data}",
            )


def audit_anova(api):
    r = post(api, "/api/stats/anova", {"session_id": api.sid, "column": "age", "group_column": "diabetes"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"], is_validation=True),
            "One-way ANOVA (age ~ diabetes) fails",
            "Tests → Hypothesis → One-way ANOVA",
            ["POST /api/stats/anova with column=age, group_column=diabetes"],
            "200 with F/p stats",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )


def audit_advanced_anova(api):
    r = post(api, "/api/advanced_anova/ancova", {"session_id": api.sid, "outcome": "age", "group_col": "sex", "covariates": ["ldl"]})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "ANCOVA endpoint fails on age ~ sex + ldl",
            "Tests → Hypothesis → ANCOVA",
            ["POST /api/advanced_anova/ancova with outcome=age, group_col=sex, covariates=[ldl]"],
            "200 with adjusted means / F / p",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        if "n_excluded" not in data and "n" not in data:
            add_finding(
                "MEDIUM",
                "ANCOVA response omits sample-size / exclusion information",
                "Tests → Hypothesis → ANCOVA",
                ["Run ANCOVA age ~ sex + ldl"],
                "Response reports n and n_excluded so user knows missing-data impact",
                f"Keys: {list(data.keys())}",
                f"response_keys={list(data.keys())}",
            )

    r2 = post(api, "/api/advanced_anova/two_way_anova", {"session_id": api.sid, "outcome": "age", "factor1": "sex", "factor2": "diabetes"})
    if r2["status"] != 200:
        add_finding(
            severity_for_status(r2["status"]),
            "Two-way ANOVA fails on age by sex and diabetes",
            "Tests → Hypothesis → Two-way ANOVA",
            ["POST /api/advanced_anova/two_way_anova with outcome=age, factor1=sex, factor2=diabetes"],
            "200 with effects table",
            f"status={r2['status']}, body={r2['text'][:500]}",
            f"HTTP {r2['status']}",
        )

    r3 = post(api, "/api/advanced_anova/mancova", {"session_id": api.sid, "outcomes": ["age", "ldl"], "group_col": "diabetes", "covariates": ["sbp"]})
    if r3["status"] != 200:
        add_finding(
            severity_for_status(r3["status"]),
            "MANCOVA fails on [age,ldl] ~ diabetes + sbp",
            "Tests → Hypothesis → MANCOVA",
            ["POST /api/advanced_anova/mancova with outcomes=[age,ldl], group_col=diabetes, covariates=[sbp]"],
            "200 with Pillai etc.",
            f"status={r3['status']}, body={r3['text'][:500]}",
            f"HTTP {r3['status']}",
        )


def audit_nonparametric(api):
    r = post(api, "/api/stats/mannwhitney", {"session_id": api.sid, "column": "bmi", "group_column": "sex"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"], is_validation=True),
            "Mann-Whitney U (bmi ~ sex) fails because sex has more than 2 levels",
            "Tests → Hypothesis → Mann-Whitney U",
            ["POST /api/stats/mannwhitney with column=bmi, group_column=sex"],
            "200 with U statistic; invalid sex codes should be recoded/dropped",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        ref = ref_mannwhitney_bmi_sex()
        n = data.get("n1", 0) + data.get("n2", 0)
        if n == 0 or has_bad_float(data.get("u_statistic")):
            add_finding(
                "HIGH",
                "Mann-Whitney U returns no usable result for dirty BMI",
                "Tests → Hypothesis → Mann-Whitney U",
                ["Run bmi ~ sex Mann-Whitney"],
                f"Valid n≈{ref['M_n']+ref['F_n']} with finite U and p",
                f"n1={data.get('n1')}, n2={data.get('n2')}, U={data.get('u_statistic')}, p={data.get('p_value')}",
                f"response={data}",
                "Backend likely fails to parse comma-decimal BMI values and treats the column as non-numeric.",
            )

    # isolate comma-decimal BMI issue using a clean 2-level group
    r2 = post(api, "/api/stats/mannwhitney", {"session_id": api.sid, "column": "bmi", "group_column": "diabetes"})
    if r2["status"] != 200:
        add_finding(
            severity_for_status(r2["status"]),
            "Mann-Whitney U crashes on comma-decimal BMI values",
            "Tests → Hypothesis → Mann-Whitney U",
            ["POST /api/stats/mannwhitney with column=bmi, group_column=diabetes"],
            "200 with U statistic after coercing comma decimals or excluding non-numeric BMI values",
            f"status={r2['status']}, body={r2['text'][:500]}",
            f"HTTP {r2['status']}",
            "The endpoint calls astype(float) on the BMI column without locale-aware parsing, so '30,6' raises a Python exception.",
        )

    r = post(api, "/api/stats/kruskal", {"session_id": api.sid, "column": "bmi", "group_column": "nyha"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Kruskal-Wallis (bmi ~ nyha) fails",
            "Tests → Hypothesis → Kruskal-Wallis",
            ["POST /api/stats/kruskal with column=bmi, group_column=nyha"],
            "200 with H statistic and per-group n",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        npg = data.get("n_per_group", {})
        if not npg:
            add_finding(
                "MEDIUM",
                "Kruskal-Wallis omits n_per_group",
                "Tests → Hypothesis → Kruskal-Wallis",
                ["Run bmi ~ nyha Kruskal-Wallis"],
                "Per-group n reported for all NYHA levels present",
                f"n_per_group absent; response keys={list(data.keys())}",
                f"response={data}",
            )

    r = post(api, "/api/stats/jonckheere_terpstra", {"session_id": api.sid, "column": "age", "group_column": "nyha"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Jonckheere-Terpstra fails on age ~ nyha",
            "Tests → Hypothesis → Jonckheere-Terpstra",
            ["POST /api/stats/jonckheere_terpstra with column=age, group_column=nyha"],
            "200 with trend p-value",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )


def audit_categorical(api):
    r = post(api, "/api/stats/chisquare", {"session_id": api.sid, "row_column": "diabetes", "col_column": "sex"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Chi-square (diabetes × sex) crashes",
            "Tests → Categorical → Chi-square",
            ["POST /api/stats/chisquare with row_column=diabetes, col_column=sex"],
            "200 with χ² and contingency table",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        table = data.get("crosstab", {})
        cols = set(table.keys())
        if any(k not in ["M", "F"] for k in cols):
            add_finding(
                "HIGH",
                "Chi-square treats invalid sex codes as separate columns",
                "Tests → Categorical → Chi-square",
                ["Run diabetes × sex chi-square"],
                "Invalid sex entries ('', 'x', 'Female') should be dropped or aggregated into a single 'other/missing' category",
                f"Returned columns include non-M/F keys: {cols}",
                f"table={table}",
                "The cross-tab does not clean/recode the grouping variable before building the table.",
            )
        p = data.get("p") if "p" in data else data.get("p_value")
        if has_bad_float(data.get("chi2")) or has_bad_float(p):
            add_finding(
                "HIGH",
                "Chi-square returns NaN/inf statistic or p-value",
                "Tests → Categorical → Chi-square",
                ["Run diabetes × sex chi-square"],
                "Finite χ² and p",
                f"chi2={data.get('chi2')}, p={p}",
                f"response={data}",
            )

    r = post(api, "/api/stats/fisher", {"session_id": api.sid, "row_column": "event", "col_column": "diabetes"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Fisher exact test fails on event × diabetes",
            "Tests → Categorical → Fisher's exact",
            ["POST /api/stats/fisher with row_column=event, col_column=diabetes"],
            "200 with OR and p",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        or_ = data.get("odds_ratio")
        if or_ is not None and (has_bad_float(or_) or or_ == float("inf")):
            add_finding(
                "HIGH",
                "Fisher exact test returns infinite/NaN odds ratio",
                "Tests → Categorical → Fisher's exact",
                ["Run event × diabetes Fisher exact"],
                "A finite odds ratio (or a note that it is undefined due to a zero cell)",
                f"odds_ratio={or_}",
                f"response={data}",
                "Zero cell in 2×2 table produces inf without guard.",
            )

    for path, payload, label in [
        ("/api/categorical/binomial", {"session_id": api.sid, "column": "event"}, "Binomial"),
        ("/api/categorical/one_proportion", {"session_id": api.sid, "column": "event"}, "One-proportion z"),
        ("/api/categorical/two_proportions", {"session_id": api.sid, "column": "event", "group_column": "sex"}, "Two-proportion z"),
        ("/api/categorical/mcnemar", {"session_id": api.sid, "col1": "event", "col2": "diabetes"}, "McNemar"),
        ("/api/categorical/cochran_q", {"session_id": api.sid, "columns": ["event", "diabetes"]}, "Cochran's Q"),
        ("/api/categorical/mantel_haenszel", {"session_id": api.sid, "row_col": "event", "col_col": "diabetes", "strata_col": "sex"}, "Mantel-Haenszel"),
        ("/api/categorical/cochran_armitage", {"session_id": api.sid, "ordinal_col": "nyha", "event_col": "event"}, "Cochran-Armitage"),
    ]:
        r = post(api, path, payload)
        if r["status"] != 200:
            add_finding(
                severity_for_status(r["status"], is_validation=True),
                f"{label} endpoint fails on cohort_test.csv",
                f"Tests → Categorical → {label}",
                [f"POST {path} with {payload}"],
                "200 with valid result",
                f"status={r['status']}, body={r['text'][:500]}",
                f"HTTP {r['status']}",
            )
        else:
            data = r["json"] or {}
            for headline in ["p_value", "statistic", "chi2", "z", "or_common", "q_statistic", "p"]:
                val = data.get(headline)
                if val is not None and has_bad_float(val):
                    add_finding(
                        "HIGH",
                        f"{label} returns NaN/inf for key stat '{headline}'",
                        f"Tests → Categorical → {label}",
                        [f"Run {label}"],
                        f"Finite value for {headline}",
                        f"{headline}={val}",
                        f"response={data}",
                    )


def audit_repeated(api):
    r = post(api, "/api/repeated/paired_ttest", {"session_id": api.sid, "col1": "sbp", "col2": "ldl"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Paired t-test fails on sbp vs ldl",
            "Tests → Repeated Measures → Paired t-test",
            ["POST /api/repeated/paired_ttest with col1=sbp, col2=ldl"],
            "200 with paired t and p",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    r = post(api, "/api/repeated/wilcoxon_signed_rank", {"session_id": api.sid, "col1": "sbp", "col2": "ldl"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Wilcoxon signed-rank fails on sbp vs ldl",
            "Tests → Repeated Measures → Wilcoxon signed-rank",
            ["POST /api/repeated/wilcoxon_signed_rank with col1=sbp, col2=ldl"],
            "200 with W and p",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    r = post(api, "/api/repeated/friedman", {"session_id": api.sid, "columns": ["age", "sbp", "ldl"]})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Friedman test fails on age/sbp/ldl",
            "Tests → Repeated Measures → Friedman",
            ["POST /api/repeated/friedman with columns=[age,sbp,ldl]"],
            "200 with χ² and p",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )


def audit_reliability(api):
    r = post(api, "/api/reliability/cronbach", {"session_id": api.sid, "items": ["bmi", "ldl", "sbp"]})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Cronbach's α fails on dirty BMI/ldl/sbp",
            "Tests → Reliability → Cronbach's α",
            ["POST /api/reliability/cronbach with items=[bmi,ldl,sbp]"],
            "200 with alpha, or a clear validation error that items must be numeric",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        alpha = data.get("alpha")
        if has_bad_float(alpha):
            add_finding(
                "HIGH",
                "Cronbach's α returns NaN/inf without explanation",
                "Tests → Reliability → Cronbach's α",
                ["Run Cronbach on bmi,ldl,sbp"],
                "A finite alpha or a clear missing-data warning",
                f"alpha={alpha}",
                f"response={data}",
                "Dirty text in bmi prevents numeric conversion, producing NaN silently.",
            )
        elif alpha is not None and (alpha < 0 or alpha > 1):
            add_finding(
                "HIGH",
                "Cronbach's α is outside [0,1]",
                "Tests → Reliability → Cronbach's α",
                ["Run Cronbach on bmi,ldl,sbp"],
                "Alpha in [0,1]",
                f"alpha={alpha}",
                f"response={data}",
            )

    r = post(api, "/api/stats/icc", {"session_id": api.sid, "rater1_col": "event", "rater2_col": "diabetes"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "ICC endpoint fails on binary raters",
            "Tests → Reliability → ICC",
            ["POST /api/stats/icc with event,diabetes"],
            "200 with ICC estimate (or validation note if not applicable)",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    r = post(api, "/api/stats/cohens_kappa", {"session_id": api.sid, "rater1_col": "event", "rater2_col": "diabetes"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Cohen's κ endpoint fails on binary raters",
            "Tests → Reliability → Cohen's κ",
            ["POST /api/stats/cohens_kappa with event,diabetes"],
            "200 with kappa",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        labels = data.get("labels", [])
        if set(labels) == {"0", "0.0", "1", "1.0"} or len(set(labels)) > 2:
            add_finding(
                "HIGH",
                "Cohen's κ treats integer and float category labels as distinct categories",
                "Tests → Reliability → Cohen's κ",
                ["Run Cohen's κ on event vs diabetes"],
                "A single 2×2 agreement table with categories 0 and 1",
                f"labels={labels}, kappa={data.get('kappa')}, se={data.get('se')}, n={data.get('n')}",
                f"confusion_matrix={data.get('confusion_matrix')}",
                "event is stored as int 0/1 and diabetes as float 0.0/1.0; κ should unify the category encoding before building the table.",
            )


def audit_correlation(api):
    for method in ["pearson", "spearman", "kendall"]:
        r = post(api, "/api/stats/correlation_pair", {"session_id": api.sid, "var1": "sbp", "var2": "ldl", "method": method})
        if r["status"] != 200:
            add_finding(
                severity_for_status(r["status"]),
                f"Correlation pair ({method}) fails on sbp vs ldl",
                f"Correlation → {method.capitalize()}",
                [f"POST /api/stats/correlation_pair method={method}"],
                "200 with r and p",
                f"status={r['status']}, body={r['text'][:500]}",
                f"HTTP {r['status']}",
            )
        else:
            data = r["json"] or {}
            r_val = data.get("r") if "r" in data else data.get("correlation")
            p_val = data.get("p_value") if "p_value" in data else data.get("p")
            if has_bad_float(r_val) or has_bad_float(p_val):
                add_finding(
                    "HIGH",
                    f"Correlation pair ({method}) returns NaN/inf",
                    f"Correlation → {method.capitalize()}",
                    [f"Run {method} on sbp vs ldl"],
                    "Finite r and p",
                    f"r={r_val}, p={p_val}",
                    f"response={data}",
                )
    r = post(api, "/api/stats/correlation_matrix", {"session_id": api.sid, "variables": ["age", "sbp", "ldl"], "method": "pearson"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Correlation matrix fails on age/sbp/ldl",
            "Correlation → Matrix",
            ["POST /api/stats/correlation_matrix"],
            "200 with correlation matrix",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )


def audit_roc(api):
    r = post(api, "/api/stats/roc", {"session_id": api.sid, "score_column": "ldl", "outcome_column": "event"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "ROC analysis fails on ldl vs event",
            "ROC → ROC curve",
            ["POST /api/stats/roc with score_column=ldl, outcome_column=event"],
            "200 with AUC and curve",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        auc = data.get("auc")
        if has_bad_float(auc):
            add_finding(
                "HIGH",
                "ROC returns NaN/inf AUC for ldl vs event",
                "ROC → ROC curve",
                ["Run ROC ldl vs event"],
                "Finite AUC; rows with missing ldl should be excluded",
                f"auc={auc}",
                f"response={data}",
                "Missing LDL values may not be dropped before roc_curve.",
            )
        elif data.get("n") != ref_roc_ldl_event()["n"]:
            add_finding(
                "MEDIUM",
                "ROC sample size does not reflect listwise deletion of missing LDL",
                "ROC → ROC curve",
                ["Run ROC ldl vs event"],
                f"n = {ref_roc_ldl_event()['n']} after dropping rows with missing ldl/event",
                f"reported n={data.get('n')}",
                f"expected_n={ref_roc_ldl_event()['n']}, response={data}",
            )

    r = post(api, "/api/stats/roc_compare", {"session_id": api.sid, "score_column_1": "ldl", "score_column_2": "sbp", "outcome_column": "event"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "DeLong ROC comparison fails on ldl vs sbp",
            "ROC → DeLong compare",
            ["POST /api/stats/roc_compare with ldl, sbp, event"],
            "200 with AUCs and p-value",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        p = data.get("p") if "p" in data else data.get("p_value")
        if has_bad_float(p):
            add_finding(
                "HIGH",
                "DeLong comparison returns NaN/inf p-value",
                "ROC → DeLong compare",
                ["Run DeLong ldl vs sbp"],
                "Finite p-value",
                f"p={p}",
                f"response={data}",
            )

    r = post(api, "/api/stats/roc_multi_compare", {"session_id": api.sid, "score_columns": ["ldl", "sbp"], "outcome_column": "event"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Multi-curve ROC comparison fails",
            "ROC → Multi-curve",
            ["POST /api/stats/roc_multi_compare with [ldl,sbp], event"],
            "200 with pairwise p matrix",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )

    r = post(api, "/api/stats/roc_combined", {"session_id": api.sid, "predictor_columns": ["ldl", "sbp", "age"], "outcome_column": "event"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Combined ROC/logistic model fails on ldl+sbp+age",
            "ROC → Combined model",
            ["POST /api/stats/roc_combined with predictors=[ldl,sbp,age], outcome=event"],
            "200 with combined AUC",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    else:
        data = r["json"]
        auc = data.get("auc")
        if auc is not None and float(auc) < 0.5:
            add_finding(
                "MEDIUM",
                "Combined ROC model reports AUC < 0.5 without flipping direction",
                "ROC → Combined model",
                ["Run combined ROC with ldl+sbp+age vs event"],
                "AUC should be ≥ 0.5 (invert predicted probability if the model direction is negative)",
                f"auc={auc}",
                f"response_keys={list(data.keys())}",
                "Logistic model coefficients produce probabilities that are inversely related to the outcome; the endpoint does not auto-flip the classifier.",
            )


def audit_noninferiority_tost(api):
    r = post(api, "/api/stats/tost", {"session_id": api.sid, "column": "sbp", "group_column": "sex", "low": -5, "high": 5})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"], is_validation=True),
            "TOST fails on sbp ~ sex because dirty sex codes create >2 groups",
            "Tests → Non-Inferiority → TOST",
            ["POST /api/stats/tost with column=sbp, group_column=sex, low=-5, high=5"],
            "200 with TOST result using only M/F or recoding invalid sex codes",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )
    r = post(api, "/api/stats/noninferiority", {"session_id": api.sid, "outcome_col": "event", "group_col": "sex", "margin": 0.15, "effect": "RD", "bound": "upper"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"], is_validation=True),
            "Non-inferiority fails on event ~ sex because dirty sex codes create >2 groups",
            "Tests → Non-Inferiority",
            ["POST /api/stats/noninferiority with outcome=event, group=sex"],
            "200 with non-inferiority conclusion using only M/F",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )


def audit_bayesian(api):
    r = post(api, "/api/bayesian", {"session_id": api.sid, "analysis_type": "ttest_ind", "outcome": "age", "predictor": "sex"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"], is_validation=True),
            "Bayesian t-test fails on age ~ sex after dropping all sex groups",
            "Tests → Bayesian Statistics → Bayesian t-test",
            ["POST /api/bayesian analysis_type=ttest_ind, outcome=age, predictor=sex"],
            "200 with BF10 (positive) using valid M/F rows",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
            "The cleaning/filtering step removes every row because it cannot reconcile the dirty sex labels, leaving zero groups.",
        )
    else:
        data = r["json"]
        bf = data.get("bf10") or data.get("bf") or data.get("bayes_factor")
        if bf is not None and float(bf) < 0:
            add_finding(
                "HIGH",
                "Bayesian t-test reports negative Bayes factor",
                "Tests → Bayesian Statistics → Bayesian t-test",
                ["Run Bayesian t-test age ~ sex"],
                "BF10 should be a positive number (or its log can be negative)",
                f"bf={bf}",
                f"response={data}",
                "Sign of the mean difference may have been returned as the BF.",
            )
    r = post(api, "/api/bayesian", {"session_id": api.sid, "analysis_type": "correlation", "outcome": "sbp", "predictor": "ldl"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"], is_validation=True),
            "Bayesian correlation fails on sbp ~ ldl",
            "Tests → Bayesian Statistics → Bayesian correlation",
            ["POST /api/bayesian analysis_type=correlation outcome=sbp predictor=ldl"],
            "200 with BF",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )


def audit_gatekeeping(api):
    r = post(api, "/api/multiplicity/gatekeeping", {
        "session_id": api.sid,
        "families": [
            {"name": "primary", "hypotheses": [{"label": "h1", "p": 0.01}]},
            {"name": "secondary", "hypotheses": [{"label": "h2", "p": 0.04}]},
        ],
        "method": "hochberg",
        "logic": "serial",
    })
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"], is_validation=True),
            "Gatekeeping endpoint fails",
            "Tests → Gatekeeping",
            ["POST /api/multiplicity/gatekeeping with two serial families"],
            "200 with adjusted alphas / rejections",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )


def audit_factor(api):
    r = post(api, "/api/factor/factor_pca", {"session_id": api.sid, "items": ["age", "sbp", "ldl"], "extraction": "pca"})
    if r["status"] != 200:
        add_finding(
            severity_for_status(r["status"]),
            "Factor/PCA fails on age/sbp/ldl",
            "Tests → Factor Analysis → PCA",
            ["POST /api/factor/factor_pca items=[age,sbp,ldl]"],
            "200 with loadings and variance",
            f"status={r['status']}, body={r['text'][:500]}",
            f"HTTP {r['status']}",
        )


def main():
    client, sid = boot()
    api = API(client, sid)
    audit_ttest(api)
    audit_anova(api)
    audit_advanced_anova(api)
    audit_nonparametric(api)
    audit_categorical(api)
    audit_repeated(api)
    audit_reliability(api)
    audit_correlation(api)
    audit_roc(api)
    audit_noninferiority_tost(api)
    audit_bayesian(api)
    audit_gatekeeping(api)
    audit_factor(api)
    write_findings()


if __name__ == "__main__":
    main()
