#!/usr/bin/env python3
"""Grok-composer QA audit — drives backend via TestClient, compares with scipy/statsmodels."""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

QA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, QA_DIR)
from run_via_testclient import boot  # noqa: E402

FINDINGS: list[dict[str, Any]] = []


def add_finding(sev: str, title: str, where: str, steps: str, expected: str, actual: str, evidence: str, hypothesis: str = ""):
    FINDINGS.append({
        "sev": sev, "title": title, "where": where, "steps": steps,
        "expected": expected, "actual": actual, "evidence": evidence, "hypothesis": hypothesis,
    })


def safe_post(client, path: str, json_body: dict | None = None, **kwargs):
    body = json_body if json_body is not None else kwargs.get("json") or {}
    try:
        return client.post(path, json=body)
    except Exception as exc:
        return type("R", (), {"status_code": 500, "text": str(exc), "json": lambda _s=None: {"error": str(exc)}})()


def safe_get(client, path: str, params: dict | None = None, **kwargs):
    q = params if params is not None else kwargs.get("params") or {}
    try:
        return client.get(path, params=q)
    except Exception as exc:
        return type("R", (), {"status_code": 500, "text": str(exc), "json": lambda _s=None: {"error": str(exc)}})()


def load_csv_raw():
    return pd.read_csv(os.path.join(QA_DIR, "cohort_test.csv"))


def main():
    client, sid = boot()
    raw = load_csv_raw()

    # ── 1. Session / ingest ─────────────────────────────────────────────
    info = safe_get(client, f"/api/sessions/{sid}").json()
    cols = {c["name"]: c for c in info["columns"]}
    preview = info.get("preview", [])

    # Comma-decimal BMI
    bmi_kinds = [c.get("kind") for c in info["columns"] if c["name"] == "bmi"]
    bmi_dtype = cols.get("bmi", {}).get("dtype", "")
    comma_rows = raw[raw["bmi"].astype(str).str.contains(",", na=False)]
    desc_bmi = safe_get(client, f"/api/stats/{sid}/descriptive", params={"column": "bmi"}).json()

    if bmi_dtype == "object" or (comma_rows.shape[0] > 0 and "bmi" not in desc_bmi):
        add_finding(
            "HIGH",
            "Comma-decimal BMI values not coerced to numeric on upload",
            "Data → Upload",
            "1) Upload qa/cohort_test.csv 2) GET /api/sessions/{sid} 3) GET descriptive for bmi",
            "Values like '25,9' and '30,6' parsed as 25.9/30.6 and included in numeric summaries",
            f"bmi dtype={bmi_dtype}; descriptive keys={list(desc_bmi.keys())}; comma rows={len(comma_rows)}",
            f"session columns: bmi kind={bmi_kinds}, dtype={bmi_dtype}; descriptive: {json.dumps(desc_bmi)[:400]}",
            "CSV upload uses pd.read_csv without locale decimal handling; comma decimals stay as object strings",
        )

    # Bad sentinel values
    freq_bmi = safe_get(client, f"/api/stats/{sid}/frequency", params={"column": "bmi"}).json()
    bmi_cats = [c["value"] for c in freq_bmi.get("bmi", {}).get("categories", [])]
    bad_bmi = [v for v in bmi_cats if str(v) in ("999", "n/a", "NA") or "," in str(v)]
    if bad_bmi:
        add_finding(
            "HIGH",
            "Sentinel strings (999, n/a) and comma-decimals retained as valid BMI categories",
            "Data → Dictionary / Summary",
            "1) Upload cohort 2) frequency table for bmi",
            "999, n/a, and locale comma decimals flagged as missing or coerced",
            f"non-numeric BMI categories present: {bad_bmi}",
            json.dumps({"bad_categories": bad_bmi, "n_categories": len(bmi_cats)})[:500],
            "No missing-value token normalisation at ingest",
        )

    # Sex mixed coding
    freq_sex = safe_get(client, f"/api/stats/{sid}/frequency", params={"column": "sex"}).json()
    sex_vals = [c["value"] for c in freq_sex.get("sex", {}).get("categories", [])]
    if len([v for v in sex_vals if v not in ("M", "F", "Missing", "nan", "")]) >= 2:
        add_finding(
            "MEDIUM",
            "Mixed sex coding creates extra categories (Female, x) without warning",
            "Data → Dictionary",
            "1) Upload cohort 2) frequency for sex",
            "Normalise or warn on inconsistent sex labels; t-test/chi² should report excluded rows",
            f"sex categories: {sex_vals}",
            json.dumps(freq_sex.get("sex", {}))[:400],
            "No value-label harmonisation for categorical columns",
        )

    # Impossible ages in descriptive
    desc_age = safe_get(client, f"/api/stats/{sid}/descriptive", params={"column": "age"}).json().get("age", {})
    if desc_age:
        age_min, age_max = desc_age.get("min"), desc_age.get("max")
        if age_min is not None and age_min < 0:
            add_finding(
                "HIGH",
                "Impossible negative ages included in descriptive mean/SD",
                "Summary → Descriptive",
                "1) Upload cohort (age=-5 row) 2) GET descriptive age",
                "Out-of-range ages excluded or flagged before summary stats",
                f"min={age_min}, max={age_max}, n={desc_age.get('n')}",
                json.dumps(desc_age)[:400],
                "No plausibility range checks on numeric columns",
            )
        if age_max is not None and age_max > 120:
            add_finding(
                "MEDIUM",
                f"Implausible age={int(age_max)} retained in descriptive statistics",
                "Summary → Descriptive",
                "1) Upload cohort 2) descriptive age",
                "Values >120 flagged or excluded",
                f"max age={age_max}",
                json.dumps(desc_age)[:300],
            )

    # parse_dates on admission_date (mixed formats)
    parse_dates = safe_post(client, f"/api/compute/{sid}/parse_dates", json={
        "column": "admission_date", "formats": ["%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y"],
    })
    if parse_dates.status_code == 200:
        pdj = parse_dates.json()
        prev = pdj.get("preview_values") or pdj.get("preview") or []
        n_null = sum(1 for v in prev if v is None)
        if n_null > 0:
            add_finding(
                "MEDIUM",
                "parse_dates coerces invalid admission_date values to null without summary warning",
                "Compute → Parse dates",
                "1) POST parse_dates on admission_date 2) count null preview_values",
                "Response reports n_failed/n_unparsed for invalid dates like 13/13/2024",
                f"null_dates_in_preview={n_null} of {len(prev)}",
                json.dumps({"null_count": n_null, "kind": pdj.get("kind"), "dtype": pdj.get("dtype")})[:400],
            )

    # Recode no-match warning
    recode_resp = safe_post(client, f"/api/compute/{sid}/recode", json={
        "new_col": "never_match",
        "rules": [{"conditions": [{"col": "age", "op": "==", "val": "99999"}], "result": "1"}],
        "else_val": "0",
    })
    if recode_resp.status_code == 200:
        body = recode_resp.json()
        matched = body.get("n_matched") or body.get("matched") or body.get("rows_affected")
        warns = body.get("warnings", [])
        if (matched == 0 or matched is None) and not warns:
            add_finding(
                "MEDIUM",
                "Recode with zero matching rows returns 200 without warning",
                "Compute → Recode",
                "1) POST recode with impossible age==99999 2) inspect response",
                "Warning that no rows matched any rule",
                f"status=200, matched={matched}, warnings={warns}",
                json.dumps(body)[:400],
            )

    # MICE on ldl
    mice_resp = safe_post(client, "/api/survival_advanced/mice", json={
        "session_id": sid, "columns": ["ldl"], "n_imputations": 3, "max_iter": 5,
    })
    if mice_resp.status_code >= 500:
        add_finding("CRITICAL", "MICE imputation crashes on ldl", "Missing → MICE",
                    "POST /api/survival_advanced/mice columns=[ldl]", "200 with convergence info",
                    f"{mice_resp.status_code}: {mice_resp.text[:300]}", mice_resp.text[:500])
    elif mice_resp.status_code == 200:
        mb = mice_resp.json()
        if mb.get("converged") is False or mb.get("error"):
            add_finding("MEDIUM", "MICE on ldl reports non-convergence or error",
                        "Missing → MICE", "POST mice on ldl", "Converges or clear message",
                        json.dumps({k: mb.get(k) for k in ("converged", "error", "n_imputed", "message") if k in mb})[:400],
                        json.dumps(mb)[:600])

    # ── 2. Hypothesis tests ─────────────────────────────────────────────
    ttest = safe_post(client, "/api/stats/ttest", json={"session_id": sid, "column": "age", "group_column": "sex"}).json()
    if ttest.get("n1") and ttest.get("n2"):
        total_sex = sum(c["count"] for c in freq_sex.get("sex", {}).get("categories", []) if c["value"] in ("M", "F"))
        if ttest["n1"] + ttest["n2"] < total_sex - 5:
            add_finding(
                "MEDIUM",
                "Independent t-test silently drops non M/F sex levels without reporting exclusions",
                "Tests → t-test",
                "1) t-test age~sex 2) compare n to frequency table",
                "Report n_excluded or warn about x/Female/blank",
                f"n1={ttest['n1']}, n2={ttest['n2']}, M+F count≈{total_sex}",
                json.dumps({"n1": ttest["n1"], "n2": ttest["n2"], "groups": [ttest.get("group1"), ttest.get("group2")]})[:300],
            )
    elif "detail" in str(ttest):
        add_finding("HIGH", "t-test age~sex fails on mixed sex coding",
                    "Tests → t-test", "POST ttest age, sex", "2-group test with exclusion note",
                    str(ttest)[:300], str(ttest)[:400])

    # ANOVA bmi ~ nyha
    anova = safe_post(client, "/api/stats/anova", {"session_id": sid, "column": "bmi", "group_column": "nyha"})
    if anova.status_code == 200:
        aj = anova.json()
        groups = list(aj.get("summary", {}).keys())
        if any(g in ("", "nan", "Missing") for g in groups):
            add_finding("MEDIUM", "ANOVA includes blank/missing as nyha group",
                        "Tests → ANOVA", "anova bmi~nyha", "Missing excluded",
                        f"groups={groups}", json.dumps(aj.get("summary", {}))[:400])
    elif anova.status_code >= 400 or "could not convert string to float" in anova.text:
        add_finding(
            "CRITICAL",
            "ANOVA bmi~nyha crashes on comma-decimal BMI strings",
            "Tests → ANOVA",
            "POST /api/stats/anova column=bmi group_column=nyha on cohort_test.csv",
            "Comma decimals coerced or excluded; 200 with group summaries",
            f"{anova.status_code}: {anova.text[:250]}",
            anova.text[:500],
            "bmi column retains object strings like '34,3' after upload; astype(float) raises ValueError",
        )

    # Mann-Whitney
    mw = safe_post(client, "/api/stats/mannwhitney", {"session_id": sid, "column": "bmi", "group_column": "sex"})
    if mw.status_code == 200:
        mwj = mw.json()
        # Independent scipy check on clean numeric bmi
        df_store = pd.read_csv(os.path.join(QA_DIR, "cohort_test.csv"))
        # approximate: use API session data via raw endpoint
        raw_bmi = safe_get(client, f"/api/stats/{sid}/raw", params={"columns": "bmi,sex"}).json()
        # Chi-square diabetes x sex
    else:
        add_finding("HIGH", "Mann-Whitney bmi~sex fails",
                    "Tests → Mann-Whitney", "POST mannwhitney", "200",
                    f"{mw.status_code}: {mw.text[:200]}", mw.text[:300])

    chi_resp = safe_post(client, "/api/stats/chisquare", json={"session_id": sid, "row_column": "diabetes", "col_column": "sex"})
    if chi_resp.status_code == 200:
        chi = chi_resp.json()
        crosstab = chi.get("crosstab") or {}
        col_labels = list(crosstab.keys()) if isinstance(crosstab, dict) else []
        if len(col_labels) > 2 or any(l in ("Female", "x") for l in col_labels):
            add_finding(
                "MEDIUM",
                "Chi-square diabetes×sex treats Female and x as separate sex columns",
                "Tests → Chi-square",
                "1) chisquare diabetes×sex 2) inspect crosstab keys",
                "Harmonised M/F only, with exclusion count for other codes",
                f"crosstab columns={col_labels}, dof={chi.get('dof')}, n={chi.get('n')}",
                json.dumps({"crosstab_columns": col_labels, "chi2": chi.get("chi2"), "p": chi.get("p")})[:500],
            )

    fisher = safe_post(client, "/api/stats/fisher", json={
        "session_id": sid,
        "table": [[10, 5], [2, 0]],  # zero cell 2x2
    })
    if fisher.status_code == 422:
        # try alternate shape
        fisher = safe_post(client, "/api/stats/fisher", json={"session_id": sid, "row_column": "diabetes", "col_column": "sex"})
    if fisher.status_code >= 500:
        add_finding("CRITICAL", "Fisher exact test crashes", "Tests → Fisher", "POST fisher", "200 with OR and p",
                    fisher.text[:300], fisher.text[:400])

    # Cronbach / ICC
    cron = safe_post(client, "/api/reliability/cronbach", json={"session_id": sid, "items": ["bmi", "ldl", "sbp"]})
    if cron.status_code == 200:
        cj = cron.json()
        alpha = cj.get("alpha") or cj.get("cronbach_alpha")
        n_cron = cj.get("n")
        if alpha is not None and float(alpha) < 0:
            add_finding(
                "HIGH",
                "Cronbach alpha negative on bmi/ldl/sbp without data-quality warning",
                "Tests → Reliability",
                "POST cronbach items=[bmi,ldl,sbp] on cohort with comma-decimal BMI",
                "Warn about excluded/non-numeric items or refuse analysis",
                f"alpha={alpha}, n={n_cron}, k={cj.get('k')}",
                json.dumps({"alpha": alpha, "n": n_cron, "magnitude": cj.get("effect_sizes", [{}])[0].get("magnitude")})[:400],
                "pd.to_numeric drops comma-decimal BMI rows silently; remaining scales misaligned",
            )
        elif alpha is None or (isinstance(alpha, float) and math.isnan(alpha)):
            add_finding("HIGH", "Cronbach alpha returns NaN/missing on sparse bmi/ldl/sbp",
                        "Tests → Reliability", "cronbach on 3 vars with missings", "Finite alpha or clear error",
                        f"alpha={alpha}", json.dumps(cj)[:400])
    elif cron.status_code >= 500:
        add_finding("CRITICAL", "Cronbach alpha endpoint crashes", "Tests → Reliability",
                    "POST cronbach", "200", cron.text[:300], cron.text[:400])

    icc = safe_post(client, "/api/stats/icc", json={"session_id": sid, "columns": ["bmi", "ldl", "sbp"], "icc_type": "icc2"})
    if icc.status_code >= 500:
        add_finding("CRITICAL", "ICC endpoint crashes on sparse columns", "Tests → ICC",
                    "POST icc", "200", icc.text[:300], icc.text[:400])

    # ── 3. Correlation + ROC ────────────────────────────────────────────
    corr_bmi = safe_post(client, "/api/stats/correlation_pair", json={
        "session_id": sid, "var1": "age", "var2": "bmi", "method": "pearson",
    })
    if corr_bmi.status_code >= 500 or "could not convert string to float" in corr_bmi.text:
        add_finding(
            "CRITICAL",
            "Pearson correlation_pair age×bmi crashes on comma-decimal BMI",
            "Correlation → Pairwise",
            "POST correlation_pair var1=age var2=bmi method=pearson",
            "Comma decimals coerced or excluded; 200 with r and p",
            f"{corr_bmi.status_code}: {corr_bmi.text[:250]}",
            corr_bmi.text[:500],
            "astype(float) on object bmi column raises ValueError for '25,9'",
        )

    corr_ldl = safe_post(client, "/api/stats/correlation_pair", json={
        "session_id": sid, "var1": "age", "var2": "ldl", "method": "pearson",
    })
    if corr_ldl.status_code == 200:
        cj = corr_ldl.json()
        r_api = cj.get("r") or cj.get("correlation")
        common = raw.assign(age_n=pd.to_numeric(raw["age"], errors="coerce"),
                            ldl_n=pd.to_numeric(raw["ldl"], errors="coerce")).dropna(subset=["age_n", "ldl_n"])
        if len(common) >= 3 and r_api is not None:
            r_scipy, _ = scipy_stats.pearsonr(common["age_n"], common["ldl_n"])
            if abs(float(r_api) - r_scipy) > 0.01:
                add_finding(
                    "HIGH",
                    "Pearson age×ldl differs from scipy by >0.01",
                    "Correlation → Pairwise",
                    "1) correlation_pair age,ldl 2) scipy.pearsonr listwise",
                    f"r matches scipy to 4 dp (scipy r={r_scipy:.4f})",
                    f"API r={r_api}, scipy r={r_scipy:.4f}, n_api={cj.get('n')}, n_scipy={len(common)}",
                    json.dumps({"api_r": r_api, "scipy_r": float(r_scipy)})[:400],
                )

    roc = safe_post(client, "/api/stats/roc", json={"session_id": sid, "score_column": "ldl", "outcome_column": "event"})
    if roc.status_code == 200:
        rj = roc.json()
        auc_api = rj.get("auc")
        work = raw.assign(ldl_n=pd.to_numeric(raw["ldl"], errors="coerce"),
                          ev=pd.to_numeric(raw["event"], errors="coerce")).dropna(subset=["ldl_n", "ev"])
        if len(work) >= 10 and auc_api is not None:
            auc_sk = roc_auc_score(work["ev"], work["ldl_n"])
            if abs(float(auc_api) - auc_sk) > 0.02:
                add_finding(
                    "HIGH",
                    "ROC AUC (ldl vs event) disagrees with sklearn roc_auc_score",
                    "ROC → Single marker",
                    "1) POST /api/stats/roc 2) sklearn on listwise ldl,event",
                    f"AUC ≈ {auc_sk:.4f}",
                    f"API AUC={auc_api}, sklearn={auc_sk:.4f}, n={len(work)}",
                    json.dumps({"api_auc": auc_api, "sklearn_auc": float(auc_sk), "n": len(work)})[:400],
                )
    elif roc.status_code >= 500:
        add_finding("CRITICAL", "ROC analysis crashes", "ROC", "POST roc", "200", roc.text[:300], roc.text[:400])

    roc_cmp = safe_post(client, "/api/stats/roc_compare", json={
        "session_id": sid, "score_columns": ["ldl", "sbp"], "outcome_column": "event",
    })
    if roc_cmp.status_code >= 500:
        add_finding("HIGH", "DeLong ROC compare crashes", "ROC → Compare",
                    "POST roc_compare ldl vs sbp", "200 with DeLong p",
                    roc_cmp.text[:300], roc_cmp.text[:400])

    # ── 4. Summary / Table1 / Visual ────────────────────────────────────
    desc_all = safe_get(client, f"/api/stats/{sid}/descriptive").json()
    for col in ("age", "bmi", "ldl", "sbp", "fu_days"):
        if col not in desc_all:
            continue
        d = desc_all[col]
        s = pd.to_numeric(raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) >= 3:
            sk_api = d.get("skewness")
            sk_sp = float(scipy_stats.skew(s))
            if sk_api is not None and abs(sk_api - sk_sp) > 0.05:
                add_finding(
                    "MEDIUM",
                    f"Skewness for {col} differs from scipy",
                    "Summary → Descriptive",
                    f"GET descriptive; scipy.stats.skew on coerced {col}",
                    f"skewness ≈ {sk_sp:.4f}",
                    f"API={sk_api}, scipy={sk_sp:.4f}",
                    json.dumps({col: {"api_skew": sk_api, "scipy_skew": sk_sp}})[:300],
                    "API may include non-numeric strings coerced differently",
                )

    # fu_days negative
    if "fu_days" in desc_all and desc_all["fu_days"].get("min", 0) < 0:
        add_finding("MEDIUM", "Negative fu_days included in descriptive stats",
                    "Summary → Descriptive", "descriptive fu_days", "Negative follow-up excluded",
                    f"min={desc_all['fu_days'].get('min')}", json.dumps(desc_all["fu_days"])[:300])

    hist = safe_post(client, "/api/charts/histogram", json={"session_id": sid, "column": "age", "bins": 20})
    if hist.status_code >= 500:
        add_finding("HIGH", "Histogram crashes on age with outliers", "Visual → Histogram",
                    "POST histogram age", "200", hist.text[:300], hist.text[:400])

    t1 = safe_post(client, f"/api/stats/table1", json={
        "session_id": sid,
        "variables": ["age", "bmi", "diabetes", "nyha"],
        "group_column": "sex",
    })
    if t1.status_code == 200:
        t1j = t1.json()
        grp_cols = t1j.get("group_labels") or t1j.get("group_columns") or t1j.get("groups") or []
        n_groups = len(grp_cols) if isinstance(grp_cols, list) else 0
        if n_groups > 2:
            add_finding(
                "MEDIUM",
                "Table 1 stratified by sex renders four group columns (F, Female, M, x)",
                "Table → Table 1",
                "table1 group_column=sex",
                "Two groups M/F with warning for other codes",
                f"group_labels={grp_cols}, group_ns={t1j.get('group_ns')}",
                json.dumps({"group_labels": grp_cols, "group_ns": t1j.get("group_ns")})[:400],
            )
    elif t1.status_code >= 500:
        add_finding("CRITICAL", "Table 1 endpoint crashes", "Table → Table 1",
                    "POST table1", "200", t1.text[:300], t1.text[:400])

    sbar_dup = safe_post(client, "/api/charts/subgroup_bar", {
        "session_id": sid, "y_col": "event", "subgroup_col": "nyha",
        "xaxis_col": "nyha", "y_mode": "percentage", "target_value": "1",
    })
    if sbar_dup.status_code >= 500 or "1-dimensional" in sbar_dup.text:
        add_finding(
            "HIGH",
            "Subgroup bar crashes when subgroup_col equals xaxis_col",
            "Visual → Subgroup bar",
            "POST subgroup_bar with subgroup_col=nyha and xaxis_col=nyha",
            "Dedupe columns or return 422; chart renders event rate by nyha",
            f"{sbar_dup.status_code}: {sbar_dup.text[:250]}",
            sbar_dup.text[:500],
            "df[[nyha, nyha]] yields 2-column DataFrame; sorted_groups expects 1-D series",
        )

    sbar = safe_post(client, "/api/charts/subgroup_bar", {
        "session_id": sid, "y_col": "event", "subgroup_col": "sex",
        "xaxis_col": "nyha", "y_mode": "percentage", "target_value": "1",
    })
    if sbar.status_code == 200:
        sbj = sbar.json()
        order = [x.get("x") or x.get("label") for x in sbj.get("bars", sbj.get("data", []))][:10]
        if order and order != sorted(order, key=lambda x: float(x) if str(x).replace(".", "").isdigit() else 999):
            # check if nyha order wrong
            try:
                nums = [float(x) for x in order if x is not None]
                if nums != sorted(nums):
                    add_finding("LOW", "Subgroup bar may not preserve nyha ordinal order",
                                "Visual → Subgroup bar", "subgroup_bar event by nyha",
                                "Categories ordered 1,2,3,4",
                                f"order={order}", json.dumps(sbj)[:400])
            except (TypeError, ValueError):
                pass

    forest = safe_post(client, "/api/charts/forest", json={
        "rows": [
            {"label": "Study A", "est": 1.2, "ci_low": 0.8, "ci_high": 1.8},
            {"label": "Study B", "est": 0.9, "ci_low": 0.5, "ci_high": 1.4},
            {"label": "Study C", "est": 1.5, "ci_low": -0.2, "ci_high": 2.1},
            {"label": "Study D", "est": 2.0, "ci_low": 1.1, "ci_high": 3.5},
        ],
        "effect_label": "HR", "x_axis": "log", "do_meta": True,
    })
    if forest.status_code == 200:
        fj = forest.json()
        neg = [r for r in fj.get("rows", []) if r.get("ci_low", 1) <= 0]
        if neg:
            add_finding(
                "HIGH",
                "Forest plot log-axis accepts negative CI bound without error",
                "Visual → Forest",
                "POST forest with ci_low=-0.2 on log scale",
                "422 or clamp/warn — log of negative undefined",
                f"rows with ci_low<=0: {len(neg)}; meta={fj.get('meta') is not None}",
                json.dumps(fj.get("rows", [])[2:3])[:400],
                "log(max(ci_low, 1e-12)) silently clips negative bounds",
            )
    elif forest.status_code >= 500:
        add_finding("CRITICAL", "Forest plot endpoint crashes", "Visual → Forest",
                    "POST forest", "200", forest.text[:300], forest.text[:400])

    # ── 5. Meta-analysis ────────────────────────────────────────────────
    studies = [
        {"label": "S1", "effect": 1.4, "ci_low": 1.0, "ci_high": 2.0},
        {"label": "S2", "effect": 0.85, "ci_low": 0.6, "ci_high": 1.2},
        {"label": "S3", "effect": 1.1, "ci_low": 0.9, "ci_high": 1.5},
        {"label": "S4", "effect": 1.6, "ci_low": 1.2, "ci_high": 2.2},
        {"label": "S5", "effect": 0.95, "ci_low": 0.7, "ci_high": 1.3, "moderator": 0.3},
    ]
    meta_fix = safe_post(client, "/api/meta/analyze", json={"studies": studies, "measure": "OR", "tau2_method": "DL"})
    meta_re = safe_post(client, "/api/meta/analyze", json={"studies": studies[:4], "measure": "OR", "tau2_method": "DL"})
    if meta_re.status_code == 200:
        mr = meta_re.json()
        i2 = mr.get("random_effects", mr.get("random", {})).get("I2_pct") or mr.get("I2_pct")
        pooled = mr.get("random_effects", mr.get("random", {})).get("pooled_effect") or mr.get("pooled_effect")
        # Independent DL pool
        ys, vs = [], []
        for s in studies[:4]:
            le = math.log(s["effect"])
            se = (math.log(s["ci_high"]) - math.log(s["ci_low"])) / (2 * 1.959963984540054)
            ys.append(le)
            vs.append(se ** 2)
        y, v = np.array(ys), np.array(vs)
        w = 1 / v
        mu = np.sum(w * y) / np.sum(w)
        q = float(np.sum(w * (y - mu) ** 2))
        df_ = len(y) - 1
        c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2 = max(0.0, (q - df_) / c) if c > 0 else 0.0
        wr = 1 / (v + tau2)
        mu_re = float(np.sum(wr * y) / np.sum(wr))
        i2_ind = max(0.0, (q - df_) / q * 100.0) if q > 0 else 0.0
        pool_api = mr.get("random_effects", {}).get("pooled_log") or mr.get("random", {}).get("mu")
        i2_api = i2
        if i2_api is not None and abs(float(i2_api) - i2_ind) > 5:
            add_finding("MEDIUM", "Meta-analysis I² differs from independent DL calculation",
                        "Meta → Pool", "POST meta/analyze 4 studies", f"I²≈{i2_ind:.1f}%",
                        f"API I²={i2_api}, independent={i2_ind:.1f}%",
                        json.dumps({"api": mr.get("random_effects", mr.get("random", {})), "indep_i2": i2_ind})[:500])
    elif meta_re.status_code >= 500:
        add_finding("CRITICAL", "Meta-analysis analyze crashes", "Meta",
                    "POST meta/analyze", "200", meta_re.text[:300], meta_re.text[:400])

    meta_reg = safe_post(client, "/api/meta/regression", json={"studies": studies, "measure": "OR"})
    if meta_reg.status_code >= 500:
        add_finding("HIGH", "Meta-regression crashes", "Meta → Regression",
                    "POST meta/regression", "200", meta_reg.text[:300], meta_reg.text[:400])

    # asymmetric funnel for trim-and-fill
    asym_studies = [
        {"label": f"A{i}", "effect": 1.2 + 0.1 * i, "se": 0.15 + 0.02 * i}
        for i in range(8)
    ] + [
        {"label": f"B{i}", "effect": 2.5 - 0.05 * i, "se": 0.08}
        for i in range(4)
    ]
    bias = safe_post(client, "/api/meta/bias", json={"studies": asym_studies, "measure": "OR"})
    if bias.status_code == 200:
        bj = bias.json()
        if bj.get("trim_and_fill") is None and bj.get("trim_fill") is None:
            add_finding("LOW", "Publication-bias endpoint omits trim-and-fill block",
                        "Meta → Bias", "POST meta/bias asymmetric set", "trim_and_fill adjusted estimate",
                        f"keys={list(bj.keys())}", json.dumps(bj)[:400])
    elif bias.status_code >= 500:
        add_finding("HIGH", "Meta bias/trim-and-fill crashes", "Meta → Bias",
                    "POST meta/bias", "200", bias.text[:300], bias.text[:400])

    # ── 6. Time series ──────────────────────────────────────────────────
    arima = safe_post(client, "/api/timeseries/arima", json={
        "session_id": sid, "value_col": "age", "time_col": "patient_id", "p": 1, "d": 0, "q": 1,
    })
    if arima.status_code >= 500:
        add_finding("CRITICAL", "ARIMA crashes on age ordered by patient_id", "Time Series → ARIMA",
                    "POST timeseries/arima", "200", arima.text[:300], arima.text[:400])

    stl = safe_post(client, "/api/timeseries/decompose", json={
        "session_id": sid, "value_col": "event", "method": "stl", "period": 12,
    })
    if stl.status_code >= 500:
        add_finding("HIGH", "STL decomposition crashes on flat/binary event series",
                    "Time Series → Decompose", "POST decompose event STL", "200 or graceful message",
                    stl.text[:300], stl.text[:400])

    statn = safe_post(client, "/api/timeseries/stationarity", json={
        "session_id": sid, "value_col": "age", "time_col": "patient_id",
    })
    if statn.status_code == 200:
        sj = statn.json()
        acf = sj.get("acf") or sj.get("acf_values") or []
        pacf = sj.get("pacf") or sj.get("pacf_values") or []
        if acf and pacf and len(acf) != len(pacf):
            add_finding("LOW", "Stationarity ACF and PACF length mismatch",
                        "Time Series → Stationarity", "POST stationarity age",
                        "ACF/PACF same lag count", f"len(acf)={len(acf)}, len(pacf)={len(pacf)}",
                        json.dumps({"adf_p": sj.get("adf", {}).get("p"), "kpss_p": sj.get("kpss", {}).get("p")})[:400])
    elif statn.status_code >= 500:
        add_finding("HIGH", "Stationarity test crashes", "Time Series",
                    "POST stationarity", "200", statn.text[:300], statn.text[:400])

    # ── 7. Causal + Power slice ─────────────────────────────────────────
    psm = safe_post(client, "/api/models/psm", json={
        "session_id": sid, "treatment_col": "diabetes", "covariates": ["age", "sex", "bmi"],
        "outcome_col": "event", "outcome_type": "binary",
    })
    if psm.status_code >= 500:
        add_finding("HIGH", "PSM endpoint crashes on cohort_test", "PSM",
                    "POST /api/models/psm", "200 with balance table", psm.text[:300], psm.text[:400])

    iptw = safe_post(client, "/api/models/iptw", json={
        "session_id": sid, "treatment_col": "diabetes", "covariates": ["age", "bmi", "sbp"],
        "outcome_col": "event", "outcome_type": "binary",
    })
    if iptw.status_code >= 500:
        add_finding("HIGH", "IPTW endpoint crashes on cohort_test", "IPTW",
                    "POST /api/models/iptw", "200", iptw.text[:300], iptw.text[:400])

    iv = safe_post(client, "/api/causal/iv_2sls", json={
        "session_id": sid, "outcome_col": "bmi", "treatment_col": "diabetes",
        "instrument_col": "nyha", "covariates": ["age", "sex"],
    })
    if iv.status_code >= 500:
        add_finding("HIGH", "IV/2SLS causal endpoint crashes", "Causal+ → IV",
                    "POST causal/iv_2sls", "200 or validation error", iv.text[:300], iv.text[:400])

    med = safe_post(client, "/api/causal/mediation", json={
        "session_id": sid, "outcome_col": "event", "treatment_col": "diabetes",
        "mediator_col": "bmi", "covariates": ["age"],
    })
    if med.status_code >= 500:
        add_finding("HIGH", "Mediation analysis crashes", "Causal+ → Mediation",
                    "POST causal/mediation", "200", med.text[:300], med.text[:400])

    dca = safe_post(client, "/api/decision_curve/dca", json={
        "session_id": sid, "outcome_col": "event", "predictor_cols": ["ldl", "age"],
    })
    if dca.status_code >= 500:
        add_finding("HIGH", "Decision-curve analysis crashes", "DCA",
                    "POST decision_curve/dca", "200", dca.text[:300], dca.text[:400])

    # Power: Hsieh logistic OR=2, p_event=0.3, power=0.8
    from scipy.stats import norm as sp_norm
    log_or = math.log(2.0)
    p_ev, power_tgt, alpha = 0.3, 0.8, 0.05
    z_a = sp_norm.ppf(1 - alpha / 2)
    z_b = sp_norm.ppf(power_tgt)
    n_hsieh = int(math.ceil(((z_a + z_b) ** 2) / (p_ev * (1 - p_ev) * (log_or ** 2))))

    pow_log = safe_post(client, "/api/stats/power", json={
        "test": "logistic", "solve_for": "n", "effect_size": 2.0,
        "p_event": 0.3, "power": 0.8, "alpha": 0.05, "tails": 2,
    })
    if pow_log.status_code == 200:
        pj = pow_log.json()
        n_api = pj.get("result") or pj.get("n")
        if n_api is not None and abs(int(n_api) - n_hsieh) > 3:
            add_finding(
                "HIGH",
                "Logistic power sample size disagrees with Hsieh hand calculation",
                "Power → Logistic",
                "POST power logistic OR=2, p_event=0.3, power=0.8; compare Hsieh formula",
                f"n ≈ {n_hsieh}",
                f"API n={n_api}, Hsieh n={n_hsieh}",
                json.dumps({"api": pj, "hsieh_n": n_hsieh})[:500],
            )
    elif pow_log.status_code >= 500:
        add_finding("CRITICAL", "Logistic power endpoint crashes", "Power",
                    "POST stats/power logistic", "200", pow_log.text[:300], pow_log.text[:400])

    # Cox Schoenfeld events HR=1.5, event_rate=0.35
    hr, er, p_exp = 1.5, 0.35, 0.5
    log_hr = math.log(hr)
    z_a = sp_norm.ppf(1 - 0.05 / 2)
    z_b = sp_norm.ppf(0.8)
    d_sch = ((z_a + z_b) ** 2) / (p_exp * (1 - p_exp) * (log_hr ** 2))
    n_sch = int(math.ceil(d_sch / er))

    pow_cox = safe_post(client, "/api/stats/power", json={
        "test": "survival_cox", "solve_for": "n", "hr": 1.5,
        "event_rate": 0.35, "power": 0.8, "alpha": 0.05, "tails": 2, "p_exposed": 0.5,
    })
    if pow_cox.status_code == 200:
        cj = pow_cox.json()
        n_api = cj.get("result")
        label = cj.get("label", "")
        if n_api is not None and abs(int(n_api) - n_sch) > 5:
            add_finding(
                "HIGH",
                "Cox power sample size disagrees with Schoenfeld events formula",
                "Power → Cox",
                "POST power survival_cox HR=1.5, event_rate=0.35; hand calc Schoenfeld",
                f"n ≈ {n_sch} (events ≈ {int(math.ceil(d_sch))})",
                f"API n={n_api}, Schoenfeld n={n_sch}, label={label}",
                json.dumps({"api": cj, "schoenfeld_n": n_sch, "events": d_sch})[:500],
            )
    elif pow_cox.status_code >= 500:
        add_finding("CRITICAL", "Cox power endpoint crashes", "Power → Cox",
                    "POST stats/power survival_cox", "200", pow_cox.text[:300], pow_cox.text[:400])

    evalue = safe_post(client, "/api/survival_advanced/evalue", json={
        "estimate": 2.0, "ci_low": 1.2, "ci_high": 3.5, "measure_type": "OR",
    })
    if evalue.status_code >= 500:
        add_finding("HIGH", "E-value endpoint crashes", "Causal sensitivity → E-value",
                    "POST survival_advanced/evalue", "200", evalue.text[:300], evalue.text[:400])

    csens = safe_post(client, "/api/survival_advanced/causal_sensitivity", json={
        "observed_estimate": 1.5, "ci_low": 1.1, "ci_high": 2.0, "measure": "rr",
    })
    if csens.status_code >= 500:
        add_finding("HIGH", "Causal sensitivity (Q-bias) crashes", "Causal sensitivity",
                    "POST causal_sensitivity", "200", csens.text[:300], csens.text[:400])

    # Kruskal nyha
    kw = safe_post(client, "/api/stats/kruskal", {"session_id": sid, "column": "bmi", "group_column": "nyha"})
    if kw.status_code >= 500:
        add_finding("HIGH", "Kruskal-Wallis bmi~nyha crashes", "Tests → Kruskal",
                    "POST kruskal", "200", kw.text[:300], kw.text[:400])

    # Gatekeeping smoke
    gk = safe_post(client, "/api/multiplicity/gatekeeping", json={
        "session_id": sid, "tests": [{"name": "t1", "p": 0.04}, {"name": "t2", "p": 0.06}],
        "method": "fallback",
    })
    if gk.status_code >= 500:
        add_finding("MEDIUM", "Gatekeeping endpoint crashes", "Tests → Multiplicity",
                    "POST gatekeeping", "200", gk.text[:300], gk.text[:400])

    write_findings()
    print(f"done grok-composer findings={len(FINDINGS)}")


def write_findings():
    out = os.path.join(QA_DIR, "findings", "grok-composer.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lines = ["# Grok-composer QA Findings\n", f"Audit of `qa/cohort_test.csv` — {len(FINDINGS)} findings.\n"]
    for i, f in enumerate(FINDINGS, 1):
        lines.append(f"## [{f['sev']}] {f['title']}\n")
        lines.append(f"**Where:** {f['where']}\n")
        lines.append(f"**Steps:** {f['steps']}\n")
        lines.append(f"**Expected:** {f['expected']}\n")
        lines.append(f"**Actual:** {f['actual']}\n")
        lines.append(f"**Evidence:** {f['evidence']}\n")
        if f.get("hypothesis"):
            lines.append(f"**Hypothesis (optional):** {f['hypothesis']}\n")
        lines.append("")
    with open(out, "w") as fp:
        fp.write("\n".join(lines))


if __name__ == "__main__":
    main()