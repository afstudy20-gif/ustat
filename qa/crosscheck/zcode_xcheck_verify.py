"""Wave-2 cross-check: ZCode independently re-runs the 9 CRITICAL findings
from kimi.md (5) and codex.md (4) on qa/cohort_test.csv via TestClient boot().

No code edits; only re-issues the exact endpoint calls and records responses.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_via_testclient import boot  # noqa: E402


def jdump(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def safe_call(client, path: str, payload: dict):
    try:
        r = client.post(path, json=payload)
        try:
            body = r.json()
        except Exception:
            body = r.text[:1500]
        return {"status": r.status_code, "body": body}
    except Exception as exc:
        return {"status": "EXC", "exception": repr(exc)}


def short(obj, max_chars=700):
    s = jdump(obj) if not isinstance(obj, str) else obj
    if len(s) > max_chars:
        s = s[:max_chars] + " …<truncated>"
    return s


def main():
    client, sid = boot()
    print("sid", sid)

    # ------------------------------------------------------------------
    # KIMI CRITICAL #1 — ANCOVA age ~ sex + ldl
    # ------------------------------------------------------------------
    print("\n=== K1 ANCOVA age ~ sex + ldl ===")
    r = safe_call(client, "/api/advanced_anova/ancova",
                  {"session_id": sid, "outcome": "age", "group_col": "sex", "covariates": ["ldl"]})
    print("status=", r["status"])
    print(short(r.get("body") or r.get("exception")))

    # ------------------------------------------------------------------
    # KIMI CRITICAL #2 — Two-way ANOVA age ~ sex + diabetes
    # ------------------------------------------------------------------
    print("\n=== K2 two_way_anova age ~ sex + diabetes ===")
    r = safe_call(client, "/api/advanced_anova/two_way_anova",
                  {"session_id": sid, "outcome": "age", "factor1": "sex", "factor2": "diabetes"})
    print("status=", r["status"])
    print(short(r.get("body") or r.get("exception")))

    # ------------------------------------------------------------------
    # KIMI CRITICAL #3 — Mann-Whitney U bmi ~ diabetes (comma-decimal BMI)
    # ------------------------------------------------------------------
    print("\n=== K3 mannwhitney bmi ~ diabetes ===")
    r = safe_call(client, "/api/stats/mannwhitney",
                  {"session_id": sid, "column": "bmi", "group_column": "diabetes"})
    print("status=", r["status"])
    print(short(r.get("body") or r.get("exception")))

    # ------------------------------------------------------------------
    # KIMI CRITICAL #4 — Kruskal-Wallis bmi ~ nyha
    # ------------------------------------------------------------------
    print("\n=== K4 kruskal bmi ~ nyha ===")
    r = safe_call(client, "/api/stats/kruskal",
                  {"session_id": sid, "column": "bmi", "group_column": "nyha"})
    print("status=", r["status"])
    print(short(r.get("body") or r.get("exception")))

    # ------------------------------------------------------------------
    # KIMI CRITICAL #5 — Mantel-Haenszel event × diabetes stratified by sex
    # ------------------------------------------------------------------
    print("\n=== K5 mantel_haenszel event*diabetes | sex ===")
    r = safe_call(client, "/api/categorical/mantel_haenszel",
                  {"session_id": sid, "row_col": "event", "col_col": "diabetes", "strata_col": "sex"})
    print("status=", r["status"])
    print(short(r.get("body") or r.get("exception")))

    # ------------------------------------------------------------------
    # CODEX CRITICAL #1 — Linear regression 500s on comma-decimal BMI
    # ------------------------------------------------------------------
    print("\n=== C1 linear bmi ~ age + sex + ldl ===")
    r = safe_call(client, "/api/models/linear",
                  {"session_id": sid, "outcome": "bmi", "predictors": ["age", "sex", "ldl"]})
    print("status=", r["status"])
    print(short(r.get("body") or r.get("exception")))

    # ------------------------------------------------------------------
    # CODEX CRITICAL #2 — Logistic models create bogus predictors for
    # dirty sex levels (sex_Female, sex_x based on single rows).
    # ------------------------------------------------------------------
    print("\n=== C2a logistic event ~ age + sex + ldl + nyha ===")
    r = safe_call(client, "/api/models/logistic",
                  {"session_id": sid, "outcome": "event", "predictors": ["age", "sex", "ldl", "nyha"]})
    print("status=", r["status"])
    body = r.get("body")
    if isinstance(body, dict):
        coefs = body.get("coefficients", [])
        bogus = [c for c in coefs if isinstance(c, dict) and
                 str(c.get("term", c.get("variable", ""))).lower() in ("sex_female", "sex_x")]
        print("coeff_terms=", [c.get("term", c.get("variable")) for c in coefs if isinstance(c, dict)])
        for c in bogus:
            print("BOGUS_ROW:", jdump({k: c.get(k) for k in ("term", "variable", "B", "or", "odds_ratio", "or_ci_low", "or_ci_high", "ci_low", "ci_high", "p")}))
    else:
        print(short(body or r.get("exception"), 1400))

    print("\n=== C2b logistic_table event ~ age + sex + ldl + nyha ===")
    r2 = safe_call(client, "/api/models/logistic_table",
                   {"session_id": sid, "outcome": "event", "predictors": ["age", "sex", "ldl", "nyha"]})
    print("status=", r2["status"])
    body2 = r2.get("body")
    if isinstance(body2, dict):
        rows = body2.get("table", [])
        bogus_rows = [row for row in rows if isinstance(row, dict) and
                      "sex_Female" in str(row.get("variable", "")) or "sex_x" in str(row.get("variable", ""))]
        print("table_vars=", [row.get("variable") for row in rows if isinstance(row, dict)])
        for row in bogus_rows:
            print("BOGUS_OR_ROW:", jdump(row))
    else:
        print(short(body2 or r2.get("exception"), 1400))

    # ------------------------------------------------------------------
    # CODEX CRITICAL #3 — Fine-Gray accepts and plots negative fu_days
    # ------------------------------------------------------------------
    print("\n=== C3 fine_gray fu_days event=1 group=nyha ===")
    r = safe_call(client, "/api/survival_advanced/fine_gray",
                  {"session_id": sid, "duration_col": "fu_days", "event_col": "event",
                   "event_of_interest": 1, "group_col": "nyha", "predictors": ["age", "ldl"]})
    print("status=", r["status"])
    body = r.get("body")
    # Pull out n, CIF curve x range, and regression n if present.
    if isinstance(body, dict):
        probe = {
            "n": body.get("n"),
            "n_excluded": body.get("n_excluded"),
            "regression_n": (body.get("regression_result") or {}).get("n") if isinstance(body.get("regression_result"), dict) else None,
            "top_keys": list(body.keys()),
        }
        # cif_data: list of {group, x, cif} points
        cif = body.get("cif_data")
        xs = []
        if isinstance(cif, list):
            for pt in cif:
                if isinstance(pt, dict):
                    xv = pt.get("x") or pt.get("time") or pt.get("t")
                    if isinstance(xv, (int, float)):
                        xs.append(float(xv))
                    elif isinstance(xv, list):
                        xs += [v for v in xv if isinstance(v, (int, float))]
        probe["cif_min_x"] = min(xs) if xs else None
        body_str = json.dumps(body, default=str)
        probe["contains_negative_ten"] = ("-10" in body_str)
        # Locate the negative time point explicitly in the plot/cif_data
        probe["plot_trace_xs"] = None
        plot = body.get("plot")
        if isinstance(plot, dict):
            data_series = plot.get("data", [])
            min_x_overall = None
            for s in data_series:
                if isinstance(s, dict):
                    xs = s.get("x") or []
                    if isinstance(xs, list):
                        for x in xs:
                            if isinstance(x, (int, float)) and (min_x_overall is None or x < min_x_overall):
                                min_x_overall = x
            probe["plot_trace_xs"] = min_x_overall
        print(jdump(probe))
    else:
        print(short(body or r.get("exception"), 1400))

    # ------------------------------------------------------------------
    # CODEX CRITICAL #4 — Survival external validation uses binary
    # calibration and omits O/E.
    # ------------------------------------------------------------------
    print("\n=== C4 setup: derive cox_lp_same_cohort on the session ===")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
    import numpy as np
    import pandas as pd
    from services import store
    from lifelines import CoxPHFitter

    df = store.get(sid).copy()
    for c in ["age", "bmi", "ldl", "nyha", "fu_days", "event"]:
        df[f"{c}_num"] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    surv = df[["fu_days_num", "event_num", "age_num", "ldl_num", "nyha_num"]].dropna()
    surv = surv[surv["fu_days_num"] > 0].copy()
    cph = CoxPHFitter()
    cph.fit(surv, duration_col="fu_days_num", event_col="event_num")
    lp = pd.Series(np.nan, index=df.index, dtype=float)
    lp.loc[surv.index] = cph.predict_log_partial_hazard(surv).astype(float).values
    df["cox_lp_same_cohort"] = lp
    store.save(sid, df, track_undo=False)
    dev_c = float(cph.concordance_index_)
    print("dev_c_index=", dev_c, "dev_n=", int(len(surv)))

    print("\n=== C4 survival external_validation cox_lp_same_cohort ===")
    r = safe_call(client, "/api/survival_advanced/external_validation",
                  {"session_id": sid, "duration_col": "fu_days_num", "event_col": "event_num",
                   "predicted_lp_col": "cox_lp_same_cohort",
                   "dev_metrics": {"c_index": dev_c, "calibration_slope": 1.0}})
    print("status=", r["status"])
    body = r.get("body")
    if isinstance(body, dict):
        probe = {
            "validation_c_index": body.get("validation_c_index"),
            "validation_calibration_slope": body.get("validation_calibration_slope"),
            "validation_calibration_intercept": body.get("validation_calibration_intercept"),
            "oe_ratio_present": any(k in body for k in ("oe_ratio", "observed_expected_ratio", "o_e_ratio", "oe")),
            "oe_in_nested": False,
            "top_keys": list(body.keys()),
        }
        nested = body.get("performance_vs_dev") or body.get("performance") or {}
        if isinstance(nested, dict):
            probe["nested_keys"] = list(nested.keys())
            probe["oe_in_nested"] = any(k in nested for k in ("oe_ratio", "observed_expected_ratio", "o_e_ratio", "oe"))
        print(jdump(probe))
    else:
        print(short(body or r.get("exception"), 1400))

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
