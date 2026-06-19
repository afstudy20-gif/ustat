# Grok-build brief — Causal stack + DCA + Power

Read `/Users/yh/Documents/projects/wiz3/qa/briefs/common.md` first.

## Scope

| Group | Endpoints |
|-------|-----------|
| Propensity-score matching (PSM) | `POST /api/psm/*` |
| Inverse-probability-of-treatment weighting (IPTW) | `POST /api/iptw/*` (or under causal router) |
| Causal+: IV/2SLS, mediation, target-trial, DiD, RDD, DAG backdoor | `POST /api/causal/{iv_2sls,mediation,target_trial,did,rdd,dag_adjustment}` |
| Decision-curve analysis | `POST /api/decision_curve/*` |
| Power / sample size: t, ANOVA, χ², proportions, correlation, **logistic, Cox** | `POST /api/stats/power` |
| Causal sensitivity / E-value / Q-bias | `POST /api/survival_advanced/{evalue,causal_sensitivity}` |

## What to probe specifically

- PSM `event ~ diabetes` adjusting on `age, sex, bmi, ldl, sbp, nyha` — does
  the matched n and SMD plot look sensible? Are unmatched cases reported?
- IPTW with the same model — does the weighted estimate move toward null vs
  PSM? Are extreme weights flagged?
- IV/2SLS: invent a plausible instrument (e.g. `admission_date` as a
  pseudo-instrument) — does the endpoint complain about weak instruments
  (first-stage F)? Does it crash on a degenerate instrument?
- Mediation: `event ← diabetes → ldl` style — does ACME + ADE add up?
- DiD: pre/post on `bmi` by `diabetes` (split admission_date into early/late) —
  does it report the interaction and a parallel-trends note?
- RDD: pick `age` ≥ 65 as cutoff for `event` — does the local-linear LATE come
  out with non-empty CI?
- DAG backdoor: minimal endpoint smoke test (the DAG must be valid input).
- DCA: pass predicted risks for `event` from a quick logistic — does net
  benefit plot make sense at threshold 0.5 vs 0.1?
- Power for **logistic** with OR=2, p_event=0.3, solve n → compare against
  Hsieh's formula by hand.
- Power for **Cox** HR=1.5, event_rate=0.35 — verify required events ≈ (zα+zβ)²/(p(1-p)log²(HR)).
- Power solve-for-effect-size with low power (0.4): does it still return a
  number or NaN?

## Output

`/Users/yh/Documents/projects/wiz3/qa/findings/grok-build.md`
</content>
