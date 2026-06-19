# Cross-check brief — Codex verifies ZCode + Grok-build CRITICAL findings

You are doing a wave-2 cross-check. Wave-1 produced findings in qa/findings/{zcode,kimi,codex,grok-build}.md.

Your job: independently reproduce the CRITICAL findings from **zcode.md** and **grok-build.md** on the same dataset (qa/cohort_test.csv via qa/run_via_testclient.py boot()). For each:
1. Re-run the exact endpoint with the same input.
2. Confirm: was the bug real? quote the actual response.
3. Or: was the finding wrong / overstated? say why.

Output to qa/crosscheck/codex_xcheck.md. Use this schema:

```
## [CONFIRM|REFUTE|PARTIAL] <agent>:<finding title>
**Verified by:** <command / endpoint hit>
**Actual response:** <key fields>
**Verdict reason:** <one sentence>
```

ZCode CRITICAL findings to verify (3):
- bmi comma-decimals silently dropped by Transform/clinical/mean
- Formula builder concatenates instead of computing on text-typed bmi (bmi*2 → "30.630.6")
- Dictionary "kind → numeric" override does not coerce the column

Grok-build CRITICAL findings to verify (2):
- IPTW omits treatment from weighted outcome GLM; no ATE returned
- IPTW reports smd_before == smd_after (no rebalancing)

Print 'done codex-xcheck verified=<C count>/<R count>/<P count>' at end (C=confirm R=refute P=partial). No code edits.
