import { useState } from "react";
import { useStore } from "../store";
import { runTTest, runChiSquare, runAnova, runMannWhitney, runFisher, runKruskal, runAncova, runTwoWayAnova } from "../api";
import ResultExporter from "./ResultExporter";

const TESTS = [
  { id: "ttest_1sample",  label: "One-sample t-test",     group: "Parametric" },
  { id: "ttest_2sample",  label: "Independent t-test",    group: "Parametric" },
  { id: "anova",          label: "One-way ANOVA",         group: "Parametric" },
  { id: "mannwhitney",    label: "Mann-Whitney U",        group: "Non-parametric" },
  { id: "kruskal",        label: "Kruskal-Wallis",        group: "Non-parametric" },
  { id: "ancova",          label: "ANCOVA",                group: "Parametric" },
  { id: "two_way",        label: "Two-way ANOVA",         group: "Parametric" },
  { id: "chisquare",      label: "Chi-square",            group: "Categorical" },
  { id: "fisher",         label: "Fisher's exact",        group: "Categorical" },
];

const TEST_GUIDANCE: Record<string, { when: string; assumptions: string; reading: string }> = {
  ttest_1sample: {
    when: "Compare one group's mean to a known reference value (e.g. population norm, clinical threshold).",
    assumptions: "Outcome is continuous and approximately normally distributed (check Q-Q plot in Summary tab). Robust to mild non-normality if n > 30 (CLT).",
    reading: "If p < 0.05, the sample mean is significantly different from the test value. Report: t(df) = X.XX, p = Y.YYY, mean difference = Z.ZZ.",
  },
  ttest_2sample: {
    when: "Compare means between two independent groups (e.g. treatment vs. control, male vs. female).",
    assumptions: "Both groups approximately normal (or n > 30 each). Levene's test checks equal variances — if violated, Welch's t-test is used automatically.",
    reading: "p < 0.05 means the groups differ significantly. Report: t(df) = X.XX, p = Y.YYY. Add Cohen's d for effect size (small 0.2, medium 0.5, large 0.8).",
  },
  anova: {
    when: "Compare means across 3+ groups simultaneously (e.g. drug A vs. B vs. C). Avoids multiple-comparison inflation from running many t-tests.",
    assumptions: "Each group approximately normal, roughly equal variances (Levene's test). Robust if groups are balanced and n > 20 per group.",
    reading: "A significant F-test (p < 0.05) means at least one group differs. Use post-hoc tests (Tukey, Bonferroni) to identify which pairs differ.",
  },
  mannwhitney: {
    when: "Non-parametric alternative to the independent t-test. Use when data are ordinal, heavily skewed, or n < 20 per group.",
    assumptions: "Both samples are independent. Tests whether one distribution is stochastically greater than the other (rank-based).",
    reading: "p < 0.05 means the groups' rank distributions differ significantly. Report U statistic and p-value. Effect size: r = Z / sqrt(N).",
  },
  kruskal: {
    when: "Non-parametric alternative to one-way ANOVA. Use when comparing 3+ groups with non-normal or ordinal data.",
    assumptions: "Groups are independent. Tests whether at least one group's distribution differs from the others (rank-based).",
    reading: "p < 0.05 means at least one group differs in rank distribution. Follow up with pairwise Mann-Whitney tests (with Bonferroni correction).",
  },
  chisquare: {
    when: "Test association between two categorical variables (e.g. treatment group vs. outcome category). Use when all expected cell counts are >= 5.",
    assumptions: "Each observation is independent. Expected frequency in each cell >= 5. If any cell < 5, use Fisher's exact test instead.",
    reading: "p < 0.05 means the variables are significantly associated. Report: \u03C7\u00B2(df) = X.XX, p = Y.YYY. Add Cramer's V for effect size.",
  },
  fisher: {
    when: "Exact test for 2\u00D72 tables, especially when sample is small or any expected cell count < 5. Preferred over chi-square for small samples.",
    assumptions: "Fixed marginals. No minimum sample size requirement — valid even for very small tables.",
    reading: "p < 0.05 means the row and column variables are significantly associated. Also report the odds ratio and its 95% CI.",
  },
  ancova: {
    when: "Compare group means on an outcome after controlling for one or more continuous covariates. Essential when groups differ on a baseline variable (e.g. age, BMI).",
    assumptions: "Normality of residuals. Homogeneity of regression slopes (covariate effect is the same in all groups). Linear relationship between covariate and outcome.",
    reading: "The F-test for the group factor shows whether groups differ AFTER adjustment. EMMs (estimated marginal means) show the adjusted group means. Report: F, p, partial \u03B7\u00B2, and EMMs.",
  },
  two_way: {
    when: "Examine the effects of two categorical factors (and their interaction) on a continuous outcome. E.g. drug type \u00D7 dose level on blood pressure.",
    assumptions: "Normality of residuals. Homogeneity of variances across all factor-level combinations.",
    reading: "Check the interaction first. If significant, main effects are qualified by the interaction. Report F, p, and partial \u03B7\u00B2 for each term. Use EMMs to understand cell means.",
  },
};

function ResultCard({ result }: { result: any }) {
  const fmt = (v: any) => {
    if (typeof v !== "number") return String(v);
    if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
    return v.toFixed(4);
  };

  const skip = ["test", "interpretation", "significant", "crosstab", "groups",
                "table", "row_labels", "col_labels", "curve", "effect_sizes",
                "assumptions", "warnings", "summary", "posthoc", "posthoc_method",
                "result_text", "export_rows"];

  const statEntries = Object.entries(result).filter(([k]) => !skip.includes(k) && typeof result[k] !== "object");
  const exportHeaders = ["Statistic", "Value"];
  const exportRows = statEntries.map(([k, v]) => [k, fmt(v)]);

  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-900">{result.test}</h4>
        <div className="flex items-center gap-2">
          <ResultExporter title={result.test ?? "hypothesis_test"} headers={exportHeaders} rows={exportRows} />
          {"significant" in result && (
            <span className={result.significant ? "badge-sig" : "badge-ns"}>
              {result.significant ? "Significant" : "Not significant"}
            </span>
          )}
        </div>
      </div>
      <p className="text-sm text-gray-500 italic">{result.interpretation}</p>

      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
        {Object.entries(result)
          .filter(([k]) => !skip.includes(k))
          .map(([k, v]) => (
            <div key={k} className="flex justify-between border-b border-gray-100 py-1">
              <span className="text-gray-400">{k}</span>
              <span className="text-gray-700 font-mono">{fmt(v)}</span>
            </div>
          ))}
      </div>

      {/* Effect Sizes */}
      {result.effect_sizes?.length > 0 && (
        <div className="mt-2 space-y-1">
          <p className="text-xs font-semibold text-gray-600">Effect Sizes</p>
          {result.effect_sizes.map((es: any, i: number) => (
            <div key={i} className="flex items-center gap-3 bg-indigo-50 rounded-lg px-3 py-1.5 text-xs">
              <span className="font-semibold text-indigo-800">{es.name?.replace(/_/g, " ")}</span>
              <span className="font-mono text-indigo-700">{es.value?.toFixed(3)}</span>
              {es.ci_low != null && es.ci_high != null && (
                <span className="text-indigo-500">95% CI: [{es.ci_low?.toFixed(3)}, {es.ci_high?.toFixed(3)}]</span>
              )}
              {es.magnitude && (
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                  es.magnitude === "large" ? "bg-red-100 text-red-700" :
                  es.magnitude === "medium" ? "bg-amber-100 text-amber-700" :
                  es.magnitude === "small" ? "bg-blue-100 text-blue-700" :
                  "bg-gray-100 text-gray-500"}`}>
                  {es.magnitude}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Assumptions */}
      {result.assumptions?.length > 0 && (
        <div className="mt-2 space-y-1">
          <p className="text-xs font-semibold text-gray-600">Assumption Checks</p>
          {result.assumptions.map((a: any, i: number) => (
            <div key={i} className={`flex items-center gap-2 text-xs px-3 py-1 rounded-lg ${a.met ? "bg-green-50 text-green-800" : "bg-amber-50 text-amber-800"}`}>
              <span>{a.met ? "✓" : "⚠"}</span>
              <span className="font-medium">{a.name}</span>
              <span className="text-gray-500">— {a.detail}</span>
            </div>
          ))}
        </div>
      )}

      {/* Warnings */}
      {result.warnings?.length > 0 && (
        <div className="mt-2 space-y-1">
          {result.warnings.map((w: string, i: number) => (
            <div key={i} className="flex items-center gap-2 text-xs px-3 py-1 rounded-lg bg-amber-50 text-amber-800">
              <span>⚠</span> {w}
            </div>
          ))}
        </div>
      )}

      {/* Post-hoc results */}
      {result.posthoc?.length > 0 && (
        <div className="mt-3">
          <p className="text-xs font-semibold text-gray-600 mb-1">
            Post-hoc: {result.posthoc_method ?? "Pairwise comparisons"}
          </p>
          <div className="overflow-auto rounded border border-gray-200">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-2 py-1 text-left">Comparison</th>
                  <th className="px-2 py-1 text-right">Statistic</th>
                  <th className="px-2 py-1 text-right">p (adj)</th>
                  <th className="px-2 py-1 text-right">Effect size</th>
                  <th className="px-2 py-1 text-center">Sig</th>
                </tr>
              </thead>
              <tbody>
                {result.posthoc.map((ph: any, i: number) => (
                  <tr key={i} className={`border-t border-gray-100 ${ph.significant ? "" : "text-gray-400"}`}>
                    <td className="px-2 py-1 font-medium">{ph.group1} vs {ph.group2}</td>
                    <td className="px-2 py-1 text-right font-mono">{ph.statistic?.toFixed(3)}</td>
                    <td className="px-2 py-1 text-right font-mono">{ph.p_adj < 0.001 ? "<0.001" : ph.p_adj?.toFixed(4)}</td>
                    <td className="px-2 py-1 text-right font-mono">
                      {ph.effect_size ? `${ph.effect_size.value?.toFixed(3)} (${ph.effect_size.magnitude})` : ph.rank_diff?.toFixed(2) ?? "—"}
                    </td>
                    <td className="px-2 py-1 text-center">{ph.significant ? "✓" : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {result.groups && (
        <div className="overflow-auto rounded border border-gray-200 mt-2">
          <table>
            <thead>
              <tr>{Object.keys(result.groups[0]).map((k) => <th key={k}>{k}</th>)}</tr>
            </thead>
            <tbody>
              {result.groups.map((g: any, i: number) => (
                <tr key={i}>{Object.values(g).map((v: any, j) => (
                  <td key={j}>{typeof v === "number" ? fmt(v) : v}</td>
                ))}</tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {result.table && (
        <div className="overflow-auto rounded border border-gray-200 mt-2">
          <table>
            <thead>
              <tr>
                <th></th>
                {result.col_labels.map((l: string) => <th key={l}>{l}</th>)}
              </tr>
            </thead>
            <tbody>
              {result.table.map((row: number[], i: number) => (
                <tr key={i}>
                  <td className="font-medium text-gray-900">{result.row_labels[i]}</td>
                  {row.map((v, j) => <td key={j}>{v}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function HypothesisPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;

  const numCols = session.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
  const catCols = session.columns.filter((c) => c.kind === "categorical").map((c) => c.name);

  const [test, setTest] = useState("ttest_1sample");
  const [col, setCol] = useState(numCols[0] ?? "");
  const [col2, setCol2] = useState(catCols[1] ?? catCols[0] ?? "");
  const [groupCol, setGroupCol] = useState(catCols[0] ?? "");
  const [mu, setMu] = useState("0");
  const [covariates, setCovariates] = useState<string[]>([]);
  const [factor2, setFactor2] = useState(catCols[1] ?? catCols[0] ?? "");
  const cached = useStore((s) => s.panelCache.hypothesis);
  const setCache = useStore((s) => s.setPanelCache);
  const [result, _setResult] = useState<any>(cached?.result ?? null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const setResult = (r: any) => { _setResult(r); setCache("hypothesis", { result: r }); };

  const isCat = test === "chisquare" || test === "fisher";
  const needsGroup = ["ttest_2sample", "anova", "mannwhitney", "kruskal", "ancova"].includes(test);
  const needsSecondCat = isCat;
  const isAncova = test === "ancova";
  const isTwoWay = test === "two_way";

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    const sid = session.session_id;
    try {
      let res: any;
      if (test === "ttest_1sample")  res = await runTTest({ session_id: sid, column: col, mu: +mu });
      else if (test === "ttest_2sample") res = await runTTest({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "anova")     res = await runAnova({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "mannwhitney") res = await runMannWhitney({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "kruskal")   res = await runKruskal({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "chisquare") res = await runChiSquare({ session_id: sid, row_column: col, col_column: col2 });
      else if (test === "fisher")    res = await runFisher({ session_id: sid, row_column: col, col_column: col2 });
      else if (test === "ancova")    res = await runAncova({ session_id: sid, outcome: col, group_col: groupCol, covariates });
      else if (test === "two_way")   res = await runTwoWayAnova({ session_id: sid, outcome: col, factor1: groupCol, factor2 });
      setResult(res?.data);
    } catch (e: any) {
      setError(e.response?.data?.detail ?? "Error");
    } finally { setLoading(false); }
  };

  const groups = ["Parametric", "Non-parametric", "Categorical"];

  return (
    <div className="flex gap-4">
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-1">
          {groups.map((grp) => (
            <div key={grp}>
              <p className="text-xs text-gray-400 uppercase tracking-wider mt-3 mb-1 first:mt-0">{grp}</p>
              {TESTS.filter((t) => t.group === grp).map(({ id, label }) => (
                <label key={id} className="flex items-center gap-2 cursor-pointer py-0.5">
                  <input type="radio" name="test" value={id} checked={test === id}
                    onChange={() => { setTest(id); setResult(null); }} className="accent-indigo-500" />
                  <span className="text-sm text-gray-700">{label}</span>
                </label>
              ))}
            </div>
          ))}
        </div>

        {/* Test guidance */}
        {TEST_GUIDANCE[test] && (
          <div className="panel bg-indigo-50 border-indigo-200 space-y-2">
            <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">When to use</p>
            <p className="text-xs text-indigo-800 leading-relaxed">{TEST_GUIDANCE[test].when}</p>
            <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mt-2">Assumptions</p>
            <p className="text-xs text-indigo-800 leading-relaxed">{TEST_GUIDANCE[test].assumptions}</p>
            <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mt-2">How to read</p>
            <p className="text-xs text-indigo-800 leading-relaxed">{TEST_GUIDANCE[test].reading}</p>
          </div>
        )}

        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
          <div>
            <label className="text-xs text-gray-400 block mb-1">
              {isCat ? "Row column" : "Outcome column"}
            </label>
            <select className="select w-full" value={col} onChange={(e) => setCol(e.target.value)}>
              {(isCat ? catCols : numCols).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          {test === "ttest_1sample" && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Test value (μ₀)</label>
              <input className="select w-full" type="number" value={mu} onChange={(e) => setMu(e.target.value)} />
            </div>
          )}

          {needsGroup && catCols.length > 0 && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Group column</label>
              <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                {catCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}

          {needsSecondCat && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Column variable</label>
              <select className="select w-full" value={col2} onChange={(e) => setCol2(e.target.value)}>
                {catCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}

          {/* ANCOVA: covariates multi-select */}
          {isAncova && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Covariates (continuous)</label>
              <select multiple className="select w-full h-24" value={covariates}
                onChange={(e) => setCovariates(Array.from(e.target.selectedOptions, o => o.value))}>
                {numCols.filter(c => c !== col).map((c) => <option key={c}>{c}</option>)}
              </select>
              <p className="text-[10px] text-gray-400 mt-1">Hold Ctrl/Cmd to select multiple</p>
            </div>
          )}

          {/* Two-way ANOVA: factor 2 */}
          {isTwoWay && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Factor 2</label>
              <select className="select w-full" value={factor2} onChange={(e) => setFactor2(e.target.value)}>
                {catCols.filter(c => c !== groupCol).map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}

          <button className="btn-primary w-full" onClick={run} disabled={loading || (isAncova && covariates.length === 0)}>
            {loading ? "Running…" : "Run Test"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1">
        {result ? <ResultCard result={result} /> : (
          <div className="panel h-64 flex items-center justify-center text-gray-400">
            Configure and run a hypothesis test
          </div>
        )}
      </div>
    </div>
  );
}
