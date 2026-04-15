import { useState, useEffect, useRef } from "react";
import Plot from "../PlotComponent";
import PlotExporter from "./PlotExporter";
import { useStore, PALETTES } from "../store";
import ResultExporter from "./ResultExporter";

const _pal = () => PALETTES[useStore.getState().plotTheme.palette] ?? PALETTES.indigo;
import {
  runCorrelationPair,
  runCorrelationMatrix,
  runICC,
  runCohensKappa,
  getRawColumns,
} from "../api";
import { Tip, LabelTip, InfoBanner } from "./Tip";

// ── shared layout ─────────────────────────────────────────────────────────────
const PLOT_BG: Record<string, unknown> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#f9fafb",
  font: { color: "#374151", size: 11 },
};

const TABS = ["Pairwise", "Matrix", "ICC", "Cohen's κ"] as const;
type Tab = (typeof TABS)[number];

// ── helpers ───────────────────────────────────────────────────────────────────
function pFmt(p: number) {
  if (p < 0.001) return "< 0.001";
  return p.toFixed(3);
}
function sig(p: number) {
  return p < 0.05 ? "text-indigo-600 font-semibold" : "text-gray-400";
}
function starsFor(p: number | null): string {
  if (p == null) return "";
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "";
}

function downloadCSV(filename: string, rows: string[][]): void {
  const csv = rows.map((r) => r.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ── PairResult type ───────────────────────────────────────────────────────────
interface PairResult {
  var1: string;
  var2: string;
  r: number;
  p: number;
  n: number;
  ci_low: number;
  ci_high: number;
  method: string;
  label: string;
  normality_test: string;   // "Shapiro-Wilk" | "Lilliefors" | "Skewness (CLT bypass)"
  normality: Record<string, {
    p: number | null;
    statistic: number | null;
    normal: boolean;
    skewness: number;
    test: string;
    bypass: string | null;
  }>;
  scatter: { x: number[]; y: number[] };
  regression_line: { x: number[]; y: number[] };
  ci_band: { x: number[]; y_upper: number[]; y_lower: number[] };
  autoSwitched: boolean;
  result_text?: string;
}

// ── PairwiseTab ───────────────────────────────────────────────────────────────
function PairwiseTab({ sessionId, columns }: { sessionId: string; columns: string[] }) {
  const showGrid = useStore((s) => s.showGrid);
  const corrScatterRef = useRef<any>(null);
  const [vars, setVars] = useState<string[]>(columns.slice(0, Math.min(4, columns.length)));
  const [varFilter, setVarFilter] = useState("");
  const [method, setMethod] = useState("auto");
  const [results, setResults] = useState<PairResult[]>([]);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const toggle = (c: string) =>
    setVars((prev) => prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]);

  const nPairs = Math.max(0, vars.length * (vars.length - 1) / 2);

  const run = async () => {
    if (vars.length < 2) { setError("Select at least 2 variables"); return; }
    setError(""); setResults([]); setActiveIdx(null); setLoading(true);

    const pairs: [string, string][] = [];
    for (let i = 0; i < vars.length; i++)
      for (let j = i + 1; j < vars.length; j++)
        pairs.push([vars[i], vars[j]]);

    try {
      const settled = await Promise.allSettled(
        pairs.map(([v1, v2]) =>
          runCorrelationPair({ session_id: sessionId, var1: v1, var2: v2, method, imputation: "listwise" })
        )
      );
      const parsed: PairResult[] = [];
      settled.forEach((s, i) => {
        if (s.status !== "fulfilled") return;
        const d = s.value.data;
        const autoSwitched =
          method === "auto" &&
          d.method === "spearman" &&
          Object.values(d.normality as Record<string, { normal: boolean }>).some((n) => !n.normal);
        parsed.push({ ...d, var1: pairs[i][0], var2: pairs[i][1], autoSwitched });
      });
      parsed.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
      setResults(parsed);
      if (parsed.length > 0) setActiveIdx(0);
    } catch {
      setError("Computation failed");
    } finally {
      setLoading(false);
    }
  };

  const pairHeaders = ["Variable 1", "Variable 2", "r / ρ", "95% CI Low", "95% CI High", "p", "n", "Method", "Stars"];
  const pairRows = results.map((res) => [
    res.var1, res.var2,
    res.r.toFixed(4),
    res.ci_low.toFixed(4),
    res.ci_high.toFixed(4),
    res.p < 0.001 ? "<0.001" : res.p.toFixed(4),
    String(res.n),
    res.method,
    starsFor(res.p),
  ]);


  const active = activeIdx != null ? results[activeIdx] : null;

  return (
    <div className="flex gap-4 h-full">
      {/* ── Left sidebar ── */}
      <div className="w-56 flex-shrink-0 space-y-3 overflow-y-auto">
        <div className="panel space-y-3">
          {/* Variable multi-select */}
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
            <div className="flex gap-1">
              <button onClick={() => setVars([...columns])} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">All</button>
              <button onClick={() => setVars([])} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">None</button>
            </div>
          </div>
          <input
            type="text"
            placeholder="Filter variables…"
            value={varFilter}
            onChange={(e) => setVarFilter(e.target.value)}
            className="select w-full text-xs py-1"
          />
          <div className="max-h-44 overflow-y-auto space-y-0.5 border border-gray-200 rounded p-1">
            {columns
              .filter((c) => c.toLowerCase().includes(varFilter.toLowerCase()))
              .map((c) => (
                <label key={c} className="flex items-center gap-2 px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                  <input type="checkbox" checked={vars.includes(c)} onChange={() => toggle(c)} className="accent-indigo-500" />
                  <span className="text-xs text-gray-700 truncate">{c}</span>
                </label>
              ))}
          </div>
          <p className="text-[10px] text-gray-400">{vars.length} selected · {nPairs} pair{nPairs !== 1 ? "s" : ""}</p>

          {/* Method */}
          <h3 className="text-sm font-semibold text-gray-700 pt-1">
            Method
            <Tip text="Auto: runs Shapiro-Wilk normality test for each pair and picks Pearson if both variables are normal (p > 0.05), or Spearman if either is not. Prevents the common mistake of running Pearson on skewed data." wide />
          </h3>
          {(["auto", "pearson", "spearman"] as const).map((m) => (
            <label key={m} className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="pw-method" value={m} checked={method === m}
                onChange={() => setMethod(m)} className="accent-indigo-500" />
              <span className="text-sm text-gray-700">{m === "auto" ? "Auto (Shapiro-Wilk)" : m === "pearson" ? "Pearson r" : "Spearman ρ"}</span>
            </label>
          ))}

          <button className="btn-primary w-full" onClick={run} disabled={loading || vars.length < 2}>
            {loading ? "Computing…" : `Compute ${nPairs > 1 ? `${nPairs} Pairs` : "Pair"}`}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>

        {/* Normality box for active pair */}
        {active && (
          <div className="panel space-y-2 text-xs">
            <p className="text-gray-500 font-semibold">
              Normality Assessment
              <Tip wide text={
                active.n <= 2000
                  ? `Shapiro-Wilk (n = ${active.n}): gold-standard test for normality in small-to-medium samples. ✓ = normal (p ≥ 0.05). If either variable is non-normal, Spearman ρ is preferred.`
                  : `Large sample (n = ${active.n} > 2000). Two-step approach: (1) Skewness check — if |skewness| ≤ 1.5, the Central Limit Theorem ensures Pearson r remains valid regardless of distribution shape. (2) If |skewness| > 1.5, Lilliefors-corrected KS test is used (standard KS adjusted for estimated parameters). Standard KS without Lilliefors correction would give anti-conservative p-values.`
              } />
            </p>
            {Object.entries(active.normality).map(([v, nm]) => (
              <div key={v} className="space-y-0.5">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 truncate max-w-[100px]" title={v}>{v}</span>
                  <div className="flex items-center gap-1.5">
                    {nm.bypass === "clt_skew" ? (
                      <span className="text-green-600 font-medium" title="CLT bypass: mild skewness at large n — Pearson is robust">
                        CLT ✓
                      </span>
                    ) : (
                      <span className={nm.normal ? "text-green-600" : "text-red-500"}>
                        {nm.p != null ? `p=${pFmt(nm.p)}` : ""} {nm.normal ? "✓" : "✗"}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex justify-between text-[10px] text-gray-400 pl-0.5">
                  <span>{nm.test}</span>
                  <span title="Skewness: 0 = symmetric. |skew| < 1 = mild, 1–2 = moderate, > 2 = severe">
                    skew={nm.skewness.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
            <p className="text-gray-400 border-t border-gray-100 pt-1.5">
              Correlation: <span className="text-indigo-600">{active.method === "pearson" ? "Pearson r" : "Spearman ρ"}</span>
            </p>
          </div>
        )}
      </div>

      {/* ── Main content ── */}
      <div className="flex-1 flex flex-col gap-3 min-h-0 overflow-y-auto">
        {results.length > 0 ? (
          <>
            {/* Results table */}
            <div className="panel flex-shrink-0">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-gray-500">Results ({results.length} pairs)</span>
                <ResultExporter title="correlation_pairwise" headers={pairHeaders} rows={pairRows} />
              </div>
              <div className="overflow-auto max-h-52">
                <table className="w-full text-xs border-collapse">
                  <thead className="sticky top-0 bg-white z-10">
                    <tr className="text-left text-gray-400 border-b border-gray-200">
                      <th className="pb-1.5 pr-3 font-medium">Variable 1</th>
                      <th className="pb-1.5 pr-3 font-medium">Variable 2</th>
                      <th className="pb-1.5 pr-3 font-medium">r / ρ</th>
                      <th className="pb-1.5 pr-3 font-medium">95% CI</th>
                      <th className="pb-1.5 pr-3 font-medium">p</th>
                      <th className="pb-1.5 pr-3 font-medium">n</th>
                      <th className="pb-1.5 font-medium">Method</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((res, i) => {
                      const highCorr = Math.abs(res.r) >= 0.70;
                      const isActive = i === activeIdx;
                      return (
                        <tr
                          key={`${res.var1}-${res.var2}`}
                          onClick={() => setActiveIdx(i)}
                          className={`cursor-pointer border-b border-gray-100 transition-colors ${
                            isActive
                              ? "bg-indigo-50"
                              : highCorr
                              ? "bg-amber-50 hover:bg-amber-100"
                              : "hover:bg-gray-50"
                          }`}
                        >
                          <td className="py-1 pr-3 font-mono text-gray-700">{res.var1}</td>
                          <td className="py-1 pr-3 font-mono text-gray-700">{res.var2}</td>
                          <td className="py-1 pr-3">
                            <span className={`font-semibold font-mono ${Math.abs(res.r) >= 0.5 ? "text-indigo-600" : "text-gray-700"}`}>
                              {res.r.toFixed(3)}
                            </span>
                            <span className="text-amber-500 ml-0.5">{starsFor(res.p)}</span>
                            {highCorr && (
                              <span className="ml-1 text-amber-500" title="High collinearity (|r| ≥ 0.70)">⚠</span>
                            )}
                            {res.autoSwitched && (
                              <span className="ml-1 text-[9px] bg-blue-100 text-blue-600 rounded px-0.5" title="Auto-switched to Spearman (non-normal data)">S</span>
                            )}
                          </td>
                          <td className="py-1 pr-3 text-gray-400 font-mono text-[10px] whitespace-nowrap">
                            [{res.ci_low.toFixed(2)}, {res.ci_high.toFixed(2)}]
                          </td>
                          <td className={`py-1 pr-3 font-mono ${res.p < 0.05 ? "text-indigo-600 font-semibold" : "text-gray-400"}`}>
                            {pFmt(res.p)}
                          </td>
                          <td className="py-1 pr-3 text-gray-400">{res.n}</td>
                          <td className="py-1 text-gray-400">{res.method === "pearson" ? "r" : "ρ"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="flex flex-wrap gap-3 mt-1.5 text-[10px] text-gray-400 border-t border-gray-100 pt-1.5">
                <span title="Significance stars show how unlikely the result is by chance. * means p < 0.05 (significant), ** means p < 0.01 (highly significant), *** means p < 0.001 (very highly significant). A result with *** has less than 0.1% probability of occurring by chance alone.">
                  <span className="text-amber-500 font-semibold">*</span> p&lt;0.05 &nbsp;
                  <span className="text-amber-500 font-semibold">**</span> p&lt;0.01 &nbsp;
                  <span className="text-amber-500 font-semibold">***</span> p&lt;0.001
                  <span className="ml-1 cursor-help text-gray-300" title="Significance stars: * p<0.05 (significant — less than 5% chance this is a fluke), ** p<0.01 (highly significant — less than 1% chance), *** p<0.001 (very highly significant — less than 0.1% chance). More stars = stronger evidence against a chance finding.">ⓘ</span>
                </span>
                <span className="text-amber-500">⚠ High collinearity (|r| ≥ 0.70)</span>
                {method === "auto" && <span className="text-blue-500">S = auto-switched to Spearman</span>}
              </div>
            </div>

            {/* Auto-switch alert */}
            {active?.autoSwitched && (
              <div className="flex items-center gap-2 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-xs text-blue-700 flex-shrink-0">
                <span>⚡</span>
                <span>
                  <strong>Switched to Spearman:</strong>{" "}
                  {active.normality_test === "Lilliefors"
                    ? `Marked skewness detected (|skew| > 1.5) and Lilliefors p < 0.05 (n = ${active.n}). Distribution is non-normal; Spearman ρ is more appropriate.`
                    : `Data is not normally distributed (${active.normality_test ?? "Shapiro-Wilk"} p < 0.05). Pearson would give misleading results on skewed data.`}
                </span>
              </div>
            )}

            {/* High collinearity alert */}
            {active && Math.abs(active.r) >= 0.70 && (
              <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-700 flex-shrink-0">
                <span>⚠</span>
                <span>
                  <strong>High Collinearity Detected (|r| = {Math.abs(active.r).toFixed(3)} ≥ 0.70).</strong>{" "}
                  Do not use both <em>{active.var1}</em> and <em>{active.var2}</em> in the same regression model — they carry redundant information and will inflate standard errors.
                </span>
              </div>
            )}

            {/* Summary stats for active pair */}
            {active && (
              <div className="panel flex-shrink-0 space-y-2">
                <div className="flex gap-5 text-sm flex-wrap items-center">
                  <span className="text-gray-500">
                    <LabelTip tip="Correlation coefficient. |r| < 0.1 negligible · 0.1–0.3 weak · 0.3–0.5 moderate · > 0.5 strong." wide>
                      {active.label}
                    </LabelTip>
                    {" = "}
                    <span className={`font-bold ${Math.abs(active.r) >= 0.5 ? "text-indigo-600" : "text-gray-700"}`}>
                      {active.r.toFixed(4)}
                    </span>
                    <span className="ml-1 text-amber-500 font-semibold">{starsFor(active.p)}</span>
                    <span className="ml-2 text-xs text-gray-400">
                      ({Math.abs(active.r) < 0.1 ? "negligible" : Math.abs(active.r) < 0.3 ? "weak" : Math.abs(active.r) < 0.5 ? "moderate" : "strong"})
                    </span>
                  </span>
                  <span className="text-gray-500">
                    <LabelTip tip="95% Confidence Interval for the correlation. Narrow CI = more precise estimate.">95% CI</LabelTip>
                    {": "}
                    <span className="text-gray-700">[{active.ci_low.toFixed(4)}, {active.ci_high.toFixed(4)}]</span>
                  </span>
                  <span className="text-gray-500">
                    p{" = "}<span className={sig(active.p)}>{pFmt(active.p)}</span>
                  </span>
                  <span className="text-gray-400">n = {active.n}</span>
                </div>
                <InfoBanner>
                  {active.p < 0.05
                    ? `Significant ${Math.abs(active.r) < 0.3 ? "weak" : Math.abs(active.r) < 0.5 ? "moderate" : "strong"} ${active.r > 0 ? "positive" : "negative"} correlation between ${active.var1} and ${active.var2} (${active.label} = ${active.r.toFixed(3)}, p ${pFmt(active.p)}). As one increases, the other tends to ${active.r > 0 ? "increase" : "decrease"}.`
                    : `No statistically significant correlation found between ${active.var1} and ${active.var2} (p = ${pFmt(active.p)}). The sample may be too small, or the relationship may be non-linear.`}
                </InfoBanner>
                {active.result_text && (
                  <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 mt-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase">Results Paragraph</span>
                      <button onClick={() => navigator.clipboard.writeText(active.result_text!)} className="text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
                    </div>
                    <p className="text-sm text-gray-700 leading-relaxed">{active.result_text}</p>
                  </div>
                )}
              </div>
            )}

            {/* Scatter plot with regression line + 95% CI band */}
            {active && (
              <div className="panel flex-1 min-h-0 relative" style={{ minHeight: 300 }}>
                <PlotExporter plotRef={corrScatterRef} title="Correlation_Scatter" />
                <Plot
                  data={[
                    {
                      type: "scatter",
                      mode: "markers",
                      x: active.scatter.x,
                      y: active.scatter.y,
                      marker: { color: _pal()[0], opacity: 0.65, size: 6 },
                      name: "Data",
                      hovertemplate: `${active.var1}: %{x:.3f}<br>${active.var2}: %{y:.3f}<extra></extra>`,
                    },
                    {
                      type: "scatter",
                      mode: "lines",
                      x: active.regression_line.x,
                      y: active.regression_line.y,
                      line: { color: "#f59e0b", width: 2 },
                      name: "Regression line",
                      hoverinfo: "skip",
                    },
                    {
                      type: "scatter",
                      mode: "lines",
                      x: [...active.ci_band.x, ...[...active.ci_band.x].reverse()],
                      y: [...active.ci_band.y_upper, ...[...active.ci_band.y_lower].reverse()],
                      fill: "toself",
                      fillcolor: "rgba(245,158,11,0.12)",
                      line: { width: 0 },
                      name: "95% CI band",
                      hoverinfo: "skip",
                      showlegend: true,
                    },
                  ]}
                  layout={{
                    ...PLOT_BG,
                    autosize: true,
                    xaxis: { title: active.var1, gridcolor: "#e5e7eb", showgrid: showGrid, zeroline: false },
                    yaxis: { title: active.var2, gridcolor: "#e5e7eb", showgrid: showGrid, zeroline: false },
                    legend: { orientation: "h", y: -0.18, font: { size: 10, color: "#374151" } },
                    annotations: [
                      {
                        xref: "paper" as const,
                        yref: "paper" as const,
                        x: 0.98,
                        y: 0.98,
                        xanchor: "right" as const,
                        yanchor: "top" as const,
                        text: `${active.label} = ${active.r.toFixed(3)}${starsFor(active.p)}, p = ${pFmt(active.p)}`,
                        showarrow: false,
                        font: { color: "#374151", size: 12 },
                        bgcolor: "rgba(249,250,251,0.9)",
                        bordercolor: "#e5e7eb",
                        borderpad: 4,
                        borderwidth: 1,
                      },
                    ],
                    margin: { t: 20, r: 20, b: 60, l: 60 },
                  }}
                  onInitialized={(_: object, gd: HTMLElement) => { corrScatterRef.current = gd; }}
                  onUpdate={(_: object, gd: HTMLElement)      => { corrScatterRef.current = gd; }}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                  config={{ responsive: true, displayModeBar: false }}
                />
              </div>
            )}
          </>
        ) : (
          <div className="panel flex-1 flex items-center justify-center text-gray-400">
            Select variables and click Compute
          </div>
        )}
      </div>
    </div>
  );
}

// ── MatrixTab ─────────────────────────────────────────────────────────────────
function MatrixTab({ sessionId, columns }: { sessionId: string; columns: string[] }) {
  const showGrid = useStore((s) => s.showGrid);
  const corrHeatmapRef = useRef<any>(null);
  const corrSplomRef = useRef<any>(null);
  const [selected, setSelected] = useState<string[]>(columns.slice(0, Math.min(8, columns.length)));
  const [colFilter, setColFilter] = useState("");
  const [method, setMethod] = useState("pearson");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [displayMode, setDisplayMode] = useState<"heatmap" | "splom">("heatmap");
  const [rawData, setRawData] = useState<Record<string, (number | null)[]> | null>(null);
  const [rawLoading, setRawLoading] = useState(false);
  const [selectedVar, setSelectedVar] = useState<string | null>(null);

  const toggle = (c: string) =>
    setSelected((prev) => prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]);

  const run = async () => {
    if (selected.length < 2) { setError("Select at least 2 variables"); return; }
    setError("");
    setLoading(true);
    try {
      const res = await runCorrelationMatrix({ session_id: sessionId, variables: selected, method, imputation: "listwise" });
      setData(res.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "Error");
    } finally {
      setLoading(false);
    }
  };

  // Fetch raw data when SPLOM mode is active
  useEffect(() => {
    if (displayMode !== "splom" || selected.length < 2) return;
    setRawLoading(true);
    getRawColumns(sessionId, selected)
      .then((r) => setRawData(r.data))
      .catch(() => {})
      .finally(() => setRawLoading(false));
  }, [displayMode, selected.join(","), sessionId]);

  const exportMatrix = () => {
    if (!data) return;
    const vars: string[] = data.variables;
    const header = ["", ...vars];
    const rows = vars.map((r: string) => [
      r,
      ...vars.map((c: string) => {
        if (r === c) return "1";
        const v = data.matrix[r][c];
        return v != null ? v.toFixed(4) : "";
      }),
    ]);
    downloadCSV("correlation_matrix.csv", [header, ...rows]);
  };

  // Build SPLOM distribution panel for a selected variable
  const renderVarDetail = (v: string) => {
    if (!rawData || !rawData[v]) return null;
    const vals = rawData[v].filter((x): x is number => x != null);
    if (vals.length === 0) return null;
    const minV = Math.min(...vals), maxV = Math.max(...vals);
    const nBins = Math.min(20, Math.max(6, Math.round(vals.length ** 0.38)));
    const binW = (maxV - minV) / nBins;
    const counts = Array(nBins).fill(0);
    vals.forEach((x) => { const b = Math.min(Math.floor((x - minV) / binW), nBins - 1); counts[b]++; });
    const xs = counts.map((_, i) => +(minV + (i + 0.5) * binW).toFixed(4));
    return (
      <div className="panel flex-shrink-0 w-48 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <p className="text-xs font-semibold text-gray-700 truncate">{v}</p>
          <button onClick={() => setSelectedVar(null)} className="text-gray-300 hover:text-gray-600 text-xs">✕</button>
        </div>
        <p className="text-[10px] text-gray-400">n = {vals.length}</p>
        <Plot
          data={[{ type: "bar" as const, x: xs, y: counts,
            marker: { color: _pal()[0], opacity: 0.8 },
            hovertemplate: "%{x:.2f}: %{y}<extra></extra>" }]}
          layout={{
            paper_bgcolor: "transparent", plot_bgcolor: "#f9fafb",
            font: { color: "#374151", size: 9 },
            height: 130, margin: { t: 8, r: 4, b: 28, l: 28 },
            xaxis: { gridcolor: "#f3f4f6", zeroline: false, title: { text: v, font: { size: 8 } } },
            yaxis: { gridcolor: "#f3f4f6", zeroline: false },
            showlegend: false,
          }}
          style={{ width: "100%", height: 130 }}
          useResizeHandler
          config={{ displayModeBar: false, responsive: true }}
        />
        {data?.matrix?.[v] && (
          <div className="space-y-0.5">
            <p className="text-[10px] text-gray-400 font-semibold">Correlations</p>
            {Object.entries(data.matrix[v] as Record<string, number>)
              .filter(([k]) => k !== v)
              .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
              .slice(0, 6)
              .map(([k, r]) => (
                <div key={k} className="flex items-center justify-between gap-1">
                  <span className="text-[9px] text-gray-500 truncate flex-1">{k}</span>
                  <span className={`text-[9px] font-mono font-semibold ${Math.abs(r) >= 0.5 ? "text-indigo-600" : "text-gray-500"}`}>
                    {r.toFixed(3)}
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex gap-4 h-full">
      {/* Controls */}
      <div className="w-52 flex-shrink-0 space-y-4 overflow-y-auto">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">
            Method
            <Tip text="Pearson: linear relationships, normally distributed data. Spearman: ranked/non-normal. Kendall: small samples or many tied ranks. Missing values handled with pairwise deletion — each pair uses its own complete cases, preserving maximum sample size." wide />
          </h3>
          {["pearson", "spearman", "kendall"].map((m) => (
            <label key={m} className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="mx-method" value={m} checked={method === m}
                onChange={() => setMethod(m)} className="accent-indigo-500" />
              <span className="text-sm text-gray-700 capitalize">{m}</span>
            </label>
          ))}

          {/* Display mode toggle */}
          <div className="pt-1 border-t border-gray-100">
            <p className="text-xs font-semibold text-gray-700 mb-2">Display</p>
            <div className="flex rounded overflow-hidden border border-gray-200">
              {(["heatmap", "splom"] as const).map((mode) => (
                <button key={mode} onClick={() => setDisplayMode(mode)}
                  className={`flex-1 text-xs py-1 transition-colors ${displayMode === mode ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}>
                  {mode === "heatmap" ? "Heatmap" : "Scatter"}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-between pt-1">
            <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
            <div className="flex gap-1">
              <button onClick={() => setSelected([...columns])} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">All</button>
              <button onClick={() => setSelected([])} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">None</button>
            </div>
          </div>
          <input
            type="text"
            placeholder="Filter variables…"
            value={colFilter}
            onChange={(e) => setColFilter(e.target.value)}
            className="select w-full text-xs py-1"
          />
          <div className="space-y-0.5 max-h-48 overflow-y-auto border border-gray-200 rounded p-1">
            {columns
              .filter((c) => c.toLowerCase().includes(colFilter.toLowerCase()))
              .map((c) => (
                <label key={c} className="flex items-center gap-2 px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                  <input type="checkbox" checked={selected.includes(c)} onChange={() => toggle(c)}
                    className="accent-indigo-500" />
                  <span className="text-xs text-gray-700 truncate">{c}</span>
                </label>
              ))}
          </div>
          <p className="text-[10px] text-gray-400">{selected.length} selected</p>

          <button className="btn-primary w-full" onClick={run} disabled={loading}>
            {loading ? "Computing…" : "Compute Matrix"}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}

          <div className="text-[10px] text-gray-400 leading-relaxed border-t border-gray-100 pt-2">
            <p>Missing values use <strong>pairwise deletion</strong> — each pair is computed on all rows where both variables are present, preserving maximum sample size.</p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex gap-3 min-h-0">
        <div className="flex-1 flex flex-col gap-3 min-h-0">
          {data ? (
            <>
              {/* Multicollinearity warnings */}
              {data.multicollinearity_warnings.length > 0 && (
                <div className="panel flex-shrink-0 space-y-1 border-amber-200 bg-amber-50">
                  <p className="text-xs font-semibold text-amber-700 flex items-center">
                    ⚠ High Collinearity Detected (|r| ≥ 0.70)
                    <Tip text="Two predictors are so strongly correlated they carry redundant information. In regression, this inflates standard errors and makes coefficients unreliable. Do not include both in the same model — remove the one that is less clinically meaningful." wide />
                  </p>
                  {data.multicollinearity_warnings.map((w: any, i: number) => (
                    <p key={i} className="text-xs text-gray-700">
                      <span className={w.severity === "high" ? "text-red-600 font-semibold" : "text-amber-600 font-medium"}>
                        {w.var1} ↔ {w.var2}
                      </span>
                      <span className="text-gray-400 ml-2">r = {w.r.toFixed(3)}</span>
                      {w.severity === "high"
                        ? <span className="text-red-500 ml-2">⚠ Very high (&gt; 0.90) — strong redundancy</span>
                        : <span className="text-amber-500 ml-2">— Do not use both in the same regression</span>}
                    </p>
                  ))}
                </div>
              )}

              {displayMode === "heatmap" ? (
                /* Heatmap */
                <div className="panel flex-1 min-h-0 flex flex-col gap-2 relative">
                  <PlotExporter plotRef={corrHeatmapRef} title="Correlation_Matrix" />
                  <div className="flex items-center justify-between flex-shrink-0">
                    <span className="text-xs font-semibold text-gray-500">Correlation Matrix</span>
                    <button
                      onClick={exportMatrix}
                      className="flex items-center gap-1 text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-gray-50 hover:text-indigo-600 hover:border-indigo-300 transition-colors"
                    >
                      ↓ Export CSV
                    </button>
                  </div>
                  <Plot
                    data={[{
                      type: "heatmap",
                      z: data.variables.map((c1: string) =>
                        data.variables.map((c2: string) => data.matrix[c1][c2])
                      ),
                      x: data.variables,
                      y: data.variables,
                      colorscale: [
                        [0,   "#2563eb"],
                        [0.25,"#93c5fd"],
                        [0.5, "#f9fafb"],
                        [0.75,"#fca5a5"],
                        [1,   "#dc2626"],
                      ],
                      zmid: 0, zmin: -1, zmax: 1,
                      text: data.variables.map((c1: string) =>
                        data.variables.map((c2: string) => {
                          if (c1 === c2) return "1";
                          const v = data.matrix[c1][c2];
                          const p = data.p_matrix?.[c1]?.[c2];
                          if (v == null) return "";
                          return `${v.toFixed(2)}${starsFor(p)}`;
                        })
                      ),
                      texttemplate: "%{text}",
                      textfont: { size: 11, color: "#111827" },
                      hovertemplate: "%{x} vs %{y}: %{z:.4f}<extra></extra>",
                    }]}
                    layout={{
                      ...PLOT_BG,
                      autosize: true,
                      xaxis: { showgrid: showGrid, gridcolor: "#e5e7eb", zeroline: false },
                      yaxis: { showgrid: showGrid, gridcolor: "#e5e7eb", zeroline: false },
                      margin: { t: 20, r: 20, b: 100, l: 100 },
                    }}
                    onInitialized={(_: object, gd: HTMLElement) => { corrHeatmapRef.current = gd; }}
                    onUpdate={(_: object, gd: HTMLElement)      => { corrHeatmapRef.current = gd; }}
                    style={{ width: "100%", height: "100%", flex: 1 }}
                    useResizeHandler
                    config={{ responsive: true, displayModeBar: false }}
                  />
                  {/* Significance legend */}
                  <div className="flex gap-4 text-[10px] text-gray-400 flex-shrink-0">
                    <span>* p &lt; 0.05</span>
                    <span>** p &lt; 0.01</span>
                    <span>*** p &lt; 0.001</span>
                    <span className="ml-2 text-blue-500">■ Negative correlation</span>
                    <span className="text-red-500">■ Positive correlation</span>
                  </div>
                </div>
              ) : (
                /* Scatter Matrix (SPLOM) */
                <div className="panel flex-1 min-h-0 flex flex-col gap-2 relative">
                  <PlotExporter plotRef={corrSplomRef} title="Scatter_Matrix" />
                  <div className="flex items-center justify-between flex-shrink-0">
                    <span className="text-xs font-semibold text-gray-500">
                      Scatter Matrix
                      <Tip text="Each cell shows the scatterplot between two variables. Diagonal shows histogram of each variable. Click a variable label to see its distribution detail on the right. Only numeric columns are included." wide />
                    </span>
                    <div className="flex items-center gap-2">
                      {rawLoading && <span className="text-[10px] text-gray-400 animate-pulse">Loading data…</span>}
                      <button onClick={exportMatrix}
                        className="flex items-center gap-1 text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-gray-50 hover:text-indigo-600 hover:border-indigo-300 transition-colors">
                        ↓ Export CSV
                      </button>
                    </div>
                  </div>
                  {rawData && Object.keys(rawData).length >= 2 ? (
                    <Plot
                      data={[{
                        type: "splom" as const,
                        dimensions: Object.keys(rawData).map((col) => ({
                          label: col,
                          values: rawData[col],
                        })),
                        marker: {
                          color: _pal()[0],
                          size: 3,
                          opacity: 0.45,
                          line: { color: "#a5b4fc", width: 0.5 },
                        },
                        diagonal: { visible: true },
                        showupperhalf: false,
                        text: Object.keys(rawData).join(", "),
                        hovertemplate: "%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<extra></extra>",
                      } as any]}
                      layout={{
                        paper_bgcolor: "transparent",
                        plot_bgcolor: "#f9fafb",
                        font: { color: "#374151", size: 10 },
                        autosize: true,
                        margin: { t: 20, r: 20, b: 20, l: 20 },
                        dragmode: "select" as const,
                      }}
                      onInitialized={(_: object, gd: HTMLElement) => { corrSplomRef.current = gd; }}
                      onUpdate={(_: object, gd: HTMLElement)      => { corrSplomRef.current = gd; }}
                      style={{ width: "100%", flex: 1 }}
                      useResizeHandler
                      config={{ responsive: true, displayModeBar: false }}
                      onClickAnnotation={(e: any) => setSelectedVar(e?.annotation?.text ?? null)}
                    />
                  ) : (
                    <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
                      {rawLoading ? "Loading scatter data…" : "Click Compute Matrix first, then switch to Scatter view"}
                    </div>
                  )}
                  <p className="text-[10px] text-gray-400 flex-shrink-0">
                    Showing {Object.keys(rawData ?? {}).length} numeric variables · up to 3 000 rows · click variable name on plot to see detail
                  </p>
                </div>
              )}
            </>
          ) : (
            <div className="panel flex-1 flex items-center justify-center text-gray-400">
              Select variables and click Compute Matrix
            </div>
          )}
        </div>

        {/* Variable detail side panel (SPLOM mode) */}
        {displayMode === "splom" && selectedVar && renderVarDetail(selectedVar)}
      </div>
    </div>
  );
}

// ── ICCTab ────────────────────────────────────────────────────────────────────
function ICCTab({ sessionId, columns }: { sessionId: string; columns: string[] }) {
  const showGrid = useStore((s) => s.showGrid);
  const blandAltmanRef = useRef<any>(null);
  const [rater1, setRater1] = useState(columns[0] ?? "");
  const [rater2, setRater2] = useState(columns[1] ?? "");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    if (!rater1 || !rater2 || rater1 === rater2) { setError("Select two different columns"); return; }
    setError("");
    setLoading(true);
    try {
      const res = await runICC({ session_id: sessionId, rater1_col: rater1, rater2_col: rater2 });
      setData(res.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "Error");
    } finally {
      setLoading(false);
    }
  };

  const interpColor = (i: string) =>
    i === "Excellent" ? "text-green-600" : i === "Good" ? "text-emerald-600" :
    i === "Moderate" ? "text-amber-600" : "text-red-500";

  const exportICC = () => {
    if (!data) return;
    const header = ["ICC(2,1)", "95% CI Low", "95% CI High", "F stat", "p", "n", "Interpretation"];
    const row = [
      data.icc.toFixed(4),
      data.ci_low.toFixed(4),
      data.ci_high.toFixed(4),
      data.f_stat.toFixed(4),
      data.f_p < 0.001 ? "<0.001" : data.f_p.toFixed(4),
      String(data.n),
      data.interpretation,
    ];
    downloadCSV("icc_result.csv", [header, row]);
  };

  return (
    <div className="flex gap-4 h-full">
      <div className="w-52 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">
            ICC(2,1) — Absolute Agreement
            <Tip text="Intraclass Correlation Coefficient measures how consistently two raters measure the same subjects. ICC(2,1) is a two-way mixed model that tests both whether raters agree AND whether their absolute values match — stricter than consistency." wide />
          </h3>
          <p className="text-xs text-gray-400 leading-tight">
            Two-way mixed model for continuous inter-observer agreement (Shrout &amp; Fleiss, 1979)
          </p>
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Rater 1 Column</label>
            <select className="select w-full text-sm" value={rater1} onChange={(e) => setRater1(e.target.value)}>
              {columns.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Rater 2 Column</label>
            <select className="select w-full text-sm" value={rater2} onChange={(e) => setRater2(e.target.value)}>
              {columns.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <button className="btn-primary w-full" onClick={run} disabled={loading}>
            {loading ? "Computing…" : "Compute"}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>

        {data && (
          <div className="panel space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <p className="text-gray-400 text-xs font-semibold">ICC(2,1) Result</p>
              <button onClick={exportICC} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50 hover:text-indigo-600 hover:border-indigo-300 transition-colors">↓ CSV</button>
            </div>
            <p className="text-2xl font-bold text-gray-900">{data.icc.toFixed(3)}</p>
            <p className="text-xs text-gray-500">95% CI: [{data.ci_low.toFixed(3)}, {data.ci_high.toFixed(3)}]</p>
            <p className="text-xs text-gray-500">F({data.n - 1}, {data.n - 1}) = {data.f_stat.toFixed(2)}, p = {pFmt(data.f_p)}</p>
            <p className={`text-xs font-bold ${interpColor(data.interpretation)}`}>{data.interpretation}</p>
            <p className="text-xs text-gray-400">n = {data.n} subjects</p>
            <div className="text-xs text-gray-400 pt-1 leading-tight border-t border-gray-200 mt-1">
              <p>≥ 0.90 Excellent</p><p>≥ 0.75 Good</p><p>≥ 0.50 Moderate</p><p>&lt; 0.50 Poor</p>
            </div>
            <InfoBanner>
              ICC = {data.icc.toFixed(3)} — {data.interpretation} agreement.{" "}
              {data.icc >= 0.75 ? "These two raters can be used interchangeably." : data.icc >= 0.5 ? "Agreement is acceptable but consider rater training." : "Poor agreement — do not treat the two raters as equivalent."}
            </InfoBanner>
          </div>
        )}
      </div>

      <div className="flex-1 panel min-h-0 relative">
        {data ? (
          <>
          <PlotExporter plotRef={blandAltmanRef} title="Bland_Altman_Plot" />
          <Plot
            data={[{
              type: "scatter", mode: "markers",
              x: data.bland_altman.means, y: data.bland_altman.diffs,
              marker: { color: _pal()[0], opacity: 0.7, size: 6 },
              name: "Subjects",
              hovertemplate: "Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>",
            }]}
            layout={{
              ...PLOT_BG,
              autosize: true,
              xaxis: { title: `Mean of ${rater1} & ${rater2}`, gridcolor: "#e5e7eb", showgrid: showGrid, zeroline: false },
              yaxis: { title: `${rater1} − ${rater2}`, gridcolor: "#e5e7eb", showgrid: showGrid, zeroline: true, zerolinecolor: "#d1d5db" },
              shapes: [
                { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: data.bland_altman.mean_diff, y1: data.bland_altman.mean_diff, line: { color: "#f59e0b", width: 2 } },
                { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: data.bland_altman.loa_upper, y1: data.bland_altman.loa_upper, line: { color: "#ef4444", width: 1.5, dash: "dash" } },
                { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: data.bland_altman.loa_lower, y1: data.bland_altman.loa_lower, line: { color: "#ef4444", width: 1.5, dash: "dash" } },
              ],
              annotations: [
                { xref: "paper", yref: "y", x: 1.01, y: data.bland_altman.mean_diff, text: `Bias: ${data.bland_altman.mean_diff.toFixed(3)}`, showarrow: false, font: { color: "#b45309", size: 10 }, xanchor: "left" },
                { xref: "paper", yref: "y", x: 1.01, y: data.bland_altman.loa_upper, text: `+1.96 SD: ${data.bland_altman.loa_upper.toFixed(3)}`, showarrow: false, font: { color: "#dc2626", size: 10 }, xanchor: "left" },
                { xref: "paper", yref: "y", x: 1.01, y: data.bland_altman.loa_lower, text: `−1.96 SD: ${data.bland_altman.loa_lower.toFixed(3)}`, showarrow: false, font: { color: "#dc2626", size: 10 }, xanchor: "left" },
              ],
              margin: { t: 20, r: 130, b: 60, l: 60 },
              title: { text: "Bland-Altman Plot", font: { color: "#374151", size: 12 } },
            }}
            onInitialized={(_: object, gd: HTMLElement) => { blandAltmanRef.current = gd; }}
            onUpdate={(_: object, gd: HTMLElement)      => { blandAltmanRef.current = gd; }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
          </>
        ) : (
          <div className="h-full flex items-center justify-center text-gray-400">
            Select two rater columns and click Compute
          </div>
        )}
      </div>
    </div>
  );
}

// ── KappaTab ──────────────────────────────────────────────────────────────────
function KappaTab({ sessionId, columns }: { sessionId: string; columns: string[] }) {
  const showGrid = useStore((s) => s.showGrid);
  const kappaMatrixRef = useRef<any>(null);
  const [rater1, setRater1] = useState(columns[0] ?? "");
  const [rater2, setRater2] = useState(columns[1] ?? "");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    if (!rater1 || !rater2 || rater1 === rater2) { setError("Select two different columns"); return; }
    setError("");
    setLoading(true);
    try {
      const res = await runCohensKappa({ session_id: sessionId, rater1_col: rater1, rater2_col: rater2 });
      setData(res.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "Error");
    } finally {
      setLoading(false);
    }
  };

  const interpColor = (i: string) =>
    i === "Almost Perfect" ? "text-green-600" : i === "Substantial" ? "text-emerald-600" :
    i === "Moderate" ? "text-amber-600" : i === "Fair" ? "text-orange-500" : "text-red-500";

  const exportKappa = () => {
    if (!data) return;
    const header = ["κ", "95% CI Low", "95% CI High", "SE", "n", "Interpretation"];
    const row = [
      data.kappa.toFixed(4),
      data.ci_low.toFixed(4),
      data.ci_high.toFixed(4),
      data.se.toFixed(4),
      String(data.n),
      data.interpretation,
    ];
    downloadCSV("cohens_kappa_result.csv", [header, row]);
  };

  return (
    <div className="flex gap-4 h-full">
      <div className="w-52 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">
            Cohen's Kappa
            <Tip text="Cohen's κ measures agreement between two raters on categorical labels, correcting for agreement that would occur purely by chance. A κ of 0 means agreement no better than chance; 1 means perfect agreement." wide />
          </h3>
          <p className="text-xs text-gray-400 leading-tight">
            Categorical inter-observer agreement (Landis &amp; Koch, 1977)
          </p>
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Rater 1 Column</label>
            <select className="select w-full text-sm" value={rater1} onChange={(e) => setRater1(e.target.value)}>
              {columns.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Rater 2 Column</label>
            <select className="select w-full text-sm" value={rater2} onChange={(e) => setRater2(e.target.value)}>
              {columns.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <button className="btn-primary w-full" onClick={run} disabled={loading}>
            {loading ? "Computing…" : "Compute"}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>

        {data && (
          <div className="panel space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <p className="text-gray-400 text-xs font-semibold">κ Result</p>
              <button onClick={exportKappa} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50 hover:text-indigo-600 hover:border-indigo-300 transition-colors">↓ CSV</button>
            </div>
            <p className="text-2xl font-bold text-gray-900">{data.kappa.toFixed(3)}</p>
            <p className="text-xs text-gray-500">95% CI: [{data.ci_low.toFixed(3)}, {data.ci_high.toFixed(3)}]</p>
            <p className="text-xs text-gray-500">SE = {data.se.toFixed(4)}</p>
            <p className={`text-xs font-bold ${interpColor(data.interpretation)}`}>{data.interpretation}</p>
            <p className="text-xs text-gray-400">n = {data.n}</p>
            <InfoBanner>
              κ = {data.kappa.toFixed(3)} ({data.interpretation}).{" "}
              {data.kappa > 0.8 ? "Excellent inter-rater reliability." : data.kappa > 0.6 ? "Good reliability — raters agree beyond chance most of the time." : data.kappa > 0.4 ? "Moderate reliability — training may help." : "Low reliability — review classification criteria."}
            </InfoBanner>
            <div className="text-xs text-gray-400 pt-1 leading-tight border-t border-gray-200 mt-1">
              <p>&gt; 0.81 Almost Perfect</p><p>0.61–0.80 Substantial</p>
              <p>0.41–0.60 Moderate</p><p>0.21–0.40 Fair</p><p>0.00–0.20 Slight</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 panel min-h-0 relative">
        {data ? (
          <>
          <PlotExporter plotRef={kappaMatrixRef} title="Kappa_Confusion_Matrix" />
          <Plot
            data={[{
              type: "heatmap",
              z: data.confusion_matrix,
              x: data.labels.map((l: string) => `Rater2: ${l}`),
              y: data.labels.map((l: string) => `Rater1: ${l}`),
              colorscale: [[0, "#f9fafb"], [1, _pal()[0]]],
              showscale: false,
              text: data.confusion_matrix.map((row: number[]) => row.map((v: number) => String(v))),
              texttemplate: "%{text}",
              hovertemplate: "Rater1=%{y}<br>Rater2=%{x}<br>Count=%{z}<extra></extra>",
            }]}
            layout={{
              ...PLOT_BG,
              autosize: true,
              title: { text: "Confusion Matrix", font: { color: "#374151", size: 13 } },
              xaxis: { side: "bottom", showgrid: showGrid, gridcolor: "#e5e7eb", zeroline: false },
              yaxis: { showgrid: showGrid, gridcolor: "#e5e7eb", zeroline: false },
              margin: { t: 50, r: 20, b: 80, l: 100 },
            }}
            onInitialized={(_: object, gd: HTMLElement) => { kappaMatrixRef.current = gd; }}
            onUpdate={(_: object, gd: HTMLElement)      => { kappaMatrixRef.current = gd; }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
          </>
        ) : (
          <div className="h-full flex items-center justify-center text-gray-400">
            Select two rater columns and click Compute
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function CorrelationPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;

  const numColumns = session.columns
    .filter((c) => c.kind === "numeric")
    .map((c) => c.name);
  const allColumns = session.columns.map((c) => c.name);

  const [activeTab, setActiveTab] = useState<Tab>("Pairwise");

  return (
    <div className="flex flex-col h-full gap-3">
      <div className="flex gap-1 flex-shrink-0">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
              activeTab === t
                ? "bg-indigo-600 text-white"
                : "bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-700"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0">
        {activeTab === "Pairwise"  && <PairwiseTab sessionId={session.session_id} columns={numColumns} />}
        {activeTab === "Matrix"    && <MatrixTab   sessionId={session.session_id} columns={numColumns} />}
        {activeTab === "ICC"       && <ICCTab      sessionId={session.session_id} columns={allColumns} />}
        {activeTab === "Cohen's κ" && <KappaTab    sessionId={session.session_id} columns={allColumns} />}
      </div>
    </div>
  );
}
