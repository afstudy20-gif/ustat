import { useState } from "react";
import Plot from "../PlotComponent";
import { runPower } from "../api";

// ── Types ───────────────────────────────────────────────────────────────────

type TestId    = "t_two" | "t_one" | "anova" | "correlation" | "proportion" | "chi2";
type SolveFor  = "n" | "power" | "effect_size";

interface CurvePoint { n: number; power: number }
interface PowerResult { result: number | null; label: string; curve: CurvePoint[] }

// ── Constants ───────────────────────────────────────────────────────────────

const TESTS: {
  id: TestId; label: string; effectLabel: string;
  hasRatio: boolean; hasGroups: boolean; hasTails: boolean; isProportions: boolean;
}[] = [
  { id: "t_two",       label: "Independent samples t-test", effectLabel: "Cohen's d", hasRatio: true,  hasGroups: false, hasTails: true,  isProportions: false },
  { id: "t_one",       label: "One-sample / Paired t-test", effectLabel: "Cohen's d", hasRatio: false, hasGroups: false, hasTails: true,  isProportions: false },
  { id: "anova",       label: "One-way ANOVA",              effectLabel: "Cohen's f", hasRatio: false, hasGroups: true,  hasTails: false, isProportions: false },
  { id: "correlation", label: "Pearson correlation",        effectLabel: "Pearson r", hasRatio: false, hasGroups: false, hasTails: true,  isProportions: false },
  { id: "proportion",  label: "Two proportions (z-test)",   effectLabel: "p₁ / p₂",  hasRatio: true,  hasGroups: false, hasTails: true,  isProportions: true  },
  { id: "chi2",        label: "Chi-square",                 effectLabel: "Cohen's w", hasRatio: false, hasGroups: true,  hasTails: false, isProportions: false },
];

// Small / Medium / Large presets per Cohen (1988)
const PRESETS: Record<string, [number, number, number]> = {
  t_two:       [0.20, 0.50, 0.80],
  t_one:       [0.20, 0.50, 0.80],
  anova:       [0.10, 0.25, 0.40],
  correlation: [0.10, 0.30, 0.50],
  chi2:        [0.10, 0.30, 0.50],
};

const COHEN_TABLE = [
  { effect: "Cohen's d (t-test)",  small: 0.20, medium: 0.50, large: 0.80 },
  { effect: "Cohen's f (ANOVA)",   small: 0.10, medium: 0.25, large: 0.40 },
  { effect: "Pearson r",           small: 0.10, medium: 0.30, large: 0.50 },
  { effect: "Cohen's w (χ²)",      small: 0.10, medium: 0.30, large: 0.50 },
  { effect: "Cohen's h (props)",   small: 0.20, medium: 0.50, large: 0.80 },
];

const BASE_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor:  "#f9fafb",
  font:          { color: "#374151", size: 11 },
  margin:        { t: 20, r: 20, b: 48, l: 56 },
  xaxis:         { gridcolor: "#e5e7eb" },
  yaxis:         { gridcolor: "#e5e7eb", range: [0, 1.05], title: { text: "Power (1−β)" } },
};

// ── Component ───────────────────────────────────────────────────────────────

export default function PowerPanel() {
  const [test,      setTest]      = useState<TestId>("t_two");
  const [solveFor,  setSolveFor]  = useState<SolveFor>("n");
  const [alpha,     setAlpha]     = useState("0.05");
  const [power,     setPower]     = useState("0.80");
  const [effectSize,setEffectSize]= useState("0.50");
  const [n,         setN]         = useState("64");
  const [tails,     setTails]     = useState("2");
  const [ratio,     setRatio]     = useState("1.0");
  const [kGroups,   setKGroups]   = useState("3");
  const [p1,        setP1]        = useState("0.50");
  const [p2,        setP2]        = useState("0.30");

  const [result,  setResult]  = useState<PowerResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const testInfo = TESTS.find((t) => t.id === test)!;
  const presets  = PRESETS[test];

  // ── Calculate ─────────────────────────────────────────────────────────────

  const calculate = async () => {
    setLoading(true); setError(null);
    try {
      const payload: Record<string, unknown> = {
        test, solve_for: solveFor,
        alpha:    parseFloat(alpha)    || 0.05,
        tails:    parseInt(tails)      || 2,
        k_groups: parseInt(kGroups)    || 3,
        ratio:    parseFloat(ratio)    || 1.0,
        p1:       parseFloat(p1),
        p2:       parseFloat(p2),
      };
      if (solveFor !== "n")            payload.n            = parseInt(n);
      if (solveFor !== "power")        payload.power        = parseFloat(power);
      if (solveFor !== "effect_size" && !testInfo.isProportions)
                                       payload.effect_size  = parseFloat(effectSize);

      const res = await runPower(payload);
      setResult(res.data);

      // Echo solved value back into the input
      if (res.data.result != null) {
        if (solveFor === "n")            setN(String(Math.ceil(res.data.result)));
        else if (solveFor === "power")   setPower(res.data.result.toFixed(4));
        else                             setEffectSize(res.data.result.toFixed(4));
      }
    } catch (e: any) {
      const msg = e.response?.data?.detail;
      setError(typeof msg === "string" ? msg : (e.message ?? "Calculation failed"));
    } finally { setLoading(false); }
  };

  const switchTest = (id: TestId) => { setTest(id); setResult(null); setError(null); };
  const switchSolve = (s: SolveFor) => { setSolveFor(s); setResult(null); setError(null); };

  // ── Plot data ─────────────────────────────────────────────────────────────

  const currentN = solveFor === "n" && result?.result
    ? Math.ceil(result.result)
    : parseInt(n) || 0;

  const xLabel = (testInfo.hasRatio || testInfo.hasGroups) ? "n per group" : "Sample size (n)";

  const plotTraces: object[] = result?.curve.length ? [
    {
      type: "scatter", mode: "lines",
      x: result.curve.map((p) => p.n),
      y: result.curve.map((p) => p.power),
      line: { color: "#6366f1", width: 2.5 },
      name: "Power",
      hovertemplate: "n = %{x}<br>Power = %{y:.3f}<extra></extra>",
    },
    {   // 80% reference line
      type: "scatter", mode: "lines",
      x: [result.curve[0]?.n ?? 4, result.curve[result.curve.length - 1]?.n ?? 200],
      y: [0.80, 0.80],
      line: { color: "#dc2626", width: 1.5, dash: "dash" },
      name: "0.80 threshold",
      hoverinfo: "skip",
    },
  ] : [];

  // Current-n dot
  if (result?.curve.length && currentN) {
    const pt = [...result.curve].reverse().find((p) => p.n <= currentN) ?? result.curve[0];
    if (pt) {
      plotTraces.push({
        type: "scatter", mode: "markers",
        x: [pt.n], y: [pt.power],
        marker: { color: "#6366f1", size: 10, line: { color: "#fff", width: 2 } },
        name: `n = ${pt.n}`,
        hovertemplate: `n = ${pt.n}<br>Power = ${pt.power.toFixed(3)}<extra></extra>`,
      });
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  const inputCls = (disabled: boolean) =>
    `w-full rounded-lg border px-2.5 py-1.5 text-sm transition-colors focus:outline-none focus:border-indigo-400 ${
      disabled ? "bg-indigo-50 border-indigo-300 text-indigo-700 font-semibold cursor-not-allowed"
               : "bg-white border-gray-300 text-gray-900"}`;

  const chipCls = (active: boolean) =>
    `flex-1 text-xs py-1 rounded border transition-colors select-none cursor-pointer ${
      active ? "bg-indigo-100 text-indigo-700 border-indigo-300 font-medium"
             : "text-gray-500 border-gray-300 hover:bg-gray-50"}`;

  return (
    <div className="flex gap-4">

      {/* ── Left sidebar ──────────────────────────────────────────────────── */}
      <div className="w-72 flex-shrink-0 space-y-3">

        {/* Test picker */}
        <div className="panel space-y-1.5">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Statistical Test</h3>
          {TESTS.map((t) => (
            <label key={t.id} className="flex items-center gap-2 cursor-pointer group">
              <input type="radio" name="power-test" value={t.id}
                checked={test === t.id} onChange={() => switchTest(t.id)}
                className="accent-indigo-500" />
              <span className={`text-sm transition-colors ${test === t.id ? "text-indigo-700 font-medium" : "text-gray-700 group-hover:text-gray-900"}`}>
                {t.label}
              </span>
            </label>
          ))}
        </div>

        {/* Solve for */}
        <div className="panel space-y-1.5">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Solve For</h3>
          {([["n", "Sample size (n)"], ["power", "Power (1−β)"], ["effect_size", "Effect size"]] as const).map(([v, l]) => (
            <label key={v} className="flex items-center gap-2 cursor-pointer group">
              <input type="radio" name="solve-for" value={v}
                checked={solveFor === v} onChange={() => switchSolve(v)}
                className="accent-indigo-500" />
              <span className={`text-sm transition-colors ${solveFor === v ? "text-indigo-700 font-medium" : "text-gray-700 group-hover:text-gray-900"}`}>
                {l}
              </span>
            </label>
          ))}
        </div>

        {/* Parameters */}
        <div className="panel space-y-3">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Parameters</h3>

          {/* Effect size / proportions */}
          {testInfo.isProportions ? (
            <div className="space-y-2">
              {[["Proportion 1 (p₁)", p1, setP1], ["Proportion 2 (p₂)", p2, setP2]].map(([label, val, setter]) => (
                <div key={label as string}>
                  <label className="text-xs text-gray-400 block mb-1">{label as string}</label>
                  <input type="number" min="0.01" max="0.99" step="0.01"
                    className={inputCls(solveFor === "effect_size")}
                    value={val as string}
                    disabled={solveFor === "effect_size"}
                    onChange={(e) => (setter as (v: string) => void)(e.target.value)} />
                </div>
              ))}
            </div>
          ) : (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Effect size ({testInfo.effectLabel})</label>
              <input type="number" min="0.001" step="0.01"
                className={inputCls(solveFor === "effect_size")}
                value={effectSize}
                disabled={solveFor === "effect_size"}
                onChange={(e) => setEffectSize(e.target.value)} />
              {presets && solveFor !== "effect_size" && (
                <div className="flex gap-1 mt-1.5">
                  {(["Small", "Medium", "Large"] as const).map((s, i) => (
                    <button key={s} onClick={() => setEffectSize(String(presets[i]))}
                      className={chipCls(parseFloat(effectSize) === presets[i])}>
                      {s}<br /><span className="text-[9px] opacity-70">{presets[i]}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Sample size */}
          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Sample size (n{testInfo.hasRatio || testInfo.hasGroups ? " per group" : ""})
            </label>
            <input type="number" min="4" step="1"
              className={inputCls(solveFor === "n")}
              value={n} disabled={solveFor === "n"}
              onChange={(e) => setN(e.target.value)} />
          </div>

          {/* Power */}
          <div>
            <label className="text-xs text-gray-400 block mb-1">Power (1−β)</label>
            <input type="number" min="0.01" max="0.999" step="0.01"
              className={inputCls(solveFor === "power")}
              value={power} disabled={solveFor === "power"}
              onChange={(e) => setPower(e.target.value)} />
          </div>

          {/* Alpha */}
          <div>
            <label className="text-xs text-gray-400 block mb-1">Significance level (α)</label>
            <div className="flex gap-1">
              {["0.01", "0.05", "0.10"].map((v) => (
                <button key={v} onClick={() => setAlpha(v)} className={chipCls(alpha === v)}>{v}</button>
              ))}
            </div>
          </div>

          {/* Tails */}
          {testInfo.hasTails && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Tails</label>
              <div className="flex gap-1">
                {[["2", "Two-tailed"], ["1", "One-tailed"]].map(([v, l]) => (
                  <button key={v} onClick={() => setTails(v)} className={chipCls(tails === v)}>{l}</button>
                ))}
              </div>
            </div>
          )}

          {/* n₂/n₁ ratio */}
          {testInfo.hasRatio && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Allocation ratio (n₂ / n₁)</label>
              <input type="number" min="0.1" step="0.1"
                className="w-full rounded-lg border border-gray-300 px-2.5 py-1.5 text-sm bg-white focus:outline-none focus:border-indigo-400"
                value={ratio} onChange={(e) => setRatio(e.target.value)} />
            </div>
          )}

          {/* Groups / bins */}
          {testInfo.hasGroups && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">
                {test === "anova" ? "Number of groups (k)" : "Number of categories"}
              </label>
              <input type="number" min="2" step="1"
                className="w-full rounded-lg border border-gray-300 px-2.5 py-1.5 text-sm bg-white focus:outline-none focus:border-indigo-400"
                value={kGroups} onChange={(e) => setKGroups(e.target.value)} />
            </div>
          )}
        </div>

        <button className="btn-primary w-full" onClick={calculate} disabled={loading}>
          {loading ? "Calculating…" : "⚡ Calculate"}
        </button>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-2.5 text-xs text-red-600">{error}</div>
        )}
      </div>

      {/* ── Right panel ───────────────────────────────────────────────────── */}
      <div className="flex-1 space-y-4 min-w-0">

        {result ? (
          <>
            {/* Result card */}
            <div className="panel">
              <div className="flex items-start gap-6">
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-gray-400 mb-0.5">Result</p>
                  <p className="text-3xl font-bold text-indigo-600 leading-none">
                    {solveFor === "n"
                      ? Math.ceil(result.result ?? 0).toLocaleString()
                      : result.result?.toFixed(4)}
                  </p>
                  <p className="text-sm text-gray-600 mt-2">{result.label}</p>
                </div>

                {/* Param summary */}
                <div className="text-xs text-gray-400 space-y-0.5 text-right flex-shrink-0 bg-gray-50 rounded-lg px-3 py-2 border border-gray-200">
                  <p><span className="text-gray-600 font-medium">α</span> = {alpha}</p>
                  {solveFor !== "power" && <p><span className="text-gray-600 font-medium">Power</span> = {power}</p>}
                  {solveFor !== "n" && <p><span className="text-gray-600 font-medium">n</span> = {n}</p>}
                  {!testInfo.isProportions && solveFor !== "effect_size" &&
                    <p><span className="text-gray-600 font-medium">ES</span> = {effectSize}</p>}
                  {testInfo.isProportions && <>
                    <p><span className="text-gray-600 font-medium">p₁</span> = {p1}</p>
                    <p><span className="text-gray-600 font-medium">p₂</span> = {p2}</p>
                  </>}
                </div>
              </div>

              {/* Power bar (only when solving for power) */}
              {solveFor === "power" && result.result != null && (
                <div className="mt-4">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">0%</span>
                    <span className={`font-semibold ${result.result >= 0.80 ? "text-emerald-600" : "text-orange-500"}`}>
                      {(result.result * 100).toFixed(1)}%
                      &nbsp;{result.result >= 0.80 ? "✓ Adequate power" : "✗ Underpowered"}
                    </span>
                    <span className="text-gray-400">100%</span>
                  </div>
                  <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${result.result >= 0.80 ? "bg-emerald-500" : "bg-orange-400"}`}
                      style={{ width: `${Math.min(100, result.result * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <div className="relative h-0">
                    <div className="absolute top-0 border-l-2 border-dashed border-gray-400 h-3" style={{ left: "80%" }} />
                  </div>
                  <p className="text-[10px] text-gray-400 mt-1 text-right" style={{ marginRight: "18%" }}>80%</p>
                </div>
              )}
            </div>

            {/* Power curve */}
            {result.curve.length > 0 && (
              <div className="panel">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Power Curve</h4>
                <Plot
                  data={plotTraces as any}
                  layout={{
                    ...BASE_LAYOUT,
                    autosize: true, height: 300,
                    xaxis: { ...BASE_LAYOUT.xaxis, title: { text: xLabel } },
                    legend: { orientation: "h", y: -0.25, font: { size: 11 } },
                    shapes: currentN ? [{
                      type: "line", xref: "x", yref: "y",
                      x0: currentN, x1: currentN, y0: 0, y1: 1,
                      line: { color: "#6366f1", width: 1.5, dash: "dot" },
                    }] : [],
                  } as any}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                  config={{ responsive: true, displaylogo: false }}
                />
              </div>
            )}

            {/* Cohen's conventions reference */}
            <div className="panel">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Cohen's Conventions</h4>
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-400">
                    <th className="text-left py-1.5 font-medium">Effect measure</th>
                    <th className="text-right py-1.5 font-medium">Small</th>
                    <th className="text-right py-1.5 font-medium">Medium</th>
                    <th className="text-right py-1.5 font-medium">Large</th>
                  </tr>
                </thead>
                <tbody>
                  {COHEN_TABLE.map((row) => (
                    <tr key={row.effect} className="border-b border-gray-100">
                      <td className="py-1.5 text-gray-600">{row.effect}</td>
                      <td className="text-right py-1.5 font-mono text-gray-700">{row.small}</td>
                      <td className="text-right py-1.5 font-mono text-gray-700">{row.medium}</td>
                      <td className="text-right py-1.5 font-mono text-gray-700">{row.large}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-[10px] text-gray-400 mt-2">
                Source: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.)
              </p>
            </div>
          </>
        ) : (
          /* Empty state */
          <div className="flex flex-col items-center justify-center gap-3 py-24 text-gray-400">
            <div className="w-16 h-16 rounded-full bg-indigo-50 flex items-center justify-center text-2xl">⚡</div>
            <p className="text-sm font-medium text-gray-600">Configure parameters and click Calculate</p>
            <ul className="text-xs space-y-1 text-center">
              <li>① Choose a statistical test</li>
              <li>② Select what to solve for (n, power, or effect size)</li>
              <li>③ Fill in the remaining parameters</li>
              <li>④ Click <strong>Calculate</strong></li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
