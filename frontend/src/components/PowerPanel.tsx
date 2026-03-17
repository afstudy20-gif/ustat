import { useState } from "react";
import Plot from "../PlotComponent";
import { runPower } from "../api";

// ── Types ───────────────────────────────────────────────────────────────────

type TestId   = "t_two" | "t_one" | "anova" | "correlation" | "proportion" | "chi2";
type SolveFor = "n" | "power" | "effect_size";

interface CurvePoint { n: number; power: number }
interface PowerResult { result: number | null; label: string; curve: CurvePoint[] }

// ── Tooltip helper ───────────────────────────────────────────────────────────

function Tip({ text, wide }: { text: string; wide?: boolean }) {
  return (
    <span className="relative group inline-block ml-1 align-middle">
      <span className="text-[10px] text-gray-300 hover:text-indigo-400 cursor-help border border-gray-200 hover:border-indigo-300 rounded-full w-3.5 h-3.5 inline-flex items-center justify-center transition-colors leading-none select-none">?</span>
      <div className={`absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 ${wide ? "w-72" : "w-56"} bg-gray-900 text-white text-[11px] leading-relaxed rounded-lg px-3 py-2.5 shadow-xl pointer-events-none`}>
        {text}
        <div className="absolute top-full left-4 border-4 border-transparent border-t-gray-900" />
      </div>
    </span>
  );
}

// ── Constants ────────────────────────────────────────────────────────────────

const TESTS: {
  id: TestId; label: string; desc: string; effectLabel: string;
  hasRatio: boolean; hasGroups: boolean; hasTails: boolean; isProportions: boolean;
}[] = [
  {
    id: "t_two", label: "Independent t-test",
    desc: "Compare the average of two separate groups — e.g. treatment vs. control.",
    effectLabel: "Cohen's d", hasRatio: true, hasGroups: false, hasTails: true, isProportions: false,
  },
  {
    id: "t_one", label: "One-sample / Paired t-test",
    desc: "Compare one group to a fixed value, or paired measurements (before vs. after).",
    effectLabel: "Cohen's d", hasRatio: false, hasGroups: false, hasTails: true, isProportions: false,
  },
  {
    id: "anova", label: "One-way ANOVA",
    desc: "Compare means across three or more groups at the same time.",
    effectLabel: "Cohen's f", hasRatio: false, hasGroups: true, hasTails: false, isProportions: false,
  },
  {
    id: "correlation", label: "Pearson correlation",
    desc: "Detect a linear relationship between two continuous variables.",
    effectLabel: "Pearson r", hasRatio: false, hasGroups: false, hasTails: true, isProportions: false,
  },
  {
    id: "proportion", label: "Two proportions (z-test)",
    desc: "Compare event rates or percentages between two groups (e.g. recovery rates).",
    effectLabel: "Cohen's h", hasRatio: true, hasGroups: false, hasTails: true, isProportions: true,
  },
  {
    id: "chi2", label: "Chi-square test",
    desc: "Test whether two categorical variables are associated (e.g. gender × diagnosis).",
    effectLabel: "Cohen's w", hasRatio: false, hasGroups: true, hasTails: false, isProportions: false,
  },
];

const SOLVE_OPTS: { id: SolveFor; label: string; desc: string }[] = [
  { id: "n",           label: "Sample size (n)",  desc: "How many participants do I need?" },
  { id: "power",       label: "Power (1−β)",       desc: "What are my chances of finding a real effect?" },
  { id: "effect_size", label: "Effect size",        desc: "What is the smallest effect I can detect?" },
];

const PRESETS: Record<string, [number, number, number]> = {
  t_two: [0.20, 0.50, 0.80], t_one: [0.20, 0.50, 0.80],
  anova: [0.10, 0.25, 0.40], correlation: [0.10, 0.30, 0.50], chi2: [0.10, 0.30, 0.50],
};

const COHEN_TABLE = [
  { effect: "Cohen's d  (t-tests)",  small: 0.20, medium: 0.50, large: 0.80 },
  { effect: "Cohen's f  (ANOVA)",    small: 0.10, medium: 0.25, large: 0.40 },
  { effect: "Pearson r  (corr.)",    small: 0.10, medium: 0.30, large: 0.50 },
  { effect: "Cohen's w  (χ²)",       small: 0.10, medium: 0.30, large: 0.50 },
  { effect: "Cohen's h  (props.)",   small: 0.20, medium: 0.50, large: 0.80 },
];

const TIP_TEXTS = {
  alpha: "The false-positive rate — the probability of concluding there is an effect when there isn't one. α = 0.05 (5%) is the standard in most fields. A stricter value (0.01) reduces false alarms but requires a larger sample.",
  power: "The probability of detecting a real effect if one truly exists. Power = 0.80 means an 80% chance of getting a significant result. The 80% threshold is the widely accepted minimum; 90% is preferred for confirmatory studies.",
  d:     "Cohen's d measures how far apart two group means are in standard-deviation units. d = 0.5 means the means differ by half a standard deviation. If unsure, use the Small/Medium/Large presets from Cohen (1988).",
  f:     "Cohen's f measures how spread out group means are in ANOVA, relative to within-group variability. f = 0.25 is a medium effect — noticeable but not dramatic.",
  r:     "Pearson r is the correlation coefficient (0 = no relationship, 1 = perfect). r = 0.3 means about 9% of variance in one variable is explained by the other.",
  w:     "Cohen's w measures departure from expected cell counts in a chi-square table. The Small/Medium/Large cutoffs are the same as for most other measures.",
  p1p2:  "Enter the expected proportion (probability or percentage) in each group. Example: if 40% of controls recover and you expect 60% in the treatment group, set p₁ = 0.40, p₂ = 0.60. A bigger difference between p₁ and p₂ means a smaller sample is needed.",
  n:     "Number of participants per group. For two-group designs the total N = n × (1 + ratio). Equal group sizes (ratio = 1) give the best efficiency per participant.",
  tails: "Two-tailed: checks for effects in both directions (A > B or A < B) — appropriate for most research questions. One-tailed: checks only one direction and needs fewer participants, but should only be used when you can rule out the opposite direction.",
  ratio: "The ratio of group 2 size to group 1 size. Use 1.0 for equal groups (most efficient). A ratio of 2.0 means group 2 has twice as many participants.",
  groups:"The number of groups being compared. More groups require a larger total sample.",
  cats:  "The number of categories (cells) in the chi-square table. For a 2 × 2 association table, enter 2 (degrees of freedom = 1). For a 3-category variable, enter 3.",
};

const BASE_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor:  "#f9fafb",
  font:   { color: "#374151", size: 11 },
  margin: { t: 20, r: 20, b: 52, l: 56 },
  xaxis:  { gridcolor: "#e5e7eb" },
  yaxis:  { gridcolor: "#e5e7eb", range: [0, 1.05], title: { text: "Power (1−β)" } },
};

// ── Plain-English interpretation ─────────────────────────────────────────────

function plainEnglish(
  test: TestId, solveFor: SolveFor, resultVal: number,
  { alpha, power, effectSize, n, p1, p2, testInfo }: {
    alpha: string; power: string; effectSize: string; n: string;
    p1: string; p2: string;
    testInfo: typeof TESTS[0];
  }
): string {
  const pct   = (x: string | number) => `${(parseFloat(String(x)) * 100).toFixed(0)}%`;
  const es    = parseFloat(effectSize);
  const presets = PRESETS[test];
  const size  = presets
    ? (es <= presets[0] ? "small" : es <= presets[1] ? "small-to-medium" : es < presets[2] ? "medium" : "large")
    : "specified";
  const perGrp = testInfo.hasRatio || testInfo.hasGroups;

  const effectDesc = testInfo.isProportions
    ? `a difference of ${parseFloat(p1) * 100}% vs ${parseFloat(p2) * 100}%`
    : `a ${size} effect (${testInfo.effectLabel} = ${es})`;

  if (solveFor === "n") {
    const nCeil = Math.ceil(resultVal);
    const perStr = perGrp ? " per group" : "";
    return `To detect ${effectDesc} with ${pct(power)} probability at α = ${alpha}, you need ${nCeil}${perStr}. If the true effect is this size, you have a ${pct(power)} chance of getting a statistically significant result.`;
  }
  if (solveFor === "power") {
    const pctVal = (resultVal * 100).toFixed(1);
    const perStr = perGrp ? ` per group` : "";
    if (resultVal >= 0.80)
      return `With ${n}${perStr}, you have a ${pctVal}% chance of detecting ${effectDesc} — above the 80% threshold. The study is adequately powered.`;
    return `With only ${n}${perStr}, you have a ${pctVal}% chance of detecting ${effectDesc}. This is below the recommended 80% — increasing sample size will reduce the risk of a false negative.`;
  }
  // effect_size
  const perStr = perGrp ? ` per group` : "";
  return `With ${n}${perStr} and ${pct(power)} power at α = ${alpha}, effects smaller than ${resultVal.toFixed(3)} (${testInfo.effectLabel}) will likely go undetected. If you expect a smaller true effect, you need more participants.`;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function PowerPanel() {
  const [test,       setTest]       = useState<TestId>("t_two");
  const [solveFor,   setSolveFor]   = useState<SolveFor>("n");
  const [alpha,      setAlpha]      = useState("0.05");
  const [power,      setPower]      = useState("0.80");
  const [effectSize, setEffectSize] = useState("0.50");
  const [n,          setN]          = useState("64");
  const [tails,      setTails]      = useState("2");
  const [ratio,      setRatio]      = useState("1.0");
  const [kGroups,    setKGroups]    = useState("3");
  const [p1,         setP1]         = useState("0.50");
  const [p2,         setP2]         = useState("0.30");

  const [result,  setResult]  = useState<PowerResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const testInfo = TESTS.find((t) => t.id === test)!;
  const presets  = PRESETS[test];

  const effectTip: Record<TestId, string> = {
    t_two: TIP_TEXTS.d, t_one: TIP_TEXTS.d,
    anova: TIP_TEXTS.f, correlation: TIP_TEXTS.r,
    proportion: TIP_TEXTS.p1p2, chi2: TIP_TEXTS.w,
  };

  // ── Calculate ───────────────────────────────────────────────────────────────

  const calculate = async () => {
    setLoading(true); setError(null);
    try {
      const payload: Record<string, unknown> = {
        test, solve_for: solveFor,
        alpha:    parseFloat(alpha)   || 0.05,
        tails:    parseInt(tails)     || 2,
        k_groups: parseInt(kGroups)   || 3,
        ratio:    parseFloat(ratio)   || 1.0,
        p1:       parseFloat(p1),
        p2:       parseFloat(p2),
      };
      if (solveFor !== "n")            payload.n           = parseInt(n);
      if (solveFor !== "power")        payload.power       = parseFloat(power);
      if (solveFor !== "effect_size" && !testInfo.isProportions)
                                       payload.effect_size = parseFloat(effectSize);
      const res = await runPower(payload);
      setResult(res.data);
      if (res.data.result != null) {
        if (solveFor === "n")          setN(String(Math.ceil(res.data.result)));
        else if (solveFor === "power") setPower(res.data.result.toFixed(4));
        else                           setEffectSize(res.data.result.toFixed(4));
      }
    } catch (e: any) {
      const msg = e.response?.data?.detail;
      setError(typeof msg === "string" ? msg : (e.message ?? "Calculation failed"));
    } finally { setLoading(false); }
  };

  const switchTest  = (id: TestId)   => { setTest(id);     setResult(null); setError(null); };
  const switchSolve = (s: SolveFor)  => { setSolveFor(s);  setResult(null); setError(null); };

  // ── Plot ────────────────────────────────────────────────────────────────────

  const currentN = solveFor === "n" && result?.result
    ? Math.ceil(result.result)
    : parseInt(n) || 0;
  const xLabel = (testInfo.hasRatio || testInfo.hasGroups) ? "n per group" : "Sample size (n)";

  const plotTraces: object[] = result?.curve.length ? [
    {
      type: "scatter", mode: "lines",
      x: result.curve.map((p) => p.n),
      y: result.curve.map((p) => p.power),
      line: { color: "#6366f1", width: 2.5 }, name: "Power",
      hovertemplate: "n = %{x}<br>Power = %{y:.3f}<extra></extra>",
    },
    {
      type: "scatter", mode: "lines",
      x: [result.curve[0]?.n ?? 4, result.curve[result.curve.length - 1]?.n ?? 200],
      y: [0.80, 0.80],
      line: { color: "#dc2626", width: 1.5, dash: "dash" },
      name: "80% threshold", hoverinfo: "skip",
    },
  ] : [];
  if (result?.curve.length && currentN) {
    const pt = [...result.curve].reverse().find((p) => p.n <= currentN) ?? result.curve[0];
    if (pt) plotTraces.push({
      type: "scatter", mode: "markers",
      x: [pt.n], y: [pt.power],
      marker: { color: "#6366f1", size: 10, line: { color: "#fff", width: 2 } },
      name: `n = ${pt.n}`,
      hovertemplate: `n = ${pt.n}<br>Power = ${pt.power.toFixed(3)}<extra></extra>`,
    });
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  const inputCls = (disabled: boolean) =>
    `w-full rounded-lg border px-2.5 py-1.5 text-sm transition-colors focus:outline-none focus:border-indigo-400 ${
      disabled
        ? "bg-indigo-50 border-indigo-300 text-indigo-700 font-semibold cursor-not-allowed"
        : "bg-white border-gray-300 text-gray-900"}`;

  const chipCls = (active: boolean) =>
    `flex-1 text-xs py-1 rounded border transition-colors select-none cursor-pointer ${
      active ? "bg-indigo-100 text-indigo-700 border-indigo-300 font-medium"
             : "text-gray-500 border-gray-300 hover:bg-gray-50"}`;

  const LabelTip = ({ children, tip, wide }: { children: string; tip: string; wide?: boolean }) => (
    <label className="text-xs text-gray-400 flex items-center mb-1">
      {children}<Tip text={tip} wide={wide} />
    </label>
  );

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="flex gap-4">

      {/* ── Left sidebar ──────────────────────────────────────────────────── */}
      <div className="w-[17rem] flex-shrink-0 space-y-3">

        {/* Test picker */}
        <div className="panel space-y-1">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Statistical Test</h3>
          {TESTS.map((t) => (
            <label key={t.id} className="flex items-start gap-2 cursor-pointer group py-0.5">
              <input type="radio" name="power-test" value={t.id}
                checked={test === t.id} onChange={() => switchTest(t.id)}
                className="accent-indigo-500 mt-0.5 flex-shrink-0" />
              <div>
                <span className={`text-sm leading-tight transition-colors ${test === t.id ? "text-indigo-700 font-medium" : "text-gray-700 group-hover:text-gray-900"}`}>
                  {t.label}
                </span>
                {test === t.id && (
                  <p className="text-[10px] text-gray-400 leading-snug mt-0.5">{t.desc}</p>
                )}
              </div>
            </label>
          ))}
        </div>

        {/* Solve for */}
        <div className="panel space-y-1">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Solve For</h3>
          {SOLVE_OPTS.map(({ id, label, desc }) => (
            <label key={id} className="flex items-start gap-2 cursor-pointer group py-0.5">
              <input type="radio" name="solve-for" value={id}
                checked={solveFor === id} onChange={() => switchSolve(id)}
                className="accent-indigo-500 mt-0.5 flex-shrink-0" />
              <div>
                <span className={`text-sm leading-tight transition-colors ${solveFor === id ? "text-indigo-700 font-medium" : "text-gray-700 group-hover:text-gray-900"}`}>
                  {label}
                </span>
                {solveFor === id && (
                  <p className="text-[10px] text-gray-400 leading-snug mt-0.5 italic">{desc}</p>
                )}
              </div>
            </label>
          ))}
        </div>

        {/* Parameters */}
        <div className="panel space-y-3">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Parameters</h3>

          {/* Effect size */}
          {testInfo.isProportions ? (
            <div className="space-y-2">
              <LabelTip tip={TIP_TEXTS.p1p2} wide>Proportions (p₁ and p₂)</LabelTip>
              <div className="flex gap-2">
                {([["p₁", p1, setP1], ["p₂", p2, setP2]] as const).map(([lbl, val, setter]) => (
                  <div key={lbl} className="flex-1">
                    <p className="text-[10px] text-gray-400 mb-0.5">{lbl}</p>
                    <input type="number" min="0.01" max="0.99" step="0.01"
                      className={inputCls(solveFor === "effect_size")}
                      value={val} disabled={solveFor === "effect_size"}
                      onChange={(e) => setter(e.target.value)} />
                  </div>
                ))}
              </div>
              {solveFor !== "effect_size" && (
                <p className="text-[10px] text-gray-400">
                  Cohen's h = {Math.abs(2 * Math.asin(Math.sqrt(parseFloat(p1) || 0)) - 2 * Math.asin(Math.sqrt(parseFloat(p2) || 0))).toFixed(3)}
                </p>
              )}
            </div>
          ) : (
            <div>
              <LabelTip tip={effectTip[test]}>{`Effect size (${testInfo.effectLabel})`}</LabelTip>
              <input type="number" min="0.001" step="0.01"
                className={inputCls(solveFor === "effect_size")}
                value={effectSize} disabled={solveFor === "effect_size"}
                onChange={(e) => setEffectSize(e.target.value)} />
              {presets && solveFor !== "effect_size" && (
                <div className="flex gap-1 mt-1.5">
                  {(["Small", "Medium", "Large"] as const).map((s, i) => (
                    <button key={s} onClick={() => setEffectSize(String(presets[i]))}
                      className={chipCls(parseFloat(effectSize) === presets[i])}>
                      {s}<br /><span className="text-[9px] opacity-60">{presets[i]}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* n */}
          <div>
            <LabelTip tip={TIP_TEXTS.n}>
              {`Sample size (n${testInfo.hasRatio || testInfo.hasGroups ? " per group" : ""})`}
            </LabelTip>
            <input type="number" min="4" step="1"
              className={inputCls(solveFor === "n")}
              value={n} disabled={solveFor === "n"}
              onChange={(e) => setN(e.target.value)} />
          </div>

          {/* Power */}
          <div>
            <LabelTip tip={TIP_TEXTS.power}>Power (1−β)</LabelTip>
            <input type="number" min="0.01" max="0.999" step="0.01"
              className={inputCls(solveFor === "power")}
              value={power} disabled={solveFor === "power"}
              onChange={(e) => setPower(e.target.value)} />
          </div>

          {/* Alpha */}
          <div>
            <LabelTip tip={TIP_TEXTS.alpha}>Significance level (α)</LabelTip>
            <div className="flex gap-1">
              {["0.01", "0.05", "0.10"].map((v) => (
                <button key={v} onClick={() => setAlpha(v)} className={chipCls(alpha === v)}>{v}</button>
              ))}
            </div>
          </div>

          {/* Tails */}
          {testInfo.hasTails && (
            <div>
              <LabelTip tip={TIP_TEXTS.tails} wide>Tails</LabelTip>
              <div className="flex gap-1">
                {[["2", "Two-tailed"], ["1", "One-tailed"]].map(([v, l]) => (
                  <button key={v} onClick={() => setTails(v)} className={chipCls(tails === v)}>{l}</button>
                ))}
              </div>
            </div>
          )}

          {/* Ratio */}
          {testInfo.hasRatio && (
            <div>
              <LabelTip tip={TIP_TEXTS.ratio}>Allocation ratio (n₂ / n₁)</LabelTip>
              <input type="number" min="0.1" step="0.1"
                className="w-full rounded-lg border border-gray-300 px-2.5 py-1.5 text-sm bg-white focus:outline-none focus:border-indigo-400"
                value={ratio} onChange={(e) => setRatio(e.target.value)} />
            </div>
          )}

          {/* Groups / bins */}
          {testInfo.hasGroups && (
            <div>
              <LabelTip tip={test === "anova" ? TIP_TEXTS.groups : TIP_TEXTS.cats}>
                {test === "anova" ? "Number of groups (k)" : "Number of categories"}
              </LabelTip>
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
                <div className="text-xs text-gray-400 space-y-0.5 text-right flex-shrink-0 bg-gray-50 rounded-lg px-3 py-2 border border-gray-200">
                  <p><span className="text-gray-600 font-medium">α</span> = {alpha}</p>
                  {solveFor !== "power" && <p><span className="text-gray-600 font-medium">Power</span> = {power}</p>}
                  {solveFor !== "n"     && <p><span className="text-gray-600 font-medium">n</span> = {n}</p>}
                  {!testInfo.isProportions && solveFor !== "effect_size" &&
                    <p><span className="text-gray-600 font-medium">ES</span> = {effectSize}</p>}
                  {testInfo.isProportions && <>
                    <p><span className="text-gray-600 font-medium">p₁</span> = {p1}</p>
                    <p><span className="text-gray-600 font-medium">p₂</span> = {p2}</p>
                  </>}
                </div>
              </div>

              {/* Power bar */}
              {solveFor === "power" && result.result != null && (
                <div className="mt-4">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">0%</span>
                    <span className={`font-semibold ${result.result >= 0.80 ? "text-emerald-600" : "text-orange-500"}`}>
                      {(result.result * 100).toFixed(1)}%
                      &nbsp;{result.result >= 0.80 ? "✓ Adequate" : "✗ Underpowered"}
                    </span>
                    <span className="text-gray-400">100%</span>
                  </div>
                  <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${result.result >= 0.80 ? "bg-emerald-500" : "bg-orange-400"}`}
                      style={{ width: `${Math.min(100, result.result * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <div className="flex justify-end">
                    <span className="text-[10px] text-gray-400 mt-0.5 mr-[18%]">← 80% target</span>
                  </div>
                </div>
              )}
            </div>

            {/* Plain-English interpretation */}
            {result.result != null && (
              <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3 flex gap-3">
                <span className="text-indigo-400 text-lg flex-shrink-0 mt-0.5">💬</span>
                <p className="text-sm text-indigo-800 leading-relaxed">
                  {plainEnglish(test, solveFor, result.result, {
                    alpha, power, effectSize, n, p1, p2, testInfo,
                  })}
                </p>
              </div>
            )}

            {/* Power curve */}
            {result.curve.length > 0 && (
              <div className="panel">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-gray-700">Power Curve</h4>
                  <p className="text-[10px] text-gray-400">
                    Each point shows the power you'd have at that sample size.
                    The red dashed line marks the 80% threshold.
                  </p>
                </div>
                <Plot
                  data={plotTraces as any}
                  layout={{
                    ...BASE_LAYOUT,
                    autosize: true, height: 290,
                    xaxis: { ...BASE_LAYOUT.xaxis, title: { text: xLabel } },
                    legend: { orientation: "h", y: -0.28, font: { size: 11 } },
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

            {/* Cohen's conventions */}
            <div className="panel">
              <h4 className="text-sm font-semibold text-gray-700 mb-1">Cohen's Effect-Size Conventions</h4>
              <p className="text-[10px] text-gray-400 mb-2">
                Use these benchmarks when you have no prior data to estimate the expected effect size.
                Medium effects are the most common assumption for planning studies.
              </p>
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-400">
                    <th className="text-left py-1.5 font-medium">Measure</th>
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
                Cohen, J. (1988). <em>Statistical Power Analysis for the Behavioral Sciences</em> (2nd ed.).
              </p>
            </div>
          </>
        ) : (
          /* ── Empty state: mini-guide ──────────────────────────────────── */
          <div className="space-y-4">
            <div className="flex flex-col items-center gap-2 py-6 text-gray-500">
              <div className="w-14 h-14 rounded-full bg-indigo-50 flex items-center justify-center text-2xl">⚡</div>
              <p className="text-base font-semibold text-gray-700">Power Analysis</p>
              <p className="text-sm text-gray-400 text-center max-w-sm">
                Helps you plan studies so you have enough participants to detect a real effect — and avoid wasting resources on underpowered research.
              </p>
            </div>

            {/* Concept cards */}
            <div className="grid grid-cols-3 gap-3">
              {[
                {
                  icon: "🎯", title: "Statistical Power",
                  body: "The probability of detecting a real effect. Aim for ≥ 80%. A study with 50% power is like flipping a coin — you'd miss half of all real effects.",
                },
                {
                  icon: "📏", title: "Effect Size",
                  body: "How large the true difference is, measured in a standardized unit. If you don't have prior estimates, Cohen's Small / Medium / Large presets are a safe starting point.",
                },
                {
                  icon: "🚦", title: "Significance (α)",
                  body: "The false-positive rate. α = 0.05 means you accept a 5% chance of declaring an effect that doesn't exist. Lower α reduces false positives but requires more participants.",
                },
              ].map(({ icon, title, body }) => (
                <div key={title} className="panel flex flex-col gap-2">
                  <div className="text-2xl">{icon}</div>
                  <p className="text-sm font-semibold text-gray-700">{title}</p>
                  <p className="text-xs text-gray-500 leading-relaxed">{body}</p>
                </div>
              ))}
            </div>

            {/* Quick-start guide */}
            <div className="panel bg-gray-50 border border-gray-200">
              <p className="text-xs font-semibold text-gray-600 mb-2">How to use</p>
              <ol className="text-xs text-gray-500 space-y-1.5 list-none">
                {[
                  ["①", "Choose the test", "that matches your study design (e.g. Independent t-test for two groups)."],
                  ["②", "Pick what to compute", "— usually \"Sample size\" if you're planning a new study."],
                  ["③", "Enter effect size", "— use the Small/Medium/Large buttons if you're unsure. Medium is the most common default."],
                  ["④", "Set Power = 0.80 and α = 0.05", "— the widely accepted standards."],
                  ["⑤", "Click Calculate", "— results include a plain-language explanation and a power curve."],
                ].map(([num, bold, rest]) => (
                  <li key={num as string} className="flex gap-2">
                    <span className="text-indigo-400 font-bold flex-shrink-0">{num}</span>
                    <span><strong className="text-gray-700">{bold}</strong> {rest}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
