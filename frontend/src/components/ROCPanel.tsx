import { useState, useEffect, useRef } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { runROC, runROCCompare, runROCCombined } from "../api";

// ── Constants ─────────────────────────────────────────────────────────────────

const PLOT_LAYOUT: Record<string, unknown> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  font: { color: "#374151", size: 12 },
  margin: { t: 40, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: "#e5e7eb", title: { text: "1 − Specificity (FPR)" }, range: [0, 1] },
  yaxis: { gridcolor: "#e5e7eb", title: { text: "Sensitivity (TPR)" }, range: [0, 1] },
};

const MULTI_PALETTE = [
  "#dc2626","#2563eb","#f59e0b","#16a34a",
  "#7c3aed","#0891b2","#be185d","#92400e",
  "#ea580c","#4f46e5",
];
const ROC_DASHES  = ["solid","dash","dot","dashdot"] as const;
const ROC_WIDTHS  = [1, 1.5, 2, 2.5, 3, 4];

// ── Helpers ───────────────────────────────────────────────────────────────────

const aucColor = (auc: number) =>
  auc >= 0.9 ? "text-green-600" : auc >= 0.8 ? "text-blue-600" : auc >= 0.7 ? "text-amber-600" : "text-red-500";
const aucLabel = (auc: number) =>
  auc >= 0.9 ? "Excellent" : auc >= 0.8 ? "Good" : auc >= 0.7 ? "Fair" : "Poor";
const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;
const fmtP   = (p: number) => p < 0.001 ? "<0.001" : p.toFixed(4);
const fmtAUC = (auc: number, lo?: number, hi?: number) =>
  lo != null && hi != null
    ? `AUC ${auc} (95% CI ${lo}–${hi})`
    : `AUC ${auc}`;

// ── Types ─────────────────────────────────────────────────────────────────────

interface CurveStyle { color: string; width: number; dash: string; }
interface MultiResult {
  col: string;
  auc: number;
  ci_lower?: number;
  ci_upper?: number;
  curve: { fpr: number; tpr: number }[];
  error?: string;
}

const defaultStyle = (i: number): CurveStyle => ({
  color: MULTI_PALETTE[i % MULTI_PALETTE.length],
  width: 2,
  dash: "solid",
});

// ── Metrics block (single mode) ───────────────────────────────────────────────

function MetricsBlock({ m, label }: { m: any; label: string }) {
  return (
    <div className="space-y-0.5">
      <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mt-2 mb-1">{label}</p>
      {[
        ["Cutoff",      m.cutoff],
        ["Sensitivity", m.sensitivity != null ? fmtPct(m.sensitivity) : "—"],
        ["Specificity", m.specificity != null ? fmtPct(m.specificity) : "—"],
        ["PPV",         m.ppv       != null ? fmtPct(m.ppv)       : "—"],
        ["NPV",         m.npv       != null ? fmtPct(m.npv)       : "—"],
        ["Accuracy",    m.accuracy  != null ? fmtPct(m.accuracy)  : "—"],
        ["LR+",         m.lr_pos    != null ? m.lr_pos.toFixed(2) : "—"],
        ["LR−",         m.lr_neg    != null ? m.lr_neg.toFixed(2) : "—"],
        ["Youden J",    m.youden_j  != null ? m.youden_j.toFixed(4) : "—"],
        ["TP", m.tp], ["TN", m.tn], ["FP", m.fp], ["FN", m.fn],
      ].map(([k, v]: any) => (
        <div key={k} className="flex justify-between border-b border-gray-100 py-0.5 text-xs">
          <span className="text-gray-400">{k}</span>
          <span className="text-gray-700 font-mono">{v}</span>
        </div>
      ))}
    </div>
  );
}

// ── StyleRow: one row of color/width/dash controls ────────────────────────────

function StyleRow({
  label, color, width, dash, onColor, onWidth, onDash,
}: {
  label: string; color: string; width: number; dash: string;
  onColor: (v: string) => void; onWidth: (v: number) => void; onDash: (v: string) => void;
}) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: color }} />
      <span className="text-xs text-gray-600 truncate flex-1 min-w-0" title={label}>{label}</span>
      <input type="color" value={color} onChange={(e) => onColor(e.target.value)}
        className="w-6 h-6 rounded cursor-pointer border border-gray-300 flex-shrink-0" />
      <select className="select text-xs py-0 px-1 flex-shrink-0" value={width}
        onChange={(e) => onWidth(Number(e.target.value))}>
        {ROC_WIDTHS.map((w) => <option key={w} value={w}>{w}px</option>)}
      </select>
      <select className="select text-xs py-0 px-1 flex-shrink-0" value={dash}
        onChange={(e) => onDash(e.target.value)}>
        {ROC_DASHES.map((d) => <option key={d} value={d}>{d}</option>)}
      </select>
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function ROCPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;

  const numCols = session.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
  const allCols = session.columns.map((c) => c.name);
  const defaultOutcome = allCols.find((c) => /mortalite|death|event|outcome|binary|status/i.test(c)) ?? allCols[0] ?? "";

  // ── Mode ──
  const [mode, setMode] = useState<"single" | "multi">("single");

  // ── Shared ──
  const [outcomeCol, setOutcomeCol] = useState(defaultOutcome);
  const rocPlotRef = useRef<any>(null);

  // ── Single-curve state ──
  const [scoreCol,     setScoreCol]     = useState(numCols[0] ?? "");
  const [manualCutoff, setManualCutoff] = useState("");
  const [useManual,    setUseManual]    = useState(false);
  const [result,       setResult]       = useState<any>(null);
  const [error,        setError]        = useState<string | null>(null);
  const [loading,      setLoading]      = useState(false);

  const [showCompare, setShowCompare] = useState(false);
  const [scoreCol2,   setScoreCol2]   = useState(numCols[1] ?? numCols[0] ?? "");
  const [cmpResult,   setCmpResult]   = useState<any>(null);
  const [cmpError,    setCmpError]    = useState<string | null>(null);
  const [cmpLoading,  setCmpLoading]  = useState(false);

  const [singleStyle, setSingleStyle] = useState<CurveStyle>({ color: "#6366f1", width: 2.5, dash: "solid" });
  const [chanceStyle, setChanceStyle] = useState<CurveStyle>({ color: "#9ca3af", width: 1,   dash: "dash"  });

  useEffect(() => {
    if (result) { setSingleStyle({ color: "#6366f1", width: 2.5, dash: "solid" }); }
  }, [result?.auc, result?.n]);

  // ── Multi-curve state ──
  const [multiCols,    setMultiCols]    = useState<string[]>([]);
  const [multiResults, setMultiResults] = useState<MultiResult[]>([]);
  const [multiStyles,  setMultiStyles]  = useState<CurveStyle[]>([]);
  const [multiLoading, setMultiLoading] = useState(false);
  const [multiError,   setMultiError]   = useState<string | null>(null);
  const [multiChance,  setMultiChance]  = useState<CurveStyle>({ color: "#9ca3af", width: 1, dash: "dash" });

  // ── Combined model state ──
  const [showCombined,     setShowCombined]     = useState(false);
  const [combinedCols,     setCombinedCols]     = useState<string[]>([]);
  const [combinedName,     setCombinedName]     = useState("Combined Model");
  const [combinedResult,   setCombinedResult]   = useState<MultiResult | null>(null);
  const [combinedStyle,    setCombinedStyle]    = useState<CurveStyle>({ color: "#dc2626", width: 3, dash: "solid" });
  const [combinedLoading,  setCombinedLoading]  = useState(false);
  const [combinedError,    setCombinedError]    = useState<string | null>(null);

  const toggleCombinedCol = (col: string) =>
    setCombinedCols((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]);

  const toggleMultiCol = (col: string) => {
    setMultiCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
    setMultiResults([]);
  };

  const runCombined = async () => {
    if (!combinedCols.length || !outcomeCol) return;
    setCombinedLoading(true); setCombinedError(null); setCombinedResult(null);
    try {
      const res = await runROCCombined({
        session_id: session.session_id,
        predictor_columns: combinedCols,
        outcome_column: outcomeCol,
        model_name: combinedName || "Combined Model",
      });
      const d = res.data;
      setCombinedResult({ col: combinedName || "Combined Model", auc: d.auc, curve: d.curve });
    } catch (e: any) {
      const msg = e.response?.data?.detail;
      setCombinedError(typeof msg === "string" ? msg : (e.message ?? "Failed"));
    } finally { setCombinedLoading(false); }
  };

  const runMulti = async () => {
    if (!multiCols.length || !outcomeCol) return;
    setMultiLoading(true); setMultiError(null); setMultiResults([]);
    setMultiStyles(multiCols.map((_, i) => defaultStyle(i)));
    // Also run combined model if enabled
    if (showCombined && combinedCols.length > 0) runCombined();
    try {
      const settled = await Promise.allSettled(
        multiCols.map((col) =>
          runROC({ session_id: session.session_id, score_column: col, outcome_column: outcomeCol })
        )
      );
      const results: MultiResult[] = settled.map((s, i) => {
        if (s.status === "fulfilled") {
          const d = s.value.data;
          return {
            col: multiCols[i],
            auc: d.auc,
            ci_lower: d.ci_lower,
            ci_upper: d.ci_upper,
            curve: d.curve,
          };
        } else {
          return { col: multiCols[i], auc: 0, curve: [], error: "Failed" };
        }
      });
      setMultiResults(results);
    } catch (e: any) {
      setMultiError(e.message ?? "Request failed");
    } finally { setMultiLoading(false); }
  };

  // ── Single ROC run ──
  const run = async () => {
    if (!scoreCol || !outcomeCol) return;
    if (scoreCol === outcomeCol) { setError("Score and outcome columns must be different"); return; }
    setLoading(true); setError(null); setResult(null); setCmpResult(null);
    const mc = useManual && manualCutoff !== "" ? parseFloat(manualCutoff) : undefined;
    try {
      const res = await runROC({
        session_id: session.session_id,
        score_column: scoreCol,
        outcome_column: outcomeCol,
        ...(mc != null && !isNaN(mc) ? { manual_cutoff: mc } : {}),
      });
      setResult(res.data);
    } catch (e: any) {
      const msg = e.response?.data?.detail;
      setError(Array.isArray(msg) ? msg.map((m: any) => m.msg).join(", ") : (msg ?? e.message ?? "Request failed"));
    } finally { setLoading(false); }
  };

  const runCompare = async () => {
    if (scoreCol === scoreCol2) { setCmpError("Select two different score columns"); return; }
    setCmpLoading(true); setCmpError(null); setCmpResult(null);
    try {
      const res = await runROCCompare({
        session_id: session.session_id,
        score_column_1: scoreCol,
        score_column_2: scoreCol2,
        outcome_column: outcomeCol,
      });
      setCmpResult(res.data);
    } catch (e: any) {
      const msg = e.response?.data?.detail;
      setCmpError(Array.isArray(msg) ? msg.map((m: any) => m.msg).join(", ") : (msg ?? "Comparison failed"));
    } finally { setCmpLoading(false); }
  };

  // ── Exports ──
  const exportSingleCSV = () => {
    if (!result) return;
    const opt = result.optimal;
    const rows = [
      "ROC Analysis Export",
      `Score,${scoreCol}`, `Outcome,${outcomeCol}`,
      `n,${result.n}`, `Positives (1),${result.n_positive}`, `Negatives (0),${result.n_negative}`,
      `AUC,${result.auc}`, `AUC interpretation,${aucLabel(result.auc)}`,
      "",
      "Optimal cutoff (Youden J)",
      `Cutoff,${opt.cutoff}`,
      `Sensitivity,${fmtPct(opt.sensitivity)}`, `Specificity,${fmtPct(opt.specificity)}`,
      `PPV,${fmtPct(opt.ppv)}`, `NPV,${fmtPct(opt.npv)}`, `Accuracy,${fmtPct(opt.accuracy)}`,
      `LR+,${opt.lr_pos ?? "—"}`, `LR-,${opt.lr_neg ?? "—"}`, `Youden J,${opt.youden_j}`,
      `TP,${opt.tp}`, `TN,${opt.tn}`, `FP,${opt.fp}`, `FN,${opt.fn}`, "",
      ...(result.manual ? [
        "Manual cutoff", `Cutoff,${result.manual.cutoff}`,
        `Sensitivity,${fmtPct(result.manual.sensitivity)}`, `Specificity,${fmtPct(result.manual.specificity)}`,
        `PPV,${fmtPct(result.manual.ppv)}`, `NPV,${fmtPct(result.manual.npv)}`,
        `Accuracy,${fmtPct(result.manual.accuracy)}`, "",
      ] : []),
      "ROC Curve", "FPR,TPR",
      ...result.curve.map((p: any) => `${p.fpr.toFixed(6)},${p.tpr.toFixed(6)}`),
    ];
    downloadCSV(rows.join("\r\n"), `ROC_${scoreCol}_vs_${outcomeCol}.csv`);
  };

  const exportMultiCSV = () => {
    if (!multiResults.length) return;
    const rows: string[] = [
      `Multi-curve ROC Export`, `Outcome,${outcomeCol}`, "",
      "Summary", "Variable,AUC,CI_Lower,CI_Upper",
      ...multiResults.map((r) =>
        `${r.col},${r.auc},${r.ci_lower ?? ""},${r.ci_upper ?? ""}`
      ),
    ];
    multiResults.forEach((r) => {
      if (r.curve.length) {
        rows.push("", `Curve — ${r.col}`, "FPR,TPR");
        r.curve.forEach((p) => rows.push(`${p.fpr.toFixed(6)},${p.tpr.toFixed(6)}`));
      }
    });
    downloadCSV(rows.join("\r\n"), `ROC_multi_${outcomeCol}.csv`);
  };

  const downloadCSV = (content: string, filename: string) => {
    const blob = new Blob(["\uFEFF" + content], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  };

  const exportPNG = (filename: string) => {
    if (!rocPlotRef.current) return;
    (window as any).Plotly?.downloadImage(rocPlotRef.current, {
      format: "png", width: 900, height: 700, filename,
    });
  };

  // ── Derived ──
  const activeMetrics = useManual && result?.manual
    ? result.manual
    : result?.optimal ?? (result ? {
        cutoff: result.optimal_cutoff, sensitivity: result.sensitivity, specificity: result.specificity,
        ppv: null, npv: null, accuracy: null, lr_pos: null, lr_neg: null, youden_j: null,
        tp: result.tp, tn: result.tn, fp: result.fp, fn: result.fn,
      } : null);

  const updateMultiStyle = (i: number, patch: Partial<CurveStyle>) =>
    setMultiStyles((prev) => prev.map((s, j) => j === i ? { ...s, ...patch } : s));

  // ── Multi-curve plot traces (combined model first, then individual, then reference) ──
  const multiTraces = [
    // Combined model — shown on top with thicker line
    ...(showCombined && combinedResult && !combinedResult.error && combinedResult.curve.length > 0 ? [{
      type: "scatter", mode: "lines",
      x: combinedResult.curve.map((p) => p.fpr),
      y: combinedResult.curve.map((p) => p.tpr),
      line: { color: combinedStyle.color, width: combinedStyle.width, dash: combinedStyle.dash },
      name: `${combinedResult.col} AUC ${combinedResult.auc}`,
    }] : []),
    // Individual predictors
    ...multiResults
      .filter((r) => !r.error && r.curve.length > 0)
      .map((r, i) => {
        const st = multiStyles[i] ?? defaultStyle(i);
        return {
          type: "scatter", mode: "lines",
          x: r.curve.map((p) => p.fpr),
          y: r.curve.map((p) => p.tpr),
          line: { color: st.color, width: st.width, dash: st.dash },
          name: `${r.col} ${fmtAUC(r.auc, r.ci_lower, r.ci_upper)}`,
        };
      }),
    // Reference diagonal
    {
      type: "scatter", mode: "lines",
      x: [0, 1], y: [0, 1],
      line: { color: multiChance.color, width: multiChance.width, dash: multiChance.dash },
      name: "Reference",
    },
  ];

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div className="flex gap-4 h-full">

      {/* ── Left sidebar ─────────────────────────────────────────────────────── */}
      <div className="w-72 flex-shrink-0 flex flex-col gap-3 overflow-y-auto">

        {/* Mode toggle */}
        <div className="flex rounded-lg overflow-hidden border border-gray-300">
          <button
            onClick={() => setMode("single")}
            className={`flex-1 text-xs py-1.5 font-medium transition-colors
              ${mode === "single" ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"}`}>
            Single curve
          </button>
          <button
            onClick={() => setMode("multi")}
            className={`flex-1 text-xs py-1.5 font-medium transition-colors
              ${mode === "multi" ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"}`}>
            Multi-curve
          </button>
        </div>

        {/* Outcome column (shared) */}
        <div className="panel space-y-2">
          <label className="text-xs text-gray-400 block">
            Binary outcome <span className="text-gray-300">(must be 0/1)</span>
          </label>
          <select className="select w-full text-xs" value={outcomeCol}
            onChange={(e) => { setOutcomeCol(e.target.value); setResult(null); setMultiResults([]); }}>
            {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        {/* ── SINGLE MODE controls ── */}
        {mode === "single" && (
          <>
            <div className="panel space-y-3">
              <h3 className="text-sm font-semibold text-gray-700">Score / predictor</h3>
              <select className="select w-full text-xs" value={scoreCol}
                onChange={(e) => { setScoreCol(e.target.value); setResult(null); }}>
                {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>

              <div>
                <label className="flex items-center gap-2 text-xs text-gray-500 cursor-pointer mb-1">
                  <input type="checkbox" className="accent-indigo-500" checked={useManual}
                    onChange={(e) => { setUseManual(e.target.checked); if (!e.target.checked) setManualCutoff(""); }} />
                  Manual cutoff
                </label>
                {useManual && (
                  <input type="number" step="any" placeholder="e.g. 42.5"
                    className="select w-full text-xs" value={manualCutoff}
                    onChange={(e) => setManualCutoff(e.target.value)} />
                )}
              </div>

              <button className="btn-primary w-full" onClick={run}
                disabled={loading || !scoreCol || !outcomeCol}>
                {loading ? "Computing…" : "Run ROC"}
              </button>

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-2">
                  <p className="text-red-600 text-xs">{error}</p>
                </div>
              )}
            </div>

            {/* Single results */}
            {result && (
              <div className="panel space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-700">Results</h3>
                  <div className="flex gap-1">
                    <button onClick={exportSingleCSV}
                      className="px-2 py-1 rounded text-xs text-gray-500 hover:text-gray-700 hover:bg-gray-100 border border-gray-300">
                      ↓ CSV
                    </button>
                    <button onClick={() => exportPNG(`ROC_${scoreCol}_vs_${outcomeCol}`)}
                      className="px-2 py-1 rounded text-xs text-gray-500 hover:text-gray-700 hover:bg-gray-100 border border-gray-300">
                      ↓ PNG
                    </button>
                  </div>
                </div>

                <div className="flex flex-col items-center bg-gray-50 border border-gray-200 rounded-lg py-3">
                  <span className="text-xs text-gray-400 mb-0.5">AUC</span>
                  <span className={`text-2xl font-bold font-mono ${aucColor(result.auc)}`}>{result.auc}</span>
                  <span className={`text-xs mt-0.5 ${aucColor(result.auc)}`}>{aucLabel(result.auc)}</span>
                  {result.ci_lower != null && (
                    <span className="text-[10px] text-gray-400 mt-0.5">
                      95% CI {result.ci_lower} – {result.ci_upper}
                    </span>
                  )}
                </div>

                {[["n", result.n], ["Positives", result.n_positive], ["Negatives", result.n_negative]].map(([k, v]: any) => (
                  <div key={k} className="flex justify-between border-b border-gray-100 py-0.5 text-xs">
                    <span className="text-gray-400">{k}</span>
                    <span className="text-gray-700 font-mono">{v}</span>
                  </div>
                ))}

                {result.manual && (
                  <div className="flex rounded overflow-hidden border border-gray-300 mt-2">
                    <button
                      className={`flex-1 text-xs py-1 transition-colors ${!useManual ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-100"}`}
                      onClick={() => setUseManual(false)}>Youden J</button>
                    <button
                      className={`flex-1 text-xs py-1 transition-colors ${useManual ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-100"}`}
                      onClick={() => setUseManual(true)}>Manual</button>
                  </div>
                )}

                {activeMetrics && (
                  <MetricsBlock m={activeMetrics}
                    label={useManual && result.manual ? "At manual cutoff" : "At optimal cutoff (Youden J)"} />
                )}
              </div>
            )}

            {/* DeLong comparison */}
            <div className="panel space-y-3">
              <button className="flex items-center w-full" onClick={() => setShowCompare((v) => !v)}>
                <span className="text-sm font-semibold text-gray-700">AUC Comparison (DeLong)</span>
                <span className="ml-auto text-gray-400 text-xs">{showCompare ? "▲" : "▼"}</span>
              </button>
              {showCompare && (
                <>
                  <p className="text-xs text-gray-400">Non-parametric test comparing two score columns.</p>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Score 1</label>
                    <p className="text-xs text-indigo-600 font-mono truncate bg-indigo-50 rounded px-2 py-1">{scoreCol || "—"}</p>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Score 2</label>
                    <select className="select w-full text-xs" value={scoreCol2}
                      onChange={(e) => { setScoreCol2(e.target.value); setCmpResult(null); }}>
                      {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <button className="btn-primary w-full" onClick={runCompare}
                    disabled={cmpLoading || !scoreCol || !scoreCol2 || !outcomeCol || scoreCol === scoreCol2}>
                    {cmpLoading ? "Testing…" : "Run DeLong Test"}
                  </button>
                  {cmpError && <p className="text-red-500 text-xs">{cmpError}</p>}
                  {cmpResult && (
                    <div className="space-y-1 mt-1">
                      <div className={`text-xs px-2 py-1.5 rounded font-semibold border
                        ${cmpResult.significant ? "border-indigo-300 bg-indigo-50 text-indigo-700" : "border-gray-200 bg-gray-50 text-gray-500"}`}>
                        {cmpResult.significant ? "Significant difference ✓" : "No significant difference"}
                      </div>
                      {[
                        [`AUC — ${cmpResult.score_1}`, cmpResult.auc_1],
                        [`AUC — ${cmpResult.score_2}`, cmpResult.auc_2],
                        ["ΔAUC",    cmpResult.difference.toFixed(4)],
                        ["Z",       cmpResult.z.toFixed(3)],
                        ["p-value", fmtP(cmpResult.p)],
                        ["n",       cmpResult.n],
                      ].map(([k, v]: any) => (
                        <div key={k} className="flex justify-between border-b border-gray-100 py-0.5 text-xs">
                          <span className="text-gray-400 truncate">{k}</span>
                          <span className="text-gray-700 font-mono ml-2 shrink-0">{v}</span>
                        </div>
                      ))}
                      <p className="text-gray-400 text-xs italic pt-1">{cmpResult.interpretation}</p>
                    </div>
                  )}
                </>
              )}
            </div>
          </>
        )}

        {/* ── MULTI MODE controls ── */}
        {mode === "multi" && (
          <>
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">Predictors</h3>
                <div className="flex gap-1">
                  <button onClick={() => { setMultiCols([...numCols]); setMultiResults([]); }}
                    className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">
                    All
                  </button>
                  <button onClick={() => { setMultiCols([]); setMultiResults([]); }}
                    className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">
                    None
                  </button>
                </div>
              </div>
              <div className="max-h-48 overflow-y-auto space-y-0.5 border border-gray-200 rounded-lg p-1">
                {numCols.map((col) => (
                  <label key={col} className="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-50 cursor-pointer">
                    <input type="checkbox" className="accent-indigo-500"
                      checked={multiCols.includes(col)}
                      onChange={() => toggleMultiCol(col)} />
                    <span className="text-xs text-gray-700 truncate">{col}</span>
                  </label>
                ))}
              </div>
              <p className="text-[10px] text-gray-400">{multiCols.length} selected</p>

              <button className="btn-primary w-full" onClick={runMulti}
                disabled={multiLoading || multiCols.length < 1 || !outcomeCol}>
                {multiLoading ? "Computing…" : `Run ${multiCols.length} ROC${multiCols.length !== 1 ? "s" : ""}`}
              </button>
              {multiError && <p className="text-red-500 text-xs">{multiError}</p>}
            </div>

            {/* ── Combined Model panel ── */}
            <div className="panel space-y-2">
              <button className="flex items-center w-full gap-2" onClick={() => setShowCombined((v) => !v)}>
                <input type="checkbox" checked={showCombined}
                  onChange={(e) => setShowCombined(e.target.checked)}
                  className="accent-indigo-500" onClick={(e) => e.stopPropagation()} />
                <span className="text-sm font-semibold text-gray-700">Combined Model</span>
                <span className="ml-auto text-gray-400 text-xs">{showCombined ? "▲" : "▼"}</span>
              </button>

              {showCombined && (
                <>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    Fits logistic regression on selected variables and plots the combined model ROC.
                  </p>

                  {/* Model name */}
                  <input
                    type="text"
                    placeholder="Combined Model"
                    value={combinedName}
                    onChange={(e) => setCombinedName(e.target.value)}
                    className="select w-full text-xs"
                  />

                  {/* Predictor checkboxes */}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">Variables</span>
                      <div className="flex gap-1">
                        <button onClick={() => setCombinedCols([...allCols.filter((c) => c !== outcomeCol)])}
                          className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">All</button>
                        <button onClick={() => setCombinedCols([])}
                          className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">None</button>
                      </div>
                    </div>
                    <div className="max-h-36 overflow-y-auto space-y-0.5 border border-gray-200 rounded-lg p-1">
                      {allCols.filter((c) => c !== outcomeCol).map((col) => (
                        <label key={col} className="flex items-center gap-2 px-2 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                          <input type="checkbox" className="accent-red-500"
                            checked={combinedCols.includes(col)}
                            onChange={() => toggleCombinedCol(col)} />
                          <span className="text-xs text-gray-700 truncate">{col}</span>
                        </label>
                      ))}
                    </div>
                    <p className="text-[10px] text-gray-400 mt-1">{combinedCols.length} predictor(s) selected</p>
                  </div>

                  <button className="btn-primary w-full" onClick={runCombined}
                    disabled={combinedLoading || combinedCols.length < 1 || !outcomeCol}>
                    {combinedLoading ? "Fitting model…" : "Run Combined Model"}
                  </button>
                  {combinedError && <p className="text-red-500 text-xs">{combinedError}</p>}

                  {combinedResult && !combinedResult.error && (
                    <div className="flex items-center justify-between border-t border-gray-100 pt-2">
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-0.5 rounded" style={{ background: combinedStyle.color, height: 3 }} />
                        <span className="text-xs text-gray-600 font-medium">{combinedResult.col}</span>
                      </div>
                      <span className={`text-sm font-bold font-mono ${aucColor(combinedResult.auc)}`}>
                        {combinedResult.auc}
                      </span>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Multi results summary */}
            {multiResults.length > 0 && (
              <div className="panel space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-700">AUC Summary</h3>
                  <div className="flex gap-1">
                    <button onClick={exportMultiCSV}
                      className="px-2 py-1 rounded text-xs text-gray-500 hover:text-gray-700 hover:bg-gray-100 border border-gray-300">
                      ↓ CSV
                    </button>
                    <button onClick={() => exportPNG(`ROC_multi_${outcomeCol}`)}
                      className="px-2 py-1 rounded text-xs text-gray-500 hover:text-gray-700 hover:bg-gray-100 border border-gray-300">
                      ↓ PNG
                    </button>
                  </div>
                </div>
                {/* Combined model in summary */}
                {showCombined && combinedResult && !combinedResult.error && (
                  <div className="flex items-center justify-between gap-2 border-b-2 border-gray-200 pb-1.5 mb-1">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <div className="w-3 h-0.5 rounded flex-shrink-0" style={{ background: combinedStyle.color, height: 3 }} />
                      <span className="text-xs text-gray-700 font-semibold truncate">{combinedResult.col}</span>
                    </div>
                    <span className={`text-xs font-mono font-bold flex-shrink-0 ${aucColor(combinedResult.auc)}`}>
                      {combinedResult.auc}
                    </span>
                  </div>
                )}
                {[...multiResults]
                  .sort((a, b) => b.auc - a.auc)
                  .map((r) => {
                    const origIdx = multiResults.findIndex((x) => x.col === r.col);
                    const st = multiStyles[origIdx] ?? defaultStyle(origIdx);
                    return (
                      <div key={r.col} className="flex items-center justify-between gap-2 border-b border-gray-100 pb-1">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: st.color }} />
                          <span className="text-xs text-gray-600 truncate">{r.col}</span>
                        </div>
                        {r.error ? (
                          <span className="text-red-500 text-xs">err</span>
                        ) : (
                          <span className={`text-xs font-mono font-semibold flex-shrink-0 ${aucColor(r.auc)}`}>
                            {r.auc}
                          </span>
                        )}
                      </div>
                    );
                  })}
              </div>
            )}
          </>
        )}
      </div>

      {/* ── Plot area ────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col gap-3 min-h-0">

        {/* Style controls bar */}
        {mode === "single" && result && (
          <div className="panel flex flex-wrap items-center gap-4 py-2">
            <StyleRow
              label="ROC curve"
              color={singleStyle.color} width={singleStyle.width} dash={singleStyle.dash}
              onColor={(v) => setSingleStyle((s) => ({ ...s, color: v }))}
              onWidth={(v) => setSingleStyle((s) => ({ ...s, width: v }))}
              onDash={(v)  => setSingleStyle((s) => ({ ...s, dash:  v }))}
            />
            <div className="w-px h-5 bg-gray-200" />
            <StyleRow
              label="Chance line"
              color={chanceStyle.color} width={chanceStyle.width} dash={chanceStyle.dash}
              onColor={(v) => setChanceStyle((s) => ({ ...s, color: v }))}
              onWidth={(v) => setChanceStyle((s) => ({ ...s, width: v }))}
              onDash={(v)  => setChanceStyle((s) => ({ ...s, dash:  v }))}
            />
          </div>
        )}

        {mode === "multi" && multiResults.length > 0 && (
          <div className="panel space-y-2 py-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-gray-500">Curve styles</span>
            </div>
            <div className="space-y-1.5">
              {/* Combined model style row — shown first when enabled */}
              {showCombined && combinedResult && !combinedResult.error && (
                <>
                  <StyleRow
                    label={combinedResult.col}
                    color={combinedStyle.color} width={combinedStyle.width} dash={combinedStyle.dash}
                    onColor={(v) => setCombinedStyle((s) => ({ ...s, color: v }))}
                    onWidth={(v) => setCombinedStyle((s) => ({ ...s, width: v }))}
                    onDash={(v)  => setCombinedStyle((s) => ({ ...s, dash:  v }))}
                  />
                  <div className="border-t border-gray-200" />
                </>
              )}
              {multiResults.map((r, i) => {
                const st = multiStyles[i] ?? defaultStyle(i);
                return (
                  <StyleRow
                    key={r.col} label={r.col}
                    color={st.color} width={st.width} dash={st.dash}
                    onColor={(v) => updateMultiStyle(i, { color: v })}
                    onWidth={(v) => updateMultiStyle(i, { width: v })}
                    onDash={(v)  => updateMultiStyle(i, { dash:  v })}
                  />
                );
              })}
              <div className="border-t border-gray-100 pt-1.5">
                <StyleRow
                  label="Reference"
                  color={multiChance.color} width={multiChance.width} dash={multiChance.dash}
                  onColor={(v) => setMultiChance((s) => ({ ...s, color: v }))}
                  onWidth={(v) => setMultiChance((s) => ({ ...s, width: v }))}
                  onDash={(v)  => setMultiChance((s) => ({ ...s, dash:  v }))}
                />
              </div>
            </div>
          </div>
        )}

        {/* Plot */}
        <div className="flex-1 panel min-h-0" style={{ minHeight: 380 }}>

          {/* ── Single plot ── */}
          {mode === "single" && result && (
            <Plot
              data={[
                {
                  type: "scatter", mode: "lines",
                  x: result.curve.map((p: any) => p.fpr),
                  y: result.curve.map((p: any) => p.tpr),
                  line: { color: singleStyle.color, width: singleStyle.width, dash: singleStyle.dash },
                  name: fmtAUC(result.auc, result.ci_lower, result.ci_upper),
                  fill: "tozeroy",
                  fillcolor: `${singleStyle.color}14`,
                },
                {
                  type: "scatter", mode: "lines",
                  x: [0, 1], y: [0, 1],
                  line: { color: chanceStyle.color, width: chanceStyle.width, dash: chanceStyle.dash },
                  name: "Reference",
                },
                ...(result.optimal ? [{
                  type: "scatter", mode: "markers",
                  x: [1 - result.optimal.specificity], y: [result.optimal.sensitivity],
                  marker: { color: "#ef4444", size: 10, symbol: "circle" },
                  name: `Optimal cutoff = ${result.optimal.cutoff}`,
                }] : result.sensitivity != null ? [{
                  type: "scatter", mode: "markers",
                  x: [1 - result.specificity], y: [result.sensitivity],
                  marker: { color: "#ef4444", size: 10, symbol: "circle" },
                  name: `Cutoff = ${result.optimal_cutoff}`,
                }] : []),
                ...(result.manual ? [{
                  type: "scatter", mode: "markers",
                  x: [1 - result.manual.specificity], y: [result.manual.sensitivity],
                  marker: { color: "#f59e0b", size: 10, symbol: "diamond" },
                  name: `Manual cutoff = ${result.manual.cutoff}`,
                }] : []),
              ]}
              layout={{
                ...PLOT_LAYOUT,
                autosize: true,
                title: { text: `ROC — ${scoreCol} → ${outcomeCol}`, font: { color: "#374151", size: 13 } },
                legend: { font: { color: "#374151", size: 11 }, bgcolor: "rgba(249,250,251,0.9)", bordercolor: "#e5e7eb", borderwidth: 1 },
                annotations: [{
                  x: 0.98, y: 0.06, xref: "paper" as const, yref: "paper" as const,
                  text: `AUC = ${result.auc}`, showarrow: false,
                  font: { color: "#374151", size: 13 },
                  bgcolor: "rgba(249,250,251,0.9)", bordercolor: "#e5e7eb", borderwidth: 1, borderpad: 5,
                  xanchor: "right" as const, yanchor: "bottom" as const,
                }],
              }}
              onInitialized={(_: object, gd: HTMLElement) => { rocPlotRef.current = gd; }}
              onUpdate={(_: object, gd: HTMLElement)      => { rocPlotRef.current = gd; }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
            />
          )}

          {/* ── Multi-curve plot ── */}
          {mode === "multi" && multiResults.length > 0 && (
            <Plot
              data={multiTraces as any}
              layout={{
                ...PLOT_LAYOUT,
                autosize: true,
                title: { text: `ROC Curves → ${outcomeCol}`, font: { color: "#374151", size: 13 } },
                legend: {
                  font: { color: "#374151", size: 11 },
                  bgcolor: "rgba(249,250,251,0.95)",
                  bordercolor: "#e5e7eb", borderwidth: 1,
                  x: 0.5, y: 0.05, xanchor: "left" as const, yanchor: "bottom" as const,
                },
              }}
              onInitialized={(_: object, gd: HTMLElement) => { rocPlotRef.current = gd; }}
              onUpdate={(_: object, gd: HTMLElement)      => { rocPlotRef.current = gd; }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
            />
          )}

          {/* ── Empty state ── */}
          {((mode === "single" && !result) || (mode === "multi" && !multiResults.length)) && (
            <div className="h-full flex flex-col items-center justify-center gap-2 text-gray-400">
              <span className="text-3xl">📈</span>
              <span className="text-sm text-center">
                {mode === "single"
                  ? "Select a continuous score and a binary outcome (0/1), then click Run ROC"
                  : "Select predictors and a binary outcome, then click Run ROCs"}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
