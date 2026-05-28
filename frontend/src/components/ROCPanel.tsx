import { useState, useEffect, useMemo, useRef } from "react";
import Plot from "../PlotComponent";
import PlotExporter from "./PlotExporter";
import ResultExporter from "./ResultExporter";
import { useStore, PALETTES } from "../store";
import { runROC, runROCCompare, runROCCombined } from "../api";
import { Tip, InfoBanner } from "./Tip";
import { MissingGuard, type ImputationStrategy } from "./MissingGuard";
import { fmtP } from "../lib/format";

// ── Helper to get current palette primary color ────────────────────────────
const _pal = () => PALETTES[useStore.getState().plotTheme.palette] ?? PALETTES.indigo;
const _p0 = () => _pal()[0];

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
const fmtAUC = (auc: number, lo?: number, hi?: number) =>
  lo != null && hi != null
    ? `AUC ${auc.toFixed(2)} (95% CI ${lo.toFixed(2)}–${hi.toFixed(2)})`
    : `AUC ${auc.toFixed(2)}`;

// ── ROC Guidance ────────────────────────────────────────────────────────────
const ROC_GUIDANCE = {
  single: {
    use: "Evaluate how well a single continuous predictor (biomarker, score) discriminates between two groups (e.g. disease vs. healthy, event vs. no event).",
    check: "Outcome must be binary 0/1. The predictor should have a reasonable spread of values. AUC = 0.5 means no better than chance.",
    interpret: "AUC 0.9-1.0 = Excellent, 0.8-0.9 = Good, 0.7-0.8 = Fair, <0.7 = Poor. Youden's index gives the optimal cut-off that maximises Sensitivity + Specificity. Report: AUC (95% CI).",
  },
  compare: {
    use: "Compare two biomarkers' discriminative ability using the DeLong test. Essential for proving a new marker outperforms an existing one.",
    check: "Both markers measured on the SAME patients (paired design). DeLong test is valid for correlated AUCs from the same sample.",
    interpret: "p < 0.05 means one AUC is significantly higher. Report: AUC\u2081 vs AUC\u2082, \u0394AUC (95% CI), DeLong p. The overlaid ROC plot visualises the difference.",
  },
  multi: {
    use: "Screen multiple biomarkers simultaneously to find the best single predictor. Overlay ROC curves for visual comparison.",
    check: "Each predictor is evaluated independently against the same binary outcome. The combined model uses cross-validated predictions to avoid overfitting bias.",
    interpret: "Compare AUCs and 95% CIs across predictors. Overlapping CIs suggest no significant difference. The combined model shows the joint discriminative power of all predictors together.",
  },
};

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

// Trimmed from full clinical explanations to one-line definitions so the
// hover popovers don't sprawl across the screen. The full glossary is still
// in the methods appendix for anyone who needs the full context.
const METRIC_TIPS: Record<string, string> = {
  "Cutoff":      "Threshold for predicting positive (score ≥ cutoff).",
  "Sensitivity": "TP rate — % of true cases correctly identified.",
  "Specificity": "TN rate — % of true non-cases correctly ruled out.",
  "PPV":         "Of predicted positives, % truly diseased. Depends on prevalence.",
  "NPV":         "Of predicted negatives, % truly healthy.",
  "Accuracy":    "(TP + TN) / total. Misleading when classes are imbalanced.",
  "LR+":         "Likelihood Ratio +: post-test odds ↑. >10 = strong evidence.",
  "LR−":         "Likelihood Ratio −: post-test odds ↓. <0.1 = strong evidence.",
  "Youden J":    "Sens + Spec − 1. Optimal cutoff maximises J.",
  "TP": "True Positives.", "TN": "True Negatives.",
  "FP": "False Positives (Type I error).",
  "FN": "False Negatives (Type II error).",
};

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
        ["Youden J",    m.youden_j  != null ? m.youden_j.toFixed(2) : "—"],
        ["TP", m.tp], ["TN", m.tn], ["FP", m.fp], ["FN", m.fn],
      ].map(([k, v]: any) => (
        <div key={k} className="flex justify-between border-b border-gray-100 py-0.5 text-xs">
          <span className="text-gray-400 flex items-center">
            {k}
            {METRIC_TIPS[k] && <Tip text={METRIC_TIPS[k]} wide />}
          </span>
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
  const showGrid = useStore((s) => s.showGrid);
  if (!session) return null;

  const numCols = session.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
  const allCols = session.columns.map((c) => c.name);

  // Binary columns (≤ 2 unique non-null values, both ∈ {0, 1}) — ROC outcome
  // must be 0/1. Falls back to allCols if no binary column is detected so
  // the user can still type-override via the Dictionary modal.
  const binaryCols = useMemo(() => {
    const out: string[] = [];
    for (const col of session.columns) {
      const vals = new Set<unknown>();
      for (const row of session.preview) {
        const v = row[col.name];
        if (v == null || v === "") continue;
        vals.add(typeof v === "number" ? v : Number(v));
        if (vals.size > 2) break;
      }
      const arr = [...vals];
      if (arr.length === 0 || arr.length > 2) continue;
      if (arr.every((v) => v === 0 || v === 1)) out.push(col.name);
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.session_id]);
  // Prefer a binary 0/1 column whose name hints at an outcome; otherwise the
  // first binary column; only fall back to allCols when no binary exists.
  const defaultOutcome =
    binaryCols.find((c) => /mortalite|death|event|outcome|binary|status/i.test(c))
    ?? binaryCols[0]
    ?? allCols.find((c) => /mortalite|death|event|outcome|binary|status/i.test(c))
    ?? allCols[0]
    ?? "";

  // ── Mode ──
  const [mode, setMode] = useState<"single" | "multi">("single");

  // ── Shared ──
  const [outcomeCol, setOutcomeCol] = useState(defaultOutcome);
  const rocPlotRef = useRef<any>(null);
  const rocSingleRef = useRef<any>(null);
  const rocCompareRef = useRef<any>(null);
  const rocMultiRef = useRef<any>(null);

  // ── Single-curve state ──
  const [scoreCol,     setScoreCol]     = useState(numCols[0] ?? "");
  const [manualCutoff, setManualCutoff] = useState("");
  // Score direction for ROC. "auto" flips the score sign when the naive
  // AUC < 0.5 so protective biomarkers (albumin, eGFR, Hb) report the
  // correct ~0.69 instead of the inverted ~0.31 — see Heinze (2002) and
  // pROC's `direction="auto"` for the same convention.
  const [scoreDirection, setScoreDirection]   = useState<"auto" | "higher" | "lower">("auto");
  const [scoreDirection2, setScoreDirection2] = useState<"auto" | "higher" | "lower">("auto");
  const [useManual,    setUseManual]    = useState(false);
  const [result,       setResult]       = useState<any>(null);
  const [error,        setError]        = useState<string | null>(null);
  const [loading,      setLoading]      = useState(false);
  const [imputation,   setImputation]   = useState<ImputationStrategy>("listwise");

  const [showCompare, setShowCompare] = useState(false);
  const [scoreCol2,   setScoreCol2]   = useState(numCols[1] ?? numCols[0] ?? "");
  const [cmpResult,   setCmpResult]   = useState<any>(null);
  const [cmpError,    setCmpError]    = useState<string | null>(null);
  const [cmpLoading,  setCmpLoading]  = useState(false);

  const [singleStyle, setSingleStyle] = useState<CurveStyle>({ color: _p0(), width: 2.5, dash: "solid" });
  const [chanceStyle, setChanceStyle] = useState<CurveStyle>({ color: "#9ca3af", width: 1,   dash: "dash"  });

  useEffect(() => {
    if (result) { setSingleStyle({ color: _p0(), width: 2.5, dash: "solid" }); }
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
  const [multiFilter,     setMultiFilter]     = useState("");
  const [combinedFilter,  setCombinedFilter]  = useState("");

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
        imputation,
        direction: scoreDirection,
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
        direction_1: scoreDirection,
        direction_2: scoreDirection2,
      });
      setCmpResult(res.data);
    } catch (e: any) {
      const msg = e.response?.data?.detail;
      setCmpError(Array.isArray(msg) ? msg.map((m: any) => m.msg).join(", ") : (msg ?? "Comparison failed"));
    } finally { setCmpLoading(false); }
  };

  // ── Exports ──
  // Two-column "metric / value" table — round-trips cleanly through CSV /
  // XLSX. ROC curve points (FPR, TPR per row) come after the summary block
  // in the same sheet so the user gets one self-contained file.
  const singleExport = useMemo(() => {
    if (!result) return null;
    const opt = result.optimal ?? {};
    const headers = ["Metric", "Value"];
    const rows: (string | number | null)[][] = [
      ["Score column", scoreCol],
      ["Outcome column", outcomeCol],
      ["n", result.n],
      ["Positives (1)", result.n_positive],
      ["Negatives (0)", result.n_negative],
      ["AUC", result.auc],
      ["AUC interpretation", aucLabel(result.auc)],
      ["AUC SE (DeLong)", result.auc_se ?? "—"],
      ["95% CI lower", result.ci_lower ?? "—"],
      ["95% CI upper", result.ci_upper ?? "—"],
      ["AUC Z (vs 0.5)", result.auc_z ?? "—"],
      ["AUC p (H₀: AUC = 0.5)", result.auc_p ?? "—"],
      ["", ""],
      ["Optimal cutoff (Youden J)", ""],
      ["Cutoff", opt.cutoff ?? "—"],
      ["Sensitivity", fmtPct(opt.sensitivity)],
      ["Specificity", fmtPct(opt.specificity)],
      ["PPV", fmtPct(opt.ppv)],
      ["NPV", fmtPct(opt.npv)],
      ["Accuracy", fmtPct(opt.accuracy)],
      ["LR+", opt.lr_pos ?? "—"],
      ["LR-", opt.lr_neg ?? "—"],
      ["Youden J", opt.youden_j != null ? Number(opt.youden_j).toFixed(2) : "—"],
      ["TP", opt.tp ?? "—"],
      ["TN", opt.tn ?? "—"],
      ["FP", opt.fp ?? "—"],
      ["FN", opt.fn ?? "—"],
    ];
    if (result.manual) {
      rows.push(["", ""], ["Manual cutoff", ""],
        ["Cutoff", result.manual.cutoff],
        ["Sensitivity", fmtPct(result.manual.sensitivity)],
        ["Specificity", fmtPct(result.manual.specificity)],
        ["PPV", fmtPct(result.manual.ppv)],
        ["NPV", fmtPct(result.manual.npv)],
        ["Accuracy", fmtPct(result.manual.accuracy)],
      );
    }
    rows.push(["", ""], ["ROC Curve (FPR, TPR)", ""]);
    for (const p of result.curve) {
      rows.push([p.fpr.toFixed(6), p.tpr.toFixed(6)]);
    }
    return { headers, rows };
  }, [result, scoreCol, outcomeCol]);

  const multiExport = useMemo(() => {
    if (!multiResults.length) return null;
    const headers = ["Variable", "AUC", "CI Lower", "CI Upper"];
    const rows: (string | number | null)[][] = multiResults.map((r) =>
      [r.col, r.auc, r.ci_lower ?? "", r.ci_upper ?? ""]
    );
    // Append each curve's FPR/TPR pairs after the summary so all data lands
    // in one file — section header rows mark the boundary.
    multiResults.forEach((r) => {
      if (r.curve.length) {
        rows.push(["", "", "", ""], [`Curve — ${r.col}`, "FPR", "TPR", ""]);
        r.curve.forEach((p) => rows.push(["", p.fpr.toFixed(6), p.tpr.toFixed(6), ""]));
      }
    });
    return { headers, rows };
  }, [multiResults]);

  // PNG / TIFF flow through ResultExporter \u2014 kept only for any legacy
  // call site (none remain after this refactor).

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

      {/* ── Left sidebar (controls + DeLong) ────────────────────────────────── */}
      {/* Holds the score / outcome / direction / manual-cutoff controls plus
          the DeLong AUC-comparison card (single mode). The single-mode
          result card (AUC tile + paragraph + metric table) lives in the
          right-hand column. */}
      <div className="w-[340px] flex-shrink-0 flex flex-col gap-3 overflow-y-auto">

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
            {binaryCols.length === 0 && <span className="ml-1 text-[10px] text-amber-600">⚠ no binary column detected — recode in Dictionary</span>}
          </label>
          <select className="select w-full text-xs" value={outcomeCol}
            onChange={(e) => { setOutcomeCol(e.target.value); setResult(null); setMultiResults([]); }}>
            {(binaryCols.length > 0 ? binaryCols : allCols).map((c) => <option key={c} value={c}>{c}</option>)}
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
                <label className="text-[10px] text-gray-400 flex items-center mb-1">
                  Direction
                  <Tip text="Auto: flip score when AUC<0.5 (protective markers). Higher: risk score. Lower: protective." />
                </label>
                <div className="flex gap-0 rounded overflow-hidden border border-gray-200">
                  {(["auto", "higher", "lower"] as const).map((d) => (
                    <button key={d}
                      onClick={() => { setScoreDirection(d); setResult(null); setCmpResult(null); }}
                      className={`flex-1 text-[10px] py-1 transition-colors ${
                        scoreDirection === d ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
                      }`}>
                      {d === "auto" ? "Auto" : d === "higher" ? "Higher = event" : "Lower = event"}
                    </button>
                  ))}
                </div>
                {result?.direction_flipped && (
                  <p className="text-[10px] text-emerald-700 mt-1">
                    ⚙ Auto-flipped: low values of <span className="font-mono">{scoreCol}</span> predict the event (AUC describes the protective direction).
                  </p>
                )}
              </div>

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

              <MissingGuard
                sessionId={session.session_id}
                columns={[scoreCol, outcomeCol].filter(Boolean)}
                imputation={imputation}
                onImputation={setImputation}
              >
                <button className="btn-primary w-full" onClick={run}
                  disabled={loading || !scoreCol || !outcomeCol}>
                  {loading ? "Computing…" : "Run ROC"}
                </button>
              </MissingGuard>

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-2">
                  <p className="text-red-600 text-xs">{error}</p>
                </div>
              )}
            </div>

            {/* Single results card lives in the right-hand column; the
                DeLong comparison card stays here in the left sidebar so
                the workflow (pick two scores → compare) sits next to the
                main score / outcome / direction controls instead of
                getting pushed underneath the results column. */}
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
                  <div>
                    <label className="text-[10px] text-gray-400 block mb-1">
                      Score 2 direction
                      <Tip text="Auto-flip when AUC<0.5 (recommended for protective markers vs risk scores)." />
                    </label>
                    <div className="flex gap-0 rounded overflow-hidden border border-gray-200">
                      {(["auto", "higher", "lower"] as const).map((d) => (
                        <button key={d}
                          onClick={() => { setScoreDirection2(d); setCmpResult(null); }}
                          className={`flex-1 text-[10px] py-1 transition-colors ${
                            scoreDirection2 === d ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
                          }`}>
                          {d === "auto" ? "Auto" : d === "higher" ? "H" : "L"}
                        </button>
                      ))}
                    </div>
                  </div>
                  {(cmpResult?.direction_1_flipped || cmpResult?.direction_2_flipped) && (
                    <div className="text-[10px] text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-2 py-1 leading-tight">
                      ⚙ DeLong recomputed with auto-flipped direction:
                      {cmpResult.direction_1_flipped && (
                        <> <span className="font-mono">{scoreCol}</span> (lower = event)</>
                      )}
                      {cmpResult.direction_1_flipped && cmpResult.direction_2_flipped && " · "}
                      {cmpResult.direction_2_flipped && (
                        <><span className="font-mono">{scoreCol2}</span> (lower = event)</>
                      )}
                    </div>
                  )}
                  <button className="btn-primary w-full" onClick={runCompare}
                    disabled={cmpLoading || !scoreCol || !scoreCol2 || !outcomeCol || scoreCol === scoreCol2}>
                    {cmpLoading ? "Testing…" : "Run DeLong Test"}
                  </button>
                  {cmpError && <p className="text-red-500 text-xs">{cmpError}</p>}
                  {cmpResult && (
                    <div className="space-y-2 mt-1">
                      <div className={`text-xs px-2 py-1.5 rounded font-semibold border flex items-center gap-1.5
                        ${cmpResult.significant ? "border-green-300 bg-green-50 text-green-700" : "border-gray-200 bg-gray-50 text-gray-500"}`}>
                        {cmpResult.significant ? "✓ Significant difference (p < 0.05)" : "No significant difference (p ≥ 0.05)"}
                      </div>

                      <div className="grid grid-cols-3 gap-1 text-center">
                        <div className="bg-blue-50 rounded p-1.5">
                          <p className="text-[9px] text-blue-400 uppercase tracking-wide font-semibold">Baseline AUC</p>
                          <p className="text-sm font-bold font-mono text-blue-700">{cmpResult.auc_2.toFixed(3)}</p>
                          <p className="text-[9px] text-blue-400">{cmpResult.ci_2_low.toFixed(3)}–{cmpResult.ci_2_high.toFixed(3)}</p>
                        </div>
                        <div className="bg-rose-50 rounded p-1.5">
                          <p className="text-[9px] text-rose-400 uppercase tracking-wide font-semibold">New AUC</p>
                          <p className="text-sm font-bold font-mono text-rose-700">{cmpResult.auc_1.toFixed(3)}</p>
                          <p className="text-[9px] text-rose-400">{cmpResult.ci_1_low.toFixed(3)}–{cmpResult.ci_1_high.toFixed(3)}</p>
                        </div>
                        <div className={`rounded p-1.5 ${cmpResult.significant ? "bg-green-50" : "bg-gray-50"}`}>
                          <p className="text-[9px] text-gray-400 uppercase tracking-wide font-semibold">ΔAUC</p>
                          <p className={`text-sm font-bold font-mono ${cmpResult.significant ? "text-green-700" : "text-gray-600"}`}>
                            {cmpResult.difference > 0 ? "+" : ""}{cmpResult.difference.toFixed(3)}
                          </p>
                          <p className="text-[9px] text-gray-400">
                            {cmpResult.ci_diff_low.toFixed(3)} to {cmpResult.ci_diff_high.toFixed(3)}
                          </p>
                        </div>
                      </div>

                      {[
                        ["Z statistic", cmpResult.z.toFixed(3)],
                        ["DeLong p-value", fmtP(cmpResult.p)],
                        ["n (paired)", cmpResult.n],
                      ].map(([k, v]: any) => (
                        <div key={k} className="flex justify-between border-b border-gray-100 py-0.5 text-xs">
                          <span className="text-gray-400">{k}</span>
                          <span className={`font-mono ml-2 shrink-0 ${k === "DeLong p-value" && cmpResult.significant ? "text-green-700 font-semibold" : "text-gray-700"}`}>{v}</span>
                        </div>
                      ))}

                      <div className="rounded bg-amber-50 border border-amber-200 px-2 py-2 text-[10px] text-amber-800 leading-relaxed">
                        <p className="font-semibold mb-0.5">Publication format (Q1/Q2):</p>
                        <p className="italic">{cmpResult.interpretation}</p>
                      </div>
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
              <input
                type="text"
                placeholder="Filter variables…"
                value={multiFilter}
                onChange={(e) => setMultiFilter(e.target.value)}
                className="select w-full text-xs py-1"
              />
              <div className="max-h-48 overflow-y-auto space-y-0.5 border border-gray-200 rounded-lg p-1">
                {numCols.filter((c) => c.toLowerCase().includes(multiFilter.toLowerCase())).map((col) => (
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
                    Fits logistic regression on selected variables using cross-validated predictions (no overfitting bias) and plots the combined model ROC.
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
                    <input
                      type="text"
                      placeholder="Filter variables…"
                      value={combinedFilter}
                      onChange={(e) => setCombinedFilter(e.target.value)}
                      className="select w-full text-xs py-1 mb-1"
                    />
                    <div className="max-h-36 overflow-y-auto space-y-0.5 border border-gray-200 rounded-lg p-1">
                      {allCols.filter((c) => c !== outcomeCol && c.toLowerCase().includes(combinedFilter.toLowerCase())).map((col) => (
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
                  {multiExport && (
                    <ResultExporter
                      title={`ROC_multi_${outcomeCol}`}
                      headers={multiExport.headers}
                      rows={multiExport.rows}
                      plotRef={rocPlotRef}
                    />
                  )}
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

      {/* ── Plot area (centre column) ───────────────────────────────────── */}
      <div className="flex-1 flex flex-col gap-3 min-h-0 overflow-y-auto">

        {/* ROC Guidance */}
        {(() => {
          const g = mode === "single" ? ROC_GUIDANCE.single : ROC_GUIDANCE.multi;
          return (
            <div className="grid grid-cols-3 gap-3">
              {[
                { icon: "🎯", title: "Use when", text: g.use },
                { icon: "✅", title: "Check", text: g.check },
                { icon: "📖", title: "Interpret", text: g.interpret },
              ].map(({ icon, title, text }) => (
                <div key={title} className="panel bg-indigo-50 border-indigo-200 p-3">
                  <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mb-1">{icon} {title}</p>
                  <p className="text-xs text-indigo-800 leading-relaxed">{text}</p>
                </div>
              ))}
            </div>
          );
        })()}

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
          {mode === "single" && result && (() => {
            // Pick the cutoff point to highlight: optimal (Youden J) unless the
            // user clicked "Manual" with a cutoff in scope.
            const cutoffPoint = result.manual
              ? { x: 1 - result.manual.specificity, y: result.manual.sensitivity,
                  cutoff: Number(result.manual.cutoff), se: result.manual.sensitivity,
                  sp: result.manual.specificity, label: "Manual cutoff", color: "#f59e0b" }
              : result.optimal
                ? { x: 1 - result.optimal.specificity, y: result.optimal.sensitivity,
                    cutoff: Number(result.optimal.cutoff), se: result.optimal.sensitivity,
                    sp: result.optimal.specificity, label: "Optimal cutoff", color: "#ef4444" }
                : result.sensitivity != null
                  ? { x: 1 - result.specificity, y: result.sensitivity,
                      cutoff: Number(result.optimal_cutoff), se: result.sensitivity,
                      sp: result.specificity, label: "Optimal cutoff", color: "#ef4444" }
                  : null;

            // AUC box: line 1 = AUC + 95% CI, line 2 = p.
            const aucCI = result.ci_lower != null && result.ci_upper != null
              ? `AUC = ${result.auc.toFixed(2)} (95% CI ${result.ci_lower.toFixed(2)}–${result.ci_upper.toFixed(2)})`
              : `AUC = ${result.auc.toFixed(2)}`;
            const aucP = result.auc_p != null
              ? (result.auc_p < 0.001 ? "p < 0.001" : `p = ${result.auc_p.toFixed(3)}`)
              : null;
            const aucBoxText = aucP ? `${aucCI}<br>${aucP}` : aucCI;

            const annotations: any[] = [
              {
                x: 0.98, y: 0.04, xref: "paper" as const, yref: "paper" as const,
                text: aucBoxText,
                showarrow: false,
                font: { color: "#374151", size: 13 },
                align: "right" as const,
                bgcolor: "rgba(249,250,251,0.92)", bordercolor: "#9ca3af", borderwidth: 1, borderpad: 6,
                xanchor: "right" as const, yanchor: "bottom" as const,
              },
            ];
            if (cutoffPoint) {
              // Arrow from text label to the cutoff dot, like matplotlib annotate().
              annotations.push({
                x: cutoffPoint.x, y: cutoffPoint.y, xref: "x" as const, yref: "y" as const,
                ax: 50, ay: 70, axref: "pixel" as const, ayref: "pixel" as const,
                text: `${cutoffPoint.label} = ${cutoffPoint.cutoff.toFixed(2)}<br>(Se ${(cutoffPoint.se * 100).toFixed(1)}%, Sp ${(cutoffPoint.sp * 100).toFixed(1)}%)`,
                showarrow: true,
                arrowhead: 0, arrowsize: 1, arrowwidth: 1.2, arrowcolor: "#6b7280",
                font: { color: "#374151", size: 11 },
                align: "left" as const,
                bgcolor: "rgba(255,255,255,0)", borderpad: 0,
                xanchor: "left" as const, yanchor: "top" as const,
              });
            }

            return (
            <div className="relative" style={{ width: "100%", height: "100%" }}>
            <PlotExporter plotRef={rocSingleRef} title="ROC_Curve" />
            <Plot
              data={[
                {
                  type: "scatter", mode: "lines",
                  x: result.curve.map((p: any) => p.fpr),
                  y: result.curve.map((p: any) => p.tpr),
                  line: { color: singleStyle.color, width: singleStyle.width, dash: singleStyle.dash, shape: "hv" as const },
                  name: fmtAUC(result.auc, result.ci_lower, result.ci_upper),
                  fill: "tozeroy",
                  fillcolor: `${singleStyle.color}22`,
                  showlegend: false,
                  hovertemplate: "FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
                },
                {
                  type: "scatter", mode: "lines",
                  x: [0, 1], y: [0, 1],
                  line: { color: chanceStyle.color, width: chanceStyle.width, dash: chanceStyle.dash },
                  name: "Reference",
                  showlegend: false,
                  hoverinfo: "skip" as const,
                },
                ...(cutoffPoint ? [{
                  type: "scatter" as const, mode: "markers" as const,
                  x: [cutoffPoint.x], y: [cutoffPoint.y],
                  marker: { color: cutoffPoint.color, size: 12, symbol: "circle" as const,
                            line: { color: "#ffffff", width: 1.5 } },
                  name: `${cutoffPoint.label} = ${cutoffPoint.cutoff.toFixed(2)}`,
                  showlegend: false,
                  hovertemplate: `${cutoffPoint.label} = ${cutoffPoint.cutoff.toFixed(2)}<br>Se: ${(cutoffPoint.se * 100).toFixed(1)}%<br>Sp: ${(cutoffPoint.sp * 100).toFixed(1)}%<extra></extra>`,
                }] : []),
              ]}
              layout={{
                ...PLOT_LAYOUT,
                xaxis: {
                  ...(PLOT_LAYOUT.xaxis as object), showgrid: showGrid,
                  title: { text: "1 − Specificity (FPR)", font: { color: "#374151", size: 12 } },
                  range: [0, 1], zeroline: false,
                },
                yaxis: {
                  ...(PLOT_LAYOUT.yaxis as object), showgrid: showGrid,
                  title: { text: "Sensitivity (TPR)", font: { color: "#374151", size: 12 } },
                  range: [0, 1.05], zeroline: false,
                },
                autosize: true,
                title: { text: `ROC — ${scoreCol} predicting ${outcomeCol}`, font: { color: "#374151", size: 13 } },
                showlegend: false,
                annotations,
              }}
              onInitialized={(_: object, gd: HTMLElement) => { rocPlotRef.current = gd; rocSingleRef.current = gd; }}
              onUpdate={(_: object, gd: HTMLElement)      => { rocPlotRef.current = gd; rocSingleRef.current = gd; }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
            />
            </div>
            );
          })()}

          {/* ── DeLong comparison plot (publication quality) ── */}
          {mode === "single" && cmpResult && cmpResult.curve_1 && cmpResult.curve_2 && (
            <div className="relative" style={{ width: "100%", height: "100%" }}>
            <PlotExporter plotRef={rocCompareRef} title="ROC_DeLong_Comparison" />
            <Plot
              data={[
                // Baseline model (blue dashed)
                {
                  type: "scatter", mode: "lines",
                  x: cmpResult.curve_2.map((p: any) => p.fpr),
                  y: cmpResult.curve_2.map((p: any) => p.tpr),
                  line: { color: "#2563eb", width: 2.5, dash: "dash" },
                  name: `Baseline (${cmpResult.score_2}): AUC = ${cmpResult.auc_2.toFixed(3)} (${cmpResult.ci_2_low.toFixed(3)}–${cmpResult.ci_2_high.toFixed(3)})`,
                },
                // New model (red solid)
                {
                  type: "scatter", mode: "lines",
                  x: cmpResult.curve_1.map((p: any) => p.fpr),
                  y: cmpResult.curve_1.map((p: any) => p.tpr),
                  line: { color: "#e11d48", width: 3, dash: "solid" },
                  name: `New model (${cmpResult.score_1}): AUC = ${cmpResult.auc_1.toFixed(3)} (${cmpResult.ci_1_low.toFixed(3)}–${cmpResult.ci_1_high.toFixed(3)})`,
                },
                // Chance line
                {
                  type: "scatter", mode: "lines",
                  x: [0, 1], y: [0, 1],
                  line: { color: "#9ca3af", width: 1.5, dash: "dot" },
                  name: "Reference (chance)",
                  showlegend: true,
                },
              ]}
              layout={{
                ...PLOT_LAYOUT,
                xaxis: { ...(PLOT_LAYOUT.xaxis as object), showgrid: showGrid },
                yaxis: { ...(PLOT_LAYOUT.yaxis as object), showgrid: showGrid },
                autosize: true,
                title: {
                  text: `ROC Analysis: Model Comparison — ${cmpResult.score_1} vs. ${cmpResult.score_2}`,
                  font: { color: "#374151", size: 13 },
                },
                legend: {
                  font: { color: "#374151", size: 11 },
                  bgcolor: "rgba(249,250,251,0.95)",
                  bordercolor: "#e5e7eb", borderwidth: 1,
                  x: 0.5, y: 0.03,
                  xanchor: "left" as const, yanchor: "bottom" as const,
                },
                annotations: [
                  // DeLong p-value box inside plot (top-left — journal standard)
                  {
                    x: 0.02, y: 0.98,
                    xref: "paper" as const, yref: "paper" as const,
                    xanchor: "left" as const, yanchor: "top" as const,
                    text: [
                      `<b>ΔAUC = ${cmpResult.difference > 0 ? "+" : ""}${cmpResult.difference.toFixed(3)}</b>`,
                      `95% CI: ${cmpResult.ci_diff_low.toFixed(3)} to ${cmpResult.ci_diff_high.toFixed(3)}`,
                      `DeLong p ${cmpResult.p < 0.001 ? "< 0.001" : "= " + cmpResult.p.toFixed(3)}`,
                    ].join("<br>"),
                    showarrow: false,
                    font: { color: cmpResult.significant ? "#15803d" : "#6b7280", size: 11 },
                    bgcolor: cmpResult.significant ? "rgba(240,253,244,0.95)" : "rgba(249,250,251,0.95)",
                    bordercolor: cmpResult.significant ? "#86efac" : "#e5e7eb",
                    borderwidth: 1, borderpad: 6,
                    align: "left" as const,
                  },
                ],
              }}
              onInitialized={(_: object, gd: HTMLElement) => { rocPlotRef.current = gd; rocCompareRef.current = gd; }}
              onUpdate={(_: object, gd: HTMLElement)      => { rocPlotRef.current = gd; rocCompareRef.current = gd; }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
            />
            </div>
          )}

          {/* ── Multi-curve plot ── */}
          {mode === "multi" && multiResults.length > 0 && (
            <div className="relative" style={{ width: "100%", height: "100%" }}>
            <PlotExporter plotRef={rocMultiRef} title="ROC_Multi_Curve" />
            <Plot
              data={multiTraces as any}
              layout={{
                ...PLOT_LAYOUT,
                xaxis: { ...(PLOT_LAYOUT.xaxis as object), showgrid: showGrid },
                yaxis: { ...(PLOT_LAYOUT.yaxis as object), showgrid: showGrid },
                autosize: true,
                title: { text: `ROC Curves → ${outcomeCol}`, font: { color: "#374151", size: 13 } },
                legend: {
                  font: { color: "#374151", size: 11 },
                  bgcolor: "rgba(249,250,251,0.95)",
                  bordercolor: "#e5e7eb", borderwidth: 1,
                  x: 0.5, y: 0.05, xanchor: "left" as const, yanchor: "bottom" as const,
                },
              }}
              onInitialized={(_: object, gd: HTMLElement) => { rocPlotRef.current = gd; rocMultiRef.current = gd; }}
              onUpdate={(_: object, gd: HTMLElement)      => { rocPlotRef.current = gd; rocMultiRef.current = gd; }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
            />
            </div>
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

      {/* ── Right results column ───────────────────────────────────────────── */}
      {/* Single-mode results (AUC tile + paragraph + metric table) and the
          DeLong comparison card moved here from the left sidebar so the
          chart no longer dominates the viewport and the result numbers
          have a proper column to live in. Only renders for single mode. */}
      {mode === "single" && (
        <div className="w-[380px] flex-shrink-0 flex flex-col gap-3 overflow-y-auto">
          {result && (
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">Results</h3>
                {singleExport && (
                  <ResultExporter
                    title={`ROC_${scoreCol}_vs_${outcomeCol}`}
                    headers={singleExport.headers}
                    rows={singleExport.rows}
                    plotRef={rocPlotRef}
                  />
                )}
              </div>

              <div className="flex flex-col items-center bg-gray-50 border border-gray-200 rounded-lg py-3">
                <span className="text-xs text-gray-400 mb-0.5 flex items-center">
                  AUC
                  <Tip text="P(score₊ > score₋). 0.5 = chance, 1.0 = perfect. ≥0.9 Excellent · ≥0.8 Good · ≥0.7 Fair." />
                </span>
                <span className={`text-2xl font-bold font-mono ${aucColor(result.auc)}`}>{result.auc}</span>
                <span className={`text-xs mt-0.5 ${aucColor(result.auc)}`}>{aucLabel(result.auc)}</span>
                {result.ci_lower != null && (
                  <span className="text-[10px] text-gray-400 mt-0.5">
                    95% CI {result.ci_lower} – {result.ci_upper}
                  </span>
                )}
                {result.auc_p != null && (
                  <span className="text-[10px] text-gray-500 mt-0.5">
                    H₀: AUC = 0.5{" "}
                    <span className="text-gray-400">|</span>{" "}
                    Z = {result.auc_z}{" "}
                    <span className="text-gray-400">|</span>{" "}
                    <span className={`font-mono ${result.auc_p < 0.05 ? "text-emerald-700 font-semibold" : "text-gray-500"}`}>
                      p = {fmtP(result.auc_p)}
                    </span>
                  </span>
                )}
              </div>

              {result.result_text && (
                <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 mt-2">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-semibold text-gray-400 uppercase">Results Paragraph</span>
                    <button onClick={() => navigator.clipboard.writeText(result.result_text)} className="text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
                  </div>
                  <p className="text-sm text-gray-700 leading-relaxed">{result.result_text}</p>
                </div>
              )}
              {result.n_excluded != null && result.n_excluded > 0 && (
                <InfoBanner>
                  {result.n_excluded} row{result.n_excluded !== 1 ? "s" : ""} excluded due to missing values
                  {result.imputation && result.imputation !== "listwise" ? ` (${result.imputation} imputation applied)` : " (listwise deletion)"}.
                </InfoBanner>
              )}
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
                    onClick={() => setUseManual(false)}>
                    Youden J
                    {!useManual && <Tip text="Maximises Sens + Spec − 1 (Youden J)." />}
                  </button>
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

          {/* DeLong comparison moved back to the left sidebar — find it next
              to the single-mode controls. */}
        </div>
      )}
    </div>
  );
}
