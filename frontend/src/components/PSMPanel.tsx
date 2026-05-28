/**
 * PSMPanel — Propensity Score Matching
 *
 * Pipeline:
 * 1. Logistic regression (Treatment ~ Covariates) → propensity scores
 * 2. Nearest-neighbor 1:1 matching with caliper (0.2 × SD of PS)
 * 3. SMD balance assessment before / after
 * 4. Love Plot — the publication-standard balance visualization
 * 5. Optional outcome analysis on matched cohort
 */
import { useState, useRef, useMemo } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { runPSM, runIPTW } from "../api";
import { Tip } from "./Tip";
import PlotExporter from "./PlotExporter";
import ResultExporter from "./ResultExporter";
import { fmtP } from "../lib/format";

// ── helpers ──────────────────────────────────────────────────────────────────
const smdColor = (smd: number) =>
  smd < 0.10 ? "text-emerald-600" : smd < 0.20 ? "text-amber-500" : "text-red-500";

const PLOT_BASE = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#f9fafb",
  font: { color: "#374151", size: 11 },
  margin: { t: 30, r: 24, b: 56, l: 130 },
};

// ── LovePlot ─────────────────────────────────────────────────────────────────
function LovePlot({
  smdBefore,
  smdAfter,
  threshold,
  showConnectors,
  showGrid,
}: {
  smdBefore: Record<string, number>;
  smdAfter: Record<string, number>;
  threshold: number;
  showConnectors: boolean;
  showGrid: boolean;
}) {
  const plotRef = useRef<any>(null);
  const covariates = Object.keys(smdBefore).reverse(); // bottom-to-top

  const xMax = Math.max(0.4, ...Object.values(smdBefore), ...Object.values(smdAfter)) * 1.15;

  const traces: any[] = [
    // Unmatched (red squares)
    {
      type: "scatter",
      mode: "markers",
      name: "Unmatched cohort",
      x: covariates.map((c) => smdBefore[c]),
      y: covariates,
      marker: { symbol: "square", size: 11, color: "#ef4444" },
      hovertemplate: "<b>%{y}</b><br>SMD (before) = %{x:.4f}<extra></extra>",
    },
    // Matched (blue circles)
    {
      type: "scatter",
      mode: "markers",
      name: "Matched cohort",
      x: covariates.map((c) => smdAfter[c]),
      y: covariates,
      marker: { symbol: "circle", size: 11, color: "#3b82f6" },
      hovertemplate: "<b>%{y}</b><br>SMD (after) = %{x:.4f}<extra></extra>",
    },
  ];

  // Connector lines
  if (showConnectors) {
    for (const cov of covariates) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x: [smdBefore[cov], smdAfter[cov]],
        y: [cov, cov],
        line: { color: "#94a3b8", width: 1.2, dash: "dot" },
        showlegend: false,
        hoverinfo: "skip",
      });
    }
  }

  const layout: any = {
    ...PLOT_BASE,
    autosize: true,
    height: Math.max(260, covariates.length * 52 + 80),
    xaxis: {
      title: { text: "Standardized Mean Difference (SMD)" },
      range: [0, xMax],
      gridcolor: showGrid ? "#e5e7eb" : "transparent",
      zeroline: false,
    },
    yaxis: {
      gridcolor: "transparent",
      automargin: true,
    },
    legend: {
      x: 1, y: 0, xanchor: "right", yanchor: "bottom",
      bgcolor: "rgba(249,250,251,0.9)",
      bordercolor: "#e5e7eb", borderwidth: 1,
      font: { size: 11 },
    },
    shapes: [
      // Threshold vertical line
      {
        type: "line",
        x0: threshold, x1: threshold,
        y0: 0, y1: 1,
        xref: "x", yref: "paper",
        line: { color: "#dc2626", width: 1.5, dash: "dash" },
      },
    ],
    annotations: [
      {
        x: threshold, y: 1.02,
        xref: "x", yref: "paper",
        text: `Threshold (${threshold})`,
        showarrow: false,
        font: { color: "#dc2626", size: 10 },
        xanchor: "center",
      },
    ],
  };

  return (
    <div className="relative">
      <Plot
        ref={plotRef}
        data={traces}
        layout={layout}
        style={{ width: "100%", height: layout.height }}
        useResizeHandler
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        onInitialized={(_: any, gd: any) => { plotRef.current = gd; }}
        onUpdate={(_: any, gd: any) => { plotRef.current = gd; }}
      />
      <PlotExporter plotRef={plotRef} title="Love_Plot_PSM" />
    </div>
  );
}

// ── PSOverlapPlot ─────────────────────────────────────────────────────────────
function PSOverlapPlot({
  psDist,
  showGrid,
}: {
  psDist: { treated_unmatched: number[]; control_unmatched: number[]; treated_matched: number[]; control_matched: number[] };
  showGrid: boolean;
}) {
  const plotRef = useRef<any>(null);
  return (
    <div className="relative">
      <Plot
        ref={plotRef}
        data={[
          {
            type: "histogram",
            name: "Treated (unmatched)",
            x: psDist.treated_unmatched,
            opacity: 0.55,
            marker: { color: "#ef4444" },
            nbinsx: 25,
            hovertemplate: "PS: %{x:.3f}<br>Count: %{y}<extra></extra>",
          },
          {
            type: "histogram",
            name: "Control (unmatched)",
            x: psDist.control_unmatched,
            opacity: 0.55,
            marker: { color: "#3b82f6" },
            nbinsx: 25,
            hovertemplate: "PS: %{x:.3f}<br>Count: %{y}<extra></extra>",
          },
        ]}
        layout={{
          ...PLOT_BASE,
          barmode: "overlay",
          autosize: true,
          height: 230,
          xaxis: {
            title: { text: "Propensity Score" },
            range: [0, 1],
            gridcolor: showGrid ? "#e5e7eb" : "transparent",
          },
          yaxis: { title: { text: "Count" }, gridcolor: showGrid ? "#e5e7eb" : "transparent" },
          legend: { x: 1, y: 1, xanchor: "right", bgcolor: "rgba(249,250,251,0.9)", bordercolor: "#e5e7eb", borderwidth: 1 },
          annotations: [{
            x: 0.5, y: 1.08, xref: "paper", yref: "paper",
            text: "Propensity Score Overlap (Common Support)",
            showarrow: false, font: { color: "#374151", size: 12 },
          }],
        } as any}
        style={{ width: "100%", height: 230 }}
        useResizeHandler
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        onInitialized={(_: any, gd: any) => { plotRef.current = gd; }}
        onUpdate={(_: any, gd: any) => { plotRef.current = gd; }}
      />
      <PlotExporter plotRef={plotRef} title="PSM_Propensity_Overlap" />
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function PSMPanel() {
  const session  = useStore((s) => s.session);
  const showGrid = useStore((s) => s.showGrid);
  if (!session) return null;

  const allCols = session.columns.map((c) => c.name);

  // Detect binary cols (0/1) from preview
  const binaryCols = useMemo(() =>
    allCols.filter((col) => {
      const vals = new Set(session.preview.map((r) => r[col]).filter((v) => v != null));
      return vals.size === 2 && [...vals].every((v) => v === 0 || v === 1);
    }),
    [session.session_id]
  );

  // Form state
  const [treatCol,   setTreatCol]   = useState(binaryCols[0] ?? allCols[0] ?? "");
  const [outcomeCol, setOutcomeCol] = useState("");
  const [covariates, setCovariates] = useState<string[]>([]);
  const [caliper,    setCaliper]    = useState(0.2);
  const [caliperScale, setCaliperScale] = useState<"logit" | "raw">("logit");
  const [trimCommonSupport, setTrimCommonSupport] = useState(false);
  const [ratio,        setRatio]        = useState(1);
  const [randomState,  setRandomState]  = useState<number>(42);
  const [scoreMethod,  setScoreMethod]  = useState<"logistic" | "probit" | "gbm">("logistic");
  const [matchingMethod, setMatchingMethod] = useState<"greedy" | "optimal">("greedy");
  const [exactMatch,   setExactMatch]   = useState<string[]>([]);
  const [outcomeType,  setOutcomeType]  = useState<"binary" | "survival">("binary");
  const [survDuration, setSurvDuration] = useState("");
  const [survEvent,    setSurvEvent]    = useState("");
  const [computeRosenbaum, setComputeRosenbaum] = useState(false);
  const [rosenbaumGammaMax, setRosenbaumGammaMax] = useState<number>(3.0);
  void setRosenbaumGammaMax;
  const [covFilter,  setCovFilter]  = useState("");

  // ── IPTW-only form state ────────────────────────────────────────────────
  const [method,            setMethod]            = useState<"psm" | "iptw">("psm");
  const [estimand,          setEstimand]          = useState<"ate" | "att" | "overlap">("ate");
  const [stabilize,         setStabilize]         = useState(true);
  const [weightTruncation,  setWeightTruncation]  = useState<"percentile" | "hard" | "none">("percentile");
  const [weightTruncLo,     setWeightTruncLo]     = useState(0.01);
  const [weightTruncHi,     setWeightTruncHi]     = useState(0.99);
  const [weightTruncMax,    setWeightTruncMax]    = useState(10);
  const [seMethod,          setSeMethod]          = useState<"robust" | "bootstrap">("robust");
  const [bootstrapReps,     setBootstrapReps]     = useState(500);

  // Result & UI
  const [result,         setResult]         = useState<any>(null);
  const [loading,        setLoading]        = useState(false);
  const [error,          setError]          = useState<string | null>(null);
  const [threshold,      setThreshold]      = useState(0.10);
  const [showConnectors, setShowConnectors] = useState(true);

  const toggleCov = (c: string) =>
    setCovariates((p) => (p.includes(c) ? p.filter((x) => x !== c) : [...p, c]));

  const run = async () => {
    if (covariates.length === 0) { setError("Select at least one covariate"); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      let res: any;
      if (method === "psm") {
        res = await runPSM({
          session_id:    session.session_id,
          treatment_col: treatCol,
          covariates,
          outcome_col:   outcomeType === "binary" ? (outcomeCol || undefined) : undefined,
          caliper,
          caliper_scale: caliperScale,
          trim_common_support: trimCommonSupport,
          ratio,
          random_state: Number.isFinite(randomState) ? randomState : undefined,
          score_method:    scoreMethod,
          matching_method: matchingMethod,
          exact_match:     exactMatch.length > 0 ? exactMatch : undefined,
          outcome_type:    outcomeType,
          survival_duration_col: outcomeType === "survival" ? (survDuration || undefined) : undefined,
          survival_event_col:    outcomeType === "survival" ? (survEvent || undefined) : undefined,
          compute_rosenbaum: outcomeType === "binary" && ratio === 1 && computeRosenbaum,
          rosenbaum_gamma_max: rosenbaumGammaMax,
        });
      } else {
        res = await runIPTW({
          session_id:    session.session_id,
          treatment_col: treatCol,
          covariates,
          outcome_col:   outcomeType === "binary" ? (outcomeCol || undefined) : undefined,
          imputation:    undefined,
          random_state:  Number.isFinite(randomState) ? randomState : undefined,
          score_method:  scoreMethod,
          estimand,
          stabilize,
          trim_common_support: trimCommonSupport,
          weight_truncation: weightTruncation,
          weight_truncation_lo:  weightTruncLo,
          weight_truncation_hi:  weightTruncHi,
          weight_truncation_max: weightTruncMax,
          outcome_type:  outcomeType,
          survival_duration_col: outcomeType === "survival" ? (survDuration || undefined) : undefined,
          survival_event_col:    outcomeType === "survival" ? (survEvent || undefined) : undefined,
          se_method:     seMethod,
          bootstrap_reps: seMethod === "bootstrap" ? bootstrapReps : undefined,
        });
      }
      setResult(res.data);
    } catch (e: any) {
      const msg = e.response?.data?.detail;
      setError(typeof msg === "string" ? msg : (e.message ?? `${method.toUpperCase()} failed`));
    } finally { setLoading(false); }
  };

  // Build export data for SMD table
  const smdExportHeaders = ["Covariate", "SMD Before", "SMD After", "Reduction %", "Balanced (<0.10)"];
  const smdExportRows = result
    ? Object.keys(result.smd_before).map((cov) => [
        cov,
        result.smd_before[cov].toFixed(4),
        result.smd_after[cov].toFixed(4),
        (((result.smd_before[cov] - result.smd_after[cov]) / result.smd_before[cov]) * 100).toFixed(1) + "%",
        result.smd_after[cov] < 0.10 ? "Yes" : "No",
      ])
    : [];

  const availableCovs = allCols.filter(
    (c) => c !== treatCol && c !== outcomeCol &&
           c.toLowerCase().includes(covFilter.toLowerCase())
  );

  return (
    <div className="flex gap-4 min-h-0">

      {/* ── Left sidebar ─────────────────────────────────────────────────── */}
      <div className="w-64 flex-shrink-0 space-y-3 overflow-y-auto">

        {/* Header */}
        <div className="panel bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-200 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xl">🧬</span>
            <h2 className="text-sm font-bold text-indigo-800">
              {method === "psm" ? "Propensity Score Matching" : "Inverse Probability of Treatment Weighting"}
            </h2>
          </div>
          <p className="text-[10px] text-indigo-600 leading-snug">
            {method === "psm"
              ? "Mimics an RCT from observational data by balancing confounders between treated and control groups via 1:k matching."
              : "Reweights every unit by the inverse of its propensity score. Keeps the full sample; supports ATE / ATT / overlap estimands; pairs with weighted GLM or weighted Cox."}
          </p>
          {/* Method toggle */}
          <div className="flex gap-1 pt-1 border-t border-indigo-200">
            {(["psm", "iptw"] as const).map((m) => (
              <button key={m}
                onClick={() => { setMethod(m); setResult(null); setError(null); }}
                className={`flex-1 px-2 py-1 text-[10px] rounded transition-colors ${
                  method === m
                    ? "bg-indigo-600 text-white font-semibold shadow-sm"
                    : "bg-white border border-indigo-200 text-indigo-600 hover:bg-indigo-100"
                }`}>
                {m === "psm" ? "Matching (PSM)" : "Weighting (IPTW)"}
              </button>
            ))}
          </div>
        </div>

        {/* ── IPTW-only controls ─────────────────────────────────────────── */}
        {method === "iptw" && (
          <div className="panel space-y-3">
            <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
              IPTW options
              <Tip wide text="Choose the estimand (target population), whether to stabilise the weights, and how to truncate the upper tail. Stabilisation rescales by P(T=1) to bring weights closer to 1 and reduces SE inflation without changing the point estimate." />
            </label>

            <div className="space-y-1">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Estimand
                <Tip wide text="ATE — average effect across the whole population (w = T/ps + (1−T)/(1−ps)). ATT — effect on those actually exposed (same target as PSM 1:1, w = T + (1−T)·ps/(1−ps)). Overlap (Crump 2009) — concentrates inference on the overlap region (w_T = 1−ps, w_C = ps); naturally bounded and robust to extreme PS." />
              </span>
              <div className="flex gap-1">
                {(["ate", "att", "overlap"] as const).map((e) => (
                  <button key={e} onClick={() => setEstimand(e)}
                    className={`flex-1 px-2 py-1 text-[10px] rounded border transition-colors ${estimand === e ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                    {e === "ate" ? "ATE ★" : e === "att" ? "ATT" : "Overlap"}
                  </button>
                ))}
              </div>
            </div>

            <label className="flex items-start gap-2 cursor-pointer">
              <input type="checkbox" className="accent-indigo-500 mt-0.5"
                checked={stabilize} onChange={(e) => setStabilize(e.target.checked)} />
              <span className="text-[10px] text-gray-600 leading-tight">
                <span className="font-medium">Stabilised weights</span>
                <Tip wide text="Rescales weights by the marginal P(T=1) (and 1−P(T=1) for controls). Standard recommendation since Hernán & Robins. Reduces extreme-weight impact, same point estimate." />
                <span className="block text-gray-400">Multiplies by marginal P(T=1)</span>
              </span>
            </label>

            <div className="space-y-1 pt-2 border-t border-gray-100">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Weight truncation
                <Tip wide text="Truncating the weight distribution controls the influence of units near the PS support boundaries. Percentile (e.g. 1st/99th) is the Cole & Hernán recommendation. Hard cap clips at an absolute maximum (e.g. 10). None preserves all weights — useful for diagnostics." />
              </span>
              <div className="flex gap-1">
                {(["percentile", "hard", "none"] as const).map((m) => (
                  <button key={m} onClick={() => setWeightTruncation(m)}
                    className={`flex-1 px-2 py-1 text-[10px] rounded border transition-colors ${weightTruncation === m ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                    {m === "percentile" ? "Pctile ★" : m === "hard" ? "Hard" : "None"}
                  </button>
                ))}
              </div>
              {weightTruncation === "percentile" && (
                <div className="grid grid-cols-2 gap-1.5">
                  <label className="flex items-center gap-1 text-[10px] text-gray-500">
                    Lo
                    <input type="number" min="0" max="0.5" step="0.005" className="select text-[10px] py-0.5 flex-1"
                      value={weightTruncLo}
                      onChange={(e) => setWeightTruncLo(parseFloat(e.target.value))} />
                  </label>
                  <label className="flex items-center gap-1 text-[10px] text-gray-500">
                    Hi
                    <input type="number" min="0.5" max="1" step="0.005" className="select text-[10px] py-0.5 flex-1"
                      value={weightTruncHi}
                      onChange={(e) => setWeightTruncHi(parseFloat(e.target.value))} />
                  </label>
                </div>
              )}
              {weightTruncation === "hard" && (
                <label className="flex items-center gap-1 text-[10px] text-gray-500">
                  Max weight
                  <input type="number" min="1" step="0.5" className="select text-[10px] py-0.5 flex-1"
                    value={weightTruncMax}
                    onChange={(e) => setWeightTruncMax(parseFloat(e.target.value))} />
                </label>
              )}
            </div>

            <div className="space-y-1 pt-2 border-t border-gray-100">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                  Score model
                  <Tip wide text="Model used to estimate the propensity score. Logistic = standard parametric. Probit = same shape, different link. GBM = gradient-boosted trees, captures non-linear / interaction effects without manual specification but may overfit on small samples." />
                </span>
                <select className="select text-[10px] py-0.5" value={scoreMethod}
                  onChange={(e) => setScoreMethod(e.target.value as any)}>
                  <option value="logistic">Logistic ★</option>
                  <option value="probit">Probit</option>
                  <option value="gbm">GBM</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                  Seed
                  <Tip text="Random seed for the propensity-score solver. Use any integer for reproducibility." />
                </span>
                <input type="number" className="select text-[10px] py-0.5 flex-1"
                  value={randomState} onChange={(e) => setRandomState(parseInt(e.target.value, 10))} />
              </div>
              <label className="flex items-start gap-2 cursor-pointer">
                <input type="checkbox" className="accent-indigo-500 mt-0.5"
                  checked={trimCommonSupport} onChange={(e) => setTrimCommonSupport(e.target.checked)} />
                <span className="text-[10px] text-gray-600 leading-tight">
                  <span className="font-medium">Trim to common support</span>
                  <Tip wide text="Crump et al. (2009): exclude treated and control units with PS outside the overlap region BEFORE weighting. Recommended when the two groups' PS distributions only partially overlap." />
                  <span className="block text-gray-400">Crump 2009 trimming (pre-weight)</span>
                </span>
              </label>
            </div>

            <div className="space-y-1 pt-2 border-t border-gray-100">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Standard error
                <Tip wide text="Robust — HC1 sandwich for GLM, Lin & Wei sandwich for Cox. Bootstrap — refit propensity scores and outcome model on resampled subjects (500 reps default) and use the percentile interval. Slower but more conservative when weights are extreme." />
              </span>
              <div className="flex gap-1">
                {(["robust", "bootstrap"] as const).map((m) => (
                  <button key={m} onClick={() => setSeMethod(m)}
                    className={`flex-1 px-2 py-1 text-[10px] rounded border transition-colors ${seMethod === m ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                    {m === "robust" ? "Robust ★" : "Bootstrap"}
                  </button>
                ))}
              </div>
              {seMethod === "bootstrap" && (
                <label className="flex items-center gap-1 text-[10px] text-gray-500">
                  Reps
                  <input type="number" min="50" max="5000" step="50" className="select text-[10px] py-0.5 flex-1"
                    value={bootstrapReps}
                    onChange={(e) => setBootstrapReps(parseInt(e.target.value, 10))} />
                </label>
              )}
            </div>
          </div>
        )}

        {/* Treatment variable */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Treatment Variable
            <Tip wide text="The binary intervention variable: 1 = Treated, 0 = Control. Examples: TAVI vs Open Surgery, Drug A vs Drug B. Must be coded 0/1." />
          </label>
          <select
            className="select w-full"
            value={treatCol}
            onChange={(e) => { setTreatCol(e.target.value); setResult(null); }}>
            {binaryCols.length > 0
              ? binaryCols.map((c) => <option key={c} value={c}>{c}</option>)
              : allCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          {!binaryCols.includes(treatCol) && (
            <p className="text-[10px] text-amber-600">⚠ Column may not be binary (0/1)</p>
          )}
        </div>

        {/* Outcome variable */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Outcome Variable <span className="normal-case font-normal text-gray-400">(optional)</span>
            <Tip wide text="The endpoint to analyse in the matched cohort (e.g. EXITUS, 30-day mortality). If binary, logistic regression is run automatically. Leave blank to only assess balance." />
          </label>
          <select
            className="select w-full"
            value={outcomeCol}
            onChange={(e) => { setOutcomeCol(e.target.value); setResult(null); }}>
            <option value="">— Skip outcome analysis —</option>
            {allCols.filter((c) => c !== treatCol).map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        {/* Covariates */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Covariates (Confounders)
            <Tip wide text="Baseline patient characteristics that influence both the treatment decision AND the outcome. Examples: Age, Sex, EF, Diabetes, Hypertension. Include all known confounders — omitting one biases the propensity score." />
          </label>
          <input
            type="text"
            placeholder="Filter covariates…"
            className="select w-full text-xs py-1"
            value={covFilter}
            onChange={(e) => setCovFilter(e.target.value)}
          />
          <div className="flex gap-1">
            <button onClick={() => setCovariates(availableCovs)}
              className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">All</button>
            <button onClick={() => setCovariates([])}
              className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">None</button>
          </div>
          <div className="max-h-52 overflow-y-auto space-y-0.5 border border-gray-200 rounded-lg p-1">
            {availableCovs.map((c) => (
              <label key={c} className="flex items-center gap-1.5 text-xs px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                <input type="checkbox" className="accent-indigo-500"
                  checked={covariates.includes(c)} onChange={() => toggleCov(c)} />
                <span className="text-gray-700 truncate">{c}</span>
              </label>
            ))}
          </div>
          <p className="text-[10px] text-gray-400">{covariates.length} selected</p>
        </div>

        {/* Caliper (PSM only) */}
        {method === "psm" && (
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Caliper
            <Tip wide text="Maximum allowed PS distance for a match, expressed as a fraction of the SD of propensity scores. Medical standard = 0.2 (Cochran & Rubin, 1973). Tighter caliper = better balance but more unmatched patients." />
          </label>
          <div className="flex gap-2 items-center">
            <input type="range" min="0.05" max="0.50" step="0.05"
              className="flex-1 accent-indigo-500"
              value={caliper}
              onChange={(e) => setCaliper(parseFloat(e.target.value))} />
            <span className="font-mono text-sm font-semibold text-indigo-700 w-10 text-right">{caliper}</span>
          </div>
          <div className="flex justify-between text-[9px] text-gray-400">
            <span>0.05 (strict)</span>
            <span className="text-indigo-500">0.20 ★ standard</span>
            <span>0.50 (loose)</span>
          </div>
          <div className="pt-2 border-t border-gray-100 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Caliper scale
                <Tip wide text="Austin (2011) recommends matching on the LOGIT of the propensity score: the raw PS is bounded [0,1] and gets compressed near the tails, so a constant caliper is far too loose at extreme values. 'Logit' is the publication standard." />
              </span>
              <div className="flex gap-1">
                {(["logit", "raw"] as const).map((s) => (
                  <button key={s} onClick={() => setCaliperScale(s)}
                    className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${caliperScale === s ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                    {s === "logit" ? "Logit ★" : "Raw PS"}
                  </button>
                ))}
              </div>
            </div>
            <label className="flex items-start gap-2 cursor-pointer">
              <input type="checkbox" className="accent-indigo-500 mt-0.5"
                checked={trimCommonSupport} onChange={(e) => setTrimCommonSupport(e.target.checked)} />
              <span className="text-[10px] text-gray-600 leading-tight">
                <span className="font-medium">Trim to common support</span>
                <Tip wide text="Crump et al. (2009): exclude treated and control units with PS outside the overlap region [max(min_treated, min_control), min(max_treated, max_control)] BEFORE matching. Recommended when the two groups' PS distributions only partially overlap." />
                <span className="block text-gray-400">Crump 2009 trimming</span>
              </span>
            </label>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-gray-500 font-medium">Ratio</span>
              <select className="select text-[10px] py-0.5 flex-1" value={ratio} onChange={(e) => setRatio(parseInt(e.target.value, 10))}>
                {[1, 2, 3, 4, 5].map((k) => <option key={k} value={k}>1 : {k}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Seed
                <Tip text="Random seed for the propensity-score logistic regression solver. Use any integer for reproducibility." />
              </span>
              <input type="number" className="select text-[10px] py-0.5 flex-1"
                value={randomState} onChange={(e) => setRandomState(parseInt(e.target.value, 10))} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Score model
                <Tip wide text="Model used to estimate the propensity score. Logistic = standard parametric. Probit = same shape, different link. GBM = gradient-boosted trees, captures non-linear / interaction effects without manual specification but may overfit on small samples." />
              </span>
              <select className="select text-[10px] py-0.5" value={scoreMethod}
                onChange={(e) => setScoreMethod(e.target.value as any)}>
                <option value="logistic">Logistic ★</option>
                <option value="probit">Probit</option>
                <option value="gbm">GBM</option>
              </select>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Matching
                <Tip wide text="Greedy: nearest neighbour with caliper, hardest-first. Standard practice. Optimal: Hungarian algorithm minimises total within-pair distance (1:1 only). Slower but yields globally best matches. Higher ratios fall back to greedy automatically." />
              </span>
              <div className="flex gap-1">
                {(["greedy", "optimal"] as const).map((m) => (
                  <button key={m} onClick={() => setMatchingMethod(m)}
                    className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${matchingMethod === m ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                    {m === "greedy" ? "Greedy ★" : "Optimal"}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-1">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Exact match strata
                <Tip wide text="Categorical columns where treated and control units MUST agree before nearest-neighbour matching runs. Common choices: sex, study site, year of enrolment. Leave empty to match across all strata." />
              </span>
              <div className="max-h-20 overflow-y-auto border border-gray-200 rounded p-1 space-y-0.5">
                {allCols.filter((c) => c !== treatCol).slice(0, 100).map((c) => (
                  <label key={c} className="flex items-center gap-1 text-[10px] px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                    <input type="checkbox" className="accent-indigo-500"
                      checked={exactMatch.includes(c)}
                      onChange={() => setExactMatch((p) => p.includes(c) ? p.filter((x) => x !== c) : [...p, c])} />
                    <span className="text-gray-700 truncate">{c}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>
        )}

        {/* Outcome type */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Outcome type
            <Tip wide text="Binary: GEE logistic on the matched cohort (matched-pair clustering). Survival: stratified Cox PH with strata = match set ID — preserves the matched-pair structure for time-to-event endpoints." />
          </label>
          <div className="flex gap-1">
            {(["binary", "survival"] as const).map((t) => (
              <button key={t} onClick={() => { setOutcomeType(t); setResult(null); }}
                className={`flex-1 px-2 py-1 text-[11px] rounded border transition-colors ${outcomeType === t ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                {t === "binary" ? "Binary" : "Survival (Cox)"}
              </button>
            ))}
          </div>
          {outcomeType === "survival" && (
            <div className="space-y-1.5">
              <div>
                <span className="text-[10px] text-gray-500 font-medium">Duration column</span>
                <select className="select w-full text-xs py-1" value={survDuration} onChange={(e) => setSurvDuration(e.target.value)}>
                  <option value="">— select —</option>
                  {allCols.filter((c) => c !== treatCol).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <span className="text-[10px] text-gray-500 font-medium">Event column (0/1)</span>
                <select className="select w-full text-xs py-1" value={survEvent} onChange={(e) => setSurvEvent(e.target.value)}>
                  <option value="">— select —</option>
                  {allCols.filter((c) => c !== treatCol).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>
          )}
          {method === "psm" && outcomeType === "binary" && ratio === 1 && outcomeCol && (
            <label className="flex items-start gap-2 cursor-pointer pt-1 border-t border-gray-100">
              <input type="checkbox" className="accent-indigo-500 mt-0.5"
                checked={computeRosenbaum} onChange={(e) => setComputeRosenbaum(e.target.checked)} />
              <span className="text-[10px] text-gray-600 leading-tight">
                <span className="font-medium">Rosenbaum bounds</span>
                <Tip wide text="Sensitivity analysis to unmeasured confounding (Rosenbaum 2002). For 1:1 matched discordant pairs with a binary outcome, computes the critical Γ — the size of hidden bias that would just nullify the treatment effect's p < 0.05." />
                <span className="block text-gray-400">1:1 binary only · Γ up to {rosenbaumGammaMax.toFixed(1)}</span>
              </span>
            </label>
          )}
        </div>

        {/* Run */}
        <button
          className="btn-primary w-full py-3 text-sm font-semibold flex items-center justify-center gap-2"
          onClick={run} disabled={loading || covariates.length === 0}>
          {loading
            ? <><span className="animate-spin inline-block">⏳</span> {method === "psm" ? "Matching…" : "Weighting…"}</>
            : <><span>{method === "psm" ? "🔗" : "⚖️"}</span> {method === "psm" ? "Run PSM" : "Run IPTW"}</>}
        </button>
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-3 py-2 text-xs text-red-600">{error}</div>
        )}
      </div>

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 overflow-y-auto space-y-4">

        {result ? (
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_480px] gap-4 auto-rows-min items-start">
            {/* ── Summary banner ── PSM and IPTW share the balance flag
                semantics but report different cohort metrics. */}
            <div className={`panel border-2 xl:col-start-2 ${result.balance_achieved ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <div className="flex items-start gap-3">
                <span className="text-2xl flex-shrink-0">{result.balance_achieved ? "✅" : "⚠️"}</span>
                <div className="flex-1">
                  <p className={`font-bold text-sm ${result.balance_achieved ? "text-emerald-800" : "text-amber-800"}`}>
                    {result.balance_achieved
                      ? "Balance achieved — all SMDs < 0.10. Publication-ready."
                      : result.method === "iptw"
                        ? "Partial balance — some SMDs ≥ 0.10. Consider trimming weights, switching to overlap weights, or adding covariates."
                        : "Partial balance — some SMDs ≥ 0.10. Consider widening caliper or adding covariates."}
                  </p>
                  {result.method === "iptw" ? (
                    <>
                      <p className="text-xs text-gray-600 mt-1">
                        IPTW · {(result.estimand ?? "ate").toUpperCase()} weights
                        {result.stabilize ? " (stabilised)" : ""}
                        {" · "}n = {result.n_total} ({result.n_treated} treated, {result.n_control} control)
                        {result.n_trimmed_common_support > 0 && <> · {result.n_trimmed_common_support} trimmed (common support)</>}
                        {result.weight_truncation?.n_trimmed > 0 && <> · {result.weight_truncation.n_trimmed} weights truncated</>}
                      </p>
                      <p className="text-[11px] text-gray-500 mt-1">
                        Score = {result.score_method} · ESS treated = {result.weight_summary?.ess_treated} · ESS control = {result.weight_summary?.ess_control} · max w = {result.weight_summary?.max}
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="text-xs text-gray-600 mt-1">
                        Matched {result.n_matched_pairs} treated : {result.n_matched_controls} control pairs
                        ({result.n_unmatched} treated patients unmatched and excluded).
                      </p>
                      <p className="text-[11px] text-gray-500 mt-1">
                        Caliper = {result.caliper_used?.toFixed(4)} on the <b>{result.caliper_scale ?? "logit"}</b> scale ({caliper} × SD = {result.caliper_used?.toFixed(4)}).
                        {result.n_trimmed_common_support > 0 && (
                          <> · {result.n_trimmed_common_support} units trimmed (common support [{result.common_support?.lo?.toFixed(3)}, {result.common_support?.hi?.toFixed(3)}]).</>
                        )}
                      </p>
                    </>
                  )}
                </div>
              </div>

              {/* Key stats */}
              <div className="grid grid-cols-5 gap-2 mt-3">
                {(result.method === "iptw"
                  ? [
                      { label: "Total N",       val: result.n_total },
                      { label: "Treated",       val: result.n_treated },
                      { label: "Controls",      val: result.n_control },
                      { label: "ESS Treated",   val: result.weight_summary?.ess_treated, highlight: true },
                      { label: "ESS Control",   val: result.weight_summary?.ess_control, highlight: true },
                    ]
                  : [
                      { label: "Total N",         val: result.n_total },
                      { label: "Treated",         val: result.n_treated },
                      { label: "Controls",        val: result.n_control },
                      { label: "Matched Pairs",   val: result.n_matched_pairs, highlight: true },
                      { label: "Unmatched",       val: result.n_unmatched, warn: result.n_unmatched > 0 },
                    ]
                ).map(({ label, val, highlight, warn }: any) => (
                  <div key={label} className={`rounded-lg px-2 py-2 text-center border ${
                    highlight ? "bg-indigo-50 border-indigo-200" :
                    warn && val > 0 ? "bg-amber-50 border-amber-200" :
                    "bg-white border-gray-200"}`}>
                    <p className="text-[9px] text-gray-400 uppercase tracking-wide">{label}</p>
                    <p className={`text-lg font-bold ${highlight ? "text-indigo-700" : warn && val > 0 ? "text-amber-600" : "text-gray-800"}`}>
                      {val}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* ── IPTW weight distribution ── */}
            {result.method === "iptw" && result.weight_distribution && (
              <div className="panel space-y-2 xl:col-start-1">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-gray-800">Weight Distribution</h3>
                    <p className="text-[10px] text-gray-400">
                      Per-unit IPTW weights by treatment group. Wider tails or large maxima indicate
                      poor PS support — consider truncation or overlap weights.
                    </p>
                  </div>
                  <div className="text-[10px] text-gray-500 text-right">
                    min {result.weight_summary?.min} · median {result.weight_summary?.median} · max {result.weight_summary?.max}
                  </div>
                </div>
                <Plot
                  data={[
                    {
                      type: "histogram", name: "Treated",
                      x: result.weight_distribution.treated,
                      opacity: 0.65, marker: { color: "#6366f1" },
                    },
                    {
                      type: "histogram", name: "Control",
                      x: result.weight_distribution.control,
                      opacity: 0.65, marker: { color: "#10b981" },
                    },
                  ] as any}
                  layout={{
                    ...PLOT_BASE,
                    barmode: "overlay",
                    height: 220, autosize: true,
                    margin: { t: 24, r: 20, b: 40, l: 60 },
                    xaxis: { title: { text: "IPTW weight" } },
                    yaxis: { title: { text: "Count" } },
                    legend: { font: { size: 10 } },
                  }}
                  style={{ width: "100%", height: 220 }}
                  config={{ responsive: true, displaylogo: false }}
                />
              </div>
            )}

            {/* ── Love Plot ── */}
            <div className="panel space-y-3 xl:col-start-1">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <h3 className="text-sm font-bold text-gray-800">Love Plot: Covariate Balance</h3>
                  <p className="text-[10px] text-gray-400">
                    Named after Dr. Thomas Love. Publication-required visual proof of balance. All matched points (blue ●) must lie left of the threshold line.
                  </p>
                </div>
                {/* Avg SMD summary */}
                <div className="flex gap-3">
                  {[
                    { label: "Avg Unmatched SMD", val: result.avg_smd_before, color: "text-red-600" },
                    { label: "Avg Matched SMD",   val: result.avg_smd_after,  color: "text-emerald-600" },
                    { label: "Reduction",          val: `${result.reduction_pct}%`, color: "text-indigo-600" },
                  ].map(({ label, val, color }) => (
                    <div key={label} className="text-center bg-gray-50 border border-gray-200 rounded-lg px-3 py-1.5">
                      <p className="text-[9px] text-gray-400 uppercase tracking-wide">{label}</p>
                      <p className={`text-base font-bold font-mono ${color}`}>
                        {typeof val === "number" ? val.toFixed(3) : val}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              <LovePlot
                smdBefore={result.smd_before}
                smdAfter={result.smd_after}
                threshold={threshold}
                showConnectors={showConnectors}
                showGrid={showGrid}
              />

              {/* Controls below plot */}
              <div className="flex flex-wrap items-center gap-6 pt-2 border-t border-gray-100">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">Balance Threshold</span>
                  <input type="range" min="0.05" max="0.25" step="0.01"
                    className="w-28 accent-indigo-500"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))} />
                  <span className="font-mono text-sm font-bold text-indigo-700 w-10">{threshold.toFixed(2)}</span>
                </div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <span className="text-xs text-gray-500">Show Connectors</span>
                  <div
                    className={`w-9 h-5 rounded-full transition-colors cursor-pointer ${showConnectors ? "bg-indigo-600" : "bg-gray-300"}`}
                    onClick={() => setShowConnectors((v) => !v)}>
                    <div className={`w-4 h-4 bg-white rounded-full shadow mt-0.5 transition-transform ${showConnectors ? "translate-x-4" : "translate-x-0.5"}`} />
                  </div>
                </label>
                <div className="flex items-center gap-3 text-xs ml-auto">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 bg-red-400 rounded-sm inline-block" /> Unmatched</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 bg-blue-500 rounded-full inline-block" /> Matched</span>
                  <span className="flex items-center gap-1"><span className="border-l-2 border-dashed border-red-500 h-3 inline-block" /> Threshold</span>
                </div>
              </div>
            </div>

            {/* ── SMD balance table ── */}
            <div className="panel space-y-2 xl:col-start-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">
                  SMD Balance Table
                  <Tip wide text="Standardized Mean Difference measures imbalance in each covariate between groups. The gold standard for publication: all SMDs after matching must be < 0.10 (Austin, 2009)." />
                </h3>
                <ResultExporter
                  title="PSM_SMD_Balance"
                  headers={smdExportHeaders}
                  rows={smdExportRows}
                />
              </div>
              <div className="overflow-auto rounded-lg border border-gray-200">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                      <th className="text-left px-3 py-2 font-medium">Covariate</th>
                      <th className="text-right px-3 py-2 font-medium">SMD Before</th>
                      <th className="text-right px-3 py-2 font-medium">SMD After</th>
                      <th className="text-right px-3 py-2 font-medium" title="Rubin's variance ratio σ²_treated / σ²_control. Target 0.5–2.0.">Var Ratio</th>
                      <th className="text-right px-3 py-2 font-medium" title="Two-sample KS test p-value after matching. Higher = better distributional balance.">KS p (after)</th>
                      <th className="text-right px-3 py-2 font-medium">Reduction</th>
                      <th className="text-center px-3 py-2 font-medium">Balanced</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(result.smd_before).map((cov) => {
                      const before = result.smd_before[cov];
                      const after  = result.smd_after[cov];
                      const reduction = before > 0 ? ((before - after) / before * 100).toFixed(1) : "—";
                      const balanced = after < threshold;
                      const vr = result.variance_ratio_after?.[cov] as number | null | undefined;
                      const vrBefore = result.variance_ratio_before?.[cov] as number | null | undefined;
                      const ksAfter = result.ks_p_after?.[cov] as number | null | undefined;
                      const vrOk = vr == null || (vr >= 0.5 && vr <= 2.0);
                      return (
                        <tr key={cov} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="px-3 py-1.5 font-mono text-gray-800">{cov}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${smdColor(before)}`}>{before.toFixed(4)}</td>
                          <td className={`px-3 py-1.5 text-right font-mono font-semibold ${smdColor(after)}`}>{after.toFixed(4)}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${vrOk ? "text-gray-600" : "text-amber-600 font-semibold"}`}
                              title={vrBefore != null ? `Before: ${vrBefore.toFixed(3)}` : ""}>
                            {vr != null ? vr.toFixed(3) : "—"}
                          </td>
                          <td className={`px-3 py-1.5 text-right font-mono ${ksAfter != null && ksAfter < 0.05 ? "text-red-600 font-semibold" : "text-gray-500"}`}>
                            {ksAfter != null ? fmtP(ksAfter) : "—"}
                          </td>
                          <td className="px-3 py-1.5 text-right text-gray-500">{reduction}%</td>
                          <td className="px-3 py-1.5 text-center">
                            <span className={`inline-block text-[9px] font-semibold border rounded-full px-1.5 py-0.5 ${
                              balanced ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-red-50 text-red-600 border-red-200"
                            }`}>
                              {balanced ? "✓" : "✗"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                  <tfoot className="bg-gray-50 border-t-2 border-gray-200">
                    <tr>
                      <td className="px-3 py-1.5 font-semibold text-gray-700">Average</td>
                      <td className={`px-3 py-1.5 text-right font-mono font-semibold ${smdColor(result.avg_smd_before)}`}>{result.avg_smd_before.toFixed(4)}</td>
                      <td className={`px-3 py-1.5 text-right font-mono font-semibold ${smdColor(result.avg_smd_after)}`}>{result.avg_smd_after.toFixed(4)}</td>
                      <td className="px-3 py-1.5 text-gray-400" colSpan={2}>—</td>
                      <td className="px-3 py-1.5 text-right text-indigo-600 font-semibold">{result.reduction_pct}%</td>
                      <td className="px-3 py-1.5 text-center">
                        <span className={`inline-block text-[9px] font-semibold border rounded-full px-1.5 py-0.5 ${
                          result.balance_achieved ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-amber-50 text-amber-700 border-amber-200"
                        }`}>
                          {result.balance_achieved ? "All ✓" : "Partial"}
                        </span>
                      </td>
                    </tr>
                  </tfoot>
                </table>
              </div>
              <p className="text-[10px] text-gray-400">
                Reference: Austin PC (2011). <em>Multivariate Behavioral Research</em>. Adequate balance requires SMD &lt; 0.10 (numerator) AND variance ratio in [0.5, 2.0] (Rubin's rule). KS p-value tests distributional balance (not just mean / variance). SMDs use the pooled SD from the unmatched sample as a fixed denominator before &amp; after matching, so any change reflects only the numerator shift.
              </p>
            </div>

            {/* ── PS Overlap ── */}
            {result.ps_distribution && (
              <div className="panel space-y-2 xl:col-start-1">
                <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
                  Propensity Score Overlap
                  <Tip wide text="The distributions must overlap substantially (common support) for PSM to be valid. If treated and control PS distributions barely overlap, matching cannot remove confounding — reconsider model specification." />
                </h3>
                <PSOverlapPlot psDist={result.ps_distribution} showGrid={showGrid} />
                <p className="text-[10px] text-gray-400">
                  Substantial overlap between red (treated) and blue (control) distributions confirms PSM is valid. Sparse overlap indicates poor common support.
                </p>
              </div>
            )}

            {/* ── Outcome analysis ── */}
            {result.outcome_result && !result.outcome_result.error && (() => {
              const t: string = result.outcome_result.type ?? "";
              const isCoxKind = t.startsWith("stratified_cox") || t.startsWith("weighted_cox");
              const isClogit = t.startsWith("conditional_logistic");
              const isWeightedGLM = t.startsWith("weighted_glm");
              const cohortLabel = result.method === "iptw" ? "Weighted Cohort" : "Matched Cohort";
              return (
              <div className="panel space-y-3 xl:col-start-2">
                <h3 className="text-sm font-semibold text-gray-700">
                  Outcome Analysis — {cohortLabel}
                  <span className="ml-2 text-[10px] font-normal text-indigo-600 bg-indigo-50 border border-indigo-200 rounded-full px-2 py-0.5">
                    {result.outcome_result.model}
                  </span>
                </h3>
                <div className="grid grid-cols-3 gap-2">
                  {isCoxKind ? (
                    [
                      [result.method === "iptw" ? "n (weighted)" : "n (matched)", result.outcome_result.n],
                      ["Events",         result.outcome_result.n_events],
                      ["C-index",        result.outcome_result.concordance?.toFixed(3)],
                    ].map(([k, v]: any) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-sm">{v}</p>
                      </div>
                    ))
                  ) : isClogit ? (
                    [
                      ["n (in fit)",           result.outcome_result.n],
                      ["Informative sets",     result.outcome_result.n_informative_sets],
                      ["Concordant sets",      result.outcome_result.n_uninformative_sets],
                    ].map(([k, v]: any) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-sm">{v}</p>
                      </div>
                    ))
                  ) : isWeightedGLM ? (
                    [
                      ["n (weighted)", result.outcome_result.n],
                      ["Estimand",      (result.estimand ?? "ate").toUpperCase()],
                      ["SE",            (result.se_method === "bootstrap" ? "Bootstrap" : "Robust HC1")],
                    ].map(([k, v]: any) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-sm">{v}</p>
                      </div>
                    ))
                  ) : (
                    [
                      ["n (matched)", result.outcome_result.n],
                      ["AIC",         result.outcome_result.aic?.toFixed(2)],
                      ["BIC",         result.outcome_result.bic?.toFixed(2)],
                    ].map(([k, v]: any) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-sm">{v}</p>
                      </div>
                    ))
                  )}
                </div>
                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                        {(isCoxKind
                          ? ["Variable", "HR", "95% CI", "β", "SE", "z", "p"]
                          : ["Variable", "OR", "95% CI", "β", "SE", "z", "p"]
                        ).map((h: string) => (
                          <th key={h} className="px-2 py-2 text-left font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.outcome_result.coefficients.map((c: any) => {
                        const effect = isCoxKind ? c.hr : c.or;
                        const lo = isCoxKind ? c.hr_low : c.or_low;
                        const hi = isCoxKind ? c.hr_high : c.or_high;
                        return (
                          <tr key={c.variable} className={`border-b border-gray-100 ${c.p < 0.05 ? "hover:bg-indigo-50/30" : "hover:bg-gray-50"}`}>
                            <td className="px-2 py-1.5 font-mono text-gray-800">{c.variable}</td>
                            <td className={`px-2 py-1.5 font-mono font-semibold ${c.p < 0.05 ? "text-indigo-700" : "text-gray-600"}`}>{effect?.toFixed(3)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-500">[{lo?.toFixed(3)}, {hi?.toFixed(3)}]</td>
                            <td className="px-2 py-1.5 font-mono text-gray-600">{c.estimate?.toFixed(4)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-500">{c.se?.toFixed(4)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-500">{c.z?.toFixed(3)}</td>
                            <td className="px-2 py-1.5">
                              <span className={`inline-block font-mono px-1.5 py-0.5 rounded text-[10px] ${
                                c.p < 0.05 ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"
                              }`}>{fmtP(c.p)}</span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="text-[10px] text-gray-400 whitespace-pre-line">
                  {result.outcome_result.method_note ?? (
                    isCoxKind
                      ? "HR = Hazard Ratio (exp(β))."
                      : "OR = Odds Ratio (exp(β))."
                  )}
                </p>
              </div>
              );
            })()}

            {result.outcome_result?.error && (
              <div className="panel bg-red-50 border border-red-200 text-xs text-red-600 xl:col-start-2">
                Outcome analysis failed: {result.outcome_result.error}
              </div>
            )}

            {/* ── Rosenbaum bounds ── */}
            {result.rosenbaum && (
              <div className="panel space-y-2 xl:col-start-2">
                <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
                  Rosenbaum Bounds — Sensitivity to Hidden Bias
                  <Tip wide text="Critical Γ is the magnitude of unmeasured confounding that would just be enough to render the observed treatment-effect p-value non-significant. Γ = 1 means no hidden bias. Γ = 2 means a hidden confounder doubling the odds of treatment. Larger critical Γ ⇒ more robust finding." />
                </h3>
                {result.rosenbaum.applicable === false ? (
                  <p className="text-xs text-gray-500">{result.rosenbaum.reason}</p>
                ) : (
                  <>
                    <div className="grid grid-cols-4 gap-2">
                      <div className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">Discordant pairs</p>
                        <p className="font-semibold text-gray-800 text-sm">{result.rosenbaum.discordant_pairs} ({result.rosenbaum.b} / {result.rosenbaum.c})</p>
                      </div>
                      <div className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">p (Γ = 1)</p>
                        <p className="font-semibold text-gray-800 text-sm">{fmtP(result.rosenbaum.p_unbiased)}</p>
                      </div>
                      <div className={`rounded-lg p-2 text-center border ${result.rosenbaum.critical_gamma != null && result.rosenbaum.critical_gamma > 1.5 ? "bg-emerald-50 border-emerald-200" : "bg-amber-50 border-amber-200"}`}>
                        <p className="text-[10px] text-gray-500">Critical Γ</p>
                        <p className={`font-semibold text-sm ${result.rosenbaum.critical_gamma != null && result.rosenbaum.critical_gamma > 1.5 ? "text-emerald-700" : "text-amber-700"}`}>
                          {result.rosenbaum.critical_gamma != null ? result.rosenbaum.critical_gamma.toFixed(2) : `> ${result.rosenbaum.gamma_max}`}
                        </p>
                      </div>
                      <div className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">α</p>
                        <p className="font-semibold text-gray-800 text-sm">{result.rosenbaum.alpha}</p>
                      </div>
                    </div>
                    <div className="overflow-auto rounded-lg border border-gray-200 max-h-48">
                      <table className="w-full text-[11px] border-collapse">
                        <thead className="sticky top-0 bg-gray-50 border-b border-gray-200 text-gray-500">
                          <tr>
                            <th className="text-left px-2 py-1">Γ</th>
                            <th className="text-right px-2 py-1">Upper-bound one-sided p</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.rosenbaum.curve?.map((r: any) => (
                            <tr key={r.gamma} className={r.p_upper > result.rosenbaum.alpha ? "bg-amber-50" : ""}>
                              <td className="px-2 py-0.5 font-mono">{r.gamma.toFixed(2)}</td>
                              <td className="px-2 py-0.5 text-right font-mono">{r.p_upper.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="text-[10px] text-gray-400">
                      Reference: Rosenbaum PR (2002). <em>Observational Studies</em>. Cardiology rule of thumb: critical Γ &gt; 2 indicates a robust effect under plausible hidden bias.
                    </p>
                  </>
                )}
              </div>
            )}

            {/* Matching method warning */}
            {result.matching_warning && (
              <div className="panel bg-amber-50 border border-amber-200 text-xs text-amber-800 xl:col-start-2">
                {result.matching_warning}
              </div>
            )}
          </div>
        ) : (
          /* ── Empty state ── */
          <div className="space-y-4">
            <div className="panel bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-3xl">🧬</span>
                <div>
                  <h2 className="text-base font-bold text-indigo-900">Propensity Score Matching</h2>
                  <p className="text-xs text-indigo-600">Advanced Epidemiology — Observational Causal Inference</p>
                </div>
              </div>
              <p className="text-sm text-gray-700 leading-relaxed">
                PSM mimics a Randomized Controlled Trial from observational data by balancing baseline characteristics between treated and control groups. It is the accepted gold standard for non-randomized cardiology studies.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {[
                { icon: "🎯", title: "Step 1 — Propensity Score", color: "indigo",
                  body: "Logistic regression (Treatment ~ Covariates) estimates each patient's probability of receiving treatment given their baseline profile. This is their Propensity Score (PS)." },
                { icon: "🔗", title: "Step 2 — Nearest-Neighbour Matching", color: "violet",
                  body: "Each treated patient is matched to the control with the closest PS. Caliper = 0.2 × SD(PS) — the medical standard prevents poor matches from degrading balance." },
                { icon: "📊", title: "Step 3 — Love Plot (SMD)", color: "blue",
                  body: "Standardized Mean Differences are calculated before and after matching for every covariate. ALL SMDs must be < 0.10 for the match to be publication-ready (Austin, 2011)." },
                { icon: "🏥", title: "Step 4 — Outcome Analysis", color: "emerald",
                  body: "Logistic regression, Kaplan-Meier, or Cox regression is run on the balanced matched cohort. Treatment effects estimated here are free from measured confounding." },
              ].map(({ icon, title, body }) => (
                <div key={title} className="panel flex gap-3 border-t-4 border-indigo-200">
                  <span className="text-2xl flex-shrink-0">{icon}</span>
                  <div>
                    <p className="text-xs font-bold text-gray-800 mb-1">{title}</p>
                    <p className="text-xs text-gray-500 leading-relaxed">{body}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="panel bg-amber-50 border border-amber-200 space-y-1.5">
              <p className="text-xs font-bold text-amber-800">⚠ Key Assumptions</p>
              {[
                ["No unmeasured confounders", "PSM only balances variables you include. Hidden confounders (not in your dataset) cannot be removed — this is PSM's fundamental limitation."],
                ["Binary treatment", "The treatment variable must be 0/1. Continuous or multi-level treatments require different methods (e.g. GPS, IPTW)."],
                ["Common support", "Treated and control propensity score distributions must overlap substantially. No overlap = no valid matches."],
              ].map(([title, body]) => (
                <div key={title as string} className="flex gap-1.5 text-[10px] text-amber-700">
                  <span className="flex-shrink-0 font-semibold">{title}:</span>
                  <span>{body}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
