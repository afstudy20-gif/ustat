import { useState, useEffect, useRef, useMemo } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { runLinear, runLogistic, runFirthLogistic, runKM, runCox, runLogisticTable, runPoisson, getSparklines } from "../api";
import { Tip, InfoBanner } from "./Tip";
import { MissingGuard, type ImputationStrategy } from "./MissingGuard";
import { PALETTES } from "../store";
import { useResizableRightCol } from "../hooks/useResizableRightCol";
import { CoefTable, ORTable, ForestPlot, PredictionPanel, CoefDetailPanel, ModelSummaryTable } from "./models/resultViews";

const _pal = () => PALETTES[useStore.getState().plotTheme.palette] ?? PALETTES.indigo;

const PLOT_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  font: { color: "#374151", size: 12 },
  margin: { t: 30, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: "#e5e7eb" },
  yaxis: { gridcolor: "#e5e7eb" },
};

// ── Model guidance ──────────────────────────────────────────────────────────
const MODEL_GUIDANCE: Record<string, { use: string; check: string; interpret: string }> = {
  linear: {
    use: "Predict a continuous outcome from one or more predictors. Best for dose-response, biomarker prediction, and adjusted mean comparisons.",
    check: "Residuals vs Fitted should show no pattern (linearity). Q-Q plot should be roughly diagonal (normality). Scale-Location should be flat (homoscedasticity). Use Robust SE if heteroscedastic.",
    interpret: "Each coefficient = change in outcome per 1-unit increase in predictor, holding others constant. R\u00B2 = proportion of variance explained. Check p-values and 95% CIs.",
  },
  logistic: {
    use: "Model a binary outcome (0/1) — e.g. death, readmission, disease presence. Returns Odds Ratios (OR) with 95% CI.",
    check: "Outcome must be binary 0/1. Check for multicollinearity (VIF > 5). Sample size rule of thumb: \u2265 10 events per predictor variable (EPV).",
    interpret: "OR > 1 = higher odds of outcome. OR < 1 = protective. OR = 1 = no effect. Report: OR (95% CI), p-value. Pseudo-R\u00B2 is NOT comparable to linear R\u00B2.",
  },
  ortable: {
    use: "Publication-standard univariate + multivariate OR table. Shows each predictor's effect both alone and adjusted for all others.",
    check: "Same as logistic. The forest plot visually compares unadjusted vs adjusted ORs — large shifts suggest confounding.",
    interpret: "Univariate OR = crude effect. Multivariate OR = adjusted effect. If they differ substantially, the variable is confounded by others in the model.",
  },
  poisson: {
    use: "Model count outcomes (0, 1, 2, 3...) — e.g. number of events, hospital visits, complications. Returns Incidence Rate Ratios (IRR).",
    check: "Outcome must be non-negative integers. Check for overdispersion: if variance >> mean, use Negative Binomial instead. Robust SE helps with mild overdispersion.",
    interpret: "IRR > 1 = higher rate. IRR = 1.5 means 50% more events. Report: IRR (95% CI), p-value.",
  },
  km: {
    use: "Visualise time-to-event data. The survival curve shows the probability of surviving beyond each time point. Log-rank test compares curves between groups.",
    check: "Event column must be binary 0/1 (1 = event occurred). Duration must be positive. Censoring is assumed non-informative.",
    interpret: "Curves that separate early = strong effect. Log-rank p < 0.05 = significant difference between groups. Median survival = time at which 50% have had the event.",
  },
  cox: {
    use: "Regression for time-to-event data with multiple predictors. Returns Hazard Ratios (HR) — the multiplicative effect on the event rate.",
    check: "Proportional hazards assumption: HR should be constant over time (check Schoenfeld residuals). Event column must be binary 0/1.",
    interpret: "HR > 1 = higher hazard (worse prognosis). HR < 1 = protective. HR = 1 = no effect. Report: HR (95% CI), p-value.",
  },
  rcs: {
    use: "Model non-linear (U/J-shaped) dose-response relationships using Restricted Cubic Splines. Essential for continuous biomarkers where the effect is not a straight line.",
    check: "Predictor must have enough unique values (\u2265 knots + 2). Logistic RCS requires binary 0/1 outcome. 4 knots is the standard default.",
    interpret: "The dose-response curve shows how OR (or outcome) changes across the predictor range. Reference value = OR 1.0. Non-linearity p-value tests whether the curve is significantly non-linear.",
  },
};


// ── Sparkline mini distribution bar ──────────────────────────────────────────
function SparklineMini({ data, type }: { data: number[]; type: string }) {
  const W = 44, H = 14;
  if (!data || data.length === 0) return null;
  const max = Math.max(...data);
  if (max === 0) return null;
  if (type === "numeric") {
    const bw = W / data.length;
    return (
      <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
        {data.map((v, i) => {
          const bh = Math.max(1, (v / max) * H);
          return <rect key={i} x={i * bw} y={H - bh} width={Math.max(bw - 0.5, 0.5)} height={bh} fill="#ef4444" opacity={0.65} />;
        })}
      </svg>
    );
  }
  // categorical → stacked horizontal proportion bars
  const total = data.reduce((a, b) => a + b, 0);
  const CATS  = _pal();
  let cx = 0;
  return (
    <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
      {data.map((v, i) => {
        const w = (v / total) * W;
        const rect = <rect key={i} x={cx} y={0} width={Math.max(w, 0.5)} height={H} fill={CATS[i % CATS.length]} />;
        cx += w;
        return rect;
      })}
    </svg>
  );
}


export default function ModelsPanel() {
  const session  = useStore((s) => s.session);
  const showGrid = useStore((s) => s.showGrid);
  const { w: rightColW, onDragStart: onResizeStart, onReset: onResizeReset } =
    useResizableRightCol("ModelsPanel.result", 480);
  if (!session) return null;

  const numCols = session.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
  const allCols = session.columns.map((c) => c.name);

  // Binary columns (≤ 2 unique non-null values, both in {0, 1}) — Cox event
  // and logistic outcome pickers should narrow to these to avoid the user
  // accidentally selecting a continuous variable as the event indicator.
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

  // Missing counts per column
  const missingCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const col of session.columns) {
      counts[col.name] = session.preview.filter(
        (row) => row[col.name] === null || row[col.name] === undefined || row[col.name] === ""
      ).length;
    }
    return counts;
  }, [session.preview, session.columns]);

  const [model, setModel] = useState("linear");
  const [outcome, setOutcome] = useState(numCols[0] ?? "");
  const [predictors, setPredictors] = useState<string[]>([]);
  // Pairwise interaction terms applied to linear / logistic / Cox / poisson.
  // Stored as [colA, colB]; rendered as a small picker below the predictor list.
  const [glmInteractions, setGlmInteractions] = useState<Array<[string, string]>>([]);
  const [glmIxA, setGlmIxA] = useState<string>("");
  const [glmIxB, setGlmIxB] = useState<string>("");

  // ── New feature state ─────────────────────────────────────────────────────
  const [selectedCoefIdx, setSelectedCoefIdx] = useState<number | null>(null);
  const [nullHyp,   setNullHyp]   = useState("eq");    // eq | leq | geq
  const [robustSE,  setRobustSE]  = useState(false);
  const [sparklines, setSparklines] = useState<Record<string, { type: string; data: number[] }>>({});

  useEffect(() => {
    getSparklines(session.session_id)
      .then((r) => setSparklines(r.data))
      .catch(() => {});
  }, [session.session_id]);

  const [scaleFactors, setScaleFactors] = useState<Record<string, string>>({}); // col → divisor string
  const [selection, setSelection] = useState("p10"); // multivariate variable selection strategy
  const [durationCol, setDurationCol] = useState(numCols[0] ?? "");
  const [eventCol, setEventCol] = useState(binaryCols[0] ?? numCols[1] ?? "");
  const [groupCol, setGroupCol] = useState("");
  const [stratifyCol, setStratifyCol] = useState("");
  const cachedModels = useStore((s) => s.panelCache.models);
  const setCacheModels = useStore((s) => s.setPanelCache);
  const [result, _setResultRaw] = useState<any>(cachedModels?.result ?? null);
  const setResult = (r: any) => { _setResultRaw(r); setCacheModels("models", { result: r }); };
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const [predFilter, setPredFilter] = useState("");

  // ── KM curve styling ────────────────────────────────────────────────────────
  const KM_PALETTE = _pal();
  const KM_DASHES  = ["solid","dash","dot","dashdot"] as const;

  interface KmStyle { color: string; width: number; dash: string; }
  const [kmStyles, setKmStyles] = useState<KmStyle[]>([]);
  const kmPlotRef  = useRef<any>(null);

  // KM display feature toggles
  const [showKMci,        setShowKMci]        = useState(true);
  const [showKMcensor,    setShowKMcensor]    = useState(true);
  const [showKMrisktable, setShowKMrisktable] = useState(true);

  // Re-init styles whenever a new KM result comes in
  useEffect(() => {
    if (result?.groups) {
      setKmStyles(
        result.groups.map((_: any, i: number) => ({
          color: KM_PALETTE[i % KM_PALETTE.length],
          width: 2,
          dash:  "solid",
        }))
      );
    }
  }, [result?.groups?.length, result?.groups?.map((g: any) => g.group).join("|")]);

  const updateKmStyle = (idx: number, patch: Partial<KmStyle>) =>
    setKmStyles((prev) => prev.map((s, i) => (i === idx ? { ...s, ...patch } : s)));

  const exportKMcsv = () => {
    if (!result?.groups) return;
    const rows = ["group,time,survival,n_at_risk,events"];
    result.groups.forEach((g: any) => {
      g.curve.forEach((p: any) => {
        rows.push(`"${g.group}",${p.time},${p.survival.toFixed(6)},${p.n_at_risk ?? ""},${p.events ?? ""}`);
      });
    });
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = `KM_${durationCol}_${eventCol}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const sid = session.session_id;

  const run = async () => {
    setLoading(true); setError(null); setResult(null); setSelectedCoefIdx(null);
    try {
      let res: any;
      const sf = buildScaleFactors();
      const interactions = glmInteractions.length > 0 ? glmInteractions : undefined;
      if (model === "linear") res = await runLinear({ session_id: sid, outcome, predictors, imputation, robust_se: robustSE, interactions });
      else if (model === "logistic") res = await runLogistic({ session_id: sid, outcome, predictors, scale_factors: sf, imputation, robust_se: robustSE, interactions });
      else if (model === "firth") res = await runFirthLogistic({ session_id: sid, outcome, predictors, scale_factors: sf, imputation, interactions });
      else if (model === "ortable") res = await runLogisticTable({ session_id: sid, outcome, predictors, scale_factors: sf, selection, imputation });
      else if (model === "firth_ortable") res = await runLogisticTable({ session_id: sid, outcome, predictors, scale_factors: sf, selection, imputation, use_firth: true });
      else if (model === "poisson") res = await runPoisson({ session_id: sid, outcome, predictors, imputation, robust_se: robustSE });
      else if (model === "km") res = await runKM({ session_id: sid, duration_col: durationCol, event_col: eventCol, group_col: groupCol || undefined, stratify_col: stratifyCol || undefined, imputation });
      else res = await runCox({ session_id: sid, duration_col: durationCol, event_col: eventCol, predictors, imputation, interactions });
      setResult(res.data);
    } catch (e: any) {
      const detail = e.response?.data?.detail;
      setError(typeof detail === "string" ? detail : (e.message ?? "Unknown error"));
    } finally { setLoading(false); }
  };

  const togglePredictor = (col: string) => {
    setPredictors((prev) => {
      if (prev.includes(col)) {
        // Removing — also clear its scale factor
        setScaleFactors((sf) => { const next = { ...sf }; delete next[col]; return next; });
        return prev.filter((c) => c !== col);
      }
      return [...prev, col];
    });
  };

  const setScaleFactor = (col: string, val: string) => {
    setScaleFactors((sf) => ({ ...sf, [col]: val }));
  };

  /** Build scale_factors object for API: only include valid factors != 1 */
  const buildScaleFactors = () => {
    const out: Record<string, number> = {};
    for (const [col, val] of Object.entries(scaleFactors)) {
      const n = parseFloat(val);
      if (!isNaN(n) && n > 0 && n !== 1) out[col] = n;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  };

  const isSurvival  = false;  // KM/Cox moved to Survival Advanced tab
  const isORTable   = model === "ortable" || model === "firth_ortable";
  const hasRobustSE = model === "linear" || model === "logistic" || model === "poisson";

  return (
    <div className="flex gap-4">
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Model</h3>
          {([
            ["linear",   "Linear Regression",       "Predict a continuous outcome (e.g. blood pressure) from one or more predictors. Output: β coefficients, R², p-values."],
            ["logistic", "Logistic Regression",      "Predict a binary outcome (0/1, yes/no) — outputs Odds Ratios showing how each predictor changes the odds of the event."],
            ["firth",    "Firth Logistic (penalized)", "Bias-corrected logistic regression (Firth 1993). Use when standard logistic fails or returns infinite ORs from rare events / separation. Same output shape as Logistic but with Jeffreys-prior penalty."],
            ["ortable",  "OR Table (Uni + Multi)",   "Run univariate logistic regression for each predictor separately, then all significant ones together in a multivariate model. Standard for clinical papers."],
            ["firth_ortable", "Firth OR Table (Uni + Multi)", "Same univariate + multivariate OR table as above but every cell is fitted via Firth's penalised likelihood — handles rare events and quasi-separation. Use for the LAR / albumin-style protective biomarker workflow when standard logistic returns ∞ or near-zero ORs."],
            ["poisson",  "Poisson Regression",       "Count outcome model (e.g. number of events). Outputs Incidence Rate Ratios (IRR = eβ). Use when the outcome is a non-negative integer (event counts, re-admissions, etc.)."],
          ] as const).map(([v, l, desc]) => (
            <label key={v} className="flex items-start gap-2 cursor-pointer group">
              <input type="radio" name="model" value={v} checked={model === v} onChange={() => { setModel(v); setResult(null); setSelectedCoefIdx(null); }} className="accent-indigo-500 mt-0.5" />
              <span className="text-sm text-gray-700 leading-tight">
                {l}
                <Tip text={desc} wide />
              </span>
            </label>
          ))}
          {hasRobustSE && (
            <label className="flex items-center gap-2 cursor-pointer mt-1 pt-2 border-t border-gray-100">
              <input type="checkbox" checked={robustSE} onChange={(e) => setRobustSE(e.target.checked)} className="accent-indigo-500" />
              <span className="text-xs text-gray-600">
                Robust SE (HC3)
                <Tip text="Heteroscedasticity-consistent standard errors (HC3). Use when residuals may have unequal variance — common in clinical data. Does not change point estimates, only SEs and p-values." wide />
              </span>
            </label>
          )}
          {(model === "linear" || model === "logistic" || model === "firth" || model === "ortable" || model === "firth_ortable") && (
            <div className="mt-1 pt-2 border-t border-gray-100 text-[10px] text-indigo-600 bg-indigo-50 border border-indigo-200 rounded px-2 py-1 leading-snug">
              Need a non-linear continuous effect? Use the
              {" "}<span className="font-semibold">Restricted Cubic Spline</span> sub-tab —
              <code>rcs(X, k)</code> with Harrell or custom knots, Wald non-linearity test,
              OR / β / HR curve with 95 % CI, and optional spline × covariate or spline × spline
              interaction (LR test + 2D contour / 3D surface).
            </div>
          )}
        </div>

        <div className="panel space-y-3">
          {isSurvival ? (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Duration column</label>
                <select className="select w-full" value={durationCol} onChange={(e) => setDurationCol(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Event column (0/1)
                  {binaryCols.length === 0 && <span className="ml-1 text-[10px] text-amber-600">⚠ no binary 0/1 column detected</span>}
                </label>
                <select className="select w-full" value={eventCol} onChange={(e) => setEventCol(e.target.value)}>
                  {(binaryCols.length > 0 ? binaryCols : numCols).map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              {model === "km" && (
                <>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Group column (optional)</label>
                    <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                      <option value="">None</option>
                      {allCols.map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Stratify by (optional)</label>
                    <select className="select w-full" value={stratifyCol} onChange={(e) => setStratifyCol(e.target.value)}>
                      <option value="">None</option>
                      {allCols.filter((c) => c !== groupCol && c !== durationCol && c !== eventCol).map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>
                </>
              )}
              {model === "cox" && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-gray-400">Predictors</label>
                    <button onClick={() => { setPredictors([]); setResult(null); }} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300 transition-colors">Clear all</button>
                  </div>
                  <div className="mb-2 text-[10px] text-indigo-600 bg-indigo-50 border border-indigo-200 rounded px-2 py-1 leading-snug">
                    Predictors enter the model linearly. For non-linear continuous effects (e.g.
                    J-shaped LDL ↔ mortality), use the
                    {" "}<span className="font-semibold">Restricted Cubic Spline</span> sub-tab —
                    it fits <code>rcs(X, 4)</code> in the same Cox PH framework with knot placement
                    (Harrell percentiles or custom), Wald non-linearity test, HR curve with 95% CI,
                    optional <code>rcs(X) × rcs(Y)</code> interaction, and 2D/3D HR surfaces.
                  </div>
                  <input
                    type="text"
                    placeholder="Filter variables…"
                    value={predFilter}
                    onChange={(e) => setPredFilter(e.target.value)}
                    className="select w-full text-xs mb-1 py-1"
                  />
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {allCols
                      .filter((c) => c !== durationCol && c !== eventCol && c.toLowerCase().includes(predFilter.toLowerCase()))
                      .map((c) => {
                        const spk = sparklines[c];
                        return (
                          <label key={c} className="flex items-center gap-2 text-sm cursor-pointer">
                            <input type="checkbox" checked={predictors.includes(c)} onChange={() => togglePredictor(c)} className="accent-indigo-500" />
                            <span className="text-gray-700 truncate flex-1">{c}</span>
                            {(missingCounts[c] ?? 0) > 0 && (
                              <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0"
                                title={`${missingCounts[c]} missing values`}>
                                {missingCounts[c]}✕
                              </span>
                            )}
                            {spk && <SparklineMini data={spk.data} type={spk.type} />}
                          </label>
                        );
                      })}
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Outcome{isORTable && <span className="text-gray-400 ml-1">(binary 0/1)</span>}
                  {(missingCounts[outcome] ?? 0) > 0 && (
                    <span className="ml-1 text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200"
                      title={`${missingCounts[outcome]} missing values in outcome`}>
                      {missingCounts[outcome]} missing
                    </span>
                  )}
                </label>
                <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
                  {allCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>

              {isORTable && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Multivariate Selection</label>
                  <select className="select w-full text-xs" value={selection} onChange={(e) => setSelection(e.target.value)}>
                    <option value="all">All variables (Enter)</option>
                    <option value="p10">Univariate p &lt; 0.10 ★</option>
                    <option value="p05">Univariate p &lt; 0.05</option>
                    <option value="forward">Stepwise Forward</option>
                    <option value="backward">Stepwise Backward</option>
                  </select>
                </div>
              )}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">Predictors</label>
                  <button onClick={() => { setPredictors([]); setResult(null); }} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300 transition-colors">Clear all</button>
                </div>
                <input
                  type="text"
                  placeholder="Filter variables…"
                  value={predFilter}
                  onChange={(e) => setPredFilter(e.target.value)}
                  className="select w-full text-xs mb-1 py-1"
                />
                <div className="max-h-48 overflow-y-auto space-y-1">
                  {allCols.filter((c) => c !== outcome && c.toLowerCase().includes(predFilter.toLowerCase())).map((c) => {
                    const checked = predictors.includes(c);
                    const showScale = checked && (model === "logistic" || model === "firth" || model === "ortable" || model === "firth_ortable");
                    const spk = sparklines[c];
                    return (
                      <div key={c} className="space-y-0.5">
                        <label className="flex items-center gap-2 text-sm cursor-pointer">
                          <input type="checkbox" checked={checked} onChange={() => togglePredictor(c)} className="accent-indigo-500" />
                          <span className="text-gray-700 truncate flex-1">{c}</span>
                          {(missingCounts[c] ?? 0) > 0 && (
                            <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0"
                              title={`${missingCounts[c]} missing values`}>
                              {missingCounts[c]}✕
                            </span>
                          )}
                          {spk && <SparklineMini data={spk.data} type={spk.type} />}
                        </label>
                        {showScale && (
                          <div className="flex items-center gap-1 ml-5 mb-0.5">
                            <span className="text-gray-400 text-xs">÷</span>
                            <input
                              type="number"
                              min="0"
                              step="any"
                              placeholder="1 (no scaling)"
                              value={scaleFactors[c] ?? ""}
                              onChange={(e) => setScaleFactor(c, e.target.value)}
                              className="w-full text-xs bg-white border border-gray-300 rounded px-1.5 py-0.5 text-gray-700 placeholder-gray-300 focus:border-indigo-500 focus:outline-none"
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
          {/* Pairwise interactions — linear / logistic / cox accept them
              server-side (Poisson + OR-table do not yet). Hidden until at
              least 2 predictors are ticked. */}
          {(model === "linear" || model === "logistic" || model === "firth" || model === "cox") && predictors.length >= 2 && (
            <div className="panel space-y-1.5">
              <p className="text-xs text-gray-500 font-medium flex items-center gap-1">
                Interactions
                <Tip wide text="Add pairwise interaction terms (e.g. AGE × SEX). Numeric × numeric is the element-wise product; numeric × categorical expands across every dummy of the categorical; categorical × categorical multiplies every dummy pair. The output coefficient table reports each interaction as 'A:B' with its own effect estimate and p-value. Use sparingly — each extra term costs degrees of freedom." />
                {glmInteractions.length > 0 && (
                  <span className="ml-1 text-indigo-600 font-semibold">{glmInteractions.length} added</span>
                )}
              </p>
              <div className="flex flex-wrap items-center gap-1.5">
                <select value={glmIxA}
                  onChange={(e) => setGlmIxA(e.target.value)}
                  className="select text-xs py-1">
                  <option value="">First term…</option>
                  {predictors.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
                <span className="text-gray-400 text-xs">×</span>
                <select value={glmIxB}
                  onChange={(e) => setGlmIxB(e.target.value)}
                  className="select text-xs py-1">
                  <option value="">Second term…</option>
                  {predictors.filter((p) => p !== glmIxA).map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
                <button
                  onClick={() => {
                    if (!glmIxA || !glmIxB || glmIxA === glmIxB) return;
                    const exists = glmInteractions.some(
                      ([a, b]) => (a === glmIxA && b === glmIxB) || (a === glmIxB && b === glmIxA),
                    );
                    if (exists) return;
                    setGlmInteractions([...glmInteractions, [glmIxA, glmIxB]]);
                    setGlmIxA(""); setGlmIxB("");
                  }}
                  disabled={!glmIxA || !glmIxB || glmIxA === glmIxB}
                  className="text-xs px-2 py-1 rounded border border-indigo-300 text-indigo-600 hover:bg-indigo-50 disabled:opacity-40 transition-colors"
                >+ Add</button>
              </div>
              {glmInteractions.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {glmInteractions.map(([a, b], i) => (
                    <span key={`${a}:${b}:${i}`}
                      className="inline-flex items-center gap-1 bg-amber-50 border border-amber-200 text-amber-800 text-[11px] rounded px-2 py-0.5">
                      {a} × {b}
                      <button onClick={() => setGlmInteractions(glmInteractions.filter((_, idx) => idx !== i))}
                        className="text-amber-500 hover:text-red-500" title="Remove">×</button>
                    </span>
                  ))}
                  <button onClick={() => setGlmInteractions([])}
                    className="text-[10px] text-gray-400 hover:text-red-500 underline">clear all</button>
                </div>
              )}
            </div>
          )}
          <MissingGuard
            sessionId={sid}
            columns={isSurvival
              ? [durationCol, eventCol, ...(model === "cox" ? predictors : [])]
              : [...predictors, outcome]}
            imputation={imputation}
            onImputation={setImputation}
          >
            <button className="btn-primary w-full" onClick={run} disabled={loading || (!isSurvival && predictors.length === 0) || (isORTable && predictors.length < 1)}>
              {loading ? "Fitting…" : "Fit Model"}
            </button>
          </MissingGuard>
          {error && <p className="text-red-400 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 space-y-4">
        {/* Model guidance */}
        {MODEL_GUIDANCE[model] && (
          <div className="grid grid-cols-3 gap-3">
            {[
              { icon: "🎯", title: "Use when", text: MODEL_GUIDANCE[model].use },
              { icon: "✅", title: "Check", text: MODEL_GUIDANCE[model].check },
              { icon: "📖", title: "Interpret", text: MODEL_GUIDANCE[model].interpret },
            ].map(({ icon, title, text }) => (
              <div key={title} className="panel bg-indigo-50 border-indigo-200 p-3">
                <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mb-1">{icon} {title}</p>
                <p className="text-xs text-indigo-800 leading-relaxed">{text}</p>
              </div>
            ))}
          </div>
        )}

        {result ? (
          <div
            className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_var(--right-col)] gap-4 auto-rows-min items-start xl:grid-flow-dense relative"
            style={{ ["--right-col" as any]: `${rightColW}px` }}
          >
            {/* Draggable column divider — desktop only */}
            <div
              role="separator"
              aria-orientation="vertical"
              title="Sürükle: orta / sağ sütun genişliği · Çift tık: sıfırla"
              onPointerDown={onResizeStart}
              onDoubleClick={onResizeReset}
              className="hidden xl:block absolute top-0 bottom-0 w-1.5 rounded-full bg-gray-300/60 hover:bg-indigo-400/80 cursor-col-resize z-20 transition-colors"
              style={{ right: `calc(${rightColW}px + 5px)` }}
            />
            {/* Summary cards */}
            <div className="panel xl:col-start-2">
              <h4 className="font-semibold text-gray-900 mb-3">{result.model}</h4>
              <div className="grid grid-cols-3 gap-3">
                {[
                  ["N",          result.n,                       "Total number of observations used to fit the model."],
                  result.r_squared != null      && ["R²",        result.r_squared?.toFixed(4),      "Proportion of variance in the outcome explained by the model (0–1). Higher is better, but add predictors only if they genuinely help."],
                  result.adj_r_squared != null  && ["Adj R²",    result.adj_r_squared?.toFixed(4),  "R² adjusted for the number of predictors — penalises adding unhelpful variables. Prefer this over R² when comparing models."],
                  result.pseudo_r2 != null      && ["Pseudo R²", result.pseudo_r2?.toFixed(4),      "McFadden's Pseudo R² for logistic regression. Analogous to R² but not directly comparable. Values 0.2–0.4 indicate good fit."],
                  result.f_stat != null         && ["F-stat",    result.f_stat?.toFixed(3),         "F-test: tests whether the model as a whole explains significantly more variance than no predictors. Large F with small p = model is useful."],
                  result.aic != null            && ["AIC",       result.aic?.toFixed(2),            "Akaike Information Criterion — lower is better. Used to compare models: the model with the lowest AIC balances fit and complexity best."],
                  result.bic != null            && ["BIC",       result.bic?.toFixed(2),            "Bayesian Information Criterion — similar to AIC but applies a larger penalty for extra parameters. Prefer the model with the lower BIC."],
                  result.concordance != null    && ["C-index",   result.concordance?.toFixed(4),    "Concordance index for Cox models — equivalent to AUC. Probability that the model ranks a higher-risk patient above a lower-risk patient."],
                ].filter(Boolean).map(([k, v, tip]: any) => (
                  <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                    <p className="text-xs text-gray-400 flex items-center">
                      {k}
                      {tip && <Tip text={tip} wide />}
                    </p>
                    <p className="text-gray-900 font-semibold">{v}</p>
                  </div>
                ))}
              </div>
              {/* Missing-data exclusion notice */}
              {result.n_excluded != null && result.n_excluded > 0 && (
                <div className="mt-3">
                  <InfoBanner>
                    {result.n_excluded} row{result.n_excluded !== 1 ? "s" : ""} were excluded due to missing values
                    {result.imputation && result.imputation !== "listwise" ? ` (${result.imputation} imputation applied to numeric columns)` : " (listwise deletion)"}.
                    Model was fitted on <strong>{result.n}</strong> of <strong>{result.n_total ?? (result.n + result.n_excluded)}</strong> rows.
                  </InfoBanner>
                </div>
              )}
              {/* Plain-English model fit interpretation */}
              {result.r_squared != null && (
                <div className="mt-3">
                  <InfoBanner>
                    The model explains <strong>{(result.r_squared * 100).toFixed(1)}%</strong> of the variance in <em>{result.outcome ?? "the outcome"}</em>.{" "}
                    {result.r_squared >= 0.7 ? "This is a strong fit." : result.r_squared >= 0.4 ? "This is a moderate fit — other factors likely also play a role." : "This is a weak fit — important predictors may be missing."}
                    {result.adj_r_squared != null && result.adj_r_squared < result.r_squared - 0.05 && " Note: Adjusted R² is notably lower than R², suggesting some predictors may not be contributing meaningfully."}
                  </InfoBanner>
                </div>
              )}
              {result.pseudo_r2 != null && !result.omnibus && (
                <div className="mt-3">
                  <InfoBanner>
                    Pseudo R² = {result.pseudo_r2?.toFixed(3)}.{" "}
                    {result.pseudo_r2 >= 0.4 ? "Excellent model fit." : result.pseudo_r2 >= 0.2 ? "Good model fit." : result.pseudo_r2 >= 0.1 ? "Moderate model fit." : "Weak model fit — consider adding more informative predictors."}
                  </InfoBanner>
                </div>
              )}

              {/* ── SPSS-style Model Summary (logistic only) ── */}
              {result.omnibus && (
                <div className="mt-3">
                  <ModelSummaryTable s={result} />
                </div>
              )}
            </div>

            {/* Coefficients table + detail panel */}
            {result.coefficients && (
              <div className="panel xl:col-start-2">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="font-semibold text-gray-900">
                    {model === "cox" ? "Coefficients (Hazard Ratios)" : (model === "logistic" || model === "firth") ? "Coefficients (Odds Ratios)" : model === "poisson" ? "Coefficients (Incidence Rate Ratios)" : "Coefficients"}
                    {model === "linear" && <Tip text="Each β coefficient shows how much the outcome changes for a 1-unit increase in that predictor, holding all others constant. Significant predictors (p < 0.05) are highlighted." wide />}
                    {(model === "logistic" || model === "firth") && <Tip text="Odds Ratio (OR) > 1 means higher odds of the outcome; OR < 1 means lower odds. E.g. OR = 2.0 means the outcome is twice as likely per unit increase. 95% CI not crossing 1 = significant." wide />}
                    {model === "cox" && <Tip text="Hazard Ratio (HR) > 1 means a higher rate of the event over time; HR < 1 means a protective effect. E.g. HR = 1.5 means 50% higher event rate per unit increase." wide />}
                    {model === "poisson" && <Tip text="Incidence Rate Ratio (IRR) = eβ. IRR > 1 means higher event rate; IRR < 1 means lower rate. Use for count outcomes (hospital admissions, episodes, etc.)." wide />}
                  </h4>
                  {/* Null hypothesis radio */}
                  <div className="flex items-center gap-3 text-xs text-gray-500">
                    <span className="text-gray-400">H₀:</span>
                    {([["eq", "β = 0"], ["leq", "β ≤ 0"], ["geq", "β ≥ 0"]] as const).map(([v, lbl]) => (
                      <label key={v} className="flex items-center gap-1 cursor-pointer">
                        <input type="radio" name="nullhyp" value={v} checked={nullHyp === v}
                          onChange={() => { setNullHyp(v); setSelectedCoefIdx(null); }}
                          className="accent-indigo-500" />
                        <span>{lbl}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <CoefTable
                  coefs={result.coefficients}
                  hrMode={model === "cox"}
                  allColumns={allCols}
                  selectedIdx={selectedCoefIdx}
                  onSelect={(i) => setSelectedCoefIdx((prev) => prev === i ? null : i)}
                  nullHyp={nullHyp}
                />
                {selectedCoefIdx != null && result.coefficients[selectedCoefIdx] && (
                  <div className="mt-3">
                    <CoefDetailPanel
                      coef={result.coefficients[selectedCoefIdx]}
                      nullHyp={nullHyp}
                      onClose={() => setSelectedCoefIdx(null)}
                    />
                  </div>
                )}
                <p className="text-[10px] text-gray-400 mt-2">Click a row to see the coefficient's sampling distribution.</p>
              </div>
            )}

            {/* Prediction Panel — linear only */}
            {model === "linear" && result.predictor_info && Object.keys(result.predictor_info).length > 0 && (
              <div className="xl:col-start-2">
                <PredictionPanel result={result} />
              </div>
            )}

            {/* Forest plot — logistic or cox */}
            {result.coefficients && (model === "logistic" || model === "firth" || model === "cox") &&
              result.coefficients.filter((c: any) => c.variable !== "const").length > 0 && (
              <div className="panel xl:col-start-1">
                <h4 className="font-semibold text-gray-900 mb-2">
                  Forest Plot
                  <Tip text="Each row shows one predictor. The square is the point estimate (OR or HR); the horizontal line is the 95% Confidence Interval. If the CI crosses 1 (the vertical dashed line), the effect is not statistically significant. Larger squares = more precise estimate." wide />
                  <span className="ml-2 text-xs font-normal text-gray-400">
                    {model === "cox" ? "HR" : "OR"} with 95% CI — colored = p&lt;0.05, square size = precision
                  </span>
                </h4>
                <ForestPlot result={result} modelType={model} outcome={result.outcome} />
              </div>
            )}

            {/* Results text for all regression models */}
            {result.result_text && !result.table && (
              <div className="panel xl:col-start-2">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900">Results Paragraph</h4>
                  <button onClick={() => navigator.clipboard.writeText(result.result_text)} className="text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
                </div>
                <p className="text-sm text-gray-700 leading-relaxed bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">{result.result_text}</p>
              </div>
            )}

            {/* OR Table (Uni + Multi) */}
            {result.table && (
              <div className="panel xl:col-start-2">
                <h4 className="font-semibold text-gray-900 mb-2">
                  Univariate &amp; Multivariate OR Table
                  <Tip text="Univariate: each predictor tested alone against the outcome. Multivariate: all selected predictors tested together, adjusting for each other. Compare both columns — a variable that is significant univariately but not multivariately may be confounded by another predictor." wide />
                </h4>
                <ORTable
                  rows={result.table}
                  outcome={result.outcome}
                  selectionMethod={result.selection_method}
                  nMulti={result.n_multi}
                  nTotal={result.n_total}
                />
              </div>
            )}

            {/* SPSS-style model stats for OR Table multivariate model */}
            {result.model_stats && (
              <div className="panel xl:col-start-2">
                <h4 className="font-semibold text-gray-900 mb-2">Multivariate Model Summary</h4>
                <ModelSummaryTable s={result.model_stats} />
              </div>
            )}

            {/* Auto-generated results text */}
            {result.result_text && (
              <div className="panel xl:col-start-2">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900">Results Paragraph</h4>
                  <button onClick={() => navigator.clipboard.writeText(result.result_text)} className="text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
                </div>
                <p className="text-sm text-gray-700 leading-relaxed bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">{result.result_text}</p>
              </div>
            )}

            {/* Forest plot — OR table */}
            {result.table && result.table.length > 0 && (
              <div className="panel xl:col-start-1">
                <h4 className="font-semibold text-gray-900 mb-2">
                  Forest Plot
                  <span className="ml-2 text-xs font-normal text-gray-400">
                    ● Univariate &nbsp;◆ Multivariate — colored = p&lt;0.05, square size = precision
                  </span>
                </h4>
                <ForestPlot result={result} modelType={model} outcome={result.outcome} />
              </div>
            )}

            {/* KM curves */}
            {result.groups && (() => {
              const kmGroups: any[]  = result.groups;
              const nG               = kmGroups.length;

              // ── 95% CI via Greenwood's formula ──────────────────────────────
              const groupCI = kmGroups.map((g: any) => {
                let cumVar = 0;
                return g.curve.map((p: any) => {
                  const n = p.n_at_risk ?? 0, d = p.events ?? 0, S = p.survival;
                  if (n > 0 && d > 0 && n > d) cumVar += d / (n * (n - d));
                  const se = S * Math.sqrt(cumVar);
                  return { lower: Math.max(0, S - 1.96 * se), upper: Math.min(1, S + 1.96 * se) };
                });
              });

              // ── Risk table time ticks ───────────────────────────────────────
              const maxT     = Math.max(...kmGroups.flatMap((g: any) => g.curve.map((p: any) => p.time)), 1);
              const N_TICKS  = 6;
              const riskTimes = Array.from({ length: N_TICKS }, (_, i) => Math.round(i * maxT / (N_TICKS - 1)));
              const getRiskAt = (curve: any[], t: number) => {
                const pts = curve.filter((p: any) => p.time <= t);
                return pts.length > 0 ? (pts[pts.length - 1].n_at_risk ?? "—") : (curve[0]?.n_at_risk ?? "—");
              };

              // ── Layout dimensions ────────────────────────────────────────────
              const riskFrac = showKMrisktable ? Math.min(0.38, 0.10 + nG * 0.08) : 0;
              const plotH    = showKMrisktable ? 460 + nG * 18 : 440;

              // ── Build traces ─────────────────────────────────────────────────
              const traces: any[] = [];
              kmGroups.forEach((g: any, i: number) => {
                const style = kmStyles[i] ?? { color: KM_PALETTE[i % KM_PALETTE.length], width: 2, dash: "solid" };
                const ci    = groupCI[i];
                const times     = g.curve.map((p: any) => p.time);
                const survivals = g.curve.map((p: any) => p.survival);

                // CI band — upper boundary (invisible line)
                if (showKMci) {
                  traces.push({
                    type: "scatter", mode: "lines",
                    x: times, y: ci.map((c: any) => c.upper),
                    line: { width: 0, shape: "hv" }, showlegend: false, hoverinfo: "skip",
                    name: `__ci_u_${i}`,
                  });
                  // CI band — lower boundary with fill back to upper
                  traces.push({
                    type: "scatter", mode: "lines",
                    x: times, y: ci.map((c: any) => c.lower),
                    fill: "tonexty", fillcolor: `${style.color}28`,
                    line: { width: 0, shape: "hv" }, showlegend: false, hoverinfo: "skip",
                    name: `__ci_l_${i}`,
                  });
                }

                // Censoring tick marks (vertical stroke)
                if (showKMcensor) {
                  const censorPts = g.curve.filter((p: any) => (p.events === 0) && p.n_at_risk != null);
                  if (censorPts.length > 0 && censorPts.length < g.curve.length) {
                    traces.push({
                      type: "scatter", mode: "markers",
                      x: censorPts.map((p: any) => p.time),
                      y: censorPts.map((p: any) => p.survival),
                      marker: { symbol: "line-ns-open", size: 9, color: style.color, line: { color: style.color, width: 1.5 } },
                      showlegend: false, hoverinfo: "skip",
                      name: `__censor_${i}`,
                    });
                  }
                }

                // Main KM step curve
                traces.push({
                  type: "scatter", mode: "lines",
                  x: times, y: survivals,
                  name: `${g.group} (n=${g.n})`,
                  line: { color: style.color, width: style.width, dash: style.dash, shape: "hv" },
                });
              });

              // ── Risk table annotations ────────────────────────────────────────
              const riskAnnotations: any[] = [];
              if (showKMrisktable) {
                const rowH = (riskFrac - 0.04) / (nG + 0.8);

                riskAnnotations.push({
                  xref: "paper", yref: "paper",
                  x: 0.0, y: riskFrac - 0.01,
                  xanchor: "left", yanchor: "top",
                  text: "<b>Number at risk</b>",
                  showarrow: false,
                  font: { color: "#374151", size: 10 },
                });

                kmGroups.forEach((g: any, i: number) => {
                  const color = kmStyles[i]?.color ?? KM_PALETTE[i % KM_PALETTE.length];
                  const yPos  = riskFrac - 0.05 - i * rowH;

                  // Group label (left margin)
                  riskAnnotations.push({
                    xref: "paper", yref: "paper",
                    x: -0.01, y: yPos,
                    xanchor: "right", yanchor: "middle",
                    text: g.group,
                    showarrow: false,
                    font: { color, size: 10 },
                  });

                  // Risk counts aligned to x-axis
                  riskTimes.forEach((t: number) => {
                    riskAnnotations.push({
                      xref: "x", yref: "paper",
                      x: t, y: yPos,
                      xanchor: "center", yanchor: "middle",
                      text: String(getRiskAt(g.curve, t)),
                      showarrow: false,
                      font: { size: 10, color: "#374151", family: "monospace" },
                    });
                  });
                });
              }

              // p-value annotation
              const pAnnotation = result.logrank?.p != null ? [{
                xref: "paper", yref: "paper",
                x: 0.02, y: 0.98,
                xanchor: "left", yanchor: "top",
                text: `Log-rank p ${result.logrank.p < 0.001 ? "< 0.001" : `= ${result.logrank.p.toFixed(3)}`}`,
                showarrow: false,
                font: { color: result.logrank.p < 0.05 ? "#6366f1" : "#374151", size: 12 },
                bgcolor: "rgba(249,250,251,0.9)", borderpad: 5, bordercolor: "#e5e7eb", borderwidth: 1,
              }] : [];

              return (
                <div className="panel space-y-3 xl:col-start-1">

                  {/* ── Header row ── */}
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500">
                      {kmGroups.map((g: any, i: number) => {
                        const c = kmStyles[i]?.color ?? KM_PALETTE[i % KM_PALETTE.length];
                        return (
                          <span key={g.group} className="flex items-center gap-1.5">
                            <span className="inline-block rounded" style={{ backgroundColor: c, width: 16, height: 3 }} />
                            <span className="text-gray-700 font-medium">{g.group}</span>
                            <span>n={g.n}, events={g.events}</span>
                            {g.median_survival != null && (
                              <span className="text-gray-400">(med {g.median_survival.toFixed(1)})</span>
                            )}
                          </span>
                        );
                      })}
                      {result.logrank && (
                        <span className="font-medium flex items-center gap-1">
                          Log-rank p
                          <Tip text="The log-rank test compares survival curves between groups. p < 0.05 means survival differs significantly between groups. It is most reliable when survival curves do not cross." wide />
                          {" "}
                          <span className={result.logrank.p < 0.05 ? "text-indigo-600" : ""}>
                            {result.logrank.p != null
                              ? (result.logrank.p < 0.001 ? "< 0.001" : `= ${result.logrank.p.toFixed(3)}`)
                              : "N/A"}
                          </span>
                          {result.logrank.p != null && (
                            <span className="text-xs font-normal text-gray-400">
                              {result.logrank.p < 0.001 ? "— highly significant" : result.logrank.p < 0.05 ? "— significant" : "— not significant"}
                            </span>
                          )}
                        </span>
                      )}
                    </div>
                    <div className="flex gap-2 flex-shrink-0">
                      <button onClick={exportKMcsv}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs text-gray-600 border border-gray-300 hover:bg-gray-100 transition-colors">
                        ↓ CSV
                      </button>
                      <button
                        onClick={() => {
                          if (!kmPlotRef.current) return;
                          const Plotly = (window as any).Plotly;
                          if (Plotly) Plotly.downloadImage(kmPlotRef.current, {
                            format: "png", width: 900, height: plotH + 80,
                            filename: `KM_${durationCol}_${eventCol}`,
                          });
                        }}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs text-gray-600 border border-gray-300 hover:bg-gray-100 transition-colors">
                        ↓ PNG
                      </button>
                    </div>
                  </div>

                  {/* ── Feature toggles + per-group style controls ── */}
                  <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-3 py-2.5 bg-gray-50 border border-gray-200 rounded-lg">
                    {/* Toggles */}
                    <div className="flex items-center gap-3 flex-shrink-0">
                      {([
                        ["95% CI",          showKMci,        setShowKMci],
                        ["Censoring marks", showKMcensor,    setShowKMcensor],
                        ["Risk table",      showKMrisktable, setShowKMrisktable],
                      ] as [string, boolean, (v: boolean) => void][]).map(([label, val, setter]) => (
                        <label key={label} className="flex items-center gap-1.5 cursor-pointer text-xs text-gray-600 select-none">
                          <input type="checkbox" checked={val} onChange={(e) => setter(e.target.checked)} className="accent-indigo-500" />
                          {label}
                        </label>
                      ))}
                    </div>

                    <div className="w-px h-5 bg-gray-300 flex-shrink-0" />

                    {/* Per-group style */}
                    {kmGroups.map((g: any, i: number) => {
                      const style = kmStyles[i] ?? { color: KM_PALETTE[i % KM_PALETTE.length], width: 2, dash: "solid" };
                      return (
                        <div key={g.group} className="flex items-center gap-2 text-xs">
                          <span className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: style.color }} />
                          <span className="text-gray-600 font-medium max-w-[80px] truncate" title={g.group}>{g.group}</span>
                          <input type="color" value={style.color}
                            onChange={(e) => updateKmStyle(i, { color: e.target.value })}
                            className="w-6 h-6 rounded cursor-pointer border border-gray-300 p-0" />
                          <select value={style.width}
                            onChange={(e) => updateKmStyle(i, { width: +e.target.value })}
                            className="select text-xs py-0.5 px-1.5 min-w-0">
                            {[1, 1.5, 2, 2.5, 3, 4].map((w) => <option key={w} value={w}>{w}px</option>)}
                          </select>
                          <select value={style.dash}
                            onChange={(e) => updateKmStyle(i, { dash: e.target.value })}
                            className="select text-xs py-0.5 px-1.5 min-w-0">
                            {KM_DASHES.map((d) => <option key={d} value={d}>{d}</option>)}
                          </select>
                        </div>
                      );
                    })}
                  </div>

                  {/* ── Plot ── */}
                  <div style={{ height: plotH }}>
                    <Plot
                      onInitialized={(_: object, gd: HTMLElement) => { kmPlotRef.current = gd; }}
                      onUpdate={(_: object, gd: HTMLElement)      => { kmPlotRef.current = gd; }}
                      data={traces}
                      layout={{
                        ...PLOT_LAYOUT,
                        autosize: true,
                        height: plotH,
                        margin: { t: 30, r: 20, b: 50, l: 90 },
                        yaxis: {
                          ...PLOT_LAYOUT.yaxis,
                          showgrid: showGrid,
                          domain: showKMrisktable ? [riskFrac, 1] : [0, 1],
                          range: [0, 1.05],
                          title: { text: "Survival probability" },
                        },
                        xaxis: { ...PLOT_LAYOUT.xaxis, showgrid: showGrid, title: { text: `Time (${durationCol})` } },
                        legend: {
                          font: { color: "#374151", size: 11 },
                          orientation: "h",
                          y: showKMrisktable ? -(riskFrac + 0.04) : -0.18,
                          bgcolor: "rgba(249,250,251,0.9)",
                          bordercolor: "#e5e7eb", borderwidth: 1,
                        },
                        annotations: [...pAnnotation, ...riskAnnotations],
                      }}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                      config={{
                        responsive: true,
                        displaylogo: false,
                        toImageButtonOptions: {
                          format: "png", filename: `KM_${durationCol}_${eventCol}`,
                          width: 900, height: plotH + 80,
                        },
                        modeBarButtonsToRemove: ["select2d", "lasso2d"],
                      }}
                    />
                  </div>
                </div>
              );
            })()}

            {/* KM stratified small-multiples grid */}
            {result.strata && (() => {
              const strata: any[] = result.strata;
              const nCols = strata.length <= 2 ? strata.length : strata.length === 3 ? 3 : 2;
              const palette = _pal();
              const miniH = 280;

              const buildMiniTraces = (groups: any[]) => {
                const traces: any[] = [];
                groups.forEach((g: any, i: number) => {
                  const color = palette[i % palette.length];
                  const times = g.curve.map((p: any) => p.time);
                  const survs = g.curve.map((p: any) => p.survival);
                  // CI via Greenwood
                  let cumVar = 0;
                  const upper: number[] = [], lower: number[] = [];
                  g.curve.forEach((p: any) => {
                    const n = p.n_at_risk ?? 0, d = p.events ?? 0, S = p.survival;
                    if (n > 0 && d > 0 && n > d) cumVar += d / (n * (n - d));
                    const se = S * Math.sqrt(cumVar);
                    upper.push(Math.min(1, S + 1.96 * se));
                    lower.push(Math.max(0, S - 1.96 * se));
                  });
                  traces.push({ type: "scatter", mode: "lines", x: times, y: upper, line: { width: 0, shape: "hv" }, showlegend: false, hoverinfo: "skip", name: `__u${i}` });
                  traces.push({ type: "scatter", mode: "lines", x: times, y: lower, fill: "tonexty", fillcolor: `${color}28`, line: { width: 0, shape: "hv" }, showlegend: false, hoverinfo: "skip", name: `__l${i}` });
                  traces.push({ type: "scatter", mode: "lines", x: times, y: survs, name: `${g.group} (n=${g.n})`, line: { color, width: 2, shape: "hv" } });
                });
                return traces;
              };

              return (
                <div className="panel space-y-3 xl:col-start-1">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold text-gray-900 text-sm">
                      Stratified by <span className="text-indigo-600">{result.stratify_col}</span>
                      {groupCol && <span className="text-gray-400 font-normal ml-2">— curves by {groupCol}</span>}
                    </h4>
                    <span className="text-xs text-gray-400">{strata.length} strata · {result.n_total} total</span>
                  </div>
                  <div className={`grid gap-4`} style={{ gridTemplateColumns: `repeat(${nCols}, minmax(0, 1fr))` }}>
                    {strata.map((stratum: any) => {
                      const pAnnot = stratum.logrank?.p != null ? [{
                        xref: "paper", yref: "paper", x: 0.02, y: 0.98,
                        xanchor: "left", yanchor: "top",
                        text: `p ${stratum.logrank.p < 0.001 ? "< 0.001" : `= ${stratum.logrank.p.toFixed(3)}`}`,
                        showarrow: false,
                        font: { size: 11, color: stratum.logrank.p < 0.05 ? "#6366f1" : "#6b7280" },
                        bgcolor: "rgba(249,250,251,0.85)", borderpad: 3, bordercolor: "#e5e7eb", borderwidth: 1,
                      }] : [];
                      return (
                        <div key={stratum.label} className="border border-gray-200 rounded-lg overflow-hidden">
                          <div className="px-3 py-1.5 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
                            <span className="text-xs font-semibold text-gray-700">{stratum.label}</span>
                            <span className="text-[10px] text-gray-400">n={stratum.n}</span>
                          </div>
                          <Plot
                            data={buildMiniTraces(stratum.groups)}
                            layout={{
                              ...PLOT_LAYOUT,
                              autosize: true,
                              height: miniH,
                              margin: { t: 10, r: 10, b: 40, l: 50 },
                              yaxis: { ...PLOT_LAYOUT.yaxis, range: [0, 1.05], title: { text: "Survival" }, showgrid: showGrid },
                              xaxis: { ...PLOT_LAYOUT.xaxis, title: { text: durationCol }, showgrid: showGrid },
                              legend: { font: { size: 9 }, orientation: "h", y: -0.22 },
                              annotations: pAnnot,
                            }}
                            style={{ width: "100%", height: miniH }}
                            useResizeHandler
                            config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] }}
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })()}
          </div>
        ) : (
          <div className="panel h-64 flex items-center justify-center text-gray-400">
            Configure and fit a model
          </div>
        )}
      </div>
    </div>
  );
}
