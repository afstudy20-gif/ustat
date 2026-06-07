import { useState } from "react";
import { useStore } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runLinear, runLogistic, runFirthLogistic, runKM, runCox, runLogisticTable, runPoisson, runCoxUniMulti } from "../api";
import { Tip, InfoBanner } from "./Tip";
import { MissingGuard, type ImputationStrategy } from "./MissingGuard";
import { PALETTES } from "../store";
import { useResizableRightCol } from "../hooks/useResizableRightCol";
import { CoefTable, ORTable, ForestPlot, PredictionPanel, CoefDetailPanel, ModelSummaryTable } from "./models/resultViews";
import CoxHRTable from "./models/CoxHRTable";
import { useModelData } from "./models/useModelData";

const _pal = () => PALETTES[useStore.getState().plotTheme.palette] ?? PALETTES.indigo;

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
  hrtable: {
    use: "Publication HR table (Table 3): each predictor's univariable HR, its parsimonious-model HR (a subset you tick), and its fully-adjusted HR (all predictors together) side by side.",
    check: "Event column must be binary 0/1, duration positive. Tick which predictors enter the parsimonious column. Categorical predictors expand to one row per level vs the reference.",
    interpret: "Univariable = crude effect. Parsimonious = adjusted for the chosen subset. Fully adjusted = adjusted for everything. A blank (—) cell means the predictor was not in that model.",
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
  // Cumulative left offset per segment, precomputed immutably (no reassigned
  // closure variable inside the render map) — react-hooks/immutability.
  const offsets = data.reduce<number[]>(
    (acc, _v, i) => [...acc, i === 0 ? 0 : acc[i - 1] + (data[i - 1] / total) * W],
    [],
  );
  return (
    <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
      {data.map((v, i) => {
        const w = (v / total) * W;
        return <rect key={i} x={offsets[i]} y={0} width={Math.max(w, 0.5)} height={H} fill={CATS[i % CATS.length]} />;
      })}
    </svg>
  );
}


export default function ModelsPanel() {
  const session  = useStore((s) => s.session);
  const { w: rightColW, onDragStart: onResizeStart, onReset: onResizeReset } =
    useResizableRightCol("ModelsPanel.result", 480);

  const { numCols: numColsRaw, allCols: allColsRaw, binaryCols: binaryColsRaw, missingCounts, sparklines } = useModelData(session);

  // Hide "exclude from analysis" columns from every variable picker in this panel.
  // The list comes from useModelData as plain names, so we filter against the set
  // of column names flagged analysis_excluded in the session metadata.
  // Computed before the early-return so it is null-safe and usable as hook defaults.
  const excludedSet = new Set((session?.columns ?? []).filter((c) => c.analysis_excluded).map((c) => c.name));
  const numCols    = numColsRaw.filter((c) => !excludedSet.has(c));
  const allCols    = allColsRaw.filter((c) => !excludedSet.has(c));
  const binaryCols = binaryColsRaw.filter((c) => !excludedSet.has(c));

  const [model, setModel] = usePersistedPanelState<string>("models", "model", "linear");
  const [outcome, setOutcome] = usePersistedPanelState<string>("models", "outcome", numCols[0] ?? "");
  const [predictors, setPredictors] = usePersistedPanelState<string[]>("models", "predictors", []);
  // HR Table (Cox uni/parsimonious/full): subset of predictors that enter the
  // parsimonious (middle) model column.
  const [parsimonious, setParsimonious] = usePersistedPanelState<string[]>("models", "parsimonious", []);
  // Pairwise interaction terms applied to linear / logistic / Cox / poisson.
  // Stored as [colA, colB]; rendered as a small picker below the predictor list.
  const [glmInteractions, setGlmInteractions] = usePersistedPanelState<Array<[string, string]>>("models", "glmInteractions", []);
  const [glmIxA, setGlmIxA] = useState<string>("");
  const [glmIxB, setGlmIxB] = useState<string>("");

  // ── New feature state ─────────────────────────────────────────────────────
  const [selectedCoefIdx, setSelectedCoefIdx] = useState<number | null>(null);
  const [nullHyp,   setNullHyp]   = useState("eq");    // eq | leq | geq
  const [robustSE,  setRobustSE]  = useState(false);
  const [scaleFactors, setScaleFactors] = useState<Record<string, string>>({}); // col → divisor string
  const [selection, setSelection] = usePersistedPanelState<string>("models", "selection", "p10"); // multivariate variable selection strategy
  const [durationCol, setDurationCol] = usePersistedPanelState<string>("models", "durationCol", numCols[0] ?? "");
  const [eventCol, setEventCol] = usePersistedPanelState<string>("models", "eventCol", binaryCols[0] ?? numCols[1] ?? "");
  const [groupCol, setGroupCol] = usePersistedPanelState<string>("models", "groupCol", "");
  const [stratifyCol, setStratifyCol] = usePersistedPanelState<string>("models", "stratifyCol", "");
  const cachedModels = useStore((s) => s.panelCache.models);
  const setCacheModels = useStore((s) => s.setPanelCache);
  const [result, _setResultRaw] = useState<any>(cachedModels?.result ?? null);
  const setResult = (r: any) => { _setResultRaw(r); setCacheModels("models", { result: r }); };
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const [predFilter, setPredFilter] = useState("");

  // All hooks above run unconditionally (react-hooks/rules-of-hooks). The
  // session guard sits here, after every hook is declared.
  if (!session) return null;

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
      else if (model === "hrtable") res = await runCoxUniMulti({ session_id: sid, duration_col: durationCol, event_col: eventCol, predictors, parsimonious: parsimonious.filter((p) => predictors.includes(p)) });
      else res = await runCox({ session_id: sid, duration_col: durationCol, event_col: eventCol, predictors, imputation, interactions });
      setResult(res.data);
    } catch (e: any) {
      const detail = e.response?.data?.detail;
      setError(typeof detail === "string" ? detail : (e.message ?? "Unknown error"));
    } finally { setLoading(false); }
  };

  const togglePredictor = (col: string) => {
    if (predictors.includes(col)) {
      // Removing — also clear its scale factor
      setScaleFactors((sf) => { const next = { ...sf }; delete next[col]; return next; });
      setPredictors(predictors.filter((c) => c !== col));
    } else {
      setPredictors([...predictors, col]);
    }
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
  const isHRTable   = model === "hrtable";
  const hasRobustSE = model === "linear" || model === "logistic" || model === "poisson";

  const toggleParsimonious = (col: string) => {
    setParsimonious(parsimonious.includes(col) ? parsimonious.filter((c) => c !== col) : [...parsimonious, col]);
  };

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
            ["hrtable",  "HR Table (Uni + Multi)",   "Cox survival version of the OR table (publication Table 3). Each predictor's univariable HR, its parsimonious-model HR (a subset you tick), and its fully-adjusted HR — side by side. Needs a duration + binary event column."],
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
          {isHRTable ? (
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
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">Predictors</label>
                  <button onClick={() => { setPredictors([]); setParsimonious([]); setResult(null); }} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300 transition-colors">Clear all</button>
                </div>
                <div className="mb-2 text-[10px] text-indigo-600 bg-indigo-50 border border-indigo-200 rounded px-2 py-1 leading-snug">
                  Tick a predictor to include it. The <strong>★</strong> box marks which predictors
                  also enter the <strong>parsimonious</strong> (middle) model column. Univariable and
                  fully-adjusted columns always use every ticked predictor.
                </div>
                <input
                  type="text"
                  placeholder="Filter variables…"
                  value={predFilter}
                  onChange={(e) => setPredFilter(e.target.value)}
                  className="select w-full text-xs mb-1 py-1"
                />
                <div className="max-h-48 overflow-y-auto space-y-1">
                  {allCols
                    .filter((c) => c !== durationCol && c !== eventCol && c.toLowerCase().includes(predFilter.toLowerCase()))
                    .map((c) => {
                      const checked = predictors.includes(c);
                      const spk = sparklines[c];
                      return (
                        <div key={c} className="flex items-center gap-2 text-sm">
                          <label className="flex items-center gap-2 cursor-pointer flex-1 min-w-0">
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
                          <label
                            className={`flex items-center gap-0.5 text-[10px] flex-shrink-0 ${checked ? "cursor-pointer text-amber-600" : "opacity-30"}`}
                            title="Include in the parsimonious model column"
                          >
                            <input
                              type="checkbox"
                              disabled={!checked}
                              checked={checked && parsimonious.includes(c)}
                              onChange={() => toggleParsimonious(c)}
                              className="accent-amber-500"
                            />
                            ★
                          </label>
                        </div>
                      );
                    })}
                </div>
              </div>
            </>
          ) : isSurvival ? (
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
            columns={isHRTable
              ? [durationCol, eventCol, ...predictors]
              : isSurvival
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
          isHRTable && result.rows ? (
            <div className="panel">
              <h4 className="font-semibold text-gray-900 mb-2">
                Univariable, Parsimonious &amp; Fully adjusted Cox HR Table
                <Tip wide text="Publication Table 3. Univariable = each predictor fitted alone. Parsimonious = the subset you ticked (★) fitted together. Fully adjusted = all predictors fitted together. A blank (—) cell means the predictor was not in that model." />
              </h4>
              <CoxHRTable
                rows={result.rows}
                columns={session.columns}
                n={result.n}
                nEvents={result.n_events}
                nPars={result.n_pars}
                nEventsPars={result.n_events_pars}
                durationCol={result.duration_col}
                eventCol={result.event_col}
              />
            </div>
          ) : (
          <div
            className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_var(--right-col)] gap-4 auto-rows-min items-start xl:grid-flow-dense relative"
            style={{ ["--right-col" as any]: `${rightColW}px` }}
          >
            {/* Draggable column divider — desktop only */}
            <div
              role="separator"
              aria-orientation="vertical"
              title="Drag: middle / right column width · Double-click: reset"
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
          </div>
          )
        ) : (
          <div className="panel h-64 flex items-center justify-center text-gray-400">
            Configure and fit a model
          </div>
        )}
      </div>
    </div>
  );
}
