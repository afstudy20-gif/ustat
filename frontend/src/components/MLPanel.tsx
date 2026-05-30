import { useState, useRef } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { usePlotLayout, usePalette } from "../plotStyle";
import { runRandomForest, runGradientBoosting, runPredictive } from "../api";
import { Tip } from "./Tip";
import PlotExporter from "./PlotExporter";
import ResultExporter from "./ResultExporter";
import ThreeCol from "./ThreeCol";

type ModelKind = "random_forest" | "gradient_boosting" | "lasso" | "svm_rbf";
type Task = "auto" | "classification" | "regression";

const MODEL_LABEL: Record<ModelKind, string> = {
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
  lasso: "Lasso",
  svm_rbf: "SVM (RBF)",
};

// Lasso / SVM use a held-out test set + GridSearchCV rather than CV-OOF.
const PREDICTIVE: ModelKind[] = ["lasso", "svm_rbf"];

// A single partial-dependence panel with its own export ref.
function PdpPlot({ feature, x, y, baseLayout, showGrid, color }: {
  feature: string; x: number[]; y: number[];
  baseLayout: Record<string, unknown>; showGrid: boolean; color: string;
}) {
  const ref = useRef<any>(null);
  return (
    <div className="relative panel" ref={ref}>
      <Plot
        data={[{ type: "scatter", mode: "lines", x, y, line: { color, width: 2.5 }, hovertemplate: `${feature} %{x:.3f}<br>risk %{y:.3f}<extra></extra>` }]}
        layout={{
          ...baseLayout,
          title: { text: `Partial dependence — ${feature}`, font: { color: "#374151", size: 12 } },
          xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, title: { text: feature } },
          yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, title: { text: "Predicted risk" } },
          showlegend: false, margin: { t: 36, r: 16, b: 44, l: 50 },
        }}
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        style={{ width: "100%", height: 240 }} useResizeHandler />
      <PlotExporter plotRef={ref} title={`PDP_${feature}`} />
    </div>
  );
}

export default function MLPanel() {
  const session = useStore((s) => s.session);
  const showGrid = useStore((s) => s.showGrid);
  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const rocRef = useRef<any>(null);
  const impRef = useRef<any>(null);
  const scatterRef = useRef<any>(null);

  const columns = session?.columns ?? [];
  const sid = session?.session_id ?? "";
  const numCols = columns.filter((c) => c.kind === "numeric").map((c) => c.name);

  const [model, setModel] = useState<ModelKind>("random_forest");
  const [task, setTask] = useState<Task>("auto");
  const [outcome, setOutcome] = useState("");
  const [predictors, setPredictors] = useState<string[]>([]);
  const [predFilter, setPredFilter] = useState("");
  const [nEstimators, setNEstimators] = useState(300);
  const [maxDepth, setMaxDepth] = useState<string>("");
  const [cvFolds, setCvFolds] = useState(5);
  const [classWeight, setClassWeight] = useState(true);
  const [learningRate, setLearningRate] = useState(0.1);
  // Lasso / SVM holdout pipeline
  const [holdoutFrac, setHoldoutFrac] = useState(0.3);
  const [spline, setSpline] = useState(false);
  const isPredictiveModel = PREDICTIVE.includes(model);

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const togglePred = (c: string) =>
    setPredictors((p) => (p.includes(c) ? p.filter((x) => x !== c) : [...p, c]));

  const run = async () => {
    if (!outcome || predictors.length === 0) {
      setError("Select an outcome and at least one predictor.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let res;
      if (isPredictiveModel) {
        res = await runPredictive({
          session_id: sid, outcome, predictors, model,
          holdout_frac: holdoutFrac, cv_folds: cvFolds,
          spline: model === "svm_rbf" ? spline : false, max_pdp: 4,
        });
      } else {
        const payload = {
          session_id: sid, outcome, predictors, task,
          n_estimators: nEstimators, max_depth: maxDepth ? parseInt(maxDepth, 10) : null,
          cv_folds: cvFolds, class_weight_balanced: classWeight, learning_rate: learningRate,
        };
        const fn = model === "random_forest" ? runRandomForest : runGradientBoosting;
        res = await fn(payload);
      }
      setResult(res.data);
    } catch (e: any) {
      const detail = e?.response?.data?.detail;
      setError(
        Array.isArray(detail)
          ? detail.map((m: any) => m.msg ?? String(m)).join(", ")
          : (typeof detail === "string" ? detail : (e?.message ?? "ML run failed")),
      );
    } finally {
      setLoading(false);
    }
  };

  const isClass = result?.task === "classification";
  const isPredictive = !!result?.holdout;
  const hd = result?.holdout;
  const topImp = (result?.importance ?? []).slice(0, 15).reverse();

  return (
    <div className="space-y-3">
      <ThreeCol
        storageKey="MLPanel"
        left={
          <>
            <div className="panel space-y-2">
              <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
                Predictive Model
                <Tip wide text="Random Forest / Gradient Boosting = tree ensembles, cross-validated (out-of-fold). Lasso (L1 logistic) and SVM (RBF kernel) use a held-out test set with stratified GridSearchCV tuning, reporting honest holdout AUC, Brier, calibration and partial-dependence plots. SVM can expand numeric predictors with cubic splines." />
              </h3>
              <div className="grid grid-cols-2 gap-1">
                {(["random_forest", "gradient_boosting", "lasso", "svm_rbf"] as const).map((m) => (
                  <button key={m} onClick={() => { setModel(m); setResult(null); }}
                    className={`px-2 py-1.5 text-xs font-medium rounded transition-colors ${
                      model === m ? "bg-indigo-600 text-white" : "border border-gray-300 text-gray-500 hover:bg-gray-50"
                    }`}>
                    {MODEL_LABEL[m]}
                  </button>
                ))}
              </div>

              <label className="flex flex-col gap-1">
                <span className="text-xs text-gray-500 font-medium">Outcome</span>
                <select value={outcome} onChange={(e) => { setOutcome(e.target.value); setResult(null); }}
                  className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
                  <option value="">— select —</option>
                  {columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-xs text-gray-500 font-medium flex items-center gap-1">
                  Task
                  <Tip text="Auto picks classification for a binary outcome and regression for a continuous one. Override if needed." />
                </span>
                <div className="flex rounded-lg overflow-hidden border border-gray-300">
                  {(["auto", "classification", "regression"] as const).map((t) => (
                    <button key={t} onClick={() => setTask(t)}
                      className={`flex-1 px-1.5 py-1 text-[11px] font-medium transition-colors ${
                        task === t ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"
                      }`}>
                      {t === "auto" ? "Auto" : t === "classification" ? "Classify" : "Regress"}
                    </button>
                  ))}
                </div>
              </label>

              {/* Predictors */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 font-medium">Predictors</span>
                  <div className="flex gap-1">
                    <button onClick={() => setPredictors(numCols.filter((c) => c !== outcome))}
                      className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">All num</button>
                    {predictors.length > 0 && (
                      <button onClick={() => setPredictors([])}
                        className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500">Clear</button>
                    )}
                  </div>
                </div>
                <input type="text" placeholder="Filter columns…" value={predFilter}
                  onChange={(e) => setPredFilter(e.target.value)}
                  className="w-full text-xs border border-gray-300 rounded-lg px-3 py-1 focus:outline-none focus:border-indigo-400" />
                <div className="max-h-44 overflow-y-auto border border-gray-200 rounded-lg p-1 space-y-0.5">
                  {columns
                    .filter((c) => c.name !== outcome && c.name.toLowerCase().includes(predFilter.toLowerCase()))
                    .map((c) => (
                      <label key={c.name} className="flex items-center gap-1.5 text-xs px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                        <input type="checkbox" className="accent-indigo-500"
                          checked={predictors.includes(c.name)} onChange={() => togglePred(c.name)} />
                        <span className="text-gray-700 truncate flex-1">{c.name}</span>
                        <span className={`text-[9px] px-1 rounded ${c.kind === "numeric" ? "bg-blue-50 text-blue-600" : "bg-purple-50 text-purple-600"}`}>
                          {c.kind === "numeric" ? "N" : "C"}
                        </span>
                      </label>
                    ))}
                </div>
                <p className="text-[10px] text-gray-400">{predictors.length} selected</p>
              </div>

              {/* Hyper-parameters */}
              <div className="grid grid-cols-2 gap-2">
                {!isPredictiveModel && (
                  <label className="flex flex-col gap-0.5">
                    <span className="text-[10px] text-gray-500">Trees (n_estimators)</span>
                    <input type="number" min={10} max={2000} step={10} value={nEstimators}
                      onChange={(e) => setNEstimators(Number(e.target.value))}
                      className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                  </label>
                )}
                {!isPredictiveModel && (
                  <label className="flex flex-col gap-0.5">
                    <span className="text-[10px] text-gray-500">Max depth (blank=auto)</span>
                    <input type="number" min={1} max={50} value={maxDepth} placeholder="auto"
                      onChange={(e) => setMaxDepth(e.target.value)}
                      className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                  </label>
                )}
                <label className="flex flex-col gap-0.5">
                  <span className="text-[10px] text-gray-500">CV folds</span>
                  <input type="number" min={2} max={10} value={cvFolds}
                    onChange={(e) => setCvFolds(Number(e.target.value))}
                    className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                </label>
                {model === "gradient_boosting" && (
                  <label className="flex flex-col gap-0.5">
                    <span className="text-[10px] text-gray-500">Learning rate</span>
                    <input type="number" min={0.01} max={1} step={0.01} value={learningRate}
                      onChange={(e) => setLearningRate(Number(e.target.value))}
                      className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                  </label>
                )}
                {isPredictiveModel && (
                  <label className="flex flex-col gap-0.5">
                    <span className="text-[10px] text-gray-500">Holdout fraction</span>
                    <input type="number" min={0.1} max={0.5} step={0.05} value={holdoutFrac}
                      onChange={(e) => setHoldoutFrac(Number(e.target.value))}
                      className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                  </label>
                )}
              </div>
              {model === "svm_rbf" && (
                <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
                  <input type="checkbox" className="accent-indigo-500" checked={spline}
                    onChange={(e) => setSpline(e.target.checked)} />
                  Cubic-spline features
                  <Tip text="Expand each numeric predictor into a natural cubic-spline (B-spline) basis before the SVM — captures non-linear effects (the 'Spline-SVM' approach)." />
                </label>
              )}
              {!isPredictiveModel && (
                <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
                  <input type="checkbox" className="accent-indigo-500" checked={classWeight}
                    onChange={(e) => setClassWeight(e.target.checked)} />
                  Balance classes
                  <Tip text="class_weight='balanced' — reweights minority class. Use for imbalanced outcomes (rare events). Ignored for regression." />
                </label>
              )}

              <button onClick={run} disabled={loading}
                className="w-full px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
                {loading ? "Training…" : isPredictiveModel ? "Train (holdout + GridSearchCV)" : "Train & cross-validate"}
              </button>
              {error && <p className="text-xs text-red-500">{error}</p>}
            </div>
          </>
        }
        middle={
          result ? (
            <div className="space-y-3">
              {/* ROC (classification) or predicted-vs-actual (regression) */}
              {isPredictive ? (
                <>
                  <div className="relative panel" ref={rocRef}>
                    <Plot
                      data={[
                        { type: "scatter", mode: "lines", x: hd.roc_curve.map((p: any) => p.fpr), y: hd.roc_curve.map((p: any) => p.tpr), line: { color: pal[0], width: 2.5, shape: "hv" as const }, fill: "tozeroy", fillcolor: `${pal[0]}22`, name: "ROC", hoverinfo: "skip" as const },
                        { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: "#9ca3af", width: 1, dash: "dash" as const }, name: "ref", hoverinfo: "skip" as const },
                      ]}
                      layout={{
                        ...baseLayout,
                        title: { text: `${result.model} — holdout ROC`, font: { color: "#374151", size: 13 } },
                        xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, title: { text: "1 − Specificity (FPR)" }, range: [0, 1], zeroline: false },
                        yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, title: { text: "Sensitivity (TPR)" }, range: [0, 1.05], zeroline: false },
                        showlegend: false,
                        annotations: [{ x: 0.98, y: 0.04, xref: "paper" as const, yref: "paper" as const, text: `AUC = ${hd.auc.toFixed(2)} · Brier ${hd.brier.toFixed(2)}`, showarrow: false, font: { color: "#374151", size: 13 }, bgcolor: "rgba(249,250,251,0.92)", bordercolor: "#9ca3af", borderwidth: 1, borderpad: 6, xanchor: "right" as const, yanchor: "bottom" as const }],
                      }}
                      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                      style={{ width: "100%", height: 360 }} useResizeHandler />
                    <PlotExporter plotRef={rocRef} title="ML_Holdout_ROC" />
                  </div>
                  {(result.pdp ?? []).map((p: any) => (
                    <PdpPlot key={p.feature} feature={p.feature} x={p.x} y={p.y} baseLayout={baseLayout} showGrid={showGrid} color={pal[0]} />
                  ))}
                </>
              ) : isClass ? (
                <div className="relative panel" ref={rocRef}>
                  <Plot
                    data={[
                      {
                        type: "scatter", mode: "lines",
                        x: result.roc_curve.map((p: any) => p.fpr),
                        y: result.roc_curve.map((p: any) => p.tpr),
                        line: { color: pal[0], width: 2.5, shape: "hv" as const },
                        fill: "tozeroy", fillcolor: `${pal[0]}22`,
                        name: "ROC", hoverinfo: "skip" as const,
                      },
                      {
                        type: "scatter", mode: "lines",
                        x: [0, 1], y: [0, 1],
                        line: { color: "#9ca3af", width: 1, dash: "dash" as const },
                        name: "Reference", hoverinfo: "skip" as const,
                      },
                    ]}
                    layout={{
                      ...baseLayout,
                      title: { text: `${result.model} ROC — ${result.outcome}`, font: { color: "#374151", size: 13 } },
                      xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, title: { text: "1 − Specificity (FPR)" }, range: [0, 1], zeroline: false },
                      yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, title: { text: "Sensitivity (TPR)" }, range: [0, 1.05], zeroline: false },
                      showlegend: false,
                      annotations: [{
                        x: 0.98, y: 0.04, xref: "paper" as const, yref: "paper" as const,
                        text: result.auc_ci_low != null
                          ? `AUC = ${result.auc.toFixed(2)} (95% CI ${result.auc_ci_low.toFixed(2)}–${result.auc_ci_high.toFixed(2)})`
                          : `AUC = ${result.auc.toFixed(2)}`,
                        showarrow: false, font: { color: "#374151", size: 13 },
                        bgcolor: "rgba(249,250,251,0.92)", bordercolor: "#9ca3af", borderwidth: 1, borderpad: 6,
                        xanchor: "right" as const, yanchor: "bottom" as const,
                      }],
                    }}
                    config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                    style={{ width: "100%", height: 360 }} useResizeHandler />
                  <PlotExporter plotRef={rocRef} title="ML_ROC" />
                </div>
              ) : (
                <div className="relative panel" ref={scatterRef}>
                  <Plot
                    data={[
                      {
                        type: "scatter", mode: "markers",
                        x: result.scatter.map((p: any) => p.actual),
                        y: result.scatter.map((p: any) => p.predicted),
                        marker: { color: pal[0], size: 5, opacity: 0.5 },
                        name: "obs", hovertemplate: "actual %{x:.2f}<br>pred %{y:.2f}<extra></extra>",
                      },
                      (() => {
                        const xs = result.scatter.map((p: any) => p.actual);
                        const lo = Math.min(...xs), hi = Math.max(...xs);
                        return { type: "scatter" as const, mode: "lines" as const, x: [lo, hi], y: [lo, hi],
                          line: { color: "#9ca3af", width: 1, dash: "dash" as const }, name: "y=x", hoverinfo: "skip" as const };
                      })(),
                    ]}
                    layout={{
                      ...baseLayout,
                      title: { text: `${result.model} — predicted vs actual (${result.outcome})`, font: { color: "#374151", size: 13 } },
                      xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, title: { text: "Actual" } },
                      yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, title: { text: "Predicted (out-of-fold)" } },
                      showlegend: false,
                    }}
                    config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                    style={{ width: "100%", height: 360 }} useResizeHandler />
                  <PlotExporter plotRef={scatterRef} title="ML_PredVsActual" />
                </div>
              )}

              {/* Feature importance bar */}
              {!isPredictive && topImp.length > 0 && (
                <div className="relative panel" ref={impRef}>
                  <Plot
                    data={[{
                      type: "bar", orientation: "h",
                      x: topImp.map((d: any) => d.permutation),
                      y: topImp.map((d: any) => d.feature),
                      error_x: { type: "data", array: topImp.map((d: any) => d.permutation_sd), visible: true, color: "#9ca3af" },
                      marker: { color: pal[1] ?? "#6366f1" },
                      hovertemplate: "%{y}<br>Δ%{x:.4f}<extra></extra>",
                    }]}
                    layout={{
                      ...baseLayout,
                      title: { text: `Permutation importance (${isClass ? "ΔAUC" : "ΔR²"})`, font: { color: "#374151", size: 13 } },
                      xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, title: { text: isClass ? "Drop in AUC when shuffled" : "Drop in R² when shuffled" } },
                      yaxis: { ...(baseLayout.yaxis as object), showgrid: false, automargin: true },
                      margin: { t: 40, r: 20, b: 50, l: 10 },
                    }}
                    config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                    style={{ width: "100%", height: Math.max(240, topImp.length * 26 + 80) }} useResizeHandler />
                  <PlotExporter plotRef={impRef} title="ML_Importance" />
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-[360px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
              Configure a model and train to see ROC / importance
            </div>
          )
        }
        right={
          result ? (
            <>
              {isPredictive && (
                <div className="panel space-y-2">
                  <h4 className="text-sm font-semibold text-gray-800">{result.model}</h4>
                  <div className="grid grid-cols-2 gap-1.5">
                    {[
                      ["Holdout AUC", hd.auc?.toFixed(3)],
                      ["Brier", hd.brier?.toFixed(3)],
                      ["O/E", hd.oe_ratio != null ? hd.oe_ratio.toFixed(2) : "—"],
                      ["CV AUC", result.cv_best_auc?.toFixed(3)],
                      ["Train n", result.n_train],
                      ["Test n", result.n_test],
                      ["Features", result.n_features],
                      ["Pos class", result.positive_class],
                    ].map(([k, v]) => (
                      <div key={String(k)} className="bg-gray-50 border border-gray-200 rounded p-1.5 text-center">
                        <p className="text-[9px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-xs font-mono">{v}</p>
                      </div>
                    ))}
                  </div>
                  {hd.confusion && (
                    <div className="text-[11px] text-gray-600 grid grid-cols-2 gap-1 mt-1">
                      <div className="bg-emerald-50 border border-emerald-100 rounded px-2 py-1">TP {hd.confusion.tp} · TN {hd.confusion.tn}</div>
                      <div className="bg-rose-50 border border-rose-100 rounded px-2 py-1">FP {hd.confusion.fp} · FN {hd.confusion.fn}</div>
                    </div>
                  )}
                  <p className="text-[10px] text-gray-400">Best: {Object.entries(result.best_params ?? {}).map(([k, v]) => `${k}=${v}`).join(", ") || "—"}</p>
                </div>
              )}
              {isPredictive && result.selected_coefficients && (
                <div className="panel space-y-1">
                  <h4 className="text-sm font-semibold text-gray-700">Lasso selected — {result.selected_coefficients.length} non-zero</h4>
                  <div className="overflow-auto rounded-lg border border-gray-200 max-h-60">
                    <table className="w-full text-[11px] border-collapse">
                      <thead className="sticky top-0 bg-gray-50 text-gray-500"><tr><th className="text-left px-1.5 py-1">Feature</th><th className="text-right px-1.5 py-1">β</th><th className="text-right px-1.5 py-1">OR</th></tr></thead>
                      <tbody>
                        {result.selected_coefficients.map((d: any) => (
                          <tr key={d.feature} className="border-b border-gray-100">
                            <td className="px-1.5 py-1 font-mono text-gray-700 truncate max-w-[140px]">{d.feature}</td>
                            <td className="px-1.5 py-1 font-mono text-right text-indigo-700">{d.coef.toFixed(3)}</td>
                            <td className="px-1.5 py-1 font-mono text-right text-gray-500">{d.or.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              {isPredictive && hd.calibration?.length > 0 && (
                <div className="panel space-y-1">
                  <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-1">Calibration (holdout)<Tip text="Observed vs mean predicted probability per decile on the held-out test set. Near-diagonal = well calibrated." /></h4>
                  <table className="w-full text-[11px]">
                    <thead className="text-gray-400"><tr><th className="text-left px-1 py-0.5">Pred</th><th className="text-left px-1 py-0.5">Obs</th><th className="text-right px-1 py-0.5">n</th></tr></thead>
                    <tbody>
                      {hd.calibration.map((c: any, i: number) => (
                        <tr key={i} className="border-t border-gray-100"><td className="px-1 py-0.5 font-mono">{c.pred.toFixed(3)}</td><td className="px-1 py-0.5 font-mono">{c.obs.toFixed(3)}</td><td className="px-1 py-0.5 font-mono text-right text-gray-400">{c.n}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {/* Metric tiles */}
              {!isPredictive && (
              <div className="panel space-y-2">
                <h4 className="text-sm font-semibold text-gray-800">{result.model}</h4>
                {isClass ? (
                  <div className="grid grid-cols-2 gap-1.5">
                    {[
                      ["AUC", result.auc?.toFixed(3)],
                      ["Accuracy", (result.accuracy * 100).toFixed(1) + "%"],
                      ["Sensitivity", result.sensitivity != null ? (result.sensitivity * 100).toFixed(1) + "%" : "—"],
                      ["Specificity", result.specificity != null ? (result.specificity * 100).toFixed(1) + "%" : "—"],
                      ["PPV", result.ppv != null ? (result.ppv * 100).toFixed(1) + "%" : "—"],
                      ["NPV", result.npv != null ? (result.npv * 100).toFixed(1) + "%" : "—"],
                      ["Brier", result.brier?.toFixed(3)],
                      ["CV folds", result.cv_folds],
                    ].map(([k, v]) => (
                      <div key={String(k)} className="bg-gray-50 border border-gray-200 rounded p-1.5 text-center">
                        <p className="text-[9px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-xs font-mono">{v}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-1.5">
                    {[
                      ["R²", result.r2?.toFixed(3)],
                      ["RMSE", result.rmse?.toFixed(3)],
                      ["MAE", result.mae?.toFixed(3)],
                      ["CV folds", result.cv_folds],
                      ["n", result.n],
                      ["features", result.n_features],
                    ].map(([k, v]) => (
                      <div key={String(k)} className="bg-gray-50 border border-gray-200 rounded p-1.5 text-center">
                        <p className="text-[9px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-xs font-mono">{v}</p>
                      </div>
                    ))}
                  </div>
                )}
                {isClass && result.confusion && (
                  <div className="text-[11px] text-gray-600 grid grid-cols-2 gap-1 mt-1">
                    <div className="bg-emerald-50 border border-emerald-100 rounded px-2 py-1">TP {result.confusion.tp} · TN {result.confusion.tn}</div>
                    <div className="bg-rose-50 border border-rose-100 rounded px-2 py-1">FP {result.confusion.fp} · FN {result.confusion.fn}</div>
                  </div>
                )}
              </div>
              )}

              {/* Importance table + export */}
              {!isPredictive && (
              <div className="panel space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-gray-700">Feature importance</h4>
                  <ResultExporter
                    title={`ML_${result.model}_${result.outcome}`}
                    headers={["Feature", "Permutation", "SD", "Impurity"]}
                    rows={result.importance.map((d: any) => [d.feature, d.permutation, d.permutation_sd, d.impurity])}
                  />
                </div>
                <div className="overflow-auto rounded-lg border border-gray-200 max-h-72">
                  <table className="w-full text-[11px] border-collapse">
                    <thead className="sticky top-0 bg-gray-50 border-b border-gray-200 text-gray-500">
                      <tr>
                        <th className="text-left px-1.5 py-1 font-medium">Feature</th>
                        <th className="text-right px-1.5 py-1 font-medium">Perm</th>
                        <th className="text-right px-1.5 py-1 font-medium">Impurity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.importance.map((d: any) => (
                        <tr key={d.feature} className="border-b border-gray-100">
                          <td className="px-1.5 py-1 font-mono text-gray-700 truncate max-w-[150px]">{d.feature}</td>
                          <td className="px-1.5 py-1 font-mono text-right text-indigo-700">{d.permutation?.toFixed(4)}</td>
                          <td className="px-1.5 py-1 font-mono text-right text-gray-500">{d.impurity != null ? d.impurity.toFixed(4) : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              )}

              {/* Calibration (classification) */}
              {!isPredictive && isClass && result.calibration?.length > 0 && (
                <div className="panel space-y-1">
                  <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
                    Calibration
                    <Tip text="Observed event rate vs mean predicted probability per decile. Points near the diagonal = well calibrated." />
                  </h4>
                  <table className="w-full text-[11px]">
                    <thead className="text-gray-400">
                      <tr><th className="text-left px-1 py-0.5">Pred</th><th className="text-left px-1 py-0.5">Obs</th><th className="text-right px-1 py-0.5">n</th></tr>
                    </thead>
                    <tbody>
                      {result.calibration.map((c: any, i: number) => (
                        <tr key={i} className="border-t border-gray-100">
                          <td className="px-1 py-0.5 font-mono">{c.pred.toFixed(3)}</td>
                          <td className="px-1 py-0.5 font-mono">{c.obs.toFixed(3)}</td>
                          <td className="px-1 py-0.5 font-mono text-right text-gray-400">{c.n}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {result.interpretation && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3 py-2 text-xs text-indigo-900 leading-relaxed">
                  {result.interpretation}
                </div>
              )}

              {!isPredictive && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[10px] text-amber-800 leading-snug">
                  Cross-validated (out-of-fold) metrics — not in-sample. Tree ensembles
                  are for prediction / screening; for inference (odds ratios, p-values)
                  use the Regression tab.
                </div>
              )}
              {isPredictive && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[10px] text-amber-800 leading-snug">
                  Honest holdout metrics (model tuned by GridSearchCV on the training split only).
                  External benchmarks (e.g. EuroSCORE II) aren't built in — add their predicted
                  risks as a column to compare.
                </div>
              )}
            </>
          ) : null
        }
      />
    </div>
  );
}
