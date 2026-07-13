import { useState, useRef, type ReactNode } from "react";
import { useStore, isNumericKind } from "../store";
import { usePlotLayout, usePalette } from "../plotStyle";
import { runRandomForest, runGradientBoosting } from "../api";
import { Tip } from "./Tip";
import TitledPlot from "./TitledPlot";
import ResultExporter from "./ResultExporter";
import ThreeCol from "./ThreeCol";
import type { PlotCaptureHandle } from "../lib/plotTypes";

type ModelKind = "random_forest" | "gradient_boosting";
type Task = "auto" | "classification" | "regression";

interface ImportanceRow {
  feature: string;
  permutation?: number;
  permutation_sd?: number;
  impurity?: number | null;
}

interface RocPoint {
  fpr: number;
  tpr: number;
}

interface ScatterPoint {
  actual: number;
  predicted: number;
}

interface CalibrationPoint {
  pred: number;
  obs: number;
  n: number;
}

interface MLResult {
  model: string;
  outcome: string;
  task: "classification" | "regression";
  cv_folds: number;
  n?: number;
  n_features?: number;
  importance: ImportanceRow[];
  roc_curve: RocPoint[];
  scatter: ScatterPoint[];
  auc?: number;
  auc_ci_low?: number | null;
  auc_ci_high?: number;
  accuracy: number;
  sensitivity?: number | null;
  specificity?: number | null;
  ppv?: number | null;
  npv?: number | null;
  brier?: number;
  r2?: number;
  rmse?: number;
  mae?: number;
  confusion?: { tp: number; tn: number; fp: number; fn: number };
  calibration?: CalibrationPoint[];
  interpretation?: string;
}

const MODEL_LABEL: Record<ModelKind, string> = {
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
};

export default function MLPanel() {
  const session = useStore((s) => s.session);
  const showGrid = useStore((s) => s.showGrid);
  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const rocRef = useRef<PlotCaptureHandle | null>(null);
  const impRef = useRef<PlotCaptureHandle | null>(null);
  const scatterRef = useRef<PlotCaptureHandle | null>(null);

  const columns = session?.columns ?? [];
  const sid = session?.session_id ?? "";
  const numCols = columns.filter((c) => isNumericKind(c.kind)).map((c) => c.name);

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

  const [result, setResult] = useState<MLResult | null>(null);
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
      const payload = {
        session_id: sid,
        outcome,
        predictors,
        task,
        n_estimators: nEstimators,
        max_depth: maxDepth ? parseInt(maxDepth, 10) : null,
        cv_folds: cvFolds,
        class_weight_balanced: classWeight,
        learning_rate: learningRate,
      };
      const fn = model === "random_forest" ? runRandomForest : runGradientBoosting;
      const res = await fn(payload);
      setResult(res.data as MLResult);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
      setError(
        Array.isArray(detail)
          ? detail.map((m) => (m as { msg?: string }).msg ?? String(m)).join(", ")
          : (typeof detail === "string" ? detail : (e instanceof Error ? e.message : "ML run failed")),
      );
    } finally {
      setLoading(false);
    }
  };

  const isClass = result?.task === "classification";
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
                <Tip wide text="Tree-ensemble machine learning. Random Forest = bagged decision trees (robust, little tuning). Gradient Boosting = sequential trees (often higher accuracy, more tuning). Performance is cross-validated (out-of-fold) so the reported AUC / R² is not optimistic in-sample." />
              </h3>
              <div className="flex rounded-lg overflow-hidden border border-gray-300">
                {(["random_forest", "gradient_boosting"] as const).map((m) => (
                  <button key={m} onClick={() => { setModel(m); setResult(null); }}
                    className={`flex-1 px-2 py-1.5 text-xs font-medium transition-colors ${
                      model === m ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"
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
                <label className="flex flex-col gap-0.5">
                  <span className="text-[10px] text-gray-500">Trees (n_estimators)</span>
                  <input type="number" min={10} max={2000} step={10} value={nEstimators}
                    onChange={(e) => setNEstimators(Number(e.target.value))}
                    className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                </label>
                <label className="flex flex-col gap-0.5">
                  <span className="text-[10px] text-gray-500">Max depth (blank=auto)</span>
                  <input type="number" min={1} max={50} value={maxDepth} placeholder="auto"
                    onChange={(e) => setMaxDepth(e.target.value)}
                    className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                </label>
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
              </div>
              <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
                <input type="checkbox" className="accent-indigo-500" checked={classWeight}
                  onChange={(e) => setClassWeight(e.target.checked)} />
                Balance classes
                <Tip text="class_weight='balanced' — reweights minority class. Use for imbalanced outcomes (rare events). Ignored for regression." />
              </label>

              <button onClick={run} disabled={loading}
                className="w-full px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
                {loading ? "Training…" : "Train & cross-validate"}
              </button>
              {error && <p className="text-xs text-red-500">{error}</p>}
            </div>
          </>
        }
        middle={
          result ? (
            <div className="space-y-3">
              {/* ROC (classification) or predicted-vs-actual (regression) */}
              {isClass ? (
                <div className="panel">
                  <TitledPlot
                    plotRefOut={rocRef}
                    storageKey="ml:roc"
                    data={[
                      {
                        type: "scatter", mode: "lines",
                        x: result.roc_curve.map((p) => p.fpr),
                        y: result.roc_curve.map((p) => p.tpr),
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
                      xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, range: [0, 1], zeroline: false },
                      yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, range: [0, 1.05], zeroline: false },
                      showlegend: false,
                      annotations: [{
                        x: 0.98, y: 0.04, xref: "paper" as const, yref: "paper" as const,
                        text: result.auc != null && result.auc_ci_low != null && result.auc_ci_high != null
                          ? `AUC = ${result.auc.toFixed(2)} (95% CI ${result.auc_ci_low.toFixed(2)}–${result.auc_ci_high.toFixed(2)})`
                          : `AUC = ${result.auc?.toFixed(2) ?? "—"}`,
                        showarrow: false, font: { color: "#374151", size: 13 },
                        bgcolor: "rgba(249,250,251,0.92)", bordercolor: "#9ca3af", borderwidth: 1, borderpad: 6,
                        xanchor: "right" as const, yanchor: "bottom" as const,
                      }],
                    }}
                    config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                    defaultTitle={`${result.model} ROC — ${result.outcome}`}
                    defaultSubtitle=""
                    defaultXAxis="1 − Specificity (FPR)"
                    defaultYAxis="Sensitivity (TPR)" />
                </div>
              ) : (
                <div className="panel">
                  <TitledPlot
                    plotRefOut={scatterRef}
                    storageKey="ml:predVsActual"
                    data={[
                      {
                        type: "scatter", mode: "markers",
                        x: result.scatter.map((p) => p.actual),
                        y: result.scatter.map((p) => p.predicted),
                        marker: { color: pal[0], size: 5, opacity: 0.5 },
                        name: "obs", hovertemplate: "actual %{x:.2f}<br>pred %{y:.2f}<extra></extra>",
                      },
                      (() => {
                        const xs = result.scatter.map((p) => p.actual);
                        const lo = Math.min(...xs), hi = Math.max(...xs);
                        return { type: "scatter" as const, mode: "lines" as const, x: [lo, hi], y: [lo, hi],
                          line: { color: "#9ca3af", width: 1, dash: "dash" as const }, name: "y=x", hoverinfo: "skip" as const };
                      })(),
                    ]}
                    layout={{
                      ...baseLayout,
                      xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid },
                      yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid },
                      showlegend: false,
                    }}
                    config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                    defaultTitle={`${result.model} — predicted vs actual (${result.outcome})`}
                    defaultSubtitle=""
                    defaultXAxis="Actual"
                    defaultYAxis="Predicted (out-of-fold)" />
                </div>
              )}

              {/* Feature importance bar */}
              {topImp.length > 0 && (
                <div className="panel">
                  <TitledPlot
                    plotRefOut={impRef}
                    storageKey="ml:importance"
                    data={[{
                      type: "bar", orientation: "h",
                      x: topImp.map((d) => d.permutation),
                      y: topImp.map((d) => d.feature),
                      error_x: { type: "data", array: topImp.map((d) => d.permutation_sd), visible: true, color: "#9ca3af" },
                      marker: { color: pal[1] ?? "#6366f1" },
                      hovertemplate: "%{y}<br>Δ%{x:.4f}<extra></extra>",
                    }]}
                    layout={{
                      ...baseLayout,
                      height: Math.max(240, topImp.length * 26 + 80),
                      xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid },
                      yaxis: { ...(baseLayout.yaxis as object), showgrid: false, automargin: true },
                      margin: { t: 40, r: 20, b: 50, l: 10 },
                    }}
                    config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                    defaultTitle={`Permutation importance (${isClass ? "ΔAUC" : "ΔR²"})`}
                    defaultSubtitle=""
                    defaultXAxis={isClass ? "Drop in AUC when shuffled" : "Drop in R² when shuffled"}
                    defaultYAxis="" />
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
              {/* Metric tiles */}
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
                    {([
                      ["R²", result.r2?.toFixed(3)],
                      ["RMSE", result.rmse?.toFixed(3)],
                      ["MAE", result.mae?.toFixed(3)],
                      ["CV folds", result.cv_folds],
                      [<i>n</i>, result.n],
                      ["features", result.n_features],
                    ] as [ReactNode, ReactNode][]).map(([k, v], i) => (
                      <div key={i} className="bg-gray-50 border border-gray-200 rounded p-1.5 text-center">
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

              {/* Importance table + export */}
              <div className="panel space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-gray-700">Feature importance</h4>
                  <ResultExporter
                    title={`ML_${result.model}_${result.outcome}`}
                    headers={["Feature", "Permutation", "SD", "Impurity"]}
                    rows={result.importance.map((d) => [d.feature, d.permutation, d.permutation_sd, d.impurity])}
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
                      {result.importance.map((d) => (
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

              {/* Calibration (classification) */}
              {isClass && (result.calibration?.length ?? 0) > 0 && (
                <div className="panel space-y-1">
                  <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
                    Calibration
                    <Tip text="Observed event rate vs mean predicted probability per decile. Points near the diagonal = well calibrated." />
                  </h4>
                  <table className="w-full text-[11px]">
                    <thead className="text-gray-400">
                      <tr><th className="text-left px-1 py-0.5">Pred</th><th className="text-left px-1 py-0.5">Obs</th><th className="text-right px-1 py-0.5"><i>n</i></th></tr>
                    </thead>
                    <tbody>
                      {(result.calibration ?? []).map((c, i: number) => (
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

              <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[10px] text-amber-800 leading-snug">
                Cross-validated (out-of-fold) metrics — not in-sample. Tree ensembles
                are for prediction / screening; for inference (odds ratios, p-values)
                use the Regression tab.
              </div>
            </>
          ) : null
        }
      />
    </div>
  );
}
