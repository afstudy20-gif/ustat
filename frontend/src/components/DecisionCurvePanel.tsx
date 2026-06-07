/**
 * DecisionCurvePanel — Standalone DCA (Phase 13)
 *
 * Professional Decision Curve Analysis panel.
 * Supports both binary outcomes (pre-computed probabilities) and survival data (risk scores).
 *
 * Features:
 * - Flexible input (probability column or risk/LP column)
 * - Survival mode with time horizon
 * - Beautiful net benefit plot with:
 *    • Shaded "clinical benefit" region (where model beats both treat-all and treat-none)
 *    • Harm threshold annotation (vertical line where NB crosses zero)
 * - Rich right column: summary metrics, assumptions, warnings, result text
 * - Full export support
 */

import { useState, useRef } from "react";
import { useStore, isNumericKind, isCategoricalKind } from "../store";
import { runDCA } from "../api";
import { Tip } from "./Tip";
import TitledPlot from "./TitledPlot";
import ThreeCol from "./ThreeCol";

interface DcaResult {
  curves?: any;
  summary?: any;
  assumptions?: string[];
  warnings?: string[];
  result_text?: string;
  prevalence?: number;
  mode?: string;
}

export default function DecisionCurvePanel() {
  const session = useStore((s) => s.session);
  // Columns live under session in the store; there is no top-level `columns`.
  // Reading a non-existent top-level field returned undefined and crashed the
  // tab on mount (`columns.filter` → "Cannot read properties of undefined").
  const columns = session?.columns ?? [];
  const sid = session?.session_id;

  const [mode, setMode] = useState<"binary" | "survival">("survival");

  // Binary mode
  const [probCol, setProbCol] = useState("");
  const [outcomeCol, setOutcomeCol] = useState("");

  // Survival mode (recommended for Phase 12/13 workflows)
  const [durationCol, setDurationCol] = useState("");
  const [eventCol, setEventCol] = useState("");
  const [riskCol, setRiskCol] = useState("");
  const [timeHorizon, setTimeHorizon] = useState<number | "">("");

  const [result, setResult] = useState<DcaResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const numericCols = columns.filter((c) => isNumericKind(c.kind)).map((c) => c.name);
  // Binary outcome can be a 0/1 numeric or a 2-level categorical column.
  const binaryCols = columns.filter((c) => isCategoricalKind(c.kind) || isNumericKind(c.kind)).map((c) => c.name);
  const dcaPlotRef = useRef<any>(null);

  const canRun = sid && (
    (mode === "binary" && probCol && outcomeCol) ||
    (mode === "survival" && durationCol && eventCol && riskCol)
  );

  async function handleRun() {
    if (!sid || !canRun) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: any = {
        session_id: sid,
        n_thresholds: 90,
        threshold_range: mode === "survival" ? [0.02, 0.55] : [0.01, 0.80],
      };

      if (mode === "survival") {
        payload.duration_col = durationCol;
        payload.event_col = eventCol;
        payload.risk_col = riskCol;
        if (timeHorizon) payload.time_horizon = Number(timeHorizon);
      } else {
        payload.probability_col = probCol;
        payload.outcome = outcomeCol;
      }

      const res = await runDCA(payload);
      setResult(res.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "Decision curve analysis failed");
    } finally {
      setLoading(false);
    }
  }

  // Build improved Plotly data with shaded benefit region + harm threshold
  function buildPlotData() {
    if (!result?.curves) return [];

    const thresholds: number[] = result.curves.thresholds || result.curves.model?.thresholds || [];
    const modelNB: number[] = result.curves.model?.net_benefit || result.curves.model_net_benefit || [];
    const allNB: number[] = result.curves.treat_all?.net_benefit || result.curves.treat_all_net_benefit || [];
    const noneNB: number[] = result.curves.treat_none?.net_benefit || result.curves.treat_none_net_benefit || [];

    const traces: any[] = [];

    // 1. Treat None (reference)
    traces.push({
      x: thresholds,
      y: noneNB,
      type: "scatter",
      mode: "lines",
      name: "Treat None",
      line: { color: "#9ca3af", width: 1.5, dash: "dot" },
      hovertemplate: "Threshold: %{x:.3f}<br>Net Benefit: %{y:.4f}<extra>Treat None</extra>",
    });

    // 2. Treat All
    traces.push({
      x: thresholds,
      y: allNB,
      type: "scatter",
      mode: "lines",
      name: "Treat All",
      line: { color: "#f59e0b", width: 1.5, dash: "dash" },
      hovertemplate: "Threshold: %{x:.3f}<br>Net Benefit: %{y:.4f}<extra>Treat All</extra>",
    });

    // 3. Main Model curve (prominent)
    traces.push({
      x: thresholds,
      y: modelNB,
      type: "scatter",
      mode: "lines",
      name: "Model",
      line: { color: "#4338ca", width: 3.5 },
      hovertemplate: "<b>Model</b><br>Threshold: %{x:.3f}<br>Net Benefit: %{y:.4f}<extra></extra>",
    });

    // 4. Shaded "Model Superior" region (green) - where model beats the best alternative and is positive
    const superiorY = thresholds.map((_, i) => {
      const m = modelNB[i] ?? -999;
      const bestAlt = Math.max(allNB[i] ?? -999, noneNB[i] ?? -999);
      return m > bestAlt && m > 0 ? m : null;
    });

    traces.push({
      x: thresholds,
      y: superiorY,
      type: "scatter",
      mode: "lines",
      fill: "tozeroy",
      fillcolor: "rgba(16, 185, 129, 0.22)",
      line: { color: "rgba(16, 185, 129, 0)", width: 0 },
      name: "Model Superior (Net Benefit)",
      hoverinfo: "skip",
      legendgroup: "benefit",
      showlegend: true,
    });

    // 5. Light red "Harm" region - where model is worse than Treat None
    const harmY = thresholds.map((_, i) => {
      const m = modelNB[i] ?? 0;
      const n = noneNB[i] ?? 0;
      return m < n ? m : null;
    });

    traces.push({
      x: thresholds,
      y: harmY,
      type: "scatter",
      mode: "lines",
      fill: "tozeroy",
      fillcolor: "rgba(239, 68, 68, 0.12)",
      line: { color: "rgba(239, 68, 68, 0)", width: 0 },
      name: "Model Harm Zone",
      hoverinfo: "skip",
      legendgroup: "harm",
      showlegend: true,
    });

    // 6. Marker for Maximum Net Benefit point
    const maxIdx = modelNB.reduce((best, val, idx) => (val > modelNB[best] ? idx : best), 0);
    const maxX = thresholds[maxIdx];
    const maxY = modelNB[maxIdx];

    traces.push({
      x: [maxX],
      y: [maxY],
      type: "scatter",
      mode: "markers",
      name: "Maximum Net Benefit",
      marker: {
        size: 11,
        color: "#059669",
        symbol: "diamond",
        line: { color: "#065f46", width: 1.5 },
      },
      hovertemplate: `<b>Best Threshold</b><br>${maxX.toFixed(3)} → Net Benefit ${maxY.toFixed(4)}<extra></extra>`,
    });

    return traces;
  }

  function buildLayout() {
    const harmThreshold = result?.summary?.harm_threshold;
    const maxThreshold = result?.summary?.max_net_benefit_threshold;

    const shapes: any[] = [];
    const annotations: any[] = [];

    // Common clinical thresholds (subtle)
    const refThresholds = [0.05, 0.10, 0.20, 0.30];
    refThresholds.forEach((t) => {
      shapes.push({
        type: "line",
        x0: t,
        x1: t,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "#e5e7eb", width: 1, dash: "dot" },
      });
    });

    // Harm threshold (red)
    if (harmThreshold != null) {
      shapes.push({
        type: "line",
        x0: harmThreshold,
        x1: harmThreshold,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "#dc2626", width: 2.5, dash: "dash" },
      });
      annotations.push({
        x: harmThreshold,
        y: 0.88,
        xref: "x",
        yref: "paper",
        text: `Harm<br>${harmThreshold.toFixed(2)}`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 0.8,
        ax: 0,
        ay: -35,
        font: { color: "#b91c1c", size: 10, weight: 600 },
        align: "center",
        bgcolor: "rgba(254, 226, 226, 0.85)",
        borderpad: 2,
      });
    }

    // Max benefit point annotation
    if (maxThreshold != null) {
      annotations.push({
        x: maxThreshold,
        y: 1.02,
        xref: "x",
        yref: "paper",
        text: "★ Max Benefit",
        showarrow: false,
        font: { color: "#059669", size: 11, weight: 700 },
        align: "center",
      });
    }

    return {
      height: 440,
      margin: { t: 35, r: 18, b: 55, l: 52 },
      xaxis: {
        title: { text: "Threshold Probability (pt)", font: { size: 12 } },
        range: [0, 0.62],
        tick0: 0,
        dtick: 0.1,
        gridcolor: "#f3f4f6",
        zeroline: false,
      },
      yaxis: {
        title: { text: "Net Benefit", font: { size: 12 } },
        zeroline: true,
        zerolinecolor: "#9ca3af",
        zerolinewidth: 1.5,
        gridcolor: "#f3f4f6",
      },
      shapes,
      annotations,
      legend: {
        orientation: "h",
        y: -0.22,
        x: 0.5,
        xanchor: "center",
        font: { size: 11 },
        bgcolor: "rgba(255,255,255,0.9)",
      },
      hovermode: "x unified",
      hoverlabel: { bgcolor: "white", bordercolor: "#4b5563" },
      paper_bgcolor: "transparent",
      plot_bgcolor: "#fafafa",
    };
  }

  const plotData = buildPlotData();

  return (
    <div className="p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Decision Curve Analysis</h2>
          <p className="text-sm text-gray-500">Net clinical benefit across threshold probabilities (Vickers & Elkin 2006)</p>
        </div>
        <div className="flex gap-2 text-xs">
          <button
            onClick={() => setMode("binary")}
            className={`px-3 py-1 rounded-lg border ${mode === "binary" ? "bg-indigo-600 text-white border-indigo-600" : "hover:bg-gray-100"}`}
          >
            Binary Outcome
          </button>
          <button
            onClick={() => setMode("survival")}
            className={`px-3 py-1 rounded-lg border ${mode === "survival" ? "bg-indigo-600 text-white border-indigo-600" : "hover:bg-gray-100"}`}
          >
            Survival / Risk Score
          </button>
        </div>
      </div>

      <ThreeCol
        left={
          <div className="space-y-4">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Inputs</div>

            {mode === "survival" ? (
              <>
                <label className="block text-xs text-gray-600">Duration / Time column</label>
                <select value={durationCol} onChange={(e) => setDurationCol(e.target.value)} className="w-full border rounded px-3 py-2 text-sm">
                  <option value="">— select —</option>
                  {numericCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>

                <label className="block text-xs text-gray-600">Event (0/1)</label>
                <select value={eventCol} onChange={(e) => setEventCol(e.target.value)} className="w-full border rounded px-3 py-2 text-sm">
                  <option value="">— select —</option>
                  {binaryCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>

                <label className="block text-xs text-gray-600">Risk Score / Linear Predictor (higher = worse prognosis)</label>
                <select value={riskCol} onChange={(e) => setRiskCol(e.target.value)} className="w-full border rounded px-3 py-2 text-sm">
                  <option value="">— select —</option>
                  {numericCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>

                <label className="block text-xs text-gray-600">Time horizon (optional)</label>
                <input
                  type="number"
                  value={timeHorizon}
                  onChange={(e) => setTimeHorizon(e.target.value ? Number(e.target.value) : "")}
                  className="w-full border rounded px-3 py-2 text-sm"
                  placeholder="e.g. 60 months"
                />
              </>
            ) : (
              <>
                <label className="block text-xs text-gray-600">Probability / Risk column</label>
                <select value={probCol} onChange={(e) => setProbCol(e.target.value)} className="w-full border rounded px-3 py-2 text-sm">
                  <option value="">— select —</option>
                  {numericCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>

                <label className="block text-xs text-gray-600">Binary Outcome (0/1)</label>
                <select value={outcomeCol} onChange={(e) => setOutcomeCol(e.target.value)} className="w-full border rounded px-3 py-2 text-sm">
                  <option value="">— select —</option>
                  {binaryCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </>
            )}

            <button
              onClick={handleRun}
              disabled={!canRun || loading}
              className="mt-3 w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white font-medium transition-colors"
            >
              {loading ? "Calculating…" : "Run Decision Curve Analysis"}
            </button>

            {error && <p className="text-xs text-red-600 mt-1">{error}</p>}

            <Tip text="DCA answers: at which decision thresholds does using this model improve patient outcomes compared with treating everyone or no one?" />
          </div>
        }
        middle={
          result && plotData.length > 0 ? (
            <TitledPlot
              plotRefOut={dcaPlotRef}
              storageKey="dca:netbenefit"
              data={plotData}
              layout={buildLayout()}
              defaultTitle="Net Benefit Curves"
              defaultSubtitle="Green = model provides clinical value over alternatives"
              defaultXAxis="Threshold Probability (pt)"
              defaultYAxis="Net Benefit"
            />
          ) : (
            <div className="h-[420px] flex items-center justify-center text-gray-400 border border-dashed rounded-2xl">
              Select columns and run DCA to see the net benefit curves
            </div>
          )
        }
        right={
          result ? (
            <div className="space-y-4 text-sm">
              {result.summary && (
                <div className="bg-white border border-gray-200 rounded-2xl p-4 shadow-sm">
                  <div className="uppercase text-[10px] tracking-[1px] font-semibold text-gray-500 mb-3">Clinical Utility Summary</div>

                  <div className="space-y-3">
                    <div>
                      <div className="text-[11px] text-gray-500">Maximum Net Benefit</div>
                      <div className="text-2xl font-semibold text-emerald-600 tabular-nums">
                        {result.summary.max_net_benefit?.toFixed(4)}
                      </div>
                      <div className="text-xs text-gray-600 mt-0.5">
                        at threshold probability <span className="font-medium text-gray-800">{result.summary.max_net_benefit_threshold?.toFixed(3)}</span>
                      </div>
                    </div>

                    {result.summary.positive_nb_range && (
                      <div className="pt-1 border-t">
                        <div className="text-[11px] text-gray-500">Range where model adds value</div>
                        <div className="font-medium text-gray-800">
                          {result.summary.positive_nb_range[0]} — {result.summary.positive_nb_range[1]}
                        </div>
                      </div>
                    )}

                    {result.summary.interventions_avoided_per_100_at_max != null && (
                      <div className="pt-1 border-t">
                        <div className="text-[11px] text-gray-500">Interventions avoided per 100 patients</div>
                        <div className="text-xl font-semibold text-indigo-600 tabular-nums">
                          {result.summary.interventions_avoided_per_100_at_max.toFixed(1)}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {result.result_text && (
                <div className="bg-gradient-to-br from-emerald-50 to-white border border-emerald-200 rounded-2xl p-4 text-sm text-emerald-800 leading-snug shadow-sm">
                  {result.result_text}
                </div>
              )}

              {result.assumptions && result.assumptions.length > 0 && (
                <div className="text-xs">
                  <div className="font-semibold text-gray-500 mb-1.5 tracking-wide">Key Assumptions</div>
                  <ul className="text-gray-600 space-y-1 pl-1">
                    {result.assumptions.map((a, i) => (
                      <li key={i} className="flex gap-2">
                        <span className="text-emerald-400 mt-px">•</span>
                        <span>{a}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {result.warnings && result.warnings.length > 0 && (
                <div className="rounded-xl bg-amber-50 border border-amber-200 p-3 text-amber-700 text-xs">
                  <span className="font-semibold">Note:</span> {result.warnings.join(" • ")}
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-center text-gray-400 border border-dashed border-gray-200 rounded-2xl p-6 text-sm">
              Run DCA to see net benefit curves,<br />clinical utility metrics, and interpretation.
            </div>
          )
        }
      />
    </div>
  );
}
