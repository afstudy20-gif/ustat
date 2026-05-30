import { useState, useRef, useEffect } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { usePlotLayout, usePalette } from "../plotStyle";
import { runSubgroupBar, getUniqueValues } from "../api";
import PlotExporter from "./PlotExporter";

export default function SubgroupBarPanel() {
  const session = useStore((s) => s.session);
  const layout = usePlotLayout();
  const pal = usePalette();
  const showGrid = useStore((s) => s.showGrid);

  if (!session) return null;

  const catCols = session.columns.filter((c) => c.kind === "categorical").map((c) => c.name);
  const numCols = session.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
  const allCols = session.columns.map((c) => c.name);

  // States
  const [yCol, setYCol] = useState(allCols[0] ?? "");
  const [yMode, setYMode] = useState("mean"); // "mean" | "percentage"
  const [targetValue, setTargetValue] = useState("");
  const [subgroupCol, setSubgroupCol] = useState(catCols[0] ?? "");
  const [xaxisCol, setXaxisCol] = useState(catCols[1] ?? catCols[0] ?? "");
  const [colorCol, setColorCol] = useState("");
  const [errorType, setErrorType] = useState("ci"); // "ci" | "se" | "sd" | "none"

  const [uniqueValues, setUniqueValues] = useState<string[]>([]);
  const [plotData, setPlotData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [chartTitle, setChartTitle] = useState("");

  const chartRef = useRef<any>(null);

  // Auto-switch mode or fetch unique values when Y column changes
  useEffect(() => {
    if (!yCol) return;
    const colMeta = session.columns.find((c) => c.name === yCol);
    if (colMeta?.kind === "categorical") {
      setYMode("percentage");
    } else {
      setYMode("mean");
    }
    
    // Fetch unique values for target category selection
    getUniqueValues(session.session_id, yCol)
      .then((res) => {
        const vals = res.data.map((v: any) => String(v));
        setUniqueValues(vals);
        setTargetValue(vals[0] ?? "");
      })
      .catch(() => {
        setUniqueValues([]);
        setTargetValue("");
      });
  }, [yCol, session.session_id, session.columns]);

  // Fetch unique values if mode changes to percentage
  useEffect(() => {
    if (yMode === "percentage" && uniqueValues.length === 0 && yCol) {
      getUniqueValues(session.session_id, yCol)
        .then((res) => {
          const vals = res.data.map((v: any) => String(v));
          setUniqueValues(vals);
          setTargetValue(vals[0] ?? "");
        })
        .catch(() => {});
    }
  }, [yMode, yCol, session.session_id, uniqueValues.length]);

  const run = async () => {
    if (!yCol || !subgroupCol || !xaxisCol) {
      setError("Please configure Y-axis, Subgroup, and X-axis variables.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await runSubgroupBar({
        session_id: session.session_id,
        y_col: yCol,
        subgroup_col: subgroupCol,
        xaxis_col: xaxisCol,
        color_col: colorCol || undefined,
        y_mode: yMode,
        target_value: yMode === "percentage" ? targetValue : undefined,
        error_type: errorType,
      });
      setPlotData(res.data);
      
      // Auto-generate a beautiful descriptive title
      const modeText = yMode === "percentage" ? `% Use / Event Rate of ${yCol} (${targetValue})` : `Mean of ${yCol}`;
      const colorText = colorCol ? ` by ${colorCol}` : "";
      setChartTitle(`${modeText} in Subgroups of ${subgroupCol}${colorText}`);
    } catch (e: any) {
      setError(e.response?.data?.detail ?? "Error generating subgroup chart");
    } finally {
      setLoading(false);
    }
  };

  // Build Plotly-ready traces
  const traces = plotData ? buildPlotlyTraces(plotData, pal) : null;

  // Layout Configuration
  const yAxisTitle = yMode === "percentage" ? `% ${yCol} (${targetValue})` : `Mean of ${yCol}`;
  const fullLayout = plotData ? {
    ...layout,
    title: { text: chartTitle, font: { color: "#374151", size: 13, weight: "bold" } },
    xaxis: {
      ...layout.xaxis,
      type: "multicategory",
      showgrid: showGrid,
      gridcolor: showGrid ? "#e5e7eb" : "transparent",
      dividercolor: "#9ca3af",
      dividerwidth: 1,
      tickfont: { size: 10 },
    },
    yaxis: {
      ...layout.yaxis,
      title: { text: yAxisTitle, font: { size: 11 } },
      showgrid: showGrid,
      gridcolor: showGrid ? "#e5e7eb" : "transparent",
      tickfont: { size: 10 },
    },
    barmode: "group", // side-by-side grouped bars
    legend: {
      orientation: "h",
      yanchor: "bottom",
      y: 1.02,
      xanchor: "right",
      x: 1,
      font: { size: 10 },
    },
    margin: { t: 70, b: 60, l: 60, r: 20 },
    autosize: true,
  } : null;

  return (
    <div className="flex gap-4 h-full">
      {/* Controls panel */}
      <div className="w-72 flex-shrink-0 space-y-4 overflow-y-auto pr-1" style={{ maxHeight: "calc(100vh - 120px)" }}>
        <div className="panel space-y-4">
          <h3 className="text-sm font-semibold text-gray-800 border-b pb-2">Subgroup Bar Settings</h3>
          
          {/* Y Axis Variable */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Y-Axis Variable (Value)</label>
            <select className="select w-full" value={yCol} onChange={(e) => setYCol(e.target.value)}>
              {allCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Y Summary Mode */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Summary Mode</label>
            <div className="flex gap-4 mt-1">
              <label className="flex items-center gap-1.5 cursor-pointer text-xs text-gray-700">
                <input
                  type="radio"
                  name="yMode"
                  value="mean"
                  checked={yMode === "mean"}
                  disabled={session.columns.find((c) => c.name === yCol)?.kind === "categorical"}
                  onChange={() => setYMode("mean")}
                  className="accent-indigo-500"
                />
                Mean
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer text-xs text-gray-700">
                <input
                  type="radio"
                  name="yMode"
                  value="percentage"
                  checked={yMode === "percentage"}
                  onChange={() => setYMode("percentage")}
                  className="accent-indigo-500"
                />
                Percentage (%)
              </label>
            </div>
          </div>

          {/* Target Category Value (Only if Percentage Mode) */}
          {yMode === "percentage" && uniqueValues.length > 0 && (
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">Target Event / Category</label>
              <select className="select w-full text-xs" value={targetValue} onChange={(e) => setTargetValue(e.target.value)}>
                {uniqueValues.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Subgroup Variable (Outer X-Axis) */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Subgroup Variable (Outer X-Axis)</label>
            <select className="select w-full" value={subgroupCol} onChange={(e) => setSubgroupCol(e.target.value)}>
              {catCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <p className="text-[10px] text-gray-400 mt-0.5">Creates separate panels/divisions (e.g. Normotensive, Hypertensive)</p>
          </div>

          {/* X Axis Variable (Inner X-Axis) */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Inner Variable (Inner X-Axis)</label>
            <select className="select w-full" value={xaxisCol} onChange={(e) => setXaxisCol(e.target.value)}>
              {catCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <p className="text-[10px] text-gray-400 mt-0.5">Defines subgroups within each panel (e.g. Year 0 vs Year 6)</p>
          </div>

          {/* Color / Group Variable */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Color / Legend Variable (Bars)</label>
            <select className="select w-full" value={colorCol} onChange={(e) => setColorCol(e.target.value)}>
              <option value="">— None (Single Bar Group) —</option>
              {catCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <p className="text-[10px] text-gray-400 mt-0.5">Groups side-by-side bars (e.g. Intervention vs Comparison)</p>
          </div>

          {/* Error Bars Type */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Error Bars</label>
            <select className="select w-full text-xs" value={errorType} onChange={(e) => setErrorType(e.target.value)}>
              <option value="ci">95% Confidence Interval (CI)</option>
              <option value="se">Standard Error (SE)</option>
              <option value="sd">Standard Deviation (SD)</option>
              <option value="none">No Error Bars</option>
            </select>
          </div>

          <button className="btn-primary w-full py-2 font-medium" onClick={run} disabled={loading}>
            {loading ? "Generating Chart…" : "Generate Chart"}
          </button>
          
          {error && <p className="text-red-500 text-xs font-medium leading-relaxed">{error}</p>}
        </div>

        {/* Informational Guidance tip */}
        <div className="panel bg-indigo-50/50 border-indigo-100 p-3.5 space-y-2">
          <p className="text-[10px] font-bold text-indigo-700 uppercase tracking-wider">Nested Subgroup Chart Guidance</p>
          <p className="text-xs text-gray-600 leading-relaxed">
            This specialized visualization uses a **multicategory hierarchical X-axis** to display summary statistics nested across three distinct grouping variables:
          </p>
          <ul className="text-xs text-gray-600 list-disc list-inside space-y-1">
            <li>**Outer Subgroup**: Splits the chart into separate visual categories (e.g., Clinical subgroups).</li>
            <li>**Inner Subgroup**: Shows secondary groupings nested within each main subgroup (e.g., Time points).</li>
            <li>**Legend Variable**: Renders side-by-side grouped colored bars (e.g., Intervention groups).</li>
          </ul>
          <p className="text-xs text-gray-600 leading-relaxed">
            Error bars show precision of estimates (standard error, standard deviation, or confidence intervals). Exact counts (N) are visible on hover.
          </p>
        </div>
      </div>

      {/* Plot area */}
      <div className="flex-1 panel min-h-0 relative flex flex-col justify-between">
        {traces ? (
          <>
            <div className="absolute right-4 top-4 z-10">
              <PlotExporter plotRef={chartRef} title={`SubgroupBarChart_${yCol}_by_${subgroupCol}`} />
            </div>
            <div className="flex-1 min-h-0">
              <Plot
                ref={chartRef}
                data={traces}
                layout={fullLayout as any}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
                config={{ responsive: true, displayModeBar: true, displaylogo: false }}
              />
            </div>
          </>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-400 p-8 text-center space-y-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span className="text-sm font-medium text-gray-500">Configure and generate a nested subgroup bar chart</span>
            <span className="text-xs text-gray-400 max-w-sm">Select Y-axis values, subgroups and inner variables on the left panel to recreate clinical publications charts.</span>
          </div>
        )}
      </div>
    </div>
  );
}

function buildPlotlyTraces(plotData: any, pal: string[]) {
  if (!plotData || !plotData.traces) return [];
  return plotData.traces.map((t: any, i: number) => {
    // Multicategory axis requires [[outer], [inner]]
    const xArray = [t.x_subgroup, t.x_xaxis];
    
    return {
      type: "bar",
      name: t.name,
      x: xArray,
      y: t.y,
      error_y: {
        type: "data",
        array: t.error,
        visible: plotData.error_type !== "none",
        color: "#4b5563", // nice dark gray error bars
        thickness: 1.5,
        width: 5,
      },
      marker: {
        color: pal[i % pal.length],
        line: {
          color: pal[i % pal.length],
          width: 0.5,
        }
      },
      customdata: t.ns,
      hovertemplate: 
        `<b>%{x}</b><br>` +
        `Summary Value: %{y:.2f}${plotData.y_mode === 'percentage' ? '%' : ''}<br>` +
        (plotData.error_type !== 'none' ? `Error (${plotData.error_type.toUpperCase()}): %{error_y.array:.2f}<br>` : '') +
        `Sample Size (N): %{customdata}<extra>${t.name}</extra>`,
    };
  });
}
