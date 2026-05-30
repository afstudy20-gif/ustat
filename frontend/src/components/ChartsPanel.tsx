import { useState, useRef } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { usePlotLayout, usePalette, useTraceDefaults } from "../plotStyle";
import { getHistogram, getScatter, getBoxplot, getBar } from "../api";
import PlotExporter from "./PlotExporter";

export default function ChartsPanel() {
  const session  = useStore((s) => s.session);
  const layout   = usePlotLayout();
  const pal      = usePalette();
  const td       = useTraceDefaults();
  if (!session) return null;

  const numCols = session.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
  const catCols = session.columns.filter((c) => c.kind === "categorical").map((c) => c.name);

  const [chartType, setChartType] = useState("histogram");
  const [x, setX] = useState(numCols[0] ?? "");
  const [y, setY] = useState(numCols[1] ?? "");
  const [color, setColor] = useState("");
  const [bins, setBins] = useState(20);
  const [plotData, setPlotData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // States for custom labels
  const [customTitle, setCustomTitle] = useState("");
  const [customXLabel, setCustomXLabel] = useState("");
  const [customYLabel, setCustomYLabel] = useState("");

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const base = { session_id: session.session_id, x, bins };
      let res: any;
      if (chartType === "histogram") res = await getHistogram(base);
      else if (chartType === "scatter") res = await getScatter({ ...base, y, color: color || undefined });
      else if (chartType === "boxplot" || chartType === "violin") res = await getBoxplot({ ...base, color: color || undefined });
      else res = await getBar({ ...base, y: y || undefined, color: color || undefined });
      setPlotData(res.data);

      // Auto-generate beautiful defaults
      const xMeta = session.columns.find((c) => c.name === x);
      const xLabelText = xMeta?.label || xMeta?.name || x;
      
      const yMeta = y ? session.columns.find((c) => c.name === y) : null;
      const yLabelText = yMeta ? (yMeta.label || yMeta.name || y) : "Count";
      
      const colorMeta = color ? session.columns.find((c) => c.name === color) : null;
      const colorLabelText = colorMeta ? (colorMeta.label || colorMeta.name || color) : "";

      let autoTitle = "";
      let autoX = "";
      let autoY = "";

      if (chartType === "histogram") {
        autoTitle = `Distribution of ${xLabelText}`;
        autoX = xLabelText;
        autoY = "Count";
      } else if (chartType === "scatter") {
        autoTitle = `${yLabelText} vs ${xLabelText}`;
        autoX = xLabelText;
        autoY = yLabelText;
      } else if (chartType === "boxplot" || chartType === "violin") {
        autoTitle = colorLabelText ? `Distribution of ${xLabelText} by ${colorLabelText}` : `Distribution of ${xLabelText}`;
        autoX = colorLabelText || "Overall";
        autoY = xLabelText;
      } else if (chartType === "bar") {
        autoTitle = y ? `${yLabelText} by ${xLabelText}` : `Count by ${xLabelText}`;
        autoX = xLabelText;
        autoY = yLabelText;
      }

      setCustomTitle(autoTitle);
      setCustomXLabel(autoX);
      setCustomYLabel(autoY);
    } catch (e: any) {
      setError(e.response?.data?.detail ?? "Error generating chart");
    } finally {
      setLoading(false);
    }
  };

  const chartRef = useRef<any>(null);
  const traces = plotData ? buildTraces(plotData, chartType, pal, td, session) : null;

  return (
    <div className="flex gap-4 h-full">
      {/* Controls */}
      <div className="w-60 flex-shrink-0 space-y-4 overflow-y-auto pr-1" style={{ maxHeight: "calc(100vh - 120px)" }}>
        <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
          <h3 className="text-sm font-semibold text-gray-700">Chart Type</h3>
          {["histogram", "scatter", "boxplot", "violin", "bar"].map((t) => (
            <label key={t} className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="chartType" value={t} checked={chartType === t}
                onChange={() => setChartType(t)} className="accent-indigo-500" />
              <span className="text-sm text-gray-700 capitalize">{t}</span>
            </label>
          ))}
        </div>

        <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
          <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
          <div>
            <label className="text-xs text-gray-400 block mb-1">X axis</label>
            <select className="select w-full" value={x} onChange={(e) => setX(e.target.value)}>
              {(chartType === "boxplot" || chartType === "violin" ? numCols : [...numCols, ...catCols]).map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>
          {(chartType === "scatter" || chartType === "bar") && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Y axis</label>
              <select className="select w-full" value={y} onChange={(e) => setY(e.target.value)}>
                <option value="">— count —</option>
                {numCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {chartType !== "histogram" && catCols.length > 0 && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Color / Group</label>
              <select className="select w-full" value={color} onChange={(e) => setColor(e.target.value)}>
                <option value="">None</option>
                {catCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {chartType === "histogram" && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Bins: {bins}</label>
              <input type="range" min={5} max={100} value={bins} onChange={(e) => setBins(+e.target.value)} className="w-full accent-indigo-500" />
            </div>
          )}
          <button className="btn-primary w-full mt-2" onClick={run} disabled={loading}>
            {loading ? "Generating…" : "Generate Chart"}
          </button>
          {error && <p className="text-red-500 text-xs mt-2">{error}</p>}
        </div>

        {/* Custom Labels Panel */}
        {plotData && (
          <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
            <h3 className="text-sm font-semibold text-gray-700 border-b pb-2">Custom Labels</h3>
            
            {/* Custom Chart Title */}
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">Chart Title</label>
              <input
                type="text"
                className="select w-full text-xs py-1 px-2 border rounded"
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
                placeholder="Title..."
              />
            </div>

            {/* Custom X Axis Label */}
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">X-Axis Label</label>
              <input
                type="text"
                className="select w-full text-xs py-1 px-2 border rounded"
                value={customXLabel}
                onChange={(e) => setCustomXLabel(e.target.value)}
                placeholder="X-axis label..."
              />
            </div>

            {/* Custom Y Axis Label */}
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">Y-Axis Label</label>
              <input
                type="text"
                className="select w-full text-xs py-1 px-2 border rounded"
                value={customYLabel}
                onChange={(e) => setCustomYLabel(e.target.value)}
                placeholder="Y-axis label..."
              />
            </div>
          </div>
        )}

        {/* Chart guidance */}
        <div className="panel bg-gray-50 border-gray-200 p-4 rounded-2xl">
          <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Tip</p>
          <p className="text-xs text-gray-600 leading-relaxed">{
            chartType === "histogram" ? "Shows the frequency distribution of a single numeric variable. The KDE (kernel density) overlay estimates the smooth probability density. Skewed histograms suggest using median instead of mean." :
            chartType === "scatter" ? "Reveals relationships between two continuous variables. The regression line and R² show linear fit strength. Add a Color variable to see group-specific patterns." :
            chartType === "boxplot" ? "Compares distributions across groups. The box shows Q1–Q3 (IQR), the line is the median, whiskers extend to 1.5×IQR. Points beyond whiskers are outliers." :
            chartType === "violin" ? "Combines a box plot with a kernel density estimate. The wider the violin, the more data points at that value. Better than box plots for showing bimodal or skewed distributions." :
            "Shows counts or aggregated values for categories. Use for comparing frequencies across groups. Add a Color variable for stacked/grouped comparisons."
          }</p>
        </div>
      </div>

      {/* Plot area */}
      <div className="flex-1 panel min-h-0 relative bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
        {traces ? (
          <>
          <div className="absolute right-4 top-4 z-10">
            <PlotExporter plotRef={chartRef} title={`Chart_${chartType}_${x}`} />
          </div>
          <Plot
            ref={chartRef}
            data={traces}
            layout={{
              ...layout,
              title: { text: customTitle || plotData?.x || "", font: { color: "#374151", size: 13, weight: "bold" } },
              xaxis: {
                ...(layout.xaxis as any),
                title: customXLabel ? { text: customXLabel, font: { size: 11 } } : undefined,
              },
              yaxis: {
                ...(layout.yaxis as any),
                title: customYLabel ? { text: customYLabel, font: { size: 11 } } : undefined,
              },
              autosize: true
            }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: true, displaylogo: false }}
          />
          </>
        ) : (
          <div className="h-full flex items-center justify-center text-gray-400">
            Configure and generate a chart
          </div>
        )}
      </div>
    </div>
  );
}

function buildTraces(d: any, chartType: string, C: string[], td: { lineWidth: number; markerSize: number; markerOpacity: number }, session: any): any[] | null {
  if (!d) return null;

  if (d.type === "histogram") {
    return [
      {
        type: "bar",
        x: d.bins.map((b: any) => (b.x0 + b.x1) / 2),
        y: d.bins.map((b: any) => b.count),
        marker: { color: C[0], opacity: 0.8 },
        name: "Count",
      },
      {
        type: "scatter",
        x: d.kde.map((k: any) => k.x),
        y: d.kde.map((k: any) => k.y * d.bins.reduce((a: number, b: any) => a + b.count, 0) * ((d.bins[0].x1 - d.bins[0].x0))),
        mode: "lines",
        line: { color: C[1], width: td.lineWidth },
        name: "KDE",
        yaxis: "y",
      },
    ];
  }

  if (d.type === "scatter") {
    if (d.color) {
      const colorColMeta = session.columns.find((c: any) => c.name === d.color);
      const colorLabels = colorColMeta?.value_labels ?? {};
      const groups = [...new Set(d.points.map((p: any) => p[d.color]))];
      return [
        ...groups.map((g, i) => ({
          type: "scatter",
          mode: "markers",
          name: colorLabels[String(g)] ?? String(g),
          x: d.points.filter((p: any) => p[d.color] === g).map((p: any) => p[d.x]),
          y: d.points.filter((p: any) => p[d.color] === g).map((p: any) => p[d.y]),
          marker: { color: C[i % C.length], size: td.markerSize, opacity: td.markerOpacity },
        })),
        {
          type: "scatter", mode: "lines",
          x: d.regression.line_x, y: d.regression.line_y,
          line: { color: "#374151", width: 1.5, dash: "dash" },
          name: `Fit (R²=${d.regression.r2.toFixed(3)})`,
        },
      ];
    }
    return [
      {
        type: "scatter", mode: "markers",
        x: d.points.map((p: any) => p[d.x]),
        y: d.points.map((p: any) => p[d.y]),
        marker: { color: C[0], size: td.markerSize, opacity: td.markerOpacity },
        name: d.y,
      },
      {
        type: "scatter", mode: "lines",
        x: d.regression.line_x, y: d.regression.line_y,
        line: { color: C[1], width: td.lineWidth },
        name: `Fit (R²=${d.regression.r2.toFixed(3)})`,
      },
    ];
  }

  if (d.type === "boxplot") {
    const colorColMeta = session.columns.find((c: any) => c.name === d.color);
    const colorLabels = colorColMeta?.value_labels ?? {};
    
    if (chartType === "violin") {
      return d.groups.map((g: any, i: number) => ({
        type: "violin",
        y: g.values,
        name: colorLabels[String(g.group)] ?? g.group,
        box: { visible: true },
        meanline: { visible: true },
        line: { color: C[i % C.length] },
        fillcolor: C[i % C.length] + "25",
        points: g.values.length < 200 ? "all" : false,
        jitter: 0.3,
        pointpos: -1.5,
        marker: { color: C[i % C.length], size: 3, opacity: 0.5 },
      }));
    }
    return d.groups.map((g: any, i: number) => ({
      type: "box",
      y: g.values,
      name: colorLabels[String(g.group)] ?? g.group,
      marker: { color: C[i % C.length] },
      boxpoints: d.groups[0].values.length < 500 ? "outliers" : false,
      text: g.row_indices?.map((idx: number) => `Row ${idx + 1}`),
      hovertemplate: "%{y}<br>%{text}<extra>%{fullData.name}</extra>",
    }));
  }

  if (d.type === "bar") {
    const xColMeta = session.columns.find((c: any) => c.name === d.x);
    const xLabels = xColMeta?.value_labels ?? {};
    return [{
      type: "bar",
      x: d.data.map((r: any) => xLabels[String(r.label)] ?? r.label),
      y: d.data.map((r: any) => r.value),
      marker: { color: C[0] },
    }];
  }

  return null;
}

