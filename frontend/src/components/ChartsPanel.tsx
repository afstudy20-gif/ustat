import { useState } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { getHistogram, getScatter, getBoxplot, getBar } from "../api";

const PLOT_LAYOUT: Record<string, unknown> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  font: { color: "#374151", size: 12 },
  margin: { t: 30, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: "#e5e7eb", zerolinecolor: "#d1d5db" },
  yaxis: { gridcolor: "#e5e7eb", zerolinecolor: "#d1d5db" },
};

export default function ChartsPanel() {
  const session  = useStore((s) => s.session);
  const showGrid = useStore((s) => s.showGrid);
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
    } catch (e: any) {
      setError(e.response?.data?.detail ?? "Error generating chart");
    } finally {
      setLoading(false);
    }
  };

  const traces = buildTraces(plotData, chartType);

  return (
    <div className="flex gap-4 h-full">
      {/* Controls */}
      <div className="w-60 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Chart Type</h3>
          {["histogram", "scatter", "boxplot", "violin", "bar"].map((t) => (
            <label key={t} className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="chartType" value={t} checked={chartType === t}
                onChange={() => setChartType(t)} className="accent-indigo-500" />
              <span className="text-sm text-gray-700 capitalize">{t}</span>
            </label>
          ))}
        </div>

        <div className="panel space-y-3">
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
          <button className="btn-primary w-full" onClick={run} disabled={loading}>
            {loading ? "Generating…" : "Generate Chart"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      {/* Plot area */}
      <div className="flex-1 panel min-h-0">
        {traces ? (
          <Plot
            data={traces}
            layout={{ ...PLOT_LAYOUT, xaxis: { ...(PLOT_LAYOUT.xaxis as object), showgrid: showGrid }, yaxis: { ...(PLOT_LAYOUT.yaxis as object), showgrid: showGrid }, title: { text: plotData?.x ?? "", font: { color: "#374151" } }, autosize: true }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: true, displaylogo: false }}
          />
        ) : (
          <div className="h-full flex items-center justify-center text-gray-400">
            Configure and generate a chart
          </div>
        )}
      </div>
    </div>
  );
}

function buildTraces(d: any, chartType = ""): any[] | null {
  if (!d) return null;
  const COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899"];

  if (d.type === "histogram") {
    return [
      {
        type: "bar",
        x: d.bins.map((b: any) => (b.x0 + b.x1) / 2),
        y: d.bins.map((b: any) => b.count),
        marker: { color: COLORS[0], opacity: 0.8 },
        name: "Count",
      },
      {
        type: "scatter",
        x: d.kde.map((k: any) => k.x),
        y: d.kde.map((k: any) => k.y * d.bins.reduce((a: number, b: any) => a + b.count, 0) * ((d.bins[0].x1 - d.bins[0].x0))),
        mode: "lines",
        line: { color: COLORS[1], width: 2 },
        name: "KDE",
        yaxis: "y",
      },
    ];
  }

  if (d.type === "scatter") {
    if (d.color) {
      const groups = [...new Set(d.points.map((p: any) => p[d.color]))];
      return [
        ...groups.map((g, i) => ({
          type: "scatter",
          mode: "markers",
          name: String(g),
          x: d.points.filter((p: any) => p[d.color] === g).map((p: any) => p[d.x]),
          y: d.points.filter((p: any) => p[d.color] === g).map((p: any) => p[d.y]),
          marker: { color: COLORS[i % COLORS.length], size: 6, opacity: 0.7 },
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
        marker: { color: COLORS[0], size: 6, opacity: 0.7 },
        name: d.y,
      },
      {
        type: "scatter", mode: "lines",
        x: d.regression.line_x, y: d.regression.line_y,
        line: { color: COLORS[1], width: 2 },
        name: `Fit (R²=${d.regression.r2.toFixed(3)})`,
      },
    ];
  }

  if (d.type === "boxplot") {
    if (chartType === "violin") {
      return d.groups.map((g: any, i: number) => ({
        type: "violin",
        y: g.values,
        name: g.group,
        box: { visible: true },
        meanline: { visible: true },
        line: { color: COLORS[i % COLORS.length] },
        fillcolor: COLORS[i % COLORS.length] + "25",
        points: g.values.length < 200 ? "all" : false,
        jitter: 0.3,
        pointpos: -1.5,
        marker: { color: COLORS[i % COLORS.length], size: 3, opacity: 0.5 },
      }));
    }
    return d.groups.map((g: any, i: number) => ({
      type: "box",
      y: g.values,
      name: g.group,
      marker: { color: COLORS[i % COLORS.length] },
      boxpoints: d.groups[0].values.length < 500 ? "outliers" : false,
    }));
  }

  if (d.type === "bar") {
    return [{
      type: "bar",
      x: d.data.map((r: any) => r.label),
      y: d.data.map((r: any) => r.value),
      marker: { color: COLORS[0] },
    }];
  }

  return null;
}
