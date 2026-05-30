import { useState, useRef } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { usePlotLayout, usePalette, useTraceDefaults } from "../plotStyle";
import { getHistogram, getScatter, getBoxplot, getBar, getGroupedBar } from "../api";
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
  const [errorType, setErrorType] = useState("ci95");
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
      else if (chartType === "grouped") res = await getGroupedBar({ session_id: session.session_id, x, series: color || undefined, y: y || undefined, error: errorType });
      else res = await getBar({ ...base, y: y || undefined, color: color || undefined });
      setPlotData(res.data);
    } catch (e: any) {
      setError(e.response?.data?.detail ?? "Error generating chart");
    } finally {
      setLoading(false);
    }
  };

  const chartRef = useRef<any>(null);
  const traces = plotData ? buildTraces(plotData, chartType, pal, td) : null;

  return (
    <div className="flex gap-4 h-full">
      {/* Controls */}
      <div className="w-60 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Chart Type</h3>
          {["histogram", "scatter", "boxplot", "violin", "bar", "grouped"].map((t) => (
            <label key={t} className="flex items-center gap-2 cursor-pointer">
              <input type="radio" name="chartType" value={t} checked={chartType === t}
                onChange={() => setChartType(t)} className="accent-indigo-500" />
              <span className="text-sm text-gray-700 capitalize">{t === "grouped" ? "Grouped bar (± CI)" : t}</span>
            </label>
          ))}
        </div>

        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
          <div>
            <label className="text-xs text-gray-400 block mb-1">{chartType === "grouped" ? "Subgroup (x)" : "X axis"}</label>
            <select className="select w-full" value={x} onChange={(e) => setX(e.target.value)}>
              {(chartType === "boxplot" || chartType === "violin" ? numCols
                : chartType === "grouped" ? [...catCols, ...numCols]
                : [...numCols, ...catCols]).map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>
          {(chartType === "scatter" || chartType === "bar" || chartType === "grouped") && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">{chartType === "grouped" ? "Value (y) — mean, or 0/1 → proportion" : "Y axis"}</label>
              <select className="select w-full" value={y} onChange={(e) => setY(e.target.value)}>
                <option value="">— count —</option>
                {numCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {chartType !== "histogram" && catCols.length > 0 && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">{chartType === "grouped" ? "Series (paired bars)" : "Color / Group"}</label>
              <select className="select w-full" value={color} onChange={(e) => setColor(e.target.value)}>
                <option value="">None</option>
                {catCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {chartType === "grouped" && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Error bars</label>
              <select className="select w-full" value={errorType} onChange={(e) => setErrorType(e.target.value)}>
                <option value="ci95">95% CI</option>
                <option value="se">± SE</option>
              </select>
              <p className="text-[10px] text-gray-400 mt-1">A binary 0/1 value is shown as a proportion with a Wilson 95% CI.</p>
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

        {/* Chart guidance */}
        <div className="panel bg-gray-50 border-gray-200">
          <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-1">Tip</p>
          <p className="text-xs text-gray-600 leading-relaxed">{
            chartType === "histogram" ? "Shows the frequency distribution of a single numeric variable. The KDE (kernel density) overlay estimates the smooth probability density. Skewed histograms suggest using median instead of mean." :
            chartType === "scatter" ? "Reveals relationships between two continuous variables. The regression line and R\u00B2 show linear fit strength. Add a Color variable to see group-specific patterns." :
            chartType === "boxplot" ? "Compares distributions across groups. The box shows Q1\u2013Q3 (IQR), the line is the median, whiskers extend to 1.5\u00D7IQR. Points beyond whiskers are outliers." :
            chartType === "violin" ? "Combines a box plot with a kernel density estimate. The wider the violin, the more data points at that value. Better than box plots for showing bimodal or skewed distributions." :
            chartType === "grouped" ? "Clustered bars by subgroup with error bars — like the 'statin use across subgroups' figure. Pick a Subgroup (x), a Series (the paired bars) and a Value: a continuous value gives mean ± 95% CI / SE; a binary 0/1 value gives a proportion with a Wilson 95% CI." :
            "Shows counts or aggregated values for categories. Use for comparing frequencies across groups. Add a Color variable for stacked/grouped comparisons."
          }</p>
        </div>
      </div>

      {/* Plot area */}
      <div className="flex-1 panel min-h-0 relative">
        {traces ? (
          <>
          <PlotExporter plotRef={chartRef} title={`Chart_${chartType}_${x}`} />
          <Plot
            ref={chartRef}
            data={traces}
            layout={{
              ...layout,
              title: {
                text: chartType === "grouped" && plotData
                  ? `${plotData.value_kind === "proportion" ? "Proportion" : (plotData.y ?? "Count")} by ${plotData.x}`
                  : (plotData?.x ?? ""),
                font: { color: "#374151" },
              },
              autosize: true,
              ...(chartType === "grouped" && plotData ? {
                barmode: "group" as const,
                bargap: 0.28,
                bargroupgap: 0.08,
                xaxis: { ...(layout as any).xaxis, title: { text: plotData.x } },
                yaxis: {
                  ...(layout as any).yaxis,
                  rangemode: "tozero" as const,
                  title: { text: plotData.value_kind === "proportion" ? "Proportion" : plotData.value_kind === "mean" ? (plotData.y ?? "Mean") : "Count" },
                },
              } : {}),
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

function buildTraces(d: any, chartType: string, C: string[], td: { lineWidth: number; markerSize: number; markerOpacity: number }): any[] | null {
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
      const groups = [...new Set(d.points.map((p: any) => p[d.color]))];
      return [
        ...groups.map((g, i) => ({
          type: "scatter",
          mode: "markers",
          name: String(g),
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
    if (chartType === "violin") {
      return d.groups.map((g: any, i: number) => ({
        type: "violin",
        y: g.values,
        name: g.group,
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
      name: g.group,
      marker: { color: C[i % C.length] },
      boxpoints: d.groups[0].values.length < 500 ? "outliers" : false,
      text: g.row_indices?.map((idx: number) => `Row ${idx + 1}`),
      hovertemplate: "%{y}<br>%{text}<extra>%{fullData.name}</extra>",
    }));
  }

  if (d.type === "bar") {
    return [{
      type: "bar",
      x: d.data.map((r: any) => r.label),
      y: d.data.map((r: any) => r.value),
      marker: { color: C[0] },
    }];
  }

  if (d.type === "grouped_bar") {
    const withErr = d.value_kind !== "count";
    return d.groups.map((g: any, i: number) => ({
      type: "bar",
      name: String(g.series),
      x: g.x,
      y: g.value,
      marker: { color: C[i % C.length] },
      error_y: withErr ? {
        type: "data",
        symmetric: false,
        array:      g.value.map((v: number, j: number) => Math.max(0, g.err_high[j] - v)),
        arrayminus: g.value.map((v: number, j: number) => Math.max(0, v - g.err_low[j])),
        color: "#374151",
        thickness: 1.2,
        width: 4,
      } : undefined,
      customdata: g.n,
      hovertemplate: `%{x} · ${g.series}<br>%{y:.3f}<br>n = %{customdata}<extra></extra>`,
    }));
  }

  return null;
}
