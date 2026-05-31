import { useEffect, useState, useCallback, useRef } from "react";
import { useStore, PALETTES } from "../store";
import { usePalette } from "../plotStyle";
import api from "../api";
import Plot from "../PlotComponent";
import ResultExporter from "./ResultExporter";
import PlotExporter from "./PlotExporter";
import { useResizableRightCol } from "../hooks/useResizableRightCol";

// ── Inline sparkline SVG (real histogram / category bars) ────────────────────

interface SparkData { type: string; data: number[]; }

function Sparkline({ spark }: { spark: SparkData }) {
  const W = 40, H = 12;
  const { type, data } = spark;
  const pal = usePalette();
  if (!data || data.length === 0) return null;
  const max = Math.max(...data);
  if (max === 0) return null;

  if (type === "numeric") {
    const bw = W / data.length;
    return (
      <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
        {data.map((v, i) => {
          const bh = Math.max(1, (v / max) * H);
          return (
            <rect key={i} x={i * bw} y={H - bh}
              width={Math.max(bw - 0.5, 0.5)} height={bh}
              fill={pal[0]} opacity={0.7} rx={0.5} />
          );
        })}
      </svg>
    );
  }

  // categorical → proportional horizontal bars
  const total = data.reduce((a, b) => a + b, 0);
  const CATS = pal;
  let cx = 0;
  return (
    <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
      {data.map((v, i) => {
        const bw = (v / total) * W;
        const rect = <rect key={i} x={cx} y={0} width={Math.max(bw - 0.5, 0.5)} height={H}
          fill={CATS[i % CATS.length]} opacity={0.8} />;
        cx += bw;
        return rect;
      })}
    </svg>
  );
}

// ── Normality Deviants component with right-click context menu ────────────────

interface NormalityDeviant { row: number; value: number; z: number; abs_residual: number; }

function NormalityDeviants({ deviants, onDelete }: { deviants: NormalityDeviant[]; onDelete: () => void }) {
  const [contextMenu, setContextMenu] = useState<{ row: number; x: number; y: number } | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const handleContextMenu = (e: React.MouseEvent, row: number) => {
    e.preventDefault();
    setContextMenu({ row, x: e.clientX, y: e.clientY });
  };

  const handleDeleteRow = async (row: number) => {
    try {
      await useStore.getState().deleteRow(row);
      // Refresh the stats immediately
      onDelete();
    } catch (err) {
      console.error("Error deleting row:", err);
    }
    setContextMenu(null);
  };

  // Close context menu on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setContextMenu(null);
      }
    };
    if (contextMenu) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [contextMenu]);

  return (
    <div className="mt-2 px-3 py-2 bg-orange-50 border border-orange-100 rounded-lg">
      <p className="text-xs font-semibold text-orange-700 mb-2 flex items-center gap-1">
        <span>🔶 Top Normality Deviants (Q-Q Deviation)</span>
        <span className="font-normal text-[10px] text-orange-400">(Shows why the distribution failed normality)</span>
      </p>
      <div className="flex flex-wrap gap-1.5 relative">
        {deviants.map((e) => (
          <div
            key={e.row}
            className="group cursor-pointer relative flex items-center gap-1 text-[10px] font-mono bg-white text-orange-800 border border-orange-200 rounded px-2 py-0.5 shadow-sm hover:border-orange-400 hover:bg-orange-100 transition-all"
            onClick={async () => {
              try {
                await useStore.getState().deleteRow(e.row);
                onDelete();
              } catch (err) {}
            }}
            onContextMenu={(ev) => handleContextMenu(ev, e.row)}
            title="Click to delete or right-click for menu"
          >
            <span className="opacity-0 w-0 overflow-hidden group-hover:w-auto group-hover:opacity-100 transition-all mr-0.5 text-orange-600">🗑</span>
            <span className="text-orange-400 font-bold">#{e.row}</span>
            <span className="w-px h-2.5 bg-orange-100 mx-0.5"></span>
            <span className="font-semibold">{e.value.toFixed(2)}</span>
            <span className="text-[9px] text-orange-400 ml-0.5">z={e.z > 0 ? "+" : ""}{e.z.toFixed(2)}</span>

            {/* Tooltip on hover */}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-gray-800 text-white text-[9px] px-2 py-1 rounded whitespace-nowrap z-10">
              Row {e.row} | Resid: {e.abs_residual.toFixed(3)}
            </div>
          </div>
        ))}

        {/* Context Menu */}
        {contextMenu && (
          <div
            ref={menuRef}
            className="fixed bg-white border border-gray-200 rounded-lg shadow-lg z-50 py-1 min-w-max"
            style={{ top: `${contextMenu.y}px`, left: `${contextMenu.x}px` }}
          >
            <button
              onClick={() => handleDeleteRow(contextMenu.row)}
              className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 flex items-center gap-2 transition-colors"
            >
              <span>🗑</span> Delete Row {contextMenu.row}
            </button>
            <div className="border-t border-gray-100 my-1"></div>
            <button
              onClick={() => setContextMenu(null)}
              className="w-full text-left px-4 py-2 text-sm text-gray-500 hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// BASE_LAYOUT kept as fallback — most charts now use usePlotLayout() instead
const BASE_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#f9fafb",
  font: { color: "#374151", size: 11 },
  margin: { t: 24, r: 16, b: 48, l: 56 },
  xaxis: { gridcolor: "#e5e7eb", zerolinecolor: "#d1d5db" },
  yaxis: { gridcolor: "#e5e7eb", zerolinecolor: "#d1d5db" },
};

// ── Main chart for numeric columns ──────────────────────────────────────────

const CHART_TABS = [
  { id: "histogram", label: "Histogram" },
  { id: "boxplot",   label: "Box Plot" },
  { id: "violin",    label: "Violin" },
  { id: "qq",        label: "Q-Q Plot" },
] as const;
function NumericView({ summary, loadSummary, selected }: { summary: any; loadSummary: (col: string) => void; selected: string }) {
  const chartTab = useStore((s) => s.descriptiveTab);
  const setChartTab = useStore((s) => s.setDescriptiveTab);
  const showGrid = useStore((s) => s.showGrid);
  const pal = usePalette();
  const plotRef = useRef<any>(null);
  const P = pal[0]; // primary color

  const histData = [{
    type: "bar" as const,
    x: summary.histogram.map((b: any) => (b.bin_start + b.bin_end) / 2),
    y: summary.histogram.map((b: any) => b.count),
    width: summary.histogram.map((b: any) => b.bin_end - b.bin_start),
    marker: { color: P, opacity: 0.85 },
    name: "Count",
    hovertemplate: "Range: %{customdata[0]}–%{customdata[1]}<br>Count: %{y}<extra></extra>",
    customdata: summary.histogram.map((b: any) => [b.bin_start.toFixed(2), b.bin_end.toFixed(2)]),
  }];

  const outliers: { row: number; value: number }[] = summary.outliers ?? [];
  const rawVals: number[] = summary.raw_values ?? [];

  const summaryHover =
    `<b>Distribution Summary</b><br>` +
    `Median: ${summary.median?.toFixed(2)}<br>` +
    `Q1: ${summary.q1?.toFixed(2)}  Q3: ${summary.q3?.toFixed(2)}<br>` +
    `Whisker: ${(summary.whisker_low ?? 0).toFixed(2)} – ${(summary.whisker_high ?? 0).toFixed(2)}<br>` +
    `Min: ${summary.min?.toFixed(2)}  Max: ${summary.max?.toFixed(2)}<br>` +
    `Mean ± SD: ${summary.mean?.toFixed(2)} ± ${summary.std?.toFixed(2)}<extra></extra>`;

  // ── Box trace ─────────────────────────────────────────────────────────────
  // Give the box an EXPLICIT x category so Plotly uses a category axis.
  // Then scatter traces with the same x value co-locate perfectly.
  // hoverinfo:"none" kills the ugly per-stat labels Plotly shows by default.
  const boxTrace: any = {
    type: "box" as const,
    x: rawVals.map(() => "Distribution"),   // explicit category → category axis
    y: rawVals,
    name: "Distribution",
    boxmean: true,
    boxpoints: false as any,                // we draw outliers ourselves
    marker: { color: P, size: 5 },
    line: { color: P },
    fillcolor: "rgba(99,102,241,0.15)",
    hoverinfo: "none" as const,             // suppress "(Distribution, max: 85)" labels
  };

  // ── Invisible summary hover scatter ──────────────────────────────────────
  // Large transparent marker at median; triggers hover anywhere over the box.
  const summaryScatter: any = {
    type: "scatter" as const,
    mode: "markers" as const,
    x: ["Distribution"],
    y: [summary.median],
    marker: { opacity: 0.001, size: 80, color: "rgba(0,0,0,0)" },
    hovertemplate: summaryHover,
    showlegend: false,
  };

  // ── Outlier scatter ────────────────────────────────────────────────────────
  // Same x category → overlaid perfectly on the box.
  const outlierTrace: any[] = outliers.length > 0 ? [{
    type: "scatter" as const,
    mode: "markers" as const,
    x: outliers.map(() => "Distribution"),
    y: outliers.map((o) => o.value),
    customdata: outliers.map((o) => [o.row, o.value.toFixed(4)]),
    hovertemplate: "<b>Outlier</b><br>Row: %{customdata[0]}<br>Value: %{customdata[1]}<extra></extra>",
    marker: { color: "#ef4444", size: 8, symbol: "circle-open", line: { width: 2, color: "#ef4444" } },
    name: "Outlier",
    showlegend: false,
  }] : [];

  const boxData: any[] = [boxTrace, summaryScatter, ...outlierTrace];


  const normalityDeviants: { row: number; value: number; z: number; qq_x: number; abs_residual: number }[] =
    summary.normality_deviants ?? [];

  const qqData = [
    {
      type: "scatter" as const, mode: "markers" as const,
      x: summary.qq.map((p: any) => p.x),
      y: summary.qq.map((p: any) => p.y),
      marker: { color: P, size: 4 },
      name: "Observed",
      hovertemplate: "Theoretical: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>",
    },
    (() => {
      const xs = summary.qq.map((p: any) => p.x);
      const ys = summary.qq.map((p: any) => p.y);
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      return {
        type: "scatter" as const, mode: "lines" as const,
        x: [xMin, xMax], y: [yMin, yMax],
        line: { color: "#9ca3af", width: 1, dash: "dash" as const },
        name: "Reference",
        hoverinfo: "skip" as const,
      };
    })(),
    // Normality deviants overlay (The ones that trigger the "Non-normal" warning)
    ...(normalityDeviants.length > 0 ? [{
      type: "scatter" as const,
      mode: "markers" as const,
      x: normalityDeviants.map((e) => e.qq_x),
      y: normalityDeviants.map((e) => e.value),
      customdata: normalityDeviants.map((e) => [e.row, e.value.toFixed(4), e.z.toFixed(3)]),
      hovertemplate:
        "<b>Disrupts normality</b><br>" +
        "Row: %{customdata[0]}<br>" +
        "Value: %{customdata[1]}<br>" +
        "Z-score: %{customdata[2]}<extra></extra>",
      marker: { 
        color: "#f97316", 
        size: 8, 
        symbol: "diamond", 
        line: { width: 1, color: "#ea580c" },
        opacity: 0.8
      },
      name: "Deviant",
      showlegend: false,
    }] : []),
  ];

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Chart type tabs */}
      <div className="flex gap-1 bg-gray-100 rounded-lg p-0.5 self-start">
        {CHART_TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setChartTab(id)}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors
              ${chartTab === id
                ? "bg-white text-indigo-600 shadow-sm"
                : "text-gray-500 hover:text-gray-700"}`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Histogram */}
      {chartTab === "histogram" && (
        <div className="relative">
        <PlotExporter plotRef={plotRef} title="Histogram" />
        <Plot ref={plotRef}
          data={histData}
          layout={{ ...BASE_LAYOUT, autosize: true, bargap: 0.02,
            xaxis: { ...BASE_LAYOUT.xaxis, showgrid: showGrid, title: { text: "Value" } },
            yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: "Count" } },
          }}
          style={{ width: "100%", height: 380 }}
          useResizeHandler config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        />
        </div>
      )}

      {/* Box Plot */}
      {chartTab === "boxplot" && (
        <div className="relative">
        <PlotExporter plotRef={plotRef} title="BoxPlot" />
        <Plot ref={plotRef}
          data={boxData}
          layout={{
            ...BASE_LAYOUT,
            autosize: true,
            yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: "Value" } },
            xaxis: { ...BASE_LAYOUT.xaxis, showticklabels: false, zeroline: false, showgrid: false },
            showlegend: false,
            annotations: [
              {
                x: 0.5, y: 1.0,
                xref: "paper" as const, yref: "paper" as const,
                text: `IQR = ${summary.iqr?.toFixed(2)}  ·  Skew = ${summary.skewness?.toFixed(3)}` +
                      (outliers.length > 0 ? `  ·  <b style="color:#ef4444">${outliers.length} outlier</b>` : ""),
                showarrow: false,
                font: { color: "#6b7280", size: 11 },
                xanchor: "center" as const,
                yanchor: "bottom" as const,
              },
            ],
          }}
          style={{ width: "100%", height: 380 }}
          useResizeHandler config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        />
        {outliers.length > 0 && (
          <div className="mt-2 px-3 py-2 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-xs font-semibold text-red-600 mb-1">
              ⚠️ {outliers.length} outlier (1.5 × IQR rule)
            </p>
            <div className="flex flex-wrap gap-1">
              {outliers.slice(0, 50).map((o) => (
                <span
                  key={o.row}
                  className="inline-block text-[10px] font-mono bg-red-100 text-red-700 border border-red-200 rounded px-1.5 py-0.5 cursor-pointer hover:bg-red-200"
                  title={`Click to delete Row ${o.row}: ${o.value}`}
                  onClick={async () => {
                     try {
                        await useStore.getState().deleteRow(o.row);
                        import("../api").then((api) => api.refreshSession(useStore.getState().session!.session_id));
                     } catch (err) {}
                  }}
                >
                  <span className="opacity-50 hover:opacity-100 mr-1">🗑</span>
                  #{o.row} · {o.value.toFixed(2)}
                </span>
              ))}
              {outliers.length > 50 && (
                <span className="text-[10px] text-red-400 italic">…and {outliers.length - 50} more</span>
              )}
            </div>
          </div>
        )}
        </div>
      )}

      {/* Violin Plot */}
      {chartTab === "violin" && (
        <div className="relative">
        <PlotExporter plotRef={plotRef} title="Violin" />
        <Plot ref={plotRef}
          data={[{
            type: "violin" as any,
            y: summary.raw_values ?? [],
            name: "Distribution",
            box: { visible: true },
            meanline: { visible: true },
            line: { color: P },
            fillcolor: P + "25",
            points: (summary.raw_values?.length ?? 0) < 200 ? "all" : false,
            jitter: 0.3,
            pointpos: -1.5,
            marker: { color: P, size: 3, opacity: 0.5 },
            hovertemplate:
              `Median: ${summary.median?.toFixed(2)}<br>` +
              `Mean: ${summary.mean?.toFixed(2)}<br>` +
              `SD: ${summary.std?.toFixed(2)}<br>` +
              `IQR: ${summary.q1?.toFixed(2)}–${summary.q3?.toFixed(2)}<extra></extra>`,
          }]}
          layout={{
            ...BASE_LAYOUT,
            autosize: true,
            yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: "Value" } },
            xaxis: { ...BASE_LAYOUT.xaxis, showticklabels: false, zeroline: false, showgrid: false },
            showlegend: false,
            annotations: [
              {
                x: 0.5, y: 1.0,
                xref: "paper" as const, yref: "paper" as const,
                text: `IQR = ${summary.iqr?.toFixed(2)}  ·  Skew = ${summary.skewness?.toFixed(3)}`,
                showarrow: false,
                font: { color: "#6b7280", size: 11 },
                xanchor: "center" as const,
                yanchor: "bottom" as const,
              },
            ],
          }}
          style={{ width: "100%", height: 380 }}
          useResizeHandler config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        />
        </div>
      )}

      {/* Q-Q Plot */}
      {chartTab === "qq" && (
        <div className="relative">
        <PlotExporter plotRef={plotRef} title="QQ_Plot" />
        <Plot ref={plotRef}
          data={qqData}
          layout={{ ...BASE_LAYOUT, autosize: true,
            title: { text: "Q-Q Plot (Normality)", font: { color: "#374151", size: 12 } },
            xaxis: { ...BASE_LAYOUT.xaxis, showgrid: showGrid, title: { text: "Theoretical quantiles" } },
            yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: "Sample quantiles" } },
          }}
          style={{ width: "100%", height: 380 }}
          useResizeHandler config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        />
        {/* List of most deviating values (The ones responsible for Non-normal label) */}
        {!summary.normal && normalityDeviants.length > 0 && (
          <NormalityDeviants deviants={normalityDeviants} onDelete={() => loadSummary(selected)} />
        )}
        </div>
      )}
    </div>
  );
}

// ── Main chart for categorical columns ──────────────────────────────────────

function CategoricalView({ summary }: { summary: any }) {
  const showGrid = useStore((s) => s.showGrid);
  const cats = summary.categories.slice(0, 20);
  const colors = ["#7c3aed", "#f59e0b", "#10b981", "#ef4444", "#06b6d4", "#ec4899"];

  const donutData = [{
    type: "pie" as const,
    values: cats.map((c: any) => c.count),
    labels: cats.map((c: any) => c.value),
    hole: 0.5,
    marker: { colors: colors },
    textinfo: "percent" as const,
    hovertemplate: "%{label}: %{value} (%{percent})<extra></extra>",
  }];

  const barData = [{
    type: "bar" as const,
    x: cats.map((c: any) => c.count),
    y: cats.map((c: any) => c.value),
    orientation: "h" as const,
    marker: { color: PALETTES[useStore.getState().plotTheme.palette]?.[0] ?? "#6366f1", opacity: 0.85 },
    text: cats.map((c: any) => `${c.count}`),
    textposition: "outside" as const,
    hovertemplate: "%{y}: %{x}<extra></extra>",
  }];

  return (
    <div className="flex flex-col gap-3 h-full">
      <Plot
        data={donutData}
        layout={{
          paper_bgcolor: "transparent", plot_bgcolor: "transparent",
          font: { color: "#374151", size: 11 }, margin: { t: 10, r: 160, b: 10, l: 10 },
          autosize: true,
          legend: { font: { color: "#374151" }, bgcolor: "transparent" },
        }}
        style={{ width: "100%", height: 220 }}
        useResizeHandler config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      />
      <Plot
        data={barData}
        layout={{ ...BASE_LAYOUT, autosize: true,
          xaxis: { ...BASE_LAYOUT.xaxis, showgrid: showGrid, title: { text: "Count" } },
          yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, automargin: true },
          margin: { ...BASE_LAYOUT.margin, l: 90 },
        }}
        style={{ width: "100%", height: Math.max(160, cats.length * 28 + 60) }}
        useResizeHandler config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      />
    </div>
  );
}

// ── Scatter view ─────────────────────────────────────────────────────────────

// Use global palette — falls back to default if not set
const _getPalette = () => PALETTES[useStore.getState().plotTheme.palette] ?? PALETTES.indigo;
const SYMBOLS  = ["circle","square","diamond","triangle-up","cross","star","hexagram","pentagon"] as const;

function ScatterView({
  sessionId,
  numCols,
  catCols,
  defaultX,
}: {
  sessionId: string;
  numCols: string[];
  catCols: string[];
  defaultX: string;
}) {
  const showGrid = useStore((s) => s.showGrid);
  const [xCol,    setXCol]    = useState(defaultX || numCols[0] || "");
  const [yCol,    setYCol]    = useState(numCols.find((c) => c !== defaultX) ?? "");
  const [color,   setColor]   = useState("");
  const [shape,   setShape]   = useState("");
  const [data,    setData]    = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const prevKey = useRef("");

  useEffect(() => {
    if (!xCol || !yCol) { setData(null); return; }
    const key = `${xCol}|${yCol}|${color}|${shape}`;
    if (key === prevKey.current) return;
    prevKey.current = key;
    setLoading(true); setError(null);
    api.post("/api/charts/scatter", {
      session_id: sessionId, x: xCol, y: yCol,
      color: color || undefined,
      shape: shape || undefined,
    })
      .then((r) => setData(r.data))
      .catch((e) => setError(e.response?.data?.detail ?? e.message))
      .finally(() => setLoading(false));
  }, [xCol, yCol, color, shape, sessionId]);

  const fmt = (v: number, d = 3) =>
    typeof v === "number" ? (Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(2) : v.toFixed(d)) : "—";

  const traces: any[] = [];
  if (data) {
    const pts = data.points as Record<string, any>[];
    const shapeUniq: string[] = shape
      ? Array.from(new Set(pts.map((p) => String(p[shape] ?? "null"))))
      : [];
    const symbolOf = (v: string) => SYMBOLS[shapeUniq.indexOf(v) % SYMBOLS.length] ?? "circle";

    if (color && data.color) {
      const groups: Record<string, { x: any[]; y: any[]; shapeLabels: string[] }> = {};
      pts.forEach((p) => {
        const g = String(p[color] ?? "null");
        if (!groups[g]) groups[g] = { x: [], y: [], shapeLabels: [] };
        groups[g].x.push(p[xCol]);
        groups[g].y.push(p[yCol]);
        if (shape) groups[g].shapeLabels.push(String(p[shape] ?? "null"));
      });
      Object.entries(groups).forEach(([g, vals], i) => {
        traces.push({
          type: "scatter", mode: "markers",
          x: vals.x, y: vals.y,
          name: g,
          marker: {
            color: _getPalette()[i % _getPalette().length],
            size: 7, opacity: 0.78,
            symbol: shape ? vals.shapeLabels.map(symbolOf) : "circle",
          },
          text: shape ? vals.shapeLabels : undefined,
          hovertemplate:
            `<b>${color}</b>: ${g}` +
            (shape ? `<br><b>${shape}</b>: %{text}` : "") +
            `<br>${xCol}: %{x}<br>${yCol}: %{y}<extra></extra>`,
        });
      });
    } else if (shape) {
      const groups: Record<string, { x: any[]; y: any[] }> = {};
      pts.forEach((p) => {
        const g = String(p[shape] ?? "null");
        if (!groups[g]) groups[g] = { x: [], y: [] };
        groups[g].x.push(p[xCol]);
        groups[g].y.push(p[yCol]);
      });
      Object.entries(groups).forEach(([g, vals], i) => {
        traces.push({
          type: "scatter", mode: "markers",
          x: vals.x, y: vals.y,
          name: g,
          marker: { color: _getPalette()[0], size: 7, opacity: 0.78, symbol: SYMBOLS[i % SYMBOLS.length] },
          hovertemplate: `<b>${shape}</b>: ${g}<br>${xCol}: %{x}<br>${yCol}: %{y}<extra></extra>`,
        });
      });
    } else {
      traces.push({
        type: "scatter", mode: "markers",
        x: pts.map((p) => p[xCol]),
        y: pts.map((p) => p[yCol]),
        name: "Data",
        marker: { color: _getPalette()[0], size: 6, opacity: 0.7, symbol: "circle" },
        hovertemplate: `${xCol}: %{x}<br>${yCol}: %{y}<extra></extra>`,
      });
    }

    const reg = data.regression;
    if (reg.line_x?.length > 0) {
      traces.push({
        type: "scatter", mode: "lines",
        x: reg.line_x, y: reg.line_y,
        name: "Fit",
        line: { color: "#ef4444", width: 2, dash: "dash" },
        hoverinfo: "skip",
        showlegend: false,
      });
    }
  }

  const hasGrouping = !!(color || shape);

  return (
    <div className="flex flex-col gap-4 h-full p-4 overflow-y-auto">
      <div className="flex gap-3 flex-wrap flex-shrink-0">
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">X axis</label>
          <select className="select text-xs min-w-[150px]" value={xCol}
            onChange={(e) => setXCol(e.target.value)}>
            {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">Y axis</label>
          <select className="select text-xs min-w-[150px]" value={yCol}
            onChange={(e) => setYCol(e.target.value)}>
            <option value="">— pick Y variable —</option>
            {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">🎨 Color by</label>
          <select className="select text-xs min-w-[150px]" value={color}
            onChange={(e) => setColor(e.target.value)}>
            <option value="">— none —</option>
            {catCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">◆ Shape by</label>
          <select className="select text-xs min-w-[150px]" value={shape}
            onChange={(e) => setShape(e.target.value)}>
            <option value="">— none —</option>
            {catCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {!yCol && (
        <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
          Select a continuous variable for the Y axis
        </div>
      )}
      {loading && (
        <div className="flex-1 flex items-center justify-center text-gray-400 animate-pulse">
          Computing…
        </div>
      )}
      {error && (
        <div className="text-red-500 text-xs bg-red-50 rounded-lg p-3">{error}</div>
      )}

      {data && !loading && (
        <>
          <div className="flex gap-3 flex-wrap flex-shrink-0">
            {[
              { label: "n",         value: String(data.points.length) },
              { label: "r",         value: fmt(data.regression.r) },
              { label: "r²",        value: fmt(data.regression.r2) },
              { label: "p",         value: data.regression.p == null ? "—" : data.regression.p < 0.001 ? "<0.001" : fmt(data.regression.p) },
              { label: "slope",     value: fmt(data.regression.slope) },
              { label: "intercept", value: fmt(data.regression.intercept) },
            ].map(({ label, value }) => (
              <div key={label} className="flex flex-col items-center bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 min-w-[60px]">
                <span className="text-[10px] text-gray-400 mb-0.5">{label}</span>
                <span className="text-xs font-mono font-semibold text-gray-800">{value}</span>
              </div>
            ))}
            {data.regression.r != null ? (
              <div className={`flex items-center px-3 py-2 rounded-lg border text-xs font-semibold
                ${Math.abs(data.regression.r) > 0.7
                  ? "bg-indigo-50 border-indigo-200 text-indigo-700"
                  : Math.abs(data.regression.r) > 0.4
                    ? "bg-amber-50 border-amber-200 text-amber-700"
                    : "bg-gray-50 border-gray-200 text-gray-500"}`}>
                {Math.abs(data.regression.r) > 0.7 ? "Strong" :
                 Math.abs(data.regression.r) > 0.4 ? "Moderate" : "Weak"}
                {" "}{data.regression.r >= 0 ? "positive" : "negative"} correlation
              </div>
            ) : (
              <div className="flex items-center px-3 py-2 rounded-lg border border-gray-200 text-xs text-gray-400">
                {data.regression.note ?? "Regression unavailable"}
              </div>
            )}
          </div>

          <div className="flex-1" style={{ minHeight: 320 }}>
            <Plot
              data={traces}
              layout={{
                ...BASE_LAYOUT,
                autosize: true,
                xaxis: { ...BASE_LAYOUT.xaxis, showgrid: showGrid, title: { text: xCol } },
                yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: yCol } },
                legend: { font: { color: "#374151", size: 11 }, bgcolor: "rgba(249,250,251,0.9)", bordercolor: "#e5e7eb", borderwidth: 1 },
                showlegend: hasGrouping,
                annotations: data.regression.r != null ? [{
                  x: 0.03, y: 0.97,
                  xref: "paper" as const, yref: "paper" as const,
                  text: `r = ${data.regression.r.toFixed(3)}   p ${data.regression.p < 0.001 ? "< 0.001" : "= " + data.regression.p.toFixed(3)}`,
                  showarrow: false,
                  font: { color: "#374151", size: 11 },
                  bgcolor: "rgba(249,250,251,0.9)",
                  bordercolor: "#e5e7eb",
                  borderwidth: 1,
                  borderpad: 5,
                  align: "left" as const,
                  xanchor: "left" as const,
                  yanchor: "top" as const,
                }] : [],
              }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
            />
          </div>
        </>
      )}
    </div>
  );
}

// ── Stats badge ───────────────────────────────────────────────────────────────

function StatBadge({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col items-center bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 min-w-[72px]">
      <span className="text-xs text-gray-400 mb-0.5">{label}</span>
      <span className="text-sm font-mono font-semibold text-gray-800">{value}</span>
    </div>
  );
}

// ── Main panel ───────────────────────────────────────────────────────────────

const KIND_CYCLE: Record<string, "numeric" | "categorical" | "text" | "date"> = {
  numeric: "categorical",
  categorical: "text",
  text: "date",
  date: "numeric",
};

const KIND_STYLE: Record<string, { label: string; cls: string }> = {
  numeric:     { label: "N", cls: "bg-blue-100 text-blue-700" },
  categorical: { label: "C", cls: "bg-purple-100 text-purple-700" },
  text:        { label: "T", cls: "bg-gray-100 text-gray-500" },
  date:        { label: "D", cls: "bg-purple-100 text-purple-700" },
};

export default function DescriptivePanel() {
  const session = useStore((s) => s.session);
  const updateColumnKind = useStore((s) => s.updateColumnKind);
  const reorderColumns   = useStore((s) => s.reorderColumns);
  const [dragIdx,  setDragIdx]  = useState<number | null>(null);
  const [dropIdx,  setDropIdx]  = useState<number | null>(null);
  const [colMeta, setColMeta] = useState<any[]>([]);
  const [sparklines, setSparklines] = useState<Record<string, SparkData>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [summary, setSummary] = useState<any | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [view, setView] = useState<"distribution" | "scatter">("distribution");
  const chartTab = useStore((s) => s.descriptiveTab);
  const setChartTab = useStore((s) => s.setDescriptiveTab);

  // For Scatter companions: which distribution plot to show next to scatter
  const [scatterCompanionType, setScatterCompanionType] = useState<"histogram" | "boxplot" | "violin" | "qq">("histogram");
  const [scatterCompanionData, setScatterCompanionData] = useState<any>(null);
  const [scatterCompanionLoading, setScatterCompanionLoading] = useState(false);

  // Resizable plot area from the right (for Summary tab graphs)
  const { w: plotWidth, onDragStart: onPlotResizeStart, onReset: resetPlotWidth } = useResizableRightCol(
    "summary-plot",
    920,   // initial width
    520,   // min
    1400   // max
  );

  // For Scatter view: width of the left panel containing the 4 distribution plots (user wants them on the LEFT of the scatter)
  const [scatterLeftWidth, setScatterLeftWidth] = useState(() => {
    if (typeof window !== "undefined") {
      const v = parseInt(localStorage.getItem("uStat.scatterLeftW") || "380", 10);
      return Math.max(280, Math.min(700, v || 380));
    }
    return 380;
  });

  const scatterLeftResizeRef = useRef<{ startX: number; startW: number } | null>(null);

  const onScatterLeftResizeMove = useCallback((e: PointerEvent) => {
    const d = scatterLeftResizeRef.current;
    if (!d) return;
    const dx = e.clientX - d.startX;
    const next = Math.max(280, Math.min(700, d.startW + dx));
    setScatterLeftWidth(next);
  }, []);

  const onScatterLeftResizeUp = useCallback(() => {
    const d = scatterLeftResizeRef.current;
    if (!d) return;
    scatterLeftResizeRef.current = null;
    document.removeEventListener("pointermove", onScatterLeftResizeMove);
    document.removeEventListener("pointerup", onScatterLeftResizeUp);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    try {
      localStorage.setItem("uStat.scatterLeftW", String(scatterLeftWidth));
    } catch {}
  }, [onScatterLeftResizeMove, scatterLeftWidth]);

  const startScatterLeftResize = (e: React.PointerEvent) => {
    e.preventDefault();
    scatterLeftResizeRef.current = { startX: e.clientX, startW: scatterLeftWidth };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("pointermove", onScatterLeftResizeMove);
    document.addEventListener("pointerup", onScatterLeftResizeUp);
  };

  // 2D resizable container for the main Distribution plot (user wants red drag lines on right + bottom)
  const [distPlotW, setDistPlotW] = useState(() => {
    if (typeof window !== "undefined") {
      const v = parseInt(localStorage.getItem("uStat.distPlotW") || "920", 10);
      return Math.max(520, Math.min(1400, v || 920));
    }
    return 920;
  });
  const [distPlotH, setDistPlotH] = useState(() => {
    if (typeof window !== "undefined") {
      const v = parseInt(localStorage.getItem("uStat.distPlotH") || "520", 10);
      return Math.max(320, Math.min(900, v || 520));
    }
    return 520;
  });

  const distResizeRef = useRef<{ startX: number; startW: number; startY: number; startH: number; mode: "right" | "bottom" } | null>(null);

  const onDistResizeMove = useCallback((e: PointerEvent) => {
    const d = distResizeRef.current;
    if (!d) return;
    if (d.mode === "right") {
      const dx = e.clientX - d.startX;
      const nextW = Math.max(520, Math.min(1400, d.startW + dx));
      setDistPlotW(nextW);
    } else {
      const dy = e.clientY - d.startY;
      const nextH = Math.max(320, Math.min(900, d.startH + dy));
      setDistPlotH(nextH);
    }
  }, []);

  const onDistResizeUp = useCallback(() => {
    const d = distResizeRef.current;
    if (!d) return;
    distResizeRef.current = null;
    document.removeEventListener("pointermove", onDistResizeMove);
    document.removeEventListener("pointerup", onDistResizeUp);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    // persist
    try {
      localStorage.setItem("uStat.distPlotW", String(distPlotW));
      localStorage.setItem("uStat.distPlotH", String(distPlotH));
    } catch {}
  }, [onDistResizeMove, distPlotW, distPlotH]);

  const startDistResize = (mode: "right" | "bottom") => (e: React.PointerEvent) => {
    e.preventDefault();
    distResizeRef.current = {
      startX: e.clientX,
      startW: distPlotW,
      startY: e.clientY,
      startH: distPlotH,
      mode,
    };
    document.body.style.cursor = mode === "right" ? "col-resize" : "row-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("pointermove", onDistResizeMove);
    document.addEventListener("pointerup", onDistResizeUp);
  };

  useEffect(() => {
    if (!session) return;
    // Fetch real sparkline histograms for all columns
    api.get(`/api/stats/${session.session_id}/sparklines`).then((r) => {
      setSparklines(r.data as Record<string, SparkData>);
    });
    api.get(`/api/stats/${session.session_id}/descriptive`).then((r) => {
      const numStats = r.data as Record<string, any>;
      const metas = session.columns.map((c) => {
        if (c.kind === "numeric" && numStats[c.name]) {
          const s = numStats[c.name];
          return { name: c.name, kind: "numeric", hist: null, shapiro_p: s.normality_p };
        }
        return { name: c.name, kind: c.kind, top2: null };
      });
      setColMeta(metas);
    });
  }, [session?.session_id]);

  const loadSummary = useCallback((colName: string, kindOverride?: string) => {
    if (!session) return;
    const kind = kindOverride ?? session.columns.find((c) => c.name === colName)?.kind ?? undefined;
    setSelected(colName);
    setSummary(null);
    setSummaryLoading(true);
    api.get(`/api/stats/${session.session_id}/column_summary`, { params: { column: colName, kind } })
      .then((r) => {
        const rawSummary = r.data;
        if (rawSummary && rawSummary.type === "categorical" && rawSummary.categories) {
          const colMeta = session.columns.find((c) => c.name === colName);
          const vLabels = colMeta?.value_labels ?? {};
          
          rawSummary.categories = rawSummary.categories.map((c: any) => ({
            ...c,
            value: vLabels[String(c.value)] ?? c.value,
          }));
        }
        setSummary(rawSummary);
      })
      .finally(() => setSummaryLoading(false));
  }, [session?.session_id]);

  useEffect(() => {
    if (session && !selected && session.columns.length > 0) {
      loadSummary(session.columns[0].name);
    }
  }, [session?.session_id]);

  if (!session) return null;

  const numCols = session.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
  const catCols = session.columns.filter((c) => c.kind !== "numeric").map((c) => c.name);

  const filtered = session.columns.filter((c) =>
    c.name.toLowerCase().includes(search.toLowerCase())
  );

  const fmt = (v: number, d = 2) => {
    if (typeof v !== "number") return "—";
    if (Math.abs(v) < 0.0001 && v !== 0) return v.toExponential(2);
    return v.toFixed(d);
  };

  return (
    <div className="flex gap-0 h-full" style={{ minHeight: 0 }}>

      {/* ── Left: column list ── */}
      <div className="w-56 flex-shrink-0 flex flex-col border-r border-gray-200 bg-white overflow-hidden">
        <div className="p-2 border-b border-gray-200">
          <input
            className="select w-full text-xs"
            placeholder="Search columns…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="overflow-y-auto flex-1">
          {filtered.map((c) => {
            const meta = colMeta.find((m) => m.name === c.name);
            const isActive = selected === c.name;
            const realIdx = session!.columns.findIndex((sc) => sc.name === c.name);
            const isDragOver = dropIdx === realIdx && dragIdx !== realIdx;
            return (
              <div
                key={c.name}
                draggable
                onDragStart={(e) => { setDragIdx(realIdx); e.dataTransfer.effectAllowed = "move"; }}
                onDragOver={(e) => { e.preventDefault(); setDropIdx(realIdx); }}
                onDragLeave={() => { if (dropIdx === realIdx) setDropIdx(null); }}
                onDrop={(e) => { e.preventDefault(); if (dragIdx !== null && dragIdx !== realIdx) reorderColumns(dragIdx, realIdx); setDragIdx(null); setDropIdx(null); }}
                onDragEnd={() => { setDragIdx(null); setDropIdx(null); }}
                onClick={() => { setView("distribution"); loadSummary(c.name); }}
                className={`flex items-center justify-between px-3 py-2 cursor-grab active:cursor-grabbing border-b border-gray-100 transition-colors select-none
                  ${dragIdx === realIdx ? "opacity-40" : ""}
                  ${isDragOver ? "border-t-2 border-t-indigo-500" : ""}
                  ${isActive ? "bg-indigo-50 border-l-2 border-l-indigo-500" : "hover:bg-gray-50"}`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-gray-300 text-[8px] flex-shrink-0">⠿</span>
                  <span
                    title={`Type: ${c.kind} — click to change`}
                    onClick={(e) => {
                      e.stopPropagation();
                      const next = KIND_CYCLE[c.kind] ?? "numeric";
                      updateColumnKind(c.name, next);
                      if (selected === c.name) loadSummary(c.name, next);
                    }}
                    className={`text-[9px] font-bold px-1 rounded flex-shrink-0 cursor-pointer hover:opacity-70
                      ${KIND_STYLE[c.kind]?.cls ?? "bg-gray-100 text-gray-500"}`}>
                    {KIND_STYLE[c.kind]?.label ?? "?"}
                  </span>
                  <span className="text-xs text-gray-700 truncate">{c.name}</span>
                </div>
                {sparklines[c.name] ? (
                  <div className="flex-shrink-0 ml-1">
                    <Sparkline spark={sparklines[c.name]} />
                  </div>
                ) : meta && (
                  <div className="w-10 h-3 bg-gray-100 rounded flex-shrink-0 ml-1 animate-pulse" />
                )}
              </div>
            );
          })}
        </div>
        <div className="p-2 border-t border-gray-200 text-xs text-gray-400 text-center">
          {session.columns.length} columns · {session.rows} rows
        </div>
      </div>

      {/* ── Right: view area ── */}
      <div className="flex-1 flex flex-col overflow-hidden bg-white">

        {/* ── Sub-tabs under Descriptive: Histogram | Box Plot | Violin | Q-Q Plot | Scatter Plot ── */}
        <div className="flex items-center gap-1 px-4 py-1.5 border-b border-gray-200 flex-shrink-0 bg-gray-50 overflow-x-auto">
          {[
            { id: "histogram", label: "Histogram" },
            { id: "boxplot",   label: "Box Plot" },
            { id: "violin",    label: "Violin" },
            { id: "qq",        label: "Q-Q Plot" },
            { id: "scatter",   label: "Scatter Plot" },
          ].map(({ id, label }) => {
            const isActive = (id === "scatter" ? view === "scatter" : view === "distribution" && chartTab === id);
            return (
              <button
                key={id}
                onClick={() => {
                  if (id === "scatter") {
                    setView("scatter");
                  } else {
                    setView("distribution");
                    setChartTab(id as any);
                  }
                }}
                className={`px-3 py-1 rounded-md text-xs font-medium whitespace-nowrap transition-colors
                  ${isActive
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-gray-600 hover:text-gray-800 hover:bg-gray-200"}`}
              >
                {label}
              </button>
            );
          })}
        </div>

        {/* ── Scatter view: 4 distribution plots (Hist/Box/Violin/QQ) on the LEFT of the scatter, red draggable divider, no extra "Distribution" title ── */}
        {view === "scatter" && (
          <div className="flex h-full" style={{ minWidth: 0 }}>
            {/* LEFT: The 4 plots (Histogram, Box, Violin, Q-Q) on the left of the scatter plot */}
            <div
              className="flex-shrink-0 min-h-0 overflow-hidden bg-gray-50 p-2 border-r border-gray-200"
              style={{ width: `${scatterLeftWidth}px` }}
            >
              {summary && summary.type === "numeric" ? (
                <div className="grid grid-cols-2 grid-rows-2 gap-2 h-full">
                  {/* Histogram */}
                  <div className="flex flex-col border bg-white rounded overflow-hidden">
                    <div className="text-[9px] font-semibold px-1.5 py-0.5 border-b bg-gray-100 flex-shrink-0">Histogram</div>
                    <div className="flex-1 p-1">
                      <Plot
                        data={[{
                          x: summary.histogram?.map((b: any) => (b.bin_start + b.bin_end) / 2) || [],
                          y: summary.histogram?.map((b: any) => b.count) || [],
                          type: "bar",
                          marker: { color: "#6366f1" },
                        }]}
                        layout={{ autosize: true, margin: { t: 4, r: 4, b: 20, l: 25 }, showlegend: false, xaxis: { showgrid: true }, yaxis: { showgrid: true } }}
                        style={{ width: "100%", height: "100%" }}
                        useResizeHandler
                      />
                    </div>
                  </div>

                  {/* Box Plot */}
                  <div className="flex flex-col border bg-white rounded overflow-hidden">
                    <div className="text-[9px] font-semibold px-1.5 py-0.5 border-b bg-gray-100 flex-shrink-0">Box Plot</div>
                    <div className="flex-1 p-1">
                      <Plot
                        data={[{
                          y: summary.raw_values || [],
                          type: "box",
                          name: selected,
                          boxpoints: "outliers",
                          marker: { color: "#6366f1" },
                        }]}
                        layout={{ autosize: true, margin: { t: 4, r: 4, b: 20, l: 25 }, showlegend: false }}
                        style={{ width: "100%", height: "100%" }}
                        useResizeHandler
                      />
                    </div>
                  </div>

                  {/* Violin */}
                  <div className="flex flex-col border bg-white rounded overflow-hidden">
                    <div className="text-[9px] font-semibold px-1.5 py-0.5 border-b bg-gray-100 flex-shrink-0">Violin</div>
                    <div className="flex-1 p-1">
                      <Plot
                        data={[{
                          y: summary.raw_values || [],
                          type: "violin",
                          name: selected,
                          box: { visible: true },
                          meanline: { visible: true },
                        }]}
                        layout={{ autosize: true, margin: { t: 4, r: 4, b: 20, l: 25 }, showlegend: false }}
                        style={{ width: "100%", height: "100%" }}
                        useResizeHandler
                      />
                    </div>
                  </div>

                  {/* Q-Q Plot */}
                  <div className="flex flex-col border bg-white rounded overflow-hidden">
                    <div className="text-[9px] font-semibold px-1.5 py-0.5 border-b bg-gray-100 flex-shrink-0">Q-Q Plot</div>
                    <div className="flex-1 p-1">
                      <Plot
                        data={[
                          {
                            x: summary.qq?.map((p: any) => p.x) || [],
                            y: summary.qq?.map((p: any) => p.y) || [],
                            type: "scatter",
                            mode: "markers",
                            marker: { color: "#6366f1", size: 3 },
                          },
                          {
                            x: summary.qq?.map((p: any) => p.x) || [],
                            y: summary.qq?.map((p: any) => p.x) || [],
                            type: "scatter",
                            mode: "lines",
                            line: { color: "#ef4444", dash: "dash" },
                          },
                        ]}
                        layout={{ autosize: true, margin: { t: 4, r: 4, b: 20, l: 25 }, showlegend: false }}
                        style={{ width: "100%", height: "100%" }}
                        useResizeHandler
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-gray-400 text-xs">
                  Select numeric column
                </div>
              )}
            </div>

            {/* Red vertical draggable line - between the left 4 plots and the scatter */}
            <div
              onPointerDown={startScatterLeftResize}
              onDoubleClick={() => setScatterLeftWidth(380)}
              className="w-[6px] cursor-col-resize bg-red-500/80 hover:bg-red-600 active:bg-red-700 transition-colors z-10 flex-shrink-0"
              title="Drag this red line to resize (left: 4 plots, right: scatter) • Double-click to reset"
            />

            {/* RIGHT: Main Scatter Plot */}
            <div className="flex-1 min-h-0 overflow-hidden">
              <ScatterView
                key={session.session_id}
                sessionId={session.session_id}
                numCols={numCols}
                catCols={catCols}
                defaultX={selected && numCols.includes(selected) ? selected : (numCols[0] ?? "")}
              />
            </div>
          </div>
        )}

        {/* ── Distribution view ── */}
        {view === "distribution" && (
          <>
            {summaryLoading && (
              <div className="flex-1 flex items-center justify-center text-gray-400 animate-pulse">
                Computing distribution…
              </div>
            )}
            {!summaryLoading && !summary && (
              <div className="flex-1 flex items-center justify-center text-gray-400">
                Select a column to view distribution
              </div>
            )}
            {!summaryLoading && summary && (
              <>
                {/* Header - compacted to single row as requested (tek sıra) */}
                <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 flex-shrink-0">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="font-semibold text-gray-900">Distribution of</span>
                    <span className="font-semibold text-indigo-600">{selected}</span>
                    <span className="text-gray-400">·</span>
                    <span className="text-gray-600">
                      {summary.type === "numeric" ? "Continuous" : "Categorical"} · n={summary.n}
                    </span>
                    {summary.missing > 0 && (
                      <span className="text-amber-600 text-xs">· {summary.missing} missing</span>
                    )}
                    {summary.type === "numeric" && summary.normality_p != null && (
                      <span className={`text-xs ${summary.normal ? "text-emerald-600" : "text-amber-600"}`}>
                        · {summary.normal ? "Normal" : "Non-normal"} (p={summary.normality_p.toFixed(3)})
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    <ResultExporter
                      title={`Summary_${selected}`}
                      headers={summary.type === "numeric"
                        ? ["Statistic", "Value"]
                        : ["Category", "Count", "Percent"]}
                      rows={summary.type === "numeric"
                        ? [
                            ["N", summary.n],
                            ["Missing", summary.missing],
                            ["Mean", summary.mean?.toFixed(4) ?? ""],
                            ["SD", summary.std?.toFixed(4) ?? ""],
                            ["Median", summary.median?.toFixed(4) ?? ""],
                            ["Q1", summary.q1?.toFixed(4) ?? ""],
                            ["Q3", summary.q3?.toFixed(4) ?? ""],
                            ["IQR", summary.iqr?.toFixed(4) ?? ""],
                            ["Min", summary.min?.toFixed(4) ?? ""],
                            ["Max", summary.max?.toFixed(4) ?? ""],
                            ["Skewness", summary.skewness?.toFixed(4) ?? ""],
                            ["Kurtosis", summary.kurtosis?.toFixed(4) ?? ""],
                            ["Normality test", summary.normality_test ?? ""],
                            ["Normality p", summary.normality_p?.toFixed(4) ?? summary.shapiro_p?.toFixed(4) ?? ""],
                          ]
                        : (summary.categories ?? []).map((c: any) => [
                            c.value, c.count,
                            c.pct != null ? `${c.pct.toFixed(1)}%` : "",
                          ])}
                    />
                  {summary.type === "numeric" && (
                    <div className={`px-3 py-1.5 rounded-lg text-xs font-semibold border
                      ${summary.normal
                        ? "bg-green-50 border-green-300 text-green-700"
                        : "bg-red-50 border-red-300 text-red-600"}`}>
                      {summary.normality_label}
                      <span className="font-normal text-gray-400 ml-1">
                        ({summary.normality_test ?? "Shapiro-Wilk"} p = {fmt(summary.normality_p ?? summary.shapiro_p, 3)})
                      </span>
                      <div className="text-[10px] font-normal text-gray-400 mt-0.5">
                        {summary.n < 50 ? "n < 50 → Shapiro-Wilk" : "n ≥ 50 → Kolmogorov-Smirnov"}
                      </div>
                    </div>
                  )}
                  {summary.type === "categorical" && (
                    <div className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-purple-300 bg-purple-50 text-purple-700">
                      {summary.n_categories} categories
                    </div>
                  )}
                  </div>
                </div>

                {/* Stats strip (numeric) */}
                {summary.type === "numeric" && (
                  <div className="flex gap-2 px-4 py-2 border-b border-gray-200 overflow-x-auto flex-shrink-0">
                    <StatBadge label="Mean" value={fmt(summary.mean)} />
                    <StatBadge label="SD" value={fmt(summary.std)} />
                    <StatBadge label="Median" value={fmt(summary.median)} />
                    <StatBadge label="Q1" value={fmt(summary.q1)} />
                    <StatBadge label="Q3" value={fmt(summary.q3)} />
                    <StatBadge label="IQR" value={fmt(summary.iqr)} />
                    <StatBadge label="Min" value={fmt(summary.min)} />
                    <StatBadge label="Max" value={fmt(summary.max)} />
                    <StatBadge label="Skew" value={fmt(summary.skewness)} />
                  </div>
                )}
                {/* Interpretation guidance */}
                {summary.type === "numeric" && (
                  <div className="px-4 py-1.5 border-b border-gray-100 bg-amber-50 flex-shrink-0">
                    <p className="text-[10px] text-amber-800 leading-relaxed">
                      {summary.normal
                        ? `Normal distribution (${summary.normality_test}, p=${summary.normality_p?.toFixed(3)}) \u2014 report Mean \u00B1 SD (${summary.mean?.toFixed(1)} \u00B1 ${summary.std?.toFixed(1)}).`
                        : `Non-normal (${summary.normality_test}, p=${summary.normality_p?.toFixed(3)}) \u2014 report Median [IQR] (${summary.median?.toFixed(1)} [${summary.q1?.toFixed(1)}\u2013${summary.q3?.toFixed(1)}]).`
                      }
                      {Math.abs(summary.skewness) > 2 ? " Highly skewed \u2014 consider log-transformation." :
                       Math.abs(summary.skewness) > 1 ? " Moderately skewed." : ""}
                    </p>
                  </div>
                )}
                {summary.type === "categorical" && (
                  <div className="px-4 py-1.5 border-b border-gray-100 bg-amber-50 flex-shrink-0">
                    <p className="text-[10px] text-amber-800 leading-relaxed">
                      {summary.categories?.length} categories, n = {summary.n}. Report as n (%). Most frequent: {summary.categories?.[0]?.value} ({summary.categories?.[0]?.pct}%).
                      {summary.missing > 0 ? ` Missing: ${summary.missing} (${(summary.missing / (summary.n + summary.missing) * 100).toFixed(1)}%).` : ""}
                    </p>
                  </div>
                )}

                {/* Charts - now inside a 2D resizable box with red drag lines on right + bottom (as requested) */}
                <div className="p-4">
                  <div
                    className="relative border border-gray-200 rounded-lg bg-white shadow-sm overflow-hidden"
                    style={{ width: `${distPlotW}px`, height: `${distPlotH}px`, minWidth: 520, minHeight: 320 }}
                  >
                    {/* The actual distribution plot content */}
                    <div className="absolute inset-0 overflow-auto p-4">
                      {summary.type === "numeric" && <NumericView summary={summary} loadSummary={loadSummary} selected={selected ?? ""} />}
                      {summary.type === "categorical" && <CategoricalView summary={summary} />}
                    </div>

                    {/* Right vertical red resize line (drag to change width) */}
                    <div
                      onPointerDown={startDistResize("right")}
                      className="absolute top-0 bottom-0 w-[5px] right-0 cursor-col-resize bg-red-500/70 hover:bg-red-600 active:bg-red-700 z-20"
                      title="Drag to resize plot width"
                    />

                    {/* Bottom horizontal red resize line (drag to change height) */}
                    <div
                      onPointerDown={startDistResize("bottom")}
                      className="absolute left-0 right-0 h-[5px] bottom-0 cursor-row-resize bg-red-500/70 hover:bg-red-600 active:bg-red-700 z-20"
                      title="Drag to resize plot height"
                    />

                    {/* Small corner handle for convenience */}
                    <div
                      onPointerDown={startDistResize("right")} // diagonal would be nicer but this is simple
                      className="absolute bottom-0 right-0 w-3 h-3 bg-red-500/80 cursor-nwse-resize z-30 rounded-tl"
                      title="Drag corner to resize both"
                    />
                  </div>

                  <div className="text-[10px] text-gray-400 mt-1">
                    Drag the red lines on the right and bottom to resize the plot area • Changes are remembered
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
