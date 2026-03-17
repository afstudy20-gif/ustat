import { useEffect, useState, useCallback, useRef } from "react";
import { useStore } from "../store";
import api from "../api";
import Plot from "../PlotComponent";

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
  { id: "qq",        label: "Q-Q Plot" },
] as const;
type ChartTab = typeof CHART_TABS[number]["id"];

function NumericView({ summary }: { summary: any }) {
  const [chartTab, setChartTab] = useState<ChartTab>("histogram");

  const histData = [{
    type: "bar" as const,
    x: summary.histogram.map((b: any) => (b.bin_start + b.bin_end) / 2),
    y: summary.histogram.map((b: any) => b.count),
    width: summary.histogram.map((b: any) => b.bin_end - b.bin_start),
    marker: { color: "#6366f1", opacity: 0.85 },
    name: "Count",
    hovertemplate: "Range: %{customdata[0]}–%{customdata[1]}<br>Count: %{y}<extra></extra>",
    customdata: summary.histogram.map((b: any) => [b.bin_start.toFixed(2), b.bin_end.toFixed(2)]),
  }];

  const boxData = [{
    type: "box" as const,
    q1: [summary.q1],
    median: [summary.median],
    mean: [summary.mean],
    q3: [summary.q3],
    lowerfence: [summary.min],
    upperfence: [summary.max],
    sd: [summary.std],
    name: "Distribution",
    boxmean: true,
    marker: { color: "#6366f1", size: 5 },
    line: { color: "#6366f1" },
    fillcolor: "rgba(99,102,241,0.15)",
    hovertemplate:
      `Median: ${summary.median?.toFixed(2)}<br>` +
      `Q1: ${summary.q1?.toFixed(2)}  Q3: ${summary.q3?.toFixed(2)}<br>` +
      `Min: ${summary.min?.toFixed(2)}  Max: ${summary.max?.toFixed(2)}<br>` +
      `Mean ± SD: ${summary.mean?.toFixed(2)} ± ${summary.std?.toFixed(2)}<extra></extra>`,
  }];

  const qqData = [
    {
      type: "scatter" as const, mode: "markers" as const,
      x: summary.qq.map((p: any) => p.x),
      y: summary.qq.map((p: any) => p.y),
      marker: { color: "#6366f1", size: 4 },
      name: "Observed",
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
      };
    })(),
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
        <Plot
          data={histData}
          layout={{ ...BASE_LAYOUT, autosize: true, bargap: 0.02,
            xaxis: { ...BASE_LAYOUT.xaxis, title: { text: "Value" } },
            yaxis: { ...BASE_LAYOUT.yaxis, title: { text: "Count" } },
          }}
          style={{ width: "100%", height: 380 }}
          useResizeHandler config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        />
      )}

      {/* Box Plot */}
      {chartTab === "boxplot" && (
        <Plot
          data={boxData}
          layout={{
            ...BASE_LAYOUT,
            autosize: true,
            yaxis: { ...BASE_LAYOUT.yaxis, title: { text: "Value" } },
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
      )}

      {/* Q-Q Plot */}
      {chartTab === "qq" && (
        <Plot
          data={qqData}
          layout={{ ...BASE_LAYOUT, autosize: true,
            title: { text: "Q-Q Plot (Normality)", font: { color: "#374151", size: 12 } },
            xaxis: { ...BASE_LAYOUT.xaxis, title: { text: "Theoretical quantiles" } },
            yaxis: { ...BASE_LAYOUT.yaxis, title: { text: "Sample quantiles" } },
          }}
          style={{ width: "100%", height: 380 }}
          useResizeHandler config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        />
      )}
    </div>
  );
}

// ── Main chart for categorical columns ──────────────────────────────────────

function CategoricalView({ summary }: { summary: any }) {
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
    marker: { color: "#6366f1", opacity: 0.85 },
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
          xaxis: { ...BASE_LAYOUT.xaxis, title: { text: "Count" } },
          yaxis: { ...BASE_LAYOUT.yaxis, automargin: true },
          margin: { ...BASE_LAYOUT.margin, l: 90 },
        }}
        style={{ width: "100%", height: Math.max(160, cats.length * 28 + 60) }}
        useResizeHandler config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      />
    </div>
  );
}

// ── Scatter view ─────────────────────────────────────────────────────────────

const PALETTE  = ["#6366f1","#f59e0b","#10b981","#ef4444","#06b6d4","#ec4899","#a78bfa","#fb923c"];
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
            color: PALETTE[i % PALETTE.length],
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
          marker: { color: "#6366f1", size: 7, opacity: 0.78, symbol: SYMBOLS[i % SYMBOLS.length] },
          hovertemplate: `<b>${shape}</b>: ${g}<br>${xCol}: %{x}<br>${yCol}: %{y}<extra></extra>`,
        });
      });
    } else {
      traces.push({
        type: "scatter", mode: "markers",
        x: pts.map((p) => p[xCol]),
        y: pts.map((p) => p[yCol]),
        name: "Data",
        marker: { color: "#6366f1", size: 6, opacity: 0.7, symbol: "circle" },
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
                xaxis: { ...BASE_LAYOUT.xaxis, title: { text: xCol } },
                yaxis: { ...BASE_LAYOUT.yaxis, title: { text: yCol } },
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

export default function DescriptivePanel() {
  const session = useStore((s) => s.session);
  const [colMeta, setColMeta] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [summary, setSummary] = useState<any | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [view, setView] = useState<"distribution" | "scatter">("distribution");

  useEffect(() => {
    if (!session) return;
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

  const loadSummary = useCallback((colName: string) => {
    if (!session) return;
    const kind = session.columns.find((c) => c.name === colName)?.kind ?? undefined;
    setSelected(colName);
    setSummary(null);
    setSummaryLoading(true);
    api.get(`/api/stats/${session.session_id}/column_summary`, { params: { column: colName, kind } })
      .then((r) => setSummary(r.data))
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
            return (
              <div
                key={c.name}
                onClick={() => { setView("distribution"); loadSummary(c.name); }}
                className={`flex items-center justify-between px-3 py-2 cursor-pointer border-b border-gray-100 transition-colors
                  ${isActive ? "bg-indigo-50 border-l-2 border-l-indigo-500" : "hover:bg-gray-50"}`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className={`text-[9px] font-bold px-1 rounded flex-shrink-0
                    ${c.kind === "numeric" ? "bg-blue-100 text-blue-700" : "bg-purple-100 text-purple-700"}`}>
                    {c.kind === "numeric" ? "N" : "C"}
                  </span>
                  <span className="text-xs text-gray-700 truncate">{c.name}</span>
                </div>
                {meta && (
                  <div className="flex-shrink-0 ml-1">
                    {c.kind === "categorical" ? (
                      <div className="w-10 h-2.5 bg-gray-200 rounded-sm overflow-hidden flex">
                        <div className="h-full bg-purple-400" style={{ width: "60%" }} />
                        <div className="h-full bg-amber-400" style={{ width: "40%" }} />
                      </div>
                    ) : (
                      <div className="w-10 h-2.5 bg-gray-200 rounded-sm overflow-hidden flex items-end gap-px px-px">
                        {[0.3, 0.6, 1, 0.8, 0.5, 0.4, 0.7, 0.9].map((h, i) => (
                          <div key={i} className="flex-1 bg-indigo-400 rounded-sm" style={{ height: `${h * 100}%` }} />
                        ))}
                      </div>
                    )}
                  </div>
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

        {/* ── View tab switcher ── */}
        <div className="flex items-center gap-1 px-4 py-2 border-b border-gray-200 flex-shrink-0 bg-gray-50">
          {([
            { id: "distribution", label: "📊 Distribution" },
            { id: "scatter",      label: "⬡ Scatter Plot" },
          ] as const).map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors
                ${view === id
                  ? "bg-indigo-600 text-white"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-200"}`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* ── Scatter view ── */}
        {view === "scatter" && (
          <ScatterView
            key={session.session_id}
            sessionId={session.session_id}
            numCols={numCols}
            catCols={catCols}
            defaultX={selected && numCols.includes(selected) ? selected : (numCols[0] ?? "")}
          />
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
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 flex-shrink-0">
                  <div>
                    <h2 className="text-base font-semibold text-gray-900">
                      Distribution of <span className="text-indigo-600">{selected}</span>
                    </h2>
                    <p className="text-xs text-gray-400 mt-0.5">
                      {summary.type === "numeric" ? "Continuous variable" : "Categorical variable"} ·{" "}
                      n = {summary.n}
                      {summary.missing > 0 && (
                        <span className="text-amber-500 ml-1">· {summary.missing} missing</span>
                      )}
                    </p>
                  </div>
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

                {/* Charts */}
                <div className="flex-1 overflow-y-auto p-4">
                  {summary.type === "numeric" && <NumericView summary={summary} />}
                  {summary.type === "categorical" && <CategoricalView summary={summary} />}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
