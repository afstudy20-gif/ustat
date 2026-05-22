import { useState, useRef, useMemo } from "react";
import Plot from "../PlotComponent";
import PlotExporter from "./PlotExporter";

// ── Types ───────────────────────────────────────────────────────────────────

interface ForestRowInput {
  label: string;
  est: number | null;
  ci_low: number | null;
  ci_high: number | null;
  p: number | null;
  extra: string;          // free-text annotation: "(48 events)", "n=110", "OR", etc.
}

type ForestLayout = {
  customTitle: string;
  customSubtitle: string;
  xLabel: string;
  xScale: "log" | "linear";
  nullLine: number;        // 1 for HR/OR, 0 for β/Δ
  colorBy: "all_one" | "significance";
  markerStyle: "square" | "circle" | "diamond";
  showValueColumns: boolean;
  showExtra: boolean;
  pointColor: string;
  sigColor: string;
  nonSigColor: string;
  height: number;
};

const DEFAULT_LAYOUT: ForestLayout = {
  customTitle: "",
  customSubtitle: "",
  xLabel: "Hazard ratio (95% CI), log scale",
  xScale: "log",
  nullLine: 1,
  colorBy: "significance",
  markerStyle: "square",
  showValueColumns: true,
  showExtra: true,
  pointColor: "#475569",
  sigColor: "#dc2626",
  nonSigColor: "#475569",
  height: 480,
};

const PRESETS: { name: string; rows: ForestRowInput[]; layout: Partial<ForestLayout> }[] = [
  {
    name: "Sensitivity — model specifications",
    rows: [
      { label: "Unadjusted",                          est: 2.03, ci_low: 1.02, ci_high: 4.03, p: 0.04,  extra: "" },
      { label: "Adjusted: age + sex + urea",          est: 1.27, ci_low: 0.61, ci_high: 2.63, p: 0.52,  extra: "" },
      { label: "Adjusted: age + EF + diabetes",       est: 1.38, ci_low: 0.69, ci_high: 2.78, p: 0.36,  extra: "" },
      { label: "Full model (age+sex+urea+EF+DM)",     est: 1.43, ci_low: 0.69, ci_high: 2.96, p: 0.34,  extra: "" },
      { label: "Landmark: deaths >30 d excluded",     est: 1.48, ci_low: 0.66, ci_high: 3.32, p: 0.34,  extra: "" },
      { label: "Landmark: deaths >180 d excluded",    est: 1.17, ci_low: 0.50, ci_high: 2.72, p: 0.72,  extra: "" },
    ],
    layout: {
      customTitle: "LDL <100 mg/dL (ref >130): hazard ratio for all-cause mortality across models",
      customSubtitle: "(n=388, 48 events; red = 95% CI excludes 1)",
      xLabel: "Hazard ratio (95% CI), log scale",
    },
  },
  {
    name: "Multiple endpoints / time horizons",
    rows: [
      { label: "30-day mortality",   est: 2.55, ci_low: 0.52, ci_high: 12.65, p: 0.25,    extra: "(6 events)"  },
      { label: "6-month mortality",  est: 2.57, ci_low: 0.83, ci_high:  7.98, p: 0.10,    extra: "(12 events)" },
      { label: "1-year mortality",   est: 5.67, ci_low: 2.15, ci_high: 14.92, p: 0.0005,  extra: "(19 events)" },
      { label: "2-year mortality",   est: 6.13, ci_low: 2.35, ci_high: 15.95, p: 0.0003,  extra: "(20 events)" },
      { label: "Overall (all-cause)", est: 2.36, ci_low: 1.34, ci_high:  4.17, p: 0.003,  extra: "(48 events)" },
    ],
    layout: {
      customTitle: "LDL <100 vs ≥100 mg/dL: crude hazard ratio for mortality by time horizon",
      customSubtitle: "(unadjusted; red = 95% CI excludes 1)",
      xLabel: "Hazard ratio (95% CI), log scale",
    },
  },
];

const emptyRow = (): ForestRowInput => ({
  label: "", est: null, ci_low: null, ci_high: null, p: null, extra: "",
});

// ── Component ───────────────────────────────────────────────────────────────

export default function ForestBuilderPanel() {
  const [rows, setRows] = useState<ForestRowInput[]>([emptyRow()]);
  const [layout, setLayout] = useState<ForestLayout>(DEFAULT_LAYOUT);
  const plotRef = useRef<any>(null);
  const [pastebox, setPastebox] = useState("");
  const [pasteOpen, setPasteOpen] = useState(false);

  const updateRow = (i: number, patch: Partial<ForestRowInput>) => {
    setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  };
  const addRow = () => setRows((p) => [...p, emptyRow()]);
  const delRow = (i: number) => setRows((p) => p.filter((_, idx) => idx !== i));
  const moveRow = (i: number, delta: -1 | 1) => {
    setRows((prev) => {
      const j = i + delta;
      if (j < 0 || j >= prev.length) return prev;
      const out = prev.slice();
      [out[i], out[j]] = [out[j], out[i]];
      return out;
    });
  };
  const clearAll = () => { setRows([emptyRow()]); setLayout(DEFAULT_LAYOUT); };

  const loadPreset = (idx: number) => {
    const p = PRESETS[idx];
    setRows(p.rows.map((r) => ({ ...r })));
    setLayout({ ...DEFAULT_LAYOUT, ...p.layout });
  };

  // Parse CSV / TSV paste: label, est, ci_low, ci_high[, p, extra]
  const applyPaste = () => {
    const lines = pastebox.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) return;
    const next: ForestRowInput[] = [];
    for (const line of lines) {
      const parts = line.split(/\t|,/).map((s) => s.trim());
      if (parts.length < 4) continue;
      const [label, est, lo, hi, p, extra] = parts;
      // Skip header-looking lines
      if (isNaN(parseFloat(est))) continue;
      next.push({
        label,
        est: parseFloat(est),
        ci_low: parseFloat(lo),
        ci_high: parseFloat(hi),
        p: p != null && p !== "" && !isNaN(parseFloat(p)) ? parseFloat(p) : null,
        extra: extra ?? "",
      });
    }
    if (next.length) setRows(next);
    setPasteOpen(false);
    setPastebox("");
  };

  // Valid rows only
  const validRows = useMemo(
    () => rows.filter((r) => r.label && r.est != null && r.ci_low != null && r.ci_high != null),
    [rows],
  );

  // Helpers
  const fmtP = (p: number | null) =>
    p == null ? "" : p < 0.001 ? "p<0.001" : `p=${p < 0.01 ? p.toFixed(3) : p.toFixed(2)}`;
  const fmtCI = (r: ForestRowInput) =>
    `${r.est!.toFixed(2)} (${r.ci_low!.toFixed(2)}–${r.ci_high!.toFixed(2)})`;
  const rowColor = (r: ForestRowInput): string => {
    if (layout.colorBy === "all_one") return layout.pointColor;
    const lo = r.ci_low ?? 0;
    const hi = r.ci_high ?? 0;
    const includesNull = lo <= layout.nullLine && hi >= layout.nullLine;
    return includesNull ? layout.nonSigColor : layout.sigColor;
  };

  const symbol = layout.markerStyle;

  // Plotly data
  const n = validRows.length;
  const yIdx = validRows.map((_, i) => n - 1 - i); // top-down order
  const colors = validRows.map((r) => rowColor(r));

  const baseTrace: any = {
    type: "scatter",
    mode: "markers",
    x: validRows.map((r) => r.est),
    y: yIdx,
    error_x: {
      type: "data", symmetric: false,
      array:      validRows.map((r) => (r.ci_high! - r.est!)),
      arrayminus: validRows.map((r) => (r.est! - r.ci_low!)),
      thickness: 2.2, width: 7,
      color: colors,
    },
    marker: {
      size: 12,
      symbol,
      color: colors,
      line: { color: "#ffffff", width: 0.5 },
    },
    hovertemplate: validRows.map((r) =>
      `<b>${r.label}</b><br>${fmtCI(r)}<br>${fmtP(r.p)}${r.extra ? `<br>${r.extra}` : ""}<extra></extra>`
    ),
    showlegend: false,
  };

  // Value-column annotations (right side)
  const showCols = layout.showValueColumns;
  const forestRight = showCols ? 0.55 : 0.95;
  const TX1 = 0.58;   // HR (95% CI), p
  const TX2 = 0.86;   // extra

  const annotations: any[] = [];
  if (showCols) {
    annotations.push({
      xref: "paper", yref: "paper", x: TX1, y: 1.04,
      xanchor: "left", yanchor: "bottom",
      text: "<b>HR (95% CI), p</b>", showarrow: false,
      font: { size: 11, color: "#1f2937" },
    });
    if (layout.showExtra && validRows.some((r) => r.extra)) {
      annotations.push({
        xref: "paper", yref: "paper", x: TX2, y: 1.04,
        xanchor: "left", yanchor: "bottom",
        text: "", showarrow: false,
      });
    }
    validRows.forEach((r, i) => {
      const c = rowColor(r);
      const yi = yIdx[i];
      annotations.push({
        xref: "paper", yref: "y", x: TX1, y: yi,
        xanchor: "left", yanchor: "middle",
        text: `${fmtCI(r)}${r.p != null ? `, ${fmtP(r.p)}` : ""}`,
        showarrow: false,
        font: { size: 11, color: c },
      });
      if (layout.showExtra && r.extra) {
        annotations.push({
          xref: "paper", yref: "y", x: TX2, y: yi,
          xanchor: "left", yanchor: "middle",
          text: r.extra,
          showarrow: false,
          font: { size: 11, color: c },
        });
      }
    });
  }

  // Title block
  const titleHtml = layout.customSubtitle
    ? `${layout.customTitle}<br><span style="font-size:11px;color:#6b7280">${layout.customSubtitle}</span>`
    : layout.customTitle;

  return (
    <div className="flex gap-4 h-full">
      {/* ── Left controls ── */}
      <div className="w-72 flex-shrink-0 space-y-4 overflow-y-auto pr-2">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Presets</h3>
          <div className="space-y-1.5">
            {PRESETS.map((p, i) => (
              <button key={i} onClick={() => loadPreset(i)}
                className="w-full text-left text-xs px-2 py-1 rounded border border-gray-300 text-gray-700 hover:bg-indigo-50 hover:border-indigo-300 transition-colors">
                {p.name}
              </button>
            ))}
            <button onClick={clearAll}
              className="w-full text-left text-xs px-2 py-1 rounded border border-orange-300 text-orange-600 hover:bg-orange-50 transition-colors">
              ✕ Clear all
            </button>
          </div>
        </div>

        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">Title</h3>
          <input
            value={layout.customTitle}
            onChange={(e) => setLayout((l) => ({ ...l, customTitle: e.target.value }))}
            placeholder="Plot title (optional)"
            className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
          />
          <input
            value={layout.customSubtitle}
            onChange={(e) => setLayout((l) => ({ ...l, customSubtitle: e.target.value }))}
            placeholder="Subtitle (optional)"
            className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
          />
          <input
            value={layout.xLabel}
            onChange={(e) => setLayout((l) => ({ ...l, xLabel: e.target.value }))}
            placeholder="X axis label"
            className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
          />
        </div>

        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">Scale & reference</h3>
          <div className="inline-flex rounded-md border border-gray-300 overflow-hidden text-[10px] w-full">
            {(["log", "linear"] as const).map((s) => (
              <button key={s}
                onClick={() => setLayout((l) => ({ ...l, xScale: s, nullLine: s === "log" ? 1 : 0 }))}
                className={`flex-1 py-1 ${layout.xScale === s ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
                {s === "log" ? "Log (HR/OR/RR)" : "Linear (β/Δ)"}
              </button>
            ))}
          </div>
          <label className="flex items-center justify-between text-[11px] text-gray-600">
            <span>Null reference line at:</span>
            <input type="number" step="any"
              value={layout.nullLine}
              onChange={(e) => setLayout((l) => ({ ...l, nullLine: parseFloat(e.target.value) || 0 }))}
              className="w-16 text-xs border border-gray-300 rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 text-right"
            />
          </label>
        </div>

        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">Color</h3>
          <div className="inline-flex rounded-md border border-gray-300 overflow-hidden text-[10px] w-full">
            {(["significance", "all_one"] as const).map((m) => (
              <button key={m}
                onClick={() => setLayout((l) => ({ ...l, colorBy: m }))}
                className={`flex-1 py-1 ${layout.colorBy === m ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
                {m === "significance" ? "By significance" : "Single color"}
              </button>
            ))}
          </div>
          {layout.colorBy === "significance" ? (
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <label className="flex flex-col gap-0.5">
                <span className="text-gray-500">CI excludes null</span>
                <input type="color" value={layout.sigColor}
                  onChange={(e) => setLayout((l) => ({ ...l, sigColor: e.target.value }))}
                  className="w-full h-6 border border-gray-300 rounded cursor-pointer" />
              </label>
              <label className="flex flex-col gap-0.5">
                <span className="text-gray-500">Includes null</span>
                <input type="color" value={layout.nonSigColor}
                  onChange={(e) => setLayout((l) => ({ ...l, nonSigColor: e.target.value }))}
                  className="w-full h-6 border border-gray-300 rounded cursor-pointer" />
              </label>
            </div>
          ) : (
            <label className="flex items-center justify-between text-[10px] text-gray-500">
              <span>Point color</span>
              <input type="color" value={layout.pointColor}
                onChange={(e) => setLayout((l) => ({ ...l, pointColor: e.target.value }))}
                className="w-10 h-6 border border-gray-300 rounded cursor-pointer" />
            </label>
          )}
        </div>

        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">Marker & value columns</h3>
          <div className="inline-flex rounded-md border border-gray-300 overflow-hidden text-[10px] w-full">
            {(["square", "circle", "diamond"] as const).map((s) => (
              <button key={s}
                onClick={() => setLayout((l) => ({ ...l, markerStyle: s }))}
                className={`flex-1 py-1 ${layout.markerStyle === s ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
                {s}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-2 text-[11px] text-gray-600">
            <input type="checkbox" checked={layout.showValueColumns}
              onChange={(e) => setLayout((l) => ({ ...l, showValueColumns: e.target.checked }))}
              className="accent-indigo-500" />
            Show <strong>HR (95% CI), p</strong> column
          </label>
          <label className="flex items-center gap-2 text-[11px] text-gray-600">
            <input type="checkbox" checked={layout.showExtra}
              onChange={(e) => setLayout((l) => ({ ...l, showExtra: e.target.checked }))}
              className="accent-indigo-500" />
            Show extra annotation column (events / n / etc.)
          </label>
          <label className="flex items-center justify-between text-[11px] text-gray-600">
            <span>Plot height</span>
            <input type="number" min={200} max={1200} step={20}
              value={layout.height}
              onChange={(e) => setLayout((l) => ({ ...l, height: parseInt(e.target.value) || 480 }))}
              className="w-16 text-xs border border-gray-300 rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 text-right"
            />
          </label>
        </div>

        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">Bulk paste (CSV/TSV)</h3>
          <p className="text-[10px] text-gray-500 leading-snug">
            One row per line:<br/>
            <code className="text-[10px]">label, est, ci_low, ci_high[, p][, extra]</code>
          </p>
          {pasteOpen ? (
            <>
              <textarea
                value={pastebox}
                onChange={(e) => setPastebox(e.target.value)}
                placeholder={"Unadjusted, 2.03, 1.02, 4.03, 0.04\nFull model, 1.43, 0.69, 2.96, 0.34, (48 events)"}
                rows={6}
                className="w-full text-[10px] font-mono border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
              />
              <div className="flex gap-1">
                <button onClick={applyPaste} className="flex-1 text-xs bg-indigo-600 text-white rounded py-1 hover:bg-indigo-700">
                  Apply
                </button>
                <button onClick={() => { setPasteOpen(false); setPastebox(""); }} className="text-xs text-gray-500 hover:text-red-500 px-2 py-1">
                  Cancel
                </button>
              </div>
            </>
          ) : (
            <button onClick={() => setPasteOpen(true)}
              className="w-full text-xs px-2 py-1 rounded border border-gray-300 text-gray-700 hover:bg-gray-50">
              📋 Paste rows
            </button>
          )}
        </div>
      </div>

      {/* ── Right pane: rows + plot ── */}
      <div className="flex-1 min-w-0 space-y-4 overflow-y-auto">
        <div className="panel space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Rows</h3>
            <span className="text-[10px] text-gray-400">{validRows.length} of {rows.length} valid</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-[10px] text-gray-500 border-b border-gray-200">
                  <th className="text-left px-1 py-1 w-8">#</th>
                  <th className="text-left px-1 py-1">Label</th>
                  <th className="text-right px-1 py-1 w-16">Est</th>
                  <th className="text-right px-1 py-1 w-16">CI low</th>
                  <th className="text-right px-1 py-1 w-16">CI high</th>
                  <th className="text-right px-1 py-1 w-16">p</th>
                  <th className="text-left px-1 py-1 w-32">Extra</th>
                  <th className="px-1 py-1 w-16"></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="px-1 py-0.5 text-gray-400">{i + 1}</td>
                    <td className="px-1 py-0.5">
                      <input value={r.label} placeholder="Label"
                        onChange={(e) => updateRow(i, { label: e.target.value })}
                        className="w-full border border-transparent rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 hover:border-gray-300" />
                    </td>
                    {(["est", "ci_low", "ci_high", "p"] as const).map((k) => (
                      <td key={k} className="px-1 py-0.5">
                        <input type="number" step="any"
                          value={r[k] == null ? "" : r[k] as number}
                          placeholder="—"
                          onChange={(e) => {
                            const v = e.target.value === "" ? null : parseFloat(e.target.value);
                            updateRow(i, { [k]: Number.isNaN(v as number) ? null : v } as any);
                          }}
                          className="w-full text-right border border-transparent rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 hover:border-gray-300" />
                      </td>
                    ))}
                    <td className="px-1 py-0.5">
                      <input value={r.extra}
                        placeholder="(48 events)"
                        onChange={(e) => updateRow(i, { extra: e.target.value })}
                        className="w-full border border-transparent rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 hover:border-gray-300" />
                    </td>
                    <td className="px-1 py-0.5 text-right whitespace-nowrap">
                      <button onClick={() => moveRow(i, -1)} className="text-gray-400 hover:text-indigo-600 px-0.5" title="Move up">▲</button>
                      <button onClick={() => moveRow(i, +1)} className="text-gray-400 hover:text-indigo-600 px-0.5" title="Move down">▼</button>
                      <button onClick={() => delRow(i)} className="text-gray-400 hover:text-red-500 px-0.5" title="Delete">✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button onClick={addRow}
            className="w-full text-xs px-2 py-1 rounded border border-dashed border-gray-300 text-gray-500 hover:border-indigo-300 hover:text-indigo-600 hover:bg-indigo-50 transition-colors">
            + Add row
          </button>
        </div>

        {validRows.length === 0 ? (
          <div className="panel h-48 flex items-center justify-center text-gray-400 text-sm">
            Enter at least one row with label, estimate, CI low, CI high.
          </div>
        ) : (
          <div className="panel space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-700">Forest plot</h3>
              <PlotExporter plotRef={plotRef} title={`Forest_custom`} />
            </div>
            <div ref={plotRef}>
              <Plot
                data={[baseTrace]}
                layout={{
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "#ffffff",
                  font: { color: "#374151", size: 12 },
                  height: layout.height,
                  autosize: true,
                  margin: { t: layout.customTitle ? 60 : 30, r: 30, b: 60, l: 240 },
                  title: layout.customTitle
                    ? { text: titleHtml, font: { size: 13, color: "#1f2937" }, x: 0.5, xanchor: "center" }
                    : undefined,
                  xaxis: {
                    type: layout.xScale,
                    gridcolor: "#e5e7eb",
                    zeroline: false,
                    domain: [0, forestRight],
                    title: { text: layout.xLabel, font: { size: 11, color: "#374151" } },
                  },
                  yaxis: {
                    tickvals: yIdx,
                    ticktext: validRows.map((r) => r.label),
                    range: [-0.6, n - 0.4],
                    gridcolor: "transparent",
                    zeroline: false,
                    tickfont: { size: 12, color: "#374151" },
                  },
                  shapes: [
                    {
                      type: "line", x0: layout.nullLine, x1: layout.nullLine,
                      xref: "x", yref: "paper", y0: 0, y1: 1,
                      line: { color: "#9ca3af", dash: "dash", width: 1.4 },
                    },
                  ],
                  annotations,
                  showlegend: false,
                }}
                style={{ width: "100%", height: layout.height }}
                useResizeHandler
                config={{ responsive: true, displaylogo: false }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
