/**
 * TitledPlot — drop-in Plotly wrapper with editable title, subtitle and
 * axis labels baked INTO the plot itself (Plotly title + annotation +
 * xaxis/yaxis titles). Whatever the user sees on screen is what the PNG
 * export captures.
 *
 * Why this exists
 * - The plain `<Plot>` placed the chart heading as HTML above the plot,
 *   so `Plotly.downloadImage` produced a bare chart without the heading.
 * - We also want the heading and the small "knot ... reference ..." line
 *   to be editable before export so users can polish wording without
 *   leaving the app.
 *
 * Usage
 * ```tsx
 * <TitledPlot
 *   data={[...]}
 *   layout={{ ... }}            // axes, traces — title/subtitle reserved here
 *   defaultTitle="LDL & exitus: Cox-RCS"
 *   defaultSubtitle="4 knots at 65/104/124/175 · reference = 114 (HR = 1.0)"
 *   defaultXAxis="LDL"
 *   defaultYAxis="Hazard Ratio (95% CI)"
 *   plotRefOut={rcsPlotRef}     // forwarded so the global exporter works
 * />
 * ```
 *
 * Notes
 * - When the user blanks a field the corresponding Plotly slot is hidden
 *   (so the chart's own data doesn't get crowded by empty headings).
 * - The control row above the plot can be hidden with `showEditor={false}`
 *   to render a read-only TitledPlot (handy for snapshots).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import Plot from "../PlotComponent";
import PlotExporter from "./PlotExporter";

export interface TitledPlotProps {
  data: any[];
  layout?: any;
  style?: React.CSSProperties;
  config?: any;
  defaultTitle?: string;
  defaultSubtitle?: string;
  defaultXAxis?: string;
  defaultYAxis?: string;
  showEditor?: boolean;
  /** Forward the inner Plotly ref so the global ResultExporter can grab it. */
  plotRefOut?: React.MutableRefObject<any>;
  /** Persist edits across re-renders. Provide a stable key per result. */
  storageKey?: string;
}

function loadStored(key: string | undefined): Record<string, string> | null {
  if (!key) return null;
  try {
    const raw = sessionStorage.getItem(`tp:${key}`);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveStored(key: string | undefined, payload: Record<string, string>): void {
  if (!key) return;
  try {
    sessionStorage.setItem(`tp:${key}`, JSON.stringify(payload));
  } catch {
    /* swallow — session storage may be unavailable */
  }
}

export default function TitledPlot({
  data,
  layout,
  style,
  config,
  defaultTitle = "",
  defaultSubtitle = "",
  defaultXAxis = "",
  defaultYAxis = "",
  showEditor = true,
  plotRefOut,
  storageKey,
}: TitledPlotProps) {
  const stored = useMemo(() => loadStored(storageKey), [storageKey]);
  const [title, setTitle]       = useState<string>(stored?.title    ?? defaultTitle);
  const [sub,   setSub]         = useState<string>(stored?.sub      ?? defaultSubtitle);
  const [xLab,  setXLab]        = useState<string>(stored?.xLab     ?? defaultXAxis);
  const [yLab,  setYLab]        = useState<string>(stored?.yLab     ?? defaultYAxis);
  const [open,  setOpen]        = useState(false);
  // Resizable figure (like the KM plot). Width 'auto' fills the column.
  const [plotW, setPlotW]       = useState<number | undefined>(undefined);
  const [plotH, setPlotH]       = useState<number>(typeof layout?.height === "number" ? layout.height : 440);
  const localRef = useRef<any>(null);
  const refToUse = plotRefOut ?? localRef;

  useEffect(() => {
    saveStored(storageKey, { title, sub, xLab, yLab });
  }, [storageKey, title, sub, xLab, yLab]);

  // Re-seed defaults when the caller swaps them for a new run.
  useEffect(() => {
    if (stored) return;
    setTitle(defaultTitle);
    setSub(defaultSubtitle);
    setXLab(defaultXAxis);
    setYLab(defaultYAxis);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultTitle, defaultSubtitle, defaultXAxis, defaultYAxis]);

  const mergedLayout = useMemo(() => {
    const base = layout || {};
    const userAnnotations = Array.isArray(base.annotations) ? base.annotations : [];
    const captionAnnotation = sub
      ? [{
          // Anchor at the plot-area bottom (y:0 paper) and offset by a FIXED
          // pixel amount, so the caption always lands inside the bottom margin
          // regardless of export height. A paper-fraction y (e.g. -0.18) scales
          // with the figure, so tall exports pushed it past the margin and it
          // was cropped out of the PNG.
          xref: "paper", yref: "paper",
          x: 0.5, y: 0, xanchor: "center", yanchor: "top",
          yshift: -70,
          text: sub,
          showarrow: false,
          font: { size: 11, color: "#6b7280" },
        }]
      : [];
    return {
      ...base,
      title: title
        ? { text: title, font: { size: 15, color: "#111827" }, x: 0.5, xanchor: "center" }
        : undefined,
      // Bottom margin needs extra room for the caption.
      margin: {
        ...(base.margin || {}),
        t: title ? Math.max(base.margin?.t ?? 30, 50) : (base.margin?.t ?? 30),
        b: sub   ? Math.max(base.margin?.b ?? 50, 90) : (base.margin?.b ?? 50),
      },
      xaxis: { ...(base.xaxis || {}), title: xLab ? { ...(base.xaxis?.title || {}), text: xLab } : (base.xaxis?.title) },
      yaxis: { ...(base.yaxis || {}), title: yLab ? { ...(base.yaxis?.title || {}), text: yLab } : (base.yaxis?.title) },
      annotations: [...userAnnotations, ...captionAnnotation],
      // The container (sized by the Width/Height sliders) drives dimensions.
      height: undefined,
      width: undefined,
      autosize: true,
    };
  }, [layout, title, sub, xLab, yLab]);

  return (
    <div className="space-y-2">
      {showEditor && (
        <div className="rounded-lg border border-gray-200 bg-gray-50">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="w-full flex items-center justify-between px-3 py-1 text-[10px] uppercase tracking-wider text-gray-500 hover:text-indigo-600"
          >
            <span>Chart labels (saved with export)</span>
            <span>{open ? "▴" : "▾"}</span>
          </button>
          {open && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 px-3 pb-2">
              <label className="text-xs space-y-0.5 col-span-1 md:col-span-2">
                <span className="block text-gray-500">Title</span>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="(no title)"
                  className="select w-full py-1 text-xs"
                />
              </label>
              <label className="text-xs space-y-0.5 col-span-1 md:col-span-2">
                <span className="block text-gray-500">Caption / subtitle</span>
                <input
                  type="text"
                  value={sub}
                  onChange={(e) => setSub(e.target.value)}
                  placeholder="(no caption)"
                  className="select w-full py-1 text-xs"
                />
              </label>
              <label className="text-xs space-y-0.5">
                <span className="block text-gray-500">X axis</span>
                <input
                  type="text"
                  value={xLab}
                  onChange={(e) => setXLab(e.target.value)}
                  placeholder="(default)"
                  className="select w-full py-1 text-xs"
                />
              </label>
              <label className="text-xs space-y-0.5">
                <span className="block text-gray-500">Y axis</span>
                <input
                  type="text"
                  value={yLab}
                  onChange={(e) => setYLab(e.target.value)}
                  placeholder="(default)"
                  className="select w-full py-1 text-xs"
                />
              </label>
              <button
                type="button"
                onClick={() => {
                  setTitle(defaultTitle);
                  setSub(defaultSubtitle);
                  setXLab(defaultXAxis);
                  setYLab(defaultYAxis);
                }}
                className="col-span-1 md:col-span-2 text-[10px] text-gray-500 hover:text-indigo-600 self-end justify-self-end"
              >
                Reset to defaults
              </button>
            </div>
          )}
        </div>
      )}

      {/* Size controls — match the KM plot; export uses these dimensions. */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-1">
        <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
          <span className="font-medium">Width</span>
          <input type="range" min={460} max={1400} step={20} value={plotW ?? 820}
            onChange={(e) => setPlotW(Number(e.target.value))} className="accent-indigo-500" style={{ width: 110 }} />
          <span className="tabular-nums w-8">{plotW ?? "auto"}</span>
          {plotW != null && <button onClick={() => setPlotW(undefined)} className="text-indigo-500 hover:text-indigo-700">auto</button>}
        </label>
        <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
          <span className="font-medium">Height</span>
          <input type="range" min={260} max={900} step={20} value={plotH}
            onChange={(e) => setPlotH(Number(e.target.value))} className="accent-indigo-500" style={{ width: 110 }} />
          <span className="tabular-nums w-8">{plotH}</span>
        </label>
      </div>

      <div className="relative" style={{ width: plotW != null ? plotW : "100%", height: plotH, maxWidth: "100%" }}>
        <Plot
          ref={refToUse as any}
          data={data}
          layout={mergedLayout}
          style={{ ...(style as object), width: "100%", height: "100%" }}
          config={config}
          useResizeHandler
        />
        <PlotExporter plotRef={refToUse} title={title || "chart"}
          defaultWidth={plotW ?? 1000} defaultHeight={plotH} />
      </div>
    </div>
  );
}
