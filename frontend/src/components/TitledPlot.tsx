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
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ComponentRef } from "react";
import Plot from "../PlotComponent";
import PlotExporter from "./PlotExporter";
import { registerPlotCaptureHooks } from "../lib/plotCapture";
import type { PlotData, PlotLayout, PlotConfig, PlotCaptureHandle } from "../lib/plotTypes";

/** Minimal shape of the Plotly module / graph-div fields we call. */
interface PlotlyRelayout {
  relayout?: (gd: HTMLElement, update: Record<string, unknown>) => Promise<unknown>;
}

/** The nested layout fields TitledPlot reads from / merges into. */
interface LayoutLike {
  height?: unknown;
  margin?: { t?: number; b?: number } & Record<string, unknown>;
  annotations?: unknown[];
  xaxis?: { title?: { text?: string } & Record<string, unknown> } & Record<string, unknown>;
  yaxis?: { title?: { text?: string } & Record<string, unknown> } & Record<string, unknown>;
}

/** Flat, typed view of the layout reads used during export relayout. */
function readLayout(layout: PlotLayout | undefined): {
  marginT?: number;
  marginB?: number;
  xAxisTitle?: string;
  yAxisTitle?: string;
} {
  const l = (layout ?? {}) as LayoutLike;
  return {
    marginT: l.margin?.t,
    marginB: l.margin?.b,
    xAxisTitle: l.xaxis?.title?.text,
    yAxisTitle: l.yaxis?.title?.text,
  };
}

export interface TitledPlotProps {
  data: PlotData[];
  layout?: PlotLayout;
  style?: React.CSSProperties;
  config?: PlotConfig;
  defaultTitle?: string;
  defaultSubtitle?: string;
  defaultXAxis?: string;
  defaultYAxis?: string;
  showEditor?: boolean;
  /** Forward the inner Plotly ref so the global ResultExporter can grab it. */
  plotRefOut?: React.MutableRefObject<PlotCaptureHandle | null>;
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
  // Hide chosen labels in the EXPORT only (kept on screen).
  const [hideExport, setHideExport] = useState({ title: false, caption: false, axes: false });
  const localRef = useRef<PlotCaptureHandle | null>(null);
  const refToUse = plotRefOut ?? localRef;

  // Caption is appended after any caller-supplied annotations.
  const captionIndex = Array.isArray(layout?.annotations) ? layout.annotations.length : 0;

  // Transiently strip labels right before a copy/download capture, then
  // restore — so the on-screen figure keeps its labels but the file omits the
  // ones the user ticked (handy when the title goes in the manuscript legend).
  const exportRelayout = useCallback(async (strip: boolean) => {
    const gd = (refToUse.current as PlotCaptureHandle | null)?.el;
    if (!gd) return;
    // react-plotly.js doesn't always attach `_Plotly` to the graph div, so fall
    // back to the dist bundle — otherwise the label-hide silently no-ops.
    let Plotly: PlotlyRelayout | undefined = (gd as { _Plotly?: PlotlyRelayout })._Plotly;
    if (!Plotly?.relayout) {
      const mod = (await import("plotly.js/dist/plotly")) as PlotlyRelayout & { default?: PlotlyRelayout };
      Plotly = mod?.relayout ? mod : mod?.default;
    }
    if (!Plotly?.relayout) return;
    const lv = readLayout(layout);
    const upd: Record<string, unknown> = {};
    const baseTop = lv.marginT ?? 30;
    const baseBottom = lv.marginB ?? 50;
    if (hideExport.title) {
      upd["title.text"] = strip ? "" : (title || "");
      upd["margin.t"] = strip ? baseTop : Math.max(baseTop, title ? 50 : baseTop);
    }
    if (hideExport.caption && sub) {
      upd[`annotations[${captionIndex}].visible`] = !strip;
      upd["margin.b"] = strip ? baseBottom : Math.max(baseBottom, 90);
    }
    if (hideExport.axes) {
      upd["xaxis.title.text"] = strip ? "" : (xLab || lv.xAxisTitle || "");
      upd["yaxis.title.text"] = strip ? "" : (yLab || lv.yAxisTitle || "");
    }
    if (Object.keys(upd).length) { try { await Plotly.relayout(gd, upd); } catch { /* non-fatal */ } }
  }, [captionIndex, hideExport, layout, refToUse, sub, title, xLab, yLab]);

  // Result panels may expose a second, shared export toolbar outside this
  // component. Register the same transient relayout hooks on the Plotly ref so
  // its PNG/TIFF/copy actions honor the label visibility choices too.
  useEffect(() => registerPlotCaptureHooks(refToUse, {
    beforeCapture: () => exportRelayout(true),
    afterCapture: () => exportRelayout(false),
  }), [exportRelayout, refToUse]);

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
    const base: PlotLayout = layout || {};
    const baseView = base as LayoutLike;
    const userAnnotations = Array.isArray(baseView.annotations) ? baseView.annotations : [];
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
        ...(baseView.margin || {}),
        t: title ? Math.max(baseView.margin?.t ?? 30, 50) : (baseView.margin?.t ?? 30),
        b: sub   ? Math.max(baseView.margin?.b ?? 50, 90) : (baseView.margin?.b ?? 50),
      },
      xaxis: { ...(baseView.xaxis || {}), title: xLab ? { ...(baseView.xaxis?.title || {}), text: xLab } : (baseView.xaxis?.title) },
      yaxis: { ...(baseView.yaxis || {}), title: yLab ? { ...(baseView.yaxis?.title || {}), text: yLab } : (baseView.yaxis?.title) },
      annotations: [...userAnnotations, ...captionAnnotation],
      // The container (sized by the Width/Height sliders) drives dimensions.
      height: undefined,
      width: undefined,
      autosize: true,
    };
  }, [layout, title, sub, xLab, yLab]);

  const exportSafeConfig = useMemo(() => ({
    ...(config || {}),
    // The built-in Plotly camera bypasses the label-hide capture hooks.
    // TitledPlot already supplies the shared exporter, so keep one reliable
    // export path for PNG/SVG/TIFF/copy.
    modeBarButtonsToRemove: Array.from(new Set([
      ...((config?.modeBarButtonsToRemove as string[] | undefined) ?? []),
      "toImage",
    ])),
  }), [config]);

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
              <div className="col-span-1 md:col-span-2 flex flex-wrap items-center gap-x-3 gap-y-1 pt-1 border-t border-gray-100">
                <span className="text-[10px] text-gray-400">Hide in export:</span>
                <label className="flex items-center gap-1 text-[10px] text-gray-500">
                  <input type="checkbox" checked={hideExport.title}
                    onChange={(e) => setHideExport((h) => ({ ...h, title: e.target.checked }))}
                    className="accent-indigo-500" /> Title
                </label>
                <label className="flex items-center gap-1 text-[10px] text-gray-500">
                  <input type="checkbox" checked={hideExport.caption}
                    onChange={(e) => setHideExport((h) => ({ ...h, caption: e.target.checked }))}
                    className="accent-indigo-500" /> Caption
                </label>
                <label className="flex items-center gap-1 text-[10px] text-gray-500">
                  <input type="checkbox" checked={hideExport.axes}
                    onChange={(e) => setHideExport((h) => ({ ...h, axes: e.target.checked }))}
                    className="accent-indigo-500" /> Axis titles
                </label>
                <span className="text-[9px] text-gray-400">(stays on screen; omitted from PNG/TIFF/copy)</span>
              </div>
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
          ref={refToUse as unknown as React.Ref<ComponentRef<typeof Plot>>}
          data={data as React.ComponentProps<typeof Plot>["data"]}
          layout={mergedLayout as React.ComponentProps<typeof Plot>["layout"]}
          style={{ ...style, width: "100%", height: "100%" }}
          config={exportSafeConfig as React.ComponentProps<typeof Plot>["config"]}
          useResizeHandler
        />
        <PlotExporter plotRef={refToUse} title={title || "chart"}
          defaultWidth={plotW ?? 1000} defaultHeight={plotH}
          onBeforeCapture={() => exportRelayout(true)}
          onAfterCapture={() => exportRelayout(false)} />
      </div>
    </div>
  );
}
