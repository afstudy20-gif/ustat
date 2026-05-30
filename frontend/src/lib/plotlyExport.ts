/**
 * Shared Plotly instance + hardened chart-image export.
 *
 * Two production-only bugs made exported charts come out with the plotted
 * shapes but NO text (axis titles, tick labels, annotations all missing):
 *
 *  1. Wrong Plotly instance.
 *     react-plotly.js renders every chart with the monolithic
 *     `plotly.js/dist/plotly` bundle but never attaches it to the graph div.
 *     The exporters read `gd._Plotly` (always undefined) and fell through to
 *     `await import("plotly.js/dist/plotly")` — a *separate* dynamic chunk
 *     whose default-export interop the production build has repeatedly
 *     mangled (see git history: "Cannot read properties of undefined
 *     (reading 'prototype')"). Importing the bundle ONCE, statically, here
 *     and exporting through it makes the export use the exact same Plotly
 *     that drew the chart on screen.
 *
 *  2. Unresolvable font in the rasteriser.
 *     `Plotly.toImage` serialises the chart to an SVG and rasterises it by
 *     loading that SVG into a detached <img>. Fonts referenced only by a CSS
 *     keyword such as `system-ui` (the app's default plot font) do not always
 *     resolve in that sandbox, so every <text> run renders blank. Re-rendering
 *     the export from a figure whose font-family is a concrete, universally
 *     installed stack keeps the text.
 */
import _Plotly from "plotly.js/dist/plotly";

// dist/plotly is UMD; under Vite the default import is the Plotly object, but
// stay defensive about the CJS/ESM interop shape.
const Plotly: any =
  (_Plotly as any)?.toImage ? _Plotly : ((_Plotly as any)?.default ?? _Plotly);

export default Plotly;

export type PlotlyExportFormat = "png" | "svg" | "jpeg";

export interface PlotlyExportOpts {
  format: PlotlyExportFormat;
  width: number;
  height: number;
  /** Raster up-scale factor (e.g. dpi / 72). Ignored for SVG. */
  scale?: number;
}

// Concrete, system-installed stack. A bare "system-ui" can fail to resolve
// inside the detached SVG image the browser rasterises, blanking all text.
const EXPORT_FONT_FAMILY = "Arial, Helvetica, system-ui, sans-serif";

/**
 * Rasterise a mounted Plotly graph div to a data URL with text preserved.
 *
 * Re-renders from the chart's own data/layout with the font family forced to
 * a guaranteed-available stack — without mutating the on-screen chart. Falls
 * back to rasterising the live graph div directly if anything goes wrong, so
 * this never regresses the previous behaviour.
 */
export async function plotlyToDataUrl(
  el: HTMLElement,
  opts: PlotlyExportOpts,
): Promise<string> {
  if (!Plotly?.toImage) throw new Error("plotly.js toImage not available");

  // Make sure any web fonts have finished loading before we snapshot.
  try {
    await (document as Document & { fonts?: { ready?: Promise<unknown> } }).fonts?.ready;
  } catch {
    /* Font Loading API not available — ignore. */
  }

  const scaleOpt = opts.scale && opts.scale !== 1 && opts.format !== "svg"
    ? { scale: opts.scale }
    : {};
  const imgOpts = {
    format: opts.format,
    width: opts.width,
    height: opts.height,
    ...scaleOpt,
  };

  const gd = el as unknown as { data?: unknown[]; layout?: Record<string, unknown> };
  if (gd?.data && gd?.layout) {
    try {
      const baseLayout = gd.layout ?? {};
      const baseFont = (baseLayout.font as Record<string, unknown>) ?? {};
      const figure = {
        data: gd.data,
        layout: {
          ...baseLayout,
          font: { ...baseFont, family: EXPORT_FONT_FAMILY },
        },
      };
      return await Plotly.toImage(figure, imgOpts);
    } catch {
      /* Fall back to rasterising the live graph div as-is. */
    }
  }
  return Plotly.toImage(el, imgOpts);
}
