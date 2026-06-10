/**
 * Shared loose-but-not-`any` types for the Plotly integration.
 *
 * The plot panels pass refs to react-plotly `<Plot>` instances and to
 * `TitledPlot`, then hand them to the shared exporter, which duck-types the
 * handle (`r.elRef?.current`, `r.el`). Plotly's own trace/layout types are
 * extremely wide, so the codebase historically used `any`. These aliases keep
 * the same flexibility while removing `any`: property access still works, but
 * the value is no longer the unsafe `any`.
 */
import type { RefObject } from "react";

/** Duck-typed handle for a Plotly/TitledPlot ref consumed by the exporter. */
export interface PlotCaptureHandle {
  el?: HTMLElement | null;
  elRef?: RefObject<HTMLElement | null>;
  [key: string]: unknown;
}

/** A ref to a capturable plot, as accepted by ResultExporter / PlotExporter. */
export type PlotRef = RefObject<PlotCaptureHandle | null>;

/** A single Plotly trace. Plotly accepts arbitrary keyed objects. */
export type PlotData = Record<string, unknown>;

/** A Plotly layout object. */
export type PlotLayout = Record<string, unknown>;

/** A Plotly config object. */
export type PlotConfig = Record<string, unknown>;
