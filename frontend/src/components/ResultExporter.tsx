/**
 * ResultExporter – standardized CSV / XLSX / 300 DPI PNG export toolbar.
 * Sits at the top-right of any result panel.
 *
 * Usage (table):
 *   <ResultExporter title="Summary" headers={["Variable","N","Mean"]} rows={data} />
 *
 * Usage (plot):
 *   <ResultExporter title="ROC Curve" plotRef={ref} />
 *
 * Usage (both):
 *   <ResultExporter title="Cox Results" headers={h} rows={r} plotRef={ref} />
 */
import { useState } from "react";
import { Download } from "lucide-react";
import { plotlyToTiffBlob, downloadBlob } from "../lib/tiffEncoder";
import { plotlyToDataUrl } from "../lib/plotlyExport";

interface Props {
  title: string;
  /** Column headers for CSV/XLSX export */
  headers?: string[];
  /** Table rows for CSV/XLSX export */
  rows?: (string | number | null | undefined)[][];
  /** Plotly chart element ref for PNG export */
  plotRef?: React.RefObject<any>;
  className?: string;
}

function downloadCSV(filename: string, headers: string[], rows: (string | number | null | undefined)[][]) {
  const escape = (v: string | number | null | undefined) =>
    `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = [headers, ...rows].map((r) => r.map(escape).join(","));
  const blob = new Blob([lines.join("\r\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename + ".csv"; a.click();
  URL.revokeObjectURL(url);
}

async function downloadXLSX(filename: string, headers: string[], rows: (string | number | null | undefined)[][]) {
  // xlsx package ships both ESM (named exports) and CJS (default export) builds.
  // Resolve whichever shape Vite delivers in this environment.
  const mod: any = await import("xlsx");
  const XLSX: any = mod?.utils ? mod : mod?.default;
  if (!XLSX?.utils?.aoa_to_sheet) {
    throw new Error("xlsx module loaded but utils.aoa_to_sheet not available");
  }
  const ws = XLSX.utils.aoa_to_sheet([headers, ...rows]);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Results");
  // Use write() + blob for macOS Safari compatibility (writeFile can fail)
  const wbout = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  const blob = new Blob([wbout], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename + ".xlsx";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

async function downloadPNG(plotRef: React.RefObject<any>, filename: string) {
  // Resolve to a Plotly graph div. react-plotly.js exposes `.el`; raw refs
  // give the DOM node directly; some wrappers nest it one deeper. A graph
  // div is recognisable by the `_fullLayout` property Plotly attaches at
  // mount time.
  const candidates: any[] = [];
  const r = plotRef.current;
  if (r) {
    candidates.push(r.el);
    candidates.push(r);
    candidates.push(r.elRef?.current);
    if (typeof r.querySelector === "function") {
      candidates.push(r.querySelector(".plotly-graph-div") || r.querySelector(".js-plotly-plot"));
    }
  }
  const el = candidates.find((c) => c && (c as any)._fullLayout) as HTMLElement | undefined;
  if (!el) {
    throw new Error("plot is not mounted yet — wait for the chart to render and try again");
  }
  // scale 3.125 ≈ 300 DPI (96 PPI × 3.125 = 300). The shared export helper
  // reuses the single statically-imported Plotly bundle and keeps axis/tick
  // text in the exported bitmap (see lib/plotlyExport).
  const dataUrl: string = await plotlyToDataUrl(el, {
    format: "png",
    width: 1200,
    height: 700,
    scale: 3.125,
  });
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${filename}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadTIFF(plotRef: React.RefObject<any>, filename: string) {
  // Resolve the Plotly graph div the same way downloadPNG does.
  const candidates: any[] = [];
  const r = plotRef.current;
  if (r) {
    candidates.push(r.el);
    candidates.push(r);
    candidates.push(r.elRef?.current);
    if (typeof r.querySelector === "function") {
      candidates.push(r.querySelector(".plotly-graph-div") || r.querySelector(".js-plotly-plot"));
    }
  }
  const el = candidates.find((c) => c && (c as any)._fullLayout) as HTMLElement | undefined;
  if (!el) {
    throw new Error("plot is not mounted yet — wait for the chart to render and try again");
  }
  const blob = await plotlyToTiffBlob(el, { width: 1200, height: 700, dpi: 300 });
  downloadBlob(blob, `${filename}.tiff`);
}

/** Render the chart to a PNG blob — shared by copy + download paths. */
async function renderPlotPngBlob(plotRef: React.RefObject<any>): Promise<Blob> {
  const candidates: any[] = [];
  const r = plotRef.current;
  if (r) {
    candidates.push(r.el);
    candidates.push(r);
    candidates.push(r.elRef?.current);
    if (typeof r.querySelector === "function") {
      candidates.push(r.querySelector(".plotly-graph-div") || r.querySelector(".js-plotly-plot"));
    }
  }
  const el = candidates.find((c) => c && (c as any)._fullLayout) as HTMLElement | undefined;
  if (!el) throw new Error("plot is not mounted yet — wait for the chart to render and try again");
  const dataUrl: string = await plotlyToDataUrl(el, {
    format: "png",
    width: 1200,
    height: 700,
    scale: 3.125,
  });
  const res = await fetch(dataUrl);
  return await res.blob();
}

/** Copy the chart to the clipboard as PNG (system clipboard). */
async function copyPlotToClipboard(plotRef: React.RefObject<any>) {
  if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
    throw new Error("Clipboard API not available in this browser");
  }
  const blob = await renderPlotPngBlob(plotRef);
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}

/** Copy the table to the clipboard as TSV (pastes into Excel / Word / Sheets). */
async function copyTableToClipboard(headers: string[], rows: (string | number | null | undefined)[][]) {
  if (!navigator.clipboard?.writeText) {
    throw new Error("Clipboard API not available in this browser");
  }
  const esc = (v: string | number | null | undefined) => {
    const s = String(v ?? "");
    // Tabs / newlines inside a cell break TSV — wrap the cell in quotes.
    return /[\t\n\r"]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const tsv = [headers, ...rows].map((r) => r.map(esc).join("\t")).join("\n");
  await navigator.clipboard.writeText(tsv);
}

export default function ResultExporter({ title, headers, rows, plotRef, className = "" }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // "Copied" pill flashes for ~1.5 s on a successful copy.
  const [copyToast, setCopyToast] = useState<string | null>(null);

  const safeTitle = title.replace(/[^\w\s-]/g, "").replace(/\s+/g, "_").slice(0, 50) || "export";
  const hasTable = headers && rows;
  const hasPlot = !!plotRef;

  const handle = async (format: "csv" | "xlsx" | "png" | "tiff" | "copy-table" | "copy-plot") => {
    if (busy) return;
    setBusy(format);
    setErr(null);
    try {
      if (format === "csv" && hasTable) downloadCSV(safeTitle, headers, rows);
      if (format === "xlsx" && hasTable) await downloadXLSX(safeTitle, headers, rows);
      if (format === "png" && hasPlot) await downloadPNG(plotRef, safeTitle);
      if (format === "tiff" && hasPlot) await downloadTIFF(plotRef, safeTitle);
      if (format === "copy-table" && hasTable) {
        await copyTableToClipboard(headers, rows);
        setCopyToast("Table copied — paste into Excel / Word");
        setTimeout(() => setCopyToast(null), 1500);
      }
      if (format === "copy-plot" && hasPlot) {
        await copyPlotToClipboard(plotRef);
        setCopyToast("Chart copied to clipboard");
        setTimeout(() => setCopyToast(null), 1500);
      }
    } catch (e) {
      console.error("Export error:", e);
      const msg = e instanceof Error ? e.message : String(e);
      setErr(`${format.toUpperCase()} export failed: ${msg}`);
    } finally {
      setBusy(null);
    }
  };

  if (!hasTable && !hasPlot) return null;

  return (
    <div className={`flex items-center gap-1 ${className}`}>
      <span className="text-[10px] text-gray-400 mr-0.5 flex items-center gap-0.5">
        <Download size={10} /> Export
      </span>
      {hasTable && (
        <>
          <button
            onClick={() => handle("csv")}
            disabled={!!busy}
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600 disabled:opacity-40 transition-colors"
          >
            {busy === "csv" ? "…" : "CSV"}
          </button>
          <button
            onClick={() => handle("xlsx")}
            disabled={!!busy}
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600 disabled:opacity-40 transition-colors"
          >
            {busy === "xlsx" ? "…" : "XLSX"}
          </button>
          <button
            onClick={() => handle("copy-table")}
            disabled={!!busy}
            title="Copy table to clipboard as TSV — paste into Excel / Word / Google Sheets"
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-emerald-50 hover:text-emerald-700 hover:border-emerald-200 disabled:opacity-40 transition-colors"
          >
            {busy === "copy-table" ? "…" : "⧉ Copy"}
          </button>
        </>
      )}
      {hasPlot && (
        <>
          <button
            onClick={() => handle("png")}
            disabled={!!busy}
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600 disabled:opacity-40 transition-colors"
          >
            {busy === "png" ? "…" : "PNG 300dpi"}
          </button>
          <button
            onClick={() => handle("tiff")}
            disabled={!!busy}
            title="Baseline uncompressed RGB TIFF (journal-ready, larger file)"
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600 disabled:opacity-40 transition-colors"
          >
            {busy === "tiff" ? "…" : "TIFF 300dpi"}
          </button>
          <button
            onClick={() => handle("copy-plot")}
            disabled={!!busy}
            title="Copy chart to clipboard as PNG — paste into PowerPoint / Word / Slack"
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-emerald-50 hover:text-emerald-700 hover:border-emerald-200 disabled:opacity-40 transition-colors"
          >
            {busy === "copy-plot" ? "…" : "⧉ Copy chart"}
          </button>
        </>
      )}
      {copyToast && (
        <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-emerald-600 text-white shadow ml-1 whitespace-nowrap">
          {copyToast}
        </span>
      )}
      {err && (
        <span
          title={err}
          className="text-[10px] text-red-600 ml-1 max-w-[280px] truncate"
        >
          {err}
        </span>
      )}
    </div>
  );
}
