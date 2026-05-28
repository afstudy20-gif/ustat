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
  // Prefer the Plotly instance react-plotly.js already attached to the
  // gd — reuses the exact bundle that drew the chart and bypasses the
  // ESM tree-shake bug entirely. Fall back to the dist subpath only when
  // _Plotly isn't present.
  let Plotly: any = (el as any)._Plotly;
  if (!Plotly?.toImage) {
    const mod: any = await import("plotly.js/dist/plotly");
    Plotly = mod?.toImage ? mod : mod?.default;
  }
  if (!Plotly?.toImage) {
    throw new Error("plotly.js toImage not available");
  }
  // scale 3.125 ≈ 300 DPI (96 PPI × 3.125 = 300). Plotly.downloadImage
  // crashes on plotly.js@3 in production builds (tree-shaking strips an
  // internal dep) — use toImage + anchor-click instead.
  const dataUrl: string = await Plotly.toImage(el, {
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

export default function ResultExporter({ title, headers, rows, plotRef, className = "" }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const safeTitle = title.replace(/[^\w\s-]/g, "").replace(/\s+/g, "_").slice(0, 50) || "export";
  const hasTable = headers && rows;
  const hasPlot = !!plotRef;

  const handle = async (format: "csv" | "xlsx" | "png" | "tiff") => {
    if (busy) return;
    setBusy(format);
    setErr(null);
    try {
      if (format === "csv" && hasTable) downloadCSV(safeTitle, headers, rows);
      if (format === "xlsx" && hasTable) await downloadXLSX(safeTitle, headers, rows);
      if (format === "png" && hasPlot) await downloadPNG(plotRef, safeTitle);
      if (format === "tiff" && hasPlot) await downloadTIFF(plotRef, safeTitle);
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
        </>
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
