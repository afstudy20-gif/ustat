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
  const XLSX = (await import("xlsx")).default;
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
  const el: HTMLElement | null = plotRef.current?.el ?? plotRef.current;
  if (!el) return;
  const Plotly = (await import("plotly.js")).default;
  // scale 3.125 ≈ 300 DPI (96 PPI × 3.125 = 300)
  await (Plotly as any).downloadImage(el, {
    format: "png",
    width: 1200,
    height: 700,
    scale: 3.125,
    filename,
  });
}

export default function ResultExporter({ title, headers, rows, plotRef, className = "" }: Props) {
  const [busy, setBusy] = useState<string | null>(null);

  const safeTitle = title.replace(/[^\w\s-]/g, "").replace(/\s+/g, "_").slice(0, 50) || "export";
  const hasTable = headers && rows;
  const hasPlot = !!plotRef;

  const handle = async (format: "csv" | "xlsx" | "png") => {
    if (busy) return;
    setBusy(format);
    try {
      if (format === "csv" && hasTable) downloadCSV(safeTitle, headers, rows);
      if (format === "xlsx" && hasTable) await downloadXLSX(safeTitle, headers, rows);
      if (format === "png" && hasPlot) await downloadPNG(plotRef, safeTitle);
    } catch (e) {
      console.error("Export error:", e);
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
        <button
          onClick={() => handle("png")}
          disabled={!!busy}
          className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600 disabled:opacity-40 transition-colors"
        >
          {busy === "png" ? "…" : "PNG 300dpi"}
        </button>
      )}
    </div>
  );
}
