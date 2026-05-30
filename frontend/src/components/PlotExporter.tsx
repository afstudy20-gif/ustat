/**
 * PlotExporter – floating ↓ button that downloads any Plotly chart as PNG/SVG.
 * Plotly is imported lazily inside the handler so this file adds no extra
 * top-level dependency on plotly.js (the chart component already loads it).
 */
import { useState, useEffect } from "react";
import { plotlyToTiffBlob, downloadBlob } from "../lib/tiffEncoder";
import { plotlyToDataUrl } from "../lib/plotlyExport";

type ExportFmt = "png" | "svg" | "tiff" | "jpeg";

interface Props {
  plotRef: React.RefObject<any>;
  title?: string;
  className?: string;
  /** Seed the export width/height from the on-screen chart size so a chart
   *  the user has shrunk exports at that same (smaller) size. */
  defaultWidth?: number;
  defaultHeight?: number;
}

export default function PlotExporter({ plotRef, title = "chart", className = "", defaultWidth, defaultHeight }: Props) {
  const [open, setOpen]       = useState(false);
  const [width, setWidth]     = useState(Math.round(defaultWidth ?? 1200));
  const [height, setHeight]   = useState(Math.round(defaultHeight ?? 700));
  // Keep export dimensions in sync with the live chart size while the popover
  // is closed; don't clobber a width/height the user is editing in the popover.
  useEffect(() => {
    if (open) return;
    if (defaultWidth)  setWidth(Math.round(defaultWidth));
    if (defaultHeight) setHeight(Math.round(defaultHeight));
  }, [defaultWidth, defaultHeight, open]);
  const [fmt, setFmt]         = useState<ExportFmt>("png");
  const [dpi, setDpi]         = useState(300);
  const [busy, setBusy]       = useState(false);
  // "Copied to clipboard" pill is shown for ~1.5s on success so the user
  // doesn't have to guess whether the click did anything.
  const [copyToast, setCopyToast] = useState<string | null>(null);

  const safeTitle = title.replace(/[^\w\s-]/g, "").replace(/\s+/g, "_").slice(0, 40) || "chart";
  const getEl = (): HTMLElement | null => {
    const ref = plotRef.current;
    if (!ref) return null;
    // react-plotly.js component instance → .el
    if (ref.el) return ref.el;
    // Direct DOM element with .data (already a plotly div)
    if (ref.data) return ref;
    // Wrapper div → find the plotly div inside
    if (ref instanceof HTMLElement) {
      const plotlyDiv = ref.querySelector(".js-plotly-plot") ?? ref.querySelector("[class*='plotly']");
      if (plotlyDiv) return plotlyDiv as HTMLElement;
      // If the ref IS the container, find first child div with data
      for (const child of ref.querySelectorAll("div")) {
        if ((child as any).data) return child as HTMLElement;
      }
      return ref;
    }
    return null;
  };

  /** Render the chart to a PNG blob at the current width/height/dpi. */
  const renderPngBlob = async (): Promise<Blob | null> => {
    const el = getEl();
    if (!el) return null;
    const dataUrl = await plotlyToDataUrl(el, { format: "png", width, height, scale: dpi / 72 });
    const res = await fetch(dataUrl);
    return await res.blob();
  };

  /** Copy the rendered chart to the system clipboard as a PNG image. */
  const copyImage = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const blob = await renderPngBlob();
      if (!blob) throw new Error("plot is not mounted yet");
      // ClipboardItem + write is supported in modern Chromium, Safari 13.4+,
      // and Firefox 127+. The user click is the gesture required by Safari.
      if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
        throw new Error("Clipboard API not available in this browser");
      }
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
      setCopyToast("Copied to clipboard");
      setTimeout(() => setCopyToast(null), 1500);
    } catch (e: unknown) {
      console.error("PlotExporter copy failed:", e);
      alert(`Copy chart failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const downloadImage = async () => {
    const el = getEl();
    if (!el) return;
    setBusy(true);
    try {
      if (fmt === "tiff") {
        // Plotly cannot emit TIFF natively — rasterise to PNG at the
        // requested DPI, then encode an uncompressed baseline RGB TIFF.
        const blob = await plotlyToTiffBlob(el, { width, height, dpi });
        downloadBlob(blob, `${safeTitle}.tiff`);
      } else {
        // Rasterise through the shared Plotly export helper: it reuses the
        // single statically-imported bundle (no flaky dynamic import) and
        // re-renders with a guaranteed-available font so axis/tick/annotation
        // text survives the SVG→bitmap step. PNG/JPEG honour DPI; SVG stays
        // vector.
        const scale = (fmt === "png" || fmt === "jpeg") ? dpi / 72 : 1;
        const dataUrl: string = await plotlyToDataUrl(el, { format: fmt, width, height, scale });
        // Data URL → Blob → anchor click. Direct anchor.href = dataUrl works
        // for small charts but Safari truncates very large data URLs; round
        // through a Blob so any chart size downloads cleanly.
        const res = await fetch(dataUrl);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${safeTitle}.${fmt}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    } catch (e: unknown) {
      console.error("PlotExporter download failed:", e);
      alert(`Chart export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
      setOpen(false);
    }
  };

  return (
    <div className={`absolute top-2 right-2 z-10 flex gap-1 ${className}`}>
      <button
        onClick={copyImage}
        disabled={busy}
        className="p-1.5 rounded-lg bg-white/80 border border-gray-200 shadow-sm text-gray-500 hover:text-emerald-600 hover:bg-white hover:border-emerald-200 transition-colors text-xs disabled:opacity-50"
        title="Copy chart to clipboard as PNG"
      >
        ⧉
      </button>
      <button
        onClick={() => setOpen(o => !o)}
        className="p-1.5 rounded-lg bg-white/80 border border-gray-200 shadow-sm text-gray-500 hover:text-indigo-600 hover:bg-white hover:border-indigo-200 transition-colors text-xs"
        title="Export chart"
      >
        ↓
      </button>
      {copyToast && (
        <span className="absolute -bottom-7 right-0 text-[10px] font-medium px-2 py-1 rounded bg-emerald-600 text-white shadow whitespace-nowrap">
          {copyToast}
        </span>
      )}

      {open && (
        <div className="absolute right-0 top-8 bg-white border border-gray-200 rounded-xl shadow-xl p-4 w-52 space-y-3 z-20">
          <p className="text-xs font-semibold text-gray-700">Export Chart</p>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-gray-400 block mb-0.5">Width px</label>
              <input type="number" value={width} onChange={e => setWidth(+e.target.value)}
                className="select w-full text-xs py-0.5" min={400} max={4000} step={100} />
            </div>
            <div>
              <label className="text-[10px] text-gray-400 block mb-0.5">Height px</label>
              <input type="number" value={height} onChange={e => setHeight(+e.target.value)}
                className="select w-full text-xs py-0.5" min={200} max={3000} step={100} />
            </div>
          </div>

          <div className="grid grid-cols-4 gap-0 rounded overflow-hidden border border-gray-200">
            {(["png", "tiff", "jpeg", "svg"] as const).map(f => (
              <button key={f} onClick={() => setFmt(f)}
                className={`text-xs py-1 transition-colors ${fmt === f ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}>
                {f.toUpperCase()}
              </button>
            ))}
          </div>

          {(fmt === "png" || fmt === "tiff" || fmt === "jpeg") && (
            <div>
              <label className="text-[10px] text-gray-400 block mb-0.5">DPI (resolution)</label>
              <div className="flex rounded overflow-hidden border border-gray-200">
                {[150, 300, 600].map(d => (
                  <button key={d} onClick={() => setDpi(d)}
                    className={`flex-1 text-xs py-1 transition-colors ${dpi === d ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}>
                    {d}
                  </button>
                ))}
              </div>
              {fmt === "tiff" && (
                <p className="text-[9px] text-gray-400 mt-1 leading-tight">
                  Uncompressed RGB baseline TIFF. Journal-ready; file sizes are larger than PNG.
                </p>
              )}
            </div>
          )}

          <button onClick={downloadImage} disabled={busy} className="btn-primary w-full text-xs py-1.5">
            {busy ? "Exporting…" : `Download ${fmt.toUpperCase()}`}
          </button>

          <button onClick={() => setOpen(false)} className="w-full text-[10px] text-gray-400 hover:text-gray-700 py-0.5">
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
