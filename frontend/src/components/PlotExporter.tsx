/**
 * PlotExporter – floating ↓ button that downloads any Plotly chart as PNG/SVG.
 * Plotly is imported lazily inside the handler so this file adds no extra
 * top-level dependency on plotly.js (the chart component already loads it).
 */
import { useState } from "react";
import { plotlyToTiffBlob, downloadBlob } from "../lib/tiffEncoder";

type ExportFmt = "png" | "svg" | "tiff" | "jpeg";

interface Props {
  plotRef: React.RefObject<any>;
  title?: string;
  className?: string;
}

export default function PlotExporter({ plotRef, title = "chart", className = "" }: Props) {
  const [open, setOpen]     = useState(false);
  const [width, setWidth]   = useState(1200);
  const [height, setHeight] = useState(700);
  const [fmt, setFmt]       = useState<ExportFmt>("png");
  const [dpi, setDpi]       = useState(300);
  const [busy, setBusy]     = useState(false);

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
        // Use Plotly.toImage instead of Plotly.downloadImage — the latter
        // crashes on plotly.js@3 in production builds with "Cannot read
        // properties of undefined (reading 'prototype')" because part of
        // its internal download chain got tree-shaken away. toImage just
        // returns a data URL / Blob, which we hand to an anchor click —
        // the same trustworthy pattern the TIFF and dataset exporters use.
        const mod: any = await import("plotly.js");
        const Plotly: any = mod?.toImage ? mod : mod?.default;
        if (!Plotly?.toImage) {
          throw new Error("plotly.js toImage not available");
        }
        const scale = (fmt === "png" || fmt === "jpeg") ? dpi / 72 : 1;  // SVG vector
        const dataUrl: string = await Plotly.toImage(el, {
          format: fmt,
          width,
          height,
          ...(scale !== 1 ? { scale } : {}),
        });
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
    <div className={`absolute top-2 right-2 z-10 ${className}`}>
      <button
        onClick={() => setOpen(o => !o)}
        className="p-1.5 rounded-lg bg-white/80 border border-gray-200 shadow-sm text-gray-500 hover:text-indigo-600 hover:bg-white hover:border-indigo-200 transition-colors text-xs"
        title="Export chart"
      >
        ↓
      </button>

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
