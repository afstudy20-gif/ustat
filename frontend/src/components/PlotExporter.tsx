/**
 * PlotExporter – floating toolbar that attaches to any <Plot> chart.
 * Usage:
 *   <div className="relative">
 *     <Plot ref={plotRef} ... />
 *     <PlotExporter plotRef={plotRef} title="My Chart" />
 *   </div>
 *
 * pptxgenjs is loaded lazily (dynamic import) so it never blocks app startup.
 */
import { useState } from "react";
import Plotly from "plotly.js";

interface Props {
  /** React ref pointing to the react-plotly.js Plot element */
  plotRef: React.RefObject<any>;
  /** Chart title used in the downloaded filenames and PPTX slide */
  title?: string;
  /** Extra className applied to the toolbar wrapper */
  className?: string;
}

export default function PlotExporter({ plotRef, title = "chart", className = "" }: Props) {
  const [open, setOpen]     = useState(false);
  const [width, setWidth]   = useState(1200);
  const [height, setHeight] = useState(700);
  const [fmt, setFmt]       = useState<"png" | "svg">("png");
  const [busy, setBusy]     = useState(false);
  const [pptxErr, setPptxErr] = useState("");

  const safeTitle = title.replace(/[^\w\s-]/g, "").replace(/\s+/g, "_").slice(0, 40) || "chart";

  const getEl = (): HTMLElement | null =>
    plotRef.current?.el ?? plotRef.current;

  const downloadImage = async () => {
    const el = getEl();
    if (!el) return;
    setBusy(true);
    try {
      await Plotly.downloadImage(el, {
        format: fmt,
        width,
        height,
        filename: safeTitle,
      });
    } finally {
      setBusy(false);
      setOpen(false);
    }
  };

  const exportPptx = async () => {
    const el = getEl();
    if (!el) return;
    setBusy(true);
    setPptxErr("");
    try {
      // Capture as PNG data URL first (before loading pptxgenjs)
      const imgData = await Plotly.toImage(el, { format: "png", width, height });

      // Dynamic import keeps pptxgenjs (and its Node.js deps) out of the
      // main bundle, so a load failure here never crashes the app.
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore – pptxgenjs CJS types may not resolve cleanly
      const pptxgenMod = await import("pptxgenjs");
      const pptxgen = pptxgenMod.default ?? pptxgenMod;

      const prs = new (pptxgen as any)();
      prs.layout = "LAYOUT_WIDE"; // 13.33" × 7.5"
      const slide = prs.addSlide();
      slide.addText(title, {
        x: 0.4, y: 0.2, w: 12.5, h: 0.5,
        fontSize: 20, bold: true, color: "111827",
      });
      slide.addImage({
        data: imgData,
        x: 0.4, y: 0.85, w: 12.5, h: 6.4,
      });
      slide.addText(`YuStat · ${new Date().toLocaleDateString()}`, {
        x: 0.4, y: 7.1, w: 12.5, h: 0.3,
        fontSize: 9, color: "9ca3af", align: "right",
      });
      await prs.writeFile({ fileName: `${safeTitle}.pptx` });
      setOpen(false);
    } catch (e: any) {
      console.error("PPTX export failed", e);
      setPptxErr("PPTX export is unavailable in this browser.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={`absolute top-2 right-2 z-10 ${className}`}>
      <button
        onClick={() => { setOpen((o) => !o); setPptxErr(""); }}
        className="p-1.5 rounded-lg bg-white/80 border border-gray-200 shadow-sm text-gray-500 hover:text-indigo-600 hover:bg-white hover:border-indigo-200 transition-colors text-xs"
        title="Export chart"
      >
        ↓
      </button>

      {open && (
        <div className="absolute right-0 top-8 bg-white border border-gray-200 rounded-xl shadow-xl p-4 w-60 space-y-3 z-20">
          <p className="text-xs font-semibold text-gray-700">Export Chart</p>

          {/* Size */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-gray-400 block mb-0.5">Width px</label>
              <input type="number" value={width} onChange={(e) => setWidth(+e.target.value)}
                className="select w-full text-xs py-0.5" min={400} max={4000} step={100} />
            </div>
            <div>
              <label className="text-[10px] text-gray-400 block mb-0.5">Height px</label>
              <input type="number" value={height} onChange={(e) => setHeight(+e.target.value)}
                className="select w-full text-xs py-0.5" min={200} max={3000} step={100} />
            </div>
          </div>

          {/* Format toggle */}
          <div>
            <label className="text-[10px] text-gray-400 block mb-1">Format</label>
            <div className="flex rounded overflow-hidden border border-gray-200">
              {(["png", "svg"] as const).map((f) => (
                <button key={f} onClick={() => setFmt(f)}
                  className={`flex-1 text-xs py-1 transition-colors ${fmt === f ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}>
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Download buttons */}
          <button onClick={downloadImage} disabled={busy}
            className="btn-primary w-full text-xs py-1.5">
            {busy ? "Exporting…" : `Download ${fmt.toUpperCase()}`}
          </button>
          <button onClick={exportPptx} disabled={busy}
            className="w-full text-xs py-1.5 rounded-lg border border-indigo-200 text-indigo-600 hover:bg-indigo-50 transition-colors">
            {busy ? "Building…" : "Export to PowerPoint (.pptx)"}
          </button>
          {pptxErr && <p className="text-[10px] text-red-500">{pptxErr}</p>}

          <button onClick={() => setOpen(false)}
            className="w-full text-[10px] text-gray-400 hover:text-gray-700 py-0.5">
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
