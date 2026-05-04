import { useCallback, useState } from "react";
import { Upload, Info, Zap, BarChart2, ShieldAlert, ListChecks, Sparkles } from "lucide-react";
import { uploadFile } from "../api";
import api from "../api";
import { useStore } from "../store";
import AboutModal from "./AboutModal";
import PowerPanel from "./PowerPanel";

export default function UploadZone() {
  const setSession = useStore((s) => s.setSession);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAbout, setShowAbout] = useState(false);
  const [mode, setMode] = useState<"home" | "power">("home");

  const handle = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      // Session JSON files → load_session endpoint
      if (file.name.endsWith(".json")) {
        const form = new FormData();
        form.append("file", file);
        const res = await api.post("/api/sessions/load_session", form);
        setSession(res.data);
      } else {
        const res = await uploadFile(file);
        setSession(res.data);
      }
    } catch (e: any) {
      const detail = e.response?.data?.detail;
      const status = e.response?.status;
      const msg = detail
        ? `${detail}`
        : e.message?.includes("Network")
        ? "Cannot connect to backend (localhost:8000). Is it running?"
        : `Upload failed (${status ?? e.message ?? "unknown error"})`;
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [setSession]);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) handle(e.dataTransfer.files[0]);
  };

  // ── Power Analysis Mode ──
  if (mode === "power") {
    return (
      <div className="flex flex-col h-screen bg-gray-50">
        {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200 flex-shrink-0">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="uSTAT" className="w-8 h-8 object-contain" />
            <span className="text-sm font-bold text-gray-800">uSTAT</span>
            <span className="text-xs text-gray-400">·</span>
            <span className="text-xs text-indigo-600 font-medium">Power Analysis</span>
          </div>
          <button onClick={() => setMode("home")}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-indigo-600 border border-gray-300 rounded-lg px-3 py-1.5 hover:border-indigo-300 transition-colors">
            <BarChart2 size={14} />
            Statistical Analysis
          </button>
        </header>
        {/* Power Panel */}
        <main className="flex-1 overflow-y-auto p-4">
          <PowerPanel />
        </main>
        <footer className="text-center py-2 border-t border-gray-100 bg-white">
          <p className="text-[11px] text-gray-300">&copy; 2026 Dr. Yusuf Ho&#x15F;o&#x11F;lu. All rights reserved.</p>
        </footer>
      </div>
    );
  }

  // ── Home / Upload Mode ──
  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-8 p-8 bg-gray-50">
      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
      <div className="flex flex-col items-center gap-3">
        <img src="/logo.png" alt="uSTAT logo" className="w-32 h-32 object-contain drop-shadow-md" />
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 leading-tight">uSTAT</h1>
          <p className="text-sm text-gray-400 leading-none mt-1">Statistical Analysis Platform</p>
        </div>
      </div>

      {/* Mode selector — symmetric tiles */}
      <div className="grid grid-cols-2 gap-3 w-full max-w-2xl">
        {/* Statistical Analysis = drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById("file-input")?.click()}
          className={`flex flex-col items-center justify-center gap-3 px-4 py-8 rounded-xl border-2 border-dashed cursor-pointer transition-colors min-h-[220px]
            ${dragging
              ? "border-indigo-500 bg-indigo-100"
              : "border-indigo-400 bg-indigo-50 hover:border-indigo-500 hover:bg-indigo-100"}`}
        >
          <div className="flex items-center gap-2 text-indigo-700">
            <BarChart2 size={22} />
            <span className="text-base font-semibold">Statistical Analysis</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <Upload size={20} className="text-indigo-400" />
            <p className="text-sm text-indigo-700 font-medium">Drop your data file here</p>
            <p className="text-xs text-indigo-400">or click to browse</p>
            <p className="text-[10px] text-indigo-300 mt-1 text-center px-2">CSV · Excel · SAS · SPSS · Stata · Session JSON</p>
          </div>
          <input
            id="file-input"
            type="file"
            className="hidden"
            accept=".csv,.xlsx,.xls,.sas7bdat,.sav,.dta,.json"
            onChange={(e) => e.target.files?.[0] && handle(e.target.files[0])}
          />
        </div>

        {/* Power Analysis — separate, equal size */}
        <button
          onClick={() => setMode("power")}
          className="flex flex-col items-center justify-center gap-3 px-4 py-8 rounded-xl border-2 border-gray-200 bg-white text-gray-600 hover:border-amber-400 hover:bg-amber-50 hover:text-amber-700 transition-colors min-h-[220px]"
        >
          <div className="flex items-center gap-2">
            <Zap size={22} />
            <span className="text-base font-semibold">Power Analysis</span>
          </div>
          <span className="text-xs text-gray-400">No data needed</span>
        </button>
      </div>

      {/* Quick facts — privacy, scope, cost */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 w-full max-w-2xl text-xs">
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
          <ShieldAlert size={14} className="text-amber-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-gray-700">Privacy</p>
            <p className="text-gray-500 leading-snug">Files held in memory only — never written to disk. Cleared 30 min after you stop using the app.</p>
          </div>
        </div>
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
          <ListChecks size={14} className="text-indigo-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-gray-700">Scope</p>
            <p className="text-gray-500 leading-snug">t-tests, ANOVA, regression, non-parametric, survival, power &amp; more.</p>
          </div>
        </div>
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
          <Sparkles size={14} className="text-emerald-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-gray-700">Cost</p>
            <p className="text-gray-500 leading-snug">Free to use. No account, no paywall.</p>
          </div>
        </div>
      </div>

      {loading && <p className="text-indigo-600 animate-pulse">Opening and parsing your data…</p>}
      {error && <p className="text-red-500 text-sm">{error}</p>}

      <button
        onClick={() => setShowAbout(true)}
        className="flex items-center gap-1.5 text-gray-400 hover:text-indigo-600 text-xs transition-colors"
      >
        <Info size={14} />
        About uSTAT — packages & methods
      </button>

      <p className="text-[11px] text-gray-300 mt-2">&copy; 2026 Dr. Yusuf Ho&#x15F;o&#x11F;lu. All rights reserved.</p>
    </div>
  );
}
