import { useCallback, useState } from "react";
import { Upload, Info, Zap, BarChart2, ShieldAlert, ListChecks, Sparkles, NotebookPen, FileText, HeartPulse, Workflow, Layers, HelpCircle } from "lucide-react";
import { uploadFile } from "../api";
import api from "../api";
import { useStore } from "../store";
import AboutModal from "./AboutModal";
import HelpModal from "./HelpModal";
import RecentSessionsPanel from "./RecentSessionsPanel";
import PowerPanel from "./PowerPanel";
import RefreshAppButton from "./RefreshAppButton";

export default function UploadZone() {
  const setSession = useStore((s) => s.setSession);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAbout, setShowAbout] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [mode, setMode] = useState<"home" | "power">("home");

  const handle = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      // Session JSON files → load_session endpoint
      const isSessionJson = file.name.endsWith(".json");
      if (isSessionJson) {
        const form = new FormData();
        form.append("file", file);
        const res = await api.post("/api/sessions/load_session", form);
        setSession(res.data);
        // Restored sessions carry server-side decimal overrides — pull them
        // into the store so the table renders with the user's formatting.
        // The setSession call above reset columnDecimals because session_id
        // flipped; this re-hydrates it from the backend snapshot.
        try {
          const { getColumnDecimalsApi } = await import("../api");
          const dres = await getColumnDecimalsApi(res.data.session_id);
          if (dres.data && Object.keys(dres.data).length > 0) {
            const { useStore: store } = await import("../store");
            store.setState({ columnDecimals: dres.data });
          }
        } catch { /* non-fatal — fall back to defaults */ }
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
        {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200 flex-shrink-0">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="uSTAT" className="w-8 h-8 object-contain" />
            <span className="text-sm font-bold text-gray-800">uSTAT</span>
            <span className="text-xs text-gray-400">·</span>
            <span className="text-xs text-indigo-600 font-medium">Power Analysis</span>
          </div>
          <div className="flex items-center gap-2">
            <RefreshAppButton />
            <button onClick={() => setShowHelp(true)}
              title="Open Help & Analysis Guide"
              className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-indigo-600 border border-gray-300 rounded-lg px-3 py-1.5 hover:border-indigo-300 transition-colors">
              <HelpCircle size={14} />
              Help &amp; Guide
            </button>
            <button onClick={() => setMode("home")}
              className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-indigo-600 border border-gray-300 rounded-lg px-3 py-1.5 hover:border-indigo-300 transition-colors">
              <BarChart2 size={14} />
              Statistical Analysis
            </button>
          </div>
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
      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
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

      {/* Recent sessions (auto-saved in IndexedDB) — only renders when
          the user has at least one saved snapshot. Lets them resume
          exactly where they left off without re-uploading the dataset. */}
      <RecentSessionsPanel />

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

      {/* ── Other drtr.uk apps ─────────────────────────────────────────── */}
      <div className="w-full max-w-2xl">
        <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider text-center mb-2">
          More tools by Dr. Yusuf Ho&#x15F;o&#x11F;lu
        </p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          {[
            { url: "https://not.drtr.uk",    Icon: NotebookPen, name: "Notepad",    desc: "Local-only text editor" },
            { url: "https://pdf.drtr.uk/",   Icon: FileText,    name: "PDF",        desc: "Annotate &amp; sign in browser" },
            { url: "https://ecgcal.drtr.uk/",Icon: HeartPulse,  name: "ECG Caliper",desc: "Digital ECG wave analyzer" },
            { url: "https://neodw.drtr.uk/", Icon: Layers,      name: "NeoDW",      desc: "DICOM workstation (any modality)" },
            { url: "https://flow.drtr.uk",   Icon: Workflow,    name: "AcademicFlow", desc: "Journal flowchart designer" },
          ].map(({ url, Icon, name, desc }) => (
            <a
              key={url}
              href={url}
              target="_blank"
              rel="noreferrer"
              className="flex flex-col items-center gap-1 px-2 py-2 rounded-lg border border-gray-200 bg-white hover:border-indigo-300 hover:bg-indigo-50 transition-colors text-center group"
            >
              <Icon size={18} className="text-indigo-500 group-hover:text-indigo-600" />
              <span className="text-xs font-semibold text-gray-700 group-hover:text-indigo-700">{name}</span>
              <span
                className="text-[9px] text-gray-400 leading-snug"
                dangerouslySetInnerHTML={{ __html: desc }}
              />
            </a>
          ))}
        </div>
      </div>

      {loading && <p className="text-indigo-600 animate-pulse">Opening and parsing your data…</p>}
      {error && <p className="text-red-500 text-sm">{error}</p>}

      <div className="flex items-center gap-4 flex-wrap justify-center">
        <button
          onClick={() => setShowHelp(true)}
          className="flex items-center gap-1.5 text-indigo-600 hover:text-indigo-800 text-xs font-semibold transition-colors border border-indigo-200 hover:border-indigo-400 bg-indigo-50/60 hover:bg-indigo-100 rounded-full px-3 py-1.5"
          title="7-section walkthrough: Quick Start, Tests, Regression, Causal, Prediction & Validation, EFA/Bayes/Meta, R Hub"
        >
          <HelpCircle size={14} />
          Help &amp; Analysis Guide
        </button>
        <button
          onClick={() => setShowAbout(true)}
          className="flex items-center gap-1.5 text-gray-400 hover:text-indigo-600 text-xs transition-colors"
        >
          <Info size={14} />
          About uSTAT — packages &amp; methods
        </button>
        <RefreshAppButton variant="inline" />
      </div>

      <p className="text-[11px] text-gray-300 mt-2">&copy; 2026 Dr. Yusuf Ho&#x15F;o&#x11F;lu. All rights reserved.</p>
      <div className="flex flex-wrap items-center gap-2 text-[10px] text-gray-400 mt-1">
        <a href="/privacy" target="_blank" rel="noreferrer" className="hover:text-indigo-500">Privacy</a>
        <span className="text-gray-200">·</span>
        <a href="/terms" target="_blank" rel="noreferrer" className="hover:text-indigo-500">Terms</a>
        <span className="text-gray-200">·</span>
        <a href="/security" target="_blank" rel="noreferrer" className="hover:text-indigo-500">Security</a>
        <span className="text-gray-200">·</span>
        <a href="https://github.com/afstudy20-gif/wiz3" target="_blank" rel="noreferrer" className="hover:text-indigo-500">Source</a>
      </div>

    </div>
  );
}
