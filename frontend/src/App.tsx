import "./index.css";
import { Component, useState, type ReactNode } from "react";
import { BarChart2, Table2, FlaskConical, GitMerge, Brain, X, TrendingUp, ClipboardList, Zap, Calculator, Grid3x3, Grid2x2, Shapes, FolderOpen, Target, Filter } from "lucide-react";
import { clearCases } from "./api";

class ErrorBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null };
  static getDerivedStateFromError(e: Error) { return { error: e.message + "\n" + e.stack }; }
  render() {
    if (this.state.error) return (
      <pre className="p-6 text-red-600 text-xs whitespace-pre-wrap bg-white min-h-screen">
        {this.state.error}
      </pre>
    );
    return this.props.children;
  }
}
import { useStore } from "./store";
import UploadZone from "./components/UploadZone";
import DataTable from "./components/DataTable";
import DescriptivePanel from "./components/DescriptivePanel";
import ChartsPanel from "./components/ChartsPanel";
import HypothesisPanel from "./components/HypothesisPanel";
import CorrelationPanel from "./components/CorrelationPanel";
import ModelsPanel from "./components/ModelsPanel";
import VisualModelPanel from "./components/VisualModelPanel";
import ROCPanel from "./components/ROCPanel";
import Table1Panel from "./components/Table1Panel";
import PowerPanel from "./components/PowerPanel";
import ComputePanel from "./components/ComputePanel";
import PSMPanel from "./components/PSMPanel";
import PlotThemeBar from "./components/PlotThemeBar";

const TABS = [
  { id: "data",        label: "Data",        icon: Table2 },
  { id: "summary",     label: "Summary",     icon: BarChart2 },
  { id: "table1",      label: "Table",       icon: ClipboardList },
  { id: "hypothesis",  label: "Hypothesis",  icon: FlaskConical },
  { id: "correlation", label: "Correlation", icon: GitMerge },
  { id: "roc",         label: "ROC",         icon: TrendingUp },
  { id: "models",      label: "Models",      icon: Brain },
  { id: "visual",      label: "Visual",      icon: Shapes },
  { id: "power",       label: "Power",       icon: Zap },
  { id: "compute",     label: "Compute",     icon: Calculator },
  { id: "psm",         label: "PSM",         icon: Target },
  { id: "charts",      label: "Charts",      icon: BarChart2 },
];

/** Save current session preview as CSV and trigger download */
function saveSessionCSV(session: { filename: string; columns: { name: string }[]; preview: Record<string, unknown>[] }) {
  const headers = session.columns.map((c) => c.name);
  const rows = session.preview.map((row) =>
    headers.map((h) => {
      const v = row[h];
      return `"${String(v ?? "").replace(/"/g, '""')}"`;
    }).join(",")
  );
  const csv = [headers.map((h) => `"${h}"`).join(","), ...rows].join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = session.filename.replace(/\.(csv|xlsx|sav|xls)$/i, "") + "_export.csv";
  a.click();
  URL.revokeObjectURL(url);
}

/** Save current session preview as XLSX and trigger download */
async function saveSessionXLSX(session: { filename: string; columns: { name: string }[]; preview: Record<string, unknown>[] }) {
  const XLSX = (await import("xlsx")).default;
  const headers = session.columns.map((c) => c.name);
  const data = [headers, ...session.preview.map((row) => headers.map((h) => row[h] ?? null))];
  const ws = XLSX.utils.aoa_to_sheet(data);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Data");
  XLSX.writeFile(wb, session.filename.replace(/\.(csv|xlsx|sav|xls)$/i, "") + "_export.xlsx");
}

/** Modal asking user to save before opening a new file */
function SaveBeforeOpenModal({
  session,
  onSave,
  onSkip,
  onCancel,
}: {
  session: { filename: string; columns: { name: string }[]; preview: Record<string, unknown>[] };
  onSave: (fmt: "csv" | "xlsx") => void;
  onSkip: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm p-6 space-y-4">
        <h3 className="font-semibold text-gray-900 text-base">Save current dataset?</h3>
        <p className="text-sm text-gray-500">
          Do you want to save <strong>{session.filename}</strong> before opening a new file?
        </p>
        <div className="flex flex-col gap-2">
          <div className="flex gap-2">
            <button
              onClick={() => onSave("csv")}
              className="flex-1 btn-primary text-sm py-2"
            >
              Save as CSV
            </button>
            <button
              onClick={() => onSave("xlsx")}
              className="flex-1 btn-primary text-sm py-2"
            >
              Save as XLSX
            </button>
          </div>
          <button
            onClick={onSkip}
            className="w-full text-sm text-gray-600 border border-gray-200 rounded-lg py-2 hover:bg-gray-50 transition-colors"
          >
            Don't save, open new file
          </button>
          <button
            onClick={onCancel}
            className="w-full text-xs text-gray-400 hover:text-gray-700 py-1"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const { session, activeTab, setActiveTab, clearSession, showGrid, toggleGrid, caseFilter, setCaseFilter } = useStore();
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [saveBusy, setSaveBusy] = useState(false);

  const handleOpenNew = () => setShowSaveModal(true);

  const handleSave = async (fmt: "csv" | "xlsx") => {
    if (!session || saveBusy) return;
    setSaveBusy(true);
    try {
      if (fmt === "csv") saveSessionCSV(session);
      else await saveSessionXLSX(session);
    } finally {
      setSaveBusy(false);
      setShowSaveModal(false);
      clearSession();
    }
  };

  const handleSkip = () => {
    setShowSaveModal(false);
    clearSession();
  };

  const handleCancel = () => setShowSaveModal(false);

  if (!session) return <UploadZone />;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-gray-50">
      {showSaveModal && (
        <SaveBeforeOpenModal
          session={session}
          onSave={handleSave}
          onSkip={handleSkip}
          onCancel={handleCancel}
        />
      )}

      {/* Header — two rows so tabs always have full width */}
      <header className="border-b border-gray-200 bg-white flex-shrink-0 shadow-sm">
        {/* Row 1: logo · filename · actions */}
        <div className="flex items-center gap-3 px-4 pt-2 pb-1.5">
          <div className="flex items-center gap-2 flex-shrink-0">
            <img src="/logo.png" alt="uStat logo" className="w-7 h-7 rounded-lg" />
            <span className="font-bold text-gray-900 text-sm tracking-tight">uStat</span>
          </div>

          <div className="flex items-center gap-1 bg-gray-50 border border-gray-200 rounded-lg px-2 py-1 min-w-0 max-w-xs">
            <span className="text-xs text-gray-600 truncate">{session.filename}</span>
            <span className="text-xs text-gray-400 ml-2 flex-shrink-0">
              {session.rows.toLocaleString()} × {session.columns.length}
            </span>
          </div>

          <div className="ml-auto flex items-center gap-1.5">
            <PlotThemeBar />
            <button
              onClick={toggleGrid}
              className={`p-1.5 rounded-lg transition-colors ${showGrid ? "text-indigo-500 bg-indigo-50 hover:bg-indigo-100" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}
              title={showGrid ? "Hide chart grid lines" : "Show chart grid lines"}
            >
              {showGrid ? <Grid3x3 size={16} /> : <Grid2x2 size={16} />}
            </button>
            <button
              onClick={handleOpenNew}
              className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors"
              title="Open new file"
            >
              <FolderOpen size={16} />
            </button>
            <button
              onClick={handleOpenNew}
              className="p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors"
              title="Close dataset"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Case filter banner */}
        {caseFilter && (
          <div className="flex items-center gap-2 px-4 py-1 bg-violet-50 border-t border-violet-200 text-xs text-violet-700">
            <Filter size={12} className="flex-shrink-0" />
            <span className="font-semibold">{caseFilter.selected.toLocaleString()} of {caseFilter.total.toLocaleString()} cases selected</span>
            <span className="text-violet-400">— all analyses use this subset</span>
            <button
              onClick={async () => {
                if (!session) return;
                await clearCases(session.session_id);
                setCaseFilter(null);
              }}
              className="ml-auto flex items-center gap-1 px-2 py-0.5 rounded bg-violet-200 hover:bg-violet-300 text-violet-800 font-medium transition-colors"
            >
              <X size={10} /> Clear filter
            </button>
          </div>
        )}

        {/* Row 2: tab strip — scrollable so tabs are never clipped */}
        <nav className="flex gap-0.5 px-3 pb-1.5 overflow-x-auto"
          style={{ scrollbarWidth: "none" }}>
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex-shrink-0
                ${activeTab === id
                  ? "bg-indigo-600 text-white"
                  : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"}`}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </nav>
      </header>

      {/* Content */}
      <main className="flex-1 overflow-hidden flex flex-col">
        <ErrorBoundary key={activeTab}>
          {activeTab === "data"        && <div className="flex-1 p-4 overflow-hidden flex flex-col" style={{minHeight:0}}><DataTable /></div>}
          {activeTab === "summary"     && <DescriptivePanel />}
          {activeTab === "table1"      && <Table1Panel />}
          {activeTab === "hypothesis"  && <div className="flex-1 p-4 overflow-y-auto"><HypothesisPanel /></div>}
          {activeTab === "correlation" && <div className="flex-1 p-4 overflow-y-auto"><CorrelationPanel /></div>}
          {activeTab === "roc"         && <ROCPanel />}
          {activeTab === "models"      && <div className="flex-1 p-4 overflow-y-auto"><ModelsPanel /></div>}
          {activeTab === "visual"      && <div className="flex-1 p-4 overflow-y-auto"><VisualModelPanel /></div>}
          {activeTab === "power"       && <div className="flex-1 p-4 overflow-y-auto"><PowerPanel /></div>}
          {activeTab === "compute"     && <div className="flex-1 p-4 overflow-y-auto"><ComputePanel /></div>}
          {activeTab === "psm"         && <div className="flex-1 p-4 overflow-y-auto"><PSMPanel /></div>}
          {activeTab === "charts"      && <div className="flex-1 p-4 overflow-y-auto"><ChartsPanel /></div>}
        </ErrorBoundary>
      </main>
    </div>
  );
}
