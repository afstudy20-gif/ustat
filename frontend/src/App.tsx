import "./index.css";
import { Component, type ReactNode } from "react";
import { FileSpreadsheet, BarChart2, Table2, FlaskConical, GitMerge, Brain, X, TrendingUp, ClipboardList, Zap, Calculator } from "lucide-react";

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
import ROCPanel from "./components/ROCPanel";
import Table1Panel from "./components/Table1Panel";
import PowerPanel from "./components/PowerPanel";
import ComputePanel from "./components/ComputePanel";

const TABS = [
  { id: "data",        label: "Data",        icon: Table2 },
  { id: "summary",     label: "Summary",     icon: BarChart2 },
  { id: "table1",      label: "Table 1",     icon: ClipboardList },
  { id: "hypothesis",  label: "Hypothesis",  icon: FlaskConical },
  { id: "correlation", label: "Correlation", icon: GitMerge },
  { id: "roc",         label: "ROC",         icon: TrendingUp },
  { id: "models",      label: "Models",      icon: Brain },
  { id: "power",       label: "Power",       icon: Zap },
  { id: "compute",     label: "Compute",     icon: Calculator },
  { id: "charts",      label: "Charts",      icon: BarChart2 },
];

export default function App() {
  const { session, activeTab, setActiveTab, clearSession } = useStore();

  if (!session) return <UploadZone />;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-gray-50">
      {/* Header */}
      <header className="flex items-center gap-4 px-4 py-2.5 border-b border-gray-200 bg-white flex-shrink-0 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-indigo-600 rounded-lg flex items-center justify-center">
            <FileSpreadsheet size={14} className="text-white" />
          </div>
          <span className="font-bold text-gray-900 text-sm tracking-tight">YuStat</span>
        </div>

        <div className="flex items-center gap-1 bg-gray-50 border border-gray-200 rounded-lg px-2 py-1.5 max-w-xs">
          <span className="text-xs text-gray-600 truncate">{session.filename}</span>
          <span className="text-xs text-gray-400 ml-auto flex-shrink-0">
            {session.rows.toLocaleString()} × {session.columns.length}
          </span>
        </div>

        <nav className="flex gap-0.5 flex-1">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
                ${activeTab === id
                  ? "bg-indigo-600 text-white"
                  : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"}`}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </nav>

        <button
          onClick={clearSession}
          className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors"
          title="Close dataset"
        >
          <X size={16} />
        </button>
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
          {activeTab === "power"       && <div className="flex-1 p-4 overflow-y-auto"><PowerPanel /></div>}
          {activeTab === "compute"     && <div className="flex-1 p-4 overflow-y-auto"><ComputePanel /></div>}
          {activeTab === "charts"      && <div className="flex-1 p-4 overflow-y-auto"><ChartsPanel /></div>}
        </ErrorBoundary>
      </main>
    </div>
  );
}
