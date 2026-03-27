import "./index.css";
import { Component, useState, type ReactNode } from "react";
import { BarChart2, Table2, FlaskConical, GitMerge, Brain, X, TrendingUp, ClipboardList, Zap, Calculator, Grid3x3, Grid2x2, Shapes, FolderOpen, Target, Filter, Info } from "lucide-react";
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

/** Download file via hidden iframe — most reliable cross-platform method */
function triggerDownload(sessionId: string, format: "csv" | "xlsx", originalFilename: string) {
  const outName = originalFilename.replace(/\.(csv|xlsx|sav|xls|sas7bdat|dta)$/i, "") + `_export.${format}`;
  const url = `/api/sessions/${sessionId}/export/${format}?filename=${encodeURIComponent(outName)}`;

  // Hidden iframe approach: browser downloads file without navigating away
  let iframe = document.getElementById("download-iframe") as HTMLIFrameElement | null;
  if (!iframe) {
    iframe = document.createElement("iframe");
    iframe.id = "download-iframe";
    iframe.style.display = "none";
    document.body.appendChild(iframe);
  }
  iframe.src = url;
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

const ABOUT_SECTIONS = [
  {
    title: "Hypothesis Testing",
    items: [
      ["Independent t-test", "scipy.stats.ttest_ind"],
      ["Paired t-test", "scipy.stats.ttest_rel"],
      ["Mann-Whitney U", "scipy.stats.mannwhitneyu"],
      ["Wilcoxon signed-rank", "scipy.stats.wilcoxon"],
      ["One-way ANOVA", "scipy.stats.f_oneway"],
      ["Kruskal-Wallis H", "scipy.stats.kruskal"],
      ["Chi-square test", "scipy.stats.chi2_contingency"],
      ["Fisher's exact test", "scipy.stats.fisher_exact"],
    ],
  },
  {
    title: "Normality & Diagnostics",
    items: [
      ["Shapiro-Wilk (n \u2264 2000)", "scipy.stats.shapiro"],
      ["Lilliefors (n > 2000)", "statsmodels.stats.diagnostic.lilliefors"],
      ["Skewness (CLT bypass)", "scipy.stats.skew"],
      ["Levene's test", "scipy.stats.levene"],
    ],
  },
  {
    title: "Correlation & Agreement",
    items: [
      ["Pearson r", "scipy.stats.pearsonr"],
      ["Spearman \u03C1", "scipy.stats.spearmanr"],
      ["ICC", "statsmodels (mixed model)"],
      ["Cohen's Kappa", "sklearn.metrics.cohen_kappa_score"],
    ],
  },
  {
    title: "Regression Models",
    items: [
      ["Linear (OLS)", "statsmodels.api.OLS"],
      ["Logistic", "statsmodels.api.Logit"],
      ["Poisson", "statsmodels.genmod.GLM (Poisson)"],
      ["Negative Binomial", "statsmodels.genmod.GLM (NegBin)"],
      ["Gamma", "statsmodels.genmod.GLM (Gamma)"],
      ["Polynomial", "statsmodels.api.OLS (poly terms)"],
      ["RCS dose-response", "statsmodels.api.Logit + custom spline basis"],
    ],
  },
  {
    title: "Mixed Models & GEE",
    items: [
      ["Linear Mixed Model (LMM)", "statsmodels.formula.api.mixedlm"],
      ["GEE (binary clustered)", "statsmodels.genmod.GEE (Binomial)"],
    ],
  },
  {
    title: "Survival Analysis",
    items: [
      ["Kaplan-Meier", "lifelines.KaplanMeierFitter"],
      ["Log-rank test", "lifelines.statistics.logrank_test"],
      ["Cox Proportional Hazards", "lifelines.CoxPHFitter"],
    ],
  },
  {
    title: "ROC & Diagnostic Accuracy",
    items: [
      ["ROC curve & AUC", "sklearn.metrics.roc_curve, roc_auc_score"],
      ["Youden's index / optimal cut-off", "numpy (argmax J = sens + spec - 1)"],
      ["DeLong test (AUC comparison)", "Custom implementation (Mann-Whitney placements)"],
      ["95% CI of AUC", "DeLong variance-covariance matrix"],
    ],
  },
  {
    title: "Power Analysis",
    items: [
      ["Two-sample t-test", "statsmodels.stats.power.TTestIndPower"],
      ["Paired t-test", "statsmodels.stats.power.TTestPower"],
      ["ANOVA (F-test)", "statsmodels.stats.power.FTestAnovaPower"],
      ["Chi-square", "statsmodels.stats.power.GofChisquarePower"],
      ["Correlation", "statsmodels.stats.power.NormalIndPower"],
      ["Survival (log-rank)", "scipy.stats.norm (Schoenfeld formula)"],
    ],
  },
  {
    title: "Propensity Score Matching",
    items: [
      ["Logistic PS model", "sklearn.linear_model.LogisticRegression"],
      ["Nearest-neighbor matching", "sklearn.neighbors.NearestNeighbors (KD-tree)"],
      ["SMD (balance check)", "Custom (Austin 2011 formula)"],
      ["Outcome analysis", "statsmodels.api.Logit (matched cohort)"],
    ],
  },
  {
    title: "Data I/O",
    items: [
      ["CSV", "pandas.read_csv"],
      ["Excel (.xlsx/.xls)", "pandas.read_excel (openpyxl)"],
      ["SPSS (.sav)", "pyreadstat.read_sav"],
      ["SAS (.sas7bdat)", "pyreadstat.read_sas7bdat"],
      ["Stata (.dta)", "pyreadstat.read_dta"],
      ["Export XLSX", "openpyxl"],
      ["Export SPSS", "pyreadstat.write_sav"],
    ],
  },
  {
    title: "Compute & Transform",
    items: [
      ["Formula engine", "pandas.DataFrame.eval"],
      ["Recode (IF-THEN)", "numpy.select"],
      ["Tertile / Quartile", "pandas.qcut"],
      ["Z-score, Ln, Log, Sqrt, etc.", "numpy / scipy"],
      ["Clinical scores (15+)", "Custom implementations (BMI, eGFR, CHA\u2082DS\u2082-VASc, GRACE, etc.)"],
    ],
  },
  {
    title: "Visualization",
    items: [
      ["All interactive charts", "Plotly.js (react-plotly.js)"],
      ["Frontend framework", "React 18 + TypeScript + Vite"],
      ["Styling", "Tailwind CSS"],
      ["Backend framework", "FastAPI + Uvicorn"],
    ],
  },
];

function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div
        className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="uSTAT" className="w-10 h-10 object-contain" />
            <div>
              <h2 className="text-lg font-bold text-gray-900">uSTAT</h2>
              <p className="text-xs text-gray-400">Statistical Analysis Platform</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
          <p className="text-sm text-gray-600">
            uSTAT is built on open-source Python and JavaScript libraries.
            Below is a reference of the statistical methods and the packages that power each analysis.
          </p>

          {ABOUT_SECTIONS.map((section) => (
            <div key={section.title}>
              <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider mb-2 border-b border-gray-100 pb-1">
                {section.title}
              </h3>
              <div className="grid grid-cols-2 gap-x-6 gap-y-0.5">
                {section.items.map(([method, pkg]) => (
                  <div key={method} className="flex items-baseline gap-2 py-0.5">
                    <span className="text-xs text-gray-700">{method}</span>
                    <span className="flex-1 border-b border-dotted border-gray-200" />
                    <code className="text-[10px] text-indigo-600 font-mono whitespace-nowrap">{pkg}</code>
                  </div>
                ))}
              </div>
            </div>
          ))}

          <div className="text-[10px] text-gray-400 pt-2 border-t border-gray-100">
            All computations run server-side via FastAPI. Data never leaves your machine.
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const { session, activeTab, setActiveTab, clearSession, showGrid, toggleGrid, caseFilter, setCaseFilter } = useStore();
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showAbout, setShowAbout] = useState(false);

  const handleOpenNew = () => setShowSaveModal(true);

  const handleSave = (fmt: "csv" | "xlsx") => {
    if (!session) return;
    // Trigger download synchronously — no async, no popup blocker issues
    triggerDownload(session.session_id, fmt, session.filename);
    // Give browser time to start the download before clearing
    setTimeout(() => {
      setShowSaveModal(false);
      clearSession();
    }, 3000);
  };

  const handleSkip = () => {
    setShowSaveModal(false);
    clearSession();
  };

  const handleCancel = () => setShowSaveModal(false);

  if (!session) return <UploadZone />;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-gray-50">
      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
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
            <img src="/logo.png" alt="uSTAT logo" className="w-7 h-7 rounded-lg" />
            <span className="font-bold text-gray-900 text-sm tracking-tight">uSTAT</span>
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
              onClick={() => setShowAbout(true)}
              className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors"
              title="About uSTAT — packages & methods"
            >
              <Info size={16} />
            </button>
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
