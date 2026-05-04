import { X } from "lucide-react";

const VERSION = "1.5.0";
const BUILD = 86;

const CHANGELOG = [
  { ver: "1.5.0", date: "2026-04-04", notes: "Ctrl+V paste from Excel/CSV, insert column left/right, copy row/column to clipboard, proprietary license" },
  { ver: "1.4.0", date: "2026-04-03", notes: "Right-click context menu, row/column operations, fill blanks (mean/median/MICE), undo/redo, variable rename, decimal formatting" },
  { ver: "1.3.0", date: "2026-04-02", notes: "Model diagnostics, calibration, decision curve analysis, model comparison, bootstrap CI, permutation tests" },
  { ver: "1.2.0", date: "2026-04-01", notes: "Repeated measures, ANCOVA, two-way ANOVA, contextual guidance panels across all analyses" },
  { ver: "1.1.0", date: "2026-03-28", notes: "Effect sizes with CI, post-hoc testing, violin plots, global palette theme, chart export at 300 DPI" },
  { ver: "1.0.0", date: "2026-03-24", notes: "Initial release with 40+ statistical methods, clinical calculators, Table 1, PSM, power analysis" },
];

export default function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div
        className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="uSTAT" className="w-10 h-10 object-contain" />
            <div>
              <div className="flex items-baseline gap-2">
                <h2 className="text-lg font-bold text-gray-900">uSTAT</h2>
                <span className="text-xs font-mono text-indigo-500 bg-indigo-50 px-1.5 py-0.5 rounded">v{VERSION}</span>
                <span className="text-[10px] text-gray-400">build {BUILD}</span>
              </div>
              <p className="text-xs text-gray-400">Statistical Analysis Platform</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">

          {/* What is uSTAT */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider border-b border-gray-100 pb-1">What is uSTAT?</h3>
            <p className="text-xs text-gray-700 leading-relaxed">
              uSTAT is a free, browser-based statistical analysis platform and SPSS alternative for clinicians, biostatisticians, and medical researchers. Upload CSV, Excel, SPSS, SAS, or Stata files and run the same analyses you would in SPSS, R, or Stata — without any installation.
            </p>
          </div>

          {/* What makes uSTAT different */}
          <div className="bg-indigo-50 rounded-xl p-4 space-y-2">
            <h3 className="text-xs font-bold text-indigo-900 uppercase tracking-wider">What makes uSTAT different</h3>
            <ul className="text-xs text-indigo-800 space-y-1.5 list-none">
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">1.</span><span><strong>Zero-code, browser-based</strong> — no syntax to learn. Point-and-click for every analysis.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">2.</span><span><strong>Free forever</strong> — no account, no paywall, no usage limits.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">3.</span><span><strong>Auto test selection</strong> — automatically picks the correct test based on normality, sample size, and variable type.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">4.</span><span><strong>Built-in clinical calculators</strong> — CHA&#x2082;DS&#x2082;-VASc, GRACE, TIMI, eGFR, H2FPEF, MAGGIC, QTc and more.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">5.</span><span><strong>One-click Table 1</strong> — publication-ready baseline characteristics with automatic p-values and Excel export.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">6.</span><span><strong>40+ statistical methods</strong> — hypothesis tests, regression, survival, ROC, PSM, power analysis, and more.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">7.</span><span><strong>Interactive charts</strong> — zoom, hover, and export at up to 600 DPI for publication.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">8.</span><span><strong>Multi-format I/O</strong> — reads and writes CSV, Excel, SPSS, SAS, and Stata files natively.</span></li>
            </ul>
          </div>

          {/* Features */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider border-b border-gray-100 pb-1">Features</h3>
            <ul className="text-xs text-gray-700 space-y-1.5 list-none pl-3">
              <li><strong>Descriptive statistics</strong> — Mean, median, IQR, normality tests, Q-Q plots</li>
              <li><strong>Hypothesis tests</strong> — t-test, ANOVA, Mann-Whitney, Kruskal-Wallis, chi-square, Fisher&apos;s exact</li>
              <li><strong>Correlation</strong> — Pearson, Spearman, Kendall with significance matrix</li>
              <li><strong>ROC curves</strong> — AUC, sensitivity/specificity, Youden index, comparing curves</li>
              <li><strong>Regression models</strong> — Linear, logistic, Poisson, polynomial, mixed effects</li>
              <li><strong>Survival analysis</strong> — Kaplan-Meier, Cox proportional hazards, Fine-Gray competing risks, landmark</li>
              <li><strong>Table 1</strong> — Publication-ready baseline characteristics with SMD</li>
              <li><strong>Propensity score matching</strong> — Balance diagnostics, treatment effect estimation</li>
              <li><strong>Power analysis</strong> — Sample size calculation, effect size estimation</li>
              <li><strong>Missing data</strong> — MICE multiple imputation</li>
            </ul>
          </div>

          {/* Usage Guide */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider border-b border-gray-100 pb-1">Usage Guide</h3>
            <ol className="text-xs text-gray-700 space-y-1.5 list-decimal pl-5">
              <li><strong>Upload your data</strong> — drop a CSV, Excel, SPSS (.sav), SAS (.sas7bdat), or Stata (.dta) file on the Statistical Analysis tile, or click to browse. Variables are auto-typed (numeric / categorical / date).</li>
              <li><strong>Inspect &amp; clean</strong> — review the data grid, rename columns, recode levels, fill blanks (mean / median / MICE), filter cases, or compute new variables.</li>
              <li><strong>Pick an analysis</strong> — choose from the left sidebar: descriptive statistics, hypothesis tests, correlation, regression, survival, ROC, PSM, Table 1, power analysis, and more.</li>
              <li><strong>Configure</strong> — pick variables, groups, options. uSTAT auto-suggests the right test based on normality, sample size, and variable type.</li>
              <li><strong>Read results</strong> — every output includes effect sizes, confidence intervals, assumption diagnostics, and a plain-English interpretation.</li>
              <li><strong>Export</strong> — download charts at up to 600 DPI, copy tables to Word/Excel, or save the full session as a JSON file to resume later.</li>
              <li><strong>Need power calc only?</strong> — click the Power Analysis tile. No data required.</li>
            </ol>
          </div>

          {/* Packages & Methods */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider border-b border-gray-100 pb-1">Packages &amp; Methods</h3>
            <p className="text-xs text-gray-600 leading-relaxed">
              uSTAT runs on a Python scientific stack. All statistical computations use peer-reviewed, open-source libraries:
            </p>
            <ul className="text-xs text-gray-700 space-y-1 list-none pl-3">
              <li><strong>SciPy</strong> 1.15 — t-tests, non-parametric tests, distributions, optimization</li>
              <li><strong>statsmodels</strong> 0.14 — linear / logistic / Poisson / mixed-effects regression, ANOVA, ANCOVA, GLM</li>
              <li><strong>lifelines</strong> 0.30 — Kaplan-Meier, Cox proportional hazards, Fine-Gray competing risks, AFT</li>
              <li><strong>scikit-learn</strong> 1.6 — ROC / AUC, calibration, propensity score matching, MICE imputation</li>
              <li><strong>pandas</strong> 2.2 / <strong>NumPy</strong> 2.2 — data wrangling, numerical core</li>
              <li><strong>pyreadstat</strong> 1.2 — native I/O for SPSS (.sav), SAS (.sas7bdat), Stata (.dta)</li>
              <li><strong>patsy</strong> 0.5 — R-style formula parsing for model specification</li>
              <li><strong>openpyxl</strong> 3.1 / <strong>xlrd</strong> 2.0 — Excel I/O</li>
              <li><strong>Plotly.js</strong> 3.4 (frontend) — interactive publication-quality charts</li>
              <li><strong>FastAPI</strong> 0.115 + <strong>Uvicorn</strong> 0.34 — backend API</li>
            </ul>
            <p className="text-[10px] text-gray-500 mt-2">
              Methods follow standard references (e.g. Hosmer-Lemeshow for logistic calibration, Schoenfeld residuals for Cox PH, Benjamini-Hochberg for FDR). Source code on request.
            </p>
          </div>

          {/* Privacy */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider border-b border-gray-100 pb-1">Privacy &amp; Data Handling</h3>
            <p className="text-xs text-gray-700 leading-relaxed">
              Your file is sent to our server only to be parsed and held in memory for the duration of your session. It is <strong>never written to disk</strong> and is automatically cleared from memory 30 minutes after you stop using the app. No account, no logs of your data, no permanent storage.
            </p>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mt-2 space-y-1.5">
              <p className="text-xs font-semibold text-amber-900 flex items-center gap-1.5">
                <span aria-hidden="true">⚠️</span> Important — uSTAT does not yet publish a formal privacy policy or data retention statement. Please:
              </p>
              <ul className="text-xs text-amber-800 space-y-1 list-disc pl-5">
                <li>Avoid uploading confidential or personally identifiable information (PII / PHI).</li>
                <li>Anonymize datasets before upload — strip names, MRNs, dates of birth, and free-text identifiers.</li>
                <li>For regulated workflows (HIPAA, GDPR clinical research), contact the developer directly for clarification on data handling and to request a self-hosted or local-only build.</li>
              </ul>
            </div>
            <p className="text-[10px] text-gray-500 mt-2">
              Contact: <a href="mailto:adycovs@gmail.com" className="text-indigo-600 hover:underline">adycovs@gmail.com</a> · A formal privacy policy and self-host option are on the roadmap.
            </p>
          </div>

          {/* Changelog */}
          <div>
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider mb-2 border-b border-gray-100 pb-1">
              Changelog
            </h3>
            <div className="space-y-2">
              {CHANGELOG.map((entry, i) => (
                <div key={entry.ver} className={`flex gap-3 text-xs ${i === 0 ? "text-gray-800" : "text-gray-500"}`}>
                  <div className="flex-shrink-0 w-24 flex items-start gap-1.5">
                    <span className={`font-mono font-semibold ${i === 0 ? "text-indigo-600" : ""}`}>v{entry.ver}</span>
                    {i === 0 && <span className="text-[8px] bg-green-100 text-green-700 px-1 rounded font-semibold">NEW</span>}
                  </div>
                  <span className="flex-shrink-0 text-gray-400 w-20">{entry.date}</span>
                  <span className="leading-relaxed">{entry.notes}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Footer */}
          <div className="text-[10px] text-gray-400 pt-3 border-t border-gray-100 space-y-1">
            <p>Files held in memory only — never written to disk. Auto-cleared 30 min after last activity.</p>
            <p>&copy; 2026 Dr. Yusuf Ho&#x15F;o&#x11F;lu. All rights reserved.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
