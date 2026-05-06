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

interface MethodGroup {
  group: string;
  items: { name: string; impl: string }[];
}

const METHODS: MethodGroup[] = [
  {
    group: "Descriptive & normality",
    items: [
      { name: "Mean / SD / median / IQR", impl: "pandas + NumPy" },
      { name: "Shapiro-Wilk normality", impl: "scipy.stats.shapiro" },
      { name: "Kolmogorov-Smirnov", impl: "scipy.stats.kstest" },
      { name: "Levene homogeneity of variance", impl: "scipy.stats.levene" },
      { name: "Q-Q plot", impl: "scipy.stats.probplot" },
      { name: "Skewness / kurtosis", impl: "scipy.stats.skew / kurtosis" },
    ],
  },
  {
    group: "Hypothesis tests — continuous",
    items: [
      { name: "Independent-samples t-test", impl: "scipy.stats.ttest_ind" },
      { name: "Paired t-test", impl: "scipy.stats.ttest_rel" },
      { name: "One-way ANOVA", impl: "scipy.stats.f_oneway" },
      { name: "Two-way ANOVA / ANCOVA", impl: "statsmodels.formula.api.ols + anova_lm" },
      { name: "Repeated-measures ANOVA / Mixed ANOVA", impl: "statsmodels.formula.api.mixedlm" },
      { name: "Tukey HSD post-hoc", impl: "scipy.stats.tukey_hsd" },
      { name: "Mann-Whitney U", impl: "scipy.stats.mannwhitneyu" },
      { name: "Wilcoxon signed-rank", impl: "scipy.stats.wilcoxon" },
      { name: "Kruskal-Wallis", impl: "scipy.stats.kruskal" },
      { name: "Friedman", impl: "scipy.stats.friedmanchisquare" },
    ],
  },
  {
    group: "Categorical tests",
    items: [
      { name: "Chi-square / Fisher's exact", impl: "scipy.stats.chi2_contingency / fisher_exact" },
      { name: "McNemar / Cochran's Q", impl: "statsmodels.stats.contingency_tables" },
      { name: "Mantel-Haenszel", impl: "statsmodels.stats.contingency_tables.StratifiedTable" },
      { name: "Binomial / one- & two-proportion", impl: "scipy.stats.binomtest, statsmodels.stats.proportion" },
    ],
  },
  {
    group: "Correlation",
    items: [
      { name: "Pearson", impl: "scipy.stats.pearsonr" },
      { name: "Spearman", impl: "scipy.stats.spearmanr" },
      { name: "Kendall's tau", impl: "scipy.stats.kendalltau" },
      { name: "Intraclass correlation (ICC)", impl: "pingouin / statsmodels mixed model" },
      { name: "Cohen's kappa", impl: "scikit-learn cohen_kappa_score" },
    ],
  },
  {
    group: "Regression",
    items: [
      { name: "Linear regression + diagnostics", impl: "statsmodels.formula.api.ols" },
      { name: "Logistic regression (with OR & CI)", impl: "statsmodels.formula.api.logit" },
      { name: "Poisson / negative binomial", impl: "statsmodels GLM (Poisson / NegativeBinomial)" },
      { name: "Polynomial & restricted cubic spline", impl: "patsy + statsmodels.ols" },
      { name: "Linear mixed-effects model", impl: "statsmodels.formula.api.mixedlm" },
      { name: "Gamma regression", impl: "statsmodels GLM (Gamma)" },
    ],
  },
  {
    group: "Survival",
    items: [
      { name: "Kaplan-Meier curves", impl: "lifelines.KaplanMeierFitter" },
      { name: "Log-rank test (multi-group)", impl: "lifelines.statistics.logrank_test / multivariate_logrank_test" },
      { name: "Cox proportional hazards", impl: "lifelines.CoxPHFitter" },
      { name: "Schoenfeld residuals (PH check)", impl: "lifelines diagnostics" },
      { name: "Fine-Gray competing risks", impl: "lifelines (CRC) / custom" },
      { name: "Landmark analysis", impl: "lifelines + custom slicing" },
    ],
  },
  {
    group: "Diagnostic accuracy & prediction",
    items: [
      { name: "ROC curve / AUC / Youden index", impl: "scikit-learn roc_curve, roc_auc_score" },
      { name: "DeLong test for ROC comparison", impl: "custom (Sun & Xu 2014 implementation)" },
      { name: "Calibration plot & Hosmer-Lemeshow", impl: "scikit-learn + custom binning" },
      { name: "Brier score", impl: "scikit-learn brier_score_loss" },
      { name: "Decision curve analysis", impl: "custom (Vickers & Elkin 2006)" },
      { name: "Nomogram", impl: "statsmodels coefs + custom rendering" },
    ],
  },
  {
    group: "Causal & missing data",
    items: [
      { name: "Propensity score matching (PSM)", impl: "scikit-learn LogisticRegression + greedy/optimal match" },
      { name: "Standardized mean differences", impl: "custom" },
      { name: "MICE multiple imputation", impl: "scikit-learn IterativeImputer" },
      { name: "Little's MCAR test", impl: "custom (Little 1988)" },
    ],
  },
  {
    group: "Agreement & reliability",
    items: [
      { name: "Bland-Altman", impl: "custom" },
      { name: "Deming / Passing-Bablok regression", impl: "custom + scipy.stats.theilslopes" },
      { name: "Cronbach's alpha", impl: "custom (covariance-based)" },
    ],
  },
  {
    group: "Power analysis",
    items: [
      { name: "t-test / ANOVA / proportions / correlation power", impl: "statsmodels.stats.power" },
      { name: "Cox / log-rank sample size", impl: "custom (Schoenfeld 1981, Freedman 1982)" },
    ],
  },
  {
    group: "Clinical calculators",
    items: [
      { name: "CHA₂DS₂-VASc, HAS-BLED, GRACE, TIMI", impl: "deterministic formulas" },
      { name: "eGFR (CKD-EPI 2021), QTc (Bazett, Fridericia, Framingham)", impl: "deterministic formulas" },
      { name: "H2FPEF, MAGGIC", impl: "deterministic formulas" },
    ],
  },
];

export default function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div
        className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Header ─────────────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="uSTAT" className="w-10 h-10 object-contain" />
            <div>
              <div className="flex items-baseline gap-2">
                <h2 className="text-lg font-bold text-gray-900">uSTAT</h2>
                <span className="text-xs font-mono text-indigo-500 bg-indigo-50 px-1.5 py-0.5 rounded">v{VERSION}</span>
                <span className="text-[10px] text-gray-400">build {BUILD}</span>
              </div>
              <p className="text-xs text-gray-500">
                Statistical Analysis Platform · by <span className="font-medium text-gray-700">Dr. Yusuf Ho&#x15F;o&#x11F;lu</span>
              </p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">

          {/* ── Validation status (prominent) ──────────────────────────────── */}
          <div className="bg-amber-50 border-l-4 border-amber-500 rounded-r-lg p-3 space-y-1">
            <p className="text-xs font-bold text-amber-900 uppercase tracking-wider">⚠️ Validation status</p>
            <p className="text-xs text-amber-900 leading-relaxed">
              uSTAT has <strong>not yet been validated through peer-reviewed publications</strong>. Independent validation against SPSS, R, and Stata is ongoing. Until validation is published, please verify any clinically or scientifically important result against an established statistics package before reporting.
            </p>
            <p className="text-[10px] text-amber-700">Use at your own discretion. Not a medical device. Not for diagnostic use.</p>
          </div>

          {/* ── What is uSTAT ──────────────────────────────────────────────── */}
          <Section title="What is uSTAT?">
            <p className="text-xs text-gray-700 leading-relaxed">
              uSTAT is a free, browser-based statistical analysis platform — an SPSS / R / Stata alternative for clinicians, biostatisticians, and medical researchers. Created and maintained by <strong>Dr. Yusuf Ho&#x15F;o&#x11F;lu</strong>. Upload CSV, Excel, SPSS (.sav), SAS (.sas7bdat), or Stata (.dta) files and run 40+ analyses in your browser. No installation, no account, no usage limits.
            </p>
          </Section>

          {/* ── Highlights (slim) ──────────────────────────────────────────── */}
          <div className="bg-indigo-50 rounded-xl p-4 space-y-2">
            <h3 className="text-xs font-bold text-indigo-900 uppercase tracking-wider">Highlights</h3>
            <ul className="text-xs text-indigo-800 space-y-1.5 list-none">
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">•</span><span><strong>Zero-code</strong> — point-and-click for every analysis. No syntax to learn.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">•</span><span><strong>Auto test selection</strong> — picks the right test from normality, sample size, and variable type.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">•</span><span><strong>Free forever</strong> — no account, no paywall, no usage limits.</span></li>
              <li className="flex gap-2"><span className="text-indigo-400 flex-shrink-0">•</span><span><strong>Publication-ready output</strong> — Table 1 (AMA), 600 DPI charts, Word/Excel export.</span></li>
            </ul>
          </div>

          {/* ── Tests & Methods (with backing implementations) ─────────────── */}
          <Section title="Statistical Tests & Methods">
            <p className="text-xs text-gray-600 leading-relaxed">
              Every test below is implemented on top of a peer-reviewed open-source library. The right column shows the exact function or class used so results are reproducible.
            </p>
            <div className="space-y-3 mt-2">
              {METHODS.map((g) => (
                <div key={g.group} className="space-y-1">
                  <p className="text-[11px] font-semibold text-indigo-700 uppercase tracking-wider">{g.group}</p>
                  <table className="w-full text-xs">
                    <tbody>
                      {g.items.map((it) => (
                        <tr key={it.name} className="border-b border-gray-50 last:border-0">
                          <td className="py-1 pr-2 text-gray-700 align-top">{it.name}</td>
                          <td className="py-1 text-gray-500 font-mono text-[10px] text-right whitespace-nowrap">{it.impl}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-gray-500 mt-3 leading-relaxed">
              Standard references applied: Hosmer-Lemeshow for logistic calibration · Schoenfeld residuals for Cox PH · Benjamini-Hochberg for FDR · DeLong (1988) for paired ROC comparison · Vickers & Elkin (2006) for decision curve analysis. Source code available on request.
            </p>
          </Section>

          {/* ── Packages ───────────────────────────────────────────────────── */}
          <Section title="Underlying packages">
            <ul className="text-xs text-gray-700 space-y-1 list-none pl-3">
              <li><strong>SciPy</strong> 1.15 · <strong>statsmodels</strong> 0.14 · <strong>lifelines</strong> 0.30 · <strong>scikit-learn</strong> 1.6</li>
              <li><strong>pandas</strong> 2.2 · <strong>NumPy</strong> 2.2 · <strong>patsy</strong> 0.5</li>
              <li><strong>pyreadstat</strong> 1.2 (SPSS / SAS / Stata I/O) · <strong>openpyxl</strong> 3.1 / <strong>xlrd</strong> 2.0 (Excel)</li>
              <li><strong>FastAPI</strong> 0.115 + <strong>Uvicorn</strong> 0.34 (backend) · <strong>React</strong> 19 + <strong>Plotly.js</strong> 3.4 (frontend)</li>
            </ul>
          </Section>

          {/* ── Usage guide ────────────────────────────────────────────────── */}
          <Section title="Usage guide">
            <ol className="text-xs text-gray-700 space-y-1.5 list-decimal pl-5">
              <li><strong>Upload</strong> — drop a CSV / Excel / SPSS / SAS / Stata file on the Statistical Analysis tile. Variables are auto-typed (numeric / categorical / date).</li>
              <li><strong>Inspect & clean</strong> — review the grid, rename columns, recode levels, fill blanks (mean / median / MICE), filter cases, compute new variables.</li>
              <li><strong>Pick an analysis</strong> from the sidebar (descriptive, hypothesis tests, correlation, regression, survival, ROC, PSM, Table 1, power…).</li>
              <li><strong>Configure</strong> — pick variables & groups. uSTAT auto-suggests the correct test from normality, sample size, and variable type.</li>
              <li><strong>Read results</strong> — every output includes effect sizes, CIs, assumption diagnostics, and a plain-English interpretation.</li>
              <li><strong>Export</strong> — 600 DPI charts, Word/Excel tables, full session as JSON to resume later.</li>
              <li><strong>Power-only?</strong> Click the Power Analysis tile. No data required.</li>
            </ol>
          </Section>

          {/* ── Privacy & Data Handling ────────────────────────────────────── */}
          <Section title="Privacy & data handling">
            <p className="text-xs text-gray-700 leading-relaxed">
              Your file is sent to our server only to be parsed and held in memory for the duration of your session. It is <strong>never written to disk</strong> and is automatically cleared from memory 30 minutes after you stop using the app. No account, no logs of your data, no permanent storage.
            </p>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mt-2 space-y-1.5">
              <p className="text-xs font-semibold text-amber-900 flex items-center gap-1.5">
                <span aria-hidden="true">⚠️</span> Important — uSTAT does not yet publish a formal privacy policy. Please:
              </p>
              <ul className="text-xs text-amber-800 space-y-1 list-disc pl-5">
                <li>Avoid uploading confidential or personally identifiable information (PII / PHI).</li>
                <li>Anonymize datasets — strip names, MRNs, dates of birth, free-text identifiers.</li>
                <li>For HIPAA / GDPR-regulated workflows, contact the developer for a self-hosted or local-only build.</li>
              </ul>
            </div>
            <p className="text-[10px] text-gray-500 mt-2">
              Contact: <a href="mailto:adycovs@gmail.com" className="text-indigo-600 hover:underline">adycovs@gmail.com</a> · Formal privacy policy and self-host option are on the roadmap.
            </p>
          </Section>

          {/* ── Changelog ──────────────────────────────────────────────────── */}
          <Section title="Changelog">
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
          </Section>

          {/* ── Footer ─────────────────────────────────────────────────────── */}
          <div className="text-[10px] text-gray-400 pt-3 border-t border-gray-100">
            <p>
              Created by <span className="font-medium text-gray-600">Dr. Yusuf Ho&#x15F;o&#x11F;lu</span>. &copy; 2026. All rights reserved.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider border-b border-gray-100 pb-1">
        {title}
      </h3>
      {children}
    </div>
  );
}
