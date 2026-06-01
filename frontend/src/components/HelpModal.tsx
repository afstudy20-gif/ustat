import { X, BookOpen, FlaskConical, Brain, Settings, ShieldCheck, HelpCircle, Code, CheckCircle2, Info } from "lucide-react";
import { useState } from "react";

type TabId = "quickstart" | "hypothesis" | "advanced" | "specialized" | "rhub";

export default function HelpModal({ onClose }: { onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<TabId>("quickstart");

  const tabs = [
    { id: "quickstart", label: "Quick Start", icon: BookOpen },
    { id: "hypothesis", label: "Hypothesis Tests", icon: FlaskConical },
    { id: "advanced",   label: "Causal & Regression", icon: Brain },
    { id: "specialized",label: "EFA, Bayes & Meta", icon: Settings },
    { id: "rhub",       label: "R Replication Hub", icon: Code },
  ] as const;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl flex flex-col h-[650px] max-h-[90vh] overflow-hidden animate-in fade-in zoom-in duration-200">

        {/* Header */}
        <div className="bg-slate-900 text-white px-6 py-4 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2">
            <div className="bg-indigo-500 p-1.5 rounded-lg">
              <HelpCircle size={18} className="text-white" />
            </div>
            <div>
              <h2 className="font-bold text-sm tracking-tight">uSTAT Help &amp; Analysis Guide</h2>
              <p className="text-[10px] text-slate-400 font-mono">uSTAT Help Center &amp; Interactive Tutorial</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors p-1 hover:bg-slate-800 rounded-lg cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Inner Content Area */}
        <div className="flex-1 flex min-h-0">

          {/* Sidebar Navigation */}
          <div className="w-56 bg-slate-50 border-r border-slate-200 flex flex-col p-3 gap-1 flex-shrink-0 overflow-y-auto">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-2.5 pb-2">
              Sections
            </p>
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all text-left cursor-pointer ${
                  activeTab === id
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-100"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                }`}
              >
                <Icon size={14} className={activeTab === id ? "text-white" : "text-slate-400"} />
                {label}
              </button>
            ))}

            <div className="mt-auto p-2 bg-indigo-50 rounded-xl border border-indigo-100">
              <p className="text-[10px] font-bold text-indigo-900 uppercase">💡 Tip</p>
              <p className="text-[9px] text-indigo-700 mt-0.5 leading-relaxed">
                Hover the <span className="font-semibold">ⓘ</span> or question-mark icons in any panel to read detailed clinical tips.
              </p>
            </div>
          </div>

          {/* Tab Panels */}
          <div className="flex-1 p-6 overflow-y-auto bg-white font-sans text-xs text-slate-700 space-y-4">

            {activeTab === "quickstart" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🚀 Quick Start &amp; Data Preparation
                  </h3>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">1. Data Upload</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Drag &amp; drop Excel (<span className="font-mono">.xlsx</span>), CSV, SPSS (<span className="font-mono">.sav</span>) or TSV files to upload. SPSS value labels and variable descriptions are read automatically.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">2. Variable Kinds</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        uSTAT classifies variables automatically. To change a variable to <span className="text-indigo-600 font-semibold">Numeric</span> (continuous) or <span className="text-teal-600 font-semibold">Categorical</span>, click the badge next to the column header in the Data tab, or open the <span className="font-semibold">Data Dictionary</span> panel.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">3. Active Filters</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Click the <span className="font-semibold">Filter</span> button in the Data tab to create subsets. When a filter is active every analysis is automatically restricted to that subset (an orange badge appears in the header).
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">4. Compute &amp; Cleaning</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Build new variables (formulas, transforms, recodes, clinical calculators) in the <span className="font-semibold">Compute</span> tab. Data-quality tools — listwise deletion, IQR / Z-score outlier removal, and find &amp; replace — live in the <span className="font-semibold">Missing</span> tab alongside MICE imputation.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "hypothesis" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🧪 Hypothesis &amp; Categorical Tests
                  </h3>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2 bg-indigo-50/50 p-3 rounded-xl border border-indigo-100/50">
                    <Info size={16} className="text-indigo-600 flex-shrink-0 mt-0.5" />
                    <p className="text-indigo-800 leading-relaxed text-[11px]">
                      <strong>Automatic distribution decision (Auto Mode):</strong> uSTAT checks normality automatically. For n ≤ 2000 it uses the Shapiro-Wilk test; for n &gt; 2000, skewness together with the Lilliefors Kolmogorov-Smirnov test. If the distribution is normal it runs parametric tests (t-test, ANOVA); otherwise it switches automatically to non-parametric tests (Mann-Whitney U, Kruskal-Wallis).
                    </p>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">T-test &amp; ANOVA</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Compare two independent groups with the Independent t-test (or Mann-Whitney U), and more than two groups with One-way ANOVA (or Kruskal-Wallis). For repeated measures use the Paired t-test or the Repeated Measures ANOVA tab.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Categorical association tests</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Compare proportions between two categorical variables with the Chi-square test, or Fisher's Exact test for small samples. Use the Cochran-Armitage test to examine dose-response or ordered-category trends.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Hierarchical endpoint testing (Gatekeeping)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Order multiple hypotheses (e.g. primary and secondary endpoints) into families and distribute Bonferroni / Hochberg / Holm weights to control the family-wise error rate (FWER).
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "advanced" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🧠 Regression &amp; Causal Inference
                  </h3>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Regression models &amp; VIF</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Fit Linear (continuous outcomes), Logistic (binary outcomes), Firth penalized logistic (for rare events and separation), Poisson, and Cox PH (survival) models. VIF (Variance Inflation Factor) for multicollinearity is computed automatically on every model.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Propensity Score Matching (PSM)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Balances confounders between treatment and control groups into 1:1 or 1:N matched cohorts. Verify balance with the Love Plot and SMD table. The <span className="font-semibold text-indigo-600">"View &amp; Analyze Matched Cohort"</span> button locks the whole app to the matched patient list; the backtracking button returns you to the original dataset.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Inverse Probability Weighting (IPTW)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Instead of matching, weight each patient by their propensity score (ATE or ATT estimands) to balance the full dataset. Lock the weighted cohort to run weighted, survey-weight-style estimates across all of your analyses.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Decision Curve Analysis (DCA)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        In the DCA tab, evaluate the clinical net benefit of a model (Cox linear predictor or an ML risk score) against "treat all" and "treat none" across threshold probabilities — for both binary and survival outcomes.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "specialized" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    ⚙️ Advanced &amp; Specialized Methods
                  </h3>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Exploratory Factor Analysis &amp; PCA</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Reveal the structure of scale and survey data. Check suitability with KMO and Bartlett's sphericity test, compute factor loadings with Varimax (orthogonal) or Promax (oblique) rotation, and build interactive Scree plots and biplots.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Bayesian hypothesis testing</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Beyond classical p-values, weigh the evidence for the alternative (H₁) vs null (H₀) with the JZS Bayes Factor (BF₁₀ / BF₀₁). Inspect prior (Cauchy) vs posterior density overlays and Savage-Dickey density ratios.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Meta-analysis &amp; publication bias</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Pool studies with fixed- and random-effects models (DerSimonian-Laird &amp; Paule-Mandel). Test for publication bias with Egger's and Begg's tests, and impute potentially missing studies with the Trim-and-Fill method.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "rhub" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    💻 R Replication Hub &amp; Reporting
                  </h3>
                </div>

                <div className="space-y-3 bg-indigo-950 text-indigo-100 p-4 rounded-2xl border border-indigo-900 shadow-lg">
                  <div className="flex gap-2.5 items-start">
                    <Info size={18} className="text-indigo-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-bold text-sm text-white">One-to-one R replication script</h4>
                      <p className="text-xs text-indigo-300 mt-1 leading-relaxed">
                        Every analysis, filter, and model you run in uSTAT is recorded chronologically in the background. Click the R Replication Hub to download a clean, optimized R script (<span className="font-mono text-white bg-indigo-900/60 px-1 py-0.5 rounded">.R file</span>) that reproduces all of your steps exactly in RStudio.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3 pt-2">
                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Publication-ready findings (Methods Appendix)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        When you finish, alongside the R script you can generate an academic Word document (DOCX) for your paper's Methods section — listing the software versions, random seeds, and analysis descriptions used.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

          </div>

        </div>

        {/* Footer */}
        <div className="bg-slate-50 border-t border-slate-200 px-6 py-3 flex items-center justify-between flex-shrink-0 text-[10px] text-slate-500 font-medium">
          <div className="flex items-center gap-1.5">
            <ShieldCheck size={13} className="text-emerald-600" />
            <span>Your data stays in your local browser session and is sent to the server only for computation, anonymously.</span>
          </div>
          <span className="font-semibold text-slate-400">uSTAT v3.1.0 Guide</span>
        </div>

      </div>
    </div>
  );
}
