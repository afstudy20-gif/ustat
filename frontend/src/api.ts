import axios from "axios";

const api = axios.create({ baseURL: "" });  // Vite proxy: /api → localhost:8000

export default api;

export const uploadFile = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/api/upload/", form);
};

export const getDescriptive = (sessionId: string, column?: string) =>
  api.get(`/api/stats/${sessionId}/descriptive`, { params: column ? { column } : {} });

export const getFrequency = (sessionId: string, column: string) =>
  api.get(`/api/stats/${sessionId}/frequency`, { params: { column } });

export const getCorrelation = (sessionId: string, method = "pearson") =>
  api.get(`/api/stats/${sessionId}/correlation`, { params: { method } });

export const runTTest = (data: object) => api.post("/api/stats/ttest", data);
export const runChiSquare = (data: object) => api.post("/api/stats/chisquare", data);
export const runAnova = (data: object) => api.post("/api/stats/anova", data);
export const runMannWhitney = (data: object) => api.post("/api/stats/mannwhitney", data);
export const runFisher = (data: object) => api.post("/api/stats/fisher", data);
export const runKruskal = (data: object) => api.post("/api/stats/kruskal", data);
export const runJonckheereTerpstra = (data: object) => api.post("/api/stats/jonckheere_terpstra", data);
export const runROC = (data: object) => api.post("/api/stats/roc", data);
export const runROCCompare = (data: object) => api.post("/api/stats/roc_compare", data);
export const runROCMultiCompare = (data: object) => api.post("/api/stats/roc_multi_compare", data);
export const runROCCombined = (data: object) => api.post("/api/stats/roc_combined", data);

export const getHistogram = (data: object) => api.post("/api/charts/histogram", data);
export const getScatter = (data: object) => api.post("/api/charts/scatter", data);
export const getBoxplot = (data: object) => api.post("/api/charts/boxplot", data);
export const getBar = (data: object) => api.post("/api/charts/bar", data);
export const getGroupedBar = (data: object) => api.post("/api/charts/grouped_bar", data);

export const runLinear   = (data: object) => api.post("/api/models/linear", data);
export const runDeltaSensitivity = (data: object) => api.post("/api/models/delta_sensitivity", data);
export const runRCS      = (data: object) => api.post("/api/models/rcs", data);
export const runLogistic = (data: object) => api.post("/api/models/logistic", data);
export const runFirthLogistic = (data: object) => api.post("/api/models/firth_logistic", data);
export const runLogisticTable = (data: object) => api.post("/api/models/logistic_table", data);
export const runPoisson  = (data: object) => api.post("/api/models/poisson", data);
export const runKM = (data: object) => api.post("/api/models/survival/km", data);
export const runCox = (data: object) => api.post("/api/models/survival/cox", data);
export const runCoxRCS = (data: object) => api.post("/api/models/survival/cox_rcs", data);
export const runPolynomial  = (data: object) => api.post("/api/models/polynomial", data);
export const runLMM         = (data: object) => api.post("/api/models/lmm", data);
export const runGamma       = (data: object) => api.post("/api/models/gamma", data);
export const runNegBinom    = (data: object) => api.post("/api/models/negbinom", data);
export const runLinearDiag  = (data: object) => api.post("/api/models/linear_diag", data);
export const runMelt          = (data: object) => api.post("/api/models/melt", data);

// Machine learning (predictive modeling)
export const runRandomForest      = (data: object) => api.post("/api/ml/random_forest", data);
export const runGradientBoosting  = (data: object) => api.post("/api/ml/gradient_boosting", data);
export const runFeatureImportance = (data: object) => api.post("/api/ml/feature_importance", data);

// Time series
export const runArima        = (data: object) => api.post("/api/timeseries/arima", data);
export const runDecompose    = (data: object) => api.post("/api/timeseries/decompose", data);
export const runStationarity = (data: object) => api.post("/api/timeseries/stationarity", data);

// Weighted descriptives (survey weights)
export const runWeightedDescriptive = (data: object) => api.post("/api/stats/weighted_descriptive", data);

// Non-inferiority / margin testing
export const runNonInferiority = (data: object) => api.post("/api/stats/noninferiority", data);

// Multiplicity / gatekeeping
export const runGatekeeping = (data: object) => api.post("/api/multiplicity/gatekeeping", data);

// Meta-analysis (study-level)
export const runMetaAnalyze    = (data: object) => api.post("/api/meta/analyze", data);
export const runMetaSubgroup   = (data: object) => api.post("/api/meta/subgroup", data);
export const runMetaRegression = (data: object) => api.post("/api/meta/regression", data);
export const runMetaBias       = (data: object) => api.post("/api/meta/bias", data);
export const refreshSession   = (sessionId: string) => api.get(`/api/stats/${sessionId}/refresh`);
export const runPSM           = (data: object) => api.post("/api/models/psm", data);
export const runIPTW          = (data: object) => api.post("/api/models/iptw", data);

export const getSparklines = (sessionId: string) =>
  api.get(`/api/stats/${sessionId}/sparklines`);

export const getRawColumns = (sessionId: string, columns: string[]) =>
  api.get(`/api/stats/${sessionId}/raw`, { params: { columns: columns.join(",") } });

export const getMissing = (sessionId: string, columns: string[]) =>
  api.get(`/api/stats/${sessionId}/missing`, { params: { columns: columns.join(",") } });

// ── Compute / Create New Variable ──────────────────────────────────────────
export const computeFormula    = (sessionId: string, data: object) => api.post(`/api/compute/${sessionId}/formula`, data);
export const computeTransform  = (sessionId: string, data: object) => api.post(`/api/compute/${sessionId}/transform`, data);
export const computeRecode     = (sessionId: string, data: object) => api.post(`/api/compute/${sessionId}/recode`, data);
export const computeClinical   = (sessionId: string, calc: string, data: object) => api.post(`/api/compute/${sessionId}/clinical/${calc}`, data);
export const deleteColumn      = (sessionId: string, col: string) => api.delete(`/api/compute/${sessionId}/column/${encodeURIComponent(col)}`);
export const getUniqueValues   = (sessionId: string, col: string) => api.get(`/api/compute/${sessionId}/unique/${encodeURIComponent(col)}`);

export const runCorrelationPair = (data: object) => api.post("/api/stats/correlation_pair", data);
export const runCorrelationMatrix = (data: object) => api.post("/api/stats/correlation_matrix", data);
export const runICC = (data: object) => api.post("/api/stats/icc", data);
export const runCohensKappa = (data: object) => api.post("/api/stats/cohens_kappa", data);
export const runFleissKappa = (data: object) => api.post("/api/stats/fleiss_kappa", data);
export const runPower       = (data: object) => api.post("/api/stats/power", data);
export const runHosmerLemeshow = (data: object) => api.post("/api/decision_curve/hosmer_lemeshow", data);
export const runTOST           = (data: object) => api.post("/api/stats/tost", data);
export const runGEE            = (data: object) => api.post("/api/models/gee", data);
export const runOrdinal        = (data: object) => api.post("/api/models/ordinal", data);
export const runCoxTV          = (data: object) => api.post("/api/models/survival/cox_tv", data);
export const runStepwise       = (data: object) => api.post("/api/models/stepwise", data);
export const runForest         = (data: object) => api.post("/api/charts/forest", data);
export const downloadMethodAppendix = (sessionId: string, title?: string) =>
  api.post("/api/pub_export/method_appendix", { session_id: sessionId, title: title ?? "Statistical Methods" }, { responseType: "blob" });

// Repeated measures
export const runPairedTTest  = (data: object) => api.post("/api/repeated/paired_ttest", data);
export const runWilcoxonSR   = (data: object) => api.post("/api/repeated/wilcoxon_signed_rank", data);
export const runFriedman     = (data: object) => api.post("/api/repeated/friedman", data);
export const runRMAnova      = (data: object) => api.post("/api/repeated/rm_anova", data);
export const runMixedAnova   = (data: object) => api.post("/api/repeated/mixed_anova", data);

// Advanced ANOVA
export const runAncova       = (data: object) => api.post("/api/advanced_anova/ancova", data);
export const runTwoWayAnova  = (data: object) => api.post("/api/advanced_anova/two_way_anova", data);

// Categorical
export const runBinomial     = (data: object) => api.post("/api/categorical/binomial", data);
export const runOneProportion = (data: object) => api.post("/api/categorical/one_proportion", data);
export const runTwoProportions = (data: object) => api.post("/api/categorical/two_proportions", data);
export const runMcNemar      = (data: object) => api.post("/api/categorical/mcnemar", data);
export const runCochranQ     = (data: object) => api.post("/api/categorical/cochran_q", data);
export const runMantelHaenszel = (data: object) => api.post("/api/categorical/mantel_haenszel", data);
export const runCochranArmitage = (data: object) => api.post("/api/categorical/cochran_armitage", data);

// Agreement
export const runBlandAltman  = (data: object) => api.post("/api/agreement/bland_altman", data);
export const runDeming       = (data: object) => api.post("/api/agreement/deming", data);
export const runPassingBablok = (data: object) => api.post("/api/agreement/passing_bablok", data);
export const runConcordance  = (data: object) => api.post("/api/agreement/concordance", data);

// Reliability
export const runCronbach     = (data: object) => api.post("/api/reliability/cronbach", data);

// Missing data
export const runMissingPattern = (data: object) => api.post("/api/missing_data/pattern", data);
export const runMCARTest     = (data: object) => api.post("/api/missing_data/mcar_test", data);
export const runImputationCompare = (data: object) => api.post("/api/missing_data/imputation_compare", data);

// Diagnostics
export const runLinearDiagFull = (data: object) => api.post("/api/diagnostics/linear_full", data);
export const runLogisticDiag   = (data: object) => api.post("/api/model_diagnostics/logistic_diagnostics", data);
export const runCoxDiag        = (data: object) => api.post("/api/model_diagnostics/cox_diagnostics", data);

// Decision curve
export const runCalibration    = (data: object) => api.post("/api/decision_curve/calibration", data);
export const runDCA            = (data: object) => api.post("/api/decision_curve/dca", data);

// Model comparison
export const runNestedLR       = (data: object) => api.post("/api/model_compare/nested_lr_test", data);
export const runCompareModels  = (data: object) => api.post("/api/model_compare/compare_models", data);

// Survival advanced
export const runMICE       = (data: object) => api.post("/api/survival_advanced/mice", data);
export const runFineGray   = (data: object) => api.post("/api/survival_advanced/fine_gray", data);
export const runEValue     = (data: object) => api.post("/api/survival_advanced/evalue", data);
export const runLandmark   = (data: object) => api.post("/api/survival_advanced/landmark", data);
export const runRMST       = (data: object) => api.post("/api/survival_advanced/rmst", data);
export const runRecurrentLWYY = (data: object) => api.post("/api/survival_advanced/recurrent_lwyy", data);

// Article parser
export const parseArticle = (file: File) => {
  const fd = new FormData();
  fd.append("file", file);
  return api.post("/api/article_parser/parse", fd);
};

// Column operations
export const renameColumn = (sessionId: string, oldName: string, newName: string) =>
  api.post(`/api/compute/${sessionId}/rename`, { old_name: oldName, new_name: newName });

// Session management
export const saveSession   = (sessionId: string) => api.get(`/api/sessions/${sessionId}/save_session`, { responseType: "blob" });
export const loadSession   = (file: File) => { const fd = new FormData(); fd.append("file", file); return api.post("/api/sessions/load_session", fd); };
export const getAuditTrail = (sessionId: string) => api.get(`/api/sessions/${sessionId}/audit`);
export const saveMetadata  = (sessionId: string, columns: Record<string, any>) => api.post(`/api/sessions/${sessionId}/metadata`, { columns });
export const setColumnKind = (sessionId: string, column: string, kind: string) => api.post(`/api/sessions/${sessionId}/kind`, { column, kind });
export const setColumnDecimalsApi = (sessionId: string, column: string, decimals: number | null) =>
  api.post(`/api/sessions/${sessionId}/decimals`, { column, decimals });
export const getColumnDecimalsApi = (sessionId: string) =>
  api.get<Record<string, number>>(`/api/sessions/${sessionId}/decimals`);
export const deleteRow     = (sessionId: string, rowIndex: number) => api.delete(`/api/sessions/${sessionId}/row/${rowIndex}`);

// Publication export
export const exportTableDocx = (data: object) => api.post("/api/pub_export/table_docx", data, { responseType: "blob" });
export const getFigureCaption = (data: object) => api.post("/api/pub_export/figure_caption", data);

// Nomogram
export const buildNomogram = (data: object) => api.post("/api/nomogram/build", data);

export const selectCases = (sessionId: string, conditions: object[]) =>
  api.post(`/api/sessions/${sessionId}/select_cases`, { conditions });
export const clearCases  = (sessionId: string) =>
  api.delete(`/api/sessions/${sessionId}/select_cases`);

// ── Code runner ────────────────────────────────────────────────────────────

export interface CodeRunnerStatus {
  enabled: boolean;
  max_timeout_s: number;
  max_code_bytes: number;
  rate_limit_per_min: number;
  rate_limit_per_hour: number;
}

export interface CodeRunRequest {
  session_id: string;
  code: string;
  timeout?: number;
}

export interface CodeRunResponse {
  stdout: string;
  stderr: string;
  figures: string[];   // base64 PNGs
  exit_code: number;
  time_used_s: number;
  error: string | null;
  timed_out: boolean;
}

export const codeRunnerStatus = () => api.get<CodeRunnerStatus>("/api/code/status");
export const runCode = (data: CodeRunRequest, signal?: AbortSignal) =>
  api.post<CodeRunResponse>("/api/code/run", data, { signal });
