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
export const runROC = (data: object) => api.post("/api/stats/roc", data);
export const runROCCompare = (data: object) => api.post("/api/stats/roc_compare", data);
export const runROCCombined = (data: object) => api.post("/api/stats/roc_combined", data);

export const getHistogram = (data: object) => api.post("/api/charts/histogram", data);
export const getScatter = (data: object) => api.post("/api/charts/scatter", data);
export const getBoxplot = (data: object) => api.post("/api/charts/boxplot", data);
export const getBar = (data: object) => api.post("/api/charts/bar", data);

export const runLinear   = (data: object) => api.post("/api/models/linear", data);
export const runRCS      = (data: object) => api.post("/api/models/rcs", data);
export const runLogistic = (data: object) => api.post("/api/models/logistic", data);
export const runLogisticTable = (data: object) => api.post("/api/models/logistic_table", data);
export const runPoisson  = (data: object) => api.post("/api/models/poisson", data);
export const runKM = (data: object) => api.post("/api/models/survival/km", data);
export const runCox = (data: object) => api.post("/api/models/survival/cox", data);
export const runPolynomial  = (data: object) => api.post("/api/models/polynomial", data);
export const runLMM         = (data: object) => api.post("/api/models/lmm", data);
export const runGamma       = (data: object) => api.post("/api/models/gamma", data);
export const runNegBinom    = (data: object) => api.post("/api/models/negbinom", data);
export const runLinearDiag  = (data: object) => api.post("/api/models/linear_diag", data);
export const runMelt          = (data: object) => api.post("/api/models/melt", data);
export const refreshSession   = (sessionId: string) => api.get(`/api/stats/${sessionId}/refresh`);
export const runPSM           = (data: object) => api.post("/api/models/psm", data);

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
export const runPower       = (data: object) => api.post("/api/stats/power", data);

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

export const selectCases = (sessionId: string, conditions: object[]) =>
  api.post(`/api/sessions/${sessionId}/select_cases`, { conditions });
export const clearCases  = (sessionId: string) =>
  api.delete(`/api/sessions/${sessionId}/select_cases`);
