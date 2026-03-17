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

export const runLinear = (data: object) => api.post("/api/models/linear", data);
export const runLogistic = (data: object) => api.post("/api/models/logistic", data);
export const runLogisticTable = (data: object) => api.post("/api/models/logistic_table", data);
export const runKM = (data: object) => api.post("/api/models/survival/km", data);
export const runCox = (data: object) => api.post("/api/models/survival/cox", data);

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
