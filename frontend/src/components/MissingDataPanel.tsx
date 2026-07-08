import { useState, type ReactNode } from "react";
import { useStore, isNumericKind } from "../store";
import {
  fillBlanks,
  getExternalImputeReferenceColumns,
  runExternalImputeApply,
  runExternalImputePreview,
  runImputationCompare,
  runMCARTest,
  runMICE,
  runMissingDiagnostics,
} from "../api";
import ResultExporter from "./ResultExporter";
import api from "../api";
import { CleaningTab } from "./CleaningTab";
import { fmtP } from "../lib/format";

interface DiagCol { name: string; n_missing: number; pct: number; kind: string; is_numeric: boolean; depends_on: string[]; likely: string }
interface DiagResult { columns: DiagCol[]; overall_hint: string; recommendation: string; any_mar: boolean }
type MissingSort = "missing-desc" | "missing-asc" | "name-asc" | "name-desc";
type QuickMethod = "__mean__" | "__median__" | "__mode__" | "__mice__";

interface MiceExportResult {
  result_text?: string;
  methods_text?: string;
  export_rows?: unknown[][];
}
interface McarResult {
  statistic: number | string;
  df: number;
  p: number | string;
  significant: boolean;
}
interface CompareColumn {
  col: string;
  before?: { mean?: number | string };
  after?: { mean?: number | string };
  ks_p?: number | null;
}
interface CompareEntry { strategy: string; columns: CompareColumn[] }
interface CompareResult { comparisons?: CompareEntry[] }
interface ExternalPreviewRow { row_index: number; imputed_value: unknown; predictors_missing: number }
interface ExternalReferenceColumn { name: string; dtype: string; kind: string; n_missing: number }
interface ExternalReferenceColumnsResult { columns: ExternalReferenceColumn[]; n_rows: number }
interface ExternalImputeResult {
  target: string;
  reference_target?: string;
  predictors: string[];
  reference_predictors?: string[];
  predictor_mappings?: Record<string, string>;
  method: string;
  mechanism: string;
  n_missing_target: number;
  n_imputed: number;
  reference_rows: number;
  reference_complete_rows: number;
  preview_rows: ExternalPreviewRow[];
  warnings?: string[];
  result_text?: string;
  methods_text?: string;
  export_rows?: unknown[][];
  applied?: boolean;
}

const QUICK_SUFFIX: Record<QuickMethod, string> = {
  __mean__: "mean",
  __median__: "median",
  __mode__: "mode",
  __mice__: "imp",
};

const errText = (e: unknown): string =>
  (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Request failed";

const normColumnName = (name: string) => name.trim().toLowerCase();

export default function MissingDataPanel() {
  const session = useStore((s) => s.session);
  const columns = session?.columns ?? [];
  const numCols = columns.filter((c) => isNumericKind(c.kind));
  const sid = session?.session_id ?? "";
  const [activeSubTab, setActiveSubTab] = useState<"overview" | "cleaning" | "reference">("overview");

  const preview = session?.preview ?? [];
  const missingInfo = columns
    .map((c) => {
      const nMiss = preview.filter(
        (r) => r[c.name] === null || r[c.name] === undefined || r[c.name] === ""
      ).length;
      return {
        name: c.name,
        kind: c.kind,
        isNum: isNumericKind(c.kind),
        nMiss,
        pct: preview.length > 0 ? (nMiss / preview.length) * 100 : 0,
      };
    })
    .filter((m) => m.nMiss > 0);

  // Selection + MICE state
  const [selected, setSelected] = useState<string[]>([]);
  const [missingSort, setMissingSort] = useState<MissingSort>("missing-desc");
  const [miceIter, setMiceIter] = useState(20);
  const [miceSeed, setMiceSeed] = useState(42);
  const [miceMechanism, setMiceMechanism] = useState<"unknown" | "MCAR" | "MAR" | "MNAR">("unknown");
  const [miceResult, setMiceResult] = useState<MiceExportResult | null>(null);
  const [miceLoading, setMiceLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null); // per-row action in flight
  const [err, setErr] = useState<string | null>(null);
  const [mutationNotice, setMutationNotice] = useState<string | null>(null);

  // Diagnostics state
  const [diag, setDiag] = useState<DiagResult | null>(null);
  const [mcar, setMcar] = useState<McarResult | null>(null);
  const [mcarNote, setMcarNote] = useState<string | null>(null);
  const [compare, setCompare] = useState<CompareResult | null>(null);
  const [externalTarget, setExternalTarget] = useState("");
  const [externalPredictors, setExternalPredictors] = useState<string[]>([]);
  const [externalReferenceTarget, setExternalReferenceTarget] = useState("");
  const [externalPredictorMappings, setExternalPredictorMappings] = useState<Record<string, string>>({});
  const [externalFile, setExternalFile] = useState<File | null>(null);
  const [externalReferenceMeta, setExternalReferenceMeta] = useState<ExternalReferenceColumnsResult | null>(null);
  const [externalMethod, setExternalMethod] = useState<"pmm" | "mice">("pmm");
  const [externalResult, setExternalResult] = useState<ExternalImputeResult | null>(null);
  const [externalLoading, setExternalLoading] = useState<"columns" | "preview" | "apply" | null>(null);

  if (!session) return <p className="text-gray-400 text-sm p-6">Upload data first.</p>;

  const refresh = async () => {
    const r = await api.get(`/api/stats/${sid}/refresh`);
    useStore.getState().setSession({ ...useStore.getState().session!, ...r.data });
    useStore.setState((s) => ({ dataVersion: s.dataVersion + 1 }));
  };

  const clearDiagnostics = () => {
    setDiag(null);
    setMcar(null);
    setMcarNote(null);
  };

  const toggle = (name: string) => {
    clearDiagnostics();
    setSelected((p) => (p.includes(name) ? p.filter((c) => c !== name) : [...p, name]));
  };

  const sortedMissingInfo = [...missingInfo].sort((a, b) => {
    if (missingSort === "name-asc" || missingSort === "name-desc") {
      const order = a.name.localeCompare(b.name, "tr", { numeric: true, sensitivity: "base" });
      return missingSort === "name-asc" ? order : -order;
    }
    const order = a.pct - b.pct || a.name.localeCompare(b.name, "tr", { numeric: true, sensitivity: "base" });
    return missingSort === "missing-asc" ? order : -order;
  });

  const toggleNameSort = () =>
    setMissingSort((current) => current === "name-asc" ? "name-desc" : "name-asc");

  const toggleMissingSort = () =>
    setMissingSort((current) => current === "missing-asc" ? "missing-desc" : "missing-asc");

  const sortArrow = (asc: MissingSort, desc: MissingSort) =>
    missingSort === asc ? "↑" : missingSort === desc ? "↓" : "↕";

  const nextColumnName = (source: string, method: QuickMethod): string => {
    const base = `${source}_${QUICK_SUFFIX[method]}`;
    const existing = new Set(columns.map((c) => c.name));
    let candidate = base;
    let index = 2;
    while (existing.has(candidate)) candidate = `${base}_${index++}`;
    return candidate;
  };

  // Per-row quick imputation (acts immediately on one column).
  const quickFill = async (col: string, method: QuickMethod) => {
    setBusy(`${col}:${method}`); setErr(null); setMutationNotice(null);
    try {
      const newColumn = nextColumnName(col, method);
      const response = await fillBlanks(sid, col, method, newColumn);
      await refresh();
      setMutationNotice(
        `${col} korundu; ${response.data.column} oluşturuldu ve ${response.data.n_filled} eksik değer tamamlandı.`
      );
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setBusy(null);
    }
  };

  const runDiagnostics = async () => {
    if (selected.length === 0) { setErr("Select at least one column to analyze"); return; }
    const selectedNumeric = selected.filter((name) => missingInfo.some((m) => m.name === name && m.isNum));
    setBusy("diag"); setErr(null); setDiag(null); setMcar(null); setMcarNote(null);
    try {
      const diagRequest = runMissingDiagnostics(sid, selected);
      const mcarRequest = selectedNumeric.length >= 2
        ? runMCARTest({ session_id: sid, columns: selectedNumeric })
        : null;
      const [d, m] = await Promise.allSettled([
        diagRequest,
        ...(mcarRequest ? [mcarRequest] : []),
      ]);
      if (d.status === "fulfilled") setDiag(d.value.data);
      else setErr(errText(d.reason));
      if (mcarRequest) {
        if (m?.status === "fulfilled") setMcar(m.value.data);
        else if (m?.status === "rejected") setMcarNote(`Little's MCAR test could not be calculated: ${errText(m.reason)}`);
      } else {
        setMcarNote("Little's MCAR test requires at least two selected numeric variables. The dependence analysis below is limited to the selected variable(s).");
      }
    } finally {
      setBusy(null);
    }
  };

  const runCompare = async () => {
    if (selected.length === 0) { setErr("Select columns to compare"); return; }
    setBusy("compare"); setErr(null); setCompare(null);
    try {
      const r = await runImputationCompare({ session_id: sid, columns: selected, strategies: ["median", "mice"] });
      setCompare(r.data);
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setBusy(null);
    }
  };

  const externalTargetName = externalTarget || missingInfo[0]?.name || "";
  const currentColumnNames = new Set(columns.map((c) => c.name));
  const currentColumnByNorm = new Map(columns.map((c) => [normColumnName(c.name), c.name]));
  const externalReferenceColumns = externalReferenceMeta?.columns ?? [];
  const autoReferenceTarget = externalReferenceColumns.find(
    (c) => normColumnName(c.name) === normColumnName(externalTargetName)
  )?.name ?? "";
  const externalReferenceTargetName = externalReferenceTarget || autoReferenceTarget;
  const externalPredictorColumns = externalReferenceColumns.filter(
    (c) => normColumnName(c.name) !== normColumnName(externalReferenceTargetName)
  );
  const predictorMappings = Object.fromEntries(
    externalPredictors.map((name) => [
      name,
      externalPredictorMappings[name] || currentColumnByNorm.get(normColumnName(name)) || "",
    ])
  );
  const externalPayload = () => ({
    sessionId: sid,
    target: externalTargetName,
    referenceTarget: externalReferenceTargetName,
    predictors: externalPredictors,
    predictorMappings,
    method: externalMethod,
    mechanism: miceMechanism,
    maxIter: miceIter,
    randomState: miceSeed,
    file: externalFile!,
  });

  const validateExternal = (): boolean => {
    if (!externalTargetName) { setErr("Select a target column with missing values"); return false; }
    if (externalPredictors.length === 0) { setErr("Select at least one predictor for the target"); return false; }
    if (!externalFile) { setErr("Upload a reference dataset first"); return false; }
    if (!externalReferenceTargetName) { setErr("Select matching target column in reference dataset"); return false; }
    if (!externalReferenceColumns.some((c) => c.name === externalReferenceTargetName)) {
      setErr(`Reference dataset must contain target column '${externalReferenceTargetName}'`);
      return false;
    }
    const missingCurrent = Object.entries(predictorMappings)
      .filter(([, currentName]) => !currentName || !currentColumnNames.has(currentName))
      .map(([refName]) => refName);
    if (missingCurrent.length > 0) {
      setErr(`Select current data match for: ${missingCurrent.join(", ")}`);
      return false;
    }
    return true;
  };

  const loadExternalReferenceColumns = async (file: File | null) => {
    setExternalFile(file);
    setExternalReferenceMeta(null);
    setExternalPredictors([]);
    setExternalReferenceTarget("");
    setExternalPredictorMappings({});
    setExternalResult(null);
    if (!file) return;
    setExternalLoading("columns"); setErr(null);
    try {
      const res = await getExternalImputeReferenceColumns(file);
      const meta = res.data as ExternalReferenceColumnsResult;
      setExternalReferenceMeta(meta);
      setExternalReferenceTarget(
        meta.columns.find((c) => normColumnName(c.name) === normColumnName(externalTargetName))?.name ?? ""
      );
      setExternalPredictorMappings(Object.fromEntries(
        meta.columns.map((c) => [c.name, currentColumnByNorm.get(normColumnName(c.name)) ?? ""])
      ));
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setExternalLoading(null);
    }
  };

  const runExternalPreview = async () => {
    if (!validateExternal()) return;
    setExternalLoading("preview"); setErr(null); setExternalResult(null); setMutationNotice(null);
    try {
      const res = await runExternalImputePreview(externalPayload());
      setExternalResult(res.data);
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setExternalLoading(null);
    }
  };

  const applyExternalImputation = async () => {
    if (!validateExternal()) return;
    setExternalLoading("apply"); setErr(null); setMutationNotice(null);
    try {
      const res = await runExternalImputeApply(externalPayload());
      setExternalResult(res.data);
      await refresh();
      setMutationNotice(`${res.data.n_imputed} value(s) transferred into ${res.data.target}.`);
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setExternalLoading(null);
    }
  };

  const handleMICE = async () => {
    if (selected.length === 0) { setErr("Select columns to impute"); return; }
    setMiceLoading(true); setErr(null); setMutationNotice(null); setMiceResult(null);
    try {
      const res = await runMICE({
        session_id: sid, columns: selected, n_imputations: 1,
        max_iter: miceIter, random_state: miceSeed, mechanism: miceMechanism,
        new_columns: true,
      });
      setMiceResult(res.data);
      await refresh();
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setMiceLoading(false);
    }
  };

  const pctClass = (pct: number) =>
    pct > 30 ? "bg-red-100 text-red-600" : pct > 10 ? "bg-amber-100 text-amber-600" : "bg-gray-100 text-gray-500";
  const QuickBtn = ({ col, method, label, show }: { col: string; method: QuickMethod; label: string; show: boolean }) =>
    !show ? null : (
      <button
        onClick={() => quickFill(col, method)}
        disabled={busy === `${col}:${method}`}
        className="text-[10px] px-1.5 py-0.5 rounded border border-gray-200 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 hover:border-indigo-300 disabled:opacity-40 transition-colors"
      >
        {busy === `${col}:${method}` ? "…" : label}
      </button>
    );

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div
        className="mb-5 flex items-center gap-1 border-b border-gray-200"
        role="tablist"
        aria-label="Missing data sections"
      >
        {([
          ["overview", "Missing Data Overview"],
          ["cleaning", "Data Cleaning"],
          ["reference", "Reference Imputation"],
        ] as const).map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={activeSubTab === id}
            onClick={() => setActiveSubTab(id)}
            className={`relative px-4 py-2.5 text-sm font-medium transition-colors ${
              activeSubTab === id
                ? "text-indigo-700"
                : "text-gray-500 hover:text-gray-800"
            }`}
          >
            {label}
            {activeSubTab === id && (
              <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-indigo-600" />
            )}
          </button>
        ))}
      </div>

      <div className={activeSubTab === "overview" ? "space-y-5" : "hidden"} role="tabpanel">
        {err && <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-600">{err}</div>}
        {mutationNotice && <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2 text-xs text-emerald-700">{mutationNotice}</div>}

        {/* ── Overview — list ── */}
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-gray-800">Missing Data Overview</h3>
              <p className="text-[11px] text-gray-400 mt-0.5">Tick rows for MICE / comparison, or impute a single column inline.</p>
            </div>
            {missingInfo.length > 0 && (
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
                  <span>Sort</span>
                  <select value={missingSort} onChange={(e) => setMissingSort(e.target.value as MissingSort)}
                    className="border border-gray-300 rounded px-2 py-1 bg-white text-[10px] text-gray-600">
                    <option value="missing-desc">Missing %: high to low</option>
                    <option value="missing-asc">Missing %: low to high</option>
                    <option value="name-asc">Name: A to Z</option>
                    <option value="name-desc">Name: Z to A</option>
                  </select>
                </label>
                <button onClick={() => {
                  clearDiagnostics();
                  setSelected(selected.length === missingInfo.length ? [] : missingInfo.map((m) => m.name));
                }}
                  className="text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-gray-100">
                  {selected.length === missingInfo.length ? "Clear all" : "Select all"}
                </button>
              </div>
            )}
          </div>
          {missingInfo.length === 0 ? (
            <div className="px-5 py-4">
              <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-700">
                ✅ No missing values detected in any column.
              </div>
            </div>
          ) : (
            <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-gray-400 border-b border-gray-100">
                <th className="px-3 py-2 w-8"></th>
                <th className="px-3 py-2">
                  <button
                    type="button"
                    onClick={toggleNameSort}
                    className="inline-flex items-center gap-1 hover:text-indigo-600 transition-colors"
                    title="Sort by variable name"
                  >
                    Variable <span aria-hidden="true">{sortArrow("name-asc", "name-desc")}</span>
                  </button>
                </th>
                <th className="px-3 py-2">Type</th>
                <th className="px-3 py-2 text-right">
                  <button
                    type="button"
                    onClick={toggleMissingSort}
                    className="inline-flex items-center justify-end gap-1 hover:text-indigo-600 transition-colors"
                    title="Sort by missing percentage"
                  >
                    Missing <span aria-hidden="true">{sortArrow("missing-asc", "missing-desc")}</span>
                  </button>
                </th>
                <th className="px-3 py-2 w-28">
                  <button
                    type="button"
                    onClick={toggleMissingSort}
                    className="inline-flex items-center gap-1 hover:text-indigo-600 transition-colors"
                    title="Sort by missing percentage"
                  >
                    % <span aria-hidden="true">{sortArrow("missing-asc", "missing-desc")}</span>
                  </button>
                </th>
                <th className="px-3 py-2 text-right">Quick impute</th>
              </tr>
            </thead>
            <tbody>
              {sortedMissingInfo.map((m) => (
                <tr key={m.name} className={`border-b border-gray-50 ${selected.includes(m.name) ? "bg-indigo-50/40" : "hover:bg-gray-50"}`}>
                  <td className="px-3 py-1.5">
                    <input type="checkbox" checked={selected.includes(m.name)} onChange={() => toggle(m.name)} className="accent-indigo-500" />
                  </td>
                  <td className="px-3 py-1.5 font-medium text-gray-800 truncate max-w-[10rem]">{m.name}</td>
                  <td className="px-3 py-1.5 text-gray-500">{m.kind}</td>
                  <td className="px-3 py-1.5 text-right text-gray-600">{m.nMiss}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-1.5">
                      <div className="flex-1 h-1.5 rounded-full bg-gray-100 overflow-hidden">
                        <div className={`h-full ${m.pct > 30 ? "bg-red-400" : m.pct > 10 ? "bg-amber-400" : "bg-gray-300"}`} style={{ width: `${Math.min(100, m.pct)}%` }} />
                      </div>
                      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${pctClass(m.pct)}`}>{m.pct.toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center justify-end gap-1">
                      <QuickBtn col={m.name} method="__mean__" label="Mean" show={m.isNum} />
                      <QuickBtn col={m.name} method="__median__" label="Median" show={m.isNum} />
                      <QuickBtn col={m.name} method="__mode__" label="Mode" show={!m.isNum} />
                      <QuickBtn col={m.name} method="__mice__" label="MICE" show={m.isNum} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
            </table>
          )}
        </div>

        {missingInfo.length > 0 && (
          <>
          {/* ── Mechanism + diagnostics ── */}
          <div className="border border-gray-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-gray-800">Missing Mechanism</h3>
                <p className="text-[11px] text-gray-400 mt-0.5">Determines which imputation is valid. Analyze, or set it manually.</p>
              </div>
              <button onClick={runDiagnostics} disabled={busy === "diag" || selected.length === 0}
                className="text-xs px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50">
                {busy === "diag" ? "Analyzing…" : `Analyze missingness${selected.length ? ` (${selected.length})` : ""}`}
              </button>
            </div>
            <div className="px-5 py-4 space-y-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {([
                  { id: "unknown", label: "Unknown", desc: "Analyze to decide" },
                  { id: "MCAR", label: "MCAR", desc: "Completely at random" },
                  { id: "MAR", label: "MAR", desc: "At random (on observed)" },
                  { id: "MNAR", label: "MNAR", desc: "Not at random" },
                ] as const).map(({ id, label, desc }) => (
                  <button key={id} onClick={() => setMiceMechanism(id)}
                    className={`flex flex-col items-start gap-0.5 px-3 py-2 rounded-lg border text-left transition-colors ${
                      miceMechanism === id ? "border-indigo-400 bg-indigo-50" : "border-gray-200 bg-white hover:border-gray-300"
                    }`}>
                    <span className={`text-xs font-semibold ${miceMechanism === id ? "text-indigo-700" : "text-gray-700"}`}>{label}</span>
                    <span className="text-[10px] text-gray-400 leading-tight">{desc}</span>
                  </button>
                ))}
              </div>

              {/* Data-driven hint (heuristic + Little's MCAR), no AI */}
              {(diag || mcar) && (
                <div className="space-y-2">
                  {mcar && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-[11px] text-blue-800">
                      <span className="font-semibold">Little's MCAR test:</span> χ²={Number(mcar.statistic).toFixed(2)}, df={mcar.df}, <i>p</i>={fmtP(Number(mcar.p))}.{" "}
                      {mcar.significant
                        ? "p < 0.05 → MCAR rejected; data are likely MAR (or MNAR). MICE is appropriate."
                        : "p ≥ 0.05 → consistent with MCAR; listwise deletion is unbiased (MICE still fine)."}
                    </div>
                  )}
                  {diag && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-[11px] text-blue-800">
                      <span className="font-semibold">Dependence check:</span> {diag.overall_hint} {diag.recommendation}
                      {diag.columns.some((c) => c.depends_on.length > 0) && (
                        <ul className="mt-1 ml-3 list-disc">
                          {diag.columns.filter((c) => c.depends_on.length > 0).map((c) => (
                            <li key={c.name}>{c.name}: missingness related to {c.depends_on.join(", ")} → MAR signal</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              )}
              {mcarNote && (
                <div className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-[11px] text-gray-600">
                  {mcarNote}
                </div>
              )}
              {miceMechanism === "MNAR" && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-700">
                  ⚠️ MNAR: MICE assumes MAR and may bias results. For &gt;40–50% missing, use a dedicated MNAR sensitivity analysis (pattern-mixture / selection model).
                </div>
              )}
            </div>
          </div>

          {/* ── MICE (multi-column) ── */}
          <div className="border border-indigo-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 bg-indigo-50 border-b border-indigo-100">
              <h3 className="text-sm font-semibold text-indigo-800">PMM Imputation (selected columns)</h3>
              <p className="text-[11px] text-indigo-400 mt-0.5">
                Keeps each original column and creates a new imputed column. For variance-correct inference, prefer the model panels' MICE option (m datasets + Rubin's-rules pooling).
              </p>
            </div>
            <div className="px-5 py-4 space-y-4">
              <div className="flex gap-4 flex-wrap">
                {([
                  ["Max iterations", miceIter, setMiceIter, 1, 100],
                  ["Seed", miceSeed, setMiceSeed, 0, 999999],
                ] as Array<[string, number, (v: number) => void, number, number]>).map(([lab, val, set, mn, mx]) => (
                  <label key={lab} className="flex flex-col gap-1">
                    <span className="text-xs text-gray-500 font-medium">{lab}</span>
                    <input type="number" value={val} onChange={(e) => set(Number(e.target.value))} min={mn} max={mx}
                      className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 w-24 focus:outline-none focus:border-indigo-400" />
                  </label>
                ))}
              </div>
              <div className="flex items-center gap-3 flex-wrap">
                <button onClick={handleMICE} disabled={miceLoading || selected.length === 0}
                  className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50">
                  {miceLoading ? "Imputing…" : `Apply PMM to ${selected.length || ""} column(s)`}
                </button>
                <button onClick={runCompare} disabled={busy === "compare" || selected.length === 0}
                  className="px-4 py-2 text-sm font-medium border border-indigo-300 text-indigo-600 rounded-lg hover:bg-indigo-50 disabled:opacity-50">
                  {busy === "compare" ? "Comparing…" : "Compare: complete-case vs MICE"}
                </button>
                {selected.length === 0 && <p className="text-xs text-gray-400">Select columns above</p>}
              </div>

              {miceResult?.result_text && (
                <div className="bg-emerald-50 border border-emerald-200 rounded-xl px-4 py-3 text-sm text-emerald-800">{miceResult.result_text}</div>
              )}
              {miceResult?.methods_text && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3">
                  <div className="flex items-center justify-between gap-3 mb-1.5">
                    <p className="text-xs font-semibold text-indigo-800">Methods</p>
                    <button
                      onClick={() => navigator.clipboard.writeText(miceResult.methods_text)}
                      className="text-[10px] px-2 py-0.5 rounded border border-indigo-200 text-indigo-600 hover:bg-white transition-colors"
                    >
                      Copy
                    </button>
                  </div>
                  <p className="text-xs text-indigo-800 leading-relaxed">{miceResult.methods_text}</p>
                </div>
              )}
              {miceResult?.export_rows?.length > 1 && (
                <>
                  <div className="overflow-auto rounded-lg border border-gray-200">
                    <table className="text-xs w-full">
                      <thead><tr className="bg-gray-50">{(miceResult.export_rows[0] as unknown[]).map((h, i: number) => <th key={i} className="px-3 py-1.5 text-left text-gray-500 font-medium">{String(h)}</th>)}</tr></thead>
                      <tbody>{miceResult.export_rows.slice(1).map((row, ri: number) => (
                        <tr key={ri} className="border-t border-gray-100">{(row as unknown[]).map((v, ci: number) => <td key={ci} className="px-3 py-1 text-gray-700">{(v as ReactNode) ?? "—"}</td>)}</tr>
                      ))}</tbody>
                    </table>
                  </div>
                  <ResultExporter title="MICE_imputation" headers={miceResult.export_rows[0]} rows={miceResult.export_rows.slice(1)} />
                </>
              )}

              {/* CCA vs MI comparison (sensitivity) */}
              {compare?.comparisons && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-gray-700">Sensitivity — observed vs imputed distribution</p>
                  <div className="overflow-auto rounded-lg border border-gray-200">
                    <table className="text-xs w-full">
                      <thead><tr className="bg-gray-50 text-left text-gray-500">
                        <th className="px-3 py-1.5">Strategy</th><th className="px-3 py-1.5">Column</th>
                        <th className="px-3 py-1.5">Mean (obs→after)</th><th className="px-3 py-1.5">KS p</th>
                      </tr></thead>
                      <tbody>
                        {compare.comparisons.flatMap((cmp) => cmp.columns.map((c) => (
                          <tr key={`${cmp.strategy}:${c.col}`} className="border-t border-gray-100">
                            <td className="px-3 py-1 text-gray-700">{cmp.strategy}</td>
                            <td className="px-3 py-1 text-gray-700">{c.col}</td>
                            <td className="px-3 py-1 text-gray-700">{c.before?.mean ?? "—"} → {c.after?.mean ?? "—"}</td>
                            <td className={`px-3 py-1 ${c.ks_p != null && c.ks_p < 0.05 ? "text-red-500" : "text-gray-700"}`}>{c.ks_p ?? "—"}</td>
                          </tr>
                        )))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[10px] text-gray-400">KS p &lt; 0.05 = imputed distribution differs from observed (expected for MAR; large shifts warrant a closer look).</p>
                </div>
              )}
            </div>
          </div>
          </>
        )}
      </div>

      <div className={activeSubTab === "cleaning" ? "" : "hidden"} role="tabpanel">
        <CleaningTab sessionId={sid} columns={columns} numCols={numCols} />
      </div>

      <div className={activeSubTab === "reference" ? "space-y-5" : "hidden"} role="tabpanel">
        {activeSubTab === "reference" && (
          <>
        {err && <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-600">{err}</div>}
        {mutationNotice && <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2 text-xs text-emerald-700">{mutationNotice}</div>}

        {missingInfo.length === 0 ? (
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-700">
            No missing values detected in any column.
          </div>
        ) : (
          <div className="border border-sky-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 bg-sky-50 border-b border-sky-100">
              <h3 className="text-sm font-semibold text-sky-800">Reference Dataset Imputation</h3>
              <p className="text-[11px] text-sky-500 mt-0.5">
                Upload a similar dataset, map reference variables to current variables, preview PMM/MICE estimates, then transfer values into the Data tab.
              </p>
            </div>
            <div className="px-5 py-4 space-y-4">
              <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Current missing target</span>
                  <select
                    value={externalTargetName}
                    onChange={(e) => {
                      const nextTarget = e.target.value;
                      const nextReferenceTarget = externalReferenceColumns.find(
                        (c) => normColumnName(c.name) === normColumnName(nextTarget)
                      )?.name ?? "";
                      setExternalTarget(nextTarget);
                      setExternalReferenceTarget(nextReferenceTarget);
                      setExternalPredictors((prev) => prev.filter((name) => name !== nextReferenceTarget));
                      setExternalResult(null);
                    }}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-sky-400"
                  >
                    {missingInfo.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Reference dataset</span>
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls,.sas7bdat,.sav,.dta"
                    onChange={(e) => void loadExternalReferenceColumns(e.target.files?.[0] ?? null)}
                    className="text-xs border border-gray-300 rounded-lg px-3 py-2 bg-white file:mr-3 file:border-0 file:bg-sky-50 file:text-sky-700 file:px-2 file:py-1 file:rounded"
                  />
                </label>
              </div>

              <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Reference target match</span>
                  <select
                    value={externalReferenceTargetName}
                    onChange={(e) => {
                      setExternalReferenceTarget(e.target.value);
                      setExternalPredictors((prev) => prev.filter((name) => name !== e.target.value));
                      setExternalResult(null);
                    }}
                    disabled={!externalReferenceMeta}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-sky-400 disabled:bg-gray-50"
                  >
                    <option value="">Select reference target</option>
                    {externalReferenceColumns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Method</span>
                  <select
                    value={externalMethod}
                    onChange={(e) => { setExternalMethod(e.target.value as "pmm" | "mice"); setExternalResult(null); }}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white focus:outline-none focus:border-sky-400"
                  >
                    <option value="pmm">PMM</option>
                    <option value="mice">MICE / PMM</option>
                  </select>
                </label>
              </div>

              <div>
                <div className="flex items-center justify-between gap-3 mb-2">
                  <p className="text-xs text-gray-500 font-medium">Reference predictors and current matches</p>
                  {externalReferenceMeta && (
                    <p className="text-[10px] text-gray-400">
                      {externalReferenceMeta.n_rows} rows, {externalReferenceMeta.columns.length} columns
                    </p>
                  )}
                </div>
                <div className="max-h-64 overflow-y-auto rounded-lg border border-gray-200">
                  {externalLoading === "columns" && (
                    <div className="px-3 py-2 text-xs text-gray-400">Reading columns...</div>
                  )}
                  {!externalLoading && externalFile && externalPredictorColumns.length === 0 && (
                    <div className="px-3 py-2 text-xs text-gray-400">No reference predictors found.</div>
                  )}
                  {!externalFile && (
                    <div className="px-3 py-2 text-xs text-gray-400">Upload reference dataset first.</div>
                  )}
                  {externalPredictorColumns.map((c) => {
                    const currentMatch = externalPredictorMappings[c.name] || currentColumnByNorm.get(normColumnName(c.name)) || "";
                    const checked = externalPredictors.includes(c.name);
                    return (
                      <div key={c.name} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-3 px-3 py-2 border-t first:border-t-0 border-gray-100 items-center">
                        <label className="flex items-center gap-2 text-xs text-gray-700 min-w-0">
                          <input
                            type="checkbox"
                            className="accent-sky-600"
                            checked={checked}
                            onChange={() => {
                              setExternalResult(null);
                              setExternalPredictors((prev) =>
                                prev.includes(c.name) ? prev.filter((x) => x !== c.name) : [...prev, c.name]
                              );
                            }}
                          />
                          <span className="truncate">{c.name}</span>
                        </label>
                        <select
                          value={currentMatch}
                          onChange={(e) => {
                            setExternalResult(null);
                            setExternalPredictorMappings((prev) => ({ ...prev, [c.name]: e.target.value }));
                          }}
                          className="text-xs border border-gray-300 rounded-md px-2 py-1 bg-white focus:outline-none focus:border-sky-400"
                        >
                          <option value="">Match current variable</option>
                          {columns
                            .filter((col) => col.name !== externalTargetName)
                            .map((col) => <option key={col.name} value={col.name}>{col.name}</option>)}
                        </select>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="flex items-center gap-3 justify-end">
                <button
                  onClick={runExternalPreview}
                  disabled={externalLoading !== null}
                  className="px-4 py-2 text-sm font-medium border border-sky-300 text-sky-700 rounded-lg hover:bg-sky-50 disabled:opacity-50"
                >
                  {externalLoading === "preview" ? "Calculating…" : "Preview"}
                </button>
                <button
                  onClick={applyExternalImputation}
                  disabled={externalLoading !== null || !externalResult}
                  className="px-4 py-2 text-sm font-medium bg-sky-600 text-white rounded-lg hover:bg-sky-700 disabled:opacity-50"
                >
                  {externalLoading === "apply" ? "Transferring…" : "Transfer data"}
                </button>
              </div>

              {externalResult?.result_text && (
                <div className="bg-sky-50 border border-sky-200 rounded-xl px-4 py-3 text-sm text-sky-800">
                  {externalResult.result_text}
                </div>
              )}
              {externalResult?.warnings?.map((w) => (
                <div key={w} className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-700">{w}</div>
              ))}
              {externalResult?.preview_rows?.length > 0 && (
                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="text-xs w-full">
                    <thead>
                      <tr className="bg-gray-50 text-left text-gray-500">
                        <th className="px-3 py-1.5">Row</th>
                        <th className="px-3 py-1.5">Imputed value</th>
                        <th className="px-3 py-1.5">Predictors missing</th>
                      </tr>
                    </thead>
                    <tbody>
                      {externalResult.preview_rows.map((row) => (
                        <tr key={row.row_index} className="border-t border-gray-100">
                          <td className="px-3 py-1 text-gray-700">{row.row_index}</td>
                          <td className="px-3 py-1 text-gray-700">{String(row.imputed_value)}</td>
                          <td className="px-3 py-1 text-gray-700">{row.predictors_missing}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
          </>
        )}
      </div>
    </div>
  );
}
