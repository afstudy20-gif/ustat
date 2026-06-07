import { useState } from "react";
import { useStore, isNumericKind } from "../store";
import { runMICE, runMCARTest, runImputationCompare, runMissingDiagnostics, fillBlanks } from "../api";
import ResultExporter from "./ResultExporter";
import api from "../api";
import { CleaningTab } from "./CleaningTab";

interface DiagCol { name: string; n_missing: number; pct: number; kind: string; is_numeric: boolean; depends_on: string[]; likely: string }
interface DiagResult { columns: DiagCol[]; overall_hint: string; recommendation: string; any_mar: boolean }

const errText = (e: unknown): string =>
  (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Request failed";

export default function MissingDataPanel() {
  const session = useStore((s) => s.session);
  const columns = session?.columns ?? [];
  const numCols = columns.filter((c) => isNumericKind(c.kind));
  const sid = session?.session_id ?? "";

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
    .filter((m) => m.nMiss > 0)
    .sort((a, b) => b.nMiss - a.nMiss);

  // Selection + MICE state
  const [selected, setSelected] = useState<string[]>([]);
  const [miceN, setMiceN] = useState(20);
  const [miceIter, setMiceIter] = useState(20);
  const [miceSeed, setMiceSeed] = useState(42);
  const [miceMechanism, setMiceMechanism] = useState<"unknown" | "MCAR" | "MAR" | "MNAR">("unknown");
  const [miceResult, setMiceResult] = useState<any>(null);
  const [miceLoading, setMiceLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null); // per-row action in flight
  const [err, setErr] = useState<string | null>(null);

  // Diagnostics state
  const [diag, setDiag] = useState<DiagResult | null>(null);
  const [mcar, setMcar] = useState<any>(null);
  const [compare, setCompare] = useState<any>(null);

  if (!session) return <p className="text-gray-400 text-sm p-6">Upload data first.</p>;

  const refresh = async () => {
    const r = await api.get(`/api/stats/${sid}/refresh`);
    useStore.getState().setSession({ ...useStore.getState().session!, ...r.data });
    useStore.setState((s) => ({ dataVersion: s.dataVersion + 1 }));
  };

  const toggle = (name: string) =>
    setSelected((p) => (p.includes(name) ? p.filter((c) => c !== name) : [...p, name]));

  // Per-row quick imputation (acts immediately on one column).
  const quickFill = async (col: string, method: "__mean__" | "__median__" | "__mode__" | "__mice__") => {
    setBusy(`${col}:${method}`); setErr(null);
    try {
      await fillBlanks(sid, col, method);
      await refresh();
    } catch (e: unknown) {
      setErr(errText(e));
    } finally {
      setBusy(null);
    }
  };

  const runDiagnostics = async () => {
    setBusy("diag"); setErr(null); setDiag(null); setMcar(null);
    try {
      const [d, m] = await Promise.allSettled([
        runMissingDiagnostics(sid),
        // No `columns` → Little's test runs across all numeric variables (it
        // needs ≥2 to assess missingness patterns).
        runMCARTest({ session_id: sid }),
      ]);
      if (d.status === "fulfilled") setDiag(d.value.data);
      if (m.status === "fulfilled") setMcar(m.value.data);
      if (d.status === "rejected" && m.status === "rejected") setErr(errText(d.reason));
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

  const handleMICE = async () => {
    if (selected.length === 0) { setErr("Select columns to impute"); return; }
    setMiceLoading(true); setErr(null); setMiceResult(null);
    try {
      const res = await runMICE({
        session_id: sid, columns: selected, n_imputations: miceN,
        max_iter: miceIter, random_state: miceSeed, mechanism: miceMechanism,
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
  const QuickBtn = ({ col, method, label, show }: { col: string; method: "__mean__" | "__median__" | "__mode__" | "__mice__"; label: string; show: boolean }) =>
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
    <div className="space-y-5 max-w-4xl mx-auto p-4">
      {err && <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-600">{err}</div>}

      {/* ── Overview — list ── */}
      <div className="border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-800">Missing Data Overview</h3>
            <p className="text-[11px] text-gray-400 mt-0.5">Tick rows for MICE / comparison, or impute a single column inline.</p>
          </div>
          {missingInfo.length > 0 && (
            <button onClick={() => setSelected(selected.length === missingInfo.length ? [] : missingInfo.map((m) => m.name))}
              className="text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-gray-100">
              {selected.length === missingInfo.length ? "Clear all" : "Select all"}
            </button>
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
                <th className="px-3 py-2">Variable</th>
                <th className="px-3 py-2">Type</th>
                <th className="px-3 py-2 text-right">Missing</th>
                <th className="px-3 py-2 w-28">%</th>
                <th className="px-3 py-2 text-right">Quick impute</th>
              </tr>
            </thead>
            <tbody>
              {missingInfo.map((m) => (
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
              <button onClick={runDiagnostics} disabled={busy === "diag"}
                className="text-xs px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50">
                {busy === "diag" ? "Analyzing…" : "Analyze missingness"}
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
                      <span className="font-semibold">Little's MCAR test:</span> χ²={mcar.statistic}, df={mcar.df}, p={mcar.p}.{" "}
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
              <h3 className="text-sm font-semibold text-indigo-800">MICE Multiple Imputation (selected columns)</h3>
              <p className="text-[11px] text-indigo-400 mt-0.5">
                Fills the session in place. For valid inference, prefer the model panels' MICE option (m datasets + Rubin's-rules pooling).
              </p>
            </div>
            <div className="px-5 py-4 space-y-4">
              <div className="flex gap-4 flex-wrap">
                {[["Imputations (m)", miceN, setMiceN, 1, 100], ["Max iterations", miceIter, setMiceIter, 1, 100], ["Seed", miceSeed, setMiceSeed, 0, 999999]].map(([lab, val, set, mn, mx]: any) => (
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
                  {miceLoading ? "Running MICE…" : `Apply MICE to ${selected.length || ""} column(s)`}
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
              {miceResult?.export_rows?.length > 1 && (
                <>
                  <div className="overflow-auto rounded-lg border border-gray-200">
                    <table className="text-xs w-full">
                      <thead><tr className="bg-gray-50">{miceResult.export_rows[0].map((h: string, i: number) => <th key={i} className="px-3 py-1.5 text-left text-gray-500 font-medium">{h}</th>)}</tr></thead>
                      <tbody>{miceResult.export_rows.slice(1).map((row: any[], ri: number) => (
                        <tr key={ri} className="border-t border-gray-100">{row.map((v: any, ci: number) => <td key={ci} className="px-3 py-1 text-gray-700">{v ?? "—"}</td>)}</tr>
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
                        {compare.comparisons.flatMap((cmp: any) => cmp.columns.map((c: any) => (
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

      {/* ── Cleaning ── */}
      <CleaningTab sessionId={sid} columns={columns} numCols={numCols} />
    </div>
  );
}
