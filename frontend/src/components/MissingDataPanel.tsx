import { useState } from "react";
import { useStore } from "../store";
import { runMICE } from "../api";
import ResultExporter from "./ResultExporter";
import api from "../api";

export default function MissingDataPanel() {
  const session = useStore((s) => s.session);
  const columns = session?.columns ?? [];
  const sid = session?.session_id ?? "";

  // Missing column detection
  const preview = session?.preview ?? [];
  const allCols = columns;
  const missingInfo = allCols
    .map((c) => {
      const nMiss = preview.filter(
        (r) => r[c.name] === null || r[c.name] === undefined || r[c.name] === ""
      ).length;
      return {
        name: c.name,
        kind: c.kind,
        nMiss,
        pct: preview.length > 0 ? ((nMiss / preview.length) * 100).toFixed(1) : "0",
      };
    })
    .filter((m) => m.nMiss > 0)
    .sort((a, b) => b.nMiss - a.nMiss);

  // MICE state
  const [miceCols, setMiceCols] = useState<string[]>([]);
  const [miceN, setMiceN] = useState(5);
  const [miceIter, setMiceIter] = useState(10);
  const [miceMechanism, setMiceMechanism] = useState<"unknown" | "MCAR" | "MAR" | "MNAR">("unknown");
  const [miceResult, setMiceResult] = useState<any>(null);
  const [miceLoading, setMiceLoading] = useState(false);
  const [miceError, setMiceError] = useState<string | null>(null);
  const [miceApplied, setMiceApplied] = useState(false);

  if (!session) return <p className="text-gray-400 text-sm p-6">Upload data first.</p>;

  const handleMICE = async () => {
    if (miceCols.length === 0) { setMiceError("Select columns to impute"); return; }
    setMiceLoading(true); setMiceError(null); setMiceApplied(false);
    try {
      const res = await runMICE({
        session_id: sid,
        columns: miceCols,
        n_imputations: miceN,
        max_iter: miceIter,
        mechanism: miceMechanism,
      });
      setMiceResult(res.data);

      // Refresh session — backend already saved imputed data via store.save()
      const refresh = await api.get(`/api/stats/${sid}/refresh`);
      useStore.getState().setSession({ ...session, ...refresh.data });
      setMiceApplied(true);
    } catch (e: any) {
      setMiceError(e?.response?.data?.detail ?? "MICE failed");
    } finally {
      setMiceLoading(false);
    }
  };

  return (
    <div className="space-y-5 max-w-4xl mx-auto p-4">

      {/* ── Missing Data Overview ── */}
      <div className="border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-800">Missing Data Overview</h3>
          <p className="text-[11px] text-gray-400 mt-0.5">Variables with missing values in the dataset</p>
        </div>
        <div className="px-5 py-4">
          {missingInfo.length === 0 ? (
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-700">
              ✅ No missing values detected in any column.
            </div>
          ) : (
            <>
              <p className="text-xs text-gray-500 mb-3">
                <span className="font-semibold text-red-600">{missingInfo.length}</span> column(s) with missing values.
                Click to select for imputation.
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                {missingInfo.map((m) => {
                  const selected = miceCols.includes(m.name);
                  const pct = parseFloat(m.pct);
                  return (
                    <button
                      key={m.name}
                      onClick={() =>
                        setMiceCols((prev) =>
                          selected ? prev.filter((c) => c !== m.name) : [...prev, m.name]
                        )
                      }
                      className={`flex items-center justify-between gap-2 px-3 py-2 rounded-lg border text-xs transition-colors ${
                        selected
                          ? "border-indigo-400 bg-indigo-50 text-indigo-700"
                          : "border-gray-200 bg-white text-gray-600 hover:border-gray-300"
                      }`}
                    >
                      <div className="text-left">
                        <p className="font-medium truncate">{m.name}</p>
                        <p className="text-[10px] text-gray-400">{m.kind} · {m.nMiss} rows</p>
                      </div>
                      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full flex-shrink-0 ${
                        pct > 30 ? "bg-red-100 text-red-600" :
                        pct > 10 ? "bg-amber-100 text-amber-600" :
                        "bg-gray-100 text-gray-500"
                      }`}>{m.pct}%</span>
                    </button>
                  );
                })}
              </div>
              {miceCols.length > 0 && (
                <p className="text-[10px] text-indigo-500 mt-2">
                  {miceCols.length} column(s) selected for imputation
                </p>
              )}
            </>
          )}
        </div>
      </div>

      {/* ── Missing Mechanism ── */}
      <div className="border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-800">Missing Data Mechanism</h3>
          <p className="text-[11px] text-gray-400 mt-0.5">Affects the choice of imputation method</p>
        </div>
        <div className="px-5 py-4 space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {([
              { id: "unknown", icon: "?", label: "Unknown", desc: "Let AI assess" },
              { id: "MCAR", icon: "~", label: "MCAR", desc: "Missing Completely At Random" },
              { id: "MAR", icon: "#", label: "MAR", desc: "Missing At Random" },
              { id: "MNAR", icon: "!", label: "MNAR", desc: "Missing Not At Random" },
            ] as const).map(({ id, icon, label, desc }) => (
              <button key={id} onClick={() => setMiceMechanism(id)}
                className={`flex flex-col items-start gap-1 px-3 py-2.5 rounded-lg border text-left transition-colors ${
                  miceMechanism === id ? "border-indigo-400 bg-indigo-50" : "border-gray-200 bg-white hover:border-gray-300"
                }`}>
                <div className="flex items-center gap-1.5">
                  <span className={`w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold ${
                    miceMechanism === id ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-500"
                  }`}>{icon}</span>
                  <span className={`text-xs font-semibold ${miceMechanism === id ? "text-indigo-700" : "text-gray-700"}`}>{label}</span>
                </div>
                <span className="text-[10px] text-gray-400 leading-tight">{desc}</span>
              </button>
            ))}
          </div>
          {miceMechanism === "MNAR" && (
            <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-700">
              ⚠️ MNAR requires specialized methods. MICE assumes MAR — results may be biased. Consider sensitivity analysis.
            </div>
          )}
          {miceMechanism === "unknown" && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-[11px] text-blue-700">
              💡 If MCAR is rejected, MAR is commonly assumed and MICE is appropriate.
            </div>
          )}
        </div>
      </div>

      {/* ── MICE Imputation ── */}
      <div className="border border-indigo-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 bg-indigo-50 border-b border-indigo-100">
          <h3 className="text-sm font-semibold text-indigo-800">MICE Multiple Imputation</h3>
          <p className="text-[11px] text-indigo-400 mt-0.5">
            Impute missing values using chained equations — imputed data replaces missing values in the session
          </p>
        </div>
        <div className="px-5 py-4 space-y-4">
          <div className="flex gap-4 flex-wrap">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-500 font-medium">Imputations (m)</span>
              <input type="number" value={miceN} onChange={(e) => setMiceN(Number(e.target.value))} min={1} max={50}
                className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 w-24 focus:outline-none focus:border-indigo-400" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-500 font-medium">Max iterations</span>
              <input type="number" value={miceIter} onChange={(e) => setMiceIter(Number(e.target.value))} min={1} max={100}
                className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 w-24 focus:outline-none focus:border-indigo-400" />
            </label>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={handleMICE}
              disabled={miceLoading || miceCols.length === 0}
              className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
            >
              {miceLoading ? "Running MICE…" : "Apply MICE Imputation"}
            </button>
            {miceCols.length === 0 && (
              <p className="text-xs text-gray-400">Select columns above to impute</p>
            )}
            {miceError && <p className="text-xs text-red-500">{miceError}</p>}
          </div>

          {/* Applied confirmation */}
          {miceApplied && (
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-700 flex items-center gap-2">
              ✅ <span>Imputation applied — all subsequent analyses will use the completed dataset.</span>
            </div>
          )}

          {/* MICE result */}
          {miceResult && (
            <div className="space-y-3">
              {miceResult.result_text && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3 text-sm text-indigo-900">
                  {miceResult.result_text}
                </div>
              )}
              {miceResult.export_rows?.length > 1 && (
                <>
                  <div className="overflow-auto rounded-lg border border-gray-200">
                    <table className="text-xs w-full">
                      <thead>
                        <tr className="bg-gray-50">
                          {miceResult.export_rows[0].map((h: string, i: number) => (
                            <th key={i} className="px-3 py-1.5 text-left text-gray-500 font-medium">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {miceResult.export_rows.slice(1).map((row: any[], ri: number) => (
                          <tr key={ri} className="border-t border-gray-100">
                            {row.map((v: any, ci: number) => (
                              <td key={ci} className="px-3 py-1 text-gray-700">{v ?? "—"}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <ResultExporter
                    title="MICE_imputation"
                    headers={miceResult.export_rows[0]}
                    rows={miceResult.export_rows.slice(1)}
                  />
                </>
              )}
              {miceResult.r_code && (
                <details className="group">
                  <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600">R Code</summary>
                  <pre className="mt-1 bg-gray-900 text-green-300 text-[11px] rounded-lg p-3 overflow-x-auto">{miceResult.r_code}</pre>
                </details>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
