import { useState, useRef } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { runMICE, runFineGray, runEValue, runLandmark } from "../api";
import ResultExporter from "./ResultExporter";
import PlotExporter from "./PlotExporter";

// ── Helpers ──────────────────────────────────────────────────────────────────

function Section({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3.5 bg-gray-50 hover:bg-gray-100 transition-colors text-left">
        <div>
          <h3 className="text-sm font-semibold text-gray-800">{title}</h3>
          <p className="text-[11px] text-gray-400 mt-0.5">{description}</p>
        </div>
        <span className="text-gray-400 text-lg">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="px-5 py-4 space-y-4 border-t border-gray-100">{children}</div>}
    </div>
  );
}

function VarSelect({ label, value, onChange, columns, kinds }: {
  label: string; value: string; onChange: (v: string) => void;
  columns: { name: string; kind: string }[]; kinds?: string[];
}) {
  const filtered = kinds ? columns.filter((c) => kinds.includes(c.kind)) : columns;
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-gray-500 font-medium">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
        <option value="">— select —</option>
        {filtered.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
      </select>
    </label>
  );
}

function MultiSelect({ label, columns, selected, onChange, kinds }: {
  label: string; columns: { name: string; kind: string }[];
  selected: string[]; onChange: (v: string[]) => void; kinds?: string[];
}) {
  const filtered = kinds ? columns.filter((c) => kinds.includes(c.kind)) : columns;
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-gray-500 font-medium">{label}</span>
      <select multiple value={selected} onChange={(e) => onChange(Array.from(e.target.selectedOptions, (o) => o.value))}
        className="text-xs border border-gray-300 rounded-lg px-2 py-1 bg-white h-28 focus:outline-none focus:border-indigo-400">
        {filtered.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
      </select>
    </label>
  );
}

function RunButton({ onClick, loading, label }: { onClick: () => void; loading: boolean; label: string }) {
  return (
    <button onClick={onClick} disabled={loading}
      className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
      {loading ? "Running…" : label}
    </button>
  );
}

function ResultBlock({ result }: { result: any }) {
  if (!result) return null;
  return (
    <div className="space-y-3 mt-3">
      {/* Result text */}
      {result.result_text && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3 text-sm text-indigo-900">
          {result.result_text}
        </div>
      )}

      {/* Assumptions */}
      {result.assumptions?.length > 0 && (
        <div className="space-y-1">
          {result.assumptions.map((a: any, i: number) => (
            <div key={i} className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg ${a.met ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
              <span>{a.met ? "✓" : "⚠"}</span>
              <span className="font-medium">{a.name}</span>
              <span className="text-gray-500">— {a.detail}</span>
            </div>
          ))}
        </div>
      )}

      {/* Export rows as table */}
      {result.export_rows?.length > 1 && (
        <div className="overflow-auto rounded-lg border border-gray-200">
          <table className="text-xs w-full">
            <thead>
              <tr className="bg-gray-50">
                {result.export_rows[0].map((h: string, i: number) => (
                  <th key={i} className="px-3 py-1.5 text-left text-gray-500 font-medium border-r border-gray-100 last:border-r-0">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.export_rows.slice(1).map((row: any[], ri: number) => (
                <tr key={ri} className="border-t border-gray-100">
                  {row.map((v: any, ci: number) => (
                    <td key={ci} className="px-3 py-1 text-gray-700 border-r border-gray-100 last:border-r-0">{v ?? "—"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* R code */}
      {result.r_code && (
        <details className="group">
          <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600">R Code</summary>
          <pre className="mt-1 bg-gray-900 text-green-300 text-[11px] rounded-lg p-3 overflow-x-auto">{result.r_code}</pre>
        </details>
      )}

      {/* Exporter */}
      {result.export_rows?.length > 1 && (
        <ResultExporter title={result.test ?? "result"} headers={result.export_rows[0]} rows={result.export_rows.slice(1)} />
      )}
    </div>
  );
}

// ── Main Panel ───────────────────────────────────────────────────────────────

export default function SurvivalAdvancedPanel() {
  const session = useStore((s) => s.session);
  const columns = session?.columns ?? [];
  const sid = session?.session_id ?? "";

  const fgPlotRef = useRef<any>(null);
  const lmPlotRef = useRef<any>(null);

  // MICE state
  const [miceCols, setMiceCols] = useState<string[]>([]);
  const [miceN, setMiceN] = useState(5);
  const [miceIter, setMiceIter] = useState(10);
  const [miceResult, setMiceResult] = useState<any>(null);
  const [miceLoading, setMiceLoading] = useState(false);
  const [miceError, setMiceError] = useState<string | null>(null);

  // Fine-Gray state
  const [fgDuration, setFgDuration] = useState("");
  const [fgEvent, setFgEvent] = useState("");
  const [fgInterest, setFgInterest] = useState(1);
  const [fgGroup, setFgGroup] = useState("");
  const [fgResult, setFgResult] = useState<any>(null);
  const [fgLoading, setFgLoading] = useState(false);
  const [fgError, setFgError] = useState<string | null>(null);

  // E-value state
  const [evEst, setEvEst] = useState("");
  const [evLo, setEvLo] = useState("");
  const [evHi, setEvHi] = useState("");
  const [evType, setEvType] = useState("OR");
  const [evP0, setEvP0] = useState("0.1");
  const [evResult, setEvResult] = useState<any>(null);
  const [evLoading, setEvLoading] = useState(false);
  const [evError, setEvError] = useState<string | null>(null);

  // Landmark state
  const [lmDuration, setLmDuration] = useState("");
  const [lmEvent, setLmEvent] = useState("");
  const [lmTime, setLmTime] = useState("");
  const [lmGroup, setLmGroup] = useState("");
  const [lmPreds, setLmPreds] = useState<string[]>([]);
  const [lmResult, setLmResult] = useState<any>(null);
  const [lmLoading, setLmLoading] = useState(false);
  const [lmError, setLmError] = useState<string | null>(null);

  if (!session) return <p className="text-gray-400 text-sm p-6">Upload data first.</p>;

  // ── MICE handler
  const handleMICE = async () => {
    if (miceCols.length === 0) { setMiceError("Select columns to impute"); return; }
    setMiceLoading(true); setMiceError(null);
    try {
      const res = await runMICE({ session_id: sid, columns: miceCols, n_imputations: miceN, max_iter: miceIter });
      setMiceResult(res.data);
      // Refresh session after imputation
      const { default: api } = await import("../api");
      const refresh = await api.get(`/api/stats/${sid}/refresh`);
      useStore.getState().setSession({ ...session, ...refresh.data });
    } catch (e: any) { setMiceError(e?.response?.data?.detail ?? "MICE failed"); }
    finally { setMiceLoading(false); }
  };

  // ── Fine-Gray handler
  const handleFineGray = async () => {
    if (!fgDuration || !fgEvent) { setFgError("Select duration and event columns"); return; }
    setFgLoading(true); setFgError(null);
    try {
      const res = await runFineGray({
        session_id: sid, duration_col: fgDuration, event_col: fgEvent,
        event_of_interest: fgInterest, group_col: fgGroup || undefined,
      });
      setFgResult(res.data);
    } catch (e: any) { setFgError(e?.response?.data?.detail ?? "Fine-Gray failed"); }
    finally { setFgLoading(false); }
  };

  // ── E-value handler
  const handleEValue = async () => {
    if (!evEst || !evLo || !evHi) { setEvError("Enter estimate and confidence interval"); return; }
    setEvLoading(true); setEvError(null);
    try {
      const res = await runEValue({
        estimate: parseFloat(evEst), ci_low: parseFloat(evLo), ci_high: parseFloat(evHi),
        measure_type: evType, baseline_risk: parseFloat(evP0),
      });
      setEvResult(res.data);
    } catch (e: any) { setEvError(e?.response?.data?.detail ?? "E-value failed"); }
    finally { setEvLoading(false); }
  };

  // ── Landmark handler
  const handleLandmark = async () => {
    if (!lmDuration || !lmEvent || !lmTime) { setLmError("Select duration, event, and landmark time"); return; }
    setLmLoading(true); setLmError(null);
    try {
      const res = await runLandmark({
        session_id: sid, duration_col: lmDuration, event_col: lmEvent,
        landmark_time: parseFloat(lmTime), group_col: lmGroup || undefined,
        predictors: lmPreds.length > 0 ? lmPreds : undefined,
      });
      setLmResult(res.data);
    } catch (e: any) { setLmError(e?.response?.data?.detail ?? "Landmark analysis failed"); }
    finally { setLmLoading(false); }
  };

  return (
    <div className="space-y-3 max-w-5xl mx-auto">
      {/* ── MICE ── */}
      <Section title="MICE Multiple Imputation" description="Impute missing values using chained equations (Predictive Mean Matching)">
        <div className="grid grid-cols-3 gap-3">
          <MultiSelect label="Columns to impute" columns={columns} selected={miceCols} onChange={setMiceCols} kinds={["numeric"]} />
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
        <div className="flex items-center gap-3">
          <RunButton onClick={handleMICE} loading={miceLoading} label="Run MICE" />
          {miceError && <p className="text-xs text-red-500">{miceError}</p>}
        </div>
        <ResultBlock result={miceResult} />
      </Section>

      {/* ── Fine-Gray ── */}
      <Section title="Fine-Gray Competing Risks" description="Cumulative incidence function with competing events (Aalen-Johansen)">
        <div className="grid grid-cols-4 gap-3">
          <VarSelect label="Duration" value={fgDuration} onChange={setFgDuration} columns={columns} kinds={["numeric"]} />
          <VarSelect label="Event (0=censor, 1,2..=events)" value={fgEvent} onChange={setFgEvent} columns={columns} />
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Event of interest</span>
            <input type="number" value={fgInterest} onChange={(e) => setFgInterest(Number(e.target.value))} min={1}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 w-20 focus:outline-none focus:border-indigo-400" />
          </label>
          <VarSelect label="Group (optional)" value={fgGroup} onChange={setFgGroup} columns={columns} kinds={["categorical"]} />
        </div>
        <div className="flex items-center gap-3">
          <RunButton onClick={handleFineGray} loading={fgLoading} label="Run Fine-Gray" />
          {fgError && <p className="text-xs text-red-500">{fgError}</p>}
        </div>
        {fgResult?.plot && (
          <div className="relative" ref={fgPlotRef}>
            <Plot data={fgResult.plot.data} layout={fgResult.plot.layout} config={{ responsive: true }} style={{ width: "100%", height: 400 }} />
            <PlotExporter plotRef={fgPlotRef} title="CIF" />
          </div>
        )}
        <ResultBlock result={fgResult} />
      </Section>

      {/* ── E-value ── */}
      <Section title="E-value (Unmeasured Confounding)" description="Quantify the minimum confounding strength to explain away an observed effect">
        <div className="grid grid-cols-5 gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Measure</span>
            <select value={evType} onChange={(e) => setEvType(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
              <option value="OR">OR</option>
              <option value="HR">HR</option>
              <option value="RR">RR</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Estimate</span>
            <input type="number" step="0.01" value={evEst} onChange={(e) => setEvEst(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" placeholder="e.g. 2.5" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">CI Low</span>
            <input type="number" step="0.01" value={evLo} onChange={(e) => setEvLo(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" placeholder="e.g. 1.2" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">CI High</span>
            <input type="number" step="0.01" value={evHi} onChange={(e) => setEvHi(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" placeholder="e.g. 5.1" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Baseline risk (p₀)</span>
            <input type="number" step="0.01" value={evP0} onChange={(e) => setEvP0(e.target.value)} min={0.01} max={0.99}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" />
          </label>
        </div>
        <div className="flex items-center gap-3">
          <RunButton onClick={handleEValue} loading={evLoading} label="Calculate E-value" />
          {evError && <p className="text-xs text-red-500">{evError}</p>}
        </div>
        {evResult && (
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3 text-center">
              <p className="text-[10px] text-indigo-400 uppercase tracking-wider font-semibold">E-value (point)</p>
              <p className="text-3xl font-bold text-indigo-700 mt-1">{evResult.evalue_point}</p>
            </div>
            <div className="bg-violet-50 border border-violet-200 rounded-xl px-4 py-3 text-center">
              <p className="text-[10px] text-violet-400 uppercase tracking-wider font-semibold">E-value (CI)</p>
              <p className="text-3xl font-bold text-violet-700 mt-1">{evResult.evalue_ci}</p>
            </div>
          </div>
        )}
        {evResult?.interpretation && (
          <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-700">
            {evResult.interpretation}
          </div>
        )}
        <ResultBlock result={evResult} />
      </Section>

      {/* ── Landmark ── */}
      <Section title="Landmark Survival Analysis" description="Survival analysis conditional on surviving beyond a landmark time point">
        <div className="grid grid-cols-4 gap-3">
          <VarSelect label="Duration" value={lmDuration} onChange={setLmDuration} columns={columns} kinds={["numeric"]} />
          <VarSelect label="Event (0/1)" value={lmEvent} onChange={setLmEvent} columns={columns} />
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Landmark time</span>
            <input type="number" step="1" value={lmTime} onChange={(e) => setLmTime(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" placeholder="e.g. 30" />
          </label>
          <VarSelect label="Group (optional)" value={lmGroup} onChange={setLmGroup} columns={columns} kinds={["categorical"]} />
        </div>
        <MultiSelect label="Predictors for Cox (optional)" columns={columns} selected={lmPreds} onChange={setLmPreds} kinds={["numeric"]} />
        <div className="flex items-center gap-3">
          <RunButton onClick={handleLandmark} loading={lmLoading} label="Run Landmark" />
          {lmError && <p className="text-xs text-red-500">{lmError}</p>}
        </div>
        {lmResult?.plot && (
          <div className="relative" ref={lmPlotRef}>
            <Plot data={lmResult.plot.data} layout={lmResult.plot.layout} config={{ responsive: true }} style={{ width: "100%", height: 400 }} />
            <PlotExporter plotRef={lmPlotRef} title="Landmark_KM" />
          </div>
        )}
        {lmResult?.cox_results && lmResult.cox_results.length > 0 && !lmResult.cox_results[0].error && (
          <div className="overflow-auto rounded-lg border border-gray-200">
            <table className="text-xs w-full">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-3 py-1.5 text-left text-gray-500">Variable</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">HR</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">95% CI</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">p</th>
                </tr>
              </thead>
              <tbody>
                {lmResult.cox_results.map((r: any, i: number) => (
                  <tr key={i} className="border-t border-gray-100">
                    <td className="px-3 py-1 text-gray-700 font-medium">{r.variable}</td>
                    <td className="px-3 py-1 text-gray-700">{r.HR}</td>
                    <td className="px-3 py-1 text-gray-500">{r.ci_low} – {r.ci_high}</td>
                    <td className={`px-3 py-1 ${r.p < 0.05 ? "text-indigo-600 font-semibold" : "text-gray-500"}`}>
                      {r.p < 0.001 ? "<0.001" : r.p?.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <ResultBlock result={lmResult} />
      </Section>
    </div>
  );
}
