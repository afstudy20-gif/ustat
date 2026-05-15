import { useState, useEffect, useRef } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { runFineGray, runEValue, runLandmark, runKM, runCox } from "../api";
import { usePlotLayout, usePalette, useTraceDefaults } from "../plotStyle";
import ResultExporter from "./ResultExporter";
import PlotExporter from "./PlotExporter";
import { Tip } from "./Tip";

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

  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const traceDefaults = useTraceDefaults();

  const fgPlotRef = useRef<any>(null);
  const lmPlotRef = useRef<any>(null);

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

  // KM state
  const [kmDuration, setKmDuration] = useState("");
  const [kmEvent, setKmEvent] = useState("");
  const [kmGroup, setKmGroup] = useState("");
  const [kmResult, setKmResult] = useState<any>(null);
  const [kmLoading, setKmLoading] = useState(false);
  const [kmError, setKmError] = useState<string | null>(null);
  const kmPlotRef = useRef<any>(null);
  // KM screening state
  const [kmScanResult, setKmScanResult] = useState<any[]>([]);
  const [kmScanLoading, setKmScanLoading] = useState(false);
  // Group rename state
  const [kmGroupLabels, setKmGroupLabels] = useState<Record<string, string>>({});
  const [kmCustomGroupTitle, setKmCustomGroupTitle] = useState("");
  const [kmCustomDurationTitle, setKmCustomDurationTitle] = useState("");
  const [kmContextMenu, setKmContextMenu] = useState<{ type: "item"|"groupTitle"|"durationTitle"; group?: string; x: number; y: number } | null>(null);
  const [kmRenameValue, setKmRenameValue] = useState("");

  // Cox state
  const [coxDuration, setCoxDuration] = useState("");
  const [coxEvent, setCoxEvent] = useState("");
  const [coxPreds, setCoxPreds] = useState<string[]>([]);
  const [coxInteractions, setCoxInteractions] = useState<Array<[string, string]>>([]);
  const [coxIxA, setCoxIxA] = useState<string>("");
  const [coxIxB, setCoxIxB] = useState<string>("");
  const [coxResult, setCoxResult] = useState<any>(null);
  const [coxLoading, setCoxLoading] = useState(false);
  const [coxError, setCoxError] = useState<string | null>(null);
  // Cox univariable screening state
  const [coxScanResult, setCoxScanResult] = useState<any[]>([]);
  const [coxScanLoading, setCoxScanLoading] = useState(false);

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

  // ── Fine-Gray handler
  const handleFineGray = async () => {
    if (!fgDuration || !fgEvent) { setFgError("Select duration and event columns"); return; }
    setFgResult(null); setFgError(null); setFgLoading(true);
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
    setEvResult(null); setEvError(null); setEvLoading(true);
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
    setLmResult(null); setLmError(null); setLmLoading(true);
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

  // ── Auto-clear stale results when their inputs change. Without this the
  // user perceives a re-Run as 'nothing happened' because the previous
  // result panel stays on screen even when the new fetch fails or returns
  // a near-identical table.
  useEffect(() => { setFgResult(null); setFgError(null); }, [fgDuration, fgEvent, fgInterest, fgGroup]);
  useEffect(() => { setEvResult(null); setEvError(null); }, [evEst, evLo, evHi, evType, evP0]);
  useEffect(() => { setLmResult(null); setLmError(null); }, [lmDuration, lmEvent, lmTime, lmGroup, lmPreds]);
  useEffect(() => { setKmResult(null); setKmError(null); }, [kmDuration, kmEvent, kmGroup]);
  useEffect(() => { setCoxResult(null); setCoxError(null); }, [coxDuration, coxEvent, coxPreds, coxInteractions]);

  return (
    <div className="space-y-3 max-w-5xl mx-auto">
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
            <Plot data={fgResult.plot.data} layout={{ ...fgResult.plot.layout, ...baseLayout, title: fgResult.plot.layout.title }} config={{ responsive: true }} style={{ width: "100%", height: 400 }} />
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
            <Plot data={lmResult.plot.data} layout={{ ...lmResult.plot.layout, ...baseLayout, title: lmResult.plot.layout.title }} config={{ responsive: true }} style={{ width: "100%", height: 400 }} />
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

      {/* ── Kaplan-Meier ── */}
      <Section title="Kaplan-Meier Survival" description="Visualise time-to-event data with survival curves and log-rank test">
        <div className="grid grid-cols-3 gap-3">
          <VarSelect label="Duration (time)" value={kmDuration} onChange={setKmDuration} columns={columns} kinds={["numeric"]} />
          <VarSelect label="Event (0/1)" value={kmEvent} onChange={setKmEvent} columns={columns} />
          <VarSelect label="Group (optional)" value={kmGroup} onChange={setKmGroup} columns={columns} kinds={["categorical"]} />
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <RunButton onClick={async () => {
            if (!kmDuration || !kmEvent) { setKmError("Select duration and event columns"); return; }
            setKmResult(null); setKmError(null); setKmLoading(true);
            try {
              const res = await runKM({ session_id: sid, duration_col: kmDuration, event_col: kmEvent, group_col: kmGroup || undefined });
              setKmResult(res.data);
            } catch (e: any) { setKmError(e?.response?.data?.detail ?? "KM failed"); }
            finally { setKmLoading(false); }
          }} loading={kmLoading} label="Run Kaplan-Meier" />

          {/* Log-rank screening button */}
          {kmDuration && kmEvent && (
            <button
              disabled={kmScanLoading}
              onClick={async () => {
                const catCols = columns.filter((c) => c.kind === "categorical").map((c) => c.name);
                if (catCols.length === 0) return;
                setKmScanLoading(true);
                const results: any[] = [];
                for (const col of catCols) {
                  try {
                    const res = await runKM({ session_id: sid, duration_col: kmDuration, event_col: kmEvent, group_col: col });
                    results.push({
                      variable: col,
                      groups: res.data.groups?.length ?? 0,
                      logrank_p: res.data.logrank?.p ?? null,
                      chi2: res.data.logrank?.chi2 ?? null,
                    });
                  } catch { results.push({ variable: col, groups: null, logrank_p: null, chi2: null }); }
                }
                results.sort((a, b) => (a.logrank_p ?? 1) - (b.logrank_p ?? 1));
                setKmScanResult(results);
                setKmScanLoading(false);
              }}
              className="px-3 py-1.5 text-xs font-medium border border-indigo-300 text-indigo-600 rounded-lg hover:bg-indigo-50 disabled:opacity-50 transition-colors"
            >
              {kmScanLoading ? "Taranıyor…" : "🔍 Log-rank Tarama"}
            </button>
          )}
          {kmError && <p className="text-xs text-red-500">{kmError}</p>}
        </div>

        {/* KM scan results */}
        {kmScanResult.length > 0 && (
          <div className="rounded-lg border border-gray-200 overflow-auto">
            <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 flex items-center justify-between">
              <p className="text-xs font-semibold text-gray-600">Log-rank Tarama — Tüm Kategorik Değişkenler</p>
              <button onClick={() => setKmScanResult([])} className="text-[10px] text-gray-400 hover:text-red-500">✕ Kapat</button>
            </div>
            <table className="text-xs w-full">
              <thead><tr className="bg-gray-50">
                <th className="px-3 py-1.5 text-left text-gray-500">Değişken</th>
                <th className="px-3 py-1.5 text-left text-gray-500">Grup sayısı</th>
                <th className="px-3 py-1.5 text-left text-gray-500">χ²</th>
                <th className="px-3 py-1.5 text-left text-gray-500">Log-rank p</th>
                <th className="px-3 py-1.5 text-left text-gray-500"></th>
              </tr></thead>
              <tbody>
                {kmScanResult.map((r, i) => (
                  <tr key={i} className={`border-t border-gray-100 ${r.logrank_p !== null && r.logrank_p < 0.05 ? "bg-indigo-50" : ""}`}>
                    <td className="px-3 py-1 font-medium text-gray-700">{r.variable}</td>
                    <td className="px-3 py-1 text-gray-500">{r.groups ?? "—"}</td>
                    <td className="px-3 py-1 text-gray-500">{r.chi2 != null ? r.chi2.toFixed(3) : "—"}</td>
                    <td className={`px-3 py-1 font-semibold ${r.logrank_p !== null && r.logrank_p < 0.05 ? "text-indigo-700" : "text-gray-500"}`}>
                      {r.logrank_p !== null ? (r.logrank_p < 0.001 ? "<0.001" : r.logrank_p.toFixed(4)) : "hata"}
                    </td>
                    <td className="px-3 py-1">
                      {r.logrank_p !== null && r.logrank_p < 0.05 && (
                        <button onClick={() => { setKmGroup(r.variable); setKmScanResult([]); }}
                          className="text-[10px] text-indigo-500 hover:text-indigo-700 underline">
                          Grafiğe ekle
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {kmResult?.groups && (() => {
          // Resolve group display name: custom rename > value_labels > raw value
          const groupColMeta = columns.find((c) => c.name === kmGroup);
          const vLabels = groupColMeta?.value_labels ?? {};
          const resolveGroupName = (raw: string) =>
            kmGroupLabels[raw] ?? vLabels[raw] ?? raw;

          return (
          <>
            <div className="relative" ref={kmPlotRef}>
              <Plot
                data={kmResult.groups.map((g: any, i: number) => ({
                  x: g.curve.map((p: any) => p.time),
                  y: g.curve.map((p: any) => p.survival),
                  type: "scatter", mode: "lines",
                  name: kmGroup
                    ? `${kmCustomGroupTitle || kmGroup} = ${resolveGroupName(String(g.group))}`
                    : resolveGroupName(String(g.group)),
                  line: { width: traceDefaults.lineWidth, color: pal[i % pal.length] },
                }))}
                layout={{
                  ...baseLayout,
                  title: { text: "Kaplan-Meier Survival Curves", font: { color: "#374151", size: 13 } },
                  xaxis: {
                    ...(baseLayout.xaxis as any),
                     // Using custom duration title if available
                    title: { text: `Time (${kmCustomDurationTitle || kmDuration})` },
                  },
                  yaxis: {
                    ...(baseLayout.yaxis as any),
                    title: { text: "Survival Probability" },
                    range: [0, 1.05],
                    tickformat: ".0%",
                  },
                  margin: { t: 44, r: 20, b: 56, l: 68 }, showlegend: true,
                  legend: { title: { text: kmCustomGroupTitle || kmGroup || "Group" } },
                }}
                config={{ responsive: true }} style={{ width: "100%", height: 400 }}
              />
              <PlotExporter plotRef={kmPlotRef} title="KM_Survival" />
            </div>

            {/* Compact Group summary table & Log-rank test */}
            <div className="overflow-hidden rounded-lg border border-gray-200 shadow-sm mt-2">
              <table className="text-xs w-full bg-white">
                <thead><tr className="bg-gray-50 border-b border-gray-200 bg-opacity-70">
                  <th className="px-3 py-1.5 text-left text-[9px] font-bold text-gray-500 uppercase tracking-wider cursor-context-menu"
                    onContextMenu={(e) => {
                      e.preventDefault();
                      setKmContextMenu({ type: "groupTitle", x: e.clientX, y: e.clientY });
                      setKmRenameValue(kmCustomGroupTitle || kmGroup || "Group");
                    }}
                  >
                    {kmCustomGroupTitle || kmGroup || "Group"}
                    <span className="ml-1 font-normal text-gray-400 normal-case tracking-normal">(right-click to rename)</span>
                  </th>
                  <th className="px-3 py-1.5 text-right text-[9px] font-bold text-gray-500 uppercase tracking-wider">N</th>
                  <th className="px-3 py-1.5 text-right text-[9px] font-bold text-gray-500 uppercase tracking-wider">Events</th>
                  <th className="px-3 py-1.5 text-right text-[9px] font-bold text-gray-500 uppercase tracking-wider cursor-context-menu"
                    onContextMenu={(e) => {
                       e.preventDefault();
                       setKmContextMenu({ type: "durationTitle", x: e.clientX, y: e.clientY });
                       setKmRenameValue(kmCustomDurationTitle || kmDuration);
                    }}
                  >
                    Median ({kmCustomDurationTitle || kmDuration})
                    <span className="ml-1 font-normal text-gray-400 normal-case tracking-normal">(right-click to rename)</span>
                  </th>
                </tr></thead>
                <tbody className="divide-y divide-gray-100">
                  {kmResult.groups.map((g: any, i: number) => {
                    const label = resolveGroupName(String(g.group));
                    const isRenamed = label !== String(g.group);
                    return (
                      <tr key={i} className="hover:bg-indigo-50/30 transition-colors"
                        onContextMenu={(e) => {
                          e.preventDefault();
                          setKmContextMenu({ type: "item", group: String(g.group), x: e.clientX, y: e.clientY });
                          setKmRenameValue(label);
                        }}
                      >
                        <td className="px-3 py-1 cursor-context-menu select-none">
                          <span className="inline-flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                              style={{ background: pal[i % pal.length] }} />
                            <span className="text-[11px] font-medium text-gray-700">{label}</span>
                            {isRenamed && (
                              <span className="text-[9px] text-gray-400">({g.group})</span>
                            )}
                          </span>
                        </td>
                        <td className="px-3 py-1 text-[11px] font-medium text-gray-600 text-right">{g.n}</td>
                        <td className="px-3 py-1 text-[11px] font-medium text-gray-600 text-right">{g.events}</td>
                        <td className="px-3 py-1 text-[11px] font-medium text-gray-600 text-right">{g.median_survival ?? "NR"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {/* Log-rank test embedded as a cohesive footer inside the same block */}
              {kmResult.logrank && (
                <div className={`px-3 py-1.5 text-[11px] border-t font-medium flex items-center justify-between ${kmResult.logrank.p < 0.05 ? "bg-indigo-50 border-indigo-100 text-indigo-700" : "bg-gray-50 border-gray-100 text-gray-500"}`}>
                  <span>Log-rank test</span>
                  <span>
                    p = {kmResult.logrank.p < 0.001 ? "<0.001" : kmResult.logrank.p?.toFixed(4)}
                    {kmResult.logrank.p < 0.05 ? " (Significant difference)" : " (No difference)"}
                  </span>
                </div>
              )}

              {/* Right-click context menu (absolute body mount replacement) */}
              {kmContextMenu && (
                <>
                  <div 
                    className="fixed inset-0 z-40" 
                    onClick={() => setKmContextMenu(null)} 
                    onContextMenu={(e) => { e.preventDefault(); setKmContextMenu(null); }} 
                  />
                  <div
                    className="fixed bg-white border border-gray-200 rounded-lg shadow-xl z-50 p-3 min-w-[200px]"
                    style={{ top: kmContextMenu.y, left: kmContextMenu.x }}
                  >
                  <p className="text-[10px] text-gray-400 mb-1.5 font-medium uppercase tracking-wide">
                    {kmContextMenu.type === "item" && `Rename group "${kmContextMenu.group}"`}
                    {kmContextMenu.type === "groupTitle" && `Rename Legend Title`}
                    {kmContextMenu.type === "durationTitle" && `Rename Time Axis Title`}
                  </p>
                  <input
                    autoFocus
                    className="w-full text-xs border border-gray-300 rounded px-2 py-1 mb-2 focus:outline-none focus:border-indigo-400"
                    value={kmRenameValue}
                    onChange={(e) => setKmRenameValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        if (kmContextMenu.type === "item" && kmContextMenu.group) {
                          setKmGroupLabels((prev) => ({ ...prev, [kmContextMenu.group!]: kmRenameValue }));
                        } else if (kmContextMenu.type === "groupTitle") {
                          setKmCustomGroupTitle(kmRenameValue);
                        } else if (kmContextMenu.type === "durationTitle") {
                          setKmCustomDurationTitle(kmRenameValue);
                        }
                        setKmContextMenu(null);
                      }
                      if (e.key === "Escape") setKmContextMenu(null);
                    }}
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => {
                        if (kmContextMenu.type === "item" && kmContextMenu.group) {
                          setKmGroupLabels((prev) => ({ ...prev, [kmContextMenu.group!]: kmRenameValue }));
                        } else if (kmContextMenu.type === "groupTitle") {
                          setKmCustomGroupTitle(kmRenameValue);
                        } else if (kmContextMenu.type === "durationTitle") {
                          setKmCustomDurationTitle(kmRenameValue);
                        }
                        setKmContextMenu(null);
                      }}
                      className="flex-1 text-xs bg-indigo-600 text-white rounded px-2 py-1 hover:bg-indigo-700"
                    >Save</button>
                    <button
                      onClick={() => {
                        if (kmContextMenu.type === "item" && kmContextMenu.group) {
                          const next = { ...kmGroupLabels };
                          delete next[kmContextMenu.group];
                          setKmGroupLabels(next);
                        } else if (kmContextMenu.type === "groupTitle") {
                          setKmCustomGroupTitle("");
                        } else if (kmContextMenu.type === "durationTitle") {
                          setKmCustomDurationTitle("");
                        }
                        setKmContextMenu(null);
                      }}
                      className="text-xs text-gray-400 hover:text-red-500 px-2 py-1"
                    >Reset</button>
                  </div>
                </div>
                </>
              )}
            </div>
          </>
          );
        })()}
      </Section>

      {/* ── Cox PH ── */}
      <Section title="Cox Proportional Hazards" description="Regression for time-to-event data — outputs Hazard Ratios (HR)">
        <div className="grid grid-cols-2 gap-3">
          <VarSelect label="Duration (time)" value={coxDuration} onChange={setCoxDuration} columns={columns} kinds={["numeric"]} />
          <VarSelect label="Event (0/1)" value={coxEvent} onChange={setCoxEvent} columns={columns} />
        </div>

        {/* Checkbox predictor list */}
        <div>
          <p className="text-xs text-gray-500 font-medium mb-1.5">
            Predictors
            {coxPreds.length > 0 && (
              <span className="ml-2 text-indigo-600 font-semibold">{coxPreds.length} selected</span>
            )}
            {coxPreds.length > 0 && (
              <button onClick={() => setCoxPreds([])} className="ml-2 text-[10px] text-gray-400 hover:text-red-500 underline">clear</button>
            )}
          </p>
          <div className="border border-gray-200 rounded-lg overflow-y-auto max-h-36 divide-y divide-gray-100">
            {columns.map((c) => (
              <label key={c.name} className={`flex items-center gap-2 px-3 py-1.5 cursor-pointer transition-colors text-xs
                ${coxPreds.includes(c.name) ? "bg-indigo-50 text-indigo-800" : "hover:bg-gray-50 text-gray-700"}`}>
                <input
                  type="checkbox"
                  checked={coxPreds.includes(c.name)}
                  onChange={(e) => {
                    if (e.target.checked) setCoxPreds([...coxPreds, c.name]);
                    else setCoxPreds(coxPreds.filter((p) => p !== c.name));
                  }}
                  className="accent-indigo-500"
                />
                <span className="font-medium">{c.name}</span>
                <span className="text-[10px] text-gray-400 ml-auto">{c.kind}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Interactions — pair selector (only meaningful when ≥2 predictors ticked) */}
        {coxPreds.length >= 2 && (
          <div>
            <p className="text-xs text-gray-500 font-medium mb-1.5 flex items-center gap-1">
              Interactions
              <Tip wide text="Add pairwise interaction terms to the linear Cox model — e.g. LDL × AGE. Numeric × numeric is the element-wise product, numeric × categorical expands across every dummy of the categorical, categorical × categorical multiplies every dummy pair. Use sparingly: each extra term costs degrees of freedom. The output table reports each interaction as 'A:B' with its own HR and p-value." />
              {coxInteractions.length > 0 && (
                <span className="ml-1 text-indigo-600 font-semibold">{coxInteractions.length} added</span>
              )}
            </p>
            <div className="flex flex-wrap items-center gap-1.5">
              <select
                value={coxIxA}
                onChange={(e) => setCoxIxA(e.target.value)}
                className="select text-xs py-1"
              >
                <option value="">First term…</option>
                {coxPreds.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
              <span className="text-gray-400 text-xs">×</span>
              <select
                value={coxIxB}
                onChange={(e) => setCoxIxB(e.target.value)}
                className="select text-xs py-1"
              >
                <option value="">Second term…</option>
                {coxPreds.filter((p) => p !== coxIxA).map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
              <button
                onClick={() => {
                  if (!coxIxA || !coxIxB || coxIxA === coxIxB) return;
                  const exists = coxInteractions.some(
                    ([a, b]) => (a === coxIxA && b === coxIxB) || (a === coxIxB && b === coxIxA),
                  );
                  if (exists) return;
                  setCoxInteractions([...coxInteractions, [coxIxA, coxIxB]]);
                  setCoxIxA(""); setCoxIxB("");
                }}
                disabled={!coxIxA || !coxIxB || coxIxA === coxIxB}
                className="text-xs px-2 py-1 rounded border border-indigo-300 text-indigo-600 hover:bg-indigo-50 disabled:opacity-40 transition-colors"
              >
                + Add
              </button>
            </div>
            {coxInteractions.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-1.5">
                {coxInteractions.map(([a, b], i) => (
                  <span
                    key={`${a}:${b}:${i}`}
                    className="inline-flex items-center gap-1 bg-amber-50 border border-amber-200 text-amber-800 text-[11px] rounded px-2 py-0.5"
                  >
                    {a} × {b}
                    <button
                      onClick={() => setCoxInteractions(coxInteractions.filter((_, idx) => idx !== i))}
                      className="text-amber-500 hover:text-red-500"
                      title="Remove"
                    >
                      ×
                    </button>
                  </span>
                ))}
                <button
                  onClick={() => setCoxInteractions([])}
                  className="text-[10px] text-gray-400 hover:text-red-500 underline"
                >
                  clear all
                </button>
              </div>
            )}
          </div>
        )}

        <div className="flex items-center gap-3 flex-wrap">
          <RunButton onClick={async () => {
            if (!coxDuration || !coxEvent || coxPreds.length === 0) { setCoxError("Select duration, event, and at least one predictor"); return; }
            setCoxResult(null); setCoxError(null); setCoxLoading(true);
            try {
              const res = await runCox({
                session_id: sid,
                duration_col: coxDuration,
                event_col: coxEvent,
                predictors: coxPreds,
                interactions: coxInteractions.length > 0 ? coxInteractions : undefined,
              });
              setCoxResult(res.data);
            } catch (e: any) { setCoxError(e?.response?.data?.detail ?? "Cox failed"); }
            finally { setCoxLoading(false); }
          }} loading={coxLoading} label="Run Cox Regression" />

          {/* Univariable screening button */}
          {coxDuration && coxEvent && coxPreds.length > 0 && (
            <button
              disabled={coxScanLoading}
              onClick={async () => {
                setCoxScanLoading(true);
                const results: any[] = [];
                for (const pred of coxPreds) {
                  try {
                    const res = await runCox({ session_id: sid, duration_col: coxDuration, event_col: coxEvent, predictors: [pred] });
                    const coef = res.data.coefficients?.[0];
                    results.push({
                      variable: pred,
                      hr: coef?.hr ?? null,
                      hr_ci_low: coef?.hr_ci_low ?? null,
                      hr_ci_high: coef?.hr_ci_high ?? null,
                      p: coef?.p ?? null,
                      n: res.data.n ?? null,
                    });
                  } catch { results.push({ variable: pred, hr: null, hr_ci_low: null, hr_ci_high: null, p: null, n: null }); }
                }
                results.sort((a, b) => (a.p ?? 1) - (b.p ?? 1));
                setCoxScanResult(results);
                setCoxScanLoading(false);
              }}
              className="px-3 py-1.5 text-xs font-medium border border-indigo-300 text-indigo-600 rounded-lg hover:bg-indigo-50 disabled:opacity-50 transition-colors"
            >
              {coxScanLoading ? "Scanning…" : "🔍 Univariable Scan"}
              <Tip wide text="Fits a separate Cox PH model for each predictor on its own (Surv(time,event) ~ X), then ranks them by p-value. Use it to triage which candidates are worth carrying into the multivariable model. Common rule: take everything with univariable p < 0.10 forward and let the multivariable fit decide what stays — variables can lose or gain significance once you adjust for confounders (e.g. SMOKER univariable p ≈ 1.0 but p < 0.001 once AGE is controlled for)." />
            </button>
          )}
          {coxError && <p className="text-xs text-red-500">{coxError}</p>}
        </div>

        {/* Cox univariable scan results */}
        {coxScanResult.length > 0 && (
          <div className="rounded-lg border border-gray-200 overflow-auto">
            <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 flex items-center justify-between">
              <p className="text-xs font-semibold text-gray-600">Univariable Cox Scan — One Predictor at a Time</p>
              <button onClick={() => setCoxScanResult([])} className="text-[10px] text-gray-400 hover:text-red-500">✕ Close</button>
            </div>
            <table className="text-xs w-full">
              <thead><tr className="bg-gray-50">
                <th className="px-3 py-1.5 text-left text-gray-500">Variable</th>
                <th className="px-3 py-1.5 text-left text-gray-500">N (events)</th>
                <th className="px-3 py-1.5 text-left text-gray-500">HR</th>
                <th className="px-3 py-1.5 text-left text-gray-500">95% CI</th>
                <th className="px-3 py-1.5 text-left text-gray-500">p</th>
              </tr></thead>
              <tbody>
                {coxScanResult.map((r, i) => (
                  <tr key={i} className={`border-t border-gray-100 ${r.p !== null && r.p < 0.05 ? "bg-indigo-50" : ""}`}>
                    <td className="px-3 py-1 font-medium text-gray-700">{r.variable}</td>
                    <td className="px-3 py-1 text-gray-500">{r.n ?? "—"}</td>
                    <td className="px-3 py-1 font-semibold text-gray-800">{r.hr != null ? r.hr.toFixed(3) : "—"}</td>
                    <td className="px-3 py-1 text-gray-500">
                      {r.hr_ci_low != null ? `${r.hr_ci_low.toFixed(3)} – ${r.hr_ci_high.toFixed(3)}` : "—"}
                    </td>
                    <td className={`px-3 py-1 font-semibold ${r.p !== null && r.p < 0.05 ? "text-indigo-700" : "text-gray-500"}`}>
                      {r.p !== null ? (r.p < 0.001 ? "<0.001" : r.p.toFixed(4)) : "error"}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="border-t border-gray-200 bg-amber-50">
                  <td colSpan={5} className="px-3 py-1.5 text-[10px] text-amber-700">
                    💡 Add variables with p &lt; 0.10 to the multivariable Cox model — the adjustment may reveal effects that look null on their own.
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        )}

        {coxResult?.coefficients && (
          <div className="overflow-auto rounded-lg border border-gray-200">
            <table className="text-xs w-full">
              <thead>
                {/* Model summary row */}
                <tr className="bg-indigo-50 border-b border-indigo-100">
                  <td colSpan={2} className="px-3 py-1.5 text-indigo-700 font-medium">
                    N (events): <span className="font-bold">{coxResult.n}</span>
                  </td>
                  <td colSpan={2} className="px-3 py-1.5 text-indigo-700 font-medium">
                    C-index: <span className="font-bold">{coxResult.concordance?.toFixed(4)}</span>
                  </td>
                  <td colSpan={2} className="px-3 py-1.5 text-indigo-700 font-medium">
                    Log-Likelihood: <span className="font-bold">{coxResult.log_likelihood?.toFixed(2)}</span>
                  </td>
                </tr>
                <tr className="bg-gray-50">
                  <th className="px-3 py-1.5 text-left text-gray-500">Variable</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">B</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">SE</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">HR</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">95% CI</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">p</th>
                </tr>
              </thead>
              <tbody>
                {coxResult.coefficients.map((c: any, i: number) => (
                  <tr key={i} className="border-t border-gray-100 hover:bg-gray-50">
                    <td className="px-3 py-1 font-medium text-gray-700">{c.variable}</td>
                    <td className="px-3 py-1 text-gray-600">{c.log_hr?.toFixed(4)}</td>
                    <td className="px-3 py-1 text-gray-600">{c.se?.toFixed(4)}</td>
                    <td className="px-3 py-1 font-semibold text-gray-800">{c.hr?.toFixed(4)}</td>
                    <td className="px-3 py-1 text-gray-500">{c.hr_ci_low?.toFixed(3)} – {c.hr_ci_high?.toFixed(3)}</td>
                    <td className={`px-3 py-1 ${c.p < 0.05 ? "text-indigo-600 font-semibold" : "text-gray-500"}`}>
                      {c.p < 0.001 ? "<0.001" : c.p?.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}
