import { useState } from "react";
import { useStore } from "../store";
import { runIV2SLS } from "../api";
import ResultExporter from "./ResultExporter";

type Method = "iv";  // mediation / target-trial added in later phases

function MultiPick({ label, accent, exclude, value, onChange, columns }: {
  label: string; accent: string; exclude: string[]; value: string[];
  onChange: (v: string[]) => void; columns: string[];
}) {
  const toggle = (c: string) => onChange(value.includes(c) ? value.filter((x) => x !== c) : [...value, c]);
  return (
    <div>
      <label className="text-xs text-gray-400 block mb-1">{label}</label>
      <div className="text-xs border border-gray-300 rounded-lg p-2 max-h-28 overflow-y-auto space-y-0.5">
        {columns.filter((c) => !exclude.includes(c)).map((c) => (
          <label key={c} className="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" className={accent} checked={value.includes(c)} onChange={() => toggle(c)} />
            <span className="text-gray-700">{c}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

function IVTab() {
  const session = useStore((s) => s.session);
  const cols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";

  const [outcome, setOutcome] = useState("");
  const [endogenous, setEndogenous] = useState("");
  const [instruments, setInstruments] = useState<string[]>([]);
  const [covariates, setCovariates] = useState<string[]>([]);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await runIV2SLS({ session_id: sid, outcome, endogenous, instruments, covariates });
      setResult(r.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "IV estimation failed.");
    } finally { setLoading(false); }
  };
  const canRun = sid && outcome && endogenous && instruments.length > 0 && !loading;

  const Tile = ({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) => (
    <div className="rounded-xl border border-gray-200 bg-white p-3">
      <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`text-xl font-semibold mt-1 ${tone ?? "text-gray-900"}`}>{value}</div>
      {sub && <div className="text-[11px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
  const p = (v: number) => (v < 0.001 ? "<0.001" : v.toFixed(4));

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">Instrumental Variable (2SLS)</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Estimates a causal effect when the exposure is <b>endogenous</b> (confounded by unmeasured factors)
            using an <b>instrument</b> — a variable that affects the exposure but the outcome only through it.
            A valid instrument must be <b>relevant</b> (first-stage F ≥ 10) and <b>exogenous</b>.
          </p>
        </div>
        <div className="panel space-y-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Outcome (continuous)</label>
            <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>
              {cols.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Endogenous exposure</label>
            <select className="select w-full" value={endogenous} onChange={(e) => setEndogenous(e.target.value)}>
              <option value="">— select —</option>
              {cols.filter((c) => c !== outcome).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <MultiPick label="Instrument(s)" accent="accent-emerald-500" columns={cols}
            exclude={[outcome, endogenous, ...covariates]} value={instruments} onChange={setInstruments} />
          <MultiPick label="Covariates (exogenous controls)" accent="accent-indigo-500" columns={cols}
            exclude={[outcome, endogenous, ...instruments]} value={covariates} onChange={setCovariates} />
          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>
            {loading ? "Running…" : "Run 2SLS"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-4">
        {!result ? (
          <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm text-center">
            Pick an outcome, an endogenous exposure, and an instrument — then run 2SLS.
          </div>
        ) : (
          <>
            <div className={`panel border ${result.first_stage.weak_instruments ? "border-amber-300 bg-amber-50" : "border-emerald-300 bg-emerald-50"}`}>
              <p className="text-sm text-gray-800 leading-relaxed">{result.result_text}</p>
              <div className="text-[11px] text-gray-500 mt-2">n = {result.n}</div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Tile label="IV effect (2SLS)" value={result.iv_estimate.estimate.toFixed(4)}
                sub={`95% CI ${result.iv_estimate.ci_low.toFixed(3)}–${result.iv_estimate.ci_high.toFixed(3)} · p=${p(result.iv_estimate.p)}`}
                tone="text-emerald-600" />
              <Tile label="Naive OLS" value={result.ols_estimate.estimate.toFixed(4)} sub={`p=${p(result.ols_estimate.p)}`} />
              <Tile label="First-stage F" value={result.first_stage.f_stat.toFixed(1)}
                sub={result.first_stage.weak_instruments ? "WEAK (<10)" : "adequate (≥10)"}
                tone={result.first_stage.weak_instruments ? "text-amber-600" : "text-emerald-600"} />
              <Tile label="Wu-Hausman" value={`p=${p(result.wu_hausman.p)}`}
                sub={result.wu_hausman.endogenous ? "endogenous → use IV" : "no strong endogeneity"} />
            </div>
            {result.sargan && (
              <div className="text-xs text-gray-600">
                Sargan over-identification: χ²={result.sargan.stat} (df={result.sargan.df}), p={p(result.sargan.p)} —
                {result.sargan.valid ? " instruments jointly valid." : " instrument validity in doubt."}
              </div>
            )}
            <ResultExporter title="iv_2sls" />
          </>
        )}
      </div>
    </div>
  );
}

export default function CausalPanel() {
  const [method] = useState<Method>("iv");
  // Sub-tab bar kept for the upcoming mediation / target-trial methods.
  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {([["iv", "Instrumental Variable (2SLS)"]] as const).map(([id, label]) => (
          <span key={id}
            className={`px-3 py-1 rounded-md text-xs font-medium ${method === id ? "bg-white text-indigo-700 shadow-sm border border-gray-200" : "text-gray-500"}`}>
            {label}
          </span>
        ))}
      </div>
      {method === "iv" && <IVTab />}
    </div>
  );
}
