import { useState } from "react";
import { useStore } from "../store";
import { runIV2SLS, runMediation, runTargetTrial } from "../api";
import ResultExporter from "./ResultExporter";

type Method = "iv" | "mediation" | "target";

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

function MediationTab() {
  const session = useStore((s) => s.session);
  const cols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";

  const [outcome, setOutcome] = useState("");
  const [treatment, setTreatment] = useState("");
  const [mediator, setMediator] = useState("");
  const [covariates, setCovariates] = useState<string[]>([]);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await runMediation({ session_id: sid, outcome, treatment, mediator, covariates, bootstrap: 1000 });
      setResult(r.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "Mediation analysis failed.");
    } finally { setLoading(false); }
  };
  const distinct = new Set([outcome, treatment, mediator].filter(Boolean)).size === 3;
  const canRun = sid && distinct && !loading;

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
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">Causal Mediation (X → M → Y)</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Splits the total effect of the treatment into the part that runs <b>through the mediator</b>
            (indirect, ACME = a·b) and the <b>direct</b> part (ADE). Significance is judged by a
            <b> bootstrap CI</b> on the indirect effect, not by p-values of separate coefficients.
            Continuous mediator &amp; outcome.
          </p>
        </div>
        <div className="panel space-y-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Outcome Y (continuous)</label>
            <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>{cols.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Treatment / exposure X</label>
            <select className="select w-full" value={treatment} onChange={(e) => setTreatment(e.target.value)}>
              <option value="">— select —</option>{cols.filter((c) => c !== outcome).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Mediator M (continuous)</label>
            <select className="select w-full" value={mediator} onChange={(e) => setMediator(e.target.value)}>
              <option value="">— select —</option>{cols.filter((c) => c !== outcome && c !== treatment).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <MultiPick label="Covariates (optional)" accent="accent-indigo-500" columns={cols}
            exclude={[outcome, treatment, mediator]} value={covariates} onChange={setCovariates} />
          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>
            {loading ? "Running…" : "Run mediation"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-4">
        {!result ? (
          <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm text-center">
            Pick outcome, treatment, and mediator — then decompose the effect.
          </div>
        ) : (
          <>
            <div className={`panel border ${result.acme_significant ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <p className="text-sm text-gray-800 leading-relaxed">{result.result_text}</p>
              <div className="text-[11px] text-gray-500 mt-2">n = {result.n}</div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Tile label="Indirect (ACME)" value={result.effects.acme.toFixed(4)}
                sub={result.effects.acme_ci ? `95% CI ${result.effects.acme_ci[0]} to ${result.effects.acme_ci[1]}` : undefined}
                tone={result.acme_significant ? "text-emerald-600" : "text-gray-900"} />
              <Tile label="Direct (ADE)" value={result.effects.ade.toFixed(4)}
                sub={result.effects.ade_ci ? `95% CI ${result.effects.ade_ci[0]} to ${result.effects.ade_ci[1]}` : undefined} />
              <Tile label="Total effect" value={result.effects.total.toFixed(4)} />
              <Tile label="Proportion mediated"
                value={result.effects.proportion_mediated != null ? (result.effects.proportion_mediated * 100).toFixed(1) + "%" : "—"}
                sub={`Sobel p=${p(result.sobel.p)}`} />
            </div>
            <div className="text-xs text-gray-500">
              Paths: a (X→M) = {result.paths.a}, b (M→Y) = {result.paths.b}, c′ (direct) = {result.paths.c_prime}.
            </div>
            <ResultExporter title="mediation" />
          </>
        )}
      </div>
    </div>
  );
}

function TargetTrialTab() {
  const session = useStore((s) => s.session);
  const cols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";

  const [treatment, setTreatment] = useState("");
  const [outcome, setOutcome] = useState("");
  const [confounders, setConfounders] = useState<string[]>([]);
  const [elig, setElig] = useState<{ column: string; op: string; value: string }[]>([]);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const eligibility = elig
        .filter((e) => e.column && e.value !== "")
        .map((e) => ({ column: e.column, op: e.op, value: Number(e.value) }));
      const r = await runTargetTrial({ session_id: sid, treatment, outcome, confounders, eligibility, bootstrap: 400 });
      setResult(r.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "Target-trial emulation failed.");
    } finally { setLoading(false); }
  };
  const canRun = sid && treatment && outcome && treatment !== outcome && confounders.length > 0 && !loading;

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">Target Trial Emulation</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Mimics an RCT from observational data: apply explicit <b>eligibility</b>, define a baseline
            <b> time zero</b>, then estimate the ITT-style effect with <b>stabilized IPTW</b> on the
            baseline confounders. Returns the effect, covariate balance, and the 7-component protocol.
          </p>
        </div>
        <div className="panel space-y-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Treatment / arm (binary 0/1)</label>
            <select className="select w-full" value={treatment} onChange={(e) => setTreatment(e.target.value)}>
              <option value="">— select —</option>{cols.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Outcome (binary 0/1)</label>
            <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>{cols.filter((c) => c !== treatment).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          <MultiPick label="Baseline confounders" accent="accent-indigo-500" columns={cols}
            exclude={[treatment, outcome]} value={confounders} onChange={setConfounders} />
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-gray-400">Eligibility (optional)</label>
              <button className="text-[10px] text-indigo-600 hover:text-indigo-800"
                onClick={() => setElig([...elig, { column: "", op: "gte", value: "" }])}>+ add</button>
            </div>
            {elig.map((e, i) => (
              <div key={i} className="flex gap-1 mb-1">
                <select className="select flex-1 text-xs" value={e.column}
                  onChange={(ev) => setElig(elig.map((x, j) => j === i ? { ...x, column: ev.target.value } : x))}>
                  <option value="">col</option>{cols.map((c) => <option key={c}>{c}</option>)}
                </select>
                <select className="select text-xs w-14" value={e.op}
                  onChange={(ev) => setElig(elig.map((x, j) => j === i ? { ...x, op: ev.target.value } : x))}>
                  {["eq", "ne", "gt", "lt", "gte", "lte"].map((o) => <option key={o}>{o}</option>)}
                </select>
                <input className="select text-xs w-16" type="number" value={e.value} placeholder="val"
                  onChange={(ev) => setElig(elig.map((x, j) => j === i ? { ...x, value: ev.target.value } : x))} />
                <button className="text-red-400 text-xs px-1" onClick={() => setElig(elig.filter((_, j) => j !== i))}>✕</button>
              </div>
            ))}
          </div>
          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>
            {loading ? "Running…" : "Emulate target trial"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-4">
        {!result ? (
          <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm text-center">
            Define treatment, outcome, confounders (and optional eligibility), then emulate the trial.
          </div>
        ) : (
          <>
            <div className={`panel border ${result.effect.significant ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <p className="text-sm text-gray-800 leading-relaxed">{result.result_text}</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="rounded-xl border border-gray-200 bg-white p-3">
                <div className="text-[10px] uppercase tracking-wider text-gray-500">Risk difference</div>
                <div className={`text-xl font-semibold mt-1 ${result.effect.significant ? "text-emerald-600" : "text-gray-900"}`}>
                  {result.effect.risk_difference > 0 ? "+" : ""}{result.effect.risk_difference}
                </div>
                <div className="text-[11px] text-gray-500 mt-0.5">{result.effect.rd_ci ? `95% CI ${result.effect.rd_ci[0]} to ${result.effect.rd_ci[1]}` : ""}</div>
              </div>
              <div className="rounded-xl border border-gray-200 bg-white p-3">
                <div className="text-[10px] uppercase tracking-wider text-gray-500">Risk ratio</div>
                <div className="text-xl font-semibold mt-1 text-gray-900">{result.effect.risk_ratio ?? "—"}</div>
                <div className="text-[11px] text-gray-500 mt-0.5">{result.effect.rr_ci ? `95% CI ${result.effect.rr_ci[0]}–${result.effect.rr_ci[1]}` : ""}</div>
              </div>
              <div className="rounded-xl border border-gray-200 bg-white p-3">
                <div className="text-[10px] uppercase tracking-wider text-gray-500">Risk treated / control</div>
                <div className="text-xl font-semibold mt-1 text-gray-900">{result.effect.risk_treated} / {result.effect.risk_control}</div>
              </div>
              <div className="rounded-xl border border-gray-200 bg-white p-3">
                <div className="text-[10px] uppercase tracking-wider text-gray-500">Cohort</div>
                <div className="text-xl font-semibold mt-1 text-gray-900">{result.n_analyzed}</div>
                <div className="text-[11px] text-gray-500 mt-0.5">of {result.n_screened} screened · balance {result.balanced ? "✓" : "⚠"}</div>
              </div>
            </div>
            <div className="panel">
              <div className="text-xs font-semibold text-gray-600 mb-1">Target-trial protocol</div>
              <table className="text-xs w-full">
                <tbody>
                  {Object.entries(result.protocol).map(([k, v]) => (
                    <tr key={k} className="border-t border-gray-100">
                      <td className="py-1 pr-3 font-medium text-gray-500 capitalize align-top whitespace-nowrap">{k.replace(/_/g, " ")}</td>
                      <td className="py-1 text-gray-700">{Array.isArray(v) ? (v as string[]).join("; ") : String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="text-[11px] text-amber-700">⚠ {result.caveats.join(" ")}</div>
            <ResultExporter title="target_trial" />
          </>
        )}
      </div>
    </div>
  );
}

export default function CausalPanel() {
  const [method, setMethod] = useState<Method>("iv");
  const tabs: [Method, string][] = [
    ["iv", "Instrumental Variable (2SLS)"],
    ["mediation", "Mediation (X→M→Y)"],
    ["target", "Target Trial Emulation"],
  ];
  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {tabs.map(([id, label]) => (
          <button key={id} onClick={() => setMethod(id)}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
              method === id ? "bg-white text-indigo-700 shadow-sm border border-gray-200" : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
            }`}>
            {label}
          </button>
        ))}
      </div>
      {method === "iv" ? <IVTab /> : method === "mediation" ? <MediationTab /> : <TargetTrialTab />}
    </div>
  );
}
