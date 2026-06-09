import { useState } from "react";
import { useStore } from "../store";
import { runIV2SLS, runMediation, runTargetTrial, runDiD, runRDD, runDAGAdjustment } from "../api";
import ResultExporter from "./ResultExporter";
import { fmtP } from "../lib/format";

type Method = "iv" | "mediation" | "target" | "did" | "rdd" | "dag";

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
                sub={`95% CI ${result.iv_estimate.ci_low.toFixed(3)}–${result.iv_estimate.ci_high.toFixed(3)} · p=${fmtP(result.iv_estimate.p)}`}
                tone="text-emerald-600" />
              <Tile label="Naive OLS" value={result.ols_estimate.estimate.toFixed(4)} sub={`p=${fmtP(result.ols_estimate.p)}`} />
              <Tile label="First-stage F" value={result.first_stage.f_stat.toFixed(1)}
                sub={result.first_stage.weak_instruments ? "WEAK (<10)" : "adequate (≥10)"}
                tone={result.first_stage.weak_instruments ? "text-amber-600" : "text-emerald-600"} />
              <Tile label="Wu-Hausman" value={`p=${fmtP(result.wu_hausman.p)}`}
                sub={result.wu_hausman.endogenous ? "endogenous → use IV" : "no strong endogeneity"} />
            </div>
            {result.sargan && (
              <div className="text-xs text-gray-600">
                Sargan over-identification: χ²={result.sargan.stat} (df={result.sargan.df}), p={fmtP(result.sargan.p)} —
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
                sub={`Sobel p=${fmtP(result.sobel.p)}`} />
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

function DiDTab() {
  const session = useStore((s) => s.session);
  const cols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";
  const [outcome, setOutcome] = useState("");
  const [groupCol, setGroupCol] = useState("");
  const [timeCol, setTimeCol] = useState("");
  const [covariates, setCovariates] = useState<string[]>([]);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await runDiD({ session_id: sid, outcome, group_col: groupCol, time_col: timeCol, covariates });
      setResult(r.data);
    } catch (e: any) { setError(e?.response?.data?.detail ?? "DiD failed."); } finally { setLoading(false); }
  };
  const canRun = sid && new Set([outcome, groupCol, timeCol].filter(Boolean)).size === 3 && !loading;
  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">Difference-in-Differences</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Effect = the extra change in the <b>treated</b> group from pre→post over and above the
            <b> control</b> group's change (the group×time interaction). Assumes <b>parallel trends</b>.
            Continuous outcome; group &amp; time coded 0/1.
          </p>
        </div>
        <div className="panel space-y-3">
          <div><label className="text-xs text-gray-400 block mb-1">Outcome (continuous)</label>
            <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>{cols.map((c) => <option key={c}>{c}</option>)}</select></div>
          <div><label className="text-xs text-gray-400 block mb-1">Group (0=control, 1=treated)</label>
            <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
              <option value="">— select —</option>{cols.filter((c) => c !== outcome).map((c) => <option key={c}>{c}</option>)}</select></div>
          <div><label className="text-xs text-gray-400 block mb-1">Time (0=pre, 1=post)</label>
            <select className="select w-full" value={timeCol} onChange={(e) => setTimeCol(e.target.value)}>
              <option value="">— select —</option>{cols.filter((c) => c !== outcome && c !== groupCol).map((c) => <option key={c}>{c}</option>)}</select></div>
          <MultiPick label="Covariates (optional)" accent="accent-indigo-500" columns={cols}
            exclude={[outcome, groupCol, timeCol]} value={covariates} onChange={setCovariates} />
          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>{loading ? "Running…" : "Run DiD"}</button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>
      <div className="flex-1 min-w-0 space-y-4">
        {!result ? <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm">Pick outcome, group, and time.</div> : (
          <>
            <div className={`panel border ${result.significant ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <p className="text-sm text-gray-800 leading-relaxed">{result.result_text}</p></div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <div className="rounded-xl border border-gray-200 bg-white p-3"><div className="text-[10px] uppercase tracking-wider text-gray-500">DiD estimate</div>
                <div className={`text-xl font-semibold mt-1 ${result.significant ? "text-emerald-600" : "text-gray-900"}`}>{result.did_estimate > 0 ? "+" : ""}{result.did_estimate}</div>
                <div className="text-[11px] text-gray-500 mt-0.5">95% CI {result.ci_low} to {result.ci_high} · p={fmtP(result.p)}</div></div>
              <div className="rounded-xl border border-gray-200 bg-white p-3"><div className="text-[10px] uppercase tracking-wider text-gray-500">Treated change</div><div className="text-xl font-semibold mt-1 text-gray-900">{result.treated_change > 0 ? "+" : ""}{result.treated_change}</div></div>
              <div className="rounded-xl border border-gray-200 bg-white p-3"><div className="text-[10px] uppercase tracking-wider text-gray-500">Control change</div><div className="text-xl font-semibold mt-1 text-gray-900">{result.control_change > 0 ? "+" : ""}{result.control_change}</div></div>
            </div>
            <div className="text-xs text-gray-500">Cell means — control: {result.cell_means.control_pre} → {result.cell_means.control_post}; treated: {result.cell_means.treated_pre} → {result.cell_means.treated_post}.</div>
            <ResultExporter title="did" />
          </>
        )}
      </div>
    </div>
  );
}

function RDDTab() {
  const session = useStore((s) => s.session);
  const cols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";
  const [outcome, setOutcome] = useState("");
  const [running, setRunning] = useState("");
  const [cutoff, setCutoff] = useState("");
  const [bandwidth, setBandwidth] = useState("");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await runRDD({ session_id: sid, outcome, running, cutoff: Number(cutoff), bandwidth: bandwidth ? Number(bandwidth) : null });
      setResult(r.data);
    } catch (e: any) { setError(e?.response?.data?.detail ?? "RDD failed."); } finally { setLoading(false); }
  };
  const canRun = sid && outcome && running && outcome !== running && cutoff !== "" && !loading;
  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">Regression Discontinuity (sharp)</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            When treatment is assigned by a <b>cutoff</b> on a running variable (e.g. a score ≥ threshold),
            the jump in the outcome at the cutoff is the local causal effect (<b>LATE</b>). Local-linear fit,
            triangular kernel within the bandwidth.
          </p>
        </div>
        <div className="panel space-y-3">
          <div><label className="text-xs text-gray-400 block mb-1">Outcome</label>
            <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>{cols.map((c) => <option key={c}>{c}</option>)}</select></div>
          <div><label className="text-xs text-gray-400 block mb-1">Running / forcing variable</label>
            <select className="select w-full" value={running} onChange={(e) => setRunning(e.target.value)}>
              <option value="">— select —</option>{cols.filter((c) => c !== outcome).map((c) => <option key={c}>{c}</option>)}</select></div>
          <div><label className="text-xs text-gray-400 block mb-1">Cutoff</label>
            <input className="select w-full" type="number" value={cutoff} onChange={(e) => setCutoff(e.target.value)} /></div>
          <div><label className="text-xs text-gray-400 block mb-1">Bandwidth (blank = auto)</label>
            <input className="select w-full" type="number" value={bandwidth} onChange={(e) => setBandwidth(e.target.value)} placeholder="auto" /></div>
          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>{loading ? "Running…" : "Run RDD"}</button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>
      <div className="flex-1 min-w-0 space-y-4">
        {!result ? <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm">Pick outcome, running variable, and cutoff.</div> : (
          <>
            <div className={`panel border ${result.significant ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <p className="text-sm text-gray-800 leading-relaxed">{result.result_text}</p></div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="rounded-xl border border-gray-200 bg-white p-3"><div className="text-[10px] uppercase tracking-wider text-gray-500">LATE at cutoff</div>
                <div className={`text-xl font-semibold mt-1 ${result.significant ? "text-emerald-600" : "text-gray-900"}`}>{result.late > 0 ? "+" : ""}{result.late}</div>
                <div className="text-[11px] text-gray-500 mt-0.5">95% CI {result.ci_low} to {result.ci_high} · p={fmtP(result.p)}</div></div>
              <div className="rounded-xl border border-gray-200 bg-white p-3"><div className="text-[10px] uppercase tracking-wider text-gray-500">Bandwidth</div><div className="text-xl font-semibold mt-1 text-gray-900">±{result.bandwidth}</div></div>
              <div className="rounded-xl border border-gray-200 bg-white p-3"><div className="text-[10px] uppercase tracking-wider text-gray-500">N in bandwidth</div><div className="text-xl font-semibold mt-1 text-gray-900">{result.n_in_bandwidth}</div><div className="text-[11px] text-gray-500 mt-0.5">{result.n_left} below / {result.n_right} above</div></div>
            </div>
            <ResultExporter title="rdd" />
          </>
        )}
      </div>
    </div>
  );
}

function DAGTab() {
  const [edgeText, setEdgeText] = useState("Z -> T\nZ -> Y\nT -> M\nM -> Y\nT -> C\nY -> C");
  const [treatment, setTreatment] = useState("T");
  const [outcome, setOutcome] = useState("Y");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const edges = edgeText.split("\n").map((l) => l.trim()).filter(Boolean)
        .map((l) => l.split(/->|→/).map((s) => s.trim())).filter((e) => e.length === 2 && e[0] && e[1]);
      const r = await runDAGAdjustment({ edges, treatment, outcome });
      setResult(r.data);
    } catch (e: any) { setError(e?.response?.data?.detail ?? "DAG analysis failed."); } finally { setLoading(false); }
  };
  const roleColor = (r: string) => r === "confounder" ? "text-amber-600" : r === "mediator" ? "text-blue-600" : r === "collider" ? "text-red-600" : "text-gray-500";
  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">DAG Backdoor Analysis</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Enter your causal graph as edges (<code>A -&gt; B</code>, one per line). Returns each node's role
            (<b>confounder / mediator / collider</b>) and the <b>minimal adjustment set</b> via the backdoor
            criterion — which variables to control for, and which to <b>never</b> adjust for.
          </p>
        </div>
        <div className="panel space-y-3">
          <div><label className="text-xs text-gray-400 block mb-1">Edges (one per line, A -&gt; B)</label>
            <textarea className="select w-full h-32 font-mono text-xs" value={edgeText} onChange={(e) => setEdgeText(e.target.value)} /></div>
          <div className="flex gap-2">
            <div className="flex-1"><label className="text-xs text-gray-400 block mb-1">Treatment</label>
              <input className="select w-full" value={treatment} onChange={(e) => setTreatment(e.target.value)} /></div>
            <div className="flex-1"><label className="text-xs text-gray-400 block mb-1">Outcome</label>
              <input className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)} /></div>
          </div>
          <button className="btn-primary w-full" onClick={run} disabled={loading || !treatment || !outcome}>{loading ? "Running…" : "Analyse DAG"}</button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>
      <div className="flex-1 min-w-0 space-y-4">
        {!result ? <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm">Enter a DAG and the treatment → outcome of interest.</div> : (
          <>
            <div className="panel border border-emerald-300 bg-emerald-50">
              <p className="text-sm text-gray-800 leading-relaxed">{result.result_text}</p></div>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-gray-200 bg-white p-3"><div className="text-[10px] uppercase tracking-wider text-gray-500">Adjust for (minimal set)</div>
                <div className="text-lg font-semibold mt-1 text-emerald-600">{result.adjustment_set.length ? result.adjustment_set.join(", ") : "∅ none"}</div></div>
              <div className="rounded-xl border border-gray-200 bg-white p-3"><div className="text-[10px] uppercase tracking-wider text-gray-500">Never adjust for</div>
                <div className="text-lg font-semibold mt-1 text-red-600">{result.do_not_adjust.length ? result.do_not_adjust.join(", ") : "—"}</div></div>
            </div>
            <div className="panel">
              <div className="text-xs font-semibold text-gray-600 mb-1">Node roles</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(result.roles).map(([nd, r]) => (
                  <span key={nd} className={`text-xs px-2 py-1 rounded border border-gray-200 bg-white ${roleColor(r as string)}`}>
                    <b>{nd}</b>: {r as string}
                  </span>
                ))}
              </div>
            </div>
            <ResultExporter title="dag_backdoor" />
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
    ["did", "Difference-in-Differences"],
    ["rdd", "Regression Discontinuity"],
    ["dag", "DAG Backdoor"],
  ];
  return (
    <div className="space-y-3">
      <div className="flex gap-1 flex-wrap">
        {tabs.map(([id, label]) => (
          <button key={id} onClick={() => setMethod(id)}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
              method === id ? "bg-white text-indigo-700 shadow-sm border border-gray-200" : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
            }`}>
            {label}
          </button>
        ))}
      </div>
      {method === "iv" ? <IVTab />
        : method === "mediation" ? <MediationTab />
        : method === "target" ? <TargetTrialTab />
        : method === "did" ? <DiDTab />
        : method === "rdd" ? <RDDTab />
        : <DAGTab />}
    </div>
  );
}
