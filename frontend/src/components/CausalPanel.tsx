import { useState } from "react";
import { useStore } from "../store";
import { runIV2SLS, runMediation, runTargetTrial, runDiD, runRDD, runDAGAdjustment, runSEM } from "../api";
import ResultExporter from "./ResultExporter";
import { fmtP } from "../lib/format";

type Method = "iv" | "mediation" | "target" | "did" | "rdd" | "dag" | "sem";

function getErrorDetail(e: unknown, fallback: string): string {
  const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  return typeof detail === "string" ? detail : fallback;
}

interface IVResult {
  result_text: string;
  n: number;
  iv_estimate: { estimate: number; ci_low: number; ci_high: number; p: number | null };
  ols_estimate: { estimate: number; p: number | null };
  first_stage: { f_stat: number; weak_instruments: boolean };
  wu_hausman: { p: number | null; endogenous: boolean };
  sargan?: { stat: number | string; df: number; p: number | null; valid: boolean } | null;
}

interface MediationResult {
  result_text: string;
  n: number;
  acme_significant: boolean;
  effects: {
    acme: number; acme_ci?: [number, number] | null;
    ade: number; ade_ci?: [number, number] | null;
    total: number; proportion_mediated?: number | null;
  };
  sobel: { p: number | null };
  paths: { a: number; b: number; c_prime: number };
}

interface TargetTrialResult {
  result_text: string;
  n_analyzed: number;
  n_screened: number;
  balanced: boolean;
  effect: {
    significant: boolean;
    risk_difference: number;
    rd_ci?: [number, number] | null;
    risk_ratio?: number | null;
    rr_ci?: [number, number] | null;
    risk_treated: number;
    risk_control: number;
  };
  protocol: Record<string, unknown>;
  caveats: string[];
}

interface DiDResult {
  result_text: string;
  significant: boolean;
  did_estimate: number;
  ci_low: number;
  ci_high: number;
  p: number | null;
  treated_change: number;
  control_change: number;
  cell_means: {
    control_pre: number; control_post: number;
    treated_pre: number; treated_post: number;
  };
}

interface RDDResult {
  result_text: string;
  significant: boolean;
  late: number;
  ci_low: number;
  ci_high: number;
  p: number | null;
  bandwidth: number;
  n_in_bandwidth: number;
  n_left: number;
  n_right: number;
}

interface DAGResult {
  result_text: string;
  adjustment_set: string[];
  do_not_adjust: string[];
  roles: Record<string, string>;
}

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
  const [result, setResult] = useState<IVResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await runIV2SLS({ session_id: sid, outcome, endogenous, instruments, covariates });
      setResult(r.data as IVResult);
    } catch (e: unknown) {
      setError(getErrorDetail(e, "IV estimation failed."));
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
  const [bootstrap, setBootstrap] = useState<number>(5000);
  const [result, setResult] = useState<MediationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const reps = Math.max(100, Math.min(20000, Math.round(bootstrap || 0)));
      const r = await runMediation({ session_id: sid, outcome, treatment, mediator, covariates, bootstrap: reps });
      setResult(r.data as MediationResult);
    } catch (e: unknown) {
      setError(getErrorDetail(e, "Mediation analysis failed."));
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
          <div>
            <label className="text-xs text-gray-400 block mb-1">Bootstrap resamples</label>
            <input
              type="number" min={100} max={20000} step={500}
              className="input w-full"
              value={bootstrap}
              onChange={(e) => setBootstrap(Number(e.target.value))}
            />
            <p className="text-[10px] text-gray-400 mt-1">PROCESS standard: 5000. Range 100–20000.</p>
          </div>
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
  const [result, setResult] = useState<TargetTrialResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const eligibility = elig
        .filter((e) => e.column && e.value !== "")
        .map((e) => ({ column: e.column, op: e.op, value: Number(e.value) }));
      const r = await runTargetTrial({ session_id: sid, treatment, outcome, confounders, eligibility, bootstrap: 400 });
      setResult(r.data as TargetTrialResult);
    } catch (e: unknown) {
      setError(getErrorDetail(e, "Target-trial emulation failed."));
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
  const [result, setResult] = useState<DiDResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await runDiD({ session_id: sid, outcome, group_col: groupCol, time_col: timeCol, covariates });
      setResult(r.data as DiDResult);
    } catch (e: unknown) { setError(getErrorDetail(e, "DiD failed.")); } finally { setLoading(false); }
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
  const [result, setResult] = useState<RDDResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await runRDD({ session_id: sid, outcome, running, cutoff: Number(cutoff), bandwidth: bandwidth ? Number(bandwidth) : null });
      setResult(r.data as RDDResult);
    } catch (e: unknown) { setError(getErrorDetail(e, "RDD failed.")); } finally { setLoading(false); }
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
  const [result, setResult] = useState<DAGResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const edges = edgeText.split("\n").map((l) => l.trim()).filter(Boolean)
        .map((l) => l.split(/->|→/).map((s) => s.trim())).filter((e) => e.length === 2 && e[0] && e[1]);
      const r = await runDAGAdjustment({ edges, treatment, outcome });
      setResult(r.data as DAGResult);
    } catch (e: unknown) { setError(getErrorDetail(e, "DAG analysis failed.")); } finally { setLoading(false); }
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

interface SEMPath { label: string | null; from: string; to: string; est: number | null; se: number | null; z: number | null; p: number | null }
interface SEMIndirect { label: string; treatment: string; chain: string[]; outcome: string; est: number | null; boot_ci: [number, number] | null; significant: boolean }
interface SEMDirect { treatment: string; outcome: string; est: number | null; se: number | null; p: number | null; ci: [number, number] | null }
interface SEMTotal { treatment: string; outcome: string; est: number | null; boot_ci: [number, number] | null }
interface SEMFit { chi2?: number | null; df?: number | null; p?: number | null; cfi?: number | null; tli?: number | null; rmsea?: number | null; srmr?: number | null; aic?: number | null; bic?: number | null; n: number }
interface SEMResult { result_text: string; n: number; lavaan_spec: string; paths: SEMPath[]; indirect_effects: SEMIndirect[]; direct_effects: SEMDirect[]; total_effects: SEMTotal[]; fit: SEMFit; serial: boolean; bootstrap_used: number }

function SEMTab() {
  const session = useStore((s) => s.session);
  const cols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";

  const [treatments, setTreatments] = useState<string[]>([]);
  const [mediators, setMediators] = useState<string[]>([]);
  const [outcomes, setOutcomes] = useState<string[]>([]);
  const [covariates, setCovariates] = useState<string[]>([]);
  const [serial, setSerial] = useState(false);
  const [bootstrap, setBootstrap] = useState<number>(5000);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [lavaanSpec, setLavaanSpec] = useState("");
  const [result, setResult] = useState<SEMResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const reps = Math.max(100, Math.min(20000, Math.round(bootstrap || 0)));
      const payload: Record<string, unknown> = {
        session_id: sid, treatments, mediators, outcomes, covariates,
        serial: serial && mediators.length >= 2, bootstrap: reps,
      };
      const spec = lavaanSpec.trim();
      if (spec) payload.lavaan_spec = spec;
      const r = await runSEM(payload);
      setResult(r.data as SEMResult);
    } catch (e: unknown) {
      setError(getErrorDetail(e, "SEM fit failed."));
    } finally { setLoading(false); }
  };

  const canRun = !!sid && !loading && (
    lavaanSpec.trim().length > 0 ||
    (treatments.length >= 1 && mediators.length >= 1 && outcomes.length >= 1)
  );
  const serialAvail = mediators.length >= 2;
  const fmtN = (v: number | null | undefined, d = 4) =>
    v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(d);
  const fmtCI = (ci?: [number, number] | null) =>
    ci ? `[${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : "—";

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">SEM / Path Analysis</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Fit a structural equation / path model with <b>multiple outcomes</b>, <b>parallel</b> or
            <b> serial mediators</b>, and global model fit indices. Equivalent to Hayes PROCESS
            Models 4 / 6 / 80 / 81 and beyond.
          </p>
        </div>
        <div className="panel space-y-3">
          <MultiPick label="Treatment(s) — exposure X" accent="accent-rose-500" columns={cols}
            exclude={[...mediators, ...outcomes, ...covariates]} value={treatments} onChange={setTreatments} />
          <MultiPick label="Mediator(s) — M" accent="accent-amber-500" columns={cols}
            exclude={[...treatments, ...outcomes, ...covariates]} value={mediators} onChange={setMediators} />
          <MultiPick label="Outcome(s) — Y (continuous)" accent="accent-emerald-500" columns={cols}
            exclude={[...treatments, ...mediators, ...covariates]} value={outcomes} onChange={setOutcomes} />
          <MultiPick label="Covariates (optional)" accent="accent-indigo-500" columns={cols}
            exclude={[...treatments, ...mediators, ...outcomes]} value={covariates} onChange={setCovariates} />
          <label className={`flex items-center gap-2 text-xs ${serialAvail ? "text-gray-700" : "text-gray-400"}`}>
            <input type="checkbox" checked={serial && serialAvail} disabled={!serialAvail}
              onChange={(e) => setSerial(e.target.checked)} />
            Serial chain (M1 → M2 → … → Y)
            {!serialAvail && <span className="text-[10px]">(needs ≥2 mediators)</span>}
          </label>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Bootstrap resamples</label>
            <input type="number" min={100} max={20000} step={500} className="input w-full"
              value={bootstrap} onChange={(e) => setBootstrap(Number(e.target.value))} />
            <p className="text-[10px] text-gray-400 mt-1">PROCESS standard: 5000. Range 100–20000.</p>
          </div>
          <div>
            <button type="button" className="text-xs text-indigo-600 hover:underline"
              onClick={() => setAdvancedOpen((o) => !o)}>
              {advancedOpen ? "▾" : "▸"} Advanced: lavaan model spec
            </button>
            {advancedOpen && (
              <div className="mt-2 space-y-1">
                <textarea className="input w-full font-mono text-[11px]" rows={6}
                  placeholder={"# overrides everything above\nM ~ a*X\nY ~ b*M + cp*X"}
                  value={lavaanSpec} onChange={(e) => setLavaanSpec(e.target.value)} />
                <p className="text-[10px] text-gray-400">When non-empty, the model is built verbatim from this spec; structure inputs above are ignored.</p>
              </div>
            )}
          </div>
          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>
            {loading ? "Fitting…" : "Run SEM"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-4">
        {!result ? (
          <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm text-center">
            Pick ≥1 treatment, ≥1 mediator, and ≥1 outcome — then run SEM.<br />
            Or supply a custom lavaan spec under Advanced.
          </div>
        ) : (
          <>
            <div className="panel border border-indigo-200 bg-indigo-50">
              <p className="text-sm text-gray-800 leading-relaxed">{result.result_text}</p>
              <div className="text-[11px] text-gray-500 mt-2">n = {result.n}</div>
            </div>

            <div className="panel">
              <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider mb-2">Model fit</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                <div>χ²({result.fit.df ?? "—"}) = {fmtN(result.fit.chi2)}</div>
                <div>p = {fmtP(result.fit.p ?? null)}</div>
                <div>CFI = {fmtN(result.fit.cfi, 3)}</div>
                <div>TLI = {fmtN(result.fit.tli, 3)}</div>
                <div>RMSEA = {fmtN(result.fit.rmsea, 3)}</div>
                <div>SRMR = {fmtN(result.fit.srmr, 3)}</div>
                <div>AIC = {fmtN(result.fit.aic, 1)}</div>
                <div>BIC = {fmtN(result.fit.bic, 1)}</div>
              </div>
              <p className="text-[10px] text-gray-400 mt-2">Good fit: CFI/TLI ≥ 0.95, RMSEA ≤ 0.06, SRMR ≤ 0.08.</p>
            </div>

            {result.indirect_effects.length > 0 && (
              <div className="panel">
                <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider mb-2">Indirect effects (bootstrap CIs)</h3>
                <table className="w-full text-xs">
                  <thead className="text-gray-500">
                    <tr><th className="text-left py-1">Path</th><th className="text-right">Est.</th><th className="text-right">95% boot CI</th><th className="text-right">Sig.</th></tr>
                  </thead>
                  <tbody>
                    {result.indirect_effects.map((ie) => (
                      <tr key={ie.label} className="border-t border-gray-100">
                        <td className="py-1">{ie.label}</td>
                        <td className="text-right">{fmtN(ie.est)}</td>
                        <td className="text-right">{fmtCI(ie.boot_ci)}</td>
                        <td className={`text-right font-semibold ${ie.significant ? "text-emerald-600" : "text-gray-400"}`}>{ie.significant ? "✓" : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {result.direct_effects.length > 0 && (
              <div className="panel">
                <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider mb-2">Direct effects (X → Y, holding M)</h3>
                <table className="w-full text-xs">
                  <thead className="text-gray-500">
                    <tr><th className="text-left py-1">Treatment</th><th className="text-left">Outcome</th><th className="text-right">Est.</th><th className="text-right">SE</th><th className="text-right">p</th><th className="text-right">95% CI</th></tr>
                  </thead>
                  <tbody>
                    {result.direct_effects.map((de, i) => (
                      <tr key={`${de.treatment}-${de.outcome}-${i}`} className="border-t border-gray-100">
                        <td className="py-1">{de.treatment}</td><td>{de.outcome}</td>
                        <td className="text-right">{fmtN(de.est)}</td>
                        <td className="text-right">{fmtN(de.se)}</td>
                        <td className="text-right">{fmtP(de.p)}</td>
                        <td className="text-right">{fmtCI(de.ci)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {result.total_effects.length > 0 && (
              <div className="panel">
                <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider mb-2">Total effects (X → Y, all paths)</h3>
                <table className="w-full text-xs">
                  <thead className="text-gray-500">
                    <tr><th className="text-left py-1">Treatment</th><th className="text-left">Outcome</th><th className="text-right">Est.</th><th className="text-right">95% boot CI</th></tr>
                  </thead>
                  <tbody>
                    {result.total_effects.map((te, i) => (
                      <tr key={`${te.treatment}-${te.outcome}-${i}`} className="border-t border-gray-100">
                        <td className="py-1">{te.treatment}</td><td>{te.outcome}</td>
                        <td className="text-right">{fmtN(te.est)}</td>
                        <td className="text-right">{fmtCI(te.boot_ci)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="panel">
              <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider mb-2">All path coefficients</h3>
              <table className="w-full text-xs">
                <thead className="text-gray-500">
                  <tr><th className="text-left py-1">From</th><th className="text-left">To</th><th className="text-left">Label</th><th className="text-right">Est.</th><th className="text-right">SE</th><th className="text-right">z</th><th className="text-right">p</th></tr>
                </thead>
                <tbody>
                  {result.paths.map((p, i) => (
                    <tr key={`${p.from}-${p.to}-${i}`} className="border-t border-gray-100">
                      <td className="py-1">{p.from}</td><td>{p.to}</td>
                      <td className="text-gray-400">{p.label ?? "—"}</td>
                      <td className="text-right">{fmtN(p.est)}</td>
                      <td className="text-right">{fmtN(p.se)}</td>
                      <td className="text-right">{fmtN(p.z, 2)}</td>
                      <td className="text-right">{fmtP(p.p)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <details className="panel">
              <summary className="text-xs font-bold text-gray-700 uppercase tracking-wider cursor-pointer">lavaan spec (echo)</summary>
              <pre className="text-[11px] mt-2 whitespace-pre-wrap">{result.lavaan_spec}</pre>
            </details>

            <ResultExporter title="sem" />
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
    ["sem", "SEM / Path analysis"],
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
        : method === "dag" ? <DAGTab />
        : <SEMTab />}
    </div>
  );
}
