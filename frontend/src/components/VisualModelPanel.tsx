/**
 * VisualModelPanel – "Visual" tab
 * Models: Polynomial, Linear Mixed, Gamma GLM, Negative Binomial
 * + Automated diagnostic plots for linear regression
 * + Per-chart export (PNG / PPTX) via PlotExporter
 * + All charts respect the global plot theme (usePlotLayout / usePalette)
 */
import { useState, useRef } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { usePlotLayout, usePalette, useTraceDefaults } from "../plotStyle";
import {
  runPolynomial, runLMM, runGamma, runNegBinom, runLinearDiag, runMelt, refreshSession,
} from "../api";
import { Tip, InfoBanner } from "./Tip";
import { MissingGuard, type ImputationStrategy } from "./MissingGuard";
import PlotExporter from "./PlotExporter";
import { fmtP } from "../lib/format";

// ── p-value formatter (centralised in lib/format.ts) ───────────────────────────
const sig   = (p: number) => p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "";

// ── CoefRow component ──────────────────────────────────────────────────────────
function CoefRow({ c, expMode = false }: { c: any; expMode?: boolean }) {
  const est = expMode ? (c.exp_estimate ?? c.irr ?? c.odds_ratio) : c.estimate;
  const adjP = c.p;
  return (
    <tr className={`border-b border-gray-100 ${adjP < 0.05 ? "hover:bg-indigo-50/40" : "hover:bg-gray-50"}`}>
      <td className="font-mono text-xs text-gray-900 pr-3 py-1">{c.variable}</td>
      <td className="pr-2 font-mono">{(expMode ? est : c.estimate)?.toFixed(4)}</td>
      {!expMode && <td className="pr-2 font-mono">{c.exp_estimate != null ? c.exp_estimate.toFixed(3) : ""}</td>}
      <td className="pr-2">{c.se?.toFixed(4)}</td>
      <td className="pr-2">{(c.z ?? c.t)?.toFixed(3)}</td>
      <td className="pr-2">
        <span className={adjP < 0.05 ? "badge-sig" : "badge-ns"}>{fmtP(adjP)}</span>
      </td>
      <td className="text-yellow-400 font-bold">{sig(adjP)}</td>
    </tr>
  );
}

function CoefTable({ coefs, expMode = false }: { coefs: any[]; expMode?: boolean }) {
  const hd = "pb-1.5 pr-2 font-medium text-xs text-gray-500";
  return (
    <div className="overflow-auto rounded border border-gray-200 mt-2">
      <table>
        <thead>
          <tr>
            <th className={hd}>Variable</th>
            <th className={hd}>Estimate</th>
            {!expMode && <th className={hd}>exp(β)</th>}
            <th className={hd}>SE</th>
            <th className={hd}>z/t</th>
            <th className={hd}>p-value</th>
            <th className={hd}></th>
          </tr>
        </thead>
        <tbody>
          {coefs.map((c: any) => <CoefRow key={c.variable} c={c} expMode={expMode} />)}
        </tbody>
      </table>
    </div>
  );
}

// ── StatCards ─────────────────────────────────────────────────────────────────
function StatCards({ pairs }: { pairs: [string, any, string?][] }) {
  return (
    <div className="grid grid-cols-4 gap-3">
      {pairs.filter(([, v]) => v != null).map(([k, v, tip]) => (
        <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
          <p className="text-xs text-gray-400 flex items-center gap-1">
            {k} {tip && <Tip text={tip} />}
          </p>
          <p className="font-semibold text-gray-900 text-sm">{typeof v === "number" ? v.toFixed(4) : v}</p>
        </div>
      ))}
    </div>
  );
}

// ── Polynomial model panel ─────────────────────────────────────────────────────
function PolynomialSection({ sessionId, numCols }: { sessionId: string; numCols: string[] }) {
  const layout = usePlotLayout();
  const pal    = usePalette();
  const td     = useTraceDefaults();
  const showGrid = useStore(s => s.showGrid);

  const [outcome,    setOutcome]    = useState(numCols[0] ?? "");
  const [predictor,  setPredictor]  = useState(numCols[1] ?? numCols[0] ?? "");
  const [degree,     setDegree]     = useState(2);
  const [covariates, setCovariates] = useState<string[]>([]);
  const [robustSE,   setRobustSE]   = useState(false);
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const [result,     setResult]     = useState<any>(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState("");
  const plotRef = useRef<any>(null);

  const run = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const r = await runPolynomial({ session_id: sessionId, outcome, predictor, degree, covariates, imputation, robust_se: robustSE });
      setResult(r.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message ?? "Error");
    } finally { setLoading(false); }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        {/* Controls */}
        <div className="w-56 flex-shrink-0 panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Polynomial / Non-linear</h3>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Outcome (continuous)</label>
            <select className="select w-full" value={outcome} onChange={e => setOutcome(e.target.value)}>
              {numCols.map(c => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Predictor</label>
            <select className="select w-full" value={predictor} onChange={e => setPredictor(e.target.value)}>
              {numCols.filter(c => c !== outcome).map(c => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Polynomial degree
              <Tip text="1 = linear, 2 = quadratic (U-shape), 3 = cubic, 4–5 = flexible curve. Choose the lowest degree that fits — higher degrees risk over-fitting." wide />
            </label>
            <div className="flex gap-1">
              {[1,2,3,4,5].map(d => (
                <button key={d} onClick={() => setDegree(d)}
                  className={`flex-1 py-1 text-xs rounded border transition-colors ${degree===d ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-600 hover:bg-gray-50"}`}>
                  {d}{d===2?" ★":""}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Covariates (optional)</label>
            <div className="max-h-28 overflow-y-auto space-y-0.5">
              {numCols.filter(c => c !== outcome && c !== predictor).map(c => (
                <label key={c} className="flex items-center gap-1.5 text-xs cursor-pointer">
                  <input type="checkbox" className="accent-indigo-500"
                    checked={covariates.includes(c)}
                    onChange={() => setCovariates(p => p.includes(c) ? p.filter(x=>x!==c) : [...p,c])} />
                  <span className="text-gray-700 truncate">{c}</span>
                </label>
              ))}
            </div>
          </div>
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={robustSE} onChange={e => setRobustSE(e.target.checked)} className="accent-indigo-500" />
            <span className="text-gray-600">Robust SE (HC3)</span>
          </label>
          <MissingGuard sessionId={sessionId} columns={[outcome, predictor, ...covariates]} imputation={imputation} onImputation={setImputation}>
            <button className="btn-primary w-full" onClick={run} disabled={loading}>
              {loading ? "Fitting…" : "Fit Polynomial"}
            </button>
          </MissingGuard>
          {error && <p className="text-red-400 text-xs">{error}</p>}
        </div>

        {/* Result */}
        {result && (
          <div className="flex-1 grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-4 auto-rows-min items-start xl:grid-flow-dense">
            <div className="panel space-y-3 xl:col-start-2">
              <h4 className="font-semibold text-gray-900">{result.model}</h4>
              <StatCards pairs={[
                ["n", result.n, "Observations used"],
                ["R²", result.r_squared, "Proportion of variance explained"],
                ["Adj R²", result.adj_r_squared],
                ["AIC", result.aic, "Lower = better fit"],
                ["BIC", result.bic],
                ["Resid SE", result.residual_se],
              ]} />
              <CoefTable coefs={result.coefficients} />
            </div>

            {/* Fitted curve plot */}
            {result.curve && (
              <div className="panel relative xl:col-start-1">
                <h4 className="font-semibold text-gray-900 mb-2">Fitted Curve (degree {result.degree})</h4>
                <div className="relative">
                  <Plot
                    ref={plotRef}
                    data={[
                      { type: "scatter" as const, mode: "markers" as const,
                        x: result.scatter.x, y: result.scatter.y,
                        marker: { color: pal[0], size: td.markerSize - 2, opacity: 0.45 },
                        name: "Data", hovertemplate: `${result.predictor}: %{x:.2f}<br>${result.outcome}: %{y:.3f}<extra></extra>` },
                      { type: "scatter" as const,
                        x: [...result.curve.x, ...result.curve.x.slice().reverse()],
                        y: [...result.curve.ci_high, ...result.curve.ci_low.slice().reverse()],
                        fill: "toself" as const, fillcolor: `${pal[0]}22`, line: { color: "transparent" },
                        hoverinfo: "skip" as const, showlegend: false, name: "95% CI" },
                      { type: "scatter" as const, mode: "lines" as const,
                        x: result.curve.x, y: result.curve.y,
                        line: { color: pal[0], width: td.lineWidth + 0.5 },
                        name: `Degree-${result.degree} fit`, hovertemplate: `${result.predictor}: %{x:.2f}<br>Ŷ: %{y:.3f}<extra></extra>` },
                    ]}
                    layout={{
                      ...layout, height: 380, autosize: true,
                      xaxis: { ...layout.xaxis as any, showgrid: showGrid, title: { text: result.predictor } },
                      yaxis: { ...layout.yaxis as any, showgrid: showGrid, title: { text: result.outcome }, zeroline: false },
                      legend: { x: 0.01, y: 0.99, xanchor: "left" as const, yanchor: "top" as const, font: { size: 10 } },
                      margin: { t: 20, r: 20, b: 50, l: 60 },
                    }}
                    style={{ width: "100%", height: 380 }}
                    useResizeHandler
                    config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                  />
                  <PlotExporter plotRef={plotRef} title={`Polynomial_${result.predictor}_${result.outcome}`} />
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── LMM helpers ────────────────────────────────────────────────────────────────

/** Detect likely patient/subject ID columns by name pattern. */
function isIdLike(col: string): boolean {
  const lo = col.toLowerCase();
  return ["id", "no", "num", "number", "patient", "subject", "case", "record"].some(
    tok => lo === tok || lo.endsWith(tok) || lo.startsWith(tok)
  );
}

/**
 * Detect repeated-measures column clusters.
 * Returns groups of columns that share a common suffix of ≥3 chars.
 * e.g. [INHOSPITALEF, EF, CONTROLEF] → group "EF"
 */
function detectRepeatClusters(cols: string[]): { base: string; members: string[] }[] {
  const groups: Map<string, string[]> = new Map();
  for (const col of cols) {
    // Try every suffix of length 2..col.length-1
    for (let len = 2; len <= col.length - 1; len++) {
      const suffix = col.slice(-len).toUpperCase();
      if (!groups.has(suffix)) groups.set(suffix, []);
      groups.get(suffix)!.push(col);
    }
  }
  // Keep only groups where ≥ 2 different columns share the suffix
  // and the suffix itself is a plausible variable name (not just digits)
  const result: { base: string; members: string[] }[] = [];
  const seen = new Set<string>();
  for (const [base, members] of groups) {
    if (members.length < 2) continue;
    if (/^\d+$/.test(base)) continue;
    const key = members.slice().sort().join("|");
    if (seen.has(key)) continue;
    seen.add(key);
    result.push({ base, members });
  }
  // Prefer longer base names (more specific) and de-duplicate subsets
  return result
    .sort((a, b) => b.base.length - a.base.length)
    .filter((g, i, arr) =>
      !arr.slice(0, i).some(prev => g.members.every(m => prev.members.includes(m)))
    )
    .slice(0, 5);  // show at most 5 suggestions
}

// ── LMM panel ──────────────────────────────────────────────────────────────────
function LMMSection({ sessionId, allCols, numCols }: { sessionId: string; allCols: string[]; numCols: string[] }) {
  const session    = useStore(s => s.session);
  const setSession = useStore(s => s.setSession);

  // Derive binary cols (exactly 2 unique values 0/1) from preview
  const binaryCols = new Set<string>(
    (session?.columns ?? [])
      .filter(c => {
        const vals = new Set((session?.preview ?? []).map(r => r[c.name]).filter(v => v != null));
        return vals.size === 2 && [...vals].every(v => v === 0 || v === 1);
      })
      .map(c => c.name)
  );

  const [outcome, setOutcome] = useState(numCols[0] ?? "");
  const [fixedEffects, setFixedEffects] = useState<string[]>([]);
  const [groupCol, setGroupCol] = useState(() => {
    // Auto-suggest first ID-like col as grouping variable
    return allCols.find(isIdLike) ?? allCols[0] ?? "";
  });
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Melt state
  const [showMelt, setShowMelt] = useState(false);
  const [meltCluster, setMeltCluster] = useState<{ base: string; members: string[] } | null>(null);
  const [meltTimeVar, setMeltTimeVar] = useState("TimePoint");
  const [meltValueVar, setMeltValueVar] = useState("Value");
  const [meltLoading, setMeltLoading] = useState(false);
  const [meltDone, setMeltDone] = useState(false);

  const repeatClusters = detectRepeatClusters(allCols);

  const toggle = (c: string) => {
    if (isIdLike(c)) return;  // hard block
    setFixedEffects(p => p.includes(c) ? p.filter(x => x !== c) : [...p, c]);
  };

  const run = async () => {
    if (fixedEffects.length === 0) { setError("Select at least one fixed effect"); return; }
    const idInFe = fixedEffects.filter(isIdLike);
    if (idInFe.length > 0) {
      setError(`"${idInFe.join(", ")}" looks like an ID column — move it to Grouping variable.`);
      return;
    }
    setLoading(true); setError(""); setResult(null);
    try {
      const r = await runLMM({ session_id: sessionId, outcome, fixed_effects: fixedEffects, group_col: groupCol, imputation });
      setResult(r.data);
    } catch (e: any) { setError(e?.response?.data?.detail ?? e.message ?? "Error"); }
    finally { setLoading(false); }
  };

  const doMelt = async () => {
    if (!meltCluster) return;
    setMeltLoading(true);
    try {
      await runMelt({
        session_id: sessionId,
        id_col: groupCol,
        value_cols: meltCluster.members,
        time_var_name: meltTimeVar,
        value_var_name: meltValueVar,
      });
      setMeltDone(true);
      // Reload session metadata to pick up new long-format columns
      if (session) {
        const refresh = await refreshSession(sessionId);
        setSession({ ...session, ...refresh.data });
      }
      setShowMelt(false);
    } catch (e: any) { setError(e?.response?.data?.detail ?? e.message ?? "Melt failed"); }
    finally { setMeltLoading(false); }
  };

  const isBinaryOutcome = binaryCols.has(outcome);

  return (
    <div className="flex gap-4">
      <div className="w-60 flex-shrink-0 panel space-y-3">
        <h3 className="text-sm font-semibold text-gray-700">
          {isBinaryOutcome ? "GEE — Binary Outcome" : "Linear Mixed Model"}
          <Tip wide text={
            isBinaryOutcome
              ? "Binary outcome detected. statsmodels MixedLM only supports continuous outcomes. GEE (Generalized Estimating Equations) with Binomial/logit link is the correct alternative for clustered binary data. Estimates are log-odds; exp(β) = Odds Ratio."
              : "LMM accounts for repeated measures or clustered data. Fixed effects = population-level predictors. Group = the clustering variable (e.g. PatientID). Uses REML estimation."
          } />
        </h3>

        {/* Binary outcome alert */}
        {isBinaryOutcome && (
          <div className="rounded-lg bg-amber-50 border border-amber-200 px-2.5 py-2 text-[10px] text-amber-800 leading-relaxed">
            <p className="font-semibold mb-0.5">⚠ Binary outcome detected</p>
            <p>Standard LMM requires a continuous outcome. Auto-routing to <strong>GEE (Binomial/Logit)</strong> — the population-averaged GLMM equivalent. exp(β) = Odds Ratio.</p>
          </div>
        )}

        <div>
          <label className="text-xs text-gray-400 block mb-1">Outcome</label>
          <select className="select w-full" value={outcome} onChange={e => { setOutcome(e.target.value); setResult(null); }}>
            {allCols.map(c => (
              <option key={c} value={c}>
                {c}{binaryCols.has(c) ? " (binary→GEE)" : ""}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-400 block mb-1">
            Grouping variable
            <Tip text="The clustering unit — e.g. PatientID for repeated measures, or HospitalID for multilevel data. This becomes the Random Intercept. Must NOT be a fixed effect." wide />
          </label>
          <select className="select w-full" value={groupCol} onChange={e => setGroupCol(e.target.value)}>
            {allCols.filter(c => c !== outcome).map(c => (
              <option key={c} value={c}>
                {c}{isIdLike(c) ? " ✓ ID" : ""}
              </option>
            ))}
          </select>
          {!isIdLike(groupCol) && (
            <p className="text-[10px] text-amber-600 mt-0.5">⚠ Selected grouping variable doesn't look like an ID column</p>
          )}
        </div>

        <div>
          <label className="text-xs text-gray-400 block mb-1">Fixed effects (predictors)</label>
          <div className="max-h-44 overflow-y-auto space-y-0.5 border border-gray-100 rounded p-1">
            {allCols.filter(c => c !== outcome && c !== groupCol).map(c => {
              const blocked = isIdLike(c);
              return (
                <label key={c}
                  className={`flex items-center gap-1.5 text-xs rounded px-1 py-0.5 ${blocked ? "opacity-50 cursor-not-allowed bg-red-50" : "cursor-pointer hover:bg-gray-50"}`}
                  title={blocked ? `"${c}" looks like an ID column. Assign it as the Grouping variable instead — using IDs as fixed effects destroys degrees of freedom.` : ""}>
                  <input type="checkbox" className="accent-indigo-500"
                    checked={fixedEffects.includes(c)}
                    disabled={blocked}
                    onChange={() => toggle(c)} />
                  <span className={`truncate ${blocked ? "text-red-500 line-through" : "text-gray-700"}`}>{c}</span>
                  {blocked && <span className="ml-auto text-[9px] bg-red-100 text-red-500 px-1 rounded flex-shrink-0">ID</span>}
                </label>
              );
            })}
          </div>
        </div>

        <MissingGuard sessionId={sessionId} columns={[outcome, groupCol, ...fixedEffects]} imputation={imputation} onImputation={setImputation}>
          <button className="btn-primary w-full" onClick={run} disabled={loading || fixedEffects.length === 0}>
            {loading ? "Fitting…" : isBinaryOutcome ? "Fit GEE (Binomial)" : "Fit LMM"}
          </button>
        </MissingGuard>
        {error && <p className="text-red-400 text-xs">{error}</p>}

        {/* Wide → Long detector */}
        {repeatClusters.length > 0 && !meltDone && (
          <div className="border-t border-gray-100 pt-2 space-y-1.5">
            <button className="flex items-center w-full text-left" onClick={() => setShowMelt(v => !v)}>
              <span className="text-xs font-medium text-indigo-600">📐 Repeated measures detected</span>
              <span className="ml-auto text-gray-400 text-xs">{showMelt ? "▲" : "▼"}</span>
            </button>
            {showMelt && (
              <div className="space-y-2 text-xs">
                <p className="text-gray-400 leading-snug">
                  Wide-format data detected. Mixed models need <strong>Long format</strong> (one row per observation). Select a column cluster to reshape:
                </p>
                <div className="space-y-1">
                  {repeatClusters.map(g => (
                    <label key={g.base} className="flex items-start gap-1.5 cursor-pointer">
                      <input type="radio" name="melt_cluster" className="accent-indigo-500 mt-0.5"
                        checked={meltCluster?.base === g.base}
                        onChange={() => { setMeltCluster(g); setMeltValueVar(g.base); }} />
                      <span>
                        <span className="font-medium text-gray-700">Base: {g.base}</span>
                        <span className="text-gray-400 ml-1">({g.members.join(", ")})</span>
                      </span>
                    </label>
                  ))}
                </div>
                {meltCluster && (
                  <div className="space-y-1.5 pt-1 border-t border-gray-100">
                    <div>
                      <label className="text-gray-400 block mb-0.5">Time variable name</label>
                      <input className="select w-full" value={meltTimeVar} onChange={e => setMeltTimeVar(e.target.value)} />
                    </div>
                    <div>
                      <label className="text-gray-400 block mb-0.5">Value variable name</label>
                      <input className="select w-full" value={meltValueVar} onChange={e => setMeltValueVar(e.target.value)} />
                    </div>
                    <p className="text-[10px] text-amber-700 bg-amber-50 rounded px-1.5 py-1">
                      ⚠ This reshapes the dataset to Long format. The session data will be replaced. Grouping = <strong>{groupCol}</strong>.
                    </p>
                    <button className="btn-primary w-full text-xs py-1" onClick={doMelt} disabled={meltLoading}>
                      {meltLoading ? "Reshaping…" : `Melt → Long Format (${meltCluster.members.length} timepoints)`}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        {meltDone && (
          <div className="text-[10px] text-green-700 bg-green-50 rounded px-2 py-1.5 border border-green-200">
            ✓ Dataset reshaped to long format. Now fit the LMM using <strong>{meltValueVar}</strong> as outcome and <strong>{meltTimeVar}</strong> as fixed effect.
          </div>
        )}
      </div>

      {result && (
        <div className="flex-1 space-y-4">
          <div className="panel space-y-3">
            <h4 className="font-semibold text-gray-900">{result.model}</h4>

            {result.model_type === "gee_binomial" && (
              <div className="rounded bg-blue-50 border border-blue-200 px-3 py-2 text-[10px] text-blue-800 leading-relaxed">
                {result.note}
              </div>
            )}

            <StatCards pairs={[
              ["n", result.n],
              ["Groups", result.n_groups, "Number of level-2 clusters (random intercepts)"],
              ...(result.icc != null ? [["ICC", result.icc.toFixed(4), "Intraclass Correlation — proportion of variance explained by grouping"] as [string, any, string]] : []),
              ["AIC", result.aic?.toFixed(2)],
              ["BIC", result.bic?.toFixed(2)],
              ...(result.random_effect_variance != null ? [["σ² RE", result.random_effect_variance.toFixed(4), "Random effect (between-group) variance"] as [string, any, string]] : []),
              ...(result.residual_variance != null ? [["σ² Resid", result.residual_variance.toFixed(4), "Within-group residual variance"] as [string, any, string]] : []),
            ]} />

            <CoefTable coefs={result.coefficients} expMode={result.model_type === "gee_binomial"} />

            <InfoBanner>
              {result.model_type === "gee_binomial" ? (
                <>GEE estimates population-averaged log-odds. <strong>exp(β) = Odds Ratio</strong>. Coefficients represent the effect on the log-odds of the outcome across the entire population, accounting for within-subject correlation via Independence working correlation.</>
              ) : (
                <>ICC = {result.icc != null ? (result.icc * 100).toFixed(1) : "—"}% of variance is between groups ({result.group}).{" "}
                {result.icc != null && result.icc > 0.05
                  ? "Clustering accounts for substantial variance — the mixed model is appropriate."
                  : "Low ICC — grouping explains little variance; standard linear regression may be sufficient."}</>
              )}
            </InfoBanner>
          </div>
        </div>
      )}
    </div>
  );
}

// ── GLM Section (Gamma + NegBinom) ────────────────────────────────────────────
function GLMSection({ sessionId, allCols, numCols }: { sessionId: string; allCols: string[]; numCols: string[] }) {
  const [glmType, setGlmType] = useState<"gamma" | "negbinom">("gamma");
  const [outcome,    setOutcome]    = useState(numCols[0] ?? "");
  const [predictors, setPredictors] = useState<string[]>([]);
  const [link,       setLink]       = useState("log");
  const [robustSE,   setRobustSE]   = useState(false);
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const [result,     setResult]     = useState<any>(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState("");

  const toggle = (c: string) => setPredictors(p => p.includes(c) ? p.filter(x=>x!==c) : [...p,c]);

  const run = async () => {
    if (predictors.length === 0) { setError("Select at least one predictor"); return; }
    setLoading(true); setError(""); setResult(null);
    try {
      const payload = { session_id: sessionId, outcome, predictors, imputation, robust_se: robustSE, ...(glmType === "gamma" ? { link } : {}) };
      const fn = glmType === "gamma" ? runGamma : runNegBinom;
      const r = await fn(payload);
      setResult(r.data);
    } catch (e: any) { setError(e?.response?.data?.detail ?? e.message ?? "Error"); }
    finally { setLoading(false); }
  };

  return (
    <div className="flex gap-4">
      <div className="w-56 flex-shrink-0 panel space-y-3">
        <h3 className="text-sm font-semibold text-gray-700">Generalized Linear Models</h3>

        {/* GLM type */}
        <div className="flex rounded overflow-hidden border border-gray-200">
          {(["gamma", "negbinom"] as const).map(g => (
            <button key={g} onClick={() => { setGlmType(g); setResult(null); }}
              className={`flex-1 text-xs py-1 transition-colors ${glmType===g ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}>
              {g === "gamma" ? "Gamma" : "Neg. Binom."}
            </button>
          ))}
        </div>

        {glmType === "gamma" && (
          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Link function
              <Tip text="Log: multiplicative effects on the outcome (most common). Identity: additive effects. Inverse: decreasing effects." wide />
            </label>
            <select className="select w-full text-xs" value={link} onChange={e => setLink(e.target.value)}>
              <option value="log">Log (default)</option>
              <option value="identity">Identity</option>
              <option value="inverse">Inverse</option>
            </select>
          </div>
        )}

        <div>
          <label className="text-xs text-gray-400 block mb-1">
            {glmType === "gamma" ? "Outcome (positive continuous)" : "Outcome (count, ≥0)"}
            <Tip text={glmType === "gamma"
              ? "Gamma regression is for strictly positive continuous outcomes with right skew — e.g. LOS, costs, lab values that cannot be negative."
              : "Negative Binomial is for count outcomes with overdispersion (variance > mean). More flexible than Poisson — use when Poisson goodness-of-fit is poor."} wide />
          </label>
          <select className="select w-full" value={outcome} onChange={e => setOutcome(e.target.value)}>
            {numCols.map(c => <option key={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Predictors</label>
          <div className="max-h-40 overflow-y-auto space-y-0.5">
            {allCols.filter(c => c !== outcome).map(c => (
              <label key={c} className="flex items-center gap-1.5 text-xs cursor-pointer">
                <input type="checkbox" className="accent-indigo-500" checked={predictors.includes(c)} onChange={() => toggle(c)} />
                <span className="text-gray-700 truncate">{c}</span>
              </label>
            ))}
          </div>
        </div>
        <label className="flex items-center gap-2 text-xs cursor-pointer">
          <input type="checkbox" checked={robustSE} onChange={e => setRobustSE(e.target.checked)} className="accent-indigo-500" />
          <span className="text-gray-600">Robust SE (HC3)</span>
        </label>
        <MissingGuard sessionId={sessionId} columns={[outcome, ...predictors]} imputation={imputation} onImputation={setImputation}>
          <button className="btn-primary w-full" onClick={run} disabled={loading || predictors.length === 0}>
            {loading ? "Fitting…" : "Fit GLM"}
          </button>
        </MissingGuard>
        {error && <p className="text-red-400 text-xs">{error}</p>}
      </div>

      {result && (
        <div className="flex-1 space-y-4">
          <div className="panel space-y-3">
            <h4 className="font-semibold text-gray-900">{result.model}</h4>
            <StatCards pairs={[
              ["n", result.n],
              ["AIC", result.aic?.toFixed(2), "Lower = better"],
              ["BIC", result.bic?.toFixed(2)],
              ["Deviance", result.deviance?.toFixed(2), "Smaller = better fit"],
              ...(result.scale != null ? [["Scale (dispersion)", result.scale?.toFixed(4)] as [string, any]] : []),
            ]} />
            <CoefTable coefs={result.coefficients} expMode={false} />
            {glmType === "gamma" && link === "log" && (
              <InfoBanner>
                Estimates are on the <strong>log scale</strong>. exp(β) = multiplicative change in the outcome per 1-unit increase in the predictor.
                E.g. exp(0.2) ≈ 1.22 means 22% higher mean outcome per unit increase.
              </InfoBanner>
            )}
            {glmType === "negbinom" && (
              <InfoBanner>
                Coefficients are log Incidence Rate Ratios (IRR). IRR = exp(β) — the ratio of expected counts per 1-unit increase.
                The negative binomial allows overdispersion (variance &gt; mean) that Poisson cannot handle.
              </InfoBanner>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Diagnostic Plots Section ──────────────────────────────────────────────────
function DiagnosticsSection({ sessionId, allCols, numCols }: { sessionId: string; allCols: string[]; numCols: string[] }) {
  const layout = usePlotLayout();
  const pal    = usePalette();
  const td     = useTraceDefaults();
  const showGrid = useStore(s => s.showGrid);

  const [outcome,    setOutcome]    = useState(numCols[0] ?? "");
  const [predictors, setPredictors] = useState<string[]>([]);
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const [diag,       setDiag]       = useState<any>(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState("");
  const rfRef  = useRef<any>(null);
  const qqRef  = useRef<any>(null);
  const slRef  = useRef<any>(null);

  const run = async () => {
    if (predictors.length === 0) { setError("Select at least one predictor"); return; }
    setLoading(true); setError(""); setDiag(null);
    try {
      const r = await runLinearDiag({ session_id: sessionId, outcome, predictors, imputation });
      setDiag(r.data);
    } catch (e: any) { setError(e?.response?.data?.detail ?? e.message ?? "Error"); }
    finally { setLoading(false); }
  };

  const sharedLayout = (title: string, xLabel: string, yLabel: string) => ({
    ...layout,
    height: 300, autosize: true,
    xaxis: { ...layout.xaxis as any, showgrid: showGrid, title: { text: xLabel, font: { size: 10 } } },
    yaxis: { ...layout.yaxis as any, showgrid: showGrid, title: { text: yLabel, font: { size: 10 } }, zeroline: false },
    title: { text: title, font: { size: 12, color: "#374151" } },
    margin: { t: 36, r: 16, b: 48, l: 56 },
    showlegend: false,
  });

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <div className="w-56 flex-shrink-0 panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">
            Regression Diagnostics
            <Tip text="Four standard diagnostic plots for a fitted linear regression model. Use these to check: (1) linearity of residuals, (2) normality of residuals, (3) homoscedasticity (constant variance). Violations may indicate the need for transformations or a different model family." wide />
          </h3>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Outcome</label>
            <select className="select w-full" value={outcome} onChange={e => setOutcome(e.target.value)}>
              {numCols.map(c => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Predictors</label>
            <div className="max-h-40 overflow-y-auto space-y-0.5">
              {allCols.filter(c => c !== outcome).map(c => (
                <label key={c} className="flex items-center gap-1.5 text-xs cursor-pointer">
                  <input type="checkbox" className="accent-indigo-500"
                    checked={predictors.includes(c)} onChange={() => setPredictors(p => p.includes(c) ? p.filter(x=>x!==c) : [...p,c])} />
                  <span className="text-gray-700 truncate">{c}</span>
                </label>
              ))}
            </div>
          </div>
          <MissingGuard sessionId={sessionId} columns={[outcome, ...predictors]} imputation={imputation} onImputation={setImputation}>
            <button className="btn-primary w-full" onClick={run} disabled={loading || predictors.length === 0}>
              {loading ? "Computing…" : "Run Diagnostics"}
            </button>
          </MissingGuard>
          {error && <p className="text-red-400 text-xs">{error}</p>}
          {diag && (
            <div className="space-y-1.5 pt-2 border-t border-gray-100">
              <p className="text-[10px] text-gray-400">Model summary</p>
              <p className="text-xs text-gray-600">n = {diag.n} &nbsp;|&nbsp; R² = {diag.r_squared.toFixed(3)}</p>
              <p className="text-xs text-gray-600">Residual SE = {diag.residual_se.toFixed(4)}</p>
            </div>
          )}
        </div>

        {diag && (
          <div className="flex-1 grid grid-cols-2 gap-4">
            {/* Residuals vs Fitted */}
            <div className="panel relative">
              <div className="relative">
                <Plot ref={rfRef}
                  data={[
                    { type: "scatter" as const, mode: "markers" as const,
                      x: diag.residuals_fitted.x, y: diag.residuals_fitted.y,
                      marker: { color: pal[0], size: td.markerSize - 2, opacity: 0.55 },
                      hovertemplate: "Fitted: %{x:.3f}<br>Resid: %{y:.4f}<extra></extra>" },
                    { type: "scatter" as const, mode: "lines" as const,
                      x: [Math.min(...diag.residuals_fitted.x), Math.max(...diag.residuals_fitted.x)],
                      y: [0, 0], line: { color: "#ef4444", dash: "dash" as const, width: 1 } },
                  ]}
                  layout={sharedLayout("Residuals vs Fitted", "Fitted Values", "Residuals")}
                  style={{ width: "100%", height: 300 }} useResizeHandler
                  config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                />
                <PlotExporter plotRef={rfRef} title="Residuals_vs_Fitted" />
              </div>
            </div>

            {/* Normal Q-Q */}
            <div className="panel relative">
              <div className="relative">
                <Plot ref={qqRef}
                  data={[
                    { type: "scatter" as const, mode: "markers" as const,
                      x: diag.qq.theoretical, y: diag.qq.sample,
                      marker: { color: pal[0], size: td.markerSize - 2, opacity: 0.6 },
                      name: "Quantiles",
                      hovertemplate: "Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>" },
                    { type: "scatter" as const, mode: "lines" as const,
                      x: diag.qq.line_x, y: diag.qq.line_y,
                      line: { color: "#ef4444", dash: "dash" as const, width: 1.5 }, name: "Normal line" },
                  ]}
                  layout={sharedLayout("Normal Q-Q", "Theoretical Quantiles", "Sample Quantiles")}
                  style={{ width: "100%", height: 300 }} useResizeHandler
                  config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                />
                <PlotExporter plotRef={qqRef} title="Normal_QQ" />
              </div>
            </div>

            {/* Scale-Location */}
            <div className="panel relative">
              <div className="relative">
                <Plot ref={slRef}
                  data={[
                    { type: "scatter" as const, mode: "markers" as const,
                      x: diag.scale_location.x, y: diag.scale_location.y,
                      marker: { color: pal[1], size: td.markerSize - 2, opacity: 0.55 },
                      hovertemplate: "Fitted: %{x:.3f}<br>√|Std Resid|: %{y:.3f}<extra></extra>" },
                  ]}
                  layout={sharedLayout("Scale-Location", "Fitted Values", "√|Standardized Residuals|")}
                  style={{ width: "100%", height: 300 }} useResizeHandler
                  config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                />
                <PlotExporter plotRef={slRef} title="Scale_Location" />
              </div>
            </div>

            {/* Residual histogram */}
            <div className="panel relative">
              <div className="relative">
                <Plot
                  data={[{
                    type: "histogram" as const,
                    x: diag.residuals_fitted.y,
                    marker: { color: pal[2], opacity: 0.75 },
                    nbinsx: 30,
                    name: "Residuals",
                  }]}
                  layout={sharedLayout("Residual Distribution", "Residual", "Count")}
                  style={{ width: "100%", height: 300 }} useResizeHandler
                  config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {diag && (
        <div className="panel">
          <InfoBanner>
            <strong>Residuals vs Fitted:</strong> Points should scatter randomly around y=0 — a pattern (curve, funnel) indicates non-linearity or heteroscedasticity. &nbsp;
            <strong>Q-Q Plot:</strong> Points close to the diagonal line indicate normally distributed residuals — heavy tails suggest non-normality. &nbsp;
            <strong>Scale-Location:</strong> A horizontal band with roughly equal spread indicates homoscedasticity (constant variance). &nbsp;
            <strong>Residual histogram:</strong> Should be approximately bell-shaped and centred at 0.
          </InfoBanner>
        </div>
      )}
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────
const SECTIONS = [
  { id: "polynomial", label: "Polynomial / Non-linear" },
  { id: "lmm",        label: "Linear Mixed Model" },
  { id: "glm",        label: "GLM (Gamma / Neg. Binom.)" },
  { id: "diag",       label: "Diagnostic Plots" },
] as const;

type SectionId = typeof SECTIONS[number]["id"];

export default function VisualModelPanel() {
  const session = useStore(s => s.session);
  if (!session) return null;

  const numCols = session.columns.filter(c => c.kind === "numeric").map(c => c.name);
  const allCols = session.columns.map(c => c.name);
  const sid = session.session_id;

  const [active, setActive] = useState<SectionId>("polynomial");

  return (
    <div className="space-y-4">
      {/* Section tabs */}
      <div className="flex gap-1 flex-wrap">
        {SECTIONS.map(({ id, label }) => (
          <button key={id} onClick={() => setActive(id)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
              ${active === id ? "bg-indigo-600 text-white" : "text-gray-600 hover:bg-gray-100"}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Active section */}
      {active === "polynomial" && <PolynomialSection sessionId={sid} numCols={numCols} />}
      {active === "lmm"        && <LMMSection        sessionId={sid} allCols={allCols} numCols={numCols} />}
      {active === "glm"        && <GLMSection        sessionId={sid} allCols={allCols} numCols={numCols} />}
      {active === "diag"       && <DiagnosticsSection sessionId={sid} allCols={allCols} numCols={numCols} />}
    </div>
  );
}
