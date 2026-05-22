import { useState, useEffect, useRef, useMemo } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { runLinear, runLogistic, runKM, runCox, runLogisticTable, runPoisson, getSparklines } from "../api";
import { Tip, InfoBanner } from "./Tip";
import ResultExporter from "./ResultExporter";
import PlotExporter from "./PlotExporter";
import { MissingGuard, type ImputationStrategy } from "./MissingGuard";
import { PALETTES } from "../store";

const _pal = () => PALETTES[useStore.getState().plotTheme.palette] ?? PALETTES.indigo;

const PLOT_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  font: { color: "#374151", size: 12 },
  margin: { t: 30, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: "#e5e7eb" },
  yaxis: { gridcolor: "#e5e7eb" },
};

// ── Model guidance ──────────────────────────────────────────────────────────
const MODEL_GUIDANCE: Record<string, { use: string; check: string; interpret: string }> = {
  linear: {
    use: "Predict a continuous outcome from one or more predictors. Best for dose-response, biomarker prediction, and adjusted mean comparisons.",
    check: "Residuals vs Fitted should show no pattern (linearity). Q-Q plot should be roughly diagonal (normality). Scale-Location should be flat (homoscedasticity). Use Robust SE if heteroscedastic.",
    interpret: "Each coefficient = change in outcome per 1-unit increase in predictor, holding others constant. R\u00B2 = proportion of variance explained. Check p-values and 95% CIs.",
  },
  logistic: {
    use: "Model a binary outcome (0/1) — e.g. death, readmission, disease presence. Returns Odds Ratios (OR) with 95% CI.",
    check: "Outcome must be binary 0/1. Check for multicollinearity (VIF > 5). Sample size rule of thumb: \u2265 10 events per predictor variable (EPV).",
    interpret: "OR > 1 = higher odds of outcome. OR < 1 = protective. OR = 1 = no effect. Report: OR (95% CI), p-value. Pseudo-R\u00B2 is NOT comparable to linear R\u00B2.",
  },
  ortable: {
    use: "Publication-standard univariate + multivariate OR table. Shows each predictor's effect both alone and adjusted for all others.",
    check: "Same as logistic. The forest plot visually compares unadjusted vs adjusted ORs — large shifts suggest confounding.",
    interpret: "Univariate OR = crude effect. Multivariate OR = adjusted effect. If they differ substantially, the variable is confounded by others in the model.",
  },
  poisson: {
    use: "Model count outcomes (0, 1, 2, 3...) — e.g. number of events, hospital visits, complications. Returns Incidence Rate Ratios (IRR).",
    check: "Outcome must be non-negative integers. Check for overdispersion: if variance >> mean, use Negative Binomial instead. Robust SE helps with mild overdispersion.",
    interpret: "IRR > 1 = higher rate. IRR = 1.5 means 50% more events. Report: IRR (95% CI), p-value.",
  },
  km: {
    use: "Visualise time-to-event data. The survival curve shows the probability of surviving beyond each time point. Log-rank test compares curves between groups.",
    check: "Event column must be binary 0/1 (1 = event occurred). Duration must be positive. Censoring is assumed non-informative.",
    interpret: "Curves that separate early = strong effect. Log-rank p < 0.05 = significant difference between groups. Median survival = time at which 50% have had the event.",
  },
  cox: {
    use: "Regression for time-to-event data with multiple predictors. Returns Hazard Ratios (HR) — the multiplicative effect on the event rate.",
    check: "Proportional hazards assumption: HR should be constant over time (check Schoenfeld residuals). Event column must be binary 0/1.",
    interpret: "HR > 1 = higher hazard (worse prognosis). HR < 1 = protective. HR = 1 = no effect. Report: HR (95% CI), p-value.",
  },
  rcs: {
    use: "Model non-linear (U/J-shaped) dose-response relationships using Restricted Cubic Splines. Essential for continuous biomarkers where the effect is not a straight line.",
    check: "Predictor must have enough unique values (\u2265 knots + 2). Logistic RCS requires binary 0/1 outcome. 4 knots is the standard default.",
    interpret: "The dose-response curve shows how OR (or outcome) changes across the predictor range. Reference value = OR 1.0. Non-linearity p-value tests whether the curve is significantly non-linear.",
  },
};

// ── p-value adjustment for one/two-tailed hypothesis ─────────────────────────


function adjustP(p: number, beta: number, nullHyp: string): number {
  if (nullHyp === "leq") return beta > 0 ? Math.min(p / 2, 1) : Math.min(1 - p / 2, 1);
  if (nullHyp === "geq") return beta < 0 ? Math.min(p / 2, 1) : Math.min(1 - p / 2, 1);
  return p; // "eq" = two-tailed default
}

// ── Mini bell-curve (sampling distribution of the estimator) ─────────────────
function MiniNormalSVG({ beta, se, p }: { beta: number; se: number; p: number }) {
  if (!isFinite(beta) || !isFinite(se) || se <= 0)
    return <span className="text-amber-400 text-[11px]">⚠</span>;
  const W = 64, H = 24, span = 3.8 * se;
  const lo = beta - span, hi = beta + span;
  const N  = 60;
  const toSX = (x: number) => ((x - lo) / (hi - lo)) * W;
  const toSY = (y: number) => H - 2 - y * (H - 4);
  const pts = Array.from({ length: N + 1 }, (_, i) => {
    const x = lo + (hi - lo) * i / N;
    return [x, Math.exp(-0.5 * ((x - beta) / se) ** 2)] as [number, number];
  });
  const curve = pts.map(([x, y]) => `${toSX(x).toFixed(1)},${toSY(y).toFixed(1)}`).join(" ");
  const fill  = [`0,${H}`, ...pts.map(([x, y]) => `${toSX(x).toFixed(1)},${toSY(y).toFixed(1)}`), `${W},${H}`].join(" ");
  const zx    = toSX(0);
  const color = p < 0.001 ? "#3730a3" : p < 0.01 ? "#4338ca" : p < 0.05 ? "#6366f1" : "#9ca3af";
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      <polygon points={fill}  fill={`${color}${p < 0.05 ? "22" : "0e"}`} />
      <polyline points={curve} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
      {zx >= 0 && zx <= W && (
        <line x1={zx.toFixed(1)} y1="1" x2={zx.toFixed(1)} y2={H}
          stroke="#9ca3af" strokeWidth="0.8" strokeDasharray="2,2" />
      )}
    </svg>
  );
}

// ── Significance bar ──────────────────────────────────────────────────────────
function SigBar({ p }: { p: number }) {
  const pct   = p < 0.001 ? 100 : p < 0.01 ? 80 : p < 0.05 ? 55 : p < 0.1 ? 22 : 7;
  const color = p < 0.001 ? "#3730a3" : p < 0.01 ? "#4338ca" : p < 0.05 ? "#6366f1" : "#d1d5db";
  return (
    <div style={{ width: 56, height: 10, backgroundColor: "#f3f4f6", borderRadius: 3, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", backgroundColor: color }} />
    </div>
  );
}

// ── Sparkline mini distribution bar ──────────────────────────────────────────
function SparklineMini({ data, type }: { data: number[]; type: string }) {
  const W = 44, H = 14;
  if (!data || data.length === 0) return null;
  const max = Math.max(...data);
  if (max === 0) return null;
  if (type === "numeric") {
    const bw = W / data.length;
    return (
      <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
        {data.map((v, i) => {
          const bh = Math.max(1, (v / max) * H);
          return <rect key={i} x={i * bw} y={H - bh} width={Math.max(bw - 0.5, 0.5)} height={bh} fill="#ef4444" opacity={0.65} />;
        })}
      </svg>
    );
  }
  // categorical → stacked horizontal proportion bars
  const total = data.reduce((a, b) => a + b, 0);
  const CATS  = _pal();
  let cx = 0;
  return (
    <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
      {data.map((v, i) => {
        const w = (v / total) * W;
        const rect = <rect key={i} x={cx} y={0} width={Math.max(w, 0.5)} height={H} fill={CATS[i % CATS.length]} />;
        cx += w;
        return rect;
      })}
    </svg>
  );
}

function CoefTable({
  coefs, hrMode = false, allColumns = [], selectedIdx = null, onSelect, nullHyp = "eq",
}: {
  coefs: any[]; hrMode?: boolean; allColumns?: string[];
  selectedIdx?: number | null; onSelect?: (i: number) => void; nullHyp?: string;
}) {
  const fmtP = (p: number) => (p < 0.001 ? "<0.001" : p.toFixed(4));
  const sig   = (p: number) => p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "";

  const isConst   = (n: string) => n === "const" || n === "Intercept";
  const isDummy   = (n: string) => !isConst(n) && allColumns.length > 0 && !allColumns.includes(n);
  const getBeta   = (c: any) => hrMode ? (c.log_hr ?? c.estimate) : (c.log_odds ?? c.estimate);

  const renderViz = (c: any) => {
    if (isConst(c.variable)) return <span className="text-gray-300 text-xs">—</span>;
    if (isDummy(c.variable)) return <span className="text-amber-400 text-xs" title="Categorical indicator variable">⚠</span>;
    const beta = getBeta(c);
    if (beta == null || c.se == null) return null;
    return <MiniNormalSVG beta={beta} se={c.se} p={adjustP(c.p, beta, nullHyp)} />;
  };
  const renderSig = (c: any) => {
    if (isConst(c.variable)) return null;
    const beta = getBeta(c) ?? 0;
    return <SigBar p={adjustP(c.p, beta, nullHyp)} />;
  };
  const rowCls = (i: number, adjP: number) =>
    `cursor-pointer border-b border-gray-100 transition-colors ${
      i === selectedIdx ? "bg-indigo-50" : adjP < 0.05 ? "hover:bg-indigo-50/40" : "hover:bg-gray-50"
    }`;
  const hd = "pb-1.5 pr-2 font-medium";

  // Detect logistic mode: coefficients have odds_ratio + or_ci_low fields
  const isLogistic = !hrMode && coefs.length > 0 && coefs[0].odds_ratio != null;
  // Detect Poisson mode
  const isPoisson  = !hrMode && !isLogistic && coefs.length > 0 && coefs[0].irr != null;

  // ── Export rows (generic) ─────────────────────────────────────────────────
  const coefExportHeaders = isPoisson
    ? ["Variable", "Log-IRR", "SE", "z", "p-value", "IRR", "CI_low", "CI_high"]
    : isLogistic
      ? ["Variable", "Log-Odds", "SE", "z", "p-value", "OR", "CI_low", "CI_high"]
      : hrMode
        ? ["Variable", "HR", "SE", "z", "p-value", "CI_low", "CI_high"]
        : ["Variable", "Estimate", "SE", "t", "p-value", "CI_low", "CI_high"];
  const coefExportRows = coefs.map((c: any) => {
    if (isPoisson) return [c.variable, c.log_irr?.toFixed(4) ?? "", c.se?.toFixed(4) ?? "", c.z?.toFixed(3) ?? "", c.p < 0.001 ? "<0.001" : c.p?.toFixed(4) ?? "", c.irr?.toFixed(3) ?? "", c.irr_ci_low?.toFixed(3) ?? "", c.irr_ci_high?.toFixed(3) ?? ""];
    if (isLogistic) return [c.variable, c.log_odds?.toFixed(4) ?? "", c.se?.toFixed(4) ?? "", c.z?.toFixed(3) ?? "", c.p < 0.001 ? "<0.001" : c.p?.toFixed(4) ?? "", c.odds_ratio?.toFixed(3) ?? "", c.or_ci_low?.toFixed(3) ?? "", c.or_ci_high?.toFixed(3) ?? ""];
    if (hrMode) return [c.variable, c.hr?.toFixed(4) ?? "", c.se?.toFixed(4) ?? "", (c.t ?? c.z)?.toFixed(3) ?? "", c.p < 0.001 ? "<0.001" : c.p?.toFixed(4) ?? "", c.hr_ci_low?.toFixed(3) ?? "", c.hr_ci_high?.toFixed(3) ?? ""];
    return [c.variable, c.estimate?.toFixed(4) ?? "", c.se?.toFixed(4) ?? "", (c.t ?? c.z)?.toFixed(3) ?? "", c.p < 0.001 ? "<0.001" : c.p?.toFixed(4) ?? "", c.ci_low?.toFixed(3) ?? "", c.ci_high?.toFixed(3) ?? ""];
  });
  const coefTitle = isPoisson ? "Poisson_Coefficients" : isLogistic ? "Logistic_Coefficients" : hrMode ? "Cox_Coefficients" : "Linear_Coefficients";

  // ── Poisson table ────────────────────────────────────────────────────────
  if (isPoisson) {
    return (
      <div>
        <div className="flex justify-end mb-1">
          <ResultExporter title={coefTitle} headers={coefExportHeaders} rows={coefExportRows} />
        </div>
      <div className="overflow-auto rounded border border-gray-200 mt-3">
        <table>
          <thead>
            <tr>
              <th className={hd}>Variable</th>
              <th className={hd} title="Log Incidence Rate Ratio">Log-IRR</th>
              <th className={hd}>SE</th><th className={hd}>z</th>
              <th className={hd}>p-value</th>
              <th className={hd} title="Incidence Rate Ratio = e^β">IRR</th>
              <th className={hd}>CI 95% (IRR)</th>
              <th className={hd}>Visualization</th>
              <th className={hd}>Significance</th>
              <th className={hd}></th>
            </tr>
          </thead>
          <tbody>
            {coefs.map((c: any, i: number) => {
              const adjP = adjustP(c.p, c.log_irr ?? 0, nullHyp);
              return (
                <tr key={c.variable} className={rowCls(i, adjP)} onClick={() => onSelect?.(i)}>
                  <td className="font-mono text-xs text-gray-900 pr-2">{c.variable}</td>
                  <td className="font-mono pr-2">{c.log_irr?.toFixed(4)}</td>
                  <td className="pr-2">{c.se?.toFixed(4)}</td>
                  <td className="pr-2">{c.z?.toFixed(3)}</td>
                  <td className="pr-2"><span className={adjP < 0.05 ? "badge-sig" : "badge-ns"}>{fmtP(adjP)}</span></td>
                  <td className={`font-mono font-semibold pr-2 ${adjP < 0.05 ? "text-indigo-600" : ""}`}>{c.irr?.toFixed(3)}</td>
                  <td className="font-mono text-xs text-gray-400 pr-2">
                    {c.irr_ci_low != null ? `${c.irr_ci_low.toFixed(3)}–${c.irr_ci_high.toFixed(3)}` : "–"}
                  </td>
                  <td className="pr-2">{renderViz(c)}</td>
                  <td className="pr-2">{renderSig(c)}</td>
                  <td className="text-yellow-400 font-bold">{sig(adjP)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      </div>
    );
  }

  // ── Logistic regression table ────────────────────────────────────────────
  if (isLogistic) {
    return (
      <div>
        <div className="flex justify-end mb-1">
          <ResultExporter title={coefTitle} headers={coefExportHeaders} rows={coefExportRows} />
        </div>
      <div className="overflow-auto rounded border border-gray-200 mt-3">
        <table>
          <thead>
            <tr>
              <th className={hd}>Variable</th>
              <th className={hd} title="Log-Odds (β)">Log-Odds</th>
              <th className={hd}>SE</th><th className={hd}>z</th>
              <th className={hd}>p-value</th>
              <th className={hd} title="Odds Ratio = e^β">OR</th>
              <th className={hd}>CI 95% (OR)</th>
              <th className={hd}>Visualization</th>
              <th className={hd}>Significance</th>
              <th className={hd}></th>
            </tr>
          </thead>
          <tbody>
            {coefs.map((c: any, i: number) => {
              const adjP = adjustP(c.p, c.log_odds ?? 0, nullHyp);
              return (
                <tr key={c.variable} className={rowCls(i, adjP)} onClick={() => onSelect?.(i)}>
                  <td className="font-mono text-xs text-gray-900 pr-2">{c.variable}</td>
                  <td className="font-mono pr-2">{c.log_odds?.toFixed(4)}</td>
                  <td className="pr-2">{c.se?.toFixed(4)}</td>
                  <td className="pr-2">{c.z?.toFixed(3)}</td>
                  <td className="pr-2"><span className={adjP < 0.05 ? "badge-sig" : "badge-ns"}>{fmtP(adjP)}</span></td>
                  <td className={`font-mono font-semibold pr-2 ${adjP < 0.05 ? "text-indigo-600" : ""}`}>{c.odds_ratio?.toFixed(3)}</td>
                  <td className="font-mono text-xs text-gray-400 pr-2">
                    {c.or_ci_low != null ? `${c.or_ci_low.toFixed(3)}–${c.or_ci_high.toFixed(3)}` : "–"}
                  </td>
                  <td className="pr-2">{renderViz(c)}</td>
                  <td className="pr-2">{renderSig(c)}</td>
                  <td className="text-yellow-400 font-bold">{sig(adjP)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      </div>
    );
  }

  // ── Linear / Cox (HR) table ──────────────────────────────────────────────
  return (
    <div>
      <div className="flex justify-end mb-1">
        <ResultExporter title={coefTitle} headers={coefExportHeaders} rows={coefExportRows} />
      </div>
    <div className="overflow-auto rounded border border-gray-200 mt-3">
      <table>
        <thead>
          <tr>
            <th className={hd}>Variable</th>
            {hrMode ? <th className={hd}>HR</th> : <th className={hd}>Estimate</th>}
            <th className={hd}>SE</th>
            {hrMode ? <th className={hd}>Z</th> : <th className={hd}>t / z</th>}
            <th className={hd}>p-value</th>
            <th className={hd}>CI (95%)</th>
            <th className={hd}>Visualization</th>
            <th className={hd}>Significance</th>
            <th className={hd}></th>
          </tr>
        </thead>
        <tbody>
          {coefs.map((c: any, i: number) => {
            const est  = hrMode ? c.hr : (c.estimate ?? c.log_hr);
            const beta = getBeta(c) ?? 0;
            const adjP = adjustP(c.p, beta, nullHyp);
            const ci   = hrMode
              ? (c.hr_ci_low != null ? `${c.hr_ci_low.toFixed(3)}–${c.hr_ci_high.toFixed(3)}` : "–")
              : (c.ci_low != null    ? `${c.ci_low.toFixed(3)}–${c.ci_high.toFixed(3)}`        : "–");
            return (
              <tr key={c.variable} className={rowCls(i, adjP)} onClick={() => onSelect?.(i)}>
                <td className="font-mono text-xs text-gray-900 pr-2">{c.variable}</td>
                <td className="pr-2">{typeof est === "number" ? est.toFixed(4) : est}</td>
                <td className="pr-2">{c.se?.toFixed(4)}</td>
                <td className="pr-2">{(c.t ?? c.z)?.toFixed(3)}</td>
                <td className="pr-2"><span className={adjP < 0.05 ? "badge-sig" : "badge-ns"}>{fmtP(adjP)}</span></td>
                <td className="font-mono text-xs pr-2">{ci}</td>
                <td className="pr-2">{renderViz(c)}</td>
                <td className="pr-2">{renderSig(c)}</td>
                <td className="text-yellow-400 font-bold">{sig(adjP)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
    </div>
  );
}

function ORTable({ rows, outcome, selectionMethod, nMulti, nTotal }: {
  rows: any[];
  outcome: string;
  selectionMethod?: string;
  nMulti?: number;
  nTotal?: number;
}) {
  const fmtP  = (p: number) => (p == null ? "–" : p < 0.001 ? "<0.001" : p.toFixed(4));
  const sig   = (p: number) => p == null ? "" : p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "";
  const fmtOR = (or: number | null, low: number | null, high: number | null) =>
    or == null ? "–" : `${or.toFixed(2)} (${low?.toFixed(2)}–${high?.toFixed(2)})`;

  const notEntered = (r: any) => r.multi_or == null && r.uni_or != null;

  const orExportHeaders = ["Variable", "Uni OR", "Uni CI low", "Uni CI high", "Uni p", "Multi OR", "Multi CI low", "Multi CI high", "Multi p"];
  const orExportRows = rows.map((r: any) => [
    r.variable,
    r.uni_or?.toFixed(4) ?? "",
    r.uni_ci_low?.toFixed(4) ?? "",
    r.uni_ci_high?.toFixed(4) ?? "",
    r.uni_p?.toFixed(6) ?? "",
    r.multi_or?.toFixed(4) ?? "",
    r.multi_ci_low?.toFixed(4) ?? "",
    r.multi_ci_high?.toFixed(4) ?? "",
    r.multi_p?.toFixed(6) ?? "",
  ]);

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-gray-400">Outcome: <span className="text-gray-700 font-mono">{outcome}</span></p>
        <ResultExporter title={`OR_Table_${outcome}`} headers={orExportHeaders} rows={orExportRows} />
      </div>
      {selectionMethod && selectionMethod !== "All variables (Enter)" && (
        <div className="flex items-center gap-2 mb-2 px-2 py-1.5 rounded bg-gray-100 border border-gray-300">
          <span className="text-yellow-400 text-xs">⚡</span>
          <span className="text-xs text-gray-400">
            <span className="text-gray-700 font-medium">{selectionMethod}</span>
            {nMulti != null && nTotal != null && (
              <span className="ml-1 text-gray-400">— {nMulti}/{nTotal} variables entered multivariate</span>
            )}
          </span>
          {nMulti != null && nTotal != null && nMulti < nTotal && (
            <span className="ml-auto text-xs text-gray-400 italic">excluded = —</span>
          )}
        </div>
      )}
      <div className="overflow-auto rounded border border-gray-200">
        <table>
          <thead>
            <tr>
              <th rowSpan={2} className="align-bottom">Variable</th>
              <th colSpan={3} className="text-center border-b border-gray-300 text-indigo-600">Univariate</th>
              <th colSpan={3} className="text-center border-b border-gray-300 text-emerald-600">Multivariate</th>
            </tr>
            <tr>
              <th className="text-indigo-600">OR (95% CI)</th>
              <th className="text-indigo-600">p-value</th>
              <th className="text-indigo-600"></th>
              <th className="text-emerald-600">OR (95% CI)</th>
              <th className="text-emerald-600">p-value</th>
              <th className="text-emerald-600"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.variable} className={notEntered(r) ? "opacity-50" : ""}>
                <td className="font-mono text-xs text-gray-900">
                  {r.variable}
                  {notEntered(r) && <span className="ml-1 text-gray-400 text-xs" title="Not selected for multivariate">↛</span>}
                </td>
                {/* Univariate */}
                <td className={`font-mono font-semibold ${r.uni_p != null && r.uni_p < 0.05 ? "text-indigo-600" : ""}`}>
                  {fmtOR(r.uni_or, r.uni_ci_low, r.uni_ci_high)}
                </td>
                <td>
                  {r.uni_p != null && (
                    <span className={r.uni_p < 0.05 ? "badge-sig" : "badge-ns"}>{fmtP(r.uni_p)}</span>
                  )}
                </td>
                <td className="text-yellow-400 font-bold">{r.uni_p != null ? sig(r.uni_p) : ""}</td>
                {/* Multivariate */}
                <td className={`font-mono font-semibold ${r.multi_p != null && r.multi_p < 0.05 ? "text-emerald-600" : ""}`}>
                  {fmtOR(r.multi_or, r.multi_ci_low, r.multi_ci_high)}
                </td>
                <td>
                  {r.multi_p != null && (
                    <span className={r.multi_p < 0.05 ? "badge-sig" : "badge-ns"}>{fmtP(r.multi_p)}</span>
                  )}
                </td>
                <td className="text-yellow-400 font-bold">{r.multi_p != null ? sig(r.multi_p) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Forest Plot ───────────────────────────────────────────────────────────────
const FOREST_BASE = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  font: { color: "#374151", size: 11 },
  xaxis: {
    type: "log" as const,
    gridcolor: "#e5e7eb",
    zeroline: false,
    tickfont: { size: 10 },
  },
  yaxis: { gridcolor: "transparent", zeroline: false, tickfont: { size: 11 } },
  shapes: [{
    type: "line" as const,
    x0: 1, x1: 1,
    xref: "x" as const, yref: "paper" as const,
    y0: 0, y1: 1,
    line: { color: "#ef4444", dash: "dot" as const, width: 1.5 },
  }],
};

/**
 * Auto-label a variable name with a clinical suffix:
 * - "Gender_Male"   → "Gender_Male  [Male vs. ref]"
 * - "Platelet (per 10000 units)" → keep as-is
 * - "Age"           → "Age  [per 1 unit ↑]"
 */
function varLabel(v: string): string {
  // Already has a unit hint → leave it alone
  if (v.includes("per ") || v.includes(" vs.") || v.includes("[")) return v;
  // Dummy from pd.get_dummies: "BaseName_Level"
  const dummyMatch = v.match(/^(.+)_([^_]+)$/);
  if (dummyMatch) {
    return `${v}  [${dummyMatch[2]} vs. ref]`;
  }
  // Binary-looking name (contains common medical binary keywords)
  const binaryKeywords = /^(DM|HT|AF|STEMI|NSTEMI|UAP|smoking|malign|redo|beating|Hypert|Heart|Chronic|periferik|occluded|Lima|LIMA)/i;
  if (binaryKeywords.test(v)) {
    return `${v}  [Yes vs. No]`;
  }
  return `${v}  [per 1 unit ↑]`;
}

/**
 * Marker size proportional to statistical precision (1/log-CI-width).
 * Narrow CI → larger square; wide CI → smaller square.
 */
function precisionSize(est: number, lo: number, hi: number): number {
  if (!est || !lo || !hi || lo <= 0 || hi <= lo) return 9;
  const logW = Math.log(hi) - Math.log(lo);
  if (logW <= 0) return 16;
  // exp decay: very narrow CI (logW~0.1) → ~15, wide CI (logW~3) → ~7
  const sz = 6 + 10 * Math.exp(-logW * 0.9);
  return Math.min(16, Math.max(6, sz));
}

type ForestLayout = "overlay" | "split";
type ForestColorMode = "series" | "significance";

function ForestPlot({ result, modelType, outcome }: {
  result: any;
  modelType: string;
  outcome?: string;
}) {
  const forestRef = useRef<any>(null);
  const isORTable = modelType === "ortable";
  const isCox     = modelType === "cox";
  const metric    = isCox ? "HR" : "OR";
  const showGrid  = useStore((s) => s.showGrid);

  // ── User-tunable layout options ───────────────────────────────────────────
  const [opts, setOpts] = useState<{
    layout: ForestLayout;
    colorBy: ForestColorMode;
    showValueColumns: boolean;
    customTitle: string;
    customSubtitle: string;
    customXLabel: string;
    markerStyle: "square" | "circle" | "diamond";
    height: number;
    sigColor: string;
    nonSigColor: string;
    showLegend: boolean;
    showArrows: boolean;
  }>({
    layout: "overlay",
    colorBy: "series",
    showValueColumns: true,
    customTitle: "",
    customSubtitle: "",
    customXLabel: "",
    markerStyle: "square",
    height: 0,                  // 0 = auto from row count
    sigColor: "#dc2626",
    nonSigColor: "#475569",
    showLegend: true,
    showArrows: true,
  });
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Significance color helper honours user-supplied palette.
  const sigColor = (est: number | null, lo: number | null, hi: number | null): string => {
    if (est == null || lo == null || hi == null) return "#9ca3af";
    const includesOne = lo <= 1 && hi >= 1;
    return includesOne ? opts.nonSigColor : opts.sigColor;
  };

  // Inline toolbar with the full option set. Advanced controls live behind
  // a disclosure to keep the bar compact.
  const Toolbar = (
    <div className="mb-2 text-[10px] text-gray-600 space-y-1.5">
      <div className="flex flex-wrap items-center gap-2">
        {isORTable && (
          <div className="inline-flex rounded-md border border-gray-300 overflow-hidden">
            {(["overlay", "split"] as const).map((v) => (
              <button key={v}
                onClick={() => setOpts((o) => ({ ...o, layout: v }))}
                className={`px-2 py-0.5 text-[10px] ${opts.layout === v ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
                {v === "overlay" ? "Overlay (one panel)" : "Split (two panels)"}
              </button>
            ))}
          </div>
        )}
        <div className="inline-flex rounded-md border border-gray-300 overflow-hidden">
          {(["series", "significance"] as const).map((v) => (
            <button key={v}
              onClick={() => setOpts((o) => ({ ...o, colorBy: v }))}
              className={`px-2 py-0.5 text-[10px] ${opts.colorBy === v ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
              {v === "series" ? "Color by series" : "Color by significance"}
            </button>
          ))}
        </div>
        <div className="inline-flex rounded-md border border-gray-300 overflow-hidden">
          {(["square", "circle", "diamond"] as const).map((v) => (
            <button key={v}
              onClick={() => setOpts((o) => ({ ...o, markerStyle: v }))}
              className={`px-2 py-0.5 text-[10px] ${opts.markerStyle === v ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
              ▪ {v}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={opts.showValueColumns}
            onChange={(e) => setOpts((o) => ({ ...o, showValueColumns: e.target.checked }))}
            className="accent-indigo-500" />
          Value columns
        </label>
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={opts.showLegend}
            onChange={(e) => setOpts((o) => ({ ...o, showLegend: e.target.checked }))}
            className="accent-indigo-500" />
          Legend
        </label>
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={opts.showArrows}
            onChange={(e) => setOpts((o) => ({ ...o, showArrows: e.target.checked }))}
            className="accent-indigo-500" />
          Direction arrows
        </label>
        <button
          onClick={() => setAdvancedOpen((v) => !v)}
          className="ml-auto text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50">
          {advancedOpen ? "▾ Hide advanced" : "▸ Show advanced"}
        </button>
      </div>
      {advancedOpen && (
        <div className="flex flex-wrap items-center gap-2 p-2 rounded border border-gray-200 bg-gray-50">
          <input type="text"
            placeholder="Custom title…"
            value={opts.customTitle}
            onChange={(e) => setOpts((o) => ({ ...o, customTitle: e.target.value }))}
            className="text-[10px] border border-gray-300 rounded px-2 py-0.5 w-60 focus:outline-none focus:border-indigo-400"
          />
          <input type="text"
            placeholder="Subtitle…"
            value={opts.customSubtitle}
            onChange={(e) => setOpts((o) => ({ ...o, customSubtitle: e.target.value }))}
            className="text-[10px] border border-gray-300 rounded px-2 py-0.5 w-60 focus:outline-none focus:border-indigo-400"
          />
          <input type="text"
            placeholder={`X-axis: ${metric} (95% CI), log scale`}
            value={opts.customXLabel}
            onChange={(e) => setOpts((o) => ({ ...o, customXLabel: e.target.value }))}
            className="text-[10px] border border-gray-300 rounded px-2 py-0.5 w-52 focus:outline-none focus:border-indigo-400"
          />
          <label className="flex items-center gap-1 text-[10px]">
            Height
            <input type="number" min={0} step={20}
              value={opts.height || ""}
              placeholder="auto"
              onChange={(e) => setOpts((o) => ({ ...o, height: parseInt(e.target.value) || 0 }))}
              className="w-14 text-[10px] border border-gray-300 rounded px-1 py-0.5 text-right focus:outline-none focus:border-indigo-400"
            />
          </label>
          {opts.colorBy === "significance" && (
            <>
              <label className="flex items-center gap-1 text-[10px]">
                CI excludes 1
                <input type="color" value={opts.sigColor}
                  onChange={(e) => setOpts((o) => ({ ...o, sigColor: e.target.value }))}
                  className="w-6 h-5 cursor-pointer border border-gray-300 rounded" />
              </label>
              <label className="flex items-center gap-1 text-[10px]">
                Includes 1
                <input type="color" value={opts.nonSigColor}
                  onChange={(e) => setOpts((o) => ({ ...o, nonSigColor: e.target.value }))}
                  className="w-6 h-5 cursor-pointer border border-gray-300 rounded" />
              </label>
            </>
          )}
        </div>
      )}
    </div>
  );

  // ── Shared helpers ────────────────────────────────────────────────────────
  const fmtP  = (p: number | null) =>
    p == null ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3);
  const fmtCI = (est: number | null, lo: number | null, hi: number | null) =>
    est == null ? "—" : `${est.toFixed(2)} (${lo?.toFixed(2)}–${hi?.toFixed(2)})`;

  // Base props for layout annotations
  const AB = { showarrow: false, xanchor: "left" as const, yanchor: "middle" as const };
  const HDR = { size: 9, color: "#374151" };

  // Directional arrow labels below x-axis (within forest domain)
  // forestRight = right edge of forest domain in paper coords
  const dirAnnotations = (forestRight: number, yPos = -0.10) => [
    {
      ...AB, xref: "paper" as const, yref: "paper" as const,
      x: 0.02, y: yPos, xanchor: "left" as const,
      text: `◀ ${isCox ? "Reduces hazard" : "Reduces risk"}`,
      font: { size: 9, color: "#10b981" }, showarrow: false,
    },
    {
      ...AB, xref: "paper" as const, yref: "paper" as const,
      x: forestRight - 0.01, y: yPos, xanchor: "right" as const,
      text: `${isCox ? "Increases hazard" : "Increases risk"} ▶`,
      font: { size: 9, color: "#ef4444" }, showarrow: false,
    },
  ];

  // ── OR Table (dual trace) ─────────────────────────────────────────────────
  if (isORTable) {
    const rows       = result.table as any[];
    const n          = rows.length;
    const yIdx       = Object.fromEntries(rows.map((r, i) => [r.variable, i]));
    const uniValid   = rows.filter((r) => r.uni_or   != null && r.uni_or   > 0);
    const multiValid = rows.filter((r) => r.multi_or != null && r.multi_or > 0);
    const plotH      = Math.max(320, n * 58 + 120);
    const splitLayout = opts.layout === "split";

    // Color resolvers honour the colorBy toggle:
    //   • "series"        — uni=slate / multi=emerald (existing palette)
    //   • "significance"  — gray when CI includes 1, red when it doesn't
    const uniColor = (r: any) =>
      opts.colorBy === "significance"
        ? sigColor(r.uni_or, r.uni_ci_low, r.uni_ci_high)
        : (r.uni_p != null && r.uni_p < 0.05 ? "#6366f1" : "#6b7280");
    const multiColor = (r: any) =>
      opts.colorBy === "significance"
        ? sigColor(r.multi_or, r.multi_ci_low, r.multi_ci_high)
        : (r.multi_p != null && r.multi_p < 0.05 ? "#10b981" : "#6b7280");

    // Forest x-domain depends on layout: overlay puts text columns to the
    // right of a single forest; split spreads two forests side-by-side and
    // skips the text columns by default (they can still be re-enabled).
    const forestRight = splitLayout
      ? 1.0           // entire width filled by two side-by-side forests
      : (opts.showValueColumns ? 0.47 : 0.95);
    const TX1 = splitLayout ? null : 0.49;
    const TX2 = splitLayout ? null : 0.76;
    const showCols = !splitLayout && opts.showValueColumns;

    const annotations: object[] = [
      ...(showCols && TX1 && TX2
        ? [
            { ...AB, xref: "paper", yref: "paper", x: TX1, y: 1.055,
              text: "<b>OR (95% CI)</b>", font: HDR },
            { ...AB, xref: "paper", yref: "paper", x: TX2, y: 1.055,
              text: "<b>p</b>", font: HDR },
            { ...AB, xref: "paper", yref: "paper", x: TX1, y: 1.012,
              text: opts.colorBy === "significance"
                ? "● Uni   ◆ Multi (red = CI excl. 1)"
                : "● Uni   ◆ Multi", font: { size: 8, color: "#4b5563" } },
          ]
        : []),
      ...(splitLayout
        ? [
            { ...AB, xref: "paper" as const, yref: "paper" as const, x: 0.215, y: 1.045,
              xanchor: "center" as const,
              text: "<b>Unadjusted</b>", font: HDR },
            { ...AB, xref: "paper" as const, yref: "paper" as const, x: 0.755, y: 1.045,
              xanchor: "center" as const,
              text: "<b>Adjusted (mutually adjusted model)</b>", font: HDR },
          ]
        : []),
      ...(splitLayout || !opts.showArrows ? [] : dirAnnotations(forestRight)),
    ];

    if (showCols && TX1 && TX2) {
      rows.forEach((r, i) => {
        if (r.uni_or != null) {
          const col = uniColor(r);
          annotations.push(
            { ...AB, xref: "paper", yref: "y", x: TX1, y: i + 0.18,
              text: fmtCI(r.uni_or, r.uni_ci_low, r.uni_ci_high), font: { size: 9, color: col } },
            { ...AB, xref: "paper", yref: "y", x: TX2, y: i + 0.18,
              text: fmtP(r.uni_p), font: { size: 9, color: col } },
          );
        }
        if (r.multi_or != null) {
          const col = multiColor(r);
          annotations.push(
            { ...AB, xref: "paper", yref: "y", x: TX1, y: i - 0.18,
              text: fmtCI(r.multi_or, r.multi_ci_low, r.multi_ci_high), font: { size: 9, color: col } },
            { ...AB, xref: "paper", yref: "y", x: TX2, y: i - 0.18,
              text: fmtP(r.multi_p), font: { size: 9, color: col } },
          );
        }
      });
    }

    // Trace builders shared by both layouts
    const uniMarker = opts.markerStyle === "square" ? "square" : "circle";
    const multiMarker = opts.markerStyle === "square" ? "square" : "diamond";
    const uniTrace: any = {
      name: splitLayout ? "Unadjusted" : "Univariate",
      type: "scatter", mode: "markers",
      x: uniValid.map((r) => r.uni_or),
      y: uniValid.map((r) => splitLayout ? yIdx[r.variable] : yIdx[r.variable] + 0.18),
      error_x: {
        type: "data", symmetric: false,
        array:      uniValid.map((r) => r.uni_ci_high - r.uni_or),
        arrayminus: uniValid.map((r) => r.uni_or - r.uni_ci_low),
        color: opts.colorBy === "series" ? "#6366f1" : "#9ca3af",
        thickness: 2, width: 7,
      },
      marker: {
        size: uniValid.map((r) => precisionSize(r.uni_or, r.uni_ci_low, r.uni_ci_high)),
        symbol: uniMarker,
        color: uniValid.map((r) => uniColor(r)),
        line: { color: "#d1d5db", width: 1 },
      },
      hovertemplate: uniValid.map((r) =>
        `<b>${r.variable}</b> (Unadjusted)<br>OR: ${r.uni_or?.toFixed(3)}<br>95% CI: ${r.uni_ci_low?.toFixed(3)} – ${r.uni_ci_high?.toFixed(3)}<br>p = ${fmtP(r.uni_p)}<extra></extra>`
      ),
      ...(splitLayout ? { xaxis: "x", yaxis: "y" } : {}),
    };
    const multiTrace: any = {
      name: splitLayout ? "Adjusted" : "Multivariate",
      type: "scatter", mode: "markers",
      x: multiValid.map((r) => r.multi_or),
      y: multiValid.map((r) => splitLayout ? yIdx[r.variable] : yIdx[r.variable] - 0.18),
      error_x: {
        type: "data", symmetric: false,
        array:      multiValid.map((r) => r.multi_ci_high - r.multi_or),
        arrayminus: multiValid.map((r) => r.multi_or - r.multi_ci_low),
        color: opts.colorBy === "series" ? "#10b981" : "#9ca3af",
        thickness: 2, width: 7,
      },
      marker: {
        size: multiValid.map((r) => precisionSize(r.multi_or, r.multi_ci_low, r.multi_ci_high)),
        symbol: multiMarker,
        color: multiValid.map((r) => multiColor(r)),
        line: { color: "#d1d5db", width: 1 },
      },
      hovertemplate: multiValid.map((r) =>
        `<b>${r.variable}</b> (Adjusted)<br>OR: ${r.multi_or?.toFixed(3)}<br>95% CI: ${r.multi_ci_low?.toFixed(3)} – ${r.multi_ci_high?.toFixed(3)}<br>p = ${fmtP(r.multi_p)}<extra></extra>`
      ),
      ...(splitLayout ? { xaxis: "x2", yaxis: "y" } : {}),
    };

    // Split layout uses two x-axes (xaxis [0, 0.43] / xaxis2 [0.50, 0.93])
    // anchored to the same y-axis, both log-scaled with the same null line.
    const splitShapes = splitLayout
      ? [
          { type: "line" as const, x0: 1, x1: 1, xref: "x" as const, yref: "paper" as const, y0: 0, y1: 1,
            line: { color: "#9ca3af", dash: "dash" as const, width: 1.2 } },
          { type: "line" as const, x0: 1, x1: 1, xref: "x2" as const, yref: "paper" as const, y0: 0, y1: 1,
            line: { color: "#9ca3af", dash: "dash" as const, width: 1.2 } },
        ]
      : [
          ...FOREST_BASE.shapes,
          ...(showCols
            ? [{ type: "line" as const, xref: "paper" as const, yref: "paper" as const,
                x0: 0.48, x1: 0.48, y0: 0, y1: 1,
                line: { color: "#e5e7eb", width: 1 } }]
            : []),
        ];

    const effHeight = opts.height || plotH;
    const titleHtml = opts.customTitle && opts.customSubtitle
      ? `${opts.customTitle}<br><span style="font-size:11px;color:#6b7280">${opts.customSubtitle}</span>`
      : opts.customTitle;
    const xLabel = opts.customXLabel
      || `${metric === "OR" ? "Odds Ratio" : "Hazard Ratio"} (95% CI)${splitLayout ? "" : (outcome ? ` — Outcome: ${outcome}` : "")}, log scale`;
    // Push legend BELOW the x-axis title so it never sits inside the data
    // area. Bottom margin grows when the legend is on to make room.
    const bottomPad = opts.showLegend ? 110 : 60;

    return (
      <div className="relative" ref={forestRef}>
      {Toolbar}
      <PlotExporter plotRef={forestRef} title={`Forest_${metric}_${outcome ?? "model"}`} />
      <Plot
        data={[uniTrace, multiTrace]}
        layout={{
          ...FOREST_BASE,
          height: effHeight,
          autosize: true,
          margin: { t: opts.customTitle ? (opts.customSubtitle ? 70 : 50) : 30, r: 20, b: bottomPad, l: 180 },
          title: opts.customTitle
            ? { text: titleHtml, font: { size: 12, color: "#1f2937" }, x: 0.5, xanchor: "center" as const }
            : undefined,
          xaxis: {
            ...FOREST_BASE.xaxis,
            showgrid: showGrid,
            domain: splitLayout ? [0, 0.43] : [0, forestRight],
            title: { text: xLabel, font: { size: 10, color: "#374151" } },
          },
          ...(splitLayout
            ? {
                xaxis2: {
                  ...FOREST_BASE.xaxis,
                  showgrid: showGrid,
                  domain: [0.50, 0.93],
                  anchor: "y" as const,
                  title: { text: opts.customXLabel || `${metric === "OR" ? "Odds Ratio" : "Hazard Ratio"} (95% CI), log scale${outcome ? ` — Outcome: ${outcome}` : ""}`, font: { size: 10, color: "#374151" } },
                },
              }
            : {}),
          yaxis: {
            ...FOREST_BASE.yaxis,
            tickvals: rows.map((_, i) => i),
            ticktext: rows.map((r) => varLabel(r.variable)),
            autorange: "reversed" as const,
            range: [-0.5, n - 0.5],
          },
          shapes: splitShapes,
          annotations,
          showlegend: opts.showLegend,
          // Legend sits below the x-axis title (yref=paper, y<0) so it never
          // overlaps short forests with few rows.
          legend: {
            font: { color: "#374151", size: 11 },
            bgcolor: "rgba(255,255,255,0.95)",
            bordercolor: "#e5e7eb", borderwidth: 1,
            orientation: "h" as const,
            x: 0.5, xanchor: "center" as const,
            y: -0.32, yanchor: "top" as const,
          },
        }}
        style={{ width: "100%", height: effHeight }}
        useResizeHandler
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      />
      </div>
    );
  }

  // ── Single model — logistic or cox ────────────────────────────────────────
  const coefs    = (result.coefficients ?? []).filter((c: any) => c.variable !== "const");
  const n        = coefs.length;
  if (n === 0) return null;

  const estimates = coefs.map((c: any) => isCox ? c.hr         : c.odds_ratio);
  const ciLow     = coefs.map((c: any) => isCox ? c.hr_ci_low  : c.or_ci_low);
  const ciHigh    = coefs.map((c: any) => isCox ? c.hr_ci_high : c.or_ci_high);
  const pVals     = coefs.map((c: any) => c.p);
  const labels    = coefs.map((c: any) => c.variable);
  const COLOR     = isCox ? "#10b981" : "#6366f1";
  const COLOR_SIG = isCox ? "#34d399" : "#818cf8";
  const plotH     = Math.max(260, n * 46 + 120);

  // Color resolver honours the colorBy toggle for the single-model branch.
  const rowColor = (i: number) =>
    opts.colorBy === "significance"
      ? sigColor(estimates[i], ciLow[i], ciHigh[i])
      : (pVals[i] < 0.05 ? COLOR_SIG : "#6b7280");

  // xaxis.domain depends on whether the value columns are visible.
  const forestRight = opts.showValueColumns ? 0.55 : 0.95;
  const TX1 = 0.57;
  const TX2 = 0.80;

  const annotations: object[] = [
    ...(opts.showValueColumns
      ? [
          { ...AB, xref: "paper", yref: "paper", x: TX1, y: 1.06,
            text: `<b>${metric} (95% CI)</b>`, font: HDR },
          { ...AB, xref: "paper", yref: "paper", x: TX2, y: 1.06,
            text: "<b>p</b>", font: HDR },
        ]
      : []),
    ...(opts.showArrows ? dirAnnotations(forestRight) : []),
    // Per-variable rows (only when value columns are shown)
    ...(opts.showValueColumns
      ? coefs.map((_: any, i: number) => {
          const col = rowColor(i);
          return [
            { ...AB, xref: "paper", yref: "y", x: TX1, y: i,
              text: fmtCI(estimates[i], ciLow[i], ciHigh[i]), font: { size: 9, color: col } },
            { ...AB, xref: "paper", yref: "y", x: TX2, y: i,
              text: fmtP(pVals[i]), font: { size: 9, color: col } },
          ];
        }).flat()
      : []),
  ];

  const effHeight = opts.height || plotH;
  const titleHtml = opts.customTitle && opts.customSubtitle
    ? `${opts.customTitle}<br><span style="font-size:11px;color:#6b7280">${opts.customSubtitle}</span>`
    : opts.customTitle;
  const xLabelSingle = opts.customXLabel || (isCox
    ? `Hazard Ratio (95% CI), log scale${outcome ? ` — Outcome: ${outcome}` : ""}`
    : `Odds Ratio (95% CI), log scale${outcome ? ` — Outcome: ${outcome}` : ""}`);

  return (
    <div className="relative" ref={forestRef}>
    {Toolbar}
    <PlotExporter plotRef={forestRef} title={`Forest_${metric}_${outcome ?? "model"}`} />
    <Plot
      data={[{
        type: "scatter", mode: "markers",
        x: estimates,
        y: coefs.map((_: any, i: number) => i),
        error_x: {
          type: "data", symmetric: false,
          array:      estimates.map((e: number, i: number) => (ciHigh[i] ?? e) - e),
          arrayminus: estimates.map((e: number, i: number) => e - (ciLow[i]  ?? e)),
          color: opts.colorBy === "series" ? COLOR : "#9ca3af",
          thickness: 2.5, width: 9,
        },
        marker: {
          size: estimates.map((_: number, i: number) => precisionSize(estimates[i], ciLow[i], ciHigh[i])),
          symbol: opts.markerStyle,
          color: coefs.map((_: any, i: number) => rowColor(i)),
          line: { color: "#d1d5db", width: 1 },
        },
        hovertemplate: coefs.map((_: any, i: number) =>
          `<b>${labels[i]}</b><br>${metric}: ${estimates[i]?.toFixed(3)}<br>95% CI: ${ciLow[i]?.toFixed(3)} – ${ciHigh[i]?.toFixed(3)}<br>p = ${fmtP(pVals[i])}<extra></extra>`
        ),
        name: isCox ? "Hazard Ratio" : "Odds Ratio",
        showlegend: opts.showLegend,
      }]}
      layout={{
        ...FOREST_BASE,
        height: effHeight,
        autosize: true,
        margin: { t: opts.customTitle ? (opts.customSubtitle ? 70 : 50) : 20, r: 20, b: opts.showLegend ? 110 : 70, l: 180 },
        title: opts.customTitle
          ? { text: titleHtml, font: { size: 12, color: "#1f2937" }, x: 0.5, xanchor: "center" as const }
          : undefined,
        xaxis: {
          ...FOREST_BASE.xaxis,
          showgrid: showGrid,
          domain: [0, forestRight],
          title: { text: xLabelSingle, font: { size: 10, color: "#374151" } },
        },
        yaxis: {
          ...FOREST_BASE.yaxis,
          tickvals: coefs.map((_: any, i: number) => i),
          ticktext: labels.map((l: string) => varLabel(l)),
          autorange: "reversed" as const,
          range: [-0.5, n - 0.5],
        },
        shapes: [
          ...FOREST_BASE.shapes,
          ...(opts.showValueColumns
            ? [{ type: "line" as const, xref: "paper" as const, yref: "paper" as const,
                x0: 0.56, x1: 0.56, y0: 0, y1: 1,
                line: { color: "#e5e7eb", width: 1 } }]
            : []),
        ],
        annotations,
        showlegend: opts.showLegend,
        legend: {
          font: { color: "#374151", size: 11 },
          bgcolor: "rgba(255,255,255,0.95)",
          bordercolor: "#e5e7eb", borderwidth: 1,
          orientation: "h" as const,
          x: 0.5, xanchor: "center" as const,
          y: -0.32, yanchor: "top" as const,
        },
      }}
      style={{ width: "100%", height: effHeight }}
      useResizeHandler
      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
    />
    </div>
  );
}

// ── Prediction Panel (interactive marginal effects + predicted value) ─────────
function PredictionPanel({ result }: { result: any }) {
  const predictorInfo: Record<string, any> = result.predictor_info ?? {};
  const coefs: any[] = result.coefficients ?? [];
  const outcome: string = result.outcome ?? "";
  const residualSe: number = result.residual_se ?? 0;
  const dfResid: number = result.df_resid ?? 100;

  // ── t quantile approximation (for PI) ─────────────────────────────────────
  const tQuantile = (ci: number) => {
    // good approximation for df > 30; exact for df → ∞
    if (dfResid > 200) return ci === 0.99 ? 2.576 : ci === 0.95 ? 1.96 : 1.645;
    const z = ci === 0.99 ? 2.576 : ci === 0.95 ? 1.96 : 1.645;
    return z * (1 + 1 / (4 * dfResid));  // simple correction
  };

  // ── Initialize slider values: numeric → median, categorical → first cat ───
  const initVals = () => {
    const v: Record<string, number | string> = {};
    for (const [col, info] of Object.entries(predictorInfo)) {
      if (info.type === "numeric") v[col] = info.median ?? info.mean ?? 0;
      else v[col] = info.categories?.[0] ?? "";
    }
    return v;
  };
  const [vals, setVals] = useState<Record<string, number | string>>(initVals);
  const [ciLevel, setCiLevel] = useState(0.95);
  const [showPI, setShowPI] = useState(false);
  const [sortByCat, setSortByCat] = useState(false);

  // ── Client-side prediction ─────────────────────────────────────────────────
  const predict = (overrides: Record<string, number | string> = {}) => {
    const v = { ...vals, ...overrides };
    let yhat = 0;
    for (const c of coefs) {
      const name: string = c.variable;
      const est: number = c.estimate ?? 0;
      if (name === "const" || name === "Intercept") {
        yhat += est;
      } else if (name in predictorInfo && predictorInfo[name].type === "numeric") {
        yhat += est * (Number(v[name]) || 0);
      } else {
        // Dummy variable — find parent by prefix match
        const parent = Object.keys(predictorInfo).find(
          (p) => predictorInfo[p]?.type === "categorical" && name.startsWith(p + "_")
        );
        if (parent) {
          const level = name.slice(parent.length + 1);
          yhat += est * (String(v[parent]) === level ? 1 : 0);
        } else {
          // Numeric predictor whose name is not in predictor_info (shouldn't happen but safe)
          if (name in v) yhat += est * (Number(v[name]) || 0);
        }
      }
    }
    return yhat;
  };

  const currentPred = predict();
  const tQ = tQuantile(ciLevel);
  const piHalf = tQ * residualSe * Math.sqrt(1 + 1 / Math.max(result.n ?? 100, 1));
  const piLow  = currentPred - piHalf;
  const piHigh = currentPred + piHalf;

  const exportCSV = () => {
    const rows: string[][] = [
      ["Variable", "Coefficient", "SE", "t", "p", "CI_low", "CI_high"],
      ...coefs.map((c: any) => [c.variable, c.estimate, c.se, c.t, c.p, c.ci_low, c.ci_high].map(String)),
      [],
      ["Outcome", outcome],
      ["R²", result.r_squared?.toFixed(4) ?? ""],
      ["Adj R²", result.adj_r_squared?.toFixed(4) ?? ""],
      ["N", String(result.n ?? "")],
      ["Residual SE", residualSe.toFixed(5)],
      [],
      ["--- Current Prediction ---"],
      ...Object.entries(vals).map(([k, v]) => [k, String(v)]),
      ["Predicted " + outcome, currentPred.toFixed(4)],
    ];
    const csv = rows.map((r) => r.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `Model_${outcome}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const numPreds = Object.entries(predictorInfo).filter(([, i]) => i.type === "numeric");
  const catPreds = Object.entries(predictorInfo).filter(([, i]) => i.type === "categorical");

  // Shared Plotly base layout
  const plotBase = {
    paper_bgcolor: "transparent", plot_bgcolor: "#ffffff",
    font: { color: "#374151", size: 10 },
    margin: { t: 32, r: 10, b: 44, l: 44 },
    showlegend: false,
  };

  return (
    <div className="panel space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-900">
          Predicted <span className="text-indigo-600">{outcome}</span>
        </h4>
        <button
          onClick={exportCSV}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs text-indigo-600 border border-indigo-200 hover:bg-indigo-50 transition-colors"
        >
          ↓ Export Model (CSV)
        </button>
      </div>

      {/* Charts grid */}
      {(numPreds.length > 0 || catPreds.length > 0) && (
        <div className="grid grid-cols-2 gap-4">
          {/* Numeric predictor line charts */}
          {numPreds.map(([col, info]) => {
            const N = 120;
            const lo = info.min, hi = info.max;
            const xs = Array.from({ length: N + 1 }, (_, i) => lo + (hi - lo) * i / N);
            const ys = xs.map((x) => predict({ [col]: x }));
            const cx = Number(vals[col]);
            const cy = predict();
            return (
              <div key={col} className="space-y-1">
                <p className="text-xs font-medium text-gray-600 text-center">
                  Predicted <em>{outcome}</em> vs. {col}
                </p>
                <Plot
                  data={[
                    { type: "scatter" as const, mode: "lines" as const, x: xs, y: ys,
                      line: { color: "#6366f1", width: 2 }, hovertemplate: `${col}: %{x:.2f}<br>Ŷ: %{y:.3f}<extra></extra>` },
                    ...(showPI ? [{
                      type: "scatter" as const, mode: "lines" as const,
                      x: [...xs, ...xs.slice().reverse()],
                      y: [...ys.map((y) => y + piHalf), ...ys.map((y) => y - piHalf).reverse()],
                      fill: "toself" as const, fillcolor: "rgba(99,102,241,0.10)",
                      line: { color: "transparent" }, hoverinfo: "skip" as const, showlegend: false,
                    }] : []),
                    { type: "scatter" as const, mode: "markers" as const, x: [cx], y: [cy],
                      marker: { color: "white", size: 11, line: { color: "#6366f1", width: 2.5 } },
                      hovertemplate: `${col} = ${cx.toFixed(1)}<br>Ŷ = ${cy.toFixed(3)}<extra></extra>` },
                  ]}
                  layout={{
                    ...plotBase, height: 200, autosize: true,
                    xaxis: { title: { text: col, font: { size: 10 } }, gridcolor: "#f3f4f6" },
                    yaxis: { title: { text: outcome, font: { size: 10 } }, gridcolor: "#f3f4f6", zeroline: false },
                  }}
                  style={{ width: "100%", height: 200 }}
                  useResizeHandler
                  config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                />
                {/* Slider */}
                <div className="flex items-center gap-2 px-1">
                  <input
                    type="number"
                    value={Number(vals[col]).toFixed(1)}
                    onChange={(e) => setVals((p) => ({ ...p, [col]: Number(e.target.value) }))}
                    className="w-16 text-xs border border-gray-300 rounded px-1.5 py-0.5 text-right font-mono"
                  />
                  <input
                    type="range"
                    min={info.min} max={info.max}
                    step={(info.max - info.min) / 200}
                    value={Number(vals[col])}
                    onChange={(e) => setVals((p) => ({ ...p, [col]: Number(e.target.value) }))}
                    className="flex-1 accent-indigo-500"
                  />
                </div>
              </div>
            );
          })}

          {/* Categorical predictor bar charts */}
          {catPreds.map(([col, info]) => {
            const cats: string[] = info.categories ?? [];
            const preds = cats.map((cat: string) => predict({ [col]: cat }));
            const selectedCat = String(vals[col]);
            const pairs = cats.map((cat: string, i: number) => ({ cat, pred: preds[i] }));
            const sorted = sortByCat ? [...pairs].sort((a, b) => b.pred - a.pred) : pairs;
            return (
              <div key={col} className="space-y-1">
                <p className="text-xs font-medium text-gray-600 text-center">
                  Predicted <em>{outcome}</em> by {col}
                </p>
                <Plot
                  data={[{
                    type: "bar" as const,
                    orientation: "h" as const,
                    x: sorted.map((p) => p.pred),
                    y: sorted.map((p) => p.cat),
                    text: sorted.map((p) => p.pred.toFixed(2)),
                    textposition: "outside" as const,
                    marker: {
                      color: sorted.map((p) => p.cat === selectedCat ? "#ef4444" : "#9ca3af"),
                    },
                    hovertemplate: `%{y}: Ŷ = %{x:.4f}<extra></extra>`,
                  }]}
                  layout={{
                    ...plotBase, height: Math.max(160, cats.length * 38 + 60), autosize: true,
                    xaxis: { title: { text: outcome, font: { size: 10 } }, gridcolor: "#f3f4f6" },
                    yaxis: { gridcolor: "transparent", zeroline: false, autorange: "reversed" as const },
                  }}
                  style={{ width: "100%", height: Math.max(160, cats.length * 38 + 60) }}
                  useResizeHandler
                  config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                />
                <div className="flex items-center gap-2 px-1">
                  <select
                    value={selectedCat}
                    onChange={(e) => setVals((p) => ({ ...p, [col]: e.target.value }))}
                    className="select text-xs flex-1"
                  >
                    {cats.map((cat: string) => <option key={cat}>{cat}</option>)}
                  </select>
                  <label className="flex items-center gap-1 text-[10px] text-gray-500 cursor-pointer whitespace-nowrap">
                    <input type="checkbox" checked={sortByCat} onChange={(e) => setSortByCat(e.target.checked)} className="accent-indigo-500" />
                    Sort by predicted
                  </label>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Big predicted value display */}
      <div className="rounded-xl bg-gray-900 text-white p-6 text-center relative">
        <p className="text-xs text-gray-400 mb-1">Predicted {outcome} =</p>
        <p className="text-5xl font-bold tracking-tight">{currentPred.toFixed(2)}</p>
        {showPI && residualSe > 0 && (
          <p className="text-sm text-gray-400 mt-2">
            {(ciLevel * 100).toFixed(0)}% PI: [{piLow.toFixed(2)}, {piHigh.toFixed(2)}]
          </p>
        )}
      </div>

      {/* PI controls */}
      <div className="flex items-center gap-4 px-1">
        <label className="flex items-center gap-2 cursor-pointer text-xs text-gray-600">
          <input type="checkbox" checked={showPI} onChange={(e) => setShowPI(e.target.checked)} className="accent-indigo-500" />
          Show prediction intervals
        </label>
        <input
          type="range" min={0.80} max={0.99} step={0.01} value={ciLevel}
          onChange={(e) => setCiLevel(Number(e.target.value))}
          disabled={!showPI}
          className={`w-28 accent-indigo-500 ${!showPI ? "opacity-30" : ""}`}
        />
        <span className="text-xs text-gray-500 font-mono">%{(ciLevel * 100).toFixed(0)}</span>
      </div>
    </div>
  );
}

// ── Coefficient Detail Panel (Plotly normal distribution on click) ────────────
function CoefDetailPanel({
  coef, nullHyp, onClose,
}: {
  coef: any; nullHyp: string; onClose: () => void;
}) {
  const beta = coef.log_odds ?? coef.log_irr ?? coef.log_hr ?? coef.estimate ?? 0;
  const se   = coef.se ?? 1;
  const adjP = adjustP(coef.p, beta, nullHyp);

  if (!isFinite(beta) || !isFinite(se) || se <= 0) return null;

  const span = 4 * se;
  const lo   = beta - span, hi = beta + span;
  const N    = 200;
  const xs   = Array.from({ length: N + 1 }, (_, i) => lo + (hi - lo) * i / N);
  const ys   = xs.map((x) => Math.exp(-0.5 * ((x - beta) / se) ** 2) / (se * Math.sqrt(2 * Math.PI)));

  const fillX = [...xs, ...xs.slice().reverse()];
  const fillY = [...ys, ...xs.map(() => 0)];

  const col = adjP < 0.001 ? "#3730a3" : adjP < 0.01 ? "#4338ca" : adjP < 0.05 ? "#6366f1" : "#9ca3af";

  return (
    <div className="panel border border-indigo-100 bg-indigo-50/30 relative">
      <button onClick={onClose} className="absolute top-2 right-2 text-gray-400 hover:text-gray-700 text-xs">✕ close</button>
      <h5 className="text-xs font-semibold text-gray-600 mb-2">
        Coefficient Detail — <span className="font-mono text-indigo-700">{coef.variable}</span>
      </h5>
      <div className="flex gap-4 items-start">
        <Plot
          data={[
            { type: "scatter" as const, x: fillX, y: fillY, fill: "toself",
              fillcolor: `${col}22`, line: { color: "transparent" }, hoverinfo: "skip", showlegend: false },
            { type: "scatter" as const, x: xs, y: ys, mode: "lines" as const,
              line: { color: col, width: 2 }, name: "N(β, SE)", hovertemplate: "x: %{x:.4f}<br>density: %{y:.4f}<extra></extra>" },
            { type: "scatter" as const, x: [0, 0], y: [0, Math.max(...ys) * 1.05],
              mode: "lines" as const, line: { color: "#9ca3af", dash: "dot" as const, width: 1.5 },
              name: "H₀ = 0", hoverinfo: "skip" as const },
            { type: "scatter" as const, x: [beta, beta], y: [0, Math.max(...ys) * 1.05],
              mode: "lines" as const, line: { color: col, dash: "dash" as const, width: 1.5 },
              name: `β = ${beta.toFixed(4)}`, hoverinfo: "skip" as const },
          ]}
          layout={{
            paper_bgcolor: "transparent", plot_bgcolor: "#ffffff",
            font: { color: "#374151", size: 11 },
            height: 200,
            margin: { t: 10, r: 20, b: 40, l: 50 },
            xaxis: { title: { text: "β (coefficient)", font: { size: 10 } }, gridcolor: "#e5e7eb", zeroline: false },
            yaxis: { title: { text: "density", font: { size: 10 } }, gridcolor: "#e5e7eb", zeroline: false },
            legend: { font: { size: 10 }, x: 0.65, y: 0.95, xanchor: "left" as const, yanchor: "top" as const },
            showlegend: true,
          }}
          style={{ width: "100%", height: 200 }}
          useResizeHandler
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        />
        <div className="flex-shrink-0 space-y-2 min-w-[130px] pt-2">
          {[
            ["β", beta.toFixed(5)],
            ["SE", se.toFixed(5)],
            ["z / t", (coef.z ?? coef.t)?.toFixed(4) ?? "–"],
            ["p (adj)", adjP < 0.001 ? "<0.001" : adjP.toFixed(4)],
            ...(coef.ci_low != null ? [["95% CI", `${coef.ci_low.toFixed(3)} – ${coef.ci_high.toFixed(3)}`]] : []),
            ...(coef.or_ci_low != null ? [["OR CI", `${coef.or_ci_low.toFixed(3)} – ${coef.or_ci_high.toFixed(3)}`]] : []),
            ...(coef.hr_ci_low  != null ? [["HR CI", `${coef.hr_ci_low.toFixed(3)} – ${coef.hr_ci_high.toFixed(3)}`]] : []),
            ...(coef.irr_ci_low != null ? [["IRR CI", `${coef.irr_ci_low.toFixed(3)} – ${coef.irr_ci_high.toFixed(3)}`]] : []),
            ...(coef.odds_ratio != null ? [["OR", coef.odds_ratio.toFixed(4)]] : []),
            ...(coef.hr != null         ? [["HR", coef.hr.toFixed(4)]] : []),
            ...(coef.irr != null        ? [["IRR", coef.irr.toFixed(4)]] : []),
          ].map(([k, v]) => (
            <div key={k}>
              <p className="text-[10px] text-gray-400">{k}</p>
              <p className={`text-xs font-mono font-semibold ${adjP < 0.05 ? "text-indigo-700" : "text-gray-700"}`}>{v}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Compact SPSS-style Model Summary Table ──────────────────────────────────

function ModelSummaryTable({ s }: { s: any }) {
  const fmtP = (p: number) => p < 0.001 ? "<0.001" : p.toFixed(4);
  const cl = s.classification;
  const hl = s.hosmer_lemeshow;
  const om = s.omnibus;

  return (
    <div className="overflow-auto rounded-xl border border-gray-200">
      <table className="text-xs w-full border-collapse">
        {/* Model Fit */}
        <thead>
          <tr className="bg-gray-50"><th colSpan={4} className="text-left px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-200">Model Fit</th></tr>
          <tr className="bg-gray-50 text-gray-400">
            <th className="px-3 py-1 text-left font-medium border-b border-gray-100">Metric</th>
            <th className="px-3 py-1 text-right font-medium border-b border-gray-100">Value</th>
            <th className="px-3 py-1 text-left font-medium border-b border-gray-100" colSpan={2}>Interpretation</th>
          </tr>
        </thead>
        <tbody>
          {om && (
            <tr className="border-b border-gray-50">
              <td className="px-3 py-1.5 text-gray-600">Omnibus χ²</td>
              <td className="px-3 py-1.5 text-right font-mono text-gray-800">{om.chi2?.toFixed(3)} <span className="text-gray-400">(df={om.df})</span></td>
              <td className="px-3 py-1.5" colSpan={2}>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${om.p < 0.05 ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                  p = {fmtP(om.p)}
                </span>
                <span className="text-gray-400 ml-1.5 text-[10px]">{om.p < 0.05 ? "Model significant" : "Not significant"}</span>
              </td>
            </tr>
          )}
          <tr className="border-b border-gray-50">
            <td className="px-3 py-1.5 text-gray-600">-2 Log Likelihood</td>
            <td className="px-3 py-1.5 text-right font-mono text-gray-800">{s.minus2ll?.toFixed(3)}</td>
            <td className="px-3 py-1.5 text-gray-400 text-[10px]" colSpan={2}>Lower = better fit</td>
          </tr>
          <tr className="border-b border-gray-50">
            <td className="px-3 py-1.5 text-gray-600">Cox &amp; Snell R²</td>
            <td className="px-3 py-1.5 text-right font-mono text-gray-800">{s.cox_snell_r2?.toFixed(4)}</td>
            <td className="px-3 py-1.5 text-gray-400 text-[10px]" colSpan={2}>Max &lt; 1.0</td>
          </tr>
          <tr className="border-b border-gray-50 bg-indigo-50/30">
            <td className="px-3 py-1.5 text-indigo-700 font-medium">Nagelkerke R²</td>
            <td className="px-3 py-1.5 text-right font-mono font-bold text-indigo-700">{s.nagelkerke_r2?.toFixed(4)}</td>
            <td className="px-3 py-1.5" colSpan={2}>
              <span className="text-[10px] text-indigo-600">{s.nagelkerke_r2 >= 0.4 ? "Excellent" : s.nagelkerke_r2 >= 0.2 ? "Good" : s.nagelkerke_r2 >= 0.1 ? "Moderate" : "Weak"} explanatory power</span>
            </td>
          </tr>
          {s.auc != null && (
            <tr className="border-b border-gray-50 bg-indigo-50/30">
              <td className="px-3 py-1.5 text-indigo-700 font-medium">AUC</td>
              <td className="px-3 py-1.5 text-right font-mono font-bold text-indigo-700">{s.auc?.toFixed(4)}</td>
              <td className="px-3 py-1.5" colSpan={2}>
                <span className="text-[10px] text-indigo-600">{s.auc >= 0.9 ? "Excellent" : s.auc >= 0.8 ? "Good" : s.auc >= 0.7 ? "Fair" : "Poor"} discrimination</span>
              </td>
            </tr>
          )}

          {/* Hosmer-Lemeshow */}
          {hl && (<>
            <tr className="bg-gray-50"><td colSpan={4} className="px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wider border-y border-gray-200">Calibration</td></tr>
            <tr className="border-b border-gray-50">
              <td className="px-3 py-1.5 text-gray-600">Hosmer-Lemeshow</td>
              <td className="px-3 py-1.5 text-right font-mono text-gray-800">{hl.chi2?.toFixed(3)} <span className="text-gray-400">(df={hl.df})</span></td>
              <td className="px-3 py-1.5" colSpan={2}>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${hl.p >= 0.05 ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                  p = {fmtP(hl.p)}
                </span>
                <span className="text-gray-400 ml-1.5 text-[10px]">{hl.p >= 0.05 ? "✓ Good calibration" : "⚠ Poor calibration"}</span>
              </td>
            </tr>
          </>)}

          {/* Classification */}
          {cl && (<>
            <tr className="bg-gray-50"><td colSpan={4} className="px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wider border-y border-gray-200">Classification (cutoff = 0.50)</td></tr>
            <tr className="border-b border-gray-50">
              <td className="px-3 py-1.5 text-gray-600">Accuracy</td>
              <td className="px-3 py-1.5 text-right font-mono font-semibold text-gray-800">{(cl.accuracy * 100).toFixed(1)}%</td>
              <td className="px-3 py-1.5 text-gray-400 text-[10px]" colSpan={2}>
                TP={cl.tp} TN={cl.tn} FP={cl.fp} FN={cl.fn}
              </td>
            </tr>
            <tr className="border-b border-gray-50">
              <td className="px-3 py-1.5 text-gray-600">Sensitivity / Specificity</td>
              <td className="px-3 py-1.5 text-right font-mono text-gray-800">{(cl.sensitivity * 100).toFixed(1)}% / {(cl.specificity * 100).toFixed(1)}%</td>
              <td className="px-3 py-1.5 text-gray-400 text-[10px]" colSpan={2}>
                PPV={(cl.ppv * 100).toFixed(1)}% NPV={(cl.npv * 100).toFixed(1)}%
              </td>
            </tr>
          </>)}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelsPanel() {
  const session  = useStore((s) => s.session);
  const showGrid = useStore((s) => s.showGrid);
  if (!session) return null;

  const numCols = session.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
  const allCols = session.columns.map((c) => c.name);

  // Binary columns (≤ 2 unique non-null values, both in {0, 1}) — Cox event
  // and logistic outcome pickers should narrow to these to avoid the user
  // accidentally selecting a continuous variable as the event indicator.
  const binaryCols = useMemo(() => {
    const out: string[] = [];
    for (const col of session.columns) {
      const vals = new Set<unknown>();
      for (const row of session.preview) {
        const v = row[col.name];
        if (v == null || v === "") continue;
        vals.add(typeof v === "number" ? v : Number(v));
        if (vals.size > 2) break;
      }
      const arr = [...vals];
      if (arr.length === 0 || arr.length > 2) continue;
      if (arr.every((v) => v === 0 || v === 1)) out.push(col.name);
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.session_id]);

  // Missing counts per column
  const missingCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const col of session.columns) {
      counts[col.name] = session.preview.filter(
        (row) => row[col.name] === null || row[col.name] === undefined || row[col.name] === ""
      ).length;
    }
    return counts;
  }, [session.preview, session.columns]);

  const [model, setModel] = useState("linear");
  const [outcome, setOutcome] = useState(numCols[0] ?? "");
  const [predictors, setPredictors] = useState<string[]>([]);

  // ── New feature state ─────────────────────────────────────────────────────
  const [selectedCoefIdx, setSelectedCoefIdx] = useState<number | null>(null);
  const [nullHyp,   setNullHyp]   = useState("eq");    // eq | leq | geq
  const [robustSE,  setRobustSE]  = useState(false);
  const [sparklines, setSparklines] = useState<Record<string, { type: string; data: number[] }>>({});

  useEffect(() => {
    getSparklines(session.session_id)
      .then((r) => setSparklines(r.data))
      .catch(() => {});
  }, [session.session_id]);

  const [scaleFactors, setScaleFactors] = useState<Record<string, string>>({}); // col → divisor string
  const [selection, setSelection] = useState("p10"); // multivariate variable selection strategy
  const [durationCol, setDurationCol] = useState(numCols[0] ?? "");
  const [eventCol, setEventCol] = useState(binaryCols[0] ?? numCols[1] ?? "");
  const [groupCol, setGroupCol] = useState("");
  const [stratifyCol, setStratifyCol] = useState("");
  const cachedModels = useStore((s) => s.panelCache.models);
  const setCacheModels = useStore((s) => s.setPanelCache);
  const [result, _setResultRaw] = useState<any>(cachedModels?.result ?? null);
  const setResult = (r: any) => { _setResultRaw(r); setCacheModels("models", { result: r }); };
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const [predFilter, setPredFilter] = useState("");

  // ── KM curve styling ────────────────────────────────────────────────────────
  const KM_PALETTE = _pal();
  const KM_DASHES  = ["solid","dash","dot","dashdot"] as const;

  interface KmStyle { color: string; width: number; dash: string; }
  const [kmStyles, setKmStyles] = useState<KmStyle[]>([]);
  const kmPlotRef  = useRef<any>(null);

  // KM display feature toggles
  const [showKMci,        setShowKMci]        = useState(true);
  const [showKMcensor,    setShowKMcensor]    = useState(true);
  const [showKMrisktable, setShowKMrisktable] = useState(true);

  // Re-init styles whenever a new KM result comes in
  useEffect(() => {
    if (result?.groups) {
      setKmStyles(
        result.groups.map((_: any, i: number) => ({
          color: KM_PALETTE[i % KM_PALETTE.length],
          width: 2,
          dash:  "solid",
        }))
      );
    }
  }, [result?.groups?.length, result?.groups?.map((g: any) => g.group).join("|")]);

  const updateKmStyle = (idx: number, patch: Partial<KmStyle>) =>
    setKmStyles((prev) => prev.map((s, i) => (i === idx ? { ...s, ...patch } : s)));

  const exportKMcsv = () => {
    if (!result?.groups) return;
    const rows = ["group,time,survival,n_at_risk,events"];
    result.groups.forEach((g: any) => {
      g.curve.forEach((p: any) => {
        rows.push(`"${g.group}",${p.time},${p.survival.toFixed(6)},${p.n_at_risk ?? ""},${p.events ?? ""}`);
      });
    });
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = `KM_${durationCol}_${eventCol}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const sid = session.session_id;

  const run = async () => {
    setLoading(true); setError(null); setResult(null); setSelectedCoefIdx(null);
    try {
      let res: any;
      const sf = buildScaleFactors();
      if (model === "linear") res = await runLinear({ session_id: sid, outcome, predictors, imputation, robust_se: robustSE });
      else if (model === "logistic") res = await runLogistic({ session_id: sid, outcome, predictors, scale_factors: sf, imputation, robust_se: robustSE });
      else if (model === "ortable") res = await runLogisticTable({ session_id: sid, outcome, predictors, scale_factors: sf, selection, imputation });
      else if (model === "poisson") res = await runPoisson({ session_id: sid, outcome, predictors, imputation, robust_se: robustSE });
      else if (model === "km") res = await runKM({ session_id: sid, duration_col: durationCol, event_col: eventCol, group_col: groupCol || undefined, stratify_col: stratifyCol || undefined, imputation });
      else res = await runCox({ session_id: sid, duration_col: durationCol, event_col: eventCol, predictors, imputation });
      setResult(res.data);
    } catch (e: any) {
      const detail = e.response?.data?.detail;
      setError(typeof detail === "string" ? detail : (e.message ?? "Unknown error"));
    } finally { setLoading(false); }
  };

  const togglePredictor = (col: string) => {
    setPredictors((prev) => {
      if (prev.includes(col)) {
        // Removing — also clear its scale factor
        setScaleFactors((sf) => { const next = { ...sf }; delete next[col]; return next; });
        return prev.filter((c) => c !== col);
      }
      return [...prev, col];
    });
  };

  const setScaleFactor = (col: string, val: string) => {
    setScaleFactors((sf) => ({ ...sf, [col]: val }));
  };

  /** Build scale_factors object for API: only include valid factors != 1 */
  const buildScaleFactors = () => {
    const out: Record<string, number> = {};
    for (const [col, val] of Object.entries(scaleFactors)) {
      const n = parseFloat(val);
      if (!isNaN(n) && n > 0 && n !== 1) out[col] = n;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  };

  const isSurvival  = false;  // KM/Cox moved to Survival Advanced tab
  const isORTable   = model === "ortable";
  const hasRobustSE = model === "linear" || model === "logistic" || model === "poisson";

  return (
    <div className="flex gap-4">
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Model</h3>
          {([
            ["linear",   "Linear Regression",       "Predict a continuous outcome (e.g. blood pressure) from one or more predictors. Output: β coefficients, R², p-values."],
            ["logistic", "Logistic Regression",      "Predict a binary outcome (0/1, yes/no) — outputs Odds Ratios showing how each predictor changes the odds of the event."],
            ["ortable",  "OR Table (Uni + Multi)",   "Run univariate logistic regression for each predictor separately, then all significant ones together in a multivariate model. Standard for clinical papers."],
            ["poisson",  "Poisson Regression",       "Count outcome model (e.g. number of events). Outputs Incidence Rate Ratios (IRR = eβ). Use when the outcome is a non-negative integer (event counts, re-admissions, etc.)."],
          ] as const).map(([v, l, desc]) => (
            <label key={v} className="flex items-start gap-2 cursor-pointer group">
              <input type="radio" name="model" value={v} checked={model === v} onChange={() => { setModel(v); setResult(null); setSelectedCoefIdx(null); }} className="accent-indigo-500 mt-0.5" />
              <span className="text-sm text-gray-700 leading-tight">
                {l}
                <Tip text={desc} wide />
              </span>
            </label>
          ))}
          {hasRobustSE && (
            <label className="flex items-center gap-2 cursor-pointer mt-1 pt-2 border-t border-gray-100">
              <input type="checkbox" checked={robustSE} onChange={(e) => setRobustSE(e.target.checked)} className="accent-indigo-500" />
              <span className="text-xs text-gray-600">
                Robust SE (HC3)
                <Tip text="Heteroscedasticity-consistent standard errors (HC3). Use when residuals may have unequal variance — common in clinical data. Does not change point estimates, only SEs and p-values." wide />
              </span>
            </label>
          )}
          {(model === "linear" || model === "logistic" || model === "ortable") && (
            <div className="mt-1 pt-2 border-t border-gray-100 text-[10px] text-indigo-600 bg-indigo-50 border border-indigo-200 rounded px-2 py-1 leading-snug">
              Need a non-linear continuous effect? Use the
              {" "}<span className="font-semibold">Restricted Cubic Spline</span> sub-tab —
              <code>rcs(X, k)</code> with Harrell or custom knots, Wald non-linearity test,
              OR / β / HR curve with 95 % CI, and optional spline × covariate or spline × spline
              interaction (LR test + 2D contour / 3D surface).
            </div>
          )}
        </div>

        <div className="panel space-y-3">
          {isSurvival ? (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Duration column</label>
                <select className="select w-full" value={durationCol} onChange={(e) => setDurationCol(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Event column (0/1)
                  {binaryCols.length === 0 && <span className="ml-1 text-[10px] text-amber-600">⚠ no binary 0/1 column detected</span>}
                </label>
                <select className="select w-full" value={eventCol} onChange={(e) => setEventCol(e.target.value)}>
                  {(binaryCols.length > 0 ? binaryCols : numCols).map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              {model === "km" && (
                <>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Group column (optional)</label>
                    <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                      <option value="">None</option>
                      {allCols.map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Stratify by (optional)</label>
                    <select className="select w-full" value={stratifyCol} onChange={(e) => setStratifyCol(e.target.value)}>
                      <option value="">None</option>
                      {allCols.filter((c) => c !== groupCol && c !== durationCol && c !== eventCol).map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>
                </>
              )}
              {model === "cox" && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-gray-400">Predictors</label>
                    <button onClick={() => { setPredictors([]); setResult(null); }} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300 transition-colors">Clear all</button>
                  </div>
                  <div className="mb-2 text-[10px] text-indigo-600 bg-indigo-50 border border-indigo-200 rounded px-2 py-1 leading-snug">
                    Predictors enter the model linearly. For non-linear continuous effects (e.g.
                    J-shaped LDL ↔ mortality), use the
                    {" "}<span className="font-semibold">Restricted Cubic Spline</span> sub-tab —
                    it fits <code>rcs(X, 4)</code> in the same Cox PH framework with knot placement
                    (Harrell percentiles or custom), Wald non-linearity test, HR curve with 95% CI,
                    optional <code>rcs(X) × rcs(Y)</code> interaction, and 2D/3D HR surfaces.
                  </div>
                  <input
                    type="text"
                    placeholder="Filter variables…"
                    value={predFilter}
                    onChange={(e) => setPredFilter(e.target.value)}
                    className="select w-full text-xs mb-1 py-1"
                  />
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {allCols
                      .filter((c) => c !== durationCol && c !== eventCol && c.toLowerCase().includes(predFilter.toLowerCase()))
                      .map((c) => {
                        const spk = sparklines[c];
                        return (
                          <label key={c} className="flex items-center gap-2 text-sm cursor-pointer">
                            <input type="checkbox" checked={predictors.includes(c)} onChange={() => togglePredictor(c)} className="accent-indigo-500" />
                            <span className="text-gray-700 truncate flex-1">{c}</span>
                            {(missingCounts[c] ?? 0) > 0 && (
                              <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0"
                                title={`${missingCounts[c]} missing values`}>
                                {missingCounts[c]}✕
                              </span>
                            )}
                            {spk && <SparklineMini data={spk.data} type={spk.type} />}
                          </label>
                        );
                      })}
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Outcome{isORTable && <span className="text-gray-400 ml-1">(binary 0/1)</span>}
                  {(missingCounts[outcome] ?? 0) > 0 && (
                    <span className="ml-1 text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200"
                      title={`${missingCounts[outcome]} missing values in outcome`}>
                      {missingCounts[outcome]} missing
                    </span>
                  )}
                </label>
                <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
                  {allCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>

              {isORTable && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Multivariate Selection</label>
                  <select className="select w-full text-xs" value={selection} onChange={(e) => setSelection(e.target.value)}>
                    <option value="all">All variables (Enter)</option>
                    <option value="p10">Univariate p &lt; 0.10 ★</option>
                    <option value="p05">Univariate p &lt; 0.05</option>
                    <option value="forward">Stepwise Forward</option>
                    <option value="backward">Stepwise Backward</option>
                  </select>
                </div>
              )}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">Predictors</label>
                  <button onClick={() => { setPredictors([]); setResult(null); }} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300 transition-colors">Clear all</button>
                </div>
                <input
                  type="text"
                  placeholder="Filter variables…"
                  value={predFilter}
                  onChange={(e) => setPredFilter(e.target.value)}
                  className="select w-full text-xs mb-1 py-1"
                />
                <div className="max-h-48 overflow-y-auto space-y-1">
                  {allCols.filter((c) => c !== outcome && c.toLowerCase().includes(predFilter.toLowerCase())).map((c) => {
                    const checked = predictors.includes(c);
                    const showScale = checked && (model === "logistic" || model === "ortable");
                    const spk = sparklines[c];
                    return (
                      <div key={c} className="space-y-0.5">
                        <label className="flex items-center gap-2 text-sm cursor-pointer">
                          <input type="checkbox" checked={checked} onChange={() => togglePredictor(c)} className="accent-indigo-500" />
                          <span className="text-gray-700 truncate flex-1">{c}</span>
                          {(missingCounts[c] ?? 0) > 0 && (
                            <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0"
                              title={`${missingCounts[c]} missing values`}>
                              {missingCounts[c]}✕
                            </span>
                          )}
                          {spk && <SparklineMini data={spk.data} type={spk.type} />}
                        </label>
                        {showScale && (
                          <div className="flex items-center gap-1 ml-5 mb-0.5">
                            <span className="text-gray-400 text-xs">÷</span>
                            <input
                              type="number"
                              min="0"
                              step="any"
                              placeholder="1 (no scaling)"
                              value={scaleFactors[c] ?? ""}
                              onChange={(e) => setScaleFactor(c, e.target.value)}
                              className="w-full text-xs bg-white border border-gray-300 rounded px-1.5 py-0.5 text-gray-700 placeholder-gray-300 focus:border-indigo-500 focus:outline-none"
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
          <MissingGuard
            sessionId={sid}
            columns={isSurvival
              ? [durationCol, eventCol, ...(model === "cox" ? predictors : [])]
              : [...predictors, outcome]}
            imputation={imputation}
            onImputation={setImputation}
          >
            <button className="btn-primary w-full" onClick={run} disabled={loading || (!isSurvival && predictors.length === 0) || (isORTable && predictors.length < 1)}>
              {loading ? "Fitting…" : "Fit Model"}
            </button>
          </MissingGuard>
          {error && <p className="text-red-400 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 space-y-4">
        {/* Model guidance */}
        {MODEL_GUIDANCE[model] && (
          <div className="grid grid-cols-3 gap-3">
            {[
              { icon: "🎯", title: "Use when", text: MODEL_GUIDANCE[model].use },
              { icon: "✅", title: "Check", text: MODEL_GUIDANCE[model].check },
              { icon: "📖", title: "Interpret", text: MODEL_GUIDANCE[model].interpret },
            ].map(({ icon, title, text }) => (
              <div key={title} className="panel bg-indigo-50 border-indigo-200 p-3">
                <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mb-1">{icon} {title}</p>
                <p className="text-xs text-indigo-800 leading-relaxed">{text}</p>
              </div>
            ))}
          </div>
        )}

        {result ? (
          <>
            {/* Summary cards */}
            <div className="panel">
              <h4 className="font-semibold text-gray-900 mb-3">{result.model}</h4>
              <div className="grid grid-cols-3 gap-3">
                {[
                  ["N",          result.n,                       "Total number of observations used to fit the model."],
                  result.r_squared != null      && ["R²",        result.r_squared?.toFixed(4),      "Proportion of variance in the outcome explained by the model (0–1). Higher is better, but add predictors only if they genuinely help."],
                  result.adj_r_squared != null  && ["Adj R²",    result.adj_r_squared?.toFixed(4),  "R² adjusted for the number of predictors — penalises adding unhelpful variables. Prefer this over R² when comparing models."],
                  result.pseudo_r2 != null      && ["Pseudo R²", result.pseudo_r2?.toFixed(4),      "McFadden's Pseudo R² for logistic regression. Analogous to R² but not directly comparable. Values 0.2–0.4 indicate good fit."],
                  result.f_stat != null         && ["F-stat",    result.f_stat?.toFixed(3),         "F-test: tests whether the model as a whole explains significantly more variance than no predictors. Large F with small p = model is useful."],
                  result.aic != null            && ["AIC",       result.aic?.toFixed(2),            "Akaike Information Criterion — lower is better. Used to compare models: the model with the lowest AIC balances fit and complexity best."],
                  result.bic != null            && ["BIC",       result.bic?.toFixed(2),            "Bayesian Information Criterion — similar to AIC but applies a larger penalty for extra parameters. Prefer the model with the lower BIC."],
                  result.concordance != null    && ["C-index",   result.concordance?.toFixed(4),    "Concordance index for Cox models — equivalent to AUC. Probability that the model ranks a higher-risk patient above a lower-risk patient."],
                ].filter(Boolean).map(([k, v, tip]: any) => (
                  <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                    <p className="text-xs text-gray-400 flex items-center">
                      {k}
                      {tip && <Tip text={tip} wide />}
                    </p>
                    <p className="text-gray-900 font-semibold">{v}</p>
                  </div>
                ))}
              </div>
              {/* Missing-data exclusion notice */}
              {result.n_excluded != null && result.n_excluded > 0 && (
                <div className="mt-3">
                  <InfoBanner>
                    {result.n_excluded} row{result.n_excluded !== 1 ? "s" : ""} were excluded due to missing values
                    {result.imputation && result.imputation !== "listwise" ? ` (${result.imputation} imputation applied to numeric columns)` : " (listwise deletion)"}.
                    Model was fitted on <strong>{result.n}</strong> of <strong>{result.n_total ?? (result.n + result.n_excluded)}</strong> rows.
                  </InfoBanner>
                </div>
              )}
              {/* Plain-English model fit interpretation */}
              {result.r_squared != null && (
                <div className="mt-3">
                  <InfoBanner>
                    The model explains <strong>{(result.r_squared * 100).toFixed(1)}%</strong> of the variance in <em>{result.outcome ?? "the outcome"}</em>.{" "}
                    {result.r_squared >= 0.7 ? "This is a strong fit." : result.r_squared >= 0.4 ? "This is a moderate fit — other factors likely also play a role." : "This is a weak fit — important predictors may be missing."}
                    {result.adj_r_squared != null && result.adj_r_squared < result.r_squared - 0.05 && " Note: Adjusted R² is notably lower than R², suggesting some predictors may not be contributing meaningfully."}
                  </InfoBanner>
                </div>
              )}
              {result.pseudo_r2 != null && !result.omnibus && (
                <div className="mt-3">
                  <InfoBanner>
                    Pseudo R² = {result.pseudo_r2?.toFixed(3)}.{" "}
                    {result.pseudo_r2 >= 0.4 ? "Excellent model fit." : result.pseudo_r2 >= 0.2 ? "Good model fit." : result.pseudo_r2 >= 0.1 ? "Moderate model fit." : "Weak model fit — consider adding more informative predictors."}
                  </InfoBanner>
                </div>
              )}

              {/* ── SPSS-style Model Summary (logistic only) ── */}
              {result.omnibus && (
                <div className="mt-3">
                  <ModelSummaryTable s={result} />
                </div>
              )}
            </div>

            {/* Coefficients table + detail panel */}
            {result.coefficients && (
              <div className="panel">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="font-semibold text-gray-900">
                    {model === "cox" ? "Coefficients (Hazard Ratios)" : model === "logistic" ? "Coefficients (Odds Ratios)" : model === "poisson" ? "Coefficients (Incidence Rate Ratios)" : "Coefficients"}
                    {model === "linear" && <Tip text="Each β coefficient shows how much the outcome changes for a 1-unit increase in that predictor, holding all others constant. Significant predictors (p < 0.05) are highlighted." wide />}
                    {model === "logistic" && <Tip text="Odds Ratio (OR) > 1 means higher odds of the outcome; OR < 1 means lower odds. E.g. OR = 2.0 means the outcome is twice as likely per unit increase. 95% CI not crossing 1 = significant." wide />}
                    {model === "cox" && <Tip text="Hazard Ratio (HR) > 1 means a higher rate of the event over time; HR < 1 means a protective effect. E.g. HR = 1.5 means 50% higher event rate per unit increase." wide />}
                    {model === "poisson" && <Tip text="Incidence Rate Ratio (IRR) = eβ. IRR > 1 means higher event rate; IRR < 1 means lower rate. Use for count outcomes (hospital admissions, episodes, etc.)." wide />}
                  </h4>
                  {/* Null hypothesis radio */}
                  <div className="flex items-center gap-3 text-xs text-gray-500">
                    <span className="text-gray-400">H₀:</span>
                    {([["eq", "β = 0"], ["leq", "β ≤ 0"], ["geq", "β ≥ 0"]] as const).map(([v, lbl]) => (
                      <label key={v} className="flex items-center gap-1 cursor-pointer">
                        <input type="radio" name="nullhyp" value={v} checked={nullHyp === v}
                          onChange={() => { setNullHyp(v); setSelectedCoefIdx(null); }}
                          className="accent-indigo-500" />
                        <span>{lbl}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <CoefTable
                  coefs={result.coefficients}
                  hrMode={model === "cox"}
                  allColumns={allCols}
                  selectedIdx={selectedCoefIdx}
                  onSelect={(i) => setSelectedCoefIdx((prev) => prev === i ? null : i)}
                  nullHyp={nullHyp}
                />
                {selectedCoefIdx != null && result.coefficients[selectedCoefIdx] && (
                  <div className="mt-3">
                    <CoefDetailPanel
                      coef={result.coefficients[selectedCoefIdx]}
                      nullHyp={nullHyp}
                      onClose={() => setSelectedCoefIdx(null)}
                    />
                  </div>
                )}
                <p className="text-[10px] text-gray-400 mt-2">Click a row to see the coefficient's sampling distribution.</p>
              </div>
            )}

            {/* Prediction Panel — linear only */}
            {model === "linear" && result.predictor_info && Object.keys(result.predictor_info).length > 0 && (
              <PredictionPanel result={result} />
            )}

            {/* Forest plot — logistic or cox */}
            {result.coefficients && (model === "logistic" || model === "cox") &&
              result.coefficients.filter((c: any) => c.variable !== "const").length > 0 && (
              <div className="panel">
                <h4 className="font-semibold text-gray-900 mb-2">
                  Forest Plot
                  <Tip text="Each row shows one predictor. The square is the point estimate (OR or HR); the horizontal line is the 95% Confidence Interval. If the CI crosses 1 (the vertical dashed line), the effect is not statistically significant. Larger squares = more precise estimate." wide />
                  <span className="ml-2 text-xs font-normal text-gray-400">
                    {model === "cox" ? "HR" : "OR"} with 95% CI — colored = p&lt;0.05, square size = precision
                  </span>
                </h4>
                <ForestPlot result={result} modelType={model} outcome={result.outcome} />
              </div>
            )}

            {/* Results text for all regression models */}
            {result.result_text && !result.table && (
              <div className="panel">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900">Results Paragraph</h4>
                  <button onClick={() => navigator.clipboard.writeText(result.result_text)} className="text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
                </div>
                <p className="text-sm text-gray-700 leading-relaxed bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">{result.result_text}</p>
              </div>
            )}

            {/* OR Table (Uni + Multi) */}
            {result.table && (
              <div className="panel">
                <h4 className="font-semibold text-gray-900 mb-2">
                  Univariate &amp; Multivariate OR Table
                  <Tip text="Univariate: each predictor tested alone against the outcome. Multivariate: all selected predictors tested together, adjusting for each other. Compare both columns — a variable that is significant univariately but not multivariately may be confounded by another predictor." wide />
                </h4>
                <ORTable
                  rows={result.table}
                  outcome={result.outcome}
                  selectionMethod={result.selection_method}
                  nMulti={result.n_multi}
                  nTotal={result.n_total}
                />
              </div>
            )}

            {/* SPSS-style model stats for OR Table multivariate model */}
            {result.model_stats && (
              <div className="panel">
                <h4 className="font-semibold text-gray-900 mb-2">Multivariate Model Summary</h4>
                <ModelSummaryTable s={result.model_stats} />
              </div>
            )}

            {/* Auto-generated results text */}
            {result.result_text && (
              <div className="panel">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900">Results Paragraph</h4>
                  <button onClick={() => navigator.clipboard.writeText(result.result_text)} className="text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
                </div>
                <p className="text-sm text-gray-700 leading-relaxed bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">{result.result_text}</p>
              </div>
            )}

            {/* Forest plot — OR table */}
            {result.table && result.table.length > 0 && (
              <div className="panel">
                <h4 className="font-semibold text-gray-900 mb-2">
                  Forest Plot
                  <span className="ml-2 text-xs font-normal text-gray-400">
                    ● Univariate &nbsp;◆ Multivariate — colored = p&lt;0.05, square size = precision
                  </span>
                </h4>
                <ForestPlot result={result} modelType={model} outcome={result.outcome} />
              </div>
            )}

            {/* KM curves */}
            {result.groups && (() => {
              const kmGroups: any[]  = result.groups;
              const nG               = kmGroups.length;

              // ── 95% CI via Greenwood's formula ──────────────────────────────
              const groupCI = kmGroups.map((g: any) => {
                let cumVar = 0;
                return g.curve.map((p: any) => {
                  const n = p.n_at_risk ?? 0, d = p.events ?? 0, S = p.survival;
                  if (n > 0 && d > 0 && n > d) cumVar += d / (n * (n - d));
                  const se = S * Math.sqrt(cumVar);
                  return { lower: Math.max(0, S - 1.96 * se), upper: Math.min(1, S + 1.96 * se) };
                });
              });

              // ── Risk table time ticks ───────────────────────────────────────
              const maxT     = Math.max(...kmGroups.flatMap((g: any) => g.curve.map((p: any) => p.time)), 1);
              const N_TICKS  = 6;
              const riskTimes = Array.from({ length: N_TICKS }, (_, i) => Math.round(i * maxT / (N_TICKS - 1)));
              const getRiskAt = (curve: any[], t: number) => {
                const pts = curve.filter((p: any) => p.time <= t);
                return pts.length > 0 ? (pts[pts.length - 1].n_at_risk ?? "—") : (curve[0]?.n_at_risk ?? "—");
              };

              // ── Layout dimensions ────────────────────────────────────────────
              const riskFrac = showKMrisktable ? Math.min(0.38, 0.10 + nG * 0.08) : 0;
              const plotH    = showKMrisktable ? 460 + nG * 18 : 440;

              // ── Build traces ─────────────────────────────────────────────────
              const traces: any[] = [];
              kmGroups.forEach((g: any, i: number) => {
                const style = kmStyles[i] ?? { color: KM_PALETTE[i % KM_PALETTE.length], width: 2, dash: "solid" };
                const ci    = groupCI[i];
                const times     = g.curve.map((p: any) => p.time);
                const survivals = g.curve.map((p: any) => p.survival);

                // CI band — upper boundary (invisible line)
                if (showKMci) {
                  traces.push({
                    type: "scatter", mode: "lines",
                    x: times, y: ci.map((c: any) => c.upper),
                    line: { width: 0, shape: "hv" }, showlegend: false, hoverinfo: "skip",
                    name: `__ci_u_${i}`,
                  });
                  // CI band — lower boundary with fill back to upper
                  traces.push({
                    type: "scatter", mode: "lines",
                    x: times, y: ci.map((c: any) => c.lower),
                    fill: "tonexty", fillcolor: `${style.color}28`,
                    line: { width: 0, shape: "hv" }, showlegend: false, hoverinfo: "skip",
                    name: `__ci_l_${i}`,
                  });
                }

                // Censoring tick marks (vertical stroke)
                if (showKMcensor) {
                  const censorPts = g.curve.filter((p: any) => (p.events === 0) && p.n_at_risk != null);
                  if (censorPts.length > 0 && censorPts.length < g.curve.length) {
                    traces.push({
                      type: "scatter", mode: "markers",
                      x: censorPts.map((p: any) => p.time),
                      y: censorPts.map((p: any) => p.survival),
                      marker: { symbol: "line-ns-open", size: 9, color: style.color, line: { color: style.color, width: 1.5 } },
                      showlegend: false, hoverinfo: "skip",
                      name: `__censor_${i}`,
                    });
                  }
                }

                // Main KM step curve
                traces.push({
                  type: "scatter", mode: "lines",
                  x: times, y: survivals,
                  name: `${g.group} (n=${g.n})`,
                  line: { color: style.color, width: style.width, dash: style.dash, shape: "hv" },
                });
              });

              // ── Risk table annotations ────────────────────────────────────────
              const riskAnnotations: any[] = [];
              if (showKMrisktable) {
                const rowH = (riskFrac - 0.04) / (nG + 0.8);

                riskAnnotations.push({
                  xref: "paper", yref: "paper",
                  x: 0.0, y: riskFrac - 0.01,
                  xanchor: "left", yanchor: "top",
                  text: "<b>Number at risk</b>",
                  showarrow: false,
                  font: { color: "#374151", size: 10 },
                });

                kmGroups.forEach((g: any, i: number) => {
                  const color = kmStyles[i]?.color ?? KM_PALETTE[i % KM_PALETTE.length];
                  const yPos  = riskFrac - 0.05 - i * rowH;

                  // Group label (left margin)
                  riskAnnotations.push({
                    xref: "paper", yref: "paper",
                    x: -0.01, y: yPos,
                    xanchor: "right", yanchor: "middle",
                    text: g.group,
                    showarrow: false,
                    font: { color, size: 10 },
                  });

                  // Risk counts aligned to x-axis
                  riskTimes.forEach((t: number) => {
                    riskAnnotations.push({
                      xref: "x", yref: "paper",
                      x: t, y: yPos,
                      xanchor: "center", yanchor: "middle",
                      text: String(getRiskAt(g.curve, t)),
                      showarrow: false,
                      font: { size: 10, color: "#374151", family: "monospace" },
                    });
                  });
                });
              }

              // p-value annotation
              const pAnnotation = result.logrank?.p != null ? [{
                xref: "paper", yref: "paper",
                x: 0.02, y: 0.98,
                xanchor: "left", yanchor: "top",
                text: `Log-rank p ${result.logrank.p < 0.001 ? "< 0.001" : `= ${result.logrank.p.toFixed(3)}`}`,
                showarrow: false,
                font: { color: result.logrank.p < 0.05 ? "#6366f1" : "#374151", size: 12 },
                bgcolor: "rgba(249,250,251,0.9)", borderpad: 5, bordercolor: "#e5e7eb", borderwidth: 1,
              }] : [];

              return (
                <div className="panel space-y-3">

                  {/* ── Header row ── */}
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500">
                      {kmGroups.map((g: any, i: number) => {
                        const c = kmStyles[i]?.color ?? KM_PALETTE[i % KM_PALETTE.length];
                        return (
                          <span key={g.group} className="flex items-center gap-1.5">
                            <span className="inline-block rounded" style={{ backgroundColor: c, width: 16, height: 3 }} />
                            <span className="text-gray-700 font-medium">{g.group}</span>
                            <span>n={g.n}, events={g.events}</span>
                            {g.median_survival != null && (
                              <span className="text-gray-400">(med {g.median_survival.toFixed(1)})</span>
                            )}
                          </span>
                        );
                      })}
                      {result.logrank && (
                        <span className="font-medium flex items-center gap-1">
                          Log-rank p
                          <Tip text="The log-rank test compares survival curves between groups. p < 0.05 means survival differs significantly between groups. It is most reliable when survival curves do not cross." wide />
                          {" "}
                          <span className={result.logrank.p < 0.05 ? "text-indigo-600" : ""}>
                            {result.logrank.p != null
                              ? (result.logrank.p < 0.001 ? "< 0.001" : `= ${result.logrank.p.toFixed(3)}`)
                              : "N/A"}
                          </span>
                          {result.logrank.p != null && (
                            <span className="text-xs font-normal text-gray-400">
                              {result.logrank.p < 0.001 ? "— highly significant" : result.logrank.p < 0.05 ? "— significant" : "— not significant"}
                            </span>
                          )}
                        </span>
                      )}
                    </div>
                    <div className="flex gap-2 flex-shrink-0">
                      <button onClick={exportKMcsv}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs text-gray-600 border border-gray-300 hover:bg-gray-100 transition-colors">
                        ↓ CSV
                      </button>
                      <button
                        onClick={() => {
                          if (!kmPlotRef.current) return;
                          const Plotly = (window as any).Plotly;
                          if (Plotly) Plotly.downloadImage(kmPlotRef.current, {
                            format: "png", width: 900, height: plotH + 80,
                            filename: `KM_${durationCol}_${eventCol}`,
                          });
                        }}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs text-gray-600 border border-gray-300 hover:bg-gray-100 transition-colors">
                        ↓ PNG
                      </button>
                    </div>
                  </div>

                  {/* ── Feature toggles + per-group style controls ── */}
                  <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-3 py-2.5 bg-gray-50 border border-gray-200 rounded-lg">
                    {/* Toggles */}
                    <div className="flex items-center gap-3 flex-shrink-0">
                      {([
                        ["95% CI",          showKMci,        setShowKMci],
                        ["Censoring marks", showKMcensor,    setShowKMcensor],
                        ["Risk table",      showKMrisktable, setShowKMrisktable],
                      ] as [string, boolean, (v: boolean) => void][]).map(([label, val, setter]) => (
                        <label key={label} className="flex items-center gap-1.5 cursor-pointer text-xs text-gray-600 select-none">
                          <input type="checkbox" checked={val} onChange={(e) => setter(e.target.checked)} className="accent-indigo-500" />
                          {label}
                        </label>
                      ))}
                    </div>

                    <div className="w-px h-5 bg-gray-300 flex-shrink-0" />

                    {/* Per-group style */}
                    {kmGroups.map((g: any, i: number) => {
                      const style = kmStyles[i] ?? { color: KM_PALETTE[i % KM_PALETTE.length], width: 2, dash: "solid" };
                      return (
                        <div key={g.group} className="flex items-center gap-2 text-xs">
                          <span className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: style.color }} />
                          <span className="text-gray-600 font-medium max-w-[80px] truncate" title={g.group}>{g.group}</span>
                          <input type="color" value={style.color}
                            onChange={(e) => updateKmStyle(i, { color: e.target.value })}
                            className="w-6 h-6 rounded cursor-pointer border border-gray-300 p-0" />
                          <select value={style.width}
                            onChange={(e) => updateKmStyle(i, { width: +e.target.value })}
                            className="select text-xs py-0.5 px-1.5 min-w-0">
                            {[1, 1.5, 2, 2.5, 3, 4].map((w) => <option key={w} value={w}>{w}px</option>)}
                          </select>
                          <select value={style.dash}
                            onChange={(e) => updateKmStyle(i, { dash: e.target.value })}
                            className="select text-xs py-0.5 px-1.5 min-w-0">
                            {KM_DASHES.map((d) => <option key={d} value={d}>{d}</option>)}
                          </select>
                        </div>
                      );
                    })}
                  </div>

                  {/* ── Plot ── */}
                  <div style={{ height: plotH }}>
                    <Plot
                      onInitialized={(_: object, gd: HTMLElement) => { kmPlotRef.current = gd; }}
                      onUpdate={(_: object, gd: HTMLElement)      => { kmPlotRef.current = gd; }}
                      data={traces}
                      layout={{
                        ...PLOT_LAYOUT,
                        autosize: true,
                        height: plotH,
                        margin: { t: 30, r: 20, b: 50, l: 90 },
                        yaxis: {
                          ...PLOT_LAYOUT.yaxis,
                          showgrid: showGrid,
                          domain: showKMrisktable ? [riskFrac, 1] : [0, 1],
                          range: [0, 1.05],
                          title: { text: "Survival probability" },
                        },
                        xaxis: { ...PLOT_LAYOUT.xaxis, showgrid: showGrid, title: { text: `Time (${durationCol})` } },
                        legend: {
                          font: { color: "#374151", size: 11 },
                          orientation: "h",
                          y: showKMrisktable ? -(riskFrac + 0.04) : -0.18,
                          bgcolor: "rgba(249,250,251,0.9)",
                          bordercolor: "#e5e7eb", borderwidth: 1,
                        },
                        annotations: [...pAnnotation, ...riskAnnotations],
                      }}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                      config={{
                        responsive: true,
                        displaylogo: false,
                        toImageButtonOptions: {
                          format: "png", filename: `KM_${durationCol}_${eventCol}`,
                          width: 900, height: plotH + 80,
                        },
                        modeBarButtonsToRemove: ["select2d", "lasso2d"],
                      }}
                    />
                  </div>
                </div>
              );
            })()}

            {/* KM stratified small-multiples grid */}
            {result.strata && (() => {
              const strata: any[] = result.strata;
              const nCols = strata.length <= 2 ? strata.length : strata.length === 3 ? 3 : 2;
              const palette = _pal();
              const miniH = 280;

              const buildMiniTraces = (groups: any[]) => {
                const traces: any[] = [];
                groups.forEach((g: any, i: number) => {
                  const color = palette[i % palette.length];
                  const times = g.curve.map((p: any) => p.time);
                  const survs = g.curve.map((p: any) => p.survival);
                  // CI via Greenwood
                  let cumVar = 0;
                  const upper: number[] = [], lower: number[] = [];
                  g.curve.forEach((p: any) => {
                    const n = p.n_at_risk ?? 0, d = p.events ?? 0, S = p.survival;
                    if (n > 0 && d > 0 && n > d) cumVar += d / (n * (n - d));
                    const se = S * Math.sqrt(cumVar);
                    upper.push(Math.min(1, S + 1.96 * se));
                    lower.push(Math.max(0, S - 1.96 * se));
                  });
                  traces.push({ type: "scatter", mode: "lines", x: times, y: upper, line: { width: 0, shape: "hv" }, showlegend: false, hoverinfo: "skip", name: `__u${i}` });
                  traces.push({ type: "scatter", mode: "lines", x: times, y: lower, fill: "tonexty", fillcolor: `${color}28`, line: { width: 0, shape: "hv" }, showlegend: false, hoverinfo: "skip", name: `__l${i}` });
                  traces.push({ type: "scatter", mode: "lines", x: times, y: survs, name: `${g.group} (n=${g.n})`, line: { color, width: 2, shape: "hv" } });
                });
                return traces;
              };

              return (
                <div className="panel space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold text-gray-900 text-sm">
                      Stratified by <span className="text-indigo-600">{result.stratify_col}</span>
                      {groupCol && <span className="text-gray-400 font-normal ml-2">— curves by {groupCol}</span>}
                    </h4>
                    <span className="text-xs text-gray-400">{strata.length} strata · {result.n_total} total</span>
                  </div>
                  <div className={`grid gap-4`} style={{ gridTemplateColumns: `repeat(${nCols}, minmax(0, 1fr))` }}>
                    {strata.map((stratum: any) => {
                      const pAnnot = stratum.logrank?.p != null ? [{
                        xref: "paper", yref: "paper", x: 0.02, y: 0.98,
                        xanchor: "left", yanchor: "top",
                        text: `p ${stratum.logrank.p < 0.001 ? "< 0.001" : `= ${stratum.logrank.p.toFixed(3)}`}`,
                        showarrow: false,
                        font: { size: 11, color: stratum.logrank.p < 0.05 ? "#6366f1" : "#6b7280" },
                        bgcolor: "rgba(249,250,251,0.85)", borderpad: 3, bordercolor: "#e5e7eb", borderwidth: 1,
                      }] : [];
                      return (
                        <div key={stratum.label} className="border border-gray-200 rounded-lg overflow-hidden">
                          <div className="px-3 py-1.5 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
                            <span className="text-xs font-semibold text-gray-700">{stratum.label}</span>
                            <span className="text-[10px] text-gray-400">n={stratum.n}</span>
                          </div>
                          <Plot
                            data={buildMiniTraces(stratum.groups)}
                            layout={{
                              ...PLOT_LAYOUT,
                              autosize: true,
                              height: miniH,
                              margin: { t: 10, r: 10, b: 40, l: 50 },
                              yaxis: { ...PLOT_LAYOUT.yaxis, range: [0, 1.05], title: { text: "Survival" }, showgrid: showGrid },
                              xaxis: { ...PLOT_LAYOUT.xaxis, title: { text: durationCol }, showgrid: showGrid },
                              legend: { font: { size: 9 }, orientation: "h", y: -0.22 },
                              annotations: pAnnot,
                            }}
                            style={{ width: "100%", height: miniH }}
                            useResizeHandler
                            config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] }}
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })()}
          </>
        ) : (
          <div className="panel h-64 flex items-center justify-center text-gray-400">
            Configure and fit a model
          </div>
        )}
      </div>
    </div>
  );
}
