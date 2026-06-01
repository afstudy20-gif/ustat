import { useState, useRef } from "react";
import Plot from "../../PlotComponent";
import { useStore } from "../../store";
import ResultExporter from "../ResultExporter";
import PlotExporter from "../PlotExporter";
import { fmtP } from "../../lib/format";

// Result-rendering subcomponents for ModelsPanel: coefficient/OR tables, the
// forest plot, prediction panel, and coefficient/summary detail views. Extracted
// from ModelsPanel.tsx (was 2509 LOC) to keep the panel focused on state + flow.

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

export function CoefTable({
  coefs, hrMode = false, allColumns = [], selectedIdx = null, onSelect, nullHyp = "eq",
}: {
  coefs: any[]; hrMode?: boolean; allColumns?: string[];
  selectedIdx?: number | null; onSelect?: (i: number) => void; nullHyp?: string;
}) {
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

export function ORTable({ rows, outcome, selectionMethod, nMulti, nTotal }: {
  rows: any[];
  outcome: string;
  selectionMethod?: string;
  nMulti?: number;
  nTotal?: number;
}) {
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
export function varLabel(v: string): string {
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

export function ForestPlot({ result, modelType, outcome }: {
  result: any;
  modelType: string;
  outcome?: string;
}) {
  const forestRef = useRef<any>(null);
  const isORTable = modelType === "ortable" || modelType === "firth_ortable";
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
  const fmtCI = (est: number | null, lo: number | null, hi: number | null) =>
    est == null ? "—" : `${est.toFixed(2)} (${lo?.toFixed(2)}–${hi?.toFixed(2)})`;

  // Base props for layout annotations
  const AB = { showarrow: false, xanchor: "left" as const, yanchor: "middle" as const };
  const HDR = { size: 9, color: "#374151" };

  // Directional arrow labels below x-axis (within forest domain).
  // forestRight = right edge of forest domain in paper coords.
  // Pushed below the x-axis tick numbers (which sit at roughly y = -0.05)
  // so "◀ Reduces risk" and "Increases risk ▶" no longer overlap with
  // 0.4, 0.6, … 2.4 tick labels.
  const dirAnnotations = (forestRight: number, yPos = -0.20) => [
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
    // area. Bottom margin grows when the legend is on to make room. Floor
    // raised so the "◀ Reduces risk / Increases risk ▶" annotations at
    // y=-0.20 land below the x-axis tick numbers rather than on top of
    // them.
    const bottomPad = opts.showLegend ? 130 : 80;

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
        margin: { t: opts.customTitle ? (opts.customSubtitle ? 70 : 50) : 20, r: 20, b: opts.showLegend ? 130 : 90, l: 180 },
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
export function PredictionPanel({ result }: { result: any }) {
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
export function CoefDetailPanel({
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

export function ModelSummaryTable({ s }: { s: any }) {
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

