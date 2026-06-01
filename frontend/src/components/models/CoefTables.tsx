import ResultExporter from "../ResultExporter";
import { fmtP } from "../../lib/format";
import { adjustP, MiniNormalSVG, SigBar } from "./shared";

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

