import { useState, useEffect, useRef } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { runLinear, runLogistic, runKM, runCox, runLogisticTable, runRCS } from "../api";
import { Tip, InfoBanner } from "./Tip";
import { MissingGuard, type ImputationStrategy } from "./MissingGuard";

const PLOT_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  font: { color: "#374151", size: 12 },
  margin: { t: 30, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: "#e5e7eb" },
  yaxis: { gridcolor: "#e5e7eb" },
};

function CoefTable({ coefs, hrMode = false }: { coefs: any[]; hrMode?: boolean }) {
  const fmtP = (p: number) => (p < 0.001 ? "<0.001" : p.toFixed(4));
  const sig   = (p: number) => p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "";

  // Detect logistic mode: coefficients have odds_ratio + or_ci_low fields
  const isLogistic = !hrMode && coefs.length > 0 && coefs[0].odds_ratio != null;

  // ── Logistic regression table ────────────────────────────────────────────
  if (isLogistic) {
    return (
      <div className="overflow-auto rounded border border-gray-200 mt-3">
        <table>
          <thead>
            <tr>
              <th>Variable</th>
              <th title="Log-Odds (β)">Estimate (Log-Odds)</th>
              <th>SE</th>
              <th>z</th>
              <th>p-value</th>
              <th title="Odds Ratio = e^β">Odds Ratio</th>
              <th title="95% CI for Odds Ratio">CI 95% (OR)</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {coefs.map((c: any) => (
              <tr key={c.variable}>
                <td className="font-mono text-xs text-gray-900">{c.variable}</td>
                <td className="font-mono">{c.log_odds?.toFixed(4)}</td>
                <td>{c.se?.toFixed(4)}</td>
                <td>{c.z?.toFixed(3)}</td>
                <td>
                  <span className={c.p < 0.05 ? "badge-sig" : "badge-ns"}>{fmtP(c.p)}</span>
                </td>
                <td className={`font-mono font-semibold ${c.p < 0.05 ? "text-indigo-600" : ""}`}>
                  {c.odds_ratio?.toFixed(3)}
                </td>
                <td className="font-mono text-xs text-gray-400">
                  {c.or_ci_low != null ? `${c.or_ci_low.toFixed(3)} – ${c.or_ci_high.toFixed(3)}` : "–"}
                </td>
                <td className="text-yellow-400 font-bold">{sig(c.p)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  // ── Linear / Cox (HR) table ──────────────────────────────────────────────
  return (
    <div className="overflow-auto rounded border border-gray-200 mt-3">
      <table>
        <thead>
          <tr>
            <th>Variable</th>
            {hrMode ? <th>HR</th> : <th>Estimate</th>}
            <th>SE</th>
            {hrMode ? <th>Z</th> : <th>t / z</th>}
            <th>p-value</th>
            <th>CI (95%)</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {coefs.map((c: any) => {
            const est = hrMode ? c.hr : c.estimate ?? c.log_hr;
            const ci = hrMode
              ? `${c.hr_ci_low?.toFixed(3)} – ${c.hr_ci_high?.toFixed(3)}`
              : c.ci_low != null
              ? `${c.ci_low.toFixed(3)} – ${c.ci_high.toFixed(3)}`
              : "–";
            return (
              <tr key={c.variable}>
                <td className="font-mono text-xs text-gray-900">{c.variable}</td>
                <td>{typeof est === "number" ? est.toFixed(4) : est}</td>
                <td>{c.se?.toFixed(4)}</td>
                <td>{(c.t ?? c.z)?.toFixed(3)}</td>
                <td>
                  <span className={c.p < 0.05 ? "badge-sig" : "badge-ns"}>{fmtP(c.p)}</span>
                </td>
                <td className="font-mono text-xs">{ci}</td>
                <td className="text-yellow-400 font-bold">{sig(c.p)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
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

  const exportCSV = () => {
    const header = "Variable,Uni OR,Uni CI low,Uni CI high,Uni p,Multi OR,Multi CI low,Multi CI high,Multi p";
    const dataRows = rows.map((r) =>
      [
        r.variable,
        r.uni_or?.toFixed(4) ?? "",
        r.uni_ci_low?.toFixed(4) ?? "",
        r.uni_ci_high?.toFixed(4) ?? "",
        r.uni_p?.toFixed(6) ?? "",
        r.multi_or?.toFixed(4) ?? "",
        r.multi_ci_low?.toFixed(4) ?? "",
        r.multi_ci_high?.toFixed(4) ?? "",
        r.multi_p?.toFixed(6) ?? "",
      ].join(",")
    );
    const csv = ["\uFEFF" + `Outcome: ${outcome}`, header, ...dataRows].join("\r\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `OR_Table_${outcome}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-gray-400">Outcome: <span className="text-gray-700 font-mono">{outcome}</span></p>
        <button
          onClick={exportCSV}
          className="flex items-center gap-1 px-2 py-1 rounded text-xs text-gray-400 hover:text-gray-900 hover:bg-gray-200 border border-gray-300 transition-colors"
          title="Export OR table as CSV"
        >
          ↓ Export CSV
        </button>
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

function ForestPlot({ result, modelType, outcome }: {
  result: any;
  modelType: string;
  outcome?: string;
}) {
  const isORTable = modelType === "ortable";
  const isCox     = modelType === "cox";
  const metric    = isCox ? "HR" : "OR";
  const showGrid  = useStore((s) => s.showGrid);

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

    // xaxis.domain = [0, 0.47] → forest occupies left 47% of the figure's plot area.
    // Text columns sit at paper x ∈ [0.49, 0.78] — safely INSIDE the SVG, no clipping.
    const TX1 = 0.49;  // OR (95% CI) column start
    const TX2 = 0.76;  // p-value column start

    const annotations: object[] = [
      { ...AB, xref: "paper", yref: "paper", x: TX1, y: 1.055,
        text: "<b>OR (95% CI)</b>", font: HDR },
      { ...AB, xref: "paper", yref: "paper", x: TX2, y: 1.055,
        text: "<b>p</b>", font: HDR },
      { ...AB, xref: "paper", yref: "paper", x: TX1, y: 1.012,
        text: "● Uni   ◆ Multi", font: { size: 8, color: "#4b5563" } },
      ...dirAnnotations(0.47),
    ];

    rows.forEach((r, i) => {
      // Univariate row — at same y-offset as uni marker (+0.18)
      if (r.uni_or != null) {
        const col = r.uni_p != null && r.uni_p < 0.05 ? "#818cf8" : "#6b7280";
        annotations.push(
          { ...AB, xref: "paper", yref: "y", x: TX1, y: i + 0.18,
            text: fmtCI(r.uni_or, r.uni_ci_low, r.uni_ci_high), font: { size: 9, color: col } },
          { ...AB, xref: "paper", yref: "y", x: TX2, y: i + 0.18,
            text: fmtP(r.uni_p), font: { size: 9, color: col } },
        );
      }
      // Multivariate row — at same y-offset as multi marker (-0.18)
      if (r.multi_or != null) {
        const col = r.multi_p != null && r.multi_p < 0.05 ? "#34d399" : "#6b7280";
        annotations.push(
          { ...AB, xref: "paper", yref: "y", x: TX1, y: i - 0.18,
            text: fmtCI(r.multi_or, r.multi_ci_low, r.multi_ci_high), font: { size: 9, color: col } },
          { ...AB, xref: "paper", yref: "y", x: TX2, y: i - 0.18,
            text: fmtP(r.multi_p), font: { size: 9, color: col } },
        );
      }
    });

    return (
      <Plot
        data={[
          {
            name: "Univariate",
            type: "scatter", mode: "markers",
            x: uniValid.map((r) => r.uni_or),
            y: uniValid.map((r) => yIdx[r.variable] + 0.18),
            error_x: {
              type: "data", symmetric: false,
              array:      uniValid.map((r) => r.uni_ci_high - r.uni_or),
              arrayminus: uniValid.map((r) => r.uni_or - r.uni_ci_low),
              color: "#6366f1", thickness: 2, width: 7,
            },
            marker: {
              size: uniValid.map((r) => precisionSize(r.uni_or, r.uni_ci_low, r.uni_ci_high)),
              symbol: "circle",
              color: uniValid.map((r) => r.uni_p != null && r.uni_p < 0.05 ? "#6366f1" : "#6b7280"),
              line: { color: "#d1d5db", width: 1 },
            },
            hovertemplate: uniValid.map((r) =>
              `<b>${r.variable}</b> (Univariate)<br>OR: ${r.uni_or?.toFixed(3)}<br>95% CI: ${r.uni_ci_low?.toFixed(3)} – ${r.uni_ci_high?.toFixed(3)}<br>p = ${fmtP(r.uni_p)}<extra></extra>`
            ),
          },
          {
            name: "Multivariate",
            type: "scatter", mode: "markers",
            x: multiValid.map((r) => r.multi_or),
            y: multiValid.map((r) => yIdx[r.variable] - 0.18),
            error_x: {
              type: "data", symmetric: false,
              array:      multiValid.map((r) => r.multi_ci_high - r.multi_or),
              arrayminus: multiValid.map((r) => r.multi_or - r.multi_ci_low),
              color: "#10b981", thickness: 2, width: 7,
            },
            marker: {
              size: multiValid.map((r) => precisionSize(r.multi_or, r.multi_ci_low, r.multi_ci_high)),
              symbol: "diamond",
              color: multiValid.map((r) => r.multi_p != null && r.multi_p < 0.05 ? "#10b981" : "#6b7280"),
              line: { color: "#d1d5db", width: 1 },
            },
            hovertemplate: multiValid.map((r) =>
              `<b>${r.variable}</b> (Multivariate)<br>OR: ${r.multi_or?.toFixed(3)}<br>95% CI: ${r.multi_ci_low?.toFixed(3)} – ${r.multi_ci_high?.toFixed(3)}<br>p = ${fmtP(r.multi_p)}<extra></extra>`
            ),
          },
        ]}
        layout={{
          ...FOREST_BASE,
          height: plotH,
          autosize: true,
          margin: { t: 20, r: 20, b: 70, l: 160 },
          xaxis: {
            ...FOREST_BASE.xaxis,
            showgrid: showGrid,
            domain: [0, 0.47],
            title: { text: `Odds Ratio (95% CI)${outcome ? ` — Outcome: ${outcome}` : ""}`, font: { size: 10, color: "#374151" } },
          },
          yaxis: {
            ...FOREST_BASE.yaxis,
            tickvals: rows.map((_, i) => i),
            ticktext: rows.map((r) => varLabel(r.variable)),
            autorange: "reversed" as const,
            range: [-0.5, n - 0.5],
          },
          shapes: [
            ...FOREST_BASE.shapes,
            { type: "line", xref: "paper", yref: "paper",
              x0: 0.48, x1: 0.48, y0: 0, y1: 1,
              line: { color: "#e5e7eb", width: 1 } },
          ],
          annotations,
          showlegend: true,
          legend: { font: { color: "#374151", size: 11 }, bgcolor: "rgba(249,250,251,0.9)", orientation: "h" as const, x: 0, y: -0.18 },
        }}
        style={{ width: "100%", height: plotH }}
        useResizeHandler
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      />
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

  // xaxis.domain = [0, 0.55] → forest left 55%; text at paper [0.57, 0.80] — no clipping.
  const TX1 = 0.57;
  const TX2 = 0.80;

  const annotations: object[] = [
    { ...AB, xref: "paper", yref: "paper", x: TX1, y: 1.06,
      text: `<b>${metric} (95% CI)</b>`, font: HDR },
    { ...AB, xref: "paper", yref: "paper", x: TX2, y: 1.06,
      text: "<b>p</b>", font: HDR },
    ...dirAnnotations(0.55),
    // Per-variable rows
    ...coefs.map((_: any, i: number) => {
      const col = pVals[i] < 0.05 ? COLOR_SIG : "#6b7280";
      return [
        { ...AB, xref: "paper", yref: "y", x: TX1, y: i,
          text: fmtCI(estimates[i], ciLow[i], ciHigh[i]), font: { size: 9, color: col } },
        { ...AB, xref: "paper", yref: "y", x: TX2, y: i,
          text: fmtP(pVals[i]), font: { size: 9, color: col } },
      ];
    }).flat(),
  ];

  return (
    <Plot
      data={[{
        type: "scatter", mode: "markers",
        x: estimates,
        y: coefs.map((_: any, i: number) => i),
        error_x: {
          type: "data", symmetric: false,
          array:      estimates.map((e: number, i: number) => (ciHigh[i] ?? e) - e),
          arrayminus: estimates.map((e: number, i: number) => e - (ciLow[i]  ?? e)),
          color: COLOR, thickness: 2.5, width: 9,
        },
        marker: {
          size: estimates.map((_: number, i: number) => precisionSize(estimates[i], ciLow[i], ciHigh[i])),
          symbol: "square",
          color: pVals.map((p: number) => p < 0.05 ? COLOR_SIG : "#6b7280"),
          line: { color: "#d1d5db", width: 1 },
        },
        hovertemplate: coefs.map((_: any, i: number) =>
          `<b>${labels[i]}</b><br>${metric}: ${estimates[i]?.toFixed(3)}<br>95% CI: ${ciLow[i]?.toFixed(3)} – ${ciHigh[i]?.toFixed(3)}<br>p = ${fmtP(pVals[i])}<extra></extra>`
        ),
        name: isCox ? "Hazard Ratio" : "Odds Ratio",
      }]}
      layout={{
        ...FOREST_BASE,
        height: plotH,
        autosize: true,
        margin: { t: 20, r: 20, b: 70, l: 160 },
        xaxis: {
          ...FOREST_BASE.xaxis,
          showgrid: showGrid,
          domain: [0, 0.55],
          title: {
            text: isCox
              ? `Hazard Ratio (95% CI)${outcome ? ` — Outcome: ${outcome}` : ""}`
              : `Odds Ratio (95% CI)${outcome ? ` — Outcome: ${outcome}` : ""}`,
            font: { size: 10, color: "#374151" },
          },
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
          { type: "line", xref: "paper", yref: "paper",
            x0: 0.56, x1: 0.56, y0: 0, y1: 1,
            line: { color: "#e5e7eb", width: 1 } },
        ],
        annotations,
        showlegend: false,
      }}
      style={{ width: "100%", height: plotH }}
      useResizeHandler
      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
    />
  );
}

export default function ModelsPanel() {
  const session  = useStore((s) => s.session);
  const showGrid = useStore((s) => s.showGrid);
  if (!session) return null;

  const numCols = session.columns.filter((c) => c.kind === "numeric").map((c) => c.name);
  const allCols = session.columns.map((c) => c.name);

  const [model, setModel] = useState("linear");
  const [outcome, setOutcome] = useState(numCols[0] ?? "");
  const [predictors, setPredictors] = useState<string[]>([]);

  // ── RCS-specific state ───────────────────────────────────────────────────
  const [rcsPredictor, setRcsPredictor] = useState(numCols[0] ?? "");
  const [rcsOutcome,   setRcsOutcome]   = useState(numCols[1] ?? numCols[0] ?? "");
  const [rcsNKnots,    setRcsNKnots]    = useState(4);
  const [rcsRefValue,  setRcsRefValue]  = useState("");
  const [rcsCovariates, setRcsCovariates] = useState<string[]>([]);
  const [scaleFactors, setScaleFactors] = useState<Record<string, string>>({}); // col → divisor string
  const [selection, setSelection] = useState("p10"); // multivariate variable selection strategy
  const [durationCol, setDurationCol] = useState(numCols[0] ?? "");
  const [eventCol, setEventCol] = useState(numCols[1] ?? "");
  const [groupCol, setGroupCol] = useState("");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const [predFilter, setPredFilter] = useState("");

  // ── KM curve styling ────────────────────────────────────────────────────────
  const KM_PALETTE = ["#6366f1","#f59e0b","#10b981","#ef4444","#8b5cf6","#06b6d4"];
  const KM_DASHES  = ["solid","dash","dot","dashdot"] as const;

  interface KmStyle { color: string; width: number; dash: string; }
  const [kmStyles, setKmStyles] = useState<KmStyle[]>([]);
  const kmPlotRef = useRef<any>(null);

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
    setLoading(true); setError(null); setResult(null);
    try {
      let res: any;
      const sf = buildScaleFactors();
      if (model === "linear") res = await runLinear({ session_id: sid, outcome, predictors, imputation });
      else if (model === "logistic") res = await runLogistic({ session_id: sid, outcome, predictors, scale_factors: sf, imputation });
      else if (model === "ortable") res = await runLogisticTable({ session_id: sid, outcome, predictors, scale_factors: sf, selection, imputation });
      else if (model === "km") res = await runKM({ session_id: sid, duration_col: durationCol, event_col: eventCol, group_col: groupCol || undefined, imputation });
      else if (model === "rcs") res = await runRCS({
        session_id: sid,
        predictor:  rcsPredictor,
        outcome:    rcsOutcome,
        covariates: rcsCovariates,
        n_knots:    rcsNKnots,
        ref_value:  rcsRefValue !== "" ? parseFloat(rcsRefValue) : undefined,
        model_type: "logistic",
      });
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

  const isSurvival = model === "km" || model === "cox";
  const isORTable  = model === "ortable";
  const isRCS      = model === "rcs";

  return (
    <div className="flex gap-4">
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Model</h3>
          {([
            ["linear",   "Linear Regression",       "Predict a continuous outcome (e.g. blood pressure) from one or more predictors. Output: β coefficients, R², p-values."],
            ["logistic", "Logistic Regression",      "Predict a binary outcome (0/1, yes/no) — outputs Odds Ratios showing how each predictor changes the odds of the event."],
            ["ortable",  "OR Table (Uni + Multi)",   "Run univariate logistic regression for each predictor separately, then all significant ones together in a multivariate model. Standard for clinical papers."],
            ["km",       "Kaplan-Meier",             "Plot survival over time, comparing curves between groups (e.g. treatment vs. control). Tests group differences with the log-rank test."],
            ["cox",      "Cox PH",                   "Regression for time-to-event data. Outputs Hazard Ratios (HR) — how much each predictor changes the rate of the event occurring over time."],
            ["rcs",      "RCS Dose-Response",        "Restricted Cubic Splines — models non-linear (U/J-shaped) relationships between a continuous predictor and a binary outcome. Outputs a publication-ready dose-response curve with 95% CI."],
          ] as const).map(([v, l, desc]) => (
            <label key={v} className="flex items-start gap-2 cursor-pointer group">
              <input type="radio" name="model" value={v} checked={model === v} onChange={() => { setModel(v); setResult(null); }} className="accent-indigo-500 mt-0.5" />
              <span className="text-sm text-gray-700 leading-tight">
                {l}
                <Tip text={desc} wide />
              </span>
            </label>
          ))}
        </div>

        <div className="panel space-y-3">
          {isRCS ? (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Predictor (continuous)</label>
                <select className="select w-full" value={rcsPredictor} onChange={(e) => setRcsPredictor(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Outcome (binary 0/1)</label>
                <select className="select w-full" value={rcsOutcome} onChange={(e) => setRcsOutcome(e.target.value)}>
                  {allCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Knots <Tip text="3 knots → [10th, 50th, 90th] percentiles. 4 knots (clinical standard) → [5th, 35th, 65th, 95th]. 5 knots → [5th, 27.5th, 50th, 72.5th, 95th]." wide />
                </label>
                <div className="flex gap-2">
                  {[3, 4, 5].map((k) => (
                    <button key={k} onClick={() => setRcsNKnots(k)}
                      className={`flex-1 py-1 text-xs rounded border transition-colors ${rcsNKnots === k ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-600 hover:bg-gray-50"}`}>
                      {k}{k === 4 ? " ★" : ""}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Reference value <Tip text="The OR = 1.0 reference point on the X-axis. Leave blank to use the median." />
                </label>
                <input type="number" placeholder="(median)" value={rcsRefValue}
                  onChange={(e) => setRcsRefValue(e.target.value)}
                  className="select w-full text-xs py-1" />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Covariates (optional)</label>
                <div className="max-h-32 overflow-y-auto space-y-1">
                  {numCols.filter((c) => c !== rcsPredictor && c !== rcsOutcome).map((c) => (
                    <label key={c} className="flex items-center gap-2 text-xs cursor-pointer">
                      <input type="checkbox" checked={rcsCovariates.includes(c)}
                        onChange={() => setRcsCovariates((prev) => prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c])}
                        className="accent-indigo-500" />
                      <span className="text-gray-700 truncate">{c}</span>
                    </label>
                  ))}
                </div>
              </div>
              <button className="btn-primary w-full" onClick={run} disabled={loading}>
                {loading ? "Fitting…" : "Fit RCS Model"}
              </button>
              {error && <p className="text-red-400 text-xs">{error}</p>}
            </>
          ) : isSurvival ? (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Duration column</label>
                <select className="select w-full" value={durationCol} onChange={(e) => setDurationCol(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Event column (0/1)</label>
                <select className="select w-full" value={eventCol} onChange={(e) => setEventCol(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              {model === "km" && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Group column (optional)</label>
                  <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                    <option value="">None</option>
                    {allCols.map((c) => <option key={c}>{c}</option>)}
                  </select>
                </div>
              )}
              {model === "cox" && (
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
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {allCols
                      .filter((c) => c !== durationCol && c !== eventCol && c.toLowerCase().includes(predFilter.toLowerCase()))
                      .map((c) => (
                        <label key={c} className="flex items-center gap-2 text-sm cursor-pointer">
                          <input type="checkbox" checked={predictors.includes(c)} onChange={() => togglePredictor(c)} className="accent-indigo-500" />
                          <span className="text-gray-700 truncate">{c}</span>
                        </label>
                      ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Outcome{isORTable && <span className="text-gray-400 ml-1">(binary 0/1)</span>}
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
                    return (
                      <div key={c} className="space-y-0.5">
                        <label className="flex items-center gap-2 text-sm cursor-pointer">
                          <input type="checkbox" checked={checked} onChange={() => togglePredictor(c)} className="accent-indigo-500" />
                          <span className="text-gray-700 truncate">{c}</span>
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
            columns={isSurvival ? [durationCol, eventCol, ...(model === "cox" ? predictors : [])] : [...predictors, outcome]}
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
        {result && isRCS ? (
          /* ── RCS dose-response result ─────────────────────────────────────── */
          <div className="panel space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="font-semibold text-gray-900">
                Dose-Response: {result.predictor} → {result.outcome}
              </h4>
              <div className="flex items-center gap-3 text-xs text-gray-500">
                <span>n = {result.n}{result.n_events != null ? `, events = ${result.n_events}` : ""}</span>
                {result.aic != null && <span>AIC = {result.aic?.toFixed(1)}</span>}
              </div>
            </div>

            {/* Knot positions */}
            <div className="flex flex-wrap gap-1.5 text-[11px]">
              <span className="text-gray-400">{result.n_knots} knots at:</span>
              {(result.knots as number[]).map((k: number, i: number) => (
                <span key={i} className="bg-indigo-50 border border-indigo-100 text-indigo-600 rounded px-1.5 py-0.5">{k}</span>
              ))}
              <span className="text-gray-400 ml-2">reference = <strong>{result.ref_value}</strong> (OR = 1.0)</span>
            </div>

            {/* Dose-response plot */}
            <Plot
              data={[
                /* CI band */
                {
                  type: "scatter" as const,
                  x: [...(result.x_values as number[]), ...(result.x_values as number[]).slice().reverse()],
                  y: [...(result.ci_high as number[]), ...(result.ci_low as number[]).slice().reverse()],
                  fill: "toself",
                  fillcolor: "rgba(99,102,241,0.12)",
                  line: { color: "transparent" },
                  hoverinfo: "skip",
                  showlegend: false,
                  name: "95% CI",
                },
                /* OR curve */
                {
                  type: "scatter" as const,
                  mode: "lines" as const,
                  x: result.x_values as number[],
                  y: result.or_values as number[],
                  line: { color: "#6366f1", width: 2.5 },
                  name: "Odds Ratio",
                  hovertemplate: `${result.predictor}: %{x:.2f}<br>OR: %{y:.3f}<extra></extra>`,
                },
                /* raw data rug */
                {
                  type: "scatter" as const,
                  mode: "markers" as const,
                  x: result.x_data as number[],
                  y: Array((result.x_data as number[]).length).fill(1),
                  marker: { color: "#6366f1", size: 3, opacity: 0.25, symbol: "line-ns-open" },
                  yaxis: "y2",
                  showlegend: false,
                  hoverinfo: "skip",
                },
              ]}
              layout={{
                ...PLOT_LAYOUT,
                autosize: true,
                height: 420,
                xaxis: { ...PLOT_LAYOUT.xaxis, showgrid: showGrid, title: { text: result.predictor } },
                yaxis: { ...PLOT_LAYOUT.yaxis, showgrid: showGrid, title: { text: "Odds Ratio (95% CI)" }, zeroline: false },
                yaxis2: { overlaying: "y" as const, range: [0.5, 1.5], fixedrange: true, showgrid: false, showticklabels: false, zeroline: false },
                shapes: [
                  /* OR = 1 reference line */
                  { type: "line" as const, xref: "paper" as const, yref: "y" as const,
                    x0: 0, x1: 1, y0: 1, y1: 1,
                    line: { color: "#9ca3af", width: 1.5, dash: "dash" as const } },
                  /* knot tick marks */
                  ...(result.knots as number[]).map((k: number) => ({
                    type: "line" as const, xref: "x" as const, yref: "paper" as const,
                    x0: k, x1: k, y0: 0, y1: 0.04,
                    line: { color: "#6366f1", width: 1, dash: "dot" as const },
                  })),
                ],
                annotations: [
                  { x: result.ref_value, y: 1, xref: "x" as const, yref: "y" as const,
                    text: `ref=${result.ref_value}`, showarrow: true, arrowhead: 2,
                    arrowcolor: "#9ca3af", font: { size: 10, color: "#6b7280" },
                    ax: 0, ay: -30 },
                ],
                legend: { font: { size: 11, color: "#374151" }, x: 0.01, y: 0.99, xanchor: "left" as const, yanchor: "top" as const },
                margin: { t: 20, r: 20, b: 50, l: 65 },
              }}
              style={{ width: "100%", height: 420 }}
              useResizeHandler
              config={{ responsive: true, displaylogo: false,
                toImageButtonOptions: { format: "png", filename: `RCS_${result.predictor}_${result.outcome}`, width: 900, height: 500 },
                modeBarButtonsToRemove: ["select2d", "lasso2d"] }}
            />

            <InfoBanner>
              The curve shows the <strong>non-linear dose-response</strong> relationship between <em>{result.predictor}</em> and the odds of <em>{result.outcome}</em>.
              The shaded band is the 95% CI — it widens at the extremes where data is sparse.
              Dashed vertical lines mark knot positions ({result.n_knots} knots at {result.n_knots === 3 ? "10th/50th/90th" : result.n_knots === 4 ? "5th/35th/65th/95th" : "5th/27.5th/50th/72.5th/95th"} percentiles).
              The horizontal dashed line at OR = 1.0 is the null reference.
            </InfoBanner>
          </div>
        ) : result ? (
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
              {result.pseudo_r2 != null && (
                <div className="mt-3">
                  <InfoBanner>
                    Pseudo R² = {result.pseudo_r2?.toFixed(3)}.{" "}
                    {result.pseudo_r2 >= 0.4 ? "Excellent model fit." : result.pseudo_r2 >= 0.2 ? "Good model fit." : result.pseudo_r2 >= 0.1 ? "Moderate model fit." : "Weak model fit — consider adding more informative predictors."}
                  </InfoBanner>
                </div>
              )}
            </div>

            {/* Coefficients table */}
            {result.coefficients && (
              <div className="panel">
                <h4 className="font-semibold text-gray-900 mb-1">
                  {model === "cox" ? "Coefficients (Hazard Ratios)" : model === "logistic" ? "Coefficients (Odds Ratios)" : "Coefficients"}
                  {model === "linear" && <Tip text="Each β coefficient shows how much the outcome changes for a 1-unit increase in that predictor, holding all others constant. Significant predictors (p < 0.05) are highlighted." wide />}
                  {model === "logistic" && <Tip text="Odds Ratio (OR) > 1 means higher odds of the outcome; OR < 1 means lower odds. E.g. OR = 2.0 means the outcome is twice as likely per unit increase. 95% CI not crossing 1 = significant." wide />}
                  {model === "cox" && <Tip text="Hazard Ratio (HR) > 1 means a higher rate of the event over time; HR < 1 means a protective effect. E.g. HR = 1.5 means 50% higher event rate per unit increase." wide />}
                </h4>
                <CoefTable coefs={result.coefficients} hrMode={model === "cox"} />
              </div>
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
