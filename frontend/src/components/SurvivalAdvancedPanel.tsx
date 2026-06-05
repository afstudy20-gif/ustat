import { useState, useEffect, useRef, useMemo } from "react";
import Plot from "../PlotComponent";
import { useStore } from "../store";
import { runFineGray, runEValue, runLandmark, runKM, runCox, runRMST, runRecurrentLWYY, runCoxHorizons } from "../api";
import { usePlotLayout, usePalette, useTraceDefaults } from "../plotStyle";
import ResultExporter from "./ResultExporter";
import PlotExporter from "./PlotExporter";
import { Tip } from "./Tip";
import ThreeCol from "./ThreeCol";

// ── Helpers ──────────────────────────────────────────────────────────────────

// In the 2-grid layout only the selected method renders, so the Section is
// always expanded — the header is a static card title, no collapse toggle.
function Section({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-800">{title}</h3>
        <p className="text-[11px] text-gray-400 mt-0.5">{description}</p>
      </div>
      <div className="px-5 py-4 space-y-4">{children}</div>
    </div>
  );
}

// Method registry — drives the left nav + which Section renders on the right.
type SurvMethod =
  | "km" | "cox" | "timehorizon" | "landmark" | "rmst"
  | "finegray" | "lwyy" | "evalue";

const SURV_METHODS: { id: SurvMethod; title: string; desc: string }[] = [
  { id: "km",          title: "Kaplan-Meier",        desc: "Survival curves + log-rank" },
  { id: "cox",         title: "Cox PH",              desc: "Hazard ratios" },
  { id: "timehorizon", title: "Time-horizon HR",     desc: "HR by follow-up window → forest" },
  { id: "landmark",    title: "Landmark",            desc: "Conditional on surviving to t" },
  { id: "rmst",        title: "RMST",                desc: "Restricted mean survival time" },
  { id: "finegray",    title: "Fine-Gray",           desc: "Competing risks (CIF)" },
  { id: "lwyy",        title: "Recurrent (LWYY)",    desc: "Repeated events" },
  { id: "evalue",      title: "E-value",             desc: "Unmeasured confounding" },
];

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

function MultiSelect({ label, columns, selected, onChange, kinds, excludeNames }: {
  label: string; columns: { name: string; kind: string }[];
  selected: string[]; onChange: (v: string[]) => void;
  kinds?: string[]; excludeNames?: string[];
}) {
  const filtered = (kinds ? columns.filter((c) => kinds.includes(c.kind)) : columns)
    .filter((c) => !(excludeNames ?? []).includes(c.name));
  const toggle = (name: string) =>
    onChange(selected.includes(name) ? selected.filter((c) => c !== name) : [...selected, name]);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500 font-medium">{label}</span>
        <div className="flex items-center gap-2 text-[10px] text-gray-400">
          <span>{selected.length} selected</span>
          {selected.length > 0 && (
            <button type="button" onClick={() => onChange([])} className="hover:text-red-500">Clear</button>
          )}
        </div>
      </div>
      <div className="text-xs border border-gray-300 rounded-lg px-2 py-1.5 bg-white max-h-40 overflow-y-auto space-y-0.5">
        {filtered.length === 0 && <p className="text-[11px] text-gray-400 italic">No columns available</p>}
        {filtered.map((c) => {
          const isNum = c.kind === "numeric";
          const isChecked = selected.includes(c.name);
          return (
            <label key={c.name} className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 px-1 py-0.5 rounded">
              <input type="checkbox" checked={isChecked} onChange={() => toggle(c.name)} className="accent-indigo-500 flex-shrink-0" />
              <span className={`flex-1 truncate ${isChecked ? "text-gray-900 font-medium" : "text-gray-700"}`}>{c.name}</span>
              <span className={`text-[9px] px-1 rounded flex-shrink-0 ${isNum ? "bg-blue-50 text-blue-600" : "bg-purple-50 text-purple-600"}`}>
                {isNum ? "N" : "C"}
              </span>
            </label>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Event columns for Cox must be binary 0/1. The session ColMeta `kind`
 * enum has no "binary" value AND a 0/1 column is frequently classified as
 * `categorical` (e.g. SEX, DM, HT), so a kind-based filter both leaks
 * continuous numerics and hides the real event columns. Detect binary
 * columns purely from the data: scan every column's preview values and
 * keep those whose non-null values are all numerically coercible with
 * ≤2 distinct values (0/1, 1/2, …) — regardless of `kind`.
 *
 * Falls back to the full column list only when nothing qualifies (e.g.
 * preview unavailable) so the user is never stuck with an empty dropdown.
 */
function binaryLikeColumns(
  columns: { name: string; kind: string }[],
  preview: Record<string, unknown>[] | undefined,
): { name: string; kind: string }[] {
  if (!preview || preview.length === 0) return columns;
  const out: { name: string; kind: string }[] = [];
  for (const c of columns) {
    const distinct = new Set<number>();
    let ok = true;
    let seen = 0;
    for (const row of preview) {
      const v = row[c.name];
      if (v === null || v === undefined || v === "") continue;
      const n = Number(v);
      if (!Number.isFinite(n)) { ok = false; break; }  // non-numeric → not an event col
      seen += 1;
      distinct.add(n);
      if (distinct.size > 2) { ok = false; break; }
    }
    if (ok && seen > 0 && distinct.size >= 1 && distinct.size <= 2) out.push(c);
  }
  return out.length > 0 ? out : columns;
}

// Okabe-Ito colourblind-safe palette + distinct line dashes for KM curves.
const OKABE_ITO = ["#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"];
const KM_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"];

/** Nice round tick set 0..xmax (~7 ticks) for the number-at-risk columns. */
function niceRiskTimes(xmax: number): number[] {
  if (!(xmax > 0) || !isFinite(xmax)) return [];
  const raw = xmax / 6;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const ticks: number[] = [];
  for (let t = 0; t <= xmax + 1e-9; t += step) ticks.push(Math.round(t));
  return ticks;
}

/**
 * Compose a publication-style standard interpretation of a KM result
 * from the returned data: overall log-rank, landmark survival, pairwise
 * comparisons, and median survival. Returns "" when not enough is present.
 */
function buildKmNarrative(
  km: any,
  labels: Record<string, string>,
  groupName: string,
): string {
  if (!km || !Array.isArray(km.groups) || km.groups.length === 0) return "";
  const lab = (g: string) => labels[g] ?? g;
  const groups = km.groups as any[];
  const gv = groupName || "group";
  const pStr = (p: number | null | undefined) =>
    p == null ? "n/a" : p < 0.001 ? "p<0.001" : `p=${p.toFixed(3)}`;
  const parts: string[] = [];

  // 0. Median follow-up (reverse KM).
  const mfu = km.median_follow_up;
  if (mfu?.median != null) {
    const iqr = (mfu.q1 != null && mfu.q3 != null)
      ? ` (IQR ${mfu.q1.toFixed(0)}–${mfu.q3.toFixed(0)})` : "";
    parts.push(`Median follow-up was ${mfu.median.toFixed(0)}${iqr}.`);
  }

  // 1. Overall difference.
  if (groups.length >= 2 && km.logrank?.p != null) {
    const sig = km.logrank.p < 0.05;
    parts.push(
      `Kaplan–Meier survival ${sig ? "differed significantly" : "did not differ significantly"} ` +
      `across the ${groups.length} ${gv} groups (log-rank ${pStr(km.logrank.p)}).`,
    );
  }

  // 2. Landmark survival at the requested time point(s).
  if (Array.isArray(km.survival_times) && km.survival_times.length > 0) {
    for (const t of km.survival_times) {
      const idx = km.survival_times.indexOf(t);
      let anyUnreliable = false;
      const frags = groups
        .map((g) => {
          const pt = (g.survival_at ?? [])[idx];
          if (!pt || pt.survival == null) return null;
          if (pt.reliable === false) anyUnreliable = true;
          return `${(pt.survival * 100).toFixed(1)}% in the ${lab(String(g.group))} group`;
        })
        .filter(Boolean);
      if (frags.length) {
        parts.push(
          `Estimated survival at t=${t} was ${frags.join(", ")}.` +
          (anyUnreliable ? " (Interpret the t=" + t + " estimate with caution — few patients remained at risk.)" : ""),
        );
      }
    }
  }

  // 3. Pairwise comparisons.
  const pw = km.pairwise?.comparisons as any[] | undefined;
  if (pw && pw.length) {
    const useAdj = pw.some((c) => c.p_adj != null);
    const val = (c: any) => (useAdj ? c.p_adj : c.p);
    const sig = pw.filter((c) => val(c) != null && val(c) < 0.05);
    const ns = pw.filter((c) => val(c) != null && val(c) >= 0.05);
    const fmtPair = (c: any) => `${lab(c.group_a)} vs ${lab(c.group_b)}, ${pStr(val(c))}`;
    const bits: string[] = [];
    if (sig.length) bits.push(`significant differences were found for ${sig.map(fmtPair).join("; ")}`);
    if (ns.length) bits.push(`no significant difference between ${ns.map(fmtPair).join("; ")}`);
    if (bits.length) {
      parts.push(
        `Pairwise comparisons (${km.pairwise.correction && km.pairwise.correction !== "none" ? km.pairwise.correction + "-adjusted" : "unadjusted"}) showed ${bits.join(", whereas ")}.`,
      );
    }
  }

  // 4. Median survival.
  const meds = groups.map((g) => ({ g: lab(String(g.group)), m: g.median_survival }));
  const reached = meds.filter((x) => x.m != null);
  if (reached.length === 0 && groups.length > 0) {
    parts.push("Median survival was not reached in any group.");
  } else if (reached.length) {
    parts.push(`Median survival: ${reached.map((x) => `${x.m} (${x.g})`).join(", ")}.`);
  }

  return parts.join(" ");
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
  // 2-grid layout: left method nav, right active method panel.
  const [activeMethod, setActiveMethod] = useState<SurvMethod>("km");
  // Binary 0/1-like columns for Cox event selectors (the `kind` enum has
  // no "binary", so a numeric filter would leak continuous columns).
  const binaryCols = useMemo(
    () => binaryLikeColumns(columns, session?.preview),
    [columns, session?.preview],
  );
  // Cross-panel handoff to the Forest Builder + Visual-tab deep link.
  const setForestHandoff = useStore((s) => s.setForestHandoff);
  const setVisualSubTab = useStore((s) => s.setVisualSubTab);
  const setActiveTab = useStore((s) => s.setActiveTab);

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
  const [fgPredictors, setFgPredictors] = useState<string[]>([]);
  const [fgPredFilter, setFgPredFilter] = useState("");
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
  const [kmStratify, setKmStratify] = useState("");
  const [kmSurvTimes, setKmSurvTimes] = useState("");          // e.g. "365, 1825"
  const [kmPairwise, setKmPairwise] = useState(false);
  const [kmCorrection, setKmCorrection] = useState("holm");    // none|bonferroni|holm|bh
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
  // Publication-quality customisation
  const [kmCustomPlotTitle, setKmCustomPlotTitle] = useState("");      // editable plot title
  const [kmShowPInTitle, setKmShowPInTitle] = useState(true);          // append " (log-rank p=…)"
  const [kmShowNInLegend, setKmShowNInLegend] = useState(true);        // append " (n=…)"
  const [kmHidePrefix, setKmHidePrefix] = useState(true);              // drop "groupCol = " prefix
  const [kmAutoZoomY, setKmAutoZoomY] = useState(true);                // zoom Y to data range
  const [kmYAxisAsPct, setKmYAxisAsPct] = useState(false);             // % vs decimal Y
  const [kmYTitle, setKmYTitle] = useState("Survival probability");    // editable Y-axis label
  const [kmPlotH, setKmPlotH] = useState(420);                         // plot height px
  const [kmPlotW, setKmPlotW] = useState<number | undefined>(undefined); // plot width px (undefined = fill)
  const [kmRiskTable, setKmRiskTable] = useState(false);              // number-at-risk table
  const [kmColorblind, setKmColorblind] = useState(false);           // Okabe-Ito + line styles
  const [kmShowCensors, setKmShowCensors] = useState(false);         // censor tick marks
  const [kmGroupColors, setKmGroupColors] = useState<Record<string, string>>({}); // per-group color
  const [kmContextMenu, setKmContextMenu] = useState<{ type: "item"|"groupTitle"|"durationTitle"|"plotTitle"; group?: string; x: number; y: number } | null>(null);
  const [kmRenameValue, setKmRenameValue] = useState("");

  // Nudge Plotly to refit when the KM plot box is resized via sliders
  // (useResizeHandler only listens to window resize, not element resize).
  useEffect(() => {
    const id = window.setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
    return () => window.clearTimeout(id);
  }, [kmPlotW, kmPlotH]);

  // Lazily augment the current KM result when the risk-table or censor-mark
  // toggles are switched on after the run: derive tick times from the
  // curves' x-max and re-fetch with risk_times / include_censors so the
  // extras render without forcing the user to re-run.
  useEffect(() => {
    if (!kmResult?.groups?.length) return;
    const needRisk = kmRiskTable && !kmResult.groups.some((g: any) => Array.isArray(g.at_risk));
    const needCens = kmShowCensors && !kmResult.groups.some((g: any) => Array.isArray(g.censors));
    if (!needRisk && !needCens) return;
    let xmax = 0;
    for (const g of kmResult.groups) {
      const c = g.curve;
      if (c?.length) xmax = Math.max(xmax, c[c.length - 1].time);
    }
    const times = niceRiskTimes(xmax);
    let cancelled = false;
    (async () => {
      try {
        const res = await runKM({
          session_id: sid, duration_col: kmDuration, event_col: kmEvent,
          group_col: kmGroup || undefined, stratify_col: kmStratify || undefined,
          survival_times: kmResult.survival_times?.length ? kmResult.survival_times : undefined,
          pairwise: !!kmResult.pairwise, pairwise_correction: kmCorrection,
          risk_times: kmRiskTable && times.length ? times : undefined,
          include_censors: kmShowCensors,
        });
        if (!cancelled) setKmResult(res.data);
      } catch { /* non-fatal — extras just won't show */ }
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kmRiskTable, kmShowCensors, kmResult]);

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


  // Time-horizon sensitivity (Cox at multiple administrative-censoring windows)
  const [chDuration, setChDuration] = useState("");
  const [chEvent, setChEvent] = useState("");
  const [chPredictor, setChPredictor] = useState("");
  const [chCovariates, setChCovariates] = useState<string[]>([]);
  const [chHorizons, setChHorizons] = useState("365, 730");
  const [chLabels, setChLabels] = useState("1 year, 2 years");
  const [chResult, setChResult] = useState<any>(null);
  const [chLoading, setChLoading] = useState(false);
  const [chError, setChError] = useState<string | null>(null);

  // RMST state — Restricted Mean Survival Time (PH-free alternative)
  const [rmstDuration, setRmstDuration] = useState("");
  const [rmstEvent, setRmstEvent] = useState("");
  const [rmstGroup, setRmstGroup] = useState("");
  const [rmstTau, setRmstTau] = useState<string>("");
  const [rmstResult, setRmstResult] = useState<any>(null);
  const [rmstLoading, setRmstLoading] = useState(false);
  const [rmstError, setRmstError] = useState<string | null>(null);
  const rmstPlotRef = useRef<any>(null);

  // Recurrent-events LWYY state
  const [lwId, setLwId] = useState("");
  const [lwStart, setLwStart] = useState("");
  const [lwStop, setLwStop] = useState("");
  const [lwEvent, setLwEvent] = useState("");
  const [lwPreds, setLwPreds] = useState<string[]>([]);
  const [lwGroup, setLwGroup] = useState("");
  const [lwResult, setLwResult] = useState<any>(null);
  const [lwLoading, setLwLoading] = useState(false);
  const [lwError, setLwError] = useState<string | null>(null);
  const lwPlotRef = useRef<any>(null);

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
        predictors: fgPredictors.length > 0 ? fgPredictors : undefined,
      });
      setFgResult(res.data);
    } catch (e: any) { setFgError(e?.response?.data?.detail ?? "Fine-Gray failed"); }
    finally { setFgLoading(false); }
  };
  const fgToggleP = (c: string) =>
    setFgPredictors((p) => p.includes(c) ? p.filter((x) => x !== c) : [...p, c]);

  // ── RMST handler
  const handleRMST = async () => {
    if (!rmstDuration || !rmstEvent) { setRmstError("Select duration and event columns"); return; }
    const tau = parseFloat(rmstTau);
    if (!Number.isFinite(tau) || tau <= 0) { setRmstError("Enter a positive time horizon τ"); return; }
    setRmstResult(null); setRmstError(null); setRmstLoading(true);
    try {
      const res = await runRMST({
        session_id: sid, duration_col: rmstDuration, event_col: rmstEvent,
        tau, group_col: rmstGroup || undefined,
      });
      setRmstResult(res.data);
    } catch (e: any) { setRmstError(e?.response?.data?.detail ?? "RMST failed"); }
    finally { setRmstLoading(false); }
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
  useEffect(() => { setFgResult(null); setFgError(null); }, [fgDuration, fgEvent, fgInterest, fgGroup, fgPredictors]);
  useEffect(() => { setRmstResult(null); setRmstError(null); }, [rmstDuration, rmstEvent, rmstGroup, rmstTau]);
  useEffect(() => { setLwResult(null); setLwError(null); }, [lwId, lwStart, lwStop, lwEvent, lwPreds, lwGroup]);

  const handleLWYY = async () => {
    if (!lwId || !lwStart || !lwStop || !lwEvent || lwPreds.length === 0) {
      setLwError("Select id, start, stop, event and at least one predictor."); return;
    }
    setLwResult(null); setLwError(null); setLwLoading(true);
    try {
      const res = await runRecurrentLWYY({
        session_id: sid, id_col: lwId, start_col: lwStart, stop_col: lwStop,
        event_col: lwEvent, predictors: lwPreds, group_col: lwGroup || undefined,
      });
      setLwResult(res.data);
    } catch (e: any) {
      const detail = e?.response?.data?.detail;
      setLwError(Array.isArray(detail) ? detail.map((m: any) => m.msg ?? String(m)).join(", ")
        : (typeof detail === "string" ? detail : (e?.message ?? "LWYY failed")));
    } finally { setLwLoading(false); }
  };
  useEffect(() => { setEvResult(null); setEvError(null); }, [evEst, evLo, evHi, evType, evP0]);
  useEffect(() => { setLmResult(null); setLmError(null); }, [lmDuration, lmEvent, lmTime, lmGroup, lmPreds]);
  useEffect(() => { setKmResult(null); setKmError(null); }, [kmDuration, kmEvent, kmGroup, kmStratify]);
  useEffect(() => { setCoxResult(null); setCoxError(null); }, [coxDuration, coxEvent, coxPreds, coxInteractions]);

  return (
    <div className="flex gap-4 max-w-[1400px] mx-auto">
      <nav className="w-52 shrink-0 space-y-1">
        {SURV_METHODS.map((m) => (
          <button key={m.id} onClick={() => setActiveMethod(m.id)}
            className={`w-full text-left px-3 py-2 rounded-lg border transition-colors ${activeMethod === m.id ? "bg-indigo-600 text-white border-indigo-600 shadow-sm" : "bg-white text-gray-700 border-gray-200 hover:border-indigo-300 hover:bg-indigo-50"}`}>
            <div className="text-xs font-semibold">{m.title}</div>
            <div className={`text-[10px] mt-0.5 ${activeMethod === m.id ? "text-indigo-100" : "text-gray-400"}`}>{m.desc}</div>
          </button>
        ))}
      </nav>
      <div className="flex-1 min-w-0 space-y-3">
      {/* ── Fine-Gray ── */}
      {activeMethod === "finegray" && (
      <Section title="Fine-Gray Competing Risks" description="Cumulative incidence function with competing events (Aalen-Johansen)">
        <ThreeCol
          storageKey="SurvivalAdvanced.FineGray"
          left={
            <>
              <div className="grid grid-cols-1 gap-2">
                <VarSelect label="Duration" value={fgDuration} onChange={setFgDuration} columns={columns} kinds={["numeric"]} />
                <VarSelect label="Event (0=censor, 1,2..=events)" value={fgEvent} onChange={setFgEvent} columns={columns} />
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Event of interest</span>
                  <input type="number" value={fgInterest} onChange={(e) => setFgInterest(Number(e.target.value))} min={1}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 w-20 focus:outline-none focus:border-indigo-400" />
                </label>
                <VarSelect label="Group (optional)" value={fgGroup} onChange={setFgGroup} columns={columns} kinds={["categorical"]} />
              </div>
              {/* Predictors for subdistribution-hazard regression (Fine-Gray 1999) */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 font-medium">
                    Predictors for sHR regression
                    <span className="ml-1 text-[10px] text-gray-400">(optional)</span>
                  </span>
                  {fgPredictors.length > 0 && (
                    <button onClick={() => setFgPredictors([])}
                      className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300">
                      Clear ({fgPredictors.length})
                    </button>
                  )}
                </div>
                <input type="text"
                  placeholder="Filter columns…"
                  value={fgPredFilter}
                  onChange={(e) => setFgPredFilter(e.target.value)}
                  className="w-full text-xs border border-gray-300 rounded-lg px-3 py-1 focus:outline-none focus:border-indigo-400" />
                <div className="max-h-32 overflow-y-auto border border-gray-200 rounded-lg p-1 space-y-0.5">
                  {columns
                    .map((c: any) => c.name)
                    .filter((n: string) =>
                      n !== fgDuration && n !== fgEvent && n !== fgGroup
                      && n.toLowerCase().includes(fgPredFilter.toLowerCase()))
                    .slice(0, 100)
                    .map((n: string) => (
                      <label key={n} className="flex items-center gap-1.5 text-xs px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                        <input type="checkbox" className="accent-indigo-500"
                          checked={fgPredictors.includes(n)} onChange={() => fgToggleP(n)} />
                        <span className="text-gray-700 truncate">{n}</span>
                      </label>
                    ))}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <RunButton onClick={handleFineGray} loading={fgLoading} label="Run Fine-Gray" />
              </div>
              {fgError && <p className="text-xs text-red-500">{fgError}</p>}
            </>
          }
          middle={
            fgResult?.plot ? (
              <div className="relative" ref={fgPlotRef}>
                <Plot data={fgResult.plot.data} layout={{ ...fgResult.plot.layout, ...baseLayout, title: fgResult.plot.layout.title }} config={{ responsive: true }} style={{ width: "100%", height: 400 }} />
                <PlotExporter plotRef={fgPlotRef} title="CIF" />
              </div>
            ) : (
              <div className="flex items-center justify-center h-[400px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
                Run Fine-Gray to render CIF
              </div>
            )
          }
          right={
            <>
              {/* ── Subdistribution-hazard regression sub-card ── */}
              {fgResult?.regression_result && (
                <div className="border border-indigo-200 bg-indigo-50/30 rounded-lg p-3 space-y-2">
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <h4 className="text-sm font-semibold text-gray-800">sHR Regression (Fine-Gray)</h4>
                  </div>
                  <p className="text-[10px] text-gray-500">{fgResult.regression_result.model}</p>
                  <div className="grid grid-cols-3 gap-1">
                    {[
                      ["n",                  fgResult.regression_result.n],
                      ["Events",             fgResult.regression_result.n_events_of_interest],
                      ["Competing",          fgResult.regression_result.n_competing],
                      ["Censored",           fgResult.regression_result.n_censored],
                      ["C-index",            fgResult.regression_result.concordance?.toFixed(3)],
                    ].map(([k, v]) => (
                      <div key={String(k)} className="bg-white border border-gray-200 rounded p-1.5 text-center">
                        <p className="text-[9px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-xs">{v}</p>
                      </div>
                    ))}
                  </div>
                  <div className="overflow-auto rounded border border-gray-200 bg-white">
                    <table className="w-full text-[11px] border-collapse">
                      <thead>
                        <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                          {["Variable", "sHR", "95% CI", "p"].map((h) => (
                            <th key={h} className="px-1.5 py-1 text-left font-medium">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {fgResult.regression_result.coefficients.map((c: any) => (
                          <tr key={c.variable} className="border-b border-gray-100">
                            <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[80px]">{c.variable}</td>
                            <td className={`px-1.5 py-1 font-mono font-semibold ${c.p != null && c.p < 0.05 ? "text-indigo-700" : "text-gray-600"}`}>{c.shr?.toFixed(2)}</td>
                            <td className="px-1.5 py-1 font-mono text-gray-500">
                              {c.shr_low != null && c.shr_high != null
                                ? `[${c.shr_low.toFixed(2)}, ${c.shr_high.toFixed(2)}]`
                                : "—"}
                            </td>
                            <td className="px-1.5 py-1">
                              <span className={`inline-block font-mono px-1 py-0.5 rounded text-[10px] ${
                                c.p != null && c.p < 0.05 ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"
                              }`}>
                                {c.p == null ? "—" : c.p < 0.001 ? "<0.001" : c.p.toFixed(3)}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[9px] text-gray-500 leading-relaxed">{fgResult.regression_result.method_note}</p>
                </div>
              )}
              <ResultBlock result={fgResult} />
            </>
          }
        />
      </Section>
      )}

      {/* ── RMST ── */}
      {activeMethod === "rmst" && (
      <Section title="Restricted Mean Survival Time (RMST)"
        description="Average event-free time over a fixed horizon τ — PH-free alternative to the hazard ratio. Robust when curves cross or the proportional-hazards assumption fails.">
        <ThreeCol
          storageKey="SurvivalAdvanced.RMST"
          left={
            <>
              <div className="grid grid-cols-1 gap-2">
                <VarSelect label="Duration" value={rmstDuration} onChange={setRmstDuration} columns={columns} kinds={["numeric"]} />
                <VarSelect label="Event (0/1)" value={rmstEvent} onChange={setRmstEvent} columns={binaryCols} />
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">τ (time horizon)</span>
                  <input type="number" min="0" step="any" value={rmstTau}
                    onChange={(e) => setRmstTau(e.target.value)}
                    placeholder="e.g. 1825"
                    className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" />
                </label>
                <VarSelect label="Group (optional)" value={rmstGroup} onChange={setRmstGroup} columns={columns} kinds={["categorical"]} />
              </div>
              <div className="flex items-center gap-3">
                <RunButton onClick={handleRMST} loading={rmstLoading} label="Run RMST" />
              </div>
              {rmstError && <p className="text-xs text-red-500">{rmstError}</p>}
            </>
          }
          middle={
            rmstResult?.plot ? (
              <div className="relative" ref={rmstPlotRef}>
                <Plot data={rmstResult.plot.data}
                  layout={{ ...rmstResult.plot.layout, ...baseLayout, title: rmstResult.plot.layout.title }}
                  config={{ responsive: true }} style={{ width: "100%", height: 400 }} />
                <PlotExporter plotRef={rmstPlotRef} title="RMST" />
              </div>
            ) : (
              <div className="flex items-center justify-center h-[400px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
                Run RMST to render KM curves
              </div>
            )
          }
          right={
            <>
              {rmstResult?.rmst_by_group && (
                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                        {["Group", "n", "RMST", "95% CI"].map((h) => (
                          <th key={h} className="px-1.5 py-1.5 text-left font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(rmstResult.rmst_by_group).map(([g, v]: any) => (
                        <tr key={g} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[60px]">{g}</td>
                          <td className="px-1.5 py-1 font-mono text-gray-600">{v.n}</td>
                          <td className="px-1.5 py-1 font-mono font-semibold text-indigo-700">{v.rmst}</td>
                          <td className="px-1.5 py-1 font-mono text-gray-500">[{v.ci_low}, {v.ci_high}]</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {rmstResult?.contrasts && rmstResult.contrasts.length > 0 && (
                <div className="overflow-auto rounded-lg border border-indigo-200 bg-indigo-50/30">
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="bg-white border-b border-indigo-200 text-gray-600">
                        {["A", "B", "ΔRMST", "p"].map((h) => (
                          <th key={h} className="px-1.5 py-1.5 text-left font-semibold">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rmstResult.contrasts.map((c: any, i: number) => (
                        <tr key={i} className={`border-b border-indigo-100 ${c.p != null && c.p < 0.05 ? "bg-indigo-50/60" : ""}`}>
                          <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[50px]">{c.group_a}</td>
                          <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[50px]">{c.group_b}</td>
                          <td className={`px-1.5 py-1 font-mono font-semibold ${c.p != null && c.p < 0.05 ? "text-indigo-700" : "text-gray-700"}`}>{c.delta_rmst}</td>
                          <td className="px-1.5 py-1">
                            <span className={`inline-block font-mono px-1 py-0.5 rounded text-[10px] ${
                              c.p != null && c.p < 0.05 ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"
                            }`}>
                              {c.p == null ? "—" : c.p < 0.001 ? "<0.001" : c.p.toFixed(3)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <ResultBlock result={rmstResult} />
            </>
          }
        />
      </Section>
      )}

      {/* ── Recurrent events — LWYY ── */}
      {activeMethod === "lwyy" && (
      <Section title="Recurrent Events (LWYY)"
        description="Modified Andersen-Gill model with Lin-Wei-Yang-Ying cluster-robust SE for recurrent events (e.g. repeat hospitalisations). Counting-process (start, stop, event] intervals; exp(β) = rate ratio.">
        <ThreeCol
          storageKey="SurvivalAdvanced.LWYY"
          left={
            <>
              <div className="grid grid-cols-1 gap-2">
                <VarSelect label="Subject id" value={lwId} onChange={setLwId} columns={columns} />
                <VarSelect label="Start (interval entry)" value={lwStart} onChange={setLwStart} columns={columns} kinds={["numeric"]} />
                <VarSelect label="Stop (interval / event time)" value={lwStop} onChange={setLwStop} columns={columns} kinds={["numeric"]} />
                <VarSelect label="Event (1 = event at stop)" value={lwEvent} onChange={setLwEvent} columns={columns} />
                <VarSelect label="Group for MCF plot (optional)" value={lwGroup} onChange={setLwGroup} columns={columns} kinds={["categorical"]} />
              </div>
              <MultiSelect label="Predictors" columns={columns} selected={lwPreds} onChange={setLwPreds}
                excludeNames={[lwId, lwStart, lwStop, lwEvent].filter(Boolean)} />
              <div className="flex items-center gap-3">
                <RunButton onClick={handleLWYY} loading={lwLoading} label="Run LWYY" />
              </div>
              {lwError && <p className="text-xs text-red-500">{lwError}</p>}
            </>
          }
          middle={
            lwResult?.plot ? (
              <div className="relative" ref={lwPlotRef}>
                <Plot data={lwResult.plot.data} layout={{ ...lwResult.plot.layout, ...baseLayout, title: lwResult.plot.layout.title }} config={{ responsive: true }} style={{ width: "100%", height: 400 }} />
                <PlotExporter plotRef={lwPlotRef} title="MCF_LWYY" />
              </div>
            ) : (
              <div className="flex items-center justify-center h-[400px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
                Run LWYY to render the mean cumulative function
              </div>
            )
          }
          right={
            <>
              {lwResult && (
                <div className="border border-indigo-200 bg-indigo-50/30 rounded-lg p-3 space-y-2">
                  <h4 className="text-sm font-semibold text-gray-800">Rate-ratio table</h4>
                  <p className="text-[10px] text-gray-500">{lwResult.model}</p>
                  <div className="grid grid-cols-3 gap-1">
                    {[
                      ["Subjects", lwResult.n_subjects],
                      ["Events", lwResult.n_events],
                      ["Ev/subj", lwResult.events_per_subject?.toFixed(2)],
                    ].map(([k, v]) => (
                      <div key={String(k)} className="bg-white border border-gray-200 rounded p-1.5 text-center">
                        <p className="text-[9px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-xs">{v}</p>
                      </div>
                    ))}
                  </div>
                  <div className="overflow-auto rounded border border-gray-200 bg-white">
                    <table className="w-full text-[11px] border-collapse">
                      <thead>
                        <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                          {["Variable", "RR", "95% CI", "p"].map((h) => (
                            <th key={h} className="px-1.5 py-1 text-left font-medium">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {lwResult.coefficients.map((c: any) => (
                          <tr key={c.variable} className="border-b border-gray-100">
                            <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[80px]">{c.variable}</td>
                            <td className={`px-1.5 py-1 font-mono font-semibold ${c.p < 0.05 ? "text-indigo-700" : "text-gray-600"}`}>{c.rate_ratio?.toFixed(2)}</td>
                            <td className="px-1.5 py-1 font-mono text-gray-500">[{c.rr_low?.toFixed(2)}, {c.rr_high?.toFixed(2)}]</td>
                            <td className="px-1.5 py-1">
                              <span className={`inline-block font-mono px-1 py-0.5 rounded text-[10px] ${c.p < 0.05 ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"}`}>
                                {c.p == null ? "—" : c.p < 0.001 ? "<0.001" : c.p.toFixed(3)}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              <ResultBlock result={lwResult} />
            </>
          }
        />
      </Section>
      )}

      {/* ── E-value ── */}
      {activeMethod === "evalue" && (
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
      )}

      {/* ── Landmark ── */}
      {activeMethod === "landmark" && (
      <Section title="Landmark Survival Analysis" description="Survival analysis conditional on surviving beyond a landmark time point">
        <ThreeCol
          storageKey="SurvivalAdvanced.Landmark"
          left={
            <>
              <div className="grid grid-cols-1 gap-2">
                <VarSelect label="Duration" value={lmDuration} onChange={setLmDuration} columns={columns} kinds={["numeric"]} />
                <VarSelect label="Event (0/1)" value={lmEvent} onChange={setLmEvent} columns={binaryCols} />
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Landmark time</span>
                  <input type="number" step="1" value={lmTime} onChange={(e) => setLmTime(e.target.value)}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" placeholder="e.g. 30" />
                </label>
                <VarSelect label="Group (optional)" value={lmGroup} onChange={setLmGroup} columns={columns} kinds={["categorical"]} />
              </div>
              <MultiSelect label="Predictors for Cox (optional)" columns={columns} selected={lmPreds} onChange={setLmPreds} excludeNames={[lmDuration, lmEvent].filter(Boolean)} />
              <div className="flex items-center gap-3">
                <RunButton onClick={handleLandmark} loading={lmLoading} label="Run Landmark" />
              </div>
              {lmError && <p className="text-xs text-red-500">{lmError}</p>}
            </>
          }
          middle={
            lmResult?.plot ? (
              <div className="relative" ref={lmPlotRef}>
                <Plot data={lmResult.plot.data} layout={{ ...lmResult.plot.layout, ...baseLayout, title: lmResult.plot.layout.title }} config={{ responsive: true }} style={{ width: "100%", height: 400 }} />
                <PlotExporter plotRef={lmPlotRef} title="Landmark_KM" />
              </div>
            ) : (
              <div className="flex items-center justify-center h-[400px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
                Run Landmark to render KM
              </div>
            )
          }
          right={
            <>
              {lmResult?.cox_results && lmResult.cox_results.length > 0 && !lmResult.cox_results[0].error && (
                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="text-[11px] w-full">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="px-1.5 py-1.5 text-left text-gray-500">Variable</th>
                        <th className="px-1.5 py-1.5 text-left text-gray-500">HR</th>
                        <th className="px-1.5 py-1.5 text-left text-gray-500">95% CI</th>
                        <th className="px-1.5 py-1.5 text-left text-gray-500">p</th>
                      </tr>
                    </thead>
                    <tbody>
                      {lmResult.cox_results.map((r: any, i: number) => (
                        <tr key={i} className="border-t border-gray-100">
                          <td className="px-1.5 py-1 text-gray-700 font-medium truncate max-w-[80px]">{r.variable}</td>
                          <td className="px-1.5 py-1 text-gray-700 font-mono">{r.HR}</td>
                          <td className="px-1.5 py-1 text-gray-500 font-mono">{r.ci_low}–{r.ci_high}</td>
                          <td className={`px-1.5 py-1 font-mono ${r.p < 0.05 ? "text-indigo-600 font-semibold" : "text-gray-500"}`}>
                            {r.p < 0.001 ? "<0.001" : r.p?.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <ResultBlock result={lmResult} />
            </>
          }
        />
      </Section>
      )}

      {/* ── Kaplan-Meier ── */}
      {activeMethod === "km" && (
      <Section title="Kaplan-Meier Survival" description="Visualise time-to-event data with survival curves and log-rank test">
        <div className="grid grid-cols-4 gap-3">
          <VarSelect label="Duration (time)" value={kmDuration} onChange={setKmDuration} columns={columns} kinds={["numeric"]} />
          <VarSelect label="Event (0/1)" value={kmEvent} onChange={setKmEvent} columns={binaryCols} />
          <VarSelect label="Group (optional)" value={kmGroup} onChange={setKmGroup} columns={columns} kinds={["categorical"]} />
          <VarSelect label="Stratify by (optional)" value={kmStratify} onChange={setKmStratify} columns={columns} kinds={["categorical"]} />
        </div>

        {/* Landmark survival + pairwise log-rank options */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">
              Survival at time(s) — comma-sep
              <Tip wide text="Landmark survival probabilities: the KM survival estimate (+ 95% CI) read off each curve at the time points you list, in the Duration column's unit. Days → '1825' gives 5-year survival; months → '60'; years → '5'. Multiple values allowed (e.g. '365, 1825'). Reported as e.g. '77.0% (95% CI 70–84)'." />
            </span>
            <input value={kmSurvTimes} onChange={(e) => setKmSurvTimes(e.target.value)}
              placeholder="365, 1825"
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400" />
            <span className="text-[10px] text-gray-400">In the Duration unit. Days → 1825 = 5-year survival.</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-600 pb-2">
            <input type="checkbox" checked={kmPairwise} onChange={(e) => setKmPairwise(e.target.checked)} className="accent-indigo-500" />
            Pairwise log-rank (≥3 groups)
            <Tip wide text="With 3+ groups the overall log-rank only says 'some difference exists'. Pairwise runs a log-rank for every group pair so you can state which pair drives it (e.g. '<100 vs 100–130, p=0.003; 100–130 vs >130, p=0.46'). Always apply a multiplicity correction → for ≥3 comparisons." />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">
              Multiplicity correction
              <Tip wide text="Adjusts pairwise p-values for testing several pairs at once (controls false positives). Holm = uniformly more powerful than Bonferroni, recommended default for confirmatory pairwise comparisons. Bonferroni = most conservative. Benjamini-Hochberg = controls false-discovery rate, for exploratory work. None = raw p (report only if pre-specified)." />
            </span>
            <select value={kmCorrection} onChange={(e) => setKmCorrection(e.target.value)} disabled={!kmPairwise}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400 disabled:opacity-50">
              <option value="none">None (raw p)</option>
              <option value="holm">Holm</option>
              <option value="bonferroni">Bonferroni</option>
              <option value="bh">Benjamini-Hochberg (FDR)</option>
            </select>
          </label>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <RunButton onClick={async () => {
            if (!kmDuration || !kmEvent) { setKmError("Select duration and event columns"); return; }
            setKmResult(null); setKmError(null); setKmLoading(true);
            try {
              const survTimes = kmSurvTimes.split(",").map((s) => parseFloat(s.trim())).filter((x) => !Number.isNaN(x) && x > 0);
              const res = await runKM({
                session_id: sid, duration_col: kmDuration, event_col: kmEvent,
                group_col: kmGroup || undefined, stratify_col: kmStratify || undefined,
                survival_times: survTimes.length ? survTimes : undefined,
                pairwise: kmPairwise && !!kmGroup,
                pairwise_correction: kmCorrection,
                include_censors: kmShowCensors,
              });
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
              {kmScanLoading ? "Scanning…" : "🔍 Log-rank Scan"}
            </button>
          )}
          {kmError && <p className="text-xs text-red-500">{kmError}</p>}
        </div>

        {/* KM scan results */}
        {kmScanResult.length > 0 && (
          <div className="rounded-lg border border-gray-200 overflow-auto">
            <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 flex items-center justify-between">
              <p className="text-xs font-semibold text-gray-600">Log-rank Scan — All Categorical Variables</p>
              <button onClick={() => setKmScanResult([])} className="text-[10px] text-gray-400 hover:text-red-500">✕ Close</button>
            </div>
            <table className="text-xs w-full">
              <thead><tr className="bg-gray-50">
                <th className="px-3 py-1.5 text-left text-gray-500">Variable</th>
                <th className="px-3 py-1.5 text-left text-gray-500">Groups</th>
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
                      {r.logrank_p !== null ? (r.logrank_p < 0.001 ? "<0.001" : r.logrank_p.toFixed(4)) : "error"}
                    </td>
                    <td className="px-3 py-1">
                      {r.logrank_p !== null && r.logrank_p < 0.05 && (
                        <button onClick={() => { setKmGroup(r.variable); setKmScanResult([]); }}
                          className="text-[10px] text-indigo-500 hover:text-indigo-700 underline">
                          Add to chart
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

          // Build legend label per group
          const legendLabel = (g: any) => {
            const resolved = resolveGroupName(String(g.group));
            const nSuffix = kmShowNInLegend ? ` (n=${g.n})` : "";
            if (!kmGroup) return `${resolved}${nSuffix}`;
            if (kmHidePrefix) return `${resolved}${nSuffix}`;
            return `${kmCustomGroupTitle || kmGroup} = ${resolved}${nSuffix}`;
          };

          // Compute Y-axis range
          let yRange: [number, number] = [0, 1.05];
          if (kmAutoZoomY) {
            let minSurv = 1;
            for (const g of kmResult.groups) {
              for (const p of g.curve) {
                if (typeof p.survival === "number" && p.survival < minSurv) minSurv = p.survival;
              }
            }
            const floor = Math.max(0, Math.floor((minSurv - 0.02) * 20) / 20);
            yRange = [floor, 1.0];
          }

          // Build plot title
          const lrP = kmResult.logrank?.p;
          const pStr = lrP == null ? null : (lrP < 0.001 ? "<0.001" : lrP.toFixed(3));
          const baseTitle = kmCustomPlotTitle || (kmGroup ? `Kaplan–Meier survival by ${kmCustomGroupTitle || kmGroup}` : "Kaplan–Meier survival");
          const titleText = (kmShowPInTitle && pStr) ? `${baseTitle} (log-rank p=${pStr})` : baseTitle;

          return (
          <>
            {/* Publication-style customisation strip */}
            <div className="flex flex-wrap items-center gap-3 text-[10px] text-gray-500 mb-2 px-1">
              <button
                onClick={(e) => {
                  setKmContextMenu({ type: "plotTitle", x: e.clientX, y: e.clientY });
                  setKmRenameValue(kmCustomPlotTitle || (kmGroup ? `Kaplan–Meier survival by ${kmCustomGroupTitle || kmGroup}` : "Kaplan–Meier survival"));
                }}
                className="text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50"
              >✏️ Plot title</button>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmShowPInTitle} onChange={(e) => setKmShowPInTitle(e.target.checked)} className="accent-indigo-500" />
                log-rank p in title
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmShowNInLegend} onChange={(e) => setKmShowNInLegend(e.target.checked)} className="accent-indigo-500" />
                (n=…) in legend
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmHidePrefix} onChange={(e) => setKmHidePrefix(e.target.checked)} className="accent-indigo-500" />
                hide "{kmCustomGroupTitle || kmGroup || "group"} =" prefix
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmAutoZoomY} onChange={(e) => setKmAutoZoomY(e.target.checked)} className="accent-indigo-500" />
                zoom Y to data
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmYAxisAsPct} onChange={(e) => setKmYAxisAsPct(e.target.checked)} className="accent-indigo-500" />
                Y as %
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmRiskTable} onChange={(e) => setKmRiskTable(e.target.checked)} className="accent-indigo-500" />
                Number at risk
                <Tip wide text="Journal-standard 'number at risk' row under the curve: subjects still in follow-up (event-free and uncensored) in each group at evenly-spaced time points. Required by most journals (CONSORT/STROBE) so readers can judge how much data supports the tail of the curve." />
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmColorblind} onChange={(e) => setKmColorblind(e.target.checked)} className="accent-indigo-500" />
                Colour-blind + line styles
                <Tip wide text="Switches to the Okabe-Ito colour-blind-safe palette and gives each group a distinct line style (solid / dashed / dotted), so curves stay distinguishable in greyscale print and for colour-blind readers. Recommended for publication figures." />
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmShowCensors} onChange={(e) => setKmShowCensors(e.target.checked)} className="accent-indigo-500" />
                Censor marks
                <Tip wide text="Overlays a small '+' on each curve wherever a subject was censored (lost to follow-up / still event-free at last contact). Shows where information thins out; common in publication KM plots." />
              </label>
              {(kmCustomPlotTitle || Object.keys(kmGroupColors).length > 0 || Object.keys(kmGroupLabels).length > 0) && (
                <button
                  onClick={() => { setKmCustomPlotTitle(""); setKmGroupColors({}); setKmGroupLabels({}); }}
                  className="text-[10px] px-2 py-0.5 rounded border border-orange-300 text-orange-600 hover:bg-orange-50"
                >✕ Reset customisation</button>
              )}
            </div>

            {/* Axis labels + size — like the Forest builder */}
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 mb-2 px-1">
              <span className="text-[10px] text-gray-400 inline-flex items-center">Axis &amp; size<Tip wide text="Rename the X/Y axis titles for publication (e.g. 'Time since primary PCI (days)', 'Overall survival'). Drag Width/Height to size the figure; Width 'auto' fills the column. Export (↓ top-right of the plot) keeps these labels — SVG/PDF are vector, journal-ready." /></span>
              <input value={kmCustomDurationTitle} onChange={(e) => setKmCustomDurationTitle(e.target.value)}
                placeholder={`X-axis (Time (${kmDuration || "time"}))`}
                className="text-[11px] border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" style={{ width: 200 }} />
              <input value={kmYTitle} onChange={(e) => setKmYTitle(e.target.value)}
                placeholder="Y-axis label"
                className="text-[11px] border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" style={{ width: 160 }} />
              <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
                <span className="font-medium">Width</span>
                <input type="range" min={420} max={1300} step={20} value={kmPlotW ?? 800}
                  onChange={(e) => setKmPlotW(Number(e.target.value))} className="accent-indigo-500" style={{ width: 100 }} />
                <span className="tabular-nums w-8">{kmPlotW ?? "auto"}</span>
                {kmPlotW != null && (
                  <button onClick={() => setKmPlotW(undefined)} className="text-indigo-500 hover:text-indigo-700">auto</button>
                )}
              </label>
              <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
                <span className="font-medium">Height</span>
                <input type="range" min={260} max={760} step={20} value={kmPlotH}
                  onChange={(e) => setKmPlotH(Number(e.target.value))} className="accent-indigo-500" style={{ width: 100 }} />
                <span className="tabular-nums w-8">{kmPlotH}</span>
              </label>
            </div>

            <div className="relative" ref={kmPlotRef} style={{ width: kmPlotW != null ? kmPlotW : "100%", height: kmPlotH, maxWidth: "100%" }}>
              <Plot
                data={[
                  ...kmResult.groups.map((g: any, i: number) => ({
                    x: g.curve.map((p: any) => p.time),
                    y: g.curve.map((p: any) => p.survival),
                    type: "scatter", mode: "lines",
                    name: legendLabel(g),
                    line: {
                      width: traceDefaults.lineWidth,
                      color: kmGroupColors[String(g.group)] ?? (kmColorblind ? OKABE_ITO[i % OKABE_ITO.length] : pal[i % pal.length]),
                      shape: "hv",
                      ...(kmColorblind ? { dash: KM_DASHES[i % KM_DASHES.length] } : {}),
                    },
                  })),
                  // Censor tick marks ('+') overlaid on each curve.
                  ...(kmShowCensors
                    ? kmResult.groups.flatMap((g: any, i: number) => {
                        if (!Array.isArray(g.censors) || g.censors.length === 0) return [];
                        const c = kmGroupColors[String(g.group)] ?? (kmColorblind ? OKABE_ITO[i % OKABE_ITO.length] : pal[i % pal.length]);
                        return [{
                          x: g.censors.map((p: any) => p.time),
                          y: g.censors.map((p: any) => p.survival),
                          type: "scatter", mode: "markers",
                          name: `${legendLabel(g)} (censored)`,
                          marker: { symbol: "cross-thin-open", size: 7, color: c, line: { width: 1.4, color: c } },
                          hoverinfo: "x", showlegend: false,
                        }];
                      })
                    : []),
                ]}
                layout={{
                  ...baseLayout,
                  title: { text: titleText, font: { color: "#374151", size: 14 } },
                  xaxis: {
                    ...(baseLayout.xaxis as any),
                     // Using custom duration title if available
                    title: { text: kmCustomDurationTitle ? kmCustomDurationTitle : `Time (${kmDuration})` },
                  },
                  yaxis: {
                    ...(baseLayout.yaxis as any),
                    title: { text: kmYTitle || "Survival probability" },
                    range: yRange,
                    tickformat: kmYAxisAsPct ? ".0%" : ".2f",
                  },
                  autosize: true,
                  margin: { t: 50, r: 20, b: 56, l: 68 }, showlegend: true,
                  legend: { title: { text: kmCustomGroupTitle || kmGroup || "Group" } },
                }}
                useResizeHandler
                config={{ responsive: true }} style={{ width: "100%", height: "100%" }}
              />
              <PlotExporter plotRef={kmPlotRef} title="KM_Survival" />
            </div>

            {/* Number at risk — journal-style row under the curve */}
            {kmRiskTable && Array.isArray(kmResult.risk_times) && kmResult.risk_times.length > 0 &&
             kmResult.groups.some((g: any) => Array.isArray(g.at_risk)) && (
              <div className="mt-1 overflow-x-auto" style={{ maxWidth: kmPlotW != null ? kmPlotW : "100%" }}>
                <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-0.5 px-1">Number at risk</div>
                <table className="text-[11px] w-full">
                  <tbody>
                    {kmResult.groups.map((g: any, i: number) => (
                      <tr key={g.group}>
                        <td className="pr-3 py-0.5 whitespace-nowrap font-medium" style={{ color: kmGroupColors[String(g.group)] ?? (kmColorblind ? OKABE_ITO[i % OKABE_ITO.length] : pal[i % pal.length]) }}>
                          {kmGroupLabels[String(g.group)] ?? g.group}
                        </td>
                        {(g.at_risk ?? []).map((n: number, j: number) => (
                          <td key={j} className="py-0.5 text-center tabular-nums text-gray-700">{n}</td>
                        ))}
                      </tr>
                    ))}
                    <tr className="border-t border-gray-100">
                      <td className="pr-3 pt-0.5 text-[9px] text-gray-400">time</td>
                      {kmResult.risk_times.map((t: number, j: number) => (
                        <td key={j} className="pt-0.5 text-center text-[9px] text-gray-400 tabular-nums">{t}</td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            )}

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
                  <span>Log-rank test (overall)</span>
                  <span>
                    p = {kmResult.logrank.p < 0.001 ? "<0.001" : kmResult.logrank.p?.toFixed(4)}
                    {kmResult.logrank.p < 0.05 ? " (Significant difference)" : " (No difference)"}
                  </span>
                </div>
              )}

              {/* Median follow-up (reverse Kaplan–Meier) */}
              {kmResult.median_follow_up?.median != null && (
                <div className="px-3 py-1.5 text-[11px] border-t border-gray-100 bg-gray-50/60 text-gray-600 flex items-center justify-between">
                  <span>Median follow-up (reverse KM)</span>
                  <span className="tabular-nums">
                    {kmResult.median_follow_up.median.toFixed(0)}
                    {kmResult.median_follow_up.q1 != null && kmResult.median_follow_up.q3 != null &&
                      ` [${kmResult.median_follow_up.q1.toFixed(0)}–${kmResult.median_follow_up.q3.toFixed(0)}]`}
                    {" "}{kmCustomDurationTitle || kmDuration}
                  </span>
                </div>
              )}

              {/* Landmark survival-at-time table */}
              {Array.isArray(kmResult.survival_times) && kmResult.survival_times.length > 0 && (
                <div className="border-t border-gray-100">
                  <div className="px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wide bg-gray-50">
                    Survival probability at time point(s)
                  </div>
                  <table className="w-full">
                    <thead>
                      <tr className="text-[10px] text-gray-400">
                        <th className="px-3 py-1 text-left font-medium">Group</th>
                        {kmResult.survival_times.map((t: number) => (
                          <th key={t} className="px-3 py-1 text-right font-medium">t = {t}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {kmResult.groups?.map((g: any) => (
                        <tr key={g.group} className="border-t border-gray-50">
                          <td className="px-3 py-1 text-[11px] text-gray-700">{kmGroupLabels[g.group] ?? g.group}</td>
                          {(g.survival_at ?? []).map((pt: any, i: number) => {
                            const unreliable = pt.reliable === false;
                            return (
                              <td key={i} className={`px-3 py-1 text-[11px] text-right tabular-nums ${unreliable ? "text-gray-300" : "text-gray-600"}`}
                                title={pt.n_at_risk != null ? `${pt.n_at_risk} at risk${unreliable ? " — too few; estimate unstable" : ""}` : undefined}>
                                {pt.survival != null ? `${(pt.survival * 100).toFixed(1)}%` : "—"}
                                {pt.ci_low != null && pt.ci_high != null && (
                                  <span className={unreliable ? "text-gray-300" : "text-gray-400"}> ({(pt.ci_low * 100).toFixed(1)}–{(pt.ci_high * 100).toFixed(1)})</span>
                                )}
                                {unreliable && <span className="text-amber-500" title="Unstable: fewer than 10 at risk or beyond max follow-up">*</span>}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {kmResult.groups?.some((g: any) => (g.survival_at ?? []).some((p: any) => p.reliable === false)) && (
                    <div className="px-3 py-1 text-[9px] text-amber-600 italic">
                      * Unstable estimate — fewer than 10 patients at risk or beyond maximum follow-up. Interpret with caution or omit.
                    </div>
                  )}
                </div>
              )}

              {/* Pairwise log-rank comparisons */}
              {kmResult.pairwise?.comparisons?.length > 0 && (
                <div className="border-t border-gray-100">
                  <div className="px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wide bg-gray-50 flex items-center justify-between">
                    <span>Pairwise log-rank</span>
                    {kmResult.pairwise.correction && kmResult.pairwise.correction !== "none" && (
                      <span className="normal-case font-normal text-gray-400">{kmResult.pairwise.correction} adjusted</span>
                    )}
                  </div>
                  <table className="w-full">
                    <thead>
                      <tr className="text-[10px] text-gray-400">
                        <th className="px-3 py-1 text-left font-medium">Comparison</th>
                        <th className="px-3 py-1 text-right font-medium">p (raw)</th>
                        {kmResult.pairwise.comparisons.some((c: any) => c.p_adj != null) && (
                          <th className="px-3 py-1 text-right font-medium">p (adj)</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {kmResult.pairwise.comparisons.map((c: any, i: number) => {
                        const pShow = (p: number | null) => p == null ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3);
                        const sig = (c.p_adj ?? c.p) != null && (c.p_adj ?? c.p) < 0.05;
                        const la = kmGroupLabels[c.group_a] ?? c.group_a;
                        const lb = kmGroupLabels[c.group_b] ?? c.group_b;
                        return (
                          <tr key={i} className={`border-t border-gray-50 ${sig ? "bg-indigo-50/40" : ""}`}>
                            <td className="px-3 py-1 text-[11px] text-gray-700">{la} vs {lb}</td>
                            <td className="px-3 py-1 text-[11px] text-gray-600 text-right tabular-nums">{pShow(c.p)}</td>
                            {kmResult.pairwise.comparisons.some((x: any) => x.p_adj != null) && (
                              <td className={`px-3 py-1 text-[11px] text-right tabular-nums ${sig ? "font-semibold text-indigo-700" : "text-gray-600"}`}>{pShow(c.p_adj)}</td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Auto-generated standard interpretation */}
              {(() => {
                const narrative = buildKmNarrative(kmResult, kmGroupLabels, kmCustomGroupTitle || kmGroup);
                if (!narrative) return null;
                return (
                  <div className="border-t border-gray-100 bg-amber-50/60 px-4 py-3">
                    <div className="text-[10px] font-semibold text-amber-700 uppercase tracking-wide mb-1">Interpretation (auto-generated)</div>
                    <p className="text-[12px] text-amber-900 leading-relaxed">{narrative}</p>
                    <p className="text-[9px] text-amber-600/80 mt-1.5 italic">Draft wording — verify against your data and reporting guidelines before publication.</p>
                  </div>
                );
              })()}

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
                    {kmContextMenu.type === "plotTitle" && `Edit plot title`}
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
                        } else if (kmContextMenu.type === "plotTitle") {
                          setKmCustomPlotTitle(kmRenameValue);
                        }
                        setKmContextMenu(null);
                      }
                      if (e.key === "Escape") setKmContextMenu(null);
                    }}
                  />
                  {kmContextMenu.type === "item" && kmContextMenu.group && (
                    <div className="mb-2">
                      <p className="text-[10px] text-gray-400 mb-1 font-medium uppercase tracking-wide">Color</p>
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {["#dc2626", "#16a34a", "#2563eb", "#ea580c", "#9333ea", "#0891b2", "#ca8a04", "#475569"].map((c) => (
                          <button key={c} title={c}
                            onClick={() => setKmGroupColors((prev) => ({ ...prev, [kmContextMenu.group!]: c }))}
                            className="w-5 h-5 rounded-full border border-gray-300 hover:scale-110 transition-transform"
                            style={{ background: c }}
                          />
                        ))}
                        <input type="color"
                          value={kmGroupColors[kmContextMenu.group] ?? "#6366f1"}
                          onChange={(e) => setKmGroupColors((prev) => ({ ...prev, [kmContextMenu.group!]: e.target.value }))}
                          className="w-6 h-6 cursor-pointer border border-gray-300 rounded"
                        />
                      </div>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <button
                      onClick={() => {
                        if (kmContextMenu.type === "item" && kmContextMenu.group) {
                          setKmGroupLabels((prev) => ({ ...prev, [kmContextMenu.group!]: kmRenameValue }));
                        } else if (kmContextMenu.type === "groupTitle") {
                          setKmCustomGroupTitle(kmRenameValue);
                        } else if (kmContextMenu.type === "durationTitle") {
                          setKmCustomDurationTitle(kmRenameValue);
                        } else if (kmContextMenu.type === "plotTitle") {
                          setKmCustomPlotTitle(kmRenameValue);
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
                          const nc = { ...kmGroupColors };
                          delete nc[kmContextMenu.group];
                          setKmGroupColors(nc);
                        } else if (kmContextMenu.type === "groupTitle") {
                          setKmCustomGroupTitle("");
                        } else if (kmContextMenu.type === "durationTitle") {
                          setKmCustomDurationTitle("");
                        } else if (kmContextMenu.type === "plotTitle") {
                          setKmCustomPlotTitle("");
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

        {/* Stratified KM (small-multiples grid) — when stratify_col is set */}
        {kmResult?.strata && (() => {
          const strata: any[] = kmResult.strata;
          const miniH = 420;

          const stratColMeta = columns.find((c) => c.name === kmStratify);
          const stratLabels = stratColMeta?.value_labels ?? {};
          const groupColMeta = columns.find((c) => c.name === kmGroup);
          const grpLabels = groupColMeta?.value_labels ?? {};

          const buildTraces = (groups: any[]) =>
            groups.map((g: any, i: number) => ({
              x: g.curve.map((p: any) => p.time),
              y: g.curve.map((p: any) => p.survival),
              type: "scatter" as const, mode: "lines" as const,
              name: kmGroup
                ? `${kmCustomGroupTitle || kmGroup} = ${grpLabels[String(g.group)] ?? g.group}`
                : String(g.group),
              line: { width: traceDefaults.lineWidth, color: pal[i % pal.length], shape: "hv" as const },
            }));

          return (
            <div className="mt-3 space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-700">
                  Stratified by <span className="text-indigo-600">{kmStratify}</span>
                  {kmGroup && <span className="text-gray-400 font-normal ml-2">— curves by {kmGroup}</span>}
                </h4>
                <span className="text-xs text-gray-400">{strata.length} strata · {kmResult.n_total ?? "?"} total</span>
              </div>
              <div className="grid gap-4 grid-cols-1">
                {strata.map((stratum: any) => {
                  const stratLabel = stratLabels[String(stratum.label)] ?? stratum.label;
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
                        <span className="text-xs font-semibold text-gray-700">{kmStratify} = {stratLabel}</span>
                        <span className="text-[10px] text-gray-400">n={stratum.n}</span>
                      </div>
                      <Plot
                        data={buildTraces(stratum.groups)}
                        layout={{
                          ...baseLayout,
                          autosize: true,
                          height: miniH,
                          margin: { t: 10, r: 10, b: 40, l: 50 },
                          xaxis: { ...(baseLayout.xaxis as any), title: { text: kmCustomDurationTitle || kmDuration } },
                          yaxis: { ...(baseLayout.yaxis as any), range: [0, 1.05], tickformat: ".0%", title: { text: "Survival" } },
                          legend: { font: { size: 9 }, orientation: "h", y: -0.22 },
                          annotations: pAnnot as any,
                        }}
                        style={{ width: "100%", height: miniH }}
                        config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] }}
                        useResizeHandler
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}
      </Section>
      )}

      {/* ── Cox PH ── */}
      {activeMethod === "cox" && (
      <Section title="Cox Proportional Hazards" description="Regression for time-to-event data — outputs Hazard Ratios (HR)">
        <div className="grid grid-cols-2 gap-3">
          <VarSelect label="Duration (time)" value={coxDuration} onChange={setCoxDuration} columns={columns} kinds={["numeric"]} />
          <VarSelect label="Event (0/1)" value={coxEvent} onChange={setCoxEvent} columns={binaryCols} />
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
      )}

      {/* ───────────────────────────────────────────────────────────────────── */}
      {/* Time-horizon sensitivity → Forest plot                               */}
      {/* ───────────────────────────────────────────────────────────────────── */}
      {activeMethod === "timehorizon" && (
      <Section
        title="Time-horizon HR (forest)"
        description="Run the same Cox model at several follow-up windows (1-year, 2-year, full) and send the hazard ratios straight to the Forest Builder. Answers 'does the effect hold early vs. late?'."
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <VarSelect label="Duration / Time" value={chDuration} onChange={setChDuration} columns={columns} kinds={["numeric"]} />
          <VarSelect label="Event (0/1)" value={chEvent} onChange={setChEvent} columns={binaryCols} />
          <VarSelect label="Predictor (HR tracked)" value={chPredictor} onChange={setChPredictor} columns={columns} />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Horizon cut-points (comma-sep)</span>
            <input value={chHorizons} onChange={(e) => setChHorizons(e.target.value)}
              placeholder="365, 730"
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400" />
            <span className="text-[10px] text-gray-400">
              In your Duration column's unit. Days → 365, 730 · months → 12, 24 · years → 1, 2.
            </span>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Labels (comma-sep, optional)</span>
            <input value={chLabels} onChange={(e) => setChLabels(e.target.value)}
              placeholder="1 year, 2 years"
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400" />
            <span className="text-[10px] text-gray-400">
              One label per cut-point. Blank → auto "≤ 365". Count must match cut-points.
            </span>
          </label>
        </div>

        <div className="mt-2">
          <MultiSelect label="Adjustment covariates (optional)" columns={columns}
            selected={chCovariates} onChange={setChCovariates}
            excludeNames={[chDuration, chEvent, chPredictor].filter(Boolean)} />
        </div>

        <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mt-1">
          <span className="text-amber-500 text-sm leading-none mt-0.5">ⓘ</span>
          <p className="text-[11px] text-amber-800 leading-relaxed">
            <b>Cut-points use the same time unit as your Duration column.</b> Follow-up in days → write
            {" "}<span className="font-mono">365, 730</span>; in months → <span className="font-mono">12, 24</span>;
            in years → <span className="font-mono">1, 2</span>. A <b>"Full follow-up"</b> row (all events,
            no censoring) is appended automatically. Each window applies administrative censoring at its
            cut-point, so shorter windows have fewer events and wider CIs.
          </p>
        </div>

        <div className="flex items-center gap-3 flex-wrap mt-2">
          <RunButton
            loading={chLoading}
            label="Run horizons"
            onClick={async () => {
              if (!chDuration || !chEvent || !chPredictor) {
                setChError("Select duration, event, and a predictor.");
                return;
              }
              const horizons = chHorizons.split(",").map((s) => parseFloat(s.trim())).filter((x) => !Number.isNaN(x) && x > 0);
              if (horizons.length === 0) {
                setChError("Enter at least one positive horizon cut-point.");
                return;
              }
              const labels = chLabels.split(",").map((s) => s.trim()).filter(Boolean);
              setChResult(null); setChError(null); setChLoading(true);
              try {
                const res = await runCoxHorizons({
                  session_id: sid,
                  duration_col: chDuration,
                  event_col: chEvent,
                  predictor: chPredictor,
                  covariates: chCovariates.length ? chCovariates : undefined,
                  horizons,
                  horizon_labels: labels.length === horizons.length ? labels : undefined,
                  include_full: true,
                });
                setChResult(res.data);
              } catch (e: any) {
                setChError(e?.response?.data?.detail ?? "Time-horizon analysis failed");
              } finally {
                setChLoading(false);
              }
            }}
          />
          {chResult?.forest_rows?.length > 0 && (
            <button
              onClick={() => {
                const cov = (chResult.covariates ?? []) as string[];
                // Keep p + event counts in the figure — richer than the
                // bare reference look. Right header reflects that content.
                setForestHandoff(chResult.forest_rows, {
                  customTitle: "",
                  customSubtitle: cov.length ? `Adjusted for ${cov.join(" + ")}` : "(unadjusted; red = 95% CI excludes 1)",
                  xLabel: `${cov.length ? "Adjusted" : "Unadjusted"} hazard ratio for ${chResult.predictor} (95% CI), log scale`,
                  leftHeader: "Time horizon",
                  rightHeader: "HR (95% CI), p",
                });
                setVisualSubTab("forest");
                setActiveTab("visual");
              }}
              className="px-4 py-2 text-sm font-medium bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors"
            >
              → Send to Forest Builder
            </button>
          )}
        </div>

        {chError && <p className="text-sm text-red-500 mt-2">{chError}</p>}

        {chResult && (
          <div className="mt-3 space-y-3">
            <div className="overflow-x-auto rounded-xl border border-gray-200">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 text-gray-500">
                  <tr>
                    <th className="text-left px-3 py-2 font-medium">Horizon</th>
                    <th className="text-right px-3 py-2 font-medium">n events</th>
                    <th className="text-right px-3 py-2 font-medium">HR</th>
                    <th className="text-right px-3 py-2 font-medium">95% CI</th>
                    <th className="text-right px-3 py-2 font-medium">p</th>
                  </tr>
                </thead>
                <tbody>
                  {(chResult.forest_rows ?? []).map((row: any, i: number) => (
                    <tr key={i} className="border-t border-gray-100">
                      <td className="px-3 py-1.5 text-gray-800">{row.label}</td>
                      <td className="px-3 py-1.5 text-right text-gray-500">{(row.extra ?? "").replace(/[()]/g, "").replace(" events", "")}</td>
                      <td className="px-3 py-1.5 text-right font-semibold text-gray-900">{row.est != null ? row.est.toFixed(2) : "—"}</td>
                      <td className="px-3 py-1.5 text-right text-gray-600">
                        {row.ci_low != null && row.ci_high != null ? `${row.ci_low.toFixed(2)}–${row.ci_high.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-3 py-1.5 text-right text-gray-600">{row.p != null ? (row.p < 0.001 ? "<0.001" : row.p.toFixed(3)) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {chResult.interpretation && (
              <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3 text-sm text-indigo-900">
                {chResult.interpretation}
              </div>
            )}
          </div>
        )}
      </Section>
      )}
      </div>
    </div>
  );
}



