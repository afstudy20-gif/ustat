import React, { useState, useEffect } from "react";
import { useStore } from "../store";
import api from "../api";
import ResultExporter from "./ResultExporter";

// ── Stat definitions ──────────────────────────────────────────────────────────

interface StatDef {
  id: string;
  label: string;
  group: "tendency" | "dispersion" | "percentile" | "counts";
}

const STAT_DEFS: StatDef[] = [
  { id: "auto",       label: "Auto (normality-based)",  group: "tendency" },
  { id: "mean_sd",    label: "Mean ± SD",               group: "tendency" },
  { id: "median_iqr", label: "Median [IQR]",            group: "tendency" },
  { id: "se",         label: "SE of Mean",              group: "dispersion" },
  { id: "ci95",       label: "95% CI",                  group: "dispersion" },
  { id: "variance",   label: "Variance",                group: "dispersion" },
  { id: "min_max",    label: "Min – Max",               group: "dispersion" },
  { id: "p10",        label: "10th Percentile",         group: "percentile" },
  { id: "p25",        label: "25th Percentile",         group: "percentile" },
  { id: "p75",        label: "75th Percentile",         group: "percentile" },
  { id: "p90",        label: "90th Percentile",         group: "percentile" },
  { id: "p95",        label: "95th Percentile",         group: "percentile" },
  { id: "n",          label: "N (non-missing)",         group: "counts" },
  { id: "missing",    label: "Missing count",           group: "counts" },
];

const STAT_GROUPS = [
  { id: "tendency",   label: "Central Tendency" },
  { id: "dispersion", label: "Dispersion" },
  { id: "percentile", label: "Percentiles" },
  { id: "counts",     label: "Counts" },
];

// ── Types ─────────────────────────────────────────────────────────────────────

interface StatRow { label: string; overall: string; group_stats: Record<string, string> }

interface T1Row {
  variable: string;
  type: "numeric" | "categorical";
  overall_n: number;
  stat_rows?: StatRow[];
  stat_label?: string;
  overall?: string;
  p_value: string | null;
  test: string | null;
  significant?: boolean;
  normal?: boolean;
  normality_test?: string;
  normality_p?: number;
  group_stats: Record<string, string>;
  sub_rows?: { category: string; overall: string; group_stats: Record<string, string> }[];
}

interface T1Result {
  group_column: string | null;
  group_labels: string[];
  group_ns: Record<string, number>;
  total_n: number;
  rows: T1Row[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function pColor(p: string | null) {
  if (!p || p === "N/A") return "text-gray-400";
  if (p === "<0.001") return "text-red-600 font-bold";
  const v = parseFloat(p);
  if (v < 0.05) return "text-amber-600 font-semibold";
  return "text-gray-400";
}

function pStars(p: string | null) {
  if (!p || p === "N/A") return "";
  if (p === "<0.001") return "***";
  const v = parseFloat(p);
  if (v < 0.001) return "***";
  if (v < 0.01) return "**";
  if (v < 0.05) return "*";
  return "ns";
}

// ── Stats selector panel ──────────────────────────────────────────────────────

function StatsSelector({
  selected,
  onChange,
}: {
  selected: Set<string>;
  onChange: (s: Set<string>) => void;
}) {
  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) {
      if (next.size === 1) return;
      next.delete(id);
    } else {
      next.add(id);
    }
    onChange(next);
  };

  return (
    <div className="border-b border-gray-200 pb-1">
      <div className="px-3 pt-2 pb-1 flex items-center justify-between">
        <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
          Statistics shown
        </h3>
        <span className="text-[9px] text-gray-400">numeric vars</span>
      </div>
      {STAT_GROUPS.map((grp) => {
        const defs = STAT_DEFS.filter((d) => d.group === grp.id);
        return (
          <div key={grp.id} className="mb-1">
            <p className="px-3 text-[9px] font-semibold text-gray-400 uppercase tracking-wider mt-1.5">
              {grp.label}
            </p>
            {defs.map((d) => (
              <label
                key={d.id}
                className={`flex items-center gap-2 px-3 py-0.5 cursor-pointer transition-colors
                  ${selected.has(d.id) ? "text-indigo-600" : "text-gray-400 hover:text-gray-600"}`}
              >
                <input
                  type="checkbox"
                  className="accent-indigo-500 flex-shrink-0"
                  checked={selected.has(d.id)}
                  onChange={() => toggle(d.id)}
                />
                <span className="text-xs">{d.label}</span>
              </label>
            ))}
          </div>
        );
      })}
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function Table1Panel() {
  const session = useStore((s) => s.session);
  const result = useStore((s) => s.table1Result) as T1Result | null;
  const setResult = useStore((s) => s.setTable1Result);
  const clearTable1 = useStore((s) => s.clearTable1);
  if (!session) return null;

  const allCols = session.columns.map((c) => c.name);

  const [groupCol, setGroupCol] = useState<string>("");
  const [selected, setSelected] = useState<Set<string>>(new Set(allCols));
  const [kindOverrides, setKindOverrides] = useState<Record<string, "numeric" | "categorical">>({});
  const [selectedStats, setSelectedStats] = useState<Set<string>>(new Set(["auto"]));
  const [showStats, setShowStats] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    setSelected(new Set(allCols.filter((c) => c !== groupCol)));
    setKindOverrides({});
    clearTable1();
    setError(null);
  }, [session.session_id]);

  const toggleKind = (col: string) => {
    setKindOverrides((prev) => {
      const base = session.columns.find((c) => c.name === col)?.kind ?? "numeric";
      const current = prev[col] ?? base;
      const next = current === "numeric" ? "categorical" : "numeric";
      if (next === base) {
        const { [col]: _, ...rest } = prev;
        return rest;
      }
      return { ...prev, [col]: next };
    });
    setResult(null);
  };

  const toggle = (col: string) =>
    setSelected((prev) => {
      const s = new Set(prev);
      s.has(col) ? s.delete(col) : s.add(col);
      return s;
    });

  const selectAll = () => setSelected(new Set(allCols.filter((c) => c !== groupCol)));
  const selectNone = () => setSelected(new Set());

  const handleGroupChange = (col: string) => {
    setGroupCol(col);
    setSelected((prev) => { const s = new Set(prev); s.delete(col); return s; });
    setResult(null);
  };

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    const variable_kinds: Record<string, string> = {};
    Array.from(selected).forEach((col) => {
      const kind = kindOverrides[col] ?? session.columns.find((c) => c.name === col)?.kind;
      if (kind) variable_kinds[col] = kind;
    });
    try {
      const res = await api.post("/api/stats/table1", {
        session_id: session.session_id,
        group_column: groupCol || null,
        variables: Array.from(selected),
        variable_kinds,
        selected_stats: Array.from(selectedStats),
      });
      setResult(res.data);
    } catch (e: any) {
      setError(e.response?.data?.detail ?? e.message ?? "Error running Table 1");
    } finally { setLoading(false); }
  };

  const buildExportData = () => {
    if (!result) return { headers: [] as string[], rows: [] as string[][] };
    const gl = result.group_labels;
    const headers = ["Variable", "Statistic",
      `Overall (n=${result.total_n})`,
      ...gl.map((g) => `${result.group_column ? result.group_column + "=" : ""}${g} (n=${result.group_ns[g] ?? ""})`),
      "p-value", "Test", "Normality test"];
    const rows: string[][] = [];
    result.rows.forEach((row: any) => {
      if (row.type === "numeric") {
        (row.stat_rows ?? []).forEach((sr: any, i: number) => {
          rows.push([
            i === 0 ? row.variable : "",
            sr.label, sr.overall,
            ...gl.map((g: string) => sr.group_stats[g] ?? ""),
            i === 0 ? (row.p_value ?? "") : "",
            i === 0 ? (row.test ?? "") : "",
            i === 0 ? `${row.normality_test} (p=${row.normality_p})` : "",
          ]);
        });
      } else {
        rows.push([row.variable, "n (%)", `n=${row.overall_n}`,
          ...gl.map(() => ""), row.p_value ?? "", row.test ?? "", ""]);
        (row.sub_rows ?? []).forEach((sr: any) => {
          rows.push([`  ${sr.category}`, "", sr.overall,
            ...gl.map((g: string) => sr.group_stats[g] ?? ""), "", "", ""]);
        });
      }
    });
    return { headers, rows };
  };

  const { headers: exportHeaders, rows: exportRows } = buildExportData();

  const filteredCols = allCols.filter(
    (c) => c !== groupCol && c.toLowerCase().includes(search.toLowerCase())
  );
  const hasGroups = result && result.group_labels.length > 0;

  const statsLabel = selectedStats.has("auto")
    ? "Auto"
    : `${selectedStats.size} stat${selectedStats.size > 1 ? "s" : ""}`;

  return (
    <div className="flex gap-0 h-full" style={{ minHeight: 0 }}>

      {/* ── Left sidebar ── */}
      <div className="w-56 flex-shrink-0 flex flex-col border-r border-gray-200 bg-white overflow-hidden">

        {/* Group column */}
        <div className="p-3 border-b border-gray-200 space-y-1.5">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Group by</h3>
          <select
            className="select w-full text-xs"
            value={groupCol}
            onChange={(e) => handleGroupChange(e.target.value)}
          >
            <option value="">— Overall only —</option>
            {session.columns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} [{c.kind === "numeric" ? "N" : "C"}]
              </option>
            ))}
          </select>
          {groupCol && (
            <p className="text-[10px] text-gray-400 leading-tight">
              Separate columns per group
            </p>
          )}
        </div>

        {/* Statistics selector (collapsible) */}
        <div className="border-b border-gray-200 flex-shrink-0">
          <button
            className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-50 transition-colors"
            onClick={() => setShowStats((v) => !v)}
          >
            <div className="flex items-center gap-1.5">
              <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                Statistics
              </h3>
              <span className="text-[9px] text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded-full">
                {statsLabel}
              </span>
            </div>
            <span className="text-gray-400 text-xs">{showStats ? "▲" : "▼"}</span>
          </button>
          {showStats && (
            <div className="overflow-y-auto max-h-64">
              <StatsSelector selected={selectedStats} onChange={setSelectedStats} />
            </div>
          )}
        </div>

        {/* Variable selector */}
        <div className="px-3 py-2 border-b border-gray-200 flex items-center justify-between flex-shrink-0">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
            Variables ({selected.size})
          </h3>
          <div className="flex gap-2">
            <button className="text-[10px] text-indigo-600 hover:text-indigo-700" onClick={selectAll}>All</button>
            <button className="text-[10px] text-indigo-600 hover:text-indigo-700" onClick={selectNone}>None</button>
          </div>
        </div>
        <div className="px-2 py-1.5 border-b border-gray-200 flex-shrink-0">
          <input className="select w-full text-xs" placeholder="Search variables…"
            value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>

        <div className="overflow-y-auto flex-1">
          {filteredCols.map((col) => {
            const baseKind = session.columns.find((c) => c.name === col)?.kind ?? "numeric";
            const effectiveKind = kindOverrides[col] ?? baseKind;
            const isOverridden = col in kindOverrides;
            const isChecked = selected.has(col);
            return (
              <div key={col}
                className={`flex items-center gap-1.5 px-2 py-1.5 border-b border-gray-100 transition-colors
                  ${isChecked ? "bg-indigo-50" : "hover:bg-gray-50"}`}>
                <input type="checkbox" className="accent-indigo-500 flex-shrink-0"
                  checked={isChecked} onChange={() => toggle(col)} />
                <button
                  onClick={(e) => { e.stopPropagation(); toggleKind(col); }}
                  title={`Currently: ${effectiveKind}. Click to switch to ${effectiveKind === "numeric" ? "categorical" : "numeric"}`}
                  className={`text-[9px] font-bold px-1.5 py-0.5 rounded flex-shrink-0 border transition-colors
                    ${effectiveKind === "numeric"
                      ? "bg-blue-100 text-blue-700 border-blue-300 hover:bg-blue-200"
                      : "bg-purple-100 text-purple-700 border-purple-300 hover:bg-purple-200"}
                    ${isOverridden ? "ring-1 ring-amber-400" : ""}`}>
                  {effectiveKind === "numeric" ? "N" : "C"}
                </button>
                <span className="text-xs text-gray-700 truncate flex-1">{col}</span>
                {isOverridden && (
                  <span className="text-[9px] text-amber-500 flex-shrink-0" title="Type overridden">★</span>
                )}
              </div>
            );
          })}
        </div>

        {/* Actions */}
        <div className="p-3 border-t border-gray-200 space-y-2 flex-shrink-0">
          <button className="btn-primary w-full text-sm py-2" onClick={run}
            disabled={loading || selected.size === 0}>
            {loading ? "Computing…" : "Generate Table"}
          </button>
          {result && (
            <ResultExporter
              title="Table1"
              headers={exportHeaders}
              rows={exportRows}
            />
          )}
          {error && (
            <p className="text-red-500 text-xs bg-red-50 rounded p-2 leading-relaxed">{error}</p>
          )}
        </div>
      </div>

      {/* ── Right: table ── */}
      <div className="flex-1 overflow-auto bg-gray-50">
        {!result && !loading && (
          <div className="h-full flex flex-col items-center justify-center text-gray-400 space-y-3">
            <div className="text-5xl opacity-20">📋</div>
            <p className="text-base font-medium text-gray-500">Table — Baseline Characteristics</p>
            <div className="text-xs text-gray-400 space-y-1 text-center leading-relaxed">
              <p>1. Pick a <span className="text-gray-500">Group by</span> column (e.g. outcome, treatment)</p>
              <p>2. Choose <span className="text-gray-500">Statistics</span> to display</p>
              <p>3. Select variables · Click <span className="text-gray-500">Generate Table</span></p>
            </div>
          </div>
        )}
        {loading && (
          <div className="h-full flex items-center justify-center text-gray-400 animate-pulse">
            Computing statistics…
          </div>
        )}

        {result && (
          <div className="p-4">
            <div className="rounded-xl border border-gray-200 overflow-hidden shadow-sm">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-gray-50 border-b border-gray-200">
                    <th className="text-left px-4 py-3 text-gray-700 font-semibold w-44 border-r border-gray-200">
                      Variable
                    </th>
                    <th className="text-center px-3 py-3 text-gray-400 font-normal text-xs w-32 border-r border-gray-200">
                      Statistic
                    </th>
                    <th className="text-center px-4 py-3 text-gray-700 font-semibold border-r border-gray-200">
                      Overall
                      <br /><span className="text-xs font-normal text-gray-400">n = {result.total_n}</span>
                    </th>
                    {result.group_labels.map((g) => (
                      <th key={g} className="text-center px-4 py-3 text-indigo-600 font-semibold border-r border-gray-200">
                        {result.group_column && <span className="text-gray-400 text-xs font-normal">{result.group_column} = </span>}
                        {g}
                        <br /><span className="text-xs font-normal text-gray-400">n = {result.group_ns[g] ?? ""}</span>
                      </th>
                    ))}
                    {hasGroups && (
                      <>
                        <th className="text-center px-3 py-3 text-gray-700 font-semibold w-24">p-value</th>
                        <th className="text-center px-3 py-3 text-gray-400 font-normal text-xs w-28">Test</th>
                      </>
                    )}
                    <th className="text-center px-2 py-3 text-gray-400 font-normal text-[10px] w-20">
                      Normality
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, ri) =>
                    row.type === "numeric" ? (
                      <React.Fragment key={`num-${ri}`}>
                      {(row.stat_rows ?? []).map((sr, si) => (
                        <tr key={`${ri}-${si}`}
                          className={`border-t transition-colors hover:bg-gray-50
                            ${si === 0 ? "border-gray-200" : "border-gray-100"}
                            ${row.significant && si === 0 ? "bg-amber-50/30" : ""}`}>
                          <td className={`px-4 py-2 border-r border-gray-200
                            ${si === 0 ? "font-medium text-gray-900" : ""}`}>
                            {si === 0 ? row.variable : ""}
                          </td>
                          <td className="px-3 py-1.5 text-center text-xs text-gray-400 border-r border-gray-200">
                            {sr.label}
                          </td>
                          <td className="px-4 py-1.5 text-center text-gray-700 font-mono text-xs border-r border-gray-200">
                            {sr.overall}
                          </td>
                          {result.group_labels.map((g) => (
                            <td key={g} className="px-4 py-1.5 text-center text-gray-700 font-mono text-xs border-r border-gray-200">
                              {sr.group_stats[g] ?? "—"}
                            </td>
                          ))}
                          {hasGroups && (
                            <>
                              <td className={`px-3 py-1.5 text-center text-xs font-mono ${si === 0 ? pColor(row.p_value) : "text-transparent"}`}>
                                {si === 0 ? (
                                  <>
                                    {row.p_value ?? "—"}
                                    {row.p_value && <span className="ml-0.5 text-[10px] opacity-70">{pStars(row.p_value)}</span>}
                                  </>
                                ) : null}
                              </td>
                              <td className="px-3 py-1.5 text-center text-xs text-gray-400">
                                {si === 0 ? row.test : ""}
                              </td>
                            </>
                          )}
                          <td className="px-2 py-1.5 text-center">
                            {si === 0 && row.normality_test ? (
                              <div className={`text-[9px] px-1 py-0.5 rounded font-medium inline-block
                                ${row.normal
                                  ? "bg-green-100 text-green-700 border border-green-300"
                                  : "bg-orange-100 text-orange-700 border border-orange-300"}`}>
                                {row.normal ? "Normal" : "Non-normal"}
                                <br />
                                <span className="text-gray-400 font-normal">
                                  {row.normality_test === "Shapiro-Wilk" ? "S-W" : "K-S"} p={row.normality_p?.toFixed(3)}
                                </span>
                              </div>
                            ) : null}
                          </td>
                        </tr>
                      ))}
                      </React.Fragment>
                    ) : (
                      <React.Fragment key={`cat-${ri}`}>
                        <tr key={`${ri}-hdr`} className="border-t-2 border-gray-200 bg-gray-50">
                          <td className="px-4 py-2 font-semibold text-indigo-600 border-r border-gray-200">{row.variable}</td>
                          <td className="px-3 py-2 text-center text-xs text-gray-400 border-r border-gray-200">n (%)</td>
                          <td className="px-4 py-2 text-center text-xs text-gray-400 border-r border-gray-200">n = {row.overall_n}</td>
                          {result.group_labels.map((g) => {
                            const gn = (row.sub_rows ?? []).reduce((s, sr) => {
                              const m = sr.group_stats[g]?.match(/^(\d+)/);
                              return s + parseInt(m?.[1] ?? "0");
                            }, 0);
                            return (
                              <td key={g} className="px-4 py-2 text-center text-xs text-gray-400 border-r border-gray-200">
                                n = {gn}
                              </td>
                            );
                          })}
                          {hasGroups && (
                            <>
                              <td className={`px-3 py-2 text-center text-xs font-mono ${pColor(row.p_value)}`}>
                                {row.p_value ?? "—"}
                                {row.p_value && <span className="ml-0.5 text-[10px] opacity-70">{pStars(row.p_value)}</span>}
                              </td>
                              <td className="px-3 py-2 text-center text-xs text-gray-400">{row.test}</td>
                            </>
                          )}
                          <td className="px-2 py-2 text-center text-[9px] text-gray-300">—</td>
                        </tr>
                        {(row.sub_rows ?? []).map((sr, si) => (
                          <tr key={`${ri}-${si}`} className="border-t border-gray-100 hover:bg-gray-50 transition-colors">
                            <td className="px-4 py-1.5 pl-8 text-gray-500 text-xs border-r border-gray-200">
                              <span className="text-gray-300 mr-1">›</span>{sr.category}
                            </td>
                            <td className="border-r border-gray-200" />
                            <td className="px-4 py-1.5 text-center text-gray-700 font-mono text-xs border-r border-gray-200">{sr.overall}</td>
                            {result.group_labels.map((g) => (
                              <td key={g} className="px-4 py-1.5 text-center text-gray-700 font-mono text-xs border-r border-gray-200">
                                {sr.group_stats[g] ?? "—"}
                              </td>
                            ))}
                            {hasGroups && <><td /><td /></>}
                            <td />
                          </tr>
                        ))}
                      </React.Fragment>
                    )
                  )}
                </tbody>
              </table>
            </div>

            {/* Interpretation banner */}
            <div className="mt-3 px-3 py-2 bg-indigo-50 border border-indigo-200 rounded-lg text-[11px] text-indigo-800 space-y-1 leading-relaxed">
              <p className="font-semibold text-indigo-900">How to read this table</p>
              <p>
                Continuous variables are tested for normality to decide how they are summarised and compared:
                <span className="font-medium"> n ≤ 2000 → Shapiro-Wilk · |skewness| ≤ 1.5 at large n → CLT bypass · otherwise → Lilliefors.</span>
              </p>
              <p>Normal → <span className="font-medium">Mean ± SD</span> with t-test/ANOVA · Non-normal → <span className="font-medium">Median [IQR]</span> with Mann-Whitney/Kruskal-Wallis.</p>
              <p>Categorical: Chi-square · Fisher's exact when any expected cell &lt; 5.</p>
              <p className="text-indigo-600">*** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · ns = not significant</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
