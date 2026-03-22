import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { useStore } from "../store";
import type { ColMeta, CaseCondition, CaseOperator } from "../store";
import api from "../api";
import { selectCases, clearCases } from "../api";

// ── Kind cycling ───────────────────────────────────────────────────────────────

const KIND_CYCLE: ColMeta["kind"][] = ["numeric", "categorical", "boolean", "text"];

const KIND_STYLE: Record<string, string> = {
  numeric:     "bg-blue-100 text-blue-700 border-blue-300 hover:bg-blue-200",
  categorical: "bg-orange-100 text-orange-700 border-orange-300 hover:bg-orange-200",
  boolean:     "bg-green-100 text-green-700 border-green-300 hover:bg-green-200",
  text:        "bg-gray-100 text-gray-500 border-gray-300 hover:bg-gray-200",
};

const KIND_LABEL: Record<string, string> = {
  numeric: "num", categorical: "cat", boolean: "bool", text: "txt",
};

// ── Select Cases Modal ──────────────────────────────────────────────────────────

const OPERATORS: { value: CaseOperator; label: string; noValue?: boolean }[] = [
  { value: "eq",          label: "=" },
  { value: "ne",          label: "≠" },
  { value: "gt",          label: ">" },
  { value: "lt",          label: "<" },
  { value: "gte",         label: "≥" },
  { value: "lte",         label: "≤" },
  { value: "contains",    label: "contains" },
  { value: "missing",     label: "is missing",     noValue: true },
  { value: "not_missing", label: "is not missing", noValue: true },
];

function SelectCasesModal({
  columns,
  sessionId,
  existing,
  onApply,
  onClear,
  onClose,
}: {
  columns: ColMeta[];
  sessionId: string;
  existing: CaseCondition[];
  onApply: (conditions: CaseCondition[], selected: number, total: number) => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const defaultCond = (): CaseCondition => ({
    column: columns[0]?.name ?? "",
    operator: "eq",
    value: "",
    join: "AND",
  });

  const [conditions, setConditions] = useState<CaseCondition[]>(
    existing.length > 0 ? existing : [defaultCond()]
  );
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<{ selected: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const updateCond = (i: number, patch: Partial<CaseCondition>) => {
    setConditions((prev) => prev.map((c, idx) => idx === i ? { ...c, ...patch } : c));
    setPreview(null);
  };

  const addCond = () => setConditions((prev) => [...prev, defaultCond()]);
  const removeCond = (i: number) => setConditions((prev) => prev.filter((_, idx) => idx !== i));

  const handlePreview = async () => {
    setBusy(true); setError(null);
    try {
      const res = await selectCases(sessionId, conditions);
      setPreview(res.data);
    } catch { setError("Preview failed"); }
    finally { setBusy(false); }
  };

  const handleApply = async () => {
    setBusy(true); setError(null);
    try {
      const res = await selectCases(sessionId, conditions);
      onApply(conditions, res.data.selected, res.data.total);
    } catch { setError("Apply failed"); }
    finally { setBusy(false); }
  };

  const handleClear = async () => {
    setBusy(true);
    try {
      await clearCases(sessionId);
      onClear();
    } finally { setBusy(false); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-xl p-6 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-gray-900">Select Cases</h2>
            <p className="text-xs text-gray-400 mt-0.5">All analyses will use only the selected subset</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100">✕</button>
        </div>

        {/* Condition rows */}
        <div className="flex flex-col gap-2">
          {conditions.map((cond, i) => {
            const opMeta = OPERATORS.find((o) => o.value === cond.operator);
            return (
              <div key={i} className="flex items-center gap-2">
                {/* AND/OR join (only for rows after the first) */}
                {i > 0 ? (
                  <select
                    value={cond.join}
                    onChange={(e) => updateCond(i, { join: e.target.value as "AND" | "OR" })}
                    className="text-xs border border-gray-300 rounded-lg px-2 py-1.5 w-16 bg-white text-gray-700 focus:outline-none focus:border-violet-400"
                  >
                    <option value="AND">AND</option>
                    <option value="OR">OR</option>
                  </select>
                ) : (
                  <div className="w-16 text-center text-xs text-gray-400 font-medium">WHERE</div>
                )}

                {/* Column */}
                <select
                  value={cond.column}
                  onChange={(e) => updateCond(i, { column: e.target.value })}
                  className="flex-1 text-xs border border-gray-300 rounded-lg px-2 py-1.5 bg-white text-gray-700 focus:outline-none focus:border-violet-400 min-w-0"
                >
                  {columns.map((c) => (
                    <option key={c.name} value={c.name}>{c.name}</option>
                  ))}
                </select>

                {/* Operator */}
                <select
                  value={cond.operator}
                  onChange={(e) => updateCond(i, { operator: e.target.value as CaseOperator })}
                  className="text-xs border border-gray-300 rounded-lg px-2 py-1.5 w-32 bg-white text-gray-700 focus:outline-none focus:border-violet-400"
                >
                  {OPERATORS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>

                {/* Value */}
                {!opMeta?.noValue ? (
                  <input
                    type="text"
                    value={cond.value}
                    onChange={(e) => updateCond(i, { value: e.target.value })}
                    placeholder="value"
                    className="w-28 text-xs border border-gray-300 rounded-lg px-2 py-1.5 bg-white text-gray-700 focus:outline-none focus:border-violet-400"
                  />
                ) : (
                  <div className="w-28" />
                )}

                {/* Remove */}
                <button
                  onClick={() => removeCond(i)}
                  disabled={conditions.length === 1}
                  className="p-1.5 text-gray-300 hover:text-red-500 disabled:opacity-30 transition-colors"
                >✕</button>
              </div>
            );
          })}
        </div>

        {/* Add condition */}
        <button
          onClick={addCond}
          className="text-xs text-violet-600 hover:text-violet-800 self-start flex items-center gap-1"
        >
          + Add condition
        </button>

        {/* Preview result */}
        {preview && (
          <div className="bg-violet-50 border border-violet-200 rounded-xl px-4 py-3 text-sm text-violet-800">
            <span className="font-bold text-lg">{preview.selected.toLocaleString()}</span>
            <span className="text-violet-500"> of {preview.total.toLocaleString()} cases match</span>
            <span className="text-violet-400 text-xs ml-2">
              ({((preview.selected / preview.total) * 100).toFixed(1)}%)
            </span>
          </div>
        )}

        {error && <p className="text-xs text-red-500">{error}</p>}

        {/* Actions */}
        <div className="flex items-center gap-2 pt-1">
          <button
            onClick={handlePreview}
            disabled={busy}
            className="px-4 py-2 text-sm border border-violet-300 text-violet-700 rounded-xl hover:bg-violet-50 transition-colors disabled:opacity-50"
          >
            Preview
          </button>
          <button
            onClick={handleApply}
            disabled={busy}
            className="flex-1 px-4 py-2 text-sm bg-violet-600 text-white rounded-xl hover:bg-violet-700 transition-colors disabled:opacity-50 font-medium"
          >
            {busy ? "Applying…" : "Apply"}
          </button>
          <button
            onClick={handleClear}
            disabled={busy}
            className="px-4 py-2 text-sm border border-gray-200 text-gray-500 rounded-xl hover:bg-gray-50 transition-colors disabled:opacity-50"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

type SortDir = "asc" | "desc";

export default function DataTable() {
  const session          = useStore((s) => s.session);
  const updateColumnKind = useStore((s) => s.updateColumnKind);
  const updatePreviewCell = useStore((s) => s.updatePreviewCell);
  const caseFilter       = useStore((s) => s.caseFilter);
  const setCaseFilter    = useStore((s) => s.setCaseFilter);

  const [sortCol,     setSortCol]     = useState<string | null>(null);
  const [sortDir,     setSortDir]     = useState<SortDir>("asc");
  const [filters,     setFilters]     = useState<Record<string, string>>({});
  const [showFilters, setShowFilters] = useState(false);
  const [editCell,       setEditCell]      = useState<{ rowIdx: number; col: string } | null>(null);
  const [editValue,      setEditValue]     = useState("");
  const [saving,         setSaving]        = useState(false);
  const [showSaveMenu,   setShowSaveMenu]  = useState(false);
  const [showMissingOnly, setShowMissingOnly] = useState(false);
  const [showSelectCases, setShowSelectCases] = useState(false);

  const inputRef   = useRef<HTMLInputElement>(null);
  const saveMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showSaveMenu) return;
    const handler = (e: MouseEvent) => {
      if (saveMenuRef.current && !saveMenuRef.current.contains(e.target as Node)) {
        setShowSaveMenu(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showSaveMenu]);

  useEffect(() => {
    if (editCell) setTimeout(() => inputRef.current?.focus(), 0);
  }, [editCell]);

  useEffect(() => {
    setSortCol(null); setFilters({}); setShowMissingOnly(false);
  }, [session?.session_id]);

  if (!session) return null;
  const { preview, columns } = session;

  type IndexedRow = Record<string, unknown> & { _idx: number };
  const indexedRows = useMemo(
    () => preview.map((row, idx): IndexedRow => ({ ...row, _idx: idx })),
    [preview]
  );

  // Per-column missing counts (computed once over full preview)
  const missingCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const col of columns) {
      counts[col.name] = preview.filter(
        (row) => row[col.name] === null || row[col.name] === undefined || row[col.name] === ""
      ).length;
    }
    return counts;
  }, [preview, columns]);

  const totalMissingRows = useMemo(
    () => indexedRows.filter((row) =>
      columns.some((col) => row[col.name] === null || row[col.name] === undefined || row[col.name] === "")
    ).length,
    [indexedRows, columns]
  );

  const filtered = useMemo(() => {
    const hasFilters = Object.values(filters).some(Boolean);
    let rows = indexedRows;

    if (showMissingOnly) {
      rows = rows.filter((row) =>
        columns.some((col) => row[col.name] === null || row[col.name] === undefined || row[col.name] === "")
      );
    }

    if (!hasFilters) return rows;
    return rows.filter((row) =>
      columns.every((col) => {
        const f = filters[col.name];
        if (!f) return true;
        const cell = row[col.name];
        if (cell === null || cell === undefined) return f === "";
        return String(cell).toLowerCase().includes(f.toLowerCase());
      })
    );
  }, [indexedRows, filters, columns, showMissingOnly]);

  const displayRows = useMemo(() => {
    if (!sortCol) return filtered;
    return [...filtered].sort((a, b) => {
      const av = a[sortCol], bv = b[sortCol];
      if (av == null) return 1;
      if (bv == null) return -1;
      const cmp =
        typeof av === "number" && typeof bv === "number"
          ? av - bv
          : String(av).localeCompare(String(bv), undefined, { numeric: true, sensitivity: "base" });
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [filtered, sortCol, sortDir]);

  const toggleSort = (colName: string) => {
    if (sortCol === colName) {
      if (sortDir === "asc") setSortDir("desc");
      else setSortCol(null);
    } else {
      setSortCol(colName);
      setSortDir("asc");
    }
  };

  const cycleKind = (colName: string) => {
    const cur = columns.find((c) => c.name === colName)?.kind ?? "numeric";
    const next = KIND_CYCLE[(KIND_CYCLE.indexOf(cur) + 1) % KIND_CYCLE.length];
    updateColumnKind(colName, next);
  };

  const startEdit = (rowIdx: number, col: string) => {
    const val = preview[rowIdx]?.[col];
    setEditCell({ rowIdx, col });
    setEditValue(val === null || val === undefined ? "" : String(val));
  };

  const commitEdit = async () => {
    if (!editCell || saving) return;
    const { rowIdx, col } = editCell;
    setEditCell(null);

    const original = preview[rowIdx]?.[col];
    const rawVal   = editValue.trim();
    const newVal   = rawVal === "" ? null : rawVal;

    if (String(original ?? "") === String(newVal ?? "")) return;

    setSaving(true);
    try {
      const res = await api.patch(`/api/sessions/${session.session_id}/cell`, {
        row_index: rowIdx,
        column: col,
        value: newVal,
      });
      updatePreviewCell(rowIdx, col, res.data.value);
    } catch {
      // On error silently revert
    } finally {
      setSaving(false);
    }
  };

  const downloadAs = useCallback((fmt: "csv" | "tsv" | "xlsx" | "sav") => {
    const base     = (session.filename ?? "data").replace(/\.[^.]+$/, "");
    const colKinds = encodeURIComponent(JSON.stringify(Object.fromEntries(columns.map((c) => [c.name, c.kind]))));
    const url      = `/api/sessions/${session.session_id}/export?fmt=${fmt}&filename=${encodeURIComponent(base)}&col_kinds=${colKinds}`;
    const a        = document.createElement("a");
    a.href         = url;
    a.download     = `${base}.${fmt}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setShowSaveMenu(false);
  }, [session.session_id, session.filename, columns]);

  const activeFilters = Object.values(filters).filter(Boolean).length;

  return (
    <div className="flex flex-col gap-2 h-full" style={{ minHeight: 0 }}>
      {showSelectCases && session && (
        <SelectCasesModal
          columns={columns}
          sessionId={session.session_id}
          existing={caseFilter?.conditions ?? []}
          onApply={(conditions, selected, total) => {
            setCaseFilter({ conditions, selected, total });
            setShowSelectCases(false);
          }}
          onClear={() => {
            setCaseFilter(null);
            setShowSelectCases(false);
          }}
          onClose={() => setShowSelectCases(false)}
        />
      )}

      {/* ── Toolbar ── */}
      <div className="flex items-center justify-between flex-shrink-0">
        <p className="text-sm text-gray-500">
          Showing{" "}
          <span className="text-gray-900 font-medium">{displayRows.length}</span>
          {displayRows.length !== preview.length && (
            <span className="text-gray-400"> of {preview.length} previewed</span>
          )}{" "}rows ·{" "}
          <span className="text-gray-900 font-medium">{session.rows.toLocaleString()}</span> total
          {" "}· {columns.length} columns
          {saving && <span className="ml-3 text-indigo-500 text-xs animate-pulse">saving…</span>}
        </p>

        <div className="flex items-center gap-2">
          {sortCol && (
            <button
              onClick={() => setSortCol(null)}
              className="text-xs text-orange-600 hover:text-orange-700 border border-orange-300 rounded-lg px-2.5 py-1 transition-colors bg-orange-50"
            >
              ✕ Sort: {sortCol} {sortDir === "asc" ? "▲" : "▼"}
            </button>
          )}
          {activeFilters > 0 && (
            <button
              onClick={() => setFilters({})}
              className="text-xs text-gray-600 hover:text-gray-800 border border-gray-300 rounded-lg px-2.5 py-1 transition-colors"
            >
              ✕ Clear {activeFilters} filter{activeFilters > 1 ? "s" : ""}
            </button>
          )}

          {/* ── Missing value button — always visible, fixed position before Filter ── */}
          <button
            onClick={() => totalMissingRows > 0 && setShowMissingOnly((v) => !v)}
            title={totalMissingRows > 0
              ? `${totalMissingRows} rows have missing values — click to show only those rows`
              : "No missing values in this dataset"}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors
              ${totalMissingRows === 0
                ? "text-gray-300 border-gray-200 cursor-default"
                : showMissingOnly
                  ? "bg-amber-100 text-amber-700 border-amber-400"
                  : "text-amber-600 border-amber-300 bg-amber-50 hover:bg-amber-100"}`}
          >
            ⚠ Missing
            {totalMissingRows > 0 && (
              <span className={`text-[9px] font-bold rounded-full px-1.5 py-0.5
                ${showMissingOnly ? "bg-amber-600 text-white" : "bg-amber-200 text-amber-800"}`}>
                {totalMissingRows}
              </span>
            )}
          </button>

          {/* Select Cases */}
          <button
            onClick={() => setShowSelectCases(true)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors
              ${caseFilter
                ? "bg-violet-100 text-violet-700 border-violet-400"
                : "text-gray-500 border-gray-300 hover:text-gray-700 hover:border-gray-400"}`}
          >
            ⊂ Cases
            {caseFilter && (
              <span className="bg-violet-600 text-white text-[9px] font-bold rounded-full px-1.5 py-0.5">
                {caseFilter.selected.toLocaleString()}
              </span>
            )}
          </button>

          <button
            onClick={() => setShowFilters((v) => !v)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors
              ${showFilters || activeFilters > 0
                ? "bg-indigo-50 text-indigo-600 border-indigo-300"
                : "text-gray-500 border-gray-300 hover:text-gray-700 hover:border-gray-400"}`}
          >
            ⟁ Filter
            {activeFilters > 0 && (
              <span className="bg-indigo-600 text-white text-[9px] font-bold rounded-full w-4 h-4 flex items-center justify-center">
                {activeFilters}
              </span>
            )}
          </button>

          {/* ── Save As ── */}
          <div className="relative" ref={saveMenuRef}>
            <button
              onClick={() => setShowSaveMenu((v) => !v)}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors
                ${showSaveMenu
                  ? "bg-emerald-50 text-emerald-700 border-emerald-300"
                  : "text-gray-500 border-gray-300 hover:text-gray-700 hover:border-gray-400"}`}
            >
              ↓ Save As
              <span className="text-gray-400 text-[10px]">{showSaveMenu ? "▲" : "▼"}</span>
            </button>

            {showSaveMenu && (
              <div className="absolute right-0 top-full mt-1 w-44 bg-white border border-gray-200 rounded-xl shadow-xl z-50 overflow-hidden">
                <p className="px-3 pt-2.5 pb-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
                  Export full dataset
                </p>
                {[
                  { fmt: "csv",  label: "CSV",          desc: "Comma-separated",  icon: "📄" },
                  { fmt: "xlsx", label: "Excel (.xlsx)", desc: "Microsoft Excel",  icon: "📊" },
                  { fmt: "sav",  label: "SPSS (.sav)",   desc: "Keeps col types",  icon: "🔬" },
                  { fmt: "tsv",  label: "TSV",           desc: "Tab-separated",    icon: "📋" },
                ].map(({ fmt, label, desc, icon }) => (
                  <button
                    key={fmt}
                    onClick={() => downloadAs(fmt as "csv" | "tsv" | "xlsx" | "sav")}
                    className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left hover:bg-gray-50 transition-colors"
                  >
                    <span className="text-base">{icon}</span>
                    <div>
                      <p className="text-xs text-gray-700 font-medium">{label}</p>
                      <p className="text-[10px] text-gray-400">{desc}</p>
                    </div>
                  </button>
                ))}
                <div className="border-t border-gray-100 px-3 py-2">
                  <p className="text-[10px] text-gray-400 leading-tight">
                    Exports the full dataset including all edits
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Table ── */}
      <div className="overflow-auto rounded-xl border border-gray-200 flex-1" style={{ minHeight: 0 }}>
        <table className="w-full text-sm border-collapse">
          <thead className="sticky top-0 z-10">

            {/* Column headers */}
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="px-3 py-2 text-left text-gray-400 text-xs font-normal w-10 border-r border-gray-200 select-none">
                #
              </th>
              {columns.map((col) => {
                const isSorted = sortCol === col.name;
                const nMissing = missingCounts[col.name] ?? 0;
                return (
                  <th key={col.name} className="px-2 py-2 border-r border-gray-200 min-w-[130px] max-w-[200px]">
                    <div className="flex items-center gap-1 justify-between">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <button
                          onClick={() => cycleKind(col.name)}
                          title={`Type: ${col.kind} — click to change`}
                          className={`text-[9px] font-bold px-1.5 py-0.5 rounded border flex-shrink-0 transition-colors ${KIND_STYLE[col.kind] ?? KIND_STYLE.text}`}
                        >
                          {KIND_LABEL[col.kind] ?? col.kind}
                        </button>
                        <span className="text-left text-gray-700 text-xs font-medium truncate">
                          {col.name}
                        </span>
                        {nMissing > 0 && (
                          <button
                            onClick={() => {
                              setShowMissingOnly(true);
                              setFilters((prev) => ({ ...prev, [col.name]: "" }));
                            }}
                            title={`${nMissing} missing values — click to filter`}
                            className="flex-shrink-0 text-[9px] font-semibold px-1 py-0.5 rounded bg-amber-100 text-amber-700 border border-amber-300 hover:bg-amber-200 transition-colors"
                          >
                            {nMissing}✕
                          </button>
                        )}
                      </div>
                      <button
                        onClick={() => toggleSort(col.name)}
                        title="Sort"
                        className={`flex-shrink-0 text-xs w-5 h-5 rounded flex items-center justify-center transition-colors
                          ${isSorted
                            ? "text-indigo-600 bg-indigo-100"
                            : "text-gray-300 hover:text-gray-500 hover:bg-gray-100"}`}
                      >
                        {isSorted ? (sortDir === "asc" ? "▲" : "▼") : "⇅"}
                      </button>
                    </div>
                  </th>
                );
              })}
            </tr>

            {/* Filter row */}
            {showFilters && (
              <tr className="bg-gray-50 border-b border-gray-200">
                <td className="border-r border-gray-200" />
                {columns.map((col) => (
                  <td key={col.name} className="px-1.5 py-1 border-r border-gray-200">
                    <input
                      className="w-full bg-white border border-gray-300 rounded px-2 py-0.5 text-xs text-gray-700
                        placeholder-gray-300 focus:outline-none focus:border-indigo-400"
                      placeholder="filter…"
                      value={filters[col.name] ?? ""}
                      onChange={(e) =>
                        setFilters((prev) => ({ ...prev, [col.name]: e.target.value }))
                      }
                    />
                  </td>
                ))}
              </tr>
            )}
          </thead>

          <tbody>
            {displayRows.map((row) => {
              const origIdx = row._idx as number;
              return (
                <tr
                  key={origIdx}
                  className="border-t border-gray-100 hover:bg-gray-50 transition-colors"
                >
                  <td className="px-3 py-1.5 text-gray-300 text-xs border-r border-gray-200 select-none text-right">
                    {origIdx + 1}
                  </td>

                  {columns.map((col) => {
                    const isEditing = editCell?.rowIdx === origIdx && editCell?.col === col.name;
                    const cellVal   = row[col.name];
                    const isNull    = cellVal === null || cellVal === undefined;

                    return (
                      <td
                        key={col.name}
                        onClick={() => !isEditing && startEdit(origIdx, col.name)}
                        className={`border-r border-gray-200 font-mono text-xs transition-colors
                          ${isEditing
                            ? "p-0 bg-indigo-50"
                            : isNull
                              ? "px-3 py-1.5 cursor-pointer bg-amber-50/60 hover:bg-amber-100/60"
                              : "px-3 py-1.5 cursor-pointer hover:bg-indigo-50/50"}`}
                      >
                        {isEditing ? (
                          <input
                            ref={inputRef}
                            className="w-full bg-white border border-indigo-400 rounded-sm px-3 py-1.5 text-xs text-gray-900 focus:outline-none"
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter")  commitEdit();
                              if (e.key === "Escape") setEditCell(null);
                            }}
                            onBlur={commitEdit}
                          />
                        ) : isNull ? (
                          <span className="text-amber-400 italic text-[10px] font-medium">null</span>
                        ) : (
                          <span className={col.kind === "numeric" ? "text-gray-700" : "text-gray-600"}>
                            {String(cellVal)}
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}

            {displayRows.length === 0 && (
              <tr>
                <td
                  colSpan={columns.length + 1}
                  className="px-6 py-16 text-center text-gray-400 text-sm"
                >
                  No rows match the current filters
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* ── Legend ── */}
      <div className="flex-shrink-0 flex items-center gap-4 text-[10px] text-gray-400 px-1">
        <span>Click a <span className="text-blue-600">type badge</span> to toggle num / cat / bool / txt</span>
        <span>·</span>
        <span>Click any <span className="text-gray-500">cell</span> to edit · Enter to save · Esc to cancel</span>
        <span>·</span>
        <span>Click <span className="text-gray-500">⇅</span> to sort</span>
      </div>
    </div>
  );
}
