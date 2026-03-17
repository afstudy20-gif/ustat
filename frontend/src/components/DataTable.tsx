import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { useStore } from "../store";
import type { ColMeta } from "../store";
import api from "../api";

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

// ── Main component ─────────────────────────────────────────────────────────────

type SortDir = "asc" | "desc";

export default function DataTable() {
  const session          = useStore((s) => s.session);
  const updateColumnKind = useStore((s) => s.updateColumnKind);
  const updatePreviewCell = useStore((s) => s.updatePreviewCell);

  const [sortCol,     setSortCol]     = useState<string | null>(null);
  const [sortDir,     setSortDir]     = useState<SortDir>("asc");
  const [filters,     setFilters]     = useState<Record<string, string>>({});
  const [showFilters, setShowFilters] = useState(false);
  const [editCell,     setEditCell]    = useState<{ rowIdx: number; col: string } | null>(null);
  const [editValue,    setEditValue]   = useState("");
  const [saving,       setSaving]      = useState(false);
  const [showSaveMenu, setShowSaveMenu] = useState(false);

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
    setSortCol(null); setFilters({});
  }, [session?.session_id]);

  if (!session) return null;
  const { preview, columns } = session;

  type IndexedRow = Record<string, unknown> & { _idx: number };
  const indexedRows = useMemo(
    () => preview.map((row, idx): IndexedRow => ({ ...row, _idx: idx })),
    [preview]
  );

  const filtered = useMemo(() => {
    const hasFilters = Object.values(filters).some(Boolean);
    if (!hasFilters) return indexedRows;
    return indexedRows.filter((row) =>
      columns.every((col) => {
        const f = filters[col.name];
        if (!f) return true;
        const cell = row[col.name];
        if (cell === null || cell === undefined) return f === "";
        return String(cell).toLowerCase().includes(f.toLowerCase());
      })
    );
  }, [indexedRows, filters, columns]);

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

  const downloadAs = useCallback((fmt: "csv" | "tsv" | "xlsx") => {
    const base = (session.filename ?? "data").replace(/\.[^.]+$/, "");
    const url  = `/api/sessions/${session.session_id}/export?fmt=${fmt}&filename=${encodeURIComponent(base)}`;
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `${base}.${fmt}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setShowSaveMenu(false);
  }, [session.session_id, session.filename]);

  const activeFilters = Object.values(filters).filter(Boolean).length;

  return (
    <div className="flex flex-col gap-2 h-full" style={{ minHeight: 0 }}>

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
                  { fmt: "tsv",  label: "TSV",           desc: "Tab-separated",    icon: "📋" },
                ].map(({ fmt, label, desc, icon }) => (
                  <button
                    key={fmt}
                    onClick={() => downloadAs(fmt as "csv" | "tsv" | "xlsx")}
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
                          <span className="text-gray-300 italic text-[10px]">null</span>
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
