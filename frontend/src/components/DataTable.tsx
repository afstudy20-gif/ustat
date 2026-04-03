import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { useStore } from "../store";
import type { ColMeta, CaseCondition, CaseOperator } from "../store";
import api from "../api";
import { selectCases, clearCases, getUniqueValues, renameColumn } from "../api";

// ── Kind cycling ───────────────────────────────────────────────────────────────

const KIND_CYCLE: ColMeta["kind"][] = ["numeric", "categorical", "text", "date"];

const KIND_STYLE: Record<string, string> = {
  numeric:     "bg-blue-100 text-blue-700 border-blue-300 hover:bg-blue-200",
  categorical: "bg-orange-100 text-orange-700 border-orange-300 hover:bg-orange-200",
  text:        "bg-gray-100 text-gray-500 border-gray-300 hover:bg-gray-200",
  date:        "bg-purple-100 text-purple-700 border-purple-300 hover:bg-purple-200",
};

const KIND_LABEL: Record<string, string> = {
  numeric: "num", categorical: "cat", text: "txt", date: "date",
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
  const [activeCond, setActiveCond] = useState(0);
  const [colValues, setColValues] = useState<Record<string, string[]>>({});
  const [valuesLoading, setValuesLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<{ selected: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [valSearch, setValSearch] = useState("");

  // Fetch unique values whenever active condition column changes
  const activeCol = conditions[activeCond]?.column ?? "";
  useEffect(() => {
    if (!activeCol || colValues[activeCol] !== undefined) return;
    setValuesLoading(true);
    getUniqueValues(sessionId, activeCol)
      .then((r) => {
        const vals: string[] = (r.data?.values ?? []).map(String);
        setColValues((prev) => ({ ...prev, [activeCol]: vals }));
      })
      .catch(() => setColValues((prev) => ({ ...prev, [activeCol]: [] })))
      .finally(() => setValuesLoading(false));
  }, [activeCol, sessionId]);

  const updateCond = (i: number, patch: Partial<CaseCondition>) => {
    setConditions((prev) => prev.map((c, idx) => idx === i ? { ...c, ...patch } : c));
    setPreview(null);
    setValSearch("");
    // If column changed, clear cached values so they reload
    if (patch.column) {
      setColValues((prev) => {
        const next = { ...prev };
        if (!(patch.column! in next)) return prev;
        return next;
      });
    }
  };

  const addCond = () => {
    setConditions((prev) => {
      const next = [...prev, defaultCond()];
      setActiveCond(next.length - 1);
      return next;
    });
    setValSearch("");
  };

  const removeCond = (i: number) => {
    setConditions((prev) => prev.filter((_, idx) => idx !== i));
    setActiveCond((prev) => Math.max(0, prev > i ? prev - 1 : prev));
  };

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

  const activeValues = colValues[activeCol] ?? [];
  const filteredValues = valSearch
    ? activeValues.filter((v) => v.toLowerCase().includes(valSearch.toLowerCase()))
    : activeValues;
  const activeOpMeta = OPERATORS.find((o) => o.value === conditions[activeCond]?.operator);
  const showValuePanel = !activeOpMeta?.noValue;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl flex flex-col" style={{ maxHeight: "90vh" }}>

        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-gray-100">
          <div>
            <h2 className="text-base font-semibold text-gray-900">Select Cases</h2>
            <p className="text-xs text-gray-400 mt-0.5">All analyses use only the selected subset</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100">✕</button>
        </div>

        {/* Body — two columns */}
        <div className="flex" style={{ minHeight: 0, overflow: "hidden" }}>

          {/* Left: condition builder */}
          <div className="flex-1 flex flex-col gap-3 px-6 py-4 overflow-y-auto border-r border-gray-100" style={{ minWidth: 0 }}>
            <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide">Conditions</p>

            {conditions.map((cond, i) => {
              const opMeta = OPERATORS.find((o) => o.value === cond.operator);
              const isActive = activeCond === i;
              return (
                <div
                  key={i}
                  onClick={() => { setActiveCond(i); setValSearch(""); }}
                  className={`flex flex-col gap-1.5 rounded-xl p-3 border cursor-pointer transition-colors
                    ${isActive ? "border-violet-300 bg-violet-50" : "border-gray-200 hover:border-gray-300 bg-white"}`}
                >
                  {/* Join label */}
                  {i > 0 && (
                    <select
                      value={cond.join}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => updateCond(i, { join: e.target.value as "AND" | "OR" })}
                      className="text-[10px] w-14 border border-gray-200 rounded px-1.5 py-0.5 bg-white text-gray-600 focus:outline-none"
                    >
                      <option value="AND">AND</option>
                      <option value="OR">OR</option>
                    </select>
                  )}
                  {i === 0 && <span className="text-[10px] text-gray-400 font-medium">WHERE</span>}

                  <div className="flex items-center gap-2">
                    {/* Column */}
                    <select
                      value={cond.column}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => { setActiveCond(i); updateCond(i, { column: e.target.value, value: "" }); }}
                      className="flex-1 text-xs border border-gray-300 rounded-lg px-2 py-1.5 bg-white text-gray-700 focus:outline-none focus:border-violet-400 min-w-0"
                    >
                      {columns.map((c) => (
                        <option key={c.name} value={c.name}>{c.name}</option>
                      ))}
                    </select>

                    {/* Operator */}
                    <select
                      value={cond.operator}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => updateCond(i, { operator: e.target.value as CaseOperator, value: "" })}
                      className="text-xs border border-gray-300 rounded-lg px-2 py-1.5 w-28 bg-white text-gray-700 focus:outline-none focus:border-violet-400 flex-shrink-0"
                    >
                      {OPERATORS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>

                    {/* Remove */}
                    <button
                      onClick={(e) => { e.stopPropagation(); removeCond(i); }}
                      disabled={conditions.length === 1}
                      className="flex-shrink-0 p-1 text-gray-300 hover:text-red-400 disabled:opacity-20 transition-colors"
                    >✕</button>
                  </div>

                  {/* Value display (read-only, click right panel to change) */}
                  {!opMeta?.noValue && (
                    <div className={`text-xs rounded-lg px-3 py-1.5 border min-h-[28px] flex items-center
                      ${cond.value
                        ? "border-violet-300 bg-white text-violet-700 font-medium"
                        : "border-dashed border-gray-300 text-gray-400 italic"}`}
                    >
                      {cond.value || (isActive ? "← click a value on the right" : "no value set")}
                    </div>
                  )}
                </div>
              );
            })}

            <button
              onClick={addCond}
              className="text-xs text-violet-600 hover:text-violet-800 self-start flex items-center gap-1 py-1"
            >
              + Add condition
            </button>
          </div>

          {/* Right: values panel */}
          <div style={{ width: 200, flexShrink: 0, display: "flex", flexDirection: "column", padding: "16px 12px", overflowY: "auto" }}>
            <p style={{ fontSize: 10, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
              Values — <span style={{ color: "#7c3aed", textTransform: "none", fontWeight: 500 }}>{activeCol}</span>
            </p>

            {showValuePanel ? (
              <>
                {/* Search */}
                <input
                  type="text"
                  placeholder="search values…"
                  value={valSearch}
                  onChange={(e) => setValSearch(e.target.value)}
                  style={{ width: "100%", fontSize: 11, border: "1px solid #e5e7eb", borderRadius: 8, padding: "5px 10px", marginBottom: 6, outline: "none", boxSizing: "border-box", color: "#111827" }}
                />

                {/* Value list */}
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {valuesLoading ? (
                    <p style={{ fontSize: 11, color: "#9ca3af", textAlign: "center", padding: "16px 0" }}>Loading…</p>
                  ) : filteredValues.length === 0 ? (
                    <p style={{ fontSize: 11, color: "#d1d5db", textAlign: "center", padding: "16px 0" }}>No values</p>
                  ) : (
                    filteredValues.map((val, vi) => {
                      const isSelected = conditions[activeCond]?.value === val;
                      return (
                        <button
                          key={`${val}-${vi}`}
                          onClick={() => updateCond(activeCond, { value: val })}
                          style={{
                            display: "block",
                            width: "100%",
                            textAlign: "left",
                            fontSize: 12,
                            padding: "8px 10px",
                            borderRadius: 6,
                            border: isSelected ? "1.5px solid #7c3aed" : "1px solid #d1d5db",
                            background: isSelected ? "#7c3aed" : "#ffffff",
                            color: isSelected ? "#ffffff" : "#1f2937",
                            fontWeight: isSelected ? 600 : 500,
                            cursor: "pointer",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            boxSizing: "border-box",
                            transition: "all 0.15s",
                          }}
                          title={val}
                        >
                          {val}
                        </button>
                      );
                    })
                  )}
                </div>

                <p style={{ fontSize: 10, color: "#d1d5db", textAlign: "center", marginTop: 6 }}>
                  {filteredValues.length} value{filteredValues.length !== 1 ? "s" : ""}
                </p>
              </>
            ) : (
              <p style={{ fontSize: 11, color: "#d1d5db", fontStyle: "italic", textAlign: "center", paddingTop: 32 }}>
                No value needed
              </p>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-100 flex flex-col gap-3">
          {/* Preview result */}
          {preview && (
            <div className="bg-violet-50 border border-violet-200 rounded-xl px-4 py-2.5 text-sm text-violet-800 flex items-baseline gap-2">
              <span className="font-bold text-xl">{preview.selected.toLocaleString()}</span>
              <span className="text-violet-500">of {preview.total.toLocaleString()} cases match</span>
              <span className="text-violet-400 text-xs ml-auto">
                {((preview.selected / preview.total) * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {error && <p className="text-xs text-red-500">{error}</p>}

          <div className="flex items-center gap-2">
            <button onClick={handlePreview} disabled={busy}
              className="px-4 py-2 text-sm border border-violet-300 text-violet-700 rounded-xl hover:bg-violet-50 transition-colors disabled:opacity-50">
              Preview
            </button>
            <button onClick={handleApply} disabled={busy}
              className="flex-1 px-4 py-2 text-sm bg-violet-600 text-white rounded-xl hover:bg-violet-700 transition-colors disabled:opacity-50 font-medium">
              {busy ? "Applying…" : "Apply"}
            </button>
            <button onClick={handleClear} disabled={busy}
              className="px-4 py-2 text-sm border border-gray-200 text-gray-500 rounded-xl hover:bg-gray-50 transition-colors disabled:opacity-50">
              Reset
            </button>
          </div>
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
  const reorderColumns   = useStore((s) => s.reorderColumns);
  const caseFilter       = useStore((s) => s.caseFilter);
  const setCaseFilter    = useStore((s) => s.setCaseFilter);
  const pushUndo         = useStore((s) => s.pushUndo);
  const undo             = useStore((s) => s.undo);
  const redo             = useStore((s) => s.redo);
  const undoLen          = useStore((s) => s.undoStack.length);
  const redoLen          = useStore((s) => s.redoStack.length);

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

  // Drag & drop column reordering
  const [dragIdx,  setDragIdx]  = useState<number | null>(null);
  const [dropIdx,  setDropIdx]  = useState<number | null>(null);

  // Column rename
  const [renameCol, setRenameCol] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState("");
  const renameRef = useRef<HTMLInputElement>(null);

  // Right-click context menu (columns)
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; col: string } | null>(null);
  const [fillMode, setFillMode] = useState<string | null>(null);
  const [fillVal, setFillVal] = useState("");
  const ctxRef = useRef<HTMLDivElement>(null);
  const fillRef = useRef<HTMLInputElement>(null);

  // Right-click context menu (rows)
  const [rowCtx, setRowCtx] = useState<{ x: number; y: number; idx: number } | null>(null);
  const rowCtxRef = useRef<HTMLDivElement>(null);

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

  // Ctrl+Z / Ctrl+Y (or Cmd+Z / Cmd+Shift+Z on Mac)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      // Don't capture when editing a cell or input
      if (editCell || renameCol) return;
      if (e.key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
      if (e.key === "z" && e.shiftKey)  { e.preventDefault(); redo(); }
      if (e.key === "y")                { e.preventDefault(); redo(); }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [undo, redo, editCell, renameCol]);

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

  useEffect(() => {
    if (renameCol) setTimeout(() => renameRef.current?.focus(), 0);
  }, [renameCol]);

  // Close context menus on outside click
  useEffect(() => {
    if (!ctxMenu && !rowCtx) return;
    const handler = (e: MouseEvent) => {
      if (ctxMenu && ctxRef.current && !ctxRef.current.contains(e.target as Node)) setCtxMenu(null);
      if (rowCtx && rowCtxRef.current && !rowCtxRef.current.contains(e.target as Node)) setRowCtx(null);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [ctxMenu, rowCtx]);

  const deleteColumn = async (colName: string) => {
    if (!session) return;
    pushUndo();
    setCtxMenu(null);
    try {
      await api.delete(`/api/compute/${session.session_id}/column/${encodeURIComponent(colName)}`);
      const updatedCols = session.columns.filter((c) => c.name !== colName);
      const updatedPreview = session.preview.map((row) => {
        const r = { ...row }; delete r[colName]; return r;
      });
      useStore.getState().setSession({ ...session, columns: updatedCols, preview: updatedPreview });
    } catch { /* ignore */ }
  };

  const deleteRow = async (rowIdx: number) => {
    if (!session) return;
    pushUndo();
    setRowCtx(null);
    try {
      await api.post(`/api/compute/${session.session_id}/delete_rows`, { row_indices: [rowIdx] });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data });
    } catch { /* ignore */ }
  };

  const sendToEnd = (colName: string) => {
    if (!session) return;
    pushUndo();
    setCtxMenu(null);
    const idx = session.columns.findIndex((c) => c.name === colName);
    if (idx < 0 || idx === session.columns.length - 1) return;
    reorderColumns(idx, session.columns.length - 1);
  };

  const fillBlanks = async (colName: string, fillValue: string) => {
    if (!session || !fillValue.trim()) return;
    pushUndo();
    setCtxMenu(null);
    try {
      await api.post(`/api/compute/${session.session_id}/fill_blanks`, {
        column: colName, value: fillValue.trim(),
      });
      // Refresh preview
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data });
    } catch { /* ignore */ }
  };

  const startRename = (colName: string) => {
    setRenameCol(colName);
    setRenameVal(colName);
  };

  const commitRename = async () => {
    if (!renameCol || !session) return;
    const oldName = renameCol;  // capture before clearing state
    const newName = renameVal.trim();
    setRenameCol(null);
    if (!newName || newName === oldName) return;
    pushUndo();
    try {
      await renameColumn(session.session_id, oldName, newName);
      // Update local state
      const updatedCols = session.columns.map((c) =>
        c.name === oldName ? { ...c, name: newName } : c
      );
      const updatedPreview = session.preview.map((row) => {
        const r = { ...row };
        if (oldName in r) { r[newName] = r[oldName]; delete r[oldName]; }
        return r;
      });
      useStore.getState().setSession({ ...session, columns: updatedCols, preview: updatedPreview });
    } catch { /* revert silently */ }
  };

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
    pushUndo();
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

    // Use window.location for direct download — backend returns Content-Disposition: attachment
    // This won't navigate away because the browser intercepts attachment responses
    window.location.assign(url);
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
          {/* Undo / Redo */}
          <button onClick={undo} disabled={undoLen === 0}
            title="Undo (Ctrl+Z)"
            className={`text-xs px-2 py-1 rounded-lg border transition-colors ${undoLen > 0 ? "text-gray-600 border-gray-300 hover:bg-gray-100" : "text-gray-300 border-gray-200 cursor-default"}`}>
            ↩ Undo
          </button>
          <button onClick={redo} disabled={redoLen === 0}
            title="Redo (Ctrl+Y)"
            className={`text-xs px-2 py-1 rounded-lg border transition-colors ${redoLen > 0 ? "text-gray-600 border-gray-300 hover:bg-gray-100" : "text-gray-300 border-gray-200 cursor-default"}`}>
            ↪ Redo
          </button>

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
              {columns.map((col, colIdx) => {
                const isSorted = sortCol === col.name;
                const nMissing = missingCounts[col.name] ?? 0;
                const isDragOver = dropIdx === colIdx && dragIdx !== colIdx;
                return (
                  <th
                    key={col.name}
                    draggable={renameCol !== col.name}
                    onDragStart={(e) => {
                      setDragIdx(colIdx);
                      e.dataTransfer.effectAllowed = "move";
                      e.dataTransfer.setData("text/plain", String(colIdx));
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      e.dataTransfer.dropEffect = "move";
                      setDropIdx(colIdx);
                    }}
                    onDragLeave={() => { if (dropIdx === colIdx) setDropIdx(null); }}
                    onDrop={(e) => {
                      e.preventDefault();
                      if (dragIdx !== null && dragIdx !== colIdx) { pushUndo(); reorderColumns(dragIdx, colIdx); }
                      setDragIdx(null);
                      setDropIdx(null);
                    }}
                    onDragEnd={() => { setDragIdx(null); setDropIdx(null); }}
                    onContextMenu={(e) => { e.preventDefault(); setCtxMenu({ x: e.clientX, y: e.clientY, col: col.name }); }}
                    className={`px-2 py-2 border-r border-gray-200 min-w-[130px] max-w-[200px]
                      ${renameCol === col.name ? "" : "cursor-grab active:cursor-grabbing select-none"}
                      ${dragIdx === colIdx ? "opacity-40" : ""}
                      ${isDragOver ? "border-l-2 border-l-indigo-500" : ""}`}
                  >
                    <div className="flex items-center gap-1 justify-between">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="text-gray-300 text-[8px] flex-shrink-0 cursor-grab" title="Drag to reorder">⠿</span>
                        <button
                          onClick={() => cycleKind(col.name)}
                          title={`Type: ${col.kind} — click to change`}
                          className={`text-[9px] font-bold px-1.5 py-0.5 rounded border flex-shrink-0 transition-colors ${KIND_STYLE[col.kind] ?? KIND_STYLE.text}`}
                        >
                          {KIND_LABEL[col.kind] ?? col.kind}
                        </button>
                        {renameCol === col.name ? (
                          <input ref={renameRef}
                            className="text-xs font-medium text-gray-900 bg-white border border-indigo-400 rounded px-1 py-0 w-24 focus:outline-none select-text"
                            value={renameVal}
                            onClick={(e) => e.stopPropagation()}
                            onMouseDown={(e) => e.stopPropagation()}
                            onChange={(e) => setRenameVal(e.target.value)}
                            onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Enter") commitRename(); if (e.key === "Escape") setRenameCol(null); }}
                            onBlur={commitRename}
                          />
                        ) : (
                          <span className="text-left text-gray-700 text-xs font-medium truncate cursor-text"
                            onDoubleClick={() => startRename(col.name)}
                            title="Double-click to rename">
                            {col.name}
                          </span>
                        )}
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
                  <td
                    className="px-3 py-1.5 text-gray-300 text-xs border-r border-gray-200 select-none text-right cursor-context-menu"
                    onContextMenu={(e) => { e.preventDefault(); setRowCtx({ x: e.clientX, y: e.clientY, idx: origIdx }); }}
                  >
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
        <span>Click a <span className="text-blue-600">type badge</span> to toggle num / cat / txt / date</span>
        <span>·</span>
        <span>Double-click <span className="text-gray-500">header</span> to rename · Right-click to delete</span>
        <span>·</span>
        <span>Click <span className="text-gray-500">cell</span> to edit · <span className="text-gray-500">⇅</span> to sort</span>
      </div>

      {/* ── Right-click context menu ── */}
      {ctxMenu && (
        <div ref={ctxRef}
          className="fixed z-50 bg-white border border-gray-200 rounded-xl shadow-xl py-1 w-48"
          style={{ left: ctxMenu.x, top: ctxMenu.y }}>
          <div className="px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100 truncate">
            {ctxMenu.col}
            {(missingCounts[ctxMenu.col] ?? 0) > 0 && (
              <span className="ml-1 text-amber-500">({missingCounts[ctxMenu.col]} missing)</span>
            )}
          </div>
          <button onClick={() => { startRename(ctxMenu.col); setCtxMenu(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ✏️ Rename
          </button>
          <button onClick={() => { cycleKind(ctxMenu.col); setCtxMenu(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            🏷️ Change type
          </button>
          <button onClick={() => { toggleSort(ctxMenu.col); setCtxMenu(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ⇅ Sort
          </button>
          <button onClick={() => sendToEnd(ctxMenu.col)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ➡️ Send to end
          </button>
          {(missingCounts[ctxMenu.col] ?? 0) > 0 && (
            <>
              <div className="border-t border-gray-100 mt-0.5" />
              <div className="px-3 py-1 text-[10px] text-amber-600 font-medium">Fill {missingCounts[ctxMenu.col]} blanks with:</div>
              <button onClick={() => { fillBlanks(ctxMenu.col, "__mean__"); }}
                className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                📊 Mean
              </button>
              <button onClick={() => { fillBlanks(ctxMenu.col, "__median__"); }}
                className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                📊 Median
              </button>
              <button onClick={() => { fillBlanks(ctxMenu.col, "0"); }}
                className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                0️⃣ Zero
              </button>
              <button onClick={() => { fillBlanks(ctxMenu.col, "__mice__"); }}
                className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                🧬 MICE (multiple imputation)
              </button>
              {fillMode === ctxMenu.col ? (
                <div className="px-3 py-1 flex items-center gap-1">
                  <input ref={fillRef} autoFocus
                    className="text-xs border border-gray-300 rounded px-1.5 py-0.5 w-20 focus:outline-none focus:border-indigo-400"
                    placeholder="value"
                    value={fillVal}
                    onChange={(e) => setFillVal(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") { fillBlanks(ctxMenu.col, fillVal); setFillMode(null); setFillVal(""); }
                      if (e.key === "Escape") { setFillMode(null); setFillVal(""); }
                    }}
                  />
                  <button onClick={() => { fillBlanks(ctxMenu.col, fillVal); setFillMode(null); setFillVal(""); }}
                    className="text-[10px] px-1.5 py-0.5 bg-indigo-600 text-white rounded hover:bg-indigo-700">Fill</button>
                </div>
              ) : (
                <button onClick={() => { setFillMode(ctxMenu.col); setFillVal(""); }}
                  className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                  ✏️ Custom value...
                </button>
              )}
            </>
          )}
          <div className="border-t border-gray-100 mt-0.5" />
          <button onClick={() => deleteColumn(ctxMenu.col)}
            className="w-full text-left px-3 py-1.5 text-xs text-red-500 hover:bg-red-50 flex items-center gap-2">
            🗑️ Delete column
          </button>
        </div>
      )}

      {/* ── Row right-click context menu ── */}
      {rowCtx && (
        <div ref={rowCtxRef}
          className="fixed z-50 bg-white border border-gray-200 rounded-xl shadow-xl py-1 w-44"
          style={{ left: rowCtx.x, top: rowCtx.y }}>
          <div className="px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100">Row {rowCtx.idx + 1}</div>
          <button onClick={() => deleteRow(rowCtx.idx)}
            className="w-full text-left px-3 py-1.5 text-xs text-red-500 hover:bg-red-50 flex items-center gap-2">
            🗑️ Delete row
          </button>
        </div>
      )}
    </div>
  );
}
