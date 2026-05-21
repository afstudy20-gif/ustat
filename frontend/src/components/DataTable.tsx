import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { BookOpen, X } from "lucide-react";
import { useStore } from "../store";
import type { ColMeta, CaseCondition, CaseOperator } from "../store";
import api from "../api";
import { selectCases, clearCases, getUniqueValues, renameColumn, saveMetadata } from "../api";
import DataDictionaryPanel from "./DataDictionaryPanel";

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
  const undo             = useStore((s) => s.undo);
  const redo             = useStore((s) => s.redo);
  const undoLen          = useStore((s) => s.undoDepth);
  const redoLen          = useStore((s) => s.redoDepth);
  const columnDecimals   = useStore((s) => s.columnDecimals);
  const setColumnDecimals = useStore((s) => s.setColumnDecimals);

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
  const [showDictionary,  setShowDictionary]  = useState(false);

  // Drag & drop column reordering
  const [dragIdx,  setDragIdx]  = useState<number | null>(null);
  const [dropIdx,  setDropIdx]  = useState<number | null>(null);

  // Frozen (pinned-left) columns. `#` row-number column is always pinned.
  // `frozenCount` = number of leading data columns to freeze.
  const [frozenCount, setFrozenCount] = useState(0);
  const HASH_COL_W = 40;       // width of `#` column (matches w-10)
  const FROZEN_COL_W = 150;    // forced width per frozen data column
  const frozenLeft = (colIdx: number) => HASH_COL_W + colIdx * FROZEN_COL_W;
  const isFrozenCol = (colIdx: number) => colIdx < frozenCount;
  // Clamp frozenCount when columns are deleted
  useEffect(() => {
    const n = session?.columns.length ?? 0;
    setFrozenCount((c) => Math.min(c, n));
  }, [session?.columns.length]);

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

  // Value labels editor
  const [valueLabelCol, setValueLabelCol] = useState<string | null>(null);
  const [valueLabelDraft, setValueLabelDraft] = useState<Record<string, string>>({});

  // Multi-cell selection
  const [selectedCells, setSelectedCells] = useState<Set<string>>(new Set());
  const [selAnchor, setSelAnchor] = useState<{ row: number; col: string } | null>(null);

  // Right-click context menu (cells)
  const [cellCtx, setCellCtx] = useState<{ x: number; y: number; row: number; col: string } | null>(null);
  const cellCtxRef = useRef<HTMLDivElement>(null);

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

  // Paste notification
  const [pasteMsg, setPasteMsg] = useState<string | null>(null);

  // Ctrl+Z / Ctrl+Y / Ctrl+V / Delete / Backspace
  useEffect(() => {
    const handler = async (e: KeyboardEvent) => {
      // Delete / Backspace clears selected cells (no modifier needed)
      if ((e.key === "Delete" || e.key === "Backspace") && !editCell && !renameCol && selectedCells.size > 0) {
        e.preventDefault();
        clearSelectedCells();
        return;
      }
      // Escape clears selection
      if (e.key === "Escape" && selectedCells.size > 0 && !editCell) {
        setSelectedCells(new Set());
        return;
      }
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      // Don't capture when editing a cell or input
      if (editCell || renameCol) return;
      if (e.key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
      if (e.key === "z" && e.shiftKey)  { e.preventDefault(); redo(); }
      if (e.key === "y")                { e.preventDefault(); redo(); }
      // Ctrl+C — copy selected cells
      if (e.key === "c" && selectedCells.size > 0) {
        e.preventDefault();
        copyCells();
        return;
      }
      // Ctrl+V — paste from clipboard
      if (e.key === "v" && session) {
        e.preventDefault();
        try {
          const text = await navigator.clipboard.readText();
          if (!text.trim()) return;

          // If we have a cell selection anchor, paste cells at that position
          if (selAnchor) {
            await pasteCellsAt(selAnchor.row, selAnchor.col, text);
            setSelectedCells(new Set());
            setPasteMsg("Cells pasted");
            setTimeout(() => setPasteMsg(null), 3000);
            return;
          }

          // Otherwise append rows (old behavior)
          const res = await api.post(`/api/compute/${session.session_id}/paste`, {
            tsv: text, has_header: true, mode: "append",
          });
          const refresh = await api.get(`/api/stats/${session.session_id}/refresh`);
          useStore.getState().setSession({ ...session, ...refresh.data }); bumpUndo();
          setPasteMsg(`${res.data.n_pasted} rows pasted`);
          setTimeout(() => setPasteMsg(null), 3000);
        } catch (err: any) {
          setPasteMsg(err?.response?.data?.detail ?? "Paste failed");
          setTimeout(() => setPasteMsg(null), 4000);
        }
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [undo, redo, editCell, renameCol, session, selectedCells]);

  useEffect(() => {
    if (editCell) setTimeout(() => inputRef.current?.focus(), 0);
  }, [editCell]);

  useEffect(() => {
    setSortCol(null); setFilters({}); setShowMissingOnly(false); setSelectedCells(new Set()); setSelAnchor(null);
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
    if (!ctxMenu && !rowCtx && !cellCtx) return;
    const handler = (e: MouseEvent) => {
      if (ctxMenu && ctxRef.current && !ctxRef.current.contains(e.target as Node)) setCtxMenu(null);
      if (rowCtx && rowCtxRef.current && !rowCtxRef.current.contains(e.target as Node)) setRowCtx(null);
      if (cellCtx && cellCtxRef.current && !cellCtxRef.current.contains(e.target as Node)) setCellCtx(null);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [ctxMenu, rowCtx, cellCtx]);

  // Bump undo depth after each backend mutation
  const bumpUndo = () => useStore.setState((s) => ({ undoDepth: s.undoDepth + 1, redoDepth: 0 }));

  const deleteColumn = async (colName: string) => {
    if (!session) return;

    setCtxMenu(null);
    try {
      await api.delete(`/api/compute/${session.session_id}/column/${encodeURIComponent(colName)}`);
      const updatedCols = session.columns.filter((c) => c.name !== colName);
      const updatedPreview = session.preview.map((row) => {
        const r = { ...row }; delete r[colName]; return r;
      });
      useStore.getState().setSession({ ...session, columns: updatedCols, preview: updatedPreview }); bumpUndo();
    } catch { /* ignore */ }
  };

  const copyRow = (rowIdx: number) => {
    if (!session) return;
    setRowCtx(null);
    const row = preview[rowIdx];
    if (!row) return;
    const headers = columns.map((c) => c.name);
    const vals = headers.map((h) => String(row[h] ?? ""));
    const tsv = headers.join("\t") + "\n" + vals.join("\t");
    navigator.clipboard.writeText(tsv).catch(() => {});
  };

  const copyColumn = (colName: string) => {
    if (!session) return;
    setCtxMenu(null);
    const vals = preview.map((row) => String(row[colName] ?? ""));
    const tsv = colName + "\n" + vals.join("\n");
    navigator.clipboard.writeText(tsv).catch(() => {});
  };

  const addRow = async (position: number) => {
    if (!session) return;
    setRowCtx(null);
    try {
      await api.post(`/api/compute/${session.session_id}/add_row`, { position });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch { /* ignore */ }
  };

  const addColumn = async (position?: number) => {
    if (!session) return;
    const name = prompt("New column name:");
    if (!name?.trim()) return;
    try {
      await api.post(`/api/compute/${session.session_id}/add_column`, { name: name.trim(), position: position ?? -1 });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch (e: any) {
      alert(e?.response?.data?.detail ?? "Failed to add column");
    }
  };

  const deleteRow = async (rowIdx: number) => {
    if (!session) return;

    setRowCtx(null);
    try {
      await api.post(`/api/compute/${session.session_id}/delete_rows`, { row_indices: [rowIdx] });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch { /* ignore */ }
  };

  // ── Cell selection helpers ──────────────────────────────────────────────────
  const cellKey = (row: number, col: string) => `${row}:${col}`;

  const selectCell = (row: number, col: string, e: React.MouseEvent) => {
    if (e.shiftKey && selAnchor) {
      // Range selection from anchor to current (works with or without Ctrl)
      const colNames = columns.map((c) => c.name);
      const c1 = colNames.indexOf(selAnchor.col);
      const c2 = colNames.indexOf(col);
      const rMin = Math.min(selAnchor.row, row);
      const rMax = Math.max(selAnchor.row, row);
      const cMin = Math.min(c1, c2);
      const cMax = Math.max(c1, c2);
      const next = new Set<string>();
      for (let r = rMin; r <= rMax; r++) {
        for (let c = cMin; c <= cMax; c++) {
          next.add(cellKey(r, colNames[c]));
        }
      }
      setSelectedCells(next);
      // Don't move anchor — allows extending from same anchor
    } else if (e.ctrlKey || e.metaKey) {
      // Ctrl+click: toggle single cell and set anchor
      setSelectedCells((prev) => {
        const next = new Set(prev);
        const k = cellKey(row, col);
        if (next.has(k)) next.delete(k); else next.add(k);
        return next;
      });
      setSelAnchor({ row, col });
    }
  };

  const clearSelectedCells = async () => {
    if (!session || selectedCells.size === 0) return;
    const cells = Array.from(selectedCells).map((k) => {
      const [r, ...cParts] = k.split(":");
      return { row_index: Number(r), column: cParts.join(":") };
    });
    try {
      await api.post(`/api/sessions/${session.session_id}/clear_cells`, { cells });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
      setSelectedCells(new Set());
    } catch { /* ignore */ }
  };

  // ── Clipboard for cell copy/paste ──────────────────────────────────────────
  const [copiedCells, setCopiedCells] = useState<{ tsv: string; rows: number; cols: number } | null>(null);

  const copyCells = () => {
    if (!session || selectedCells.size === 0) return;
    const cells = Array.from(selectedCells).map((k) => {
      const [r, ...cParts] = k.split(":");
      return { row: Number(r), col: cParts.join(":") };
    });
    const rows = [...new Set(cells.map((c) => c.row))].sort((a, b) => a - b);
    const cols = [...new Set(cells.map((c) => c.col))];
    const colOrder = columns.map((c) => c.name);
    cols.sort((a, b) => colOrder.indexOf(a) - colOrder.indexOf(b));
    const tsv = rows.map((r) =>
      cols.map((c) => {
        const val = preview[r]?.[c];
        return val === null || val === undefined ? "" : String(val);
      }).join("\t")
    ).join("\n");
    setCopiedCells({ tsv, rows: rows.length, cols: cols.length });
    navigator.clipboard.writeText(tsv).catch(() => {});
  };

  const pasteCellsAt = async (startRow: number, startCol: string, tsv: string) => {
    if (!session) return;
    try {
      await api.post(`/api/compute/${session.session_id}/paste_cells`, {
        start_row: startRow, start_col: startCol, tsv,
      });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch { /* ignore */ }
  };

  const duplicateColumn = async (colName: string) => {
    if (!session) return;
    setCtxMenu(null);
    try {
      await api.post(`/api/compute/${session.session_id}/duplicate_column`, { column: colName });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch { /* ignore */ }
  };

  const sendToEnd = (colName: string) => {
    if (!session) return;

    setCtxMenu(null);
    const idx = session.columns.findIndex((c) => c.name === colName);
    if (idx < 0 || idx === session.columns.length - 1) return;
    reorderColumns(idx, session.columns.length - 1);
  };

  const fillBlanks = async (colName: string, fillValue: string) => {
    if (!session || !fillValue.trim()) return;

    setCtxMenu(null);
    try {
      await api.post(`/api/compute/${session.session_id}/fill_blanks`, {
        column: colName, value: fillValue.trim(),
      });
      // Refresh preview
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
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
      useStore.getState().setSession({ ...session, columns: updatedCols, preview: updatedPreview }); bumpUndo();
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
      bumpUndo();
    } catch {
      // On error silently revert
    } finally {
      setSaving(false);
    }
  };

  const downloadAs = useCallback(async (fmt: "csv" | "tsv" | "xlsx" | "sav") => {
    const base = (session.filename ?? "data").replace(/\.[^.]+$/, "");
    // Iframe-based downloads silently swallowed any backend error (500, 413,
    // 422) — the iframe loaded the error HTML in the hidden frame and the
    // user saw nothing happen. Switching to fetch + blob + anchor click
    // surfaces failures via the alert() catch path and avoids the X-Frame /
    // CSP edge cases that come with iframe loads of API responses.
    const colKinds = encodeURIComponent(JSON.stringify(Object.fromEntries(columns.map((c) => [c.name, c.kind]))));
    const url = `/api/sessions/${session.session_id}/export?fmt=${fmt}&filename=${encodeURIComponent(base)}&col_kinds=${colKinds}`;
    setShowSaveMenu(false);
    try {
      const res = await api.get(url, { responseType: "blob" });
      // Backend may return a JSON error body with the same 200 envelope on
      // some misconfigurations — sniff the content-type and bail if so.
      const ct = (res.headers["content-type"] || "").toString();
      if (ct.includes("application/json")) {
        const txt = await (res.data as Blob).text();
        throw new Error(`Server returned JSON instead of ${fmt.toUpperCase()}: ${txt.slice(0, 200)}`);
      }
      const ext = fmt;
      const mime = fmt === "xlsx" ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 : fmt === "sav"  ? "application/x-spss-sav"
                 : fmt === "tsv"  ? "text/tab-separated-values"
                 : "text/csv";
      const blob = new Blob([res.data], { type: mime });
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = `${base}.${ext}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (e: any) {
      // Try to extract the server's detail message from a Blob-typed error body.
      let detail = e?.message ?? String(e);
      const blobBody = e?.response?.data;
      if (blobBody instanceof Blob) {
        try {
          const txt = await blobBody.text();
          const parsed = JSON.parse(txt);
          detail = parsed?.detail ?? txt.slice(0, 200);
        } catch {
          // not JSON, leave detail as-is
        }
      }
      console.error(`Export as ${fmt} failed:`, e);
      alert(`Export as ${fmt.toUpperCase()} failed: ${detail}`);
    }
  }, [session.session_id, session.filename, columns]);

  const downloadSession = useCallback(async () => {
    try {
      const res = await api.get(`/api/sessions/${session.session_id}/save_session`, { responseType: "blob" });
      const blob = new Blob([res.data], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const base = (session.filename ?? "session").replace(/\.[^.]+$/, "");
      const a = document.createElement("a");
      a.href = url;
      a.download = `${base}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      console.error("Save session failed:", e);
      alert(`Save session failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setShowSaveMenu(false);
    }
  }, [session.session_id, session.filename]);

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
          {pasteMsg && <span className="ml-3 text-emerald-600 text-xs">{pasteMsg}</span>}
          {selectedCells.size > 1 && (
            <span className="ml-3 text-blue-600 text-xs font-medium">
              {selectedCells.size} cells selected
              <button onClick={() => setSelectedCells(new Set())} className="ml-1 text-blue-400 hover:text-blue-600">✕</button>
            </span>
          )}
          {copiedCells && (
            <span className="ml-2 text-green-600 text-xs">
              {copiedCells.rows}x{copiedCells.cols} copied
            </span>
          )}
        </p>

        <div className="flex items-center gap-2">
          {/* Dictionary modal opener — moved from the Compute combo so the
              variable-metadata view sits next to the data grid it describes. */}
          <button
            onClick={() => setShowDictionary(true)}
            title="Edit variable labels, value labels, and column metadata"
            className="text-xs px-2 py-1 rounded-lg border border-indigo-300 text-indigo-600 hover:bg-indigo-50 transition-colors flex items-center gap-1"
          >
            <BookOpen size={12} /> Dictionary
          </button>

          <div className="w-px h-5 bg-gray-200" />

          {/* Add Row / Add Column */}
          <button onClick={() => addRow(-1)}
            className="text-xs px-2 py-1 rounded-lg border border-emerald-300 text-emerald-600 hover:bg-emerald-50 transition-colors">
            + Row
          </button>
          <button onClick={() => addColumn()}
            className="text-xs px-2 py-1 rounded-lg border border-emerald-300 text-emerald-600 hover:bg-emerald-50 transition-colors">
            + Column
          </button>

          <div className="w-px h-5 bg-gray-200" />

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

          <div className="w-px h-5 bg-gray-200" />

          {/* Freeze (pin-left) columns */}
          <div className="flex items-center gap-0.5 text-xs">
            <span className="text-gray-500 mr-1" title="Freeze leading columns so they stay visible while scrolling right">❄ Freeze</span>
            <button
              onClick={() => setFrozenCount((n) => Math.max(0, n - 1))}
              disabled={frozenCount === 0}
              title="Freeze one fewer column"
              className={`w-6 h-6 rounded border transition-colors flex items-center justify-center ${frozenCount > 0 ? "text-gray-600 border-gray-300 hover:bg-gray-100" : "text-gray-300 border-gray-200 cursor-default"}`}
            >−</button>
            <span className="w-6 text-center font-medium text-gray-700">{frozenCount}</span>
            <button
              onClick={() => setFrozenCount((n) => Math.min(columns.length, n + 1))}
              disabled={frozenCount >= columns.length}
              title="Freeze one more column"
              className={`w-6 h-6 rounded border transition-colors flex items-center justify-center ${frozenCount < columns.length ? "text-gray-600 border-gray-300 hover:bg-gray-100" : "text-gray-300 border-gray-200 cursor-default"}`}
            >+</button>
            {frozenCount > 0 && (
              <button
                onClick={() => setFrozenCount(0)}
                title="Unfreeze all"
                className="ml-1 text-[10px] text-orange-600 hover:text-orange-700 border border-orange-300 rounded px-1.5 py-0.5"
              >✕</button>
            )}
          </div>

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
                  { fmt: "xlsx", label: "Excel (.xlsx)", desc: "With value labels sheet",  icon: "📊" },
                  { fmt: "sav",  label: "SPSS (.sav)",   desc: "Native value labels",  icon: "🔬" },
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
                <div className="border-t border-gray-100" />
                <p className="px-3 pt-2 pb-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
                  Session
                </p>
                <button
                  onClick={downloadSession}
                  className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left hover:bg-gray-50 transition-colors"
                >
                  <span className="text-base">💾</span>
                  <div>
                    <p className="text-xs text-gray-700 font-medium">Session (.json)</p>
                    <p className="text-[10px] text-gray-400">Data + labels + filters + audit</p>
                  </div>
                </button>
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
              <th
                className="px-3 py-2 text-left text-gray-400 text-xs font-normal border-r border-gray-200 select-none sticky left-0 bg-gray-50 z-20"
                style={{ width: HASH_COL_W, minWidth: HASH_COL_W, maxWidth: HASH_COL_W }}
              >
                #
              </th>
              {columns.map((col, colIdx) => {
                const isSorted = sortCol === col.name;
                const nMissing = missingCounts[col.name] ?? 0;
                const isDragOver = dropIdx === colIdx && dragIdx !== colIdx;
                const frozen = isFrozenCol(colIdx);
                const draggable = !frozen && renameCol !== col.name;
                return (
                  <th
                    key={col.name}
                    draggable={draggable}
                    onDragStart={(e) => {
                      if (!draggable) { e.preventDefault(); return; }
                      setDragIdx(colIdx);
                      e.dataTransfer.effectAllowed = "move";
                      e.dataTransfer.setData("text/plain", String(colIdx));
                    }}
                    onDragOver={(e) => {
                      // Block dropping unfrozen → frozen region and vice versa
                      if (dragIdx === null) return;
                      const srcFrozen = isFrozenCol(dragIdx);
                      if (srcFrozen !== frozen) return;
                      e.preventDefault();
                      e.dataTransfer.dropEffect = "move";
                      setDropIdx(colIdx);
                    }}
                    onDragLeave={() => { if (dropIdx === colIdx) setDropIdx(null); }}
                    onDrop={(e) => {
                      e.preventDefault();
                      if (dragIdx !== null && dragIdx !== colIdx && isFrozenCol(dragIdx) === frozen) {
                        reorderColumns(dragIdx, colIdx);
                      }
                      setDragIdx(null);
                      setDropIdx(null);
                    }}
                    onDragEnd={() => { setDragIdx(null); setDropIdx(null); }}
                    onContextMenu={(e) => { e.preventDefault(); setCtxMenu({ x: e.clientX, y: e.clientY, col: col.name }); }}
                    className={`px-2 py-2 border-r border-gray-200
                      ${frozen ? "sticky bg-gray-50 z-20" : "min-w-[130px] max-w-[200px]"}
                      ${renameCol === col.name || frozen ? "" : "cursor-grab active:cursor-grabbing select-none"}
                      ${dragIdx === colIdx ? "opacity-40" : ""}
                      ${isDragOver ? "border-l-2 border-l-indigo-500" : ""}`}
                    style={frozen ? { left: frozenLeft(colIdx), width: FROZEN_COL_W, minWidth: FROZEN_COL_W, maxWidth: FROZEN_COL_W } : undefined}
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
                <td
                  className="border-r border-gray-200 sticky left-0 bg-gray-50 z-20"
                  style={{ width: HASH_COL_W, minWidth: HASH_COL_W, maxWidth: HASH_COL_W }}
                />
                {columns.map((col, colIdx) => {
                  const frozen = isFrozenCol(colIdx);
                  return (
                    <td
                      key={col.name}
                      className={`px-1.5 py-1 border-r border-gray-200 ${frozen ? "sticky bg-gray-50 z-20" : ""}`}
                      style={frozen ? { left: frozenLeft(colIdx), width: FROZEN_COL_W, minWidth: FROZEN_COL_W, maxWidth: FROZEN_COL_W } : undefined}
                    >
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
                  );
                })}
              </tr>
            )}
          </thead>

          <tbody>
            {displayRows.map((row, visualIdx) => {
              const origIdx = row._idx as number;
              return (
                <tr
                  key={origIdx}
                  className="group border-t border-gray-100 hover:bg-gray-50 transition-colors"
                >
                  <td
                    className="px-3 py-1.5 text-gray-300 text-xs border-r border-gray-200 select-none text-right cursor-context-menu sticky left-0 bg-white group-hover:bg-gray-50 z-10"
                    style={{ width: HASH_COL_W, minWidth: HASH_COL_W, maxWidth: HASH_COL_W }}
                    onContextMenu={(e) => { e.preventDefault(); setRowCtx({ x: e.clientX, y: e.clientY, idx: origIdx }); }}
                    title={`Original row #${origIdx + 1} in the dataset`}
                  >
                    {visualIdx + 1}
                  </td>

                  {columns.map((col, colIdx) => {
                    const isEditing = editCell?.rowIdx === origIdx && editCell?.col === col.name;
                    const cellVal   = row[col.name];
                    const isNull    = cellVal === null || cellVal === undefined;
                    const isSel     = selectedCells.has(cellKey(origIdx, col.name));
                    const frozen    = isFrozenCol(colIdx);

                    return (
                      <td
                        key={col.name}
                        onClick={(e) => {
                          if (isEditing) return;
                          if (e.shiftKey || e.ctrlKey || e.metaKey) {
                            // Multi-select mode — don't open editor
                            selectCell(origIdx, col.name, e);
                          } else {
                            // Normal click: clear selection, open editor
                            setSelectedCells(new Set());
                            setSelAnchor(null);
                            startEdit(origIdx, col.name);
                          }
                        }}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          // If right-clicking an unselected cell, select just that cell
                          if (!selectedCells.has(cellKey(origIdx, col.name))) {
                            setSelectedCells(new Set([cellKey(origIdx, col.name)]));
                            setSelAnchor({ row: origIdx, col: col.name });
                          }
                          setCellCtx({ x: e.clientX, y: e.clientY, row: origIdx, col: col.name });
                        }}
                        className={`border-r border-gray-200 font-mono text-xs transition-colors
                          ${frozen ? "sticky z-10 group-hover:bg-gray-50" : ""}
                          ${isEditing
                            ? "p-0 bg-indigo-50"
                            : isSel
                              ? "px-3 py-1.5 cursor-pointer bg-blue-100 outline outline-1 outline-blue-400"
                              : isNull
                                ? "px-3 py-1.5 cursor-pointer bg-amber-50/60 hover:bg-amber-100/60"
                                : `px-3 py-1.5 cursor-pointer hover:bg-indigo-50/50 ${frozen ? "bg-white" : ""}`}`}
                        style={frozen ? { left: frozenLeft(colIdx), width: FROZEN_COL_W, minWidth: FROZEN_COL_W, maxWidth: FROZEN_COL_W } : undefined}
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
                            {col.name in columnDecimals && typeof cellVal === "number"
                              ? cellVal.toFixed(columnDecimals[col.name])
                              : String(cellVal)}
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
        <span>Click <span className="text-gray-500">cell</span> to edit · Ctrl+click to select · Shift+click for range · Delete to clear</span>
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
          <button onClick={() => copyColumn(ctxMenu.col)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📋 Copy column
          </button>
          <button onClick={() => { cycleKind(ctxMenu.col); setCtxMenu(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            🏷️ Change type
          </button>
          <button onClick={() => {
            const col = columns.find((c) => c.name === ctxMenu.col);
            setValueLabelDraft(col?.value_labels ? { ...col.value_labels } : {});
            setValueLabelCol(ctxMenu.col);
            setCtxMenu(null);
          }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            🔤 Value Labels
          </button>
          {/* Decimal places selector */}
          {columns.find((c) => c.name === ctxMenu.col)?.kind === "numeric" && (
            <div className="px-3 py-1 flex items-center gap-1.5">
              <span className="text-xs text-gray-500">🔢 Decimals:</span>
              {[0, 1, 2, 3, 4, "auto"].map((d) => (
                <button key={String(d)}
                  onClick={() => {
                    if (d === "auto") {
                      const next = { ...columnDecimals }; delete next[ctxMenu.col];
                      useStore.setState({ columnDecimals: next });
                    } else {
                      setColumnDecimals(ctxMenu.col, d as number);
                    }
                    setCtxMenu(null);
                  }}
                  className={`text-[10px] w-6 h-5 rounded flex items-center justify-center transition-colors ${
                    (d === "auto" && !(ctxMenu.col in columnDecimals)) || columnDecimals[ctxMenu.col] === d
                      ? "bg-indigo-600 text-white"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                  }`}>
                  {d === "auto" ? "A" : d}
                </button>
              ))}
            </div>
          )}
          <button onClick={() => { toggleSort(ctxMenu.col); setCtxMenu(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ⇅ Sort
          </button>
          <button onClick={() => sendToEnd(ctxMenu.col)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ➡️ Send to end
          </button>
          <button
            onClick={() => {
              const idx = columns.findIndex((c) => c.name === ctxMenu.col);
              if (idx >= 0) setFrozenCount(idx + 1);
              setCtxMenu(null);
            }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ❄ Freeze up to here
          </button>
          {frozenCount > 0 && (
            <button
              onClick={() => { setFrozenCount(0); setCtxMenu(null); }}
              className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
              ❄ Unfreeze all
            </button>
          )}
          <div className="border-t border-gray-100 mt-0.5" />
          <button onClick={() => { const idx = columns.findIndex((c) => c.name === ctxMenu.col); setCtxMenu(null); addColumn(idx); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ⬅️ Insert column left
          </button>
          <button onClick={() => { const idx = columns.findIndex((c) => c.name === ctxMenu.col); setCtxMenu(null); addColumn(idx + 1); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ➡️ Insert column right
          </button>
          <button onClick={() => duplicateColumn(ctxMenu.col)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📑 Duplicate column
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

      {/* ── Cell right-click context menu ── */}
      {cellCtx && (
        <div ref={cellCtxRef}
          className="fixed z-50 bg-white border border-gray-200 rounded-xl shadow-xl py-1 w-48"
          style={{ left: cellCtx.x, top: cellCtx.y }}>
          <div className="px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100 truncate">
            {selectedCells.size > 1
              ? `${selectedCells.size} cells selected`
              : `Row ${cellCtx.row + 1}, ${cellCtx.col}`}
          </div>
          <button onClick={() => { clearSelectedCells(); setCellCtx(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            🧹 Clear {selectedCells.size > 1 ? `${selectedCells.size} cells` : "cell"}
          </button>
          <button onClick={() => { copyCells(); setCellCtx(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📋 Copy {selectedCells.size > 1 ? `${selectedCells.size} cells` : "cell"}
          </button>
          <button onClick={async () => {
            setCellCtx(null);
            try {
              const text = await navigator.clipboard.readText();
              if (text.trim()) await pasteCellsAt(cellCtx.row, cellCtx.col, text);
            } catch { /* clipboard denied */ }
          }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📌 Paste here
          </button>
        </div>
      )}

      {/* ── Row right-click context menu ── */}
      {rowCtx && (
        <div ref={rowCtxRef}
          className="fixed z-50 bg-white border border-gray-200 rounded-xl shadow-xl py-1 w-44"
          style={{ left: rowCtx.x, top: rowCtx.y }}>
          <div className="px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100">Row {rowCtx.idx + 1}</div>
          <button onClick={() => copyRow(rowCtx.idx)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📋 Copy row
          </button>
          <div className="border-t border-gray-100 mt-0.5" />
          <button onClick={() => addRow(rowCtx.idx)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ⬆️ Insert row above
          </button>
          <button onClick={() => addRow(rowCtx.idx + 1)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ⬇️ Insert row below
          </button>
          <div className="border-t border-gray-100 mt-0.5" />
          <button onClick={() => deleteRow(rowCtx.idx)}
            className="w-full text-left px-3 py-1.5 text-xs text-red-500 hover:bg-red-50 flex items-center gap-2">
            🗑️ Delete row
          </button>
        </div>
      )}

      {/* ── Value Labels Modal ── */}
      {valueLabelCol && (() => {
        const col = columns.find((c) => c.name === valueLabelCol);
        // Get unique values from preview data
        const uniqueVals = Array.from(
          new Set(preview.map((r) => r[valueLabelCol]).filter((v) => v !== null && v !== undefined && v !== ""))
        ).map(String).sort((a, b) => {
          const na = Number(a), nb = Number(b);
          return (!isNaN(na) && !isNaN(nb)) ? na - nb : a.localeCompare(b);
        });

        const handleSaveLabels = async () => {
          // Save to store
          const updatedCols = session.columns.map((c) =>
            c.name === valueLabelCol ? { ...c, value_labels: { ...valueLabelDraft } } : c
          );
          useStore.getState().setSession({ ...session, columns: updatedCols });
          // Save to backend
          try {
            await saveMetadata(session.session_id, {
              [valueLabelCol]: { value_labels: valueLabelDraft },
            });
          } catch { /* ignore */ }
          setValueLabelCol(null);
        };

        return (
          <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center" onClick={() => setValueLabelCol(null)}>
            <div className="bg-white rounded-xl shadow-2xl w-96 max-h-[80vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
              {/* Header */}
              <div className="px-5 py-3.5 border-b border-gray-200 flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-gray-800">Value Labels</h3>
                  <p className="text-[11px] text-gray-400 mt-0.5">
                    {valueLabelCol}
                    {col?.kind && <span className="ml-1 text-indigo-500">({col.kind})</span>}
                  </p>
                </div>
                <button onClick={() => setValueLabelCol(null)} className="text-gray-400 hover:text-gray-600 text-lg">✕</button>
              </div>

              {/* Labels list */}
              <div className="flex-1 overflow-y-auto px-5 py-3 space-y-2">
                {uniqueVals.length === 0 ? (
                  <p className="text-xs text-gray-400 text-center py-4">No values found</p>
                ) : (
                  uniqueVals.map((val) => (
                    <div key={val} className="flex items-center gap-2">
                      <span className="w-14 text-xs font-mono text-gray-500 bg-gray-100 px-2 py-1 rounded text-center flex-shrink-0">
                        {val}
                      </span>
                      <span className="text-gray-400 text-xs">=</span>
                      <input
                        className="flex-1 text-xs border border-gray-300 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-200"
                        placeholder={`Label for ${val}`}
                        value={valueLabelDraft[val] ?? ""}
                        onChange={(e) => setValueLabelDraft((prev) => ({ ...prev, [val]: e.target.value }))}
                      />
                    </div>
                  ))
                )}
              </div>

              {/* Footer */}
              <div className="px-5 py-3 border-t border-gray-200 flex items-center justify-between">
                <button
                  onClick={() => { setValueLabelDraft({}); }}
                  className="text-xs text-gray-400 hover:text-red-500"
                >Clear all</button>
                <div className="flex gap-2">
                  <button onClick={() => setValueLabelCol(null)}
                    className="px-3 py-1.5 text-xs text-gray-500 border border-gray-200 rounded-lg hover:bg-gray-50">
                    Cancel
                  </button>
                  <button onClick={handleSaveLabels}
                    className="px-3 py-1.5 text-xs bg-indigo-600 text-white rounded-lg hover:bg-indigo-700">
                    Save Labels
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      {/* ── Dictionary modal ────────────────────────────────────────────── */}
      {showDictionary && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
          onClick={() => setShowDictionary(false)}
        >
          <div
            className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-gray-100">
              <div className="flex items-center gap-2">
                <BookOpen size={18} className="text-indigo-500" />
                <h2 className="font-semibold text-gray-800">Variable Dictionary</h2>
                <span className="text-xs text-gray-400">
                  Edit labels, value codings, and metadata for every column.
                </span>
              </div>
              <button
                onClick={() => setShowDictionary(false)}
                className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100"
                aria-label="Close dictionary"
              >
                <X size={18} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <DataDictionaryPanel />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
