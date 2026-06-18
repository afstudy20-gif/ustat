import { useState, useMemo, useRef, useEffect, useLayoutEffect } from "react";
import type { CSSProperties, RefObject } from "react";
import { BookOpen, X } from "lucide-react";
import { useStore } from "../store";
import type { ColMeta, Session } from "../store";
import api from "../api";
import { renameColumn } from "../api";
import DataDictionaryPanel from "./DataDictionaryPanel";

// ── Kind cycling ───────────────────────────────────────────────────────────────

const KIND_CYCLE: ColMeta["kind"][] = ["numeric", "categorical", "ordinal", "text", "date"];

const KIND_STYLE: Record<string, string> = {
  numeric:     "bg-blue-100 text-blue-700 border-blue-300 hover:bg-blue-200",
  categorical: "bg-orange-100 text-orange-700 border-orange-300 hover:bg-orange-200",
  ordinal:     "bg-teal-100 text-teal-700 border-teal-300 hover:bg-teal-200",
  text:        "bg-gray-100 text-gray-500 border-gray-300 hover:bg-gray-200",
  date:        "bg-purple-100 text-purple-700 border-purple-300 hover:bg-purple-200",
};

const KIND_LABEL: Record<string, string> = {
  numeric: "num", categorical: "cat", ordinal: "ord", text: "txt", date: "date",
};

import { SelectCasesModal } from "./datatable/SelectCasesModal";
import { ValueLabelsModal } from "./datatable/ValueLabelsModal";
import { FindReplaceModal } from "./datatable/FindReplaceModal";
import { ParseDatesModal } from "./datatable/ParseDatesModal";
type SortDir = "asc" | "desc";

type ContextMenuAnchor = { x: number; y: number };

const CONTEXT_MENU_MARGIN = 8;

function useViewportContextMenuStyle(
  anchor: ContextMenuAnchor | null,
  menuRef: RefObject<HTMLDivElement | null>,
  fallbackWidth: number,
): CSSProperties {
  const [position, setPosition] = useState({
    left: CONTEXT_MENU_MARGIN,
    top: CONTEXT_MENU_MARGIN,
    // Start tall, not 0 — a 0 here (or a transient innerHeight=0 in PWA/iframe
    // contexts) collapses the menu to ~nothing, so it looks like it closes the
    // instant you try to click an item.
    maxHeight: 9999,
  });

  useLayoutEffect(() => {
    if (!anchor) return;

    const updatePosition = () => {
      const menu = menuRef.current;
      const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1024;
      // innerHeight can briefly report 0 in embedded/PWA/iframe contexts; fall
      // back to the document height and never clamp below a usable minimum.
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 768;
      const maxHeight = Math.max(200, viewportHeight - CONTEXT_MENU_MARGIN * 2);
      const menuWidth = menu?.offsetWidth || fallbackWidth;
      const menuHeight = Math.min(
        menu?.scrollHeight || menu?.offsetHeight || maxHeight,
        maxHeight,
      );

      const preferredLeft =
        anchor.x + menuWidth <= viewportWidth - CONTEXT_MENU_MARGIN
          ? anchor.x
          : anchor.x - menuWidth;
      const left = Math.max(
        CONTEXT_MENU_MARGIN,
        Math.min(preferredLeft, viewportWidth - menuWidth - CONTEXT_MENU_MARGIN),
      );

      const spaceBelow = viewportHeight - CONTEXT_MENU_MARGIN - anchor.y;
      const spaceAbove = anchor.y - CONTEXT_MENU_MARGIN;
      let top: number;
      if (menuHeight <= spaceBelow) {
        top = anchor.y;
      } else if (menuHeight <= spaceAbove) {
        top = anchor.y - menuHeight;
      } else {
        top = CONTEXT_MENU_MARGIN;
      }

      setPosition((current) =>
        current.left === left &&
        current.top === top &&
        current.maxHeight === maxHeight
          ? current
          : { left, top, maxHeight },
      );
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);

    const resizeObserver = new ResizeObserver(updatePosition);
    const mutationObserver = new MutationObserver(updatePosition);
    if (menuRef.current) {
      resizeObserver.observe(menuRef.current);
      mutationObserver.observe(menuRef.current, { childList: true, subtree: true });
    }

    return () => {
      window.removeEventListener("resize", updatePosition);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
    };
  }, [anchor, fallbackWidth, menuRef]);

  return {
    ...position,
    maxWidth: `calc(100vw - ${CONTEXT_MENU_MARGIN * 2}px)`,
    overflowY: "auto",
    overscrollBehavior: "contain",
    scrollbarGutter: "stable",
  };
}

export default function DataTable() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <DataTableBody session={session} />;
}

function DataTableBody({ session }: { session: Session }) {
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
  const clearColumnDecimals = useStore((s) => s.clearColumnDecimals);

  const [sortCol,     setSortCol]     = useState<string | null>(null);
  const [sortDir,     setSortDir]     = useState<SortDir>("asc");
  const [filters,     setFilters]     = useState<Record<string, string>>({});
  const [showFilters, setShowFilters] = useState(false);
  const [editCell,       setEditCell]      = useState<{ rowIdx: number; col: string } | null>(null);
  const [editValue,      setEditValue]     = useState("");
  const [saving,         setSaving]        = useState(false);
  const [showMissingOnly, setShowMissingOnly] = useState(false);
  const [showSelectCases, setShowSelectCases] = useState(false);
  const [showDictionary,  setShowDictionary]  = useState(false);

  // Drag & drop column reordering
  const [dragIdx,  setDragIdx]  = useState<number | null>(null);
  const [dropIdx,  setDropIdx]  = useState<number | null>(null);

  // Frozen (pinned-left) columns. `#` row-number column is always pinned.
  // `frozenCount` = number of leading data columns to freeze.
  const [frozenCount, setFrozenCount] = useState(0);
  const HASH_COL_W = 30;       // width of `#` (row-number) column — kept narrow
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
  const [findReplaceCol, setFindReplaceCol] = useState<string | null>(null);
  const [parseDateCol, setParseDateCol] = useState<string | null>(null);
  const [valueLabelDraft, setValueLabelDraft] = useState<Record<string, string>>({});

  // Analysis-exclude flag + move-to-position + name-suggestion modals
  const setColumnAnalysisExcluded = useStore((s) => s.setColumnAnalysisExcluded);
  const [moveCol, setMoveCol] = useState<string | null>(null);
  const [suggestOpen, setSuggestOpen] = useState(false);
  const [suggestDraft, setSuggestDraft] = useState<Record<string, string>>({});  // col → target name (editable)
  const [suggestAccept, setSuggestAccept] = useState<Record<string, boolean>>({});
  const [suggestBusy, setSuggestBusy] = useState(false);

  // Multi-cell selection
  const [selectedCells, setSelectedCells] = useState<Set<string>>(new Set());
  const [selAnchor, setSelAnchor] = useState<{ row: number; col: string } | null>(null);
  const [selFocus, setSelFocus] = useState<{ row: number; col: string } | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const dragSelectingRef = useRef(false);
  const dragAnchorRef = useRef<{ row: number; col: string } | null>(null);
  const dragAdditiveRef = useRef(false);
  const dragBaseSelectionRef = useRef<Set<string>>(new Set());

  // Right-click context menu (cells)
  const [cellCtx, setCellCtx] = useState<{ x: number; y: number; row: number; col: string } | null>(null);
  const cellCtxRef = useRef<HTMLDivElement>(null);

  // Right-click context menu (rows)
  const [rowCtx, setRowCtx] = useState<{ x: number; y: number; idx: number } | null>(null);
  const rowCtxRef = useRef<HTMLDivElement>(null);
  const columnMenuStyle = useViewportContextMenuStyle(ctxMenu, ctxRef, 192);
  const cellMenuStyle = useViewportContextMenuStyle(cellCtx, cellCtxRef, 192);
  const rowMenuStyle = useViewportContextMenuStyle(rowCtx, rowCtxRef, 176);

  const inputRef   = useRef<HTMLInputElement>(null);
  const committingCellsRef = useRef<Set<string>>(new Set());

  // Paste notification
  const [pasteMsg, setPasteMsg] = useState<string | null>(null);

  useEffect(() => {
    if (editCell) setTimeout(() => inputRef.current?.focus(), 0);
  }, [editCell]);

  useEffect(() => {
    setSortCol(null); setFilters({}); setShowMissingOnly(false); setSelectedCells(new Set());
    setSelAnchor(null); setSelFocus(null);
  }, [session?.session_id]);

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
  const bumpUndo = () => useStore.setState((s) => ({ undoDepth: s.undoDepth + 1, redoDepth: 0, dataVersion: s.dataVersion + 1 }));

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
    const trimmed = name?.trim();
    if (!trimmed) return;
    // Client-side duplicate guard so the user sees the conflict before the
    // round-trip; backend also validates as a safety net.
    if (session.columns.some((c) => c.name === trimmed)) {
      alert(`Column "${trimmed}" already exists. Pick a different name.`);
      return;
    }
    try {
      await api.post(`/api/compute/${session.session_id}/add_column`, { name: trimmed, position: position ?? -1 });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch (e: unknown) {
      alert((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Failed to add column");
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

  type CellPosition = { row: number; col: string };

  const visibleRowIds = () => displayRows.map((row) => row._idx as number);

  const rangeKeys = (anchor: CellPosition, focus: CellPosition): Set<string> => {
    const rowIds = visibleRowIds();
    const colNames = columns.map((c) => c.name);
    const r1 = rowIds.indexOf(anchor.row);
    const r2 = rowIds.indexOf(focus.row);
    const c1 = colNames.indexOf(anchor.col);
    const c2 = colNames.indexOf(focus.col);
    const next = new Set<string>();
    if (r1 < 0 || r2 < 0 || c1 < 0 || c2 < 0) return next;
    for (let r = Math.min(r1, r2); r <= Math.max(r1, r2); r++) {
      for (let c = Math.min(c1, c2); c <= Math.max(c1, c2); c++) {
        next.add(cellKey(rowIds[r], colNames[c]));
      }
    }
    return next;
  };

  const selectRange = (anchor: CellPosition, focus: CellPosition, additive = false) => {
    const range = rangeKeys(anchor, focus);
    setSelectedCells((prev) => additive ? new Set([...prev, ...range]) : range);
    setSelAnchor(anchor);
    setSelFocus(focus);
  };

  const selectSingleCell = (row: number, col: string) => {
    const pos = { row, col };
    setSelectedCells(new Set([cellKey(row, col)]));
    setSelAnchor(pos);
    setSelFocus(pos);
  };

  const beginCellSelection = (row: number, col: string, e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    gridRef.current?.focus({ preventScroll: true });
    const mod = e.ctrlKey || e.metaKey;
    const pos = { row, col };
    dragBaseSelectionRef.current = new Set(selectedCells);

    if (e.shiftKey) {
      const anchor = selAnchor ?? pos;
      selectRange(anchor, pos, mod && selAnchor !== null);
      dragAnchorRef.current = anchor;
      dragAdditiveRef.current = mod && selAnchor !== null;
    } else if (mod) {
      setSelectedCells((prev) => {
        const next = new Set(prev);
        const key = cellKey(row, col);
        if (next.has(key)) next.delete(key); else next.add(key);
        return next;
      });
      setSelAnchor(pos);
      setSelFocus(pos);
      dragAnchorRef.current = null;
      dragSelectingRef.current = false;
      return;
    } else {
      selectSingleCell(row, col);
      dragAnchorRef.current = pos;
      dragAdditiveRef.current = false;
    }
    dragSelectingRef.current = true;
  };

  const extendMouseSelection = (row: number, col: string, e: React.MouseEvent) => {
    if (!dragSelectingRef.current || e.buttons !== 1 || !dragAnchorRef.current) {
      if (e.buttons !== 1) dragSelectingRef.current = false;
      return;
    }
    const focus = { row, col };
    const range = rangeKeys(dragAnchorRef.current, focus);
    setSelectedCells(
      dragAdditiveRef.current
        ? new Set([...dragBaseSelectionRef.current, ...range])
        : range
    );
    setSelFocus(focus);
  };

  const selectVisibleRow = (row: number, additive: boolean) => {
    const keys = columns.map((c) => cellKey(row, c.name));
    setSelectedCells((prev) => additive ? new Set([...prev, ...keys]) : new Set(keys));
    const anchor = { row, col: columns[0]?.name ?? "" };
    const focus = { row, col: columns[columns.length - 1]?.name ?? "" };
    setSelAnchor(anchor);
    setSelFocus(focus);
    gridRef.current?.focus({ preventScroll: true });
  };

  const selectVisibleColumn = (col: string, additive: boolean) => {
    const rows = visibleRowIds();
    const keys = rows.map((row) => cellKey(row, col));
    setSelectedCells((prev) => additive ? new Set([...prev, ...keys]) : new Set(keys));
    if (rows.length > 0) {
      setSelAnchor({ row: rows[0], col });
      setSelFocus({ row: rows[rows.length - 1], col });
    }
    gridRef.current?.focus({ preventScroll: true });
  };

  const moveSelectionFocus = (rowDelta: number, colDelta: number, extend: boolean, toEdge: boolean) => {
    if (!selFocus || columns.length === 0 || displayRows.length === 0) return;
    const rows = visibleRowIds();
    const currentRow = Math.max(0, rows.indexOf(selFocus.row));
    const currentCol = Math.max(0, columns.findIndex((c) => c.name === selFocus.col));
    const nextRow = toEdge
      ? (rowDelta < 0 ? 0 : rowDelta > 0 ? rows.length - 1 : currentRow)
      : Math.max(0, Math.min(rows.length - 1, currentRow + rowDelta));
    const nextCol = toEdge
      ? (colDelta < 0 ? 0 : colDelta > 0 ? columns.length - 1 : currentCol)
      : Math.max(0, Math.min(columns.length - 1, currentCol + colDelta));
    const next = { row: rows[nextRow], col: columns[nextCol].name };
    if (extend) selectRange(selAnchor ?? selFocus, next);
    else selectSingleCell(next.row, next.col);
    requestAnimationFrame(() => {
      gridRef.current
        ?.querySelector<HTMLElement>(`[data-grid-row="${next.row}"][data-grid-col="${nextCol}"]`)
        ?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
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
      setSelAnchor(null);
      setSelFocus(null);
    } catch { /* ignore */ }
  };

  // ── Clipboard for cell copy/paste ──────────────────────────────────────────
  const [copiedCells, setCopiedCells] = useState<{ tsv: string; rows: number; cols: number } | null>(null);

  const selectedCellsTsv = (): { tsv: string; rows: number; cols: number } | null => {
    if (!session || selectedCells.size === 0) return null;
    const cells = Array.from(selectedCells).map((k) => {
      const [r, ...cParts] = k.split(":");
      return { row: Number(r), col: cParts.join(":") };
    });
    const visibleOrder = visibleRowIds();
    const rows = [...new Set(cells.map((c) => c.row))].sort(
      (a, b) => visibleOrder.indexOf(a) - visibleOrder.indexOf(b)
    );
    const cols = [...new Set(cells.map((c) => c.col))];
    const colOrder = columns.map((c) => c.name);
    cols.sort((a, b) => colOrder.indexOf(a) - colOrder.indexOf(b));
    const tsv = rows.map((r) =>
      cols.map((c) => {
        if (!selectedCells.has(cellKey(r, c))) return "";
        const val = preview[r]?.[c];
        return val === null || val === undefined ? "" : String(val);
      }).join("\t")
    ).join("\n");
    return { tsv, rows: rows.length, cols: cols.length };
  };

  const copyCells = async (): Promise<boolean> => {
    const copied = selectedCellsTsv();
    if (!copied) return false;
    try {
      await navigator.clipboard.writeText(copied.tsv);
      setCopiedCells(copied);
      setPasteMsg(`${copied.rows}×${copied.cols} cells copied`);
      setTimeout(() => setPasteMsg(null), 2500);
      return true;
    } catch {
      setPasteMsg("Clipboard access was denied");
      setTimeout(() => setPasteMsg(null), 3000);
      return false;
    }
  };

  const cutCells = async () => {
    if (await copyCells()) {
      await clearSelectedCells();
      setPasteMsg("Cells cut to clipboard");
      setTimeout(() => setPasteMsg(null), 2500);
    }
  };

  const pasteCellsAt = async (startRow: number, startCol: string, tsv: string) => {
    if (!session) return;
    try {
      const rowOrder = visibleRowIds();
      const rowPos = rowOrder.indexOf(startRow);
      const colPos = columns.findIndex((c) => c.name === startCol);
      const res = await api.post(`/api/compute/${session.session_id}/paste_cells`, {
        start_row: startRow,
        start_col: startCol,
        row_indices: rowPos >= 0 ? rowOrder.slice(rowPos) : undefined,
        target_columns: colPos >= 0 ? columns.slice(colPos).map((c) => c.name) : undefined,
        tsv,
      });
      const refresh = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...refresh.data }); bumpUndo();
      setPasteMsg(`${res.data.pasted} cells pasted`);
      setTimeout(() => setPasteMsg(null), 2500);
    } catch (err: unknown) {
      setPasteMsg((err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Paste failed");
      setTimeout(() => setPasteMsg(null), 3500);
    }
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

  // Move a column to an explicit 1-based position (shifts the rest along).
  const moveToPosition = (colName: string, oneBased: number) => {
    if (!session) return;
    const idx = session.columns.findIndex((c) => c.name === colName);
    if (idx < 0) return;
    const target = Math.max(0, Math.min(oneBased - 1, session.columns.length - 1));
    if (target !== idx) reorderColumns(idx, target);
    setMoveCol(null);
  };

  // Open the bulk rename modal seeded with Sentence-case suggestions for EVERY
  // column (editable, so the user can also rename acronyms the auto-suggester
  // intentionally leaves untouched). Columns with a suggestion are pre-ticked.
  const openSuggestNames = async () => {
    if (!session) return;
    setCtxMenu(null);
    setSuggestBusy(true);
    try {
      const { getNameSuggestions } = await import("../api");
      const res = await getNameSuggestions(session.session_id);
      const s: Record<string, string> = res.data?.suggestions ?? {};
      const draft: Record<string, string> = {};
      const acc: Record<string, boolean> = {};
      for (const c of session.columns) {
        draft[c.name] = s[c.name] ?? c.name;
        acc[c.name] = c.name in s;  // pre-tick only the ones we actually changed
      }
      setSuggestDraft(draft);
      setSuggestAccept(acc);
      setSuggestOpen(true);
    } catch { /* ignore */ } finally { setSuggestBusy(false); }
  };

  // Apply ticked rows whose target differs from the current name, then refresh.
  const applySuggestions = async () => {
    if (!session) return;
    const pairs = session.columns
      .map((c) => c.name)
      .filter((n) => suggestAccept[n] && suggestDraft[n]?.trim() && suggestDraft[n].trim() !== n)
      .map((n) => [n, suggestDraft[n].trim()] as [string, string]);
    if (pairs.length === 0) { setSuggestOpen(false); return; }
    setSuggestBusy(true);
    try {
      const { renameColumn } = await import("../api");
      for (const [oldName, newName] of pairs) {
        try { await renameColumn(session.session_id, oldName, newName); } catch { /* skip dup/invalid */ }
      }
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      const cur = useStore.getState().session;
      if (cur) { useStore.getState().setSession({ ...cur, ...res.data }); bumpUndo(); }
    } catch { /* ignore */ } finally { setSuggestBusy(false); setSuggestOpen(false); }
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
    } catch (e: unknown) {
      // Surface the failure — previously swallowed, so MICE/fill errors looked
      // like nothing happened.
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      alert(detail ?? "Could not fill blanks for this column.");
    }
  };

  const startRename = (colName: string) => {
    setRenameCol(colName);
    setRenameVal(colName);
  };

  const commitRename = async () => {
    if (!renameCol || !session) return;
    const oldName = renameCol;  // capture before clearing state
    const newName = renameVal.trim();
    if (!newName || newName === oldName) {
      setRenameCol(null);
      return;
    }

    // Client-side duplicate-name guard. Keeps editor open so the user can
    // adjust the name instead of losing the input on a silent revert.
    const existingNames = new Set(session.columns.map((c) => c.name));
    if (existingNames.has(newName)) {
      alert(`Column "${newName}" already exists. Pick a different name.`);
      // Keep the input open with the rejected name selected for quick re-edit.
      setTimeout(() => renameRef.current?.select(), 0);
      return;
    }

    setRenameCol(null);
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
      // Remap per-column decimal formatting so the rename carries the user's
      // formatting choice over to the new column name.
      if (oldName in columnDecimals) {
        const next: Record<string, number> = { ...columnDecimals };
        next[newName] = next[oldName];
        delete next[oldName];
        useStore.setState({ columnDecimals: next });
      }
      useStore.getState().setSession({ ...session, columns: updatedCols, preview: updatedPreview }); bumpUndo();
    } catch (e: unknown) {
      // Surface backend errors (422 duplicate, network, etc.) instead of
      // silently dropping the rename. Falls back to a generic message.
      const detail =
        (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? (e instanceof Error ? e.message : String(e));
      alert(`Rename failed: ${detail}`);
    }
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

  const startEdit = (rowIdx: number, col: string, initialValue?: string) => {
    const val = preview[rowIdx]?.[col];
    selectSingleCell(rowIdx, col);
    setEditCell({ rowIdx, col });
    setEditValue(initialValue ?? (val === null || val === undefined ? "" : String(val)));
  };

  const commitEdit = async (restoreGridFocus = false) => {
    if (!editCell) return;

    const { rowIdx, col } = editCell;
    const commitKey = cellKey(rowIdx, col);
    if (committingCellsRef.current.has(commitKey)) return;
    committingCellsRef.current.add(commitKey);
    setEditCell(null);
    if (restoreGridFocus) {
      requestAnimationFrame(() => gridRef.current?.focus({ preventScroll: true }));
    }

    const original = preview[rowIdx]?.[col];
    const rawVal   = editValue.trim();
    const newVal   = rawVal === "" ? null : rawVal;

    if (String(original ?? "") === String(newVal ?? "")) {
      committingCellsRef.current.delete(commitKey);
      return;
    }

    const colKind = columns.find((c) => c.name === col)?.kind;
    const parsedNumber = rawVal === "" ? null : Number(rawVal);
    const optimisticValue =
      colKind === "numeric" && parsedNumber !== null && Number.isFinite(parsedNumber)
        ? parsedNumber
        : newVal;

    // Show the edit immediately; the backend response below normalizes the
    // value to the column dtype. Revert only when persistence fails.
    updatePreviewCell(rowIdx, col, optimisticValue);
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
      updatePreviewCell(rowIdx, col, original);
      setPasteMsg("Cell update failed; the previous value was restored");
      setTimeout(() => setPasteMsg(null), 3500);
    } finally {
      committingCellsRef.current.delete(commitKey);
      setSaving(false);
    }
  };

  const handleGridKeyDown = async (e: React.KeyboardEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    if (
      editCell || renameCol ||
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target instanceof HTMLButtonElement ||
      target instanceof HTMLAnchorElement ||
      target.isContentEditable
    ) return;

    const mod = e.metaKey || e.ctrlKey;

    if (mod && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (e.shiftKey) redo(); else undo();
      return;
    }
    if (mod && e.key.toLowerCase() === "y") {
      e.preventDefault();
      redo();
      return;
    }
    if (mod && e.key.toLowerCase() === "a") {
      e.preventDefault();
      const all = new Set<string>();
      for (const row of visibleRowIds()) {
        for (const col of columns) all.add(cellKey(row, col.name));
      }
      setSelectedCells(all);
      if (displayRows.length && columns.length) {
        setSelAnchor({ row: displayRows[0]._idx as number, col: columns[0].name });
        setSelFocus({
          row: displayRows[displayRows.length - 1]._idx as number,
          col: columns[columns.length - 1].name,
        });
      }
      return;
    }
    if (mod && e.key.toLowerCase() === "c" && selectedCells.size > 0) {
      e.preventDefault();
      await copyCells();
      return;
    }
    if (mod && e.key.toLowerCase() === "x" && selectedCells.size > 0) {
      e.preventDefault();
      await cutCells();
      return;
    }
    if (mod && e.key.toLowerCase() === "v" && session) {
      e.preventDefault();
      try {
        const text = await navigator.clipboard.readText();
        if (!text.trim()) return;
        const destination = selAnchor ?? selFocus;
        if (destination) {
          await pasteCellsAt(destination.row, destination.col, text);
          return;
        }
        const res = await api.post(`/api/compute/${session.session_id}/paste`, {
          tsv: text, has_header: true, mode: "append",
        });
        const refresh = await api.get(`/api/stats/${session.session_id}/refresh`);
        useStore.getState().setSession({ ...session, ...refresh.data }); bumpUndo();
        setPasteMsg(`${res.data.n_pasted} rows pasted`);
        setTimeout(() => setPasteMsg(null), 3000);
      } catch (err: unknown) {
        setPasteMsg((err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Paste failed");
        setTimeout(() => setPasteMsg(null), 4000);
      }
      return;
    }

    if ((e.key === "Delete" || e.key === "Backspace") && selectedCells.size > 0) {
      e.preventDefault();
      await clearSelectedCells();
      return;
    }
    if (e.key === "Escape" && selectedCells.size > 0) {
      e.preventDefault();
      setSelectedCells(new Set());
      setSelAnchor(null);
      setSelFocus(null);
      return;
    }

    const directions: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };
    if (e.key in directions && selFocus) {
      e.preventDefault();
      const [rowDelta, colDelta] = directions[e.key];
      moveSelectionFocus(rowDelta, colDelta, e.shiftKey, mod);
      return;
    }
    if (e.key === "Tab" && selFocus) {
      e.preventDefault();
      moveSelectionFocus(0, e.shiftKey ? -1 : 1, false, false);
      return;
    }
    if ((e.key === "Enter" || e.key === "F2") && selFocus) {
      e.preventDefault();
      startEdit(selFocus.row, selFocus.col);
      return;
    }
    if (!mod && !e.altKey && e.key.length === 1 && selFocus) {
      e.preventDefault();
      startEdit(selFocus.row, selFocus.col, e.key);
    }
  };

  const activeFilters = Object.values(filters).filter(Boolean).length;

  return (
    <div
      ref={gridRef}
      tabIndex={0}
      onKeyDown={handleGridKeyDown}
      onMouseUp={() => { dragSelectingRef.current = false; }}
      className="flex flex-col gap-2 h-full focus:outline-none"
      style={{ minHeight: 0 }}
    >
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
            ⊂ Select Cases
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

        </div>
      </div>

      {/* ── Table ── */}
      <div className="overflow-auto rounded-xl border border-gray-200 flex-1" style={{ minHeight: 0 }}>
        <table className="w-full text-sm border-collapse">
          <thead className="sticky top-0 z-10">

            {/* Column-number row (1, 2, 3 … above each column) */}
            <tr className="bg-gray-50 border-b border-gray-100">
              <th
                className="py-0.5 text-center text-gray-300 text-[9px] font-normal border-r border-gray-200 select-none sticky left-0 bg-gray-50 z-20"
                style={{ width: HASH_COL_W, minWidth: HASH_COL_W, maxWidth: HASH_COL_W }}
              />
              {columns.map((col, colIdx) => {
                const frozen = isFrozenCol(colIdx);
                return (
                  <th
                    key={col.name}
                    className={`py-0.5 text-center text-gray-300 text-[9px] font-normal border-r border-gray-200 select-none ${frozen ? "sticky bg-gray-50 z-20" : ""}`}
                    style={frozen ? { left: frozenLeft(colIdx), width: FROZEN_COL_W, minWidth: FROZEN_COL_W, maxWidth: FROZEN_COL_W } : undefined}
                  >
                    {colIdx + 1}
                  </th>
                );
              })}
            </tr>

            {/* Column headers */}
            <tr className="bg-gray-50 border-b border-gray-200">
              <th
                className="px-1 py-2 text-center text-gray-400 text-xs font-normal border-r border-gray-200 select-none sticky left-0 bg-gray-50 z-20"
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
                    onMouseDown={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                        e.preventDefault();
                        e.stopPropagation();
                        selectVisibleColumn(col.name, true);
                      }
                    }}
                    onClickCapture={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                        e.preventDefault();
                        e.stopPropagation();
                      }
                    }}
                    onDragStart={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                        e.preventDefault();
                        return;
                      }
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
                    title="Ctrl/Cmd+Shift+click selects the visible column"
                  >
                    <div className="flex flex-col gap-1">
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
                            <span className={`text-left text-xs font-medium truncate cursor-text ${col.analysis_excluded ? "text-gray-400 line-through" : "text-gray-700"}`}
                              onDoubleClick={() => startRename(col.name)}
                              title={col.analysis_excluded ? "Excluded from analysis · double-click to rename" : "Double-click to rename"}>
                              {col.name}
                            </span>
                          )}
                          {col.analysis_excluded && (
                            <span className="flex-shrink-0 text-[8px] font-bold px-1 py-0.5 rounded bg-violet-100 text-violet-600 border border-violet-300"
                              title="Excluded from analysis (kept in the dataset)">excl</span>
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
                      {nMissing > 0 && (() => {
                        const pct = preview.length ? (nMissing / preview.length) * 100 : 0;
                        const pctLabel = pct >= 10 ? pct.toFixed(0) : pct.toFixed(1);
                        return (
                          <div className="flex justify-start">
                            <button
                              onClick={() => {
                                setShowMissingOnly(true);
                                setFilters((prev) => ({ ...prev, [col.name]: "" }));
                              }}
                              title={`${nMissing} missing values (${pctLabel}% of ${preview.length} rows) — click to filter`}
                              className="flex-shrink-0 text-[9px] font-semibold px-1 py-0.5 rounded bg-amber-100 text-amber-700 border border-amber-300 hover:bg-amber-200 transition-colors"
                            >
                              {nMissing}✕ · {pctLabel}%
                            </button>
                          </div>
                        );
                      })()}
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
                    className="px-1 py-1.5 text-gray-300 text-[11px] border-r border-gray-200 select-none text-center cursor-context-menu sticky left-0 bg-white group-hover:bg-gray-50 z-10"
                    style={{ width: HASH_COL_W, minWidth: HASH_COL_W, maxWidth: HASH_COL_W }}
                    onMouseDown={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                        e.preventDefault();
                        selectVisibleRow(origIdx, true);
                      }
                    }}
                    onContextMenu={(e) => { e.preventDefault(); setRowCtx({ x: e.clientX, y: e.clientY, idx: origIdx }); }}
                    title={`Original row #${origIdx + 1} in the dataset · Ctrl/Cmd+Shift+click selects the row`}
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
                        data-grid-row={origIdx}
                        data-grid-col={colIdx}
                        onMouseDown={(e) => {
                          if (!isEditing) beginCellSelection(origIdx, col.name, e);
                        }}
                        onMouseEnter={(e) => {
                          if (!isEditing) extendMouseSelection(origIdx, col.name, e);
                        }}
                        onDoubleClick={() => {
                          if (!isEditing) startEdit(origIdx, col.name);
                        }}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          // If right-clicking an unselected cell, select just that cell
                          if (!selectedCells.has(cellKey(origIdx, col.name))) {
                            setSelectedCells(new Set([cellKey(origIdx, col.name)]));
                            setSelAnchor({ row: origIdx, col: col.name });
                            setSelFocus({ row: origIdx, col: col.name });
                          }
                          gridRef.current?.focus({ preventScroll: true });
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
                              if (e.key === "Enter") {
                                e.preventDefault();
                                void commitEdit(true);
                                moveSelectionFocus(1, 0, false, false);
                              }
                              if (e.key === "Tab") {
                                e.preventDefault();
                                void commitEdit(true);
                                moveSelectionFocus(0, e.shiftKey ? -1 : 1, false, false);
                              }
                              if (e.key === "Escape") {
                                setEditCell(null);
                                requestAnimationFrame(() => gridRef.current?.focus({ preventScroll: true }));
                              }
                            }}
                            onBlur={() => { void commitEdit(false); }}
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
        <span>Click to select · Double-click / Enter to edit · Drag or Shift+arrows for range · Ctrl/Cmd+C/X/V</span>
      </div>

      {/* ── Right-click context menu ── */}
      {ctxMenu && (
        <div ref={ctxRef}
          className="fixed z-50 bg-white border border-gray-200 rounded-xl shadow-xl py-1 w-48"
          style={columnMenuStyle}
          role="menu">
          <div className="sticky top-0 z-10 bg-white px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100 truncate">
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
          <button onClick={() => { setFindReplaceCol(ctxMenu.col); setCtxMenu(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            🔁 Find &amp; Replace…
          </button>
          <button onClick={() => { setParseDateCol(ctxMenu.col); setCtxMenu(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📅 Parse as date…
          </button>
          <button onClick={() => {
            const col = columns.find((c) => c.name === ctxMenu.col);
            setColumnAnalysisExcluded(ctxMenu.col, !(col?.analysis_excluded ?? false));
            setCtxMenu(null);
          }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            {columns.find((c) => c.name === ctxMenu.col)?.analysis_excluded
              ? "✅ Include in analysis" : "🚫 Exclude from analysis"}
          </button>
          <button onClick={openSuggestNames}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            💡 Suggest names…
          </button>
          {/* Decimal places selector */}
          {columns.find((c) => c.name === ctxMenu.col)?.kind === "numeric" && (
            <div className="px-3 py-1">
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-gray-500">🔢 Decimals:</span>
                {[0, 1, 2, 3, 4, "auto"].map((d) => (
                  <button key={String(d)}
                    onClick={() => {
                      if (d === "auto") {
                        clearColumnDecimals(ctxMenu.col);
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
              {/* Hint — surfaces that the decimals control now affects more
                  than just this data table. */}
              <p className="text-[9px] text-gray-400 mt-1 leading-snug">
                Summary, Histogram &amp; Tablo 1 metriklerine de uygulanır. "A" =
                otomatik (integer kolonlar 0, diğerleri 2 ondalık).
              </p>
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
          <button onClick={() => { setMoveCol(ctxMenu.col); setCtxMenu(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📍 Move to position…
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
          style={cellMenuStyle}
          role="menu">
          <div className="sticky top-0 z-10 bg-white px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100 truncate">
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
          <button onClick={() => { void cutCells(); setCellCtx(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ✂ Cut {selectedCells.size > 1 ? `${selectedCells.size} cells` : "cell"}
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
          style={rowMenuStyle}
          role="menu">
          <div className="sticky top-0 z-10 bg-white px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100">Row {rowCtx.idx + 1}</div>
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
      {valueLabelCol && (
        <ValueLabelsModal
          colName={valueLabelCol}
          columns={columns}
          preview={preview}
          draft={valueLabelDraft}
          setDraft={setValueLabelDraft}
          session={session}
          onClose={() => setValueLabelCol(null)}
        />
      )}

      {/* ── Find & Replace Modal ── */}
      {findReplaceCol && (
        <FindReplaceModal
          colName={findReplaceCol}
          columns={columns}
          preview={preview}
          session={session}
          onClose={() => setFindReplaceCol(null)}
          onApplied={bumpUndo}
        />
      )}

      {/* ── Parse as Date Modal ── */}
      {parseDateCol && (
        <ParseDatesModal
          colName={parseDateCol}
          columns={columns}
          session={session}
          onClose={() => setParseDateCol(null)}
          onApplied={bumpUndo}
        />
      )}

      {/* ── Move column to position ── */}
      {moveCol && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setMoveCol(null)}>
          <div className="bg-white rounded-xl shadow-2xl w-72" onClick={(e) => e.stopPropagation()}>
            <div className="px-4 py-3 border-b border-gray-200">
              <h3 className="text-sm font-semibold text-gray-800">Move column</h3>
              <p className="text-[11px] text-gray-400 mt-0.5 truncate">{moveCol}</p>
            </div>
            <div className="px-4 py-3 space-y-2">
              <label className="text-xs text-gray-500">New position (1–{columns.length})</label>
              <input type="number" min={1} max={columns.length} autoFocus
                defaultValue={columns.findIndex((c) => c.name === moveCol) + 1}
                onKeyDown={(e) => { if (e.key === "Enter") moveToPosition(moveCol, parseInt((e.target as HTMLInputElement).value, 10)); if (e.key === "Escape") setMoveCol(null); }}
                id="move-pos-input"
                className="w-full text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" />
              <p className="text-[10px] text-gray-400">Other columns shift to make room. The “#” and frozen columns stay pinned.</p>
            </div>
            <div className="px-4 py-3 border-t border-gray-200 flex justify-end gap-2">
              <button onClick={() => setMoveCol(null)} className="px-3 py-1.5 text-xs text-gray-500 border border-gray-200 rounded-lg hover:bg-gray-50">Cancel</button>
              <button onClick={() => { const el = document.getElementById("move-pos-input") as HTMLInputElement | null; moveToPosition(moveCol, el ? parseInt(el.value, 10) : 1); }}
                className="px-3 py-1.5 text-xs bg-indigo-600 text-white rounded-lg hover:bg-indigo-700">Move</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Suggest names (bulk review) ── */}
      {suggestOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setSuggestOpen(false)}>
          <div className="bg-white rounded-xl shadow-2xl w-[28rem] max-h-[80vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
            <div className="px-5 py-3.5 border-b border-gray-200">
              <h3 className="text-sm font-semibold text-gray-800">Rename columns</h3>
              <p className="text-[11px] text-gray-400 mt-0.5">Edit any target name. Sentence-case suggestions are pre-filled and ticked; medical acronyms (LDL, DM…) keep their case — tick &amp; edit to change them too. Applying renames the ticked rows.</p>
            </div>
            <div className="flex-1 overflow-y-auto px-5 py-3 space-y-1">
              {(session?.columns ?? []).map((c) => {
                const changed = (suggestDraft[c.name] ?? c.name).trim() !== c.name;
                return (
                  <div key={c.name} className="flex items-center gap-2 text-xs">
                    <input type="checkbox" checked={suggestAccept[c.name] ?? false}
                      onChange={(e) => setSuggestAccept((p) => ({ ...p, [c.name]: e.target.checked }))}
                      className="accent-indigo-500 flex-shrink-0" />
                    <span className="font-mono text-gray-400 truncate w-32 flex-shrink-0" title={c.name}>{c.name}</span>
                    <span className="text-gray-300 flex-shrink-0">→</span>
                    <input
                      value={suggestDraft[c.name] ?? c.name}
                      onChange={(e) => {
                        setSuggestDraft((p) => ({ ...p, [c.name]: e.target.value }));
                        setSuggestAccept((p) => ({ ...p, [c.name]: e.target.value.trim() !== c.name }));
                      }}
                      className={`flex-1 text-xs border rounded px-2 py-0.5 focus:outline-none focus:border-indigo-400 ${changed ? "border-indigo-300 text-gray-900" : "border-gray-200 text-gray-500"}`} />
                  </div>
                );
              })}
            </div>
            <div className="px-5 py-3 border-t border-gray-200 flex items-center justify-between">
              <button onClick={() => setSuggestAccept((session?.columns ?? []).reduce((a, c) => ({ ...a, [c.name]: false }), {}))}
                className="text-xs text-gray-400 hover:text-gray-700">Untick all</button>
              <div className="flex gap-2">
                <button onClick={() => setSuggestOpen(false)} className="px-3 py-1.5 text-xs text-gray-500 border border-gray-200 rounded-lg hover:bg-gray-50">Cancel</button>
                <button onClick={applySuggestions} disabled={suggestBusy}
                  className="px-3 py-1.5 text-xs bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50">
                  {suggestBusy ? "Applying…" : "Apply ticked"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

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
