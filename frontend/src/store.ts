import { create } from "zustand";

export interface ColMeta {
  name: string;
  dtype: string;
  kind: "numeric" | "categorical" | "text" | "date";
  label?: string;
  description?: string;
  units?: string;
  value_labels?: Record<string, string>;
  role?: "outcome" | "predictor" | "covariate" | "id" | "time" | "event" | "";
}

export interface Session {
  session_id: string;
  filename: string;
  rows: number;
  columns: ColMeta[];
  preview: Record<string, unknown>[];
}

export type PaletteName = "indigo" | "clinical" | "nature" | "grayscale" | "warm" | "jama";

export interface PlotTheme {
  palette: PaletteName;
  fontFamily: string;
  fontSize: number;
  lineWidth: number;
  markerSize: number;
  markerOpacity: number;
  plotBg: string;
}

export const DEFAULT_THEME: PlotTheme = {
  palette: "indigo",
  fontFamily: "system-ui, sans-serif",
  fontSize: 11,
  lineWidth: 2,
  markerSize: 6,
  markerOpacity: 0.7,
  plotBg: "#ffffff",
};

export const PALETTES: Record<PaletteName, string[]> = {
  indigo:    ["#6366f1","#f59e0b","#10b981","#ef4444","#8b5cf6","#06b6d4","#84cc16","#f97316"],
  clinical:  ["#1a5276","#2874a6","#5dade2","#27ae60","#d35400","#8e44ad","#c0392b","#2c3e50"],
  nature:    ["#27ae60","#2ecc71","#f39c12","#e67e22","#8e44ad","#3498db","#e74c3c","#1abc9c"],
  grayscale: ["#111827","#374151","#6b7280","#9ca3af","#d1d5db","#4b5563","#1f2937","#374151"],
  warm:      ["#dc2626","#ea580c","#d97706","#ca8a04","#65a30d","#16a34a","#0891b2","#7c3aed"],
  jama:      ["#003087","#7f0000","#003b00","#5e0070","#663300","#004c4c","#004080","#380038"],
};

export type CaseOperator = "eq" | "ne" | "gt" | "lt" | "gte" | "lte" | "contains" | "missing" | "not_missing";

export interface CaseCondition {
  column: string;
  operator: CaseOperator;
  value: string;
  join: "AND" | "OR";
}

export interface CaseFilter {
  conditions: CaseCondition[];
  selected: number;
  total: number;
}

interface AppState {
  session: Session | null;
  activeTab: string;
  showGrid: boolean;
  plotTheme: PlotTheme;
  caseFilter: CaseFilter | null;
  setSession: (s: Session) => void;
  setActiveTab: (t: string) => void;
  toggleGrid: () => void;
  clearSession: () => void;
  setPlotTheme: (patch: Partial<PlotTheme>) => void;
  setCaseFilter: (f: CaseFilter | null) => void;
  // Column kind override (data tab kind badge)
  updateColumnKind: (name: string, kind: ColMeta["kind"]) => void;
  // Inline cell editing
  updatePreviewCell: (rowIdx: number, col: string, value: unknown) => void;
  // Computed columns (Compute tab)
  addSessionColumn: (col: ColMeta, previewValues: (number | string | null)[]) => void;
  removeSessionColumn: (name: string) => void;
  // Column reordering (drag & drop)
  reorderColumns: (fromIndex: number, toIndex: number) => void;
  // Table 1 persistence across tab switches
  table1Result: any;
  setTable1Result: (r: any) => void;
  clearTable1: () => void;
  // Generic panel result cache — persists results across tab switches
  panelCache: Record<string, any>;
  setPanelCache: (panel: string, data: any) => void;
  clearPanelCache: (panel: string) => void;
  // Column decimal formatting
  columnDecimals: Record<string, number>;  // col name → decimal places
  setColumnDecimals: (col: string, decimals: number) => void;
  // Undo / Redo (backend-driven)
  undoDepth: number;
  redoDepth: number;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  deleteRow: (rowIdx: number) => Promise<void>;
  
  // Descriptive tab UI state
  descriptiveTab: "histogram" | "boxplot" | "violin" | "qq";
  setDescriptiveTab: (tab: "histogram" | "boxplot" | "violin" | "qq") => void;
}

const loadTheme = (): PlotTheme => {
  try { return { ...DEFAULT_THEME, ...JSON.parse(localStorage.getItem("plotTheme") ?? "{}") }; }
  catch { return DEFAULT_THEME; }
};

export const useStore = create<AppState>((set) => ({
  session: null,
  activeTab: "data",
  descriptiveTab: "histogram",
  setDescriptiveTab: (t) => set({ descriptiveTab: t }),
  showGrid: localStorage.getItem("showGrid") !== "false",
  plotTheme: loadTheme(),
  table1Result: null,
  caseFilter: null,
  setSession: (s) => set({ session: s, activeTab: "data", table1Result: null, caseFilter: null, panelCache: {}, undoDepth: 0, redoDepth: 0, columnDecimals: {} }),
  setActiveTab: (t) => set({ activeTab: t }),
  setCaseFilter: (f) => set({ caseFilter: f }),
  toggleGrid: () => set((state) => {
    const next = !state.showGrid;
    localStorage.setItem("showGrid", String(next));
    return { showGrid: next };
  }),
  setPlotTheme: (patch) => set((state) => {
    const next = { ...state.plotTheme, ...patch };
    localStorage.setItem("plotTheme", JSON.stringify(next));
    return { plotTheme: next };
  }),
  clearSession: () => set({ session: null, activeTab: "data", table1Result: null, caseFilter: null, panelCache: {}, undoDepth: 0, redoDepth: 0 }),
  updateColumnKind: (name, kind) => {
    set((state) => {
      if (!state.session) return state;
      // Fire-and-forget persistence to the backend so save_session captures
      // the user's classification override. Import done lazily to avoid a
      // circular import (api.ts → store).
      import("./api").then(({ setColumnKind }) => {
        setColumnKind(state.session!.session_id, name, kind).catch(() => {
          /* network errors here are non-fatal — UI state is the source of
             truth for the current session; the next save attempt re-syncs. */
        });
      });
      return {
        session: {
          ...state.session,
          columns: state.session.columns.map((c) =>
            c.name === name ? { ...c, kind } : c
          ),
        },
      };
    });
  },
  updatePreviewCell: (rowIdx, col, value) =>
    set((state) => {
      if (!state.session) return state;
      const preview = [...state.session.preview];
      preview[rowIdx] = { ...preview[rowIdx], [col]: value };
      return { session: { ...state.session, preview } };
    }),
  addSessionColumn: (col, previewValues) =>
    set((state) => {
      if (!state.session) return state;
      // Replace existing column with same name, or append
      const columns = [
        ...state.session.columns.filter((c) => c.name !== col.name),
        col,
      ];
      const preview = state.session.preview.map((row, i) => ({
        ...row,
        [col.name]: previewValues[i] ?? null,
      }));
      return { session: { ...state.session, columns, preview } };
    }),
  removeSessionColumn: (name) =>
    set((state) => {
      if (!state.session) return state;
      const columns = state.session.columns.filter((c) => c.name !== name);
      const preview = state.session.preview.map((row) => {
        const r = { ...row };
        delete r[name];
        return r;
      });
      return { session: { ...state.session, columns, preview } };
    }),
  reorderColumns: (fromIndex, toIndex) =>
    set((state) => {
      if (!state.session || fromIndex === toIndex) return state;
      const cols = [...state.session.columns];
      const [moved] = cols.splice(fromIndex, 1);
      cols.splice(toIndex, 0, moved);
      // Sync to backend so column order persists across refresh/export
      const colNames = cols.map((c) => c.name);
      import("./api").then((api) => {
        api.default.post(`/api/sessions/${state.session!.session_id}/reorder_columns`, { columns: colNames }).catch(() => {});
      });
      return { session: { ...state.session, columns: cols } };
    }),
  setTable1Result: (r) => set({ table1Result: r }),
  clearTable1: () => set({ table1Result: null }),
  panelCache: {},
  setPanelCache: (panel, data) => set((state) => ({ panelCache: { ...state.panelCache, [panel]: data } })),
  clearPanelCache: (panel) => set((state) => {
    const next = { ...state.panelCache };
    delete next[panel];
    return { panelCache: next };
  }),
  // Column decimal formatting
  columnDecimals: {},
  setColumnDecimals: (col, decimals) => set((state) => ({ columnDecimals: { ...state.columnDecimals, [col]: decimals } })),
  // Undo / Redo — backend-driven (DataFrame snapshots on server)
  undoDepth: 0,
  redoDepth: 0,
  undo: async () => {
    const state = useStore.getState();
    if (!state.session) return;
    try {
      const { default: api } = await import("./api");
      const res = await api.post(`/api/sessions/${state.session.session_id}/undo`);
      const d = res.data;
      set({ session: { ...state.session, rows: d.rows, columns: d.columns, preview: d.preview },
            undoDepth: d.undo_depth ?? 0, redoDepth: d.redo_depth ?? 0 });
    } catch { /* nothing to undo */ }
  },
  redo: async () => {
    const state = useStore.getState();
    if (!state.session) return;
    try {
      const { default: api } = await import("./api");
      const res = await api.post(`/api/sessions/${state.session.session_id}/redo`);
      const d = res.data;
      set({ session: { ...state.session, rows: d.rows, columns: d.columns, preview: d.preview },
            undoDepth: d.undo_depth ?? 0, redoDepth: d.redo_depth ?? 0 });
    } catch { /* nothing to redo */ }
  },
  deleteRow: async (rowIdx: number) => {
    const state = useStore.getState();
    if (!state.session) return;
    try {
      const { deleteRow } = await import("./api");
      const res = await deleteRow(state.session.session_id, rowIdx);
      const d = res.data;
      // Same refresh payload structure as undo/redo, triggers global re-renders
      set({ 
        session: { ...state.session, rows: d.rows, columns: d.columns, preview: d.preview },
        // Add 1 to undo depth because delete is a destructive action we pushed
        undoDepth: state.undoDepth + 1,
        redoDepth: 0 
      });
    } catch (e) {
      console.error("Failed to delete row", e);
      throw e;
    }
  },
}));
